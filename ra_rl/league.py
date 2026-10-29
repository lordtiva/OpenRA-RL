"""Phase 4 League & Multi-Bot Curriculum Training for the Commander Network.

Trains the commander network against a diverse league of opponents:
  - hard (OpenRA normal bot): macro and balanced army
  - brutal (OpenRA rush bot): aggressive early pressure / anti-rush
  - turtle (OpenRA turtle bot): fortified defenses / siege challenge

Collapse-prevention safeguards:
  - Strict entropy floor (--ent-min 0.004) preventing policy sharpening / argmax collapse.
  - Multi-opponent sampling in every rollout iteration (e.g. 50% hard, 30% brutal, 20% turtle).
  - Dual evaluation (hard + brutal) on selection seeds (20000+), tracking harmonic mean win rate.
  - Comprehensive TensorBoard logging across all opponent types and diagnostic metrics.
"""

import argparse
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import numpy as np
import torch
from torch.distributions import Categorical

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    SummaryWriter = None
    HAS_TB = False

from ra_env import Bridge, GameSession, MatchConfig, daemons, pb
from ra_rl.eval import DEFAULT_DAEMONS, evaluate
from ra_rl.features import Features
from ra_rl.model import CommanderNet
from ra_rl.ppo import ARMY_KEYS, PROD_KEYS, TERMINAL_REWARDS, GameTrajectory, pack, ppo_update, to_torch_dict
from ra_rl.vocab import QUEUE_INDEX


def collect_game_league(bridge: Bridge, net: CommanderNet, match: MatchConfig, seed: int,
                        gamma: float, gae_lambda: float, step_ticks: int = 500) -> tuple[dict, dict, dict]:
    session = GameSession(bridge, match)
    feats = Features()
    traj = GameTrajectory()
    t0 = time.perf_counter()
    decisions = 0
    error = ""
    outcome = "error"

    try:
        step = session.reset(seed)
        while not (step.done or step.truncated):
            feats.observe(step.obs)
            answer: list[pb.Decision] = []
            if step.pending:
                g_tensor = torch.from_numpy(feats.g).unsqueeze(0)
                with torch.no_grad():
                    val = float(net.value(g_tensor).item())

                for req in step.pending:
                    if req.kind == pb.DECIDE_ARMY:
                        cand, mask = feats.candidates(req)
                        cand_t = torch.from_numpy(cand).unsqueeze(0)
                        mask_t = torch.from_numpy(mask).unsqueeze(0)
                        with torch.no_grad():
                            logits = net.army_logits(g_tensor, cand_t, mask_t)
                            dist = Categorical(logits=logits)
                            action = int(dist.sample().item())
                            lp = float(dist.log_prob(torch.tensor(action)).item())

                        d = feats.decision(req, action)
                        feats.record(req, d)
                        if d is not None:
                            answer.append(d)
                        traj.add_army(feats.g, cand, mask, action, lp, val)
                        decisions += 1
                    else:
                        q_idx = QUEUE_INDEX.get(req.queue, 0)
                        q_t = torch.tensor([q_idx])
                        mask = feats.item_mask(req)
                        mask_t = torch.from_numpy(mask).unsqueeze(0)
                        with torch.no_grad():
                            logits = net.item_logits(g_tensor, q_t, mask_t)
                            dist = Categorical(logits=logits)
                            action = int(dist.sample().item())
                            lp = float(dist.log_prob(torch.tensor(action)).item())

                        d = feats.decision(req, action)
                        feats.record(req, d)
                        if d is not None:
                            answer.append(d)
                        traj.add_item(feats.g, q_idx, mask, action, lp, val)
                        decisions += 1

            step = session.step(answer, step_ticks)
        outcome = step.result or "truncated"
    except (grpc.RpcError, RuntimeError) as e:
        error = repr(e)
    finally:
        session.close()

    term_reward = TERMINAL_REWARDS.get(outcome, -0.5) if not error else 0.0
    prod_data, army_data = traj.finish(term_reward, gamma, gae_lambda) if not error else ({k: [] for k in PROD_KEYS}, {k: [] for k in ARMY_KEYS})

    meta = {
        "seed": seed, "bot": match.bot, "outcome": outcome, "ticks": session.tick,
        "decisions": decisions, "wall_s": round(time.perf_counter() - t0, 1),
        "error": error,
    }
    return meta, prod_data, army_data


def collect_league_rollouts(addresses: list[str], net: CommanderNet, games: int, seed_base: int,
                            opponents: list[str], opponent_weights: list[float],
                            map_name: str, max_ticks: int, gamma: float, gae_lambda: float,
                            recycle_every: int = 60, step_ticks: int = 500) -> tuple[dict, dict, dict]:
    net_cpu = copy.deepcopy(net).cpu().eval()
    
    # Assign opponent bot per game based on weights
    probs = np.array(opponent_weights, dtype=np.float64)
    probs /= probs.sum()
    counts = np.random.multinomial(games, probs)
    bot_assignments = []
    for bot_name, count in zip(opponents, counts):
        bot_assignments.extend([bot_name] * count)
    while len(bot_assignments) < games:
        bot_assignments.append(opponents[0])
    np.random.shuffle(bot_assignments)

    job_list = [(seed_base + i, bot_assignments[i]) for i in range(games)]
    lock = threading.Lock()
    todo = list(job_list)

    all_prod: dict[str, list] = {k: [] for k in PROD_KEYS}
    all_army: dict[str, list] = {k: [] for k in ARMY_KEYS}
    metas: list[dict] = []

    def worker(addr: str) -> None:
        bridge = Bridge(addr)
        played = 0
        try:
            while True:
                with lock:
                    if not todo:
                        return
                    seed, bot_name = todo.pop(0)

                if recycle_every and played >= recycle_every:
                    daemons.restart(addr)
                    played = 0

                match = MatchConfig(map_name=map_name, bot=bot_name, max_ticks=max_ticks, brain="external")
                meta, p_data, a_data = collect_game_league(bridge, net_cpu, match, seed, gamma, gae_lambda, step_ticks)
                played += 1

                if meta["error"] and "never became ready" in meta["error"]:
                    daemons.restart(addr)
                    played = 0
                    meta, p_data, a_data = collect_game_league(bridge, net_cpu, match, seed, gamma, gae_lambda, step_ticks)
                    played += 1

                with lock:
                    metas.append(meta)
                    if not meta["error"]:
                        for k in PROD_KEYS:
                            all_prod[k].extend(p_data[k])
                        for k in ARMY_KEYS:
                            all_army[k].extend(a_data[k])
        finally:
            bridge.close()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(addresses)) as pool:
        list(pool.map(worker, addresses))

    packed_prod = pack(all_prod, PROD_KEYS)
    packed_army = pack(all_army, ARMY_KEYS)

    if len(packed_prod.get("adv", [])) > 1:
        adv = packed_prod["adv"]
        packed_prod["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
    if len(packed_army.get("adv", [])) > 1:
        adv = packed_army["adv"]
        packed_army["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)

    valid_metas = [m for m in metas if not m["error"]]
    outcomes: dict[str, int] = {}
    bot_stats: dict[str, dict] = {}

    for bot in opponents:
        bot_metas = [m for m in valid_metas if m["bot"] == bot]
        w = sum(1 for m in bot_metas if m["outcome"] == "win")
        bot_stats[bot] = {
            "games": len(bot_metas),
            "wins": w,
            "win_rate": round(w / max(1, len(bot_metas)), 3),
        }

    for m in valid_metas:
        outcomes[m["outcome"]] = outcomes.get(m["outcome"], 0) + 1

    stats = {
        "games_collected": len(valid_metas),
        "games_failed": len(metas) - len(valid_metas),
        "outcomes": outcomes,
        "win_rate": round(outcomes.get("win", 0) / max(1, len(valid_metas)), 3),
        "mean_ticks": round(sum(m["ticks"] for m in valid_metas) / max(1, len(valid_metas))),
        "mean_decisions": round(sum(m["decisions"] for m in valid_metas) / max(1, len(valid_metas)), 1),
        "n_prod": len(packed_prod.get("g", [])),
        "n_army": len(packed_army.get("g", [])),
        "collect_s": round(time.perf_counter() - t0, 1),
        "bot_stats": bot_stats,
    }

    return stats, packed_prod, packed_army


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="runs/phase3/ppo_v1/checkpoint_best.pt", help="starting checkpoint (Phase 3 best)")
    ap.add_argument("--out", required=True, help="output directory for checkpoints and logs")
    ap.add_argument("--iterations", type=int, default=100, help="total league training iterations")
    ap.add_argument("--games-per-iter", type=int, default=48, help="number of games collected per iteration")
    ap.add_argument("--opponents", default="hard,brutal,turtle", help="comma-separated opponent bot types")
    ap.add_argument("--opponent-weights", default="0.5,0.3,0.2", help="relative sampling weights for opponents")
    ap.add_argument("--map", default=MatchConfig.map_name)
    ap.add_argument("--max-ticks", type=int, default=54000)
    ap.add_argument("--daemons", default=DEFAULT_DAEMONS)
    ap.add_argument("--seed-base", type=int, default=300000, help="starting seed for rollouts (never overlaps with eval)")
    ap.add_argument("--lr", type=float, default=2e-4, help="slightly reduced lr for fine-tuning")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.997, help="decision discount factor")
    ap.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    ap.add_argument("--ppo-epochs", type=int, default=4, help="optimization epochs per iteration")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--val-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--ent-decay", type=float, default=0.992, help="slower decay to prevent premature sharpening")
    ap.add_argument("--ent-min", type=float, default=0.004, help="strict minimum entropy floor to prevent argmax collapse")
    ap.add_argument("--army-weight", type=float, default=1.0)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.04)
    ap.add_argument("--eval-every", type=int, default=10, help="evaluate every N iterations")
    ap.add_argument("--eval-games-per-bot", type=int, default=50, help="eval games for each tested bot")
    ap.add_argument("--eval-seeds-base", type=int, default=20000, help="selection seeds (20000-20099)")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--recycle-every", type=int, default=60)
    ap.add_argument("--tb", action=argparse.BooleanOptionalAction, default=True, help="log metrics to TensorBoard")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    daemons_list = [d.strip() for d in args.daemons.split(",") if d.strip()]
    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    weights = [float(w.strip()) for w in args.opponent_weights.split(",") if w.strip()]

    writer = SummaryWriter(log_dir=str(out / "tb")) if (args.tb and HAS_TB) else None

    print(json.dumps({
        "msg": "Starting Phase 4 League Training",
        "init": args.init,
        "out": str(out),
        "device": device,
        "daemons": len(daemons_list),
        "opponents": opponents,
        "weights": weights,
        "ent_min": args.ent_min,
        "games_per_iter": args.games_per_iter,
    }), flush=True)

    net = CommanderNet.load(args.init, device=device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cur_seed = args.seed_base
    cur_ent = args.ent_coef
    best_harmonic_score = -1.0
    log = (out / "metrics.jsonl").open("a", encoding="utf-8")

    t_start = time.perf_counter()

    for it in range(1, args.iterations + 1):
        t_it = time.perf_counter()

        # 1. Collect league rollouts across diverse opponents
        collect_stats, prod_np, army_np = collect_league_rollouts(
            addresses=daemons_list,
            net=net,
            games=args.games_per_iter,
            seed_base=cur_seed,
            opponents=opponents,
            opponent_weights=weights,
            map_name=args.map,
            max_ticks=args.max_ticks,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            recycle_every=args.recycle_every,
        )
        cur_seed += args.games_per_iter

        # 2. PPO update
        prod_t = to_torch_dict(prod_np, device)
        army_t = to_torch_dict(army_np, device)

        update_metrics = ppo_update(
            net=net,
            optimizer=optimizer,
            prod=prod_t,
            army=army_t,
            epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            clip_eps=args.clip_eps,
            val_coef=args.val_coef,
            ent_coef=cur_ent,
            army_weight=args.army_weight,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
        )

        # Decay entropy smoothly down to ent_min floor
        cur_ent = max(args.ent_min, cur_ent * args.ent_decay)

        # 3. Checkpointing
        net.save(str(out / "checkpoint_latest.pt"), iteration=it, stats=collect_stats)
        if args.save_every > 0 and it % args.save_every == 0:
            net.save(str(out / f"checkpoint_iter_{it:04d}.pt"), iteration=it, stats=collect_stats)

        it_record = {
            "iteration": it,
            "ent_coef": round(cur_ent, 5),
            "it_s": round(time.perf_counter() - t_it, 1),
            "total_s": round(time.perf_counter() - t_start, 1),
            **collect_stats,
            **update_metrics,
        }

        # 4. Multi-Bot Dual Evaluation (hard + brutal)
        if args.eval_every > 0 and it % args.eval_every == 0:
            print(f"[Iter {it}] Running League Evaluation (vs hard & brutal)...", flush=True)
            eval_results = {}
            for test_bot in ("hard", "brutal"):
                eval_match = MatchConfig(map_name=args.map, bot=test_bot, max_ticks=args.max_ticks)
                eval_summary = evaluate(
                    daemons=daemons_list,
                    policy=f"net:{out / 'checkpoint_latest.pt'}",
                    match=eval_match,
                    games=args.eval_games_per_bot,
                    seed_base=args.eval_seeds_base,
                    step_ticks=500,
                    log=lambda _: None,
                )
                eval_results[test_bot] = {
                    "win_rate": eval_summary["win_rate"],
                    "ci95": eval_summary["win_rate_ci95"],
                    "outcomes": eval_summary["outcomes"],
                }

            wr_h = eval_results["hard"]["win_rate"]
            wr_b = eval_results["brutal"]["win_rate"]
            harmonic_score = round(2.0 * (wr_h * wr_b) / (wr_h + wr_b + 1e-8), 3)

            it_record["eval"] = {
                "hard": eval_results["hard"],
                "brutal": eval_results["brutal"],
                "harmonic_score": harmonic_score,
            }

            if harmonic_score > best_harmonic_score:
                best_harmonic_score = harmonic_score
                net.save(str(out / "checkpoint_best.pt"), iteration=it, eval=it_record["eval"])
                it_record["new_best"] = True

        log.write(json.dumps(it_record) + "\n")
        log.flush()
        print(json.dumps(it_record), flush=True)

        # 5. TensorBoard Logging
        if writer is not None:
            writer.add_scalar("rollout/win_rate_total", collect_stats["win_rate"], it)
            for bot, bstat in collect_stats.get("bot_stats", {}).items():
                writer.add_scalar(f"rollout/win_rate_{bot}", bstat["win_rate"], it)

            writer.add_scalar("rollout/mean_ticks", collect_stats["mean_ticks"], it)
            writer.add_scalar("rollout/mean_decisions", collect_stats["mean_decisions"], it)
            writer.add_scalar("rollout/collect_s", collect_stats["collect_s"], it)

            writer.add_scalar("loss/policy_prod", update_metrics["pi_loss_prod"], it)
            writer.add_scalar("loss/policy_army", update_metrics["pi_loss_army"], it)
            writer.add_scalar("loss/value_prod", update_metrics["val_loss_prod"], it)
            writer.add_scalar("loss/value_army", update_metrics["val_loss_army"], it)

            writer.add_scalar("policy/entropy_prod", update_metrics["ent_prod"], it)
            writer.add_scalar("policy/entropy_army", update_metrics["ent_army"], it)
            writer.add_scalar("policy/kl_prod", update_metrics["kl_prod"], it)
            writer.add_scalar("policy/kl_army", update_metrics["kl_army"], it)
            writer.add_scalar("policy/ent_coef", cur_ent, it)

            if "eval" in it_record:
                ev = it_record["eval"]
                writer.add_scalar("eval/win_rate_hard", ev["hard"]["win_rate"], it)
                writer.add_scalar("eval/win_rate_brutal", ev["brutal"]["win_rate"], it)
                writer.add_scalar("eval/harmonic_score", ev["harmonic_score"], it)

            writer.flush()

    log.close()
    if writer is not None:
        writer.close()
    print(f"League training complete in {time.perf_counter() - t_start:.0f}s. Best harmonic score: {best_harmonic_score}", flush=True)


if __name__ == "__main__":
    main()
