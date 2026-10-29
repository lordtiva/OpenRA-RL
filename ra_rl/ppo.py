"""PPO training for the commander network (Phase 3).

    python -m ra_rl.ppo --init runs/phase2/bc_s2/model.pt --out runs/phase3/ppo_v1

Features and architecture:
- Shared MLP trunk, item head (BUILD/TRAIN), army head (ARMY pointer + hold/keep),
  and scalar value head V(s).
- Rollouts collected in parallel across multi-session OpenRA daemons.
- GAE per game trajectory with decision-level discounting (default gamma=0.997).
- Standard PPO clipped objective + MSE value loss + entropy bonus + KL early stopping.
- Periodic evaluation following the official protocol on non-training seeds (20000+).
"""

import argparse
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import grpc
import numpy as np
import torch
import torch.nn.functional as F
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
from ra_rl.vocab import QUEUE_INDEX

TERMINAL_REWARDS = {
    "win": 1.0,
    "lose": -1.0,
    "draw": 0.0,
    "truncated": -0.5,
}

PROD_KEYS = ("g", "queue", "mask", "action", "old_lp", "adv", "ret")
ARMY_KEYS = ("g", "cand", "mask", "action", "old_lp", "adv", "ret")


def pack(d: dict[str, list], keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    res = {}
    for k in keys:
        if not d[k]:
            res[k] = np.zeros((0,), dtype=np.float32)
        elif k in ("g", "cand", "adv", "ret", "old_lp"):
            res[k] = np.asarray(d[k], dtype=np.float32)
        elif k in ("queue", "action"):
            res[k] = np.asarray(d[k], dtype=np.int64)
        elif k == "mask":
            res[k] = np.asarray(d[k], dtype=bool)
    return res


@dataclass
class Transition:
    is_army: bool
    g: np.ndarray             # (GLOBAL_DIM,)
    queue: int                # for prod
    cand: np.ndarray | None   # for army (MAX_TARGETS, CAND_DIM)
    mask: np.ndarray          # bool mask
    action: int
    log_prob: float
    val: float


class GameTrajectory:
    def __init__(self) -> None:
        self.transitions: list[Transition] = []

    def add_item(self, g: np.ndarray, queue: int, mask: np.ndarray, action: int, log_prob: float, val: float) -> None:
        self.transitions.append(Transition(
            is_army=False, g=g, queue=queue, cand=None, mask=mask,
            action=action, log_prob=log_prob, val=val,
        ))

    def add_army(self, g: np.ndarray, cand: np.ndarray, mask: np.ndarray, action: int, log_prob: float, val: float) -> None:
        self.transitions.append(Transition(
            is_army=True, g=g, queue=0, cand=cand, mask=mask,
            action=action, log_prob=log_prob, val=val,
        ))

    def finish(self, terminal_reward: float, gamma: float, gae_lambda: float) -> tuple[dict, dict]:
        T = len(self.transitions)
        if T == 0:
            return {k: [] for k in PROD_KEYS}, {k: [] for k in ARMY_KEYS}

        vals = np.array([t.val for t in self.transitions], dtype=np.float32)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            reward = terminal_reward if t == T - 1 else 0.0
            next_val = 0.0 if t == T - 1 else vals[t + 1]
            delta = reward + gamma * next_val - vals[t]
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages[t] = last_gae
            returns[t] = last_gae + vals[t]

        prod_data: dict[str, list] = {k: [] for k in PROD_KEYS}
        army_data: dict[str, list] = {k: [] for k in ARMY_KEYS}

        for t, tr in enumerate(self.transitions):
            adv, ret = advantages[t], returns[t]
            if tr.is_army:
                army_data["g"].append(tr.g)
                army_data["cand"].append(tr.cand)
                army_data["mask"].append(tr.mask)
                army_data["action"].append(tr.action)
                army_data["old_lp"].append(tr.log_prob)
                army_data["adv"].append(adv)
                army_data["ret"].append(ret)
            else:
                prod_data["g"].append(tr.g)
                prod_data["queue"].append(tr.queue)
                prod_data["mask"].append(tr.mask)
                prod_data["action"].append(tr.action)
                prod_data["old_lp"].append(tr.log_prob)
                prod_data["adv"].append(adv)
                prod_data["ret"].append(ret)

        return prod_data, army_data


def collect_game(bridge: Bridge, net: CommanderNet, match: MatchConfig, seed: int,
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


def collect_rollouts(addresses: list[str], net: CommanderNet, games: int, seed_base: int,
                     match: MatchConfig, gamma: float, gae_lambda: float,
                     recycle_every: int = 60, step_ticks: int = 500) -> tuple[dict, dict, dict]:
    net_cpu = copy.deepcopy(net).cpu().eval()
    seeds = [seed_base + i for i in range(games)]
    lock = threading.Lock()
    todo = list(seeds)

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
                    seed = todo.pop(0)

                if recycle_every and played >= recycle_every:
                    daemons.restart(addr)
                    played = 0

                meta, p_data, a_data = collect_game(bridge, net_cpu, match, seed, gamma, gae_lambda, step_ticks)
                played += 1

                if meta["error"] and "never became ready" in meta["error"]:
                    daemons.restart(addr)
                    played = 0
                    meta, p_data, a_data = collect_game(bridge, net_cpu, match, seed, gamma, gae_lambda, step_ticks)
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

    # Convert to numpy arrays
    packed_prod = pack(all_prod, PROD_KEYS)
    packed_army = pack(all_army, ARMY_KEYS)

    # Normalize advantages per head across the rollout
    if len(packed_prod.get("adv", [])) > 1:
        adv = packed_prod["adv"]
        packed_prod["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
    if len(packed_army.get("adv", [])) > 1:
        adv = packed_army["adv"]
        packed_army["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)

    valid_metas = [m for m in metas if not m["error"]]
    outcomes: dict[str, int] = {}
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
    }

    return stats, packed_prod, packed_army


def to_torch_dict(d: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in d.items():
        if len(v) == 0:
            out[k] = torch.empty(0, device=device)
            continue
        t = torch.from_numpy(v)
        if k in ("g", "cand", "adv", "ret", "old_lp"):
            t = t.float()
        elif k in ("queue", "action"):
            t = t.long()
        elif k == "mask":
            t = t.bool()
        out[k] = t.to(device)
    return out


def ppo_update(net: CommanderNet, optimizer: torch.optim.Optimizer,
               prod: dict[str, torch.Tensor], army: dict[str, torch.Tensor],
               epochs: int = 4, batch_size: int = 2048, clip_eps: float = 0.2,
               val_coef: float = 0.5, ent_coef: float = 0.01, army_weight: float = 1.0,
               max_grad_norm: float = 0.5, target_kl: float = 0.04) -> dict:
    net.train()
    device = next(net.parameters()).device
    n_prod = len(prod.get("g", []))
    n_army = len(army.get("g", []))

    if n_prod == 0 and n_army == 0:
        return {}

    steps_per_epoch = max(1, n_prod // batch_size) if n_prod > 0 else max(1, n_army // batch_size)
    army_batch = max(1, batch_size * n_army // max(1, n_prod)) if n_prod > 0 and n_army > 0 else batch_size

    metrics = {
        "pi_loss_prod": 0.0, "pi_loss_army": 0.0,
        "val_loss_prod": 0.0, "val_loss_army": 0.0,
        "ent_prod": 0.0, "ent_army": 0.0,
        "kl_prod": 0.0, "kl_army": 0.0,
        "steps": 0, "epochs_ran": 0, "early_stopped": False,
    }

    for epoch in range(epochs):
        metrics["epochs_ran"] += 1
        perm_prod = torch.randperm(n_prod, device=device) if n_prod > 0 else None

        for s in range(steps_per_epoch):
            total_loss = torch.tensor(0.0, device=device)

            # Item head (BUILD / TRAIN)
            if n_prod > 0 and perm_prod is not None:
                pi = perm_prod[s * batch_size:min(n_prod, (s + 1) * batch_size)]
                logits = net.item_logits(prod["g"][pi], prod["queue"][pi], prod["mask"][pi])
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(prod["action"][pi])
                ratio = torch.exp(new_lp - prod["old_lp"][pi])
                surr1 = ratio * prod["adv"][pi]
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * prod["adv"][pi]
                pi_loss_p = -torch.min(surr1, surr2).mean()

                v_pred = net.value(prod["g"][pi])
                v_loss_p = F.mse_loss(v_pred, prod["ret"][pi])

                ent_p = dist.entropy().mean()
                kl_p = ((ratio - 1) - torch.log(ratio + 1e-8)).mean()

                total_loss = total_loss + pi_loss_p + val_coef * v_loss_p - ent_coef * ent_p
                metrics["pi_loss_prod"] += float(pi_loss_p.item())
                metrics["val_loss_prod"] += float(v_loss_p.item())
                metrics["ent_prod"] += float(ent_p.item())
                metrics["kl_prod"] += float(kl_p.item())

            # Army head (ARMY)
            if n_army > 0:
                ai = torch.randint(0, n_army, (army_batch,), device=device)
                logits_a = net.army_logits(army["g"][ai], army["cand"][ai], army["mask"][ai])
                dist_a = Categorical(logits=logits_a)
                new_lp_a = dist_a.log_prob(army["action"][ai])
                ratio_a = torch.exp(new_lp_a - army["old_lp"][ai])
                surr1_a = ratio_a * army["adv"][ai]
                surr2_a = torch.clamp(ratio_a, 1.0 - clip_eps, 1.0 + clip_eps) * army["adv"][ai]
                pi_loss_a = -torch.min(surr1_a, surr2_a).mean()

                v_pred_a = net.value(army["g"][ai])
                v_loss_a = F.mse_loss(v_pred_a, army["ret"][ai])

                ent_a = dist_a.entropy().mean()
                kl_a = ((ratio_a - 1) - torch.log(ratio_a + 1e-8)).mean()

                total_loss = total_loss + army_weight * pi_loss_a + val_coef * v_loss_a - ent_coef * ent_a
                metrics["pi_loss_army"] += float(pi_loss_a.item())
                metrics["val_loss_army"] += float(v_loss_a.item())
                metrics["ent_army"] += float(ent_a.item())
                metrics["kl_army"] += float(kl_a.item())

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()
            metrics["steps"] += 1

        # Check early stopping KL
        cur_steps = max(1, metrics["steps"])
        avg_kl_p = metrics["kl_prod"] / cur_steps if n_prod > 0 else 0.0
        avg_kl_a = metrics["kl_army"] / cur_steps if n_army > 0 else 0.0
        mean_kl = (avg_kl_p + avg_kl_a) / (int(n_prod > 0) + int(n_army > 0))
        if mean_kl > target_kl:
            metrics["early_stopped"] = True
            break

    n = max(1, metrics["steps"])
    return {
        "pi_loss_prod": round(metrics["pi_loss_prod"] / n, 4),
        "pi_loss_army": round(metrics["pi_loss_army"] / n, 4),
        "val_loss_prod": round(metrics["val_loss_prod"] / n, 4),
        "val_loss_army": round(metrics["val_loss_army"] / n, 4),
        "ent_prod": round(metrics["ent_prod"] / n, 4),
        "ent_army": round(metrics["ent_army"] / n, 4),
        "kl_prod": round(metrics["kl_prod"] / n, 4),
        "kl_army": round(metrics["kl_army"] / n, 4),
        "epochs_ran": metrics["epochs_ran"],
        "early_stopped": metrics["early_stopped"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="runs/phase2/bc_s2/model.pt", help="starting checkpoint (BC or previous PPO)")
    ap.add_argument("--out", required=True, help="output directory for checkpoints and logs")
    ap.add_argument("--iterations", type=int, default=100, help="total PPO iterations")
    ap.add_argument("--games-per-iter", type=int, default=48, help="number of games collected per iteration")
    ap.add_argument("--bot", default="hard", help="opponent difficulty during training")
    ap.add_argument("--map", default=MatchConfig.map_name)
    ap.add_argument("--max-ticks", type=int, default=54000)
    ap.add_argument("--daemons", default=DEFAULT_DAEMONS)
    ap.add_argument("--seed-base", type=int, default=100000, help="starting seed for rollouts (never overlaps with eval)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.997, help="decision discount factor")
    ap.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    ap.add_argument("--ppo-epochs", type=int, default=4, help="optimization epochs per iteration")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--val-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--ent-decay", type=float, default=0.98)
    ap.add_argument("--ent-min", type=float, default=0.001)
    ap.add_argument("--army-weight", type=float, default=1.0)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.04)
    ap.add_argument("--eval-every", type=int, default=10, help="evaluate every N iterations")
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--eval-seeds-base", type=int, default=20000, help="selection seeds (20000-20199)")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--recycle-every", type=int, default=60)
    ap.add_argument("--tb", action=argparse.BooleanOptionalAction, default=True, help="log metrics to TensorBoard")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    daemons_list = [d.strip() for d in args.daemons.split(",") if d.strip()]
    writer = SummaryWriter(log_dir=str(out / "tb")) if (args.tb and HAS_TB) else None

    print(json.dumps({"msg": "Starting Phase 3 PPO", "init": args.init, "out": str(out),
                      "device": device, "daemons": len(daemons_list), "games_per_iter": args.games_per_iter}),
          flush=True)

    net = CommanderNet.load(args.init, device=device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cur_seed = args.seed_base
    cur_ent = args.ent_coef
    best_eval_wr = -1.0
    log = (out / "metrics.jsonl").open("a", encoding="utf-8")

    t_start = time.perf_counter()
    match = MatchConfig(map_name=args.map, bot=args.bot, max_ticks=args.max_ticks, brain="external")

    for it in range(1, args.iterations + 1):
        t_it = time.perf_counter()

        # 1. Collect rollouts across daemons
        collect_stats, prod_np, army_np = collect_rollouts(
            addresses=daemons_list,
            net=net,
            games=args.games_per_iter,
            seed_base=cur_seed,
            match=match,
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

        # Decay entropy
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

        # 4. Evaluation
        if args.eval_every > 0 and it % args.eval_every == 0:
            print(f"[Iter {it}] Running evaluation ({args.eval_games} games vs {args.bot})...", flush=True)
            eval_match = MatchConfig(map_name=args.map, bot=args.bot, max_ticks=args.max_ticks)
            eval_summary = evaluate(
                daemons=daemons_list,
                policy=f"net:{out / 'checkpoint_latest.pt'}",
                match=eval_match,
                games=args.eval_games,
                seed_base=args.eval_seeds_base,
                step_ticks=500,
                log=lambda _: None,
            )
            it_record["eval"] = {
                "games": eval_summary["games"],
                "win_rate": eval_summary["win_rate"],
                "win_rate_ci95": eval_summary["win_rate_ci95"],
                "outcomes": eval_summary["outcomes"],
            }
            if eval_summary["win_rate"] > best_eval_wr:
                best_eval_wr = eval_summary["win_rate"]
                net.save(str(out / "checkpoint_best.pt"), iteration=it, eval=eval_summary)
                it_record["new_best"] = True

        log.write(json.dumps(it_record) + "\n")
        log.flush()
        print(json.dumps(it_record), flush=True)

        if writer is not None:
            writer.add_scalar("rollout/win_rate", collect_stats["win_rate"], it)
            writer.add_scalar("rollout/mean_ticks", collect_stats["mean_ticks"], it)
            writer.add_scalar("rollout/mean_decisions", collect_stats["mean_decisions"], it)
            writer.add_scalar("rollout/collect_s", collect_stats["collect_s"], it)
            writer.add_scalar("rollout/decisions_prod", collect_stats["n_prod"], it)
            writer.add_scalar("rollout/decisions_army", collect_stats["n_army"], it)

            writer.add_scalar("loss/policy_prod", update_metrics["pi_loss_prod"], it)
            writer.add_scalar("loss/policy_army", update_metrics["pi_loss_army"], it)
            writer.add_scalar("loss/value_prod", update_metrics["val_loss_prod"], it)
            writer.add_scalar("loss/value_army", update_metrics["val_loss_army"], it)

            writer.add_scalar("policy/entropy_prod", update_metrics["ent_prod"], it)
            writer.add_scalar("policy/entropy_army", update_metrics["ent_army"], it)
            writer.add_scalar("policy/kl_prod", update_metrics["kl_prod"], it)
            writer.add_scalar("policy/kl_army", update_metrics["kl_army"], it)
            writer.add_scalar("policy/ent_coef", cur_ent, it)
            writer.add_scalar("policy/early_stopped", float(update_metrics["early_stopped"]), it)

            if "eval" in it_record:
                writer.add_scalar("eval/win_rate", eval_summary["win_rate"], it)
                writer.add_scalar("eval/ci95_low", eval_summary["win_rate_ci95"][0], it)
                writer.add_scalar("eval/ci95_high", eval_summary["win_rate_ci95"][1], it)
                writer.add_scalar("eval/mean_ticks", eval_summary["mean_ticks"], it)

            writer.flush()

    log.close()
    if writer is not None:
        writer.close()
    print(f"PPO complete in {time.perf_counter() - t_start:.0f}s. Best eval win rate: {best_eval_wr}", flush=True)


if __name__ == "__main__":
    main()
