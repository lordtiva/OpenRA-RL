r"""Fase 4.3: Hybrid Self-Play + PFSP League Training for Commander v4.

    python -m ra_rl.selfplay --init runs/v4/league_warmstart/checkpoint_best.pt \
        --ref runs/v4/bc_v4/model.pt --out runs/v4/selfplay

Architecture (OpenAI Five / AlphaStar hybrid league):
  - 30% Live Mirror Self-Play (`pi_theta vs pi_theta` via `SelfPlaySession`):
    collects on-policy SMDP trajectories from BOTH `Multi1` and `Multi0` in lock-step
    (2x sample efficiency per engine tick, exact zero-sum PBRS symmetry).
  - 20% Hall-of-Fame Historical Snapshots (`pi_theta vs pi_snap` via `SelfPlaySession`):
    sampled via PFSP (`(1 - WR)^1.5`) with automatic `Multi1`/`Multi0` slot swapping.
  - 35% C# Bot Exploiter Pool (`hard`, `brutal`, `turtle` via `GameSession`):
    weighted via PFSP to prevent cyclic rock-paper-scissors forgetting.
  - 15% Anchor Pool (`pi_theta vs bc_v4` via `SelfPlaySession`, slot-swapped):
    prevents drift against the supervised bilateral baseline.
  - Stratified advantage normalization per pool + dual cosine KL decay
    (`kl_prod: 0.04 -> 0.005`, `kl_army: 0.04 -> 0.002`) + live Elo & faction telemetry.
"""

import argparse
import copy
import json
import math
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import grpc
import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    SummaryWriter = None
    HAS_TB = False

from ra_env import Bridge, MatchConfig, SelfPlayConfig, SelfPlaySession, daemons
from ra_rl.eval import DEFAULT_DAEMONS, evaluate, evaluate_selfplay, is_win
from ra_rl.features import Features, opponent_id
from ra_rl.league import assign_opponents, compute_harmonic_score, pfsp_probabilities
from ra_rl.model import CommanderNet
from ra_rl.policies import make_policy
from ra_rl.ppo import (
    ARMY_KEYS,
    PROD_KEYS,
    STEP_KEYS,
    TERMINAL_REWARDS,
    GameTrajectory,
    collect_game,
    compute_behavior_stats,
    normalize_advantages,
    pack,
    policy_spec,
    ppo_update,
    step_trajectory,
    to_torch_dict,
)


@dataclass(frozen=True)
class RolloutJob:
    seed: int
    pool: str           # "mirror" | "snapshot" | "bots" | "anchor"
    opponent: str       # "selfplay" | snapshot_id | bot_name | "bc"
    opponent_spec: str  # "" | policy spec for snapshot/anchor
    learner_is_p1: bool
    map_name: str


def elo_update(r_a: float, r_b: float, score_a: float, k: float = 16.0) -> tuple[float, float]:
    """Standard logistic Elo update for a single match outcome score_a in {0.0, 0.5, 1.0}."""
    exp_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
    delta = float(k) * (float(score_a) - exp_a)
    return r_a + delta, r_b - delta


def schedule_kl_coefs(
    it: int,
    total_iters: int,
    kl_prod_init: float = 0.04,
    kl_prod_min: float = 0.005,
    kl_army_init: float = 0.04,
    kl_army_min: float = 0.002,
) -> tuple[float, float]:
    """Cosine decay from init -> min, releasing tactical exploration faster than macro."""
    if total_iters <= 1:
        return kl_prod_min, kl_army_min
    progress = min(1.0, max(0.0, float(it - 1) / float(total_iters - 1)))
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    kl_p = kl_prod_min + (kl_prod_init - kl_prod_min) * cos_factor
    kl_a = kl_army_min + (kl_army_init - kl_army_min) * cos_factor
    return round(kl_p, 5), round(kl_a, 5)


def plan_hybrid_jobs(
    games: int,
    seed_base: int,
    pool_shares: tuple[float, float, float, float],  # (mirror, snapshot, bots, anchor)
    bots: list[str],
    bot_ema_wrs: dict[str, float],
    snapshots: dict[str, Path],
    snap_ema_wrs: dict[str, float],
    anchor_spec: str,
    maps: list[str],
    snap_temp: float = 0.5,
    pfsp_power: float = 1.5,
    rng: np.random.Generator | None = None,
) -> list[RolloutJob]:
    rng = rng or np.random.default_rng(seed_base)
    w_mirror, w_snap, w_bots, w_anchor = pool_shares
    if not snapshots:
        w_mirror += w_snap
        w_snap = 0.0
    if not bots:
        w_mirror += w_bots
        w_bots = 0.0

    probs = np.array([w_mirror, w_snap, w_bots, w_anchor], dtype=np.float64)
    probs /= probs.sum()
    counts = rng.multinomial(games, probs)
    n_mirror, n_snap, n_bots, n_anchor = (int(x) for x in counts)

    bot_list: list[str] = []
    if n_bots > 0:
        b_probs = pfsp_probabilities(bot_ema_wrs, bots, power=pfsp_power, min_prob=0.18)
        bot_list = assign_opponents(bots, b_probs, n_bots, rng)

    snap_ids = sorted(snapshots.keys())
    snap_list: list[str] = []
    if n_snap > 0 and snap_ids:
        s_probs = pfsp_probabilities(snap_ema_wrs, snap_ids, power=pfsp_power, min_prob=0.05)
        snap_list = assign_opponents(snap_ids, s_probs, n_snap, rng)

    raw_jobs: list[tuple[str, str, str]] = []
    for _ in range(n_mirror):
        raw_jobs.append(("mirror", "selfplay", ""))
    for sid in snap_list:
        spec = policy_spec(snapshots[sid], snap_temp)
        raw_jobs.append(("snapshot", sid, spec))
    for b_name in bot_list:
        raw_jobs.append(("bots", b_name, ""))
    for _ in range(n_anchor):
        raw_jobs.append(("anchor", "bc", anchor_spec))

    rng.shuffle(raw_jobs)
    jobs: list[RolloutJob] = []
    for i, (pool, opp, spec) in enumerate(raw_jobs[:games]):
        seed = seed_base + i
        map_name = maps[seed % len(maps)]
        learner_is_p1 = True if pool in ("mirror", "bots") else bool(i % 2 == 0)
        jobs.append(RolloutJob(
            seed=seed,
            pool=pool,
            opponent=opp,
            opponent_spec=spec,
            learner_is_p1=learner_is_p1,
            map_name=map_name,
        ))
    return jobs


def collect_mirror_game(
    bridge: Bridge,
    net: CommanderNet,
    cfg: SelfPlayConfig,
    seed: int,
    gamma: float = 0.995,
    gae_lambda: float = 0.98,
    step_ticks: int = 500,
    gamma_ticks: float = 250.0,
    pbrs_coef: float = 0.15,
) -> tuple[dict, dict, dict, dict]:
    """Live mirror `pi_theta vs pi_theta`: collects on-policy trajectories from BOTH players."""
    session = SelfPlaySession(bridge, cfg)
    feats1, feats0 = Features(), Features()
    traj1, traj0 = GameTrajectory(), GameTrajectory()
    b_idx = opponent_id("selfplay")
    t0 = time.perf_counter()
    dec1 = dec0 = 0
    error = ""
    out1 = out0 = "error"
    fac1 = fac0 = ""

    try:
        dual = session.reset(seed)
        fac1 = dual.p1.obs.player_faction
        fac0 = dual.p0.obs.player_faction
        while not (dual.done or dual.truncated):
            d1, n1 = step_trajectory(dual.p1, feats1, traj1, net, "selfplay", b_idx)
            d0, n0 = step_trajectory(dual.p0, feats0, traj0, net, "selfplay", b_idx)
            dec1 += n1
            dec0 += n0
            dual = session.step(d1, d0, step_ticks)
        out1 = dual.p1.result or "truncated"
        out0 = dual.p0.result or "truncated"
    except (grpc.RpcError, RuntimeError) as e:
        error = repr(e)
    finally:
        session.close()

    prod_data: dict[str, list] = {k: [] for k in PROD_KEYS}
    army_data: dict[str, list] = {k: [] for k in ARMY_KEYS}
    step_data: dict[str, list] = {k: [] for k in STEP_KEYS}

    if not error:
        for tr, out_side in ((traj1, out1), (traj0, out0)):
            r_term = TERMINAL_REWARDS.get(out_side, -0.5)
            p_d, a_d, s_d = tr.finish(
                r_term,
                end_tick=session.tick,
                gamma=gamma,
                gae_lambda=gae_lambda,
                gamma_ticks=gamma_ticks,
                pbrs_coef=pbrs_coef,
            )
            for k in PROD_KEYS:
                prod_data[k].extend(p_d[k])
            for k in ARMY_KEYS:
                army_data[k].extend(a_d[k])
            for k in STEP_KEYS:
                step_data[k].extend(s_d[k])

    meta = {
        "seed": seed,
        "pool": "mirror",
        "bot": "selfplay",
        "opponent": "selfplay",
        "faction": fac1,
        "peer_faction": fac0,
        "outcome": out1,
        "peer_outcome": out0,
        "ticks": session.tick,
        "decisions": dec1 + dec0,
        "steps_joint": len(traj1.steps) + len(traj0.steps),
        "wall_s": round(time.perf_counter() - t0, 1),
        "error": error,
    }
    return meta, prod_data, army_data, step_data


def collect_vs_policy_game(
    bridge: Bridge,
    net: CommanderNet,
    opponent_spec: str,
    pool_name: str,
    opponent_label: str,
    cfg: SelfPlayConfig,
    seed: int,
    learner_is_p1: bool = True,
    gamma: float = 0.995,
    gae_lambda: float = 0.98,
    step_ticks: int = 500,
    gamma_ticks: float = 250.0,
    pbrs_coef: float = 0.15,
) -> tuple[dict, dict, dict, dict]:
    """Learner `pi_theta` vs frozen policy (`snapshot` or `bc`/`scripted`) via SelfPlaySession."""
    opp_pol = make_policy(opponent_spec)
    opp_pol.reset(seed * 2 + 7)
    sp_cfg = SelfPlayConfig(
        map_name=cfg.map_name,
        player_faction=cfg.player_faction,
        enemy_faction=cfg.enemy_faction,
        max_ticks=cfg.max_ticks,
        brain="external" if learner_is_p1 else opp_pol.brain,
        peer_brain=opp_pol.brain if learner_is_p1 else "external",
    )
    session = SelfPlaySession(bridge, sp_cfg)
    feats = Features()
    traj = GameTrajectory()
    critic_tag = "snapshot" if pool_name == "snapshot" else "bc"
    b_idx = opponent_id(critic_tag)
    t0 = time.perf_counter()
    decisions = 0
    error = ""
    outcome = "error"
    faction = ""

    try:
        dual = session.reset(seed)
        l_step = dual.p1 if learner_is_p1 else dual.p0
        faction = l_step.obs.player_faction
        while not (dual.done or dual.truncated):
            if learner_is_p1:
                d1, n_dec = step_trajectory(dual.p1, feats, traj, net, critic_tag, b_idx)
                d0 = opp_pol.decide(dual.p0)
            else:
                d1 = opp_pol.decide(dual.p1)
                d0, n_dec = step_trajectory(dual.p0, feats, traj, net, critic_tag, b_idx)
            decisions += n_dec
            dual = session.step(d1, d0, step_ticks)
        l_final = dual.p1 if learner_is_p1 else dual.p0
        outcome = l_final.result or "truncated"
    except (grpc.RpcError, RuntimeError) as e:
        error = repr(e)
    finally:
        session.close()

    if not error:
        r_term = TERMINAL_REWARDS.get(outcome, -0.5)
        prod_data, army_data, step_data = traj.finish(
            r_term,
            end_tick=session.tick,
            gamma=gamma,
            gae_lambda=gae_lambda,
            gamma_ticks=gamma_ticks,
            pbrs_coef=pbrs_coef,
        )
    else:
        prod_data = {k: [] for k in PROD_KEYS}
        army_data = {k: [] for k in ARMY_KEYS}
        step_data = {k: [] for k in STEP_KEYS}

    meta = {
        "seed": seed,
        "pool": pool_name,
        "bot": critic_tag,
        "opponent": opponent_label,
        "slot": "Multi1" if learner_is_p1 else "Multi0",
        "faction": faction,
        "outcome": outcome,
        "ticks": session.tick,
        "decisions": decisions,
        "steps_joint": len(traj.steps),
        "wall_s": round(time.perf_counter() - t0, 1),
        "error": error,
    }
    return meta, prod_data, army_data, step_data


def _run_job(
    bridge: Bridge,
    net_cpu: CommanderNet,
    job: RolloutJob,
    player_faction: str,
    enemy_faction: str,
    max_ticks: int,
    gamma: float,
    gae_lambda: float,
    step_ticks: int,
    gamma_ticks: float,
    pbrs_coef: float,
) -> tuple[dict, dict, dict, dict]:
    if job.pool == "mirror":
        cfg = SelfPlayConfig(
            map_name=job.map_name,
            player_faction=player_faction,
            enemy_faction=enemy_faction,
            max_ticks=max_ticks,
        )
        return collect_mirror_game(
            bridge, net_cpu, cfg, job.seed, gamma, gae_lambda, step_ticks, gamma_ticks, pbrs_coef
        )
    if job.pool in ("snapshot", "anchor"):
        cfg = SelfPlayConfig(
            map_name=job.map_name,
            player_faction=player_faction,
            enemy_faction=enemy_faction,
            max_ticks=max_ticks,
        )
        return collect_vs_policy_game(
            bridge, net_cpu, job.opponent_spec, job.pool, job.opponent, cfg, job.seed,
            learner_is_p1=job.learner_is_p1, gamma=gamma, gae_lambda=gae_lambda,
            step_ticks=step_ticks, gamma_ticks=gamma_ticks, pbrs_coef=pbrs_coef,
        )
    match = MatchConfig(
        map_name=job.map_name,
        bot=job.opponent,
        player_faction=player_faction,
        enemy_faction=enemy_faction,
        max_ticks=max_ticks,
        brain="external",
    )
    meta, p_d, a_d, s_d = collect_game(
        bridge, net_cpu, match, job.seed, gamma, gae_lambda, step_ticks, gamma_ticks, pbrs_coef
    )
    meta["pool"] = "bots"
    meta["opponent"] = job.opponent
    return meta, p_d, a_d, s_d


def collect_hybrid_rollouts(
    addresses: list[str],
    net: CommanderNet,
    jobs: list[RolloutJob],
    player_faction: str = "Random",
    enemy_faction: str = "Random",
    max_ticks: int = 54000,
    gamma: float = 0.995,
    gae_lambda: float = 0.98,
    recycle_every: int = 45,
    step_ticks: int = 500,
    gamma_ticks: float = 250.0,
    pbrs_coef: float = 0.15,
) -> tuple[dict, dict, dict, dict, list[dict]]:
    net_cpu = copy.deepcopy(net).cpu().eval()
    lock = threading.Lock()
    todo = list(jobs)

    all_prod: dict[str, list] = {k: [] for k in PROD_KEYS}
    all_army: dict[str, list] = {k: [] for k in ARMY_KEYS}
    all_step: dict[str, list] = {k: [] for k in STEP_KEYS}
    metas: list[dict] = []

    def worker(addr: str) -> None:
        bridge = Bridge(addr)
        played = 0
        try:
            while True:
                with lock:
                    if not todo:
                        return
                    job = todo.pop(0)

                if recycle_every and played >= recycle_every:
                    bridge.close()
                    daemons.restart(addr)
                    bridge = Bridge(addr)
                    played = 0

                meta, p_data, a_data, s_data = _run_job(
                    bridge, net_cpu, job, player_faction, enemy_faction, max_ticks,
                    gamma, gae_lambda, step_ticks, gamma_ticks, pbrs_coef,
                )
                played += 1

                if meta["error"]:
                    bridge.close()
                    daemons.restart(addr)
                    bridge = Bridge(addr)
                    played = 0
                    meta, p_data, a_data, s_data = _run_job(
                        bridge, net_cpu, job, player_faction, enemy_faction, max_ticks,
                        gamma, gae_lambda, step_ticks, gamma_ticks, pbrs_coef,
                    )
                    played += 1

                with lock:
                    metas.append(meta)
                    if not meta["error"]:
                        for k in PROD_KEYS:
                            all_prod[k].extend(p_data[k])
                        for k in ARMY_KEYS:
                            all_army[k].extend(a_data[k])
                        for k in STEP_KEYS:
                            all_step[k].extend(s_data[k])
        finally:
            bridge.close()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(addresses)) as pool:
        list(pool.map(worker, addresses))

    packed_prod = pack(all_prod, PROD_KEYS)
    packed_army = pack(all_army, ARMY_KEYS)
    packed_step = pack(all_step, STEP_KEYS)

    # Stratified advantage normalization per pool/archetype (`bot_idx`).
    normalize_advantages(packed_prod, per_bot=True)
    normalize_advantages(packed_army, per_bot=True)
    beh = compute_behavior_stats(packed_prod, packed_army, packed_step)

    valid_metas = [m for m in metas if not m["error"]]
    pool_stats: dict[str, dict] = {}
    for pname in ("mirror", "snapshot", "bots", "anchor"):
        sub = [m for m in valid_metas if m.get("pool") == pname]
        if not sub:
            continue
        if pname == "mirror":
            p1_wins = sum(1 for m in sub if is_win(m["outcome"]))
            p0_wins = sum(1 for m in sub if is_win(m.get("peer_outcome", "")))
            truncs = sum(1 for m in sub if m["outcome"] == "truncated")
            pool_stats[pname] = {
                "games": len(sub),
                "p1_win_rate": round(p1_wins / len(sub), 3),
                "p0_win_rate": round(p0_wins / len(sub), 3),
                "decisive_rate": round((p1_wins + p0_wins) / len(sub), 3),
                "trunc_rate": round(truncs / len(sub), 3),
            }
        else:
            w = sum(1 for m in sub if is_win(m["outcome"]))
            pool_stats[pname] = {
                "games": len(sub),
                "wins": w,
                "win_rate": round(w / len(sub), 3),
            }

    faction_wins: dict[str, list[int]] = {}
    for m in valid_metas:
        if m.get("pool") == "mirror":
            for fk, ok in (("faction", "outcome"), ("peer_faction", "peer_outcome")):
                fac = m.get(fk) or ""
                if fac:
                    faction_wins.setdefault(fac, [0, 0])
                    faction_wins[fac][0] += 1
                    faction_wins[fac][1] += int(is_win(m.get(ok, "")))
        else:
            fac = m.get("faction") or ""
            if fac:
                faction_wins.setdefault(fac, [0, 0])
                faction_wins[fac][0] += 1
                faction_wins[fac][1] += int(is_win(m["outcome"]))

    faction_stats = {
        f: {"games": g_cnt, "win_rate": round(w_cnt / max(1, g_cnt), 3)}
        for f, (g_cnt, w_cnt) in sorted(faction_wins.items())
    }

    non_mirror = [m for m in valid_metas if m.get("pool") != "mirror"]
    ext_wr = round(
        sum(1 for m in non_mirror if is_win(m["outcome"])) / max(1, len(non_mirror)), 3
    )

    stats = {
        "games_collected": len(valid_metas),
        "games_failed": len(metas) - len(valid_metas),
        "win_rate_vs_external": ext_wr,
        "pool_stats": pool_stats,
        "faction_stats": faction_stats,
        "mean_ticks": round(sum(m["ticks"] for m in valid_metas) / max(1, len(valid_metas))),
        "mean_decisions": round(sum(m["decisions"] for m in valid_metas) / max(1, len(valid_metas)), 1),
        "n_prod": len(packed_prod.get("g", [])),
        "n_army": len(packed_army.get("g", [])),
        "n_steps": len(packed_step.get("g", [])),
        "collect_s": round(time.perf_counter() - t0, 1),
        **beh,
    }
    return stats, packed_prod, packed_army, packed_step, valid_metas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="runs/v4/league_warmstart/checkpoint_best.pt", help="starting checkpoint")
    ap.add_argument("--ref", default="runs/v4/bc_v4/model.pt", help="frozen BC anchor checkpoint")
    ap.add_argument("--out", required=True, help="output directory for checkpoints, snapshots, and logs")
    ap.add_argument("--iterations", type=int, default=80)
    ap.add_argument("--games-per-iter", type=int, default=48)
    ap.add_argument("--pool-shares", default="0.30,0.20,0.35,0.15",
                    help="mirror,snapshot,bots,anchor rollout shares")
    ap.add_argument("--opponents", default="hard,brutal,turtle", help="C# bot exploiters")
    ap.add_argument("--eval-bots", default="hard,brutal,turtle")
    ap.add_argument("--snapshot-every", type=int, default=8, help="add learner to Hall of Fame every N iterations")
    ap.add_argument("--max-snapshots", type=int, default=24)
    ap.add_argument("--snap-temp", type=float, default=0.5, help="sampling temperature for snapshot/anchor opponents")
    ap.add_argument("--pfsp-power", type=float, default=1.5)
    ap.add_argument("--player-faction", default="Random")
    ap.add_argument("--enemy-faction", default="Random")
    ap.add_argument("--maps", default="singles.oramap", help="comma-separated map pool")
    ap.add_argument("--max-ticks", type=int, default=54000)
    ap.add_argument("--daemons", default=DEFAULT_DAEMONS)
    ap.add_argument("--seed-base", type=int, default=600000)
    ap.add_argument("--lr", type=float, default=3.5e-5)
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--critic-warmup-iters", type=int, default=4)
    ap.add_argument("--reset-critic", action="store_true")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gamma-ticks", type=float, default=250.0)
    ap.add_argument("--gae-lambda", type=float, default=0.98)
    ap.add_argument("--pbrs-coef", type=float, default=0.15)
    ap.add_argument("--kl-prod-init", type=float, default=0.04)
    ap.add_argument("--kl-prod-min", type=float, default=0.005)
    ap.add_argument("--kl-army-init", type=float, default=0.04)
    ap.add_argument("--kl-army-min", type=float, default=0.002)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.004)
    ap.add_argument("--ent-decay", type=float, default=0.996)
    ap.add_argument("--ent-min", type=float, default=0.0015)
    ap.add_argument("--army-weight", type=float, default=1.0)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.025)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games-per-bot", type=int, default=40)
    ap.add_argument("--eval-vs-bc-games", type=int, default=40)
    ap.add_argument("--eval-seeds-base", type=int, default=20000)
    ap.add_argument("--eval-temp", type=float, default=0.0)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--recycle-every", type=int, default=45)
    ap.add_argument("--tb", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    out = Path(args.out)
    snap_dir = out / "snapshots"
    out.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    daemons_list = [d.strip() for d in args.daemons.split(",") if d.strip()]
    bots = [o.strip() for o in args.opponents.split(",") if o.strip()]
    eval_bots = [o.strip() for o in args.eval_bots.split(",") if o.strip()]
    maps = [m.strip() for m in args.maps.split(",") if m.strip()]
    shares_raw = [float(x.strip()) for x in args.pool_shares.split(",")]
    if len(shares_raw) != 4:
        raise SystemExit("--pool-shares must have 4 comma-separated floats: mirror,snapshot,bots,anchor")
    pool_shares = (shares_raw[0], shares_raw[1], shares_raw[2], shares_raw[3])

    ref_path = args.ref if (args.ref and Path(args.ref).exists()) else args.init
    anchor_spec = policy_spec(Path(ref_path), args.snap_temp)
    writer = SummaryWriter(log_dir=str(out / "tb")) if (args.tb and HAS_TB) else None

    net = CommanderNet.load(args.init, device=device)
    if args.reset_critic:
        net.reset_critic()
    ref_net = CommanderNet.load(ref_path, device=device).eval()
    ref_net.requires_grad_(False)

    # Seed Hall of Fame with initial checkpoint
    snapshots: dict[str, Path] = {}
    snap_ema_wrs: dict[str, float] = {}
    snap_elos: dict[str, float] = {"bc": 1200.0}
    init_snap = snap_dir / "snap_0000_init.pt"
    if not init_snap.exists():
        shutil.copy2(args.init, init_snap)
    for existing in sorted(snap_dir.glob("snap_*.pt")):
        sid = existing.stem
        snapshots[sid] = existing
        snap_ema_wrs[sid] = 0.5
        snap_elos[sid] = 1250.0 if sid == "snap_0000_init" else 1250.0

    learner_elo = 1250.0
    bot_ema_wrs: dict[str, float] = {b: 0.5 for b in bots}

    actor_opt = torch.optim.AdamW(list(net.actor_parameters()), lr=args.lr, weight_decay=args.weight_decay)
    critic_opt = torch.optim.AdamW(list(net.critic_parameters()), lr=args.critic_lr, weight_decay=args.weight_decay)

    cur_seed = args.seed_base
    cur_ent = args.ent_coef
    best_composite = -1.0
    rng = np.random.default_rng(args.seed_base)
    log = (out / "metrics.jsonl").open("a", encoding="utf-8")

    print(json.dumps({
        "msg": "Starting Fase 4.3 Hybrid Self-Play League",
        "init": args.init, "ref": ref_path, "out": str(out),
        "device": device, "daemons": len(daemons_list),
        "pool_shares": pool_shares, "bots": bots, "maps": maps,
        "player_faction": args.player_faction, "games_per_iter": args.games_per_iter,
    }), flush=True)

    t_start = time.perf_counter()

    for it in range(1, args.iterations + 1):
        t_it = time.perf_counter()
        in_warmup = it <= args.critic_warmup_iters
        kl_p, kl_a = schedule_kl_coefs(
            it, args.iterations,
            args.kl_prod_init, args.kl_prod_min,
            args.kl_army_init, args.kl_army_min,
        )

        jobs = plan_hybrid_jobs(
            games=args.games_per_iter,
            seed_base=cur_seed,
            pool_shares=pool_shares,
            bots=bots,
            bot_ema_wrs=bot_ema_wrs,
            snapshots=snapshots,
            snap_ema_wrs=snap_ema_wrs,
            anchor_spec=anchor_spec,
            maps=maps,
            snap_temp=args.snap_temp,
            pfsp_power=args.pfsp_power,
            rng=rng,
        )
        cur_seed += args.games_per_iter

        collect_stats, prod_np, army_np, step_np, valid_metas = collect_hybrid_rollouts(
            addresses=daemons_list,
            net=net,
            jobs=jobs,
            player_faction=args.player_faction,
            enemy_faction=args.enemy_faction,
            max_ticks=args.max_ticks,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            recycle_every=args.recycle_every,
            gamma_ticks=args.gamma_ticks,
            pbrs_coef=args.pbrs_coef,
        )

        # Update PFSP EMAs & live Elo against Hall-of-Fame snapshots + frozen BC anchor
        for b_name in bots:
            b_rows = [m for m in valid_metas if m.get("pool") == "bots" and m.get("opponent") == b_name]
            if b_rows:
                wr = sum(1 for m in b_rows if is_win(m["outcome"])) / len(b_rows)
                bot_ema_wrs[b_name] = 0.7 * bot_ema_wrs[b_name] + 0.3 * wr

        for m in valid_metas:
            pool = m.get("pool")
            opp = m.get("opponent", "")
            if pool in ("snapshot", "anchor"):
                s_val = 1.0 if is_win(m["outcome"]) else (0.5 if m["outcome"] in ("draw", "truncated") else 0.0)
                opp_elo = snap_elos.get(opp, 1200.0)
                learner_elo, new_opp_elo = elo_update(learner_elo, opp_elo, s_val, k=12.0)
                if pool == "snapshot" and opp in snap_elos:
                    snap_elos[opp] = new_opp_elo  # keep "bc" anchored at 1200.0
                    snap_ema_wrs[opp] = 0.8 * snap_ema_wrs.get(opp, 0.5) + 0.2 * float(is_win(m["outcome"]))

        prod_t = to_torch_dict(prod_np, device)
        army_t = to_torch_dict(army_np, device)
        step_t = to_torch_dict(step_np, device)

        update_metrics = ppo_update(
            net=net,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            prod=prod_t,
            army=army_t,
            steps=step_t,
            ref_net=ref_net,
            critic_only=in_warmup,
            epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            clip_eps=args.clip_eps,
            ent_coef=cur_ent,
            kl_coef=kl_p,
            kl_coef_army=kl_a,
            army_weight=args.army_weight,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
        )

        if not in_warmup:
            cur_ent = max(args.ent_min, cur_ent * args.ent_decay)

        net.save(str(out / "checkpoint_latest.pt"), iteration=it, elo=round(learner_elo, 1), stats=collect_stats)
        if args.save_every > 0 and it % args.save_every == 0:
            net.save(str(out / f"checkpoint_iter_{it:04d}.pt"), iteration=it, elo=round(learner_elo, 1))

        if args.snapshot_every > 0 and it % args.snapshot_every == 0 and not in_warmup:
            sid = f"snap_{it:04d}"
            spath = snap_dir / f"{sid}.pt"
            net.save(str(spath), iteration=it, elo=round(learner_elo, 1))
            snapshots[sid] = spath
            snap_ema_wrs[sid] = 0.5
            snap_elos[sid] = learner_elo
            if len(snapshots) > args.max_snapshots:
                removable = [k for k in sorted(snapshots.keys()) if k != "snap_0000_init"]
                if removable:
                    drop_id = removable[0]
                    snapshots.pop(drop_id, None)
                    snap_ema_wrs.pop(drop_id, None)

        it_record = {
            "iteration": it,
            "elo": round(learner_elo, 1),
            "hall_of_fame_size": len(snapshots),
            "kl_coef_prod": kl_p,
            "kl_coef_army": kl_a,
            "ent_coef": round(cur_ent, 5),
            "bot_ema_wrs": {k: round(v, 3) for k, v in bot_ema_wrs.items()},
            "it_s": round(time.perf_counter() - t_it, 1),
            "total_s": round(time.perf_counter() - t_start, 1),
            **collect_stats,
            **update_metrics,
        }

        if args.eval_every > 0 and it % args.eval_every == 0:
            print(f"[Iter {it}] Running Hybrid Self-Play Evaluation (bots + head-to-head vs BC)...", flush=True)
            cand_spec = policy_spec(out / "checkpoint_latest.pt", args.eval_temp)
            eval_results: dict[str, dict] = {}
            total_eval_errors = 0

            for test_bot in eval_bots:
                eval_match = MatchConfig(
                    map_name=maps[0],
                    bot=test_bot,
                    player_faction=args.player_faction,
                    enemy_faction=args.enemy_faction,
                    max_ticks=args.max_ticks,
                )
                ev = evaluate(
                    daemons=daemons_list,
                    policy=cand_spec,
                    match=eval_match,
                    games=args.eval_games_per_bot,
                    seed_base=args.eval_seeds_base,
                    step_ticks=500,
                    log=lambda _: None,
                    recycle_every=args.recycle_every,
                    pre_recycle=True,
                )
                total_eval_errors += ev["outcomes"].get("error", 0)
                eval_results[test_bot] = {
                    "win_rate": ev["win_rate"],
                    "ci95": ev["win_rate_ci95"],
                    "outcomes": ev["outcomes"],
                }

            sp_eval_cfg = SelfPlayConfig(
                map_name=maps[0],
                player_faction=args.player_faction,
                enemy_faction=args.enemy_faction,
                max_ticks=args.max_ticks,
            )
            ev_bc = evaluate_selfplay(
                daemons=daemons_list,
                policy_a=cand_spec,
                policy_b=f"net:{ref_path}",
                cfg=sp_eval_cfg,
                games=args.eval_vs_bc_games,
                seed_base=args.eval_seeds_base,
                step_ticks=500,
                swap_slots=True,
                log=lambda _: None,
                recycle_every=args.recycle_every,
                pre_recycle=True,
            )
            total_eval_errors += ev_bc["outcomes"].get("error", 0)
            eval_results["vs_bc"] = {
                "win_rate": ev_bc["win_rate"],
                "ci95": ev_bc["win_rate_ci95"],
                "outcomes": ev_bc["outcomes"],
            }

            bot_harmonic = compute_harmonic_score(*(eval_results[b]["win_rate"] for b in eval_bots))
            composite = compute_harmonic_score(
                *(eval_results[b]["win_rate"] for b in eval_bots),
                eval_results["vs_bc"]["win_rate"],
            )
            it_record["eval"] = {
                **eval_results,
                "bot_harmonic_score": bot_harmonic,
                "composite_score": composite,
                "total_errors": total_eval_errors,
            }

            max_err = max(4, (len(eval_bots) * args.eval_games_per_bot + args.eval_vs_bc_games) // 10)
            if composite > best_composite and total_eval_errors <= max_err:
                best_composite = composite
                net.save(str(out / "checkpoint_best.pt"), iteration=it, elo=round(learner_elo, 1), eval=it_record["eval"])
                it_record["new_best"] = True

        log.write(json.dumps(it_record) + "\n")
        log.flush()
        print(json.dumps(it_record), flush=True)

        if writer is not None:
            writer.add_scalar("selfplay/elo", learner_elo, it)
            writer.add_scalar("selfplay/kl_coef_prod", kl_p, it)
            writer.add_scalar("selfplay/kl_coef_army", kl_a, it)
            writer.add_scalar("rollout/win_rate_vs_external", collect_stats["win_rate_vs_external"], it)
            writer.add_scalar("rollout/explained_var", collect_stats["explained_var"], it)
            writer.add_scalar("behavior/army_allin_pct", collect_stats["beh_army_allin_pct"], it)
            writer.add_scalar("behavior/harass_atk_pct", collect_stats["beh_harass_atk_pct"], it)
            if "eval" in it_record:
                writer.add_scalar("eval/bot_harmonic", it_record["eval"]["bot_harmonic_score"], it)
                writer.add_scalar("eval/vs_bc_win_rate", it_record["eval"]["vs_bc"]["win_rate"], it)
                writer.add_scalar("eval/composite_score", it_record["eval"]["composite_score"], it)
            writer.flush()

    log.close()
    if writer is not None:
        writer.close()
    print(f"Hybrid self-play complete in {time.perf_counter() - t_start:.0f}s. Best composite: {best_composite}", flush=True)


if __name__ == "__main__":
    main()
