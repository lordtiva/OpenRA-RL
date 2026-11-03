"""Standard & head-to-head evaluation: N games with fixed seeds, one session per daemon.

    # Against scripted C# bots:
    python -m ra_rl.eval --policy scripted --bot hard --games 100 --player-faction Random

    # Head-to-head RL-vs-RL (slot-swapped for zero spawn bias):
    python -m ra_rl.eval --policy net:runs/v4/selfplay/checkpoint_best.pt \
        --opponent-policy net:runs/v4/bc_v4/model.pt --games 60

Win rate counts real wins ("win" and "win_early"); truncated games are not wins.
Daemon containers are automatically recycled every --recycle-every games and
retried once on transient container OOM / poisoned sessions so evaluations are clean.
"""

import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import grpc

from ra_env import Bridge, GameSession, MatchConfig, SelfPlayConfig, SelfPlaySession, daemons as daemon_ctl
from ra_rl.policies import make_policy

DEFAULT_DAEMONS = ",".join(f"localhost:{p}" for p in range(9999, 10005))


def is_win(outcome: str) -> bool:
    return outcome in ("win", "win_early")


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win rate."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _faction_breakdown(results: list[dict]) -> dict[str, dict]:
    by_fac: dict[str, list[dict]] = {}
    for r in results:
        if r.get("outcome") == "error":
            continue
        fac = r.get("faction") or "unknown"
        by_fac.setdefault(fac, []).append(r)
    out: dict[str, dict] = {}
    for fac, rows in sorted(by_fac.items()):
        w = sum(1 for r in rows if is_win(r["outcome"]))
        out[fac] = {"games": len(rows), "wins": w, "win_rate": round(w / max(1, len(rows)), 3)}
    return out


def play_game(bridge: Bridge, policy_name: str, match: MatchConfig, seed: int, step_ticks: int) -> dict:
    policy = make_policy(policy_name)
    session = GameSession(bridge, MatchConfig(**{**match.__dict__, "brain": policy.brain}))
    t0 = time.perf_counter()
    steps = decisions = teacher = 0
    faction = ""
    try:
        policy.reset(seed)
        step = session.reset(seed)
        faction = step.obs.player_faction
        while not (step.done or step.truncated):
            answer = policy.decide(step)
            decisions += len(answer)
            step = session.step(answer, step_ticks)
            steps += 1
            teacher += len(step.teacher)
            if not faction and step.obs.player_faction:
                faction = step.obs.player_faction
        outcome, error = (step.result or "truncated"), ""
    except (grpc.RpcError, RuntimeError) as e:
        outcome, error = "error", repr(e)
    finally:
        session.close()
    return {
        "seed": seed, "faction": faction, "outcome": outcome, "ticks": session.tick,
        "steps": steps, "decisions": decisions, "teacher_decisions": teacher,
        "wall_s": round(time.perf_counter() - t0, 1), "error": error,
    }


def play_selfplay_game(
    bridge: Bridge,
    policy_a_name: str,
    policy_b_name: str,
    cfg: SelfPlayConfig,
    seed: int,
    step_ticks: int,
    a_is_p1: bool = True,
) -> dict:
    pol_a = make_policy(policy_a_name)
    pol_b = make_policy(policy_b_name)
    p1_pol, p0_pol = (pol_a, pol_b) if a_is_p1 else (pol_b, pol_a)
    sp_cfg = SelfPlayConfig(
        map_name=cfg.map_name,
        player_faction=cfg.player_faction if a_is_p1 else cfg.enemy_faction,
        enemy_faction=cfg.enemy_faction if a_is_p1 else cfg.player_faction,
        max_ticks=cfg.max_ticks,
        brain=p1_pol.brain,
        peer_brain=p0_pol.brain,
    )
    session = SelfPlaySession(bridge, sp_cfg)
    t0 = time.perf_counter()
    steps = decisions = 0
    faction_a = ""
    faction_b = ""
    try:
        p1_pol.reset(seed * 2 + 1)
        p0_pol.reset(seed * 2 + 2)
        dual = session.reset(seed)
        a_step = dual.p1 if a_is_p1 else dual.p0
        b_step = dual.p0 if a_is_p1 else dual.p1
        faction_a = a_step.obs.player_faction
        faction_b = b_step.obs.player_faction
        while not (dual.done or dual.truncated):
            d1 = p1_pol.decide(dual.p1)
            d0 = p0_pol.decide(dual.p0)
            decisions += len(d1 if a_is_p1 else d0)
            dual = session.step(d1, d0, step_ticks)
            steps += 1
        a_final = dual.p1 if a_is_p1 else dual.p0
        outcome, error = (a_final.result or "truncated"), ""
    except (grpc.RpcError, RuntimeError) as e:
        outcome, error = "error", repr(e)
    finally:
        session.close()
    return {
        "seed": seed,
        "slot_a": "Multi1" if a_is_p1 else "Multi0",
        "faction": faction_a,
        "opponent_faction": faction_b,
        "outcome": outcome,
        "ticks": session.tick,
        "steps": steps,
        "decisions": decisions,
        "teacher_decisions": 0,
        "wall_s": round(time.perf_counter() - t0, 1),
        "error": error,
    }


def evaluate(
    daemons: list[str],
    policy: str,
    match: MatchConfig,
    games: int,
    seed_base: int,
    step_ticks: int,
    log=print,
    recycle_every: int = 50,
    pre_recycle: bool = False,
) -> dict:
    maps = [m.strip() for m in match.map_name.split(",") if m.strip()]
    lock = threading.Lock()
    next_game = [0]
    results: list[dict] = []

    def worker(addr: str) -> None:
        if pre_recycle:
            daemon_ctl.restart(addr)
        bridge = Bridge(addr)
        played = 0
        try:
            while True:
                with lock:
                    g = next_game[0]
                    next_game[0] += 1
                if g >= games:
                    return
                if recycle_every and played >= recycle_every:
                    bridge.close()
                    daemon_ctl.restart(addr)
                    bridge = Bridge(addr)
                    played = 0

                seed = seed_base + g
                cur_map = maps[g % len(maps)]
                cur_match = MatchConfig(
                    map_name=cur_map,
                    bot=match.bot,
                    player_faction=match.player_faction,
                    enemy_faction=match.enemy_faction,
                    max_ticks=match.max_ticks,
                    brain=match.brain,
                )
                r = play_game(bridge, policy, cur_match, seed, step_ticks)
                played += 1

                if r["outcome"] == "error":
                    bridge.close()
                    if daemon_ctl.restart(addr):
                        bridge = Bridge(addr)
                        played = 0
                        r = play_game(bridge, policy, cur_match, seed, step_ticks)
                        played += 1
                    else:
                        bridge = Bridge(addr)

                with lock:
                    results.append(r)
                    log(json.dumps(r))
        finally:
            bridge.close()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, len(daemons))) as pool:
        list(pool.map(worker, daemons))

    outcomes: dict[str, int] = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    finished = [r for r in results if r["outcome"] != "error"]
    wins = sum(1 for r in finished if is_win(r["outcome"]))
    lo, hi = wilson(wins, len(finished))
    return {
        "policy": policy,
        "bot": match.bot,
        "map": match.map_name,
        "player_faction": match.player_faction,
        "games": len(finished),
        "outcomes": outcomes,
        "win_rate": round(wins / max(1, len(finished)), 3),
        "win_rate_ci95": [round(lo, 3), round(hi, 3)],
        "factions": _faction_breakdown(finished),
        "mean_ticks": round(sum(r["ticks"] for r in finished) / max(1, len(finished))),
        "mean_decisions": round(sum(r["decisions"] for r in finished) / max(1, len(finished)), 1),
        "mean_teacher_decisions": round(sum(r["teacher_decisions"] for r in finished) / max(1, len(finished)), 1),
        "wall_s": round(time.perf_counter() - t0, 1),
        "errors": [r["error"] for r in results if r["error"]][:5],
    }


def evaluate_selfplay(
    daemons: list[str],
    policy_a: str,
    policy_b: str,
    cfg: SelfPlayConfig,
    games: int,
    seed_base: int,
    step_ticks: int = 500,
    swap_slots: bool = True,
    log=print,
    recycle_every: int = 50,
    pre_recycle: bool = False,
) -> dict:
    maps = [m.strip() for m in cfg.map_name.split(",") if m.strip()]
    lock = threading.Lock()
    next_game = [0]
    results: list[dict] = []

    def worker(addr: str) -> None:
        if pre_recycle:
            daemon_ctl.restart(addr)
        bridge = Bridge(addr)
        played = 0
        try:
            while True:
                with lock:
                    g = next_game[0]
                    next_game[0] += 1
                if g >= games:
                    return
                if recycle_every and played >= recycle_every:
                    bridge.close()
                    daemon_ctl.restart(addr)
                    bridge = Bridge(addr)
                    played = 0

                seed = seed_base + g
                a_is_p1 = True if not swap_slots else (g % 2 == 0)
                cur_map = maps[g % len(maps)]
                cur_cfg = SelfPlayConfig(
                    map_name=cur_map,
                    player_faction=cfg.player_faction,
                    enemy_faction=cfg.enemy_faction,
                    max_ticks=cfg.max_ticks,
                )
                r = play_selfplay_game(bridge, policy_a, policy_b, cur_cfg, seed, step_ticks, a_is_p1=a_is_p1)
                played += 1

                if r["outcome"] == "error":
                    bridge.close()
                    if daemon_ctl.restart(addr):
                        bridge = Bridge(addr)
                        played = 0
                        r = play_selfplay_game(bridge, policy_a, policy_b, cur_cfg, seed, step_ticks, a_is_p1=a_is_p1)
                        played += 1
                    else:
                        bridge = Bridge(addr)

                with lock:
                    results.append(r)
                    log(json.dumps(r))
        finally:
            bridge.close()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, len(daemons))) as pool:
        list(pool.map(worker, daemons))

    outcomes: dict[str, int] = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    finished = [r for r in results if r["outcome"] != "error"]
    wins = sum(1 for r in finished if is_win(r["outcome"]))
    lo, hi = wilson(wins, len(finished))
    return {
        "policy": policy_a,
        "opponent_policy": policy_b,
        "map": cfg.map_name,
        "player_faction": cfg.player_faction,
        "games": len(finished),
        "outcomes": outcomes,
        "win_rate": round(wins / max(1, len(finished)), 3),
        "win_rate_ci95": [round(lo, 3), round(hi, 3)],
        "factions": _faction_breakdown(finished),
        "mean_ticks": round(sum(r["ticks"] for r in finished) / max(1, len(finished))),
        "mean_decisions": round(sum(r["decisions"] for r in finished) / max(1, len(finished)), 1),
        "wall_s": round(time.perf_counter() - t0, 1),
        "errors": [r["error"] for r in results if r["error"]][:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="scripted",
                    help="builtin | scripted | random | net:<path> | net-sample:<path> | net-t0.25:<path>")
    ap.add_argument("--opponent-policy", "--peer-policy", default="",
                    help="if set, evaluate RL-vs-RL via SelfPlaySession instead of vs --bot")
    ap.add_argument("--daemons", default=DEFAULT_DAEMONS)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--bot", default="hard")
    ap.add_argument("--player-faction", default=MatchConfig.player_faction)
    ap.add_argument("--enemy-faction", default=MatchConfig.enemy_faction)
    ap.add_argument("--map", default=MatchConfig.map_name)
    ap.add_argument("--max-ticks", type=int, default=54000, help="truncate games after this (~36 min)")
    ap.add_argument("--step-ticks", type=int, default=500, help="max ticks per step")
    ap.add_argument("--seed-base", type=int, default=10000)
    ap.add_argument("--swap-slots", action=argparse.BooleanOptionalAction, default=True,
                    help="alternate Multi1/Multi0 in --opponent-policy mode")
    ap.add_argument("--recycle-every", type=int, default=50, help="restart each daemon every N games (0=never)")
    ap.add_argument("--pre-recycle", action="store_true", help="restart daemons before starting evaluation")
    ap.add_argument("--out", default="", help="append per-game lines and the summary to this jsonl")
    args = ap.parse_args()

    def log(line: str) -> None:
        print(line, flush=True)
        if args.out:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    daemons = [d.strip() for d in args.daemons.split(",") if d.strip()]
    if args.opponent_policy:
        sp_cfg = SelfPlayConfig(
            map_name=args.map,
            player_faction=args.player_faction,
            enemy_faction=args.enemy_faction,
            max_ticks=args.max_ticks,
        )
        summary = evaluate_selfplay(
            daemons, args.policy, args.opponent_policy, sp_cfg, args.games, args.seed_base,
            args.step_ticks, swap_slots=args.swap_slots, log=log,
            recycle_every=args.recycle_every, pre_recycle=args.pre_recycle,
        )
    else:
        match = MatchConfig(
            map_name=args.map,
            bot=args.bot,
            player_faction=args.player_faction,
            enemy_faction=args.enemy_faction,
            max_ticks=args.max_ticks,
        )
        summary = evaluate(
            daemons, args.policy, match, args.games, args.seed_base, args.step_ticks, log,
            recycle_every=args.recycle_every, pre_recycle=args.pre_recycle,
        )
    log(json.dumps({"summary": summary}))


if __name__ == "__main__":
    main()
