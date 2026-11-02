"""Record imitation data (Commander v4): (features, teacher action) per decision, one .npz per game.

    # Bilateral Allies+Soviet scripted demonstrations vs bots + mirror self-play:
    python -m ra_rl.record --teacher scripted --bots hard,brutal,turtle --player-faction Random \
        --selfplay-prob 0.25 --eps 0.15 --eps-army 0.10 --games 1800 --out data/scripted_v4

    # DAgger: the network plays, the bilateral scripted policy labels the states it reaches:
    python -m ra_rl.record --teacher scripted --executor net:runs/v4/bc_s0/model.pt \
        --bots hard,brutal,turtle --player-faction Random --selfplay-prob 0.25 --games 1200 --out data/dagger1_v4

Seeds 10000-10099 (eval) and 20000-20199 (selection) are reserved and refused.
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import numpy as np

from ra_env import Bridge, GameSession, MatchConfig, SelfPlayConfig, SelfPlaySession, StepResult, daemons, pb
from ra_rl.eval import DEFAULT_DAEMONS
from ra_rl.features import (
    ALL_IN_OFFSET,
    FEATURE_VERSION,
    HOLD,
    WAIT,
    Features,
    is_tactical_request,
    tactical_channel_id,
)
from ra_rl.policies import make_policy
from ra_rl.policies.scripted import ScriptedPolicy
from ra_rl.vocab import QUEUE_INDEX

EVAL_SEEDS = range(10000, 10100)
SELECT_SEEDS = range(20000, 20200)


class GameBuffer:
    def __init__(self) -> None:
        self.prod: dict[str, list] = {"g": [], "queue": [], "mask": [], "y": []}
        self.army: dict[str, list] = {"g": [], "group": [], "cand": [], "mask": [], "y": []}

    def add(self, feats: Features, req: pb.DecisionRequest, y: int) -> None:
        if is_tactical_request(req):
            cand, mask = feats.candidates(req)
            if not mask[y]:
                if ALL_IN_OFFSET <= y < HOLD and mask[y - ALL_IN_OFFSET]:
                    y = y - ALL_IN_OFFSET
                else:
                    return
            grp = tactical_channel_id(req)
            for k, v in (("g", feats.g.copy()), ("group", grp), ("cand", cand), ("mask", mask), ("y", y)):
                self.army[k].append(v)
        else:
            mask = feats.item_mask(req)
            if not mask[y]:
                y = WAIT
            for k, v in (("g", feats.g.copy()), ("queue", QUEUE_INDEX.get(req.queue, 0)),
                         ("mask", mask), ("y", y)):
                self.prod[k].append(v)

    def save(self, path: Path, meta: dict) -> None:
        def arr(rows: list, dtype) -> np.ndarray:
            return np.asarray(rows, dtype) if rows else np.zeros((0,), dtype)

        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            prod_g=arr(self.prod["g"], np.float16),
            prod_queue=arr(self.prod["queue"], np.int8),
            prod_mask=arr(self.prod["mask"], bool),
            prod_y=arr(self.prod["y"], np.int16),
            army_g=arr(self.army["g"], np.float16),
            army_group=arr(self.army["group"], np.int8),
            army_cand=arr(self.army["cand"], np.float16),
            army_mask=arr(self.army["mask"], bool),
            army_y=arr(self.army["y"], np.int16),
            meta=np.asarray(json.dumps(meta)),
        )
        tmp.replace(path)


def _step_side(
    step: StepResult,
    feats: Features,
    expert: ScriptedPolicy,
    executor,
    buf: GameBuffer,
    rng: random.Random,
    eps: float,
    eps_army: float,
) -> list[pb.Decision]:
    feats.observe(step.obs)
    answer: list[pb.Decision] = []
    before = expert.army_state()
    labels = {d.request_id: d for d in expert.decide(step)}
    chosen = executor.actions(step) if executor is not None else None
    for i, req in enumerate(step.pending):
        teacher = labels.get(req.id)
        y = feats.label(req, teacher)
        if y is not None:
            buf.add(feats, req, y)
        tactical = is_tactical_request(req)
        a = chosen[i] if chosen is not None else y
        if rng.random() < (eps_army if tactical else eps):
            mask = feats.candidates(req)[1] if tactical else feats.item_mask(req)
            a = rng.choice(np.flatnonzero(mask).tolist())
        d = teacher if a is None or a == y else feats.decision(req, a)
        feats.record(req, d)
        if executor is not None:
            executor.feats.record(req, d)
        if req.kind == pb.DECIDE_ARMY and a != y:
            expert.sync_army(before, req, None if d is None else d.target_index, step.obs)
        if d is not None:
            answer.append(d)
    return answer


def record_selfplay_game(bridge: Bridge, args, seed: int) -> dict:
    maps = [m.strip() for m in args.map.split(",") if m.strip()]
    map_name = maps[seed % len(maps)]
    sp_cfg = SelfPlayConfig(
        map_name=map_name,
        player_faction=args.player_faction,
        enemy_faction=args.enemy_faction,
        max_ticks=args.max_ticks,
        brain="external",
        peer_brain="external",
    )
    session = SelfPlaySession(bridge, sp_cfg)
    rng1 = random.Random(seed * 2 + 1)
    rng0 = random.Random(seed * 2 + 2)
    feats1, feats0 = Features(), Features()
    expert1, expert0 = ScriptedPolicy(), ScriptedPolicy()
    exec1 = make_policy(args.executor) if args.executor else None
    exec0 = make_policy(args.executor) if args.executor else None
    if exec1 is not None:
        exec1.reset(seed * 2 + 1)
    if exec0 is not None:
        exec0.reset(seed * 2 + 2)
    buf = GameBuffer()
    t0 = time.perf_counter()
    factions = ("", "")
    try:
        dual = session.reset(seed)
        factions = (dual.p1.obs.player_faction, dual.p0.obs.player_faction)
        while not (dual.done or dual.truncated):
            d1 = _step_side(dual.p1, feats1, expert1, exec1, buf, rng1, args.eps, args.eps_army)
            d0 = _step_side(dual.p0, feats0, expert0, exec0, buf, rng0, args.eps, args.eps_army)
            dual = session.step(d1, d0, args.step_ticks)
        outcome, error = (dual.p1.result or "truncated"), ""
    except (grpc.RpcError, RuntimeError) as e:
        outcome, error = "error", repr(e)
    finally:
        session.close()

    meta = {
        "seed": seed, "bot": "selfplay", "factions": list(factions),
        "teacher": args.teacher, "executor": args.executor,
        "eps": args.eps, "eps_army": args.eps_army,
        "outcome": outcome, "ticks": session.tick,
        "prod_rows": len(buf.prod["y"]), "army_rows": len(buf.army["y"]),
        "feature_version": FEATURE_VERSION,
        "wall_s": round(time.perf_counter() - t0, 1), "error": error,
    }
    if not error:
        buf.save(Path(args.out) / f"g{seed:06d}.npz", meta)
    return meta


def record_game(bridge: Bridge, args, seed: int, bot: str) -> dict:
    if bot == "selfplay":
        return record_selfplay_game(bridge, args, seed)

    builtin = args.teacher == "builtin"
    maps = [m.strip() for m in args.map.split(",") if m.strip()]
    map_name = maps[seed % len(maps)]
    match = MatchConfig(
        map_name=map_name,
        bot=bot,
        player_faction=args.player_faction,
        enemy_faction=args.enemy_faction,
        max_ticks=args.max_ticks,
        brain="builtin" if builtin else "external",
    )
    session = GameSession(bridge, match)
    rng = random.Random(seed)
    feats = Features()
    expert = None if builtin else ScriptedPolicy()
    executor = make_policy(args.executor) if args.executor else None
    if executor is not None:
        executor.reset(seed)
    buf = GameBuffer()
    t0 = time.perf_counter()
    faction = ""
    try:
        step = session.reset(seed)
        faction = step.obs.player_faction
        while not (step.done or step.truncated):
            if builtin:
                feats.observe(step.obs)
                answer: list[pb.Decision] = []
                labeled = [(td.request, td.decision) for td in step.teacher]
                for req, d in labeled:
                    y = feats.label(req, d)
                    if y is not None:
                        buf.add(feats, req, y)
                for req, d in labeled:
                    feats.record(req, d)
            else:
                assert expert is not None
                answer = _step_side(step, feats, expert, executor, buf, rng, args.eps, args.eps_army)
            step = session.step(answer, args.step_ticks)
        outcome, error = (step.result or "truncated"), ""
    except (grpc.RpcError, RuntimeError) as e:
        outcome, error = "error", repr(e)
    finally:
        session.close()

    meta = {
        "seed": seed, "bot": bot, "faction": faction,
        "teacher": args.teacher, "executor": args.executor,
        "eps": args.eps, "eps_army": args.eps_army,
        "outcome": outcome, "ticks": session.tick,
        "prod_rows": len(buf.prod["y"]), "army_rows": len(buf.army["y"]),
        "feature_version": FEATURE_VERSION,
        "wall_s": round(time.perf_counter() - t0, 1), "error": error,
    }
    if not error:
        buf.save(Path(args.out) / f"g{seed:06d}.npz", meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", choices=("builtin", "scripted"), required=True)
    ap.add_argument("--executor", default="", help="policy that plays instead of the teacher (DAgger), e.g. net:<path>")
    ap.add_argument("--eps", type=float, default=0.0, help="chance of a random BUILD/TRAIN action (scripted)")
    ap.add_argument("--eps-army", type=float, default=0.0, help="chance of a random tactical action (scripted)")
    ap.add_argument("--selfplay-prob", type=float, default=0.0,
                    help="fraction of games recorded as bilateral RL-vs-RL mirror self-play")
    ap.add_argument("--player-faction", default="Random", help="Multi1 faction (Random covers Allies + Soviet)")
    ap.add_argument("--enemy-faction", default="Random", help="Multi0 faction")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--seed-base", type=int, default=1)
    ap.add_argument("--bots", default="hard,brutal,turtle", help="opponents, cycled by seed")
    ap.add_argument("--daemons", default=DEFAULT_DAEMONS)
    ap.add_argument("--map", default=MatchConfig.map_name)
    ap.add_argument("--max-ticks", type=int, default=54000)
    ap.add_argument("--step-ticks", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recycle-every", type=int, default=60,
                    help="restart each daemon (docker) after this many games; 0 = never")
    args = ap.parse_args()
    if args.teacher == "builtin" and (args.executor or args.eps or args.eps_army or args.selfplay_prob > 0):
        ap.error("the builtin teacher plays its own games (no --executor / --eps / --selfplay-prob)")

    seeds = [args.seed_base + i for i in range(args.games)]
    if any(s in EVAL_SEEDS or s in SELECT_SEEDS or s <= 0 for s in seeds):
        ap.error(
            f"seeds must be > 0 and outside reserved eval ({EVAL_SEEDS.start}-{EVAL_SEEDS.stop - 1}) "
            f"and selection ({SELECT_SEEDS.start}-{SELECT_SEEDS.stop - 1}) ranges"
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seeds if not (out / f"g{s:06d}.npz").exists()]
    bots = [b.strip() for b in args.bots.split(",") if b.strip()]
    print(f"{len(todo)} games to record ({len(seeds) - len(todo)} already there)", flush=True)

    lock = threading.Lock()
    log = (out / "games.jsonl").open("a", encoding="utf-8")

    def pick_opponent(seed: int) -> str:
        if args.selfplay_prob > 0.0 and random.Random(seed * 7919).random() < args.selfplay_prob:
            return "selfplay"
        return bots[seed % len(bots)]

    def worker(address: str) -> None:
        bridge = Bridge(address)
        played = 0
        try:
            while True:
                with lock:
                    if not todo:
                        return
                    seed = todo.pop(0)
                if args.recycle_every and played >= args.recycle_every:
                    bridge.close()
                    daemons.restart(address)
                    bridge = Bridge(address)
                    played = 0
                bot_name = pick_opponent(seed)
                meta = record_game(bridge, args, seed, bot_name)
                played += 1
                if meta["error"]:
                    bridge.close()
                    daemons.restart(address)
                    bridge = Bridge(address)
                    played = 0
                    meta = record_game(bridge, args, seed, bot_name)
                    played += 1
                with lock:
                    log.write(json.dumps(meta) + "\n")
                    log.flush()
                    print(json.dumps(meta), flush=True)
        finally:
            bridge.close()

    addresses = [d.strip() for d in args.daemons.split(",") if d.strip()]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(addresses)) as pool:
        list(pool.map(worker, addresses))
    log.close()
    print(f"done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
