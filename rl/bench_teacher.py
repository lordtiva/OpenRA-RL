#!/usr/bin/env python3
"""bench_teacher.py — ScriptedTeacher quality bench vs beginner (phase-A matched).

Mirrors onboard phase A / BC teacher games as closely as practical:
  scenario a_short, bot_type beginner, macro_ticks=20, max_steps=2800
  (same macro path as rl/rollout.collect_one_episode: step + advance).

Uses ScriptedTeacher (not ScriptedBot). Per-instance rush_attack_move override
only — does NOT mutate ScriptedTeacher.RUSH_ATTACK_MOVE class default.

Recommendation rule (pick rush for train/onboard):
  1. Maximize winrate (wins / n). Primary objective.
  2. REJECT a candidate if lose_rate rises materially vs baseline rush=8
     even if incompletes fall (fear: 8→5 turns incompletes into loses).
     "Materially" ≈ +5pp absolute lose_rate or more loses at same n.
  3. Prefer lower incompletes only as a tie-break among equal winrate /
     non-worse lose_rate configs.
  4. Use Wilson 95% CI for directional read; do not promote on pilot n alone.

Usage:
  .\\.venv\\Scripts\\python.exe rl/bench_teacher.py --help
  .\\.venv\\Scripts\\python.exe rl/bench_teacher.py --rush 8,6,5 --games 8 \\
      --url http://localhost:8000,http://localhost:8010 \\
      --out rl/ckpts_v2/bench_teacher_pilot.json
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openra_env.client import OpenRAEnv
from openra_env.models import ActionType, CommandModel, OpenRAAction
from rl.scripted_teacher import ScriptedTeacher


def _wilson(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lo = max(0.0, (centre - margin) / denom)
    hi = min(1.0, (centre + margin) / denom)
    return (round(lo, 4), round(hi, 4))


def _scenario_reset_kwargs(scenario: str, bot_type: str, seed: int) -> dict:
    mapa = Path("rl/scenarios") / f"fase2_{scenario.lower()}.oramap"
    if not mapa.exists():
        raise SystemExit(f"escenario inexistente: {mapa}")
    return {
        "seed": int(seed),
        "bot_type": bot_type,
        "map_data": base64.b64encode(mapa.read_bytes()).decode(),
        "map_name": f"fase2_{scenario.lower()}.oramap",
    }


async def play_one(
    url: str,
    *,
    rush: int,
    seed: int,
    scenario: str,
    bot_type: str,
    macro_ticks: int,
    max_steps: int,
    message_timeout_s: float = 300.0,
) -> dict:
    """One ScriptedTeacher game with train/BC-style macro advance."""
    teacher = ScriptedTeacher(rush_attack_move=int(rush))
    reset_kwargs = _scenario_reset_kwargs(scenario, bot_type, seed)
    t0 = time.time()
    async with OpenRAEnv(base_url=url, message_timeout_s=message_timeout_s) as env:
        result = await env.reset(**reset_kwargs)
        obs = result.observation
        done = bool(result.done)
        decisions = 0
        advanced_total = 0
        while not done and decisions < max_steps:
            action = teacher.decide(obs)
            result = await env.step(action)
            obs = result.observation
            done = bool(result.done)
            decisions += 1
            if done or macro_ticks <= 0:
                continue
            # Mirror rl/rollout.py macro: step(+2 ticks) then advance rest.
            restante = max(0, int(macro_ticks) - 2)
            while restante > 0 and not done:
                adv = await env.advance(min(50, restante))
                advanced_total += int(adv.get("actual_ticks_advanced", 0) or 0)
                done = bool(adv.get("done", False))
                if done:
                    # Prefer advance result if game ended mid-block.
                    final = adv.get("result")
                    if final:
                        # stitch a minimal done obs view
                        try:
                            obs.result = final  # type: ignore[attr-defined]
                            obs.done = True
                            obs.tick = int(adv.get("tick", getattr(obs, "tick", 0)) or 0)
                        except Exception:
                            pass
                    break
                restante -= int(adv.get("actual_ticks_advanced", 0) or 0)
            if not done:
                result = await env.step(OpenRAAction(
                    commands=[CommandModel(action=ActionType.NO_OP)]))
                obs = result.observation
                done = bool(result.done)
        outcome = (getattr(obs, "result", None) or "incomplete") if done else "incomplete"
        if not isinstance(outcome, str):
            outcome = str(outcome or "incomplete")
        outcome_l = outcome.lower()
        if outcome_l.startswith("win"):
            bucket = "win"
        elif outcome_l.startswith("lose") or outcome_l.startswith("loss"):
            bucket = "lose"
        else:
            bucket = "incomplete"
        return {
            "rush": int(rush),
            "seed": int(seed),
            "url": url,
            "result": outcome,
            "bucket": bucket,
            "done": bool(done),
            "ticks": int(getattr(obs, "tick", 0) or 0),
            "decisions": int(decisions),
            "advanced_ticks": int(advanced_total),
            "elapsed_s": round(time.time() - t0, 1),
            "n_own_buildings": len(getattr(obs, "buildings", None) or []),
            "n_own_units": len(getattr(obs, "units", None) or []),
            "n_ene_units": len(getattr(obs, "visible_enemies", None) or []),
            "n_ene_buildings": len(getattr(obs, "visible_enemy_buildings", None) or []),
        }


def summarize(rows: List[dict]) -> dict:
    n = len(rows)
    wins = sum(1 for r in rows if r.get("bucket") == "win")
    loses = sum(1 for r in rows if r.get("bucket") == "lose")
    incs = sum(1 for r in rows if r.get("bucket") == "incomplete")
    ticks_all = [int(r.get("ticks") or 0) for r in rows]
    ticks_win = [int(r.get("ticks") or 0) for r in rows if r.get("bucket") == "win"]
    mean_all = round(sum(ticks_all) / n, 1) if n else 0.0
    mean_win = round(sum(ticks_win) / len(ticks_win), 1) if ticks_win else None
    wr = round(wins / n, 4) if n else 0.0
    lr = round(loses / n, 4) if n else 0.0
    ir = round(incs / n, 4) if n else 0.0
    lo, hi = _wilson(wins, n)
    return {
        "n": n,
        "wins": wins,
        "loses": loses,
        "incompletes": incs,
        "winrate": wr,
        "lose_rate": lr,
        "incomplete_rate": ir,
        "mean_ticks": mean_all,
        "mean_ticks_wins": mean_win,
        "wilson95_winrate": [lo, hi],
    }


def recommend(per_rush: Dict[str, dict], baseline: int = 8) -> dict:
    """Apply docstring rule; return structured recommendation."""
    if not per_rush:
        return {"pick": None, "reason": "no data"}
    base = per_rush.get(str(baseline)) or per_rush.get(baseline)
    base_lose = float(base["lose_rate"]) if base else 0.0
    candidates = []
    rejected = []
    for k, s in per_rush.items():
        rush = int(k)
        if base is not None and rush != baseline:
            # reject if lose_rate rises materially vs baseline
            if float(s["lose_rate"]) - base_lose >= 0.05:
                rejected.append({
                    "rush": rush,
                    "reason": (
                        f"lose_rate {s['lose_rate']} vs baseline "
                        f"{base_lose} (+{round(float(s['lose_rate']) - base_lose, 4)})"
                    ),
                })
                continue
        candidates.append((rush, s))
    if not candidates:
        return {
            "pick": baseline if base else None,
            "reason": "all non-baseline rejected; keep baseline",
            "rejected": rejected,
        }
    # max winrate; tie-break lower incomplete_rate, then lower lose_rate, then lower rush
    candidates.sort(
        key=lambda x: (
            -float(x[1]["winrate"]),
            float(x[1]["incomplete_rate"]),
            float(x[1]["lose_rate"]),
            x[0],
        )
    )
    pick, s = candidates[0]
    return {
        "pick": pick,
        "reason": (
            f"max winrate={s['winrate']} (n={s['n']}); "
            f"lose_rate={s['lose_rate']} incomplete_rate={s['incomplete_rate']}"
        ),
        "rejected": rejected,
    }


async def run_bench(args) -> dict:
    rushes = [int(x.strip()) for x in args.rush.split(",") if x.strip()]
    urls = [u.strip() for u in args.url.split(",") if u.strip()]
    if not urls:
        raise SystemExit("need at least one --url")
    games = int(args.games)
    print(
        f"ScriptedTeacher bench vs {args.bot_type} on {args.scenario} "
        f"macro={args.macro_ticks} max_steps={args.max_steps} "
        f"rushes={rushes} games/rush={games} urls={urls}",
        flush=True,
    )
    # Build job list: (rush, game_idx, seed)
    jobs: List[Tuple[int, int, int]] = []
    for rush in rushes:
        for g in range(games):
            seed = int(args.seed) + rush * 1000 + g
            jobs.append((rush, g, seed))

    sem = asyncio.Semaphore(max(1, len(urls)))
    url_cycle = 0
    url_lock = asyncio.Lock()
    raw: List[dict] = []
    raw_lock = asyncio.Lock()

    async def next_url() -> str:
        nonlocal url_cycle
        async with url_lock:
            u = urls[url_cycle % len(urls)]
            url_cycle += 1
            return u

    async def one_job(rush: int, g: int, seed: int):
        async with sem:
            url = await next_url()
            tag = f"rush={rush} #{g + 1}/{games} seed={seed} url={url}"
            try:
                r = await play_one(
                    url,
                    rush=rush,
                    seed=seed,
                    scenario=args.scenario,
                    bot_type=args.bot_type,
                    macro_ticks=args.macro_ticks,
                    max_steps=args.max_steps,
                )
            except Exception as e:
                r = {
                    "rush": rush,
                    "seed": seed,
                    "url": url,
                    "result": f"ERROR:{e}",
                    "bucket": "incomplete",
                    "done": False,
                    "ticks": 0,
                    "decisions": 0,
                    "advanced_ticks": 0,
                    "elapsed_s": 0.0,
                    "n_own_buildings": 0,
                    "n_own_units": 0,
                    "n_ene_units": 0,
                    "n_ene_buildings": 0,
                    "error": str(e),
                }
                print(f"  [{tag}] ERROR: {e}", flush=True)
            print(
                f"  [{tag}] {r['bucket']:11s} result={r['result']!s:12s} "
                f"tick={r['ticks']:6d} dec={r.get('decisions', 0):4d} "
                f"({r.get('elapsed_s', 0):.0f}s)",
                flush=True,
            )
            async with raw_lock:
                raw.append(r)

    # Run all jobs with concurrency = number of URLs
    await asyncio.gather(*[one_job(rush, g, seed) for rush, g, seed in jobs])

    per_rush: Dict[str, Any] = {}
    for rush in rushes:
        rows = [r for r in raw if int(r["rush"]) == int(rush)]
        # stable order by seed
        rows.sort(key=lambda r: int(r.get("seed") or 0))
        per_rush[str(rush)] = summarize(rows)

    rec = recommend(per_rush, baseline=8)
    out = {
        "config": {
            "scenario": args.scenario,
            "bot_type": args.bot_type,
            "macro_ticks": args.macro_ticks,
            "max_steps": args.max_steps,
            "seed_base": args.seed,
            "games_per_rush": games,
            "rushes": rushes,
            "urls": urls,
            "class_default_rush": ScriptedTeacher.RUSH_ATTACK_MOVE,
        },
        "per_rush": per_rush,
        "recommendation": rec,
        "raw": raw,
    }
    return out


def print_table(per_rush: dict) -> None:
    hdr = (
        f"{'rush':>4} {'n':>3} {'W':>3} {'L':>3} {'I':>3} "
        f"{'winrate':>8} {'lose_r':>8} {'inc_r':>8} "
        f"{'mean_t':>8} {'mean_t_W':>8} {'wilson95':>16}"
    )
    print("\n=== TABLE ===", flush=True)
    print(hdr, flush=True)
    for k in sorted(per_rush.keys(), key=lambda x: int(x)):
        s = per_rush[k]
        wci = s.get("wilson95_winrate") or [0, 0]
        mtw = s.get("mean_ticks_wins")
        mtw_s = f"{mtw:.1f}" if mtw is not None else "-"
        print(
            f"{int(k):>4} {s['n']:>3} {s['wins']:>3} {s['loses']:>3} {s['incompletes']:>3} "
            f"{s['winrate']:>8.3f} {s['lose_rate']:>8.3f} {s['incomplete_rate']:>8.3f} "
            f"{s['mean_ticks']:>8.1f} {mtw_s:>8} "
            f"[{wci[0]:.3f},{wci[1]:.3f}]",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark ScriptedTeacher rush thresholds vs beginner")
    ap.add_argument("--url", default="http://localhost:8000,http://localhost:8010",
                    help="Comma-separated OpenRA daemon URLs (concurrency=len)")
    ap.add_argument("--rush", default="8,6,5",
                    help="Comma-separated RUSH_ATTACK_MOVE overrides")
    ap.add_argument("--games", type=int, default=8,
                    help="Games per rush value")
    ap.add_argument("--scenario", default="a_short")
    ap.add_argument("--bot-type", default="beginner")
    ap.add_argument("--macro-ticks", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=2800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="rl/ckpts_v2/bench_teacher.json")
    args = ap.parse_args()
    out = asyncio.run(run_bench(args))
    print_table(out["per_rush"])
    rec = out.get("recommendation") or {}
    print(f"\nrecommendation: pick={rec.get('pick')} — {rec.get('reason')}", flush=True)
    if rec.get("rejected"):
        for rj in rec["rejected"]:
            print(f"  rejected rush={rj['rush']}: {rj['reason']}", flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
