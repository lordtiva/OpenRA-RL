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


def _as_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    return getattr(obj, "__dict__", None) or {}


def _gs_side(gs, side: str) -> dict:
    raw = _as_dict(gs)
    s = raw.get(side) if isinstance(raw, dict) else None
    if s is None and gs is not None:
        s = getattr(gs, side, None)
    s = _as_dict(s)
    return {
        "cash": int(s.get("cash", 0) or 0),
        "earned": int(s.get("earned", 0) or 0),
        "building_value": int(s.get("building_value", 0) or 0),
        "unit_value": int(s.get("unit_value", 0) or 0),
        "n_buildings": int(s.get("n_buildings", 0) or 0),
    }


def _proc_cells(obs, attr: str = "buildings") -> List[List[int]]:
    out: List[List[int]] = []
    for b in getattr(obs, attr, None) or []:
        if str(getattr(b, "type", "") or "").lower() != "proc":
            continue
        try:
            out.append([int(b.cell_x), int(b.cell_y)])
        except (TypeError, ValueError):
            continue
    return out


def _gs_snap(tick: int, gs, obs=None) -> dict:
    own = _gs_side(gs, "own")
    ene = _gs_side(gs, "enemy")
    t = max(int(tick or 0), 1)
    own_e = int(own["earned"])
    ene_e = int(ene["earned"])
    return {
        "tick": int(tick or 0),
        "own_earned": own_e,
        "ene_earned": ene_e,
        "own_h1k": round(own_e / t * 1000.0, 1),
        "ene_h1k": round(ene_e / t * 1000.0, 1),
        "edge": round((own_e - ene_e) / t * 1000.0, 1),
        "own_bv": int(own["building_value"]),
        "ene_bv": int(ene["building_value"]),
        "own_uv": int(own["unit_value"]),
        "ene_uv": int(ene["unit_value"]),
        "own_nb": int(own["n_buildings"]),
        "ene_nb": int(ene["n_buildings"]),
        "own_proc": _proc_cells(obs) if obs is not None else [],
        "ene_proc": (
            _proc_cells(obs, "visible_enemy_buildings")
            if obs is not None else []),
    }


def _fmt_proc(cells) -> str:
    if not cells:
        return "-"
    return "+".join(f"{c[0]},{c[1]}" for c in cells)


def _print_trace(tag: str, snap: dict) -> None:
    print(
        f"  [trace {tag} t={snap['tick']:5d}] "
        f"earned={snap['own_earned']}/{snap['ene_earned']} "
        f"eco={snap['own_h1k']}/{snap['ene_h1k']} edge={snap['edge']} "
        f"bv={snap['own_bv']}/{snap['ene_bv']} "
        f"uv={snap['own_uv']}/{snap['ene_uv']} "
        f"nb={snap['own_nb']}/{snap['ene_nb']} "
        f"proc={_fmt_proc(snap.get('own_proc'))} "
        f"eneproc={_fmt_proc(snap.get('ene_proc'))}",
        flush=True,
    )


def _obs_gs(obs):
    gs = getattr(obs, "global_summary", None)
    if gs is None:
        md = getattr(obs, "metadata", None)
        if isinstance(md, dict):
            gs = md.get("global_summary")
    return gs


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
    mode: str = "rush",
    message_timeout_s: float = 300.0,
    trace_every: int = 0,
) -> dict:
    """One ScriptedTeacher game with train/BC-style macro advance."""
    teacher = ScriptedTeacher(rush_attack_move=int(rush), mode=mode)
    reset_kwargs = _scenario_reset_kwargs(scenario, bot_type, seed)
    t0 = time.time()
    async with OpenRAEnv(base_url=url, message_timeout_s=message_timeout_s) as env:
        result = await env.reset(**reset_kwargs)
        obs = result.observation
        done = bool(result.done)
        decisions = 0
        advanced_total = 0
        last_gs = None
        traces: List[dict] = []
        last_trace_bucket = -1
        tag = f"seed={seed}"
        peak = {
            "buildings": 0, "weap": 0, "pbox": 0, "tnk": 0,
            "harv": 0, "proc": 0,
        }
        proc_seen: List[List[int]] = []

        def _maybe_trace(tick: int, gs, o) -> None:
            nonlocal last_trace_bucket
            if int(trace_every) <= 0 or gs is None:
                return
            bucket = int(tick or 0) // int(trace_every)
            if bucket <= last_trace_bucket:
                return
            last_trace_bucket = bucket
            snap = _gs_snap(tick, gs, o)
            traces.append(snap)
            _print_trace(tag, snap)

        def _note(o):
            blds = list(getattr(o, "buildings", None) or [])
            units = list(getattr(o, "units", None) or [])
            types_b = {
                str(getattr(b, "type", "") or "").lower() for b in blds
            }
            peak["buildings"] = max(peak["buildings"], len(blds))
            peak["proc"] = max(
                peak["proc"],
                sum(1 for b in blds
                    if str(getattr(b, "type", "") or "").lower() == "proc"),
            )
            if "weap" in types_b:
                peak["weap"] = 1
            n_pbox = sum(
                1 for b in blds
                if str(getattr(b, "type", "") or "").lower()
                in ("pbox", "hbox", "ftur")
            )
            peak["pbox"] = max(peak["pbox"], n_pbox)
            n_tnk = sum(
                1 for u in units
                if str(getattr(u, "type", "") or "").lower()
                in ("1tnk", "2tnk", "3tnk")
            )
            peak["tnk"] = max(peak["tnk"], n_tnk)
            n_harv = sum(
                1 for u in units
                if "harv" in str(getattr(u, "type", "") or "").lower()
            )
            eco_h = int(getattr(getattr(o, "economy", None), "harvester_count", 0) or 0)
            peak["harv"] = max(peak["harv"], n_harv, eco_h)
            for cell in _proc_cells(o):
                if cell not in proc_seen:
                    proc_seen.append(cell)

        _note(obs)
        while not done and decisions < max_steps:
            action = teacher.decide(obs)
            result = await env.step(action)
            obs = result.observation
            done = bool(result.done)
            decisions += 1
            _note(obs)
            if done or macro_ticks <= 0:
                continue
            # Mirror rl/rollout.py macro: step(+2 ticks) then advance rest.
            restante = max(0, int(macro_ticks) - 2)
            while restante > 0 and not done:
                adv = await env.advance(min(50, restante))
                advanced_total += int(adv.get("actual_ticks_advanced", 0) or 0)
                gs_adv = adv.get("global_summary")
                if gs_adv:
                    last_gs = gs_adv
                    _maybe_trace(int(adv.get("tick", 0) or 0), gs_adv, obs)
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
                _note(obs)
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
        gs = _obs_gs(obs) or last_gs
        if int(trace_every) > 0 and gs is not None:
            snap = _gs_snap(int(getattr(obs, "tick", 0) or 0), gs, obs)
            if (not traces
                    or int(snap["tick"]) != int(traces[-1].get("tick") or 0)):
                traces.append(snap)
                _print_trace(tag, snap)
        own = _gs_side(gs, "own")
        ene = _gs_side(gs, "enemy")
        tick = max(int(getattr(obs, "tick", 0) or 0), 1)
        own_e = int(own["earned"])
        ene_e = int(ene["earned"])
        own_h1k = round(own_e / tick * 1000.0, 1)
        ene_h1k = round(ene_e / tick * 1000.0, 1)
        eco = getattr(obs, "economy", None)
        own_cash = int(getattr(eco, "cash", 0) or 0) if eco is not None else 0
        own_ore = int(getattr(eco, "ore", 0) or 0) if eco is not None else 0
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
            "peak_buildings": int(peak["buildings"]),
            "ever_weap": int(peak["weap"]),
            "ever_pbox": int(peak["pbox"]),
            "peak_tnk": int(peak["tnk"]),
            "peak_harv": int(peak["harv"]),
            "peak_proc": int(peak["proc"]),
            "own_earned": own_e,
            "ene_earned": ene_e,
            "own_harvest_per_1k": own_h1k,
            "ene_harvest_per_1k": ene_h1k,
            "harvest_edge": round(own_h1k - ene_h1k, 1),
            "own_cash": own_cash,
            "own_ore": own_ore,
            "own_proc_cells": proc_seen,
            "trace": traces,
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
        f"mode={getattr(args, 'mode', 'rush')} "
        f"macro={args.macro_ticks} max_steps={args.max_steps} "
        f"rushes={rushes} games/rush={games} urls={urls}"
        + (f" trace={int(getattr(args, 'trace', 0) or 0)}"
           if int(getattr(args, "trace", 0) or 0) > 0 else ""),
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
                    mode=str(getattr(args, "mode", "rush") or "rush"),
                    trace_every=int(getattr(args, "trace", 0) or 0),
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
                f"({r.get('elapsed_s', 0):.0f}s) "
                f"weap={r.get('ever_weap', '-')} pbox={r.get('ever_pbox', '-')} "
                f"tnk={r.get('peak_tnk', '-')} harv={r.get('peak_harv', '-')} "
                f"proc={r.get('peak_proc', '-')}@{_fmt_proc(r.get('own_proc_cells'))} "
                f"eco={r.get('own_harvest_per_1k', '-')}/{r.get('ene_harvest_per_1k', '-')} "
                f"edge={r.get('harvest_edge', '-')} "
                f"earned={r.get('own_earned', '-')}/{r.get('ene_earned', '-')} "
                f"res={r.get('own_cash', '-')}+{r.get('own_ore', '-')}",
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
    ap.add_argument("--mode", default="rush", choices=("rush", "expand"),
                    help="ScriptedTeacher mode (expand = C weap/tanks).")
    ap.add_argument("--macro-ticks", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=2800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="rl/ckpts_v2/bench_teacher.json")
    ap.add_argument(
        "--trace", nargs="?", const=5000, default=0, type=int,
        help="Spectator fogless dump every N ticks (default off; "
             "--trace alone = 5000). Prints earned/bv/uv/nb + own proc cells.",
    )
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
