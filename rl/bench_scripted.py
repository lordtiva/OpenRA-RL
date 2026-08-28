#!/usr/bin/env python3
"""
bench_scripted.py — Benchmark del ScriptedBot (futuro maestro BC) contra
beginner / easy / hard, SIN tocar el server.

¿Para qué sirve?
  El pipeline BC (doc 12, Capa 1) clonará al ScriptedBot para que el agente
  nazca sabiendo proc→harv→barracks→push (resuelve el dolor: harvest 61 vs
  663 del bot C#). Antes de grabar datos BC, este benchmark dice qué tan
  buen maestro es el ScriptedBot y contra quién grabar:
    - si le gana a beginner pero pierde con hard -> grabar BC vs beginner
      (economía limpia, no defensa perdedora).
    - si pierde siempre -> el maestro es malo y hay que usar el bot C# (tocar
      el engine para loguear sus IOrder: "Cambio B" de la doc 12).

Uso:
  python rl/bench_scripted.py                # 5 partidas por dificultad
  python rl/bench_scripted.py --games 3 --diffs beginner hard
  python rl/bench_scripted.py --url http://localhost:8000

No usa modo macro (el maestro juega con step de 2 ticks, igual que el
scripted_bot original). Reporta result + carrera de económica vía
obs.global_summary (own/enemy) cuando el server lo trae.
"""
import argparse
import asyncio
import base64
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from openra_env.client import OpenRAEnv
from openra_env.models import OpenRAAction
from examples.scripted_bot import ScriptedBot


def _gs_side(gs, side: str) -> dict:
    """Extrae un lado del global_summary con defaults."""
    s = (gs or {}).get(side) or {}
    return {
        "cash": s.get("cash", 0) or 0,
        "unit_value": s.get("unit_value", 0) or 0,
        "building_value": s.get("building_value", 0) or 0,
        "n_buildings": s.get("n_buildings", 0) or 0,
        "earned": s.get("earned", 0) or 0,
    }


async def play_one(url: str, bot_type: str | None, seed: int, max_steps: int,
                   verbose: bool, scenario: str | None) -> dict:
    bot = ScriptedBot(verbose=False)
    reset_kwargs: dict = {"seed": seed}
    if bot_type is not None:
        reset_kwargs["bot_type"] = bot_type
    if scenario is not None:
        # Mismo mecanismo que rl/train.py --scenario: pasa map_data para usar
        # un mapa con slot enemigo (el mapa por defecto del contenedor NO lo
        # tiene -> el bot C# no spawna y el bench juega contra el vacio).
        mapa = Path("rl/scenarios") / f"fase2_{scenario.lower()}.oramap"
        if not mapa.exists():
            raise SystemExit(f"escenario inexistente: {mapa}")
        reset_kwargs["map_data"] = base64.b64encode(mapa.read_bytes()).decode()
        reset_kwargs["map_name"] = f"fase2_{scenario.lower()}.oramap"
    async with OpenRAEnv(base_url=url, message_timeout_s=300.0) as env:
        result = await env.reset(**reset_kwargs)
        obs = result.observation
        step = 0
        while not result.done and step < max_steps:
            action: OpenRAAction = bot.decide(result.observation)
            result = await env.step(action)
            step += 1
            obs = result.observation
        gs = getattr(obs, "global_summary", None)
        gs = gs.__dict__ if hasattr(gs, "__dict__") else (gs or {})
        own = _gs_side(gs, "own")
        ene = _gs_side(gs, "enemy")
        # Conteos CRUDOS de la obs (no dependen de global_summary) para
        # diagnosticar si el bot construyó y si spawnó el enemigo.
        n_own_b = len(obs.buildings or [])
        n_own_u = len(obs.units or [])
        n_ene_u = len(obs.visible_enemies or [])
        n_ene_b = len(obs.visible_enemy_buildings or [])
        return {
            "bot_type": bot_type,
            "seed": seed,
            "result": (obs.result or "incomplete") if obs.done else "incomplete",
            "done": obs.done,
            "ticks": obs.tick,
            "own_bv": own["building_value"],
            "own_cash": own["cash"],
            "own_earned": own["earned"],
            "ene_bv": ene["building_value"],
            "ene_cash": ene["cash"],
            "ene_earned": ene["earned"],
            "kills": obs.military.kills_cost,
            "deaths": obs.military.deaths_cost,
            "b_killed": obs.military.buildings_killed,
            "b_lost": obs.military.buildings_lost,
            "n_own_buildings": n_own_b,
            "n_own_units": n_own_u,
            "n_ene_units": n_ene_u,
            "n_ene_buildings": n_ene_b,
        }


def summarize(rows: List[dict]) -> dict:
    if not rows:
        return {}
    wins = sum(1 for r in rows if r["result"].startswith("win"))
    n = len(rows)
    def mean(k):
        return round(sum(r[k] for r in rows) / n, 1)
    return {
        "games": n,
        "winrate": round(wins / n, 2),
        "wins": wins,
        "mean_ticks": mean("ticks"),
        "mean_own_bv": mean("own_bv"),
        "mean_ene_bv": mean("ene_bv"),
        "mean_own_earned": mean("own_earned"),
        "mean_ene_earned": mean("ene_earned"),
        "mean_b_killed": mean("b_killed"),
    }


async def main_async(args):
    diffs = args.diffs
    print(f"Benchmark ScriptedBot vs {diffs} "
          f"({args.games} partidas c/u, max_steps={args.max_steps})")
    print(f"URL: {args.url}\n")
    all_rows: List[dict] = []
    for diff in diffs:
        # "container" = no pasar bot_type (hereda BOT_TYPE del contenedor,
        # reset() vacío como examples/scripted_bot.py).
        bt = None if diff == "container" else diff
        rows: List[dict] = []
        for g in range(args.games):
            seed = args.base_seed + hash((diff, g)) % 100000
            t0 = time.time()
            try:
                r = await play_one(args.url, bt, seed, args.max_steps,
                                   args.verbose, args.scenario)
            except Exception as e:
                r = {"bot_type": diff, "seed": seed, "result": f"ERROR:{e}",
                     "ticks": 0, "own_bv": 0, "own_cash": 0, "own_earned": 0,
                     "ene_bv": 0, "ene_cash": 0, "ene_earned": 0,
                     "kills": 0, "deaths": 0, "b_killed": 0, "b_lost": 0,
                     "n_own_buildings": 0, "n_own_units": 0,
                     "n_ene_units": 0, "n_ene_buildings": 0}
                print(f"  [{diff} #{g+1}] ERROR: {e}")
            rows.append(r)
            dt = time.time() - t0
            print(f"  [{diff} #{g+1}] {r['result']:10s} tick={r['ticks']:6d} "
                  f"ownB={r['n_own_buildings']} ownU={r['n_own_units']} "
                  f"eneU={r['n_ene_units']} eneB={r['n_ene_buildings']} "
                  f"eneBV={r['ene_bv']:.0f}(fogless) bKill={r['b_killed']:.0f} ({dt:.0f}s)")
        s = summarize(rows)
        print(f"  -> {diff}: winrate={s.get('winrate')} "
              f"({s.get('wins')}/{s.get('games')}) "
              f"meanTicks={s.get('mean_ticks')} "
              f"ownBV={s.get('mean_own_bv')} vs eneBV={s.get('mean_ene_bv')} "
              f"ownEarned={s.get('mean_own_earned')} vs eneEarned={s.get('mean_ene_earned')}\n")
        all_rows.extend(rows)
    out = {"per_diff": {}, "raw": all_rows}
    by_diff: Dict[str, list] = defaultdict(list)
    for r in all_rows:
        by_diff[r["bot_type"]].append(r)
    for d, rs in by_diff.items():
        out["per_diff"][d] = summarize(rs)
    print("=== RESUMEN ===")
    print(json.dumps(out["per_diff"], indent=2))
    # Guardar para referencia
    try:
        with open("rl/ckpts/bench_scripted.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("\nguardado en rl/ckpts/bench_scripted.json")
    except Exception as e:
        print(f"(no se pudo guardar json: {e})")


def main():
    ap = argparse.ArgumentParser(description="Benchmark ScriptedBot vs dificultades")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--games", type=int, default=5, help="partidas por dificultad")
    ap.add_argument("--diffs", nargs="+",
                    default=["beginner", "easy", "hard"],
                    help="dificultades: beginner/easy/hard (reset bot_type) "
                         "o 'container' (reset() vacio, hereda BOT_TYPE)")
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--scenario", default=None,
                    help="usa rl/scenarios/fase2_<X>.oramap (con slot enemigo). "
                         "Sin esto el mapa por defecto NO spawna bot C#.")
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
