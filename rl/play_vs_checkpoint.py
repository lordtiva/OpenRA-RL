#!/usr/bin/env python3
"""
play_vs_checkpoint.py — Enfrenta un checkpoint PPO contra el bot y lo miras
en el navegador (Docker + noVNC).

El server OpenRA corre headless en Docker; este script es el AGENTE que
carga el .pt y juega. Vos MIRAS la partida con:

    openra-rl replay watch          # abre http://localhost:6080 en el navegador
    # o:  openra-rl replay watch --port 6080

Uso (PowerShell, desde la raiz del repo):

    # 1) Levantar el server (una vez)
    openra-rl server start
    openra-rl server status

    # 2) Jugar 1 partida vs beginner con el checkpoint más nuevo
    $env:PYTHONPATH=""
    python -m rl.play_vs_checkpoint --ckpt rl/ckpts/latest.pt

    # 3) Mientras juega, en OTRA terminal abrir el visor:
    openra-rl replay watch

    # Opciones útiles:
    python -m rl.play_vs_checkpoint --ckpt rl/ckpts/iter0010.pt --bot-type hard --episodes 3 --greedy
    python -m rl.play_vs_checkpoint --ckpt rl/ckpts/latest.pt --scenario a --bot-type beginner
    python -m rl.play_vs_checkpoint --ckpt rl/ckpts/latest.pt --temperature 0.0  # greedy
    python -m rl.play_vs_checkpoint --ckpt rl/ckpts/latest.pt --url http://localhost:8000 --verbose

Notas:
  - --scenario a usa rl/scenarios/fase2_a.oramap (tiene slot enemigo; el mapa
    por defecto del contenedor NO spawna bot C# -> partida vacía).
  - --greedy equivale a --temperature 0 (determinístico, mejor para ver qué
    sabe hacer el checkpoint).
  - El replay (.orarep) se guarda en el contenedor y en ~/.openra-rl/replays/
    si tenés RECORD_REPLAYS=true. `openra-rl replay list` los lista.
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

import torch

# Permitir `python -m rl.play_vs_checkpoint` y `python rl/play_vs_checkpoint.py`
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openra_env.client import OpenRAEnv
from rl.action_adapter import Vocab
from rl.network import AlphaLiteNet
from rl.rollout import collect_one_episode
from rl.trainer import load_checkpoint


def pick_device(req: str) -> str:
    if req != "auto":
        return req
    return "cuda" if torch.cuda.is_available() else "cpu"


async def play_one(env: OpenRAEnv, net, vocab, device, args, ep_idx: int):
    reset_kwargs: dict = {}
    if args.scenario:
        mapa = Path(f"rl/scenarios/fase2_{args.scenario.lower()}.oramap")
        if not mapa.exists():
            # fallback: probar sin prefijo fase2_
            alt = Path(f"rl/scenarios/{args.scenario}.oramap")
            if alt.exists():
                mapa = alt
            else:
                raise SystemExit(f"Escenario no encontrado: {mapa} (ni {alt})")
        reset_kwargs["map_data"] = base64.b64encode(mapa.read_bytes()).decode()
        reset_kwargs["map_name"] = mapa.name
    if args.bot_type:
        reset_kwargs["bot_type"] = args.bot_type
    # seed distinto por episodio para no repetir partida
    reset_kwargs["seed"] = args.seed + ep_idx * 1009 if args.seed is not None else None
    # limpiar None
    reset_kwargs = {k: v for k, v in reset_kwargs.items() if v is not None}

    if reset_kwargs and args.verbose:
        print(f"  reset_kwargs: { {k: (len(v) if k=='map_data' else v) for k,v in reset_kwargs.items()} }")

    t0 = time.time()
    traj, outcome = await collect_one_episode(
        env, net, vocab, device,
        k_skip=args.k_skip,
        temperature=args.temperature,
        max_steps=args.max_steps,
        macro_ticks=args.macro_ticks,
        reset_kwargs=reset_kwargs,
        shaper_preset=args.shaper_preset,
        auto_support=args.auto_support,
    )
    dt = time.time() - t0
    return traj, outcome, dt


async def amain(args):
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        # Permitir ruta relativa aunque el PS esté en C:\Users\lordc
        repo_root = Path(__file__).resolve().parent.parent
        alt = repo_root / args.ckpt
        if alt.exists():
            ckpt_path = alt
        else:
            raise SystemExit(f"Checkpoint no encontrado: {args.ckpt} (probado: {ckpt_path.resolve()} y {alt}) — corre desde C:\\Users\\lordc\\Desktop\\OpenRA-RL o usa ruta absoluta")
    # also try rl/ckpts/ prefix if bare name
    device = pick_device(args.device)
    print(f"Device: {device} | ckpt: {ckpt_path}")

    net = AlphaLiteNet()
    vocab = Vocab()
    it = load_checkpoint(str(ckpt_path), net, vocab=vocab)
    net.to(device)
    net.eval()
    print(f"Checkpoint iter {it} | vocab {len(vocab.type_to_id)} tipos | params {sum(p.numel() for p in net.parameters())/1e6:.2f}M")
    if args.verbose:
        # mostrar algunos tipos del vocab para debug
        sample = list(vocab.type_to_id.items())[:12]
        print(f"  vocab sample: {sample}")

    # temperatura
    if args.greedy:
        args.temperature = 0.0
    print(f"Server: {args.url} | bot={args.bot_type or 'container default'} "
          f"scenario={args.scenario or 'default'} episodes={args.episodes} "
          f"T={args.temperature} k_skip={args.k_skip} macro={args.macro_ticks} "
          f"max_steps={args.max_steps}")

    ws_timeout = max(60.0, args.max_steps * 3.0)
    env = OpenRAEnv(base_url=args.url, message_timeout_s=ws_timeout)
    try:
        await env.connect()
    except Exception as e:
        print(f"\n[ERROR] No pude conectar a {args.url}: {e}")
        print("  ¿Está el server levantado?")
        print("    openra-rl server start")
        print("    openra-rl server status")
        print("    openra-rl server logs")
        raise SystemExit(1)

    print(f"Conectado a {args.url} (ws_timeout={ws_timeout:.0f}s)")
    print(f"\n>>> Mientras juega, abrí en OTRA terminal:  openra-rl replay watch")
    print(f"    (abre http://localhost:6080 en el navegador para ver la partida)\n")

    wins = 0
    for ep in range(1, args.episodes + 1):
        print(f"--- Episodio {ep}/{args.episodes} ---")
        try:
            traj, outcome, dt = await play_one(env, net, vocab, device, args, ep)
        except Exception as e:
            print(f"  [ERROR episodio {ep}] {e}")
            # intentar sanear la sesión con un reset best-effort
            try:
                await asyncio.wait_for(env.reset(), timeout=20)
            except Exception:
                pass
            continue

        result = outcome.get("result", "?")
        is_win = str(result).startswith("win")
        wins += 1 if is_win else 0
        ticks = outcome.get("ticks", 0)
        decs = outcome.get("decisions", len(traj))
        rew = outcome.get("episode_reward", 0)
        sup = outcome.get("supremacy", {})
        hist = outcome.get("action_hist", {})
        n_bld = outcome.get("n_buildings", {})
        adv = outcome.get("advanced_ticks", 0)
        print(f"  result={result} ticks={ticks} decs={decs} reward={rew} wall={dt:.1f}s "
              f"adv={adv} buildings={n_bld} sup={sup}")
        if hist:
            # top 5 acciones
            top = sorted(hist.items(), key=lambda x: -x[1])[:6]
            print(f"  hist: {dict(top)}")
        rc = outcome.get("reward_components", {})
        if rc:
            print(f"  reward_comp: {rc}")

        # si el server guarda replay, avisar dónde mirarlo
        # el path real está en el contenedor; el usuario lo ve con `replay watch`
        if ep < args.episodes:
            await asyncio.sleep(1.0)

    print(f"\n=== Resumen: {wins}/{args.episodes} wins ===")
    print("Para ver el replay en el navegador:")
    print("  openra-rl replay watch          # último replay")
    print("  openra-rl replay list           # listar replays")
    print("  openra-rl replay copy           # copiar a ~/.openra-rl/replays/")

    try:
        await env.close()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Jugar vs checkpoint y mirar en navegador (Docker noVNC)")
    ap.add_argument("--ckpt", default="rl/ckpts/latest.pt", help="ruta al .pt (default: rl/ckpts/latest.pt)")
    ap.add_argument("--url", default="http://localhost:8000", help="URL del server (default: http://localhost:8000)")
    ap.add_argument("--bot-type", default="beginner", help="dificultad rival: beginner/easy/hard/brutal/rush/turtle (default: beginner, ''=deshabilitar)")
    ap.add_argument("--scenario", default=None, help="escenario fase2 (ej: a, a_short, amin160_short). Usa rl/scenarios/fase2_<X>.oramap con slot enemigo")
    ap.add_argument("--episodes", type=int, default=1, help="cuántas partidas jugar (default: 1)")
    ap.add_argument("--seed", type=int, default=42, help="seed base (se suma offset por episodio)")
    ap.add_argument("--temperature", type=float, default=1.0, help="temperatura de muestreo (0=greedy)")
    ap.add_argument("--greedy", action="store_true", help="atajo para --temperature 0 (determinístico)")
    ap.add_argument("--k-skip", type=int, default=8, help="frame-skip (default: 8)")
    ap.add_argument("--macro-ticks", type=int, default=0, help="ticks por decisión en modo macro (0=deshabilitado, ej: 160)")
    ap.add_argument("--max-steps", type=int, default=4000, help="máx decisiones por episodio (default: 4000)")
    ap.add_argument("--shaper-preset", default="eradicate", choices=["eradicate", "eradicate_v3", "standard"], help="preset de reward shaping")
    ap.add_argument("--auto-support", action="store_true", help="activa soporte automático (harvest/deploy gratis)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="device")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    # bot_type "" deshabilita enemigo
    if args.bot_type == "":
        args.bot_type = None
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
