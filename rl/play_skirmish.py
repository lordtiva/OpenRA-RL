#!/usr/bin/env python3
"""
play_skirmish.py — el PPO se engancha al OpenRA de escritorio como bot de lobby.

El engine GUI (launch-game.cmd) ya lista "PPO Agent" junto a beginner/easy.
Cuando ese slot arranca, ExternalBotBridge abre gRPC :9999, pausa, y (si
encontró el venv) lanza este script. El mundo corre a 25 tps: FastAdvance
con ticks=0 solo inyecta órdenes, no acelera.

Uso (PowerShell, desde la raiz; PYTHONPATH vacío):

    # A) Solo el juego: make.cmd all + launch-game.cmd Game.Mod=ra
    #    Skirmish → oponente "PPO Agent". El engine spawnea este script.

    # B) Sidecar a mano (si desactivaste el autostart):
    $env:PYTHONPATH=""
    $env:OPENRA_RL_AUTOSTART="0"
    .\\.venv\\Scripts\\python.exe -m rl.play_skirmish --attach --ckpt rl/ckpts/best.pt

El agente se entrenó en a_short/Singles. En otro mapa corre (la red es
agnóstica al H×W) pero el dest/beacon es el de Singles.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from openra_env.models import (
    BuildingInfoModel,
    CommandModel,
    EconomyInfo,
    MapInfoModel,
    MilitaryInfo,
    OpenRAAction,
    OpenRAObservation,
    ProductionInfoModel,
    UnitInfoModel,
)
from openra_env.server.bridge_client import (
    BridgeClient,
    commands_to_proto,
    observation_to_dict,
)
from rl.action_adapter import Vocab, index_to_command_effective
from rl.auto_support import apply_dest_credit, support_commands
from rl.network import ACTION_TYPES, HIDDEN_DIM, AlphaLiteNet
from rl.rollout import _batch_of
from rl.trainer import load_checkpoint


def pick_device(req: str) -> str:
    if req != "auto":
        return req
    return "cuda" if torch.cuda.is_available() else "cpu"


def action_to_dicts(action: OpenRAAction) -> list[dict]:
    """OpenRAAction -> list[dict] for commands_to_proto."""
    out = []
    for c in action.commands or []:
        act = c.action
        name = act.value if hasattr(act, "value") else str(act)
        out.append({
            "action": name,
            "actor_id": int(getattr(c, "actor_id", 0) or 0),
            "target_actor_id": int(getattr(c, "target_actor_id", 0) or 0),
            "target_x": int(getattr(c, "target_x", 0) or 0),
            "target_y": int(getattr(c, "target_y", 0) or 0),
            "item_type": str(getattr(c, "item_type", "") or ""),
            "queued": bool(getattr(c, "queued", False)),
        })
    return out


def obs_from_dict(d: dict) -> OpenRAObservation:
    """observation_to_dict output -> OpenRAObservation (skirmish attach)."""
    return OpenRAObservation(
        tick=int(d.get("tick") or 0),
        economy=EconomyInfo(**(d.get("economy") or {})),
        military=MilitaryInfo(**(d.get("military") or {})),
        units=[UnitInfoModel(**u) for u in (d.get("units") or [])],
        buildings=[BuildingInfoModel(**b) for b in (d.get("buildings") or [])],
        production=[ProductionInfoModel(**p) for p in (d.get("production") or [])],
        visible_enemies=[UnitInfoModel(**u) for u in (d.get("visible_enemies") or [])],
        visible_enemy_buildings=[
            BuildingInfoModel(**b) for b in (d.get("visible_enemy_buildings") or [])
        ],
        map_info=MapInfoModel(**(d.get("map_info") or {})),
        available_production=list(d.get("available_production") or []),
        done=bool(d.get("done")),
        reward=float(d.get("reward") or 0.0),
        result=str(d.get("result") or ""),
        spatial_map=str(d.get("spatial_map") or ""),
        spatial_channels=int(d.get("spatial_channels") or 0),
    )


def proto_to_obs(pb) -> OpenRAObservation:
    return obs_from_dict(observation_to_dict(pb))


def wait_for_match(client: BridgeClient, timeout_s: float | None = None) -> bool:
    """Poll GetState until the bridge is in 'playing' (match started)."""
    t0 = time.time()
    n = 0
    while True:
        try:
            if not client.is_connected:
                client.connect()
            st = client.get_state()
            phase = str(getattr(st, "phase", "") or "")
            if phase == "playing":
                print(f"[skirmish] bridge playing tick={st.tick} session={st.episode_id}",
                      flush=True)
                return True
            if phase == "game_over":
                print("[skirmish] game_over before first step", flush=True)
                return False
            if n % 10 == 0:
                print(f"[skirmish] esperando partida… phase={phase or 'no_bridge'}",
                      flush=True)
        except Exception as e:
            if n % 10 == 0:
                print(f"[skirmish] gRPC aún no listo: {e}", flush=True)
        n += 1
        if timeout_s is not None and (time.time() - t0) >= timeout_s:
            print("[skirmish] timeout esperando la partida", flush=True)
            return False
        time.sleep(0.5)


def decide(net, vocab, device, hidden, obs, args, last_push_cell):
    """One policy + auto_support decision. Returns action, hidden, last_push, atype."""
    batch, aidx = _batch_of(obs, vocab, device)
    with torch.no_grad():
        out = net.act(batch, hidden, temperature=args.temperature)
    hidden = out["hidden"].detach()
    action, (eff_t, _eff_u, _eff_i, eff_c) = index_to_command_effective(
        obs, int(out["type"]), int(out["unit_slot"]),
        int(out["cell_flat"]), int(out["item_slot"]), aidx)
    atype_str = ACTION_TYPES[eff_t]
    H = max(int(obs.map_info.height or 1), 1)
    W = max(int(obs.map_info.width or 1), 1)
    for c in action.commands:
        if c.target_x >= W or c.target_y >= H:
            c.target_x = min(int(c.target_x), W - 1)
            c.target_y = min(int(c.target_y), H - 1)
    if args.auto_support:
        new_c, _xy = apply_dest_credit(
            obs, action, atype_str, int(eff_c), aidx, last_push=last_push_cell)
        eff_c = int(new_c)
        if atype_str in ("army_attack_move", "attack_move") and action.commands:
            c0 = action.commands[0]
            if getattr(c0, "target_x", None) is not None:
                last_push_cell = (int(c0.target_x), int(c0.target_y))
        for cmd in support_commands(obs, last_push=last_push_cell, aidx=aidx, war_nudge=not getattr(args, "no_war_nudge", False)):
            action.commands.append(cmd)
    return action, hidden, last_push_cell, atype_str


def play_loop(client: BridgeClient, net, vocab, device, args) -> str:
    hidden = torch.zeros(1, HIDDEN_DIM, device=device)
    last_push_cell = None
    last_decision_tick = -10**9
    pending: list = []
    result = ""
    while True:
        st = client.get_state()
        phase = str(getattr(st, "phase", "") or "")
        if phase == "game_over":
            result = str(getattr(st, "winner", "") or "game_over")
            break
        if phase != "playing":
            time.sleep(0.2)
            continue
        tick = int(getattr(st, "tick", 0) or 0)
        if tick - last_decision_tick < args.macro_ticks:
            time.sleep(0.04)
            continue
        pb = client.fast_advance_unary(0, pending)
        pending = []
        obs = proto_to_obs(pb)
        if pb.done:
            result = str(pb.result or "done")
            break
        last_decision_tick = int(obs.tick)
        action, hidden, last_push_cell, atype = decide(
            net, vocab, device, hidden, obs, args, last_push_cell)
        pending = list(commands_to_proto(action_to_dicts(action)).commands)
        cash = getattr(obs.economy, "cash", 0)
        print(
            f"[skirmish] t={obs.tick} {atype} units={len(obs.units or [])} "
            f"cash={cash} cmds={len(pending)}",
            flush=True,
        )
    return result


def main():
    p = argparse.ArgumentParser(description="PPO sidecar for OpenRA skirmish")
    p.add_argument("--attach", action="store_true",
                   help="Wait for the local GUI bridge (default and only mode).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=10001,
                   help="GUI skirmish gRPC (10001). Train/docker keeps 9999.")
    p.add_argument("--ckpt", default="rl/ckpts/best.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--macro-ticks", type=int, default=80)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true",
                   help="temperature=0 (deterministic). Default is sample, like train.")
    p.add_argument("--no-auto-support", action="store_true")
    p.add_argument("--no-war-nudge", action="store_true")
    p.add_argument("--wait-timeout", type=float, default=0.0,
                   help="Seconds to wait for the match (0 = forever).")
    args = p.parse_args()
    args.auto_support = not args.no_auto_support
    if args.greedy:
        args.temperature = 0.0

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint no existe: {ckpt}")

    device = pick_device(args.device)
    vocab = Vocab()
    net = AlphaLiteNet().to(device)
    net.eval()
    extra = {}
    it = load_checkpoint(str(ckpt), net, vocab=vocab, extra_out=extra)
    print(f"[skirmish] ckpt={ckpt} iter={it} device={device} T={args.temperature}",
          flush=True)

    client = BridgeClient(host=args.host, port=args.port, timeout_s=30.0)
    wait_s = None if args.wait_timeout <= 0 else float(args.wait_timeout)
    if not wait_for_match(client, timeout_s=wait_s):
        raise SystemExit(2)
    result = play_loop(client, net, vocab, device, args)
    print(f"[skirmish] fin result={result}", flush=True)
    try:
        client.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
