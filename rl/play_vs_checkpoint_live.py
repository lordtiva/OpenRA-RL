#!/usr/bin/env python3
"""
play_vs_checkpoint_live.py — visor EN VIVO del checkpoint en el navegador.

Por default replica el recipe de auto_train (a_short / beginner / eradicate_v4 /
auto-support / macro 80 / max-steps 624) y sirve el canvas en :8786.
Recarga latest.pt entre partidas si el train lo actualizó.

Uso (PowerShell, UN comando):
    cd C:/Users/lordc/Desktop/OpenRA-RL
    $env:PYTHONPATH=""
    .\.venv\Scripts\python.exe -m rl.play_vs_checkpoint_live

    # visor:
    http://localhost:8786/

Ctrl+C corta el live, no el train.
"""
import argparse
import asyncio
import base64
import time
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from openra_env.client import OpenRAEnv
from rl.action_adapter import Vocab
from rl.network import AlphaLiteNet
from rl.trainer import load_checkpoint
from rl.live_server import LiveBroadcaster
from rl.obs_encoding import BEACON_BY_MAP, decode_spatial
from rl.rollout import _batch_of
from rl.action_adapter import index_to_command_effective
from openra_env.models import ActionType, CommandModel, OpenRAAction
from rl.reward_shaping import PRESETS, ShapedReward
from rl.supremacy import evaluate_supremacy
from rl.network import ACTION_TYPES, HIDDEN_DIM
from rl.auto_support import support_commands


def pick_device(req: str) -> str:
    if req != "auto":
        return req
    return "cuda" if torch.cuda.is_available() else "cpu"


def _obs_to_live_state(obs, beacon, hist, decs, rew, adv_ticks, last_action_str, status, done, result):
    """Convierte observación a dict liviano para el visor."""
    H = obs.map_info.height or 64
    W = obs.map_info.width or 64
    # intentar extraer recursos / niebla del spatial (muestreo)
    resources = []
    fog = []
    try:
        sp = decode_spatial(obs.spatial_map, H, W, obs.spatial_channels or 9, beacon=beacon)
        if sp is not None:
            # ch2 recurso >0.3, ch4 niebla==1
            import numpy as np
            ch2 = sp[2]
            ch4 = sp[4]
            # muestrear cada 2 para no mandar 4k puntos
            for y in range(0, H, 1):
                for x in range(0, W, 1):
                    if ch2[y, x] > 0.3:
                        resources.append([x, y])
                    if ch4[y, x] > 0.8:
                        fog.append([x, y])
            # cap para payload
            if len(resources) > 800:
                resources = resources[:: max(1, len(resources)//800)]
            if len(fog) > 2000:
                fog = fog[:: max(1, len(fog)//2000)]
    except Exception:
        pass

    sup = None
    try:
        gs = getattr(obs, "global_summary", None)
        ev = evaluate_supremacy(obs, gs=gs if isinstance(gs, dict) else None)
        sup = ev
    except Exception:
        pass

    return {
        "tick": obs.tick,
        "map_w": W, "map_h": H,
        "cash": getattr(obs.economy, "cash", 0),
        "ore": getattr(obs.economy, "ore", 0),
        "cap": getattr(obs.economy, "resource_capacity", 0),
        "power_provided": getattr(obs.economy, "power_provided", 0),
        "power_drained": getattr(obs.economy, "power_drained", 0),
        "power": f"{getattr(obs.economy,'power_provided',0)}/{getattr(obs.economy,'power_drained',0)}",
        "n_units": len(obs.units or []),
        "n_bld": len(obs.buildings or []),
        "n_ene_u": len(obs.visible_enemies or []),
        "n_ene_b": len(obs.visible_enemy_buildings or []),
        "buildings": [{"x": b.cell_x, "y": b.cell_y, "hp": getattr(b, "hp_percent", 1.0)} for b in (obs.buildings or [])],
        "units": [{"x": u.cell_x, "y": u.cell_y, "can_attack": getattr(u, "can_attack", True)} for u in (obs.units or [])],
        "ene_buildings": [{"x": b.cell_x, "y": b.cell_y} for b in (obs.visible_enemy_buildings or [])],
        "ene_units": [{"x": u.cell_x, "y": u.cell_y} for u in (obs.visible_enemies or [])],
        "resources": resources,
        "fog": fog,
        "beacon": beacon,
        "decisions": decs,
        "episode_reward": round(rew, 2) if rew is not None else None,
        "advanced_ticks": adv_ticks,
        "last_action": last_action_str,
        "hist": hist,
        "sup_own": round(sup["own"], 0) if sup else None,
        "sup_ene": round(sup["enemy"], 0) if sup else None,
        "sup_diff": round(sup["diff"], 0) if sup else None,
        "result": result,
        "status": status,
        "done": bool(done),
    }


async def run_episode_live(env: OpenRAEnv, net, vocab, device, args, broadcaster: LiveBroadcaster):
    reset_kwargs = {}
    if args.scenario:
        mapa = Path(f"rl/scenarios/fase2_{args.scenario.lower()}.oramap")
        if not mapa.exists():
            alt = Path(f"rl/scenarios/{args.scenario}.oramap")
            if alt.exists():
                mapa = alt
            else:
                raise SystemExit(f"Escenario no encontrado: {mapa}")
        reset_kwargs["map_data"] = base64.b64encode(mapa.read_bytes()).decode()
        reset_kwargs["map_name"] = mapa.name
    # bot_type="" -> dummy (pasivo), ai_slot="" -> sin enemigo. No filtrar "".
    if args.bot_type is not None:
        reset_kwargs["bot_type"] = args.bot_type
    if hasattr(args, "ai_slot") and args.ai_slot is not None:
        reset_kwargs["ai_slot"] = args.ai_slot
    reset_kwargs["seed"] = args.seed

    beacon = BEACON_BY_MAP.get(reset_kwargs.get("map_name")) if reset_kwargs.get("map_name") else None

    result = await env.reset(**reset_kwargs)
    obs = result.observation
    hidden = torch.zeros(1, HIDDEN_DIM, device=device)
    shaper = ShapedReward(preset=args.shaper_preset)
    shaper.reset(obs)

    hist = {}
    decs = 0
    episode_reward = 0.0
    adv_total = 0
    done = False
    last_action_str = "—"
    macro_final = None
    last_push_cell = None

    # estado inicial
    broadcaster.update(_obs_to_live_state(obs, beacon, hist, decs, episode_reward, adv_total, last_action_str, "jugando…", done, ""))

    use_macro = args.macro_ticks > 0
    for step in range(args.max_steps):
        can_decide = use_macro or step % args.k_skip == 0
        action = None
        atype_str = "no_op"
        if can_decide:
            batch, aidx = _batch_of(obs, vocab, device)
            h_in = hidden.detach().clone()
            with torch.no_grad():
                out = net.act(batch, hidden, temperature=args.temperature)
            hidden = out["hidden"].detach()
            had_item = aidx.item_mask.any().view(1).to(device)
            action, (eff_t, eff_u, eff_i) = index_to_command_effective(
                obs, int(out["type"]), int(out["unit_slot"]), int(out["cell_flat"]), int(out["item_slot"]), aidx)
            # recalc log_prob si hubo coerción (igual que rollout, pero sin grad)
            sampled = (int(out["type"]), int(out["unit_slot"]), int(out["item_slot"]))
            effective = (eff_t, eff_u, eff_i)
            if sampled != effective:
                with torch.no_grad():
                    re_lp, _, _ = net.evaluate_actions(batch, h_in, {
                        "type": torch.tensor([eff_t], device=device),
                        "unit_slot": torch.tensor([eff_u], device=device),
                        "cell_flat": out["cell_flat"],
                        "item_slot": torch.tensor([eff_i], device=device),
                        "had_item": had_item,
                    })
            atype_str = ACTION_TYPES[eff_t]
            hist[atype_str] = hist.get(atype_str, 0) + 1
            decs += 1
            # texto acción
            item_name = aidx.items[int(out["item_slot"])] if int(out["item_slot"]) < len(aidx.items) else "—"
            last_action_str = f"{atype_str}  cell={int(out['cell_flat'])%aidx.w},{int(out['cell_flat'])//aidx.w}  item={item_name}  units={len(obs.units)} cash={obs.economy.cash}"
            # clamp
            ep_dims = (obs.map_info.height, obs.map_info.width)
            for c in action.commands:
                if c.target_x >= ep_dims[1] or c.target_y >= ep_dims[0]:
                    c.target_x = min(c.target_x, ep_dims[1]-1)
                    c.target_y = min(c.target_y, ep_dims[0]-1)
            # Pilar B: auto-harvest/repair gratis (no roba decisión PPO)
            if atype_str in ("army_attack_move", "attack_move") and action.commands:
                c0 = action.commands[0]
                if getattr(c0, "target_x", None) is not None:
                    last_push_cell = (int(c0.target_x), int(c0.target_y))
            if args.auto_support:
                for cmd in support_commands(obs, last_push=last_push_cell):
                    action.commands.append(cmd)
        else:
            # mantener último comando (frame-skip)
            action = action  # type: ignore

        if action is None:
            action = OpenRAAction(commands=[CommandModel(action=ActionType.NO_OP)])

        try:
            result = await env.step(action)
        except RuntimeError as e:
            # degradar a NO_OP
            try:
                result = await env.step(OpenRAAction(commands=[CommandModel(action=ActionType.NO_OP)]))
            except Exception:
                break
        obs = result.observation
        r_frame = shaper.step(obs, done=bool(result.done), action_type=atype_str)
        episode_reward += r_frame
        done = bool(result.done)

        # macro advance
        if use_macro and not done:
            restante = max(0, args.macro_ticks - 2)
            try:
                while restante > 0 and not done:
                    adv = await env.advance(min(50, restante))
                    adv_total += int(adv.get("actual_ticks_advanced", 0) or 0)
                    done = bool(adv.get("done", False))
                    if done:
                        macro_final = adv.get("result")
                    restante -= int(adv.get("actual_ticks_advanced", 0) or 0)
                if not done:
                    result = await env.step(OpenRAAction(commands=[CommandModel(action=ActionType.NO_OP)]))
                    obs = result.observation
                    done = bool(result.done)
                    r_close = shaper.step(obs, done=done, action_type=atype_str, closing=True)
                    episode_reward += r_close
            except Exception as e:
                if "DEADLINE" in str(e):
                    broadcaster.update({"status": "DEADLINE — sesión envenenada, abortando"})
                    break

        # push live cada decisión (throttle: solo si can_decide para no spamear)
        if can_decide:
            broadcaster.update(_obs_to_live_state(obs, beacon, hist, decs, episode_reward, adv_total, last_action_str, f"dec {decs} tick {obs.tick}", done, getattr(obs, "result", "") or macro_final or ""))

        if done:
            break
        # pequeño yield para que el http server respire
        await asyncio.sleep(0)

    r_final = shaper.finalize(truncated=not done, result=str(getattr(obs, "result", "") or macro_final or ""))
    episode_reward += r_final
    final_result = getattr(obs, "result", None) or macro_final or ("incomplete" if not done else "")
    broadcaster.update(_obs_to_live_state(obs, beacon, hist, decs, episode_reward, adv_total, last_action_str, f"final: {final_result}", True, final_result))
    return {"result": final_result, "ticks": obs.tick, "decisions": decs, "episode_reward": round(episode_reward, 3), "hist": hist, "advanced_ticks": adv_total}


async def amain(args):
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        repo_root = Path(__file__).resolve().parent.parent
        alt = repo_root / args.ckpt
        if alt.exists():
            ckpt_path = alt
        else:
            raise SystemExit(f"Checkpoint no encontrado: {args.ckpt}")

    device = pick_device(args.device)
    print(f"Device: {device} | ckpt: {ckpt_path}")
    net = AlphaLiteNet()
    vocab = Vocab()
    it = load_checkpoint(str(ckpt_path), net, vocab=vocab)
    net.to(device)
    net.eval()
    print(f"Checkpoint iter {it} | vocab {len(vocab.type_to_id)} tipos | params {sum(p.numel() for p in net.parameters())/1e6:.2f}M")
    if args.greedy:
        args.temperature = 0.0
    print(f"Server: {args.url} | bot={args.bot_type} scenario={args.scenario} T={args.temperature} macro={args.macro_ticks}")

    bc = LiveBroadcaster(port=args.port)
    bc.start()
    print(f"Abrí en el navegador: http://localhost:{args.port}/  (se actualiza solo)")

    ws_timeout = max(60.0, args.max_steps * 3.0)
    env = OpenRAEnv(base_url=args.url, message_timeout_s=ws_timeout)
    try:
        await env.connect()
    except Exception as e:
        print(f"[ERROR] No pude conectar a {args.url}: {e}")
        print("  openra-rl: .\\.venv\\Scripts\\openra-rl.exe server start")
        raise SystemExit(1)
    print(f"Conectado a {args.url}")

    last_mtime = ckpt_path.stat().st_mtime if ckpt_path.exists() else 0.0
    ep = 0
    try:
        while True:
            ep += 1
            if args.episodes > 0 and ep > args.episodes:
                break
            try:
                mtime = ckpt_path.stat().st_mtime
            except OSError:
                mtime = last_mtime
            if mtime != last_mtime:
                it = load_checkpoint(str(ckpt_path), net, vocab=vocab)
                net.to(device)
                net.eval()
                last_mtime = mtime
                print(f"Recargué ckpt iter {it}")
            label = str(ep) if args.episodes <= 0 else f"{ep}/{args.episodes}"
            print(f"\n=== Episodio {label} ===")
            bc.update({"status": f"episodio {label} — iniciando…", "done": False, "ckpt_iter": it})
            outcome = await run_episode_live(env, net, vocab, device, args, bc)
            print(f"  result={outcome['result']} ticks={outcome['ticks']} decs={outcome['decisions']} rew={outcome['episode_reward']} hist={outcome['hist']}")
        print(f"\nVisor sigue en http://localhost:{args.port}/  (Ctrl+C para salir)")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            await env.close()
        except Exception:
            pass
        bc.stop()


def main():
    ap = argparse.ArgumentParser(description="Checkpoint vs bot con visor EN VIVO en http://localhost:8786/")
    ap.add_argument("--ckpt", default="rl/ckpts/latest.pt")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--bot-type", default="beginner")
    ap.add_argument("--ai-slot", default=None, help='slot IA: "Multi0" (default) o "" para sin enemigo')
    ap.add_argument("--scenario", default="a_short")
    ap.add_argument("--episodes", type=int, default=0, help="0 = loop infinito; recarga latest.pt entre partidas")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--k-skip", type=int, default=8)
    ap.add_argument("--macro-ticks", type=int, default=80)
    ap.add_argument("--max-steps", type=int, default=624)
    ap.add_argument("--shaper-preset", default="eradicate_v4", choices=list(PRESETS))
    ap.add_argument("--auto-support", action=argparse.BooleanOptionalAction, default=True,
                    help="harvest/repair/power automático (Pilar B); default on como el train")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--port", type=int, default=8786, help="puerto del visor live (default 8786)")
    args = ap.parse_args()
    # --bot-type "" mantiene "" (dummy), solo None es "no tocar". --ai-slot "" desactiva enemigo.
    if args.bot_type == "__none__":
        args.bot_type = None
    asyncio.run(amain(args))

if __name__ == "__main__":
    main()
