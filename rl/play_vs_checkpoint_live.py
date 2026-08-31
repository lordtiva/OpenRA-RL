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

Al terminar cada partida append a rl/ckpts/live_games.jsonl (histograma,
destino de asalto, centroide vs beacon, tape cada 10 decs). Cada decisión
también va a rl/ckpts/live_tape.jsonl (política, dest de soporte, harv xy,
centroide). Ctrl+C no pisa el train.
"""
import argparse
import asyncio
import base64
import json
import time
from datetime import datetime
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
from rl.auto_support import apply_dest_credit, support_commands


def pick_device(req: str) -> str:
    if req != "auto":
        return req
    return "cuda" if torch.cuda.is_available() else "cpu"


def _tally_types(objs) -> dict:
    c: dict[str, int] = {}
    for o in objs or []:
        if isinstance(o, dict):
            k = str(o.get("type") or "?")
        else:
            k = str(getattr(o, "type", "") or "?")
        c[k] = c.get(k, 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def _unit_xy(u):
    try:
        return [int(u.cell_x), int(u.cell_y)]
    except (TypeError, ValueError):
        return None


def _is_noncombat_type(typ: str) -> bool:
    t = (typ or "").lower()
    return "harv" in t or "mcv" in t


def _combat_centroid(units):
    pts = []
    for u in units or []:
        if _is_noncombat_type(str(getattr(u, "type", "") or "")):
            continue
        xy = _unit_xy(u)
        if xy is not None:
            pts.append(xy)
    if not pts:
        return None
    return [round(sum(p[0] for p in pts) / len(pts), 1),
            round(sum(p[1] for p in pts) / len(pts), 1)]


def _harv_xy(units, cap: int = 6):
    out = []
    for u in units or []:
        if "harv" not in str(getattr(u, "type", "") or "").lower():
            continue
        xy = _unit_xy(u)
        if xy is None:
            continue
        out.append(xy)
        if len(out) >= cap:
            break
    return out


def _n_combat(units) -> int:
    n = 0
    for u in units or []:
        if not _is_noncombat_type(str(getattr(u, "type", "") or "")):
            n += 1
    return n


def _n_near(units, cell, radius: int, combat_only: bool = True) -> int:
    if cell is None:
        return 0
    r2 = int(radius) * int(radius)
    n = 0
    for u in units or []:
        if combat_only and _is_noncombat_type(str(getattr(u, "type", "") or "")):
            continue
        xy = _unit_xy(u)
        if xy is None:
            continue
        if (xy[0] - int(cell[0])) ** 2 + (xy[1] - int(cell[1])) ** 2 <= r2:
            n += 1
    return n


def _n_combat_home(obs, radius: int = 18) -> int:
    n = 0
    for u in getattr(obs, "units", None) or []:
        if _is_noncombat_type(str(getattr(u, "type", "") or "")):
            continue
        xy = _unit_xy(u)
        if xy is None:
            continue
        for b in getattr(obs, "buildings", None) or []:
            bxy = _unit_xy(b)
            if bxy is None:
                continue
            if (xy[0] - bxy[0]) ** 2 + (xy[1] - bxy[1]) ** 2 <= radius * radius:
                n += 1
                break
    return n


def _cmd_xy(cmd):
    try:
        x, y = int(getattr(cmd, "target_x", -1)), int(getattr(cmd, "target_y", -1))
    except (TypeError, ValueError):
        return None
    if x < 0 or y < 0:
        return None
    return [x, y]


def _tape_row(obs, *, ep, ckpt, dec, pol, cell, item, sup, supk, iss=None):
    dest = sup or cell
    return {
        "ep": ep,
        "ckpt": int(ckpt),
        "dec": int(dec),
        "tick": int(getattr(obs, "tick", 0) or 0),
        "pol": pol,
        "cell": cell,
        "item": item,
        "iss": iss,
        "sup": sup,
        "supk": supk,
        "cent": _combat_centroid(obs.units),
        "harv": _harv_xy(obs.units),
        "nc": _n_combat(obs.units),
        "nu": len(obs.units or []),
        "nb": len(getattr(obs, "buildings", None) or []),
        "nh": _n_combat_home(obs),
        "nd": _n_near(obs.units, dest, 8),
        "ne": (len(obs.visible_enemies or [])
               + len(obs.visible_enemy_buildings or [])),
        "cash": int(getattr(getattr(obs, "economy", None), "cash", 0) or 0),
    }


def _dist(a, b):
    if not a or not b:
        return None
    return round(((float(a[0]) - float(b[0])) ** 2
                  + (float(a[1]) - float(b[1])) ** 2) ** 0.5, 1)


def _remember_cell(bucket: list, xy, cap: int = 24) -> None:
    if xy is None or len(bucket) >= cap:
        return
    cell = [int(xy[0]), int(xy[1])]
    if cell not in bucket:
        bucket.append(cell)


def _append_live_game(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _obs_to_live_state(obs, beacon, hist, decs, rew, adv_ticks, last_action_str, status, done, result):
    """Convierte observación a dict liviano para el visor."""
    H = obs.map_info.height or 64
    W = obs.map_info.width or 64
    # intentar extraer recursos / niebla del spatial (muestreo)
    resources = []
    fog = []
    try:
        sp = decode_spatial(obs.spatial_map, H, W, obs.spatial_channels or 9, beacon=None)
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
        "buildings": [{"x": b.cell_x, "y": b.cell_y, "hp": getattr(b, "hp_percent", 1.0),
                         "type": str(getattr(b, "type", "") or "")} for b in (obs.buildings or [])],
        "units": [{"x": u.cell_x, "y": u.cell_y,
                    "type": str(getattr(u, "type", "") or ""),
                    "can_attack": bool(getattr(u, "can_attack", False))} for u in (obs.units or [])],
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


async def run_episode_live(env: OpenRAEnv, net, vocab, device, args,
                           broadcaster: LiveBroadcaster, ckpt_iter: int = 0):
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
    ep_id = f"{int(ckpt_iter)}-{int(time.time())}"
    tape_path = getattr(args, "tape_file", "") or ""
    trace = {
        "policy_push_cells": [],
        "support_dests": [],
        "n_support_army": 0,
        "n_support_am": 0,
        "centroid": [],
        "tape": [],
    }

    # estado inicial
    broadcaster.update(_obs_to_live_state(obs, beacon, hist, decs, episode_reward, adv_total, last_action_str, "jugando…", done, ""))

    use_macro = args.macro_ticks > 0
    for step in range(args.max_steps):
        can_decide = use_macro or step % args.k_skip == 0
        action = None
        atype_str = "no_op"
        step_meta = None
        if can_decide:
            batch, aidx = _batch_of(obs, vocab, device)
            h_in = hidden.detach().clone()
            with torch.no_grad():
                out = net.act(batch, hidden, temperature=args.temperature)
            hidden = out["hidden"].detach()
            had_item = aidx.item_mask.any().view(1).to(device)
            action, (eff_t, eff_u, eff_i, eff_c) = index_to_command_effective(
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
            # clamp first so last_action shows the cell actually issued
            ep_dims = (obs.map_info.height, obs.map_info.width)
            for c in action.commands:
                if c.target_x >= ep_dims[1] or c.target_y >= ep_dims[0]:
                    c.target_x = min(c.target_x, ep_dims[1]-1)
                    c.target_y = min(c.target_y, ep_dims[0]-1)
            # Mismo crédito que el train: army/attack_move muestra el dest
            # de soporte (visor: last_push ≈ support_dests, no mill en casa).
            if args.auto_support:
                new_c, _dest_xy = apply_dest_credit(
                    obs, action, atype_str, int(eff_c), aidx,
                    last_push=last_push_cell)
                eff_c = int(new_c)
            # texto acción: the cell ACTUALLY issued (after water/OOB remap).
            # TRAIN/BUILD ignore cell — show em-dash so live does not display south-water.
            # item = rol EFECTIVO (post-coerce). iss = item_type del comando C#.
            item_name = (aidx.items[int(eff_i)] if int(eff_i) < len(aidx.items) else "—")
            issued_item = None
            if action.commands:
                issued_item = getattr(action.commands[0], "item_type", None) or None
            if atype_str in ("train", "build", "no_op", "deploy", "harvest",
                             "stop", "cancel_production"):
                cell_txt = "cell=—"
            elif action.commands:
                c0 = action.commands[0]
                cell_txt = f"cell={int(getattr(c0, 'target_x', 0))},{int(getattr(c0, 'target_y', 0))}"
            else:
                cell_txt = f"cell={int(eff_c) % aidx.w},{int(eff_c) // aidx.w}"
            last_action_str = f"{atype_str}  {cell_txt}  item={issued_item or item_name}  units={len(obs.units)} cash={obs.economy.cash}"
            pol_cell = None
            if atype_str in ("army_attack_move", "attack_move", "move", "attack") and action.commands:
                pol_cell = _cmd_xy(action.commands[0])
            # Pilar B: auto-harvest/repair gratis (no roba decisión PPO)
            if atype_str in ("army_attack_move", "attack_move") and action.commands:
                c0 = action.commands[0]
                if getattr(c0, "target_x", None) is not None:
                    last_push_cell = (int(c0.target_x), int(c0.target_y))
                    _remember_cell(trace["policy_push_cells"], last_push_cell)
            sup_xy = None
            sup_kind = None
            if args.auto_support:
                for cmd in support_commands(obs, last_push=last_push_cell, aidx=aidx):
                    action.commands.append(cmd)
                    name = getattr(getattr(cmd, "action", None), "value", None) or str(
                        getattr(cmd, "action", ""))
                    if name in ("army_attack_move", "attack_move"):
                        dest = [int(cmd.target_x), int(cmd.target_y)]
                        if name == "army_attack_move":
                            trace["n_support_army"] += 1
                            sup_kind = "army"
                        else:
                            trace["n_support_am"] += 1
                            if sup_kind is None:
                                sup_kind = "am"
                        sup_xy = dest
                        _remember_cell(trace["support_dests"], dest)
            step_meta = {
                "pol": atype_str,
                "cell": pol_cell,
                "item": (item_name if atype_str in ("train", "build", "place_building",
                                                    "cancel_production")
                         else None),
                "iss": issued_item,
                "sup": sup_xy,
                "supk": sup_kind,
            }
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
            if step_meta is not None:
                row = _tape_row(
                    obs, ep=ep_id, ckpt=ckpt_iter, dec=decs,
                    pol=step_meta["pol"], cell=step_meta["cell"],
                    item=step_meta["item"], sup=step_meta["sup"],
                    supk=step_meta["supk"], iss=step_meta.get("iss"))
                trace["tape"].append(row)
                if tape_path:
                    p_tape = Path(tape_path)
                    if not p_tape.is_absolute():
                        p_tape = Path(__file__).resolve().parent.parent / p_tape
                    try:
                        _append_live_game(p_tape, row)
                    except OSError:
                        pass
            if decs == 1 or decs % 50 == 0:
                n_cbt = _n_combat(obs.units)
                dest = (step_meta or {}).get("sup") or last_push_cell
                dest_xy = list(dest) if dest is not None else None
                trace["centroid"].append({
                    "dec": decs, "tick": obs.tick,
                    "xy": _combat_centroid(obs.units),
                    "harv": _harv_xy(obs.units),
                    "n_combat": n_cbt,
                    "n_home": _n_combat_home(obs),
                    "n_at_dest": _n_near(obs.units, dest_xy, 8),
                    "n_units": len(obs.units or []),
                    "n_ene": (len(obs.visible_enemies or [])
                              + len(obs.visible_enemy_buildings or [])),
                })
            broadcaster.update(_obs_to_live_state(obs, beacon, hist, decs, episode_reward, adv_total, last_action_str, f"dec {decs} tick {obs.tick}", done, getattr(obs, "result", "") or macro_final or ""))

        if done:
            break
        # pequeño yield para que el http server respire
        await asyncio.sleep(0)

    r_final = shaper.finalize(truncated=not done, result=str(getattr(obs, "result", "") or macro_final or ""))
    episode_reward += r_final
    final_result = getattr(obs, "result", None) or macro_final or ("incomplete" if not done else "")
    broadcaster.update(_obs_to_live_state(obs, beacon, hist, decs, episode_reward, adv_total, last_action_str, f"final: {final_result}", True, final_result))
    xy_end = _combat_centroid(obs.units)
    beacon_xy = list(beacon) if beacon else None
    log_row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "ckpt": str(getattr(args, "ckpt", "")),
        "ckpt_iter": int(ckpt_iter),
        "bot_type": args.bot_type,
        "scenario": args.scenario,
        "map_name": str(getattr(getattr(obs, "map_info", None), "map_name", "") or ""),
        "result": final_result,
        "ticks": int(obs.tick or 0),
        "decisions": decs,
        "episode_reward": round(episode_reward, 3),
        "advanced_ticks": adv_total,
        "hist": dict(hist),
        "beacon": beacon_xy,
        "policy_push_cells": trace["policy_push_cells"],
        "support_dests": trace["support_dests"],
        "n_support_army": trace["n_support_army"],
        "n_support_am": trace["n_support_am"],
        "centroid": trace["centroid"],
        "tape": [t for i, t in enumerate(trace["tape"])
                 if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(trace["tape"])],
        "ep": ep_id,
        "centroid_end": xy_end,
        "dist_to_beacon": _dist(xy_end, beacon_xy),
        "n_units_end": len(obs.units or []),
        "n_bld_end": len(obs.buildings or []),
        "n_ene_vis": (len(obs.visible_enemies or [])
                      + len(obs.visible_enemy_buildings or [])),
        "units": _tally_types(obs.units),
        "buildings": _tally_types(obs.buildings),
        "last_action": last_action_str,
        "last_push": list(last_push_cell) if last_push_cell else None,
    }
    log_path = getattr(args, "log_file", "") or ""
    if log_path:
        p = Path(log_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / p
        try:
            _append_live_game(p, log_row)
            print(f"  logged {p}  dist_beacon={log_row['dist_to_beacon']} "
                  f"tape={len(trace['tape'])} support_dests={trace['support_dests']}",
                  flush=True)
        except OSError as e:
            print(f"  [live log] no pude escribir {p}: {e}", flush=True)
    return {"result": final_result, "ticks": obs.tick, "decisions": decs,
            "episode_reward": round(episode_reward, 3), "hist": hist,
            "advanced_ticks": adv_total, "dist_to_beacon": log_row["dist_to_beacon"]}


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
            outcome = await run_episode_live(
                env, net, vocab, device, args, bc, ckpt_iter=it)
            print(f"  result={outcome['result']} ticks={outcome['ticks']} "
                  f"decs={outcome['decisions']} rew={outcome['episode_reward']} "
                  f"dist_beacon={outcome.get('dist_to_beacon')} hist={outcome['hist']}")
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
    ap.add_argument("--macro-ticks", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--shaper-preset", default="eradicate_v4", choices=list(PRESETS))
    ap.add_argument("--auto-support", action=argparse.BooleanOptionalAction, default=True,
                    help="harvest/repair/power automático (Pilar B); default on como el train")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--port", type=int, default=8786, help="puerto del visor live (default 8786)")
    ap.add_argument("--log-file", default="rl/ckpts/live_games.jsonl",
                    help="jsonl por partida del visor. Vacío = no loguear.")
    ap.add_argument("--tape-file", default="rl/ckpts/live_tape.jsonl",
                    help="jsonl una línea por decisión (política, dest, harv, centroide). Vacío = off.")
    args = ap.parse_args()
    # --bot-type "" mantiene "" (dummy), solo None es "no tocar". --ai-slot "" desactiva enemigo.
    if args.bot_type == "__none__":
        args.bot_type = None
    asyncio.run(amain(args))

if __name__ == "__main__":
    main()
