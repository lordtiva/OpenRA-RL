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

    # una partida, greedy, el visor retiene el frame final:
    .\.venv\Scripts\python.exe -m rl.play_vs_checkpoint_live `
      --ckpt rl\ckpts_v2\latest.pt --episodes 1 --step-delay 0.2

    # visor:
    http://localhost:8786/

    # live-only lobby overrides (train defaults untouched):
    python -m rl.play_vs_checkpoint_live --enemy-faction russia --spawn ne --episodes 1
    python -m rl.play_vs_checkpoint_live --enemy-faction Random --spawn random

Loop infinito (default --episodes 0) pausa --pause-sec entre partidas
para que dé tiempo a mirar el canvas. --pause-sec 0 vuelve al turbo.

Al terminar cada partida (win/lose/incomplete) append a
rl/ckpts_v2/live_games.jsonl y el tape completo a rl/ckpts_v2/live_tape.jsonl.
El visor puede grabar WebM (mapa+HUD) a {ckpt_dir}/live_recordings/{episode_id}.webm
con el mismo episode_id que va en el jsonl.
Ctrl+C / DEADLINE no escriben: un chequeo corto no deja basura del run
siguiente. No pisa el train.
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
from openra_env.mcp_ws_client import OpenRAMCPClient
from rl.action_adapter import Vocab
from rl.network import AlphaLiteNet
from rl.trainer import load_checkpoint
from rl.live_server import LiveBroadcaster
from rl.live_lobby import (
    ENEMY_FACTION_CHOICES,
    PLAYER_FACTION_CHOICES,
    SPAWN_CHOICES,
    normalize_enemy_faction,
    normalize_player_faction,
    normalize_spawn,
    patch_oramap_bytes,
    resolve_episode_spawn,
)
from rl.obs_encoding import EnemyBeliefStore, decode_spatial
from rl.war_objective import war_objective
from rl.rollout import (
    _batch_of, episode_progress, is_progress_activity, should_idle_truncate,
)
from rl.action_adapter import index_to_command_effective, filter_army_push_hysteresis
from openra_env.models import ActionType, CommandModel, OpenRAAction
from rl.reward_shaping import PRESETS, ShapedReward
from rl.supremacy import evaluate_supremacy
from rl.network import ACTION_TYPES, COMBAT_PUSH_TYPES, HIDDEN_DIM
from rl.auto_support import apply_dest_credit, support_commands
from rl.imitation import student_combat_ready


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
    obj = None
    try:
        obj = war_objective(obs)
    except Exception:
        obj = None
    dest = obj or sup or cell
    return {
        "ep": ep,
        "episode_id": ep,
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


def _resolve_log_path(path_str: str) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    return p


def _append_live_game(path: Path, row: dict) -> None:
    _append_live_rows(path, [row])


def _append_live_rows(path: Path, rows: list) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


_ALLIED_TELLS = {
    "tent", "pbox", "hbox", "gun", "agun", "gap", "atek", "pdox",
    "e7", "medi", "mech", "spy", "ctnk", "stnk", "mh60", "apc",
}
_SOVIET_TELLS = {
    "barr", "ftur", "tsla", "sam", "iron", "kenn", "mslo", "stek",
    "3tnk", "ttnk", "v2rl", "dog", "shok",
}


def _side_from_names(names) -> str:
    n = {str(x).lower() for x in (names or []) if x}
    if n & _ALLIED_TELLS:
        return "allies"
    if n & _SOVIET_TELLS:
        return "soviet"
    return ""


def _infer_side(obs) -> str:
    names = set()
    for x in getattr(obs, "available_production", None) or []:
        names.add(str(x).lower())
    for b in getattr(obs, "buildings", None) or []:
        names.add(str(getattr(b, "type", "") or "").lower())
    for u in getattr(obs, "units", None) or []:
        names.add(str(getattr(u, "type", "") or "").lower())
    return _side_from_names(names)


def _infer_enemy_side(obs) -> str:
    names = set()
    for u in getattr(obs, "visible_enemies", None) or []:
        names.add(str(getattr(u, "type", "") or "").lower())
    for b in getattr(obs, "visible_enemy_buildings", None) or []:
        names.add(str(getattr(b, "type", "") or "").lower())
    return _side_from_names(names)


_COUNTRY_FACTIONS = frozenset({
    "england", "france", "germany", "russia", "ukraine",
})


def _better_faction(old: str, new: str) -> str:
    """Prefer InternalName (russia) over side guess (soviet) over empty."""
    new = str(new or "").strip()
    old = str(old or "").strip()
    if new in _COUNTRY_FACTIONS:
        return new
    if old in _COUNTRY_FACTIONS:
        return old
    return new or old


def _faction_from_mcp(gs: dict) -> tuple[str, str]:
    pf = str(gs.get("faction") or gs.get("player_faction") or "")
    ef = str(gs.get("enemy_faction") or "")
    if not ef:
        names = []
        for row in (gs.get("enemy_summary") or []) + (gs.get("enemy_buildings_summary") or []):
            if isinstance(row, dict):
                names.append(row.get("type"))
        ef = _side_from_names(names)
    return pf, ef


def _faction_pair(obs) -> tuple[str, str]:
    meta = getattr(obs, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    pf = str(getattr(obs, "player_faction", "") or meta.get("player_faction") or "")
    ef = str(getattr(obs, "enemy_faction", "") or meta.get("enemy_faction") or "")
    if not pf:
        pf = _infer_side(obs)
    if not ef:
        ef = _infer_enemy_side(obs)
    return pf, ef


async def _mcp_game_state(env) -> dict:
    """Docker stamps faction on GameState, not always on the step observation."""
    n = int(getattr(env, "_rpc_n", 0) or 0) + 1
    env._rpc_n = n
    req = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": "get_game_state", "arguments": {}},
        "id": f"fac{n}",
    }
    resp = await env._send_and_receive({"type": "mcp", "data": req})
    result = (resp.get("data") or {}).get("result", {})
    if isinstance(result, dict):
        return OpenRAMCPClient._unwrap_mcp_result(result) or {}
    return {}


def _obs_to_live_state(obs, beacon, hist, decs, rew, adv_ticks, last_action_str, status, done, result, episode_id="", player_faction="", enemy_faction=""):
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

    inf_pf, inf_ef = _faction_pair(obs)
    player_faction = player_faction or inf_pf
    enemy_faction = enemy_faction or inf_ef
    return {
        "tick": obs.tick,
        "player_faction": player_faction,
        "enemy_faction": enemy_faction,
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
        "ene_buildings": [{"x": b.cell_x, "y": b.cell_y,
                              "type": str(getattr(b, "type", "") or "")}
                             for b in (obs.visible_enemy_buildings or [])],
        "ene_units": [{"x": u.cell_x, "y": u.cell_y,
                        "type": str(getattr(u, "type", "") or ""),
                        "can_attack": bool(getattr(u, "can_attack", False))}
                       for u in (obs.visible_enemies or [])],
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
        "episode_id": episode_id or "",
    }


async def run_episode_live(env: OpenRAEnv, net, vocab, device, args,
                           broadcaster: LiveBroadcaster, ckpt_iter: int = 0,
                           ep_index: int = 1):
    lobby = broadcaster.take_lobby() if broadcaster is not None else {
        "player_faction": getattr(args, "player_faction", "RandomAllies"),
        "enemy_faction": getattr(args, "enemy_faction", "Random"),
        "spawn": getattr(args, "spawn", "random"),
    }
    player_faction = normalize_player_faction(lobby.get("player_faction"))
    enemy_faction = normalize_enemy_faction(lobby.get("enemy_faction"))
    spawn_asked = normalize_spawn(lobby.get("spawn"))
    import random as _random
    rng = _random.Random((args.seed or 0) + int(ep_index) * 1009)
    agent_side = resolve_episode_spawn(spawn_asked, rng=rng)

    reset_kwargs = {}
    spawn_meta = {}
    if args.scenario:
        mapa = Path(f"rl/scenarios/fase2_{args.scenario.lower()}.oramap")
        if not mapa.exists():
            alt = Path(f"rl/scenarios/{args.scenario}.oramap")
            if alt.exists():
                mapa = alt
            else:
                raise SystemExit(f"Escenario no encontrado: {mapa}")
        raw_map = mapa.read_bytes()
        try:
            raw_map, spawn_meta = patch_oramap_bytes(raw_map, agent_side)
        except ValueError as e:
            print(f"  [live] spawn pin skipped ({e}); using stock map", flush=True)
            spawn_meta = {"agent_side": agent_side, "error": str(e)}
        reset_kwargs["map_data"] = base64.b64encode(raw_map).decode()
        reset_kwargs["map_name"] = mapa.name
    # bot_type="" -> dummy (pasivo), ai_slot="" -> sin enemigo. No filtrar "".
    if args.bot_type is not None:
        reset_kwargs["bot_type"] = args.bot_type
    if hasattr(args, "ai_slot") and args.ai_slot is not None:
        reset_kwargs["ai_slot"] = args.ai_slot
    reset_kwargs["seed"] = args.seed
    reset_kwargs["player_faction"] = player_faction
    reset_kwargs["enemy_faction"] = enemy_faction

    beacon = tuple(spawn_meta["enemy_cell"]) if spawn_meta.get("enemy_cell") else None
    print(
        f"  lobby player={player_faction} enemy={enemy_faction} "
        f"spawn_asked={spawn_asked} agent_side={agent_side} "
        f"beacon={beacon}",
        flush=True,
    )
    if broadcaster is not None:
        broadcaster.update({
            "lobby": {
                "player_faction": player_faction,
                "enemy_faction": enemy_faction,
                "spawn": spawn_asked,
                "agent_side": agent_side,
            },
            "spawn_meta": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in spawn_meta.items()
                if k != "spawns"
            },
        })

    result = await env.reset(**reset_kwargs)
    obs = result.observation
    pf, ef = _faction_pair(obs)
    try:
        gs = await _mcp_game_state(env)
        mcp_pf, mcp_ef = _faction_from_mcp(gs if isinstance(gs, dict) else {})
        pf = _better_faction(pf, mcp_pf)
        ef = _better_faction(ef, mcp_ef)
    except Exception as e:
        print(f"  [live] get_game_state falló: {e}")
    if pf:
        try:
            obs.player_faction = pf
            obs.enemy_faction = ef
        except Exception:
            pass
    print(f"  faction={pf or '?'} enemy={ef or '?'} "
          f"(lock Aliados = england|france|germany)")
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
    last_army_push_cell = None  # hysteresis for executed army_attack_move
    last_activity_tick = int(getattr(obs, "tick", 0) or 0)
    _idle_kills = _idle_deaths = _idle_earned = 0
    last_gs = None
    belief = EnemyBeliefStore()
    ep_id = (
        f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_ep{int(ep_index)}_ck{int(ckpt_iter)}"
    )
    aborted = False
    trace = {
        "policy_push_cells": [],
        "support_dests": [],
        "n_support_army": 0,
        "n_support_am": 0,
        "centroid": [],
        "tape": [],
    }

    def _push(obs_, status_, done_, result_=""):
        nonlocal pf, ef
        inf_pf, inf_ef = _faction_pair(obs_)
        pf = _better_faction(pf, inf_pf)
        ef = _better_faction(ef, inf_ef)
        broadcaster.update(_obs_to_live_state(
            obs_, beacon, hist, decs, episode_reward, adv_total, last_action_str,
            status_, done_, result_, episode_id=ep_id,
            player_faction=pf, enemy_faction=ef))

    # estado inicial
    _push(obs, "jugando…", done, "")

    use_macro = args.macro_ticks > 0
    for step in range(args.max_steps):
        can_decide = use_macro or step % args.k_skip == 0
        action = None
        atype_str = "no_op"
        step_meta = None
        if can_decide:
            try:
                obs.belief = belief
            except Exception:
                pass
            batch, aidx = _batch_of(obs, vocab, device, belief=belief)
            h_in = hidden.detach().clone()
            with torch.no_grad():
                out = net.act(batch, hidden, temperature=args.temperature)
            hidden = out["hidden"].detach()
            out_ctx = out.get("_ctx")
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
            # Mismo crédito que el train: army/attack_move muestra el dest
            # de soporte (visor: last_push ≈ support_dests, no mill en casa).
            if args.auto_support:
                new_c, _dest_xy = apply_dest_credit(
                    obs, action, atype_str, int(eff_c), aidx,
                    last_push=last_push_cell)
                eff_c = int(new_c)
            # K=2 eco+push Alt-B: micro-step GRU then combat AR (same tick).
            if (out_ctx is not None and student_combat_ready(obs, belief)
                    and atype_str not in COMBAT_PUSH_TYPES):
                with torch.no_grad():
                    _h_in2, h_push = net.k2_micro_hidden(out_ctx, hidden)
                    out2 = net.act_combat(
                        batch, h_push, temperature=args.temperature, ctx=out_ctx)
                if out2 is not None:
                    hidden = h_push.detach()
                    push_action, (pt, pu, pi, pc) = index_to_command_effective(
                        obs, int(out2["type"]), int(out2["unit_slot"]),
                        int(out2["cell_flat"]), int(out2["item_slot"]), aidx)
                    ptype = ACTION_TYPES[int(pt)]
                    if args.auto_support:
                        new_pc, _ = apply_dest_credit(
                            obs, push_action, ptype, int(pc), aidx,
                            last_push=last_push_cell)
                        pc = int(new_pc)
                    action.commands.extend(push_action.commands or [])
                    hist[ptype] = hist.get(ptype, 0) + 1
            # clamp after dual-emit so last_action / last_push see issued cells
            ep_dims = (obs.map_info.height, obs.map_info.width)
            for c in action.commands:
                if c.target_x >= ep_dims[1] or c.target_y >= ep_dims[0]:
                    c.target_x = min(c.target_x, ep_dims[1]-1)
                    c.target_y = min(c.target_y, ep_dims[0]-1)
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
            if atype_str in ("army_attack_move", "infantry_attack_move", "vehicle_attack_move", "harvesters_move", "attack_move", "move", "attack") and action.commands:
                pol_cell = _cmd_xy(action.commands[0])
            # Pilar B: auto-harvest/repair gratis (no roba decisión PPO)
            _push_names = ("army_attack_move", "infantry_attack_move",
                           "vehicle_attack_move", "harvesters_move",
                           "attack_move", "attack")
            for c in action.commands:
                cname = getattr(getattr(c, "action", None), "value", None) or str(
                    getattr(c, "action", ""))
                if cname in _push_names and getattr(c, "target_x", None) is not None:
                    last_push_cell = (int(c.target_x), int(c.target_y))
                    _remember_cell(trace["policy_push_cells"], last_push_cell)
            sup_xy = None
            sup_kind = None
            if args.auto_support:
                for cmd in support_commands(obs, last_push=last_push_cell, aidx=aidx, war_nudge=not args.no_war_nudge):
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
            # Executed-command hysteresis: suppress near-duplicate army pushes.
            filtered, last_army_push_cell = filter_army_push_hysteresis(
                action.commands, last_army_push_cell)
            action.commands[:] = filtered
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
                aborted = True
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
                    gs_adv = adv.get("global_summary")
                    if isinstance(gs_adv, dict):
                        last_gs = gs_adv
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
                    aborted = True
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
            if decs == 1 or decs % 50 == 0:
                n_cbt = _n_combat(obs.units)
                obj = None
                try:
                    obj = war_objective(obs)
                except Exception:
                    obj = None
                dest = obj or (step_meta or {}).get("sup") or last_push_cell
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
            _push(obs, f"dec {decs} tick {obs.tick}", done, getattr(obs, "result", "") or macro_final or "")

        tick_now, kills_now, deaths_now, earned_now = episode_progress(
            obs, last_gs)
        if is_progress_activity(
                kills_now, deaths_now, earned_now,
                _idle_kills, _idle_deaths, _idle_earned):
            last_activity_tick = tick_now
        _idle_kills, _idle_deaths, _idle_earned = (
            kills_now, deaths_now, earned_now)
        if (not done) and should_idle_truncate(
                tick_now, last_activity_tick,
                after_tick=int(getattr(args, "idle_truncate_after_tick", 35000) or 0),
                idle_ticks=int(getattr(args, "idle_truncate_idle_ticks", 5000) or 0)):
            break

        if done:
            break
        # yield al http del visor; --step-delay frena el turbo para poder mirar
        delay = float(getattr(args, "step_delay", 0.0) or 0.0)
        await asyncio.sleep(delay if delay > 0 else 0)

    r_final = shaper.finalize(truncated=not done, result=str(getattr(obs, "result", "") or macro_final or ""))
    episode_reward += r_final
    final_result = getattr(obs, "result", None) or macro_final or ("incomplete" if not done else "")
    inf_pf, inf_ef = _faction_pair(obs)
    pf = pf or inf_pf
    ef = ef or inf_ef
    _push(obs, f"final: {final_result}", True, final_result)
    xy_end = _combat_centroid(obs.units)
    beacon_xy = list(beacon) if beacon else None
    log_row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "ckpt": str(getattr(args, "ckpt", "")),
        "ckpt_iter": int(ckpt_iter),
        "player_faction": pf,
        "enemy_faction": ef,
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
        "episode_id": ep_id,
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
    if aborted:
        print("  abort (DEADLINE/sesión): no logueo partida incompleta",
              flush=True)
    else:
        tape_path = _resolve_log_path(getattr(args, "tape_file", "") or "")
        if tape_path is not None:
            try:
                _append_live_rows(tape_path, trace["tape"])
            except OSError as e:
                print(f"  [live tape] no pude escribir {tape_path}: {e}",
                      flush=True)
        log_path = _resolve_log_path(getattr(args, "log_file", "") or "")
        if log_path is not None:
            try:
                _append_live_game(log_path, log_row)
                print(f"  logged {log_path}  dist_beacon={log_row['dist_to_beacon']} "
                      f"tape={len(trace['tape'])} support_dests={trace['support_dests']}",
                      flush=True)
            except OSError as e:
                print(f"  [live log] no pude escribir {log_path}: {e}", flush=True)
    return {"result": final_result, "ticks": obs.tick, "decisions": decs,
            "episode_reward": round(episode_reward, 3), "hist": hist,
            "advanced_ticks": adv_total, "dist_to_beacon": log_row["dist_to_beacon"],
            "player_faction": pf, "enemy_faction": ef}


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
    net.xf_topk = int(getattr(args, "xf_topk", 16) or 0)
    net.qsa_topk = int(getattr(args, "qsa_topk", 8) or 0)
    net.qsa_block = int(getattr(args, "qsa_block", 8) or 8)
    vocab = Vocab()
    it = load_checkpoint(str(ckpt_path), net, vocab=vocab)
    net.to(device)
    net.eval()
    print(f"Checkpoint iter {it} | vocab {len(vocab.type_to_id)} tipos | params {sum(p.numel() for p in net.parameters())/1e6:.2f}M")
    if args.greedy:
        args.temperature = 0.0
    print(f"Server: {args.url} | bot={args.bot_type} scenario={args.scenario} T={args.temperature} macro={args.macro_ticks}")
    print(f"Lobby live: player={args.player_faction} enemy={args.enemy_faction} spawn={args.spawn}")

    bc = LiveBroadcaster(port=args.port)
    bc.set_lobby_defaults(
        player_faction=args.player_faction,
        enemy_faction=args.enemy_faction,
        spawn=args.spawn,
    )
    bc.start()
    print(f"Abrí en el navegador: http://localhost:{args.port}/  (se actualiza solo)")
    print("  UI: selectores enemy/spawn hacen POST /api/config (valen desde la próxima partida)")

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
            bc.update({"status": f"episodio {label} — iniciando…", "done": False, "ckpt_iter": it, "episode_id": ""})
            outcome = await run_episode_live(
                env, net, vocab, device, args, bc, ckpt_iter=it, ep_index=ep)
            print(f"  result={outcome['result']} ticks={outcome['ticks']} "
                  f"decs={outcome['decisions']} rew={outcome['episode_reward']} "
                  f"faction={outcome.get('player_faction') or '?'} "
                  f"enemy={outcome.get('enemy_faction') or '?'} "
                  f"dist_beacon={outcome.get('dist_to_beacon')} hist={outcome['hist']}")
            pause = max(0.0, float(getattr(args, "pause_sec", 15.0) or 0.0))
            looping = args.episodes <= 0 or ep < args.episodes
            if pause > 0 and looping:
                print(f"  pausa {pause:.0f}s — el visor retiene el frame. "
                      f"--pause-sec 0 para no esperar")
                bc.update({
                    "status": f"final {outcome['result']} — pausa {pause:.0f}s",
                    "done": True,
                })
                await asyncio.sleep(pause)
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
    ap.add_argument("--ckpt", default="rl/ckpts_v2/latest.pt")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--bot-type", default="beginner")
    ap.add_argument("--ai-slot", default=None, help='slot IA: "Multi0" (default) o "" para sin enemigo')
    ap.add_argument("--scenario", default="a_short")
    ap.add_argument("--episodes", type=int, default=0, help="0 = loop infinito; recarga latest.pt entre partidas")
    ap.add_argument("--pause-sec", type=float, default=10.0,
                    help="segundos a retener el frame final entre partidas (0 = turbo)")
    ap.add_argument("--step-delay", type=float, default=0.0,
                    help="segundos extra por decisión (0.2 hace el canvas seguible)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--k-skip", type=int, default=8)
    ap.add_argument("--macro-ticks", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--idle-truncate-after-tick", type=int, default=35000,
                    help="After this many ticks, allow idle early-truncate (0=off).")
    ap.add_argument("--idle-truncate-idle-ticks", type=int, default=5000,
                    help="No combat/income ticks before idle-truncate (0=off).")
    ap.add_argument("--shaper-preset", default="eradicate_v4", choices=list(PRESETS))
    ap.add_argument("--auto-support", action=argparse.BooleanOptionalAction, default=True,
                    help="harvest/repair/power automático (Pilar B); default on como el train")
    ap.add_argument("--no-war-nudge", action="store_true",
                    help="igual que el train Run 43: apaga raid/push/fog-scout; PPO manda la guerra")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--xf-topk", type=int, default=16,
                    help="entity XF top-k (0=dense; default 16 = Run46)")
    ap.add_argument("--qsa-topk", type=int, default=8,
                    help="map QSA top-k blocks (0=off)")
    ap.add_argument("--qsa-block", type=int, default=8,
                    help="map QSA block size")
    ap.add_argument(
        "--player-faction", default="RandomAllies",
        help="facción del RL agent (Allies only): RandomAllies|england|france|germany",
    )
    ap.add_argument(
        "--enemy-faction", default="Random",
        help="facción del bot: Random|RandomAllies|RandomSoviet|england|france|germany|russia|ukraine",
    )
    ap.add_argument(
        "--spawn", default="sw",
        help="spawn del agent: random|sw|ne (pin via LockSpawn en el .oramap; agent=Multi1)",
    )
    ap.add_argument("--port", type=int, default=8786, help="puerto del visor live (default 8786)")
    ap.add_argument("--log-file", default=None,
                    help="jsonl por partida (default: <ckpt-dir>/live_games.jsonl). Vacío = off.")
    ap.add_argument("--tape-file", default=None,
                    help="jsonl del tape (default: <ckpt-dir>/live_tape.jsonl). Vacío = off.")
    args = ap.parse_args()
    try:
        args.player_faction = normalize_player_faction(args.player_faction)
        args.enemy_faction = normalize_enemy_faction(args.enemy_faction)
        args.spawn = normalize_spawn(args.spawn)
    except ValueError as e:
        raise SystemExit(f"lobby flag error: {e}")
    # Defaults next to the ckpt (v2: rl/ckpts_v2; v1.1: pass --ckpt rl/ckpts/...)
    if args.log_file is None:
        args.log_file = (Path(args.ckpt).parent / "live_games.jsonl").as_posix()
    if args.tape_file is None:
        args.tape_file = (Path(args.ckpt).parent / "live_tape.jsonl").as_posix()
    from rl import live_server as _ls
    _ls.RECORDINGS_DIR = Path(args.ckpt).resolve().parent / "live_recordings"
    # --bot-type "" mantiene "" (dummy), solo None es "no tocar". --ai-slot "" desactiva enemigo.
    if args.bot_type == "__none__":
        args.bot_type = None
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nVisor cerrado.")

if __name__ == "__main__":
    main()
