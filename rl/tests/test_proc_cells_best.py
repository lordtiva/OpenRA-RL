# -*- coding: utf-8 -*-
"""Unit tests: combat-train mask, legal attack-move cells, best.pt comparator."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rl.action_adapter import (
    ActionIndex,
    Vocab,
    apply_passability,
    economy_ready_for_combat,
    index_to_command_effective,
    owns_proc,
    remap_move_cell,
)
from openra_env.models import ActionType, CommandModel, OpenRAAction
from rl.auto_support import (
    HUNT_OFFSETS, STANCE_ATTACK_ANYTHING, apply_dest_credit, support_commands,
)
from rl.best_ckpt import (
    batch_is_dead,
    dead_policy_reason,
    is_attack_spam_collapse,
    is_dead_policy,
    is_strictly_better,
    maybe_update_best,
    viability_breakdown,
    viability_score,
)
from rl.network import TYPE_TO_IDX
from rl.obs_encoding import BEACON_BY_MAP

ok = True


def check(name, cond):
    global ok
    print(f"  [{'OK' if cond else 'FALLA'}] {name}")
    ok = ok and bool(cond)


def _u(actor_id=1, typ="e1", x=12, y=16, idle=True):
    return NS(actor_id=actor_id, type=typ, cell_x=x, cell_y=y,
              is_idle=idle, hp_percent=1.0, can_attack=True)


def _b(typ="fact", actor_id=10, x=12, y=16, hp=1.0):
    return NS(type=typ, actor_id=actor_id, cell_x=x, cell_y=y,
              hp_percent=hp, is_repairing=False, is_powered=True,
              rally_x=-1, rally_y=-1)


def _obs(*, cash=5000, harv=0, bldgs=("fact",), units=None, prod=(),
         avail=("e1", "proc", "powr", "barr"), w=128, h=128,
         map_name="fase2_a_short.oramap", enemies=(), enemy_bldgs=(),
         tick=100):
    if units is None:
        units = [_u(1, "mcv", 12, 16)]
    return NS(
        tick=tick,
        map_info=NS(height=h, width=w, map_name=map_name),
        economy=NS(cash=cash, ore=0, harvester_count=harv,
                   power_provided=100, power_drained=60, resource_capacity=5000),
        military=NS(kills_cost=0, deaths_cost=0, assets_value=2000,
                    units_killed=0, units_dead=0, army_value=0),
        buildings=[_b(t, 100 + i) for i, t in enumerate(bldgs)],
        units=list(units),
        production=list(prod),
        available_production=list(avail),
        visible_enemies=list(enemies),
        visible_enemy_buildings=list(enemy_bldgs),
    )


def _collapse_row(reward=50.0):
    return {
        "iter": 10, "iter_winrate": 0.0, "winrate": 0.0, "winrate_rolling20": 0.0,
        "mean_episode_reward": reward,
        "outcomes": ["lose", "lose", "lose", "lose"],
        "ticks": [9000, 9100, 8900, 9200],
        "n_buildings": {"own": 0.0, "enemy": 6.0},
        "action_hist": {"deploy": 4, "no_op": 400, "attack_move": 0},
        "reward_components": {"raze": 0.0, "mining": 0.0, "first_ore": 0.0, "garrison": 8.0},
        "economy_race": {"own_harvest_per_1k": 0.0},
    }


def _builder_row():
    return {
        "iter": 11, "iter_winrate": 0.0, "winrate": 0.0, "winrate_rolling20": 0.0,
        "mean_episode_reward": 5.0,
        "outcomes": ["lose", "lose", "lose", "lose"],
        "ticks": [20000, 21000, 19000, 22000],
        "n_buildings": {"own": 4.0, "enemy": 3.0},
        "action_hist": {"build": 20, "place_building": 8, "army_attack_move": 30,
                        "train": 10, "no_op": 5},
        "reward_components": {"raze": 1.5, "mining": 1.0, "first_ore": 1.5, "garrison": 0.0},
        "economy_race": {"own_harvest_per_1k": 200.0},
    }


print("=== proc-first / cell remap / best.pt ===")

obs = _obs(bldgs=("fact", "powr", "barr"), avail=("e1", "harv", "dog", "proc", "powr", "barr"))
check("sin proc no esta listo para combate", economy_ready_for_combat(obs) is False)
check("sin proc owns_proc False", owns_proc(obs) is False)
aidx = ActionIndex(obs, Vocab())
check("TRAIN apagado sin proc (ni harv)", bool(aidx.type_mask[TYPE_TO_IDX["train"]]) is False)
check("ningun slot train on", bool(aidx.train_slot_mask.any()) is False)
check("BUILD power/refinery sigue legal", bool(aidx.type_mask[TYPE_TO_IDX["build"]]) is True)
if "barracks" in aidx.build_items:
    bslot = len(aidx.train_items) + aidx.build_items.index("barracks")
    check("slot barracks off sin proc", bool(aidx.build_slot_mask[bslot]) is False)
if "refinery" in aidx.build_items:
    rslot = len(aidx.train_items) + aidx.build_items.index("refinery")
    check("slot refinery on", bool(aidx.build_slot_mask[rslot]) is True)
check("DEPLOY sigue legal", bool(aidx.type_mask[TYPE_TO_IDX["deploy"]]) is True)
check("sin proc army_attack_move off", bool(aidx.type_mask[TYPE_TO_IDX["army_attack_move"]]) is False)
check("sin proc attack_move off", bool(aidx.type_mask[TYPE_TO_IDX["attack_move"]]) is False)
check("sin proc attack off", bool(aidx.type_mask[TYPE_TO_IDX["attack"]]) is False)
check("sin proc move sigue on", bool(aidx.type_mask[TYPE_TO_IDX["move"]]) is True)
check("PLACE no se toca (proc en build_items)", "refinery" in aidx.build_items)

obs2 = _obs(harv=1, bldgs=("fact", "proc", "barr"), avail=("e1", "harv", "proc"),
            units=[_u(1, "harv", 14, 16)])
check("proc+harv listo para combate", economy_ready_for_combat(obs2) is True)
aidx2 = ActionIndex(obs2, Vocab())
check("TRAIN legal con proc+harv", bool(aidx2.type_mask[TYPE_TO_IDX["train"]]) is True)
check("con proc army_attack_move on", bool(aidx2.type_mask[TYPE_TO_IDX["army_attack_move"]]) is True)
check("con proc attack_move on", bool(aidx2.type_mask[TYPE_TO_IDX["attack_move"]]) is True)
slot = aidx2.train_items.index("infantry_basic")
check("slot infantry_basic on", bool(aidx2.train_slot_mask[slot]) is True)

prod = [NS(queue_type="Vehicle", item="harv", progress=0.4, paused=False)]
check("harv en cola cuenta como in-flight",
      economy_ready_for_combat(_obs(harv=0, bldgs=("fact", "proc"), avail=("e1", "harv"),
                                    prod=prod, units=[])) is True)

obs3 = _obs(bldgs=("fact", "barr"), avail=("e1", "proc", "powr"))
aidx3 = ActionIndex(obs3, Vocab())
t = TYPE_TO_IDX["train"]
slot = aidx3.items.index("infantry_basic") if "infantry_basic" in aidx3.items else 0
action, _eff = index_to_command_effective(obs3, t, 0, 0, slot, aidx3)
cmd = action.commands[0]
check("adapter no entrena e1 sin proc",
      cmd.action.value != "train" or cmd.item_type != "e1")
check("adapter tampoco entrena harv sin proc",
      cmd.action.value != "train")
action_am, _ = index_to_command_effective(
    obs3, TYPE_TO_IDX["army_attack_move"], 0, 0, 0, aidx3)
check("adapter army_attack_move -> no_op sin proc",
      action_am.commands[0].action.value == "no_op")

cmds = support_commands(_obs(cash=5000, bldgs=("fact",), avail=("e1", "proc", "powr", "barr")))
kinds = [(c.action.value, c.item_type) for c in cmds]
check("auto-support pushea BUILD proc", ("build", "proc") in kinds)

prod_ready = [NS(queue_type="Building", item="proc", progress=1.0, paused=False)]
cmds_p = support_commands(_obs(bldgs=("fact",), avail=("proc",), prod=prod_ready))
check("auto-support PLACE proc listo",
      any(c.action.value == "place_building" and c.item_type == "proc" for c in cmds_p))
cmds_push = support_commands(_obs(bldgs=("fact",), units=[_u(1, "e1", 12, 16)]), last_push=(90, 12))
check("support no keep-alive attack sin proc",
      not any(c.action.value == "attack_move" for c in cmds_push))
cmds_push2 = support_commands(
    _obs(bldgs=("fact", "proc"), units=[_u(1, "e1", 12, 16)]), last_push=(90, 12))
check("support keep-alive attack CON proc",
      any(c.action.value == "attack_move" for c in cmds_push2))

army4 = [_u(i, "e1", 12 + i, 16) for i in range(1, 5)] + [_u(9, "harv", 14, 16)]
cmds_assault = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army4, enemies=[_u(99, "e1", 90, 12)]))
check("asalto sostenido: army_attack_move con 4 rifles + proc + harv",
      any(c.action.value == "army_attack_move" for c in cmds_assault))
am = next(c for c in cmds_assault if c.action.value == "army_attack_move")
check("asalto apunta al enemigo visible", am.target_x == 90 and am.target_y == 12)
cmds_no_eco = support_commands(
    _obs(bldgs=("fact",), units=army4, enemies=[_u(99, "e1", 90, 12)]))
check("asalto NO arranca sin proc",
      not any(c.action.value == "army_attack_move" for c in cmds_no_eco))
cmds_no_harv = support_commands(
    _obs(harv=0, bldgs=("fact", "proc"), units=army4[:4],
         enemies=[_u(99, "e1", 90, 12)]))
check("asalto NO arranca sin harv",
      not any(c.action.value == "army_attack_move" for c in cmds_no_harv))
cmds_beacon = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army4))
check("asalto sin enemigo visible usa beacon a_short",
      any(c.action.value == "army_attack_move" and c.target_x == 95 and c.target_y == 11
          for c in cmds_beacon))
cmds_home = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army4), last_push=(12, 16))
check("asalto ignora last_push en casa, usa beacon",
      any(c.action.value == "army_attack_move" and c.target_x == 95 and c.target_y == 11
          for c in cmds_home))
cmds_title = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army4, map_name="Singles"),
    last_push=(12, 16))
check("asalto con Title=Singles (obs real) usa beacon, no last_push",
      any(c.action.value == "army_attack_move" and c.target_x == 95 and c.target_y == 11
          for c in cmds_title))
army_at_beacon = [_u(i, "e1", 94 + (i % 3), 10 + (i % 2)) for i in range(1, 5)] + [
    _u(9, "harv", 90, 12)]
cmds_hunt = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army_at_beacon, tick=100))
hx, hy = 95 + HUNT_OFFSETS[0][0], 11 + HUNT_OFFSETS[0][1]
check("pile en beacon sin vis: hunt, no idle sobre (95,11)",
      any(c.action.value == "army_attack_move" and c.target_x == hx and c.target_y == hy
          for c in cmds_hunt)
      and not any(c.action.value == "army_attack_move" and c.target_x == 95
                  and c.target_y == 11 for c in cmds_hunt))
cmds_bldg = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army_at_beacon,
         enemies=[_u(99, "e1", 50, 28)],
         enemy_bldgs=[_b("powr", 200, 88, 22)]))
check("edificio visible gana a unidad lejana y al hunt",
      any(c.action.value == "army_attack_move" and c.target_x == 88 and c.target_y == 22
          for c in cmds_bldg))
cmds_stray = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army_at_beacon,
         enemies=[_u(99, "e1", 48, 29)]))
check("stray lejos del beacon con pile: hunt, no persigue el scout",
      any(c.action.value == "army_attack_move" and c.target_x == hx and c.target_y == hy
          for c in cmds_stray))
cmds_raid = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army_at_beacon,
         enemies=[_u(99, "e1", 14, 17)]))
check("raid junto a fact gana a hunt (easy pega en casa)",
      any(c.action.value == "army_attack_move" and c.target_x == 14 and c.target_y == 17
          for c in cmds_raid))
cmds_raid_bldg = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army_at_beacon,
         enemies=[_u(99, "e1", 14, 17)],
         enemy_bldgs=[_b("powr", 200, 88, 22)]))
check("raid en casa gana tambien a edificio enemigo en el beacon",
      any(c.action.value == "army_attack_move" and c.target_x == 14 and c.target_y == 17
          for c in cmds_raid_bldg))
army_walk_beacon = [
    _u(i, "e1", 94 + (i % 3), 10 + (i % 2), idle=False) for i in range(1, 5)
] + [_u(9, "harv", 90, 12)]
cmds_defend_walk = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army_walk_beacon,
         enemies=[_u(99, "e1", 14, 17)]))
check("raid: recall aunque el blob ya camina al beacon",
      any(c.action.value == "army_attack_move" and c.target_x == 14
          and c.target_y == 17 for c in cmds_defend_walk))
army_fight_home = [
    _u(i, "e1", 13 + (i % 3), 16, idle=False) for i in range(1, 5)
] + [_u(9, "harv", 14, 16)]
cmds_fight = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army_fight_home,
         enemies=[_u(99, "e1", 14, 17)]))
check("ya pelean el raid en casa: no cancelar path",
      not any(c.action.value == "army_attack_move" for c in cmds_fight))
aidx_cred = NS(w=128, h=64)
home_flat = 16 * 128 + 12
act_army = OpenRAAction(commands=[CommandModel(
    action=ActionType.ARMY_ATTACK_MOVE, target_x=12, target_y=16)])
flat_b, xy_b = apply_dest_credit(
    _obs(harv=1, bldgs=("fact", "proc"), units=army4),
    act_army, "army_attack_move", home_flat, aidx_cred)
check("credit army_attack_move en casa -> beacon", xy_b == (95, 11))
check("credit muta el comando a beacon",
      act_army.commands[0].target_x == 95 and act_army.commands[0].target_y == 11)
check("credit cell_flat = y*w+x", flat_b == 11 * 128 + 95)
act_train = OpenRAAction(commands=[CommandModel(
    action=ActionType.TRAIN, item_type="e1")])
flat_t, xy_t = apply_dest_credit(
    _obs(harv=1, bldgs=("fact", "proc"), units=army4),
    act_train, "train", home_flat, aidx_cred)
check("credit no toca TRAIN", xy_t is None and flat_t == home_flat)
act_raid = OpenRAAction(commands=[CommandModel(
    action=ActionType.ARMY_ATTACK_MOVE, target_x=12, target_y=16)])
_flat_r, xy_r = apply_dest_credit(
    _obs(harv=1, bldgs=("fact", "proc"), units=army4,
         enemies=[_u(99, "e1", 14, 17)]),
    act_raid, "army_attack_move", home_flat, aidx_cred)
check("credit dest en raid = celda del raid", xy_r == (14, 17))
grid_w = np.ones((64, 128), dtype=np.float32)
grid_w[40:, :] = 0.0
aidx_wat = NS(w=128, h=64, pass_grid=grid_w, cell_mask=grid_w.reshape(-1) > 0.5)
obs_wat = _obs(h=64, w=128, map_name="no-beacon", harv=1,
               bldgs=("fact", "proc"), units=army4)
act_wat = OpenRAAction(commands=[CommandModel(
    action=ActionType.ARMY_ATTACK_MOVE, target_x=12, target_y=16)])
_flat_w, xy_w = apply_dest_credit(
    obs_wat, act_wat, "army_attack_move", home_flat, aidx_wat,
    last_push=(0, 53))
check("credit no deja dest en agua", xy_w is None or xy_w[1] < 40)
cmds_rally = support_commands(
    _obs(harv=1, bldgs=("fact", "proc", "tent"), units=army4))
check("rally de tent al beacon",
      any(c.action.value == "set_rally_point" and c.target_x == 95
          and c.target_y == 11 for c in cmds_rally))
cmds_weap = support_commands(
    _obs(harv=1, bldgs=("fact", "proc", "weap"), units=army4))
check("weap no rally al beacon (HARV sale de weap)",
      not any(c.action.value == "set_rally_point" for c in cmds_weap))
act_harv_am = OpenRAAction(commands=[CommandModel(
    action=ActionType.ATTACK_MOVE, actor_id=9, target_x=12, target_y=16)])
flat_h, xy_h = apply_dest_credit(
    _obs(harv=1, bldgs=("fact", "proc"), units=army4),
    act_harv_am, "attack_move", home_flat, aidx_cred)
check("credit no manda attack_move de harv al beacon",
      xy_h is None and act_harv_am.commands[0].target_x == 12
      and act_harv_am.commands[0].target_y == 16
      and flat_h == home_flat)
act_e1_am = OpenRAAction(commands=[CommandModel(
    action=ActionType.ATTACK_MOVE, actor_id=1, target_x=12, target_y=16)])
_flat_e, xy_e = apply_dest_credit(
    _obs(harv=1, bldgs=("fact", "proc"), units=army4),
    act_e1_am, "attack_move", home_flat, aidx_cred)
check("credit attack_move de rifle SI va al beacon", xy_e == (95, 11))
obs_h = _obs(harv=1, bldgs=("fact", "proc"), units=army4)
aidx_h = ActionIndex(obs_h, Vocab())
hslot = aidx_h.unit_ids.index(9)
e1slot = aidx_h.unit_ids.index(1)
act_h_am, _ = index_to_command_effective(
    obs_h, TYPE_TO_IDX["attack_move"], hslot, home_flat, 0, aidx_h)
check("adapter attack_move sobre harv -> harvest",
      act_h_am.commands[0].action.value == "harvest")
act_h_mv, _ = index_to_command_effective(
    obs_h, TYPE_TO_IDX["move"], hslot, home_flat, 0, aidx_h)
check("adapter move sobre harv -> harvest",
      act_h_mv.commands[0].action.value == "harvest")
act_e1_am2, _ = index_to_command_effective(
    obs_h, TYPE_TO_IDX["attack_move"], e1slot, home_flat, 0, aidx_h)
check("adapter attack_move sobre e1 sigue attack_move",
      act_e1_am2.commands[0].action.value == "attack_move")
cmds_stance = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"),
         units=[_u(i, "e1", 12 + i, 16) for i in range(1, 5)]
         + [_u(9, "harv", 14, 16)]))
# _u sin stance → getattr default AttackAnything, no spam
check("sin campo stance no spamea set_stance",
      not any(c.action.value == "set_stance" for c in cmds_stance))
e1_def = [_u(i, "e1", 12 + i, 16) for i in range(1, 5)]
for u in e1_def:
    u.stance = 2
cmds_aa = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"),
         units=e1_def + [_u(9, "harv", 14, 16)]))
check("Defend -> AttackAnything al nacer",
      any(c.action.value == "set_stance" and c.target_x == STANCE_ATTACK_ANYTHING
          for c in cmds_aa))
cmds_sell = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army4))
# no wrecks
check("no vende fact/proc sanos",
      not any(c.action.value == "sell" for c in cmds_sell))
wreck = _obs(harv=1, bldgs=("fact", "proc", "tent"), units=army4)
wreck.buildings[-1].hp_percent = 0.05
cmds_wreck = support_commands(wreck)
check("vende tent en ruinas",
      any(c.action.value == "sell" and c.actor_id == wreck.buildings[-1].actor_id
          for c in cmds_wreck))
army_walk = [_u(i, "e1", 12 + i, 16, idle=False) for i in range(1, 5)] + [
    _u(9, "harv", 14, 16)]
cmds_walk = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army_walk,
         enemies=[_u(99, "e1", 90, 12)]))
check("no re-emite army_attack si el blob ya camina",
      not any(c.action.value in ("army_attack_move", "attack_move") for c in cmds_walk))
army_walk_home = [_u(i, "e1", 12 + i, 16, idle=False) for i in range(1, 5)] + [
    _u(9, "harv", 14, 16)]
cmds_reassault = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army_walk_home))
check("re-asalto: blob en casa caminando (post-recall) va al beacon",
      any(c.action.value == "army_attack_move" and c.target_x == 95
          and c.target_y == 11 for c in cmds_reassault))
army_mid = [_u(i, "e1", 48 + i, 20, idle=False) for i in range(1, 5)] + [
    _u(9, "harv", 50, 20)]
cmds_mid = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army_mid))
check("mid-map caminando al beacon: no re-order (visor 817)",
      not any(c.action.value in ("army_attack_move", "attack_move") for c in cmds_mid))
army_mix = [_u(1, "e1", 12, 16, idle=True)] + [
    _u(i, "e1", 12 + i, 16, idle=False) for i in range(2, 5)] + [_u(9, "harv", 14, 16)]
cmds_mix = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army_mix,
         enemies=[_u(99, "e1", 90, 12)]))
check("blob caminando: solo ATTACK_MOVE al ocioso, no army_attack",
      any(c.action.value == "attack_move" and c.actor_id == 1
          and c.target_x == 90 for c in cmds_mix)
      and not any(c.action.value == "army_attack_move" for c in cmds_mix))

h, w = 64, 128
grid = np.ones((h, w), dtype=np.float32)
grid[40:, :] = 0.0
obs_w = _obs(h=h, w=w, enemies=[_u(99, "e1", 90, 12)])
aidx_w = ActionIndex(obs_w, Vocab())
apply_passability(aidx_w, grid)
check("celda agua (0,53) enmascarada", bool(aidx_w.cell_mask[53 * w + 0]) is False)
x, y = remap_move_cell(obs_w, aidx_w, 0, 53, actor_id=1)
check("remap agua -> enemigo visible (y<40)", y < 40 and bool(aidx_w.pass_grid[y, x]))

obs_b = _obs(h=h, w=w, enemies=())
aidx_b = ActionIndex(obs_b, Vocab())
apply_passability(aidx_b, grid)
bx, by = remap_move_cell(obs_b, aidx_b, 0, 53, actor_id=1)
check("remap agua sin enemigo -> beacon a_short (95,11)",
      (bx, by) == BEACON_BY_MAP["fase2_a_short.oramap"])

obs_m = _obs(h=h, w=w, bldgs=("fact", "proc"), units=[_u(1, "e1", 12, 16)])
aidx_m = ActionIndex(obs_m, Vocab())
apply_passability(aidx_m, grid)
water_flat = 53 * w + 0
action_m, (_et, _eu, _ei, eff_c) = index_to_command_effective(
    obs_m, TYPE_TO_IDX["attack_move"], 0, water_flat, 0, aidx_m)
cmd_m = action_m.commands[0]
check("attack_move emitido no es agua", cmd_m.target_y < 40 and bool(aidx_m.pass_grid[cmd_m.target_y, cmd_m.target_x]))
check("eff_cell_flat = celda emitida", eff_c == cmd_m.target_y * w + cmd_m.target_x)

obs_t = _obs(harv=1, bldgs=("fact", "proc", "barr"), avail=("e1",), units=[_u(1, "harv")])
aidx_t = ActionIndex(obs_t, Vocab())
slot_t = aidx_t.items.index("infantry_basic")
action_t, _ = index_to_command_effective(
    obs_t, TYPE_TO_IDX["train"], 0, water_flat, slot_t, aidx_t)
cmd_t = action_t.commands[0]
check("TRAIN ignora celda (item e1, target 0,0)",
      cmd_t.action.value == "train" and cmd_t.item_type == "e1"
      and cmd_t.target_x == 0 and cmd_t.target_y == 0)

collapse = _collapse_row(99.0)
builder = _builder_row()
check("collapse detectado", viability_breakdown(collapse)["collapse"] is True)
check("viability builder > collapse (no usa mean_episode_reward)",
      viability_score(builder) > viability_score(collapse))
better, reason = is_strictly_better(builder, collapse)
check("builder strictly better por viability", better and reason == "higher_viability")
better2, _ = is_strictly_better(collapse, builder)
check("collapse NO reemplaza builder", better2 is False)

spam = {
    "iter": 275, "winrate_rolling20": 0.0,
    "n_buildings": {"own": 0.0, "enemy": 6.0},
    "action_hist": {"army_attack_move": 400, "no_op": 5, "build": 0},
}
check("attack-spam collapse detectado", is_attack_spam_collapse(spam) is True)
check("builder no es attack-spam", is_attack_spam_collapse(builder) is False)
check("deploy-noop collapse no es attack-spam", is_attack_spam_collapse(collapse) is False)
check("deploy-noop ES dead policy", is_dead_policy(collapse) is True)
check("reason deploy-noop", dead_policy_reason(collapse) == "deploy-noop")
check("attack-spam ES dead policy", dead_policy_reason(spam) == "attack-spam")
check("builder NO es dead policy", is_dead_policy(builder) is False)
noop_spam = {
    "iter": 329, "winrate_rolling20": 0.0, "entropy": 0.015,
    "n_buildings": {"own": 0.0, "enemy": 0.0},
    "action_hist": {"no_op": 391, "deploy": 4},
    "outcomes": ["lose", "lose", "lose", "lose"],
    "ticks": [8500, 8400, 8300, 8200],
    "reward_components": {},
}
check("Run8 no_op-spam es dead policy",
      is_dead_policy(noop_spam) is True
      and dead_policy_reason(noop_spam) in ("no_op-spam", "deploy-noop"))
check("batch 80%+ no_op se salta", batch_is_dead([
    {"action_hist": {"no_op": 90, "deploy": 2}},
    {"action_hist": {"no_op": 95, "deploy": 1}},
]) is True)
check("batch sano no se salta", batch_is_dead([
    {"action_hist": {"train": 40, "attack_move": 50, "no_op": 10}},
]) is False)

cmds_mcv = support_commands(_obs(bldgs=(), units=[_u(1, "mcv", 12, 16)],
                                 avail=("proc", "powr")))
kinds_mcv = [(c.action.value, c.item_type) for c in cmds_mcv]
check("auto-support DEPLOY MCV sin fact",
      any(c.action.value == "deploy" for c in cmds_mcv))
check("sin fact no BUILD proc el mismo bloque",
      ("build", "proc") not in kinds_mcv)

a = {"iter_winrate": 0.25, "winrate_rolling20": 0.2, "iter": 1}
b = {"iter_winrate": 0.5, "winrate_rolling20": 0.2, "iter": 2}
ok_wr, reason_wr = is_strictly_better(b, a)
check("winrate per-iter gana", ok_wr and reason_wr == "higher_iter_winrate")
ok_eq, reason_eq = is_strictly_better(a, dict(a))
check("empate no flap", (not ok_eq) and reason_eq == "not_strictly_better")

with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    latest = d / "latest.pt"
    latest.write_bytes(b"CKPT-A")
    row = _builder_row()
    row["iter"] = 3
    check("first best.pt se escribe", maybe_update_best(d, row, latest_path=str(latest)) is True)
    check("best.pt es copia (no symlink)", (d / "best.pt").read_bytes() == b"CKPT-A"
          and not (d / "best.pt").is_symlink())
    meta = json.loads((d / "best.json").read_text(encoding="utf-8"))
    check("best.json iter/reason first", meta["iter"] == 3 and meta["reason"] == "first")
    latest.write_bytes(b"CKPT-B")
    check("collapse no pisa best.pt",
          maybe_update_best(d, _collapse_row(), latest_path=str(latest)) is False
          and (d / "best.pt").read_bytes() == b"CKPT-A")
    latest.write_bytes(b"CKPT-C")
    check("igual no flap",
          maybe_update_best(d, row, latest_path=str(latest)) is False
          and (d / "best.pt").read_bytes() == b"CKPT-A")

print("\n" + ("TODOS LOS TESTS OK" if ok else "HAY FALLAS"))
sys.exit(0 if ok else 1)