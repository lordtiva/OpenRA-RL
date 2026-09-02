# -*- coding: utf-8 -*-
"""Unit tests: combat-train mask, legal attack-move cells, best.pt comparator."""
from __future__ import annotations

import json
import math
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
    HUNT_Y_MAX, MIN_ARMY_FOR_ASSAULT, MIN_HARVESTERS, MIN_PILE_FOR_HUNT,
    RAID_HOME_ORDERS, STAGING_STEPS,
    STANCE_ATTACK_ANYTHING, SUPPORT_ASSAULT, SUPPORT_REMNANT, SUPPORT_WAR_NUDGE,
    apply_dest_credit, remate_sweep_cell, support_commands, war_nudge_cell,
)
from rl.best_ckpt import (
    DROUGHT_PEAK,
    DROUGHT_STREAK,
    DROUGHT_WR20,
    batch_is_dead,
    dead_policy_reason,
    is_attack_spam_collapse,
    is_dead_policy,
    is_strictly_better,
    is_wr20_drought,
    maybe_update_best,
    viability_breakdown,
    viability_score,
)
from rl.network import TYPE_TO_IDX
from rl.obs_encoding import (
    BEACON_BY_MAP, MAX_ENEMIES, MAX_TOKENS, MAX_UNITS, SCALAR_DIM, UNIT_FEAT_DIM,
    select_unit_slots, unit_slots, unit_tokens,
)

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
         tick=100, spatial_map=""):
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
        spatial_map=spatial_map,
        spatial_channels=9,
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
cmds_tent = support_commands(
    _obs(harv=1, cash=5000, bldgs=("fact", "proc"),
         avail=("e1", "tent", "proc", "powr"), units=[_u(9, "harv", 14, 16)]))
check("auto-tent BUILD tent con proc",
      any(c.action.value == "build" and c.item_type == "tent" for c in cmds_tent))
cmds_barr = support_commands(
    _obs(harv=1, cash=5000, bldgs=("fact", "proc"),
         avail=("e1", "barr", "proc"), units=[_u(9, "harv", 14, 16)]))
check("auto-tent usa barr si no hay tent en avail",
      any(c.action.value == "build" and c.item_type == "barr" for c in cmds_barr))
cmds_tent_no = support_commands(
    _obs(harv=1, cash=5000, bldgs=("fact",),
         avail=("e1", "tent", "proc"), units=[_u(9, "harv", 14, 16)]))
check("auto-tent NO antes de proc",
      not any(c.action.value == "build" and c.item_type in ("tent", "barr")
              for c in cmds_tent_no))
prod_tent = [NS(queue_type="Building", item="tent", progress=1.0, paused=False)]
cmds_ptent = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), avail=("tent",), prod=prod_tent,
         units=[_u(9, "harv", 14, 16)]))
check("auto-tent PLACE tent listo",
      any(c.action.value == "place_building" and c.item_type == "tent"
          for c in cmds_ptent))
obs_wall = _obs(harv=1, bldgs=("fact", "proc", "tent"),
                avail=("e1", "sbag", "brik", "fenc", "tent", "proc"),
                units=[_u(1, "e1", 12, 16), _u(9, "harv", 14, 16)])
aidx_wall = ActionIndex(obs_wall, Vocab())
check("sbag/brik son BUILD civic, no TRAIN",
      "civic" in aidx_wall.build_items and "civic" not in aidx_wall.train_items)
cslot = (len(aidx_wall.train_items) + aidx_wall.build_items.index("civic")
         if "civic" in aidx_wall.build_items else 0)
act_wall, _ = index_to_command_effective(
    obs_wall, TYPE_TO_IDX["train"], 0, 0, cslot, aidx_wall)
iss_wall = act_wall.commands[0].item_type
check("train+muro no emite sbag/brik",
      act_wall.commands[0].action.value != "train"
      or str(iss_wall).lower() not in ("sbag", "brik", "fenc"))

# PLACE cola Defense (pbox/gun): el bug era queue_type=="Building" only.
obs_def = _obs(
    harv=1, bldgs=("fact", "proc", "tent"),
    avail=("e1", "tent", "pbox", "gun", "agun", "proc", "powr"),
    units=[_u(1, "e1", 12, 16), _u(9, "harv", 14, 16)],
    prod=[NS(queue_type="Defense", item="gun", progress=1.0, paused=False)],
)
aidx_def = ActionIndex(obs_def, Vocab())
check("PLACE legal con cola Defense",
      bool(aidx_def.type_mask[TYPE_TO_IDX["place_building"]]))
check("defense_gun concreto es pbox (más barato)",
      aidx_def.rol_a_concreto.get("defense_gun") == "pbox")
dslot = (len(aidx_def.train_items) + aidx_def.build_items.index("defense_gun")
         if "defense_gun" in aidx_def.build_items else 0)
act_def, _ = index_to_command_effective(
    obs_def, TYPE_TO_IDX["place_building"], 0, 16 * 128 + 12, dslot, aidx_def)
check("PLACE Defense emite el gun listo (no tent)",
      act_def.commands[0].action.value == "place_building"
      and act_def.commands[0].item_type == "gun")

obs_both = _obs(
    harv=1, bldgs=("fact", "proc", "tent"),
    avail=("e1", "tent", "pbox", "gun", "proc", "powr"),
    units=[_u(1, "e1", 12, 16), _u(9, "harv", 14, 16)],
    prod=[
        NS(queue_type="Building", item="tent", progress=1.0, paused=False),
        NS(queue_type="Defense", item="gun", progress=1.0, paused=False),
    ],
)
aidx_both = ActionIndex(obs_both, Vocab())
dslot2 = (len(aidx_both.train_items)
          + aidx_both.build_items.index("defense_gun"))
act_both, _ = index_to_command_effective(
    obs_both, TYPE_TO_IDX["place_building"], 0, 16 * 128 + 12, dslot2, aidx_both)
check("PLACE sampled defense_gun prefiere gun listo, no tent",
      act_both.commands[0].item_type == "gun")
tslot = (len(aidx_both.train_items)
         + aidx_both.build_items.index("barracks"))
act_tent_p, _ = index_to_command_effective(
    obs_both, TYPE_TO_IDX["place_building"], 0, 16 * 128 + 12, tslot, aidx_both)
check("PLACE sampled barracks sigue poniendo tent",
      act_tent_p.commands[0].item_type == "tent")

# Harvest: idle nunca apunta al argmax del mapa (Ch2=0 en la proc).
# Migajas junto a casa sí retargetean al parche rico CERCANO, no al lejano.
import base64 as _b64
_h, _w = 32, 32
_sp = np.zeros((_h, _w, 9), dtype=np.float32)
_sp[:, :, 3] = 1.0
_sp[:, :, 4] = 1.0
_sp[16, 10, 2] = 0.2   # migajas junto a proc (12,16)
_sp[16, 14, 2] = 5.0   # parche rico de casa
_sp[16, 28, 2] = 12.0  # más rico lejos (no ir)
_b64map = _b64.b64encode(_sp.tobytes()).decode("ascii")
obs_ore = _obs(
    harv=1, bldgs=("fact", "proc"), w=_w, h=_h,
    avail=("e1", "proc"),
    units=[_u(9, "harv", 10, 16, idle=False)],
    spatial_map=_b64map,
)
cmds_ore = support_commands(obs_ore)
harv_cmds = [c for c in cmds_ore if c.action.value == "harvest"]
check("harv en migajas de casa va al parche cercano, no al lejano",
      len(harv_cmds) == 1 and harv_cmds[0].target_x == 14
      and harv_cmds[0].target_y == 16)
obs_idle_far = _obs(
    harv=1, bldgs=("fact", "proc"), w=_w, h=_h,
    units=[_u(9, "harv", 12, 16, idle=True)],
    spatial_map=_b64map,
)
cmds_if = support_commands(obs_idle_far)
ih = [c for c in cmds_if if c.action.value == "harvest"]
check("harv idle en proc no apunta al ore lejano",
      len(ih) == 1 and int(ih[0].target_x or 0) == 0
      and int(ih[0].target_y or 0) == 0)
obs_idle_h = _obs(
    harv=1, bldgs=("fact", "proc"),
    units=[_u(9, "harv", 14, 16, idle=True)])
cmds_ih = support_commands(obs_idle_h)
check("harv idle sin spatial sigue harvest",
      any(c.action.value == "harvest" for c in cmds_ih))
check("piso easy: 2 harvs", MIN_HARVESTERS == 2)
cmds_h2 = support_commands(
    _obs(harv=1, cash=5000, bldgs=("fact", "proc"),
         avail=("e1", "harv", "proc", "tent"),
         units=[_u(9, "harv", 14, 16)]))
check("con 1 harv TRAIN el segundo (easy nace con 2)",
      any(c.action.value == "train" and c.item_type == "harv" for c in cmds_h2))
cmds_h2ok = support_commands(
    _obs(harv=2, cash=5000, bldgs=("fact", "proc"),
         avail=("e1", "harv", "proc", "tent"),
         units=[_u(9, "harv", 14, 16), _u(10, "harv", 15, 16)]))
check("con 2 harvs no TRAIN un tercero",
      not any(c.action.value == "train" and c.item_type == "harv" for c in cmds_h2ok))
cmds_husk = support_commands(
    _obs(cash=5000, bldgs=("fact", "proc"),
         avail=("harv", "proc"),
         units=[_u(9, "harv.fullhusk", 14, 16)]))
check("husk no cuenta como harv vivo",
      any(c.action.value == "train" and c.item_type == "harv" for c in cmds_husk))
obs_idle2 = _obs(
    harv=2, bldgs=("fact", "proc"),
    units=[_u(9, "harv", 12, 16, idle=True),
           _u(10, "harv", 13, 16, idle=True)])
cmds_i2 = support_commands(obs_idle2)
ih2 = [c for c in cmds_i2 if c.action.value == "harvest"]
check("2 idle: despierta ambos untargeted",
      len(ih2) == 2
      and all(int(c.target_x or 0) == 0 and int(c.target_y or 0) == 0 for c in ih2)
      and {int(c.actor_id) for c in ih2} == {9, 10})

cmds_push = support_commands(_obs(bldgs=("fact",), units=[_u(1, "e1", 12, 16)]), last_push=(90, 12))
check("support no keep-alive attack sin proc",
      not any(c.action.value == "attack_move" for c in cmds_push))
cmds_push2 = support_commands(
    _obs(bldgs=("fact", "proc"), units=[_u(1, "e1", 12, 16)]), last_push=(90, 12))
check("support keep-alive NO drip 1 rifle CON proc",
      not any(c.action.value in ("attack_move", "army_attack_move")
              for c in cmds_push2))

army4 = [_u(i, "e1", 12 + i, 16) for i in range(1, 5)] + [_u(9, "harv", 14, 16)]
army12 = [_u(i, "e1", 12 + (i % 4), 16 + (i // 4)) for i in range(1, 13)] + [
    _u(9, "harv", 14, 16)]
check("pack size", MIN_ARMY_FOR_ASSAULT == 12 and STAGING_STEPS == 10)
check("SUPPORT_ASSAULT off (no pack/hunt/rally/crédito)", SUPPORT_ASSAULT is False)
check("SUPPORT_WAR_NUDGE on (raid + contacto visible)", SUPPORT_WAR_NUDGE is True)
check("SUPPORT_REMNANT on (sweep+commit idle de campo)", SUPPORT_REMNANT is True)
check("raid peel cap", RAID_HOME_ORDERS == 6)
check("pile remate", MIN_PILE_FOR_HUNT == 4)
cmds_has_tent = support_commands(
    _obs(harv=1, cash=5000, bldgs=("fact", "proc", "tent"),
         avail=("e1", "tent", "proc"), units=army4))
check("auto-tent no spamea si ya hay tent",
      not any(c.action.value == "build" and c.item_type in ("tent", "barr")
              for c in cmds_has_tent))
cmds_drip4 = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army4, enemies=[_u(99, "e1", 90, 12)]))
check("4 rifles NO asaltan contacto lejano (pack 12)",
      not any(c.action.value in ("army_attack_move", "attack_move")
              for c in cmds_drip4))
cmds_assault = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army12, enemies=[_u(99, "e1", 90, 12)]))
aam_as = [c for c in cmds_assault if c.action.value == "army_attack_move"]
check("12 idle en casa + enemigo visible: army_attack_move a ese contacto",
      len(aam_as) == 1 and int(aam_as[0].target_x) == 90
      and int(aam_as[0].target_y) == 12)
cmds_beacon = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army12))
check("sin contacto visible: no manda al beacon (95,11)",
      not any(c.action.value in ("army_attack_move", "attack_move")
              for c in cmds_beacon))
cmds_raid = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army12,
         enemies=[_u(99, "e1", 14, 17)]))
aam_raid = [c for c in cmds_raid if c.action.value == "army_attack_move"]
am_raid = [c for c in cmds_raid if c.action.value == "attack_move"]
check("raid en casa: NO army_attack_move (no yank de grupo)",
      len(aam_raid) == 0)
check("raid en casa: attack_move idle local al raid",
      len(am_raid) == RAID_HOME_ORDERS
      and all(int(c.target_x) == 14 and int(c.target_y) == 17 for c in am_raid)
      and all(int(c.actor_id) in range(1, 13) for c in am_raid))
cmds_raid4 = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army4,
         enemies=[_u(99, "e1", 14, 17)]))
am_r4 = [c for c in cmds_raid4 if c.action.value == "attack_move"]
check("raid no espera pack 12 (peel local)",
      len(am_r4) == 4 and all(int(c.target_x) == 14 for c in am_r4)
      and not any(c.action.value == "army_attack_move" for c in cmds_raid4))
near_powr = _b("powr", 200, 35, 16)
far_e1 = _u(99, "e1", 90, 12)
cell, is_raid = war_nudge_cell(
    _obs(harv=1, bldgs=("fact", "proc"), units=army12,
         enemies=[far_e1], enemy_bldgs=[near_powr]))
check("contacto: el más lejano gana al powr de la puerta",
      is_raid is False and cell == (90, 12))
cmds_near_b = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army12,
         enemies=[far_e1], enemy_bldgs=[near_powr]))
aam_nb = [c for c in cmds_near_b if c.action.value == "army_attack_move"]
check("12 idle: marchan al contacto lejano, no al powr (35,16) ni beacon",
      len(aam_nb) == 1 and int(aam_nb[0].target_x) == 90
      and int(aam_nb[0].target_y) == 12)
far_fact = _b("fact", 201, 80, 12)
cell_prod, is_raid_p = war_nudge_cell(
    _obs(harv=1, bldgs=("fact", "proc"), units=army12,
         enemies=[far_e1], enemy_bldgs=[_b("tent", 200, 35, 16), far_fact]))
check("prod al fondo gana al tent adelantado",
      is_raid_p is False and cell_prod == (80, 12))
aidx_cred = NS(w=128, h=64)
home_flat = 16 * 128 + 12
act_army = OpenRAAction(commands=[CommandModel(
    action=ActionType.ARMY_ATTACK_MOVE, target_x=12, target_y=16)])
flat_b, xy_b = apply_dest_credit(
    _obs(harv=1, bldgs=("fact", "proc"), units=army4),
    act_army, "army_attack_move", home_flat, aidx_cred)
check("dest credit off: no pisa el click de la red",
      xy_b is None and flat_b == home_flat
      and act_army.commands[0].target_x == 12
      and act_army.commands[0].target_y == 16)
cmds_rally12 = support_commands(
    _obs(harv=1, bldgs=("fact", "proc", "tent"), units=army12))
check("sin rally de guerra al beacon",
      not any(c.action.value == "set_rally_point" for c in cmds_rally12))
cmds_weap = support_commands(
    _obs(harv=1, bldgs=("fact", "proc", "weap"), units=army4))
check("weap no rally al beacon (HARV sale de weap)",
      not any(c.action.value == "set_rally_point" for c in cmds_weap))
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
cmds_walk_raid = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army_walk,
         enemies=[_u(99, "e1", 14, 17)]))
check("raid: blob caminando en casa no se yanka (idle only)",
      not any(c.action.value in ("army_attack_move", "attack_move")
              for c in cmds_walk_raid))
field = [_u(i, "e1", 50 + i, 20, idle=False) for i in range(1, 9)]
home_idle = [_u(i, "e1", 12 + (i % 2), 16, idle=True) for i in range(9, 13)]
cmds_peel = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"),
         units=field + home_idle + [_u(99, "harv", 14, 16)],
         enemies=[_u(200, "e1", 14, 17)]))
am_peel = [c for c in cmds_peel if c.action.value == "attack_move"]
check("raid: peel solo idle en casa, el field no se toca",
      not any(c.action.value == "army_attack_move" for c in cmds_peel)
      and len(am_peel) == 4
      and set(int(c.actor_id) for c in am_peel) == {9, 10, 11, 12})
field_idle = [_u(i, "e1", 80 + i, 12, idle=True) for i in range(1, 5)]
home_few = [_u(i, "e1", 12 + (i % 2), 16, idle=True) for i in range(20, 22)]
leftover = _b("tent", 200, 90, 12)
cmds_rem = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"),
         units=field_idle + home_few + [_u(9, "harv", 14, 16)],
         enemy_bldgs=[leftover]))
am_rem = [c for c in cmds_rem if c.action.value == "attack_move"]
check("remate commit: no army_attack_move (no yank casa)",
      not any(c.action.value == "army_attack_move" for c in cmds_rem))
check("remate commit: idle de campo al leftover, casa quieta",
      len(am_rem) == 4
      and set(int(c.actor_id) for c in am_rem) == {1, 2, 3, 4}
      and all(int(c.target_x) == 90 and int(c.target_y) == 12 for c in am_rem))
cmds_drip_f = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"),
         units=field_idle[:3] + [_u(9, "harv", 14, 16)]))
check("remate no drip 3 idle de campo",
      not any(c.action.value in ("army_attack_move", "attack_move")
              for c in cmds_drip_f))
field_walk = [_u(i, "e1", 80 + i, 12, idle=False) for i in range(1, 5)]
cmds_fw = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"),
         units=field_walk + [_u(9, "harv", 14, 16)],
         enemy_bldgs=[leftover]))
check("remate no re-ordena campo que ya camina",
      not any(c.action.value in ("army_attack_move", "attack_move")
              for c in cmds_fw))
cmds_sw = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"),
         units=field_idle + home_few + [_u(9, "harv", 14, 16)],
         tick=100))
am_sw = [c for c in cmds_sw if c.action.value == "attack_move"]
sw_cell = remate_sweep_cell(
    _obs(harv=1, bldgs=("fact", "proc"), units=field_idle, tick=100),
    field_idle)
check("remate sweep: waypoint de tierra, no beacon, no casa",
      sw_cell is not None
      and sw_cell != (95, 11)
      and sw_cell[1] <= HUNT_Y_MAX
      and (abs(sw_cell[0] - 12) + abs(sw_cell[1] - 16)) > 18)
check("remate sweep: idle de campo al waypoint, casa quieta, no army",
      not any(c.action.value == "army_attack_move" for c in cmds_sw)
      and len(am_sw) == 4
      and set(int(c.actor_id) for c in am_sw) == {1, 2, 3, 4}
      and all(int(c.target_x) == sw_cell[0] and int(c.target_y) == sw_cell[1]
              for c in am_sw))
cmds_raid_field = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"),
         units=field_idle + home_few + [_u(9, "harv", 14, 16)],
         enemies=[_u(200, "e1", 14, 17)]))
am_rf = [c for c in cmds_raid_field if c.action.value == "attack_move"]
check("raid gana a remate: peel casa, campo idle no se toca",
      not any(c.action.value == "army_attack_move" for c in cmds_raid_field)
      and set(int(c.actor_id) for c in am_rf) == {20, 21})
army_walk_home = [
    _u(i, "e1", 12 + (i % 4), 16 + (i // 4), idle=False) for i in range(1, 13)
] + [_u(9, "harv", 14, 16)]
cmds_reassault = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army_walk_home))
check("re-asalto off: pack 12 caminando no va al beacon",
      not any(c.action.value in ("army_attack_move", "attack_move")
              for c in cmds_reassault))
cmds_walk12_far = support_commands(
    _obs(harv=1, bldgs=("fact", "proc"), units=army_walk_home,
         enemies=[_u(99, "e1", 90, 12)]))
check("12 caminando en casa: no yank a contacto lejano (hace falta idle)",
      not any(c.action.value in ("army_attack_move", "attack_move")
              for c in cmds_walk12_far))
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
check("blob caminando: 1 ocioso no drip al dest",
      not any(c.action.value in ("army_attack_move", "attack_move")
              for c in cmds_mix))

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

def _wr20_rows(pairs):
    return [{"iter": i, "winrate_rolling20": w} for i, w in pairs]

check("sequia: era start wr20=0 sin pico no dispara",
      is_wr20_drought(_wr20_rows([(1082 + i, 0.0) for i in range(8)])) is False)
peaked = _wr20_rows(
    [(1082, 0.40)] + [(1083 + i, 0.25) for i in range(4)]
    + [(1087 + i, 0.0) for i in range(DROUGHT_STREAK)])
check("sequia: pico 0.40 luego wr20=0 x5 dispara", is_wr20_drought(peaked) is True)
almost = _wr20_rows(
    [(1082, 0.40)] + [(1083 + i, 0.0) for i in range(DROUGHT_STREAK - 1)]
    + [(1082 + DROUGHT_STREAK, 0.10)])
check("sequia: 4 ceros y un 0.10 no dispara", is_wr20_drought(almost) is False)
floor = _wr20_rows(
    [(1082, 0.40)] + [(1083 + i, DROUGHT_WR20) for i in range(DROUGHT_STREAK)])
check("sequia: wr20==piso 0.05 x5 dispara", is_wr20_drought(floor) is True)
low_peak = _wr20_rows(
    [(1082, DROUGHT_PEAK - 0.05)] + [(1083 + i, 0.0) for i in range(DROUGHT_STREAK)])
check("sequia: pico 0.15 < 0.20 no dispara", is_wr20_drought(low_peak) is False)
old_peak = _wr20_rows(
    [(1070, 0.40)] + [(1147 + i, 0.0) for i in range(DROUGHT_STREAK)])
check("sequia: pico anterior al restore no cuenta",
      is_wr20_drought(old_peak, since_iter=1146) is False)
alive = {
    "iter": 1148, "winrate_rolling20": 0.0, "entropy": 1.6,
    "n_buildings": {"own": 1.4, "enemy": 8.0},
    "action_hist": {"no_op": 40, "train": 20, "harvest": 5, "build": 10},
}
check("Run32 wr20=0 con H alta NO es dead_policy", is_dead_policy(alive) is False)
check("DROUGHT_STREAK es 5", DROUGHT_STREAK == 5)

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

with tempfile.TemporaryDirectory() as td_bt:
    d = Path(td_bt)
    latest = d / "latest.pt"
    latest.write_bytes(b"BEG")
    beg = dict(_builder_row())
    beg.update({"iter": 951, "iter_winrate": 1.0, "winrate": 0.879,
                "bot_type": "beginner", "winrate_rolling20": 1.0})
    check("beginner 4/4 se escribe", maybe_update_best(d, beg, latest_path=str(latest)))
    latest.write_bytes(b"EASY")
    easy = dict(_builder_row())
    easy.update({"iter": 970, "iter_winrate": 0.25, "winrate": 0.066,
                 "bot_type": "easy", "winrate_rolling20": 0.2})
    check("easy pisa beginner (nuevo rival, no iwr 1.0)",
          maybe_update_best(d, easy, latest_path=str(latest)) is True
          and (d / "best.pt").read_bytes() == b"EASY")
    meta_e = json.loads((d / "best.json").read_text(encoding="utf-8"))
    check("best.json bot_type easy", meta_e.get("bot_type") == "easy"
          and meta_e.get("reason") == "new_bot_type")

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

from rl.play_skirmish import action_to_dicts, obs_from_dict
from rl.play_vs_checkpoint_live import (
    _append_live_rows, _harv_xy, _n_combat, _resolve_log_path, _tape_row,
)
check("tape path vacío no escribe", _resolve_log_path("") is None)
with tempfile.TemporaryDirectory() as td:
    p_tape = Path(td) / "live_tape.jsonl"
    _append_live_rows(p_tape, [{"ep": "a", "dec": 1}, {"ep": "a", "dec": 2}])
    lines = p_tape.read_text(encoding="utf-8").strip().splitlines()
    check("tape se flushea junto al terminar", len(lines) == 2)
obs_t = _obs(harv=1, bldgs=("fact", "proc"),
             units=[_u(1, "e1", 12, 16), _u(9, "harv", 14, 17)])
check("tape harv xy", _harv_xy(obs_t.units) == [[14, 17]])
check("tape n_combat ignora harv", _n_combat(obs_t.units) == 1)
row_t = _tape_row(obs_t, ep="x", ckpt=899, dec=3, pol="train",
                  cell=None, item="infantry_basic", sup=[95, 11], supk="army")
check("tape row dest y harv",
      row_t["sup"] == [95, 11] and row_t["harv"] == [[14, 17]]
      and row_t["pol"] == "train" and row_t["nh"] == 1)

import torch
from rl.network import AlphaLiteNet
net_nan = AlphaLiteNet()
bad = torch.full((1, 22), float("nan"))
cat_nan = AlphaLiteNet._categorical(bad)
check("categorical nan no crashea", torch.isfinite(cat_nan.logits).all().item())
check("categorical nan cae a no_op (slot 0)",
      int(cat_nan.probs.argmax(-1).item()) == 0)

from rl.trainer import PPOTrainer


class _BoomNet(torch.nn.Module):
    """evaluate_actions_seq returns canned tensors; w exists so Adam has a param."""

    def __init__(self, lp_new, entropy, value):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self._lp = lp_new
        self._ent = entropy
        self._val = value

    def evaluate_actions_seq(self, seg, device):
        n = len(seg)
        extra = self.w.expand(n)
        return self._lp[:n] + extra, self._ent[:n], self._val[:n]


def _seg_sample(lp_old, adv=-1.0, ret=0.0, value_pred=0.0):
    return {
        "_ep": 0,
        "action": {"log_prob": torch.tensor([lp_old])},
        "adv": adv,
        "ret": ret,
        "value_pred": value_pred,
    }


boom = _BoomNet(
    lp_new=torch.tensor([0.0]),
    entropy=torch.tensor([1.0]),
    value=torch.tensor([0.0]),
)
tr = PPOTrainer(boom, lr=1e-3, device="cpu")
st_boom = tr.update([_seg_sample(-1000.0, adv=-1.0)], epochs=1, batch_size=1)
check("ppo ratio inf no envenena pi_loss", math.isfinite(st_boom.get("pi_loss", 0)))
check("ppo ratio inf no envenena pesos", torch.isfinite(boom.w).all().item())

sil_net = _BoomNet(
    lp_new=torch.full((1,), -20.0),
    entropy=torch.tensor([1.0]),
    value=torch.tensor([0.0]),
)
tr_sil = PPOTrainer(sil_net, lr=1e-3, device="cpu")
sil_nll = tr_sil.imitation_update(
    [_seg_sample(-20.0)], coef=0.5, epochs=1, batch_size=1)
check("sil nll saturado se salta (no 20)", sil_nll == 0.0)
check("sil nll saturado no envenena pesos", torch.isfinite(sil_net.w).all().item())

act_sk = OpenRAAction(commands=[
    CommandModel(action=ActionType.ARMY_ATTACK_MOVE, target_x=95, target_y=11),
    CommandModel(action=ActionType.TRAIN, item_type="e1"),
])
dicts_sk = action_to_dicts(act_sk)
from rl.network import (
    CELL_HEAD_OLD_IN, HIDDEN_DIM, ROLE_EMB_DIM, SCATTER_CH, SPATIAL_CH,
    TYPE_TO_IDX, UNIT_COND_DIM, UNIT_MLP_IN, adapt_capa2_state_dict,
    adapt_capa2c_state_dict,
)
from rl.obs_encoding import MAX_UNITS
from rl.trainer import load_checkpoint as _load_ckpt

net_c2 = AlphaLiteNet()
n_params = sum(p.numel() for p in net_c2.parameters())
check("capa2 params en rango 2.5-8M", 2.5e6 < n_params < 8e6)
check("cell_head Capa 2 in_ch",
      net_c2.cell_head.weight.shape[1]
      == SPATIAL_CH + SCATTER_CH + 64 + 64 + UNIT_COND_DIM)

H, W = 16, 16
feats = torch.zeros(1, MAX_UNITS, UNIT_FEAT_DIM)
feats[0, 0, 7], feats[0, 0, 8] = 10 / 128.0, 10 / 128.0
feats[0, 1, 7], feats[0, 1, 8] = 12 / 128.0, 4 / 128.0
valid = torch.zeros(1, MAX_UNITS, dtype=torch.bool)
valid[0, 0] = True
valid[0, 1] = True
spatial = torch.zeros(1, 9, H, W)
scalars = torch.zeros(1, SCALAR_DIM)
h0 = torch.zeros(1, HIDDEN_DIM)
torch.nn.init.normal_(net_c2.unit_cond_proj.weight, 0.0, 0.3)
torch.nn.init.zeros_(net_c2.unit_cond_proj.bias)
fmap, _, h, tokens = net_c2.encode(spatial, scalars, feats, valid, h0)
t_am = torch.tensor([TYPE_TO_IDX["attack_move"]])
c_mask = torch.ones(1, H * W, dtype=torch.bool)
lc0 = net_c2._logits_cell(fmap, t_am, c_mask, h, tokens, feats, valid,
                          torch.tensor([0]))
lc1 = net_c2._logits_cell(fmap, t_am, c_mask, h, tokens, feats, valid,
                          torch.tensor([1]))
check("dist_cell condiciona al slot (rifle ≠ MCV)",
      not torch.allclose(lc0, lc1))

torch.nn.init.ones_(net_c2.scatter_proj.weight)
torch.nn.init.ones_(net_c2.scatter_proj.bias)
sc = net_c2._scatter_units(feats, valid, (H, W))
check("scatter pinta la celda de la unidad",
      float(sc[0, :, 10, 10].abs().sum()) > 0)

old = {k: v.detach().clone() for k, v in net_c2.state_dict().items()
       if not k.startswith(("unit_xf", "scatter_proj", "unit_cond",
                            "cell_head"))}
old["cell_head.weight"] = torch.randn(1, CELL_HEAD_OLD_IN, 1, 1)
old["cell_head.bias"] = torch.zeros(1)
fresh = AlphaLiteNet()
adapted = adapt_capa2_state_dict(fresh, old)
check("Net2Net cell_head shape Capa 2",
      adapted["cell_head.weight"].shape == fresh.cell_head.weight.shape)
check("Net2Net copia fmap 96",
      torch.allclose(adapted["cell_head.weight"][:, :SPATIAL_CH],
                     old["cell_head.weight"][:, :SPATIAL_CH]))
check("Net2Net scatter extra es 0",
      float(adapted["cell_head.weight"][:, SPATIAL_CH:SPATIAL_CH + SCATTER_CH]
            .abs().sum()) == 0.0)

ckpt_922 = Path("rl/ckpts/best.pt")
if ckpt_922.exists():
    loaded = AlphaLiteNet()
    it_c2 = _load_ckpt(str(ckpt_922), loaded)
    check("Capa 2 carga best.pt", it_c2 >= 900)
    batch = {
        "spatial": torch.zeros(1, 9, 8, 8),
        "scalars": torch.zeros(1, SCALAR_DIM),
        "unit_feats": feats[:, :, :].contiguous()[:, :MAX_UNITS],
        "unit_valid": valid,
        "type_mask": torch.ones(1, 22, dtype=torch.bool),
        "cell_mask": torch.ones(1, 8 * 8, dtype=torch.bool),
        "item_indices": torch.zeros(1, 4, dtype=torch.long),
        "item_mask": torch.zeros(1, 4, dtype=torch.bool),
        "train_slot_mask": torch.zeros(1, 4, dtype=torch.bool),
        "build_slot_mask": torch.zeros(1, 4, dtype=torch.bool),
    }
    # feats is 16-map coords; act on 8x8 still ok (scatter clamps)
    out_c2 = loaded.act(batch, torch.zeros(1, HIDDEN_DIM))
    check("Capa 2 act(922) log_prob finito",
          torch.isfinite(out_c2["log_prob"]).all().item())

check("skirmish action_to_dicts names",
      dicts_sk[0]["action"] == "army_attack_move" and dicts_sk[1]["item_type"] == "e1")
obs_sk = obs_from_dict({
    "tick": 80,
    "economy": {"cash": 5000},
    "military": {},
    "units": [{"actor_id": 1, "type": "e1", "cell_x": 12, "cell_y": 16}],
    "buildings": [],
    "production": [],
    "visible_enemies": [],
    "visible_enemy_buildings": [],
    "map_info": {"width": 112, "height": 54, "map_name": "Singles"},
    "available_production": ["e1"],
    "done": False,
    "spatial_map": "",
    "spatial_channels": 9,
})
check("skirmish obs_from_dict",
      obs_sk.tick == 80 and obs_sk.map_info.map_name == "Singles"
      and obs_sk.units[0].type == "e1" and obs_sk.economy.cash == 5000)

# --- Capa 2c-A: 96 slots + combat-first ---
check("MAX_UNITS 96", MAX_UNITS == 96)
check("UNIT_FEAT_DIM 11", UNIT_FEAT_DIM == 11)
check("MAX_TOKENS 128", MAX_TOKENS == MAX_UNITS + MAX_ENEMIES == 128)
check("UNIT_MLP_IN 19", UNIT_MLP_IN == 11 + ROLE_EMB_DIM)

raid_enemy = _u(900, "e1", 52, 16)
old_home = [_u(i + 1, "e1", 12, 16) for i in range(80)]
new_raid = [_u(200 + i, "e1", 50, 16) for i in range(30)]
obs_raid = _obs(units=old_home + new_raid, enemies=(raid_enemy,),
                bldgs=("fact",))
picked_raid = select_unit_slots(obs_raid)
ids_raid = [u.actor_id for u in picked_raid]
check("combat-first: 30 del raid entran",
      all((200 + i) in ids_raid for i in range(30)))
check("combat-first: cap 96", len(picked_raid) == 96)
check("combat-first: tensor ordenado por actor_id",
      ids_raid == sorted(ids_raid))
check("combat-first: no hay huecos de sort",
      ids_raid == sorted(u.actor_id for u in picked_raid))

tiny = [_u(i + 1, "e1", 12, 16) for i in range(10)]
obs_tiny = _obs(units=tiny)
feats_t, valid_t = unit_slots(obs_tiny)
check("10 unidades: feats N=10", feats_t.shape == (10, UNIT_FEAT_DIM))
check("10 unidades: valid 10", int(valid_t.sum()) == 10)
check("10 unidades team=0", float(feats_t[0, 10]) == 0.0)

aidx_raid = ActionIndex(obs_raid, Vocab())
enc_ids = [u.actor_id for u in select_unit_slots(obs_raid)]
check("adapter y encoding eligen los mismos actor_id",
      list(aidx_raid.unit_ids) == enc_ids)
check("adapter valid[:len] True",
      bool(aidx_raid.unit_valid[:len(enc_ids)].all())
      and not bool(aidx_raid.unit_valid[len(enc_ids):].any()))

harvs = [_u(i + 1, "harv", 12, 16) for i in range(20)]
for h in harvs:
    h.can_attack = False
e1s = [_u(100 + i, "e1", 40, 16) for i in range(40)]
obs_h = _obs(units=harvs + e1s)
ids_h = [u.actor_id for u in select_unit_slots(obs_h)]
check("harv no pisa a combate",
      all((100 + i) in ids_h for i in range(40)))
check("con cupo, harv rellena",
      len(ids_h) == 60 and any(i <= 20 for i in ids_h))

home_raid_e = _u(901, "e1", 14, 16)
old_ore = [_u(i + 1, "e1", 80, 16) for i in range(80)]
new_yard = [_u(300 + i, "e1", 12, 16) for i in range(30)]
obs_home = _obs(units=old_ore + new_yard, enemies=(home_raid_e,),
                bldgs=("fact",))
ids_home = [u.actor_id for u in select_unit_slots(obs_home)]
check("raid en casa: e1 del yard entran",
      all((300 + i) in ids_home for i in range(30)))

empty_obs = _obs(units=[])
feats_e, valid_e = unit_slots(empty_obs)
check("sin unidades: feats (0,11)", feats_e.shape == (0, UNIT_FEAT_DIM))

from rl.roles import ROLE_VOCAB, role_id_of
check("e1 y e3 roles distintos",
      role_id_of("e1") != role_id_of("e3")
      and role_id_of("e1") == ROLE_VOCAB["infantry_basic"]
      and role_id_of("e3") == ROLE_VOCAB["infantry_antiarmor"])
check("role pad id 0", ROLE_VOCAB["pad"] == 0)
tok_f, tok_r, tok_v, tok_o = unit_tokens(obs_tiny)
check("tokens padded 128", tok_f.shape == (MAX_TOKENS, UNIT_FEAT_DIM))
check("tokens 10 own", int(tok_o.sum()) == 10 and int(tok_v.sum()) == 10)
obs_ene = _obs(units=[_u(1, "e1", 12, 16)],
               enemies=(_u(50, "2tnk", 40, 16),))
tf, tr, tv, to = unit_tokens(obs_ene)
check("enemigo team=1", float(tf[MAX_UNITS, 10]) == 1.0)
check("enemigo 2tnk tank_medium",
      int(tr[MAX_UNITS]) == ROLE_VOCAB["tank_medium"])
check("enemigo valid no-own", bool(tv[MAX_UNITS]) and not bool(to[MAX_UNITS]))
check("propia team=0", float(tf[0, 10]) == 0.0)

ckpt_a = Path("rl/ckpts/Run 25 (a_short harvest-fix smoke 1000-1013)/best.pt")
if not ckpt_a.exists():
    ckpt_a = Path("rl/ckpts/Run 23 (a_short capa2c-A 96 977-1002)/best.pt")
if ckpt_a.exists():
    blob_a = __import__("torch").load(
        ckpt_a, map_location="cpu", weights_only=False)
    net_a = AlphaLiteNet()
    adapted_a = adapt_capa2c_state_dict(
        net_a, adapt_capa2_state_dict(net_a, blob_a["net"]))
    old_mlp = blob_a["net"]["unit_mlp.0.weight"]
    check("2c-B mlp copia feats 10",
          torch.allclose(adapted_a["unit_mlp.0.weight"][:, :10], old_mlp))
    check("2c-B mlp extra ≈0",
          float(adapted_a["unit_mlp.0.weight"][:, 10:].abs().sum()) == 0.0)
    old_sc = blob_a["net"]["scatter_proj.weight"]
    check("2c-B scatter copia 10",
          torch.allclose(adapted_a["scatter_proj.weight"][:, :10], old_sc))
    inc_a = net_a.load_state_dict(adapted_a, strict=False)
    check("2c-B missing role_emb",
          any("role_emb" in k for k in inc_a.missing_keys))
    check("2c-B unexpected 0", len(inc_a.unexpected_keys) == 0)
    check("2c-B resume iter >=970", int(blob_a.get("iteration", 0) or 0) >= 970)
    batch_a = {
        "spatial": torch.zeros(1, 9, 8, 8),
        "scalars": torch.zeros(1, SCALAR_DIM),
        "unit_feats": torch.zeros(1, MAX_TOKENS, UNIT_FEAT_DIM),
        "unit_valid": torch.zeros(1, MAX_TOKENS, dtype=torch.bool),
        "unit_role_ids": torch.zeros(1, MAX_TOKENS, dtype=torch.long),
        "unit_own_mask": torch.zeros(1, MAX_TOKENS, dtype=torch.bool),
        "type_mask": torch.ones(1, 22, dtype=torch.bool),
        "cell_mask": torch.ones(1, 8 * 8, dtype=torch.bool),
        "item_indices": torch.zeros(1, 4, dtype=torch.long),
        "item_mask": torch.zeros(1, 4, dtype=torch.bool),
        "train_slot_mask": torch.zeros(1, 4, dtype=torch.bool),
        "build_slot_mask": torch.zeros(1, 4, dtype=torch.bool),
    }
    batch_a["unit_valid"][0, 0] = True
    batch_a["unit_own_mask"][0, 0] = True
    batch_a["unit_valid"][0, MAX_UNITS] = True
    batch_a["unit_feats"][0, MAX_UNITS, 10] = 1.0
    out_a = net_a.act(batch_a, torch.zeros(1, HIDDEN_DIM))
    check("2c-B act(128 tokens) log_prob finito",
          torch.isfinite(out_a["log_prob"]).all().item())
    net_a.eval()
    with torch.no_grad():
        hits = []
        for _ in range(40):
            o = net_a.act(batch_a, torch.zeros(1, HIDDEN_DIM), temperature=1.0)
            hits.append(int(o["unit_slot"].item()))
    check("dist_unit no samplea slot enemigo",
          all(h != MAX_UNITS for h in hits) and all(h < MAX_UNITS for h in hits))

print("\n" + ("TODOS LOS TESTS OK" if ok else "HAY FALLAS"))
sys.exit(0 if ok else 1)

