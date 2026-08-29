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
from rl.auto_support import support_commands
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


def _b(typ="fact", actor_id=10, x=12, y=16):
    return NS(type=typ, actor_id=actor_id, cell_x=x, cell_y=y,
              hp_percent=1.0, is_repairing=False, is_powered=True)


def _obs(*, cash=5000, harv=0, bldgs=("fact",), units=None, prod=(),
         avail=("e1", "proc", "powr", "barr"), w=128, h=128,
         map_name="fase2_a_short.oramap", enemies=(), enemy_bldgs=()):
    if units is None:
        units = [_u(1, "mcv", 12, 16)]
    return NS(
        tick=100,
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