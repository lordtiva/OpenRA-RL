# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openra_env.models import ActionType, CommandModel
from rl.action_adapter import ActionIndex, TYPE_TO_IDX, Vocab
from rl.imitation import (
    EliteBuffer,
    command_to_indices,
    lambda_bc_at,
    pick_bc_command,
)
from rl.scripted_teacher import ScriptedTeacher


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

ok = True


def check(name, cond):
    global ok
    print(f"  [{'OK' if cond else 'FALLA'}] {name}")
    ok = ok and bool(cond)


print("=== capa 1 imitation ===")
check("lambda start", abs(lambda_bc_at(100, 100, warmup=80) - 1.0) < 1e-9)
check("lambda mid", abs(lambda_bc_at(140, 100, warmup=80) - 0.5) < 1e-9)
check("lambda end", abs(lambda_bc_at(180, 100, warmup=80) - 0.0) < 1e-9)
check("lambda after", abs(lambda_bc_at(200, 100, warmup=80) - 0.0) < 1e-9)

cmds = [
    CommandModel(action=ActionType.GUARD, actor_id=1, target_actor_id=2),
    CommandModel(action=ActionType.BUILD, item_type="proc"),
]
picked = pick_bc_command(cmds)
check("pick salta GUARD, toma BUILD", picked.action == ActionType.BUILD)

obs = _obs(harv=1, bldgs=("fact", "proc", "barr"),
           avail=("e1", "harv", "proc", "powr", "barr"),
           units=[_u(1, "e1", 12, 16), _u(2, "harv", 14, 16)])
v = Vocab()
v.seed_roles()
aidx = ActionIndex(obs, v)

t, u, c, i = command_to_indices(
    obs, CommandModel(action=ActionType.DEPLOY, actor_id=1), aidx)
check("deploy type", t == TYPE_TO_IDX["deploy"])
check("deploy unit slot 0", u == 0)

t, u, c, i = command_to_indices(
    obs, CommandModel(action=ActionType.ARMY_ATTACK_MOVE, target_x=95, target_y=11),
    aidx)
check("army type", t == TYPE_TO_IDX["army_attack_move"])
check("army cell y=11", c // aidx.w == 11)

t, u, c, i = command_to_indices(
    obs, CommandModel(action=ActionType.TRAIN, item_type="e1"), aidx)
check("train e1 -> infantry_basic slot",
      aidx.items[i] == "infantry_basic" or "infantry" in str(aidx.items[i]))

buf = EliteBuffer(cap_steps=3)
n = buf.add_episode([{"a": 1}, {"a": 2}], {"result": "lose", "reward_components": {"raze": 0}})
check("lose sin raze no entra", n == 0 and len(buf) == 0)
n = buf.add_episode([{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}],
                    {"result": "win", "reward_components": {"raze": 1.0}})
check("win entra y cap recorta", n == 4 and len(buf) == 3)

th = ScriptedTeacher()
check("teacher proc antes de barracks",
      th.BUILD_PRIORITY.index("proc") < th.BUILD_PRIORITY.index("barracks"))

print("\n" + ("TODOS LOS TESTS OK" if ok else "HAY FALLAS"))
sys.exit(0 if ok else 1)
