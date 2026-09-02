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
    SIL_PREFER_TICKS,
    balance_bc_samples,
    command_to_indices,
    lambda_bc_at,
    pick_bc_command,
    sample_type_name,
)
from rl.scripted_teacher import ScriptedTeacher


def _u(actor_id=1, typ="e1", x=12, y=16, idle=True):
    return NS(actor_id=actor_id, type=typ, cell_x=x, cell_y=y,
              is_idle=idle, hp_percent=1.0, can_attack=True)


def _b(typ="fact", actor_id=10, x=12, y=16):
    return NS(type=typ, actor_id=actor_id, cell_x=x, cell_y=y,
              hp_percent=1.0, is_repairing=False, is_powered=True,
              can_produce=())


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
picked = pick_bc_command([
    CommandModel(action=ActionType.ARMY_ATTACK_MOVE, target_x=90, target_y=10),
    CommandModel(action=ActionType.TRAIN, item_type="e1"),
])
check("pick TRAIN gana a ARMY_ATTACK_MOVE", picked.action == ActionType.TRAIN)

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
check("sample_recent respeta max", len(buf.sample_recent(2)) == 2)
n_r = buf.add_episode([{"a": 9}],
                      {"result": "lose", "reward_components": {"raze": 0.5}})
check("raze flojo no entra", n_r == 0)
n_farm = buf.add_episode(
    [{"a": 8}], {"result": "lose", "reward_components": {"raze": 12.0}})
check("lose con raze alto no entra (SIL solo wins)", n_farm == 0)
n_inc = buf.add_episode(
    [{"a": 7}], {"result": "incomplete", "reward_components": {"raze": 22.0}})
check("incomplete con raze no entra", n_inc == 0)
n_early = buf.add_episode(
    [{"a": 6}], {"result": "win_early", "reward_components": {"raze": 0.0}})
check("win_early entra", n_early == 1)
check("SIL_PREFER_TICKS 40k", SIL_PREFER_TICKS == 40000)


def _sil_step(tag):
    return {"tag": tag}


buf_s = EliteBuffer(cap_steps=500, prefer_ticks=40000)
buf_s.add_episode([_sil_step(("L", i)) for i in range(80)],
                  {"result": "win", "ticks": 50000})
buf_s.add_episode([_sil_step(("Sa", i)) for i in range(80)],
                  {"result": "win", "ticks": 25000})
got_s = buf_s.sample_recent(20)
tags_s = [s["tag"][0] for s in got_s]
check("SIL prefiere win corto, no la cola del largo",
      set(tags_s) == {"Sa"} and len(got_s) == 20)

buf_e = EliteBuffer(cap_steps=500, prefer_ticks=40000)
buf_e.add_episode([_sil_step(("Sa", i)) for i in range(80)],
                  {"result": "win", "ticks": 20000})
buf_e.add_episode([_sil_step(("Sb", i)) for i in range(80)],
                  {"result": "win", "ticks": 22000})
got_e = buf_e.sample_recent(20)
from_a = [s["tag"][1] for s in got_e if s["tag"][0] == "Sa"]
from_b = [s["tag"][1] for s in got_e if s["tag"][0] == "Sb"]
check("SIL even-pick por win, no solo el ultimo",
      len(from_a) == 10 and len(from_b) == 10)
check("SIL even-pick recorre el win, no la cola",
      from_a[0] == 0 and from_a[-1] == 79)

buf_l = EliteBuffer(cap_steps=500, prefer_ticks=40000)
buf_l.add_episode([_sil_step(("L", i)) for i in range(80)],
                  {"result": "win", "ticks": 50000})
got_l = buf_l.sample_recent(20)
check("SIL sin corto cae al win largo",
      len(got_l) == 20 and all(s["tag"][0] == "L" for s in got_l))

buf_t = EliteBuffer(cap_steps=100, prefer_ticks=40000)
buf_t.add_episode([_sil_step(("L", i)) for i in range(80)],
                  {"result": "win", "ticks": 50000})
buf_t.add_episode([_sil_step(("Sa", i)) for i in range(80)],
                  {"result": "win", "ticks": 20000})
check("trim echa el win largo primero",
      len(buf_t) == 80 and all(s["tag"][0] == "Sa" for s in buf_t.snapshot()))

th = ScriptedTeacher()
check("teacher proc antes de barracks",
      th.BUILD_PRIORITY.index("proc") < th.BUILD_PRIORITY.index("barracks"))
check("teacher army umbral > 6", th.INFANTRY_TRAIN_TARGET >= 16)
th.phase = "attack"
obs_atk = _obs(
    cash=5000, harv=1,
    bldgs=("fact", "proc", "barr"),
    avail=("e1", "harv", "proc", "powr", "barr"),
    units=[_u(i, "e1", 12, 16) for i in range(1, 22)],
)
prod_cmds = th._handle_production(obs_atk)
check("teacher TRAIN en attack aunque ya hay 16+ e1",
      any(c.action == ActionType.TRAIN for c in prod_cmds))

import torch
from rl.action_adapter import TYPE_TO_IDX as _T


def _step(typ):
    return {"action": {"type": torch.tensor([_T[typ]])}}


tape = [_step("train")] * 12 + [_step("army_attack_move")] * 200
bal = balance_bc_samples(tape, per_type_cap=20, combat_cap=40)
n_army = sum(1 for s in bal if sample_type_name(s) == "army_attack_move")
n_train = sum(1 for s in bal if sample_type_name(s) == "train")
check("balance capea army", n_army == 40)
check("balance conserva train", n_train == 12)
check("balance no alarga", len(bal) == 52)

print("\n" + ("TODOS LOS TESTS OK" if ok else "HAY FALLAS"))
sys.exit(0 if ok else 1)
