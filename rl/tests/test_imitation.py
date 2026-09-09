# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openra_env.models import ActionType, CommandModel
from rl.action_adapter import ActionIndex, TYPE_TO_IDX, Vocab
from rl.imitation import (
    EliteBuffer,
    TeacherWinBuffer,
    SIL_PREFER_TICKS,
    TAPE_SCHEMA,
    balance_bc_samples,
    command_to_indices,
    lambda_bc_at,
    merge_teacher_wins,
    pick_bc_command,
    pick_bc_commands,
    sample_type_name,
    student_combat_ready,
)
from rl.network import COMBAT_PUSH_TYPES, TYPE_TO_IDX as NET_TYPE_TO_IDX, build_combat_type_mask
from rl.obs_encoding import EnemyBeliefStore
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
                    units_killed=0, units_dead=0, units_lost=0, army_value=0),
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
check("lambda piso", abs(lambda_bc_at(200, 100, warmup=80, end=0.25) - 0.25) < 1e-9)

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

obs_open = _obs(harv=1, bldgs=("fact", "proc"),
                units=[_u(1, "e1", 12, 16)])
both_cmds = [
    CommandModel(action=ActionType.ARMY_ATTACK_MOVE, target_x=90, target_y=10),
    CommandModel(action=ActionType.TRAIN, item_type="e1"),
]
open_picks = pick_bc_commands(both_cmds, obs_open)
check("opening: solo TRAIN (sin leftover, 1 rifle)",
      len(open_picks) == 1 and open_picks[0].action == ActionType.TRAIN)

obs_atk = _obs(
    harv=1, bldgs=("fact", "proc", "barr"),
    units=[_u(i, "e1", 12, 16) for i in range(1, 9)],
    enemy_bldgs=[_b("proc", 200, 80, 20)],
)
atk_picks = pick_bc_commands(both_cmds, obs_atk)
check("attack leftover: TRAIN + army",
      len(atk_picks) == 2
      and atk_picks[0].action == ActionType.TRAIN
      and atk_picks[1].action == ActionType.ARMY_ATTACK_MOVE)

obs_rush = _obs(
    harv=1, bldgs=("fact", "proc", "barr"),
    units=[_u(i, "e1", 12, 16) for i in range(1, 9)],
)
rush_picks = pick_bc_commands(both_cmds, obs_rush)
check("rush 8 sin leftover: TRAIN + army",
      len(rush_picks) == 2
      and rush_picks[1].action == ActionType.ARMY_ATTACK_MOVE)

# student dual-emit readiness + combat type mask
check("student not ready opening", not student_combat_ready(obs_open))
check("student ready leftover", student_combat_ready(obs_atk))
check("student ready rush pack", student_combat_ready(obs_rush))
bel = EnemyBeliefStore()
bel.enemy_base_xy = (70, 22)
bel.enemy_base_count = 3
check("student ready via belief base", student_combat_ready(obs_open, bel))
import torch as _torch
_base = _torch.ones(1, len(NET_TYPE_TO_IDX), dtype=_torch.bool)
# disable non-combat to mirror a sparse mask — keep combat on
_cm = build_combat_type_mask(_base)
check("combat mask not None", _cm is not None)
check("combat mask only push types",
      all(bool(_cm[0, NET_TYPE_TO_IDX[n]]) for n in COMBAT_PUSH_TYPES
          if n in NET_TYPE_TO_IDX)
      and sum(int(x) for x in _cm[0].tolist()) == len(
          [n for n in COMBAT_PUSH_TYPES if n in NET_TYPE_TO_IDX]))
_empty = _torch.zeros(1, len(NET_TYPE_TO_IDX), dtype=_torch.bool)
check("combat mask None if illegal", build_combat_type_mask(_empty) is None)

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

import tempfile as _tf
_el_dir = Path(_tf.mkdtemp(prefix="elite_"))
buf_e.save(_el_dir / "elite.pt")
buf_load = EliteBuffer(cap_steps=500, prefer_ticks=40000)
n_load = buf_load.load(_el_dir / "elite.pt")
check("elite save/load steps", n_load == len(buf_e) and n_load > 0)
check("elite load missing = 0",
      EliteBuffer().load(_el_dir / "nope.pt") == 0)

th = ScriptedTeacher()
from rl.action_adapter import PACK_ARMY as _PACK

check("teacher proc antes de barracks",
      th.BUILD_PRIORITY.index("proc") < th.BUILD_PRIORITY.index("barracks"))
check("teacher rush sin weap", "weap" not in th.BUILD_PRIORITY)
check("teacher pack = PACK_ARMY", th.INFANTRY_TRAIN_TARGET == _PACK)
check("teacher rush AM 8", th.RUSH_ATTACK_MOVE == 8)
check("teacher 0 guards", th.GUARD_COUNT == 0)
th.phase = "build_base"
th._update_phase(_obs(
    cash=5000, harv=1,
    bldgs=("fact", "proc", "barr"),
    units=[_u(1, "e1", 12, 16)],
))
check("barracks => train_army (no espera weap)", th.phase == "train_army")
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
check("teacher no encola APC",
      all(getattr(c, "item_type", "") != "apc" for c in prod_cmds))

obs_fog = _obs(
    cash=5000, harv=1,
    bldgs=("fact", "proc", "barr"),
    units=[_u(i, "e1", 12, 16) for i in range(1, 6)],
)
th.phase = "train_army"
scout = th._handle_combat(obs_fog)
check("sin pack no army_attack_move",
      all(c.action != ActionType.ARMY_ATTACK_MOVE for c in scout))
check("sin pack scoutea",
      any(c.action == ActionType.ATTACK_MOVE for c in scout))
obs_rush = _obs(
    cash=5000, harv=1,
    bldgs=("fact", "proc", "barr"),
    units=[_u(i, "e1", 12, 16) for i in range(1, 9)],
    enemy_bldgs=[_b("proc", 200, 80, 20)],
)
th.phase = "attack"
rush = th._handle_combat(obs_rush)
check("rush 8 no army_attack_move",
      all(c.action != ActionType.ARMY_ATTACK_MOVE for c in rush))
check("rush 8 mueve el blob",
      sum(1 for c in rush if c.action == ActionType.ATTACK_MOVE) >= 8)
dest = th._find_attack_target(obs_fog)
check("dest no es beacon GPS", dest != (95, 11))
check("dest no es el yard", dest != (12, 16))

obs_pack = _obs(
    cash=5000, harv=1,
    bldgs=("fact", "proc", "barr"),
    units=[_u(i, "e1", 12, 16) for i in range(1, 14)],
)
th.phase = "attack"
push = th._handle_combat(obs_pack)
check("con pack emite army_attack_move",
      any(c.action == ActionType.ARMY_ATTACK_MOVE for c in push))
am = next(c for c in push if c.action == ActionType.ARMY_ATTACK_MOVE)
check("army no va al beacon GPS", (am.target_x, am.target_y) != (95, 11))

# Visible leftover beats beacon GPS even if map_name has a beacon entry.
obs_vis = _obs(
    cash=5000, harv=1,
    bldgs=("fact", "proc", "barr"),
    units=[_u(i, "e1", 12, 16) for i in range(1, 14)],
    enemy_bldgs=[_b("proc", 200, 70, 22)],
)
th2 = ScriptedTeacher()
th2.phase = "attack"
cell = th2._push_cell(obs_vis)
check("push_cell prefiere visible enemigo sobre beacon",
      cell == (70, 22))
check("push_cell visible != beacon", cell != (95, 11))

# Ghost last_seen: after seeing an enemy then fog, hunt the ghost cell.
th3 = ScriptedTeacher()
obs_see = _obs(
    cash=5000, harv=1,
    bldgs=("fact", "proc", "barr"),
    units=[_u(i, "e1", 12, 16) for i in range(1, 10)],
    enemies=[_u(99, "e1", 80, 18)],
)
th3._push_cell(obs_see)
obs_fog2 = _obs(
    cash=5000, harv=1,
    bldgs=("fact", "proc", "barr"),
    units=[_u(i, "e1", 12, 16) for i in range(1, 10)],
)
cell_g = th3._push_cell(obs_fog2)
check("push_cell usa last_seen ghost tras fog", cell_g == (80, 18))

# Mental enemy-base: first building sets belief; denser cluster updates;
# push uses mental base after leftovers clear; GPS only if belief empty.
from rl.obs_encoding import EnemyBeliefStore, scalar_features, SCALAR_DIM
bel = EnemyBeliefStore()
obs_b1 = _obs(
    cash=5000, harv=1,
    bldgs=("fact", "proc", "barr"),
    units=[_u(i, "e1", 12, 16) for i in range(1, 10)],
    enemy_bldgs=[_b("powr", 201, 60, 20)],
)
bel.update(obs_b1)
check("first enemy building sets mental base", bel.enemy_base_xy == (60, 20))
check("first building count=1", bel.enemy_base_count == 1)
obs_sparse = _obs(
    cash=5000, harv=1,
    bldgs=("fact", "proc", "barr"),
    units=[_u(i, "e1", 12, 16) for i in range(1, 10)],
    enemy_bldgs=[_b("powr", 202, 90, 40)],
)
bel.update(obs_sparse)
check("sparser/equal count does not move base", bel.enemy_base_xy == (60, 20))
obs_dense = _obs(
    cash=5000, harv=1,
    bldgs=("fact", "proc", "barr"),
    units=[_u(i, "e1", 12, 16) for i in range(1, 10)],
    enemy_bldgs=[
        _b("proc", 210, 88, 18),
        _b("tent", 211, 90, 18),
        _b("powr", 212, 89, 20),
    ],
)
bel.update(obs_dense)
check("denser cluster updates mental base", bel.enemy_base_count == 3)
check("denser cluster near (88-90,18-20)",
      bel.enemy_base_xy is not None
      and 86 <= bel.enemy_base_xy[0] <= 92
      and 16 <= bel.enemy_base_xy[1] <= 22)

th4 = ScriptedTeacher()
th4.phase = "attack"
# Seed belief via visible buildings then fog — push should use mental base.
obs_see_b = _obs(
    cash=5000, harv=1,
    bldgs=("fact", "proc", "barr"),
    units=[_u(i, "e1", 12, 16) for i in range(1, 14)],
    enemy_bldgs=[
        _b("proc", 220, 70, 22),
        _b("tent", 221, 72, 22),
        _b("powr", 222, 71, 24),
    ],
)
cell_vis = th4._push_cell(obs_see_b)
check("push prefers visible leftover over mental", cell_vis is not None)
obs_fog_base = _obs(
    cash=5000, harv=1,
    bldgs=("fact", "proc", "barr"),
    units=[_u(i, "e1", 12, 16) for i in range(1, 14)],
)
cell_mb = th4._push_cell(obs_fog_base)
check("push uses mental base after leftovers cleared",
      cell_mb == th4.belief.enemy_base_xy)
check("mental base push != GPS beacon", cell_mb != (95, 11))

sc = scalar_features(obs_fog_base, belief=th4.belief)
check("SCALAR_DIM is 33", SCALAR_DIM == 33 and sc.shape == (33,))
check("has_enemy_base_belief scalar on", float(sc[25]) == 1.0)
check("base_conf > 0", float(sc[28]) > 0.0)

obs_mill = _obs(
    cash=5000, harv=1,
    bldgs=("fact", "proc", "barr"),
    units=[_u(i, "e1", 95, 11, idle=False) for i in range(1, 14)],
)
th.phase = "attack"
th._last_contact = (95, 11)
mill = th._handle_combat(obs_mill)
mill_am = [c for c in mill if c.action == ActionType.ARMY_ATTACK_MOVE]
check("mill pile lejos (no idle) reemite army", bool(mill_am))
check("mill hunt no se queda en la pila",
      mill_am and (mill_am[0].target_x, mill_am[0].target_y) != (95, 11))

obs_rush_mill = _obs(
    cash=5000, harv=1,
    bldgs=("fact", "proc", "barr"),
    units=[_u(i, "e1", 95, 11, idle=False) for i in range(1, 9)],
)
th.phase = "attack"
th._last_contact = (95, 11)
rmill = th._handle_combat(obs_rush_mill)
check("rush mill no idle emite AM",
      any(c.action == ActionType.ATTACK_MOVE for c in rmill))
check("rush mill hunt no todos en la pila",
      any((c.target_x, c.target_y) != (95, 11)
          for c in rmill if c.action == ActionType.ATTACK_MOVE))

# BC label: beacon-aimed army retargets to visible leftover.
from rl.obs_encoding import resolve_beacon
beacon_cmd = CommandModel(
    action=ActionType.ARMY_ATTACK_MOVE, target_x=95, target_y=11)
train_cmd = CommandModel(action=ActionType.TRAIN, item_type="e1")
bc_ret = pick_bc_commands([beacon_cmd, train_cmd], obs_vis)
check("BC attack retargetea lejos del beacon",
      len(bc_ret) == 2
      and bc_ret[1].action == ActionType.ARMY_ATTACK_MOVE
      and (bc_ret[1].target_x, bc_ret[1].target_y) == (70, 22))

obs_raid = _obs(
    cash=5000, harv=1,
    bldgs=("fact", "proc", "barr"),
    units=[_u(i, "e1", 12, 16) for i in range(1, 14)],
    enemies=[_u(99, "e1", 14, 16)],
)
peel = th._handle_combat(obs_raid)
check("raid peel no yank de grupo",
      all(c.action != ActionType.ARMY_ATTACK_MOVE for c in peel))
check("raid peel attack_move",
      any(c.action == ActionType.ATTACK_MOVE for c in peel))

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
import inspect
_sig = inspect.signature(balance_bc_samples)
check("balance defaults 512/512",
      _sig.parameters["per_type_cap"].default == 512
      and _sig.parameters["combat_cap"].default == 512)

# Opening prior: fresh teacher, empty belief → beacon when map has GPS.
th_open = ScriptedTeacher()
obs_open = _obs(
    cash=5000, harv=1,
    bldgs=("fact", "proc", "barr"),
    units=[_u(i, "e1", 12, 16) for i in range(1, 14)],
)
cell_open = th_open._push_cell(obs_open)
check("empty belief push uses beacon opening prior",
      cell_open == resolve_beacon(obs_open) == (95, 11))

win_s = [_step("train")] * 3
lose_s = [_step("no_op")] * 5
kept, meta, kept_eps = merge_teacher_wins([
    (win_s, {"result": "win", "ticks": 20000}),
    (lose_s, {"result": "lose", "ticks": 9000}),
    (win_s, {"result": "win_early", "ticks": 15000}),
])
check("merge descarta lose", len(kept) == 6)
check("merge cuenta 2 wins", meta["bc_n_win_eps"] == 2)
check("merge n_eps 3", meta["bc_n_eps"] == 3)
check("merge kept_eps 2", len(kept_eps) == 2)
empty, meta0, empty_eps = merge_teacher_wins([
    (lose_s, {"result": "incomplete", "ticks": 52000}),
])
check("merge 0 si no hay win", empty == [] and meta0["bc_n_win_eps"] == 0
      and empty_eps == [])
inc_kept, meta_i, inc_eps = merge_teacher_wins(
    [(lose_s, {"result": "incomplete", "ticks": 52000}),
     (win_s, {"result": "lose", "ticks": 9000})],
    keep_incomplete=True,
)
check("bc-only guarda incomplete largo", len(inc_kept) == 5)
check("bc-only sigue tirando lose", meta_i["bc_n_win_eps"] == 0)
check("bc-only kept_eps incomplete", len(inc_eps) == 1)

# TeacherWinBuffer: add / sample / save / load roundtrip
import tempfile
import shutil

tw_dir = Path(tempfile.mkdtemp(prefix="twbuf_"))
try:
    tw = TeacherWinBuffer(cap_steps=50, prefer_ticks=40000, path=tw_dir)
    n0 = tw.add_episode([{"tag": ("L", i)} for i in range(30)],
                        {"result": "lose", "ticks": 9000})
    check("TW lose no entra", n0 == 0 and tw.n_episodes == 0)
    n1 = tw.add_episode([{"tag": ("Sa", i)} for i in range(20)],
                        {"result": "win", "ticks": 20000})
    n2 = tw.add_episode([{"tag": ("Sb", i)} for i in range(20)],
                        {"result": "win_early", "ticks": 22000})
    check("TW wins entran", n1 == 20 and n2 == 20 and tw.n_episodes == 2)
    got = tw.sample(10)
    check("TW sample size", len(got) == 10)
    tags = {s["tag"][0] for s in got}
    check("TW sample even-pick ambos wins", tags == {"Sa", "Sb"})
    tw.save()
    check("TW manifest existe", (tw_dir / "manifest.json").is_file())
    tw2 = TeacherWinBuffer(cap_steps=50, prefer_ticks=40000, path=tw_dir)
    check("TW load eps", tw2.n_episodes == 2)
    check("TW load steps", len(tw2) == 40)
    got2 = tw2.sample(8)
    check("TW sample tras load", len(got2) == 8)
    tw3 = TeacherWinBuffer(cap_steps=20, prefer_ticks=40000)
    tw3.add_episode([{"tag": ("L", i)} for i in range(20)],
                    {"result": "win", "ticks": 50000})
    tw3.add_episode([{"tag": ("S", i)} for i in range(20)],
                    {"result": "win", "ticks": 18000})
    check("TW trim echa largo",
          tw3.n_episodes == 1 and all(s["tag"][0] == "S" for s in tw3.snapshot()))
    tw20 = TeacherWinBuffer(cap_steps=25, prefer_ticks=20000)
    tw20.add_episode([{"tag": ("L", i)} for i in range(20)],
                     {"result": "win", "ticks": 33456})
    tw20.add_episode([{"tag": ("S", i)} for i in range(20)],
                     {"result": "win", "ticks": 12000})
    check("TW prefer 20k echa win 33k antes que rush 12k",
          tw20.n_episodes == 1 and all(s["tag"][0] == "S" for s in tw20.snapshot()))
    man = json.loads((tw_dir / "manifest.json").read_text(encoding="utf-8"))
    check("TW schema hunt_v2", man.get("schema") == TAPE_SCHEMA)
    check("TW schema string", TAPE_SCHEMA == "eco_and_combat_mental_v4")
    stale = Path(tempfile.mkdtemp(prefix="twstale_"))
    try:
        (stale / "manifest.json").write_text(
            json.dumps({"cap": 50, "episodes": []}), encoding="utf-8")
        tw_stale = TeacherWinBuffer(cap_steps=50, path=stale)
        check("TW schema viejo no hidrata", tw_stale.n_episodes == 0)
    finally:
        shutil.rmtree(stale, ignore_errors=True)
    stale_v1 = Path(tempfile.mkdtemp(prefix="twv1_"))
    try:
        (stale_v1 / "manifest.json").write_text(
            json.dumps({"schema": "eco_and_combat_v1", "cap": 50,
                        "episodes": []}), encoding="utf-8")
        tw_v1 = TeacherWinBuffer(cap_steps=50, path=stale_v1)
        check("TW schema beacon v1 no hidrata", tw_v1.n_episodes == 0)
    finally:
        shutil.rmtree(stale_v1, ignore_errors=True)
finally:
    shutil.rmtree(tw_dir, ignore_errors=True)

print("\n" + ("TODOS LOS TESTS OK" if ok else "HAY FALLAS"))
sys.exit(0 if ok else 1)
