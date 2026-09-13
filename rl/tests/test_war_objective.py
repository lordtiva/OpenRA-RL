# -*- coding: utf-8 -*-
"""Spawn-agnostic war dest: no BEACON_BY_MAP / (95,11) GPS."""
from __future__ import annotations

from types import SimpleNamespace as NS

from rl.obs_encoding import resolve_beacon
from rl.scripted_teacher import ScriptedTeacher
from rl.war_objective import (
    FEET_PACK,
    guard_push_cell,
    own_anchor,
    reject_feet_push_cell,
    war_objective,
)


def _u(actor_id=1, typ="e1", x=12, y=16, idle=True):
    return NS(actor_id=actor_id, type=typ, cell_x=x, cell_y=y,
              is_idle=idle, hp_percent=1.0, can_attack=True,
              speed=50, attack_range=1000, experience_level=0, stance=0,
              facing=0)


def _b(typ="fact", actor_id=10, x=12, y=16):
    return NS(type=typ, actor_id=actor_id, cell_x=x, cell_y=y,
              hp_percent=1.0, is_repairing=False, is_powered=True,
              can_produce=())


def _obs(*, enemy_bldgs=(), enemies=(), map_name="Singles",
         bldgs=None, units=None, width=128, height=64, own_xy=(12, 16)):
    if bldgs is None:
        bldgs = [_b("fact", 100, own_xy[0], own_xy[1]),
                 _b("proc", 101, own_xy[0] + 2, own_xy[1])]
    if units is None:
        units = [_u(i, "e1", own_xy[0], own_xy[1]) for i in range(1, 14)]
    return NS(
        tick=100,
        map_info=NS(height=height, width=width, map_name=map_name),
        economy=NS(cash=5000, ore=0, harvester_count=1,
                   power_provided=100, power_drained=60, resource_capacity=5000),
        military=NS(kills_cost=0, deaths_cost=0, assets_value=2000,
                    units_killed=0, units_lost=0, army_value=1000),
        buildings=list(bldgs),
        units=list(units),
        production=[],
        available_production=["e1"],
        visible_enemies=list(enemies),
        visible_enemy_buildings=list(enemy_bldgs),
        belief=None,
    )


class _Aidx:
    def __init__(self, h=64, w=128):
        self.h = h
        self.w = w
        self.pass_grid = None


def test_visible_leftover_not_gps():
    obs = _obs(enemy_bldgs=[_b("proc", 200, 70, 22)])
    assert resolve_beacon(obs) == (95, 11)
    cell = war_objective(obs)
    assert cell == (70, 22)
    assert cell != (95, 11)


def test_opening_fog_not_beacon():
    obs = _obs()
    th = ScriptedTeacher()
    cell = th._push_cell(obs)
    assert cell != (95, 11)
    assert cell != (12, 16)
    assert war_objective(obs) != (95, 11)


def test_ne_spawn_visible_goes_to_enemy_not_gps():
    """Agent at NE (95,11), enemy building SW — dest is the building."""
    obs = _obs(
        own_xy=(95, 11),
        enemy_bldgs=[_b("proc", 200, 20, 40)],
        units=[_u(i, "e1", 95, 11) for i in range(1, 14)],
    )
    assert own_anchor(obs)[0] >= 90
    cell = war_objective(obs)
    assert cell == (20, 40)
    assert cell != (95, 11)


def test_mental_base_after_fog():
    th = ScriptedTeacher()
    th._push_cell(_obs(enemy_bldgs=[
        _b("proc", 210, 88, 18),
        _b("tent", 211, 90, 18),
        _b("powr", 212, 89, 20),
    ]))
    denser = th.belief.enemy_base_xy
    cell = th._push_cell(_obs())
    assert cell == denser
    assert cell != (95, 11)


def test_raid_beats_distant_building():
    obs = _obs(
        enemy_bldgs=[_b("proc", 200, 80, 20)],
        enemies=[_u(99, "e1", 14, 16)],
    )
    cell = war_objective(obs)
    assert cell == (14, 16)


def test_feet_reject_rewrites_local_am():
    leftover = _b("weap", 200, 80, 18)
    units = [_u(i, "e1", 12 + (i % 3), 16 + (i % 2), idle=True)
             for i in range(1, FEET_PACK + 2)]
    obs = _obs(enemy_bldgs=[leftover], units=units)
    aidx = _Aidx()
    cx, cy = reject_feet_push_cell(obs, aidx, 13, 17)
    assert (cx, cy) == (80, 18)
    assert (cx, cy) != (95, 11)


def test_feet_reject_keeps_far_click():
    leftover = _b("weap", 200, 80, 18)
    units = [_u(i, "e1", 12, 16, idle=True) for i in range(1, FEET_PACK + 2)]
    obs = _obs(enemy_bldgs=[leftover], units=units)
    aidx = _Aidx()
    cx, cy = reject_feet_push_cell(obs, aidx, 80, 18)
    assert (cx, cy) == (80, 18)


def test_guard_does_not_return_gps():
    units = [_u(i, "e1", 80 + (i % 4), 12 + (i % 3), idle=True)
             for i in range(10)]
    obs = _obs(units=units, own_xy=(12, 16), enemy_bldgs=[_b("proc", 200, 90, 14)])
    aidx = _Aidx()
    gx, gy = guard_push_cell(obs, aidx, 40, 35)
    assert (gx, gy) != (95, 11)
    assert (gx, gy) == (90, 14)


def test_teacher_opening_matches_war_objective():
    obs = _obs()
    th = ScriptedTeacher()
    assert th._push_cell(obs) == war_objective(obs, last_contact=None, belief=th.belief)
