# -*- coding: utf-8 -*-
"""Mental enemy-base belief (map-agnostic) — no GPS BEACON_BY_MAP."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.obs_encoding import (
    BASE_CLUSTER_RADIUS,
    EnemyBeliefStore,
    SCALAR_DIM,
    resolve_beacon,
    scalar_features,
)
from rl.scripted_teacher import ScriptedTeacher


def _u(actor_id=1, typ="e1", x=12, y=16, idle=True):
    return NS(actor_id=actor_id, type=typ, cell_x=x, cell_y=y,
              is_idle=idle, hp_percent=1.0, can_attack=True,
              speed=50, attack_range=1000, experience_level=0, stance=0,
              facing=0)


def _b(typ="fact", actor_id=10, x=12, y=16):
    return NS(type=typ, actor_id=actor_id, cell_x=x, cell_y=y,
              hp_percent=1.0, is_repairing=False, is_powered=True,
              can_produce=())


def _obs(*, enemy_bldgs=(), enemies=(), map_name="fase2_a_short.oramap",
         bldgs=("fact", "proc", "barr"), units=None):
    if units is None:
        units = [_u(i, "e1", 12, 16) for i in range(1, 10)]
    return NS(
        tick=100,
        map_info=NS(height=64, width=128, map_name=map_name),
        economy=NS(cash=5000, ore=0, harvester_count=1,
                   power_provided=100, power_drained=60, resource_capacity=5000),
        military=NS(kills_cost=0, deaths_cost=0, assets_value=2000,
                    units_killed=0, units_lost=0, army_value=1000),
        buildings=[_b(t, 100 + i) for i, t in enumerate(bldgs)],
        units=list(units),
        production=[],
        available_production=["e1"],
        visible_enemies=list(enemies),
        visible_enemy_buildings=list(enemy_bldgs),
    )


def test_first_building_sets_belief():
    bel = EnemyBeliefStore()
    bel.update(_obs(enemy_bldgs=[_b("powr", 201, 60, 20)]))
    assert bel.enemy_base_xy == (60, 20)
    assert bel.enemy_base_count == 1
    assert bel.enemy_base_strength == 1.0


def test_denser_cluster_updates():
    bel = EnemyBeliefStore()
    bel.update(_obs(enemy_bldgs=[_b("powr", 201, 60, 20)]))
    assert bel.enemy_base_xy == (60, 20)
    # Equal/sparser far sighting must NOT move the hypothesis.
    bel.update(_obs(enemy_bldgs=[_b("powr", 202, 90, 40)]))
    assert bel.enemy_base_xy == (60, 20)
    assert bel.enemy_base_count == 1
    # Denser cluster updates centroid + strength.
    bel.update(_obs(enemy_bldgs=[
        _b("proc", 210, 88, 18),
        _b("tent", 211, 90, 18),
        _b("powr", 212, 89, 20),
    ]))
    assert bel.enemy_base_count == 3
    assert bel.enemy_base_xy is not None
    assert 86 <= bel.enemy_base_xy[0] <= 92
    assert 16 <= bel.enemy_base_xy[1] <= 22


def test_push_prefers_denser_update_not_gps():
    th = ScriptedTeacher()
    # Seed with one building, then denser cluster.
    th._push_cell(_obs(enemy_bldgs=[_b("powr", 201, 55, 15)]))
    assert th.belief.enemy_base_xy == (55, 15)
    th._push_cell(_obs(enemy_bldgs=[
        _b("proc", 210, 88, 18),
        _b("tent", 211, 90, 18),
        _b("powr", 212, 89, 20),
    ]))
    assert th.belief.enemy_base_count == 3
    denser = th.belief.enemy_base_xy
    # Fog: leftovers gone → push uses denser mental base, never GPS beacon.
    cell = th._push_cell(_obs(enemy_bldgs=(), enemies=()))
    assert cell == denser
    assert cell != (95, 11)
    assert resolve_beacon(_obs()) == (95, 11)  # GPS exists but unused


def test_gps_beacon_not_used_for_belief_or_scalars():
    bel = EnemyBeliefStore()
    obs = _obs(enemy_bldgs=[], map_name="fase2_a_short.oramap")
    bel.update(obs)
    assert bel.enemy_base_xy is None
    assert resolve_beacon(obs) == (95, 11)
    sc = scalar_features(obs, belief=bel)
    assert SCALAR_DIM == 33
    assert sc.shape == (33,)
    assert float(sc[25]) == 0.0  # has_enemy_base_belief
    # After a real sighting, scalars light up from belief — still not GPS.
    bel.update(_obs(enemy_bldgs=[_b("proc", 201, 70, 22)]))
    sc2 = scalar_features(_obs(enemy_bldgs=[]), belief=bel)
    assert float(sc2[25]) == 1.0
    assert float(sc2[28]) > 0.0


def test_push_beacon_when_belief_empty():
    """Opening-SFT prior: beacon only when visible/ghost/mental absent."""
    th = ScriptedTeacher()
    obs = _obs(enemy_bldgs=(), enemies=())
    assert th.belief.enemy_base_xy is None
    assert resolve_beacon(obs) == (95, 11)
    cell = th._push_cell(obs)
    assert cell == (95, 11)


def test_push_fog_when_no_beacon_map():
    th = ScriptedTeacher()
    obs = _obs(enemy_bldgs=(), enemies=(), map_name="unknown_map.oramap")
    assert resolve_beacon(obs) is None
    cell = th._push_cell(obs)
    # Falls through to fog scout (may be None on tiny synthetic obs).
    assert cell != (95, 11)


def test_keep_until_better_evidence():
    bel = EnemyBeliefStore()
    bel.update(_obs(enemy_bldgs=[
        _b("proc", 210, 88, 18),
        _b("tent", 211, 90, 18),
    ]))
    xy, n = bel.enemy_base_xy, bel.enemy_base_count
    # Empty fog: keep.
    bel.update(_obs(enemy_bldgs=[]))
    assert bel.enemy_base_xy == xy and bel.enemy_base_count == n
    # Weaker single building: keep.
    bel.update(_obs(enemy_bldgs=[_b("powr", 202, 40, 40)]))
    assert bel.enemy_base_xy == xy and bel.enemy_base_count == n


if __name__ == "__main__":
    test_first_building_sets_belief()
    test_denser_cluster_updates()
    test_push_prefers_denser_update_not_gps()
    test_gps_beacon_not_used_for_belief_or_scalars()
    test_push_beacon_when_belief_empty()
    test_push_fog_when_no_beacon_map()
    test_keep_until_better_evidence()
    print("test_mental_base: OK")
    print(f"BASE_CLUSTER_RADIUS={BASE_CLUSTER_RADIUS} SCALAR_DIM={SCALAR_DIM}")
