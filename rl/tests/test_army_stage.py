# -*- coding: utf-8 -*-
"""Unit tests: army staging past choke + push hysteresis + east-push guards."""
import numpy as np
from types import SimpleNamespace as NS

from rl.action_adapter import (
    stage_army_attack_cell,
    should_emit_army_push,
    filter_army_push_hysteresis,
    _wider_flank_passable,
    remap_move_cell,
    guard_army_push_cell,
    nearest_passable,
)


class _U:
    def __init__(self, **kw):
        self.actor_id = kw.get("actor_id", 1)
        self.type = kw.get("type", "e1")
        self.can_attack = kw.get("can_attack", True)
        self.cell_x = kw.get("cell_x", 10)
        self.cell_y = kw.get("cell_y", 10)


class _Obs:
    def __init__(self, units, enemies=None, enemy_bldgs=None, map_name="Singles",
                 enemy_base_xy=None, belief=None):
        self.units = units
        self.visible_enemies = list(enemies or [])
        self.visible_enemy_buildings = list(enemy_bldgs or [])
        self.map_info = NS(map_name=map_name, height=40, width=128)
        self.enemy_base_xy = enemy_base_xy
        self.belief = belief
        self.buildings = []


class _Aidx:
    def __init__(self, h, w, grid):
        self.h = h
        self.w = w
        self.pass_grid = grid


def _lake_grid(h=40, w=80, lake_x0=30, lake_x1=45):
    """Passable land with a vertical lake band (central choke)."""
    g = np.ones((h, w), dtype=bool)
    g[:, lake_x0:lake_x1] = False
    # Leave N/S land corridors
    g[0:3, :] = True
    g[h - 3:h, :] = True
    return g


def test_stage_past_choke_returns_target():
    grid = _lake_grid()
    # Army already east of home choke (x>35)
    obs = _Obs([_U(cell_x=40, cell_y=20), _U(actor_id=2, cell_x=42, cell_y=18)])
    aidx = _Aidx(40, 80, grid)
    cx, cy = stage_army_attack_cell(obs, aidx, 70, 20)
    assert (cx, cy) == (70, 20)


def test_stage_flank_near_cen_returns_target():
    # Wide lake + thin N/S corridors; army already on the flank attractor with
    # cen.x <= 35 so the past-choke early-out does not mask this guard.
    h, w = 40, 80
    grid = np.ones((h, w), dtype=bool)
    grid[:, 15:50] = False
    grid[5, :] = True
    grid[30, :] = True
    aidx = _Aidx(h, w, grid)
    obs = _Obs([_U(cell_x=34, cell_y=5), _U(actor_id=2, cell_x=33, cell_y=5)])
    flank = _wider_flank_passable(int((34 + 40) / 2), int((5 + 35) / 2), grid, h, w)
    assert flank is not None
    assert (flank[0] - 34) ** 2 + (flank[1] - 5) ** 2 < 36
    cx, cy = stage_army_attack_cell(obs, aidx, 40, 35)
    assert (cx, cy) == (40, 35)



def test_stage_n_advanced_keeps_east_target():
    """>=8 units past choke keep east target even if home units drag centroid west."""
    grid = _lake_grid()
    units = []
    # Many home reinforcements (x~12) pull mean centroid below 35
    for i in range(12):
        units.append(_U(actor_id=100 + i, cell_x=12, cell_y=10 + (i % 5)))
    # Front group already past home choke
    for i in range(8):
        units.append(_U(actor_id=200 + i, cell_x=40 + (i % 3), cell_y=18 + (i % 4)))
    obs = _Obs(units)
    aidx = _Aidx(40, 80, grid)
    # Sanity: mean centroid is west of choke due to home mass
    xs = [u.cell_x for u in units]
    assert sum(xs) / len(xs) < 35
    cx, cy = stage_army_attack_cell(obs, aidx, 70, 20)
    assert (cx, cy) == (70, 20)


def test_stage_few_advanced_still_flanks():
    """Few/no advanced units + blocked midline still stages to flank."""
    # Wider lake so west-army to east-target midline is majority-blocked
    # (default _lake_grid band is too thin for this long eastbound segment).
    h, w = 40, 80
    grid = np.ones((h, w), dtype=bool)
    grid[:, 15:50] = False
    grid[0:3, :] = True
    grid[h - 3:h, :] = True
    obs = _Obs([
        _U(actor_id=1, cell_x=12, cell_y=20),
        _U(actor_id=2, cell_x=14, cell_y=18),
        _U(actor_id=3, cell_x=13, cell_y=22),
    ])
    aidx = _Aidx(h, w, grid)
    cx, cy = stage_army_attack_cell(obs, aidx, 70, 20)
    assert (cx, cy) != (70, 20)
    assert 0 <= cy < h


def test_should_emit_army_push():
    assert should_emit_army_push(None, 10, 10) is True
    assert should_emit_army_push((10, 10), 12, 12) is False  # dist2=8 < 64
    assert should_emit_army_push((10, 10), 18, 10) is True   # dist2=64
    assert should_emit_army_push((10, 10), 20, 10) is True
    assert should_emit_army_push((10, 10), 10, 10, eps=8) is False


def test_filter_army_push_hysteresis():
    from openra_env.models import ActionType, CommandModel
    c1 = CommandModel(action=ActionType.ARMY_ATTACK_MOVE, target_x=50, target_y=10)
    c2 = CommandModel(action=ActionType.ARMY_ATTACK_MOVE, target_x=52, target_y=11)
    kept, last = filter_army_push_hysteresis([c1], None)
    assert len(kept) == 1 and last == (50, 10)
    kept2, last2 = filter_army_push_hysteresis([c2], last)
    assert kept2[0].action == ActionType.NO_OP or len(kept2) == 1
    # near duplicate suppressed -> only no_op fallback if that was sole cmd
    assert last2 == (50, 10)
    assert all(
        getattr(getattr(c, "action", None), "value", None) != "army_attack_move"
        for c in kept2
    )
    # far target emits and updates
    c3 = CommandModel(action=ActionType.ARMY_ATTACK_MOVE, target_x=70, target_y=10)
    kept3, last3 = filter_army_push_hysteresis([c3], last2)
    assert last3 == (70, 10)
    assert kept3[0].action == ActionType.ARMY_ATTACK_MOVE


def test_remap_illegal_midmap_water_near_click_not_south_flank():
    """Illegal mid-map water -> nearest_passable near click, NOT south flank y~40."""
    h, w = 48, 100
    grid = np.ones((h, w), dtype=bool)
    # Central lake blob (illegal mid-map click)
    grid[18:28, 45:55] = False
    # South ore corridor land (what old _wider_flank_passable preferred ~y=36)
    # already passable via ones; flank_ys = (h//8=6, 3h//4=36)
    aidx = _Aidx(h, w, grid)
    obs = _Obs([_U(cell_x=20, cell_y=20)], map_name="Singles")
    cx, cy = remap_move_cell(obs, aidx, 50, 22)  # water cell inside lake
    assert bool(grid[cy, cx]), (cx, cy)
    # Must stay near the click, not snap to south flank y~36-40
    assert abs(cx - 50) <= 12 and abs(cy - 22) <= 12, (cx, cy)
    assert cy < 34, f"snapped to south flank y={cy}"
    # Sanity: nearest_passable agrees
    nx, ny = nearest_passable(50, 22, grid, h, w)
    assert (cx, cy) == (nx, ny)


def test_guard_west_target_with_advanced_front():
    """Ore click behind a vanguard must not stay on deep-home ore."""
    h, w = 40, 128
    grid = np.ones((h, w), dtype=bool)
    aidx = _Aidx(h, w, grid)
    units = [_U(actor_id=i, cell_x=78 + (i % 3), cell_y=10 + (i % 4)) for i in range(10)]
    obs = _Obs(units, enemies=[], enemy_bldgs=[_U(type="proc", cell_x=90, cell_y=12)],
               map_name="Singles")
    obs.buildings = [NS(type="fact", cell_x=12, cell_y=16)]
    gx, gy = guard_army_push_cell(obs, aidx, 42, 38)
    assert (gx, gy) != (95, 11), (gx, gy)
    assert (gx, gy) == (90, 12), (gx, gy)


def test_guard_fog_east_retargets_beacon():
    """No GPS: fog click behind a vanguard goes to war_objective, not (95,11)."""
    h, w = 40, 128
    grid = np.ones((h, w), dtype=bool)
    aidx = _Aidx(h, w, grid)
    units = [_U(actor_id=i, cell_x=80 + (i % 4), cell_y=12 + (i % 3)) for i in range(10)]
    obs = _Obs(units, enemies=[], enemy_bldgs=[_U(type="proc", cell_x=88, cell_y=14)],
               map_name="Singles")
    obs.buildings = [NS(type="fact", cell_x=12, cell_y=16)]
    gx, gy = guard_army_push_cell(obs, aidx, 40, 35)
    assert (gx, gy) != (95, 11), (gx, gy)
    assert (gx, gy) == (88, 14), (gx, gy)


if __name__ == "__main__":
    test_stage_past_choke_returns_target()
    test_stage_flank_near_cen_returns_target()
    test_stage_n_advanced_keeps_east_target()
    test_stage_few_advanced_still_flanks()
    test_should_emit_army_push()
    test_filter_army_push_hysteresis()
    test_remap_illegal_midmap_water_near_click_not_south_flank()
    test_guard_west_target_with_advanced_front()
    test_guard_fog_east_retargets_beacon()
    print("OK army stage/hysteresis/east-push tests")
