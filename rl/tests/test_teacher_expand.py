# -*- coding: utf-8 -*-
"""Expand teacher: weap save/build as soon as legal; tanks after rush."""
from openra_env.models import ActionType
from rl.scripted_teacher import ScriptedTeacher


class _B:
    def __init__(self, typ="proc", x=5, y=5, aid=99):
        self.type = typ
        self.cell_x = x
        self.cell_y = y
        self.actor_id = aid
        self.pos_x = x * 1024
        self.pos_y = y * 1024


class _P:
    def __init__(self, queue_type, item, progress=1.0):
        self.queue_type = queue_type
        self.item = item
        self.progress = progress


class _U:
    def __init__(self, typ="harv", aid=1, x=8, y=8, is_idle=True):
        self.type = typ
        self.actor_id = aid
        self.cell_x = x
        self.cell_y = y
        self.hp_percent = 1.0
        self.can_attack = typ in ("e1", "e3", "1tnk", "2tnk")
        self.is_idle = is_idle
        self.speed = 50
        self.attack_range = 0
        self.experience_level = 0
        self.stance = 0
        self.facing = 0


class _Obs:
    def __init__(self, buildings=None, units=None, available=None, cash=5000):
        self.units = units or [_U()]
        self.buildings = buildings if buildings is not None else [
            _B("proc"), _B("powr", aid=98), _B("tent", aid=97)]
        self.visible_enemies = []
        self.visible_enemy_buildings = []
        self.production = []
        self.available_production = list(available or [])
        self.economy = type("E", (), {
            "cash": cash, "ore": 0, "resource_capacity": 2000,
            "power_provided": 100, "power_drained": 50,
            "harvester_count": sum(1 for u in self.units if "harv" in u.type),
        })()
        self.military = type("M", (), {
            "units_killed": 0, "units_lost": 0, "buildings_killed": 0,
            "buildings_lost": 0, "army_value": 0, "active_unit_count": 0,
            "kills_cost": 0, "deaths_cost": 0, "assets_value": 0,
        })()
        self.map_info = type("MI", (), {
            "height": 32, "width": 32, "map_name": "fase2_a_short.oramap",
        })()
        self.spatial_map = ""
        self.spatial_channels = 9
        self.tick = 1


class _T(ScriptedTeacher):
    def _can_produce_item(self, obs, item_type):
        return item_type in (obs.available_production or [])


def _rifles(n):
    return [_U("harv")] + [_U("e1", aid=10 + i) for i in range(n)]


def test_default_mode_is_rush():
    t = ScriptedTeacher()
    assert t.mode == "rush"
    assert "weap" not in t.BUILD_PRIORITY


def test_expand_weap_before_rush_when_legal():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    obs = _Obs(available=["weap", "pbox", "e1", "e3", "1tnk"],
               units=_rifles(2), cash=5000)
    out = t._handle_production(obs)
    assert any(c.action == ActionType.BUILD and c.item_type == "weap"
               for c in out)


def test_expand_trains_1tnk_before_rush():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    obs = _Obs(
        buildings=[_B("proc"), _B("powr", aid=2), _B("tent", aid=3),
                   _B("weap", aid=4)],
        available=["1tnk", "2tnk", "e1", "e3", "pbox"],
        units=_rifles(2), cash=5000)
    out = t._apply_expand(obs, [])
    tanks = [c for c in out if c.action == ActionType.TRAIN
             and c.item_type in ("1tnk", "2tnk")]
    assert len(tanks) == 1
    assert tanks[0].item_type == "1tnk"


def test_expand_pumps_garrison_before_weap_save():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    obs = _Obs(available=["weap", "e1", "e3"], units=_rifles(3), cash=1500)
    from openra_env.models import CommandModel
    cmds = [CommandModel(action=ActionType.TRAIN, item_type="e1")]
    out = t._apply_expand(obs, cmds)
    assert not any(c.action == ActionType.BUILD and c.item_type == "weap"
                   for c in out)
    assert any(c.action == ActionType.TRAIN and c.item_type == "e1"
               for c in out)


def test_expand_saves_e1_once_garrison_up():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    obs = _Obs(available=["weap", "e1", "e3"], units=_rifles(6), cash=1500)
    from openra_env.models import CommandModel
    cmds = [CommandModel(action=ActionType.TRAIN, item_type="e1")]
    out = t._apply_expand(obs, cmds)
    assert not any(c.action == ActionType.BUILD and c.item_type == "weap"
                   for c in out)
    assert not any(c.action == ActionType.TRAIN and c.item_type == "e1"
                   for c in out)


def test_expand_keeps_harv_during_weap_save():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    obs = _Obs(available=["weap", "e1", "harv"], units=_rifles(6), cash=1500)
    from openra_env.models import CommandModel
    cmds = [CommandModel(action=ActionType.TRAIN, item_type="harv")]
    out = t._apply_expand(obs, cmds)
    assert any(c.action == ActionType.TRAIN and c.item_type == "harv"
               for c in out)


def test_expand_keeps_e1_below_save():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    obs = _Obs(available=["weap", "e1"], units=_rifles(2), cash=800)
    from openra_env.models import CommandModel
    cmds = [CommandModel(action=ActionType.TRAIN, item_type="e1")]
    out = t._apply_expand(obs, cmds)
    assert any(c.action == ActionType.TRAIN and c.item_type == "e1"
               for c in out)


def test_expand_builds_weap_after_rush():
    t = _T(verbose=False, mode="expand")
    t.phase = "attack"
    obs = _Obs(available=["weap", "pbox", "e1", "e3", "1tnk"],
               units=_rifles(8), cash=5000)
    out = t._handle_production(obs)
    weap = [c for c in out if c.action == ActionType.BUILD and c.item_type == "weap"]
    assert len(weap) == 1


def test_expand_weap_before_pbox():
    t = _T(verbose=False, mode="expand")
    t.phase = "attack"
    obs = _Obs(available=["weap", "pbox", "dome", "e1"],
               units=_rifles(8), cash=5000)
    out = t._apply_expand(obs, [])
    assert len(out) == 1
    assert out[0].item_type == "weap"


def test_expand_second_proc_before_weap():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    obs = _Obs(available=["proc", "weap", "powr", "pbox", "e1"],
               units=_rifles(2), cash=5000)
    out = t._handle_production(obs)
    assert any(c.action == ActionType.BUILD and c.item_type == "proc"
               for c in out)
    assert not any(c.action == ActionType.BUILD and c.item_type == "weap"
                   for c in out)


def test_expand_second_proc_after_weap():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    obs = _Obs(
        buildings=[_B("proc"), _B("powr", aid=2), _B("tent", aid=3),
                   _B("weap", aid=4)],
        available=["proc", "pbox", "1tnk"],
        units=_rifles(6), cash=5000)
    out = t._apply_expand(obs, [])
    assert any(c.action == ActionType.BUILD and c.item_type == "proc"
               for c in out)


def test_expand_builds_third_proc_after_tanks_mass():
    """After ~4 tanks, spend on 3rd proc (mid if reachable, else home)."""
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    units = (_rifles(6)
             + [_U("harv", aid=2), _U("harv", aid=3), _U("harv", aid=4)]
             + [_U("1tnk", aid=80 + i) for i in range(4)])
    obs = _Obs(
        buildings=[_B("fact", x=12, y=16, aid=1),
                   _B("proc", x=15, y=16, aid=5),
                   _B("proc", x=10, y=19, aid=6),
                   _B("powr", x=14, y=18, aid=2),
                   _B("tent", x=16, y=18, aid=3),
                   _B("weap", x=18, y=16, aid=4)],
        available=["proc", "pbox", "1tnk", "powr", "harv"],
        units=units, cash=5000)
    obs.map_info = type("MI", (), {
        "height": 50, "width": 108, "map_name": "fase2_a_short.oramap",
    })()
    out = t._apply_expand(obs, [])
    assert any(c.action == ActionType.BUILD and c.item_type == "proc"
               for c in out)
    assert any(c.action == ActionType.TRAIN
               and c.item_type in ("1tnk", "2tnk") for c in out)


def test_expand_mcv_moves_then_deploys_at_mid():
    t = _T(verbose=False, mode="expand")
    obs = _Obs(
        buildings=[_B("fact", x=12, y=16, aid=1),
                   _B("proc", x=15, y=16, aid=5),
                   _B("weap", x=18, y=16, aid=4)],
        units=[_U("mcv", aid=50, x=20, y=20, is_idle=True)],
        cash=100)
    obs.map_info = type("MI", (), {
        "height": 50, "width": 108, "map_name": "fase2_a_short.oramap",
    })()
    obs.tick = 100
    out = t._handle_mcv_expand(obs)
    assert any(c.action == ActionType.MOVE and c.actor_id == 50 for c in out)
    mv = next(c for c in out if c.action == ActionType.MOVE and c.actor_id == 50)
    # Approach is clear fringe toward home, still near mid.
    assert abs(mv.target_x - 56) <= 10 and abs(mv.target_y - 35) <= 10
    obs2 = _Obs(
        buildings=[_B("fact", x=12, y=16, aid=1)],
        units=[_U("mcv", aid=50, x=50, y=34, is_idle=True)],
        cash=100)
    obs2.map_info = obs.map_info
    obs2.tick = 200
    out2 = t._handle_mcv_expand(obs2)
    assert any(c.action == ActionType.DEPLOY and c.actor_id == 50
               for c in out2)


def test_expand_builds_mid_proc_when_in_reach():
    """Once Adjacent reaches mid fringe, spend on mid proc (not home-3rd)."""
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    units = (_rifles(6)
             + [_U("harv", aid=2), _U("harv", aid=3), _U("harv", aid=4)]
             + [_U("1tnk", aid=80 + i) for i in range(4)])
    # Forward powr near mid makes mid fringe placeable.
    obs = _Obs(
        buildings=[_B("fact", x=12, y=16, aid=1),
                   _B("proc", x=15, y=16, aid=5),
                   _B("proc", x=28, y=30, aid=6),
                   _B("powr", x=48, y=34, aid=2),
                   _B("tent", x=16, y=18, aid=3),
                   _B("weap", x=18, y=16, aid=4)],
        available=["proc", "pbox", "1tnk", "powr", "harv"],
        units=units, cash=5000)
    obs.map_info = type("MI", (), {
        "height": 50, "width": 108, "map_name": "fase2_a_short.oramap",
    })()
    out = t._apply_expand(obs, [])
    assert any(c.action == ActionType.BUILD and c.item_type == "proc"
               for c in out)
    assert any(c.action == ActionType.TRAIN
               and c.item_type in ("1tnk", "2tnk") for c in out)


def test_expand_third_proc_waits_for_tanks():
    """v1 0/8: mid contest right after weap starved the 12-tank hold."""
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    units = (_rifles(6)
             + [_U("harv", aid=2), _U("harv", aid=3), _U("harv", aid=4)]
             + [_U("1tnk", aid=80), _U("1tnk", aid=81)])
    obs = _Obs(
        buildings=[_B("proc"), _B("proc", aid=6), _B("powr", aid=2),
                   _B("tent", aid=3), _B("weap", aid=4)],
        available=["proc", "pbox", "1tnk", "weap", "powr"],
        units=units, cash=5000)
    out = t._apply_expand(obs, [])
    assert not any(c.action == ActionType.BUILD and c.item_type == "proc"
                   for c in out)


def test_expand_does_not_mill_fourth_proc():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    units = (_rifles(6)
             + [_U("harv", aid=2), _U("harv", aid=3), _U("harv", aid=4)]
             + [_U("1tnk", aid=80 + i) for i in range(4)])
    # 2 home + mid already claimed.
    obs = _Obs(
        buildings=[_B("proc", x=15, y=16, aid=5),
                   _B("proc", x=28, y=30, aid=6),
                   _B("proc", x=52, y=34, aid=9),
                   _B("powr", aid=2), _B("tent", aid=3), _B("weap", aid=4)],
        available=["proc", "pbox", "1tnk", "weap", "harv", "powr"],
        units=units, cash=5000)
    obs.map_info = type("MI", (), {
        "height": 50, "width": 108, "map_name": "fase2_a_short.oramap",
    })()
    out = t._apply_expand(obs, [])
    assert not any(c.action == ActionType.BUILD and c.item_type == "proc"
                   for c in out)


def test_expand_counts_queued_proc():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    obs = _Obs(
        buildings=[_B("proc"), _B("powr", aid=2), _B("tent", aid=3)],
        available=["proc", "weap", "pbox"],
        units=_rifles(6), cash=5000)
    obs.production = [_P("Building", "proc", progress=0.5)]
    out = t._apply_expand(obs, [])
    assert not any(c.action == ActionType.BUILD and c.item_type == "proc"
                   for c in out)


def test_expand_powr_before_weap_when_two_proc():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    obs = _Obs(
        buildings=[_B("proc"), _B("proc", aid=6), _B("powr", aid=2),
                   _B("tent", aid=3)],
        available=["weap", "powr", "e1"],
        units=_rifles(6), cash=5000)
    obs.economy.power_provided = 100
    obs.economy.power_drained = 80
    out = t._apply_expand(obs, [])
    assert any(c.action == ActionType.BUILD and c.item_type == "powr"
               for c in out)
    assert not any(c.action == ActionType.BUILD and c.item_type == "weap"
                   for c in out)


def test_expand_tanks_from_ore_not_just_cash():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    units = _rifles(6) + [_U("harv", aid=2)]
    obs = _Obs(
        buildings=[_B("proc"), _B("proc", aid=6), _B("powr", aid=2),
                   _B("tent", aid=3), _B("weap", aid=4)],
        available=["1tnk", "e1", "pbox"],
        units=units, cash=0)
    obs.economy.ore = 2000
    out = t._apply_expand(obs, [])
    tanks = [c for c in out if c.action == ActionType.TRAIN
             and c.item_type in ("1tnk", "2tnk")]
    assert len(tanks) == 1


def test_expand_tanks_despite_parent_e1():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    units = _rifles(6) + [_U("harv", aid=2)]
    obs = _Obs(
        buildings=[_B("proc"), _B("proc", aid=6), _B("powr", aid=2),
                   _B("tent", aid=3), _B("weap", aid=4)],
        available=["1tnk", "e1"],
        units=units, cash=5000)
    from openra_env.models import CommandModel
    cmds = [CommandModel(action=ActionType.TRAIN, item_type="e1")]
    out = t._apply_expand(obs, cmds)
    assert any(c.action == ActionType.TRAIN and c.item_type == "1tnk"
               for c in out)


def test_expand_trains_harv_before_tanks():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    units = _rifles(6) + [_U("harv", aid=2)]
    obs = _Obs(
        buildings=[_B("proc"), _B("proc", aid=6), _B("powr", aid=2),
                   _B("tent", aid=3), _B("weap", aid=4)],
        available=["harv", "1tnk", "e1"],
        units=units, cash=5000)
    out = t._apply_expand(obs, [])
    assert any(c.action == ActionType.TRAIN and c.item_type == "harv"
               for c in out)
    assert not any(c.action == ActionType.TRAIN and c.item_type == "1tnk"
                   for c in out)


def test_expand_harvesters_flee_raid():
    t = _T(verbose=False, mode="expand")
    harv = _U("harv", aid=11, x=6, y=6)
    obs = _Obs(
        buildings=[_B("proc", x=5, y=5, aid=1), _B("powr", aid=2),
                   _B("tent", aid=3)],
        units=[harv], cash=500)
    obs.visible_enemies = [_U("e1", aid=99, x=8, y=6)]
    out = t._handle_harvesters(obs)
    assert any(c.action == ActionType.MOVE and c.actor_id == 11 for c in out)


def _ore_blob(arr, x0, y0, n=4, dens=8.0):
    for i in range(n):
        arr[2, y0, x0 + i] = dens
        arr[2, y0 + 1, x0 + i] = dens


def test_expand_proc_place_prefers_ore():
    t = _T(verbose=False, mode="expand")
    cy = _B("fact", x=12, y=16, aid=1)
    obs = _Obs(buildings=[cy], units=_rifles(2), cash=500)
    obs.map_info = type("MI", (), {
        "height": 50, "width": 108, "map_name": "fase2_a_short.oramap",
    })()
    x, y = t._ore_place_cell(obs, cy)
    # Adjacent=8: must stay in the CY halo (else C# dumps on the yard ring).
    assert max(abs(x - 12), abs(y - 16)) <= t.EXPAND_PROC_BASE_REACH
    # West spawn's nearest mine is (23, 5) — far NE, not CY+3 (15,16).
    assert x > 12
    assert y < 16
    assert max(abs(x - 12), abs(y - 16)) >= 6
    assert max(abs(x - 23), abs(y - 5)) < max(abs(12 - 23), abs(16 - 5))


def test_expand_second_proc_other_ore_zone():
    t = _T(verbose=False, mode="expand")
    cy = _B("fact", x=12, y=16, aid=1)
    # First proc already claimed the NE mine (23,5) from the west spawn.
    first = _B("proc", x=18, y=10, aid=2)
    obs = _Obs(buildings=[cy, first], units=_rifles(2), cash=500)
    obs.map_info = type("MI", (), {
        "height": 50, "width": 108, "map_name": "fase2_a_short.oramap",
    })()
    x, y = t._ore_place_cell(obs, cy)
    assert max(abs(x - 12), abs(y - 16)) <= t.EXPAND_PROC_BASE_REACH or \
        max(abs(x - 18), abs(y - 10)) <= t.EXPAND_PROC_BASE_REACH
    assert max(abs(x - 18), abs(y - 10)) >= t.EXPAND_PROC_ZONE_SEP
    # 2nd proc prefers mid-facing home mine (32,28) over far SW (4,30).
    assert not (x >= 16 and y <= 12)
    assert max(abs(x - 32), abs(y - 28)) <= max(abs(x - 4), abs(y - 30))


def test_expand_proc_ignores_fog_crumbs_near_cy():
    """Visible ore next to the CY is the same mine as (23,5), not a new zone."""
    import numpy as np
    t = _T(verbose=False, mode="expand")
    cy = _B("fact", x=12, y=16, aid=1)
    obs = _Obs(buildings=[cy], units=_rifles(2), cash=500)
    obs.map_info = type("MI", (), {
        "height": 50, "width": 108, "map_name": "fase2_a_short.oramap",
    })()
    arr = np.zeros((9, 50, 108), dtype=np.float32)
    arr[3] = 1.0
    arr[4] = 1.0
    # Fog-revealed crumbs hugging the yard (what live spatial looks like
    # at first PLACE). Must not steal the (23,5) identity.
    for x0 in range(14, 18):
        for y0 in range(14, 18):
            arr[2, y0, x0] = 8.0
    obs._spatial_chw = arr
    x, y = t._ore_place_cell(obs, cy)
    assert x > 12 and y < 16
    assert max(abs(x - 12), abs(y - 16)) >= 6
    assert (x, y) != (15, 16)


def test_expand_idle_harv_harvests_home_mine():
    t = _T(verbose=False, mode="expand")
    harv = _U("harv", aid=11, x=14, y=16, is_idle=True)
    obs = _Obs(
        buildings=[_B("fact", x=12, y=16, aid=1),
                   _B("proc", x=15, y=16, aid=2)],
        units=[harv], cash=500)
    obs.map_info = type("MI", (), {
        "height": 50, "width": 108, "map_name": "fase2_a_short.oramap",
    })()
    out = t._handle_harvesters(obs)
    hits = [c for c in out if c.action == ActionType.HARVEST
            and c.actor_id == 11]
    assert len(hits) == 1
    # Nearest west mine is (23,5), not crumbs at the CY-stacked proc.
    assert (hits[0].target_x, hits[0].target_y) == (23, 5)


def test_expand_busy_harv_not_reordered():
    t = _T(verbose=False, mode="expand")
    harv = _U("harv", aid=11, x=18, y=10, is_idle=False)
    obs = _Obs(
        buildings=[_B("fact", x=12, y=16, aid=1),
                   _B("proc", x=15, y=16, aid=2)],
        units=[harv], cash=500)
    out = t._handle_harvesters(obs)
    assert out == []


def test_expand_east_proc_toward_se_mine():
    t = _T(verbose=False, mode="expand")
    cy = _B("fact", x=95, y=11, aid=1)
    obs = _Obs(buildings=[cy], units=_rifles(2), cash=500)
    obs.map_info = type("MI", (), {
        "height": 50, "width": 108, "map_name": "fase2_a_short.oramap",
    })()
    x, y = t._ore_place_cell(obs, cy)
    assert max(abs(x - 95), abs(y - 11)) <= t.EXPAND_PROC_BASE_REACH
    # East nearest mine is (104, 25) — SE, not CY+3 (98,11).
    assert x > 95
    assert y > 11
    assert max(abs(x - 95), abs(y - 11)) >= 6
    assert (x, y) != (98, 11)


def test_expand_proc_sits_on_clear_fringe_not_on_ore():
    """RA cannot PLACE on Ore. Easy's 100,24 is the clear fringe of (104,25)."""
    import numpy as np
    t = _T(verbose=False, mode="expand")
    cy = _B("fact", x=12, y=16, aid=1)
    obs = _Obs(buildings=[cy], units=_rifles(2), cash=500)
    obs.map_info = type("MI", (), {
        "height": 50, "width": 108, "map_name": "fase2_a_short.oramap",
    })()
    arr = np.zeros((9, 50, 108), dtype=np.float32)
    arr[3] = 1.0
    arr[4] = 1.0
    for yy in range(50):
        for xx in range(108):
            if max(abs(xx - 23), abs(yy - 5)) <= 4:
                arr[2, yy, xx] = 8.0
    obs._spatial_chw = arr
    x, y = t._ore_place_cell(obs, cy)
    assert x > 12 and y < 16
    assert (x, y) != (15, 16)
    for fx, fy in t._fp_cells("proc", x, y):
        if 0 <= fy < 50 and 0 <= fx < 108:
            assert arr[2, fy, fx] <= 0.0
    # Closer to the mine than the CY is, but not sitting on it.
    assert max(abs(x - 23), abs(y - 5)) < max(abs(12 - 23), abs(16 - 5))
    assert max(abs(x - 23), abs(y - 5)) >= t.EXPAND_MINE_NO_BUILD


def test_expand_extra_harv_without_tanks():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    obs = _Obs(
        buildings=[_B("proc"), _B("proc", aid=6), _B("powr", aid=2),
                   _B("tent", aid=3), _B("weap", aid=4), _B("pbox", aid=5)],
        available=["harv", "1tnk", "e1"],
        units=_rifles(6), cash=5000)
    out = t._apply_expand(obs, [])
    assert any(c.action == ActionType.TRAIN and c.item_type == "harv"
               for c in out)


def test_expand_extra_harv_after_push_tanks():
    t = _T(verbose=False, mode="expand")
    t.phase = "attack"
    units = _rifles(6) + [
        _U("1tnk", aid=80), _U("1tnk", aid=81),
        _U("1tnk", aid=82), _U("1tnk", aid=83)]
    obs = _Obs(
        buildings=[_B("proc"), _B("powr", aid=2), _B("tent", aid=3),
                   _B("weap", aid=4), _B("pbox", aid=5)],
        available=["harv", "1tnk", "e1"],
        units=units, cash=5000)
    out = t._apply_expand(obs, [])
    assert any(c.action == ActionType.TRAIN and c.item_type == "harv"
               for c in out)


def test_expand_trains_1tnk_once_weap_exists():
    t = _T(verbose=False, mode="expand")
    t.phase = "attack"
    obs = _Obs(
        buildings=[_B("proc"), _B("powr", aid=2), _B("tent", aid=3),
                   _B("weap", aid=4)],
        available=["1tnk", "2tnk", "e1", "e3", "pbox"],
        units=_rifles(8), cash=5000)
    out = t._apply_expand(obs, [])
    tanks = [c for c in out if c.action == ActionType.TRAIN
             and c.item_type in ("1tnk", "2tnk")]
    assert len(tanks) == 1
    assert tanks[0].item_type == "1tnk"


def test_expand_mixes_e3_without_weap_if_broke_for_weap():
    t = _T(verbose=False, mode="expand")
    t.phase = "attack"
    # cash < weap: drop e1, do not BUILD weap, may TRAIN e3/e1 if leftover.
    obs = _Obs(available=["weap", "e1", "e3"], units=_rifles(10), cash=1500)
    from openra_env.models import CommandModel
    cmds = [CommandModel(action=ActionType.TRAIN, item_type="e1")]
    out = t._apply_expand(obs, cmds)
    assert not any(c.action == ActionType.BUILD and c.item_type == "weap"
                   for c in out)
    assert not any(c.action == ActionType.TRAIN and c.item_type == "e1"
                   for c in out)


def test_rush_mode_does_not_force_weap():
    t = _T(verbose=False, mode="rush")
    t.phase = "attack"
    obs = _Obs(available=["weap", "pbox", "e1"], units=_rifles(8), cash=5000)
    out = t._optional_tech_defense(obs, [])
    assert out and out[0].item_type in ("pbox", "hbox", "ftur")


def test_expand_holds_blob_without_tanks():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    t._push_cell = lambda obs: (90, 10)
    obs = _Obs(available=["weap", "e1"], units=_rifles(8), cash=500)
    obs.visible_enemy_buildings = [_B("fact", x=90, y=10, aid=50)]
    out = t._handle_combat(obs)
    assert not any(c.action == ActionType.ARMY_ATTACK_MOVE for c in out)
    # No tanks yet => rifle-harass gate closed; only fog scout.
    n_am = sum(1 for c in out if c.action == ActionType.ATTACK_MOVE)
    assert n_am <= t.EXPAND_HOLD_SCOUTS



def test_expand_pushes_once_tanks_ready():
    t = _T(verbose=False, mode="expand")
    t.phase = "attack"
    t._push_cell = lambda obs: (90, 10)
    units = _rifles(8) + [
        _U("1tnk", aid=80 + i) for i in range(t.EXPAND_PUSH_TANKS)]
    obs = _Obs(
        buildings=[_B("proc"), _B("powr", aid=2), _B("tent", aid=3),
                   _B("pbox", aid=4)],
        available=["1tnk", "e1"], units=units, cash=500)
    obs.visible_enemy_buildings = [_B("fact", x=90, y=10, aid=50)]
    out = t._handle_combat(obs)
    assert any(
        c.action in (ActionType.ARMY_ATTACK_MOVE, ActionType.ATTACK_MOVE)
        for c in out)
    n_am = sum(1 for c in out
               if c.action in (ActionType.ARMY_ATTACK_MOVE,
                               ActionType.ATTACK_MOVE))
    assert n_am >= t.RUSH_ATTACK_MOVE or any(
        c.action == ActionType.ARMY_ATTACK_MOVE for c in out)


def test_expand_holds_eight_tanks():
    """Bench 0/8: 8-tank push dribbled into easy's 20-40 building base."""
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    t._push_cell = lambda obs: (90, 10)
    units = _rifles(8) + [
        _U("1tnk", aid=80 + i) for i in range(8)]
    obs = _Obs(
        buildings=[_B("proc"), _B("powr", aid=2), _B("tent", aid=3),
                   _B("pbox", aid=4)],
        available=["1tnk", "e1"], units=units, cash=500)
    obs.visible_enemy_buildings = [_B("fact", x=90, y=10, aid=50)]
    out = t._handle_combat(obs)
    assert not any(c.action == ActionType.ARMY_ATTACK_MOVE for c in out)
    n_am = sum(1 for c in out if c.action == ActionType.ATTACK_MOVE)
    assert n_am <= t.EXPAND_HOLD_SCOUTS



def test_expand_push_sends_tanks_not_rifles():
    t = _T(verbose=False, mode="expand")
    t.phase = "attack"
    t._push_cell = lambda obs: (90, 10)
    units = _rifles(8) + [
        _U("1tnk", aid=80 + i) for i in range(t.EXPAND_PUSH_TANKS)]
    obs = _Obs(
        buildings=[_B("fact", x=12, y=16, aid=1), _B("weap", x=14, y=16, aid=2),
                   _B("pbox", x=15, y=16, aid=3)],
        available=["1tnk", "e1"], units=units, cash=500)
    obs.visible_enemy_buildings = [_B("fact", x=90, y=10, aid=50)]
    out = t._handle_combat(obs)
    e1_ids = set(range(10, 18))
    assert not any(c.actor_id in e1_ids for c in out)
    assert not any(c.action == ActionType.ARMY_ATTACK_MOVE for c in out)
    n_tnk = sum(1 for c in out if c.action == ActionType.ATTACK_MOVE
                and c.actor_id is not None and c.actor_id >= 80)
    assert n_tnk >= 8


def test_expand_retreats_understrength_tanks():
    t = _T(verbose=False, mode="expand")
    t.phase = "attack"
    t._expand_committed = True
    t._push_cell = lambda obs: (90, 10)
    tanks = [_U("1tnk", aid=80 + i, x=90, y=10) for i in range(5)]
    obs = _Obs(
        buildings=[_B("fact", x=12, y=16, aid=1),
                   _B("weap", x=14, y=16, aid=2)],
        units=_rifles(2) + tanks, cash=500)
    obs.visible_enemy_buildings = [_B("fact", x=90, y=10, aid=50)]
    out = t._handle_combat(obs)
    assert not any(c.action == ActionType.ARMY_ATTACK_MOVE for c in out)
    homes = [c for c in out
             if c.action in (ActionType.ATTACK_MOVE, ActionType.MOVE)
             and c.actor_id is not None and c.actor_id >= 80]
    assert homes
    assert all((c.target_x, c.target_y) == (12, 16) for c in homes)


def test_expand_yard_raid_peels_to_threat_not_enemy_base():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    t._push_cell = lambda obs: (90, 10)
    tank = _U("1tnk", aid=80, x=12, y=16)
    obs = _Obs(
        buildings=[_B("fact", x=12, y=16, aid=1), _B("weap", x=14, y=16, aid=2),
                   _B("pbox", x=15, y=16, aid=3)],
        units=_rifles(2) + [tank], cash=500)
    obs.visible_enemies = [_U("e1", aid=99, x=16, y=16)]
    obs.visible_enemy_buildings = [_B("fact", x=90, y=10, aid=50)]
    out = t._handle_combat(obs)
    hits = [c for c in out if c.action == ActionType.ATTACK_MOVE
            and c.actor_id == 80]
    assert hits
    assert (hits[0].target_x, hits[0].target_y) == (16, 16)
    assert not any(c.action == ActionType.ARMY_ATTACK_MOVE for c in out)


def test_expand_hold_ignores_scout_on_forward_proc():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    tank = _U("1tnk", aid=80, x=12, y=16)
    obs = _Obs(
        buildings=[_B("fact", x=12, y=16, aid=1),
                   _B("proc", x=19, y=10, aid=2)],
        units=_rifles(2) + [tank], cash=500)
    # Scout on the NE mine, 18 from CY — old DEFEND_CELLS peel yanked the hold.
    obs.visible_enemies = [_U("e1", aid=99, x=32, y=8)]
    out = t._handle_combat(obs)
    assert not any(c.actor_id == 80 for c in out)


def test_expand_harv_flees_to_fact_not_raided_proc():
    t = _T(verbose=False, mode="expand")
    harv = _U("harv", aid=11, x=20, y=8)
    obs = _Obs(
        buildings=[_B("fact", x=12, y=16, aid=1),
                   _B("proc", x=19, y=10, aid=2)],
        units=[harv], cash=500)
    obs.visible_enemies = [_U("e1", aid=99, x=21, y=8)]
    out = t._handle_harvesters(obs)
    hits = [c for c in out if c.action == ActionType.MOVE and c.actor_id == 11]
    assert hits
    assert (hits[0].target_x, hits[0].target_y) == (12, 16)


def test_expand_safe_harv_does_not_flee_with_raided_mate():
    t = _T(verbose=False, mode="expand")
    raided = _U("harv", aid=11, x=20, y=8)
    safe = _U("harv", aid=12, x=9, y=24)
    obs = _Obs(
        buildings=[_B("fact", x=12, y=16, aid=1),
                   _B("proc", x=19, y=10, aid=2),
                   _B("proc", x=9, y=23, aid=3)],
        units=[raided, safe], cash=500)
    obs.visible_enemies = [_U("e1", aid=99, x=21, y=8)]
    out = t._handle_harvesters(obs)
    moved = {c.actor_id for c in out if c.action == ActionType.MOVE}
    assert 11 in moved
    assert 12 not in moved


def test_expand_holds_four_tanks():
    """Bench 0/8: 4-tank all-in died at 17-23k into easy SquadSize 8."""
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    t._push_cell = lambda obs: (90, 10)
    units = _rifles(8) + [
        _U("1tnk", aid=80), _U("1tnk", aid=81),
        _U("1tnk", aid=82), _U("1tnk", aid=83)]
    obs = _Obs(
        buildings=[_B("proc"), _B("powr", aid=2), _B("tent", aid=3),
                   _B("pbox", aid=4)],
        available=["1tnk", "e1"], units=units, cash=500)
    obs.visible_enemy_buildings = [_B("fact", x=90, y=10, aid=50)]
    out = t._handle_combat(obs)
    assert not any(c.action == ActionType.ARMY_ATTACK_MOVE for c in out)
    n_am = sum(1 for c in out if c.action == ActionType.ATTACK_MOVE)
    assert n_am <= t.EXPAND_HOLD_SCOUTS


def test_expand_holds_two_tanks_with_pbox():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    t._push_cell = lambda obs: (90, 10)
    units = _rifles(8) + [_U("1tnk", aid=80), _U("1tnk", aid=81)]
    obs = _Obs(
        buildings=[_B("proc"), _B("powr", aid=2), _B("tent", aid=3),
                   _B("pbox", aid=4)],
        available=["1tnk", "e1"], units=units, cash=500)
    obs.visible_enemy_buildings = [_B("fact", x=90, y=10, aid=50)]
    out = t._handle_combat(obs)
    assert not any(c.action == ActionType.ARMY_ATTACK_MOVE for c in out)
    n_am = sum(1 for c in out if c.action == ActionType.ATTACK_MOVE)
    assert n_am <= t.EXPAND_HOLD_SCOUTS


def test_expand_hold_scout_skips_tanks():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    units = [_U("harv"), _U("e1", aid=10), _U("1tnk", aid=80)]
    obs = _Obs(available=["1tnk", "e1"], units=units, cash=500)
    out = t._handle_combat(obs)
    moved = [c.actor_id for c in out if c.action == ActionType.ATTACK_MOVE]
    assert 80 not in moved
    assert moved == [10] or not moved


def test_expand_holds_two_tanks_without_pbox():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    t._push_cell = lambda obs: (90, 10)
    units = _rifles(8) + [_U("1tnk", aid=80), _U("1tnk", aid=81)]
    obs = _Obs(available=["1tnk", "e1"], units=units, cash=500)
    obs.visible_enemy_buildings = [_B("fact", x=90, y=10, aid=50)]
    out = t._handle_combat(obs)
    assert not any(c.action == ActionType.ARMY_ATTACK_MOVE for c in out)
    n_am = sum(1 for c in out if c.action == ActionType.ATTACK_MOVE)
    assert n_am <= t.EXPAND_HOLD_SCOUTS


def test_expand_second_pbox():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    obs = _Obs(
        buildings=[_B("proc"), _B("proc", aid=6), _B("powr", aid=2),
                   _B("tent", aid=3), _B("pbox", aid=5)],
        available=["pbox", "1tnk", "e1"],
        units=_rifles(6), cash=5000)
    out = t._apply_expand(obs, [])
    assert any(c.action == ActionType.BUILD and c.item_type == "pbox"
               for c in out)


def test_expand_second_weap_after_four_tanks():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    units = (_rifles(6)
             + [_U("harv", aid=2), _U("harv", aid=3), _U("harv", aid=4)]
             + [_U("1tnk", aid=80 + i) for i in range(4)])
    obs = _Obs(
        buildings=[_B("proc", x=52, y=34, aid=9), _B("proc"), _B("proc", aid=6), _B("proc", x=52, y=34, aid=9),
                   _B("powr", aid=2), _B("tent", aid=3), _B("weap", aid=4)],
        available=["weap", "1tnk", "2tnk", "pbox", "powr"],
        units=units, cash=5000)
    obs.map_info = type("MI", (), {
        "height": 50, "width": 108, "map_name": "fase2_a_short.oramap",
    })()
    out = t._apply_expand(obs, [])
    assert any(c.action == ActionType.BUILD and c.item_type == "weap"
               for c in out)
    assert not any(c.action == ActionType.TRAIN
                   and c.item_type in ("1tnk", "2tnk") for c in out)


def test_expand_no_third_weap():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    units = (_rifles(6)
             + [_U("harv", aid=2), _U("harv", aid=3), _U("harv", aid=4)]
             + [_U("1tnk", aid=80 + i) for i in range(4)])
    obs = _Obs(
        buildings=[_B("proc"), _B("proc", aid=6), _B("proc", aid=9),
                   _B("powr", aid=2), _B("tent", aid=3), _B("weap", aid=4),
                   _B("weap", aid=8)],
        available=["weap", "1tnk", "2tnk", "pbox"],
        units=units, cash=5000)
    obs.map_info = type("MI", (), {
        "height": 50, "width": 108, "map_name": "fase2_a_short.oramap",
    })()
    out = t._apply_expand(obs, [])
    assert not any(c.action == ActionType.BUILD and c.item_type == "weap"
                   for c in out)
    assert any(c.action == ActionType.TRAIN
               and c.item_type in ("1tnk", "2tnk") for c in out)


def test_expand_does_not_mill_third_pbox():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    obs = _Obs(
        buildings=[_B("proc"), _B("proc", aid=6), _B("powr", aid=2),
                   _B("tent", aid=3), _B("pbox", aid=5), _B("pbox", aid=7)],
        available=["pbox", "1tnk", "e1"],
        units=_rifles(6), cash=5000)
    out = t._apply_expand(obs, [])
    assert not any(c.action == ActionType.BUILD and c.item_type == "pbox"
                   for c in out)


def test_expand_pbox_while_weap_building():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    obs = _Obs(
        buildings=[_B("proc"), _B("proc", aid=6), _B("powr", aid=2),
                   _B("tent", aid=3)],
        available=["pbox", "1tnk", "e1"],
        units=_rifles(6), cash=5000)
    obs.production = [_P("Building", "weap", progress=0.4)]
    out = t._apply_expand(obs, [])
    assert any(c.action == ActionType.BUILD and c.item_type == "pbox"
               for c in out)


def test_expand_places_defense_queue():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    obs = _Obs(
        buildings=[_B("fact", x=10, y=10, aid=1), _B("proc", aid=2),
                   _B("tent", aid=3)],
        available=["pbox"],
        units=_rifles(2), cash=500)
    obs.production = [_P("Defense", "pbox", progress=1.0)]
    out = t._handle_placement(obs)
    placed = [c for c in out if c.action == ActionType.PLACE_BUILDING
              and c.item_type == "pbox"]
    assert len(placed) == 1


def test_expand_low_power_builds_powr_before_weap():
    t = _T(verbose=False, mode="expand")
    t.phase = "attack"
    obs = _Obs(available=["weap", "pbox", "powr", "e1", "1tnk"],
               units=_rifles(8), cash=5000)
    obs.economy.power_provided = 100
    obs.economy.power_drained = 460
    obs.economy.ore = 12000
    out = t._apply_expand(obs, [])
    assert len(out) == 1
    assert out[0].action == ActionType.BUILD
    assert out[0].item_type == "powr"


def test_optional_tech_low_power_before_pbox():
    t = _T(verbose=False, mode="rush")
    t.phase = "attack"
    obs = _Obs(available=["pbox", "powr", "dome"], units=_rifles(8), cash=0)
    obs.economy.power_provided = 100
    obs.economy.power_drained = 200
    obs.economy.ore = 8000
    out = t._optional_tech_defense(obs, [])
    assert len(out) == 1
    assert out[0].item_type == "powr"


def test_handle_production_strips_e1_when_low_power():
    t = _T(verbose=False, mode="rush")
    t.phase = "attack"
    obs = _Obs(available=["powr", "e1", "pbox"], units=_rifles(8), cash=5000)
    obs.economy.power_provided = 100
    obs.economy.power_drained = 300
    from openra_env.models import CommandModel
    # parent-like e1 already in the list
    t._can_produce_item = lambda obs, item: item in (obs.available_production or [])
    out = t._handle_production(obs)
    assert not any(c.action == ActionType.TRAIN and c.item_type == "e1"
                   for c in out)
    assert any(c.action == ActionType.BUILD and c.item_type == "powr"
               for c in out)

def test_expand_committed_hunts_fog_when_no_leftover():
    """Bench seed 8047: 76 tanks, ene_nb=1 fogged, never finished."""
    t = _T(verbose=False, mode="expand")
    t.phase = "attack"
    t._expand_committed = True
    t._last_contact = (90, 10)
    t._push_cell = lambda obs: None
    units = _rifles(2) + [
        _U("1tnk", aid=80 + i, x=40, y=20, is_idle=True)
        for i in range(t.EXPAND_PUSH_TANKS)]
    obs = _Obs(
        buildings=[_B("fact", x=12, y=16, aid=1),
                   _B("weap", x=14, y=16, aid=2)],
        units=units, cash=500)
    # No visible leftover — must still issue tank AM via hunt/fog.
    out = t._handle_combat(obs)
    assert any(c.action == ActionType.ATTACK_MOVE and c.actor_id >= 80
               for c in out)


def test_expand_third_proc_not_before_weap():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    obs = _Obs(
        buildings=[_B("proc"), _B("proc", aid=6), _B("powr", aid=2),
                   _B("tent", aid=3)],
        available=["proc", "weap", "pbox", "powr", "e1"],
        units=_rifles(6), cash=5000)
    out = t._apply_expand(obs, [])
    # Still saving/building weap; must not mill a 3rd proc pre-weap.
    assert not any(c.action == ActionType.BUILD and c.item_type == "proc"
                   for c in out)
    assert any(c.action == ActionType.BUILD and c.item_type == "weap"
               for c in out)


def test_expand_second_proc_prefers_mid_facing_home_mine():
    """2nd proc claims the home mine closest to mid (bridgehead), not far corner."""
    t = _T(verbose=False, mode="expand")
    # West CY; first proc already on (23,5); next should prefer (32,28) over (4,30).
    cy = _B("fact", x=12, y=16, aid=1)
    obs = _Obs(
        buildings=[cy, _B("proc", x=20, y=8, aid=5), _B("powr", x=14, y=18, aid=2)],
        cash=2000)
    obs.map_info = type("MI", (), {
        "height": 50, "width": 108, "map_name": "fase2_a_short.oramap",
    })()
    x, y = t._ore_place_cell(obs, cy)
    # Toward SE home mine (32,28), not SW (4,30).
    assert x >= 18
    assert y >= 16


def test_expand_mid_proc_place_near_mid_when_anchored():
    t = _T(verbose=False, mode="expand")
    cy = _B("fact", x=12, y=16, aid=1)
    obs = _Obs(
        buildings=[cy,
                   _B("proc", x=15, y=16, aid=5),
                   _B("proc", x=28, y=30, aid=6),
                   _B("powr", x=48, y=34, aid=2)],
        cash=2000)
    obs.map_info = type("MI", (), {
        "height": 50, "width": 108, "map_name": "fase2_a_short.oramap",
    })()
    x, y = t._ore_place_cell(obs, cy)
    assert abs(x - 56) <= 14
    assert abs(y - 35) <= 14


def test_expand_idle_harv_can_target_mid():
    t = _T(verbose=False, mode="expand")
    obs = _Obs(
        buildings=[_B("fact", x=12, y=16, aid=1),
                   _B("proc", x=15, y=16, aid=5),
                   _B("proc", x=28, y=30, aid=6)],
        units=[_U("harv", aid=1 + i, x=16, y=16, is_idle=True) for i in range(2)],
        cash=100)
    obs.map_info = type("MI", (), {
        "height": 50, "width": 108, "map_name": "fase2_a_short.oramap",
    })()
    obs.tick = 500
    out = t._handle_harvesters(obs)
    assert any(c.action == ActionType.HARVEST for c in out)
    dests = {(c.target_x, c.target_y) for c in out}
    assert (56, 35) in dests


def test_expand_mcv_ignores_opening_before_cy():
    t = _T(verbose=False, mode="expand")
    obs = _Obs(
        buildings=[],
        units=[_U("mcv", aid=50, x=12, y=16, is_idle=True)],
        cash=5000)
    obs.map_info = type("MI", (), {
        "height": 50, "width": 108, "map_name": "fase2_a_short.oramap",
    })()
    out = t._handle_mcv_expand(obs)
    assert out == []

