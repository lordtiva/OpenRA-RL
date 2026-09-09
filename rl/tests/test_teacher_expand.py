# -*- coding: utf-8 -*-
"""Expand teacher: weap/1tnk/e3 after rush; opening intact before."""
from openra_env.models import ActionType
from rl.scripted_teacher import ScriptedTeacher


class _B:
    def __init__(self, typ="proc", x=5, y=5, aid=99):
        self.type = typ
        self.cell_x = x
        self.cell_y = y
        self.actor_id = aid


class _U:
    def __init__(self, typ="harv", aid=1, x=8, y=8):
        self.type = typ
        self.actor_id = aid
        self.cell_x = x
        self.cell_y = y
        self.hp_percent = 1.0
        self.can_attack = typ in ("e1", "e3", "1tnk", "2tnk")
        self.is_idle = True
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


def test_expand_noop_before_rush():
    t = _T(verbose=False, mode="expand")
    t.phase = "train_army"
    obs = _Obs(available=["weap", "pbox", "e1", "e3", "1tnk"],
               units=_rifles(2), cash=5000)
    out = t._handle_production(obs)
    assert not any(c.action == ActionType.BUILD and c.item_type == "weap"
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
