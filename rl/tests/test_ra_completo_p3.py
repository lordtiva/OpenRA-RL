# -*- coding: utf-8 -*-
"""P3 RA-completo: map catalog, naval masks, water map pool hook."""
import base64
from pathlib import Path

from rl import map_catalog as mapcat
from rl.action_adapter import ActionIndex, Vocab, owns_proc, economy_ready_for_combat
from rl.map_catalog import allows_naval, parse_pool_arg, reset_payload_for


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
        self.can_attack = False
        self.is_idle = True
        self.speed = 50
        self.attack_range = 0
        self.experience_level = 0
        self.stance = 0
        self.facing = 0


class _Obs:
    def __init__(self, map_name="fase2_a_short.oramap", buildings=None,
                 units=None, available=None):
        self.units = units or [_U()]
        self.buildings = buildings if buildings is not None else [_B("proc"), _B("powr", aid=98)]
        self.visible_enemies = []
        self.visible_enemy_buildings = []
        self.production = []
        self.available_production = list(available or [])
        self.economy = type("E", (), {
            "cash": 5000, "ore": 0, "resource_capacity": 2000,
            "power_provided": 100, "power_drained": 50,
            "harvester_count": sum(1 for u in self.units if "harv" in u.type),
        })()
        self.military = type("M", (), {
            "units_killed": 0, "units_lost": 0, "buildings_killed": 0,
            "buildings_lost": 0, "army_value": 0, "active_unit_count": 0,
            "kills_cost": 0, "deaths_cost": 0, "assets_value": 0,
        })()
        self.map_info = type("MI", (), {
            "height": 32, "width": 32, "map_name": map_name,
        })()
        self.spatial_map = ""
        self.spatial_channels = 9
        self.tick = 1


def _aidx(obs):
    v = Vocab()
    v.seed_roles()
    return ActionIndex(obs, v)


def test_catalog_inventory_has_land_and_water():
    keys = set(mapcat.MAP_CATALOG)
    assert "a_short" in keys
    assert "doughnut" in keys
    assert "bombardment_islands" in keys
    assert mapcat.get_entry("a_short").has_water is False
    assert mapcat.get_entry("doughnut").has_water is True
    assert mapcat.get_entry("bombardment_islands").has_water is True


def test_stock_water_files_exist():
    for key in ("doughnut", "bombardment_islands"):
        path = mapcat.resolve_map_path(key)
        assert path.exists(), path
        assert path.stat().st_size > 1000


def test_allows_naval_hints():
    assert allows_naval("fase2_a_short.oramap") is False
    assert allows_naval("singles.oramap") is False
    assert allows_naval("stock_doughnut.oramap") is True
    assert allows_naval("doughnut.oramap") is True
    assert allows_naval("archipelago.oramap") is True


def test_named_pools_expose_two_plus_maps():
    water = parse_pool_arg("water")
    mixed = parse_pool_arg("mixed")
    assert water is not None and len(water) >= 2
    assert mixed is not None and len(mixed) >= 2
    assert "a_short" in mixed
    assert "doughnut" in mixed


def test_reset_payload_doughnut_stable_name():
    payload = reset_payload_for("doughnut")
    assert payload["map_name"] == "stock_doughnut.oramap"
    raw = base64.b64decode(payload["map_data"])
    assert raw[:2] == Path(mapcat.resolve_map_path("doughnut")).read_bytes()[:2]


def test_naval_build_masked_on_land_a_short():
    avail = ["powr", "proc", "tent", "weap", "syrd", "hpad", "e1", "harv", "dd"]
    obs = _Obs(map_name="fase2_a_short.oramap", available=avail,
               buildings=[_B("proc"), _B("powr", aid=2), _B("tent", aid=3)],
               units=[_U("harv"), _U("e1", aid=2)])
    assert owns_proc(obs) and economy_ready_for_combat(obs)
    aidx = _aidx(obs)
    assert "naval" in aidx.build_items
    n_train = len(aidx.train_items)
    naval_slot = n_train + aidx.build_items.index("naval")
    assert bool(aidx.build_slot_mask[naval_slot]) is False
    assert bool(aidx.item_mask[naval_slot]) is False
    if "ship_combat" in aidx.train_items:
        s = aidx.train_items.index("ship_combat")
        assert bool(aidx.train_slot_mask[s]) is False
    if "airbase" in aidx.build_items:
        aslot = n_train + aidx.build_items.index("airbase")
        assert bool(aidx.build_slot_mask[aslot]) is True


def test_naval_build_unmasked_on_water_doughnut():
    avail = ["powr", "proc", "tent", "weap", "syrd", "hpad", "e1", "harv", "dd"]
    obs = _Obs(map_name="stock_doughnut.oramap", available=avail,
               buildings=[_B("proc"), _B("powr", aid=2), _B("tent", aid=3)],
               units=[_U("harv"), _U("e1", aid=2)])
    aidx = _aidx(obs)
    assert "naval" in aidx.build_items
    n_train = len(aidx.train_items)
    naval_slot = n_train + aidx.build_items.index("naval")
    assert bool(aidx.build_slot_mask[naval_slot]) is True
    if "ship_combat" in aidx.train_items:
        s = aidx.train_items.index("ship_combat")
        assert bool(aidx.train_slot_mask[s]) is True


def test_teacher_optional_naval_noop_on_land():
    from rl.scripted_teacher import ScriptedTeacher
    t = ScriptedTeacher(verbose=False)
    obs = _Obs(map_name="fase2_a_short.oramap",
               available=["syrd", "spen", "hpad"],
               buildings=[_B("proc"), _B("tent", aid=2)],
               units=[_U("harv")])
    obs.economy.cash = 5000
    out = t._optional_naval_air(obs, [])
    assert out == []


def test_teacher_optional_naval_on_water():
    from rl.scripted_teacher import ScriptedTeacher
    from openra_env.models import ActionType

    class _T(ScriptedTeacher):
        def _can_produce_item(self, obs, item_type):
            return item_type in (obs.available_production or [])

    t = _T(verbose=False)
    obs = _Obs(map_name="stock_doughnut.oramap",
               available=["syrd", "spen", "hpad", "tent"],
               buildings=[_B("proc"), _B("tent", aid=2)],
               units=[_U("harv")])
    obs.economy.cash = 5000
    out = t._optional_naval_air(obs, [])
    assert len(out) == 1
    assert out[0].action == ActionType.BUILD
    assert out[0].item_type in ("syrd", "spen")

