# -*- coding: utf-8 -*-
"""P3 RA-completo: map catalog, naval masks, official map pool hook."""
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
    assert "tournament_island" in keys
    assert "x_lake" in keys
    a = mapcat.get_entry("a_short")
    assert a.has_water is True  # lakes present
    assert a.naval_viable is False  # navy still gated
    assert mapcat.get_entry("doughnut").has_water is True
    assert mapcat.get_entry("doughnut").naval_viable is True
    assert mapcat.get_entry("doughnut").display_name == "Doughnut"
    assert mapcat.get_entry("bombardment_islands").display_name == "Bombardment Islands"


def test_official_map_files_resolve():
    for key in ("doughnut", "bombardment_islands", "tournament_island", "x_lake"):
        path = mapcat.resolve_map_path(key)
        assert path.exists(), path
        assert path.stat().st_size > 1000
        # Prefer official basename (not stock_*)
        assert "stock_" not in path.name


def test_allows_naval_hints():
    # Lakes on a_short ≠ navy theatre
    assert allows_naval("fase2_a_short.oramap") is False
    assert allows_naval("a_short") is False
    assert allows_naval("singles.oramap") is False
    assert allows_naval("stock_doughnut.oramap") is True
    assert allows_naval("doughnut.oramap") is True
    assert allows_naval("bombardment-islands.oramap") is True
    assert allows_naval("archipelago.oramap") is True
    assert allows_naval("x-lake.oramap") is True


def test_named_pools_expose_official_2p_small():
    water = parse_pool_arg("water")
    mixed = parse_pool_arg("mixed")
    small = parse_pool_arg("official_2p_small")
    assert water is not None and len(water) >= 2
    assert mixed is not None and len(mixed) >= 2
    assert "a_short" in mixed
    assert "doughnut" in mixed
    assert small is not None and 4 <= len(small) <= 6
    assert small[0] == "a_short" or "a_short" in small
    assert "doughnut" in small
    assert "bombardment_islands" in small


def test_reset_payload_doughnut_official_name():
    payload = reset_payload_for("doughnut")
    assert payload["map_name"] == "doughnut.oramap"
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
    obs = _Obs(map_name="doughnut.oramap", available=avail,
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
    obs = _Obs(map_name="doughnut.oramap",
               available=["syrd", "spen", "hpad", "tent"],
               buildings=[_B("proc"), _B("tent", aid=2)],
               units=[_U("harv")])
    obs.economy.cash = 5000
    out = t._optional_naval_air(obs, [])
    assert len(out) == 1
    assert out[0].action == ActionType.BUILD
    assert out[0].item_type in ("syrd", "spen")


def test_legacy_key_a_aliases_a_short():
    e = mapcat.get_entry("a")
    assert e.key == "a_short"
    assert e.file_name == "fase2_a_short.oramap"
    assert mapcat.normalize_key("a") == "a_short"
    assert mapcat.normalize_key("fase2_a") == "a_short"


def test_fase2_a_only_in_archive():
    active = mapcat.SCENARIOS_DIR / "fase2_a.oramap"
    archived = mapcat.SCENARIOS_DIR / "_archive" / "fase2_a.oramap"
    assert not active.exists(), "long fase2_a should be archived"
    assert archived.exists()
    assert (mapcat.SCENARIOS_DIR / "fase2_a_short.oramap").exists()


def test_onboard_map_pool_injects_argv():
    from rl.onboard import build_train_argv, new_curriculum
    base = ["python", "-m", "rl.train", "--scenario", "a_short", "--sil"]
    cfg = new_curriculum({"map_pool": "official_2p_small"})
    argv = build_train_argv(base, "C", cfg)
    assert "--map-pool" in argv
    assert argv[argv.index("--map-pool") + 1] == "official_2p_small"
    # empty pool: no --map-pool (C resume / MAIN land)
    cfg2 = new_curriculum({"map_pool": ""})
    argv2 = build_train_argv(base, "C", cfg2)
    assert "--map-pool" not in argv2


def test_teacher_optional_naval_trains_ship_after_yard():
    from rl.scripted_teacher import ScriptedTeacher
    from openra_env.models import ActionType

    class _T(ScriptedTeacher):
        def _can_produce_item(self, obs, item_type):
            return item_type in (obs.available_production or [])

    t = _T(verbose=False)
    obs = _Obs(map_name="doughnut.oramap",
               available=["dd", "pt", "heli"],
               buildings=[_B("proc"), _B("tent", aid=2), _B("syrd", aid=3)],
               units=[_U("harv")])
    obs.economy.cash = 2000
    out = t._optional_naval_air(obs, [])
    assert len(out) == 1
    assert out[0].action == ActionType.TRAIN
    assert out[0].item_type in ("dd", "pt", "ss", "ca", "msub")
