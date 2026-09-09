# -*- coding: utf-8 -*-
"""RA Aliados (rl/docs/contract/ra-aliados.md): lock, catálogo, APC, IDENTITY_ITEMS, patrol, Chronosphere, Chrono Tank."""
from openra_env.models import ActionType
from rl.allies import (
    ALLIED_CATALOG_BUILDINGS, ALLIED_CATALOG_UNITS, DEFAULT_ENEMY_FACTION,
    DEFAULT_PLAYER_FACTION, is_rl_agent_bot, resolve_enemy_faction,
    resolve_player_faction,
)
from rl.action_adapter import (
    BUILDING_ITEM_TYPES, ENABLED_TYPES, ActionIndex, Vocab,
    can_issue_enter_transport, can_issue_unload, index_to_command,
)
from rl.network import TYPE_TO_IDX, TYPES_USE_UNIT
from rl.roles import role_of


class _U:
    def __init__(self, **kw):
        self.actor_id = kw.get("actor_id", 1)
        self.type = kw.get("type", "e1")
        self.hp_percent = kw.get("hp_percent", 1.0)
        self.can_attack = kw.get("can_attack", True)
        self.is_idle = kw.get("is_idle", True)
        self.speed = kw.get("speed", 50)
        self.attack_range = kw.get("attack_range", 1000)
        self.experience_level = kw.get("experience_level", 0)
        self.stance = kw.get("stance", 0)
        self.cell_x = kw.get("cell_x", 10)
        self.cell_y = kw.get("cell_y", 10)
        self.facing = kw.get("facing", 0)
        self.passenger_count = kw.get("passenger_count", 0)


class _B:
    def __init__(self, typ="proc", x=5, y=5, aid=99):
        self.type = typ
        self.cell_x = x
        self.cell_y = y
        self.actor_id = aid


class _Obs:
    def __init__(self, units=None, buildings=None, enemies=None, harv_count=None,
                 available=None):
        self.units = units or []
        self.buildings = buildings if buildings is not None else [_B()]
        self.visible_enemies = enemies or []
        self.visible_enemy_buildings = []
        self.production = []
        self.available_production = available or []
        hc = harv_count
        if hc is None:
            hc = sum(1 for u in self.units if "harv" in str(u.type).lower())
        self.economy = type("E", (), {
            "cash": 5000, "ore": 0, "resource_capacity": 2000,
            "power_provided": 100, "power_drained": 50, "harvester_count": hc,
        })()
        self.military = type("M", (), {
            "units_killed": 0, "units_lost": 0, "buildings_killed": 0,
            "buildings_lost": 0, "army_value": 0, "active_unit_count": 0,
            "kills_cost": 0, "deaths_cost": 0, "assets_value": 0,
        })()
        self.map_info = type("MI", (), {
            "height": 32, "width": 32, "map_name": "t",
        })()
        self.spatial_map = ""
        self.spatial_channels = 9
        self.tick = 1


def _aidx(obs):
    v = Vocab()
    v.seed_roles()
    return ActionIndex(obs, v)


def test_faction_lock_defaults():
    assert DEFAULT_PLAYER_FACTION == "RandomAllies"
    assert DEFAULT_ENEMY_FACTION == "Random"
    assert resolve_player_faction("") == "RandomAllies"
    assert resolve_player_faction(None) == "RandomAllies"
    assert resolve_player_faction("Random") == "RandomAllies"
    assert resolve_player_faction("random") == "RandomAllies"
    assert resolve_player_faction("england") == "england"
    assert resolve_player_faction("germany") == "germany"
    assert resolve_enemy_faction("") == "Random"
    assert resolve_enemy_faction("RandomSoviet") == "RandomSoviet"
    assert is_rl_agent_bot("rl-agent")
    assert is_rl_agent_bot("rl")
    assert not is_rl_agent_bot("beginner")


def test_allied_catalog_not_misc():
    missing = [it for it in ALLIED_CATALOG_UNITS + ALLIED_CATALOG_BUILDINGS
               if role_of(it) == "misc"]
    assert missing == [], missing


def test_superweapons_and_fakes_are_buildings():
    for it in ("pdox", "iron", "mslo", "fpwr", "tenf", "facf"):
        assert it in BUILDING_ITEM_TYPES, it


def test_pdox_does_not_enter_train_split():
    obs = _Obs(
        units=[_U(actor_id=1, type="harv")],
        buildings=[_B("proc"), _B("atek", aid=2)],
        available=["e1", "pdox", "atek", "harv"],
    )
    aidx = _aidx(obs)
    # IDENTITY_ITEMS: pdox/atek are their own build keys (not folded into tech).
    assert "pdox" in aidx.build_items
    assert "atek" in aidx.build_items
    assert aidx.rol_a_concreto.get("pdox") == "pdox"
    assert "pdox" not in aidx.train_items
    if aidx.train_items:
        for role in aidx.train_items:
            assert aidx.rol_a_concreto[role] not in BUILDING_ITEM_TYPES


def test_identity_items_not_cheapest_of():
    from rl.roles import IDENTITY_ITEMS
    obs = _Obs(
        units=[_U(actor_id=1, type="harv")],
        buildings=[_B("proc"), _B("tent", aid=2)],
        available=["e1", "e6", "e7", "medi", "mech", "spy", "harv"],
    )
    aidx = _aidx(obs)
    for it in ("e7", "medi", "mech", "spy"):
        assert it in IDENTITY_ITEMS
        assert it in aidx.train_items, it
        assert aidx.rol_a_concreto[it] == it


def test_patrol_emits():
    obs = _Obs(units=[_U(actor_id=1, type="e1", can_attack=True)])
    aidx = _aidx(obs)
    assert "patrol" in ENABLED_TYPES
    assert bool(aidx.type_mask[TYPE_TO_IDX["patrol"]])
    action = index_to_command(
        obs, TYPE_TO_IDX["patrol"], 0, 20 * aidx.w + 20, 0, aidx)
    cmd = action.commands[0]
    assert cmd.action == ActionType.PATROL
    assert cmd.actor_id == 1
    assert cmd.target_x == 20 and cmd.target_y == 20


def test_support_power_masked_without_ready():
    obs = _Obs(units=[_U(actor_id=1, type="e1")])
    aidx = _aidx(obs)
    assert not bool(aidx.type_mask[TYPE_TO_IDX["support_power"]])


def test_support_power_emits_chronoshift():
    obs = _Obs(units=[_U(actor_id=1, type="e1", cell_x=8, cell_y=8)])
    obs.ready_support_powers = ["Chronoshift", "GpsPowerOrder"]
    aidx = _aidx(obs)
    assert bool(aidx.type_mask[TYPE_TO_IDX["support_power"]])
    action = index_to_command(
        obs, TYPE_TO_IDX["support_power"], 0, 20 * aidx.w + 20, 0, aidx)
    cmd = action.commands[0]
    assert cmd.action == ActionType.SUPPORT_POWER
    assert cmd.item_type == "Chronoshift"
    assert cmd.actor_id == 1
    assert cmd.target_x == 20 and cmd.target_y == 20


def test_chrono_tank_deploy_keeps_cell():
    obs = _Obs(units=[_U(actor_id=7, type="ctnk", cell_x=10, cell_y=10)])
    aidx = _aidx(obs)
    action = index_to_command(
        obs, TYPE_TO_IDX["deploy"], 0, 20 * aidx.w + 25, 0, aidx)
    cmd = action.commands[0]
    assert cmd.action == ActionType.DEPLOY
    assert cmd.actor_id == 7
    assert cmd.target_x == 25 and cmd.target_y == 20


def test_enter_unload_enabled_no_type_head_grow():
    assert "enter_transport" in ENABLED_TYPES
    assert "unload" in ENABLED_TYPES
    assert "enter_transport" in TYPES_USE_UNIT
    assert "unload" in TYPES_USE_UNIT
    # Rows already existed in ACTION_TYPES (not append-only this cut).
    assert TYPE_TO_IDX["enter_transport"] < TYPE_TO_IDX["army_attack_move"]


def test_enter_masked_without_transport():
    obs = _Obs(units=[_U(actor_id=1, type="e1")])
    aidx = _aidx(obs)
    assert not can_issue_enter_transport(obs)
    assert not bool(aidx.type_mask[TYPE_TO_IDX["enter_transport"]])
    assert not can_issue_unload(obs)
    assert not bool(aidx.type_mask[TYPE_TO_IDX["unload"]])


def test_enter_unmasked_with_apc_and_emits():
    obs = _Obs(units=[
        _U(actor_id=1, type="e1", cell_x=10, cell_y=10),
        _U(actor_id=2, type="apc", cell_x=12, cell_y=10, passenger_count=0),
    ])
    aidx = _aidx(obs)
    assert can_issue_enter_transport(obs)
    assert bool(aidx.type_mask[TYPE_TO_IDX["enter_transport"]])
    action = index_to_command(
        obs, TYPE_TO_IDX["enter_transport"], 0, 10 * aidx.w + 10, 0, aidx)
    cmd = action.commands[0]
    assert cmd.action == ActionType.ENTER_TRANSPORT
    assert cmd.actor_id == 1
    assert cmd.target_actor_id == 2


def test_unload_emits_loaded_apc():
    obs = _Obs(units=[
        _U(actor_id=1, type="e1"),
        _U(actor_id=2, type="apc", passenger_count=3),
    ])
    aidx = _aidx(obs)
    assert can_issue_unload(obs)
    assert bool(aidx.type_mask[TYPE_TO_IDX["unload"]])
    action = index_to_command(
        obs, TYPE_TO_IDX["unload"], 0, 0, 0, aidx)
    cmd = action.commands[0]
    assert cmd.action == ActionType.UNLOAD
    assert cmd.actor_id == 2
