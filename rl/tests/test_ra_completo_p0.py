# -*- coding: utf-8 -*-
"""P0 RA-completo: per-harvester cell path + naval/air macros (ckpt-safe append)."""
from openra_env.models import ActionType, CommandModel
from rl.network import (
    ACTION_TYPES, N_ACTION_TYPES, TYPE_TO_IDX, TYPES_USE_CELL,
    adapt_v2_state_dict, AlphaLiteNet,
)
from rl.action_adapter import (
    ActionIndex, Vocab, ENABLED_TYPES, group_actor_ids,
    index_to_command, index_to_command_effective,
    n_naval_total, n_air_total, n_vehicle_combat_total,
)


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


class _B:
    def __init__(self, typ="proc", x=5, y=5, aid=99):
        self.type = typ
        self.cell_x = x
        self.cell_y = y
        self.actor_id = aid


class _Obs:
    def __init__(self, units=None, buildings=None, enemies=None, harv_count=None):
        self.units = units or []
        self.buildings = buildings or [_B()]
        self.visible_enemies = enemies or []
        self.visible_enemy_buildings = []
        self.production = []
        self.available_production = []
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


def test_action_types_append_only():
    assert ACTION_TYPES[-2:] == ["naval_attack_move", "air_attack_move"]
    assert N_ACTION_TYPES == TYPE_TO_IDX["air_attack_move"] + 1
    assert "naval_attack_move" in ENABLED_TYPES
    assert "harvest" in TYPES_USE_CELL


def test_vehicle_bucket_excludes_naval_air():
    units = [
        _U(actor_id=1, type="1tnk"),
        _U(actor_id=2, type="dd"),
        _U(actor_id=3, type="heli"),
        _U(actor_id=4, type="e1"),
    ]
    obs = _Obs(units=units)
    assert group_actor_ids(obs, "vehicle") == [1]
    assert group_actor_ids(obs, "naval") == [2]
    assert group_actor_ids(obs, "air") == [3]
    assert n_vehicle_combat_total(obs) == 1
    assert n_naval_total(obs) == 1
    assert n_air_total(obs) == 1


def test_one_harvester_move_emits_single_move():
    h1 = _U(actor_id=11, type="harv", can_attack=False, cell_x=3, cell_y=3)
    h2 = _U(actor_id=12, type="harv", can_attack=False, cell_x=4, cell_y=4)
    tank = _U(actor_id=20, type="1tnk")
    obs = _Obs(units=[h1, h2, tank])
    aidx = _aidx(obs)
    # unit slot for h2
    assert 12 in aidx.unit_ids
    slot = aidx.unit_ids.index(12)
    cx, cy = 15, 10
    cell = cy * aidx.w + cx
    t = TYPE_TO_IDX["move"]
    action = index_to_command(obs, t, slot, cell, 0, aidx)
    assert len(action.commands) == 1
    cmd = action.commands[0]
    assert cmd.action == ActionType.MOVE
    assert cmd.actor_id == 12
    assert cmd.target_x == cx and cmd.target_y == cy


def test_attack_on_harvester_becomes_move_not_any_harvest():
    h1 = _U(actor_id=11, type="harv", can_attack=False)
    h2 = _U(actor_id=12, type="harv", can_attack=False)
    obs = _Obs(units=[h1, h2])
    aidx = _aidx(obs)
    slot = aidx.unit_ids.index(12)
    cx, cy = 8, 9
    cell = cy * aidx.w + cx
    action, eff = index_to_command_effective(
        obs, TYPE_TO_IDX["attack"], slot, cell, 0, aidx)
    assert len(action.commands) == 1
    cmd = action.commands[0]
    assert cmd.action == ActionType.MOVE
    assert cmd.actor_id == 12
    assert ACTION_TYPES[eff[0]] == "move"


def test_harvesters_move_still_n_cmds():
    harvs = [
        _U(actor_id=11, type="harv", can_attack=False),
        _U(actor_id=12, type="harv", can_attack=False),
        _U(actor_id=13, type="harv", can_attack=False),
    ]
    obs = _Obs(units=harvs + [_U(actor_id=20, type="1tnk")])
    aidx = _aidx(obs)
    cx, cy = 20, 21
    cell = cy * aidx.w + cx
    action = index_to_command(
        obs, TYPE_TO_IDX["harvesters_move"], 0, cell, 0, aidx)
    assert len(action.commands) == 3
    aids = sorted(c.actor_id for c in action.commands)
    assert aids == [11, 12, 13]
    for c in action.commands:
        assert c.action == ActionType.MOVE
        assert c.target_x == cx and c.target_y == cy


def test_harvest_uses_selected_id_and_cell():
    h1 = _U(actor_id=11, type="harv", can_attack=False, is_idle=True)
    h2 = _U(actor_id=12, type="harv", can_attack=False, is_idle=True)
    obs = _Obs(units=[h1, h2])
    aidx = _aidx(obs)
    slot = aidx.unit_ids.index(12)
    cx, cy = 7, 6
    cell = cy * aidx.w + cx
    action, eff = index_to_command_effective(
        obs, TYPE_TO_IDX["harvest"], slot, cell, 0, aidx)
    assert len(action.commands) == 1
    cmd = action.commands[0]
    assert cmd.action == ActionType.HARVEST
    assert cmd.actor_id == 12
    assert cmd.target_x == cx and cmd.target_y == cy
    assert aidx.unit_ids[eff[1]] == 12


def test_naval_air_macros_and_mask():
    # No ships/air -> masked
    land = _Obs(units=[_U(actor_id=1, type="1tnk"), _U(actor_id=2, type="e1")])
    aidx = _aidx(land)
    assert not bool(aidx.type_mask[TYPE_TO_IDX["naval_attack_move"]])
    assert not bool(aidx.type_mask[TYPE_TO_IDX["air_attack_move"]])
    # With ships/air -> legal (proc owned via default building)
    sea = _Obs(units=[
        _U(actor_id=1, type="dd"),
        _U(actor_id=2, type="pt"),
        _U(actor_id=3, type="heli"),
        _U(actor_id=4, type="1tnk"),
    ])
    aidx2 = _aidx(sea)
    assert bool(aidx2.type_mask[TYPE_TO_IDX["naval_attack_move"]])
    assert bool(aidx2.type_mask[TYPE_TO_IDX["air_attack_move"]])
    cx, cy = 10, 11
    cell = cy * aidx2.w + cx
    naval = index_to_command(
        sea, TYPE_TO_IDX["naval_attack_move"], 0, cell, 0, aidx2)
    assert len(naval.commands) == 2
    assert sorted(c.actor_id for c in naval.commands) == [1, 2]
    assert all(c.action == ActionType.ATTACK_MOVE for c in naval.commands)
    air = index_to_command(
        sea, TYPE_TO_IDX["air_attack_move"], 0, cell, 0, aidx2)
    assert len(air.commands) == 1
    assert air.commands[0].actor_id == 3
    assert air.commands[0].action == ActionType.ATTACK_MOVE


def test_partial_type_head_load_from_shorter_ckpt():
    net = AlphaLiteNet()
    # Simulate older ckpt with fewer action types (pre naval/air)
    old_n = N_ACTION_TYPES - 2
    raw = net.state_dict()
    raw = {k: v.clone() for k, v in raw.items()}
    raw["head_type.weight"] = raw["head_type.weight"][:old_n].clone()
    raw["head_type.bias"] = raw["head_type.bias"][:old_n].clone()
    raw["type_embedding.weight"] = raw["type_embedding.weight"][:old_n].clone()
    adapted = adapt_v2_state_dict(net, raw)
    assert adapted["head_type.weight"].shape[0] == N_ACTION_TYPES
    assert adapted["type_embedding.weight"].shape[0] == N_ACTION_TYPES
    # Overlapping rows preserved
    assert (adapted["head_type.weight"][:old_n] == raw["head_type.weight"]).all()


if __name__ == "__main__":
    test_action_types_append_only()
    test_vehicle_bucket_excludes_naval_air()
    test_one_harvester_move_emits_single_move()
    test_attack_on_harvester_becomes_move_not_any_harvest()
    test_harvesters_move_still_n_cmds()
    test_harvest_uses_selected_id_and_cell()
    test_naval_air_macros_and_mask()
    test_partial_type_head_load_from_shorter_ckpt()
    print("OK ra-completo-p0 tests")
