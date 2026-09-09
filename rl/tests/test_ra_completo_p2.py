# -*- coding: utf-8 -*-
"""P2 RA-completo micro: guard, group stop/stance, focus fire."""
import torch
from openra_env.models import ActionType, CommandModel
from rl.network import (
    ACTION_TYPES, TYPE_TO_IDX, TYPES_USE_CELL, TYPES_USE_UNIT,
    TYPES_GROUP_MACRO, N_ACTION_TYPES, adapt_v2_state_dict, AlphaLiteNet,
    HIDDEN_DIM,
)
from rl.action_adapter import (
    ActionIndex, Vocab, ENABLED_TYPES, GROUP_MACRO_TYPES,
    index_to_command, index_to_command_effective,
    resolve_guard_target, has_guard_target, can_issue_guard, group_actor_ids,
)
from rl.obs_encoding import MAX_TOKENS, UNIT_FEAT_DIM, SCALAR_DIM


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


class _E:
    def __init__(self, aid=200, x=20, y=20, hp=1.0, typ="e1"):
        self.actor_id = aid
        self.cell_x = x
        self.cell_y = y
        self.hp_percent = hp
        self.type = typ


class _Obs:
    def __init__(self, units=None, buildings=None, enemies=None,
                 enemy_buildings=None, harv_count=None):
        self.units = units or []
        self.buildings = buildings if buildings is not None else [_B()]
        self.visible_enemies = enemies or []
        self.visible_enemy_buildings = enemy_buildings or []
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


def test_p2_types_append_only_and_enabled():
    assert ACTION_TYPES[-5:-2] == ["army_stop", "army_set_stance", "army_guard"]
    assert ACTION_TYPES[-2:] == ["patrol", "support_power"]
    assert N_ACTION_TYPES == TYPE_TO_IDX["support_power"] + 1
    for name in ("guard", "army_stop", "army_set_stance", "army_guard"):
        assert name in ENABLED_TYPES
        assert name in ACTION_TYPES
    assert "guard" in TYPES_USE_UNIT
    assert "guard" in TYPES_USE_CELL
    assert "army_stop" in TYPES_GROUP_MACRO
    assert "army_set_stance" in TYPES_GROUP_MACRO
    assert "army_guard" in TYPES_GROUP_MACRO
    for name in ("army_stop", "army_set_stance", "army_guard"):
        assert name in GROUP_MACRO_TYPES


def test_guard_emits_actor_and_target():
    tank = _U(actor_id=1, type="1tnk", cell_x=8, cell_y=8)
    rifle = _U(actor_id=2, type="e1", cell_x=9, cell_y=9)
    harv = _U(actor_id=11, type="harv", can_attack=False, cell_x=12, cell_y=12)
    obs = _Obs(units=[tank, rifle, harv], buildings=[_B("proc", aid=50, x=5, y=5)],
               harv_count=1)
    aidx = _aidx(obs)
    assert bool(aidx.type_mask[TYPE_TO_IDX["guard"]])
    slot = aidx.unit_ids.index(1)
    # Cell near harv
    cell = 12 * aidx.w + 12
    action, eff = index_to_command_effective(
        obs, TYPE_TO_IDX["guard"], slot, cell, 0, aidx)
    assert len(action.commands) == 1
    cmd = action.commands[0]
    assert cmd.action == ActionType.GUARD
    assert cmd.actor_id == 1
    assert cmd.target_actor_id == 11  # prefer harv over building
    assert ACTION_TYPES[eff[0]] == "guard"


def test_guard_prefers_harv_over_building():
    obs = _Obs(
        units=[_U(actor_id=1, type="1tnk"), _U(actor_id=11, type="harv",
                                               can_attack=False, cell_x=4, cell_y=4)],
        buildings=[_B("proc", aid=50, x=4, y=4)],
        harv_count=1,
    )
    assert resolve_guard_target(obs, 4, 4, exclude_id=1) == 11


def test_guard_masked_without_target_or_combat():
    # Only harv — no combat escort
    obs = _Obs(units=[_U(actor_id=11, type="harv", can_attack=False)],
               buildings=[], harv_count=1)
    aidx = _aidx(obs)
    assert not bool(aidx.type_mask[TYPE_TO_IDX["guard"]])
    assert not bool(aidx.type_mask[TYPE_TO_IDX["army_guard"]])
    # Combat but nothing to guard (no buildings, no other units)
    obs2 = _Obs(units=[_U(actor_id=1, type="1tnk")], buildings=[], harv_count=0)
    assert not has_guard_target(obs2, exclude_id=1)
    assert not can_issue_guard(obs2)
    aidx2 = _aidx(obs2)
    assert not bool(aidx2.type_mask[TYPE_TO_IDX["guard"]])


def test_army_guard_n_combat_one_target():
    units = [
        _U(actor_id=1, type="1tnk"),
        _U(actor_id=2, type="e1"),
        _U(actor_id=3, type="2tnk"),
        _U(actor_id=11, type="harv", can_attack=False, cell_x=15, cell_y=15),
    ]
    obs = _Obs(units=units, harv_count=1)
    aidx = _aidx(obs)
    assert bool(aidx.type_mask[TYPE_TO_IDX["army_guard"]])
    cell = 15 * aidx.w + 15
    action = index_to_command(
        obs, TYPE_TO_IDX["army_guard"], 0, cell, 0, aidx)
    assert len(action.commands) == 3
    assert all(c.action == ActionType.GUARD for c in action.commands)
    assert all(c.target_actor_id == 11 for c in action.commands)
    assert sorted(c.actor_id for c in action.commands) == [1, 2, 3]


def test_army_stop_and_set_stance_group():
    units = [_U(actor_id=i, type="e1") for i in (1, 2, 3)]
    units.append(_U(actor_id=11, type="harv", can_attack=False))
    obs = _Obs(units=units, harv_count=1)
    aidx = _aidx(obs)
    assert bool(aidx.type_mask[TYPE_TO_IDX["army_stop"]])
    assert bool(aidx.type_mask[TYPE_TO_IDX["army_set_stance"]])
    stop = index_to_command(obs, TYPE_TO_IDX["army_stop"], 0, 0, 0, aidx)
    assert len(stop.commands) == 3
    assert all(c.action == ActionType.STOP for c in stop.commands)
    assert sorted(c.actor_id for c in stop.commands) == [1, 2, 3]
    # stance from cx % 4 → cx=5 → 1 (ReturnFire)
    cell = 0 * aidx.w + 5
    st = index_to_command(
        obs, TYPE_TO_IDX["army_set_stance"], 0, cell, 0, aidx)
    assert len(st.commands) == 3
    assert all(c.action == ActionType.SET_STANCE for c in st.commands)
    assert all(c.target_x == 1 for c in st.commands)


def test_single_set_stance_encodes_target_x():
    obs = _Obs(units=[_U(actor_id=1, type="1tnk")], harv_count=1)
    aidx = _aidx(obs)
    action = index_to_command(obs, TYPE_TO_IDX["set_stance"], 0, 0, 0, aidx)
    cmd = action.commands[0]
    assert cmd.action == ActionType.SET_STANCE
    assert cmd.actor_id == 1
    assert cmd.target_x == 3  # AttackAnything default


def test_focus_fire_attack_binds_enemy_actor():
    tank = _U(actor_id=1, type="1tnk", cell_x=10, cell_y=10)
    ene_far = _E(aid=201, x=25, y=25, hp=1.0)
    ene_near_hurt = _E(aid=202, x=12, y=12, hp=0.3)
    ene_near_full = _E(aid=203, x=12, y=12, hp=1.0)
    obs = _Obs(units=[tank], enemies=[ene_far, ene_near_hurt, ene_near_full],
               harv_count=1)
    aidx = _aidx(obs)
    assert bool(aidx.type_mask[TYPE_TO_IDX["attack"]])
    cell = 12 * aidx.w + 12
    action = index_to_command(
        obs, TYPE_TO_IDX["attack"], 0, cell, 0, aidx)
    cmd = action.commands[0]
    assert cmd.action == ActionType.ATTACK
    assert cmd.target_actor_id == 202  # wounded preferred at same cell


def test_attack_masked_without_visible_enemies():
    obs = _Obs(units=[_U(actor_id=1, type="1tnk")], enemies=[], harv_count=1)
    aidx = _aidx(obs)
    assert not bool(aidx.type_mask[TYPE_TO_IDX["attack"]])
    # Safety: if forced, degrades to attack_move
    cell = 5 * aidx.w + 5
    action, eff = index_to_command_effective(
        obs, TYPE_TO_IDX["attack"], 0, cell, 0, aidx)
    assert action.commands[0].action == ActionType.ATTACK_MOVE
    assert ACTION_TYPES[eff[0]] == "attack_move"


def test_land_macros_still_intact():
    units = [_U(actor_id=i, type="e1") for i in range(1, 14)]
    units.append(_U(actor_id=50, type="harv", can_attack=False))
    obs = _Obs(units=units, harv_count=1)
    aidx = _aidx(obs)
    assert bool(aidx.type_mask[TYPE_TO_IDX["army_attack_move"]])
    assert bool(aidx.type_mask[TYPE_TO_IDX["infantry_attack_move"]])
    cell = 10 * aidx.w + 10
    army = index_to_command(
        obs, TYPE_TO_IDX["army_attack_move"], 0, cell, 0, aidx)
    assert len(army.commands) == 1
    assert army.commands[0].action == ActionType.ARMY_ATTACK_MOVE


def test_partial_load_expands_type_head_for_p2():
    """Old ckpt (pre army_stop) soft-expands type head under strict=False."""
    net = AlphaLiteNet()
    raw = {k: v.clone() for k, v in net.state_dict().items()}
    # Shrink type head to pre-P2 size (before army_stop; patrol/support after P2)
    n_old = TYPE_TO_IDX["army_stop"]
    assert ACTION_TYPES[n_old] == "army_stop"
    w = raw["head_type.weight"]
    raw["head_type.weight"] = w[:n_old].clone()
    raw["head_type.bias"] = raw["head_type.bias"][:n_old].clone()
    raw["type_embedding.weight"] = raw["type_embedding.weight"][:n_old].clone()
    net2 = AlphaLiteNet()
    adapted = adapt_v2_state_dict(net2, raw)
    incompat = net2.load_state_dict(adapted, strict=False)
    assert adapted["head_type.weight"].shape[0] == N_ACTION_TYPES
    assert adapted["type_embedding.weight"].shape[0] == N_ACTION_TYPES
    # Smoke act
    B, U = 1, MAX_TOKENS
    own = torch.zeros(B, U, dtype=torch.bool)
    own[0, 0] = True
    batch = {
        "spatial": torch.zeros(B, 9, 8, 8),
        "scalars": torch.zeros(B, SCALAR_DIM),
        "unit_feats": torch.zeros(B, U, UNIT_FEAT_DIM),
        "unit_valid": own.clone(),
        "unit_own_mask": own.clone(),
        "unit_role_ids": torch.zeros(B, U, dtype=torch.long),
        "type_mask": torch.zeros(B, N_ACTION_TYPES, dtype=torch.bool),
        "cell_mask": torch.ones(B, 64, dtype=torch.bool),
        "item_indices": torch.zeros(B, 128, dtype=torch.long),
        "item_mask": torch.zeros(B, 128, dtype=torch.bool),
        "train_slot_mask": torch.zeros(B, 128, dtype=torch.bool),
        "build_slot_mask": torch.zeros(B, 128, dtype=torch.bool),
    }
    batch["type_mask"][0, TYPE_TO_IDX["army_stop"]] = True
    h = torch.zeros(1, HIDDEN_DIM)
    out = net2.act(batch, h, temperature=1.0)
    assert int(out["type"]) == TYPE_TO_IDX["army_stop"]
    _ = incompat  # unused; soft-add may be empty after expand


def test_p0_naval_air_still_last_before_p2():
    i = TYPE_TO_IDX["army_stop"]
    assert ACTION_TYPES[i - 2:i] == ["naval_attack_move", "air_attack_move"]


if __name__ == "__main__":
    test_p2_types_append_only_and_enabled()
    test_guard_emits_actor_and_target()
    test_guard_prefers_harv_over_building()
    test_guard_masked_without_target_or_combat()
    test_army_guard_n_combat_one_target()
    test_army_stop_and_set_stance_group()
    test_single_set_stance_encodes_target_x()
    test_focus_fire_attack_binds_enemy_actor()
    test_attack_masked_without_visible_enemies()
    test_land_macros_still_intact()
    test_partial_load_expands_type_head_for_p2()
    test_p0_naval_air_still_last_before_p2()
    print("OK ra-completo-p2 micro tests")
