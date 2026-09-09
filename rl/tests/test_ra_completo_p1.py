# -*- coding: utf-8 -*-
"""P1 RA-completo: building-slot sell/repair/rally/power_down/set_primary (usable)."""
import torch
from openra_env.models import ActionType, CommandModel
from rl.network import (
    ACTION_TYPES, TYPE_TO_IDX, TYPES_USE_CELL, TYPES_USE_UNIT,
    TYPES_USE_BUILDING, N_ACTION_TYPES, adapt_v2_state_dict, AlphaLiteNet,
    HIDDEN_DIM, _slot_legal_for_types,
)
from rl.action_adapter import (
    ActionIndex, Vocab, ENABLED_TYPES, BUILDING_SLOT_TYPES,
    index_to_command, index_to_command_effective,
)
from rl.imitation import command_to_indices
from rl.obs_encoding import MAX_UNITS, MAX_TOKENS, UNIT_FEAT_DIM, SCALAR_DIM


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
        self.buildings = buildings if buildings is not None else [_B()]
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


def _mini_batch(aidx, h=8, w=8):
    """Minimal act() batch with building_valid for P1 sampling tests."""
    B, U = 1, MAX_TOKENS
    own = torch.zeros(B, U, dtype=torch.bool)
    own[0, 0] = True  # only 1 own unit slot
    bvalid = aidx.building_valid.unsqueeze(0).clone()
    # building_valid is MAX_UNITS; pad to MAX_TOKENS like unit masks
    if bvalid.size(-1) < U:
        pad = torch.zeros(B, U - bvalid.size(-1), dtype=torch.bool)
        bvalid = torch.cat([bvalid, pad], dim=-1)
    return {
        "spatial": torch.zeros(B, 9, h, w),
        "scalars": torch.zeros(B, SCALAR_DIM),
        "unit_feats": torch.zeros(B, U, UNIT_FEAT_DIM),
        "unit_valid": own.clone(),
        "unit_own_mask": own.clone(),
        "unit_role_ids": torch.zeros(B, U, dtype=torch.long),
        "type_mask": aidx.type_mask.unsqueeze(0).clone(),
        "cell_mask": torch.ones(B, h * w, dtype=torch.bool),
        "item_indices": aidx.item_indices.unsqueeze(0),
        "item_mask": aidx.item_mask.unsqueeze(0),
        "train_slot_mask": aidx.train_slot_mask.unsqueeze(0),
        "build_slot_mask": aidx.build_slot_mask.unsqueeze(0),
        "building_valid": bvalid,
    }


def test_building_types_enabled_append_only():
    for name in ("sell", "repair", "set_rally_point", "power_down", "set_primary"):
        assert name in ENABLED_TYPES
        assert name in BUILDING_SLOT_TYPES
        assert name in TYPES_USE_BUILDING
        assert name in TYPES_USE_UNIT
        assert name in ACTION_TYPES
    assert "set_rally_point" in TYPES_USE_CELL
    # Still append-only naval/air at end of ACTION_TYPES
    assert ACTION_TYPES[-2:] == ["naval_attack_move", "air_attack_move"]
    assert N_ACTION_TYPES == TYPE_TO_IDX["air_attack_move"] + 1


def test_building_slot_masked_without_buildings():
    obs = _Obs(units=[_U(actor_id=1, type="1tnk")], buildings=[])
    aidx = _aidx(obs)
    assert aidx.building_ids == []
    for name in BUILDING_SLOT_TYPES:
        assert not bool(aidx.type_mask[TYPE_TO_IDX[name]])


def test_building_slot_legal_with_buildings():
    obs = _Obs(
        units=[_U(actor_id=1, type="1tnk"), _U(actor_id=2, type="harv", can_attack=False)],
        buildings=[_B("proc", aid=50), _B("powr", aid=51), _B("weap", aid=52)],
        harv_count=1,
    )
    aidx = _aidx(obs)
    assert aidx.building_ids == [50, 51, 52]
    assert bool(aidx.building_valid[0]) and bool(aidx.building_valid[2])
    for name in BUILDING_SLOT_TYPES:
        assert bool(aidx.type_mask[TYPE_TO_IDX[name]]), name


def test_sell_repair_power_primary_use_building_slot():
    obs = _Obs(
        units=[_U(actor_id=1, type="1tnk")],
        buildings=[_B("proc", aid=50), _B("powr", aid=51)],
        harv_count=1,
    )
    aidx = _aidx(obs)
    for tname, at in (
        ("sell", ActionType.SELL),
        ("repair", ActionType.REPAIR),
        ("power_down", ActionType.POWER_DOWN),
        ("set_primary", ActionType.SET_PRIMARY),
    ):
        action, eff = index_to_command_effective(
            obs, TYPE_TO_IDX[tname], 1, 0, 0, aidx)
        assert len(action.commands) == 1
        cmd = action.commands[0]
        assert cmd.action == at
        assert cmd.actor_id == 51  # building slot 1
        assert ACTION_TYPES[eff[0]] == tname
        assert eff[1] == 1


def test_set_rally_point_uses_cell_and_building():
    obs = _Obs(
        units=[_U(actor_id=1, type="1tnk")],
        buildings=[_B("weap", aid=77)],
        harv_count=1,
    )
    aidx = _aidx(obs)
    cx, cy = 12, 8
    cell = cy * aidx.w + cx
    action, eff = index_to_command_effective(
        obs, TYPE_TO_IDX["set_rally_point"], 0, cell, 0, aidx)
    cmd = action.commands[0]
    assert cmd.action == ActionType.SET_RALLY_POINT
    assert cmd.actor_id == 77
    assert cmd.target_x == cx and cmd.target_y == cy


def test_building_oob_slot_falls_back_to_first():
    obs = _Obs(
        units=[_U(actor_id=1, type="1tnk")],
        buildings=[_B("proc", aid=50)],
        harv_count=1,
    )
    aidx = _aidx(obs)
    action = index_to_command(
        obs, TYPE_TO_IDX["sell"], 99, 0, 0, aidx)
    assert action.commands[0].actor_id == 50


def test_land_path_macros_still_work():
    """Enabling building types must not break vehicle/army macros."""
    units = [
        _U(actor_id=i, type="1tnk") for i in range(1, 14)
    ] + [_U(actor_id=20, type="harv", can_attack=False)]
    obs = _Obs(units=units, buildings=[_B("proc", aid=50)], harv_count=1)
    aidx = _aidx(obs)
    assert bool(aidx.type_mask[TYPE_TO_IDX["army_attack_move"]])
    assert bool(aidx.type_mask[TYPE_TO_IDX["vehicle_attack_move"]])
    cx, cy = 10, 11
    cell = cy * aidx.w + cx
    army = index_to_command(
        obs, TYPE_TO_IDX["army_attack_move"], 0, cell, 0, aidx)
    assert army.commands[0].action == ActionType.ARMY_ATTACK_MOVE


def test_partial_load_unchanged_n_types():
    """P1 reuses existing ACTION_TYPES rows — no type-head growth."""
    net = AlphaLiteNet()
    raw = {k: v.clone() for k, v in net.state_dict().items()}
    adapted = adapt_v2_state_dict(net, raw)
    assert adapted["head_type.weight"].shape[0] == N_ACTION_TYPES


def test_slot_legal_switches_to_building_valid():
    """With 1 unit and 3 buildings, sell must legalize 3 slots not 1."""
    obs = _Obs(
        units=[_U(actor_id=1, type="1tnk")],
        buildings=[_B("proc", aid=50), _B("powr", aid=51), _B("weap", aid=52)],
        harv_count=1,
    )
    aidx = _aidx(obs)
    B, U = 1, MAX_UNITS
    own = torch.zeros(B, U, dtype=torch.bool)
    own[0, 0] = True
    bv = aidx.building_valid.unsqueeze(0)
    t_sell = torch.tensor([TYPE_TO_IDX["sell"]])
    t_move = torch.tensor([TYPE_TO_IDX["move"]])
    legal_sell = _slot_legal_for_types(t_sell, own, bv)
    legal_move = _slot_legal_for_types(t_move, own, bv)
    assert int(legal_sell[0].sum()) == 3
    assert bool(legal_sell[0, 2])
    assert int(legal_move[0].sum()) == 1
    assert bool(legal_move[0, 0]) and not bool(legal_move[0, 1])


def test_act_samples_building_slot_beyond_unit_own():
    """Force-mask types to sell; sampled unit_slot must be in building_valid."""
    torch.manual_seed(0)
    obs = _Obs(
        units=[_U(actor_id=1, type="1tnk")],
        buildings=[_B("proc", aid=50), _B("powr", aid=51), _B("weap", aid=52),
                   _B("barr", aid=53)],
        harv_count=1,
    )
    aidx = _aidx(obs)
    batch = _mini_batch(aidx)
    # Only sell legal
    tm = torch.zeros_like(batch["type_mask"])
    tm[0, TYPE_TO_IDX["sell"]] = True
    batch["type_mask"] = tm
    net = AlphaLiteNet()
    net.eval()
    h = torch.zeros(1, HIDDEN_DIM)
    seen = set()
    for _ in range(40):
        out = net.act(batch, h, temperature=1.0)
        assert int(out["type"]) == TYPE_TO_IDX["sell"]
        slot = int(out["unit_slot"])
        assert slot < len(aidx.building_ids), slot
        assert bool(aidx.building_valid[slot])
        seen.add(slot)
        action, eff = index_to_command_effective(
            obs, int(out["type"]), slot, int(out["cell_flat"]),
            int(out["item_slot"]), aidx)
        assert action.commands[0].action == ActionType.SELL
        assert action.commands[0].actor_id in (50, 51, 52, 53)
        assert eff[1] == slot
    # With 4 buildings and only 1 own unit, must not be stuck on slot 0 only
    assert len(seen) >= 2, seen


def test_command_to_indices_building_actor():
    obs = _Obs(
        units=[_U(actor_id=1, type="1tnk")],
        buildings=[_B("proc", aid=50), _B("powr", aid=51)],
        harv_count=1,
    )
    aidx = _aidx(obs)
    cmd = CommandModel(action=ActionType.REPAIR, actor_id=51)
    t, u, c, i = command_to_indices(obs, cmd, aidx)
    assert ACTION_TYPES[t] == "repair"
    assert u == 1
    cmd2 = CommandModel(action=ActionType.SET_RALLY_POINT, actor_id=50,
                        target_x=3, target_y=4)
    t2, u2, c2, _ = command_to_indices(obs, cmd2, aidx)
    assert ACTION_TYPES[t2] == "set_rally_point"
    assert u2 == 0
    assert c2 == 4 * aidx.w + 3


def test_guard_still_disabled_in_enabled_types():
    """Guard is documented as P2 — must NOT sneak into ENABLED_TYPES here."""
    assert "guard" in ACTION_TYPES
    assert "guard" not in ENABLED_TYPES


if __name__ == "__main__":
    test_building_types_enabled_append_only()
    test_building_slot_masked_without_buildings()
    test_building_slot_legal_with_buildings()
    test_sell_repair_power_primary_use_building_slot()
    test_set_rally_point_uses_cell_and_building()
    test_building_oob_slot_falls_back_to_first()
    test_land_path_macros_still_work()
    test_partial_load_unchanged_n_types()
    test_slot_legal_switches_to_building_valid()
    test_act_samples_building_slot_beyond_unit_own()
    test_command_to_indices_building_actor()
    test_guard_still_disabled_in_enabled_types()
    print("OK ra-completo-p1 usable tests")
