# -*- coding: utf-8 -*-
"""Smoke tests AlphaLiteNet v2 full pack (fog ghosts + group macros + dims)."""
import numpy as np
import torch

from rl.obs_encoding import (
    UNIT_FEAT_DIM, EnemyBeliefStore, unit_tokens, GHOST_TTL, MAX_TOKENS,
    SCALAR_DIM,
)
from rl.network import (
    COMBAT_PUSH_TYPES,
    build_combat_type_mask,
    AlphaLiteNet, ACTION_TYPES, HIDDEN_DIM, SPATIAL_CH, SPATIAL_MID,
    TYPE_TO_IDX, TYPES_USE_CELL, N_ACTION_TYPES,
)
from rl.action_adapter import (
    group_actor_ids, ENABLED_TYPES, MOVE_CELL_TYPES,
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


class _Obs:
    def __init__(self, units=None, enemies=None):
        self.units = units or []
        self.visible_enemies = enemies or []
        self.visible_enemy_buildings = []
        self.buildings = []
        self.production = []
        self.available_production = []
        self.economy = type("E", (), {
            "cash": 0, "ore": 0, "resource_capacity": 1,
            "power_provided": 1, "power_drained": 0, "harvester_count": 0,
        })()
        self.military = type("M", (), {
            "units_killed": 0, "units_lost": 0, "buildings_killed": 0,
            "buildings_lost": 0, "army_value": 0, "active_unit_count": 0,
            "kills_cost": 0, "deaths_cost": 0, "assets_value": 0,
        })()
        self.map_info = type("MI", (), {"height": 64, "width": 64, "map_name": "t"})()
        self.spatial_map = ""
        self.spatial_channels = 9
        self.tick = 1


def test_unit_feat_dim():
    assert UNIT_FEAT_DIM == 14


def test_belief_ghost_persists():
    bel = EnemyBeliefStore()
    e = _U(actor_id=7, type="e1", cell_x=20, cell_y=30, hp_percent=0.6)
    obs_vis = _Obs(enemies=[e])
    feats, roles, valid, own = unit_tokens(obs_vis, belief=bel)
    assert valid[96]  # first enemy slot
    assert feats[96, 11] == 1.0  # visible
    assert feats[96, 12] == 1.0  # conf
    # disappear into fog
    obs_fog = _Obs(enemies=[])
    feats2, _, valid2, _ = unit_tokens(obs_fog, belief=bel)
    assert valid2[96]
    assert feats2[96, 11] == 0.0  # not visible
    assert feats2[96, 12] < 1.0
    assert feats2[96, 7] == 20 / 128.0  # last_seen x
    assert feats2[96, 8] == 30 / 128.0
    # TTL drop
    for _ in range(GHOST_TTL + 2):
        bel.update(obs_fog)
    feats3, _, valid3, _ = unit_tokens(obs_fog, belief=bel)
    assert not valid3[96]


def test_group_helpers():
    units = [
        _U(actor_id=1, type="e1"),
        _U(actor_id=2, type="e3"),
        _U(actor_id=3, type="1tnk"),
        _U(actor_id=4, type="harv", can_attack=False),
        _U(actor_id=5, type="dd"),
        _U(actor_id=6, type="heli"),
    ]
    obs = _Obs(units=units)
    assert group_actor_ids(obs, "infantry") == [1, 2]
    assert group_actor_ids(obs, "vehicle") == [3]  # land-only; dd/heli excluded
    assert group_actor_ids(obs, "harvesters") == [4]
    assert group_actor_ids(obs, "naval") == [5]
    assert group_actor_ids(obs, "air") == [6]
    assert "infantry_attack_move" in ENABLED_TYPES
    assert "infantry_attack_move" in MOVE_CELL_TYPES
    assert "infantry_attack_move" in TYPES_USE_CELL
    assert TYPE_TO_IDX["support_power"] == N_ACTION_TYPES - 1
    assert TYPE_TO_IDX["patrol"] == N_ACTION_TYPES - 2
    assert TYPE_TO_IDX["army_guard"] == N_ACTION_TYPES - 3
    assert TYPE_TO_IDX["air_attack_move"] == N_ACTION_TYPES - 6
    assert TYPE_TO_IDX["naval_attack_move"] == N_ACTION_TYPES - 7
    assert "naval_attack_move" in ENABLED_TYPES
    assert "air_attack_move" in TYPES_USE_CELL


def test_net_forward_smoke():
    net = AlphaLiteNet()
    B, H, W = 2, 32, 32
    batch = {
        "spatial": torch.zeros(B, 9, H, W),
        "scalars": torch.zeros(B, SCALAR_DIM),
        "unit_feats": torch.zeros(B, MAX_TOKENS, UNIT_FEAT_DIM),
        "unit_valid": torch.zeros(B, MAX_TOKENS, dtype=torch.bool),
        "unit_role_ids": torch.zeros(B, MAX_TOKENS, dtype=torch.long),
        "unit_own_mask": torch.zeros(B, MAX_TOKENS, dtype=torch.bool),
        "type_mask": torch.ones(B, N_ACTION_TYPES, dtype=torch.bool),
        "cell_mask": torch.ones(B, H * W, dtype=torch.bool),
        "item_indices": torch.zeros(B, 8, dtype=torch.long),
        "item_mask": torch.zeros(B, 8, dtype=torch.bool),
    }
    batch["unit_valid"][:, 0] = True
    batch["unit_own_mask"][:, 0] = True
    h = torch.zeros(B, HIDDEN_DIM)
    out = net.act(batch, h)
    assert out["type"].shape == (B,)
    assert out["log_prob"].shape == (B,)
    fmap, _, h2, tokens = net.encode(
        batch["spatial"], batch["scalars"], batch["unit_feats"],
        batch["unit_valid"], h,
        unit_role_ids=batch["unit_role_ids"],
        unit_own_mask=batch["unit_own_mask"])
    assert fmap.shape[1] == SPATIAL_CH
    assert SPATIAL_MID == 64
    assert tokens.shape[-1] == 128



def test_act_combat_mask():
    """Student K=2: act_combat restricts type to COMBAT_PUSH_TYPES."""
    net = AlphaLiteNet()
    B, H, W = 1, 16, 16
    batch = {
        "spatial": torch.zeros(B, 9, H, W),
        "scalars": torch.zeros(B, SCALAR_DIM),
        "unit_feats": torch.zeros(B, MAX_TOKENS, UNIT_FEAT_DIM),
        "unit_valid": torch.zeros(B, MAX_TOKENS, dtype=torch.bool),
        "unit_role_ids": torch.zeros(B, MAX_TOKENS, dtype=torch.long),
        "unit_own_mask": torch.zeros(B, MAX_TOKENS, dtype=torch.bool),
        "type_mask": torch.ones(B, N_ACTION_TYPES, dtype=torch.bool),
        "cell_mask": torch.ones(B, H * W, dtype=torch.bool),
        "item_indices": torch.zeros(B, 8, dtype=torch.long),
        "item_mask": torch.zeros(B, 8, dtype=torch.bool),
    }
    batch["unit_valid"][:, 0] = True
    batch["unit_own_mask"][:, 0] = True
    h = torch.zeros(B, HIDDEN_DIM)
    out = net.act(batch, h, temperature=0.0)
    assert "_ctx" in out
    assert len(out["_ctx"]) >= 7 and out["_ctx"][6] is not None  # fused
    # Alt-B: micro-step GRU then combat sample under advanced hidden
    h_in2, h2 = net.k2_micro_hidden(out["_ctx"], out["hidden"])
    assert torch.equal(h_in2, out["hidden"])
    assert not torch.equal(h2, out["hidden"]), "K=2 micro GRU must move hidden"
    out2 = net.act_combat(batch, h2, temperature=0.0, ctx=out["_ctx"])
    assert out2 is not None
    tname = ACTION_TYPES[int(out2["type"].item())]
    assert tname in COMBAT_PUSH_TYPES, tname
    assert torch.equal(out2["hidden"], h2)
    # Illegal combat -> None
    batch_off = dict(batch)
    batch_off["type_mask"] = torch.zeros(B, N_ACTION_TYPES, dtype=torch.bool)
    assert net.act_combat(batch_off, h) is None
    assert build_combat_type_mask(batch_off["type_mask"]) is None


if __name__ == "__main__":
    test_unit_feat_dim()
    test_belief_ghost_persists()
    test_group_helpers()
    test_net_forward_smoke()
    test_act_combat_mask()
    print("OK all v2 smoke tests")
