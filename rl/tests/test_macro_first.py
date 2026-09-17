# -*- coding: utf-8 -*-
"""Macro-first curriculum: tech items, PBRS, MAB, spatial extras."""
from types import SimpleNamespace as NS

import numpy as np
import torch

from openra_env.models import ActionType
from rl.action_adapter import (
    ActionIndex, Vocab, should_save_for_weap, WEAP_SAVE_CASH,
)
from rl.auto_support import SUPPORT_WEAP_CASH, support_commands
from rl.network import AlphaLiteNet, HIDDEN_DIM, N_ACTION_TYPES
from rl.obs_encoding import (
    SCALAR_DIM, SPATIAL_EXTRA_CH, SPATIAL_TOTAL_CH,
    augment_spatial, scalar_features,
)
from rl.pfsp import BotPFSP
from rl.reward_shaping import ShapedReward, tech_tier_level
from rl.roles import IDENTITY_ITEMS, TECH_ITEMS, cheapest_of, sample_of
from rl.scripted_teacher import ScriptedTeacher


class _B:
    def __init__(self, typ="proc", x=5, y=5, aid=99):
        self.type = typ
        self.cell_x = x
        self.cell_y = y
        self.actor_id = aid
        self.hp_percent = 1.0
        self.is_repairing = False
        self.is_powered = True
        self.rally_x = -1
        self.rally_y = -1
        self.can_produce = []
        self.power_amount = 0
        self.sell_value = 0
        self.is_producing = False


class _U:
    def __init__(self, typ="e1", aid=1, x=8, y=8):
        self.type = typ
        self.actor_id = aid
        self.cell_x = x
        self.cell_y = y
        self.hp_percent = 1.0
        self.can_attack = typ not in ("harv", "mcv")
        self.is_idle = True
        self.speed = 50
        self.attack_range = 0
        self.experience_level = 0
        self.stance = 0
        self.facing = 0


class _Mil:
    def __init__(self, **kw):
        self.kills_cost = kw.get("kills_cost", 0)
        self.deaths_cost = kw.get("deaths_cost", 0)
        self.assets_value = kw.get("assets_value", 0)
        self.units_killed = 0
        self.units_lost = 0
        self.army_value = 0
        self.active_unit_count = 0
        self.buildings_killed = 0


def _obs(**kw):
    bldgs = kw.get("bldgs", ("fact", "proc", "powr", "tent"))
    units = kw.get("units") or [_U("harv", 1), _U("e1", 2)]
    return NS(
        tick=kw.get("tick", 1000),
        map_info=NS(height=32, width=32, map_name="doughnut"),
        economy=NS(
            cash=kw.get("cash", 2500), ore=kw.get("ore", 0),
            harvester_count=1, power_provided=200, power_drained=50,
            resource_capacity=2000),
        military=_Mil(),
        buildings=[_B(t, aid=100 + i) for i, t in enumerate(bldgs)],
        units=list(units),
        production=list(kw.get("prod", ())),
        available_production=list(kw.get(
            "avail", ("e1", "e3", "1tnk", "2tnk", "weap", "proc", "powr", "harv"))),
        visible_enemies=list(kw.get("enemies", ())),
        visible_enemy_buildings=list(kw.get("ene_bldgs", ())),
        spatial_map="",
        spatial_channels=9,
        result="",
    )


def test_tech_items_are_identity():
    for it in ("1tnk", "2tnk", "e3"):
        assert it in IDENTITY_ITEMS
        assert it in TECH_ITEMS
    assert cheapest_of(["1tnk", "2tnk"]) == "1tnk"
    assert sample_of(["1tnk", "2tnk"], rng=__import__("random").Random(1)) in (
        "1tnk", "2tnk")


def test_adapter_exposes_1tnk_and_2tnk():
    obs = _obs(bldgs=("fact", "proc", "powr", "tent", "weap"), cash=500)
    v = Vocab()
    v.seed_roles()
    aidx = ActionIndex(obs, v)
    assert "1tnk" in aidx.train_items
    assert "2tnk" in aidx.train_items
    assert "e3" in aidx.train_items


def test_weap_save_masks_cheap_infantry():
    obs = _obs(cash=1500, ore=0)
    assert should_save_for_weap(obs) is True
    assert 1500 >= WEAP_SAVE_CASH
    v = Vocab()
    v.seed_roles()
    aidx = ActionIndex(obs, v)
    if "infantry_basic" in aidx.train_items:
        slot = aidx.train_items.index("infantry_basic")
        assert bool(aidx.train_slot_mask[slot]) is False


def test_auto_support_builds_weap_at_2500():
    obs = _obs(cash=SUPPORT_WEAP_CASH,
               bldgs=("fact", "proc", "proc", "powr", "tent"))
    cmds = support_commands(obs, war_nudge=False)
    assert any(c.action == ActionType.BUILD and c.item_type == "weap"
               for c in cmds)


def test_teacher_default_expand_no_hardcoded_mines():
    t = ScriptedTeacher()
    assert t.mode == "expand"
    assert not hasattr(t, "A_SHORT_MINES")


def test_eradicate_v5_symmetric_combat_punishes_suicide_wave():
    sh = ShapedReward(preset="eradicate_v5")
    assert float(sh.w_exchange) == 0.0
    obs = _obs()
    sh.reset(obs)
    obs.military = _Mil(kills_cost=1000, deaths_cost=3000)
    r = sh.step(obs, done=False)
    # (0.15*1000 - 0.15*3000) / 1000 = -0.3
    assert sh.last_components["combat"] < -0.2
    assert sh.last_components["exchange"] == 0.0
    assert r < 0.0


def test_eradicate_v5_pbrs_pays_weap_tier():
    sh = ShapedReward(preset="eradicate_v5")
    obs = _obs(bldgs=("fact", "proc", "tent"))
    sh.reset(obs)
    assert tech_tier_level(obs) == 1
    obs2 = _obs(bldgs=("fact", "proc", "tent", "weap"))
    r = sh.step(obs2, done=False)
    assert sh.last_components["tier"] > 0.0
    assert r > 0.0


def test_mab_crushed_bot_hits_floor():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        league = BotPFSP(
            Path(td), anchor="beginner",
            pool=["beginner", "easy", "medium"],
            mab=True, mab_tau=0.25, mab_floor=0.05,
            rng=__import__("random").Random(0))
        for _ in range(40):
            league.record("beginner", "win")
        for _ in range(10):
            league.record("easy", "lose")
        dist = league.mab_probs()
        assert dist["beginner"] >= 0.05
        assert dist["easy"] > dist["beginner"]
        samples = [league.sample() for _ in range(200)]
        assert "easy" in samples


def test_augment_spatial_adds_typed_channels():
    obs = _obs()
    base = np.zeros((9, 32, 32), dtype=np.float32)
    out = augment_spatial(base, obs)
    assert out.shape[0] == SPATIAL_TOTAL_CH
    assert SPATIAL_EXTRA_CH == 11
    # own prod (fact/proc/tent) painted on extra ch 0
    assert float(out[9].max()) > 0.0


def test_scalar_dim_41_weap_save():
    obs = _obs(cash=2000)
    sc = scalar_features(obs)
    assert sc.shape == (SCALAR_DIM,)
    assert SCALAR_DIM == 41
    assert abs(float(sc[34]) - 1.0) < 1e-5  # cash/2000
    assert float(sc[40]) == 0.0  # no weap


def test_net_accepts_9ch_and_20ch():
    net = AlphaLiteNet()
    B, H, W = 1, 16, 16
    h = torch.zeros(B, HIDDEN_DIM)
    def _batch(c):
        return {
            "spatial": torch.zeros(B, c, H, W),
            "scalars": torch.zeros(B, SCALAR_DIM),
            "unit_feats": torch.zeros(B, 128, 14),
            "unit_valid": torch.zeros(B, 128, dtype=torch.bool),
            "unit_role_ids": torch.zeros(B, 128, dtype=torch.long),
            "unit_own_mask": torch.zeros(B, 128, dtype=torch.bool),
            "type_mask": torch.ones(B, N_ACTION_TYPES, dtype=torch.bool),
            "cell_mask": torch.ones(B, H * W, dtype=torch.bool),
            "item_indices": torch.zeros(B, 8, dtype=torch.long),
            "item_mask": torch.zeros(B, 8, dtype=torch.bool),
        }
    out9 = net.act(_batch(9), h)
    assert "log_prob_macro" in out9 and "log_prob_micro" in out9
    out20 = net.act(_batch(20), h)
    assert out20["log_prob"].shape == (B,)
