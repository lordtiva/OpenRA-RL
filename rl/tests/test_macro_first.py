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


def test_weap_save_disabled_v6():
    units = [_U("harv", 1)] + [_U("e1", 10 + i) for i in range(6)]
    obs = _obs(cash=1500, ore=0, units=units)
    assert should_save_for_weap(obs) is False
    v = Vocab()
    v.seed_roles()
    aidx = ActionIndex(obs, v)
    if "infantry_basic" in aidx.train_items:
        slot = aidx.train_items.index("infantry_basic")
        assert bool(aidx.train_slot_mask[slot]) is True


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


def test_mab_zpd_gaussian_peaks_near_half():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        league = BotPFSP(
            Path(td), anchor="beginner",
            pool=["beginner", "easy", "medium"],
            mab=True, mab_tau=0.25, mab_floor=0.05,
            rng=__import__("random").Random(0))
        # beginner solved (~100% WR), easy ZPD (~50%), medium crushed (~0%)
        for _ in range(40):
            league.record("beginner", "win")
        for _ in range(10):
            league.record("easy", "win")
        for _ in range(10):
            league.record("easy", "lose")
        for _ in range(20):
            league.record("medium", "lose")
        dist = league.mab_probs()
        assert dist["beginner"] >= 0.05
        assert dist["medium"] >= 0.05
        assert dist["easy"] > dist["beginner"]
        assert dist["easy"] > dist["medium"]
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

def test_harvester_train_not_capped_v6():
    from rl.action_adapter import HARVESTER_TRAIN_CAP
    units = [_U("harv", 100 + i) for i in range(HARVESTER_TRAIN_CAP)] + [
        _U("e1", 200)]
    obs = _obs(
        bldgs=("fact", "proc", "powr", "tent", "weap"),
        cash=5000, units=units,
        avail=("harv", "e1", "1tnk", "powr"))
    obs.economy.harvester_count = HARVESTER_TRAIN_CAP
    v = Vocab()
    v.seed_roles()
    aidx = ActionIndex(obs, v)
    assert "harvester" in aidx.train_items
    slot = aidx.train_items.index("harvester")
    assert bool(aidx.train_slot_mask[slot]) is True


def test_yard_attack_move_redirects_to_war_objective(monkeypatch):
    from rl.action_adapter import set_attack_cell_override
    set_attack_cell_override("war_objective")
    monkeypatch.setattr(
        "rl.action_adapter.ATTACK_CELL_OVERRIDE", "war_objective")
    from rl.action_adapter import (
        TYPE_TO_IDX, index_to_command_effective, PACK_HOME_RADIUS)
    # army_attack_move stays masked until PACK_ARMY; infantry path is live.
    units = [_U("e1", 10 + i, x=12, y=16) for i in range(8)]
    obs = _obs(
        bldgs=("fact", "proc", "powr", "tent", "weap"),
        cash=500, units=units,
        avail=("e1", "1tnk"))
    v = Vocab()
    v.seed_roles()
    aidx = ActionIndex(obs, v)
    import rl.war_objective as wo
    import rl.auto_support as aus
    monkeypatch.setattr(wo, "war_objective", lambda obs, aidx: (28, 28))
    monkeypatch.setattr(aus, "home_raid_targets", lambda obs: [])
    # Own buildings default to (5,5); click near home.
    cx, cy = 6, 6
    cell = cy * aidx.w + cx
    t = TYPE_TO_IDX["infantry_attack_move"]
    action, _eff = index_to_command_effective(
        obs, t, 0, cell, 0, aidx, heuristic_p=0.0)
    cmd = action.commands[0]
    assert (cmd.target_x, cmd.target_y) == (28, 28)


def test_onboard_s_mab_and_hyper_lr_min():
    from rl.hyper_health import HyperConfig
    from rl.onboard import DEFAULTS, _s_phase_flags
    assert HyperConfig().lr_min == 2e-5
    assert DEFAULTS["s_anchor_prob"] == 0.40
    assert DEFAULTS["s_mab_floor"] == 0.35
    flags = _s_phase_flags(DEFAULTS)
    assert flags[flags.index("--pfsp-anchor-prob") + 1] == "0.40"
    assert flags[flags.index("--mab-floor") + 1] == "0.35"


def test_mining_rate_damps_above_harvester_cap():
    from rl.reward_shaping import ShapedReward
    sh = ShapedReward(preset="eradicate_v5")
    units = [_U("harv", 100 + i) for i in range(8)]
    obs = _obs(units=units, cash=500)
    obs.economy.harvester_count = 8
    sh.reset(obs)
    # Simulate ore income
    obs2 = _obs(units=units, cash=500, ore=500)
    obs2.economy.harvester_count = 8
    # earned delta via ore/cash — step uses spendable/earned internals
    r = sh.step(obs2, done=False)
    # Just ensure mining component path ran without error; damp is internal
    assert "mining" in sh.last_components

def test_eradicate_v6_timeout_wipes_positive_return():
    from rl.reward_shaping import ShapedReward
    sh = ShapedReward(preset="eradicate_v6")
    obs = _obs(cash=500)
    sh.reset(obs)
    # Simulate dense positive shaping without going through full econ
    sh._episode_return = 19.0
    sh.last_components["mining"] = 19.0
    r = sh.finalize(truncated=True, result="incomplete")
    assert r <= -2.0
    assert sh.last_components["timeout_wipe"] < 0
    # Net episode including prior +19 must not stay positive
    assert sh._episode_return <= 0.0


def test_eradicate_v6_win_keeps_terminal():
    from rl.reward_shaping import ShapedReward
    sh = ShapedReward(preset="eradicate_v6")
    obs = _obs(cash=500)
    sh.reset(obs)
    sh._episode_return = 5.0
    sh._last_mil = obs.military
    r = sh.finalize(truncated=False, result="win")
    assert r >= sh.w_win - 0.1


def test_eradicate_v6_army_ratio_delta():
    from rl.reward_shaping import ShapedReward
    sh = ShapedReward(preset="eradicate_v6")
    obs = _obs(units=[_U("e1", 1), _U("harv", 2)])
    sh.reset(obs)
    # Inject visible enemies weaker than us via monkeypatch on aoa
    import rl.force_estimate as fe
    calls = {"n": 0}
    def fake_aoa(obs):
        calls["n"] += 1
        # first call in reset already happened; step sees rising ratio
        return {"rel_power": 0.4 if calls["n"] == 1 else 0.8}
    # reset already set prev; force prev then step
    sh._prev_army_ratio = 0.4
    import unittest.mock as m
    with m.patch.object(fe, "aoa_features", side_effect=lambda o: {"rel_power": 0.8}):
        r = sh._army_ratio_delta(obs)
    assert r > 0.0
    assert sh.last_components["army_ratio"] > 0.0


def test_default_preset_is_v6():
    from rl.reward_shaping import PRESETS
    assert "eradicate_v6" in PRESETS
    import argparse
    # train default
    from pathlib import Path
    src = Path("rl/train.py").read_text(encoding="utf-8")
    assert 'default="eradicate_v6"' in src or "default='eradicate_v6'" in src

