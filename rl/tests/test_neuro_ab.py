# -*- coding: utf-8 -*-
"""A/B measurements: Kenyon novelty (B) + RingGoalBias (A)."""
from types import SimpleNamespace as NS

import numpy as np
import torch

from rl.state_hash import novelty_pick, diversity_stats
from rl.imitation import _even_pick, _pick_steps, EliteBuffer
from rl.network import AlphaLiteNet, RingGoalBias, HIDDEN_DIM, SCALAR_DIM


def _fake_step(i: int, t: int = 2, scalars=None):
    sc = torch.zeros(SCALAR_DIM)
    if scalars is not None:
        sc = torch.tensor(scalars, dtype=torch.float32)
    else:
        sc[0] = (i % 7) * 0.1
        sc[25] = 1.0 if i % 3 == 0 else 0.0
        sc[26] = 0.01 * i
        sc[27] = -0.01 * i
    return {
        "batch": {"scalars": sc},
        "action": {"type": torch.tensor(t if i % 5 else 0),
                   "cell": torch.tensor(float(i * 10))},
    }


def test_novelty_pick_higher_unique_ratio_than_even():
    # Many near-duplicates + a few distinct combat frames
    steps = []
    for i in range(40):
        steps.append(_fake_step(i // 8, t=0))  # clumps of identical-ish
    for i in range(10):
        steps.append(_fake_step(100 + i * 17, t=2))
    even = _even_pick(steps, 16)
    nov = novelty_pick(steps, 16, min_hamming=8)
    d_even = diversity_stats(even)
    d_nov = diversity_stats(nov)
    assert d_nov["unique_ratio"] >= d_even["unique_ratio"] - 1e-6
    assert d_nov["n_unique_hash"] >= d_even["n_unique_hash"]


def test_elite_buffer_novelty_flag():
    buf = EliteBuffer(cap_steps=50, use_novelty=True)
    samples = [_fake_step(i) for i in range(30)]
    n = buf.add_episode(samples, {"result": "win", "ticks": 5000})
    assert n == 30
    out = buf.sample_recent(12)
    assert len(out) <= 12
    assert "unique_ratio" in buf.last_sample_stats
    assert buf.last_sample_stats.get("novelty") is True


def test_ring_goal_bias_shapes_and_stats():
    ring = RingGoalBias(n_ring=16, hidden_dim=HIDDEN_DIM)
    B, H, W = 2, 8, 8
    hidden = torch.zeros(B, HIDDEN_DIM)
    scalars = torch.zeros(B, SCALAR_DIM)
    scalars[:, 25] = 1.0
    scalars[:, 26] = 1.0
    scalars[:, 27] = 0.0
    bias = ring(hidden, scalars, (H, W))
    assert bias.shape == (B, H, W)
    assert ring.last_stats["ring_has_goal"] > 0.0
    assert ring.last_stats["ring_bias_peak"] > 0.0


def test_net_ring_in_cell_logits():
    net = AlphaLiteNet()
    net.use_ring_goal = True
    B, H, W = 1, 16, 16
    fmap = torch.zeros(B, 96, H, W)
    hidden = torch.zeros(B, HIDDEN_DIM)
    scalars = torch.zeros(B, SCALAR_DIM)
    scalars[0, 25] = 1.0
    scalars[0, 26] = 0.5
    scalars[0, 27] = 0.5
    tokens = torch.zeros(B, 128, 128)
    feats = torch.zeros(B, 128, 14)
    valid = torch.zeros(B, 128, dtype=torch.bool)
    valid[0, 0] = True
    cell_mask = torch.ones(B, H * W, dtype=torch.bool)
    t_idx = torch.zeros(B, dtype=torch.long)
    u_idx = torch.zeros(B, dtype=torch.long)
    logits = net._logits_cell(
        fmap, t_idx, cell_mask, hidden, tokens, feats, valid, u_idx,
        scalars=scalars)
    assert logits.shape == (B, H * W)
    assert net.ring_goal.last_stats["ring_has_goal"] > 0.0


def test_spatial_corr_snapshot_api():
    from rl.action_adapter import (
        reset_spatial_corr_stats, spatial_corr_snapshot, SPATIAL_CORR_STATS)
    reset_spatial_corr_stats()
    SPATIAL_CORR_STATS["n_attack_macro"] = 10
    SPATIAL_CORR_STATS["n_redirected"] = 4
    snap = spatial_corr_snapshot()
    assert abs(snap["redirect_rate"] - 0.4) < 1e-6

def test_unit_xf_scale_nonzero():
    net = AlphaLiteNet()
    assert float(net.unit_xf_scale.item()) > 0.0


def test_attack_override_off_keeps_cell(monkeypatch):
    from rl.action_adapter import (
        set_attack_cell_override, TYPE_TO_IDX, index_to_command_effective,
        ActionIndex, Vocab,
    )
    from rl.tests.test_macro_first import _obs, _U
    set_attack_cell_override("off")
    units = [_U("e1", 10 + i, x=12, y=16) for i in range(8)]
    obs = _obs(bldgs=("fact", "proc", "powr", "tent", "weap"),
               cash=500, units=units, avail=("e1", "1tnk"))
    v = Vocab(); v.seed_roles(); aidx = ActionIndex(obs, v)
    import rl.war_objective as wo
    import rl.auto_support as aus
    monkeypatch.setattr(wo, "war_objective", lambda obs, aidx: (28, 28))
    monkeypatch.setattr(aus, "home_raid_targets", lambda obs: [])
    cx, cy = 6, 6
    cell = cy * aidx.w + cx
    t = TYPE_TO_IDX["infantry_attack_move"]
    action, _eff = index_to_command_effective(
        obs, t, 0, cell, 0, aidx, heuristic_p=0.0)
    cmd = action.commands[0]
    # Must NOT snap to war_objective (28,28) when override off
    assert (cmd.target_x, cmd.target_y) != (28, 28)

