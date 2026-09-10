"""Training corrections: SMDP gamma_eff, K=2 h advance, building_kind parity, topk LN."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from rl.network import (
    ACTION_TYPES, AlphaLiteNet, XF_DIM, HIDDEN_DIM, SCALAR_DIM,
    MAX_UNITS, UNIT_FEAT_DIM, N_ACTION_TYPES, MAX_BUILDINGS, BUILDING_FEAT_DIM,
    _building_kind_legal, _building_kind_legal_loop, COMBAT_PUSH_TYPES,
)
from rl.rollout import add_advantages, smdp_k_ref, STEP_TICKS


def test_smdp_k_ref_defaults():
    assert smdp_k_ref(macro_ticks=50, k_skip=8) == 50.0
    assert smdp_k_ref(macro_ticks=0, k_skip=8) == float(STEP_TICKS * 8)
    assert smdp_k_ref(macro_ticks=0, k_skip=8, explicit=20) == 20.0
    assert smdp_k_ref(macro_ticks=50, k_skip=8, explicit=0) == 50.0  # 0 → auto


def test_smdp_gamma_eff_math():
    """gamma_eff = gamma ** (delta_t / k_ref); Δt=k_ref → gamma; Δt=0 → 1."""
    gamma, lam, k_ref = 0.99, 0.95, 16.0
    traj = [
        {"reward": 1.0, "value_pred": 0.0, "delta_t": 0.0, "k_ref": k_ref},
        {"reward": 0.0, "value_pred": 0.0, "delta_t": 16.0, "k_ref": k_ref},
        {"reward": 2.0, "value_pred": 0.0, "delta_t": 8.0, "k_ref": k_ref},
    ]
    add_advantages(traj, gamma=gamma, lam=lam, k_ref=k_ref, last_value=0.0)
    assert abs(traj[0]["gamma_eff"] - 1.0) < 1e-12
    assert abs(traj[1]["gamma_eff"] - gamma) < 1e-12
    assert abs(traj[2]["gamma_eff"] - (gamma ** (8.0 / 16.0))) < 1e-12
    # Shorter dt -> milder discount (higher gamma_eff) than nominal full step
    assert traj[2]["gamma_eff"] > traj[1]["gamma_eff"]
    assert math.isclose(traj[2]["gamma_eff"], gamma ** 0.5, rel_tol=0, abs_tol=1e-12)


def test_smdp_uniform_fallback_without_delta_t():
    gamma = 0.995
    traj = [
        {"reward": 1.0, "value_pred": 0.5},
        {"reward": 0.0, "value_pred": 0.4},
    ]
    add_advantages(traj, gamma=gamma, lam=0.95, last_value=0.0, k_ref=16.0)
    # missing delta_t → treated as k_ref → gamma_eff == gamma
    assert abs(traj[0]["gamma_eff"] - gamma) < 1e-12
    assert abs(traj[1]["gamma_eff"] - gamma) < 1e-12


def test_k2_micro_hidden_differs():
    torch.manual_seed(0)
    net = AlphaLiteNet()
    net.eval()
    B, H, W = 1, 16, 16
    batch = {
        "spatial": torch.zeros(B, 9, H, W),
        "scalars": torch.zeros(B, SCALAR_DIM),
        "unit_feats": torch.zeros(B, MAX_UNITS, UNIT_FEAT_DIM),
        "unit_valid": torch.zeros(B, MAX_UNITS, dtype=torch.bool),
        "unit_role_ids": torch.zeros(B, MAX_UNITS, dtype=torch.long),
        "unit_own_mask": torch.zeros(B, MAX_UNITS, dtype=torch.bool),
        "type_mask": torch.ones(B, N_ACTION_TYPES, dtype=torch.bool),
        "cell_mask": torch.ones(B, H * W, dtype=torch.bool),
        "item_indices": torch.zeros(B, 8, dtype=torch.long),
        "item_mask": torch.zeros(B, 8, dtype=torch.bool),
    }
    batch["unit_valid"][:, 0] = True
    batch["unit_own_mask"][:, 0] = True
    h0 = torch.zeros(B, HIDDEN_DIM)
    out = net.act(batch, h0, temperature=0.0)
    h1 = out["hidden"]
    h_in2, h2 = net.k2_micro_hidden(out["_ctx"], h1)
    assert torch.equal(h_in2, h1)
    assert not torch.allclose(h2, h1), "second GRU step must change hidden"
    out2 = net.act_combat(batch, h2, temperature=0.0, ctx=out["_ctx"])
    assert out2 is not None
    assert ACTION_TYPES[int(out2["type"].item())] in COMBAT_PUSH_TYPES
    # Regression: traj bookkeeping — second h_in != first h_in
    assert not torch.equal(h_in2, h0)


def test_topk_mha_ln_present_and_near_identity():
    net = AlphaLiteNet()
    assert hasattr(net, "topk_mha_ln")
    assert isinstance(net.topk_mha_ln, nn.LayerNorm)
    assert net.topk_mha_ln.normalized_shape == (XF_DIM,)
    # Default LN ≈ identity (weight=1, bias=0)
    assert torch.allclose(net.topk_mha_ln.weight, torch.ones(XF_DIM))
    assert torch.allclose(net.topk_mha_ln.bias, torch.zeros(XF_DIM))
    # Soft-load: missing key leaves identity — simulate load without LN key
    sd = {k: v for k, v in net.state_dict().items() if not k.startswith("topk_mha_ln")}
    net2 = AlphaLiteNet()
    # perturb LN then soft-load old dict (missing LN keys)
    with torch.no_grad():
        net2.topk_mha_ln.weight.fill_(0.5)
        net2.topk_mha_ln.bias.fill_(0.1)
    incompat = net2.load_state_dict(sd, strict=False)
    assert any(k.startswith("topk_mha_ln") for k in incompat.missing_keys)
    # After soft-load of sd without LN, the perturbed LN stays (missing keys
    # keep current params). Fresh net soft-load from old ckpt keeps identity
    # because __init__ set identity before load. Document that contract:
    net3 = AlphaLiteNet()
    incompat3 = net3.load_state_dict(sd, strict=False)
    assert any("topk_mha_ln" in k for k in incompat3.missing_keys)
    assert torch.allclose(net3.topk_mha_ln.weight, torch.ones(XF_DIM))
    assert torch.allclose(net3.topk_mha_ln.bias, torch.zeros(XF_DIM))
    # Forward with xf_topk uses LN path without crash
    net3.xf_topk = 4
    B, H, W = 1, 16, 16
    batch = {
        "spatial": torch.zeros(B, 9, H, W),
        "scalars": torch.zeros(B, SCALAR_DIM),
        "unit_feats": torch.zeros(B, MAX_UNITS, UNIT_FEAT_DIM),
        "unit_valid": torch.zeros(B, MAX_UNITS, dtype=torch.bool),
        "unit_role_ids": torch.zeros(B, MAX_UNITS, dtype=torch.long),
        "unit_own_mask": torch.zeros(B, MAX_UNITS, dtype=torch.bool),
        "type_mask": torch.ones(B, N_ACTION_TYPES, dtype=torch.bool),
        "cell_mask": torch.ones(B, H * W, dtype=torch.bool),
        "item_indices": torch.zeros(B, 8, dtype=torch.long),
        "item_mask": torch.zeros(B, 8, dtype=torch.bool),
    }
    batch["unit_valid"][:, 0] = True
    batch["unit_own_mask"][:, 0] = True
    h = torch.zeros(B, HIDDEN_DIM)
    out = net3.act(batch, h, temperature=0.0)
    assert out["hidden"].shape == (B, HIDDEN_DIM)


def test_building_kind_legal_parity():
    torch.manual_seed(3)
    B, Nb = 8, MAX_BUILDINGS
    bv = torch.zeros(B, Nb, dtype=torch.bool)
    bv[:, :5] = True
    feats = torch.zeros(B, Nb, BUILDING_FEAT_DIM)
    # feat5 can_produce, feat6 power, feat7 sellable
    feats[:, 0, 7] = 1.0  # sellable
    feats[:, 1, 5] = 1.0  # can_prod
    feats[:, 2, 6] = 0.5  # power
    feats[:, 3, 7] = 1.0
    feats[:, 3, 5] = 1.0
    types = ["sell", "set_rally_point", "set_primary", "power_down",
             "repair", "move", "sell", "power_down"]
    t_idx = torch.tensor([ACTION_TYPES.index(n) for n in types])
    # One row where sell filter empties: no sellable among valid
    bv2 = bv.clone()
    feats2 = feats.clone()
    bv2[6, :] = False
    bv2[6, 4] = True  # only slot 4 valid, not sellable
    new = _building_kind_legal(t_idx, bv2, feats2)
    old = _building_kind_legal_loop(t_idx, bv2, feats2)
    assert torch.equal(new, old), (new ^ old).nonzero()[:10]
    # Broader random parity
    for _ in range(20):
        bv_r = torch.randint(0, 2, (B, Nb), dtype=torch.bool)
        feats_r = torch.randn(B, Nb, BUILDING_FEAT_DIM)
        t_r = torch.randint(0, len(ACTION_TYPES), (B,))
        assert torch.equal(
            _building_kind_legal(t_r, bv_r, feats_r),
            _building_kind_legal_loop(t_r, bv_r, feats_r),
        )


if __name__ == "__main__":
    test_smdp_k_ref_defaults()
    test_smdp_gamma_eff_math()
    test_smdp_uniform_fallback_without_delta_t()
    test_k2_micro_hidden_differs()
    test_topk_mha_ln_present_and_near_identity()
    test_building_kind_legal_parity()
    print("OK training corrections tests")
