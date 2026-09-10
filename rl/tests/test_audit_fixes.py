"""Tests for fixes identified in DeepSeek & Gemini technical audits.

Validates:
1. metrics_lock contextmanager generator exception handling.
2. reward_shaping w_refinery_early bonus in v3/v4 presets.
3. trainer save_checkpoint atomic file write.
4. network _scores_building fp16 safety and _heads_used device caching.
5. network _unit_cond_vector parity with _unit_cond_map.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from types import SimpleNamespace

import pytest
import torch

from rl.metrics_lock import metrics_lock
from rl.network import (
    AlphaLiteNet, _heads_used, _tables_for,
    _ILLEGAL_FP16, _ILLEGAL_FP32, MAX_BUILDINGS, HIDDEN_DIM,
)
from rl.reward_shaping import ShapedReward, PRESETS
from rl.trainer import save_checkpoint, load_checkpoint


def check(name: str, cond: bool) -> None:
    print(f"  [{'OK' if cond else 'FALLA'}] {name}")
    assert cond, f"Fallo en assertion: {name}"


def test_metrics_lock_exception_inside_with():
    print("=== test_metrics_lock_exception_inside_with ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        metric_file = os.path.join(tmpdir, "test_metrics.jsonl")

        # 1. Normal usage
        with metrics_lock(metric_file, exclusive=True):
            with open(metric_file, "a", encoding="utf-8") as f:
                f.write('{"test": 1}\n')
        check("normal write completed", os.path.exists(metric_file))

        # 2. Exception inside with block must propagate cleanly (not RuntimeError)
        caught = False
        try:
            with metrics_lock(metric_file, exclusive=True):
                raise ValueError("custom_test_error")
        except ValueError as e:
            caught = (str(e) == "custom_test_error")
        except RuntimeError as re:
            # Bug 2.1 was RuntimeError: generator didn't stop after throw()
            caught = False

        check("exception inside with propagates cleanly without RuntimeError", caught)


def test_reward_shaping_early_refinery():
    print("=== test_reward_shaping_early_refinery ===")
    # Under eradicate_v4, w_refinery_early should be applied on the first proc placement
    shaper = ShapedReward(preset="eradicate_v4")

    # Initial state: only base buildings, no proc
    obs_init = SimpleNamespace(
        military=SimpleNamespace(kills_cost=0, deaths_cost=0, assets_value=1000, buildings_killed=0),
        buildings=[SimpleNamespace(type="fact", cell_x=10, cell_y=10)],
        units=[],
        economy=SimpleNamespace(harvester_count=0),
        tick=100,
        visible_enemies=[],
    )
    shaper.reset(obs_init)
    check("initial _refinery_paid is False", shaper._refinery_paid is False)

    # Step with new 'proc' building placed at tick=1000
    obs_step1 = SimpleNamespace(
        military=SimpleNamespace(kills_cost=0, deaths_cost=0, assets_value=3000, buildings_killed=0),
        buildings=[
            SimpleNamespace(type="fact", cell_x=10, cell_y=10),
            SimpleNamespace(type="proc", cell_x=12, cell_y=12),
        ],
        units=[SimpleNamespace(type="harv", cell_x=12, cell_y=13, can_attack=False)],
        economy=SimpleNamespace(harvester_count=1),
        tick=1000,
        visible_enemies=[],
    )
    gs = {"own": {"earned": 0}}
    r1 = shaper.step(obs_step1, False, gs=gs, closing=True)

    check("refinery marked paid", shaper._refinery_paid is True)
    check("early_refinery component > 0", shaper.last_components["early_refinery"] > 0.0)

    # Expected early bonus: 2.0 * (1.0 - 1000 / 6000) = 2.0 * (5/6) = 1.6667
    expected_early = shaper.w_refinery_early * (1.0 - 1000.0 / shaper.refinery_target_tick)
    check("early_refinery component matches formula",
          abs(shaper.last_components["early_refinery"] - expected_early) < 1e-4)

    # Next step: should NOT re-trigger early refinery
    obs_step2 = SimpleNamespace(
        military=SimpleNamespace(kills_cost=0, deaths_cost=0, assets_value=3000, buildings_killed=0),
        buildings=[
            SimpleNamespace(type="fact", cell_x=10, cell_y=10),
            SimpleNamespace(type="proc", cell_x=12, cell_y=12),
        ],
        units=[SimpleNamespace(type="harv", cell_x=12, cell_y=13, can_attack=False)],
        economy=SimpleNamespace(harvester_count=1),
        tick=1500,
        visible_enemies=[],
    )
    shaper.step(obs_step2, False, gs=gs, closing=True)
    check("early_refinery did not increase on subsequent steps",
          abs(shaper.last_components["early_refinery"] - expected_early) < 1e-4)


def test_save_checkpoint_atomic():
    print("=== test_save_checkpoint_atomic ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "subdir", "test_ckpt.pt")
        net = AlphaLiteNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        save_checkpoint(ckpt_path, net, opt, iteration=42, extra={"tag": "v1"})

        check("destination checkpoint exists", os.path.isfile(ckpt_path))
        check("temporary .tmp file was cleaned up", not os.path.exists(f"{ckpt_path}.tmp"))

        # Verify load works correctly
        it = load_checkpoint(ckpt_path, net, opt)
        check("loaded iteration matches", it == 42)


def test_network_scores_building_fp16():
    print("=== test_network_scores_building_fp16 ===")
    net = AlphaLiteNet()

    # Float32 hidden with None building_feats
    h32 = torch.randn(2, HIDDEN_DIM, dtype=torch.float32)
    t_idx = torch.tensor([0, 1])
    legal = torch.ones(2, MAX_BUILDINGS, dtype=torch.bool)

    s32 = net._scores_building(h32, t_idx, building_feats=None, building_legal=legal)
    check("fp32 scores_building returns _ILLEGAL_FP32", (s32 == _ILLEGAL_FP32).all().item())

    # Float16 hidden with None building_feats
    h16 = h32.half()
    s16 = net._scores_building(h16, t_idx, building_feats=None, building_legal=legal)
    check("fp16 scores_building returns _ILLEGAL_FP16 without inf/nan",
          (s16 == _ILLEGAL_FP16).all().item())
    check("fp16 scores_building has no inf or nan", not torch.isinf(s16).any() and not torch.isnan(s16).any())


def test_network_heads_used_caching():
    print("=== test_network_heads_used_caching ===")
    t = torch.tensor([0, 1, 2, 3])
    dev = torch.device("cpu")

    tabs1 = _tables_for(dev)
    tabs2 = _tables_for(dev)
    check("_tables_for returns cached tuple instance", tabs1 is tabs2)

    u, c, i = _heads_used(t, dev)
    check("heads_used returns correct types", u.dtype == torch.bool and c.dtype == torch.bool and i.dtype == torch.bool)


def test_network_unit_cond_vector_parity():
    print("=== test_network_unit_cond_vector_parity ===")
    net = AlphaLiteNet()
    B = 2
    tokens = torch.randn(B, 16, net.unit_cond_proj.in_features)
    unit_valid = torch.ones(B, 16, dtype=torch.bool)
    unit_slot = torch.tensor([3, 7])
    chosen_type = torch.tensor([1, 2])

    v = net._unit_cond_vector(tokens, unit_valid, unit_slot, chosen_type)
    m = net._unit_cond_map(tokens, unit_valid, unit_slot, chosen_type, hw=(12, 12))

    check("unit_cond_map matches unit_cond_vector broadcast across (H, W)",
          torch.allclose(m[:, :, 0, 0], v, atol=1e-6))


if __name__ == "__main__":
    test_metrics_lock_exception_inside_with()
    test_reward_shaping_early_refinery()
    test_save_checkpoint_atomic()
    test_network_scores_building_fp16()
    test_network_heads_used_caching()
    test_network_unit_cond_vector_parity()
    print("\nTODOS LOS TESTS DE AUDITORIA PASARON EXITOSAMENTE.")
