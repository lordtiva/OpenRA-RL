"""P0/P1 training fixes: seq-eval smoke, item_slot fallback, heuristic_p, metrics lock."""
from __future__ import annotations

import json
import os
import random
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

import pytest
import torch

from rl.metrics_lock import metrics_lock
from rl.network import AlphaLiteNet, HIDDEN_DIM, SCALAR_DIM, MAX_UNITS, UNIT_FEAT_DIM, N_ACTION_TYPES
from rl import action_adapter as aa


def _tiny_batch(B=1, H=16, W=16, device="cpu"):
    spatial = torch.zeros(B, 9, H, W, device=device)
    scalars = torch.zeros(B, SCALAR_DIM, device=device)
    unit_feats = torch.zeros(B, MAX_UNITS, UNIT_FEAT_DIM, device=device)
    unit_valid = torch.zeros(B, MAX_UNITS, dtype=torch.bool, device=device)
    unit_valid[:, 0] = True
    unit_role_ids = torch.zeros(B, MAX_UNITS, dtype=torch.long, device=device)
    unit_own_mask = unit_valid.clone()
    type_mask = torch.ones(B, N_ACTION_TYPES, dtype=torch.bool, device=device)
    cell_mask = torch.ones(B, H * W, dtype=torch.bool, device=device)
    item_indices = torch.zeros(B, 8, dtype=torch.long, device=device)
    item_mask = torch.zeros(B, 8, dtype=torch.bool, device=device)
    item_mask[:, 0] = True
    return {
        "spatial": spatial,
        "scalars": scalars,
        "unit_feats": unit_feats,
        "unit_valid": unit_valid,
        "unit_role_ids": unit_role_ids,
        "unit_own_mask": unit_own_mask,
        "type_mask": type_mask,
        "cell_mask": cell_mask,
        "item_indices": item_indices,
        "item_mask": item_mask,
    }


def _make_seg(net, T=3, device="cpu"):
    """Build a short BPTT segment with deterministic actions."""
    torch.manual_seed(0)
    seg = []
    h = torch.zeros(1, HIDDEN_DIM, device=device)
    for t in range(T):
        batch = _tiny_batch(1, device=device)
        actions = {
            "type": torch.tensor([0], device=device),  # no_op
            "unit_slot": torch.tensor([0], device=device),
            "cell_flat": torch.tensor([0], device=device),
            "item_slot": torch.tensor([0], device=device),
            "had_item": torch.tensor([True], device=device),
        }
        seg.append({
            "batch": {k: v.detach().cpu() for k, v in batch.items()},
            "action": {k: v.detach().cpu() for k, v in actions.items()},
            "h_in": h.detach().cpu(),
            "_burn": False,
        })
        # advance h like rollout would (encode)
        with torch.no_grad():
            _, _, h, _ = net.encode(
                batch["spatial"], batch["scalars"],
                batch["unit_feats"], batch["unit_valid"], h,
                unit_role_ids=batch["unit_role_ids"],
                unit_own_mask=batch["unit_own_mask"])
            h = h.detach()
    return seg


def test_encode_features_matches_encode_core():
    torch.manual_seed(1)
    net = AlphaLiteNet()
    net.eval()
    batch = _tiny_batch(2)
    h = torch.zeros(2, HIDDEN_DIM)
    fmap, tokens, fused = net.encode_features(
        batch["spatial"], batch["scalars"],
        batch["unit_feats"], batch["unit_valid"],
        unit_role_ids=batch["unit_role_ids"],
        unit_own_mask=batch["unit_own_mask"])
    fmap2, flat, h2, tokens2 = net.encode(
        batch["spatial"], batch["scalars"],
        batch["unit_feats"], batch["unit_valid"], h,
        unit_role_ids=batch["unit_role_ids"],
        unit_own_mask=batch["unit_own_mask"])
    assert torch.allclose(fmap, fmap2)
    assert torch.allclose(tokens, tokens2)
    h_from_fused = net.core(fused, h)
    assert torch.allclose(h_from_fused, h2)


def test_evaluate_actions_seq_batch_smoke():
    """Seq eval still runs (per-t encode; P0.1 BxT path reverted)."""
    torch.manual_seed(42)
    net = AlphaLiteNet()
    net.train()
    segs = [_make_seg(net, T=4), _make_seg(net, T=3)]
    lp, ent, val, valid = net.evaluate_actions_seq_batch(segs, "cpu")
    assert lp.shape == (2, 4)
    assert ent.shape == (2, 4)
    assert val.shape == (2, 4)
    assert valid.shape == (2, 4)
    assert valid.dtype == torch.bool
    # padded shorter seg: T=3 active, last pad invalid
    assert bool(valid[0].all())
    assert bool(valid[1, :3].all()) and not bool(valid[1, 3])
    assert torch.isfinite(lp[valid]).all()
    assert torch.isfinite(ent[valid]).all()
    assert torch.isfinite(val[valid]).all()
    # no legacy vectorized helper after revert
    assert not hasattr(net, "_evaluate_actions_seq_batch_legacy")


def test_item_slot_fallback_logs_and_counts(caplog):
    aa.drain_item_slot_fallback()  # reset
    class _A:
        items = ["power", "barracks"]
    with caplog.at_level("WARNING", logger="rl.action_adapter"):
        slot = aa._item_slot_of("totally_unknown_xyz", _A())
    assert slot == 0
    assert aa.drain_item_slot_fallback() == 1
    assert aa.drain_item_slot_fallback() == 0
    assert any("fallback" in r.message for r in caplog.records)
    assert any("totally_unknown_xyz" in r.message for r in caplog.records)


def test_heuristic_p_gates_stage_and_guard_seeded():
    """heuristic_p=0 skips guard only; remap+stage always called."""
    calls = {"remap": 0, "stage": 0, "guard": 0}

    def fake_remap(obs, aidx, cx, cy, actor_id=0):
        calls["remap"] += 1
        return cx, cy

    def fake_stage(obs, aidx, cx, cy):
        calls["stage"] += 1
        return cx + 1, cy

    def fake_guard(obs, aidx, cx, cy):
        calls["guard"] += 1
        return cx, cy + 1

    class _Aidx:
        h = 64
        w = 64
        unit_ids = [1]
        building_ids = []
        items = ["e1"]
        train_items = ["e1"]
        build_items = []
        rol_a_concreto = {}

    # Minimal obs stub
    class _Eco:
        cash = 0
        ore = 0
    class _Mil:
        assets_value = 0
    class _Obs:
        units = []
        production = []
        economy = _Eco()
        military = _Mil()
        visible_enemies = []
        visible_enemy_buildings = []

    from rl.network import TYPE_TO_IDX
    t = TYPE_TO_IDX["army_attack_move"]
    cell = 10 * 64 + 20

    with mock.patch.object(aa, "remap_move_cell", fake_remap), \
         mock.patch.object(aa, "stage_army_attack_cell", fake_stage), \
         mock.patch.object(aa, "guard_army_push_cell", fake_guard), \
         mock.patch.object(aa, "owns_proc", return_value=True), \
         mock.patch.object(aa, "n_combat_total", return_value=20):
        random.seed(0)
        aa.index_to_command_effective(
            _Obs(), t, 0, cell, 0, _Aidx(), heuristic_p=0.0)
        assert calls["remap"] == 1
        assert calls["stage"] == 1
        assert calls["guard"] == 0

        calls["remap"] = calls["stage"] = calls["guard"] = 0
        random.seed(0)
        aa.index_to_command_effective(
            _Obs(), t, 0, cell, 0, _Aidx(), heuristic_p=1.0)
        assert calls["remap"] == 1
        assert calls["stage"] == 1
        assert calls["guard"] == 1

        # Seeded stochastic: stage always on; guard anneals with p=0.5
        n_stage = 0
        n_guard = 0
        for i in range(40):
            calls["stage"] = 0
            calls["guard"] = 0
            random.seed(i)
            aa.index_to_command_effective(
                _Obs(), t, 0, cell, 0, _Aidx(), heuristic_p=0.5)
            n_stage += calls["stage"]
            n_guard += calls["guard"]
        assert n_stage == 40
        assert 5 <= n_guard <= 35


def test_metrics_lock_smoke():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "metrics.jsonl")
        # exclusive write
        with metrics_lock(path, exclusive=True):
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"iter": 1, "ok": True}) + "\n")
                f.flush()
        # shared read
        rows = []
        with metrics_lock(path, exclusive=False):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    rows.append(json.loads(line))
        assert rows[0]["iter"] == 1

        # concurrent writers should not corrupt (best-effort)
        def writer(n):
            for i in range(20):
                with metrics_lock(path, exclusive=True):
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"iter": 1000 + n * 100 + i}) + "\n")
                        f.flush()
                time.sleep(0.001)

        threads = [threading.Thread(target=writer, args=(k,)) for k in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(x) for x in f if x.strip()]
        assert all("iter" in r for r in lines)
        # Best-effort lock: allow tiny loss under contention on Windows msvcrt,
        # but require the large majority of appends and no parse corruption.
        assert len(lines) >= 1 + 50, len(lines)


def test_compute_heuristic_p_phase_a_and_anneal():
    from rl.train import compute_heuristic_p
    class A: pass
    a = A()
    a.heuristic_p = None
    a.onboard_phase = "A"
    a.heuristic_phase_start = 100
    a.heuristic_anneal_iters = 60
    assert compute_heuristic_p(a, 150) == 1.0

    a.onboard_phase = "B"
    assert compute_heuristic_p(a, 100) == 1.0
    assert abs(compute_heuristic_p(a, 130) - 0.5) < 1e-9
    assert compute_heuristic_p(a, 160) == 0.0

    a.heuristic_p = 0.25
    assert compute_heuristic_p(a, 999) == 0.25
