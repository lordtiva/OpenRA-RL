"""Unit tests for HyperHealthMonitor rules (synthetic metric series)."""
from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest import mock

import pytest

from rl.hyper_health import (
    HyperConfig,
    HyperHealthMonitor,
    resolve_auto_hyper,
)


def _mon(**kw):
    cfg = HyperConfig(**{k: v for k, v in kw.items() if k in HyperConfig.__dataclass_fields__})
    enabled = kw.get("enabled", True)
    return HyperHealthMonitor(cfg=cfg, enabled=enabled)


def test_disabled_always_noop():
    m = _mon(enabled=False)
    d = m.decide(kl=0.1, clip_frac=0.9, entropy=0.1, grad_norm=99.0,
                 lr=1e-4, ent_lo=0.01, epochs=2)
    assert d.action == "noop"
    assert d.reason == "disabled"


def test_phase_a_resolve_off_b_on():
    assert resolve_auto_hyper(SimpleNamespace(auto_hyper=None, onboard_phase="A")) is False
    assert resolve_auto_hyper(SimpleNamespace(auto_hyper=None, onboard_phase="B")) is True
    assert resolve_auto_hyper(SimpleNamespace(auto_hyper=None, onboard_phase="C")) is True
    assert resolve_auto_hyper(SimpleNamespace(auto_hyper=None, onboard_phase=None)) is False
    assert resolve_auto_hyper(SimpleNamespace(auto_hyper=True, onboard_phase="A")) is True
    assert resolve_auto_hyper(SimpleNamespace(auto_hyper=False, onboard_phase="B")) is False


def test_kl_high_lr_down():
    m = _mon()
    d = m.decide(kl=0.04, clip_frac=0.1, entropy=1.5, grad_norm=1.0,
                 lr=1e-4, ent_lo=0.01)
    assert d.action == "lr_down"
    assert "kl=" in d.reason


def test_kl_low_lr_up():
    m = _mon()
    d = m.decide(kl=0.005, clip_frac=0.1, entropy=1.5, grad_norm=1.0,
                 lr=1e-5, ent_lo=0.01)
    assert d.action == "lr_up"


def test_kl_early_stop_when_epochs_gt1():
    m = _mon()
    d = m.decide(kl=0.06, clip_frac=0.2, entropy=1.5, grad_norm=1.0,
                 lr=1e-4, ent_lo=0.01, epochs=2)
    assert d.action == "early_stop"
    assert d.next_epochs == 1


def test_kl_early_stop_falls_to_lr_down_if_epochs1():
    m = _mon()
    d = m.decide(kl=0.06, clip_frac=0.2, entropy=1.5, grad_norm=1.0,
                 lr=1e-4, ent_lo=0.01, epochs=1)
    assert d.action == "lr_down"


def test_clip_high_but_kl_healthy_noop():
    """Prefer KL bands: inflated clip with healthy KL must NOT lr_down."""
    m = _mon()
    d = m.decide(kl=0.005, clip_frac=0.55, entropy=1.5, grad_norm=1.0,
                 lr=1e-5, ent_lo=0.01)
    # kl low wins first → lr_up, not clip-driven lr_down
    assert d.action == "lr_up"

    d2 = m.decide(kl=0.015, clip_frac=0.55, entropy=1.5, grad_norm=1.0,
                  lr=1e-5, ent_lo=0.01)
    # kl in soft-high band + clip → tie-break lr_down
    assert d2.action == "lr_down"
    assert "tie-break" in d2.reason

    d3 = m.decide(kl=0.009, clip_frac=0.55, entropy=1.5, grad_norm=1.0,
                  lr=1e-5, ent_lo=0.01)
    # kl between low and soft, clip alone → noop (healthy KL preference)
    assert d3.action == "noop"
    assert "KL healthy" in d3.reason


def test_grad_norm_isolated_spike_noop():
    m = _mon()
    d = m.decide(kl=0.015, clip_frac=0.1, entropy=1.5, grad_norm=8.0,
                 lr=1e-4, ent_lo=0.01, grad_high_streak=1)
    assert d.action == "noop"
    assert "streak=" in d.reason


def test_grad_norm_lr_down():
    m = _mon()
    d = m.decide(kl=0.015, clip_frac=0.1, entropy=1.5, grad_norm=8.0,
                 lr=1e-4, ent_lo=0.01, grad_high_streak=2)
    assert d.action == "lr_down"
    assert "grad_norm" in d.reason


def test_grad_norm_two_step_consecutive():
    trainer = mock.MagicMock()
    trainer.opt.param_groups = [{"lr": 1e-4}]
    trainer.ent_lo = 0.01
    m = _mon()
    stats = {"kl": 0.015, "clip_frac": 0.1, "entropy": 1.5, "grad_norm": 8.0}
    d1 = m.step(trainer, stats)
    assert d1.action == "noop"
    assert "streak=" in d1.reason
    d2 = m.step(trainer, stats)
    assert d2.action == "lr_down"
    assert trainer.opt.param_groups[0]["lr"] == pytest.approx(1e-4 * 0.7)


def test_lr_min_floor_5e6():
    trainer = mock.MagicMock()
    trainer.opt.param_groups = [{"lr": 6e-6}]
    trainer.ent_lo = 0.01
    m = _mon()
    stats = {"kl": 0.015, "clip_frac": 0.1, "entropy": 1.5, "grad_norm": 8.0}
    m.step(trainer, stats)
    m.step(trainer, stats)
    assert trainer.opt.param_groups[0]["lr"] == pytest.approx(5e-6)


def test_entropy_ent_up():
    m = _mon()
    d = m.decide(kl=0.015, clip_frac=0.1, entropy=0.6, grad_norm=1.0,
                 lr=1e-4, ent_lo=0.01)
    assert d.action == "ent_up"


def test_priority_kl_over_grad_and_entropy():
    m = _mon()
    d = m.decide(kl=0.04, clip_frac=0.9, entropy=0.2, grad_norm=8.0,
                 lr=1e-4, ent_lo=0.01, epochs=1)
    # gn=8 < explode(50) so KL still wins over regular grad_high
    assert d.action == "lr_down"
    assert d.reason.startswith("kl=")


def test_pause_and_cooldown():
    trainer = mock.MagicMock()
    trainer.opt.param_groups = [{"lr": 1e-4}]
    trainer.ent_lo = 0.01

    m = _mon()
    m.pause(5)
    d = m.step(trainer, {"kl": 0.05, "clip_frac": 0.5, "entropy": 0.5,
                         "grad_norm": 10.0}, epochs=1)
    assert d.action == "noop"
    assert d.paused is True
    assert "paused" in d.reason

    # drain pause
    for _ in range(4):
        m.step(trainer, {"kl": 0.015, "clip_frac": 0.1, "entropy": 1.5,
                         "grad_norm": 1.0})
    # now can act
    d2 = m.step(trainer, {"kl": 0.04, "clip_frac": 0.1, "entropy": 1.5,
                          "grad_norm": 1.0})
    assert d2.action == "lr_down"
    # cooldown blocks next
    d3 = m.step(trainer, {"kl": 0.04, "clip_frac": 0.1, "entropy": 1.5,
                          "grad_norm": 1.0})
    assert d3.action == "noop"
    assert "cooldown" in d3.reason


def test_apply_mutates_lr_and_ent():
    trainer = mock.MagicMock()
    trainer.opt.param_groups = [{"lr": 1e-4}]
    trainer.ent_lo = 0.01
    m = _mon()

    d = m.step(trainer, {"kl": 0.04, "clip_frac": 0.1, "entropy": 1.5,
                         "grad_norm": 1.0})
    assert d.action == "lr_down"
    assert trainer.opt.param_groups[0]["lr"] == pytest.approx(1e-4 * 0.7)

    # finish cooldown
    for _ in range(3):
        m.step(trainer, {"kl": 0.015, "clip_frac": 0.1, "entropy": 1.5,
                         "grad_norm": 1.0})
    d2 = m.step(trainer, {"kl": 0.015, "clip_frac": 0.1, "entropy": 0.5,
                          "grad_norm": 1.0})
    assert d2.action == "ent_up"
    assert trainer.ent_lo == pytest.approx(0.02)


def test_early_stop_overrides_next_epochs():
    trainer = mock.MagicMock()
    trainer.opt.param_groups = [{"lr": 1e-4}]
    trainer.ent_lo = 0.01
    m = _mon()
    d = m.step(trainer, {"kl": 0.06, "clip_frac": 0.2, "entropy": 1.5,
                         "grad_norm": 1.0}, epochs=2)
    assert d.action == "early_stop"
    assert m.consume_epochs_override(2) == 1
    assert m.consume_epochs_override(2) == 2  # consumed


def test_skipped_update_noop():
    m = _mon()
    d = m.decide(kl=0.1, clip_frac=0.9, entropy=0.1, grad_norm=99.0,
                 lr=1e-4, ent_lo=0.01, skipped=True)
    assert d.action == "noop"
    assert d.reason == "update_skipped"


def test_metrics_payload():
    d = _mon().decide(kl=0.04, clip_frac=0.1, entropy=1.5, grad_norm=1.0,
                      lr=1e-4, ent_lo=0.01)
    # decide alone does not set action onto metrics until apply; build manually
    from rl.hyper_health import HyperDecision
    x = HyperDecision(action="lr_down", reason="kl", lr=7e-5, entropy_coef=0.01)
    m = x.as_metrics()
    assert m["hyper_action"] == "lr_down"
    assert m["lr"] == pytest.approx(7e-5)
    assert "entropy_coef" in m


def test_argparse_boolean_optional():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto-hyper", action=argparse.BooleanOptionalAction,
                    default=None)
    assert ap.parse_args([]).auto_hyper is None
    assert ap.parse_args(["--auto-hyper"]).auto_hyper is True
    assert ap.parse_args(["--no-auto-hyper"]).auto_hyper is False


def test_grad_explode_immediate_lr_down():
    """gn>>max_grad_norm (1920 vs clip 1.0) lr_downs on the first iter."""
    m = _mon()
    d = m.decide(kl=0.015, clip_frac=0.88, entropy=1.5, grad_norm=1921.0,
                 lr=1e-4, ent_lo=0.01, grad_high_streak=1)
    assert d.action == "lr_down"
    assert "grad_explode" in d.reason


def test_grad_explode_at_lr_floor_early_stop():
    """Incident: gn~1921, kl~0.066, lr=5e-6, hyper was noop — now early_stop."""
    m = _mon()
    d = m.decide(kl=0.066, clip_frac=0.88, entropy=1.5, grad_norm=1921.0,
                 lr=5e-6, ent_lo=0.01, epochs=1)
    assert d.action == "early_stop"
    assert "grad_explode" in d.reason
