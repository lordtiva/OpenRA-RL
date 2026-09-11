"""Tests for Phase 1.5 hang mitigation (heartbeat + recreate policy)."""
from __future__ import annotations

import time
from pathlib import Path

from rl.collect_heartbeat import (
    HEARTBEAT_NAME,
    heartbeat_advanced,
    heartbeat_age_s,
    read_heartbeat,
    write_heartbeat,
)
from rl import auto_train as at


def test_heartbeat_write_read_roundtrip(tmp_path: Path):
    assert write_heartbeat(
        tmp_path, tick=1200, step=40, phase="collect",
        session="http://localhost:8000", iter=7, worker=1, force=True,
    )
    hb = read_heartbeat(tmp_path)
    assert hb is not None
    assert hb["tick"] == 1200
    assert hb["step"] == 40
    assert hb["phase"] == "collect"
    assert hb["iter"] == 7
    assert (tmp_path / HEARTBEAT_NAME).is_file()


def test_heartbeat_throttle(tmp_path: Path):
    assert write_heartbeat(tmp_path, tick=1, force=True, min_interval_s=60)
    assert write_heartbeat(tmp_path, tick=2, force=False, min_interval_s=60) is False
    assert write_heartbeat(tmp_path, tick=3, force=True, min_interval_s=60) is True
    assert read_heartbeat(tmp_path)["tick"] == 3


def test_heartbeat_advanced():
    assert heartbeat_advanced(None, {"ts": 1.0, "tick": 1})
    assert not heartbeat_advanced({"ts": 10.0, "tick": 5}, None)
    assert heartbeat_advanced({"ts": 10.0, "tick": 5}, {"ts": 12.0, "tick": 5})
    assert heartbeat_advanced({"ts": 10.0, "tick": 5}, {"ts": 10.1, "tick": 9})
    assert not heartbeat_advanced({"ts": 10.0, "tick": 5}, {"ts": 10.2, "tick": 5})


def test_heartbeat_age():
    now = 1000.0
    assert heartbeat_age_s({"ts": 900.0}, now=now) == 100.0
    assert heartbeat_age_s(None) is None


def test_effective_hang_threshold_tightens_on_gpu_low(monkeypatch):
    monkeypatch.setattr(at, "_onboard", None)
    base = at.hang_threshold()
    # GPU busy / unknown streak → base
    assert at.effective_hang_threshold(50, 100) == base
    assert at.effective_hang_threshold(None, 100) == base
    # GPU low but streak too short → base
    short = max(0, (at.GPU_LOW_STREAK_FOR_TIGHT_S // at.CHECK_EVERY_S) - 1)
    assert at.effective_hang_threshold(0, short) == base
    # GPU low long streak → tightened
    long = (at.GPU_LOW_STREAK_FOR_TIGHT_S // at.CHECK_EVERY_S) + 2
    thr = at.effective_hang_threshold(0, long)
    assert thr == min(base, at.GPU_IDLE_HANG_S)
    assert thr <= base


def test_effective_hang_threshold_onboard_tightens(monkeypatch):
    monkeypatch.setattr(at, "_onboard", {"phase": "B"})
    base = at.hang_threshold()
    assert base == at.ONBOARD_A_THRESHOLD_S
    long = (at.GPU_LOW_STREAK_FOR_TIGHT_S // at.CHECK_EVERY_S) + 5
    thr = at.effective_hang_threshold(1, long)
    assert thr == at.GPU_IDLE_HANG_S
    assert thr < base


def test_should_recreate_on_python_hang():
    assert at.should_recreate_on_python_hang(
        markers=0, gpu=0, idle=700, thr=600) is True
    # GPU busy → never
    assert at.should_recreate_on_python_hang(
        markers=0, gpu=40, idle=700, thr=600) is False
    # Not yet idle enough
    assert at.should_recreate_on_python_hang(
        markers=0, gpu=0, idle=100, thr=600) is False
    # markers>=1 is other branch (function is for markers==0 case)
    assert at.should_recreate_on_python_hang(
        markers=2, gpu=0, idle=700, thr=600) is False


def test_recreate_allowed_cooldown_and_max():
    at.reset_recreate_history()
    t0 = 1_700_000_000.0
    ok, why = at.recreate_allowed(now=t0)
    assert ok and why == "ok"
    at.note_recreate(now=t0)
    ok2, why2 = at.recreate_allowed(now=t0 + 10)
    assert not ok2 and "cooldown" in why2
    # After cooldown, still under max
    ok3, _ = at.recreate_allowed(now=t0 + at.RECREATE_COOLDOWN_S + 1)
    assert ok3
    # Fill to max
    at.reset_recreate_history()
    for i in range(at.RECREATE_MAX_PER_HOUR):
        at.note_recreate(now=t0 + i * (at.RECREATE_COOLDOWN_S + 1))
    ok4, why4 = at.recreate_allowed(
        now=t0 + at.RECREATE_MAX_PER_HOUR * (at.RECREATE_COOLDOWN_S + 1))
    assert not ok4 and "max" in why4
    # Hour later → pruned
    ok5, _ = at.recreate_allowed(now=t0 + 3600 + 10)
    assert ok5
    at.reset_recreate_history()