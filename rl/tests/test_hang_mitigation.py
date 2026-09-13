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
    # Default consecutive=0 → no recreate (need streak)
    assert at.should_recreate_on_python_hang(
        markers=0, gpu=0, idle=1000, thr=900) is False
    # GPU busy → never
    assert at.should_recreate_on_python_hang(
        markers=0, gpu=40, idle=1000, thr=900,
        consecutive_python_idles=5) is False
    # Not yet idle enough
    assert at.should_recreate_on_python_hang(
        markers=0, gpu=0, idle=100, thr=900,
        consecutive_python_idles=5) is False
    # markers>=1 is other branch (function is for markers==0 case)
    assert at.should_recreate_on_python_hang(
        markers=2, gpu=0, idle=1000, thr=900,
        consecutive_python_idles=5) is False


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


def test_should_recreate_requires_consecutive_streak():
    """No Docker recreate on first python-idle; yes after streak."""
    # First idle hang → train-only (consecutive=1 < default 3)
    assert at.should_recreate_on_python_hang(
        markers=0, gpu=0, idle=1000, thr=900,
        consecutive_python_idles=1) is False
    assert at.should_recreate_on_python_hang(
        markers=0, gpu=0, idle=1000, thr=900,
        consecutive_python_idles=2) is False
    assert at.should_recreate_on_python_hang(
        markers=0, gpu=0, idle=1000, thr=900,
        consecutive_python_idles=3) is True
    # Configurable recreate_after=2
    assert at.should_recreate_on_python_hang(
        markers=0, gpu=0, idle=1000, thr=900,
        consecutive_python_idles=1, recreate_after=2) is False
    assert at.should_recreate_on_python_hang(
        markers=0, gpu=0, idle=1000, thr=900,
        consecutive_python_idles=2, recreate_after=2) is True
    # GPU busy / not idle enough still False
    assert at.should_recreate_on_python_hang(
        markers=0, gpu=40, idle=1000, thr=900,
        consecutive_python_idles=5) is False
    assert at.should_recreate_on_python_hang(
        markers=0, gpu=0, idle=100, thr=900,
        consecutive_python_idles=5) is False
    # markers!=0 is other branch
    assert at.should_recreate_on_python_hang(
        markers=2, gpu=0, idle=1000, thr=900,
        consecutive_python_idles=5) is False


def test_gpu_idle_hang_raised():
    assert at.GPU_IDLE_HANG_S >= 900
    assert at.PYTHON_IDLE_RECREATE_STREAK >= 2


def test_fast_advance_timeout_hard_capped(monkeypatch):
    from openra_env.server import bridge_client as bc
    monkeypatch.delenv("OPENRA_RL_FAST_ADVANCE_DEADLINE_S", raising=False)
    assert bc.fast_advance_deadline_s() == 90.0
    # ticks must NOT inflate past deadline (old bug: max(90, ticks+30))
    assert bc.fast_advance_grpc_timeout_s(50) == 90.0
    assert bc.fast_advance_grpc_timeout_s(200) == 90.0
    assert bc.fast_advance_grpc_timeout_s(1) == 90.0
    monkeypatch.setenv("OPENRA_RL_FAST_ADVANCE_DEADLINE_S", "75")
    assert bc.fast_advance_deadline_s() == 75.0
    assert bc.fast_advance_grpc_timeout_s(500, deadline_s=75) == 75.0
    monkeypatch.setenv("OPENRA_RL_FAST_ADVANCE_DEADLINE_S", "bad")
    assert bc.fast_advance_deadline_s() == 90.0


def test_op_message_timeout_near_deadline(monkeypatch):
    from openra_env import client as cl
    monkeypatch.delenv("OPENRA_RL_FAST_ADVANCE_DEADLINE_S", raising=False)
    t = cl.default_op_message_timeout_s()
    assert 90.0 < t <= 120.0
    monkeypatch.setenv("OPENRA_RL_FAST_ADVANCE_DEADLINE_S", "90")
    assert cl.default_op_message_timeout_s() == 110.0


def test_is_collect_timeout_empty_asyncio():
    from rl.collect_timeout import is_collect_timeout
    assert is_collect_timeout(TimeoutError())
    assert is_collect_timeout(TimeoutError(""))
    assert is_collect_timeout(TimeoutError("WS op timed out after 110s"))
    assert is_collect_timeout(RuntimeError("FastAdvance DEADLINE_EXCEEDED"))
    assert not is_collect_timeout(RuntimeError("bridge failed to start"))


def test_wait_for_ready_wall_clock():
    """GetState hang must not run 40x30s — wall clock wins."""
    import time
    from openra_env.server.bridge_client import BridgeClient
    bc = BridgeClient()
    bc._connected = True
    bc._stub = object()

    def slow(_timeout_s=None):
        time.sleep(0.2)
        raise RuntimeError("hung GetState")

    bc.get_state = slow
    t0 = time.monotonic()
    assert bc.wait_for_ready(max_retries=40, retry_interval=0.5, deadline_s=0.45) is False
    elapsed = time.monotonic() - t0
    assert elapsed < 1.5, elapsed


def test_send_and_receive_op_times_out(monkeypatch):
    """Whole send+recv (not just recv) fails within the cap."""
    import asyncio
    import time
    from openra_env.client import OpenRAEnv

    env = OpenRAEnv.__new__(OpenRAEnv)
    env._message_timeout = 3000.0
    env._ws = object()

    async def hang(_msg):
        await asyncio.sleep(30)

    env._send_and_receive = hang
    monkeypatch.setenv("OPENRA_RL_FAST_ADVANCE_DEADLINE_S", "1")

    async def run():
        t0 = time.monotonic()
        try:
            await env._send_and_receive_op({"type": "mcp"}, timeout_s=0.35)
            raise AssertionError("should have timed out")
        except TimeoutError as e:
            assert "timed out" in str(e).lower()
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, elapsed
        assert env._ws is None

    asyncio.run(run())


def test_hung_advance_cannot_block_past_timeout(tmp_path):
    """Simulated hung advance must fail ~timeout and write collect_timeout hb."""
    import asyncio
    import time
    from rl.collect_heartbeat import read_heartbeat
    from rl.collect_timeout import await_env_op

    class Hung:
        async def advance(self, ticks):
            await asyncio.sleep(3600)

    async def run():
        t0 = time.monotonic()
        try:
            await await_env_op(
                Hung().advance(50),
                timeout_s=0.3,
                heartbeat_path=tmp_path,
                what="advance",
                error="advance_deadline",
                iter=124, worker=0,
            )
            raise AssertionError("hung advance must timeout")
        except TimeoutError as e:
            assert "timed out" in str(e)
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, elapsed
        hb = read_heartbeat(tmp_path)
        assert hb is not None
        assert hb["phase"] == "collect_timeout"
        assert hb["error"] == "advance_deadline"
        assert hb["iter"] == 124

    asyncio.run(run())


def test_heartbeat_written_on_timeout_path(tmp_path):
    from rl.collect_heartbeat import read_heartbeat
    from rl.collect_timeout import note_collect_timeout
    write_heartbeat(tmp_path, phase="collect", force=True, tick=100)
    note_collect_timeout(tmp_path, error="step_timeout", iter=7, worker=2)
    hb = read_heartbeat(tmp_path)
    assert hb["phase"] == "collect_timeout"
    assert hb["error"] == "step_timeout"
    assert hb["iter"] == 7


def test_send_and_receive_op_uncancellable_clears_ws(monkeypatch):
    """Wedged send that ignores CancelledError still times out and drops _ws."""
    import asyncio
    import time
    from openra_env.client import OpenRAEnv

    env = OpenRAEnv.__new__(OpenRAEnv)
    env._message_timeout = 3000.0
    env._ws = object()  # sentinel; must be cleared on timeout

    async def uncancellable(_msg):
        # Ignore first CancelledError briefly (wedged ws.send), then cooperate
        # so asyncio.run shutdown cannot hang.
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(0.4)
            raise

    env._send_and_receive = uncancellable
    monkeypatch.setenv("OPENRA_RL_FAST_ADVANCE_DEADLINE_S", "1")

    async def run():
        t0 = time.monotonic()
        try:
            await env._send_and_receive_op({"type": "mcp"}, timeout_s=0.25)
            raise AssertionError("should have timed out")
        except TimeoutError as e:
            assert "timed out" in str(e).lower()
            assert "force-dropped" in str(e).lower() or "timed out" in str(e).lower()
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, elapsed
        assert env._ws is None

    asyncio.run(run())


def test_await_env_op_uncancellable_bounded(tmp_path):
    """await_env_op must not hang waiting for cancel of an uncancellable coro."""
    import asyncio
    import time
    from rl.collect_heartbeat import read_heartbeat
    from rl.collect_timeout import await_env_op

    async def uncancellable():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(0.4)
            raise

    async def run():
        t0 = time.monotonic()
        try:
            await await_env_op(
                uncancellable(),
                timeout_s=0.25,
                heartbeat_path=tmp_path,
                what="uncancellable_op",
                error="op_timeout",
                iter=1,
            )
            raise AssertionError("must timeout")
        except TimeoutError as e:
            assert "timed out" in str(e)
        elapsed = time.monotonic() - t0
        # timeout 0.25 + cancel join <=1s + slack
        assert elapsed < 2.0, elapsed
        hb = read_heartbeat(tmp_path)
        assert hb is not None
        assert hb["phase"] == "collect_timeout"

    asyncio.run(run())


def test_hang_threshold_for_dead_hb():
    """markers=0 + GPU low + dead collect hb → ~HB_DEAD_HANG_S, not 900."""
    base = 900
    # Healthy: hb fresh during collect → no tighten
    thr = at.hang_threshold_for_dead_hb(
        base, markers=0, gpu=0,
        hb={"phase": "collect"}, hb_age=30.0)
    assert thr == base
    # Dead hb age during collect → tight
    thr2 = at.hang_threshold_for_dead_hb(
        base, markers=0, gpu=0,
        hb={"phase": "collect"}, hb_age=float(at.HB_DEAD_HANG_S))
    assert thr2 == at.HB_DEAD_HANG_S
    assert thr2 < base
    # Missing hb → tight
    thr3 = at.hang_threshold_for_dead_hb(
        base, markers=0, gpu=0, hb=None, hb_age=None)
    assert thr3 == at.HB_DEAD_HANG_S
    # GPU busy → never tighten
    thr4 = at.hang_threshold_for_dead_hb(
        base, markers=0, gpu=40,
        hb={"phase": "collect"}, hb_age=999.0)
    assert thr4 == base
    # markers>=1 → other branch; leave base
    thr5 = at.hang_threshold_for_dead_hb(
        base, markers=2, gpu=0,
        hb={"phase": "collect"}, hb_age=999.0)
    assert thr5 == base
    # Non-collect phase with dead age → do not tighten
    thr6 = at.hang_threshold_for_dead_hb(
        base, markers=0, gpu=0,
        hb={"phase": "update"}, hb_age=999.0)
    assert thr6 == base
    # collect_timeout phase → tight when dead
    thr7 = at.hang_threshold_for_dead_hb(
        base, markers=0, gpu=0,
        hb={"phase": "collect_timeout"}, hb_age=999.0)
    assert thr7 == at.HB_DEAD_HANG_S
