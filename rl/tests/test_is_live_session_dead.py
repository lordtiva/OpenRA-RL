"""Tiny tests for live session-dead detection (no train side effects)."""
from __future__ import annotations

import asyncio

from rl.collect_timeout import is_collect_timeout, is_live_session_dead


def test_is_live_session_dead_covers_no_progress_and_unavailable():
    assert is_live_session_dead(RuntimeError("FastAdvance NO-PROGRESS"))
    assert is_live_session_dead(RuntimeError("channel UNAVAILABLE"))
    assert is_live_session_dead(RuntimeError("session poisoned"))
    assert is_live_session_dead(RuntimeError("CreateSession required"))
    assert is_live_session_dead(RuntimeError("DEADLINE_EXCEEDED"))
    assert is_live_session_dead(RuntimeError("ABORTED: world hung"))
    assert is_live_session_dead(TimeoutError(""))
    assert is_live_session_dead(asyncio.TimeoutError())
    # not a session-dead / not a collect timeout
    assert not is_live_session_dead(RuntimeError("bridge failed to start"))
    assert not is_live_session_dead(None)


def test_is_live_session_dead_superset_of_collect_timeout():
    samples = [
        TimeoutError("WS op timed out after 110s"),
        RuntimeError("FastAdvance DEADLINE_EXCEEDED"),
        RuntimeError("poisoned session"),
        RuntimeError("CreateSession required before advance"),
        RuntimeError("ABORTED"),
    ]
    for exc in samples:
        assert is_collect_timeout(exc)
        assert is_live_session_dead(exc)
