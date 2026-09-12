"""Idle early-truncate: stop 53k-tick timeout poison."""
from types import SimpleNamespace as NS

from rl.rollout import (
    episode_progress,
    is_progress_activity,
    should_idle_truncate,
)


def test_should_idle_truncate_gates():
    assert not should_idle_truncate(20000, 0)
    assert not should_idle_truncate(35000, 34000)
    assert should_idle_truncate(35000, 0)
    assert should_idle_truncate(40000, 35000)
    assert not should_idle_truncate(40000, 36000)
    assert not should_idle_truncate(40000, 0, after_tick=0)
    assert not should_idle_truncate(40000, 0, idle_ticks=0)


def test_episode_progress_and_activity():
    obs = NS(tick=12000, military=NS(kills_cost=100, deaths_cost=20))
    tick, kills, deaths, earned = episode_progress(
        obs, {"own": {"earned": 500}})
    assert (tick, kills, deaths, earned) == (12000, 100, 20, 500)
    assert is_progress_activity(100, 20, 500, 0, 0, 0)
    assert not is_progress_activity(100, 20, 500, 100, 20, 500)
    assert is_progress_activity(
        100, 20, 500, 100, 20, 500, interrupt_reason="under_attack")
    assert not is_progress_activity(
        100, 20, 500, 100, 20, 500, interrupt_reason="production_complete")
