"""Integration tests against a real daemon: docker compose up -d, then
    pytest -m daemon
Address override: RA_DAEMON=host:port.
"""

import os

import pytest

from ra_env import Bridge, GameSession, MatchConfig
from ra_env.determinism import describe, trace
from ra_env.probe import deploy_mcv_commands

pytestmark = pytest.mark.daemon

ADDRESS = os.environ.get("RA_DAEMON", "localhost:9999")


@pytest.fixture(scope="module")
def bridge():
    b = Bridge(ADDRESS)
    yield b
    b.close()


def test_game_starts_with_an_mcv_that_deploys(bridge):
    s = GameSession(bridge, MatchConfig(bot="easy", max_ticks=2000))
    try:
        step = s.reset(seed=42)
        cmds = deploy_mcv_commands(step.obs)
        assert cmds, "agent should start with an MCV"
        step = s.step(cmds, 200)
        assert any(b.type == "fact" for b in step.obs.buildings)
        assert step.obs.map_info.width > 0
    finally:
        s.close()


def test_same_seed_same_game(bridge):
    match = MatchConfig(bot="hard", max_ticks=4000)
    a = trace(bridge, match, 99, 100)
    b = trace(bridge, match, 99, 100)
    assert describe(a, b) == "identical"
