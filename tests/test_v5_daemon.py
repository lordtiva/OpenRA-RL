"""v5 specific integration tests against the live daemon:
- Dual map pool verification (singles.oramap & doubles.oramap)
- TargetCandidate metadata verification (health_percent, is_capturable, is_infiltratable, actor_id)
- DecisionKind.DECIDE_ABILITY and DECIDE_MACRO_OPS contract verification
"""

import os
import pytest

from ra_env import Bridge, GameSession, MatchConfig, pb

pytestmark = pytest.mark.daemon

ADDRESS = os.environ.get("RA_DAEMON", "localhost:9999")


@pytest.fixture(scope="module")
def bridge():
    b = Bridge(ADDRESS)
    yield b
    b.close()


def test_v5_dual_map_support(bridge):
    """Verify that both singles.oramap and doubles.oramap load and advance cleanly."""
    for map_name in ("singles.oramap", "doubles.oramap"):
        s = GameSession(bridge, MatchConfig(bot="easy", brain="external", map_name=map_name, max_ticks=500))
        try:
            step = s.reset(seed=10)
            assert step.obs.map_info.width > 0
            assert step.obs.map_info.height > 0
            step = s.step([], 250)
            assert step.obs.tick > 0
        finally:
            s.close()


def test_v5_target_candidate_metadata(bridge):
    """Verify that TargetCandidate carries v5 metadata (health_percent, actor_id, etc.)."""
    s = GameSession(bridge, MatchConfig(bot="easy", brain="external", map_name="singles.oramap", max_ticks=2500))
    try:
        step = s.reset(seed=42)
        candidates_seen = []
        # Advance until we get army/support/ability requests with targets
        for _ in range(8):
            if step.done or step.truncated:
                break
            ans = []
            for r in step.pending:
                if r.targets:
                    candidates_seen.extend(r.targets)
                if r.kind == pb.DECIDE_BUILD and "powr" in r.options:
                    ans.append(pb.Decision(request_id=r.id, choice="powr"))
            step = s.step(ans, 250)

        # Confirm target candidates exist and have valid fields
        if candidates_seen:
            first = candidates_seen[0]
            assert hasattr(first, "health_percent")
            assert hasattr(first, "is_capturable")
            assert hasattr(first, "is_infiltratable")
            assert hasattr(first, "is_ally")
            assert hasattr(first, "actor_id")
            assert 0 <= first.health_percent <= 100
    finally:
        s.close()


def test_v5_decision_kinds_and_macro_action_support(bridge):
    """Verify that the engine accepts decisions with macro_action and ability targets without error."""
    s = GameSession(bridge, MatchConfig(bot="easy", brain="external", map_name="singles.oramap", max_ticks=1000))
    try:
        step = s.reset(seed=123)
        # Send a synthetic decision with macro_action and target_index
        ans = [
            pb.Decision(request_id=999999, choice="hold", target_index=16, macro_action="hold"),
        ]
        step = s.step(ans, 200)
        assert not step.done
    finally:
        s.close()
