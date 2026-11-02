import pytest
from unittest.mock import MagicMock

from ra_env._proto import rl_bridge_pb2 as pb
from ra_rl.play_skirmish import find_default_checkpoint, run_skirmish_loop
from ra_rl.policies.scripted import ScriptedPolicy


class FakeSkirmishBridge:
    def __init__(self, total_ticks: int = 50):
        self.address = "127.0.0.1:10001"
        self.tick = 0
        self.total_ticks = total_ticks
        self.submitted_decisions = []

    def get_state(self, session_id: str = "", timeout_s: float = 1.0):
        return pb.GameState(phase="playing", episode_id="skirmish-test", tick=self.tick)

    def fast_advance(self, session_id: str, ticks: int, decisions=None, commands=None):
        assert ticks == 0
        if decisions:
            self.submitted_decisions.extend(decisions)
        self.tick += 10
        done = self.tick >= self.total_ticks
        return pb.GameObservation(
            tick=self.tick,
            done=done,
            result="win" if done else "",
            pending_decisions=[],
        )


def test_find_default_checkpoint():
    ckpt = find_default_checkpoint()
    # In this repo, checkpoint_best.pt exists in runs/v4/selfplay
    assert ckpt is not None
    assert ckpt.endswith(".pt")


def test_skirmish_loop_termination():
    bridge = FakeSkirmishBridge(total_ticks=30)
    policy = ScriptedPolicy()
    result = run_skirmish_loop(bridge, policy, step_delay=0.0)
    assert result == "win"
    assert bridge.tick >= 30
