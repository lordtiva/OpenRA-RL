"""Smoke tests for RL-vs-RL plumbing (no Docker)."""
from pathlib import Path
import tempfile

from openra_env.generated import rl_bridge_pb2
from rl.pfsp import BotPFSP, VALID_BOTS


def test_proto_has_peer_fields():
    req = rl_bridge_pb2.FastAdvanceRequest()
    assert hasattr(req, "peer_commands")
    assert hasattr(req, "peer_slot")
    obs = rl_bridge_pb2.GameObservation()
    assert hasattr(obs, "player_slot")
    assert hasattr(rl_bridge_pb2, "ObservationRequest")


def test_pfsp_accepts_rl_and_pick_ckpt():
    assert "rl" in VALID_BOTS
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "latest.pt").write_bytes(b"x")
        (d / "best.pt").write_bytes(b"y")
        pfsp = BotPFSP(ckpt_dir=d, pool=["easy", "rl"], anchor="easy")
        assert "rl" in pfsp.pool
        ck = pfsp.pick_rl_ckpt()
        assert ck is not None
        assert ck.name in ("latest.pt", "best.pt")


def test_openra_action_peer_commands():
    from openra_env.models import OpenRAAction, CommandModel, ActionType
    a = OpenRAAction(commands=[], peer_commands=[CommandModel(action=ActionType.NO_OP)])
    assert len(a.peer_commands) == 1
