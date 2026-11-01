"""Lock-step Python client for the OpenRA RL daemon (gRPC, multi-session)."""

from ra_env._proto import rl_bridge_pb2 as pb
from ra_env.bridge import Bridge, SessionPoisoned
from ra_env.selfplay_session import DualStepResult, SelfPlayConfig, SelfPlaySession
from ra_env.session import BOT_TYPES, GameSession, MatchConfig, StepResult

__all__ = [
    "BOT_TYPES",
    "Bridge",
    "DualStepResult",
    "GameSession",
    "MatchConfig",
    "SelfPlayConfig",
    "SelfPlaySession",
    "SessionPoisoned",
    "StepResult",
    "pb",
]
