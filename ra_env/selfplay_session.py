"""Two-agent RL-vs-RL game (episode) in single-RPC lock-step.

Both Multi1 (primary, p1) and Multi0 (peer, p0) run the `rl-agent` commander.
A single `FastAdvance` RPC injects both players' decisions, ticks the world
until `ticks` elapse, the game ends, or EITHER commander has pending decisions,
and returns both fog-honest observations (`p1.obs` and `p0.obs`) serialized at
the end of `WorldTick` (`PostWorldTick`), eliminating player-order races.
"""

from dataclasses import dataclass

from ra_env._proto import rl_bridge_pb2 as pb
from ra_env.bridge import Bridge, SessionPoisoned
from ra_env.session import AGENT_SLOT, BOT_SLOT, BRAINS, StepResult


@dataclass(frozen=True)
class SelfPlayConfig:
    map_name: str = "singles.oramap"
    player_faction: str = "Random"   # Multi1 (p1)
    enemy_faction: str = "Random"    # Multi0 (p0)
    max_ticks: int = 30000
    brain: str = "external"          # Multi1 brain ("external" | "builtin")
    peer_brain: str = "external"     # Multi0 brain ("external" | "builtin")

    def bots_spec(self) -> str:
        return f"{AGENT_SLOT}:rl-agent,{BOT_SLOT}:rl-agent"


@dataclass
class DualStepResult:
    p1: StepResult  # Multi1 (primary slot)
    p0: StepResult  # Multi0 (peer slot)
    tick: int
    done: bool
    truncated: bool


class SelfPlaySession:
    def __init__(self, bridge: Bridge, config: SelfPlayConfig = SelfPlayConfig()):
        if config.brain not in BRAINS:
            raise ValueError(f"unknown brain {config.brain!r}; expected one of {BRAINS}")
        if config.peer_brain not in BRAINS:
            raise ValueError(f"unknown peer_brain {config.peer_brain!r}; expected one of {BRAINS}")
        self.bridge = bridge
        self.config = config
        self.session_id = ""
        self.tick = 0

    def reset(self, seed: int) -> DualStepResult:
        """Start a fresh RL-vs-RL game. seed=0 asks the engine for a random seed."""
        self.close()
        self.session_id = self.bridge.create_session(
            map_name=self.config.map_name,
            bots=self.config.bots_spec(),
            seed=seed,
            player_faction=self.config.player_faction,
            enemy_faction=self.config.enemy_faction,
            agent_brain=self.config.brain,
            peer_brain=self.config.peer_brain,
        )
        if not self.bridge.wait_ready(self.session_id):
            self.close()
            raise RuntimeError(f"self-play session on {self.bridge.address} never became ready")
        self.tick = 0
        return self.step([], [], ticks=1)

    def step(
        self,
        p1_decisions: list[pb.Decision],
        p0_decisions: list[pb.Decision],
        ticks: int,
        p1_commands: list[pb.Command] | None = None,
        p0_commands: list[pb.Command] | None = None,
    ) -> DualStepResult:
        """Answer both players' pending decisions and advance up to `ticks` ticks."""
        if not self.session_id:
            raise RuntimeError("step() before reset()")
        ticks = max(1, min(ticks, self.config.max_ticks - self.tick))
        try:
            obs = self.bridge.fast_advance(
                self.session_id,
                ticks,
                decisions=p1_decisions,
                commands=p1_commands,
                peer_decisions=p0_decisions,
                peer_commands=p0_commands,
            )
        except SessionPoisoned:
            self.session_id = ""
            raise

        if not obs.HasField("peer_observation"):
            raise RuntimeError(f"session {self.session_id} returned no peer_observation at tick {obs.tick}")

        peer_obs = pb.GameObservation()
        peer_obs.CopyFrom(obs.peer_observation)
        obs.ClearField("peer_observation")

        self.tick = obs.tick
        done = bool(obs.done or peer_obs.done)
        truncated = not done and obs.tick >= self.config.max_ticks

        p1 = StepResult(
            obs=obs,
            tick=obs.tick,
            done=done,
            truncated=truncated,
            result=obs.result if done else "",
            pending=list(obs.pending_decisions),
            teacher=list(obs.teacher_decisions),
        )
        p0 = StepResult(
            obs=peer_obs,
            tick=peer_obs.tick,
            done=done,
            truncated=truncated,
            result=peer_obs.result if done else "",
            pending=list(peer_obs.pending_decisions),
            teacher=list(peer_obs.teacher_decisions),
        )
        return DualStepResult(p1=p1, p0=p0, tick=obs.tick, done=done, truncated=truncated)

    def close(self) -> None:
        if self.session_id:
            self.bridge.destroy_session(self.session_id)
            self.session_id = ""
