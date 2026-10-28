"""One game (episode) against a scripted OpenRA bot, in lock-step.

The game only advances inside step(): the agent decides, the engine runs
`ticks` ticks headless, the observation comes back. No wall-clock pacing.
"""

from dataclasses import dataclass

from ra_env._proto import rl_bridge_pb2 as pb
from ra_env.bridge import Bridge, SessionPoisoned

# Friendly difficulty -> OpenRA bot type (mods/ra/rules/ai.yaml).
BOT_TYPES = {
    "beginner": "beginner",
    "easy": "easy",
    "medium": "medium",
    "hard": "normal",
    "brutal": "rush",
    "normal": "normal",
    "rush": "rush",
    "turtle": "turtle",
    "naval": "naval",
}

AGENT_SLOT = "Multi1"
BOT_SLOT = "Multi0"


@dataclass(frozen=True)
class MatchConfig:
    map_name: str = "singles.oramap"
    bot: str = "hard"
    player_faction: str = "RandomAllies"
    enemy_faction: str = "Random"
    max_ticks: int = 30000  # truncation cap (~20 min of game time at 25 t/s)

    def bots_spec(self) -> str:
        bot_type = BOT_TYPES.get(self.bot)
        if bot_type is None:
            raise ValueError(f"unknown bot {self.bot!r}; expected one of {sorted(BOT_TYPES)}")
        return f"{AGENT_SLOT}:rl-agent,{BOT_SLOT}:{bot_type}"


@dataclass
class StepResult:
    obs: pb.GameObservation
    tick: int
    done: bool        # game over (win / lose / draw)
    truncated: bool   # hit match.max_ticks without a result
    result: str       # "win" | "lose" | "draw" | ""


class GameSession:
    def __init__(self, bridge: Bridge, match: MatchConfig):
        self.bridge = bridge
        self.match = match
        self.session_id = ""
        self.tick = 0

    def reset(self, seed: int) -> StepResult:
        """Start a fresh game. seed=0 asks the engine for a random seed."""
        self.close()
        self.session_id = self.bridge.create_session(
            map_name=self.match.map_name,
            bots=self.match.bots_spec(),
            seed=seed,
            player_faction=self.match.player_faction,
            enemy_faction=self.match.enemy_faction,
        )
        if not self.bridge.wait_ready(self.session_id):
            self.close()
            raise RuntimeError(f"session on {self.bridge.address} never became ready")
        self.tick = 0
        return self.step([], ticks=1)

    def step(self, commands: list[pb.Command], ticks: int) -> StepResult:
        if not self.session_id:
            raise RuntimeError("step() before reset()")
        ticks = max(1, min(ticks, self.match.max_ticks - self.tick))
        try:
            obs = self.bridge.fast_advance(self.session_id, ticks, commands)
        except SessionPoisoned:
            self.session_id = ""  # the daemon already destroyed it
            raise
        self.tick = obs.tick
        done = bool(obs.done)
        return StepResult(
            obs=obs,
            tick=obs.tick,
            done=done,
            truncated=not done and obs.tick >= self.match.max_ticks,
            result=obs.result if done else "",
        )

    def close(self) -> None:
        if self.session_id:
            self.bridge.destroy_session(self.session_id)
            self.session_id = ""
