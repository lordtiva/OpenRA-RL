"""Hand-written commander policy (Phase 1 baseline: must beat `hard` >= 60%).

Fixed Allied plan: economy first, power kept ahead of demand, every queue kept
busy while there is money (extra factories when the bank piles up), a
medium-tank core with rockets and artillery, and one big push once the army is
worth enough. Only the agent's own observation is used (no spectator summary).

ARMY contract: -1 = hold (recall), k = attack targets[k], no answer = keep going.
"""

from collections import Counter
from dataclasses import dataclass

from ra_env import StepResult, pb

# Structures in the order we want them, with how many of each by then.
BUILD_ORDER = [
    ("powr", 1), ("tent", 1), ("proc", 1), ("proc", 2), ("powr", 2), ("weap", 1),
    ("apwr", 1), ("dome", 1), ("proc", 3), ("fix", 1), ("apwr", 2), ("weap", 2),
    ("atek", 1), ("apwr", 3), ("proc", 4),
]
DEFENSES = [("pbox", 2), ("gun", 3)]
POWER = {"powr": 100, "apwr": 200}

COMBAT_COST = {
    "e1": 100, "e3": 300, "e7": 1200, "medi": 200, "jeep": 600, "1tnk": 700, "2tnk": 850,
    "arty": 600, "apc": 800, "ctnk": 1350, "stnk": 1350, "mgg": 1200, "heli": 1200, "mh60": 1000,
}
HOME_RADIUS = 20  # cells around the construction yard that count as "home"


@dataclass
class Params:
    power_margin: int = 40
    rich_bank: int = 3000        # with this much banked, add factories
    max_weap: int = 3
    max_tent: int = 2
    max_infantry: int = 40
    infantry_bank: int = 0       # train infantry only with this much banked
    rifles_per_rocket: int = 2
    harv_per_proc: int = 2
    max_harv: int = 8
    arty_every: int = 4          # 1 artillery per this many tanks
    attack_value: int = 6000     # push when the army is worth this much
    reinforce_value: int = 2000  # while attacking, send the home group once it is worth this
    recall_fraction: float = 0.35  # recall when the army falls below this share of its peak
    max_aim_decisions: int = 8   # give up on a target still listed after this many army decisions


def counts(obs: pb.GameObservation) -> tuple[Counter, Counter]:
    return Counter(b.type for b in obs.buildings), Counter(u.type for u in obs.units)


def army_value(units: Counter) -> int:
    return sum(COMBAT_COST.get(t, 0) * n for t, n in units.items())


def bank(obs: pb.GameObservation) -> int:
    return obs.economy.cash + obs.economy.ore


def home_army_value(obs: pb.GameObservation) -> int:
    yards = [(b.cell_x, b.cell_y) for b in obs.buildings if b.type == "fact"]
    if not yards:
        return 0
    hx, hy = yards[0]
    return sum(COMBAT_COST.get(u.type, 0) for u in obs.units
               if (u.cell_x - hx) ** 2 + (u.cell_y - hy) ** 2 <= HOME_RADIUS ** 2)


class ScriptedPolicy:
    brain = "external"

    def __init__(self, params: Params | None = None):
        self.p = params or Params()
        self.reset(0)

    def reset(self, seed: int) -> None:
        self.attacking = False
        self.peak = 0
        self.target: tuple[str, int, int] | None = None
        self.aim_age = 0
        self.skipped: set[tuple[str, int, int]] = set()

    def decide(self, step: StepResult) -> list[pb.Decision]:
        buildings, units = counts(step.obs)
        out = []
        for req in step.pending:
            if req.kind == pb.DECIDE_BUILD:
                out.append(pb.Decision(request_id=req.id, choice=self.build(req, step.obs, buildings)))
            elif req.kind == pb.DECIDE_TRAIN:
                out.append(pb.Decision(request_id=req.id, choice=self.train(req, step.obs, buildings, units)))
            elif req.kind == pb.DECIDE_ARMY:
                index = self.army(req, step.obs, units)
                if index is not None:
                    out.append(pb.Decision(request_id=req.id, target_index=index))
        return out

    def power_item(self, options: set[str]) -> str:
        return "apwr" if "apwr" in options else ("powr" if "powr" in options else "")

    def build(self, req: pb.DecisionRequest, obs: pb.GameObservation, have: Counter) -> str:
        options = set(req.options)
        if req.queue == "Defense":
            if have["weap"] == 0:
                return ""
            for item, n in DEFENSES:
                if item in options and have[item] < n:
                    return item
            return ""

        spare = obs.economy.power_provided - obs.economy.power_drained
        if spare < self.p.power_margin and have["fact"]:
            return self.power_item(options)
        for item, n in BUILD_ORDER:
            if have[item] < n and item in options:
                return item
        if bank(obs) >= self.p.rich_bank:
            if "weap" in options and have["weap"] < self.p.max_weap:
                return "weap"
            if "tent" in options and have["tent"] < self.p.max_tent:
                return "tent"
        return ""

    def train(self, req: pb.DecisionRequest, obs: pb.GameObservation, have: Counter, units: Counter) -> str:
        options = set(req.options)
        if req.queue == "Infantry":
            infantry = units["e1"] + units["e3"]
            if infantry >= self.p.max_infantry or (bank(obs) < self.p.infantry_bank and have["weap"]):
                return ""
            if "e3" in options and units["e3"] * self.p.rifles_per_rocket < units["e1"]:
                return "e3"
            return "e1" if "e1" in options else ""
        if req.queue == "Vehicle":
            if "harv" in options and units["harv"] < min(self.p.max_harv, self.p.harv_per_proc * have["proc"]):
                return "harv"
            tanks = units["2tnk"] + units["1tnk"]
            if "arty" in options and tanks >= self.p.arty_every * (units["arty"] + 1):
                return "arty"
            for tank in ("2tnk", "1tnk"):
                if tank in options:
                    return tank
            return ""
        return ""  # aircraft / naval: not part of this plan

    def army(self, req: pb.DecisionRequest, obs: pb.GameObservation, units: Counter) -> int | None:
        value = army_value(units)
        if not req.targets:
            return None
        if not self.attacking:
            if value >= self.p.attack_value:
                self.attacking, self.peak = True, value
                return self.aim(req)
            return None
        self.peak = max(self.peak, value)
        if value < self.p.recall_fraction * self.peak:
            self.attacking, self.target = False, None
            return -1
        self.aim_age += 1
        if self.target not in {(t.kind, t.cell_x, t.cell_y) for t in req.targets}:
            return self.aim(req)  # target destroyed / area explored: next one
        if self.aim_age >= self.p.max_aim_decisions:
            self.skipped.add(self.target)  # stuck (unreachable, or nothing there): move on
            return self.aim(req)
        if home_army_value(obs) >= self.p.reinforce_value:
            return self.aim(req)  # fold the home group into the push
        return None  # keep going

    def aim(self, req: pb.DecisionRequest) -> int:
        i = self.pick_target(req, self.skipped)
        t = req.targets[i]
        new = (t.kind, t.cell_x, t.cell_y)
        if new != self.target:
            self.aim_age = 0
        self.target = new
        return i

    @staticmethod
    def pick_target(req: pb.DecisionRequest, skipped: set = frozenset()) -> int:
        # Targets come sorted nearest-first within each kind: known buildings, then the
        # enemy spawn, then unexplored and stale (not seen lately) sectors where an expansion may hide.
        for kind in ("enemy_building", "enemy_spawn", "unexplored", "stale"):
            for i, t in enumerate(req.targets):
                if t.kind == kind and (t.kind, t.cell_x, t.cell_y) not in skipped:
                    return i
        return 0
