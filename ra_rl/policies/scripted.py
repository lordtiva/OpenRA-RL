"""Bilateral (Allies + Soviet) multi-squad commander policy (Commander v4 teacher).

Fog-honest reactive macro + tactical coordination across Allies and Soviet against
rush (`brutal`), macro (`hard`), fortified (`turtle`), and self-play opponents:
  - Bilateral build trees: Allied (`tent`, `weap`, `dome`, `fix`, `hpad`, `atek`)
    and Soviet (`barr`, `weap`, `dome`, `afld`, `stek`), keeping power ahead of drain
    and expanding with a second MCV if base placement saturates (`placement_failures`).
  - Strict Aircraft Rearm Rule: only trains `heli`/`mh60` up to `have["hpad"]` and
    `yak`/`mig` up to `have["afld"]` so every aircraft has a dedicated rearm pad.
  - Multi-squad tactics:
      * `MAIN`: rallies until push threshold, skirmishes with normal micro, and switches
        to fearless `all_in` when massing decisive superiority on an enemy base.
      * `HARASS`: fast scout/raider group (`jeep`/`ftrk`/`1tnk`) revealing `unexplored`
        sectors early and hunting exposed `enemy_harvester` targets.
      * `AIR`: strikes `enemy_harvester`, artillery/V2s, and low-AA structures.
      * `SUPPORT`: triggers SpyPlane on fog, Parabombs/Paratroopers/Nuke on structures,
        and Iron Curtain / Chronoshift during active `MAIN` pushes.
  - Only the agent's own fog-of-war observation is used (never `global_summary`).
"""

from collections import Counter
from dataclasses import dataclass

from ra_env import StepResult, pb
from ra_rl.vocab import ALLIED_FACTIONS, SOVIET_FACTIONS, base_type

# Unified Allied + Soviet build order (unavailable faction buildings are skipped automatically).
BUILD_ORDER = [
    ("powr", 1),
    ("tent", 1), ("barr", 1),
    ("proc", 1),
    ("proc", 2),                 # 2nd Refinery early: double income by tick 2500!
    ("powr", 2),
    ("weap", 1),
    ("powr", 3),                 # 3rd Power Plant: prevents brownouts during tech & defense ramp!
    ("fix", 1),                  # unlocks 3tnk (Heavy Tank) for Soviet / 2tnk for Allies ($1200)
    ("proc", 3),                 # 3rd Refinery: 3rd harvester + dock for booming midgame economy!
    ("weap", 2),                 # 2nd War Factory: double tank production speed!
    ("dome", 1),                 # radar + unlocks siege units & hpad & afld ($1000)
    ("afld", 1),                 # unlocks Soviet SpyPlane ($600) early to uncover the fog of war!
    ("hpad", 1),                 # unlocks Allied Longbow Helicopters ($500)
    ("hpad", 2),                 # 2nd Helipad for Allied air fleet ($500)
    ("afld", 2),                 # 2nd Airfield for Soviet air fleet ($500)
    ("stek", 1), ("atek", 1),
]
TECH_BUILDINGS = frozenset({"dome", "afld", "hpad", "atek", "stek"})
EARLY_DEFENSES = [("pbox", 2), ("ftur", 2), ("gun", 1)]
MID_DEFENSES = [("gun", 2)]
LATE_DEFENSES = [("tsla", 1)]
DEFENSES = EARLY_DEFENSES + MID_DEFENSES + LATE_DEFENSES
RUSH_DEFENSES = [("pbox", 2), ("ftur", 2), ("gun", 1), ("hbox", 1)]
AA_DEFENSES = [("agun", 2), ("sam", 2)]
SUPER_DEFENSES = [("iron", 1), ("pdox", 1)]

STATIC_DEFENSE_TYPES = frozenset({"pbox", "hbox", "gun", "ftur", "tsla", "sam", "agun"})
GROUND_DEFENSE_TYPES = frozenset({"pbox", "hbox", "gun", "ftur", "tsla"})
AA_TYPES = frozenset({"sam", "agun", "ftrk", "e3"})
AIR_TYPES = frozenset({"heli", "hind", "mh60", "yak", "mig"})
AIR_BUILDING_TYPES = frozenset({"hpad", "afld"})

COMBAT_COST = {
    "e1": 100, "e2": 150, "e3": 300, "e4": 300, "e7": 1200, "shok": 350, "medi": 200, "dog": 200,
    "jeep": 600, "1tnk": 700, "2tnk": 850, "3tnk": 1150, "4tnk": 2000,
    "arty": 600, "v2rl": 900, "apc": 800, "ctnk": 1350, "stnk": 1350, "ttnk": 1350,
    "ftrk": 600, "mgg": 1200, "heli": 1200, "hind": 1500, "mh60": 1000, "yak": 1350, "mig": 2000,
}
HOME_RADIUS = 28  # cells around the construction yard that count as "home"


@dataclass
class Params:
    power_margin: int = 40
    rich_bank: int = 3000          # with this much banked, add extra factories
    max_weap: int = 3
    max_tent: int = 2
    max_infantry: int = 20
    infantry_bank: int = 800
    rifles_per_rocket: int = 1
    harv_per_proc: int = 2
    max_harv: int = 6
    harass_raiders: int = 2        # fast scouts/raiders (jeep / ftrk) once >=2 refineries exist
    arty_every: int = 3            # 1 artillery/V2 per this many tanks (standard)
    siege_arty_every: int = 2      # 1 artillery/V2 per this many tanks vs fortified turtle
    siege_def_threshold: int = 3
    attack_value: int = 13000      # MAIN push threshold: ~8 tanks + 2 escorts + 12 infantry (deathball)
    all_in_value: int = 16000      # MAIN fearless assault threshold when overwhelming enemy (endgame mass)
    siege_step_value: int = 0
    max_attack_value: int = 12000
    reinforce_value: int = 2500
    recall_fraction: float = 0.35
    min_attack_fraction: float = 0.35
    max_aim_decisions: int = 6
    rush_threat_threshold: int = 600


def counts(obs: pb.GameObservation) -> tuple[Counter, Counter]:
    return Counter(base_type(b.type) for b in obs.buildings), Counter(base_type(u.type) for u in obs.units)


def army_value(units: Counter) -> int:
    return sum(COMBAT_COST.get(t, 0) * n for t, n in units.items())


def ground_main_value(obs: pb.GameObservation) -> int:
    harass_ids = set(obs.harass_unit_ids)
    total = 0
    for u in obs.units:
        ut = base_type(u.type)
        if ut in AIR_TYPES or u.actor_id in harass_ids:
            continue
        total += COMBAT_COST.get(ut, 0)
    return total


def bank(obs: pb.GameObservation) -> int:
    return obs.economy.cash + obs.economy.ore


def base_center(obs: pb.GameObservation) -> tuple[int, int] | None:
    yards = [(b.cell_x, b.cell_y) for b in obs.buildings if base_type(b.type) == "fact"]
    if yards:
        return yards[0]
    if obs.buildings:
        bx = int(sum(b.cell_x for b in obs.buildings) / len(obs.buildings))
        by = int(sum(b.cell_y for b in obs.buildings) / len(obs.buildings))
        return bx, by
    return None


def home_army_value(obs: pb.GameObservation) -> int:
    yards = [(b.cell_x, b.cell_y) for b in obs.buildings if base_type(b.type) == "fact"]
    if not yards:
        return 0
    hx, hy = yards[0]
    harass_ids = set(obs.harass_unit_ids)
    return sum(
        COMBAT_COST.get(base_type(u.type), 0)
        for u in obs.units
        if base_type(u.type) not in AIR_TYPES
        and u.actor_id not in harass_ids
        and (u.cell_x - hx) ** 2 + (u.cell_y - hy) ** 2 <= HOME_RADIUS ** 2
    )


def home_threat_value(obs: pb.GameObservation) -> int:
    center = base_center(obs)
    if center is None:
        return 0
    hx, hy = center
    return sum(
        COMBAT_COST.get(base_type(u.type), 0)
        for u in obs.visible_enemies
        if (u.cell_x - hx) ** 2 + (u.cell_y - hy) ** 2 <= HOME_RADIUS ** 2
    )


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
        self.recalls = 0

        self.harass_target: tuple[str, int, int] | None = None
        self.harass_age = 0
        self.harass_skipped: set[tuple[str, int, int]] = set()

        self.air_target: tuple[str, int, int] | None = None
        self.air_age = 0

        self.seen_defenses: set[int] = set()
        self.aa_locations: dict[int, tuple[int, int]] = {}
        self.seen_air = False
        self.rush_detected = False
        self.harass_built = 0
        self.recent_threat_peak = 0

    def _observe_intel(self, obs: pb.GameObservation, have: Counter) -> int:
        for b in obs.visible_enemy_buildings:
            bt = base_type(b.type)
            if bt in STATIC_DEFENSE_TYPES:
                self.seen_defenses.add(b.actor_id)
            if bt in AA_TYPES:
                self.aa_locations[b.actor_id] = (b.cell_x, b.cell_y)
            if bt in AIR_BUILDING_TYPES:
                self.seen_air = True
        for u in obs.visible_enemies:
            ut = base_type(u.type)
            if ut in AIR_TYPES:
                self.seen_air = True
            if ut in AA_TYPES:
                self.aa_locations[u.actor_id] = (u.cell_x, u.cell_y)
        threat = home_threat_value(obs)
        if threat >= self.p.rush_threat_threshold and (obs.tick <= 5000 or have["weap"] == 0):
            self.rush_detected = True
        return threat

    def decide(self, step: StepResult) -> list[pb.Decision]:
        buildings, units = counts(step.obs)
        threat = self._observe_intel(step.obs, buildings)
        for req in step.pending:
            if req.kind == pb.DECIDE_ARMY:
                for t in req.targets:
                    if t.kind == "enemy_building" and base_type(t.type) in ("hpad", "afld"):
                        self.seen_air = True
                    elif t.kind == "enemy_mobile" and base_type(t.type) in AIR_TYPES:
                        self.seen_air = True
        out: list[pb.Decision] = []
        for req in step.pending:
            if req.kind == pb.DECIDE_BUILD:
                out.append(pb.Decision(request_id=req.id, choice=self.build(req, step.obs, buildings, units, threat)))
            elif req.kind == pb.DECIDE_TRAIN:
                out.append(pb.Decision(request_id=req.id, choice=self.train(req, step.obs, buildings, units, threat)))
            elif req.kind == pb.DECIDE_ARMY:
                grp = req.queue or "MAIN"
                if grp == "HARASS":
                    d = self.decide_harass(req, step.obs, threat)
                elif grp == "AIR":
                    d = self.decide_air(req, step.obs)
                else:
                    d = self.decide_main(req, step.obs, units, threat)
                if d is not None:
                    out.append(d)
            elif req.kind == pb.DECIDE_SUPPORT:
                out.append(self.decide_support(req, step.obs))
        return out

    def army_state(self) -> tuple:
        return (
            self.attacking,
            self.peak,
            self.target,
            self.aim_age,
            set(self.skipped),
            self.recalls,
            self.harass_target,
            self.harass_age,
            set(self.harass_skipped),
            self.air_target,
            self.air_age,
        )

    def sync_army(
        self,
        before: tuple,
        req: pb.DecisionRequest,
        index: int | None,
        obs: pb.GameObservation,
    ) -> None:
        """DAgger / exploration state sync when another policy executed `index` instead."""
        (
            self.attacking,
            self.peak,
            self.target,
            self.aim_age,
            self.skipped,
            self.recalls,
            self.harass_target,
            self.harass_age,
            self.harass_skipped,
            self.air_target,
            self.air_age,
        ) = before

        if req.kind != pb.DECIDE_ARMY:
            return
        grp = req.queue or "MAIN"
        if grp == "HARASS":
            if index is None:
                self.harass_age += 1
            elif index < 0:
                self.harass_target, self.harass_age = None, 0
            elif index < len(req.targets):
                t = req.targets[index]
                new = (t.kind, t.cell_x, t.cell_y)
                self.harass_age = self.harass_age + 1 if new == self.harass_target else 0
                self.harass_target = new
            return

        if grp == "AIR":
            if index is None:
                self.air_age += 1
            elif index < 0:
                self.air_target, self.air_age = None, 0
            elif index < len(req.targets):
                t = req.targets[index]
                new = (t.kind, t.cell_x, t.cell_y)
                self.air_age = self.air_age + 1 if new == self.air_target else 0
                self.air_target = new
            return

        if index is None:
            if self.attacking:
                self.aim_age += 1
            return
        if index < 0:
            if self.attacking:
                self.recalls += 1
            self.attacking, self.target = False, None
            return
        if not self.attacking:
            self.attacking, self.peak = True, max(ground_main_value(obs), army_value(counts(obs)[1]))
        if index < len(req.targets):
            t = req.targets[index]
            new = (t.kind, t.cell_x, t.cell_y)
            self.aim_age = self.aim_age + 1 if new == self.target else 0
            self.target = new

    def power_item(self, options: set[str]) -> str:
        return "apwr" if "apwr" in options else ("powr" if "powr" in options else "")

    def build(
        self,
        req: pb.DecisionRequest,
        obs: pb.GameObservation,
        have: Counter,
        units: Counter | None = None,
        threat: int = 0,
    ) -> str:
        options = {base_type(o) for o in req.options}
        raw_map = {base_type(o): o for o in req.options}

        def pick(item: str) -> str:
            return raw_map.get(item, "")

        tanks = (units["1tnk"] + units["2tnk"] + units["3tnk"] + units["4tnk"] + units["ctnk"]) if units else 0
        b_cash = bank(obs)

        if req.queue == "Defense":
            if have["weap"] == 0:
                if (self.rush_detected or threat >= 400) and have["proc"] >= 1:
                    for item, n in RUSH_DEFENSES:
                        if item in options and have[item] < n:
                            return pick(item)
                return ""
            for item, n in EARLY_DEFENSES:
                if item in options and have[item] < n:
                    return pick(item)
            if self.seen_air and b_cash >= 600:
                for item, n in AA_DEFENSES:
                    if item in options and have[item] < n:
                        return pick(item)
            if b_cash >= 700 and have["proc"] >= 2 and (tanks >= 2 or threat >= 800):
                for item, n in MID_DEFENSES + LATE_DEFENSES:
                    if item in options and have[item] < n:
                        return pick(item)
            if b_cash >= 3000 and (have["atek"] or have["stek"]):
                for item, n in SUPER_DEFENSES:
                    if item in options and have[item] < n:
                        return pick(item)
            return ""

        spare = obs.economy.power_provided - obs.economy.power_drained
        needed = 100 if (have["dome"] or have["afld"] or have["tsla"]) else self.p.power_margin
        if spare < needed and have["fact"]:
            p_item = self.power_item(options)
            return pick(p_item) if p_item else ""
        for item, n in BUILD_ORDER:
            current = have[item] + (have["apwr"] if item == "powr" else 0)
            if current < n and item in options:
                if item == "proc" and n >= 4 and units and tanks < 2:
                    continue
                if item in ("afld", "hpad") and units and tanks < 2:
                    continue
                if item == "dome" and units and tanks < 6:
                    continue
                if item in ("stek", "atek") and units and tanks < 10:
                    continue
                return pick(item)

        if (b_cash >= 1500 and tanks >= 3) or b_cash >= 2500:
            if "proc" in options and have["proc"] < 4:
                return pick("proc")
            if "weap" in options and have["weap"] < self.p.max_weap:
                return pick("weap")
            for b_inf in ("tent", "barr"):
                if b_inf in options and have[b_inf] < self.p.max_tent:
                    return pick(b_inf)
        return ""

    def train(
        self,
        req: pb.DecisionRequest,
        obs: pb.GameObservation,
        have: Counter,
        units: Counter,
        threat: int = 0,
    ) -> str:
        options = {base_type(o) for o in req.options}
        raw_map = {base_type(o): o for o in req.options}

        def pick(item: str) -> str:
            return raw_map.get(item, "")

        if req.queue == "Infantry":
            if "dog" in options and units["dog"] < 3:
                return pick("dog")
            infantry = units["e1"] + units["e2"] + units["e3"] + units["e4"] + units["shok"]
            under_attack = threat >= self.p.rush_threat_threshold
            if infantry >= self.p.max_infantry:
                return ""
            if not under_attack and have["weap"] == 0 and infantry >= 4:
                return ""
            if not under_attack and bank(obs) < self.p.infantry_bank:
                return ""
            ratio = 1 if self.seen_air else self.p.rifles_per_rocket
            if "e3" in options and units["e3"] * ratio < max(1, units["e1"] + units["e2"]):
                return pick("e3")
            if "shok" in options and units["shok"] * 3 < infantry + 1:
                return pick("shok")
            if "e2" in options and units["e2"] * 2 < units["e1"]:
                return pick("e2")
            return pick("e1") if "e1" in options else ""

        if req.queue == "Vehicle":
            tanks = units["2tnk"] + units["1tnk"] + units["3tnk"] + units["4tnk"] + units["ctnk"]
            if "harv" in options and units["harv"] < min(self.p.max_harv, self.p.harv_per_proc * have["proc"]):
                return pick("harv")

            if "mcv" in options and have["fact"] == 0 and units["mcv"] == 0:
                return pick("mcv")

            tanks = units["2tnk"] + units["1tnk"] + units["3tnk"] + units["4tnk"] + units["ctnk"]
            flaks = units["ftrk"]
            target_flaks = min(4, max(1, tanks // 3))
            if self.seen_air:
                target_flaks = max(target_flaks, 3)
            if tanks >= 3 and flaks < target_flaks and "ftrk" in options:
                return pick("ftrk")
            if "jeep" in options and self.harass_built < 1 and have["proc"] >= 2:
                self.harass_built += 1
                return pick("jeep")

            arties = units["arty"] + units["v2rl"]
            siege_mode = len(self.seen_defenses) >= self.p.siege_def_threshold
            arty_ratio = self.p.siege_arty_every if siege_mode else self.p.arty_every
            if tanks >= arty_ratio * (arties + 1):
                for siege_unit in ("v2rl", "arty"):
                    if siege_unit in options:
                        return pick(siege_unit)

            if "4tnk" in options and (bank(obs) >= 2200 or siege_mode) and units["4tnk"] * 2 <= units["3tnk"] + 1:
                return pick("4tnk")
            if "ctnk" in options and bank(obs) >= 2000 and units["ctnk"] * 3 <= units["2tnk"] + 1:
                return pick("ctnk")
            for tank in ("3tnk", "2tnk", "1tnk", "4tnk"):
                if tank in options:
                    return pick(tank)
            return ""

        if req.queue == "Aircraft":
            # Strict Rearm Rule: 1 helicopter per hpad, 1 fixed-wing plane per afld.
            if have["hpad"] > 0 and (units["heli"] + units["mh60"] + units["hind"] < have["hpad"]):
                for heli in ("heli", "mh60", "hind"):
                    if heli in options:
                        return pick(heli)
            if have["afld"] > 0 and (units["yak"] + units["mig"] < have["afld"]):
                if "mig" in options and units["mig"] <= units["yak"]:
                    return pick("mig")
                for plane in ("yak", "mig"):
                    if plane in options:
                        return pick(plane)
            return ""

        return ""

    # ── tactical heads ───────────────────────────────────────────────────────

    def decide_main(
        self,
        req: pb.DecisionRequest,
        obs: pb.GameObservation,
        units: Counter,
        threat: int = 0,
    ) -> pb.Decision | None:
        index = self.army(req, obs, units, threat)
        if index is None:
            return None
        if index < 0:
            return pb.Decision(request_id=req.id, choice="hold", target_index=-1)
        val = ground_main_value(obs) or army_value(units)
        tgt = req.targets[index]
        crit_buildings = {"fact", "weap", "proc"}
        is_crit = tgt.kind == "enemy_building" and base_type(tgt.type) in crit_buildings
        use_all_in = (
            (val >= self.p.all_in_value or (val >= 8500 and is_crit) or (obs.tick >= 20000 and val >= 5500))
            and tgt.kind == "enemy_building"
            and threat < 1500
        )
        return pb.Decision(
            request_id=req.id,
            choice="all_in" if use_all_in else "attack",
            target_index=index,
        )

    def army(self, req: pb.DecisionRequest, obs: pb.GameObservation, units: Counter, threat: int = 0) -> int | None:
        value = ground_main_value(obs) or army_value(units)
        if not req.targets:
            return None
        if not self.attacking:
            if threat >= 400 and value > 0:
                self.recent_threat_peak = max(self.recent_threat_peak, threat)
                idx = self.aim(req, defending=True)
                if idx is not None and 0 <= idx < len(req.targets):
                    if req.targets[idx].kind == "enemy_mobile":
                        return idx
                return -1

            self.recent_threat_peak = 0

            is_allied = obs.player_faction in ALLIED_FACTIONS
            base_push = 8500 if is_allied else self.p.attack_value
            push_threshold = base_push
            if obs.tick >= 22000:
                push_threshold = min(push_threshold, 6000 if is_allied else 7500)
            elif obs.tick >= 12000:
                push_threshold = min(push_threshold, 8500 if is_allied else 9500)

            if value >= push_threshold:
                self.attacking, self.peak = True, value
                return self.aim(req)
            return None
        self.peak = max(self.peak, value)
        if threat >= 1500 and threat > 0.3 * value:
            self.attacking, self.target = False, None
            self.recalls += 1
            return -1
        if value < self.p.recall_fraction * self.peak:
            self.attacking, self.target = False, None
            self.recalls += 1
            return -1
        self.aim_age += 1
        if self.target not in {(t.kind, t.cell_x, t.cell_y) for t in req.targets}:
            return self.aim(req)
        # If currently tracking an exploratory target but real enemy targets were spotted, retarget immediately!
        if self.target[0] in ("enemy_spawn", "unexplored", "stale"):
            if any(t.kind in ("enemy_building", "enemy_harvester", "enemy_mobile") for t in req.targets):
                return self.aim(req)
        # If currently tracking a harvester but enemy combat units appear, prioritize them!
        if self.target[0] == "enemy_harvester":
            if any(t.kind == "enemy_mobile" for t in req.targets):
                return self.aim(req)
        if self.aim_age >= self.p.max_aim_decisions:
            self.skipped.add(self.target)
            return self.aim(req)
        if home_army_value(obs) >= self.p.reinforce_value:
            return self.aim(req)
        return None

    def decide_harass(self, req: pb.DecisionRequest, obs: pb.GameObservation, threat: int = 0) -> pb.Decision | None:
        if not req.targets:
            return None
        if threat >= 1200 and not self.attacking:
            self.harass_target, self.harass_age = None, 0
            return pb.Decision(request_id=req.id, choice="hold", target_index=-1)

        listed = {(t.kind, t.cell_x, t.cell_y) for t in req.targets}
        if self.harass_target in listed and self.harass_age < 4:
            self.harass_age += 1
            return None
        if self.harass_target in listed and self.harass_age >= 4:
            self.harass_skipped.add(self.harass_target)

        prio = (
            ("unexplored", "enemy_harvester", "stale", "enemy_spawn", "enemy_building", "enemy_mobile")
            if obs.explored_percent < 50.0
            else ("enemy_harvester", "unexplored", "stale", "enemy_spawn", "enemy_building", "enemy_mobile")
        )
        idx = self._pick_by_priority(req, prio, self.harass_skipped)
        if idx is None:
            self.harass_skipped.clear()
            idx = self._pick_by_priority(req, prio, self.harass_skipped) or 0
        t = req.targets[idx]
        self.harass_target = (t.kind, t.cell_x, t.cell_y)
        self.harass_age = 0
        return pb.Decision(request_id=req.id, choice="attack", target_index=idx)

    def decide_air(self, req: pb.DecisionRequest, obs: pb.GameObservation) -> pb.Decision | None:
        if not req.targets:
            return None
        listed = {(t.kind, t.cell_x, t.cell_y) for t in req.targets}
        if self.air_target in listed and self.air_age < 3:
            self.air_age += 1
            return None

        best_i = 0
        best_score = -1e9
        for i, t in enumerate(req.targets):
            aa_near = sum(
                1 for ax, ay in self.aa_locations.values()
                if (ax - t.cell_x) ** 2 + (ay - t.cell_y) ** 2 <= 12 ** 2
            )
            base_score = {
                "enemy_harvester": 50.0,
                "enemy_mobile": 35.0,
                "enemy_building": 30.0,
                "enemy_spawn": 20.0,
                "unexplored": 15.0,
                "stale": 10.0,
            }.get(t.kind, 5.0)
            score = base_score - 18.0 * aa_near - 0.1 * i
            if score > best_score:
                best_score = score
                best_i = i

        t = req.targets[best_i]
        self.air_target = (t.kind, t.cell_x, t.cell_y)
        self.air_age = 0
        return pb.Decision(request_id=req.id, choice="attack", target_index=best_i)

    def decide_support(self, req: pb.DecisionRequest, obs: pb.GameObservation) -> pb.Decision:
        if not req.targets:
            return pb.Decision(request_id=req.id, choice="hold", target_index=-1)

        power = req.queue
        if power == "SovietSpyPlane":
            idx = self._pick_by_priority(
                req, ("unexplored", "stale", "enemy_spawn", "enemy_building", "enemy_harvester")
            )
            return pb.Decision(request_id=req.id, choice="attack", target_index=idx or 0)

        if power in ("GrantExternalConditionPowerInfoOrder", "Chronoshift", "AdvancedChronoshift"):
            if not self.attacking:
                return pb.Decision(request_id=req.id, choice="hold", target_index=-1)
            idx = self._pick_by_priority(req, ("enemy_building", "enemy_mobile", "enemy_spawn"))
            if idx is None:
                return pb.Decision(request_id=req.id, choice="hold", target_index=-1)
            return pb.Decision(request_id=req.id, choice="attack", target_index=idx)

        # Offensive powers: UkraineParabombs, SovietParatroopers, NukePowerInfoOrder
        idx = self._pick_by_priority(
            req, ("enemy_building", "enemy_harvester", "enemy_mobile", "enemy_spawn", "unexplored")
        )
        return pb.Decision(request_id=req.id, choice="attack", target_index=idx or 0)

    def aim(self, req: pb.DecisionRequest, defending: bool = False) -> int | None:
        if not defending and self.target is not None and self.target not in self.skipped:
            is_exploratory = self.target[0] in ("enemy_spawn", "unexplored", "stale")
            has_real_targets = any(t.kind in ("enemy_building", "enemy_harvester", "enemy_mobile") for t in req.targets)
            if is_exploratory and has_real_targets:
                pass
            elif self.target[0] == "enemy_harvester" and any(t.kind == "enemy_mobile" for t in req.targets):
                pass
            else:
                for i, t in enumerate(req.targets):
                    if (t.kind, t.cell_x, t.cell_y) == self.target:
                        return i
        i = self.pick_target(req, self.skipped, defending=defending)
        if i is None:
            if defending:
                return None
            self.skipped.clear()
            i = self.pick_target(req, self.skipped, defending=defending) or 0
        t = req.targets[i]
        new = (t.kind, t.cell_x, t.cell_y)
        if new != self.target:
            self.aim_age = 0
        self.target = new
        return i

    @staticmethod
    def _pick_by_priority(
        req: pb.DecisionRequest,
        kinds: tuple[str, ...],
        skipped: set = frozenset(),
    ) -> int | None:
        for kind in kinds:
            for i, t in enumerate(req.targets):
                if t.kind == kind and (t.kind, t.cell_x, t.cell_y) not in skipped:
                    return i
        return None

    @staticmethod
    def pick_target(req: pb.DecisionRequest, skipped: set = frozenset(), defending: bool = False) -> int | None:
        if defending:
            for i, t in enumerate(req.targets):
                if t.kind == "enemy_mobile" and (t.kind, t.cell_x, t.cell_y) not in skipped:
                    return i
            return None

        # Attacking: kill harvesters and critical production buildings first!
        for i, t in enumerate(req.targets):
            if t.kind == "enemy_harvester" and (t.kind, t.cell_x, t.cell_y) not in skipped:
                return i

        crit = {"fact", "weap", "proc", "barr", "tent", "afld", "hpad"}
        for i, t in enumerate(req.targets):
            if (
                t.kind == "enemy_building"
                and base_type(t.type) in crit
                and (t.kind, t.cell_x, t.cell_y) not in skipped
            ):
                return i

        for i, t in enumerate(req.targets):
            if t.kind == "enemy_building" and (t.kind, t.cell_x, t.cell_y) not in skipped:
                return i

        for i, t in enumerate(req.targets):
            if t.kind == "enemy_spawn" and (t.kind, t.cell_x, t.cell_y) not in skipped:
                return i

        for kind in ("enemy_mobile", "unexplored", "stale"):
            for i, t in enumerate(req.targets):
                if t.kind == kind and (t.kind, t.cell_x, t.cell_y) not in skipped:
                    return i
        return 0 if not skipped else None
