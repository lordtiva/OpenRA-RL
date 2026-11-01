"""Observation -> network inputs, and decisions <-> network actions (Commander v4).

The same code encodes the teacher's decisions when recording and the policy's
inputs when playing, so both see exactly the same features. Only the agent's own
knowledge is used in `Features.g`: its units and buildings, its faction, what it
sees, what it remembers having seen, and its own past production/tactical orders.
The spectator summary (`global_summary.enemy`) is NEVER read by `Features.observe`
or `Features.g`.

For Centralized Training with Decentralized Execution (CTDE) in RL (Phases 4.2-4.3),
`privileged_features()` and `state_potential()` separately read `global_summary`
(which is serialized player-relatively for both Multi1 and Multi0) for the
training-only critic and exact zero-sum Potential-Based Reward Shaping (PBRS).
"""

from dataclasses import dataclass

import numpy as np

from ra_env import pb
from ra_rl.vocab import (
    AIRCRAFT,
    ALLIED_FACTIONS,
    ALLIED_ONLY_ACTORS,
    ARMY_GROUPS,
    BUILDING_INDEX,
    BUILDINGS,
    COST,
    FACTION_INDEX,
    FACTIONS,
    ITEM_INDEX,
    ITEMS,
    KIND_INDEX,
    QUEUE_INDEX,
    QUEUES,
    SOVIET_FACTIONS,
    SOVIET_ONLY_ACTORS,
    TACTICAL_INDEX,
    TARGET_KINDS,
    UNIT_INDEX,
    UNITS,
    base_type,
    combat_value,
)

FEATURE_VERSION = 3

MAX_TARGETS = 16                     # RLCommander.MaxTargets in v4
WAIT = len(ITEMS)                    # item head: "produce nothing for now"
N_ITEM_ACTIONS = len(ITEMS) + 1

ALL_IN_OFFSET = MAX_TARGETS          # 16..31: fearless assault on targets[k] (MAIN only)
HOLD = 2 * MAX_TARGETS               # 32: recall/hold (MAIN/AIR), merge (HARASS), wait (SUPPORT)
KEEP = 2 * MAX_TARGETS + 1           # 33: leave ARMY request unanswered (keep current orders)
N_ARMY_ACTIONS = 2 * MAX_TARGETS + 2

HOME_CELLS = 20                      # around the base center counts as "home"
NEAR_CELLS = 10                      # around a target counts as "at the target"
AA_CELLS = 12                        # anti-air threat radius around candidate cell
ENEMY_UNIT_MEMORY = 750              # ticks a seen enemy unit is still counted
ENEMY_BUILDING_MEMORY = 3000         # ticks a seen enemy building is still counted

STATIC_DEFENSES = frozenset({"pbox", "hbox", "gun", "ftur", "tsla", "agun", "sam"})
AA_ACTORS = frozenset({"sam", "agun", "ftrk", "e3"})
HEAVY_ARMOR = frozenset({"2tnk", "3tnk", "4tnk", "v2rl", "ttnk", "ctnk", "qtnk", "stnk"})
AIR_COMBAT_SET = frozenset({"heli", "hind", "mh60", "yak", "mig"})
WALL_ITEMS = frozenset({"sbag", "fenc", "brik"})
FORBIDDEN_TRAIN_ITEMS = frozenset({"tran", "truk", "e6", "spy", "thf", "mnly"})

# Sanity caps for non-combat / redundant / rearm-bound spam in external commander mode.
MAX_HARVESTERS = 10
MIN_HARVESTER_CAP = 4
MAX_BARRACKS = 4
MAX_LIGHT_DEFENSES = 6
MAX_HEAVY_DEFENSES = 8
MAX_AA_DEFENSES = 6
MAX_AIR_PADS = 4
MAX_MCV_TOTAL = 2

N_SCALARS = 45
N_MEMORY = 23
GLOBAL_DIM = (
    N_SCALARS
    + len(FACTIONS)
    + len(QUEUES)
    + len(BUILDINGS)
    + len(UNITS)
    + len(ITEMS)
    + len(BUILDINGS)
    + len(UNITS)
    + N_MEMORY
)
CAND_DIM = len(TARGET_KINDS) + len(BUILDINGS) + 12

# Training-only asymmetric critic inputs (never used by the actor or at eval time).
OPPONENTS = ("hard", "brutal", "turtle", "selfplay", "snapshot", "bc", "other")
OPPONENT_INDEX = {
    "hard": 0, "normal": 0,
    "brutal": 1, "rush": 1,
    "turtle": 2,
    "selfplay": 3, "mirror": 3,
    "snapshot": 4, "pfsp": 4,
    "bc": 5, "scripted": 5, "anchor": 5,
    "other": 6, "medium": 6, "easy": 6, "beginner": 6, "naval": 6,
}
N_PRIV_SCALARS = 14
PRIV_DIM = N_PRIV_SCALARS + len(OPPONENTS)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def opponent_id(bot_name: str) -> int:
    return OPPONENT_INDEX.get(bot_name, OPPONENT_INDEX["other"])


def tactical_group_name(req: pb.DecisionRequest) -> str:
    if req.kind == pb.DECIDE_SUPPORT:
        return req.queue if req.queue in TACTICAL_INDEX else "OTHER_SUPPORT"
    q = req.queue or "MAIN"
    return q if q in ARMY_GROUPS else "MAIN"


def tactical_channel_id(req: pb.DecisionRequest) -> int:
    return TACTICAL_INDEX[tactical_group_name(req)]


def is_tactical_request(req: pb.DecisionRequest) -> bool:
    return req.kind in (pb.DECIDE_ARMY, pb.DECIDE_SUPPORT)


def privileged_features(obs: pb.GameObservation, bot_name: str = "other") -> np.ndarray:
    """Training-only spectator features for the asymmetric critic V(g, priv)."""
    own = obs.global_summary.own
    ene = obs.global_summary.enemy

    own_u = float(max(0, own.unit_value))
    own_b = float(max(0, own.building_value))
    own_nb = float(max(0, own.n_buildings))
    own_earn = float(max(0, own.earned))

    ene_cash = float(max(0, ene.cash))
    ene_u = float(max(0, ene.unit_value))
    ene_b = float(max(0, ene.building_value))
    ene_nb = float(max(0, ene.n_buildings))
    ene_earn = float(max(0, ene.earned))

    own_tot = own_u + own_b
    ene_tot = ene_u + ene_b

    scalars = np.asarray([
        np.clip(ene_cash / 5000.0, 0.0, 4.0),
        np.log1p(ene_cash) / 10.0,
        np.clip(ene_u / 10000.0, 0.0, 4.0),
        np.clip(ene_b / 10000.0, 0.0, 4.0),
        np.clip(ene_nb / 20.0, 0.0, 3.0),
        np.clip(ene_earn / 20000.0, 0.0, 4.0),
        np.clip(own_earn / 20000.0, 0.0, 4.0),
        np.clip((own_u - ene_u) / 10000.0, -3.0, 3.0),
        own_u / max(500.0, own_u + ene_u),
        np.clip((own_b - ene_b) / 10000.0, -3.0, 3.0),
        own_b / max(500.0, own_b + ene_b),
        np.clip((own_nb - ene_nb) / 15.0, -3.0, 3.0),
        np.clip((own_earn - ene_earn) / 15000.0, -3.0, 3.0),
        own_tot / max(1000.0, own_tot + ene_tot),
    ], dtype=np.float32)

    opp_oh = np.zeros(len(OPPONENTS), dtype=np.float32)
    opp_oh[opponent_id(bot_name)] = 1.0
    out = np.concatenate([scalars, opp_oh]).astype(np.float32)
    assert out.shape == (PRIV_DIM,)
    return out


def state_potential(obs: pb.GameObservation) -> float:
    """Bounded zero-sum symmetric potential Phi(s) in [-1, 1] for exact PBRS."""
    own = obs.global_summary.own
    ene = obs.global_summary.enemy
    own_b = float(max(0, own.building_value))
    ene_b = float(max(0, ene.building_value))
    own_u = float(max(0, own.unit_value))
    ene_u = float(max(0, ene.unit_value))

    if own_b == 0.0 and ene_b == 0.0 and own_u == 0.0 and ene_u == 0.0:
        return 0.0

    bld_adv = (own_b - ene_b) / max(1500.0, own_b + ene_b)
    own_tot = own_b + own_u
    ene_tot = ene_b + ene_u
    tot_adv = (own_tot - ene_tot) / max(2000.0, own_tot + ene_tot)
    return float(np.clip(0.65 * bld_adv + 0.35 * tot_adv, -1.0, 1.0))


@dataclass
class ArmyMemory:
    """The agent's own squad orders for a tactical group (MAIN, HARASS, AIR)."""

    attacking: bool = False
    all_in: bool = False
    attack_tick: int = 0
    order_tick: int = 0
    peak: int = 0
    target: tuple[str, int, int] | None = None
    aim_age: int = 0
    last: int = 0  # 0 keep, 1 hold, 2 attack, 3 all_in


class Features:
    def __init__(self) -> None:
        self.reset()

    @property
    def army(self) -> ArmyMemory:
        return self.armies["MAIN"]

    def reset(self) -> None:
        self.enemy_units: dict[int, tuple[str, int, tuple[int, int]]] = {}
        self.enemy_buildings: dict[int, tuple[str, int, tuple[int, int]]] = {}
        self.seen_enemy_def_ids: set[int] = set()
        self.seen_enemy_aa_ids: set[int] = set()
        self.seen_enemy_heavy_ids: set[int] = set()
        self.seen_enemy_air: bool = False
        self.seen_enemy_allies: bool = False
        self.seen_enemy_soviet: bool = False
        self.early_rush_seen: bool = False
        self.max_early_threat: float = 0.0

        self.armies: dict[str, ArmyMemory] = {grp: ArmyMemory() for grp in ARMY_GROUPS}
        self.last_build_tick: int = 0
        self.consec_build_waits: int = 0
        self.ready_idle_tick: int = 0

        self.tick = 0
        self.g = np.zeros(GLOBAL_DIM, np.float32)
        self._own_b = np.zeros(len(BUILDINGS), np.float32)
        self._own_u = np.zeros(len(UNITS), np.float32)
        self._producing = np.zeros(len(ITEMS), np.float32)
        self.base: tuple[float, float] = (0.0, 0.0)
        self.away_center: tuple[float, float] | None = None
        self.harass_center: tuple[float, float] | None = None
        self.air_center: tuple[float, float] | None = None
        self.own_units: list[tuple[int, int, int]] = []   # (x, y, combat value)
        self.army_value = 0
        self.main_value = 0
        self.harass_value = 0
        self.air_value = 0

    # ── per step ─────────────────────────────────────────────────────────────

    def observe(self, obs: pb.GameObservation) -> None:
        self.tick = tick = obs.tick
        for u in obs.visible_enemies:
            ut = base_type(u.type)
            self.enemy_units[u.actor_id] = (ut, tick, (u.cell_x, u.cell_y))
            if ut in HEAVY_ARMOR:
                self.seen_enemy_heavy_ids.add(u.actor_id)
            if ut in AA_ACTORS:
                self.seen_enemy_aa_ids.add(u.actor_id)
            if ut in AIR_COMBAT_SET:
                self.seen_enemy_air = True
            if ut in ALLIED_ONLY_ACTORS:
                self.seen_enemy_allies = True
            elif ut in SOVIET_ONLY_ACTORS:
                self.seen_enemy_soviet = True

        for b in obs.visible_enemy_buildings:
            bt = base_type(b.type)
            self.enemy_buildings[b.actor_id] = (bt, tick, (b.cell_x, b.cell_y))
            if bt in STATIC_DEFENSES:
                self.seen_enemy_def_ids.add(b.actor_id)
            if bt in AA_ACTORS:
                self.seen_enemy_aa_ids.add(b.actor_id)
            if bt in ("hpad", "afld"):
                self.seen_enemy_air = True
            if bt in ALLIED_ONLY_ACTORS:
                self.seen_enemy_allies = True
            elif bt in SOVIET_ONLY_ACTORS:
                self.seen_enemy_soviet = True

        self.enemy_units = {k: v for k, v in self.enemy_units.items() if tick - v[1] <= ENEMY_UNIT_MEMORY}
        self.enemy_buildings = {k: v for k, v in self.enemy_buildings.items()
                                if tick - v[1] <= ENEMY_BUILDING_MEMORY}

        yards = [(b.cell_x, b.cell_y) for b in obs.buildings if base_type(b.type) == "fact"]
        if yards:
            self.base = yards[0]
        elif obs.buildings:
            self.base = (float(np.mean([b.cell_x for b in obs.buildings])),
                         float(np.mean([b.cell_y for b in obs.buildings])))
        elif obs.units:
            self.base = (float(np.mean([u.cell_x for u in obs.units])),
                         float(np.mean([u.cell_y for u in obs.units])))

        harass_ids = set(obs.harass_unit_ids)
        self.own_units = [(u.cell_x, u.cell_y, combat_value(u.type)) for u in obs.units]
        self.army_value = sum(v for _, _, v in self.own_units)

        main_pts: list[tuple[int, int, int]] = []
        harass_pts: list[tuple[int, int, int]] = []
        air_pts: list[tuple[int, int, int]] = []
        for u in obs.units:
            v = combat_value(u.type)
            if v <= 0:
                continue
            ut = base_type(u.type)
            if ut in AIRCRAFT:
                air_pts.append((u.cell_x, u.cell_y, v))
            elif u.actor_id in harass_ids:
                harass_pts.append((u.cell_x, u.cell_y, v))
            else:
                main_pts.append((u.cell_x, u.cell_y, v))

        self.main_value = sum(v for _, _, v in main_pts)
        self.harass_value = sum(v for _, _, v in harass_pts)
        self.air_value = sum(v for _, _, v in air_pts)

        home = sum(v for x, y, v in main_pts if _dist((x, y), self.base) <= HOME_CELLS)
        away = [(x, y, v) for x, y, v in main_pts if _dist((x, y), self.base) > HOME_CELLS]
        self.away_center = (
            (sum(x * v for x, _, v in away) / sum(v for _, _, v in away),
             sum(y * v for _, y, v in away) / sum(v for _, _, v in away))
            if away else None
        )
        self.harass_center = (
            (sum(x * v for x, _, v in harass_pts) / self.harass_value,
             sum(y * v for _, y, v in harass_pts) / self.harass_value)
            if self.harass_value > 0 else None
        )
        self.air_center = (
            (sum(x * v for x, _, v in air_pts) / self.air_value,
             sum(y * v for _, y, v in air_pts) / self.air_value)
            if self.air_value > 0 else None
        )

        a_main = self.armies["MAIN"]
        if a_main.attacking:
            a_main.peak = max(a_main.peak, self.main_value)
        if a_main.attacking or self.main_value < 4000:
            self.ready_idle_tick = tick

        visible_value = sum(combat_value(u.type) for u in obs.visible_enemies)
        threat = sum(combat_value(u.type) for u in obs.visible_enemies
                     if _dist((u.cell_x, u.cell_y), self.base) <= HOME_CELLS)
        recent_value = sum(combat_value(t) for t, _, _ in self.enemy_units.values())

        if tick <= 4500 and threat > 0:
            self.early_rush_seen = True
        if tick <= 6000:
            self.max_early_threat = max(self.max_early_threat, float(threat))

        busy = np.zeros(len(QUEUES), np.float32)
        producing = np.zeros(len(ITEMS), np.float32)
        for p in obs.production:
            if p.queue_type in QUEUE_INDEX:
                busy[QUEUE_INDEX[p.queue_type]] = 1
            i = ITEM_INDEX.get(base_type(p.item))
            if i is not None:
                producing[i] += 1

        own_b = np.zeros(len(BUILDINGS), np.float32)
        for b in obs.buildings:
            i = BUILDING_INDEX.get(base_type(b.type))
            if i is not None:
                own_b[i] += 1
        own_u = np.zeros(len(UNITS), np.float32)
        for u in obs.units:
            i = UNIT_INDEX.get(base_type(u.type))
            if i is not None:
                own_u[i] += 1
        enemy_b = np.zeros(len(BUILDINGS), np.float32)
        for t, _, _ in self.enemy_buildings.values():
            i = BUILDING_INDEX.get(t)
            if i is not None:
                enemy_b[i] += 1
        enemy_u = np.zeros(len(UNITS), np.float32)
        for t, _, _ in self.enemy_units.values():
            i = UNIT_INDEX.get(t)
            if i is not None:
                enemy_u[i] += 1

        self._own_b = own_b
        self._own_u = own_u
        self._producing = producing

        faction_oh = np.zeros(len(FACTIONS), np.float32)
        fac = (obs.player_faction or "").lower()
        if fac in FACTION_INDEX:
            faction_oh[FACTION_INDEX[fac]] = 1.0
        is_allies = float(
            fac in ALLIED_FACTIONS
            or (not fac and (own_b[BUILDING_INDEX["tent"]] + own_u[UNIT_INDEX["2tnk"]] > 0))
        )
        is_soviet = float(
            fac in SOVIET_FACTIONS
            or (not fac and (own_b[BUILDING_INDEX["barr"]] + own_u[UNIT_INDEX["3tnk"]] > 0))
        )
        enemy_side_unknown = float(not self.seen_enemy_allies and not self.seen_enemy_soviet)

        own_proc = float(own_b[BUILDING_INDEX["proc"]])
        own_harv_total = float(own_u[UNIT_INDEX["harv"]] + producing[ITEM_INDEX["harv"]])
        has_weap = float(own_b[BUILDING_INDEX["weap"]] + producing[ITEM_INDEX["weap"]] > 0)
        has_radar = float(own_b[BUILDING_INDEX["dome"]] > 0)
        has_tech = float(own_b[BUILDING_INDEX["atek"]] + own_b[BUILDING_INDEX["stek"]] > 0)
        own_pads = float(own_b[BUILDING_INDEX["hpad"]] + own_b[BUILDING_INDEX["afld"]])

        e, m = obs.economy, obs.military
        scalars = [
            tick / 20000, e.cash / 3000, e.ore / 3000, np.log1p(max(0, e.cash + e.ore)) / 10,
            e.power_provided / 500, e.power_drained / 500,
            np.clip((e.power_provided - e.power_drained) / 200, -3, 3),
            e.resource_capacity / 5000, e.harvester_count / 8, obs.explored_percent / 100,
            m.kills_cost / 20000, m.deaths_cost / 20000,
            self.army_value / 10000, home / 10000, (self.main_value - home) / 10000,
            len(obs.visible_enemies) / 20, visible_value / 10000, threat / 5000, recent_value / 10000,
            float(bool(yards)), len(obs.buildings) / 30,
            # Economy saturation, tech gates & persistent fog-honest scouting
            np.clip(own_harv_total / max(1.0, 2.0 * own_proc), 0.0, 3.0),
            np.clip((own_harv_total - 2.0 * own_proc) / 4.0, -2.0, 2.0),
            has_weap,
            has_radar,
            has_tech,
            float(self.early_rush_seen),
            min(2.0, self.max_early_threat / 5000.0),
            min(3.0, len(self.seen_enemy_def_ids) / 6.0),
            min(3.0, len(self.seen_enemy_heavy_ids) / 10.0),
            np.clip((m.kills_cost - m.deaths_cost) / 10000.0, -2.0, 2.0),
            # v4 bilateral faction, multi-squad, air, AA & expansion saturation signals
            is_allies,
            is_soviet,
            float(self.seen_enemy_allies),
            float(self.seen_enemy_soviet),
            enemy_side_unknown,
            self.main_value / 10000.0,
            self.harass_value / 3000.0,
            len(harass_pts) / 4.0,
            self.air_value / 6000.0,
            len(air_pts) / 4.0,
            own_pads / 4.0,
            min(3.0, len(self.seen_enemy_aa_ids) / 6.0),
            float(self.seen_enemy_air),
            min(3.0, float(obs.placement_failures) / 5.0),
        ]

        ready_idle = (
            min(2.0, max(0, tick - self.ready_idle_tick) / 2500.0)
            if (not a_main.attacking and self.main_value >= 4000)
            else 0.0
        )
        a_har = self.armies["HARASS"]
        a_air = self.armies["AIR"]
        memory = [
            float(a_main.attacking),
            float(a_main.all_in),
            min(2.0, (tick - a_main.attack_tick) / 5000) if a_main.attacking else 0.0,
            min(2.0, (tick - a_main.order_tick) / 2000),
            a_main.peak / 10000,
            self.main_value / a_main.peak if a_main.attacking and a_main.peak else 0.0,
            min(2.0, a_main.aim_age / 8),
            *(float(a_main.target is not None and a_main.target[0] == k) for k in TARGET_KINDS),
            *(float(a_main.last == i) for i in (1, 2, 3)),
            min(2.0, max(0, tick - self.last_build_tick) / 2000.0),
            min(2.0, self.consec_build_waits / 5.0),
            ready_idle,
            float(a_har.attacking),
            min(2.0, a_har.aim_age / 8),
            float(a_air.attacking),
            min(2.0, a_air.aim_age / 8),
        ]

        self.g = np.concatenate([
            np.asarray(scalars, np.float32),
            faction_oh,
            busy,
            np.minimum(own_b, 20) / 4,
            np.minimum(own_u, 40) / 8,
            producing,
            np.minimum(enemy_b, 20) / 4,
            np.minimum(enemy_u, 40) / 8,
            np.asarray(memory, np.float32),
        ]).astype(np.float32)
        assert self.g.shape == (GLOBAL_DIM,)

    # ── per request ──────────────────────────────────────────────────────────

    def item_mask(self, req: pb.DecisionRequest) -> np.ndarray:
        mask = np.zeros(N_ITEM_ACTIONS, bool)
        for o in req.options:
            bt = base_type(o)
            if bt in WALL_ITEMS or bt in FORBIDDEN_TRAIN_ITEMS:
                continue
            i = ITEM_INDEX.get(bt)
            if i is not None:
                mask[i] = True

        # Sanity caps on non-combat / redundant spam (external mode bypasses YAML UnitLimits).
        own_proc = int(self._own_b[BUILDING_INDEX["proc"]])
        harv_cap = min(MAX_HARVESTERS, max(MIN_HARVESTER_CAP, 2 * own_proc + 2))
        if self._own_u[UNIT_INDEX["harv"]] + self._producing[ITEM_INDEX["harv"]] >= harv_cap:
            mask[ITEM_INDEX["harv"]] = False

        mcv_total = (self._own_b[BUILDING_INDEX["fact"]]
                     + self._own_u[UNIT_INDEX["mcv"]]
                     + self._producing[ITEM_INDEX["mcv"]])
        if mcv_total >= MAX_MCV_TOTAL:
            mask[ITEM_INDEX["mcv"]] = False

        barr_total = (self._own_b[BUILDING_INDEX["tent"]] + self._own_b[BUILDING_INDEX["barr"]]
                      + self._producing[ITEM_INDEX["tent"]] + self._producing[ITEM_INDEX["barr"]])
        if barr_total >= MAX_BARRACKS:
            mask[ITEM_INDEX["tent"]] = False
            mask[ITEM_INDEX["barr"]] = False

        for singleton in ("kenn", "dome", "fix", "atek", "stek"):
            if self._own_b[BUILDING_INDEX[singleton]] + self._producing[ITEM_INDEX[singleton]] >= 1:
                mask[ITEM_INDEX[singleton]] = False
        if self._own_b[BUILDING_INDEX["silo"]] + self._producing[ITEM_INDEX["silo"]] >= 2:
            mask[ITEM_INDEX["silo"]] = False

        light_def = (
            self._own_b[BUILDING_INDEX["pbox"]] + self._own_b[BUILDING_INDEX["hbox"]] + self._own_b[BUILDING_INDEX["ftur"]]
            + self._producing[ITEM_INDEX["pbox"]] + self._producing[ITEM_INDEX["hbox"]] + self._producing[ITEM_INDEX["ftur"]]
        )
        if light_def >= MAX_LIGHT_DEFENSES:
            for d_name in ("pbox", "hbox", "ftur"):
                mask[ITEM_INDEX[d_name]] = False

        heavy_def = (
            self._own_b[BUILDING_INDEX["gun"]] + self._own_b[BUILDING_INDEX["tsla"]]
            + self._producing[ITEM_INDEX["gun"]] + self._producing[ITEM_INDEX["tsla"]]
        )
        if heavy_def >= MAX_HEAVY_DEFENSES:
            mask[ITEM_INDEX["gun"]] = False
            mask[ITEM_INDEX["tsla"]] = False

        aa_def = (
            self._own_b[BUILDING_INDEX["agun"]] + self._own_b[BUILDING_INDEX["sam"]]
            + self._producing[ITEM_INDEX["agun"]] + self._producing[ITEM_INDEX["sam"]]
        )
        if aa_def >= MAX_AA_DEFENSES:
            mask[ITEM_INDEX["agun"]] = False
            mask[ITEM_INDEX["sam"]] = False

        # Aircraft Rearm Rule: every yak/mig requires 1 dedicated afld; every heli/mh60/hind requires 1 hpad.
        own_hpad = int(self._own_b[BUILDING_INDEX["hpad"]])
        if own_hpad + int(self._producing[ITEM_INDEX["hpad"]]) >= MAX_AIR_PADS:
            mask[ITEM_INDEX["hpad"]] = False
        heli_total = sum(
            self._own_u[UNIT_INDEX[u]] + self._producing[ITEM_INDEX[u]]
            for u in ("heli", "mh60", "hind")
        )
        if heli_total >= own_hpad:
            for u in ("heli", "mh60", "hind"):
                mask[ITEM_INDEX[u]] = False

        own_afld = int(self._own_b[BUILDING_INDEX["afld"]])
        if own_afld + int(self._producing[ITEM_INDEX["afld"]]) >= MAX_AIR_PADS:
            mask[ITEM_INDEX["afld"]] = False
        plane_total = sum(
            self._own_u[UNIT_INDEX[u]] + self._producing[ITEM_INDEX[u]]
            for u in ("yak", "mig")
        )
        if plane_total >= own_afld:
            mask[ITEM_INDEX["yak"]] = False
            mask[ITEM_INDEX["mig"]] = False

        for sup_unit, cap in (("dog", 4), ("medi", 2), ("mech", 2), ("apc", 2), ("mrj", 1), ("mgg", 1)):
            if self._own_u[UNIT_INDEX[sup_unit]] + self._producing[ITEM_INDEX[sup_unit]] >= cap:
                mask[ITEM_INDEX[sup_unit]] = False

        mask[WAIT] = True
        return mask

    def candidates(self, req: pb.DecisionRequest) -> tuple[np.ndarray, np.ndarray]:
        cand = np.zeros((MAX_TARGETS, CAND_DIM), np.float32)
        mask = np.zeros(N_ARMY_ACTIONS, bool)
        mask[HOLD] = True
        is_support = req.kind == pb.DECIDE_SUPPORT
        grp = tactical_group_name(req)
        allow_all_in = (not is_support) and (grp == "MAIN")
        if not is_support:
            mask[KEEP] = True

        grp_mem = self.armies.get(grp, self.armies["MAIN"])
        main_mem = self.armies["MAIN"]

        rank: dict[str, int] = {}
        visible = [(c, combat_value(t)) for t, _, c in self.enemy_units.values()]
        aa_sources: list[tuple[tuple[int, int], float]] = [
            (c, float(COST.get(t, 300))) for t, _, c in self.enemy_units.values() if t in AA_ACTORS
        ] + [
            (c, float(COST.get(t, 700))) for t, _, c in self.enemy_buildings.values() if t in AA_ACTORS
        ]

        for k, t in enumerate(req.targets[:MAX_TARGETS]):
            mask[k] = True
            if allow_all_in:
                mask[ALL_IN_OFFSET + k] = True
            cell = (t.cell_x, t.cell_y)
            row = cand[k]
            if t.kind in KIND_INDEX:
                row[KIND_INDEX[t.kind]] = 1
            o = len(TARGET_KINDS)
            bt = base_type(t.type)
            if bt in BUILDING_INDEX:
                row[o + BUILDING_INDEX[bt]] = 1
            o += len(BUILDINGS)
            r = rank.get(t.kind, 0)
            rank[t.kind] = r + 1
            row[o:o + 12] = [
                COST.get(bt, 0) / 2000.0,
                _dist(cell, self.base) / 100.0,
                _dist(cell, self.away_center) / 100.0 if self.away_center else 1.0,
                _dist(cell, self.harass_center) / 100.0 if self.harass_center else 1.0,
                _dist(cell, self.air_center) / 100.0 if self.air_center else 1.0,
                r / MAX_TARGETS,
                k / MAX_TARGETS,
                sum(v for c, v in visible if _dist(c, cell) <= NEAR_CELLS) / 5000.0,
                sum(v for c, v in aa_sources if _dist(c, cell) <= AA_CELLS) / 3000.0,
                sum(v for x, y, v in self.own_units if _dist((x, y), cell) <= NEAR_CELLS) / 5000.0,
                float(grp_mem.target == (t.kind, t.cell_x, t.cell_y)),
                float(main_mem.target == (t.kind, t.cell_x, t.cell_y)),
            ]
        return cand, mask

    # ── actions ──────────────────────────────────────────────────────────────

    @staticmethod
    def label(req: pb.DecisionRequest, decision: pb.Decision | None) -> int | None:
        """Network action for a decision; None if it is outside the vocabulary."""
        if req.kind == pb.DECIDE_SUPPORT:
            if decision is None or decision.target_index < 0 or decision.choice == "hold":
                return HOLD
            return decision.target_index if decision.target_index < min(MAX_TARGETS, len(req.targets)) else None

        if req.kind == pb.DECIDE_ARMY:
            if decision is None:
                return KEEP
            if decision.target_index < 0 or decision.choice == "hold":
                return HOLD
            if decision.target_index >= min(MAX_TARGETS, len(req.targets)):
                return None
            if decision.choice == "all_in" and tactical_group_name(req) == "MAIN":
                return ALL_IN_OFFSET + decision.target_index
            return decision.target_index

        if decision is None or decision.choice not in req.options:
            return WAIT
        bt = base_type(decision.choice)
        if bt in WALL_ITEMS or bt in FORBIDDEN_TRAIN_ITEMS:
            return WAIT
        return ITEM_INDEX.get(bt)

    @staticmethod
    def decision(req: pb.DecisionRequest, action: int) -> pb.Decision | None:
        """Inverse of label(): the Decision to send (None = leave ARMY request unanswered)."""
        if req.kind == pb.DECIDE_SUPPORT:
            if action == HOLD or action == KEEP:
                return pb.Decision(request_id=req.id, choice="hold", target_index=-1)
            idx = action - ALL_IN_OFFSET if ALL_IN_OFFSET <= action < HOLD else action
            return pb.Decision(request_id=req.id, choice="attack", target_index=idx)

        if req.kind == pb.DECIDE_ARMY:
            if action == KEEP:
                return None
            if action == HOLD:
                return pb.Decision(request_id=req.id, choice="hold", target_index=-1)
            if ALL_IN_OFFSET <= action < HOLD:
                return pb.Decision(request_id=req.id, choice="all_in", target_index=action - ALL_IN_OFFSET)
            return pb.Decision(request_id=req.id, choice="attack", target_index=action)

        if action == WAIT:
            return pb.Decision(request_id=req.id, choice="")
        name = ITEMS[action]
        choice = next((o for o in req.options if base_type(o) == name), name)
        return pb.Decision(request_id=req.id, choice=choice)

    def record(self, req: pb.DecisionRequest, decision: pb.Decision | None) -> None:
        """Remember an executed decision (Building waits + tactical squad orders feed memory)."""
        if req.kind == pb.DECIDE_BUILD and req.queue == "Building":
            if decision is None or not decision.choice:
                self.consec_build_waits += 1
            else:
                self.consec_build_waits = 0
                self.last_build_tick = self.tick
            return

        if req.kind != pb.DECIDE_ARMY:
            return

        grp = tactical_group_name(req)
        a = self.armies[grp]
        if decision is None:
            a.last = 0
            if a.attacking:
                a.aim_age += 1
            return
        a.order_tick = self.tick
        if decision.target_index < 0 or decision.choice == "hold":
            a.attacking, a.all_in, a.target, a.last = False, False, None, 1
            return
        if decision.target_index >= len(req.targets):
            return
        t = req.targets[decision.target_index]
        new = (t.kind, t.cell_x, t.cell_y)
        grp_val = (
            self.harass_value if grp == "HARASS"
            else self.air_value if grp == "AIR"
            else self.main_value
        )
        if not a.attacking:
            a.attacking, a.attack_tick, a.peak = True, self.tick, grp_val
        a.all_in = decision.choice == "all_in"
        a.aim_age = a.aim_age + 1 if new == a.target else 0
        a.target, a.last = new, (3 if a.all_in else 2)
