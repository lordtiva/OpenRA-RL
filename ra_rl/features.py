"""Observation -> network inputs, and decisions <-> network actions.

The same code encodes the teacher's decisions when recording and the policy's
inputs when playing, so both see exactly the same features. Only the agent's own
knowledge is used: its units and buildings, what it sees, what it remembers
having seen, and its own past army orders. The spectator summary
(`global_summary.enemy`) is never read.

Per game:  feats.reset() once, then each step  feats.observe(step.obs)  before
encoding that step's requests, and  feats.record(req, decision)  for every
decision actually executed (an ARMY request left unanswered is decision None).
"""

from dataclasses import dataclass

import numpy as np

from ra_env import pb
from ra_rl.vocab import (
    BUILDING_INDEX,
    BUILDINGS,
    COST,
    ITEM_INDEX,
    ITEMS,
    KIND_INDEX,
    QUEUE_INDEX,
    QUEUES,
    TARGET_KINDS,
    UNIT_INDEX,
    UNITS,
    base_type,
    combat_value,
)

FEATURE_VERSION = 1

MAX_TARGETS = 12               # RLCommander.MaxTargets
WAIT = len(ITEMS)              # item head: "produce nothing for now"
N_ITEM_ACTIONS = len(ITEMS) + 1
HOLD = MAX_TARGETS             # army head: call off the attack
KEEP = MAX_TARGETS + 1         # army head: no answer, keep the current orders
N_ARMY_ACTIONS = MAX_TARGETS + 2

HOME_CELLS = 20                # around the base center counts as "home"
NEAR_CELLS = 10                # around a target counts as "at the target"
ENEMY_UNIT_MEMORY = 750        # ticks a seen enemy unit is still counted
ENEMY_BUILDING_MEMORY = 3000   # ticks a seen enemy building is still counted

N_SCALARS = 21
N_MEMORY = 12
GLOBAL_DIM = (N_SCALARS + len(QUEUES) + len(BUILDINGS) + len(UNITS) + len(ITEMS)
              + len(BUILDINGS) + len(UNITS) + N_MEMORY)
CAND_DIM = len(TARGET_KINDS) + len(BUILDINGS) + 8


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


@dataclass
class ArmyMemory:
    """The agent's own army orders (what it decided, not what the bot did)."""

    attacking: bool = False
    attack_tick: int = 0
    order_tick: int = 0
    peak: int = 0
    target: tuple[str, int, int] | None = None
    aim_age: int = 0
    last: int = 0  # last ARMY answer: 0 keep, 1 hold, 2 attack


class Features:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.enemy_units: dict[int, tuple[str, int, tuple[int, int]]] = {}
        self.enemy_buildings: dict[int, tuple[str, int]] = {}
        self.army = ArmyMemory()
        self.tick = 0
        self.g = np.zeros(GLOBAL_DIM, np.float32)
        self.base: tuple[float, float] = (0.0, 0.0)
        self.away_center: tuple[float, float] | None = None
        self.own_units: list[tuple[int, int, int]] = []   # (x, y, combat value)
        self.army_value = 0

    # ── per step ─────────────────────────────────────────────────────────────

    def observe(self, obs: pb.GameObservation) -> None:
        self.tick = tick = obs.tick
        for u in obs.visible_enemies:
            self.enemy_units[u.actor_id] = (base_type(u.type), tick, (u.cell_x, u.cell_y))
        for b in obs.visible_enemy_buildings:
            self.enemy_buildings[b.actor_id] = (base_type(b.type), tick)
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
            self.base = (float(np.mean([u.cell_x for u in obs.units])), float(np.mean([u.cell_y for u in obs.units])))

        self.own_units = [(u.cell_x, u.cell_y, combat_value(u.type)) for u in obs.units]
        self.army_value = sum(v for _, _, v in self.own_units)
        home = sum(v for x, y, v in self.own_units if _dist((x, y), self.base) <= HOME_CELLS)
        away = [(x, y, v) for x, y, v in self.own_units if v and _dist((x, y), self.base) > HOME_CELLS]
        self.away_center = None
        if away:
            w = sum(v for _, _, v in away)
            self.away_center = (sum(x * v for x, _, v in away) / w, sum(y * v for _, y, v in away) / w)
        if self.army.attacking:
            self.army.peak = max(self.army.peak, self.army_value)

        visible_value = sum(combat_value(u.type) for u in obs.visible_enemies)
        threat = sum(combat_value(u.type) for u in obs.visible_enemies
                     if _dist((u.cell_x, u.cell_y), self.base) <= HOME_CELLS)
        recent_value = sum(combat_value(t) for t, _, _ in self.enemy_units.values())

        e, m = obs.economy, obs.military
        scalars = [
            tick / 20000, e.cash / 3000, e.ore / 3000, np.log1p(max(0, e.cash + e.ore)) / 10,
            e.power_provided / 500, e.power_drained / 500,
            np.clip((e.power_provided - e.power_drained) / 200, -3, 3),
            e.resource_capacity / 5000, e.harvester_count / 8, obs.explored_percent / 100,
            m.kills_cost / 20000, m.deaths_cost / 20000,
            self.army_value / 10000, home / 10000, (self.army_value - home) / 10000,
            len(obs.visible_enemies) / 20, visible_value / 10000, threat / 5000, recent_value / 10000,
            float(bool(yards)), len(obs.buildings) / 30,
        ]

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
        for t, _ in self.enemy_buildings.values():
            i = BUILDING_INDEX.get(t)
            if i is not None:
                enemy_b[i] += 1
        enemy_u = np.zeros(len(UNITS), np.float32)
        for t, _, _ in self.enemy_units.values():
            i = UNIT_INDEX.get(t)
            if i is not None:
                enemy_u[i] += 1

        a = self.army
        memory = [
            float(a.attacking), min(2.0, (tick - a.attack_tick) / 5000) if a.attacking else 0.0,
            min(2.0, (tick - a.order_tick) / 2000), a.peak / 10000,
            self.army_value / a.peak if a.attacking and a.peak else 0.0, min(2.0, a.aim_age / 8),
            *(float(a.target is not None and a.target[0] == k) for k in TARGET_KINDS),
            *(float(a.last == i) for i in (1, 2)),
        ]

        self.g = np.concatenate([
            np.asarray(scalars, np.float32), busy,
            np.minimum(own_b, 20) / 4, np.minimum(own_u, 40) / 8, producing,
            np.minimum(enemy_b, 20) / 4, np.minimum(enemy_u, 40) / 8,
            np.asarray(memory, np.float32),
        ]).astype(np.float32)
        assert self.g.shape == (GLOBAL_DIM,)

    # ── per request ──────────────────────────────────────────────────────────

    @staticmethod
    def item_mask(req: pb.DecisionRequest) -> np.ndarray:
        mask = np.zeros(N_ITEM_ACTIONS, bool)
        for o in req.options:
            i = ITEM_INDEX.get(base_type(o))
            if i is not None:
                mask[i] = True
        mask[WAIT] = True
        return mask

    def candidates(self, req: pb.DecisionRequest) -> tuple[np.ndarray, np.ndarray]:
        cand = np.zeros((MAX_TARGETS, CAND_DIM), np.float32)
        mask = np.zeros(N_ARMY_ACTIONS, bool)
        mask[HOLD] = mask[KEEP] = True
        rank: dict[str, int] = {}
        visible = [(c, combat_value(t)) for t, _, c in self.enemy_units.values()]
        for k, t in enumerate(req.targets[:MAX_TARGETS]):
            mask[k] = True
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
            row[o:o + 8] = [
                COST.get(bt, 0) / 2000,
                _dist(cell, self.base) / 100,
                _dist(cell, self.away_center) / 100 if self.away_center else 1.0,
                r / MAX_TARGETS,
                k / MAX_TARGETS,
                sum(v for c, v in visible if _dist(c, cell) <= NEAR_CELLS) / 5000,
                sum(v for x, y, v in self.own_units if _dist((x, y), cell) <= NEAR_CELLS) / 5000,
                float(self.army.target == (t.kind, t.cell_x, t.cell_y)),
            ]
        return cand, mask

    # ── actions ──────────────────────────────────────────────────────────────

    @staticmethod
    def label(req: pb.DecisionRequest, decision: pb.Decision | None) -> int | None:
        """Network action for a decision; None if it is outside the vocabulary."""
        if req.kind == pb.DECIDE_ARMY:
            if decision is None:
                return KEEP
            if decision.target_index < 0:
                return HOLD
            return decision.target_index if decision.target_index < min(MAX_TARGETS, len(req.targets)) else None
        # A choice the queue does not offer starts nothing (the builtin bot asks for an
        # MCV before it can build one): that is a wait.
        if decision is None or decision.choice not in req.options:
            return WAIT
        return ITEM_INDEX.get(base_type(decision.choice))

    @staticmethod
    def decision(req: pb.DecisionRequest, action: int) -> pb.Decision | None:
        """Inverse of label(): the Decision to send (None = leave the ARMY request unanswered)."""
        if req.kind == pb.DECIDE_ARMY:
            if action == KEEP:
                return None
            return pb.Decision(request_id=req.id, target_index=-1 if action == HOLD else action)
        if action == WAIT:
            return pb.Decision(request_id=req.id, choice="")
        name = ITEMS[action]
        choice = next((o for o in req.options if base_type(o) == name), name)
        return pb.Decision(request_id=req.id, choice=choice)

    def record(self, req: pb.DecisionRequest, decision: pb.Decision | None) -> None:
        """Remember an executed decision (only ARMY orders feed the memory)."""
        if req.kind != pb.DECIDE_ARMY:
            return
        a = self.army
        if decision is None:
            a.last = 0
            if a.attacking:
                a.aim_age += 1
            return
        a.order_tick = self.tick
        if decision.target_index < 0:
            a.attacking, a.target, a.last = False, None, 1
            return
        t = req.targets[decision.target_index]
        new = (t.kind, t.cell_x, t.cell_y)
        if not a.attacking:
            a.attacking, a.attack_tick, a.peak = True, self.tick, self.army_value
        a.aim_age = a.aim_age + 1 if new == a.target else 0
        a.target, a.last = new, 2
