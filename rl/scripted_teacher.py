"""Maestro BC para Capa 1: ScriptedBot alineado al MDP de train.

No clona el ModularBot hard (IOrder ≠ ActionIndex). Copia *estrategia*
leíble de `mods/ra/rules/ai.yaml`:

  - beginner: 1 harv, SquadSize 3, MinimumAttackForceDelay 150000
    (casi no rushea en a_short @53k)
  - rush: ataca en cuanto hay squad, no espera weap/tech
  - normal/hard: 4 harvs + weap + SquadSize 20 — demasiado lento acá

El example original: barracks antes de proc, 2 scouts, pack 12 idle en
casa hasta army_attack_move, weap en el build order. Eso mill@52k.

Este teacher:
  - proc antes de barracks
  - rush: powr → proc → tent. Sin weap en el camino crítico
  - a 8 rifles: attack_move de TODO el idle (legal sin pack mask)
  - a PACK_ARMY (12): army_attack_move (lo que el alumno puede emitir)
  - hunt map-agnostic: home raid > visible leftover > mental enemy-base
    belief > last_seen ghosts > hunt/sweep cerca del último contacto >
    fog scout. Never resolve_beacon / BEACON_BY_MAP (tapes
    eco_and_combat_scout_v1).
  - early fog scout away from home once a few combat exist (relative fog).
  - peel de raid; TRAIN e1 durante el push; 2 harvs; 0 guards / 0 APC
  - P3 land-first tech/defense AFTER eco/barracks (optional, not BUILD_PRIORITY):
    cheap defense → dome → weap → fix → atek/stek. Gates on rush rolling so
    a_short Allies rush stays intact. Navy/air stays light via _optional_naval_air.
  - mode=expand (onboard C/D/E): 2nd powr/proc from starting cash, then weap,
    tanks, two pbox. 2nd proc prefers the mid-facing home mine; after ~4
    tanks, 3rd proc (mid fringe if Adjacent allows, else home). Idle harvs
    deny shared mid ore. Fix+MCV mid expand tried/reverted (needs fix;
    walk rarely completed). Spend cash+ore. Hold to 12 tanks, tanks-only
    push, retreat under 7, 2nd weap, finish-hunt when committed.
    A/B keep mode=rush.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from examples.scripted_bot import ScriptedBot
from openra_env.models import ActionType, CommandModel, OpenRAAction, OpenRAObservation
from rl.action_adapter import (
    PACK_ARMY,
    POWR_COST,
    _harv_units,
    _home_xy,
    _spatial_chw,
    building_queue_busy,
    n_combat_total,
    power_in_deficit,
    power_in_flight,
    spendable_resources,
)
from rl.auto_support import (
    ARRIVED_CELLS,
    DEFEND_CELLS,
    MIN_PILE_FOR_HUNT,
    fog_scout_destinations,
    home_raid_targets,
    hunt_near_cell,
)
from rl.obs_encoding import EnemyBeliefStore
from rl.war_objective import own_anchor, war_objective


def _xy(obj) -> Tuple[int, int]:
    return int(obj.cell_x), int(obj.cell_y)


def _cheb(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


class ScriptedTeacher(ScriptedBot):
    BUILD_PRIORITY = [
        "powr",
        "proc",
        "barracks",
    ]
    # Seguir TRAIN hasta el pack legal. El primer push es antes (RUSH).
    INFANTRY_TRAIN_TARGET = PACK_ARMY
    # Beginner rushea con ~3. A 8 el blob camina; a 12 el alumno puede
    # clonar army_attack_move (máscara PACK_ARMY).
    RUSH_ATTACK_MOVE = 8
    N_SCOUTS = 4
    EARLY_SCOUT_COMBAT = 3  # fog scout away from home once a few rifles exist
    MIN_HARVS = 2
    GUARD_COUNT = 0
    # Expand (C/D/E): 2nd proc before weap (free harv), then weap/tanks,
    # then contest mid ore (not a 3rd home stack). Easy keeps expanding
    # after 10k; home-3rd peaked proc=3 but still 0/7/1 — easy wins mid.
    # Weap 2000: save at 1200. Spend cash+ore, not cash.
    EXPAND_WEAP_SAVE = 1200
    EXPAND_WEAP_COST = 2000
    EXPAND_WEAP_CASH = EXPAND_WEAP_COST  # alias; build still costs 2000
    EXPAND_TANK_CASH = 700
    EXPAND_PBOX_CASH = 600
    EXPAND_HARV_COST = 1100
    EXPAND_PUSH_TANKS = 12  # easy UnitLimits 7 tanks; 8/10-tank pushes lost
    EXPAND_RETREAT_TANKS = 7  # hysteresis: come home before another dribble
    EXPAND_HOLD_SCOUTS = 1  # e1 fog only; never send a tank
    EXPAND_MIN_PBOX = 2  # easy BuildingFractions pbox: 4; 1 pbox lost the hold
    EXPAND_MIN_WEAP = 2  # easy BuildingLimits weap: 1; leftover cash sat idle
    EXPAND_WEAP2_TANKS = 4  # first weap must already be producing
    EXPAND_HARV_FLEE = 10  # easy ProtectionScanRadius; THREAT_RADIUS=18 yanks all 4
    EXPAND_PBOX_SEP = 3
    # Scout on a forward proc (DEFEND_CELLS=18) used to yank the hold
    # across the map (bench 8042: tnk=3 dead at 15k). Yard only.
    EXPAND_YARD_RAID = 12
    EXPAND_YARD_TYPES = frozenset({
        "fact", "afac", "weap", "tent", "barr", "powr", "apwr",
        "pbox", "hbox", "ftur",
    })
    EXPAND_GARRISON = 6  # keep this many e1 at home before saving for weap
    EXPAND_MIN_PROC = 2  # 2nd proc before weap (easy AdditionalMinimumRefineryCount 1)
    EXPAND_TARGET_PROC = 3  # 2 home + 1 mid; do not mill a 4th home stack
    EXPAND_PROC3_TANKS = 4  # after weap producing; 3rd proc / mid if reachable
    EXPAND_PROC_COST = 1400
    EXPAND_PROC_POWER = 30
    EXPAND_WEAP_POWER = 30
    EXPAND_POWER_PAD = 10
    EXPAND_MIN_HARVS = 4  # easy UnitLimits harv: 4; mid proc adds a free truck
    # a_short: 7 mines, 3 next to each MCV, 1 shared mid. Easy sells
    # stacked procs (SellRefineryTooCloseCellDistance 6) and puts the
    # 2nd on another indice, not the same patch.
    # RA Adjacent=8: a cell farther than that fails IsCloseEnoughToBase
    # and ActionHandler dumps the building on the CY annulus (bench:
    # proc=13,20+16,20 / 96,15+93,14). Mid (56,35) needs a powr bridge
    # from the mid-facing home mine (or a forward MCV).
    EXPAND_PROC_ZONE_SEP = 6
    EXPAND_PROC_BASE_REACH = 8
    EXPAND_ORE_ZONE_MIN = 3
    EXPAND_MID_CLAIM = 12  # proc within this of mid counts as mid claim
    EXPAND_BRIDGE_SEP = 2  # pack powr stepping stones toward mid
    EXPAND_FIX_COST = 1200  # MCV Prerequisites: fix, ~techlevel.medium
    EXPAND_MCV_COST = 2000
    EXPAND_MCV_DEPLOY = 10  # deploy when within this of mid mine
    EXPAND_MCV_MOVE_TICKS = 40  # do not spam MOVE every macro tick
    # RA Building TerrainTypes: Clear, Road — not Ore. Requesting a cell
    # on the mine (20,8 next to 23,5) makes CanPlaceBuilding fail and the
    # engine dumps on the CY ring (bench: 15,16+10,19 / 98,11+93,14).
    # Easy's visible proc at 100,24 is the *clear fringe* of (104,25).
    EXPAND_MINE_NO_BUILD = 4
    # Occupied + bib cells (not '_'). PROC _X_ xxx X== === ; POWR xx xx ==.
    _FP_PROC = (
        (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
    )
    _FP_POWR = (
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
    )
    _FP_FACT = tuple((x, y) for y in range(4) for x in range(3))
    # Do not re-issue Harvest every macro tick: walking harvs have local
    # ore=0, which looked "stale" and yanked them onto CY crumbs (bench
    # seed 8047: earned 575 at t=5k and still 575 at t=10k).
    HARV_REORDER_TICKS = 200
    A_SHORT_MINES = (
        (23, 5), (4, 30), (32, 28),
        (80, 4), (77, 22), (104, 25),
        (56, 35),
    )
    _PROD = frozenset({
        "fact", "afac", "proc", "weap", "tent", "barr", "kenn",
        "hpad", "afld", "syrd",
    })

    def __init__(self, verbose: bool = False, rush_attack_move: int | None = None,
                 mode: str = "rush"):
        """Optional per-instance RUSH override for benches; class default unchanged.

        mode=rush: A/B rifle teacher. mode=expand: C/D/E weap+tank overlay.
        """
        super().__init__(verbose=verbose)
        if rush_attack_move is not None:
            self.RUSH_ATTACK_MOVE = int(rush_attack_move)
        m = str(mode or "rush").lower().strip()
        self.mode = m if m in ("rush", "expand") else "rush"
        # Per-episode fog belief + last visible contact (map-agnostic hunt).
        self.belief = EnemyBeliefStore()
        self._last_contact: Optional[Tuple[int, int]] = None
        self._last_tick: Optional[int] = None
        self._harv_cmd_at: dict[int, int] = {}
        self._expand_committed = False
        self._mcv_cmd_at: dict[int, int] = {}

    def decide(self, obs: OpenRAObservation):
        """Reset belief between episodes if the same instance is reused."""
        try:
            tick = int(getattr(obs, "tick", 0) or 0)
        except (TypeError, ValueError):
            tick = 0
        if self._last_tick is not None and tick < self._last_tick:
            self.belief.reset()
            self._last_contact = None
            self._harv_cmd_at = {}
            self._expand_committed = False
            self._mcv_cmd_at = {}
        self._last_tick = tick
        action = super().decide(obs)
        extra = self._handle_harvesters(obs)
        # MCV expand disabled: fix+mcv path 0/8 vs easy (rarely deployed).
        # Mid contested by harvester denial + 3rd proc when Adjacent allows.
        if not extra:
            return action
        cmds = [
            c for c in (getattr(action, "commands", None) or [])
            if getattr(c, "action", None) != ActionType.NO_OP
        ]
        return OpenRAAction(commands=cmds + extra)

    def _handle_guards(self, obs: OpenRAObservation) -> List[CommandModel]:
        return []

    def _handle_transport(self, obs: OpenRAObservation) -> List[CommandModel]:
        return []

    def _update_phase(self, obs: OpenRAObservation):
        has_cy = any(b.type == "fact" for b in obs.buildings)
        has_barracks = any(b.type in self.BARRACKS_TYPES for b in obs.buildings)
        n_combat = n_combat_total(obs)
        if self.phase == "deploy_mcv" and has_cy:
            self.phase = "build_base"
            self._log("Phase → build_base")
        elif self.phase == "build_base" and has_barracks:
            self.phase = "train_army"
            self._log("Phase → train_army (barracks up, no wait for weap)")
        elif (self.phase == "train_army"
              and n_combat >= self.RUSH_ATTACK_MOVE
              and (self.mode != "expand" or self._expand_push_ready(obs))):
            self.phase = "attack"
            self._log(f"Phase → attack ({n_combat} combat, rush {self.RUSH_ATTACK_MOVE})")

    def _handle_production(self, obs: OpenRAObservation) -> List[CommandModel]:
        commands = super()._handle_production(obs)
        # El padre puede haber encolado APC; no lo queremos en a_short.
        commands = [
            c for c in commands
            if not (c.action == ActionType.TRAIN
                    and str(c.item_type or "") == self.TRANSPORT_TYPE)
        ]
        n_harv = sum(
            1 for u in obs.units
            if "harv" in str(getattr(u, "type", "") or "").lower()
        )
        vehicle_training = any(
            p.queue_type == "Vehicle" and p.progress < 0.99
            for p in obs.production
        )
        if (n_harv < self.MIN_HARVS and not vehicle_training
                and self._can_produce_item(obs, "harv")
                and spendable_resources(obs) >= 1100):
            commands.append(CommandModel(action=ActionType.TRAIN, item_type="harv"))
            self._log("Training harv (teacher eco)")
        if power_in_deficit(obs):
            commands = self._strip_power_competitors(commands)
            commands.extend(self._try_build_power(obs, "low-power"))
            commands.extend(self._optional_naval_air(obs, commands))
            return commands
        if self.mode == "expand":
            commands = self._apply_expand(obs, commands)
            commands.extend(self._optional_naval_air(obs, commands))
            return commands
        if self.phase != "attack":
            commands.extend(self._optional_tech_defense(obs, commands))
            commands.extend(self._optional_naval_air(obs, commands))
            return commands
        already_train = any(c.action == ActionType.TRAIN for c in commands)
        if already_train:
            # Still allow land tech BUILD (does not stack TRAIN).
            commands.extend(self._optional_tech_defense(obs, commands))
            return commands
        has_barracks = any(b.type in self.BARRACKS_TYPES for b in obs.buildings)
        infantry_training = any(
            p.queue_type == "Infantry" and p.progress < 0.99
            for p in obs.production
        )
        if (has_barracks and not infantry_training
                and self._can_produce_item(obs, "e1")
                and obs.economy.cash >= 100):
            commands.append(CommandModel(action=ActionType.TRAIN, item_type="e1"))
            self._log("Training e1 (sustain during attack)")
        commands.extend(self._optional_tech_defense(obs, commands))
        commands.extend(self._optional_naval_air(obs, commands))
        return commands

    def _count_type(self, obs: OpenRAObservation, *names: str) -> int:
        want = {str(n).lower() for n in names}
        n = 0
        for u in obs.units or []:
            if str(getattr(u, "type", "") or "").lower() in want:
                n += 1
        for b in obs.buildings or []:
            if str(getattr(b, "type", "") or "").lower() in want:
                n += 1
        return n

    def _rush_rolling(self, obs: OpenRAObservation) -> bool:
        return (self.phase == "attack"
                or n_combat_total(obs) >= self.RUSH_ATTACK_MOVE)

    def _queued_count(self, obs: OpenRAObservation, *names: str) -> int:
        want = {str(n).lower() for n in names}
        n = 0
        for p in obs.production or []:
            if str(getattr(p, "item", "") or "").lower() in want:
                n += 1
        return n

    def _is_tank(self, u) -> bool:
        return str(getattr(u, "type", "") or "").lower() in ("1tnk", "2tnk", "3tnk")

    def _expand_push_ready(self, obs: OpenRAObservation) -> bool:
        """Mass 12 tanks at home (easy UnitLimits 7) before the tank push."""
        n_tnk = self._count_type(obs, "1tnk", "2tnk", "3tnk")
        return n_tnk >= int(self.EXPAND_PUSH_TANKS)

    def _strip_power_competitors(
        self, commands: List[CommandModel],
    ) -> List[CommandModel]:
        """Drop e1 / non-power BUILD so a plant can actually queue."""
        out: List[CommandModel] = []
        for c in commands or []:
            act = getattr(c, "action", None)
            item = str(getattr(c, "item_type", "") or "").lower()
            if act == ActionType.TRAIN and item == "e1":
                continue
            if act == ActionType.BUILD and item not in ("powr", "apwr", "power"):
                continue
            out.append(c)
        return out

    def _try_build_power(
        self, obs: OpenRAObservation, reason: str,
    ) -> List[CommandModel]:
        if building_queue_busy(obs) or power_in_flight(obs):
            return []
        if spendable_resources(obs) < POWR_COST:
            return []
        if self._count_type(obs, "powr", "apwr") >= 8:
            return []
        if not self._can_produce_item(obs, "powr"):
            return []
        self._log(f"{reason} BUILD powr (low power)")
        return [CommandModel(action=ActionType.BUILD, item_type="powr")]

    def _building_cells(self, obs: OpenRAObservation, *names: str) -> List[Tuple[int, int]]:
        want = {str(n).lower() for n in names}
        out: List[Tuple[int, int]] = []
        for b in obs.buildings or []:
            if str(getattr(b, "type", "") or "").lower() not in want:
                continue
            try:
                out.append((int(b.cell_x), int(b.cell_y)))
            except (TypeError, ValueError):
                continue
        return out

    def _ore_zones(self, arr) -> List[dict]:
        """Connected explored-ore patches (the 7 mines on a_short)."""
        if arr is None or getattr(arr, "shape", (0,))[0] < 5:
            return []
        ch2, ch4 = arr[2], arr[4]
        h, w = ch2.shape
        seen = [[False] * w for _ in range(h)]
        zones: List[dict] = []
        min_n = int(self.EXPAND_ORE_ZONE_MIN)
        neigh = ((-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1))
        for y in range(h):
            for x in range(w):
                if seen[y][x] or float(ch2[y, x]) <= 0.0 or float(ch4[y, x]) < 0.45:
                    continue
                stack = [(x, y)]
                seen[y][x] = True
                cells: List[Tuple[int, int]] = []
                dens = 0.0
                while stack:
                    cx, cy = stack.pop()
                    cells.append((cx, cy))
                    dens += float(ch2[cy, cx])
                    for dx, dy in neigh:
                        nx, ny = cx + dx, cy + dy
                        if not (0 <= nx < w and 0 <= ny < h) or seen[ny][nx]:
                            continue
                        if float(ch2[ny, nx]) <= 0.0 or float(ch4[ny, nx]) < 0.45:
                            continue
                        seen[ny][nx] = True
                        stack.append((nx, ny))
                if len(cells) < min_n:
                    continue
                sx = sum(c[0] for c in cells) // len(cells)
                sy = sum(c[1] for c in cells) // len(cells)
                zones.append({
                    "xy": (int(sx), int(sy)),
                    "n": len(cells),
                    "dens": dens,
                })
        return zones

    def _map_wh(self, obs: OpenRAObservation, arr) -> Tuple[int, int]:
        if arr is not None and getattr(arr, "shape", (0, 0, 0))[0] >= 1:
            return int(arr.shape[2]), int(arr.shape[1])
        info = getattr(obs, "map_info", None)
        return (
            int(getattr(info, "width", 64) or 64),
            int(getattr(info, "height", 64) or 64),
        )

    def _mid_ore_target(
        self, obs: OpenRAObservation,
    ) -> Optional[Tuple[int, int]]:
        """Shared mid mine on a_short; else densest central ore zone."""
        arr = _spatial_chw(obs)
        w, h = self._map_wh(obs, arr)
        name = str(getattr(getattr(obs, "map_info", None), "map_name", "")
                   or "").lower()
        known = ("a_short" in name or "singles" in name or "fase2" in name)
        if known:
            # Fixed a_short mid mine; ignore possibly-wrong unit-test map_info size.
            return (56, 35)
        zones = self._ore_zones(arr)
        if not zones:
            return None
        cx, cy = w // 2, h // 2
        zones.sort(key=lambda z: (_cheb(z["xy"], (cx, cy)), -float(z["dens"])))
        return zones[0]["xy"]

    def _all_mine_points(
        self, obs: OpenRAObservation,
    ) -> List[Tuple[int, int]]:
        arr = _spatial_chw(obs)
        w, h = self._map_wh(obs, arr)
        name = str(getattr(getattr(obs, "map_info", None), "map_name", "")
                   or "").lower()
        pts: List[Tuple[int, int]] = []
        known = ("a_short" in name or "singles" in name or "fase2" in name)
        if known:
            for mx, my in self.A_SHORT_MINES:
                if 0 <= mx < w and 0 <= my < h:
                    pts.append((int(mx), int(my)))
        if not pts:
            for z in self._ore_zones(arr):
                pts.append(z["xy"])
        uniq: List[Tuple[int, int]] = []
        for p in pts:
            if any(_cheb(p, q) < 8 for q in uniq):
                continue
            uniq.append(p)
        return uniq

    def _home_ore_targets(
        self, obs: OpenRAObservation, origin: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Home-side mines only (exclude shared mid).

        a_short uses the 7 known mine actors. Fog-limited spatial centroids
        sit on the CY-adjacent crumbs of those same fields and used to
        steal (23,5)/(4,30) via <8 dedupe — first proc then sat at CY+3
        (bench: 15,16+10,19 / 98,11+93,14). Mid is contested separately.
        """
        mid = self._mid_ore_target(obs)
        claim = int(self.EXPAND_MID_CLAIM)
        uniq = []
        for p in self._all_mine_points(obs):
            if mid is not None and _cheb(p, mid) < claim:
                continue
            uniq.append(p)
        uniq.sort(key=lambda p: _cheb(p, origin))
        return uniq[:3]

    def _expand_anchors(self, obs: OpenRAObservation) -> List[Tuple[int, int]]:
        anchors = self._building_cells(
            obs, "fact", "afac", "proc", "powr", "apwr", "tent", "barr", "weap")
        if anchors:
            return anchors
        home = _home_xy(obs)
        return [home] if home is not None else []

    def _mid_claimed(self, obs: OpenRAObservation) -> bool:
        mid = self._mid_ore_target(obs)
        if mid is None:
            return False
        claim = int(self.EXPAND_MID_CLAIM)
        for p in self._building_cells(obs, "proc"):
            if _cheb(p, mid) <= claim:
                return True
        return False

    def _mid_in_reach(self, obs: OpenRAObservation) -> bool:
        """True if Adjacent can sit a proc on the mid fringe (not just toward it)."""
        mid = self._mid_ore_target(obs)
        if mid is None:
            return False
        anchors = self._expand_anchors(obs)
        if not anchors:
            return False
        existing = self._building_cells(obs, "proc")
        placed = self._halo_toward(
            anchors, mid, existing, self._blocked_cells(obs),
            _spatial_chw(obs), item="proc", mines=[mid])
        if placed is None:
            return False
        # Halo always finds *some* cell on the base edge toward mid; require
        # that cell actually sit on the mid fringe (else keep bridging).
        return _cheb(placed, mid) <= int(self.EXPAND_MID_CLAIM)

    def _bridge_place_cell(
        self, obs: OpenRAObservation, cy, target: Tuple[int, int],
        item: str = "powr",
    ) -> Tuple[int, int]:
        """Furthest clear cell toward mid within Adjacent — powr stepping stone."""
        fallback = self._placement_offset(cy)
        anchors = self._expand_anchors(obs)
        if not anchors:
            return fallback
        # Prefer progressing from the forward-most anchor.
        anchors = sorted(anchors, key=lambda a: _cheb(a, target))
        existing = self._building_cells(obs, "powr", "apwr", "proc", "fact")
        placed = self._halo_toward(
            anchors[-3:] if len(anchors) > 3 else anchors,
            target, existing, self._blocked_cells(obs),
            _spatial_chw(obs), item=item, mines=[target],
            sep=int(self.EXPAND_BRIDGE_SEP))
        return placed if placed is not None else fallback

    def _fp_offsets(self, item: str) -> Tuple[Tuple[int, int], ...]:
        it = str(item or "proc").lower()
        if it in ("powr", "apwr"):
            return self._FP_POWR
        if it in ("fact", "afac"):
            return self._FP_FACT
        if it in ("pbox", "hbox", "ftur", "gun"):
            return ((0, 0),)
        return self._FP_PROC

    def _fp_cells(self, item: str, x: int, y: int) -> List[Tuple[int, int]]:
        return [(x + ox, y + oy) for ox, oy in self._fp_offsets(item)]

    def _blocked_cells(self, obs: OpenRAObservation) -> set:
        blocked = set()
        for b in obs.buildings or []:
            try:
                bx, by = int(b.cell_x), int(b.cell_y)
            except (TypeError, ValueError):
                continue
            typ = str(getattr(b, "type", "") or "fact").lower()
            for cx, cy in self._fp_cells(typ, bx, by):
                blocked.add((cx, cy))
        return blocked

    def _halo_toward(
        self,
        anchors: List[Tuple[int, int]],
        target: Tuple[int, int],
        existing: List[Tuple[int, int]],
        blocked: set,
        arr=None,
        item: str = "proc",
        mines: Optional[List[Tuple[int, int]]] = None,
        sep: Optional[int] = None,
    ) -> Optional[Tuple[int, int]]:
        """Clear-fringe cell toward `target`. Ore/trees fail CanPlace → CY dump."""
        reach = int(self.EXPAND_PROC_BASE_REACH)
        sep = int(self.EXPAND_PROC_ZONE_SEP if sep is None else sep)
        no_build = int(self.EXPAND_MINE_NO_BUILD)
        offsets = self._fp_offsets(item)
        mines = list(mines or ())
        best: Optional[Tuple[int, int]] = None
        best_score = -1e9
        ch2 = arr[2] if arr is not None and getattr(arr, "shape", (0,))[0] >= 3 else None
        ch3 = arr[3] if arr is not None and getattr(arr, "shape", (0,))[0] >= 4 else None
        h = int(arr.shape[1]) if arr is not None else 10**9
        w = int(arr.shape[2]) if arr is not None else 10**9
        neigh = ((-1, 0), (1, 0), (0, -1), (0, 1))
        for ax, ay in anchors:
            for dx in range(-reach, reach + 1):
                for dy in range(-reach, reach + 1):
                    if max(abs(dx), abs(dy)) > reach:
                        continue
                    x, y = ax + dx, ay + dy
                    fp = [(x + ox, y + oy) for ox, oy in offsets]
                    if any(c in blocked for c in fp):
                        continue
                    if w < 10**9:
                        if any(not (0 <= fx < w and 0 <= fy < h) for fx, fy in fp):
                            continue
                    bad = False
                    adj_ore = 0
                    for fx, fy in fp:
                        if ch2 is not None:
                            if 0 <= fy < ch2.shape[0] and 0 <= fx < ch2.shape[1]:
                                if float(ch2[fy, fx]) > 0.0:
                                    bad = True
                                    break
                        if ch3 is not None:
                            if 0 <= fy < ch3.shape[0] and 0 <= fx < ch3.shape[1]:
                                if float(ch3[fy, fx]) <= 0.5:
                                    bad = True
                                    break
                        for mx, my in mines:
                            if _cheb((fx, fy), (mx, my)) < no_build:
                                bad = True
                                break
                        if bad:
                            break
                    if bad:
                        continue
                    if existing:
                        d_proc = min(_cheb((x, y), p) for p in existing)
                        if d_proc < sep:
                            continue
                    else:
                        d_proc = 99
                    if ch2 is not None:
                        seen = set()
                        for fx, fy in fp:
                            for ndx, ndy in neigh:
                                nx, ny = fx + ndx, fy + ndy
                                if (nx, ny) in seen:
                                    continue
                                seen.add((nx, ny))
                                if 0 <= ny < ch2.shape[0] and 0 <= nx < ch2.shape[1]:
                                    if float(ch2[ny, nx]) > 0.0:
                                        adj_ore += 1
                    d_tgt = min(_cheb(c, target) for c in fp)
                    d_manh = min(abs(c[0] - target[0]) + abs(c[1] - target[1])
                                 for c in fp)
                    # Sit on Clear next to ore (easy 100,24), never ON ore.
                    score = (-10.0 * d_tgt - 0.1 * d_manh
                             + 3.0 * min(adj_ore, 12)
                             + 0.25 * min(d_proc, 12))
                    if score > best_score:
                        best_score = score
                        best = (int(x), int(y))
        return best

    def _ore_place_cell(
        self, obs: OpenRAObservation, cy, item: str = "proc",
    ) -> Tuple[int, int]:
        """Clear cell toward next ore target. On-ore requests CY-dump in ActionHandler.

        Proc 1: nearest home mine. Proc 2: remaining home mine closest to mid
        (bridgehead). Proc 3+: shared mid fringe once Adjacent reaches it.
        """
        fallback = self._placement_offset(cy)
        try:
            origin = (int(cy.cell_x), int(cy.cell_y))
        except (TypeError, ValueError, AttributeError):
            origin = _home_xy(obs) or (0, 0)
        existing = self._building_cells(obs, "proc")
        anchors = self._expand_anchors(obs) or [origin]
        home = self._home_ore_targets(obs, origin)
        mid = self._mid_ore_target(obs)
        claim_r = int(self.EXPAND_MID_CLAIM)

        def _claim_index(pts, px, py):
            best_i, best_d = None, 10**9
            for i, t in enumerate(pts):
                d = max(abs(px - t[0]), abs(py - t[1]))
                if d < best_d:
                    best_d = d
                    best_i = i
            return best_i, best_d

        target: Optional[Tuple[int, int]] = None
        mine_list: List[Tuple[int, int]] = list(home)
        if mid is not None:
            mine_list = list(home) + [mid]

        if len(existing) == 0 and home:
            target = home[0]
        elif len(existing) == 1 and home:
            claimed_home: set[int] = set()
            for px, py in existing:
                i, d = _claim_index(home, px, py)
                if i is not None and d < claim_r:
                    claimed_home.add(i)
            rest = [t for i, t in enumerate(home) if i not in claimed_home]
            if rest and mid is not None:
                rest.sort(key=lambda p: _cheb(p, mid))
                target = rest[0]
            elif rest:
                target = rest[0]
            elif mid is not None:
                target = mid
        else:
            # 3rd+ : mid contest (skip if already claimed).
            if mid is not None and not self._mid_claimed(obs):
                target = mid
            elif home:
                claimed_home = set()
                for px, py in existing:
                    i, d = _claim_index(home, px, py)
                    if i is not None and d < claim_r:
                        claimed_home.add(i)
                for i, t in enumerate(home):
                    if i not in claimed_home:
                        target = t
                        break
            if target is None and mid is not None:
                target = mid
            if target is None and home:
                target = home[-1]

        if target is None:
            return fallback
        arr = _spatial_chw(obs)
        placed = self._halo_toward(
            anchors, target, existing, self._blocked_cells(obs), arr,
            item=item, mines=mine_list or [target])
        return placed if placed is not None else fallback

    def _approach_place_cell(
        self, obs: OpenRAObservation, cy, item: str = "pbox",
    ) -> Tuple[int, int]:
        """Pbox on the attack path (map center), not stacked on the CY +3,0."""
        fallback = self._placement_offset(cy)
        try:
            origin = (int(cy.cell_x), int(cy.cell_y))
        except (TypeError, ValueError, AttributeError):
            origin = _home_xy(obs) or (0, 0)
        info = getattr(obs, "map_info", None)
        try:
            target = (
                int(getattr(info, "width", 64) or 64) // 2,
                int(getattr(info, "height", 32) or 32) // 2,
            )
        except (TypeError, ValueError):
            target = (54, 27)
        existing = self._building_cells(obs, "pbox", "hbox", "ftur", "gun")
        anchors = self._building_cells(
            obs, "fact", "afac", "powr", "tent", "barr", "weap", "pbox")
        if not anchors:
            anchors = [origin]
        placed = self._halo_toward(
            anchors, target, existing, self._blocked_cells(obs),
            _spatial_chw(obs), item=item, sep=int(self.EXPAND_PBOX_SEP))
        return placed if placed is not None else fallback

    def _yard_raid_targets(self, obs: OpenRAObservation) -> list:
        """Threats at the yard. A scout on a forward proc is not a raid."""
        cores: List[Tuple[int, int]] = []
        for b in obs.buildings or []:
            if str(getattr(b, "type", "") or "").lower() not in self.EXPAND_YARD_TYPES:
                continue
            try:
                cores.append(_xy(b))
            except (TypeError, ValueError):
                continue
        if not cores:
            return []
        rad = int(self.EXPAND_YARD_RAID)
        out = []
        for t in list(getattr(obs, "visible_enemies", None) or []) + list(
                getattr(obs, "visible_enemy_buildings", None) or []):
            try:
                xy = _xy(t)
            except (TypeError, ValueError):
                continue
            if min(_cheb(xy, c) for c in cores) <= rad:
                out.append(t)
        return out

    def _nearest_xy(
        self, origin: Tuple[int, int], pts: List[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        if not pts:
            return None
        ox, oy = origin
        return min(pts, key=lambda p: max(abs(p[0] - ox), abs(p[1] - oy)))

    def _threatened_harvs(self, obs: OpenRAObservation, harvs) -> list:
        """Only the trucks next to a threat. Radius 18 used to yank all 4."""
        threats = list(getattr(obs, "visible_enemies", None) or []) + list(
            getattr(obs, "visible_enemy_buildings", None) or [])
        if not threats or not harvs:
            return []
        rad = int(self.EXPAND_HARV_FLEE)
        out = []
        for u in harvs:
            try:
                xy = _xy(u)
            except (TypeError, ValueError):
                continue
            for t in threats:
                try:
                    if _cheb(xy, _xy(t)) <= rad:
                        out.append(u)
                        break
                except (TypeError, ValueError):
                    continue
        return out

    def _handle_mcv_expand(self, obs: OpenRAObservation) -> List[CommandModel]:
        """Move expansion MCV to mid ore and deploy (easy McvExpansionManager)."""
        if self.mode != "expand":
            return []
        # Never touch the opening MCV — parent deploy_mcv must land the first CY.
        n_fact = self._count_type(obs, "fact", "afac")
        if n_fact < 1:
            return []
        mid = self._mid_ore_target(obs)
        if mid is None:
            return []
        if self._mid_claimed(obs):
            return []
        mcvs = [
            u for u in (obs.units or [])
            if "mcv" in str(getattr(u, "type", "") or "").lower()
        ]
        if not mcvs:
            return []
        try:
            tick = int(getattr(obs, "tick", 0) or 0)
        except (TypeError, ValueError):
            tick = 0
        out: List[CommandModel] = []
        deploy_r = int(self.EXPAND_MCV_DEPLOY)
        cooldown = int(self.EXPAND_MCV_MOVE_TICKS)
        for u in mcvs[:1]:
            try:
                uid = int(u.actor_id)
                xy = _xy(u)
            except (TypeError, ValueError):
                continue
            # Deploy on clear fringe (~5 toward home), not ON the ore cell.
            # Do NOT deploy merely "near mid" — that landed fact at ~70 on east
            # spawn (bench 8042) short of the shared patch.
            home = _home_xy(obs) or xy
            dx, dy = int(home[0] - mid[0]), int(home[1] - mid[1])
            span = max(abs(dx), abs(dy), 1)
            approach = (
                int(mid[0] + int(round(5.0 * dx / span))),
                int(mid[1] + int(round(5.0 * dy / span))),
            )
            if _cheb(xy, approach) <= 5:
                out.append(CommandModel(
                    action=ActionType.DEPLOY, actor_id=uid))
                self._mcv_cmd_at[uid] = tick
                self._log(f"Expand DEPLOY mcv @ {xy} near mid {mid}")
                continue
            last = self._mcv_cmd_at.get(uid, -10**9)
            if tick - last < cooldown:
                continue
            out.append(CommandModel(
                action=ActionType.MOVE,
                actor_id=uid,
                target_x=int(approach[0]),
                target_y=int(approach[1]),
            ))
            self._mcv_cmd_at[uid] = tick
            self._log(f"Expand MOVE mcv {xy} -> approach {approach}")
            # Escort: peel up to 2 idle tanks along the MCV path.
            escorts = 0
            for u2 in (obs.units or []):
                if escorts >= 2:
                    break
                if not self._is_tank(u2):
                    continue
                if not bool(getattr(u2, "is_idle", False)):
                    continue
                try:
                    out.append(CommandModel(
                        action=ActionType.ATTACK_MOVE,
                        actor_id=int(u2.actor_id),
                        target_x=int(approach[0]),
                        target_y=int(approach[1]),
                    ))
                    escorts += 1
                except (TypeError, ValueError):
                    continue
            if escorts:
                self._log(f"Expand MCV escort tanks={escorts}")
        return out

    def _handle_harvesters(self, obs: OpenRAObservation) -> List[CommandModel]:
        """Flee raids; send idle trucks to a home mine. Do not micro walkers.

        Easy's HarvesterBotModule leaves pathing to the engine. The old
        stale-yank used _best_ore_near(proc, r=16): CY-stacked procs cannot
        see (23,5) (manhattan 19) so every walking harv was re-Harvested
        onto crumbs next to the yard.
        """
        if self.mode != "expand":
            return []
        harvs = _harv_units(obs)
        if not harvs:
            return []
        try:
            tick = int(getattr(obs, "tick", 0) or 0)
        except (TypeError, ValueError):
            tick = 0
        procs = self._building_cells(obs, "proc")
        home = _home_xy(obs)
        threatened = self._threatened_harvs(obs, harvs)
        if threatened:
            out: List[CommandModel] = []
            fact = self._own_fact(obs)
            for u in threatened[:4]:
                dest = fact or home
                if dest is None:
                    continue
                try:
                    uid = int(u.actor_id)
                    out.append(CommandModel(
                        action=ActionType.MOVE,
                        actor_id=uid,
                        target_x=int(dest[0]),
                        target_y=int(dest[1]),
                    ))
                    self._harv_cmd_at[uid] = tick
                except (TypeError, ValueError):
                    continue
            if out:
                self._log(f"Harv flee raid {len(out)}")
            return out
        idle = [u for u in harvs if bool(getattr(u, "is_idle", False))]
        if not idle:
            return []
        origin = home or (0, 0)
        mines = list(self._home_ore_targets(obs, origin))
        mid = self._mid_ore_target(obs)
        # Mid-ore denial: once 2 procs exist, send half the idle trucks to the
        # shared mid patch even without a mid refinery (long haul, denies easy).
        if mid is not None and (len(procs) >= 2 or self._mid_claimed(obs)):
            mines = [m for m in mines if _cheb(m, mid) >= 8]
            if mines:
                mines = [mines[0], mid] + mines[1:]
            else:
                mines = [mid]
        out = []
        cooldown = int(self.HARV_REORDER_TICKS)
        for i, u in enumerate(idle[:4]):
            try:
                uid = int(u.actor_id)
            except (TypeError, ValueError):
                continue
            last = self._harv_cmd_at.get(uid, -10**9)
            if tick - last < cooldown:
                continue
            dest = None
            if mines:
                dest = mines[i % len(mines)]
            elif procs:
                dest = procs[i % len(procs)]
            try:
                if dest is not None:
                    out.append(CommandModel(
                        action=ActionType.HARVEST,
                        actor_id=uid,
                        target_x=int(dest[0]),
                        target_y=int(dest[1]),
                    ))
                else:
                    out.append(CommandModel(
                        action=ActionType.HARVEST, actor_id=uid))
                self._harv_cmd_at[uid] = tick
            except (TypeError, ValueError):
                continue
        if out:
            dests = [(c.target_x, c.target_y) for c in out]
            self._log(f"Harv idle {len(out)} -> {dests}")
        return out

    def _handle_placement(self, obs: OpenRAObservation) -> List[CommandModel]:
        """Parent only places Building-queue. Pbox sits on Defense.

        Expand: snap proc onto Clear next to ore; powr toward mid when bridging.
        """
        commands = super()._handle_placement(obs)
        cy = self._find_building(obs, "fact")
        if not cy:
            return commands
        if self.mode == "expand":
            rewritten: List[CommandModel] = []
            n_proc = self._count_type(obs, "proc")
            mid = self._mid_ore_target(obs)
            bridging = (
                n_proc >= int(self.EXPAND_MIN_PROC)
                and mid is not None
                and not self._mid_claimed(obs)
                and not self._mid_in_reach(obs)
            )
            for c in commands:
                item = str(getattr(c, "item_type", "") or "").lower()
                if (getattr(c, "action", None) == ActionType.PLACE_BUILDING
                        and item == "proc"):
                    x, y = self._ore_place_cell(obs, cy, item=item)
                    rewritten.append(CommandModel(
                        action=ActionType.PLACE_BUILDING,
                        item_type=item,
                        target_x=int(x),
                        target_y=int(y),
                    ))
                elif (getattr(c, "action", None) == ActionType.PLACE_BUILDING
                        and item in ("powr", "apwr")):
                    if bridging and mid is not None:
                        x, y = self._bridge_place_cell(obs, cy, mid, item=item)
                    else:
                        x, y = self._ore_place_cell(obs, cy, item=item)
                    rewritten.append(CommandModel(
                        action=ActionType.PLACE_BUILDING,
                        item_type=item,
                        target_x=int(x),
                        target_y=int(y),
                    ))
                elif (getattr(c, "action", None) == ActionType.PLACE_BUILDING
                        and item in ("pbox", "hbox", "ftur", "gun")):
                    x, y = self._approach_place_cell(obs, cy, item=item)
                    rewritten.append(CommandModel(
                        action=ActionType.PLACE_BUILDING,
                        item_type=item,
                        target_x=int(x),
                        target_y=int(y),
                    ))
                else:
                    rewritten.append(c)
            commands = rewritten
        for prod in obs.production or []:
            qt = str(getattr(prod, "queue_type", "") or "").lower()
            if qt not in ("defense", "defences", "defenses"):
                continue
            try:
                prog = float(getattr(prod, "progress", 0) or 0)
            except (TypeError, ValueError):
                prog = 0.0
            if prog < 0.99:
                continue
            item = str(getattr(prod, "item", "") or "")
            if self.mode == "expand":
                x, y = self._approach_place_cell(obs, cy, item=item or "pbox")
            else:
                x, y = self._placement_offset(cy)
            self._log(
                f"Placing {prod.item} (defense) at cell ({x}, {y})")
            commands.append(CommandModel(
                action=ActionType.PLACE_BUILDING,
                item_type=str(getattr(prod, "item", "") or ""),
                target_x=x,
                target_y=y,
            ))
        return commands

    def _power_surplus(self, obs: OpenRAObservation) -> int:
        eco = getattr(obs, "economy", None)
        try:
            return (int(getattr(eco, "power_provided", 0) or 0)
                    - int(getattr(eco, "power_drained", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _apply_expand(
        self, obs: OpenRAObservation, commands: List[CommandModel],
    ) -> List[CommandModel]:
        """2nd proc (free harv) then weap/tanks/pbox, then mid-ore contest.

        Opening (powr/proc/tent/e1) stays parent BUILD_PRIORITY. Cap at
        EXPAND_TARGET_PROC (3 = 2 home + mid). Skip optional dome-first tech
        so weap is the C lesson. Vehicle TRAIN is independent of the infantry
        queue — parent e1 must not starve 1tnk. After tanks mass: powr-bridge
        toward mid until Adjacent reaches it, then mid proc (not home-3rd).
        """
        if power_in_deficit(obs):
            commands = self._strip_power_competitors(commands)
            commands.extend(self._try_build_power(obs, "Expand"))
            return commands
        cash = spendable_resources(obs)
        own = {
            str(getattr(b, "type", "") or "").lower()
            for b in (obs.buildings or [])
        }
        queued = {
            str(getattr(p, "item", "") or "").lower()
            for p in (obs.production or [])
        }
        has_weap_built = "weap" in own
        has_weap = has_weap_built or "weap" in queued
        has_build = any(c.action == ActionType.BUILD for c in commands)
        building_busy = any(
            str(getattr(p, "queue_type", "") or "").lower() in
            ("building", "buildings", "structure", "structures")
            and float(getattr(p, "progress", 0) or 0) < 0.99
            for p in (obs.production or [])
        )
        vehicle_busy = any(
            str(getattr(p, "queue_type", "") or "").lower() == "vehicle"
            and float(getattr(p, "progress", 0) or 0) < 0.99
            for p in (obs.production or [])
        )
        infantry_busy = any(
            str(getattr(p, "queue_type", "") or "").lower() == "infantry"
            and float(getattr(p, "progress", 0) or 0) < 0.99
            for p in (obs.production or [])
        )
        defense_busy = any(
            str(getattr(p, "queue_type", "") or "").lower() in
            ("defense", "defences", "defenses")
            and float(getattr(p, "progress", 0) or 0) < 0.99
            for p in (obs.production or [])
        )
        n_proc = (self._count_type(obs, "proc")
                  + self._queued_count(obs, "proc"))
        n_e1 = self._count_type(obs, "e1")
        n_e3 = self._count_type(obs, "e3")
        n_harv = sum(
            1 for u in (obs.units or [])
            if "harv" in str(getattr(u, "type", "") or "").lower()
        )
        n_tnk = self._count_type(obs, "1tnk", "2tnk", "3tnk")
        n_1tnk = self._count_type(obs, "1tnk")
        surplus = self._power_surplus(obs)
        # Parent trains to PACK_ARMY=12; surplus e1 starves proc/weap/tanks.
        if n_e1 >= int(self.EXPAND_GARRISON):
            commands = [
                c for c in commands
                if not (c.action == ActionType.TRAIN
                        and str(c.item_type or "") in ("e1", "e3"))
            ]

        def _try_powr(reason: str) -> bool:
            nonlocal has_build
            if has_build or building_busy:
                return False
            if cash < POWR_COST or not self._can_produce_item(obs, "powr"):
                return False
            self._log(f"Expand BUILD powr ({reason})")
            commands.append(CommandModel(action=ActionType.BUILD, item_type="powr"))
            has_build = True
            return True

        # 1) 2nd proc before weap (each proc spawns a free harv). Easy is 2 proc.
        need_proc = (
            n_proc < int(self.EXPAND_MIN_PROC)
            and "proc" in own
            and self._can_produce_item(obs, "proc")
        )
        if need_proc:
            pad = int(self.EXPAND_PROC_POWER) + int(self.EXPAND_POWER_PAD)
            if surplus < pad and _try_powr("before 2nd proc"):
                return commands
            if (not has_build and not building_busy
                    and cash >= int(self.EXPAND_PROC_COST)):
                self._log("Expand BUILD proc (2nd refinery)")
                commands.append(CommandModel(
                    action=ActionType.BUILD, item_type="proc"))
                return commands
            # Still short for 2nd proc: do not dump the opening cash into weap.
            return commands

        weap_legal = has_weap or self._can_produce_item(obs, "weap")
        # 2) Weap save/build. 2nd powr first if weap would brown out (2 proc).
        if weap_legal and not has_weap:
            pad = int(self.EXPAND_WEAP_POWER) + int(self.EXPAND_POWER_PAD)
            if surplus < pad and _try_powr("before weap"):
                return commands
            if cash >= self.EXPAND_WEAP_SAVE and cash < self.EXPAND_WEAP_COST:
                keep_e1 = n_e1 < int(self.EXPAND_GARRISON)
                commands = [
                    c for c in commands
                    if c.action != ActionType.TRAIN
                    or (keep_e1 and str(c.item_type or "") == "e1")
                    or (n_harv < self.MIN_HARVS
                        and str(c.item_type or "") == "harv")
                ]
            if (not has_build and not building_busy
                    and cash >= self.EXPAND_WEAP_COST
                    and self._can_produce_item(obs, "weap")):
                self._log("Expand BUILD weap")
                commands.append(CommandModel(
                    action=ActionType.BUILD, item_type="weap"))
                return commands
            if (cash >= self.EXPAND_WEAP_SAVE
                    and n_e1 >= int(self.EXPAND_GARRISON)):
                return commands

        # Defense queue is independent of Building (weap/proc/powr). Easy
        # pbox delay is 3000; prereq is tent. After 2nd proc is in flight.
        n_pbox = (self._count_type(obs, "pbox", "hbox", "ftur")
                  + self._queued_count(obs, "pbox", "hbox", "ftur"))
        has_tent = bool(own & self.BARRACKS_TYPES)
        if (has_tent and n_proc >= int(self.EXPAND_MIN_PROC)
                and n_pbox < int(self.EXPAND_MIN_PBOX)
                and not defense_busy and cash >= self.EXPAND_PBOX_CASH):
            for item in ("pbox", "hbox", "ftur"):
                if self._can_produce_item(obs, item):
                    self._log(f"Expand BUILD {item}")
                    commands.append(CommandModel(
                        action=ActionType.BUILD, item_type=item))
                    break

        # Eco: 3rd proc after tanks mass. Prefer mid fringe when Adjacent
        # reaches it; else home mine (home-3rd). Fix+MCV expand was too
        # slow/fragile (bench 0/8: MCV rarely deployed; easy took mid).
        # Mid contest is via harvester denial (see _handle_harvesters).
        need_proc3 = (
            has_weap_built
            and n_proc < int(self.EXPAND_TARGET_PROC)
            and n_tnk >= int(self.EXPAND_PROC3_TANKS)
            and self._can_produce_item(obs, "proc")
        )
        if need_proc3:
            pad = int(self.EXPAND_PROC_POWER) + int(self.EXPAND_POWER_PAD)
            if surplus < pad and _try_powr("before 3rd proc"):
                return commands
            if (not has_build and not building_busy
                    and cash >= int(self.EXPAND_PROC_COST)
                    and cash >= int(self.EXPAND_PROC_COST) + int(self.EXPAND_TANK_CASH)):
                where = "mid" if self._mid_in_reach(obs) else "home"
                self._log(f"Expand BUILD proc (3rd refinery, {where})")
                commands.append(CommandModel(
                    action=ActionType.BUILD, item_type="proc"))
                has_build = True
                # fall through — Vehicle TRAIN still allowed this tick
        # 2nd weap after harv floor + 4 tanks. Easy BuildingLimits weap: 1;
        # leftover 2k sat unspent while 1 weap dribbled (bench 8047).
        n_weap = (self._count_type(obs, "weap")
                  + self._queued_count(obs, "weap"))
        need_weap2 = (
            has_weap_built
            and n_weap < int(self.EXPAND_MIN_WEAP)
            and n_harv >= int(self.EXPAND_MIN_HARVS)
            and n_tnk >= int(self.EXPAND_WEAP2_TANKS)
            and self._can_produce_item(obs, "weap")
        )
        if need_weap2:
            pad = int(self.EXPAND_WEAP_POWER) + int(self.EXPAND_POWER_PAD)
            if surplus < pad and _try_powr("before 2nd weap"):
                return commands
            if cash >= self.EXPAND_WEAP_SAVE and cash < self.EXPAND_WEAP_COST:
                if (n_harv < int(self.EXPAND_MIN_HARVS)
                        and not vehicle_busy
                        and cash >= int(self.EXPAND_HARV_COST)
                        and self._can_produce_item(obs, "harv")):
                    self._log("Expand TRAIN harv (save 2nd weap)")
                    commands.append(CommandModel(
                        action=ActionType.TRAIN, item_type="harv"))
                return commands
            if (not has_build and not building_busy
                    and cash >= self.EXPAND_WEAP_COST):
                self._log("Expand BUILD weap (2nd factory)")
                commands.append(CommandModel(
                    action=ActionType.BUILD, item_type="weap"))
                return commands

        # Vehicle queue: harv floor before tanks. Easy UnitLimits harv: 4;
        # waiting for tanks first starved eco (bench 0/8, edge ~-2200).
        if has_weap_built and not vehicle_busy:
            want = None
            if (n_harv < int(self.EXPAND_MIN_HARVS)
                    and cash >= int(self.EXPAND_HARV_COST)
                    and self._can_produce_item(obs, "harv")):
                want = "harv"
            elif cash >= self.EXPAND_TANK_CASH:
                if (n_1tnk >= 1 and self._can_produce_item(obs, "2tnk")
                        and cash >= 1500):
                    want = "2tnk"
                elif self._can_produce_item(obs, "1tnk"):
                    want = "1tnk"
            if want:
                _veh = {"harv", "1tnk", "2tnk", "3tnk", self.TRANSPORT_TYPE}
                commands = [
                    c for c in commands
                    if not (c.action == ActionType.TRAIN
                            and str(c.item_type or "") in _veh)
                ]
                self._log(f"Expand TRAIN {want}")
                commands.append(CommandModel(
                    action=ActionType.TRAIN, item_type=want))
                return commands

        if not infantry_busy and cash >= 100:
            already_inf = any(
                c.action == ActionType.TRAIN
                and str(c.item_type or "") in ("e1", "e3")
                for c in commands
            )
            if not already_inf:
                want_e3 = (n_e1 >= 8 and n_e3 * 2 < n_e1
                           and self._can_produce_item(obs, "e3")
                           and cash >= 300)
                item = "e3" if want_e3 else "e1"
                if (n_e1 < int(self.EXPAND_GARRISON)
                        or want_e3) and self._can_produce_item(obs, item):
                    self._log(f"Expand TRAIN {item}")
                    commands.append(CommandModel(
                        action=ActionType.TRAIN, item_type=item))
                    return commands
        return commands

    def _optional_tech_defense(
        self, obs: OpenRAObservation, existing: List[CommandModel] | None = None,
    ) -> List[CommandModel]:
        """P3 land-first radar/tech/defense after eco/barracks (roles).

        Keeps BUILD_PRIORITY = powr→proc→barracks (rush critical path). Once
        barracks+proc exist and the rush is rolling (phase attack or enough
        combat), queue one next structure from the roles catalog:
          cheap defense_gun/turret → powr (if tight) → dome → weap → fix → tech.
        One BUILD per decide; skips if a BUILD is already queued this tick.
        """
        if any(getattr(c, "action", None) == ActionType.BUILD for c in (existing or [])):
            return []
        own = {
            str(getattr(b, "type", "") or "").lower()
            for b in (obs.buildings or [])
        }
        if not (own & self.BARRACKS_TYPES) or "proc" not in own:
            return []
        if power_in_deficit(obs):
            return self._try_build_power(obs, "P3 low-power")
        n_combat = n_combat_total(obs)
        # Do not divert cash from the a_short rifle rush.
        if not (self.phase == "attack" or n_combat >= self.RUSH_ATTACK_MOVE):
            return []
        queued = {
            str(getattr(p, "item", "") or "").lower()
            for p in (obs.production or [])
        }
        building_busy = any(
            str(getattr(p, "queue_type", "") or "").lower() in
            ("building", "buildings", "structure", "structures")
            and float(getattr(p, "progress", 0) or 0) < 0.99
            for p in (obs.production or [])
        )
        defense_busy = any(
            str(getattr(p, "queue_type", "") or "").lower() in
            ("defense", "defences", "defenses")
            and float(getattr(p, "progress", 0) or 0) < 0.99
            for p in (obs.production or [])
        )
        cash = int(getattr(obs.economy, "cash", 0) or 0)
        try:
            power_bal = (
                int(getattr(obs.economy, "power_provided", 0) or 0)
                - int(getattr(obs.economy, "power_drained", 0) or 0)
            )
        except (TypeError, ValueError):
            power_bal = 0

        def _try(items, min_cash: int, *, defense_queue: bool = False):
            if cash < min_cash:
                return None
            if defense_queue and defense_busy:
                return None
            if (not defense_queue) and building_busy:
                return None
            for item in items:
                if item in own or item in queued:
                    return None  # category already owned/queued (first hit)
            for item in items:
                if self._can_produce_item(obs, item):
                    return item
            return None

        # 1) Cheap base defense (roles defense_gun / defense_turret).
        if not (own & {"pbox", "hbox", "gun", "agun", "ftur", "tsla", "sam"}):
            item = _try(("pbox", "hbox", "ftur"), 700, defense_queue=True)
            if item:
                self._log(f"P3 tech/defense BUILD {item} (cheap defense)")
                return [CommandModel(action=ActionType.BUILD, item_type=item)]

        # 2) Power cushion before radar/tech drains (extra powr OK).
        if power_bal < 40 and "powr" not in queued and "apwr" not in queued:
            n_powr = sum(
                1 for b in (obs.buildings or [])
                if str(getattr(b, "type", "") or "").lower() in ("powr", "apwr")
            )
            if (n_powr < 3 and spendable_resources(obs) >= POWR_COST
                    and not building_busy
                    and self._can_produce_item(obs, "powr")):
                self._log("P3 tech/defense BUILD powr (power cushion)")
                return [CommandModel(action=ActionType.BUILD, item_type="powr")]

        # 3) Radar (dome) — ROLE_TECH entry point.
        if "dome" not in own and "dome" not in queued:
            item = _try(("dome",), 1600)
            if item:
                self._log(f"P3 tech/defense BUILD {item} (radar)")
                return [CommandModel(action=ActionType.BUILD, item_type=item)]

        # 4) War factory — still off BUILD_PRIORITY critical path.
        if "weap" not in own and "weap" not in queued:
            item = _try(("weap",), 2100)
            if item:
                self._log(f"P3 tech/defense BUILD {item} (warf)")
                return [CommandModel(action=ActionType.BUILD, item_type=item)]

        # 5) Service depot.
        if "weap" in own and "fix" not in own and "fix" not in queued:
            item = _try(("fix",), 1300)
            if item:
                self._log(f"P3 tech/defense BUILD {item} (repair)")
                return [CommandModel(action=ActionType.BUILD, item_type=item)]

        # 6) Tech center (Allied atek / Soviet stek).
        if ("dome" in own and "weap" in own
                and not (own & {"atek", "stek"})
                and not (queued & {"atek", "stek"})):
            item = _try(("atek", "stek"), 1600)
            if item:
                self._log(f"P3 tech/defense BUILD {item} (tech)")
                return [CommandModel(action=ActionType.BUILD, item_type=item)]

        # 7) One advanced defense once radar or weap unlocks it.
        if not (own & {"gun", "agun", "tsla", "sam"}):
            item = _try(("agun", "sam", "gun", "tsla"), 900, defense_queue=True)
            if item:
                self._log(f"P3 tech/defense BUILD {item} (adv defense)")
                return [CommandModel(action=ActionType.BUILD, item_type=item)]
        return []

    def _optional_naval_air(self, obs: OpenRAObservation, existing: List[CommandModel] | None = None) -> List[CommandModel]:
        """P3 light optional navy/air + production on water maps.

        Land/a_short: no-op (catalog forbids navy). Water maps: after barracks,
        queue one shipyard if available and cash allows; once a yard exists,
        TRAIN one light ship. Air: one helipad after war factory/barracks, then
        TRAIN one heli. Does not force navy on land-only maps.
        """
        from rl.map_catalog import allows_naval, allows_airbase_optional
        if any(getattr(c, 'action', None) == ActionType.BUILD for c in (existing or [])):
            return []
        # Avoid stacking another TRAIN on the same decide tick.
        if any(getattr(c, 'action', None) == ActionType.TRAIN for c in (existing or [])):
            return []
        map_name = str(getattr(getattr(obs, "map_info", None), "map_name", "") or "")
        if not allows_naval(map_name):
            return []
        out: List[CommandModel] = []
        own = {str(getattr(b, "type", "") or "").lower() for b in (obs.buildings or [])}
        queued = {
            str(getattr(p, "item", "") or "").lower()
            for p in (obs.production or [])
        }
        building_busy = any(
            str(getattr(p, "queue_type", "") or "").lower() in
            ("building", "buildings", "structure", "structures")
            and float(getattr(p, "progress", 0) or 0) < 0.99
            for p in (obs.production or [])
        )
        ship_busy = any(
            str(getattr(p, "queue_type", "") or "").lower() in
            ("ship", "ships", "naval", "boat", "boats")
            and float(getattr(p, "progress", 0) or 0) < 0.99
            for p in (obs.production or [])
        )
        air_busy = any(
            str(getattr(p, "queue_type", "") or "").lower() in
            ("aircraft", "plane", "planes", "heli", "helicopter")
            and float(getattr(p, "progress", 0) or 0) < 0.99
            for p in (obs.production or [])
        )
        cash = int(getattr(obs.economy, "cash", 0) or 0)
        # Naval yard: one syrd/spen.
        if not building_busy and cash >= 1500:
            if not (own & {"syrd", "spen"}) and not (queued & {"syrd", "spen"}):
                for item in ("syrd", "spen"):
                    if self._can_produce_item(obs, item):
                        out.append(CommandModel(action=ActionType.BUILD, item_type=item))
                        self._log(f"P3 optional naval BUILD {item} ({map_name})")
                        return out
        # Airbase after basic production exists.
        if allows_airbase_optional(map_name) and not building_busy and cash >= 1500:
            if not (own & {"hpad", "afld"}) and not (queued & {"hpad", "afld"}):
                has_prod = bool(own & ({"weap"} | self.BARRACKS_TYPES))
                if has_prod:
                    for item in ("hpad", "afld"):
                        if self._can_produce_item(obs, item):
                            out.append(CommandModel(action=ActionType.BUILD, item_type=item))
                            self._log(f"P3 optional airbase BUILD {item} ({map_name})")
                            return out
        # Production: one light ship once a yard is up.
        if (own & {"syrd", "spen"}) and not ship_busy and cash >= 500:
            for item in ("dd", "pt", "ss", "ca", "msub"):
                if self._can_produce_item(obs, item):
                    out.append(CommandModel(action=ActionType.TRAIN, item_type=item))
                    self._log(f"P3 optional naval TRAIN {item} ({map_name})")
                    return out
        # Production: one heli once a pad is up.
        if (own & {"hpad", "afld"}) and not air_busy and cash >= 900:
            for item in ("heli", "hind", "yak", "mig"):
                if self._can_produce_item(obs, item):
                    out.append(CommandModel(action=ActionType.TRAIN, item_type=item))
                    self._log(f"P3 optional air TRAIN {item} ({map_name})")
                    return out
        return out

    def _refresh_contact(self, obs: OpenRAObservation) -> None:
        """Update last_seen belief + last visible contact cell."""
        self.belief.update(obs)
        bldgs = list(obs.visible_enemy_buildings or [])
        ene = list(obs.visible_enemies or [])
        if bldgs or ene:
            origin = self._own_fact(obs) or own_anchor(obs)
            prod = [b for b in bldgs if str(b.type or "").lower() in self._PROD]
            pool = prod or bldgs or ene
            try:
                t = max(pool, key=lambda o: _cheb(origin, _xy(o)))
                self._last_contact = _xy(t)
            except (TypeError, ValueError):
                pass

    def _ghost_cell(self, obs: OpenRAObservation) -> Optional[Tuple[int, int]]:
        """Best belief ghost cell (last_seen), if any still confident."""
        slots = self.belief.select_enemies(obs)
        ghosts = [
            (u, conf) for (u, vis, conf, _t) in slots
            if float(vis) < 0.5 and float(conf) > 0.05
        ]
        if not ghosts:
            return None
        ghosts.sort(key=lambda t: -float(t[1]))
        try:
            return _xy(ghosts[0][0])
        except (TypeError, ValueError):
            return None

    def _push_cell(self, obs: OpenRAObservation) -> Optional[Tuple[int, int]]:
        """Raid > visible leftover > mental base > ghost > last contact > fog.

        Never resolve_beacon / BEACON_BY_MAP. Shared dest with adapter/live
        via war_objective. Mental base: densest seen enemy-building cluster.
        """
        self._refresh_contact(obs)
        return war_objective(
            obs, last_contact=self._last_contact, belief=self.belief)

    def _own_fact(self, obs: OpenRAObservation) -> Optional[Tuple[int, int]]:
        for b in obs.buildings or []:
            if str(getattr(b, "type", "") or "").lower() in ("fact", "afac"):
                try:
                    return _xy(b)
                except (TypeError, ValueError):
                    continue
        return None

    def _all_combat(self, obs: OpenRAObservation) -> list:
        return [
            u for u in obs.units
            if (u.type in self.COMBAT_UNIT_TYPES
                and "harv" not in str(u.type or "").lower())
        ]

    def _idle_combat(self, obs: OpenRAObservation) -> list:
        return [u for u in self._all_combat(obs) if u.is_idle]

    def _piled_on(self, units, dest, radius: int) -> int:
        if dest is None:
            return 0
        n = 0
        for u in units or []:
            try:
                if _cheb(_xy(u), dest) <= int(radius):
                    n += 1
            except (TypeError, ValueError):
                continue
        return n

    def _near_home(self, obs: OpenRAObservation, u) -> bool:
        origin = self._own_fact(obs)
        if origin is None:
            return True
        try:
            return _cheb(origin, _xy(u)) <= DEFEND_CELLS
        except (TypeError, ValueError):
            return False

    def _away_pile(self, obs: OpenRAObservation, combat) -> Optional[Tuple[int, int]]:
        """Centroid of combat units far from home if they form a pile."""
        origin = self._own_fact(obs) or own_anchor(obs)
        away = []
        for u in combat or []:
            try:
                if _cheb(origin, _xy(u)) > DEFEND_CELLS:
                    away.append(u)
            except (TypeError, ValueError):
                continue
        if len(away) < MIN_PILE_FOR_HUNT:
            return None
        try:
            cx = sum(int(u.cell_x) for u in away) // len(away)
            cy = sum(int(u.cell_y) for u in away) // len(away)
        except (TypeError, ValueError):
            return None
        pile = (cx, cy)
        if self._piled_on(away, pile, ARRIVED_CELLS) < MIN_PILE_FOR_HUNT:
            return None
        return pile

    def _handle_expand_combat(
        self, obs: OpenRAObservation, combat, idle, dest,
    ) -> List[CommandModel]:
        """Turtle to 12 tanks, send tanks only, retreat under 7.

        Bench 0/8 after the 8-tank hold: ARMY_ATTACK_MOVE walked e1+tanks
        into easy's 20-40 building base (UnitLimits 7 tanks + 11 guns).
        Peak tnk=23 still lost because replacements dribbled across the map.
        When committed with no visible leftover, fog/remnant-hunt so a lead
        finishes (bench seed 8047: 76 tanks, ene_nb=1 fogged, incomplete).
        """
        commands: List[CommandModel] = []
        n_tnk = self._count_type(obs, "1tnk", "2tnk", "3tnk")
        if n_tnk >= int(self.EXPAND_PUSH_TANKS):
            self._expand_committed = True
        if n_tnk < int(self.EXPAND_RETREAT_TANKS):
            self._expand_committed = False
        tanks = [u for u in combat if self._is_tank(u)]
        home = self._own_fact(obs) or own_anchor(obs)

        if not self._expand_committed:
            away = [u for u in tanks if not self._near_home(obs, u)]
            if away and home is not None:
                for u in away[:12]:
                    try:
                        commands.append(CommandModel(
                            action=ActionType.ATTACK_MOVE,
                            actor_id=int(u.actor_id),
                            target_x=int(home[0]),
                            target_y=int(home[1]),
                        ))
                    except (TypeError, ValueError):
                        continue
                if commands:
                    self._log(
                        f"Expand retreat {len(commands)}/{n_tnk} -> {home}")
                    return commands
            scouts = [u for u in idle if not self._is_tank(u)]
            n_scout = min(len(scouts), int(self.EXPAND_HOLD_SCOUTS))
            if n_scout:
                fog = fog_scout_destinations(obs, n_scout)
                for u, d in zip(scouts[:n_scout], fog):
                    commands.append(CommandModel(
                        action=ActionType.ATTACK_MOVE,
                        actor_id=int(u.actor_id),
                        target_x=int(d[0]),
                        target_y=int(d[1]),
                    ))
                if commands:
                    self._log(f"Expand hold fog scout {len(commands)}")
            return commands

        leftover = bool(
            getattr(obs, "visible_enemy_buildings", None)
            or getattr(obs, "visible_enemies", None)
        )
        # Fogged remnant: war_objective may be None / feet. Spread hunt.
        if dest is None or not leftover:
            pile = None if leftover else self._away_pile(obs, tanks)
            hunt_anchor = self._last_contact or pile
            if hunt_anchor is not None:
                dest = hunt_near_cell(obs, hunt_anchor)
            if dest is None:
                fog = fog_scout_destinations(obs, 1)
                if fog:
                    dest = (int(fog[0][0]), int(fog[0][1]))
        if dest is None:
            return commands
        # Idle only: re-issuing AM every macro tick stutters the ball.
        movers = [u for u in tanks if bool(getattr(u, "is_idle", False))]
        # Piled on empty ground (not idle while AM): re-task a few.
        if not movers and not leftover:
            pile = self._away_pile(obs, tanks)
            if pile is not None:
                for u in tanks:
                    try:
                        if _cheb(_xy(u), pile) <= ARRIVED_CELLS:
                            movers.append(u)
                    except (TypeError, ValueError):
                        continue
        n_go = min(len(movers), 12)
        # Spread across fog cells when hunting shroud leftovers.
        dests: List[Tuple[int, int]] = [(int(dest[0]), int(dest[1]))]
        if not leftover and n_go > 1:
            fog = fog_scout_destinations(obs, max(0, n_go - 1))
            dests.extend((int(d[0]), int(d[1])) for d in fog)
        while len(dests) < n_go:
            dests.append(dests[0])
        for u, d in zip(movers[:n_go], dests[:n_go]):
            try:
                commands.append(CommandModel(
                    action=ActionType.ATTACK_MOVE,
                    actor_id=int(u.actor_id),
                    target_x=int(d[0]),
                    target_y=int(d[1]),
                ))
            except (TypeError, ValueError):
                continue
        if commands:
            self._log(
                f"Expand tank AM {len(commands)}/{n_tnk} toward {dests[:n_go]}")
        return commands

    def _handle_combat(self, obs: OpenRAObservation) -> List[CommandModel]:
        commands: List[CommandModel] = []
        idle = self._idle_combat(obs)
        combat = self._all_combat(obs)
        dest = self._push_cell(obs)
        raids = (self._yard_raid_targets(obs) if self.mode == "expand"
                 else home_raid_targets(obs))
        n_combat = n_combat_total(obs)
        leftover = bool(obs.visible_enemy_buildings or obs.visible_enemies)

        if raids:
            origin = self._own_fact(obs) or own_anchor(obs)
            threat = self._nearest_xy(origin, [_xy(t) for t in raids]) if origin else None
            if threat is None:
                try:
                    threat = _xy(raids[0])
                except (TypeError, ValueError):
                    threat = dest
            home_units = [u for u in combat if self._near_home(obs, u)]
            if not home_units:
                home_units = [u for u in idle if self._near_home(obs, u)]
            if not home_units or threat is None:
                return commands
            rx, ry = int(threat[0]), int(threat[1])
            for u in home_units[:12]:
                try:
                    commands.append(CommandModel(
                        action=ActionType.ATTACK_MOVE,
                        actor_id=int(u.actor_id),
                        target_x=rx,
                        target_y=ry,
                    ))
                except (TypeError, ValueError):
                    continue
            self._log(f"Peel yard raid {len(commands)} -> ({rx},{ry})")
            return commands

        if self.mode == "expand":
            return self._handle_expand_combat(obs, combat, idle, dest)

        # Visible contact is 0 but army is piled away from home (fog leftovers).
        # AttackMove units are often not is_idle, so idle-only scout never fired.
        pile = None if leftover else self._away_pile(obs, combat)
        if pile is not None:
            hunt_anchor = self._last_contact or pile
            hunt = hunt_near_cell(obs, hunt_anchor)
            if n_combat >= PACK_ARMY:
                commands.append(CommandModel(
                    action=ActionType.ARMY_ATTACK_MOVE,
                    target_x=int(hunt[0]),
                    target_y=int(hunt[1]),
                ))
                self._log(f"Remnant army AM {n_combat} toward {hunt}")
                return commands
            movers = list(idle)
            if not movers:
                for u in combat:
                    try:
                        if _cheb(_xy(u), pile) <= ARRIVED_CELLS:
                            movers.append(u)
                    except (TypeError, ValueError):
                        continue
            dests = self._remnant_dests(obs, hunt, min(len(movers), 12))
            n_go = min(len(movers), len(dests), 12)
            for u, d in zip(movers[:n_go], dests):
                commands.append(CommandModel(
                    action=ActionType.ATTACK_MOVE,
                    actor_id=int(u.actor_id),
                    target_x=int(d[0]),
                    target_y=int(d[1]),
                ))
            if commands:
                self._log(f"Remnant sweep {len(commands)} toward {dests[:n_go]}")
            return commands

        if not idle:
            return commands

        # Early fog scout: once a few combat exist and we still lack a mental
        # base / leftover, open relative fog away from home (no GPS coords).
        has_base = getattr(self.belief, "enemy_base_xy", None) is not None
        if (n_combat >= self.EARLY_SCOUT_COMBAT
                and not leftover and not raids and not has_base
                and n_combat < self.RUSH_ATTACK_MOVE):
            n_scout = min(len(idle), self.N_SCOUTS)
            fog = fog_scout_destinations(obs, n_scout)
            for u, d in zip(idle[:n_scout], fog):
                commands.append(CommandModel(
                    action=ActionType.ATTACK_MOVE,
                    actor_id=int(u.actor_id),
                    target_x=int(d[0]),
                    target_y=int(d[1]),
                ))
            if commands:
                self._log(
                    f"Early fog scout {len(commands)}/{n_combat} toward "
                    f"{fog[:len(commands)]}"
                )
                return commands

        piled = 0
        if dest is not None:
            piled = sum(1 for u in idle if _cheb(_xy(u), dest) <= 6)

        push_ok = self.mode != "expand" or self._expand_push_ready(obs)

        # Pack legal: group order the student can clone (mask PACK_ARMY).
        if push_ok and n_combat >= PACK_ARMY and dest is not None:
            if leftover or piled < max(4, n_combat // 2):
                commands.append(CommandModel(
                    action=ActionType.ARMY_ATTACK_MOVE,
                    target_x=int(dest[0]),
                    target_y=int(dest[1]),
                ))
                self._log(
                    f"Army attack-move {len(idle)}/{n_combat} toward {dest}"
                )
                return commands

        # Rush: walk the whole idle blob. attack_move is legal without pack 12.
        # If already piled on an empty dest, fall through to fog scout.
        if (push_ok and n_combat >= self.RUSH_ATTACK_MOVE and dest is not None
                and (leftover or piled < max(3, n_combat // 3))):
            n_go = min(len(idle), 12)
            for u in idle[:n_go]:
                commands.append(CommandModel(
                    action=ActionType.ATTACK_MOVE,
                    actor_id=int(u.actor_id),
                    target_x=int(dest[0]),
                    target_y=int(dest[1]),
                ))
            self._log(f"Rush AM {n_go}/{n_combat} toward {dest}")
            return commands

        n_scout = self.N_SCOUTS
        dests = self._remnant_dests(obs, dest, n_scout) if not leftover else []
        if leftover or not dests:
            dests = []
            if dest is not None:
                dests.append(dest)
            fog = fog_scout_destinations(obs, max(0, n_scout - len(dests)))
            dests.extend(fog)
        for u, d in zip(idle[:n_scout], dests):
            commands.append(CommandModel(
                action=ActionType.ATTACK_MOVE,
                actor_id=int(u.actor_id),
                target_x=int(d[0]),
                target_y=int(d[1]),
            ))
        if commands:
            self._log(f"Scout {len(commands)} toward {dests[:len(commands)]}")
        return commands

    def _remnant_dests(
        self, obs: OpenRAObservation, dest: Optional[Tuple[int, int]], n: int,
    ) -> List[Tuple[int, int]]:
        """Hunt + fog cells. Never pad with map beacon GPS."""
        n = max(0, int(n))
        if n <= 0:
            return []
        dests: List[Tuple[int, int]] = []
        if dest is not None:
            dests.append((int(dest[0]), int(dest[1])))
        fog = fog_scout_destinations(obs, max(0, n - len(dests)))
        dests.extend(fog)
        return dests[:n]

    def _find_attack_target(self, obs: OpenRAObservation) -> Tuple[int, int]:
        d = self._push_cell(obs)
        if d is not None:
            return d
        if self._last_contact is not None:
            return hunt_near_cell(obs, self._last_contact)
        fog = fog_scout_destinations(obs, 1)
        if fog:
            return int(fog[0][0]), int(fog[0][1])
        fact = self._own_fact(obs)
        if fact is not None:
            return fact
        return own_anchor(obs)
