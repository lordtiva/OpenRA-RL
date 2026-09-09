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
    fog scout. When belief empty: resolve_beacon opening-SFT prior, else fog
    (tapes eco_and_combat_mental_v4).
  - early fog scout away from home once a few combat exist (relative fog).
  - peel de raid; TRAIN e1 durante el push; 2 harvs; 0 guards / 0 APC
  - P3 land-first tech/defense AFTER eco/barracks (optional, not BUILD_PRIORITY):
    cheap defense → dome → weap → fix → atek/stek. Gates on rush rolling so
    a_short Allies rush stays intact. Navy/air stays light via _optional_naval_air.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from examples.scripted_bot import ScriptedBot
from openra_env.models import ActionType, CommandModel, OpenRAObservation
from rl.action_adapter import PACK_ARMY, n_combat_total
from rl.auto_support import (
    ARRIVED_CELLS,
    DEFEND_CELLS,
    MIN_PILE_FOR_HUNT,
    fog_scout_destinations,
    home_raid_targets,
    hunt_near_cell,
    war_nudge_cell,
)
from rl.obs_encoding import EnemyBeliefStore, resolve_beacon


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
    _PROD = frozenset({
        "fact", "afac", "proc", "weap", "tent", "barr", "kenn",
        "hpad", "afld", "syrd",
    })

    def __init__(self, verbose: bool = False, rush_attack_move: int | None = None):
        """Optional per-instance RUSH override for benches; class default unchanged."""
        super().__init__(verbose=verbose)
        if rush_attack_move is not None:
            self.RUSH_ATTACK_MOVE = int(rush_attack_move)
        # Per-episode fog belief + last visible contact (map-agnostic hunt).
        self.belief = EnemyBeliefStore()
        self._last_contact: Optional[Tuple[int, int]] = None
        self._last_tick: Optional[int] = None

    def decide(self, obs: OpenRAObservation):
        """Reset belief between episodes if the same instance is reused."""
        try:
            tick = int(getattr(obs, "tick", 0) or 0)
        except (TypeError, ValueError):
            tick = 0
        if self._last_tick is not None and tick < self._last_tick:
            self.belief.reset()
            self._last_contact = None
        self._last_tick = tick
        return super().decide(obs)

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
              and n_combat >= self.RUSH_ATTACK_MOVE):
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
                and obs.economy.cash >= 1100):
            commands.append(CommandModel(action=ActionType.TRAIN, item_type="harv"))
            self._log("Training harv (teacher eco)")
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
            if (n_powr < 3 and cash >= 350 and not building_busy
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
            origin = self._own_fact(obs) or (12, 16)
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
        """Raid > visible leftover > mental base > ghost > last contact >
        beacon (opening prior) > fog.

        Visible / leftover / ghost / mental-base stay first. Only when those
        are absent, prefer resolve_beacon(obs) as opening-SFT prior; else
        fog_scout_destinations. Mental base: densest seen enemy-building
        cluster from belief store.
        """
        self._refresh_contact(obs)
        raids = home_raid_targets(obs)
        if raids:
            origin = self._own_fact(obs) or (12, 16)
            t = min(raids, key=lambda o: _cheb(origin, _xy(o)))
            return _xy(t)
        nudge, is_raid = war_nudge_cell(obs)
        if nudge is not None and not is_raid:
            return int(nudge[0]), int(nudge[1])
        bldgs = list(obs.visible_enemy_buildings or [])
        prod = [b for b in bldgs if str(b.type or "").lower() in self._PROD]
        if prod:
            origin = self._own_fact(obs) or (12, 16)
            t = max(prod, key=lambda b: _cheb(origin, _xy(b)))
            return _xy(t)
        if bldgs:
            return _xy(bldgs[0])
        if obs.visible_enemies:
            return _xy(obs.visible_enemies[0])
        # Strategic: remembered enemy base after leftovers cleared (not GPS).
        base = getattr(self.belief, "enemy_base_xy", None)
        if base is not None:
            return int(base[0]), int(base[1])
        ghost = self._ghost_cell(obs)
        if ghost is not None:
            return int(ghost[0]), int(ghost[1])
        combat = self._all_combat(obs)
        anchor = self._last_contact
        if anchor is not None:
            n_at = self._piled_on(combat, anchor, ARRIVED_CELLS)
            if n_at >= MIN_PILE_FOR_HUNT:
                return hunt_near_cell(obs, anchor)
            return int(anchor[0]), int(anchor[1])
        # Belief empty: beacon is opening-SFT prior when present; else fog.
        beacon = resolve_beacon(obs)
        if beacon is not None:
            return int(beacon[0]), int(beacon[1])
        fog = fog_scout_destinations(obs, 1)
        if fog:
            return int(fog[0][0]), int(fog[0][1])
        return None

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
        origin = self._own_fact(obs) or (12, 16)
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

    def _handle_combat(self, obs: OpenRAObservation) -> List[CommandModel]:
        commands: List[CommandModel] = []
        idle = self._idle_combat(obs)
        combat = self._all_combat(obs)
        dest = self._push_cell(obs)
        raids = home_raid_targets(obs)
        n_combat = n_combat_total(obs)
        leftover = bool(obs.visible_enemy_buildings or obs.visible_enemies)

        if raids:
            home_idle = [u for u in idle if self._near_home(obs, u)]
            if not home_idle:
                return commands
            rx, ry = dest if dest is not None else _xy(raids[0])
            for u in home_idle[:8]:
                commands.append(CommandModel(
                    action=ActionType.ATTACK_MOVE,
                    actor_id=int(u.actor_id),
                    target_x=int(rx),
                    target_y=int(ry),
                ))
            self._log(f"Peel raid {len(commands)} idle -> ({rx},{ry})")
            return commands

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

        # Pack legal: group order the student can clone (mask PACK_ARMY).
        if n_combat >= PACK_ARMY and dest is not None:
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
        if (n_combat >= self.RUSH_ATTACK_MOVE and dest is not None
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
        return 12, 16
