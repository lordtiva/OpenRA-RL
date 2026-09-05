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
  - leftover visible > beacon; si el blob ya está en el dest vacío, scout
  - peel de raid; TRAIN e1 durante el push; 2 harvs; 0 guards / 0 APC
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from examples.scripted_bot import ScriptedBot
from openra_env.models import ActionType, CommandModel, OpenRAObservation
from rl.action_adapter import PACK_ARMY, n_combat_total
from rl.auto_support import (
    DEFEND_CELLS,
    fog_scout_destinations,
    home_raid_targets,
    war_nudge_cell,
)
from rl.obs_encoding import resolve_beacon


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
    MIN_HARVS = 2
    GUARD_COUNT = 0
    _PROD = frozenset({
        "fact", "afac", "proc", "weap", "tent", "barr", "kenn",
        "hpad", "afld", "syrd",
    })

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
            return commands
        already_train = any(c.action == ActionType.TRAIN for c in commands)
        if already_train:
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
        return commands

    def _push_cell(self, obs: OpenRAObservation) -> Optional[Tuple[int, int]]:
        """Raid en casa > prod visible > edificio/unidad > beacon. Nunca centro."""
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
        beacon = resolve_beacon(obs)
        if beacon is not None:
            return int(beacon[0]), int(beacon[1])
        return None

    def _own_fact(self, obs: OpenRAObservation) -> Optional[Tuple[int, int]]:
        for b in obs.buildings or []:
            if str(getattr(b, "type", "") or "").lower() in ("fact", "afac"):
                try:
                    return _xy(b)
                except (TypeError, ValueError):
                    continue
        return None

    def _idle_combat(self, obs: OpenRAObservation) -> list:
        return [
            u for u in obs.units
            if (u.type in self.COMBAT_UNIT_TYPES
                and u.is_idle
                and "harv" not in str(u.type or "").lower())
        ]

    def _near_home(self, obs: OpenRAObservation, u) -> bool:
        origin = self._own_fact(obs)
        if origin is None:
            return True
        try:
            return _cheb(origin, _xy(u)) <= DEFEND_CELLS
        except (TypeError, ValueError):
            return False

    def _handle_combat(self, obs: OpenRAObservation) -> List[CommandModel]:
        commands: List[CommandModel] = []
        idle = self._idle_combat(obs)
        if not idle:
            return commands
        dest = self._push_cell(obs)
        raids = home_raid_targets(obs)
        n_combat = n_combat_total(obs)
        home_idle = [u for u in idle if self._near_home(obs, u)]
        leftover = bool(obs.visible_enemy_buildings or obs.visible_enemies)

        if raids:
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
            # Blob already on an empty dest (beacon mill): scout leftovers.

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
        dests: List[Tuple[int, int]] = []
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

    def _find_attack_target(self, obs: OpenRAObservation) -> Tuple[int, int]:
        d = self._push_cell(obs)
        if d is not None:
            return d
        beacon = resolve_beacon(obs)
        if beacon is not None:
            return int(beacon[0]), int(beacon[1])
        fact = self._own_fact(obs)
        if fact is not None:
            return fact
        return 12, 16
