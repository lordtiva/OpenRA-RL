"""Maestro BC para Capa 1: ScriptedBot alineado al MDP de train.

El example original construye barracks ANTES de proc y ataca con attack_move
por unidad. Este MDP enmascara BUILD no-económico y TRAIN de combate hasta
proc+harv, y el asalto de grupo es army_attack_move. El teacher clona el
build order legal y el remate de ejército.
"""
from __future__ import annotations

from typing import List

from examples.scripted_bot import ScriptedBot
from openra_env.models import ActionType, CommandModel, OpenRAObservation


class ScriptedTeacher(ScriptedBot):
    BUILD_PRIORITY = [
        "powr",
        "proc",
        "barracks",
        "weap",
        "powr",
    ]

    def _handle_combat(self, obs: OpenRAObservation) -> List[CommandModel]:
        commands: List[CommandModel] = []
        if self.phase != "attack":
            return commands
        commands.extend(self._handle_unload(obs))
        idle_fighters = [
            u for u in obs.units
            if (u.type in self.COMBAT_UNIT_TYPES
                and u.is_idle
                and u.actor_id not in self._guards_assigned)
        ]
        if len(idle_fighters) < 2:
            return commands
        target_x, target_y = self._find_attack_target(obs)
        commands.append(CommandModel(
            action=ActionType.ARMY_ATTACK_MOVE,
            target_x=target_x,
            target_y=target_y,
        ))
        self._log(
            f"Army attack-move {len(idle_fighters)} units "
            f"toward ({target_x}, {target_y})"
        )
        return commands
