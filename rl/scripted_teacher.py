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
    # 6 rifles no cierran a_short vs beginner; entrar en attack más tarde
    # y seguir produciendo durante el push.
    INFANTRY_TRAIN_TARGET = 16

    def _handle_production(self, obs: OpenRAObservation) -> List[CommandModel]:
        commands = super()._handle_production(obs)
        if self.phase != "attack":
            return commands
        already_train = any(
            getattr(getattr(c, "action", None), "value", None) == "train"
            or str(getattr(c, "action", "")) == "train"
            for c in commands
        )
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
