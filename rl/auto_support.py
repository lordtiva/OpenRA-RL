"""Autonomía de soporte — capa de confort APM (Pilar B, full-stack Run3).

Resuelve el cuello de botella 1 decisión / 80 ticks sin robar ancho de banda
a la red. El agente decide ESTRATEGIA (qué entrenar, dónde atacar); esta capa
resuelve micro-mantenimiento obvio en paralelo, igual que BuildingRepairBotModule
y PowerDownBotModule del bot hard.

Diseño:
- Se inyecta en rollout.py DESPUÉS de index_to_command_effective(), extendiendo
  action.commands con comandos extra. El log_prob de PPO solo cubre el comando
  estratégico — el soporte es gratis y no entra en el buffer.
- Flag --auto-support (default False en Run2, True en Run3/v4). Sin flag, 0 impacto.
- Reparación: si cash>500 y edificio hp<35% y no está ya reparándose, emite 1
  repair por bloque (máx 2 para no spamear). Es el umbral del hard.
- Energía: si power_drained > power_provided, apaga dome/tsla/mslo (prioridad baja).

No genera reward — evita defense_loss/hold_zero ya existentes.
"""

from openra_env.models import ActionType, CommandModel


# Tipos que el hard apaga cuando hay brownout (ai.yaml PowerDownBotModule)
_POWER_DOWN_TYPES = {"dome", "tsla", "mslo", "atag", "stag"}


def support_commands(obs, max_repairs: int = 2):
    """Lista de CommandModel de soporte para esta observación.

    Llamar con la obs que ve la red ANTES de ejecutar el step. Devuelve [] si
    no hay nada que hacer. No toca el estado del shaper ni el buffer de PPO.
    """
    out = []
    eco = getattr(obs, "economy", None)
    blds = getattr(obs, "buildings", []) or []
    if not blds:
        return out

    cash = int(getattr(eco, "cash", 0) or 0) if eco else 0

    # 1) Auto-repair — umbral hard (35%)
    if cash > 500:
        repairs = 0
        for b in blds:
            if repairs >= max_repairs:
                break
            hp = float(getattr(b, "hp_percent", 1.0) or 1.0)
            if hp < 0.35 and not bool(getattr(b, "is_repairing", False)):
                out.append(CommandModel(action=ActionType.REPAIR, actor_id=int(b.actor_id)))
                repairs += 1

    # 2) Auto-power_down — solo si balance negativo
    if eco is not None:
        provided = int(getattr(eco, "power_provided", 0) or 0)
        drained = int(getattr(eco, "power_drained", 0) or 0)
        if drained > provided:
            for b in blds:
                if b.type in _POWER_DOWN_TYPES and bool(getattr(b, "is_powered", True)):
                    out.append(CommandModel(action=ActionType.POWER_DOWN, actor_id=int(b.actor_id)))
                    break  # uno por bloque

    return out
