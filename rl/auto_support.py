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
- Cosecha: si harv está idle y hay proc, emite 1 harvest (auto al ore más cercano).
- Energía: si power_drained > power_provided, apaga dome/tsla/mslo (prioridad baja).
- Push keep-alive: si la política ya eligió army/attack_move, los ociosos de
  combate siguen hacia esa celda entre decisiones (APM de confort, como repair).

No genera reward — evita defense_loss/hold_zero ya existentes.
"""

from openra_env.models import ActionType, CommandModel

# Tipos que el hard apaga cuando hay brownout (ai.yaml PowerDownBotModule)
_POWER_DOWN_TYPES = {"dome", "tsla", "mslo", "atag", "stag"}
_NON_COMBAT = ("harv", "mcv")

def support_commands(obs, last_push=None, max_repairs: int = 2):
    """Lista de CommandModel de soporte para esta observación.

    Llamar con la obs que ve la red ANTES de ejecutar el step. Devuelve [] si
    no hay nada que hacer. No toca el estado del shaper ni el buffer de PPO.
    """
    out = []
    eco = getattr(obs, "economy", None)
    blds = getattr(obs, "buildings", []) or []
    units = getattr(obs, "units", []) or []
    cash = int(getattr(eco, "cash", 0) or 0) if eco else 0

    # 0) Auto-harvest — si harv está idle y hay proc, mandarlo a cosechar.
    #    Gratis para PPO (igual que repair): no roba decisión estratégica.
    #    El engine con actor_id solo hace auto-harvest al ore más cercano.
    has_proc = any(getattr(b, "type", "") == "proc" for b in blds)
    if has_proc:
        for u in units:
            ut = str(getattr(u, "type", "")).lower()
            if "harv" in ut and bool(getattr(u, "is_idle", False)):
                out.append(CommandModel(action=ActionType.HARVEST, actor_id=int(u.actor_id)))
                break  # uno por bloque (no spamear)

    # 0b) Auto-proc — si no hay proc, hay cash y hay fact/barr para producirlo, pedirlo.
    #     Evita el starve infinito del iter100 (6 edificios sin refinería → cap 0 → ore 0).
    #     Gratis, igual que harvest: el shaper ya premia proc, esto solo rompe el deadlock.
    if not has_proc and cash >= 1400:
        has_fact = any(getattr(b, "type", "") == "fact" for b in blds)
        # available_production viene de obs.available_production (si la red puede ver proc)
        avail = set(getattr(obs, "available_production", []) or [])
        can_build_proc = "proc" in avail or has_fact  # fact puede producir proc
        if can_build_proc and not any(getattr(p, "item", "") == "proc" for p in getattr(obs, "production", []) or []):
            # emitir BUILD proc (el engine lo encola, luego place_building vendrá de la red o de otro tick)
            out.append(CommandModel(action=ActionType.BUILD, item_type="proc"))
            # no break: puede coexistir con harvest (pero harvest no dispara sin proc, así que ok)

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

    # 3) Keep-alive de push. army_attack_move del C# mueve a todos EN ESE
    #    bloque; entre decisiones (train/build/etc.) los rifles ociosos se
    #    quedan parados. Si hay un destino de push vivo, re-emitimos
    #    AttackMove a los idle de combate. Gratis para PPO.
    if last_push is not None:
        px, py = last_push
        n_push = 0
        for u in units:
            ut = str(getattr(u, "type", "")).lower()
            if any(tag in ut for tag in _NON_COMBAT):
                continue
            if not bool(getattr(u, "is_idle", False)):
                continue
            out.append(CommandModel(
                action=ActionType.ATTACK_MOVE,
                actor_id=int(u.actor_id),
                target_x=int(px),
                target_y=int(py),
            ))
            n_push += 1
            if n_push >= 8:
                break

    return out
