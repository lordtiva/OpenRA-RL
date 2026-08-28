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

def _place_near_base(obs):
    """A cell next to the construction yard / any own building (not a spawn rewrite)."""
    for b in getattr(obs, "buildings", None) or []:
        t = str(getattr(b, "type", "")).lower()
        if t in ("fact", "proc", "powr", "apwr", "barr", "tent"):
            return int(b.cell_x) + 4, int(b.cell_y) + 2
    for u in getattr(obs, "units", None) or []:
        return int(u.cell_x) + 3, int(u.cell_y) + 1
    return 0, 0


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

    # 0b) Auto-proc + auto-harv — push the economy if the policy is rifle-spamming.
    #     Missing proc: BUILD if it is in available_production (do not deadlock
    #     BUILD/PLACE of proc — those stay unmasked even when we cannot build yet).
    #     Proc ready in the queue: PLACE near the conyard. Has proc but no harvester:
    #     TRAIN harv if the war factory lists it.
    avail = set(getattr(obs, "available_production", []) or [])
    prod = list(getattr(obs, "production", []) or [])
    proc_queued = any(str(getattr(p, "item", "")).lower() == "proc" for p in prod)
    proc_ready = any(
        str(getattr(p, "item", "")).lower() == "proc"
        and float(getattr(p, "progress", 0) or 0) >= 1.0
        for p in prod
    )
    if not has_proc:
        if proc_ready:
            ax, ay = _place_near_base(obs)
            out.append(CommandModel(
                action=ActionType.PLACE_BUILDING, item_type="proc",
                target_x=ax, target_y=ay))
        elif (not proc_queued) and "proc" in avail and cash >= 2000:
            out.append(CommandModel(action=ActionType.BUILD, item_type="proc"))
    else:
        has_harv = False
        if eco is not None and int(getattr(eco, "harvester_count", 0) or 0) > 0:
            has_harv = True
        if not has_harv:
            has_harv = any("harv" in str(getattr(u, "type", "")).lower() for u in units)
        harv_queued = any("harv" in str(getattr(p, "item", "")).lower() for p in prod)
        if (not has_harv) and (not harv_queued) and "harv" in avail:
            out.append(CommandModel(action=ActionType.TRAIN, item_type="harv"))

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
