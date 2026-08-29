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
- Asalto sostenido (Capa 0, corte Run9 iter 442): con proc+harv y ≥N rifles,
  el entorno emite army_attack_move cada bloque aunque la red haya elegido
  train/build. Sin esto el push nunca arranca (hist army ~0% a iter 443).

No genera reward — evita defense_loss/hold_zero ya existentes.
"""

from openra_env.models import ActionType, CommandModel
from rl.obs_encoding import BEACON_BY_MAP

# Tipos que el hard apaga cuando hay brownout (ai.yaml PowerDownBotModule)
_POWER_DOWN_TYPES = {"dome", "tsla", "mslo", "atag", "stag"}
_NON_COMBAT = ("harv", "mcv")
# Don't march a 1-rifle scout; wait for a real army (Run7 collapse was
# combat-without-eco; this gate is the army half of that lesson).
MIN_ARMY_FOR_ASSAULT = 4

def _place_near_base(obs):
    """A cell next to the construction yard / any own building (not a spawn rewrite)."""
    for b in getattr(obs, "buildings", None) or []:
        t = str(getattr(b, "type", "")).lower()
        if t in ("fact", "proc", "powr", "apwr", "barr", "tent"):
            return int(b.cell_x) + 4, int(b.cell_y) + 2
    for u in getattr(obs, "units", None) or []:
        return int(u.cell_x) + 3, int(u.cell_y) + 1
    return 0, 0


def _is_combat(u) -> bool:
    ut = str(getattr(u, "type", "")).lower()
    return not any(tag in ut for tag in _NON_COMBAT)


def _combat_units(units):
    return [u for u in units if _is_combat(u)]


def _has_harvester(obs, eco, units, prod) -> bool:
    if eco is not None and int(getattr(eco, "harvester_count", 0) or 0) > 0:
        return True
    if any("harv" in str(getattr(u, "type", "")).lower() for u in units):
        return True
    return any("harv" in str(getattr(p, "item", "")).lower() for p in prod)


def _push_cell(obs, last_push):
    """Celda de asalto: último push de la política, si no enemigo visible, si no beacon."""
    if last_push is not None:
        return int(last_push[0]), int(last_push[1])
    enemies = list(getattr(obs, "visible_enemies", None) or []) + list(
        getattr(obs, "visible_enemy_buildings", None) or [])
    if enemies:
        e = enemies[0]
        return int(e.cell_x), int(e.cell_y)
    map_name = str(getattr(getattr(obs, "map_info", None), "map_name", "") or "")
    beacon = BEACON_BY_MAP.get(map_name)
    if beacon is not None:
        return int(beacon[0]), int(beacon[1])
    return None


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

    # 0a) Auto-deploy MCV — without a conyard, auto-proc cannot fire and the
    #     policy collapses to deploy-once-then-no_op (Run 8). Gratis for PPO.
    has_fact = any(str(getattr(b, "type", "")).lower() in ("fact", "afac")
                   for b in blds)
    if not has_fact:
        for u in units:
            if "mcv" in str(getattr(u, "type", "")).lower():
                out.append(CommandModel(
                    action=ActionType.DEPLOY, actor_id=int(u.actor_id)))
                break

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
    if not has_fact:
        pass  # wait for the deploy above; BUILD proc needs a conyard
    elif not has_proc:
        if proc_ready:
            ax, ay = _place_near_base(obs)
            out.append(CommandModel(
                action=ActionType.PLACE_BUILDING, item_type="proc",
                target_x=ax, target_y=ay))
        elif (not proc_queued) and "proc" in avail and cash >= 2000:
            out.append(CommandModel(action=ActionType.BUILD, item_type="proc"))
    else:
        has_harv = _has_harvester(obs, eco, units, prod)
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

    # 3) Asalto sostenido + keep-alive. Gratis para PPO.
    #    Con eco lista y ≥MIN_ARMY de combate, army_attack_move cada bloque
    #    (el C# salta harvesters). No espera a que la red lo muestreé —
    #    Run9 iters 340-443: army hist → 0% e incomplete 24%→60%.
    #    Si aún no hay ejército, solo keep-alive del last_push de la política.
    combat = _combat_units(units)
    has_harv = _has_harvester(obs, eco, units, prod)
    dest = _push_cell(obs, last_push)
    assault = (has_proc and has_harv and dest is not None
               and len(combat) >= MIN_ARMY_FOR_ASSAULT)
    if assault:
        px, py = dest
        out.append(CommandModel(
            action=ActionType.ARMY_ATTACK_MOVE, target_x=px, target_y=py))
    elif last_push is not None and has_proc:
        px, py = last_push
        n_push = 0
        for u in combat:
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
