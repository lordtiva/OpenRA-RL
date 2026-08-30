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
- Push keep-alive: ociosos de combate hacia enemigo visible / beacon / hunt.
- Asalto sostenido (Capa 0, corte Run9 iter 442 + visor 817): con proc+harv
  y ≥N rifles, el entorno manda el ejército al enemigo visible o al beacon
  aunque la red elija train/build. NO usa last_push de la política (la
  cabeza de celda apunta al blob propio → hormiguero en casa). NO re-emite
  army_attack_move a unidades que ya caminan (el re-order cada 80 ticks
  cancelaba el path).
- Hunt (iter ~854): si ≥N combate ya están en el beacon y no hay objetivo
  visible, el destino rota por waypoints al sur/oeste del NE. El pile-up
  en (95,11) dejaba edificios resagados en niebla y el episodio iba a
  timeout (incomplete con enB 5–10).

No genera reward — evita defense_loss/hold_zero ya existentes.
"""

from openra_env.models import ActionType, CommandModel
from rl.obs_encoding import resolve_beacon

# Tipos que el hard apaga cuando hay brownout (ai.yaml PowerDownBotModule)
_POWER_DOWN_TYPES = {"dome", "tsla", "mslo", "atag", "stag"}
_NON_COMBAT = ("harv", "mcv")
# Don't march a 1-rifle scout; wait for a real army (Run7 collapse was
# combat-without-eco; this gate is the army half of that lesson).
MIN_ARMY_FOR_ASSAULT = 4
# Pile-up at the beacon (visor 851: 230 e1, dist 2.8, then timeout if a
# powr/tent sits in fog 15 cells south). Hunt only after this many combat
# units are actually there — home spawns must not start the sweep.
ARRIVED_CELLS = 8
# Ignore a stray scout in mid-map once the army is already at the enemy base.
STRAY_FROM_BEACON = 20
HUNT_PERIOD_TICKS = 1600  # ~20 macro decisions; infantry can walk a waypoint
# Water on Singles is y≳40. Stay on the enemy half (never last_push / home ore).
HUNT_Y_MAX = 36
HUNT_X_MIN = 40
HUNT_OFFSETS = (
    (0, 14),
    (-16, 6),
    (10, 6),
    (-12, 20),
    (8, 22),
    (-24, 12),
    (-20, 26),
    (6, 28),
)

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


def _xy(obj) -> tuple[int, int]:
    return int(obj.cell_x), int(obj.cell_y)


def _dist2(a, b) -> int:
    return (int(a[0]) - int(b[0])) ** 2 + (int(a[1]) - int(b[1])) ** 2


def _nearest_xy(targets, origin) -> tuple[int, int]:
    return _xy(min(targets, key=lambda t: _dist2(_xy(t), origin)))


def _n_combat_at(combat, cell, radius: int) -> int:
    r2 = int(radius) * int(radius)
    n = 0
    for u in combat:
        try:
            if _dist2(_xy(u), cell) <= r2:
                n += 1
        except (TypeError, ValueError):
            continue
    return n


def _hunt_cell(obs, beacon) -> tuple[int, int]:
    """Next sweep cell around the enemy half. Stateless: index from obs.tick."""
    info = getattr(obs, "map_info", None)
    w = int(getattr(info, "width", 128) or 128)
    h = int(getattr(info, "height", 64) or 64)
    bx, by = int(beacon[0]), int(beacon[1])
    tick = int(getattr(obs, "tick", 0) or 0)
    dx, dy = HUNT_OFFSETS[(tick // HUNT_PERIOD_TICKS) % len(HUNT_OFFSETS)]
    min_x = max(HUNT_X_MIN, bx - 45)
    max_x = max(min_x + 1, w - 2)
    max_y = min(h - 2, HUNT_Y_MAX)
    x = min(max(bx + dx, min_x), max_x)
    y = min(max(by + dy, 2), max_y)
    return int(x), int(y)


def _push_cell(obs, last_push):
    """Celda de asalto: edificio visible, unidad visible, hunt, beacon.

    last_push de la política es veneno en a_short: la cabeza de celda no está
    condicionada a la unidad y cae en Ch6 (densidad propia). Visor 2026-08-30:
    280 e1 attack-move al mineral de casa, beginner intacto en niebla.

    Iter ~854: clavar el dest en el beacon después de llegar deja el ejército
    idle sobre (95,11). Edificios resagados en niebla (enB 5–10) → timeout.
    Hunt solo si ya hay masa en el beacon; el acercamiento sigue siendo beacon.
    """
    beacon = resolve_beacon(obs)
    origin = beacon if beacon is not None else (0, 0)
    bldgs = list(getattr(obs, "visible_enemy_buildings", None) or [])
    if bldgs:
        return _nearest_xy(bldgs, origin)
    ene_u = list(getattr(obs, "visible_enemies", None) or [])
    combat = _combat_units(getattr(obs, "units", None) or [])
    n_at = _n_combat_at(combat, beacon, ARRIVED_CELLS) if beacon is not None else 0
    piled = n_at >= MIN_ARMY_FOR_ASSAULT
    if ene_u:
        e = _nearest_xy(ene_u, origin)
        if piled and beacon is not None and _dist2(e, beacon) > STRAY_FROM_BEACON ** 2:
            return _hunt_cell(obs, beacon)
        return e
    if beacon is not None:
        if piled:
            return _hunt_cell(obs, beacon)
        return int(beacon[0]), int(beacon[1])
    if last_push is not None:
        return int(last_push[0]), int(last_push[1])
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
    #    Destino = edificio visible / unidad / hunt-si-pile / beacon
    #    (no last_push). army_attack_move de grupo SOLO si hay ≥MIN_ARMY
    #    ociosos; si no, ATTACK_MOVE a ociosos (cap 16). Re-emitir el grupo
    #    cada bloque cancelaba el path (visor).
    combat = _combat_units(units)
    has_harv = _has_harvester(obs, eco, units, prod)
    dest = _push_cell(obs, last_push)
    assault = (has_proc and has_harv and dest is not None
               and len(combat) >= MIN_ARMY_FOR_ASSAULT)
    if dest is not None and has_proc:
        px, py = dest
        idles = [u for u in combat if bool(getattr(u, "is_idle", False))]
        if assault and len(idles) >= MIN_ARMY_FOR_ASSAULT:
            out.append(CommandModel(
                action=ActionType.ARMY_ATTACK_MOVE, target_x=px, target_y=py))
        else:
            n_push = 0
            for u in idles:
                out.append(CommandModel(
                    action=ActionType.ATTACK_MOVE,
                    actor_id=int(u.actor_id),
                    target_x=int(px),
                    target_y=int(py),
                ))
                n_push += 1
                if n_push >= 16:
                    break

    return out
