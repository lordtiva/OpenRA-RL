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
- Defend recall (Run 11): raid a ≤DEFEND_CELLS de un edificio propio cancela
  el path de asalto (army_attack_move de grupo aunque el blob ya camine).
  Sin eso el easy razeaba la casa vacía (defense_loss≈−4, 0/140).
- Re-asalto (Run 12): si el dest volvió a beacon/hunt y el blob sigue en
  casa caminando (post-recall), army_attack_move de grupo. Sin eso el blob
  terminaba el viaje a casa y el episodio iba a timeout (incomplete ~70%).
  NO re-ordenar mid-map ni un asalto a enemigo visible (visor 817).
- Crédito de dest: apply_dest_credit reescribe cell_flat de army/attack_move
  al dest de soporte ANTES del buffer PPO/SIL (el win no acredita click en casa).
  El dest se remapea a celda pasable; hunt en agua era sil_nll ~1e9/128 (Run 13).
- Rally al dest en tent/barr/kenn (Capa 0): los e1 salen caminando, no idle en el tent.
  weap/hpad/syrd NO: HARV sale de Vehicle queue y el visor 921 la veía
  caminar al beacon. Tanques idle los recoge army_attack_move.
- Stance AttackAnything al nacer (Capa 0): Defend no caza; el scripted sí.
  Solo combate (no harv/mcv).

No genera reward — evita defense_loss/hold_zero ya existentes.
"""

from openra_env.models import ActionType, CommandModel
from rl.action_adapter import nearest_passable, remap_move_cell
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
# NOT if that unit is next to our buildings: easy raids home, beginner didn't.
STRAY_FROM_BEACON = 20
DEFEND_CELLS = 18
HUNT_PERIOD_TICKS = 1600  # ~20 macro decisions; infantry can walk a waypoint
# Water on Singles is y≳40. Stay on the enemy half (never last_push / home ore).
HUNT_Y_MAX = 32  # water on Singles is y≳40; 36 still pisa mask (SIL -1e9)
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


def _near_own_base(obs, xy, radius: int = DEFEND_CELLS) -> bool:
    r2 = int(radius) * int(radius)
    for b in getattr(obs, "buildings", None) or []:
        try:
            if _dist2(_xy(b), xy) <= r2:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _n_combat_near_own_base(obs, combat, radius: int = DEFEND_CELLS) -> int:
    n = 0
    for u in combat or []:
        try:
            if _near_own_base(obs, _xy(u), radius):
                n += 1
        except (TypeError, ValueError):
            continue
    return n


def _is_beacon_or_hunt(obs, dest) -> bool:
    """True si dest es el waypoint de asalto (beacon/hunt), no un actor visible."""
    if dest is None:
        return False
    beacon = resolve_beacon(obs)
    if beacon is None:
        return False
    d = (int(dest[0]), int(dest[1]))
    if d == (int(beacon[0]), int(beacon[1])):
        return True
    hx, hy = _hunt_cell(obs, beacon)
    return d == (int(hx), int(hy))


def home_raid_targets(obs):
    """Enemigos (unidad o edificio) a ≤DEFEND_CELLS de un edificio propio."""
    bldgs = list(getattr(obs, "visible_enemy_buildings", None) or [])
    ene_u = list(getattr(obs, "visible_enemies", None) or [])
    out = []
    for t in bldgs + ene_u:
        try:
            if _near_own_base(obs, _xy(t)):
                out.append(t)
        except (TypeError, ValueError):
            continue
    return out


def _cmd_name(cmd) -> str:
    act = getattr(cmd, "action", None)
    if act is None:
        return ""
    return str(getattr(act, "value", None) or act)


COMBAT_PUSH_TYPES = frozenset({"army_attack_move", "attack_move"})
STANCE_ATTACK_ANYTHING = 3
# Infantry only. weap/hpad/syrd/afld produce HARV (or mix combat+eco);
# rally-to-dest marched ore trucks to the enemy beacon (visor 921).
_RALLY_BUILDINGS = frozenset({
    "tent", "barr", "kenn",
})
_NO_SELL = frozenset({"fact", "afac", "proc"})
SELL_HP = 0.12


def _snap_passable(obs, dest, aidx=None):
    """Dest de soporte/hunt en tierra. Hunt y=36 era agua → sil_nll 7e6."""
    if dest is None:
        return None
    x, y = int(dest[0]), int(dest[1])
    if aidx is not None:
        return remap_move_cell(obs, aidx, x, y)
    info = getattr(obs, "map_info", None)
    w = int(getattr(info, "width", 128) or 128)
    h = int(getattr(info, "height", 64) or 64)
    x = min(max(x, 0), max(0, w - 1))
    y = min(max(y, 0), min(max(0, h - 1), HUNT_Y_MAX))
    return int(x), int(y)


def _cell_mask_ok(aidx, cell_flat) -> bool:
    mask = getattr(aidx, "cell_mask", None)
    if mask is None:
        return True
    try:
        import torch
        m = mask.reshape(-1)
        i = int(cell_flat)
        if i < 0 or i >= int(m.numel() if torch.is_tensor(m) else len(m)):
            return False
        v = m[i]
        return bool(v.item() if torch.is_tensor(v) else v)
    except (TypeError, ValueError, IndexError, RuntimeError):
        return True


def _is_noncombat_actor(obs, actor_id) -> bool:
    """True if actor_id is a harvester or MCV (not a combat unit)."""
    try:
        aid = int(actor_id or 0)
    except (TypeError, ValueError):
        return False
    if aid <= 0:
        return False
    for u in getattr(obs, "units", None) or []:
        try:
            if int(getattr(u, "actor_id", 0) or 0) == aid:
                return not _is_combat(u)
        except (TypeError, ValueError):
            continue
    return False


def apply_dest_credit(obs, action, type_name, cell_flat, aidx, last_push=None):
    """cell_flat de army/attack_move = dest de soporte (el que mueve el ejército).

    PPO/SIL veían el sample de la cabeza de celda (Ch6 en casa) mientras el
    engine ganaba por el comando de soporte al beacon. El gradiente leía
    'clickeaste el mineral y ganaste'. Mutar el comando de política y devolver
    el flat del dest; el caller recalcula log π(a_ejecutada|s).
    TRAIN/BUILD/PLACE no se tocan.
    attack_move per-unit sobre harv/mcv tampoco: el C# de army_attack_move
    salta Harvester, pero AttackMove por actor_id no, y el crédito mandaba
    la recolectora al beacon (visor 921).
    """
    if type_name not in COMBAT_PUSH_TYPES:
        return int(cell_flat), None
    if type_name == "attack_move":
        actor_id = 0
        for c in getattr(action, "commands", None) or []:
            if _cmd_name(c) == "attack_move":
                actor_id = int(getattr(c, "actor_id", 0) or 0)
                break
        if _is_noncombat_actor(obs, actor_id):
            return int(cell_flat), None
    dest = _push_cell(obs, last_push)
    dest = _snap_passable(obs, dest, aidx)
    if dest is None:
        return int(cell_flat), None
    w = int(getattr(aidx, "w", 0) or 0)
    h = int(getattr(aidx, "h", 0) or 0)
    x, y = int(dest[0]), int(dest[1])
    if w > 0:
        x = min(max(x, 0), w - 1)
    if h > 0:
        y = min(max(y, 0), h - 1)
    new_flat = (int(y) * w + int(x)) if w > 0 else int(cell_flat)
    if not _cell_mask_ok(aidx, new_flat):
        return int(cell_flat), None
    for c in getattr(action, "commands", None) or []:
        if _cmd_name(c) not in COMBAT_PUSH_TYPES:
            continue
        if (_cmd_name(c) == "attack_move"
                and _is_noncombat_actor(obs, getattr(c, "actor_id", 0))):
            continue
        c.target_x = x
        c.target_y = y
    return int(new_flat), (x, y)


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

    Easy (Capa 3): un enemigo junto a nuestros edificios es raid, no scout.
    Defender eso gana a hunt / beacon. El stray-ignore del beginner no aplica.
    """
    beacon = resolve_beacon(obs)
    origin = beacon if beacon is not None else (0, 0)
    bldgs = list(getattr(obs, "visible_enemy_buildings", None) or [])
    ene_u = list(getattr(obs, "visible_enemies", None) or [])
    home = home_raid_targets(obs)
    if home:
        fact = None
        for b in getattr(obs, "buildings", None) or []:
            if str(getattr(b, "type", "")).lower() in ("fact", "afac"):
                fact = _xy(b)
                break
        return _nearest_xy(home, fact or origin)
    if bldgs:
        return _nearest_xy(bldgs, origin)
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


def support_commands(obs, last_push=None, max_repairs: int = 2, aidx=None):

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

    # 1b) Sell wrecks (not fact/proc). APM tonto; no cabe en la política.
    for b in blds:
        t = str(getattr(b, "type", "")).lower()
        if t in _NO_SELL:
            continue
        hp = float(getattr(b, "hp_percent", 1.0) or 1.0)
        if hp < SELL_HP:
            out.append(CommandModel(action=ActionType.SELL, actor_id=int(b.actor_id)))
            break

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
    #    Destino = raid-en-casa / edificio visible / unidad / hunt / beacon
    #    (no last_push). army_attack_move de grupo SOLO si hay ≥MIN_ARMY
    #    ociosos; si no, ATTACK_MOVE a ociosos (cap 16). Re-emitir el grupo
    #    cada bloque cancelaba el path (visor) — EXCEPTO:
    #    - defend recall: raid en casa y el blob no está ahí (Run 11).
    #    - re-asalto: dest volvió a beacon/hunt y el blob sigue en casa
    #      caminando (Run 12: post-recall el viaje a casa no se cancelaba).
    combat = _combat_units(units)
    has_harv = _has_harvester(obs, eco, units, prod)
    raw_dest = _push_cell(obs, last_push)
    waypoint = _is_beacon_or_hunt(obs, raw_dest)
    dest = _snap_passable(obs, raw_dest, aidx)
    defending = bool(home_raid_targets(obs))
    assault = (has_proc and has_harv and dest is not None
               and len(combat) >= MIN_ARMY_FOR_ASSAULT)
    if dest is not None and (has_proc or defending):
        px, py = dest
        idles = [u for u in combat if bool(getattr(u, "is_idle", False))]
        n_at_dest = _n_combat_at(combat, (px, py), ARRIVED_CELLS)
        n_home = _n_combat_near_own_base(obs, combat)
        recall = bool(defending and combat and n_at_dest < MIN_ARMY_FOR_ASSAULT)
        reassault = bool(
            assault
            and not defending
            and n_at_dest < MIN_ARMY_FOR_ASSAULT
            and n_home >= MIN_ARMY_FOR_ASSAULT
            and waypoint
        )
        if recall or reassault or (assault and len(idles) >= MIN_ARMY_FOR_ASSAULT):
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

    # 4) Rally al dest — e1 spawnea y camina (Capa 0, no la política).
    #    Solo tents/barr/kenn. weap produce HARV: no marchar ore al dest.
    if dest is not None and has_proc:
        px, py = dest
        for b in blds:
            if str(getattr(b, "type", "")).lower() not in _RALLY_BUILDINGS:
                continue
            rx = int(getattr(b, "rally_x", -1) if getattr(b, "rally_x", -1) is not None else -1)
            ry = int(getattr(b, "rally_y", -1) if getattr(b, "rally_y", -1) is not None else -1)
            if rx == int(px) and ry == int(py):
                continue
            out.append(CommandModel(
                action=ActionType.SET_RALLY_POINT,
                actor_id=int(b.actor_id),
                target_x=int(px),
                target_y=int(py),
            ))
            break

    # 5) Stance AttackAnything al nacer (Defend no caza).
    for u in combat:
        st = int(getattr(u, "stance", STANCE_ATTACK_ANYTHING) or 0)
        if st != STANCE_ATTACK_ANYTHING:
            out.append(CommandModel(
                action=ActionType.SET_STANCE,
                actor_id=int(u.actor_id),
                target_x=STANCE_ATTACK_ANYTHING,
            ))
            break

    return out
