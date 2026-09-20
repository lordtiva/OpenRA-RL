"""Autonomía de soporte — APM mínimo (no estrategia).

Una decisión cada ~50 ticks no cubre repair/power/harvest-idle. Esta capa
inyecta esos tres comandos DESPUÉS de index_to_command_effective(); no
entran al buffer de PPO.

No BUILD/PLACE/TRAIN, no deploy, no war nudge, no fog scout, no remate.
Eso lo muestrea la red (máscara anti-rush en action_adapter). Helpers de
dest (war_nudge_cell, fog_scout_destinations, …) siguen en el módulo para
tests y para apply_dest_credit, pero support_commands no los emite.
"""

from openra_env.models import ActionType, CommandModel
from rl.action_adapter import PACK_ARMY, nearest_passable, remap_move_cell
from rl.obs_encoding import decode_spatial

# Tipos que el hard apaga cuando hay brownout (ai.yaml PowerDownBotModule)
_POWER_DOWN_TYPES = {"dome", "tsla", "mslo", "atag", "stag"}
_NON_COMBAT = ("harv", "mcv")
# War script (pack/hunt/rally/dest-credit). Off: the policy owns targeting.
# Eco/micro above stays. Flip True only for a controlled ablation.
SUPPORT_ASSAULT = False
# War/fog/remnant stay off: the net owns targeting and scouting.
SUPPORT_WAR_NUDGE = False
# Leftover sweep+commit. Off: Run 34 wr 33%→17% (agua/beacon + AM spam).
SUPPORT_REMNANT = False
# Late leftover hunt. Off: APM-only support; the net must close leftovers.
SUPPORT_LATE_REMNANT = False
REMNANT_MIN_TICK = 25000
REMNANT_ENEMY_WEALTH = 6000
REMNANT_SWEEP_N = 4
# Fog scout helpers remain; support_commands does not emit them.
SUPPORT_FOG_SCOUT = False
FOG_SCOUT_N_BASE = 2          # until we have a bigger home army
FOG_SCOUT_N_MORE = 3          # once home combat >= FOG_SCOUT_ARMY_FOR_MORE
FOG_SCOUT_ARMY_FOR_MORE = 20  # "2 hasta tener más army"
FOG_SCOUT_MIN_SEP = 16        # min chebyshev sep between scout dests
# Raid peel: per-unit AttackMove, not group army_attack_move (Run 29 yank).
# Se sube a 24 para que toda la guarnición ociosa en casa defienda ante raids
# del rival (que ataca con 8-12 unidades), sin cuentagotas de 6.
RAID_HOME_ORDERS = 24
# Production buildings beat a forward powr/scout when choosing the push dest.
_PROD_BUILDINGS = frozenset({
    "fact", "afac", "proc", "weap", "tent", "barr", "kenn",
    "hpad", "afld", "syrd",
})
# Don't march a 1-rifle scout; wait for a real army (Run7 collapse was
# combat-without-eco; this gate is the army half of that lesson).
# Visor 6am 947: 4 idle → army_attack_move + rally-to-beacon = oleada de 4
# que muere en x≈45 (nd≥4≈0, incomplete 81%). Pack at HOME, not total nc.
MIN_ARMY_FOR_ASSAULT = PACK_ARMY
# Already piled on the enemy half: 4 is enough to hunt leftover buildings.
MIN_PILE_FOR_HUNT = 4
# Rally staging: toward dest from the conyard, not the beacon, until the pack
# is ready. STAGING_STEPS cells (~10) keeps new e1 in the yard.
STAGING_STEPS = 10
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


def _is_harv_type(typ) -> bool:
    t = str(typ or "").lower()
    return "harv" in t and "husk" not in t


def _n_harvesters(units, prod) -> int:
    alive = sum(1 for u in units or [] if _is_harv_type(getattr(u, "type", "")))
    queued = sum(1 for p in prod or [] if _is_harv_type(getattr(p, "item", "")))
    return int(alive + queued)


def _has_harvester(obs, eco, units, prod) -> bool:
    if eco is not None and int(getattr(eco, "harvester_count", 0) or 0) > 0:
        return True
    return _n_harvesters(units, prod) > 0


def _xy(obj) -> tuple[int, int]:
    return int(obj.cell_x), int(obj.cell_y)


def _dist2(a, b) -> int:
    return (int(a[0]) - int(b[0])) ** 2 + (int(a[1]) - int(b[1])) ** 2


def _nearest_xy(targets, origin) -> tuple[int, int]:
    return _xy(min(targets, key=lambda t: _dist2(_xy(t), origin)))


def _farthest_xy(targets, origin) -> tuple[int, int]:
    return _xy(max(targets, key=lambda t: _dist2(_xy(t), origin)))


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
    """True if dest is the war_objective / hunt around it, not a visible actor."""
    if dest is None:
        return False
    from rl.war_objective import war_objective
    obj = war_objective(obs)
    if obj is None:
        return False
    d = (int(dest[0]), int(dest[1]))
    if d == (int(obj[0]), int(obj[1])):
        return True
    hx, hy = hunt_near_cell(obs, obj)
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


def war_nudge_cell(obs):
    """Dest del nudge: contacto visible. Nunca beacon.

    Returns (cell, is_raid) or (None, False).
    Raid: amenaza ≤DEFEND_CELLS, más cercana a la fact (defensores locales).
    Push: edificio de producción más lejano; si no, contacto más lejano.
    Un tent/powr de la puerta no gana contra un fact visible al fondo.
    """
    origin = _own_anchor(obs) or _map_center(obs)
    home = home_raid_targets(obs)
    if home:
        return _nearest_xy(home, origin), True
    bldgs = list(getattr(obs, "visible_enemy_buildings", None) or [])
    prod = [
        b for b in bldgs
        if str(getattr(b, "type", "") or "").lower() in _PROD_BUILDINGS
    ]
    if prod:
        return _farthest_xy(prod, origin), False
    contacts = bldgs + list(getattr(obs, "visible_enemies", None) or [])
    if contacts:
        return _farthest_xy(contacts, origin), False
    return None, False


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
_BARRACKS_ITEMS = ("tent", "barr")
TENT_COST = 500
HARV_COST = 1100
WEAP_COST = 2000
DOME_COST = 1800
# Assist weap/dome once the infantry opening is standing (doc 1.md).
SUPPORT_WEAP_CASH = 2500
_WEAP_ITEMS = ("weap",)
_DOME_ITEMS = ("dome",)
# Easy InitialHarvesters: 2. We only replaced at 0 → 1 truck vs their 2.
MIN_HARVESTERS = 2
# Apertura estándar: 2 refinerías (2 muelles de descarga + 2 cosechadoras).
MAX_SUPPORT_PROCS = 2
_ORE_IDLE_ORDERS = 2


def _map_center(obs) -> tuple[int, int]:
    info = getattr(obs, "map_info", None)
    w = max(int(getattr(info, "width", 128) or 128), 1)
    h = max(int(getattr(info, "height", 64) or 64), 1)
    return w // 2, h // 2


def _own_anchor(obs):
    """Construction yard / first civic building (rally origin)."""
    for b in getattr(obs, "buildings", None) or []:
        t = str(getattr(b, "type", "") or "").lower()
        if t in ("fact", "afac", "proc", "tent", "barr", "powr", "apwr"):
            try:
                return _xy(b)
            except (TypeError, ValueError):
                continue
    return None


def _staging_cell(obs, dest, aidx=None):
    """A cell ~STAGING_STEPS toward dest from the yard. Pack gathers here."""
    origin = _own_anchor(obs) or _map_center(obs)
    if dest is None:
        raw = (int(origin[0]) + STAGING_STEPS, int(origin[1]))
    else:
        dx = int(dest[0]) - int(origin[0])
        dy = int(dest[1]) - int(origin[1])
        n = max(abs(dx), abs(dy), 1)
        raw = (
            int(origin[0]) + int(round(STAGING_STEPS * dx / n)),
            int(origin[1]) + int(round(STAGING_STEPS * dy / n)),
        )
    snapped = _snap_passable(obs, raw, aidx)
    return snapped if snapped is not None else (int(raw[0]), int(raw[1]))


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
    snap = _nearest_land(obs, (x, y))
    return snap if snap is not None else (int(x), int(y))


def _nearest_land(obs, dest, max_r: int = 16):
    """Pasable más cercana (cualquier componente) o None sin terreno."""
    pas, _ = _terrain_grids(obs)
    if pas is None:
        return None
    try:
        x0, y0 = int(dest[0]), int(dest[1])
    except (TypeError, ValueError, IndexError):
        return None
    h, w = pas.shape
    if 0 <= x0 < w and 0 <= y0 < h and bool(pas[y0, x0]):
        return x0, y0
    for r in range(1, int(max_r) + 1):
        for dx in range(-r, r + 1):
            for nx, ny in ((x0 + dx, y0 - r), (x0 + dx, y0 + r)):
                if 0 <= nx < w and 0 <= ny < h and bool(pas[ny, nx]):
                    return nx, ny
        for dy in range(-r + 1, r):
            for nx, ny in ((x0 - r, y0 + dy), (x0 + r, y0 + dy)):
                if 0 <= nx < w and 0 <= ny < h and bool(pas[ny, nx]):
                    return nx, ny
    return None


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

    No-op si SUPPORT_ASSAULT=False: PPO ve el click de la red, no (95,11).

    PPO/SIL veían el sample de la cabeza de celda (Ch6 en casa) mientras el
    engine ganaba por el comando de soporte al beacon. El gradiente leía
    'clickeaste el mineral y ganaste'. Mutar el comando de política y devolver
    el flat del dest; el caller recalcula log π(a_ejecutada|s).
    TRAIN/BUILD/PLACE no se tocan.
    attack_move per-unit sobre harv/mcv tampoco: el C# de army_attack_move
    salta Harvester, pero AttackMove por actor_id no, y el crédito mandaba
    la recolectora al beacon (visor 921).
    """
    if not SUPPORT_ASSAULT:
        return int(cell_flat), None
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


# Offsets relative to last contact / army pile (map-agnostic; no Singles half).
HUNT_NEAR_OFFSETS = (
    (0, 12),
    (-14, 4),
    (12, 4),
    (-10, -10),
    (10, -8),
    (-18, 10),
    (16, 12),
    (0, -14),
)


def hunt_near_cell(obs, anchor, from_xy=None) -> tuple[int, int]:
    """Sweep cell around last_seen / pile. Map-agnostic (no BEACON_BY_MAP).

    Con from_xy, el barrido cae en tierra alcanzable desde el ejército.
    """
    info = getattr(obs, "map_info", None)
    w = int(getattr(info, "width", 128) or 128)
    h = int(getattr(info, "height", 64) or 64)
    ax, ay = int(anchor[0]), int(anchor[1])
    tick = int(getattr(obs, "tick", 0) or 0)
    dx, dy = HUNT_NEAR_OFFSETS[(tick // HUNT_PERIOD_TICKS) % len(HUNT_NEAR_OFFSETS)]
    x = min(max(ax + dx, 1), max(1, w - 2))
    y = min(max(ay + dy, 1), max(1, h - 2))
    if from_xy is not None:
        nr = nearest_reachable(obs, (x, y), from_xy)
        if nr is not None:
            return int(nr[0]), int(nr[1])
    return int(x), int(y)


def _push_cell(obs, last_push):
    """Celda de asalto: war_objective (raid/visible/mental/fog). Never GPS.

    last_push de la política es veneno: la cabeza de celda cae en Ch6
    (densidad propia). No se usa como dest.
    """
    from rl.war_objective import war_objective
    obj = war_objective(obs, last_contact=last_push)
    if obj is not None:
        return int(obj[0]), int(obj[1])
    return None


# Migajas junto a casa (no el argmax del mapa: idle en proc tiene Ch2=0).
_ORE_STALE_FRAC = 0.35
_ORE_HOME_RADIUS = 12


def _spatial_chw(obs):
    """Tensor espacial (C,H,W) o None. Ch2=ore, Ch3=passable, Ch4=fog."""
    info = getattr(obs, "map_info", None)
    h = int(getattr(info, "height", 0) or 0)
    w = int(getattr(info, "width", 0) or 0)
    raw = getattr(obs, "spatial_map", "") or ""
    ch = int(getattr(obs, "spatial_channels", 0) or 9)
    if not raw or h < 1 or w < 1:
        return None
    try:
        return decode_spatial(raw, h, w, ch, beacon=None)
    except (ValueError, TypeError):
        return None


def _proc_xy(blds):
    for b in blds or []:
        if str(getattr(b, "type", "") or "").lower() == "proc":
            try:
                return int(b.cell_x), int(b.cell_y)
            except (TypeError, ValueError):
                return None
    return None


# Terreno por episodio: pasabilidad (Ch3) + componentes conexas 8-dir.
# El hunt ciego ordenaba niebla al otro lado del lago (a_short oeste:
# 20 rifles idle 54k ticks hacia (109,32) sin contacto). Sin spatial se
# comporta como antes (sin filtro). Cache por hash del grid: el mapa no
# cambia dentro del episodio.
_TERRAIN = {"key": None, "pass": None, "comp": None}


def _terrain_grids(obs):
    """(pass_bool_HxW, comp_int32_HxW) cacheado, o (None, None)."""
    import hashlib
    import numpy as np
    arr = _spatial_chw(obs)
    if arr is None or int(arr.shape[0]) < 4:
        return None, None
    ch3 = arr[3]
    h, w = int(ch3.shape[0]), int(ch3.shape[1])
    if h < 1 or w < 1:
        return None, None
    pas = np.asarray(ch3 > 0.5, dtype=bool)
    key = (h, w, hashlib.md5(pas.tobytes()).hexdigest())
    c = _TERRAIN
    if c["key"] == key and c["pass"] is not None:
        return c["pass"], c["comp"]
    comp = np.full((h, w), -1, dtype=np.int32)
    ncomp = 0
    for y in range(h):
        for x in range(w):
            if not bool(pas[y, x]) or int(comp[y, x]) >= 0:
                continue
            stack = [(x, y)]
            comp[y, x] = ncomp
            while stack:
                cx, cy = stack.pop()
                for dy in (-1, 0, 1):
                    ny = cy + dy
                    if ny < 0 or ny >= h:
                        continue
                    row_p = pas[ny]
                    row_c = comp[ny]
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx = cx + dx
                        if (0 <= nx < w and bool(row_p[nx])
                                and int(row_c[nx]) < 0):
                            row_c[nx] = ncomp
                            stack.append((nx, ny))
            ncomp += 1
    c["key"] = key
    c["pass"] = pas
    c["comp"] = comp
    return pas, comp


def _comp_id(comp, xy):
    try:
        x, y = int(xy[0]), int(xy[1])
    except (TypeError, ValueError, IndexError):
        return None
    h, w = comp.shape
    if not (0 <= x < w and 0 <= y < h):
        return None
    v = int(comp[y, x])
    return v if v >= 0 else None


def land_connected(obs, a, b) -> bool:
    """Misma isla de tierra. Sin terreno → True (comportamiento previo)."""
    _, comp = _terrain_grids(obs)
    if comp is None:
        return True
    ca, cb = _comp_id(comp, a), _comp_id(comp, b)
    if ca is None or cb is None:
        return False
    return ca == cb


def nearest_reachable(obs, xy, from_xy, max_r: int = 40):
    """Pasable más cercana a xy en la componente de from_xy (o None)."""
    _, comp = _terrain_grids(obs)
    if comp is None:
        return None
    fc = _comp_id(comp, from_xy)
    if fc is None:
        return None
    try:
        x0, y0 = int(xy[0]), int(xy[1])
    except (TypeError, ValueError, IndexError):
        return None
    h, w = comp.shape
    if 0 <= x0 < w and 0 <= y0 < h and int(comp[y0, x0]) == fc:
        return x0, y0
    for r in range(1, int(max_r) + 1):
        best = None
        best_d = None
        # Anillo exterior del cuadrado (Chebyshev == r).
        ring = []
        for dx in range(-r, r + 1):
            ring.append((x0 + dx, y0 - r))
            ring.append((x0 + dx, y0 + r))
        for dy in range(-r + 1, r):
            ring.append((x0 - r, y0 + dy))
            ring.append((x0 + r, y0 + dy))
        for nx, ny in ring:
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if int(comp[ny, nx]) != fc:
                continue
            d = max(abs(nx - x0), abs(ny - y0))
            if best_d is None or d < best_d:
                best, best_d = (nx, ny), d
        if best is not None:
            return best[0], best[1]
    return None


def _keep_reachable(obs, dests, from_xy):
    """Filtra dests a la componente de from_xy (ancla si None).

    Cada dest al otro lado del agua se reemplaza por la pasable más
    cercana alcanzable; si no hay en radio, se dropea el sector.
    Sin terreno → lista intacta.
    """
    _, comp = _terrain_grids(obs)
    if comp is None:
        return list(dests)
    origin = from_xy if from_xy is not None else (
        _own_anchor(obs) or _map_center(obs))
    try:
        origin = (int(origin[0]), int(origin[1]))
    except (TypeError, ValueError, IndexError):
        return list(dests)
    if _comp_id(comp, origin) is None:
        return list(dests)
    out = []
    for d in dests:
        try:
            dd = (int(d[0]), int(d[1]))
        except (TypeError, ValueError, IndexError):
            continue
        if _comp_id(comp, dd) == _comp_id(comp, origin):
            out.append(dd)
            continue
        nr = nearest_reachable(obs, dd, origin)
        if nr is not None:
            out.append((int(nr[0]), int(nr[1])))
    return out


def _best_ore_near(arr, origin, radius: int = _ORE_HOME_RADIUS):
    """Celda de mineral más rica en radio de la proc, explorada y pasable."""
    import numpy as np
    if arr is None or arr.shape[0] < 5 or origin is None:
        return None
    ox, oy = int(origin[0]), int(origin[1])
    ch2, ch3, ch4 = arr[2], arr[3], arr[4]
    h, w = ch2.shape
    yy, xx = np.ogrid[:h, :w]
    near = (np.abs(xx - ox) + np.abs(yy - oy)) <= int(radius)
    mask = near & (ch4 >= 0.45) & (ch3 > 0.5)
    if not bool(mask.any()):
        return None
    vals = np.where(mask, ch2, -1.0)
    if float(vals.max()) <= 0.0:
        return None
    y, x = [int(i) for i in np.unravel_index(int(vals.argmax()), vals.shape)]
    return int(x), int(y), float(vals[y, x])


def _harv_local_ore(u, arr) -> float:
    if arr is None:
        return 0.0
    try:
        x, y = int(u.cell_x), int(u.cell_y)
    except (TypeError, ValueError):
        return 0.0
    _, h, w = arr.shape
    if not (0 <= y < h and 0 <= x < w):
        return 0.0
    return float(arr[2, y, x])



def fog_scout_count(n_home_combat: int) -> int:
    """2 exploradores hasta tener más army; 3 cuando casa ≥ umbral."""
    if int(n_home_combat) >= int(FOG_SCOUT_ARMY_FOR_MORE):
        return int(FOG_SCOUT_N_MORE)
    return int(FOG_SCOUT_N_BASE)


def _fog_scout_angle_dests(obs, n: int, aidx=None):
    """N rumbos equiespaciados desde el ancla (sin beacon). Fallback sin spatial."""
    import math
    origin = _own_anchor(obs) or _map_center(obs)
    info = getattr(obs, "map_info", None)
    w = int(getattr(info, "width", 128) or 128)
    h = int(getattr(info, "height", 64) or 64)
    r = max(24, min(w, h) // 2)
    # Small phase from map size so two maps of same size still share logic
    # but we never hardcode enemy spawn.
    phase = 0.37
    out = []
    for i in range(max(0, int(n))):
        ang = phase + (2.0 * math.pi * i) / max(1, int(n))
        raw = (int(origin[0] + r * math.cos(ang)),
               int(origin[1] + r * math.sin(ang)))
        dest = _snap_passable(obs, raw, aidx)
        if dest is None:
            dest = (max(0, min(w - 1, raw[0])), max(0, min(h - 1, raw[1])))
        out.append((int(dest[0]), int(dest[1])))
    return out


def fog_scout_destinations(obs, n: int, aidx=None, from_xy=None):
    """Hasta n celdas distintas en niebla pasable, una por sector angular.

    Usa Ch3 (passable) + Ch4 (fog). Sin spatial → rumbos equiespaciados.
    Nunca resolve_beacon / BEACON_BY_MAP. Con from_xy (o el ancla), los
    sectores al otro lado del agua se reemplazan por tierra alcanzable.
    """
    import math
    import numpy as np
    n = max(0, int(n))
    if n <= 0:
        return []
    origin = _own_anchor(obs) or _map_center(obs)
    ox, oy = int(origin[0]), int(origin[1])
    arr = _spatial_chw(obs)
    if arr is None or arr.shape[0] < 5:
        return _keep_reachable(obs, _fog_scout_angle_dests(obs, n, aidx),
                               from_xy)
    ch3, ch4 = arr[3], arr[4]
    h, w = int(ch3.shape[0]), int(ch3.shape[1])
    # Unexplored / heavy shroud, walkable, not on top of the yard.
    ys, xs = np.where((ch3 > 0.5) & (ch4 < 0.45))
    if len(xs) == 0:
        return _keep_reachable(obs, _fog_scout_angle_dests(obs, n, aidx),
                               from_xy)
    # Bin by angle from origin; pick farthest cell in each of n sectors.
    sectors = [[] for _ in range(n)]
    for x, y in zip(xs.tolist(), ys.tolist()):
        dx, dy = int(x) - ox, int(y) - oy
        if dx * dx + dy * dy < 10 * 10:
            continue
        ang = math.atan2(dy, dx)
        if ang < 0:
            ang += 2.0 * math.pi
        si = int((ang / (2.0 * math.pi)) * n) % n
        sectors[si].append((int(x), int(y), dx * dx + dy * dy))
    dests = []
    for bucket in sectors:
        if not bucket:
            continue
        bucket.sort(key=lambda t: t[2], reverse=True)
        x, y, _ = bucket[0]
        # Enforce separation from already chosen dests.
        if any(max(abs(x - dx), abs(y - dy)) < FOG_SCOUT_MIN_SEP
               for dx, dy in dests):
            picked = None
            for x2, y2, _ in bucket[1:]:
                if all(max(abs(x2 - dx), abs(y2 - dy)) >= FOG_SCOUT_MIN_SEP
                       for dx, dy in dests):
                    picked = (x2, y2)
                    break
            if picked is None:
                continue
            x, y = picked
        snap = _snap_passable(obs, (x, y), aidx)
        dests.append((int(snap[0]), int(snap[1])) if snap else (x, y))
        if len(dests) >= n:
            break
    if len(dests) < n:
        # Fill missing sectors with angle fallback, skipping near-dupes.
        for raw in _fog_scout_angle_dests(obs, n, aidx):
            if len(dests) >= n:
                break
            if any(max(abs(raw[0] - dx), abs(raw[1] - dy)) < FOG_SCOUT_MIN_SEP
                   for dx, dy in dests):
                continue
            dests.append(raw)
    return _keep_reachable(obs, dests[:n], from_xy)


def _spectator_enemy_wealth(obs):
    """Spectator wealth if attached (rollout gs); None under fog-only obs."""
    gs = getattr(obs, "global_summary", None)
    if not isinstance(gs, dict):
        md = getattr(obs, "metadata", None)
        if isinstance(md, dict):
            gs = md.get("global_summary")
    if not isinstance(gs, dict):
        return None
    ene = gs.get("enemy") or {}
    try:
        return (int(ene.get("cash") or 0)
                + int(ene.get("unit_value") or 0)
                + int(ene.get("building_value") or 0))
    except (TypeError, ValueError):
        return None


def remnant_hunt_needed(obs) -> bool:
    """True when the enemy is crushed / fog-empty after the rush window.

    Visible leftovers stay a fight (nudge / policy). Late fog-empty games
    (own 34k vs enemy 4.5k, timeout 53k) need an edge sweep.
    """
    if not SUPPORT_LATE_REMNANT:
        return False
    try:
        tick = int(getattr(obs, "tick", 0) or 0)
    except (TypeError, ValueError):
        tick = 0
    if tick < REMNANT_MIN_TICK:
        return False
    if home_raid_targets(obs):
        return False
    vis_b = list(getattr(obs, "visible_enemy_buildings", None) or [])
    vis_u = list(getattr(obs, "visible_enemies", None) or [])
    if vis_b or vis_u:
        return False
    # Fog-empty after 25k (and optional spectator wealth<6k crushed leftover).
    return True


def _unexplored_edge_dests(obs, n: int, aidx=None):
    """Fog / map-edge cells, one per quadrant. Never beacon GPS."""
    n = max(0, int(n))
    if n <= 0:
        return []
    dests = fog_scout_destinations(obs, n, aidx)
    if len(dests) >= n:
        return dests[:n]
    info = getattr(obs, "map_info", None)
    w = int(getattr(info, "width", 128) or 128)
    h = int(getattr(info, "height", 64) or 64)
    corners = ((w - 3, 2), (w - 3, h - 3), (2, h - 3), (2, 2))
    for raw in corners:
        if len(dests) >= n:
            break
        snap = _snap_passable(obs, raw, aidx) or raw
        xy = (int(snap[0]), int(snap[1]))
        if any(max(abs(xy[0] - dx), abs(xy[1] - dy)) < FOG_SCOUT_MIN_SEP
               for dx, dy in dests):
            continue
        dests.append(xy)
    return dests[:n]


def _emit_late_remnant(obs, combat, aidx, out):
    """Fan-out idle / circling field units to unexplored edges."""
    if not remnant_hunt_needed(obs) or not combat:
        return
    idle = [u for u in combat if bool(getattr(u, "is_idle", False))]
    if idle:
        movers = idle
    else:
        away = []
        for u in combat:
            try:
                if not _near_own_base(obs, _xy(u)):
                    away.append(u)
            except (TypeError, ValueError):
                continue
        if len(away) < MIN_PILE_FOR_HUNT:
            return
        movers = away
    n = min(len(movers), REMNANT_SWEEP_N)
    dests = _unexplored_edge_dests(obs, n, aidx)
    if not dests:
        return
    n_go = min(n, len(dests))
    for u, dest in zip(movers[:n_go], dests):
        try:
            aid = int(getattr(u, "actor_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if aid <= 0:
            continue
        out.append(CommandModel(
            action=ActionType.ATTACK_MOVE,
            actor_id=aid,
            target_x=int(dest[0]),
            target_y=int(dest[1]),
        ))


def _scout_preference_key(u):
    """Prefer fast rifles; never medi/harv (harv already excluded from combat)."""
    ut = str(getattr(u, "type", "") or "").lower()
    if "medi" in ut:
        return (2, int(getattr(u, "actor_id", 0) or 0))
    if ut in ("e1", "e2", "dog", "e3"):
        return (0, int(getattr(u, "actor_id", 0) or 0))
    return (1, int(getattr(u, "actor_id", 0) or 0))


def _emit_fog_scouts(obs, combat, idles_home, aidx, out):
    """Append AttackMove for up to N fog scouts. No-op if contacts/raid."""
    if not SUPPORT_FOG_SCOUT:
        return
    n_want = fog_scout_count(len(idles_home))
    if n_want <= 0 or not idles_home:
        return
    # Units already away from base count as active scouts (sticky).
    active = []
    for u in combat:
        try:
            if not _near_own_base(obs, _xy(u)):
                active.append(u)
        except (TypeError, ValueError):
            continue
    slots = max(0, n_want - len(active))
    if slots <= 0:
        return
    dests = fog_scout_destinations(obs, slots, aidx)
    if not dests:
        return
    cand = sorted(
        [u for u in idles_home
         if "medi" not in str(getattr(u, "type", "") or "").lower()],
        key=_scout_preference_key,
    )
    for u, dest in zip(cand, dests):
        try:
            aid = int(getattr(u, "actor_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if aid <= 0:
            continue
        out.append(CommandModel(
            action=ActionType.ATTACK_MOVE,
            actor_id=aid,
            target_x=int(dest[0]),
            target_y=int(dest[1]),
        ))


def support_commands(obs, last_push=None, max_repairs: int = 2, aidx=None, war_nudge=None):

    """APM only: harvest-idle (no cell), repair, power_down.

    last_push / aidx / war_nudge are ignored (kept so call sites stay stable).
    """
    _ = (last_push, aidx, war_nudge)
    out = []
    eco = getattr(obs, "economy", None)
    blds = getattr(obs, "buildings", []) or []
    units = getattr(obs, "units", []) or []
    cash = int(getattr(eco, "cash", 0) or 0) if eco else 0

    # Idle harvs → HARVEST without a cell (engine picks nearest ore).
    n_h = 0
    for u in units:
        if n_h >= _ORE_IDLE_ORDERS:
            break
        if not _is_harv_type(getattr(u, "type", "")):
            continue
        if not bool(getattr(u, "is_idle", False)):
            continue
        out.append(CommandModel(
            action=ActionType.HARVEST, actor_id=int(u.actor_id)))
        n_h += 1

    # Repair: cash>500 and hp<35%, max 2 per block (hard-bot threshold).
    if cash > 500:
        repairs = 0
        for b in blds:
            if repairs >= max_repairs:
                break
            hp = float(getattr(b, "hp_percent", 1.0) or 1.0)
            if hp < 0.35 and not bool(getattr(b, "is_repairing", False)):
                out.append(CommandModel(
                    action=ActionType.REPAIR, actor_id=int(b.actor_id)))
                repairs += 1

    # Power down one low-priority building on brownout.
    if eco is not None:
        provided = int(getattr(eco, "power_provided", 0) or 0)
        drained = int(getattr(eco, "power_drained", 0) or 0)
        if drained > provided:
            for b in blds:
                if b.type in _POWER_DOWN_TYPES and bool(getattr(b, "is_powered", True)):
                    out.append(CommandModel(
                        action=ActionType.POWER_DOWN, actor_id=int(b.actor_id)))
                    break

    return out
