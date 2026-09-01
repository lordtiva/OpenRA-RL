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
- Cosecha: idle/migajas → Harvest CON celda al parche de CASA con menos
  camiones (radio 26 de la proc, no el argmax del mapa). Untargeted iba
  siempre al mismo yacimiento; el easy reparte. Sin Ch2 global (Run 24).
- Energía: si power_drained > power_provided, apaga dome/tsla/mslo (prioridad baja).
- Asalto FULL / hunt / recall / rally-al-beacon / crédito de dest: APAGADOS
  (corte 950). Era estrategia spawn-asimétrica (siempre (95,11)).
  Re-activar: SUPPORT_ASSAULT=True (no este corte).
- Nudge de guerra (SUPPORT_WAR_NUDGE): raid → attack_move solo idle en
  casa (no army_attack_move: visor 1099 ping-pong x=8↔70). Push: ≥12 idle
  en casa + contacto visible → army_attack_move al más lejano / prod, no
  al tent de la puerta. Sin beacon, sin crédito de dest.
- Stance AttackAnything al nacer (Capa 0): Defend no caza; el scripted sí.
  Solo combate (no harv/mcv). Micro, no “andá al NE”.
- Auto-tent (Capa 0, corte 987): con proc en pie y sin tent/barr, BUILD/PLACE
  el cuartel (como auto-proc). Sin eso latest 985 TRAIN-eaba sbag y moría
  a los 10k sin un rifle.

No genera reward — evita defense_loss/hold_zero ya existentes.
"""

from openra_env.models import ActionType, CommandModel
from rl.action_adapter import nearest_passable, remap_move_cell
from rl.obs_encoding import decode_spatial, resolve_beacon

# Tipos que el hard apaga cuando hay brownout (ai.yaml PowerDownBotModule)
_POWER_DOWN_TYPES = {"dome", "tsla", "mslo", "atag", "stag"}
_NON_COMBAT = ("harv", "mcv")
# War script (pack/hunt/rally/dest-credit). Off: the policy owns targeting.
# Eco/micro above stays. Flip True only for a controlled ablation.
SUPPORT_ASSAULT = False
# Cheap war nudge vs easy. Visible contact only — never beacon.
# Independent of SUPPORT_ASSAULT (that flag stays False).
SUPPORT_WAR_NUDGE = True
# Raid peel: per-unit AttackMove, not group army_attack_move (Run 29 yank).
RAID_HOME_ORDERS = 6
# Production buildings beat a forward powr/scout when choosing the push dest.
_PROD_BUILDINGS = frozenset({
    "fact", "afac", "proc", "weap", "tent", "barr", "kenn",
    "hpad", "afld", "syrd",
})
# Don't march a 1-rifle scout; wait for a real army (Run7 collapse was
# combat-without-eco; this gate is the army half of that lesson).
# Visor 6am 947: 4 idle → army_attack_move + rally-to-beacon = oleada de 4
# que muere en x≈45 (nd≥4≈0, incomplete 81%). Pack at HOME, not total nc.
MIN_ARMY_FOR_ASSAULT = 12
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


def war_nudge_cell(obs):
    """Dest del nudge: contacto visible. Nunca beacon.

    Returns (cell, is_raid) or (None, False).
    Raid: amenaza ≤DEFEND_CELLS, más cercana a la fact (defensores locales).
    Push: edificio de producción más lejano; si no, contacto más lejano.
    Un tent/powr de la puerta no gana contra un fact visible al fondo.
    """
    origin = _own_anchor(obs) or (12, 16)
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
    origin = _own_anchor(obs) or (12, 16)
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
    piled = n_at >= MIN_PILE_FOR_HUNT
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


# Casa, no el mapa: idle untargeted = todas al mismo pozo (visor 2+ en
# [15,17]; easy cosecha 2.7×). Run 24 argmax global = yank al enemigo.
_ORE_STALE_FRAC = 0.35
_ORE_HOME_RADIUS = 12
_ORE_ASSIGN_RADIUS = 26
_ORE_PATCH_SEP = 8
_ORE_MIN_DENSITY = 0.25
_ORE_HARVEST_ORDERS = 2


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


def _ore_patches(arr, origin, radius: int = _ORE_ASSIGN_RADIUS,
                 sep: int = _ORE_PATCH_SEP):
    """Yacimientos explorados cerca de la proc. Nunca el argmax global."""
    import numpy as np
    if arr is None or origin is None or arr.shape[0] < 5:
        return []
    ox, oy = int(origin[0]), int(origin[1])
    ch2, ch3, ch4 = arr[2], arr[3], arr[4]
    h, w = ch2.shape
    yy, xx = np.ogrid[:h, :w]
    near = (np.abs(xx - ox) + np.abs(yy - oy)) <= int(radius)
    mask = near & (ch4 >= 0.45) & (ch3 > 0.5) & (ch2 >= _ORE_MIN_DENSITY)
    if arr.shape[0] >= 9:
        mask = mask & ((arr[7] + arr[8]) < 0.15)
    if not bool(mask.any()):
        return []
    vals = np.where(mask, ch2, 0.0)
    taken = np.zeros((h, w), dtype=bool)
    patches = []
    rsep = int(sep)
    for _ in range(6):
        work = np.where(taken, -1.0, vals)
        peak = float(work.max())
        if peak < _ORE_MIN_DENSITY:
            break
        y, x = [int(i) for i in np.unravel_index(int(work.argmax()), work.shape)]
        patches.append((int(x), int(y), peak))
        taken |= (np.abs(xx - x) + np.abs(yy - y)) <= rsep
    return patches


def _nearest_patch_idx(xy, patches, sep: int = _ORE_PATCH_SEP):
    if not patches or xy is None:
        return None
    best_i, best_d = None, None
    try:
        hx, hy = int(xy[0]), int(xy[1])
    except (TypeError, ValueError, IndexError):
        return None
    for i, (px, py, _den) in enumerate(patches):
        d = abs(hx - px) + abs(hy - py)
        if best_i is None or d < best_d:
            best_i, best_d = i, d
    if best_i is not None and best_d <= int(sep):
        return best_i
    return None


def _pick_understaffed_patch(xy, patches, counts):
    """Parche con menos harvs; empate = más denso, luego más cerca."""
    try:
        hx, hy = int(xy[0]), int(xy[1])
    except (TypeError, ValueError, IndexError):
        hx, hy = 0, 0
    best_i = min(
        range(len(patches)),
        key=lambda i: (
            int(counts[i]),
            -float(patches[i][2]),
            abs(int(patches[i][0]) - hx) + abs(int(patches[i][1]) - hy),
        ),
    )
    return patches[best_i], best_i


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

    # 0) Auto-harvest. Idle/migajas → celda del parche de casa con menos
    #    harvs. Untargeted (engine nearest) amontonaba 2+ en un pozo.
    #    Sin parches explorados: idle sin celda (no yank al mapa).
    has_proc = any(getattr(b, "type", "") == "proc" for b in blds)
    if has_proc:
        arr = _spatial_chw(obs)
        origin = _proc_xy(blds)
        patches = _ore_patches(arr, origin)
        harvs = [
            u for u in units
            if "harv" in str(getattr(u, "type", "")).lower()
        ]
        counts = [0] * len(patches)
        for hv in harvs:
            try:
                hxy = _xy(hv)
            except (TypeError, ValueError):
                continue
            pi = _nearest_patch_idx(hxy, patches)
            if pi is not None:
                counts[pi] += 1
        n_h = 0
        for u in harvs:
            if n_h >= _ORE_HARVEST_ORDERS:
                break
            idle = bool(getattr(u, "is_idle", False))
            local = _harv_local_ore(u, arr)
            try:
                uxy = _xy(u)
            except (TypeError, ValueError):
                continue
            idx = _nearest_patch_idx(uxy, patches)
            stale = False
            if (not idle) and patches and local > 0.0:
                best_den = max(p[2] for p in patches)
                stale = best_den > 0.0 and local < _ORE_STALE_FRAC * best_den
            if not idle and not stale:
                continue
            if not idle and not patches:
                home = _best_ore_near(arr, origin)
                if home is None:
                    continue
                bx, by, bden = home
                if not (bden > 0.0 and local < _ORE_STALE_FRAC * bden):
                    continue
                out.append(CommandModel(
                    action=ActionType.HARVEST, actor_id=int(u.actor_id),
                    target_x=int(bx), target_y=int(by)))
                n_h += 1
                continue
            if not patches:
                if idle:
                    out.append(CommandModel(
                        action=ActionType.HARVEST, actor_id=int(u.actor_id)))
                    n_h += 1
                continue
            stay = (
                idx is not None
                and counts[idx] < min(counts) + 2
                and not stale
            )
            if stay:
                px, py, _den = patches[idx]
                new_i = idx
            else:
                _patch, new_i = _pick_understaffed_patch(uxy, patches, counts)
                px, py, _den = _patch
                if idx is not None:
                    counts[idx] = max(0, counts[idx] - 1)
                counts[new_i] += 1
            out.append(CommandModel(
                action=ActionType.HARVEST, actor_id=int(u.actor_id),
                target_x=int(px), target_y=int(py)))
            n_h += 1

    # 0b) Auto-proc + auto-harv + auto-tent — push eco then the first barracks.
    #     Missing proc: BUILD if it is in available_production (do not deadlock
    #     BUILD/PLACE of proc — those stay unmasked even when we cannot build yet).
    #     Proc ready in the queue: PLACE near the conyard. Has proc but no harvester:
    #     TRAIN harv if the war factory lists it. Has proc, no tent/barr: BUILD/PLACE
    #     the faction barracks (visor 985: sin cuartel, TRAIN sbag, lose 10k).
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
        # Auto-tent: primer cuartel, después de proc. tent (allies) o barr.
        has_barracks = any(
            str(getattr(b, "type", "")).lower() in _BARRACKS_ITEMS for b in blds)
        barr_ready = next(
            (str(getattr(p, "item", "")).lower()
             for p in prod
             if str(getattr(p, "item", "")).lower() in _BARRACKS_ITEMS
             and float(getattr(p, "progress", 0) or 0) >= 1.0),
            None,
        )
        barr_queued = any(
            str(getattr(p, "item", "")).lower() in _BARRACKS_ITEMS for p in prod)
        barr_item = next((n for n in _BARRACKS_ITEMS if n in avail), None)
        if not has_barracks:
            if barr_ready:
                ax, ay = _place_near_base(obs)
                out.append(CommandModel(
                    action=ActionType.PLACE_BUILDING, item_type=barr_ready,
                    target_x=ax, target_y=ay))
            elif (not barr_queued) and barr_item and cash >= TENT_COST:
                out.append(CommandModel(
                    action=ActionType.BUILD, item_type=barr_item))

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

    combat = _combat_units(units)

    # 3) War nudge — visible contact only. No beacon, no dest-credit.
    #    Raid: AttackMove idle-at-home only (group army_attack_move yanks
    #    the field army back to the door — visor 1099 ping-pong).
    #    Push: ≥12 idle at home, army_attack_move to farthest/prod contact.
    if SUPPORT_WAR_NUDGE and not SUPPORT_ASSAULT and combat:
        raw_dest, is_raid = war_nudge_cell(obs)
        idles_home = [
            u for u in combat
            if bool(getattr(u, "is_idle", False)) and _near_own_base(obs, _xy(u))
        ]
        if raw_dest is not None:
            dest = _snap_passable(obs, raw_dest, aidx)
            if dest is None:
                dest = raw_dest
            if is_raid:
                n_peel = 0
                for u in idles_home:
                    if n_peel >= RAID_HOME_ORDERS:
                        break
                    try:
                        aid = int(getattr(u, "actor_id", 0) or 0)
                    except (TypeError, ValueError):
                        continue
                    if aid <= 0:
                        continue
                    out.append(CommandModel(
                        action=ActionType.ATTACK_MOVE,
                        actor_id=aid,
                        target_x=int(dest[0]), target_y=int(dest[1])))
                    n_peel += 1
            elif len(idles_home) >= MIN_ARMY_FOR_ASSAULT:
                out.append(CommandModel(
                    action=ActionType.ARMY_ATTACK_MOVE,
                    target_x=int(dest[0]), target_y=int(dest[1])))

    # 3b-4) Asalto FULL / hunt / recall / rally-al-dest: off. Ablation only.
    if SUPPORT_ASSAULT:
        has_harv = _has_harvester(obs, eco, units, prod)
        raw_dest = _push_cell(obs, last_push)
        waypoint = _is_beacon_or_hunt(obs, raw_dest)
        dest = _snap_passable(obs, raw_dest, aidx)
        defending = bool(home_raid_targets(obs))
        n_home = _n_combat_near_own_base(obs, combat)
        n_at_dest = 0
        if dest is not None:
            n_at_dest = _n_combat_at(combat, dest, ARRIVED_CELLS)
        beacon = resolve_beacon(obs)
        n_at_beacon = (_n_combat_at(combat, beacon, ARRIVED_CELLS)
                       if beacon is not None else 0)
        assault = (has_proc and has_harv and dest is not None
                   and n_home >= MIN_ARMY_FOR_ASSAULT)
        if dest is not None and (has_proc or defending):
            px, py = dest
            idles = [u for u in combat if bool(getattr(u, "is_idle", False))]
            idles_home = [u for u in idles if _near_own_base(obs, _xy(u))]
            recall = bool(defending and combat and n_at_dest < MIN_PILE_FOR_HUNT)
            reassault = bool(
                assault
                and not defending
                and n_at_dest < MIN_ARMY_FOR_ASSAULT
                and n_home >= MIN_ARMY_FOR_ASSAULT
                and waypoint
            )
            pack_idle = assault and len(idles_home) >= MIN_ARMY_FOR_ASSAULT
            sweep = (not defending and has_proc and has_harv
                     and n_at_beacon >= MIN_PILE_FOR_HUNT
                     and len(idles) >= MIN_PILE_FOR_HUNT)
            if recall or reassault or pack_idle or sweep:
                out.append(CommandModel(
                    action=ActionType.ARMY_ATTACK_MOVE, target_x=px, target_y=py))
        if has_proc:
            pack_committed = bool(
                defending
                or n_home >= MIN_ARMY_FOR_ASSAULT
                or n_at_dest >= MIN_PILE_FOR_HUNT
                or n_at_beacon >= MIN_PILE_FOR_HUNT
            )
            if dest is not None and pack_committed:
                rx_t, ry_t = dest
            else:
                rx_t, ry_t = _staging_cell(obs, dest, aidx)
            for b in blds:
                if str(getattr(b, "type", "")).lower() not in _RALLY_BUILDINGS:
                    continue
                rx = int(getattr(b, "rally_x", -1) if getattr(b, "rally_x", -1) is not None else -1)
                ry = int(getattr(b, "rally_y", -1) if getattr(b, "rally_y", -1) is not None else -1)
                if rx == int(rx_t) and ry == int(ry_t):
                    continue
                out.append(CommandModel(
                    action=ActionType.SET_RALLY_POINT,
                    actor_id=int(b.actor_id),
                    target_x=int(rx_t),
                    target_y=int(ry_t),
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
