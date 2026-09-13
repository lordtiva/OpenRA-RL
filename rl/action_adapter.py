"""Traducción entre índices de la red y CommandModel del engine.

Responsabilidades:
    - Construir ActionIndex: máscaras legales por cabeza para una observación
    - Convertir la elección de la red en OpenRAAction ejecutable

Lecciones de la primera corrida real (crash C# "Exception was thrown by
handler"):
    - TRAIN con un tipo de EDIFICIO revienta el handler -> los ítems se
      separan en unidades vs edificios usando can_produce de nuestros propios
      edificios de producción (dato que ya viene en la observación)
    - ATTACK sin enemigo visible cerca -> target_actor_id=0 -> NRE en C# ->
      se degrada a ATTACK_MOVE hacia la celda (siempre seguro)
    - Acciones de EDIFICIO (sell/repair/rally/power_down/set_primary):
      P1 usable — unit_slot indexa ActionIndex.building_ids; act/eval usan
      building_valid (no unit_own). Cabeza dedicada puede venir después.
"""

import logging
import random

import numpy as np
import torch

from openra_env.models import ActionType, CommandModel, OpenRAAction
from rl.network import TYPE_TO_IDX, build_type_masks
from rl.obs_encoding import (
    BEACON_BY_MAP, MAX_UNITS, pending_place_item, resolve_beacon,
    select_unit_slots,
)

_log = logging.getLogger(__name__)

# P1.3: silent _item_slot_of fallback counter (drained by rollout → action_hist).
_ITEM_SLOT_FALLBACK_COUNT = 0


def drain_item_slot_fallback() -> int:
    """Return and reset module fallback hits since last drain."""
    global _ITEM_SLOT_FALLBACK_COUNT
    n = int(_ITEM_SLOT_FALLBACK_COUNT)
    _ITEM_SLOT_FALLBACK_COUNT = 0
    return n


# Tipos habilitados en v0.1 (el resto ni entra en la máscara)
ENABLED_TYPES = {
    "no_op", "move", "attack_move", "attack", "stop", "harvest",
    "set_stance", "deploy", "train", "build", "place_building",
    "cancel_production", "army_attack_move",
    "infantry_attack_move", "vehicle_attack_move", "harvesters_move",
    "naval_attack_move", "air_attack_move",
    # P1 scaffold: building slot (masked until buildings exist)
    "sell", "repair", "set_rally_point", "power_down", "set_primary",
    # P2 micro: guard (existing ActionType) + group stop/stance/guard macros
    "guard", "army_stop", "army_set_stance", "army_guard",
    # Aliados: already in ACTION_TYPES; unmask for APC / LST / chinook.
    "enter_transport", "unload",
    "patrol", "support_power",
}

# Types that pick a building actor via building_head -> building_ids.
BUILDING_SLOT_TYPES = {
    "sell", "repair", "set_rally_point", "power_down", "set_primary",
}

UNIT_ACTION_TYPES = {"move", "attack_move", "attack", "stop", "set_stance",
                     "harvest", "guard", "enter_transport", "unload", "patrol"}


# Techo del vocabulario de tipos de actor (debe == n_item_types de
# AlphaLiteNet). El embedding es fijo; si id_of() asignara ids >= n el
# lookup haría device-side assert en CUDA (ocurrió en Fase 2: el mapa con
# base completa desbloquea TODO el árbol de RA y superó 64 tipos).
MAX_ITEM_TYPES = 128   # RA completo supera 64 (3tnk era el #65 al explotar)


class Vocab:
    """Vocabulario estable de ROLES funcionales (agnóstico a facción).

    La cabeza de ítems ya no indexa nombres de actor concretos (1tnk, e1...)
    que cambian por facción, sino ROLES (rl.roles: 'tank_medium', 'power'...).
    El vocab se puede SEMBRAR con todos los roles del catálogo para que los ids
    sean estables entre reinicios/facciones (traductor universal).
    """

    def __init__(self):
        self.type_to_id = {}

    def seed_roles(self):
        """Pre-asigna un id estable a cada rol del catálogo RA.

        IDENTITY_ITEMS se agregan al final (append-only) para no reordenar
        los ids de roles en un scratch nuevo vs un ckpt viejo que no los tenía.
        """
        from rl.roles import IDENTITY_ITEMS, ROLE_OF_ITEM
        roles = sorted(set(ROLE_OF_ITEM.values()))
        for rol in roles:
            self.id_of(rol)
        for it in sorted(IDENTITY_ITEMS):
            self.id_of(it)
        return self.type_to_id

    def id_of(self, type_str: str) -> int:
        i = self.type_to_id.get(type_str)
        if i is None:
            if len(self.type_to_id) >= MAX_ITEM_TYPES:
                raise RuntimeError(
                    f"vocab lleno ({MAX_ITEM_TYPES} tipos): '{type_str}' sin "
                    f"id libre. Ampliar n_item_types en network.py Y "
                    f"MAX_ITEM_TYPES juntos.")
            i = len(self.type_to_id)
            self.type_to_id[type_str] = i
        return i


# Edificios conocidos del mod RA (nombres internos del engine).
# Clasificación ESTÁTICA: el can_produce del cuartel de mando lista tanto
# unidades como edificios, así que la lógica vieja (intersección con la
# unión de can_produce) mandaba las refinerías al cubo equivocado y dejaba
# la acción 'build' casi siempre enmascarada.
BUILDING_ITEM_TYPES = {
    # economía
    "proc", "silo",
    # energía
    "powr", "apwr",
    # producción militar
    "barr", "tent", "kenn", "weap", "hpad", "afld", "dome", "fix", "atek", "stek",
    # defensa y navales
    "gun", "ftur", "tsla", "agun", "pbox", "hbox", "sam", "gap",
    "spen", "syrd",
    # Superweapons (pdox=Chronosphere Allied). Missing → TRAIN crash like sbag.
    "pdox", "iron", "mslo",
    # France fakes (Defense queue). Same TRAIN trap if omitted.
    "fpwr", "tenf", "syrf", "spef", "weaf", "domf", "fixf", "fapw",
    "atef", "pdof", "mslf", "facf",
    # muros: si faltan, _split_production los manda a TRAIN y el C# los
    # mete en la cola de edificios (visor 985: train:sbag / train:brik,
    # 0 rifles, lose ~10k).
    "sbag", "brik", "fenc",
}

# Combat TRAIN roles: masked until proc + harvester. Without a standing
# refinery we also mask ALL TRAIN (including harvester) and BUILD of
# anything except power/refinery, so the 5000 cannot dump into harvs or
# barracks before proc is placed.
COMBAT_TRAIN_ROLES = {
    "infantry_basic", "infantry_antiinf", "infantry_antiarmor",
    "rocket_truck", "tank_light", "tank_medium", "tank_heavy",
    "artillery", "scout", "specialist",
    "air_fighter", "air_bomber", "heli", "transporter",
    "ship_sub", "ship_combat", "ship_amphib",
    # IDENTITY_ITEMS that are still combat (gated until proc+harv).
    "e7", "medi", "mech", "spy", "ctnk", "stnk", "mh60",
}
ECONOMY_BUILD_ROLES = {"power", "refinery"}  # legal BUILD before proc exists
MOVE_CELL_TYPES = {
    "move", "attack_move", "attack", "army_attack_move",
    "infantry_attack_move", "vehicle_attack_move", "harvesters_move",
    "naval_attack_move", "air_attack_move", "harvest",
    "set_rally_point",
    "guard", "army_guard", "army_set_stance",
    "patrol", "support_power", "deploy",
}
# Combat movement: masked until a refinery stands. Otherwise PPO
# reward-hacks army_attack_move / attack_move (the 201-309 collapse).
COMBAT_MOVE_TYPES = (
    "army_attack_move", "attack_move", "attack",
    "infantry_attack_move", "vehicle_attack_move",
    "naval_attack_move", "air_attack_move",
)
GROUP_MACRO_TYPES = (
    "army_attack_move", "infantry_attack_move",
    "vehicle_attack_move", "harvesters_move",
    "naval_attack_move", "air_attack_move",
    "army_stop", "army_set_stance", "army_guard",
)
# Group push: legal once TOTAL combat >= PACK_ARMY (home OR field).
# Run 43: home-only gate froze 200+ units mid-map (incomplete @53k) because
# n_home dropped below 12 after the march. Still blocks drip-4 at home.
# attack_move per-unit stays on (raid peel).
PACK_ARMY = 12
PACK_HOME_RADIUS = 18
_NON_COMBAT_TAGS = ("harv", "mcv")


def owns_proc(obs) -> bool:
    """True if a refinery (proc) is already standing."""
    for b in getattr(obs, "buildings", None) or []:
        if str(getattr(b, "type", "")).lower() == "proc":
            return True
    return False


def economy_ready_for_combat(obs) -> bool:
    """True once a refinery is standing AND a harvester exists or is queued."""
    if not owns_proc(obs):
        return False
    eco = getattr(obs, "economy", None)
    if int(getattr(eco, "harvester_count", 0) or 0) > 0:
        return True
    for u in getattr(obs, "units", None) or []:
        if "harv" in str(getattr(u, "type", "")).lower():
            return True
    for p in getattr(obs, "production", None) or []:
        if "harv" in str(getattr(p, "item", "")).lower():
            return True
    return False


# Low-power plant: e1 ($100) vacuums cash+ore so a $300 powr never queues.
# Same style as proc-before-combat. Runtime — no scratch. PLACE stays legal.
POWR_COST = 300
POWER_PLANT_CAP = 8
_POWER_ITEMS = frozenset({"powr", "apwr", "power"})
_BUILDING_QUEUE = frozenset(
    {"building", "buildings", "structure", "structures"})


def spendable_resources(obs) -> int:
    """Cash + silo ore. OpenRA spends the combined pool."""
    eco = getattr(obs, "economy", None)
    return (int(getattr(eco, "cash", 0) or 0)
            + int(getattr(eco, "ore", 0) or 0))


def power_in_deficit(obs) -> bool:
    eco = getattr(obs, "economy", None)
    try:
        provided = int(getattr(eco, "power_provided", 0) or 0)
        drained = int(getattr(eco, "power_drained", 0) or 0)
    except (TypeError, ValueError):
        return False
    return drained > provided


def building_queue_busy(obs) -> bool:
    for p in getattr(obs, "production", None) or []:
        qt = str(getattr(p, "queue_type", "") or "").lower()
        if qt not in _BUILDING_QUEUE:
            continue
        try:
            if float(getattr(p, "progress", 0) or 0) < 0.99:
                return True
        except (TypeError, ValueError):
            continue
    return False


def power_in_flight(obs) -> bool:
    """True if a plant is queued or ready to PLACE."""
    for p in getattr(obs, "production", None) or []:
        if str(getattr(p, "item", "") or "").lower() in _POWER_ITEMS:
            return True
    pending = pending_place_item(obs)
    return bool(pending) and str(pending).lower() in _POWER_ITEMS


def can_produce_power(obs) -> bool:
    for it in getattr(obs, "available_production", None) or []:
        if str(it or "").lower() in _POWER_ITEMS:
            return True
    return False


def n_power_plants(obs) -> int:
    n = 0
    for b in getattr(obs, "buildings", None) or []:
        if str(getattr(b, "type", "") or "").lower() in ("powr", "apwr"):
            n += 1
    return n


def needs_power_build(obs) -> bool:
    """Deficit + can pay + queue free + plant not already in flight."""
    if not power_in_deficit(obs) or not owns_proc(obs):
        return False
    if spendable_resources(obs) < POWR_COST:
        return False
    if power_in_flight(obs) or building_queue_busy(obs):
        return False
    if n_power_plants(obs) >= POWER_PLANT_CAP:
        return False
    return can_produce_power(obs)


def _apply_power_priority(obs, t_name: str, item_type: str, aidx):
    """When low-power: PLACE/BUILD powr beats train e1 / build pbox."""
    if not power_in_deficit(obs) or not owns_proc(obs):
        return t_name, item_type
    pending = pending_place_item(obs)
    if pending and str(pending).lower() in _POWER_ITEMS:
        if t_name in ("train", "build", "no_op"):
            return "place_building", pending
        return t_name, item_type
    item = str(item_type or "")
    has_power_slot = "power" in (getattr(aidx, "build_items", None) or [])
    if t_name == "train" and item in COMBAT_TRAIN_ROLES:
        if needs_power_build(obs) and has_power_slot:
            return "build", "power"
        return "no_op", item_type
    if t_name == "build" and item not in _POWER_ITEMS:
        if needs_power_build(obs) and has_power_slot:
            return "build", "power"
        if power_in_flight(obs):
            return "no_op", item_type
    return t_name, item_type


def _is_combat_unit(u) -> bool:
    ut = str(getattr(u, "type", "") or "").lower()
    return not any(tag in ut for tag in _NON_COMBAT_TAGS)


def n_combat_total(obs) -> int:
    """Combat units anywhere on the map (excludes harv/mcv)."""
    n = 0
    for u in getattr(obs, "units", None) or []:
        if _is_combat_unit(u):
            n += 1
    return int(n)


_INFANTRY_TYPE_PREFIXES = ("e1", "e2", "e3", "e4", "e6", "e7", "dog", "spy",
                           "med", "medi", "mech", "shok", "thf", "chan", "delphi")
_TRANSPORT_TYPES = frozenset({"apc", "lst", "tran", "stnk"})
# Exact internal names from rl.roles (RA air / navy). Keep vehicle land-only.
_NAVAL_TYPES = frozenset({"ss", "msub", "dd", "ca", "pt", "lst"})
_AIR_TYPES = frozenset({
    "mig", "yak", "u2", "badr", "heli", "hind", "mh60", "tran",
})


def _utype(u) -> str:
    return str(getattr(u, "type", "") or "").lower()


def _is_infantry_unit(u) -> bool:
    t = _utype(u)
    return any(t == p or t.startswith(p) for p in _INFANTRY_TYPE_PREFIXES)


def _is_naval_unit(u) -> bool:
    return _utype(u) in _NAVAL_TYPES


def _is_air_unit(u) -> bool:
    return _utype(u) in _AIR_TYPES


def _is_vehicle_combat(u) -> bool:
    """Land combat vehicle only (excludes infantry / naval / air)."""
    if not _is_combat_unit(u):
        return False
    if _is_infantry_unit(u) or _is_naval_unit(u) or _is_air_unit(u):
        return False
    return True


def _is_harvester_unit(u) -> bool:
    return "harv" in _utype(u)


def _is_transport_unit(u) -> bool:
    return _utype(u) in _TRANSPORT_TYPES


def _passenger_count(u) -> int:
    try:
        return int(getattr(u, "passenger_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _unit_by_id(obs, actor_id):
    try:
        aid = int(actor_id or 0)
    except (TypeError, ValueError):
        return None
    if aid <= 0:
        return None
    for u in getattr(obs, "units", None) or []:
        try:
            if int(getattr(u, "actor_id", 0) or 0) == aid:
                return u
        except (TypeError, ValueError):
            continue
    return None


def _is_chrono_tank_unit(u) -> bool:
    return _utype(u) == "ctnk"


def _any_chrono_tank(obs):
    for u in getattr(obs, "units", None) or []:
        if _is_chrono_tank_unit(u):
            try:
                aid = int(getattr(u, "actor_id", 0) or 0)
            except (TypeError, ValueError):
                continue
            if aid > 0:
                return aid
    return None


def _pick_support_power(obs) -> str:
    ready = [str(x) for x in (getattr(obs, "ready_support_powers", None) or []) if x]
    ready = [k for k in ready if "gps" not in k.lower()]
    if not ready:
        return ""
    pref = (
        "Chronoshift", "AdvancedChronoshift", "NukePowerOrder",
        "GrantExternalConditionPowerOrder", "SovietSpyPlane",
        "SovietParatroopers", "UkraineParabombs",
    )
    for p in pref:
        if p in ready:
            return p
    return ready[0]


def _first_infantry_id(obs) -> int:
    for u in getattr(obs, "units", None) or []:
        if not _is_infantry_unit(u):
            continue
        try:
            aid = int(getattr(u, "actor_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if aid > 0:
            return aid
    return 0


def can_issue_enter_transport(obs) -> bool:
    """Infantry + a friendly transport (APC / LST / chinook / phase)."""
    units = list(getattr(obs, "units", None) or [])
    has_inf = any(_is_infantry_unit(u) for u in units)
    has_tr = any(_is_transport_unit(u) for u in units)
    return bool(has_inf and has_tr)


def can_issue_unload(obs) -> bool:
    """A friendly transport with at least one passenger."""
    for u in getattr(obs, "units", None) or []:
        if _is_transport_unit(u) and _passenger_count(u) > 0:
            return True
    return False


def _nearest_own_transport(obs, from_id: int):
    """Closest own transport actor_id to from_id (or first transport)."""
    units = list(getattr(obs, "units", None) or [])
    origin = None
    for u in units:
        try:
            if int(getattr(u, "actor_id", 0) or 0) == int(from_id or 0):
                origin = (int(u.cell_x), int(u.cell_y))
                break
        except (TypeError, ValueError):
            continue
    best, best_d = None, None
    for u in units:
        if not _is_transport_unit(u):
            continue
        try:
            aid = int(getattr(u, "actor_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if aid <= 0 or aid == int(from_id or 0):
            continue
        if origin is None:
            return aid
        try:
            d = (int(u.cell_x) - origin[0]) ** 2 + (int(u.cell_y) - origin[1]) ** 2
        except (TypeError, ValueError):
            d = 0
        if best_d is None or d < best_d:
            best, best_d = aid, d
    return best


def _any_loaded_transport(obs):
    for u in getattr(obs, "units", None) or []:
        if _is_transport_unit(u) and _passenger_count(u) > 0:
            try:
                aid = int(getattr(u, "actor_id", 0) or 0)
            except (TypeError, ValueError):
                continue
            if aid > 0:
                return aid
    return None


def n_infantry_total(obs) -> int:
    return sum(1 for u in (getattr(obs, "units", None) or [])
               if _is_infantry_unit(u))


def n_vehicle_combat_total(obs) -> int:
    return sum(1 for u in (getattr(obs, "units", None) or [])
               if _is_vehicle_combat(u))


def n_naval_total(obs) -> int:
    return sum(1 for u in (getattr(obs, "units", None) or [])
               if _is_naval_unit(u))


def n_air_total(obs) -> int:
    return sum(1 for u in (getattr(obs, "units", None) or [])
               if _is_air_unit(u))


def n_harvester_total(obs) -> int:
    return sum(1 for u in (getattr(obs, "units", None) or [])
               if _is_harvester_unit(u))


def group_actor_ids(obs, group: str, limit: int = 64) -> list:
    """Actor ids for a role-group macro (stable actor_id order)."""
    units = list(getattr(obs, "units", None) or [])
    units.sort(key=lambda u: int(getattr(u, "actor_id", 0) or 0))
    out = []
    for u in units:
        aid = int(getattr(u, "actor_id", 0) or 0)
        if aid <= 0:
            continue
        if group == "army":
            ok = _is_combat_unit(u)
        elif group == "infantry":
            ok = _is_infantry_unit(u)
        elif group == "vehicle":
            ok = _is_vehicle_combat(u)
        elif group == "naval":
            ok = _is_naval_unit(u)
        elif group == "air":
            ok = _is_air_unit(u)
        elif group == "harvesters":
            ok = _is_harvester_unit(u)
        else:
            ok = False
        if ok:
            out.append(aid)
        if len(out) >= limit:
            break
    return out


def n_combat_near_own_base(obs, radius: int = PACK_HOME_RADIUS) -> int:
    """Combat units within radius of an own building (yard, not the field)."""
    r2 = int(radius) * int(radius)
    blds = list(getattr(obs, "buildings", None) or [])
    n = 0
    for u in getattr(obs, "units", None) or []:
        if not _is_combat_unit(u):
            continue
        try:
            ux, uy = int(u.cell_x), int(u.cell_y)
        except (TypeError, ValueError):
            continue
        for b in blds:
            try:
                dx = ux - int(b.cell_x)
                dy = uy - int(b.cell_y)
            except (TypeError, ValueError):
                continue
            if dx * dx + dy * dy <= r2:
                n += 1
                break
    return n


def apply_passability(aidx, pass_hw) -> None:
    """Mask the cell head with spatial channel 3 (passable=1). No-op if empty."""
    import numpy as _np
    if pass_hw is None:
        return
    grid = _np.asarray(pass_hw)
    if grid.ndim != 2 or grid.shape != (aidx.h, aidx.w):
        return
    legal = grid > 0.5
    if not legal.any():
        return  # keep all-true rather than deadlock the Categorical
    aidx.pass_grid = legal
    aidx.cell_mask = torch.from_numpy(_np.ascontiguousarray(legal.reshape(-1)))


def nearest_passable(x: int, y: int, pass_grid, h: int, w: int, max_r: int = 16):
    """Closest passable cell to (x,y). If no grid, just clamp to the map."""
    x = int(max(0, min(w - 1, x)))
    y = int(max(0, min(h - 1, y)))
    if pass_grid is None:
        return x, y
    if 0 <= y < h and 0 <= x < w and bool(pass_grid[y, x]):
        return x, y
    for r in range(1, max_r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) != r and abs(dy) != r:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= ny < h and 0 <= nx < w and bool(pass_grid[ny, nx]):
                    return nx, ny
    ys, xs = np.where(pass_grid)
    if len(xs):
        i = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
        return int(xs[i]), int(ys[i])
    return x, y


def _passable_width(x: int, y: int, grid, h: int, w: int, rad: int = 2) -> int:
    """How many passable cells in a (2*rad+1)^2 window — proxy for corridor width."""
    if grid is None:
        return 0
    n = 0
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            nx, ny = x + dx, y + dy
            if 0 <= ny < h and 0 <= nx < w and bool(grid[ny, nx]):
                n += 1
    return n


def _wider_flank_passable(tx: int, ty: int, grid, h: int, w: int):
    """Prefer north/south land corridors over the closest water-edge cell.

    Singles-style maps have a central lake; nearest_passable to a water cell
    often lands on a 1-2-wide choke. Stage via the wider flank at similar x.
    """
    if grid is None:
        return None
    tx = int(max(0, min(w - 1, tx)))
    # Relative flanks (Informe-3 ~y=8 / y=30 on ~40-tall maps).
    flank_ys = (max(1, h // 8), min(h - 2, (3 * h) // 4))
    best = None
    best_score = -1
    for fy in flank_ys:
        px, py = nearest_passable(tx, fy, grid, h, w)
        score = _passable_width(px, py, grid, h, w)
        # Prefer the flank that is actually passable and wide.
        if score > best_score:
            best_score = score
            best = (px, py)
    return best



def _army_centroid(obs):
    combat = []
    for u in getattr(obs, "units", None) or []:
        ut = str(getattr(u, "type", "") or "").lower()
        if "harv" in ut or "mcv" in ut:
            continue
        if not bool(getattr(u, "can_attack", True)):
            continue
        try:
            combat.append((int(u.cell_x), int(u.cell_y)))
        except (TypeError, ValueError):
            continue
    if not combat:
        return None
    sx = sum(c[0] for c in combat) / len(combat)
    sy = sum(c[1] for c in combat) / len(combat)
    return sx, sy


def _midline_blocked(x0: float, y0: float, x1: float, y1: float, grid, h: int, w: int) -> bool:
    """True if the straight segment crosses mostly impassable cells (lake)."""
    if grid is None:
        return False
    steps = 12
    blocked = 0
    for i in range(1, steps):
        t = i / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        if not (0 <= x < w and 0 <= y < h) or not bool(grid[y, x]):
            blocked += 1
    return blocked >= steps // 2


def stage_army_attack_cell(obs, aidx, cx: int, cy: int):
    """If army→target crosses a lake, stage via the wider N/S flank first."""
    grid = getattr(aidx, "pass_grid", None)
    h, w = aidx.h, aidx.w
    # Combat list (same filters as _army_centroid). n_advanced exists because a
    # split army / base spawn pulls the mean centroid west — enough units past
    # the choke must not remap the attack cell backward to the south flank.
    combat = []
    for u in getattr(obs, "units", None) or []:
        ut = str(getattr(u, "type", "") or "").lower()
        if "harv" in ut or "mcv" in ut:
            continue
        if not bool(getattr(u, "can_attack", True)):
            continue
        try:
            combat.append((int(u.cell_x), int(u.cell_y)))
        except (TypeError, ValueError):
            continue
    n_advanced = sum(1 for c in combat if c[0] > 35)
    if n_advanced >= 8:
        return cx, cy
    cen = None
    if combat:
        sx = sum(c[0] for c in combat) / len(combat)
        sy = sum(c[1] for c in combat) / len(combat)
        cen = (sx, sy)
    if cen is None or grid is None:
        return cx, cy
    # Already past home choke on east-facing maps — keep the attack cell.
    if cen[0] > 35:
        return cx, cy
    if not _midline_blocked(cen[0], cen[1], cx, cy, grid, h, w):
        return cx, cy
    flank = _wider_flank_passable(int((cen[0] + cx) / 2), int((cen[1] + cy) / 2), grid, h, w)
    if flank is None:
        return cx, cy
    # Map-agnostic attractor guard: army already standing on/near the flank
    # waypoint — do not re-stage onto the same cell they occupy.
    if (flank[0] - cen[0]) ** 2 + (flank[1] - cen[1]) ** 2 < 36:
        return cx, cy
    # Only stage if flank is meaningfully different from final target.
    if (flank[0] - cx) ** 2 + (flank[1] - cy) ** 2 < 36:
        return cx, cy
    return int(flank[0]), int(flank[1])


def should_emit_army_push(prev, cx, cy, eps=8) -> bool:
    """True if army push cell moved enough to warrant re-emitting.

    Hysteresis for executed OpenRA commands only (rollout / live). Keep BC
    labels as the intended attack cell — call this when building final
    OpenRAAction.commands, not inside index_to_command.
    """
    if prev is None:
        return True
    try:
        dx = int(cx) - int(prev[0])
        dy = int(cy) - int(prev[1])
    except (TypeError, ValueError, IndexError):
        return True
    return dx * dx + dy * dy >= int(eps) * int(eps)


def filter_army_push_hysteresis(commands, last_army_push, eps=8):
    """Drop near-duplicate army_attack_move from a final command list.

    Returns (kept_commands, new_last_army_push). Falls back to [no_op] if empty.
    """
    if not commands:
        return list(commands or []), last_army_push
    kept = []
    new_last = last_army_push
    for c in commands:
        cname = getattr(getattr(c, "action", None), "value", None) or str(
            getattr(c, "action", ""))
        if cname == "army_attack_move" and getattr(c, "target_x", None) is not None:
            tx, ty = int(c.target_x), int(c.target_y)
            if not should_emit_army_push(new_last, tx, ty, eps=eps):
                continue
            new_last = (tx, ty)
        kept.append(c)
    if not kept:
        from openra_env.models import ActionType, CommandModel
        kept = [CommandModel(action=ActionType.NO_OP)]
    return kept, new_last

def remap_move_cell(obs, aidx, cx: int, cy: int, actor_id: int = 0):
    """If (cx,cy) is water/OOB/unpathable, retarget near the click — not south flank.

    Order: legal keep; else nearest_passable(cx,cy); else nearest enemy;
    else resolve_beacon + nearest_passable; else near unit.
    Does NOT use _wider_flank_passable (that snap sent illegal mid-map water
    clicks to the south ore corridor y~38-40). Flank staging stays only in
    stage_army_attack_cell (opening choke).
    Does NOT use a hardcoded y<40 water line — passability comes from obs/spatial.
    If there is no passability grid, only OOB is illegal.
    """
    h, w = aidx.h, aidx.w
    grid = getattr(aidx, "pass_grid", None)

    def legal(x, y):
        if not (0 <= x < w and 0 <= y < h):
            return False
        if grid is None:
            return True
        return bool(grid[y, x])

    if legal(cx, cy):
        return cx, cy

    # Prefer land near the illegal click (not Informe-3 N/S ore flank).
    if grid is not None:
        px, py = nearest_passable(cx, cy, grid, h, w)
        if legal(px, py):
            return int(px), int(py)

    enemies = list(getattr(obs, "visible_enemies", None) or []) + list(
        getattr(obs, "visible_enemy_buildings", None) or [])
    if enemies:
        e = min(enemies, key=lambda e: (int(e.cell_x) - cx) ** 2 + (int(e.cell_y) - cy) ** 2)
        return nearest_passable(int(e.cell_x), int(e.cell_y), grid, h, w)

    beacon = resolve_beacon(obs)
    if beacon:
        return nearest_passable(int(beacon[0]), int(beacon[1]), grid, h, w)

    ux, uy = cx, cy
    units = list(getattr(obs, "units", None) or [])
    found = False
    if actor_id:
        for u in units:
            if int(getattr(u, "actor_id", 0) or 0) == int(actor_id):
                ux, uy = int(u.cell_x), int(u.cell_y)
                found = True
                break
    if not found:
        if units:
            ux, uy = int(units[0].cell_x), int(units[0].cell_y)
        else:
            blds = list(getattr(obs, "buildings", None) or [])
            if blds:
                ux, uy = int(blds[0].cell_x), int(blds[0].cell_y)
    return nearest_passable(ux, uy, grid, h, w)


def _combat_xy_list(obs):
    """Own combat cell positions (skip harv/mcv / non-attackers)."""
    out = []
    for u in getattr(obs, "units", None) or []:
        ut = str(getattr(u, "type", "") or "").lower()
        if "harv" in ut or "mcv" in ut:
            continue
        if not bool(getattr(u, "can_attack", True)):
            continue
        try:
            out.append((int(u.cell_x), int(u.cell_y)))
        except (TypeError, ValueError):
            continue
    return out


def _east_push_objective(obs, aidx, cx: int, cy: int, front_xs):
    """Beacon / visible contact / mental base / clamp to front — never hardcode GPS."""
    grid = getattr(aidx, "pass_grid", None)
    h, w = aidx.h, aidx.w
    contacts = list(getattr(obs, "visible_enemy_buildings", None) or []) + list(
        getattr(obs, "visible_enemies", None) or [])
    if contacts:
        e = max(contacts, key=lambda o: int(getattr(o, "cell_x", 0) or 0))
        return nearest_passable(int(e.cell_x), int(e.cell_y), grid, h, w)
    mental = getattr(obs, "enemy_base_xy", None)
    if mental is None:
        bel = getattr(obs, "belief", None)
        if bel is not None:
            mental = getattr(bel, "enemy_base_xy", None)
    if mental is not None:
        try:
            return nearest_passable(int(mental[0]), int(mental[1]), grid, h, w)
        except (TypeError, ValueError, IndexError):
            pass
    beacon = resolve_beacon(obs)
    if beacon is not None:
        return nearest_passable(int(beacon[0]), int(beacon[1]), grid, h, w)
    # Map-agnostic: keep at least the forward blob x (75th percentile).
    if front_xs:
        xs = sorted(int(x) for x in front_xs)
        px = xs[int(0.75 * (len(xs) - 1))]
        return nearest_passable(max(int(cx), int(px)), int(cy), grid, h, w)
    return int(cx), int(cy)


def guard_army_push_cell(obs, aidx, cx: int, cy: int):
    """Don't yank an eastern vanguard west to ore; fog-push east when blind.

    Cause 2 — advanced contingent: if >=~8 combat with cell_x>70 and no base
    threat (no visible enemies/buildings with x<40), refuse targets far behind
    the front (cx < 50, or cx < min_front_x - 15). Retarget via
    _east_push_objective (visible / mental / resolve_beacon / front clamp).

    Cause 3 — fog penetration: when the front is deep east (>=8 with x>75 or
    army front/centroid x>75) and no visible_enemy_buildings, ensure the push
    goes toward the enemy quadrant (beacon/fog-east), not mid-map ore.
    Light: only retarget when the issued cell is behind/west of the front.
    """
    combat = _combat_xy_list(obs)
    if not combat:
        return int(cx), int(cy)
    xs = [c[0] for c in combat]
    n70 = sum(1 for x in xs if x > 70)
    n75 = sum(1 for x in xs if x > 75)
    front70 = [x for x in xs if x > 70]
    front75 = [x for x in xs if x > 75]
    cen_x = sum(xs) / len(xs)
    max_x = max(xs)

    def _base_threat() -> bool:
        for e in list(getattr(obs, "visible_enemies", None) or []) + list(
                getattr(obs, "visible_enemy_buildings", None) or []):
            try:
                if int(e.cell_x) < 40:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    behind = False
    if n70 >= 8 and not _base_threat():
        if int(cx) < 50:
            behind = True
        elif front70 and int(cx) < (min(front70) - 15):
            behind = True

    n_enemy_bldgs = len(getattr(obs, "visible_enemy_buildings", None) or [])
    fog_east = False
    if n_enemy_bldgs == 0 and (n75 >= 8 or max_x > 75 or cen_x > 75):
        ref = min(front75) if front75 else (max_x if max_x > 75 else None)
        if ref is not None and int(cx) < int(ref) - 10:
            fog_east = True
        elif int(cx) < 70 and (n75 >= 8 or cen_x > 75):
            fog_east = True

    if behind or fog_east:
        return _east_push_objective(obs, aidx, cx, cy, front70 or front75 or xs)
    return int(cx), int(cy)


def _split_production(obs):
    """Separa available_production en (roles de entrenables, roles de
    construibles) + mapa rol->item concreto.

    TRAductor universal (agnóstico a facción): la cabeza de ítems indexa ROLES
    estables (rl.roles), no nombres internos que varían por facción. Cada
    rol->[items concretos disponibles]. Al armar el comando, el adapter elige
    el item concreto de la facción actual para el rol muestreado.
    """
    available = set(str(x).lower() for x in (obs.available_production or []) if x)
    buildables = available & BUILDING_ITEM_TYPES
    trainables = available - buildables
    # roles disponibles + concreto más barato (pbox no agun; ftur no tsla).
    # IDENTITY_ITEMS keep their internName so Tanya/pdox are not cheapest_of'd.
    def _roles(items):
        from rl.roles import IDENTITY_ITEMS, role_of
        por_rol: dict[str, list[str]] = {}
        for it in sorted(items):
            key = it if it in IDENTITY_ITEMS else role_of(it)
            por_rol.setdefault(key, []).append(it)
        return por_rol

    from rl.roles import cheapest_of
    train_por_rol = _roles(trainables)
    build_por_rol = _roles(buildables)
    train_roles = sorted(train_por_rol)
    build_roles = sorted(build_por_rol)
    rol_a_concreto = {r: cheapest_of(items) for r, items in train_por_rol.items()}
    for r, items in build_por_rol.items():
        rol_a_concreto[r] = cheapest_of(items)
    return train_roles, build_roles, rol_a_concreto



# P2 micro helpers -----------------------------------------------------------

_STANCE_ATTACK_ANYTHING = 3  # proto SET_STANCE target_x


def _any_visible_enemy(obs) -> bool:
    return bool(getattr(obs, "visible_enemies", None) or []) or bool(
        getattr(obs, "visible_enemy_buildings", None) or [])


def _guard_candidate_entries(obs):
    """Own actors worth escorting: harv/MCV first, then buildings, else units."""
    out = []
    for u in getattr(obs, "units", None) or []:
        try:
            aid = int(getattr(u, "actor_id", 0) or 0)
            cx = int(getattr(u, "cell_x", 0) or 0)
            cy = int(getattr(u, "cell_y", 0) or 0)
        except (TypeError, ValueError):
            continue
        if aid <= 0:
            continue
        ut = str(getattr(u, "type", "") or "").lower()
        if "harv" in ut:
            prio = 0
        elif "mcv" in ut:
            prio = 1
        else:
            prio = 3
        out.append((prio, aid, cx, cy))
    for b in getattr(obs, "buildings", None) or []:
        try:
            aid = int(getattr(b, "actor_id", 0) or 0)
            cx = int(getattr(b, "cell_x", 0) or 0)
            cy = int(getattr(b, "cell_y", 0) or 0)
        except (TypeError, ValueError):
            continue
        if aid <= 0:
            continue
        out.append((2, aid, cx, cy))
    return out


def has_guard_target(obs, exclude_id: int = 0) -> bool:
    ex = int(exclude_id or 0)
    for _p, aid, _x, _y in _guard_candidate_entries(obs):
        if aid != ex:
            return True
    return False


def can_issue_guard(obs) -> bool:
    """True if some combat escort has a distinct own target (harv/MCV/bld/unit)."""
    for aid in group_actor_ids(obs, "army"):
        if has_guard_target(obs, exclude_id=aid):
            return True
    return False


def resolve_guard_target(obs, cx: int, cy: int, exclude_id: int = 0):
    """Nearest preferred escort target to cell (harv/MCV/building/own)."""
    ex = int(exclude_id or 0)
    best, best_key = None, None
    for prio, aid, ax, ay in _guard_candidate_entries(obs):
        if aid == ex:
            continue
        d = (ax - int(cx)) ** 2 + (ay - int(cy)) ** 2
        key = (prio, d, aid)
        if best_key is None or key < best_key:
            best, best_key = aid, key
    return best


def _is_escort_unit(obs, actor_id) -> bool:
    """Combat unit suitable to Guard-on-actor (not harv/mcv)."""
    try:
        aid = int(actor_id or 0)
    except (TypeError, ValueError):
        return False
    if aid <= 0:
        return False
    for u in getattr(obs, "units", None) or []:
        try:
            if int(getattr(u, "actor_id", 0) or 0) != aid:
                continue
        except (TypeError, ValueError):
            continue
        if not _is_combat_unit(u):
            return False
        return bool(getattr(u, "can_attack", True))
    return False


def _stance_from_cell(cx: int, cy: int = 0) -> int:
    """Map cell to stance bucket 0..3 (HoldFire..AttackAnything)."""
    return int(cx) % 4


def _focus_enemy_at_cell(obs, cx: int, cy: int):
    """Focus-fire target: nearest visible enemy to cell, prefer wounded units.

    Strengthens ATTACK vs pure attack_move-to-cell by always binding a
    concrete target_actor_id when enemies are visible.
    """
    best, best_key = None, None
    units = list(getattr(obs, "visible_enemies", None) or [])
    blds = list(getattr(obs, "visible_enemy_buildings", None) or [])
    for is_bld, pool in ((0, units), (1, blds)):
        for e in pool:
            try:
                aid = int(getattr(e, "actor_id", 0) or 0)
                ex = int(getattr(e, "cell_x", 0) or 0)
                ey = int(getattr(e, "cell_y", 0) or 0)
            except (TypeError, ValueError):
                continue
            if aid <= 0:
                continue
            try:
                hp = float(getattr(e, "hp_percent", 1.0) or 1.0)
            except (TypeError, ValueError):
                hp = 1.0
            d = (ex - int(cx)) ** 2 + (ey - int(cy)) ** 2
            # Prefer units over buildings; among equals, lower HP (focus fire).
            key = (d, is_bld, hp, aid)
            if best_key is None or key < best_key:
                best, best_key = aid, key
    return best


class ActionIndex:
    """Todo lo que la red necesita para decidir sobre una observación."""

    __slots__ = ("type_mask", "unit_valid", "cell_mask", "item_indices",
                 "item_mask", "unit_ids", "items", "train_items",
                 "build_items", "rol_a_concreto", "h", "w",
                 "train_slot_mask", "build_slot_mask", "pass_grid",
                 "building_ids", "building_valid")

    def __init__(self, obs, vocab: Vocab, device="cpu"):
        self.h = max(obs.map_info.height, 1)
        self.w = max(obs.map_info.width, 1)

        # Cabeza 1: tipos legales, acotados a los habilitados en v0.1
        raw_mask = build_type_masks(obs)
        m = np.zeros_like(raw_mask.numpy())
        for name in ENABLED_TYPES:
            m[TYPE_TO_IDX[name]] = raw_mask[TYPE_TO_IDX[name]]
        # harvest solo si hay cosechadoras
        if obs.economy.harvester_count <= 0:
            m[TYPE_TO_IDX["harvest"]] = False
        else:
            # Informe-3: camiones cosechan solos. Si ninguno esta idle, mask
            # harvest para no quemar 40-60% del ancho de banda en spam.
            idle_harv = any(
                "harv" in str(getattr(u, "type", "") or "").lower()
                and bool(getattr(u, "is_idle", False))
                for u in (getattr(obs, "units", None) or [])
            )
            if not idle_harv:
                m[TYPE_TO_IDX["harvest"]] = False
        # train/build requieren ítems de su categoría
        self.train_items, self.build_items, self.rol_a_concreto = \
            _split_production(obs)
        if not self.train_items:
            m[TYPE_TO_IDX["train"]] = False
        if not self.build_items:
            m[TYPE_TO_IDX["build"]] = False
        self.type_mask = torch.from_numpy(m)

        # Cabeza 2: SOLO propias (combat-first, ≤MAX_UNITS). Enemigos van
        # en tokens 96..127 del xf; dist_unit los enmascara (2c-B).
        units = select_unit_slots(obs)
        self.unit_ids = [u.actor_id for u in units]
        self.unit_valid = torch.zeros(MAX_UNITS, dtype=torch.bool)
        for i in range(len(self.unit_ids)):
            self.unit_valid[i] = True

        # P1: building slots (MAX_BUILDINGS==MAX_UNITS; building_head index)
        blds = list(getattr(obs, "buildings", None) or [])[:MAX_UNITS]
        self.building_ids = [
            int(getattr(b, "actor_id", 0) or 0) for b in blds
            if int(getattr(b, "actor_id", 0) or 0) > 0
        ]
        self.building_valid = torch.zeros(MAX_UNITS, dtype=torch.bool)
        for i in range(len(self.building_ids)):
            self.building_valid[i] = True

        # Cabeza 3: mapa completo como candidatos de celda
        self.cell_mask = torch.ones(self.h * self.w, dtype=torch.bool)

        # Cabeza 4: ROLES disponibles (traductor universal agnóstico a facción).
        # items = roles (train + build), estables entre facciones.
        items = self.train_items + self.build_items
        self.items = items
        # Tamaño FIJO = MAX_ITEM_TYPES: el vocab crece dinamicamente y si los
        # tensores cambiaran de largo entre episodios el torch.cat del update
        # explota ("Expected size 64 but got 65"). Padding estable siempre.
        n_vocab = MAX_ITEM_TYPES
        self.item_indices = torch.zeros(n_vocab, dtype=torch.long)
        self.item_mask = torch.zeros(n_vocab, dtype=torch.bool)
        for slot, it in enumerate(items[:n_vocab]):
            self.item_indices[slot] = vocab.id_of(it)
            self.item_mask[slot] = True
        # Enmascaramiento jerárquico ESTRICTO: máscaras de slots por categoría
        # de producción, para que la cabeza de ítems SOLO pueda elegir dentro
        # de la categoría del tipo (train->train_roles, build->build_roles).
        # Esto elimina las coerciones post-hoc del adapter.
        self.train_slot_mask = torch.zeros(n_vocab, dtype=torch.bool)
        self.build_slot_mask = torch.zeros(n_vocab, dtype=torch.bool)
        n_train = len(self.train_items)
        for slot in range(min(n_train, n_vocab)):
            self.train_slot_mask[slot] = True
        for slot in range(n_train, min(n_vocab, n_train + len(self.build_items))):
            self.build_slot_mask[slot] = True
        self.pass_grid = None

        # P3: when not naval_viable, forbid naval BUILD + ship TRAIN
        # (Informe-2 anti-syrd). Lakes alone (a_short) stay gated; navy maps OK.
        # Airbase / air units stay legal on land. Defense already unmasked
        # once proc stands (not in ECONOMY_BUILD_ROLES-only gate).
        from rl.map_catalog import (
            FORBIDDEN_BUILD_ROLES_LAND, NAVAL_TRAIN_ROLES, allows_naval,
        )
        map_name = str(getattr(getattr(obs, 'map_info', None), 'map_name', '') or '')
        naval_ok = allows_naval(map_name)
        if not naval_ok:
            for slot, role in enumerate(self.build_items):
                bslot = n_train + slot
                if bslot >= n_vocab:
                    break
                if role in FORBIDDEN_BUILD_ROLES_LAND:
                    self.build_slot_mask[bslot] = False
                    self.item_mask[bslot] = False
            for slot, role in enumerate(self.train_items):
                if slot >= n_vocab:
                    break
                if role in NAVAL_TRAIN_ROLES:
                    self.train_slot_mask[slot] = False
                    self.item_mask[slot] = False
            if not bool(self.build_slot_mask.any()):
                m[TYPE_TO_IDX['build']] = False
            if not bool(self.train_slot_mask.any()):
                m[TYPE_TO_IDX['train']] = False
            self.type_mask = torch.from_numpy(m)

        # Hard constraint: no combat TRAIN until proc + harvester.
        # Without a standing proc, also freeze ALL train (no harv spam) and
        # BUILD of barracks/weap/etc. Power + refinery stay legal so we
        # cannot deadlock. PLACE of a queued proc stays legal.
        if not owns_proc(obs):
            for slot, role in enumerate(self.train_items):
                if slot >= n_vocab:
                    break
                self.train_slot_mask[slot] = False
                self.item_mask[slot] = False
            for slot, role in enumerate(self.build_items):
                bslot = n_train + slot
                if bslot >= n_vocab:
                    break
                if role not in ECONOMY_BUILD_ROLES:
                    self.build_slot_mask[bslot] = False
                    self.item_mask[bslot] = False
            m[TYPE_TO_IDX["train"]] = False
            if not bool(self.build_slot_mask.any()):
                m[TYPE_TO_IDX["build"]] = False
            for name in COMBAT_MOVE_TYPES:
                m[TYPE_TO_IDX[name]] = False
            self.type_mask = torch.from_numpy(m)
        elif not economy_ready_for_combat(obs):
            for slot, role in enumerate(self.train_items):
                if slot >= n_vocab:
                    break
                if role in COMBAT_TRAIN_ROLES:
                    self.train_slot_mask[slot] = False
                    self.item_mask[slot] = False
            if not bool(self.train_slot_mask.any()):
                m[TYPE_TO_IDX["train"]] = False
                self.type_mask = torch.from_numpy(m)
        # Low-power: e1/pbox cannot steal the $300 plant. PLACE stays on.
        if power_in_deficit(obs) and owns_proc(obs):
            for slot, role in enumerate(self.train_items):
                if slot >= n_vocab:
                    break
                if role in COMBAT_TRAIN_ROLES:
                    self.train_slot_mask[slot] = False
                    self.item_mask[slot] = False
            if not bool(self.train_slot_mask.any()):
                m[TYPE_TO_IDX["train"]] = False
            freeze_build = needs_power_build(obs) or power_in_flight(obs)
            if freeze_build:
                for slot, role in enumerate(self.build_items):
                    bslot = n_train + slot
                    if bslot >= n_vocab:
                        break
                    if role != "power":
                        self.build_slot_mask[bslot] = False
                        self.item_mask[bslot] = False
                if not bool(self.build_slot_mask.any()):
                    m[TYPE_TO_IDX["build"]] = False
            self.type_mask = torch.from_numpy(m)
        # Pack-12: group push with a real army anywhere (Run 44 field remate).
        if n_combat_total(obs) < PACK_ARMY:
            m[TYPE_TO_IDX["army_attack_move"]] = False
        # v2 role-group macros: >=2 matching (harvesters >=1).
        if n_infantry_total(obs) < 2:
            m[TYPE_TO_IDX["infantry_attack_move"]] = False
        if n_vehicle_combat_total(obs) < 2:
            m[TYPE_TO_IDX["vehicle_attack_move"]] = False
        if n_harvester_total(obs) < 1:
            m[TYPE_TO_IDX["harvesters_move"]] = False
        # P0: naval/air macros masked until at least one matching unit exists.
        if n_naval_total(obs) < 1:
            m[TYPE_TO_IDX["naval_attack_move"]] = False
        if n_air_total(obs) < 1:
            m[TYPE_TO_IDX["air_attack_move"]] = False
        # P1: building-slot actions masked until at least one building exists.
        if len(self.building_ids) < 1:
            for name in BUILDING_SLOT_TYPES:
                m[TYPE_TO_IDX[name]] = False
        # P2 micro: guard needs escort + distinct target (prefer harv/MCV/building).
        if not can_issue_guard(obs):
            m[TYPE_TO_IDX["guard"]] = False
            m[TYPE_TO_IDX["army_guard"]] = False
        # Group stop/stance: at least one combat unit.
        if n_combat_total(obs) < 1:
            m[TYPE_TO_IDX["army_stop"]] = False
            m[TYPE_TO_IDX["army_set_stance"]] = False
        # Focus fire: ATTACK only when a visible enemy actor exists.
        if not _any_visible_enemy(obs):
            m[TYPE_TO_IDX["attack"]] = False
        # Aliados: APC load/unload. Coarse type mask is refined here.
        if not can_issue_enter_transport(obs):
            m[TYPE_TO_IDX["enter_transport"]] = False
        if not can_issue_unload(obs):
            m[TYPE_TO_IDX["unload"]] = False
        if n_combat_total(obs) < 1:
            m[TYPE_TO_IDX["patrol"]] = False
        if not _pick_support_power(obs):
            m[TYPE_TO_IDX["support_power"]] = False
        self.type_mask = torch.from_numpy(m)


def index_to_command(obs, chosen_type: int, unit_slot: int, cell_flat: int,
                     item_slot: int, aidx: ActionIndex) -> OpenRAAction:
    """Convierte la salida cruda de la red en un comando válido y SEGURO."""
    action, _ = index_to_command_effective(
        obs, chosen_type, unit_slot, cell_flat, item_slot, aidx)
    return action


def _item_slot_of(item_type: str, aidx) -> int:
    """Concreto (proc/gun) o rol -> slot en aidx.items (roles).

    PLACE/cancel mutan item_type al InternalName; TRAIN/BUILD ya traen rol.
    Mismo orden que SIL en imitation.py: rol primero, concreto si ya está.
    Fallback a slot 0 se loguea (P1.3) y cuenta en action_hist via rollout.
    """
    global _ITEM_SLOT_FALLBACK_COUNT
    if not item_type or not getattr(aidx, "items", None):
        items = getattr(aidx, "items", None) or []
        _ITEM_SLOT_FALLBACK_COUNT += 1
        _log.warning(
            "_item_slot_of fallback: item_type=%r role=%r len(items)=%d slot0",
            item_type, None, len(items))
        return 0
    items = aidx.items
    if item_type in items:
        return int(items.index(item_type))
    from rl.roles import role_of
    role = role_of(item_type)
    if role in items:
        return int(items.index(role))
    _ITEM_SLOT_FALLBACK_COUNT += 1
    _log.warning(
        "_item_slot_of fallback: item_type=%r role=%r len(items)=%d slot0",
        item_type, role, len(items))
    return 0


def index_to_command_effective(obs, chosen_type: int, unit_slot: int,
                               cell_flat: int, item_slot: int,
                               aidx: ActionIndex,
                               heuristic_p: float = 1.0):
    """Igual que index_to_command pero TAMBIÉN devuelve los índices EFECTIVOS.

    heuristic_p (P1.4): probabilidad de aplicar guard_army_push_cell en
    army/infantry/vehicle_attack_move. remap_move_cell y
    stage_army_attack_cell siempre corren (safety: orilla de lago / choke).
    Phase A / default = 1.0; anneal post Phase B solo apaga guard.

    Las correcciones de seguridad mutan la acción muestreada (ej. 'train'
    con ítem de edificio -> primer entrenable). Guardar el log_prob de la
    acción MUESTREADA cuando se ejecutó otra viola el teorema del gradiente
    de política (revisión externa 2026-08-24): con los índices efectivos el
    rollout puede recalcular log π(a_ejecutada|s) y atribuir el crédito a lo
    que realmente ocurrió.
    """
    idx_to_type = {v: k for k, v in TYPE_TO_IDX.items()}
    t_name = idx_to_type.get(chosen_type, "no_op")
    if t_name not in ENABLED_TYPES:
        t_name = "no_op"

    actor_id = aidx.unit_ids[unit_slot] if unit_slot < len(aidx.unit_ids) else 0

    cx, cy = 0, 0
    if cell_flat < aidx.h * aidx.w:
        cy, cx = divmod(int(cell_flat), aidx.w)

    item_type = aidx.items[item_slot] if item_slot < len(aidx.items) else ""
    train_set, build_set = set(aidx.train_items), set(aidx.build_items)

    # --- Correcciones de seguridad (determinísticas, mantienen índices) ---
    if t_name == "train":
        concrete = str(aidx.rol_a_concreto.get(item_type, item_type) or "").lower()
        if item_type not in train_set or concrete in BUILDING_ITEM_TYPES:
            # Slot de edificio/muro (sbag/brik) no es unidad. Primer entrenable.
            if aidx.train_items:
                item_type = aidx.train_items[0]
            else:
                t_name = "no_op"
        # Defensa: sin proc no se entrena NADA (ni harv). Con proc pero sin
        # harv, los rifles se tiran a no_op (el support se encarga del harv).
        if t_name == "train" and not owns_proc(obs):
            t_name = "no_op"
        elif t_name == "train" and not economy_ready_for_combat(obs):
            if item_type in COMBAT_TRAIN_ROLES:
                t_name = "no_op"
    elif t_name == "build":
        if item_type not in build_set:
            if aidx.build_items:
                item_type = aidx.build_items[0]
            else:
                t_name = "no_op"
        if t_name == "build" and not owns_proc(obs) and item_type not in ECONOMY_BUILD_ROLES:
            if "refinery" in build_set:
                item_type = "refinery"
            elif "power" in build_set:
                item_type = "power"
            else:
                t_name = "no_op"
    elif t_name == "place_building":
        from rl.roles import concretos_de
        concrete = str(aidx.rol_a_concreto.get(item_type, item_type) or "").lower()
        also = concretos_de(item_type) if item_type else ()
        pending = pending_place_item(obs, prefer=concrete, also=also)
        if not pending:
            t_name = "no_op"
        else:
            item_type = pending
    elif t_name == "cancel_production":
        if not obs.production:
            t_name = "no_op"
        else:
            item_type = obs.production[0].item

    if t_name in UNIT_ACTION_TYPES and actor_id == 0:
        t_name = "no_op"  # acción de unidad sin unidad válida

    # F1 (auditoría): attack SIN enemigo resolvible se degrada a attack_move
    # AQUÍ, antes de computar los índices efectivos — así el tipo efectivo
    # refleja la acción realmente ejecutada (antes quedaba "attack" y el
    # buffer atribuía el crédito al tipo equivocado).
    # P2: use focus resolver (same visibility set; prefers wounded).
    if t_name == "attack" and _focus_enemy_at_cell(obs, cx, cy) is None:
        t_name = "attack_move"

    if t_name in COMBAT_MOVE_TYPES and not owns_proc(obs):
        t_name = "no_op"
    if t_name == "army_attack_move" and n_combat_total(obs) < PACK_ARMY:
        t_name = "no_op"
    if t_name == "infantry_attack_move" and n_infantry_total(obs) < 2:
        t_name = "no_op"
    if t_name == "vehicle_attack_move" and n_vehicle_combat_total(obs) < 2:
        t_name = "no_op"
    if t_name == "harvesters_move" and n_harvester_total(obs) < 1:
        t_name = "no_op"
    if t_name == "naval_attack_move" and n_naval_total(obs) < 1:
        t_name = "no_op"
    if t_name == "air_attack_move" and n_air_total(obs) < 1:
        t_name = "no_op"
    if t_name in BUILDING_SLOT_TYPES and not aidx.building_ids:
        t_name = "no_op"
    # P2 micro gates
    if t_name in ("guard", "army_guard"):
        if not can_issue_guard(obs):
            t_name = "no_op"
    if t_name in ("army_stop", "army_set_stance") and n_combat_total(obs) < 1:
        t_name = "no_op"
    if t_name == "enter_transport" and not can_issue_enter_transport(obs):
        t_name = "no_op"
    if t_name == "unload" and not can_issue_unload(obs):
        t_name = "no_op"
    if t_name == "patrol" and n_combat_total(obs) < 1:
        t_name = "no_op"
    if t_name == "support_power" and not _pick_support_power(obs):
        t_name = "no_op"

    # Per-harvester cell path (P0): move/attack_move/attack on a selected
    # harvester stays MOVE to (cx,cy) for THAT actor_id — do not rewrite to
    # harvest (which used to drop the cell and swap to _any_harvester).
    if t_name in ("move", "attack_move", "attack") and _is_harvester(obs, actor_id):
        t_name = "move"

    if t_name == "train" and not owns_proc(obs):
        t_name = "no_op"
    elif t_name == "train" and not economy_ready_for_combat(obs):
        if item_type in COMBAT_TRAIN_ROLES:
            t_name = "no_op"
    t_name, item_type = _apply_power_priority(obs, t_name, item_type, aidx)

    # Índices EFECTIVOS tras las correcciones (para el log_prob honesto).
    # Se computan ANTES de armar el comando, reflejando cada mutación.
    eff_type = TYPE_TO_IDX.get(t_name, chosen_type)
    eff_unit_slot = unit_slot if 0 <= unit_slot < len(aidx.unit_ids) else 0
    eff_item_slot = (item_slot if 0 <= item_slot < len(aidx.items)
                     else 0)
    # Remap illegal move cells BEFORE computing the issued cell_flat.
    # TRAIN/BUILD/PLACE ignore this (place keeps the sampled cell).
    if t_name in MOVE_CELL_TYPES:
        cx, cy = remap_move_cell(obs, aidx, cx, cy, actor_id)
        if t_name in ("army_attack_move", "infantry_attack_move",
                      "vehicle_attack_move"):
            # Lake/choke staging is always-on safety (like remap). Binary
            # heuristic_p=0 left armies stuck on the west shore.
            cx, cy = stage_army_attack_cell(obs, aidx, cx, cy)
            # P1.4: only fog-east / no-west-yank guard anneals to 0.
            hp = 1.0 if heuristic_p is None else float(heuristic_p)
            if hp >= 1.0 or (hp > 0.0 and random.random() < hp):
                cx, cy = guard_army_push_cell(obs, aidx, cx, cy)
    eff_cell_flat = int(cy) * aidx.w + int(cx)
    if t_name in ("train", "build", "place_building", "cancel_production"):
        # PLACE/cancel dejan item_type concreto (proc/gun/tent); aidx.items
        # son roles. Sin role_of el slot queda el muestreado (auditoría 1.4).
        eff_item_slot = _item_slot_of(item_type, aidx)
    if t_name == "harvest":
        # Prefer unit-head selection when it is a harvester; else any harv.
        if _is_harvester(obs, actor_id) and actor_id in aidx.unit_ids:
            eff_unit_slot = aidx.unit_ids.index(actor_id)
        else:
            h_id = _any_harvester(obs)
            if h_id and h_id in aidx.unit_ids:
                eff_unit_slot = aidx.unit_ids.index(h_id)
    elif t_name == "deploy":
        m_id = _any_mcv(obs)
        if m_id and m_id in aidx.unit_ids:
            eff_unit_slot = aidx.unit_ids.index(m_id)
    elif t_name in BUILDING_SLOT_TYPES and aidx.building_ids:
        # unit_slot indexes building_ids (act masks via building_valid).
        if 0 <= unit_slot < len(aidx.building_ids):
            eff_unit_slot = unit_slot
        else:
            eff_unit_slot = 0
        actor_id = aidx.building_ids[eff_unit_slot]
        if actor_id <= 0:
            t_name = "no_op"
            eff_type = TYPE_TO_IDX.get("no_op", chosen_type)

    cmd = None
    group_cmds = None  # v2: multi CommandModel for role-group macros
    t = ActionType(t_name)
    if t == ActionType.NO_OP:
        cmd = CommandModel(action=t)
    elif t in (ActionType.MOVE, ActionType.ATTACK_MOVE):
        cmd = CommandModel(action=t, actor_id=actor_id,
                           target_x=cx, target_y=cy)
    elif t == ActionType.ARMY_ATTACK_MOVE:
        # Fase 2: sin actor_id — el C# itera TODAS las unidades de combate
        # propias y les emite AttackMove hacia la celda.
        cmd = CommandModel(action=t, target_x=cx, target_y=cy)
    elif t in (ActionType.INFANTRY_ATTACK_MOVE, ActionType.VEHICLE_ATTACK_MOVE,
               ActionType.HARVESTERS_MOVE, ActionType.NAVAL_ATTACK_MOVE,
               ActionType.AIR_ATTACK_MOVE):
        # v2/P0: N per-unit cmds (engine has no typed group macros besides army).
        gkey = {
            ActionType.INFANTRY_ATTACK_MOVE: "infantry",
            ActionType.VEHICLE_ATTACK_MOVE: "vehicle",
            ActionType.HARVESTERS_MOVE: "harvesters",
            ActionType.NAVAL_ATTACK_MOVE: "naval",
            ActionType.AIR_ATTACK_MOVE: "air",
        }[t]
        ids = group_actor_ids(obs, gkey)
        atk = (ActionType.MOVE if t == ActionType.HARVESTERS_MOVE
               else ActionType.ATTACK_MOVE)
        group_cmds = [
            CommandModel(action=atk, actor_id=aid, target_x=cx, target_y=cy)
            for aid in ids
        ]
        cmd = group_cmds[0] if group_cmds else CommandModel(action=ActionType.NO_OP)
    elif t == ActionType.ATTACK:
        # P2 focus fire: bind concrete enemy actor (wounded/nearest to cell).
        target = _focus_enemy_at_cell(obs, cx, cy)
        if target is None:
            cmd = CommandModel(action=ActionType.ATTACK_MOVE,
                               actor_id=actor_id, target_x=cx, target_y=cy)
        else:
            cmd = CommandModel(action=t, actor_id=actor_id,
                               target_actor_id=target,
                               target_x=cx, target_y=cy)
    elif t == ActionType.GUARD:
        # Escort unit (unit head) guards preferred own harv/MCV/building.
        if not _is_escort_unit(obs, actor_id):
            escort = next((uid for uid in aidx.unit_ids
                           if _is_escort_unit(obs, uid)), None)
            if escort is None:
                actor_id = 0
            else:
                actor_id = int(escort)
                eff_unit_slot = aidx.unit_ids.index(actor_id)
        tgt = (resolve_guard_target(obs, cx, cy, exclude_id=actor_id)
               if actor_id > 0 else None)
        if actor_id <= 0 or tgt is None:
            cmd = CommandModel(action=ActionType.NO_OP)
        else:
            cmd = CommandModel(action=ActionType.GUARD, actor_id=actor_id,
                               target_actor_id=int(tgt))
    elif t == ActionType.ARMY_GUARD:
        tgt = resolve_guard_target(obs, cx, cy)
        ids = group_actor_ids(obs, "army")
        if tgt is None or not ids:
            cmd = CommandModel(action=ActionType.NO_OP)
        else:
            group_cmds = [
                CommandModel(action=ActionType.GUARD, actor_id=aid,
                             target_actor_id=int(tgt))
                for aid in ids if aid != int(tgt)
            ]
            cmd = group_cmds[0] if group_cmds else CommandModel(
                action=ActionType.NO_OP)
    elif t == ActionType.ARMY_STOP:
        ids = group_actor_ids(obs, "army")
        group_cmds = [
            CommandModel(action=ActionType.STOP, actor_id=aid) for aid in ids
        ]
        cmd = group_cmds[0] if group_cmds else CommandModel(action=ActionType.NO_OP)
    elif t == ActionType.ARMY_SET_STANCE:
        ids = group_actor_ids(obs, "army")
        stance = _stance_from_cell(cx, cy)
        group_cmds = [
            CommandModel(action=ActionType.SET_STANCE, actor_id=aid,
                         target_x=stance)
            for aid in ids
        ]
        cmd = group_cmds[0] if group_cmds else CommandModel(action=ActionType.NO_OP)
    elif t == ActionType.STOP:
        cmd = CommandModel(action=t, actor_id=actor_id)
    elif t == ActionType.SET_STANCE:
        # Proto: target_x encodes stance; default AttackAnything (land-safe).
        stance = _stance_from_cell(cx, cy) if (cx or cy) else _STANCE_ATTACK_ANYTHING
        cmd = CommandModel(action=t, actor_id=actor_id, target_x=stance)
    elif t == ActionType.HARVEST:
        # Selected harvester + cell when possible (CommandModel/HARVEST accepts
        # target_x/y; MCP harvest uses the same fields).
        hid = actor_id if _is_harvester(obs, actor_id) else (
            _any_harvester(obs) or actor_id)
        cmd = CommandModel(action=t, actor_id=hid, target_x=cx, target_y=cy)
    elif t == ActionType.DEPLOY:
        subj = _unit_by_id(obs, actor_id)
        if subj is not None and _is_chrono_tank_unit(subj):
            cmd = CommandModel(action=t, actor_id=int(actor_id),
                               target_x=cx, target_y=cy)
        elif _any_mcv(obs):
            cmd = CommandModel(action=t, actor_id=_any_mcv(obs) or actor_id)
        elif _any_chrono_tank(obs):
            cid = int(_any_chrono_tank(obs))
            if cid in aidx.unit_ids:
                eff_unit_slot = aidx.unit_ids.index(cid)
            cmd = CommandModel(action=t, actor_id=cid, target_x=cx, target_y=cy)
        else:
            cmd = CommandModel(action=t, actor_id=_any_mcv(obs) or actor_id)
    elif t == ActionType.TRAIN:
        cmd = CommandModel(action=t,
                           item_type=aidx.rol_a_concreto.get(item_type,
                                                             item_type))
    elif t == ActionType.BUILD:
        cmd = CommandModel(action=t,
                           item_type=aidx.rol_a_concreto.get(item_type,
                                                             item_type))
    elif t == ActionType.PLACE_BUILDING:
        cmd = CommandModel(action=t, item_type=item_type,
                           target_x=cx, target_y=cy)
    elif t == ActionType.CANCEL_PRODUCTION:
        cmd = CommandModel(action=t, item_type=item_type)
    elif t in (ActionType.SELL, ActionType.REPAIR, ActionType.POWER_DOWN,
               ActionType.SET_PRIMARY):
        # P1: actor_id = selected building (no cell/item fields needed).
        if actor_id <= 0:
            cmd = CommandModel(action=ActionType.NO_OP)
        else:
            cmd = CommandModel(action=t, actor_id=actor_id)
    elif t == ActionType.SET_RALLY_POINT:
        if actor_id <= 0:
            cmd = CommandModel(action=ActionType.NO_OP)
        else:
            cmd = CommandModel(action=t, actor_id=actor_id,
                               target_x=cx, target_y=cy)
    elif t == ActionType.ENTER_TRANSPORT:
        subj = _unit_by_id(obs, actor_id)
        if subj is None or not _is_infantry_unit(subj):
            actor_id = _first_infantry_id(obs)
            if actor_id and actor_id in aidx.unit_ids:
                eff_unit_slot = aidx.unit_ids.index(actor_id)
        tgt = _nearest_own_transport(obs, actor_id) if actor_id else None
        if not actor_id or tgt is None:
            cmd = CommandModel(action=ActionType.NO_OP)
        else:
            cmd = CommandModel(action=ActionType.ENTER_TRANSPORT,
                               actor_id=int(actor_id),
                               target_actor_id=int(tgt))
    elif t == ActionType.UNLOAD:
        subj = _unit_by_id(obs, actor_id)
        if (subj is None or not _is_transport_unit(subj)
                or _passenger_count(subj) <= 0):
            loaded = _any_loaded_transport(obs)
            if loaded:
                actor_id = int(loaded)
                if actor_id in aidx.unit_ids:
                    eff_unit_slot = aidx.unit_ids.index(actor_id)
            else:
                actor_id = 0
        if not actor_id:
            cmd = CommandModel(action=ActionType.NO_OP)
        else:
            cmd = CommandModel(action=ActionType.UNLOAD, actor_id=int(actor_id))
    elif t == ActionType.PATROL:
        if actor_id <= 0:
            ids = group_actor_ids(obs, "army", limit=1)
            actor_id = ids[0] if ids else 0
            if actor_id and actor_id in aidx.unit_ids:
                eff_unit_slot = aidx.unit_ids.index(actor_id)
        if actor_id <= 0:
            cmd = CommandModel(action=ActionType.NO_OP)
        else:
            cmd = CommandModel(action=ActionType.PATROL, actor_id=int(actor_id),
                               target_x=cx, target_y=cy)
    elif t == ActionType.SUPPORT_POWER:
        key = _pick_support_power(obs)
        if not key:
            cmd = CommandModel(action=ActionType.NO_OP)
        else:
            cmd = CommandModel(action=ActionType.SUPPORT_POWER,
                               actor_id=int(actor_id or 0),
                               target_x=cx, target_y=cy, item_type=key)
    else:
        cmd = CommandModel(action=ActionType.NO_OP)
    if group_cmds:
        out_cmds = group_cmds
    else:
        out_cmds = [cmd] if cmd is not None else []
    return OpenRAAction(commands=out_cmds), (eff_type, eff_unit_slot,
                                             eff_item_slot, eff_cell_flat)


def _nearest_enemy_at_cell(obs, cx: int, cy: int):
    """ID del enemigo visible más cercano a la celda (legacy; prefer focus)."""
    return _focus_enemy_at_cell(obs, cx, cy)


def _pending_building_type(obs):
    """Tipo del edificio terminado esperando colocación (Building o Defense)."""
    return pending_place_item(obs)


def _is_harvester(obs, actor_id) -> bool:
    try:
        aid = int(actor_id or 0)
    except (TypeError, ValueError):
        return False
    if aid <= 0:
        return False
    for u in getattr(obs, "units", None) or []:
        try:
            if int(getattr(u, "actor_id", 0) or 0) == aid:
                return "harv" in str(getattr(u, "type", "")).lower()
        except (TypeError, ValueError):
            continue
    return False


def _any_harvester(obs):
    for u in obs.units:
        if "harv" in u.type.lower():
            return u.actor_id
    return None


def _any_mcv(obs):
    for u in obs.units:
        if "mcv" in u.type.lower():
            return u.actor_id
    return None
