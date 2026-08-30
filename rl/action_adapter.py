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
    - Acciones que apuntan a EDIFICIOS (sell/repair/rally/power_down/
      set_primary) quedan fuera de la v0.1: requerirían una cabeza de slots
      de edificios; se agregará cuando el núcleo sea estable
"""

import numpy as np
import torch

from openra_env.models import ActionType, CommandModel, OpenRAAction
from rl.network import TYPE_TO_IDX, build_type_masks
from rl.obs_encoding import BEACON_BY_MAP, MAX_UNITS, resolve_beacon

# Tipos habilitados en v0.1 (el resto ni entra en la máscara)
ENABLED_TYPES = {
    "no_op", "move", "attack_move", "attack", "stop", "harvest",
    "set_stance", "deploy", "train", "build", "place_building",
    "cancel_production", "army_attack_move",
}

UNIT_ACTION_TYPES = {"move", "attack_move", "attack", "stop", "set_stance",
                     "harvest"}


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
        """Pre-asigna un id estable a cada rol del catálogo RA."""
        from rl.roles import ROLE_OF_ITEM
        roles = sorted(set(ROLE_OF_ITEM.values()))
        for rol in roles:
            self.id_of(rol)
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
    "barr", "tent", "kenn", "weap", "hpad", "dome", "fix", "atek", "stek",
    # defensa y navales
    "gun", "ftur", "tsla", "agun", "pbox", "hbox", "sam", "gap",
    "spen", "syrd",
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
}
ECONOMY_BUILD_ROLES = {"power", "refinery"}  # legal BUILD before proc exists
MOVE_CELL_TYPES = {"move", "attack_move", "attack", "army_attack_move"}
# Combat movement: masked until a refinery stands. Otherwise PPO
# reward-hacks army_attack_move / attack_move (the 201-309 collapse).
COMBAT_MOVE_TYPES = ("army_attack_move", "attack_move", "attack")


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


def remap_move_cell(obs, aidx, cx: int, cy: int, actor_id: int = 0):
    """If (cx,cy) is water/OOB/unpathable, retarget: visible enemy, else beacon, else near unit.

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


def _split_production(obs):
    """Separa available_production en (roles de entrenables, roles de
    construibles) + mapa rol->item concreto.

    TRAductor universal (agnóstico a facción): la cabeza de ítems indexa ROLES
    estables (rl.roles), no nombres internos que varían por facción. Cada
    rol->[items concretos disponibles]. Al armar el comando, el adapter elige
    el item concreto de la facción actual para el rol muestreado.
    """
    available = set(obs.available_production or [])
    buildables = available & BUILDING_ITEM_TYPES
    trainables = available - buildables
    # roles disponibles + su mejor item concreto (estable, más barato primero)
    def _roles(items):
        from rl.roles import role_of
        por_rol: dict[str, list[str]] = {}
        for it in sorted(items):
            por_rol.setdefault(role_of(it), []).append(it)
        return por_rol

    train_por_rol = _roles(trainables)
    build_por_rol = _roles(buildables)
    train_roles = sorted(train_por_rol)
    build_roles = sorted(build_por_rol)
    # rol -> ítem concreto preferido de la facción actual
    rol_a_concreto = {r: items[0] for r, items in train_por_rol.items()}
    for r, items in build_por_rol.items():
        rol_a_concreto[r] = items[0]
    return train_roles, build_roles, rol_a_concreto


class ActionIndex:
    """Todo lo que la red necesita para decidir sobre una observación."""

    __slots__ = ("type_mask", "unit_valid", "cell_mask", "item_indices",
                 "item_mask", "unit_ids", "items", "train_items",
                 "build_items", "rol_a_concreto", "h", "w",
                 "train_slot_mask", "build_slot_mask", "pass_grid")

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
        # train/build requieren ítems de su categoría
        self.train_items, self.build_items, self.rol_a_concreto = \
            _split_production(obs)
        if not self.train_items:
            m[TYPE_TO_IDX["train"]] = False
        if not self.build_items:
            m[TYPE_TO_IDX["build"]] = False
        self.type_mask = torch.from_numpy(m)

        # Cabeza 2: slots de unidades propias móviles
        units = sorted(obs.units, key=lambda u: u.actor_id)[:MAX_UNITS]
        self.unit_ids = [u.actor_id for u in units]
        self.unit_valid = torch.zeros(MAX_UNITS, dtype=torch.bool)
        for i in range(len(self.unit_ids)):
            self.unit_valid[i] = True

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


def index_to_command(obs, chosen_type: int, unit_slot: int, cell_flat: int,
                     item_slot: int, aidx: ActionIndex) -> OpenRAAction:
    """Convierte la salida cruda de la red en un comando válido y SEGURO."""
    action, _ = index_to_command_effective(
        obs, chosen_type, unit_slot, cell_flat, item_slot, aidx)
    return action


def index_to_command_effective(obs, chosen_type: int, unit_slot: int,
                               cell_flat: int, item_slot: int,
                               aidx: ActionIndex):
    """Igual que index_to_command pero TAMBIÉN devuelve los índices EFECTIVOS.

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
        if item_type not in train_set:
            # elegimos un ítem de edificio: usar el primer entrenable
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
        pending = _pending_building_type(obs)
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
    if t_name == "attack" and _nearest_enemy_at_cell(obs, cx, cy) is None:
        t_name = "attack_move"

    if t_name in COMBAT_MOVE_TYPES and not owns_proc(obs):
        t_name = "no_op"

    if t_name == "train" and not owns_proc(obs):
        t_name = "no_op"
    elif t_name == "train" and not economy_ready_for_combat(obs):
        if item_type in COMBAT_TRAIN_ROLES:
            t_name = "no_op"

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
    eff_cell_flat = int(cy) * aidx.w + int(cx)
    if t_name in ("train", "build", "place_building", "cancel_production"):
        # item_type pudo ser corregido arriba -> localizar su slot real
        if item_type in aidx.items:
            eff_item_slot = aidx.items.index(item_type)
    if t_name == "harvest":
        h_id = _any_harvester(obs)
        if h_id and h_id in aidx.unit_ids:
            eff_unit_slot = aidx.unit_ids.index(h_id)
    elif t_name == "deploy":
        m_id = _any_mcv(obs)
        if m_id and m_id in aidx.unit_ids:
            eff_unit_slot = aidx.unit_ids.index(m_id)

    cmd = None
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
    elif t == ActionType.ATTACK:
        # Con la degradación temprana F1, acá solo se llega CON enemigo
        # resolvible; el if queda como defensa en profundidad.
        target = _nearest_enemy_at_cell(obs, cx, cy)
        if target is None:
            cmd = CommandModel(action=ActionType.ATTACK_MOVE,
                               actor_id=actor_id, target_x=cx, target_y=cy)
        else:
            cmd = CommandModel(action=t, actor_id=actor_id,
                               target_actor_id=target,
                               target_x=cx, target_y=cy)
    elif t in (ActionType.STOP, ActionType.SET_STANCE):
        cmd = CommandModel(action=t, actor_id=actor_id)
    elif t == ActionType.HARVEST:
        cmd = CommandModel(action=t, actor_id=_any_harvester(obs) or actor_id)
    elif t == ActionType.DEPLOY:
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
    else:
        cmd = CommandModel(action=ActionType.NO_OP)
    return OpenRAAction(commands=[cmd]), (eff_type, eff_unit_slot,
                                          eff_item_slot, eff_cell_flat)


def _nearest_enemy_at_cell(obs, cx: int, cy: int):
    """ID del enemigo visible más cercano a la celda (para attack)."""
    best, best_d = None, float("inf")
    candidates = list(obs.visible_enemies) + list(obs.visible_enemy_buildings)
    for e in candidates:
        d = (e.cell_x - cx) ** 2 + (e.cell_y - cy) ** 2
        if d < best_d:
            best, best_d = e.actor_id, d
    return best


def _pending_building_type(obs):
    """Tipo del edificio terminado esperando colocación."""
    for p in obs.production:
        if p.queue_type == "Building" and p.progress >= 1.0:
            return p.item
    return None


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
