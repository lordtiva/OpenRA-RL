"""Decodificación de observaciones OpenRA-RL a tensores para la red.

Fuente de verdad de los canales espaciales (ObservationSerializer.cs,
OpenRA.Mods.Common/Traits/Player/, submodule OpenRA @ 9b271c1):
    Ch 0: terrain type index     Ch 5: own building (0/1)
    Ch 1: height                 Ch 6: own unit density (conteo)
    Ch 2: resource density       Ch 7: enemy building (0/1)
    Ch 3: passability (0/1)      Ch 8: enemy unit density (conteo)
    Ch 4: fog of war (0/0.5/1)

Formato wire: base64(float32), row-major H*W*C -> channels-last.
"""

import base64
import math

import numpy as np

SPATIAL_CHANNELS = 9
# Colas que terminan en PLACE (no Infantry/Vehicle). Defense = pbox/gun/ftur.
PLACE_QUEUE_TYPES = frozenset({"building", "defense"})

# Normalización por canal (escala aproximada a [0,1])
CHANNEL_SCALE = {
    0: 1.0 / 16.0,  # índices de terreno (decenas)
    1: 1.0 / 10.0,  # altura en celdas
    2: 1.0,         # densidad recurso ya viene ~[0,1]
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: 1.0 / 8.0,   # densidad de unidades: rara vez >8 apiladas visibles
    7: 1.0,
    8: 1.0 / 8.0,
}

# Beacon del objetivo (curriculum Escenario A): revela en Ch7/Ch8 la posición
# de la base enemiga SIN revelar la niebla del resto. Pedagogía: el A enseña
# combate/destrucción de base, no navegación a ciegas. Clave = map_name.
BEACON_BY_MAP = {
    # amin160_allies: bot base powr@98,8 + barr@98,12 + MCV auto ~ (96,11)
    "fase2_amin160_allies.oramap": (96, 10),
    "fase2_amin160_short.oramap": (96, 10),
    # a_short / amin160: misma base bot (~95,11)
    "fase2_a_short.oramap": (95, 11),
    "fase2_a_minus_short.oramap": (95, 11),
    # Run2/Runs generales (sin sufijo _short) — mismo spawn
    "fase2_a.oramap": (95, 11),
    "fase2_amin160.oramap": (96, 10),
    "fase2_a_minus.oramap": (95, 11),
    # ObservationSerializer writes world.Map.Title, not the filename.
    # All fase2_*.oramap currently ship Title: Singles (live_games 2026-08-30).
    "Singles": (95, 11),
    "A-short (PPO train)": (95, 11),
}


def resolve_beacon(obs) -> tuple[int, int] | None:
    """Beacon for this obs. Matches filename, Title ('Singles') or stem."""
    info = getattr(obs, "map_info", None)
    name = str(getattr(info, "map_name", "") or "")
    if not name:
        return None
    hit = BEACON_BY_MAP.get(name)
    if hit is not None:
        return int(hit[0]), int(hit[1])
    low = name.lower().replace("\\", "/").rsplit("/", 1)[-1]
    for k, v in BEACON_BY_MAP.items():
        kl = k.lower()
        stem = kl[:-7] if kl.endswith(".oramap") else kl
        if low == kl or low == stem or stem in low:
            return int(v[0]), int(v[1])
    return None


def decode_spatial(spatial_b64: str, height: int, width: int, channels: int,
                   beacon=None):
    """base64 -> array float32 (C,H,W) normalizado. Devuelve None si vacío.
    beacon=(cx, cy): posicion del target enemigo a revelar en Ch7/Ch8 sin
    tocar la niebla (Ch4). Pedagogia del Escenario A (curriculum): aprender
    combate/destruccion de base, no navegacion a ciegas. Escenario A -> el
    gradiente VE la base rival y puede muestrear army_attack_move hacia ella.
    """
    if not spatial_b64:
        return None
    raw = base64.b64decode(spatial_b64)
    arr = np.frombuffer(raw, dtype=np.float32).copy()
    expected = height * width * channels
    if arr.size != expected:
        raise ValueError(f"spatial_map: esperados {expected} floats, llegaron {arr.size}")
    # channels-last (H,W,C) -> channels-first (C,H,W)
    arr = arr.reshape(height, width, channels).transpose(2, 0, 1)
    for ch, scale in CHANNEL_SCALE.items():
        arr[ch] *= scale
    if beacon is not None:
        apply_beacon(arr, beacon[0], beacon[1], height, width)
    return np.ascontiguousarray(arr, dtype=np.float32)


def apply_beacon(spatial, cx: int, cy: int, height: int, width: int,
                 radius: int = 3):
    """Marca la celda enemiga en Ch7 (building) y Ch8 (unit) en un radio.

    Solo enciende la señal de 'hay objetivo aqui' en un radio pequeno; la
    niebla (Ch4) y el resto del mapa quedan intactos -> no se rompe la
    percepcion parcial. Es el 'radar' del spawn rival que propuso el operador.
    """
    ch7 = 7
    ch8 = 8
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < width and 0 <= y < height:
                spatial[ch7, y, x] = max(spatial[ch7, y, x], 0.85)
                spatial[ch8, y, x] = max(spatial[ch8, y, x], 0.85)
    return spatial


SCALAR_DIM = 21


def scalar_features(obs) -> np.ndarray:
    """Vector de escalares económicos/militares normalizado (~[0,1])."""
    eco = obs.economy
    mil = obs.military
    capacity = max(eco.resource_capacity, 1)
    cash_norm = min(eco.cash / 10000.0, 1.0)
    ore_norm = min(eco.ore / capacity, 1.0)
    silo_full = eco.ore / capacity  # señal separada: silos llenos => construir más
    power_in = max(eco.power_provided, 1)
    power_ratio = min(eco.power_drained / power_in, 2.0) / 2.0
    n_units = len(obs.units)
    n_buildings = len(obs.buildings)
    n_enemies = len(obs.visible_enemies)
    n_enemy_bldgs = len(obs.visible_enemy_buildings)
    prod_active = sum(1 for p in obs.production if not p.paused)
    prod_progress = (
        sum(p.progress for p in obs.production) / len(obs.production)
        if obs.production else 0.0
    )

    # ---- Escalares estratégicos (3 lecciones del bot scripted) ----
    # 1. Refinería: la red debe saber si puede pagarla y si ya la tiene.
    has_refinery = 1.0 if any(b.type == "proc" for b in obs.buildings) else 0.0
    can_afford_proc = 1.0 if (eco.cash >= 2000 and has_refinery == 0.0) else 0.0

    # 2. Guarnición: nº de unidades de combate ≈≤10 celdas de un edificio propio.
    garrison_count = 0
    if obs.buildings and obs.units:
        bpts = [(b.cell_x, b.cell_y) for b in obs.buildings]
        for u in obs.units:
            if getattr(u, "can_attack", True) and \
                    "harv" not in getattr(u, "type", "").lower():
                dmin = min(abs(u.cell_x - bx) + abs(u.cell_y - by)
                           for bx, by in bpts)
                if dmin <= 10:
                    garrison_count += 1
    garrison_ratio = min(garrison_count / 2.0, 1.0)

    # 3. Full-stack Run3: conciencia Lanchester + tier tecnológico (congelado acá)
    own_val = float(getattr(mil, "army_value", 0) or 0)
    ene_est = n_enemies * 400 + n_enemy_bldgs * 1000
    military_ratio = min(own_val / max(ene_est, 300), 3.0) / 3.0
    btypes = {getattr(b, "type", "") for b in obs.buildings}
    if "stek" in btypes:
        tier = 4
    elif "atek" in btypes:
        tier = 3
    elif "weap" in btypes:
        tier = 2
    elif "barr" in btypes or "tent" in btypes:
        tier = 1
    else:
        tier = 0
    tech_tier = tier / 4.0

    return np.array([
        cash_norm,
        ore_norm,
        silo_full,
        power_ratio,
        1.0 if eco.power_drained > eco.power_provided else 0.0,  # low power
        min(eco.harvester_count / 10.0, 1.0),
        min(n_units / 60.0, 1.0),
        min(n_buildings / 25.0, 1.0),
        min(n_enemies / 40.0, 1.0),
        min(n_enemy_bldgs / 15.0, 1.0),
        min(mil.units_killed / 50.0, 1.0),
        min(mil.units_lost / 50.0, 1.0),
        min(mil.army_value / 20000.0, 1.0),
        min(prod_active / 4.0, 1.0),
        prod_progress,
        min(obs.tick / 18000.0, 1.0),  # ~12 min de juego a 25 tps
        has_refinery,
        can_afford_proc,
        garrison_ratio,
        military_ratio,
        tech_tier,
    ], dtype=np.float32)


MAX_UNITS = 96  # Capa 2c-A: techo de slots (pesos del xf no crecen con n)
UNIT_FEAT_DIM = 10
# Mismo radio que el dest-defend viejo: combate que ve el raid entra al set
# aunque tenga actor_id alto (los e1 nuevos no cabían en oldest-48).
THREAT_RADIUS = 18


def _actor_id(u) -> int:
    try:
        return int(getattr(u, "actor_id", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _xy(u) -> tuple[int, int]:
    try:
        return int(getattr(u, "cell_x", 0) or 0), int(getattr(u, "cell_y", 0) or 0)
    except (TypeError, ValueError):
        return 0, 0


def _is_harv_or_mcv(u) -> bool:
    t = str(getattr(u, "type", "") or "").lower()
    return "harv" in t or "mcv" in t


def _is_combat(u) -> bool:
    if _is_harv_or_mcv(u):
        return False
    return bool(getattr(u, "can_attack", False))


def _manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _near_any(u, others, radius: int) -> bool:
    ux, uy = _xy(u)
    for o in others:
        if _manhattan((ux, uy), _xy(o)) <= radius:
            return True
    return False


def _base_under_attack(obs, radius: int = THREAT_RADIUS) -> bool:
    bldgs = list(getattr(obs, "buildings", None) or [])
    threats = list(getattr(obs, "visible_enemies", None) or []) + list(
        getattr(obs, "visible_enemy_buildings", None) or [])
    if not bldgs or not threats:
        return False
    return any(_near_any(t, bldgs, radius) for t in threats)


def select_unit_slots(obs, max_units: int = MAX_UNITS):
    """Hasta max_units propias: amenaza, resto combate, no-combate.

    Cubos (actor_id ascendente dentro de cada uno; el tensor final se
    reordena por id para que el slot no salte entre bloques):
      1. combate a ≤THREAT_RADIUS de enemigo visible / edificio enemigo,
         o combate a ≤THREAT_RADIUS de edificio propio si la base está
         bajo ataque (raid en casa)
      2. resto de combate
      3. harv / mcv / otros
    """
    units = list(getattr(obs, "units", None) or [])
    if not units:
        return []
    threats = list(getattr(obs, "visible_enemies", None) or []) + list(
        getattr(obs, "visible_enemy_buildings", None) or [])
    bldgs = list(getattr(obs, "buildings", None) or [])
    raid = _base_under_attack(obs)
    c1, c2, c3 = [], [], []
    for u in units:
        if _is_combat(u):
            near_ene = bool(threats) and _near_any(u, threats, THREAT_RADIUS)
            near_home = raid and _near_any(u, bldgs, THREAT_RADIUS)
            if near_ene or near_home:
                c1.append(u)
            else:
                c2.append(u)
        else:
            c3.append(u)
    picked = []
    for cubo in (c1, c2, c3):
        cubo.sort(key=_actor_id)
        need = max_units - len(picked)
        if need <= 0:
            break
        picked.extend(cubo[:need])
    picked.sort(key=_actor_id)
    return picked


def _unit_feat(u) -> list:
    return [
        float(getattr(u, "hp_percent", 1.0) or 0.0),
        1.0 if getattr(u, "can_attack", False) else 0.0,
        1.0 if getattr(u, "is_idle", True) else 0.0,
        min(float(getattr(u, "speed", 0) or 0) / 100.0, 1.0),
        min(float(getattr(u, "attack_range", 0) or 0) / 6000.0, 1.0),
        float(getattr(u, "experience_level", 0) or 0) / 3.0,
        float(getattr(u, "stance", 0) or 0) / 3.0,
        float(getattr(u, "cell_x", 0) or 0) / 128.0,
        float(getattr(u, "cell_y", 0) or 0) / 128.0,
        float(getattr(u, "facing", 0) or 0) / 1023.0,
    ]


def unit_slots(obs):
    """Lista de hasta MAX_UNITS unidades propias como vectores de features.

    Selección = select_unit_slots (combat-first). Devuelve
    (features [N,UNIT_FEAT_DIM], válidos bool[N]) ordenadas por id.
    """
    units = select_unit_slots(obs)
    if not units:
        return (np.zeros((0, UNIT_FEAT_DIM), dtype=np.float32),
                np.zeros(0, dtype=bool))
    feats = np.array([_unit_feat(u) for u in units], dtype=np.float32)
    valid = np.ones(len(units), dtype=bool)
    return feats, valid


def ready_place_items(obs) -> list:
    """Ítems con progress≥1 en colas Building o Defense (se pueden PLACE)."""
    out = []
    for p in getattr(obs, "production", None) or []:
        qt = str(getattr(p, "queue_type", "") or "").lower()
        if qt not in PLACE_QUEUE_TYPES:
            continue
        if float(getattr(p, "progress", 0) or 0) < 1.0:
            continue
        item = str(getattr(p, "item", "") or "").lower()
        if item:
            out.append(item)
    return out


def pending_place_item(obs, prefer: str | None = None,
                       also=None) -> str | None:
    """Edificio listo para PLACE.

    1) `prefer` si está listo (concreto barato del rol)
    2) el primero de `also` que esté listo (otros concretos del mismo rol:
       encoló gun, el rol prefiere pbox)
    3) el primero listo de cualquier cola Building/Defense
    """
    ready = ready_place_items(obs)
    if not ready:
        return None
    if prefer:
        pref = str(prefer).lower()
        if pref in ready:
            return pref
    if also:
        also_set = {str(x).lower() for x in also if x}
        for it in ready:
            if it in also_set:
                return it
    return ready[0]


def type_to_index(type_str: str, vocab: dict) -> int:
    """Índice estable para un tipo de actor ('e1', '1tnk', ...)."""
    idx = vocab.get(type_str)
    if idx is None:
        idx = len(vocab)
        vocab[type_str] = idx
    return idx


def safe_log(p: float) -> float:
    """Guard log(0) para probabilidades exactamente cero."""
    return math.log(max(p, 1e-12))
