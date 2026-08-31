# -*- coding: utf-8 -*-
"""Traductor universal faccion -> roles funcionales.

Objetivo (concepto del operador): el agente debe ser AGNOSTICO A LA FACCION.
Cada faccion de RA tiene unidades equivalentes con nombre interno distinto
(ej. tanque ligero es 1TNK en aliados, pero el catalogo difiere por faction).
En vez de entrenar la cabeza de items sobre nombres concretos (1TNK, 2TNK,
STNK...) que cambian segun la faccion y rompen el resume, la red aprende ROLES
funcionales estables: 'harvester', 'infantry_basic', 'anti_armor', 'tank', ...

El server ya emite `available_production` filtrado por la faccion del jugador
(solo los items que puede construir). Este modulo mapea cada item concreto ->
ROL, y de vuelta rol -> items concretos de la faccion disponible.

Flujo:
  available_production (por faccion) --ROLE_OF_ITEM--> roles_disponibles
  rol_elegido por la red --concretos_de[rol] & available--> item concreto a ordenar

Asi la cabeza de items es estable entre facciones y episodios (indice = rol),
y el adapter decide el nombre interno concreto en el momento del comando.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# ROLE_OF_ITEM: catalogo RA completo (nombre interno del engine -> rol).
# Roles = equivalencia funcional ENTRE facciones. Si un item nuevo del mod
# aparece y no esta aqui, cae a 'misc' (defensivo/simple), nunca estrangula.
# --------------------------------------------------------------------------

# --- Economia / produccion de base ---
ROLE_HARVESTER = "harvester"

# --- Roles de UNIDADES entrenables ---
ROLE_INFANTRY_BASIC = "infantry_basic"        # e1 (rifle) — anti-infanteria ligera
ROLE_INFANTRY_ANTIINF = "infantry_antiinf"    # e2 granadero, e4/med flame, dog
ROLE_INFANTRY_ANTIARMOR = "infantry_antiarmor"  # e3 rocket, e6? no (combat) — cohetes
ROLE_ROCKET_TRUCK = "rocket_truck"            # v2rl lanzadora anti-estructura
ROLE_TANK_LIGHT = "tank_light"                # 1tnk, jeep
ROLE_TANK_MEDIUM = "tank_medium"              # 2tnk (aliados), 4tnk (soviet)
ROLE_TANK_HEAVY = "tank_heavy"                # 3tnk, qtnk, stnk, ctnk, ttnk, mgg
ROLE_ARTILLERY = "artillery"                  # arty
ROLE_APC_TRANSPORT = "transport"              # apc, truk, mnly, dtrk, ftrk
ROLE_SCOUT = "scout"                          # jeep (si no tanque)
ROLE_MCV = "mcv"                              # construction vehicle
ROLE_COMMANDO_SPY = "specialist"              # e6, spy, delphi, chan, shok, thf

# --- Aviación / naval (rápidos de clasificar, fuera de la escalera principal) ---
ROLE_AIRFIGHTER = "air_fighter"
ROLE_AIRBOMBER = "air_bomber"
ROLE_HELI = "heli"
ROLE_TRANSPORTER = "transporter"
ROLE_SHIP_ASW = "ship_sub"
ROLE_SHIP_COMBAT = "ship_combat"
ROLE_SHIP_AMPHIB = "ship_amphib"

# --- Roles de EDIFICIOS construibles (build) ---
ROLE_REFINERY = "refinery"                    # proc, silo
ROLE_POWER_INFRA = "power"                    # powr, apwr
ROLE_BARRACKS = "barracks"                    # barr, tent, kenn
ROLE_WARFACTORY = "warf"                      # weap
ROLE_TECH = "tech"                            # dome, atek, stek, gap
ROLE_DEFENSE_GUN = "defense_gun"              # gun, agun, pbox, hbox
ROLE_DEFENSE_TURRET = "defense_turret"        # ftur, tsla, sam
ROLE_REPAIR = "repair"                        # fix
ROLE_AIRBASE = "airbase"                      # hpad, afld
ROLE_NAVAL = "naval"                          # spen, syrd
ROLE_CIVIL = "civic"                          # fenc, brik, misc walls

# --------------------------------------------------------------------------
ROLE_OF_ITEM: dict[str, str] = {
    # Infantería
    "e1": ROLE_INFANTRY_BASIC,
    "e2": ROLE_INFANTRY_ANTIINF,
    "e4": ROLE_INFANTRY_ANTIINF,
    "med": ROLE_INFANTRY_ANTIINF,   # flame
    "dog": ROLE_INFANTRY_ANTIINF,
    "e3": ROLE_INFANTRY_ANTIARMOR,
    "e6": ROLE_COMMANDO_SPY,
    "spy": ROLE_COMMANDO_SPY,
    "delphi": ROLE_COMMANDO_SPY,
    "chan": ROLE_COMMANDO_SPY,
    "shok": ROLE_COMMANDO_SPY,
    "thf": ROLE_COMMANDO_SPY,
    # Vehículos
    "v2rl": ROLE_ROCKET_TRUCK,
    "1tnk": ROLE_TANK_MEDIUM,
    "2tnk": ROLE_TANK_MEDIUM,
    "4tnk": ROLE_TANK_MEDIUM,
    "3tnk": ROLE_TANK_HEAVY,
    "qtnk": ROLE_TANK_HEAVY,
    "stnk": ROLE_TANK_HEAVY,
    "ctnk": ROLE_TANK_HEAVY,
    "ttnk": ROLE_TANK_HEAVY,
    "mgg": ROLE_TANK_HEAVY,
    "arty": ROLE_ARTILLERY,
    "apc": ROLE_APC_TRANSPORT,
    "truk": ROLE_APC_TRANSPORT,
    "mnly": ROLE_APC_TRANSPORT,
    "dtrk": ROLE_APC_TRANSPORT,
    "ftrk": ROLE_APC_TRANSPORT,
    "jeep": ROLE_SCOUT,
    "mrj": ROLE_SCOUT,
    "harv": ROLE_HARVESTER,
    "mcv": ROLE_MCV,
    # Aviación
    "mig": ROLE_AIRFIGHTER, "yak": ROLE_AIRFIGHTER, "u2": ROLE_AIRFIGHTER,
    "badr": ROLE_AIRBOMBER,
    "heli": ROLE_HELI, "hind": ROLE_HELI, "mh60": ROLE_HELI,
    "tran": ROLE_TRANSPORTER,
    # Naval
    "ss": ROLE_SHIP_ASW, "msub": ROLE_SHIP_ASW,
    "dd": ROLE_SHIP_COMBAT, "ca": ROLE_SHIP_COMBAT, "pt": ROLE_SHIP_COMBAT,
    "lst": ROLE_SHIP_AMPHIB,
    # Edificios (build) — coherente con BUILDING_ITEM_TYPES
    "proc": ROLE_REFINERY, "silo": ROLE_REFINERY,
    "powr": ROLE_POWER_INFRA, "apwr": ROLE_POWER_INFRA,
    "barr": ROLE_BARRACKS, "tent": ROLE_BARRACKS, "kenn": ROLE_BARRACKS,
    "weap": ROLE_WARFACTORY,
    "dome": ROLE_TECH, "atek": ROLE_TECH, "stek": ROLE_TECH, "gap": ROLE_TECH,
    "gun": ROLE_DEFENSE_GUN, "agun": ROLE_DEFENSE_GUN,
    "pbox": ROLE_DEFENSE_GUN, "hbox": ROLE_DEFENSE_GUN,
    "ftur": ROLE_DEFENSE_TURRET, "tsla": ROLE_DEFENSE_TURRET,
    "sam": ROLE_DEFENSE_TURRET,
    "fix": ROLE_REPAIR,
    "hpad": ROLE_AIRBASE, "afld": ROLE_AIRBASE,
    "spen": ROLE_NAVAL, "syrd": ROLE_NAVAL,
    "fenc": ROLE_CIVIL, "brik": ROLE_CIVIL, "sbag": ROLE_CIVIL,
}

# --------------------------------------------------------------------------
# Helper: facción conocida -> NO hace falta, available_production ya viene
# filtrada. Pero para depurar/test mantenemos un alias de facciones comunes.
# --------------------------------------------------------------------------

def role_of(item: str) -> str:
    """Item concreto -> rol funcional. Desconocido -> 'misc'."""
    return ROLE_OF_ITEM.get(item.lower(), "misc")


def item_cost(item: str) -> float:
    """Costo de catálogo RA; desconocido = caro (no gana el desempate)."""
    try:
        from openra_env.game_data import get_building_stats, get_unit_stats
    except Exception:
        return 1e9
    key = str(item or "").lower()
    st = get_building_stats(key) or get_unit_stats(key)
    if not st:
        return 1e9
    return float(st.get("cost", 1e9) or 1e9)


def cheapest_of(items) -> str:
    """Concreto más barato del rol (pbox antes que gun/agun; ftur antes que tsla).

    El adapter decía 'más barato primero' pero ordenaba alfabético (agun < gun
    < pbox). Desempate alfabético para estabilidad.
    """
    its = [str(x).lower() for x in (items or []) if x]
    if not its:
        return ""
    return min(its, key=lambda it: (item_cost(it), it))


def concretos_de(rol: str) -> list[str]:
    """Todos los items concretos (del catalogo) que cumplen un rol."""
    return [it for it, r in ROLE_OF_ITEM.items() if r == rol]


def roles_de_available(available) -> dict[str, list[str]]:
    """De la lista de items disponibles (ya filtrada por faccion del server),
    devuelve {rol: [items concretos disponibles de ese rol]} ordenado de forma
    estable. Vacio si available es None/vacio.
    """
    out: dict[str, list[str]] = {}
    for it in sorted(available or []):
        rol = role_of(it)
        out.setdefault(rol, []).append(it)
    return out


def rol_estable(items_por_rol: dict[str, list[str]],
                preferencia) -> str | None:
    """Picks a role from available per learner preference order (unused for
    now, kept for symmetry)."""
    for r in preferencia:
        if items_por_rol.get(r):
            return r
    return None