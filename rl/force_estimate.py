# -*- coding: utf-8 -*-
"""Proxies de OpenRA AttackOrFleeFuzzy para scalars / reward.

OpenRA (AttackOrFleeFuzzy) compara:
  RelativeAttackPower, RelativeHealth, RelativeSpeed.
  Squads Rush: atacan solo si Strong (own/enemy ~= 1.1).

No tenemos Damage/Reload en game_data, asi que:
  power  ~= sum(cost) de unidades de combate
  health ~= sum(hp_percent * catalog_hp)
  speed  ~= media de speed de combate movil
"""
from __future__ import annotations

from openra_env.game_data import get_unit_stats

# Mismo umbral que Rush fuzzy (~1.1).
STRONG_RATIO = 1.1
# Combate a <= este Manhattan de un edificio propio = "en casa".
HOME_RADIUS = 10


def _is_harv_or_mcv(u) -> bool:
    t = str(getattr(u, "type", "") or "").lower()
    return "harv" in t or "mcv" in t


def _is_combat(u) -> bool:
    if _is_harv_or_mcv(u):
        return False
    return bool(getattr(u, "can_attack", False))


def _catalog(tipo: str) -> dict:
    st = get_unit_stats(str(tipo or ""))
    return st if isinstance(st, dict) else {}


def _side_force(units) -> dict:
    """Agrega poder / HP / speed de una lista de actores (solo combate)."""
    power = 0.0
    hp = 0.0
    speed_sum = 0.0
    n_speed = 0
    n = 0
    for u in units or []:
        if not _is_combat(u):
            continue
        n += 1
        tipo = getattr(u, "type", "") or ""
        st = _catalog(tipo)
        cost = float(st.get("cost", 300.0) or 300.0)
        power += cost
        cat_hp = float(st.get("hp", 5000.0) or 5000.0)
        hp_pct = float(getattr(u, "hp_percent", 1.0) or 0.0)
        hp += max(0.0, hp_pct) * cat_hp
        # Preferir speed del wire; si no, catalogo.
        spd = float(getattr(u, "speed", 0) or 0)
        if spd <= 0:
            spd = float(st.get("speed", 0) or 0)
        if spd > 0:
            speed_sum += spd
            n_speed += 1
    mean_spd = (speed_sum / n_speed) if n_speed else 0.0
    return {
        "n": n,
        "power": power,
        "hp": hp,
        "speed": mean_spd,
    }


def aoa_features(obs) -> dict:
    """Features estilo AttackOrFlee; enemigo = solo visibles (niebla).

    Si no hay enemigos visibles, ratios = 0.5 (neutro) y strong = 0
    (no inventamos superioridad bajo fog).
    """
    own = _side_force(getattr(obs, "units", None) or [])
    ene = _side_force(getattr(obs, "visible_enemies", None) or [])
    eps = 1e-3
    if ene["n"] <= 0:
        rel_power = 0.5
        rel_health = 0.5
        rel_speed = 0.5
        strong = 0.0
        ratio = 1.0
    else:
        rel_power = own["power"] / (own["power"] + ene["power"] + eps)
        rel_health = own["hp"] / (own["hp"] + ene["hp"] + eps)
        rel_speed = own["speed"] / (own["speed"] + ene["speed"] + eps)
        ratio = own["power"] / max(ene["power"], eps)
        strong = 1.0 if ratio > STRONG_RATIO else 0.0
    return {
        "own_power": own["power"],
        "ene_power": ene["power"],
        "own_hp": own["hp"],
        "ene_hp": ene["hp"],
        "own_speed": own["speed"],
        "ene_speed": ene["speed"],
        "rel_power": float(rel_power),
        "rel_health": float(rel_health),
        "rel_speed": float(rel_speed),
        "strong": float(strong),
        "ratio": float(ratio),
        "n_own": own["n"],
        "n_ene": ene["n"],
    }


def combat_away_frac(obs, radius: int = HOME_RADIUS) -> float:
    """Fraccion de combate propio a >radius de TODO edificio propio.

    1.0 = todo el ejercito lejos de base (push); 0.0 = turtling / sin combate.
    """
    units = [u for u in (getattr(obs, "units", None) or []) if _is_combat(u)]
    if not units:
        return 0.0
    bldgs = list(getattr(obs, "buildings", None) or [])
    if not bldgs:
        return 1.0
    bpts = [(int(b.cell_x), int(b.cell_y)) for b in bldgs]
    away = 0
    for u in units:
        ux, uy = int(u.cell_x), int(u.cell_y)
        dmin = min(abs(ux - bx) + abs(uy - by) for bx, by in bpts)
        if dmin > radius:
            away += 1
    return away / float(len(units))


def force_edge_reward(obs, w: float, step_scale: float = 0.02,
                      min_away: float = 0.35) -> float:
    """Reward chico si Strong y el ejercito no esta turtling en casa.

    Magitud tipica: w * 0.02 * edge * away  (con w=0.5 ~= 0.01/step max).
    No castiga turtling (ya esta w_timeout / garrison cap).
    """
    if w <= 0:
        return 0.0
    feats = aoa_features(obs)
    if feats["strong"] < 0.5:
        return 0.0
    away = combat_away_frac(obs)
    if away < min_away:
        return 0.0
    # ratio 1.1 -> 0, ~3.1 -> 1
    edge = max(0.0, min(1.0, (feats["ratio"] - STRONG_RATIO) / 2.0))
    return float(w * step_scale * edge * min(1.0, away))
