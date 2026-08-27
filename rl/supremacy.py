# -*- coding: utf-8 -*-
"""Evaluador de supremacía estilo motor de ajedrez para OpenRA.

Nivel 1 (este módulo): score estático de estado a partir de la observación
— valor de ejército + edificios + ingreso, propio vs enemigo visible.
Nivel 3 (futuro): P(victoria|estado) calibrada con episodios logueados.
"""
import math

from openra_env.game_data import get_unit_stats, get_building_stats


def _cost_of(tipo: str) -> float:
    """Costo de una unidad o edificio; desconocidos valen 300 (estimación)."""
    st = get_unit_stats(tipo)
    if st is None:
        st = get_building_stats(tipo)
    if st is None:
        return 300.0
    return float(st.get("cost", 300.0))


def evaluate_supremacy(obs, enemy_known: bool = True, gs=None) -> dict:
    """Score de supremacía desde UNA observación completa.

    Si hay modo espectador (gs explícito o obs.global_summary del server),
    usa esos valores EXACTOS; si no, cae a niebla (solo visible).

    Devuelve dict con:
      own/enemy   valor total estimado de cada bando ($)
      diff        own − enemy
      lead_ratio  tanh(diff / escala) ∈ [−1, +1]  (la 'barra' estilo ajedrez)
      p_win_est   prob. de victoria ESTIMADA heurística (nivel 1; la versión
                  calibrada con datos reales es el nivel 3)
      exact       True si vino del modo espectador (datos completos reales)
    """
    if gs is None:
        gs = getattr(obs, "global_summary", None)
    if gs and isinstance(gs, dict) and gs.get("own") and gs.get("enemy"):
        SCALE = 6000.0
        # Valor militar + edificios + efectivo de CADA bando, exacto
        def total(side):
            return float(side.get("unit_value", 0)
                         + side.get("building_value", 0)
                         + side.get("cash", 0))
        own, enemy = total(gs["own"]), total(gs["enemy"])
        diff = own - enemy
        lead_ratio = math.tanh(diff / SCALE)
        return {
            "own": round(own, 0),
            "enemy": round(enemy, 0),
            "diff": round(diff, 0),
            "lead_ratio": round(lead_ratio, 3),
            "p_win_est": round(0.5 * (1.0 + lead_ratio), 3),
            "exact": True,
        }

    own = sum(_cost_of(u.type) for u in obs.units)
    own += sum(_cost_of(b.type) for b in obs.buildings)
    # Efectivo + cosecha acumulada: poder de compra inmediato
    eco = getattr(obs.economy, "cash", 0) + getattr(obs.economy, "ore", 0)
    own += eco

    enemy = sum(_cost_of(u.type) for u in obs.visible_enemies)
    enemy += sum(_cost_of(b.type) for b in obs.visible_enemy_buildings)

    # Escala: el tanh satura alrededor de ±$6000 de diferencia
    SCALE = 6000.0
    diff = own - enemy
    lead_ratio = math.tanh(diff / SCALE)

    # Heurística nivel-1: mapear la barra a probabilidad con pendiente fija.
    # Nivel 3 reemplazará esto por un modelo ajustado con resultados reales.
    p_win_est = 0.5 * (1.0 + lead_ratio)

    return {
        "own": round(own, 0),
        "enemy": round(enemy, 0),
        "diff": round(diff, 0),
        "lead_ratio": round(lead_ratio, 3),
        "p_win_est": round(p_win_est, 3),
        "fog_note": ("enemy solo visible (niebla)" if not enemy_known else ""),
        "exact": False,
    }
