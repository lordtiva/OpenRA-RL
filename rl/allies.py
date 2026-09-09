# -*- coding: utf-8 -*-
"""Contrato RA Aliados (rl/docs/contract/ra-aliados.md). Constantes de lobby / catálogo.

La red no ve estos strings: el engine resuelve RandomAllies a un país
antes del primer tick y `available_production` ya viene filtrado.
"""

# Slot rl-agent. RandomAllies → england | france | germany.
DEFAULT_PLAYER_FACTION = "RandomAllies"
# Rival scripted: Random (Allies o Soviet). No es lock vs soviet.
DEFAULT_ENEMY_FACTION = "Random"

ALLIED_COUNTRIES = ("england", "france", "germany")
SOVIET_COUNTRIES = ("russia", "ukraine")
ALLIED_SIDES = frozenset(ALLIED_COUNTRIES) | {"allies", "RandomAllies"}

# Bot types that occupy the learner slot (not the scripted rival).
RL_AGENT_BOT_TYPES = frozenset({"rl-agent", "rl", "self"})


def resolve_player_faction(asked: str | None) -> str:
    """Mirror of RLSessionManager.ResolvePlayerFaction (keep in sync)."""
    s = str(asked or "").strip()
    if not s or s.lower() == "random":
        return DEFAULT_PLAYER_FACTION
    return s


def resolve_enemy_faction(asked: str | None) -> str:
    s = str(asked or "").strip()
    return s if s else DEFAULT_ENEMY_FACTION


def is_rl_agent_bot(bot_type: str | None) -> bool:
    return str(bot_type or "").strip().lower() in RL_AGENT_BOT_TYPES

# InternNames that must not fall to misc (crash / TRAIN-as-building).
ALLIED_CATALOG_UNITS = (
    "e1", "e3", "e6", "e7", "medi", "mech", "spy", "dog",
    "1tnk", "2tnk", "ctnk", "stnk", "jeep", "arty", "apc", "harv", "mcv",
    "heli", "mh60", "tran", "dd", "ca", "pt", "lst",
)
ALLIED_CATALOG_BUILDINGS = (
    "powr", "apwr", "proc", "silo", "tent", "weap", "dome", "atek", "fix",
    "hpad", "syrd", "pbox", "hbox", "gun", "agun", "gap", "pdox",
    "sbag", "brik", "fenc",
    "fpwr", "tenf", "syrf", "spef", "weaf", "domf", "fixf", "fapw",
    "atef", "pdof", "mslf", "facf",
)
