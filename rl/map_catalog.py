"""P3 map inventory: scenarios, water/land flags, reset payloads, pool hook.

Curriculum default remains a_short (land / no competitive navy). Water stock
maps live under rl/scenarios/stock_*.oramap so train can select them without
depending on the OpenRA submodule checkout.

Spawn / Multi rotation: OpenRA lobby spawn pick is not hooked here. Variety
for P3 = map pool (2+ maps). True Multi1/Multi2 spawn rotation needs engine
or session overrides — see TODO in docs/23.
"""
from __future__ import annotations

import base64
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Repo-relative scenarios dir (train cwd is usually repo root).
SCENARIOS_DIR = Path("rl/scenarios")

# Build roles forbidden on land-only / puddle maps (Informe-2 anti-shipyard).
FORBIDDEN_BUILD_ROLES_LAND = frozenset({"naval"})  # spen, syrd
# Train roles that need a real navy theatre.
NAVAL_TRAIN_ROLES = frozenset({"ship_sub", "ship_combat", "ship_amphib"})


@dataclass(frozen=True)
class MapEntry:
    key: str
    file_name: str
    has_water: bool
    kind: str  # "fase2" | "stock"
    notes: str = ""
    has_air_start: bool = False  # pre-placed airfield (none of ours yet)

    @property
    def path(self) -> Path:
        return SCENARIOS_DIR / self.file_name


# Explicit inventory (CLI keys). fase2_* keep legacy --scenario names.
MAP_CATALOG: dict[str, MapEntry] = {
    "a": MapEntry("a", "fase2_a.oramap", False, "fase2", "prebuilt land"),
    "a_short": MapEntry(
        "a_short", "fase2_a_short.oramap", False, "fase2",
        "default land curriculum; Multi1-style short"),
    "a_minus_short": MapEntry(
        "a_minus_short", "fase2_a_minus_short.oramap", False, "fase2"),
    "amin160_short": MapEntry(
        "amin160_short", "fase2_amin160_short.oramap", False, "fase2"),
    "amin160_allies": MapEntry(
        "amin160_allies", "fase2_amin160_allies.oramap", False, "fase2"),
    "probe": MapEntry("probe", "fase2_probe.oramap", False, "fase2"),
    "probe_short": MapEntry(
        "probe_short", "fase2_probe_short.oramap", False, "fase2"),
    # Water-capable stock skirmish maps (copied from OpenRA mods/ra/maps).
    "doughnut": MapEntry(
        "doughnut", "stock_doughnut.oramap", True, "stock",
        "water ring; naval viable"),
    "bombardment_islands": MapEntry(
        "bombardment_islands", "stock_bombardment_islands.oramap", True,
        "stock", "islands; naval viable"),
}

# Name substrings → water / land when map_name is not an exact catalog key
# (engine may report stock_doughnut.oramap, doughnut.oramap, etc.).
_WATER_NAME_HINTS = (
    "doughnut", "bombardment", "archipelago", "tournament-island",
    "shuriken-island", "island",
)
_LAND_NAME_HINTS = (
    "a_short", "fase2_a", "singles", "probe", "amin160", "a_minus",
    "forgotten-plains", "green-belt", "desert",
)

LAND_POOL = ("a_short",)
WATER_POOL = ("doughnut", "bombardment_islands")
MIXED_POOL = ("a_short", "doughnut")
NAMED_POOLS = {
    "land": LAND_POOL,
    "water": WATER_POOL,
    "mixed": MIXED_POOL,
}


def list_inventory() -> list[MapEntry]:
    return [MAP_CATALOG[k] for k in sorted(MAP_CATALOG)]


def normalize_key(raw: str | None) -> str:
    s = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if s.endswith(".oramap"):
        s = s[: -len(".oramap")]
    if s.startswith("fase2_"):
        s = s[len("fase2_") :]
    if s.startswith("stock_"):
        s = s[len("stock_") :]
    # Legacy CLI: --scenario A / a
    if s == "a":
        return "a"
    return s


def get_entry(key: str) -> MapEntry:
    k = normalize_key(key)
    if k not in MAP_CATALOG:
        known = ", ".join(sorted(MAP_CATALOG))
        raise KeyError(f"unknown map/scenario {key!r}; known: {known}")
    return MAP_CATALOG[k]


def resolve_map_path(key: str) -> Path:
    entry = get_entry(key)
    p = entry.path
    if not p.exists():
        raise FileNotFoundError(
            f"map file missing: {p} (key={entry.key}). "
            f"Copy stock maps into rl/scenarios/ or generate fase2_*.oramap.")
    return p


def reset_payload_for(key: str) -> dict:
    """kwargs for env.reset: map_data (b64) + stable map_name."""
    entry = get_entry(key)
    path = resolve_map_path(key)
    return {
        "map_data": base64.b64encode(path.read_bytes()).decode(),
        "map_name": entry.file_name,
    }


def parse_pool_arg(raw: str | None) -> list[str] | None:
    """Parse --map-pool: named (land|water|mixed) or comma keys.

    None / empty → no pool (single --scenario or full game).
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    low = s.lower()
    if low in NAMED_POOLS:
        return list(NAMED_POOLS[low])
    keys = [normalize_key(p) for p in s.split(",") if p.strip()]
    if not keys:
        return None
    for k in keys:
        get_entry(k)  # validate
    return keys


def sample_pool(pool: Iterable[str], rng: random.Random | None = None) -> str:
    keys = list(pool)
    if not keys:
        raise ValueError("empty map pool")
    r = rng or random
    return r.choice(keys)


def allows_naval(map_name: str | None) -> bool:
    """True if navy BUILD/TRAIN should be legal for this map_name."""
    n = str(map_name or "").lower()
    if not n:
        return False
    # Exact / normalized catalog hit
    try:
        return bool(get_entry(n).has_water)
    except KeyError:
        pass
    stem = n.replace("\\", "/").split("/")[-1]
    if stem.endswith(".oramap"):
        stem = stem[: -len(".oramap")]
    try:
        return bool(get_entry(stem).has_water)
    except KeyError:
        pass
    if any(h in n for h in _WATER_NAME_HINTS):
        return True
    if any(h in n for h in _LAND_NAME_HINTS):
        return False
    # Conservative default: forbid navy (Singles puddle / unknown land).
    return False


def allows_airbase_optional(map_name: str | None) -> bool:
    """Helipad/airfield BUILD is legal on land; always True for mask purposes.

    Teacher may still gate optional air push on cash/tech.
    """
    return True


def inventory_summary() -> str:
    lines = ["key | water | kind | file | notes"]
    for e in list_inventory():
        w = "yes" if e.has_water else "no"
        lines.append(
            f"{e.key} | {w} | {e.kind} | {e.file_name} | {e.notes}")
    lines.append(
        "pools: land=" + ",".join(LAND_POOL)
        + " water=" + ",".join(WATER_POOL)
        + " mixed=" + ",".join(MIXED_POOL))
    lines.append(
        "spawn: Multi spawn rotation not hooked; use --map-pool for layout variety.")
    return "\n".join(lines)
