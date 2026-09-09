"""P3 map inventory: scenarios, water/land flags, reset payloads, pool hook.

Curriculum default remains a_short (lakes present; navy gated — not a navy
theatre). Prefer official OpenRA map paths/titles under OpenRA/mods/ra/maps
when the submodule (or sibling MAIN checkout) is populated; thin copies under
rl/scenarios/ are fallbacks only.

Pools:
  land              — a_short only (default curriculum)
  mixed             — a_short + a few official mixed land-water maps
  official_2p_small — curated ~5-map pool for 1v1 / 2-player training
                      (official maps may declare >2 Multi slots; we still run 2p)
  water             — alias of naval-viable official mixed maps (NOT "pure water")

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

# Repo root (rl/map_catalog.py -> parents[1])
_REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = _REPO_ROOT / "rl" / "scenarios"

# Official RA maps relative to an OpenRA checkout.
_OFFICIAL_MAPS_REL = Path("OpenRA") / "mods" / "ra" / "maps"

# Extra OpenRA checkouts to probe when worktree submodule is empty
# (MAIN desktop clone sits next to OpenRA-RL-p0 worktree).
_EXTRA_OPENRA_ROOTS = (
    _REPO_ROOT.parent / "OpenRA-RL",
)

# Build roles forbidden when navy is not viable (Informe-2 anti-shipyard).
FORBIDDEN_BUILD_ROLES_LAND = frozenset({"naval"})  # spen, syrd
# Train roles that need a real navy theatre.
NAVAL_TRAIN_ROLES = frozenset({"ship_sub", "ship_combat", "ship_amphib"})


@dataclass(frozen=True)
class MapEntry:
    key: str
    file_name: str
    has_water: bool
    kind: str  # "fase2" | "official"
    display_name: str = ""  # original OpenRA Title / human label
    notes: str = ""
    has_air_start: bool = False
    # Navy BUILD/TRAIN legal only when True (puddles/lakes alone are not enough).
    naval_viable: bool = False
    # Basename under OpenRA/mods/ra/maps (official). None => scenarios-only.
    official_file: str | None = None

    @property
    def path(self) -> Path:
        """Scenarios-dir fallback path (may be missing; use resolve_map_path)."""
        return SCENARIOS_DIR / self.file_name


# Explicit inventory (CLI keys). fase2_* keep legacy --scenario names.
MAP_CATALOG: dict[str, MapEntry] = {
    # Soft-deprecated: long fase2_a.oramap lives in rl/scenarios/_archive/.
    # Key "a" aliases a_short so old CLIs keep working.
    "a": MapEntry(
        "a", "fase2_a_short.oramap", True, "fase2",
        display_name="DEPRECATED alias of a_short",
        notes="use a_short; long fase2_a.oramap archived under _archive/",
        naval_viable=False,
    ),
    "a_short": MapEntry(
        "a_short", "fase2_a_short.oramap", True, "fase2",
        display_name="A-short / Singles short",
        notes="default curriculum; lakes present; navy gated",
        naval_viable=False,
    ),
    # Official mixed land-water skirmish maps (prefer OpenRA/mods/ra/maps).
    "doughnut": MapEntry(
        "doughnut", "doughnut.oramap", True, "official",
        display_name="Doughnut",
        notes="water ring; naval viable; official 6p used as 2p",
        naval_viable=True,
        official_file="doughnut.oramap",
    ),
    "bombardment_islands": MapEntry(
        "bombardment_islands", "bombardment-islands.oramap", True, "official",
        display_name="Bombardment Islands",
        notes="islands; naval viable; official 6p used as 2p",
        naval_viable=True,
        official_file="bombardment-islands.oramap",
    ),
    "tournament_island": MapEntry(
        "tournament_island", "tournament-island.oramap", True, "official",
        display_name="Tournament Island",
        notes="island theatre; naval viable; official 4p used as 2p",
        naval_viable=True,
        official_file="tournament-island.oramap",
    ),
    "x_lake": MapEntry(
        "x_lake", "x-lake.oramap", True, "official",
        display_name="X-Lake",
        notes="lake/cross; naval viable; official 4p used as 2p",
        naval_viable=True,
        official_file="x-lake.oramap",
    ),
    "archipelago": MapEntry(
        "archipelago", "archipelago.oramap", True, "official",
        display_name="Archipelago",
        notes="multi-island; naval viable; official 8p used as 2p",
        naval_viable=True,
        official_file="archipelago.oramap",
    ),
}

# Name substrings → water / land when map_name is not an exact catalog key
# (engine may report stock_doughnut.oramap, doughnut.oramap, etc.).
_WATER_NAME_HINTS = (
    "doughnut", "bombardment", "archipelago", "tournament-island",
    "tournament_island", "shuriken-island", "x-lake", "x_lake", "island",
)
_LAND_NAME_HINTS = (
    # Keep singles/forgotten-plains as non-navy (puddles ≠ navy theatre).
    "singles", "forgotten-plains", "green-belt", "desert",
)
# a_short / fase2_* have lakes but navy stays gated via entry.naval_viable.
_NAVY_GATED_HINTS = (
    "a_short", "fase2_a", "fase2_a_short",
)

LAND_POOL = ("a_short",)
# Naval-viable official mixed land-water maps (not "pure water").
WATER_POOL = (
    "doughnut", "bombardment_islands", "tournament_island", "x_lake",
)
MIXED_POOL = ("a_short", "doughnut", "bombardment_islands")
# Curated small pool for 2-player training (a_short default + official mixed).
OFFICIAL_2P_SMALL = (
    "a_short",
    "doughnut",
    "bombardment_islands",
    "tournament_island",
    "x_lake",
)
NAMED_POOLS = {
    "land": LAND_POOL,
    "water": WATER_POOL,  # naval-viable mixed; not pure-water framing
    "mixed": MIXED_POOL,
    "official_2p_small": OFFICIAL_2P_SMALL,
}


def list_inventory() -> list[MapEntry]:
    return [MAP_CATALOG[k] for k in sorted(MAP_CATALOG)]


def normalize_key(raw: str | None) -> str:
    s = str(raw or "").strip().lower().replace(" ", "_")
    if s.endswith(".oramap"):
        s = s[: -len(".oramap")]
    if s.startswith("fase2_"):
        s = s[len("fase2_") :]
    if s.startswith("stock_"):
        s = s[len("stock_") :]
    # Official hyphenated basenames → catalog keys
    s = s.replace("-", "_")
    # Legacy CLI: --scenario A / a → a_short (long fase2_a archived)
    if s == "a":
        return "a_short"
    return s


def get_entry(key: str) -> MapEntry:
    k = normalize_key(key)
    if k not in MAP_CATALOG:
        known = ", ".join(sorted(MAP_CATALOG))
        raise KeyError(f"unknown map/scenario {key!r}; known: {known}")
    return MAP_CATALOG[k]


def _candidate_paths(entry: MapEntry) -> list[Path]:
    """Ordered search: official OpenRA trees, then thin rl/scenarios copy."""
    out: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path) -> None:
        try:
            rp = p.resolve()
        except OSError:
            rp = p
        if rp in seen:
            return
        seen.add(rp)
        out.append(p)

    official_name = entry.official_file or (
        entry.file_name if entry.kind == "official" else None
    )
    if official_name:
        _add(_REPO_ROOT / _OFFICIAL_MAPS_REL / official_name)
        for root in _EXTRA_OPENRA_ROOTS:
            _add(root / _OFFICIAL_MAPS_REL / official_name)
    # Thin / vendored fallback (official basename preferred)
    _add(SCENARIOS_DIR / entry.file_name)
    # Legacy stock_* copies lived here; still accept if present / archived
    if official_name and official_name != entry.file_name:
        _add(SCENARIOS_DIR / official_name)
    stem = entry.key
    _add(SCENARIOS_DIR / f"stock_{stem}.oramap")
    _add(SCENARIOS_DIR / "_archive" / f"stock_{stem}.oramap")
    if entry.kind == "fase2":
        _add(SCENARIOS_DIR / entry.file_name)
        _add(SCENARIOS_DIR / "_archive" / entry.file_name)
    return out


def resolve_map_path(key: str) -> Path:
    entry = get_entry(key)
    for p in _candidate_paths(entry):
        if p.is_file():
            return p
    tried = ", ".join(str(p) for p in _candidate_paths(entry)[:6])
    raise FileNotFoundError(
        f"map file missing for key={entry.key!r} (display={entry.display_name!r}). "
        f"Tried: {tried}. Populate OpenRA/mods/ra/maps or rl/scenarios/."
    )


def reset_payload_for(key: str) -> dict:
    """kwargs for env.reset: map_data (b64) + stable map_name (official basename)."""
    entry = get_entry(key)
    path = resolve_map_path(key)
    # Prefer official basename for engine / allows_naval hints.
    map_name = entry.official_file or entry.file_name
    return {
        "map_data": base64.b64encode(path.read_bytes()).decode(),
        "map_name": map_name,
    }


def parse_pool_arg(raw: str | None) -> list[str] | None:
    """Parse --map-pool: named pools or comma keys.

    Named: land | water | mixed | official_2p_small
    None / empty → no pool (single --scenario or full game).
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    low = s.lower().replace("-", "_")
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
    """True if navy BUILD/TRAIN should be legal for this map_name.

    Lakes / puddles (a_short) set has_water but naval_viable=False → False here.
    """
    n = str(map_name or "").lower()
    if not n:
        return False
    # Exact / normalized catalog hit
    try:
        return bool(get_entry(n).naval_viable)
    except KeyError:
        pass
    stem = n.replace("\\", "/").split("/")[-1]
    if stem.endswith(".oramap"):
        stem = stem[: -len(".oramap")]
    try:
        return bool(get_entry(stem).naval_viable)
    except KeyError:
        pass
    if any(h in n for h in _NAVY_GATED_HINTS):
        return False
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
    lines = ["key | water | navy | kind | display | file | notes"]
    for e in list_inventory():
        w = "yes" if e.has_water else "no"
        nv = "yes" if e.naval_viable else "no"
        dn = e.display_name or e.key
        lines.append(
            f"{e.key} | {w} | {nv} | {e.kind} | {dn} | {e.file_name} | {e.notes}")
    lines.append(
        "pools: land=" + ",".join(LAND_POOL)
        + " mixed=" + ",".join(MIXED_POOL)
        + " official_2p_small=" + ",".join(OFFICIAL_2P_SMALL)
        + " water(naval-viable)=" + ",".join(WATER_POOL))
    lines.append(
        "spawn: Multi spawn rotation not hooked; use --map-pool for layout variety.")
    lines.append(
        "paths: prefer OpenRA/mods/ra/maps/<official>; fallback rl/scenarios/.")
    return "\n".join(lines)
