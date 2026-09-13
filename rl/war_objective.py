# -*- coding: utf-8 -*-
"""Spawn-agnostic war dest: raid / visible / mental-base / ghost / fog.

Never BEACON_BY_MAP / map-title GPS / hardcoded (95,11). Own conyard is the
anchor; the dest moves with sightings and shroud. Shared by teacher, adapter,
and live/rollout dest accounting.
"""
from __future__ import annotations

from rl.auto_support import (
    ARRIVED_CELLS,
    MIN_PILE_FOR_HUNT,
    _PROD_BUILDINGS,
    _combat_units,
    _farthest_xy,
    _n_combat_at,
    _nearest_xy,
    _own_anchor,
    _xy,
    fog_scout_destinations,
    home_raid_targets,
    hunt_near_cell,
)

# Chebyshev: AM onto own centroid / yard counts as "clicking your feet".
FEET_CELLS = 8
# Idle pack size to rewrite a local group-AM (same as PACK_ARMY).
FEET_PACK = 12


def map_center(obs) -> tuple[int, int]:
    info = getattr(obs, "map_info", None)
    w = max(int(getattr(info, "width", 128) or 128), 1)
    h = max(int(getattr(info, "height", 64) or 64), 1)
    return w // 2, h // 2


def own_anchor(obs) -> tuple[int, int]:
    """Construction yard / civic building, else a unit, else map center."""
    hit = _own_anchor(obs)
    if hit is not None:
        return int(hit[0]), int(hit[1])
    for u in getattr(obs, "units", None) or []:
        try:
            return int(u.cell_x), int(u.cell_y)
        except (TypeError, ValueError):
            continue
    return map_center(obs)


def _cheb(a, b) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _mental_xy(obs, belief=None) -> tuple[int, int] | None:
    bel = belief if belief is not None else getattr(obs, "belief", None)
    if bel is not None:
        xy = getattr(bel, "enemy_base_xy", None)
        if xy is not None:
            try:
                return int(xy[0]), int(xy[1])
            except (TypeError, ValueError, IndexError):
                pass
    xy = getattr(obs, "enemy_base_xy", None)
    if xy is not None:
        try:
            return int(xy[0]), int(xy[1])
        except (TypeError, ValueError, IndexError):
            return None
    return None


def _ghost_xy(obs, belief=None) -> tuple[int, int] | None:
    bel = belief if belief is not None else getattr(obs, "belief", None)
    if bel is None or not hasattr(bel, "select_enemies"):
        return None
    slots = bel.select_enemies(obs)
    ghosts = [
        (u, conf) for (u, vis, conf, _t) in slots
        if float(vis) < 0.5 and float(conf) > 0.05
    ]
    if not ghosts:
        return None
    ghosts.sort(key=lambda t: -float(t[1]))
    try:
        return _xy(ghosts[0][0])
    except (TypeError, ValueError):
        return None


def _snap(obs, aidx, xy):
    if xy is None:
        return None
    x, y = int(xy[0]), int(xy[1])
    if aidx is None:
        return x, y
    from rl.action_adapter import nearest_passable
    grid = getattr(aidx, "pass_grid", None)
    h = int(getattr(aidx, "h", 0) or 0)
    w = int(getattr(aidx, "w", 0) or 0)
    if h < 1 or w < 1:
        return x, y
    return nearest_passable(x, y, grid, h, w)


def war_objective(obs, aidx=None, last_contact=None, belief=None):
    """Raid > visible leftover > mental base > ghost > last contact > fog.

    Never resolve_beacon / BEACON_BY_MAP. Returns (x, y) or None.
    """
    origin = own_anchor(obs)
    raids = home_raid_targets(obs)
    if raids:
        return _snap(obs, aidx, _nearest_xy(raids, origin))

    bldgs = list(getattr(obs, "visible_enemy_buildings", None) or [])
    prod = [
        b for b in bldgs
        if str(getattr(b, "type", "") or "").lower() in _PROD_BUILDINGS
    ]
    if prod:
        return _snap(obs, aidx, _farthest_xy(prod, origin))
    if bldgs:
        return _snap(obs, aidx, _farthest_xy(bldgs, origin))

    ene = list(getattr(obs, "visible_enemies", None) or [])
    if ene:
        return _snap(obs, aidx, _farthest_xy(ene, origin))

    mental = _mental_xy(obs, belief)
    if mental is not None:
        return _snap(obs, aidx, mental)

    ghost = _ghost_xy(obs, belief)
    if ghost is not None:
        return _snap(obs, aidx, ghost)

    if last_contact is not None:
        try:
            anchor = (int(last_contact[0]), int(last_contact[1]))
        except (TypeError, ValueError, IndexError):
            anchor = None
        if anchor is not None:
            combat = _combat_units(getattr(obs, "units", None) or [])
            n_at = _n_combat_at(combat, anchor, ARRIVED_CELLS)
            if n_at >= MIN_PILE_FOR_HUNT:
                return _snap(obs, aidx, hunt_near_cell(obs, anchor))
            return _snap(obs, aidx, anchor)

    fog = fog_scout_destinations(obs, 1, aidx)
    if fog:
        return _snap(obs, aidx, fog[0])
    return None


def _idle_combat_xy(obs) -> list[tuple[int, int]]:
    out = []
    for u in getattr(obs, "units", None) or []:
        ut = str(getattr(u, "type", "") or "").lower()
        if "harv" in ut or "mcv" in ut:
            continue
        if not bool(getattr(u, "can_attack", True)):
            continue
        if not bool(getattr(u, "is_idle", False)):
            continue
        try:
            out.append((int(u.cell_x), int(u.cell_y)))
        except (TypeError, ValueError):
            continue
    return out


def _combat_centroid_xy(obs) -> tuple[float, float] | None:
    pts = []
    for u in getattr(obs, "units", None) or []:
        ut = str(getattr(u, "type", "") or "").lower()
        if "harv" in ut or "mcv" in ut:
            continue
        if not bool(getattr(u, "can_attack", True)):
            continue
        try:
            pts.append((int(u.cell_x), int(u.cell_y)))
        except (TypeError, ValueError):
            continue
    if not pts:
        return None
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def reject_feet_push_cell(obs, aidx, cx: int, cy: int, belief=None,
                          last_contact=None):
    """Rewrite group-AM that lands on the army / own yard when a real dest exists.

    Illegal-cell safety (like remap), not GPS. Raid at home is the dest.
    """
    click = (int(cx), int(cy))
    idle = _idle_combat_xy(obs)
    if len(idle) < FEET_PACK:
        return click
    cen = _combat_centroid_xy(obs)
    if cen is None:
        return click
    near_feet = _cheb(click, (int(round(cen[0])), int(round(cen[1])))) <= FEET_CELLS
    anchor = own_anchor(obs)
    near_home = _cheb(click, anchor) <= 18
    if not (near_feet or near_home):
        return click
    obj = war_objective(obs, aidx, last_contact=last_contact, belief=belief)
    if obj is None:
        return click
    if _cheb(click, obj) <= FEET_CELLS:
        return click
    return _snap(obs, aidx, obj) or click


def guard_push_cell(obs, aidx, cx: int, cy: int, belief=None, last_contact=None):
    """Don't yank a vanguard away from war_objective back toward the yard.

    Spawn-agnostic: 'behind' = farther from the objective than the front.
    Home raid: leave the issued cell alone.
    """
    click = (int(cx), int(cy))
    if home_raid_targets(obs):
        return click
    obj = war_objective(obs, aidx, last_contact=last_contact, belief=belief)
    if obj is None:
        return click
    pts = []
    for u in getattr(obs, "units", None) or []:
        ut = str(getattr(u, "type", "") or "").lower()
        if "harv" in ut or "mcv" in ut:
            continue
        if not bool(getattr(u, "can_attack", True)):
            continue
        try:
            pts.append((int(u.cell_x), int(u.cell_y)))
        except (TypeError, ValueError):
            continue
    if len(pts) < 8:
        return click
    anchor = own_anchor(obs)
    front = [p for p in pts if _cheb(p, obj) + 8 < _cheb(p, anchor)]
    if len(front) < 8:
        return click
    d_front = min(_cheb(p, obj) for p in front)
    if _cheb(click, obj) > d_front + 12:
        return _snap(obs, aidx, obj) or click
    return click
