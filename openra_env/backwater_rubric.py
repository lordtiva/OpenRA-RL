"""Rubric scoring for the Backwater Battle benchmark."""

from __future__ import annotations

from typing import Any


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _action_names(events: list[dict[str, Any]], messages: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for event in events:
        if event.get("type") == "tool":
            tool = event.get("tool")
            if isinstance(tool, str):
                names.append(tool)
    for message in messages:
        for tc in message.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                names.append(fn["name"])
    return names


def compute_backwater_score(
    *,
    result: str,
    ticks: int,
    kills_cost: int,
    deaths_cost: int,
    assets_value: int,
    explored_percent: float,
    buildings_killed: int = 0,
    buildings_lost: int = 0,
    own_units: int = 0,
    own_buildings: int = 0,
    events: list[dict[str, Any]] | None = None,
    messages: list[dict[str, Any]] | None = None,
    encountered_agent_error: bool = False,
) -> dict[str, Any]:
    """Compute a positive 0-100 Backwater benchmark rubric score.

    This score rewards both outcome and scenario-specific behavior. It is robust
    to partial games: a loss can still earn points for efficient defense and
    objective pressure, while crashes and invalid-command loops are penalized.
    """
    events = events or []
    messages = messages or []
    result_normalized = (result or "").lower()
    win = result_normalized == "win"
    kd_ratio = kills_cost / max(deaths_cost, 1)
    event_types = {str(e.get("type", "")) for e in events}

    primary_result = 30.0 if win else 0.0
    if win:
        # Full speed credit by 7000 ticks, taper to 0 by 18000.
        primary_result += 10.0 * _clamp((18000 - ticks) / 11000)
    else:
        # Partial credit for surviving long enough to contest the battle.
        primary_result += 5.0 * _clamp(ticks / 12000)

    lure_and_pincer = 0.0
    if "first_attack_order" in event_types:
        lure_and_pincer += 4.0
    lure_and_pincer += 6.0 * _clamp(buildings_killed / 4)
    lure_and_pincer += 3.0 * _clamp(kills_cost / 8000)
    lure_and_pincer += 2.0 * _clamp(ticks / 3000)

    combat = 20.0 * _clamp(kd_ratio / 1.5)
    camp_raid = 15.0 * _clamp(buildings_killed / 5)
    pincer_pressure = 10.0 * _clamp((kills_cost + buildings_killed * 2000) / 18000)
    han_survival = 5.0 * _clamp(assets_value / 12000)

    if own_units or own_buildings:
        han_survival += 5.0 * _clamp((own_units + 2 * own_buildings) / 20)

    scouting = 5.0 * _clamp(explored_percent / 70.0)

    action_names = _action_names(events, messages)
    invalid_tool_results = sum(
        1 for m in messages
        if m.get("role") == "tool"
        and isinstance(m.get("content"), str)
        and "\"error\"" in m["content"]
    )
    operational_quality = max(0.0, 5.0 - min(5.0, invalid_tool_results * 0.5))

    naval_loop_penalty = 0.0
    if action_names.count("lookup_building") >= 3:
        naval_mentions = sum(
            1 for m in messages
            if isinstance(m.get("content"), str)
            and ("syrd" in m["content"].lower() or "naval" in m["content"].lower())
        )
        naval_loop_penalty = min(5.0, naval_mentions * 1.0)

    crash_penalty = 15.0 if encountered_agent_error else 0.0

    earned_score = (
        primary_result
        + lure_and_pincer
        + combat
        + camp_raid
        + pincer_pressure
        + han_survival
        + scouting
        + operational_quality
    )
    raw_penalty = naval_loop_penalty + crash_penalty
    # Provider failures and invalid loops should hurt, but not erase observed
    # survival/scouting/progress credit from a partial run.
    applied_penalty = min(raw_penalty, earned_score * 0.5)
    total = earned_score - applied_penalty
    total = round(_clamp(total, 0.0, 100.0), 1)

    return {
        "benchmark": "backwater-hanxin",
        "score": total,
        "max_score": 100,
        "components": {
            "primary_result": round(primary_result, 1),
            "lure_and_pincer": round(lure_and_pincer, 1),
            "camp_raid": round(camp_raid, 1),
            "combat_efficiency": round(combat, 1),
            "pincer_pressure": round(pincer_pressure, 1),
            "han_survival": round(han_survival, 1),
            "scouting": round(scouting, 1),
            "operational_quality": round(operational_quality, 1),
        },
        "penalties": {
            "invalid_tool_results": invalid_tool_results,
            "irrelevant_naval_loop": round(naval_loop_penalty, 1),
            "agent_error": round(crash_penalty, 1),
            "applied_total": round(applied_penalty, 1),
        },
        "metrics": {
            "win": win,
            "ticks": ticks,
            "kills_cost": kills_cost,
            "deaths_cost": deaths_cost,
            "kd_ratio": round(kd_ratio, 2),
            "assets_value": assets_value,
            "explored_percent": explored_percent,
            "buildings_killed": buildings_killed,
            "buildings_lost": buildings_lost,
            "own_units": own_units,
            "own_buildings": own_buildings,
        },
    }
