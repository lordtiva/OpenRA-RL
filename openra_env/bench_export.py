"""Build bench export JSON from a final game observation.

Custom agents that use OpenRAEnv directly (CNN, RL, multi-agent, etc.)
can call build_bench_export() after their game loop to produce a bench
submission JSON — the same format the built-in LLM agent auto-exports.

Usage:
    from openra_env.bench_export import build_bench_export

    obs = await env.step(action)  # final observation (obs.done == True)
    export = build_bench_export(
        obs,
        agent_name="DeathBot-9000",
        agent_type="RL",
        opponent="Normal",
    )
    print(f"Saved to {export['path']}")

    # Then submit:
    #   openra-rl bench submit <path>
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from openra_env.backwater_rubric import compute_backwater_score


def build_bench_export(
    obs: Any,
    agent_name: str,
    agent_type: str = "RL",
    opponent: str = "Normal",
    agent_url: str = "",
    replay_path: str = "",
    hf_token: str = "",
    export_dir: Optional[Path] = None,
    benchmark: str = "",
    events: Optional[list[dict[str, Any]]] = None,
    messages: Optional[list[dict[str, Any]]] = None,
    encountered_agent_error: bool = False,
) -> Dict[str, Any]:
    """Build and save a bench export JSON from a final observation.

    Args:
        obs: Final observation — either a dict or a Pydantic model with
             .military, .economy, .tick, .result, .explored_percent attributes.
        agent_name: Display name for the leaderboard.
        agent_type: One of "Scripted", "LLM", "RL".
        opponent: Difficulty tier (Beginner/Easy/Medium/Normal/Hard).
        agent_url: Optional GitHub/project URL.
        replay_path: Optional path to .orarep replay file.
        hf_token: Optional HuggingFace token for verified bench submissions.
        export_dir: Where to save the JSON (default: ~/.openra-rl/bench-exports/).
        benchmark: Optional benchmark id. Use "backwater-hanxin" to attach
            the Backwater rubric score.
        events: Optional event timeline from the run.
        messages: Optional LLM/tool trace from the run.
        encountered_agent_error: Whether the run hit an agent runtime error.

    Returns:
        Dict with all submission fields plus "path" pointing to the saved file.
    """
    # Normalize obs to dict
    if hasattr(obs, "model_dump"):
        obs_dict = obs.model_dump()
    elif hasattr(obs, "__dict__") and not isinstance(obs, dict):
        obs_dict = vars(obs)
    else:
        obs_dict = dict(obs)

    mil = obs_dict.get("military") or {}
    kills = mil.get("kills_cost", 0)
    deaths = mil.get("deaths_cost", 0)

    sub = {
        "agent_name": agent_name,
        "agent_type": agent_type,
        "agent_url": agent_url,
        "opponent": opponent,
        "games": 1,
        "result": obs_dict.get("result", ""),
        "win": obs_dict.get("result") == "win",
        "ticks": obs_dict.get("tick", 0),
        "kills_cost": kills,
        "deaths_cost": deaths,
        "kd_ratio": round(kills / max(deaths, 1), 2),
        "assets_value": mil.get("assets_value", 0),
        "explored_percent": obs_dict.get("explored_percent", 0),
        "reward_vector": obs_dict.get("reward_vector", {}),
        "replay_path": replay_path,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if benchmark:
        sub["benchmark"] = benchmark
    if benchmark == "backwater-hanxin":
        rubric = compute_backwater_score(
            result=obs_dict.get("result", ""),
            ticks=obs_dict.get("tick", 0),
            kills_cost=kills,
            deaths_cost=deaths,
            assets_value=mil.get("assets_value", 0),
            explored_percent=obs_dict.get("explored_percent", 0),
            buildings_killed=mil.get("buildings_killed", 0),
            buildings_lost=mil.get("buildings_lost", 0),
            own_units=obs_dict.get("own_units", len(obs_dict.get("units", []))),
            own_buildings=obs_dict.get("own_buildings", len(obs_dict.get("buildings", []))),
            events=events or [],
            messages=messages or [],
            encountered_agent_error=encountered_agent_error,
        )
        sub["backwater_score"] = rubric["score"]
        sub["backwater_rubric"] = rubric
    if hf_token:
        sub["hf_token"] = hf_token

    # Save to disk
    if export_dir is None:
        export_dir = Path.home() / ".openra-rl" / "bench-exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = agent_name.replace("/", "_").replace(" ", "_")[:40]
    export_path = export_dir / f"bench-{slug}-{ts}.json"
    export_path.write_text(json.dumps(sub, indent=2))

    sub["path"] = str(export_path)
    return sub
