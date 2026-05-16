from openra_env.backwater_rubric import compute_backwater_score
from openra_env.bench_export import build_bench_export


def test_backwater_rubric_gives_positive_partial_score():
    rubric = compute_backwater_score(
        result="loss",
        ticks=6000,
        kills_cost=5000,
        deaths_cost=3000,
        assets_value=4000,
        explored_percent=35,
        buildings_killed=1,
        own_units=6,
        own_buildings=4,
        events=[{"type": "first_attack_order", "tick": 1000, "detail": "attack"}],
    )

    assert rubric["score"] > 0
    assert rubric["components"]["lure_and_pincer"] > 0
    assert rubric["components"]["camp_raid"] > 0
    assert rubric["metrics"]["kd_ratio"] > 1.0


def test_backwater_rubric_provider_error_does_not_erase_progress():
    rubric = compute_backwater_score(
        result="",
        ticks=244,
        kills_cost=0,
        deaths_cost=0,
        assets_value=11300,
        explored_percent=93.8,
        own_units=12,
        own_buildings=7,
        encountered_agent_error=True,
    )

    assert rubric["score"] > 0
    assert rubric["components"]["han_survival"] > 0
    assert rubric["components"]["scouting"] > 0
    assert rubric["penalties"]["applied_total"] < rubric["penalties"]["agent_error"]


def test_build_bench_export_attaches_backwater_score(tmp_path):
    export = build_bench_export(
        {
            "result": "win",
            "tick": 5000,
            "military": {
                "kills_cost": 12000,
                "deaths_cost": 4000,
                "assets_value": 8000,
                "buildings_killed": 3,
                "buildings_lost": 1,
            },
            "explored_percent": 55,
            "own_units": 8,
            "own_buildings": 5,
        },
        agent_name="Mercury",
        agent_type="LLM",
        opponent="Beginner",
        benchmark="backwater-hanxin",
        export_dir=tmp_path,
    )

    assert export["benchmark"] == "backwater-hanxin"
    assert export["backwater_score"] > 0
    assert export["backwater_rubric"]["score"] == export["backwater_score"]
