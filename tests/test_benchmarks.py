"""Tests for named benchmark presets."""

from openra_env.benchmarks import benchmark_ids, get_benchmark


def test_backwater_hanxin_preset():
    preset = get_benchmark("backwater-hanxin")

    assert preset.map_name == "backwater-battle-hanxin"
    assert preset.bot_type == "beginner"
    assert preset.agent_type == "LLM"
    assert preset.max_time_s == 300
    assert preset.config_overrides()["agent"]["max_time_s"] == 300
    assert preset.reset_options() == {
        "map_name": "backwater-battle-hanxin",
        "bot_type": "beginner",
    }


def test_benchmark_ids():
    assert "backwater-hanxin" in benchmark_ids()
