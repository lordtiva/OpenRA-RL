"""Tests for cross-episode memory and in-game event tracking."""

import json
import pytest
from pathlib import Path

from openra_env.memory import (
    EventTracker,
    GameMemory,
    build_reflection_prompt,
    parse_reflection_response,
)


class TestEventTracker:
    """Tests for in-game event tracking via state diffing."""

    def test_first_building_milestone(self):
        tracker = EventTracker()
        state = {"tick": 100, "building_types": ["powr"], "units_summary": [],
                 "military": {}, "economy": {"cash": 5000}}
        tracker.update_from_state(state)
        assert len(tracker.events) == 1
        assert tracker.events[0]["type"] == "first_powr"
        assert tracker.events[0]["tick"] == 100

    def test_first_unit_milestone(self):
        tracker = EventTracker()
        state = {"tick": 200, "building_types": [],
                 "units_summary": [{"type": "e1"}],
                 "military": {}, "economy": {"cash": 5000}}
        tracker.update_from_state(state)
        assert any(e["type"] == "first_e1" for e in tracker.events)

    def test_milestone_deduplication(self):
        tracker = EventTracker()
        state = {"tick": 100, "building_types": ["powr"], "units_summary": [],
                 "military": {}, "economy": {"cash": 5000}}
        tracker.update_from_state(state)
        tracker.update_from_state(state)  # same state again
        powr_events = [e for e in tracker.events if e["type"] == "first_powr"]
        assert len(powr_events) == 1

    def test_kill_tracking(self):
        tracker = EventTracker()
        state1 = {"tick": 100, "building_types": [], "units_summary": [],
                  "military": {"units_killed": 0, "units_lost": 0,
                               "buildings_killed": 0, "buildings_lost": 0},
                  "economy": {"cash": 5000}}
        tracker.update_from_state(state1)

        state2 = {**state1, "tick": 200,
                  "military": {"units_killed": 3, "units_lost": 0,
                               "buildings_killed": 0, "buildings_lost": 0}}
        tracker.update_from_state(state2)
        assert any(e["type"] == "first_kill" for e in tracker.events)

    def test_loss_tracking(self):
        tracker = EventTracker()
        state1 = {"tick": 100, "building_types": [], "units_summary": [],
                  "military": {"units_killed": 0, "units_lost": 0,
                               "buildings_killed": 0, "buildings_lost": 0},
                  "economy": {"cash": 5000}}
        tracker.update_from_state(state1)

        state2 = {**state1, "tick": 300,
                  "military": {"units_killed": 0, "units_lost": 4,
                               "buildings_killed": 0, "buildings_lost": 0}}
        tracker.update_from_state(state2)
        assert any(e["type"] == "first_loss" for e in tracker.events)

    def test_low_cash_milestone(self):
        tracker = EventTracker()
        state = {"tick": 1000, "building_types": [], "units_summary": [],
                 "military": {}, "economy": {"cash": 50}}
        tracker.update_from_state(state)
        assert any(e["type"] == "low_cash" for e in tracker.events)

    def test_low_cash_not_early(self):
        """Low cash before tick 500 should not trigger."""
        tracker = EventTracker()
        state = {"tick": 100, "building_types": [], "units_summary": [],
                 "military": {}, "economy": {"cash": 50}}
        tracker.update_from_state(state)
        assert not any(e["type"] == "low_cash" for e in tracker.events)

    def test_tool_result_build_success(self):
        tracker = EventTracker()
        result = {"tick": 500, "note": "Queued 1x e1"}  # no "error" key = success
        tracker.update_from_tool_result("build_unit", {"unit_type": "e1"}, result, 500)
        assert any(e["type"] == "first_build_e1" for e in tracker.events)

    def test_tool_result_build_failure(self):
        tracker = EventTracker()
        result = {"error": "Not enough funds", "tick": 500}
        tracker.update_from_tool_result("build_unit", {"unit_type": "e1"}, result, 500)
        assert not any("first_build" in e["type"] for e in tracker.events)

    def test_tool_result_attack(self):
        tracker = EventTracker()
        result = {"tick": 600, "commanded_units": [{"id": 1}, {"id": 2}]}
        tracker.update_from_tool_result("attack_move", {}, result, 600)
        assert any(e["type"] == "first_attack_order" for e in tracker.events)

    def test_format_timeline_empty(self):
        tracker = EventTracker()
        assert tracker.format_timeline() == ""

    def test_format_timeline_capped(self):
        tracker = EventTracker()
        for i in range(30):
            tracker.record(f"event_{i}", tick=i * 100, detail=f"Event {i}")
        timeline = tracker.format_timeline(max_events=10)
        lines = timeline.strip().split("\n")
        assert len(lines) <= 11  # header + 10 events

    def test_summary_sorted(self):
        tracker = EventTracker()
        tracker.record("b", tick=200, detail="second")
        tracker.record("a", tick=100, detail="first")
        summary = tracker.summary()
        assert summary[0]["tick"] == 100
        assert summary[1]["tick"] == 200


class TestGameMemory:
    """Tests for cross-episode memory persistence."""

    def test_save_and_load(self, tmp_path):
        mem = GameMemory(tmp_path)
        mem.add_episode(
            result="lose", ticks=5000, faction="russia", opponent="easy",
            stats={"units_killed": 3, "units_lost": 5, "kills_cost": 1000,
                   "deaths_cost": 2000, "buildings_killed": 0,
                   "buildings_lost": 1, "army_value": 500, "cash_remaining": 100},
            reflection="Too slow", lessons=["Build faster"],
        )
        mem.save()

        mem2 = GameMemory(tmp_path)
        assert mem2.episode_count == 1
        assert mem2.episodes[0]["result"] == "lose"
        assert mem2.episodes[0]["lessons"] == ["Build faster"]

    def test_win_rate(self, tmp_path):
        mem = GameMemory(tmp_path)
        for result in ["win", "lose", "win"]:
            mem.add_episode(
                result=result, ticks=5000, faction="russia", opponent="easy",
                stats={}, reflection="", lessons=[],
            )
        assert mem.win_rate == pytest.approx(2 / 3)

    def test_get_context_empty(self, tmp_path):
        mem = GameMemory(tmp_path)
        assert mem.get_context() == ""

    def test_get_context_with_episodes(self, tmp_path):
        mem = GameMemory(tmp_path)
        mem.add_episode(
            result="lose", ticks=5000, faction="russia", opponent="easy",
            stats={"units_killed": 3, "units_lost": 5},
            reflection="Bad timing", lessons=["Be faster"],
        )
        ctx = mem.get_context()
        assert "1 games" in ctx or "1W" in ctx or "0W 1L" in ctx
        assert "Bad timing" in ctx

    def test_load_corrupt_json(self, tmp_path):
        """Corrupt memory.json should not crash."""
        mem_file = tmp_path / "memory.json"
        mem_file.write_text("not valid json", encoding="utf-8")
        mem = GameMemory(tmp_path)
        assert mem.episode_count == 0

    def test_load_non_dict_json(self, tmp_path):
        """JSON that is a list instead of dict should not crash."""
        mem_file = tmp_path / "memory.json"
        mem_file.write_text("[1, 2, 3]", encoding="utf-8")
        mem = GameMemory(tmp_path)
        assert mem.episode_count == 0

    def test_events_stored(self, tmp_path):
        mem = GameMemory(tmp_path)
        events = [{"type": "first_powr", "tick": 100, "detail": "First building: powr"}]
        mem.add_episode(
            result="lose", ticks=5000, faction="russia", opponent="easy",
            stats={}, reflection="", lessons=[], events=events,
        )
        assert mem.episodes[0]["events"] == events


class TestReflectionPrompt:
    """Tests for reflection prompt builder and parser."""

    def test_build_prompt_contains_stats(self):
        prompt = build_reflection_prompt(
            result="lose", ticks=5000, faction="russia", opponent="easy",
            stats={"units_killed": 3, "units_lost": 5, "kills_cost": 1000,
                   "deaths_cost": 2000, "buildings_killed": 0,
                   "buildings_lost": 1, "army_value": 500, "cash_remaining": 100},
        )
        assert "LOSE" in prompt
        assert "russia" in prompt
        assert "5000" in prompt

    def test_build_prompt_with_timeline(self):
        prompt = build_reflection_prompt(
            result="win", ticks=3000, faction="england", opponent="hard",
            stats={}, event_timeline="Key event timeline:\n  t=100: First building",
        )
        assert "Key event timeline" in prompt

    def test_parse_response_standard(self):
        text = """Analysis: The game was lost due to slow tank production.
Lesson1: Must have tanks before t=3000.
Lesson2: Build refinery by t=1000."""
        reflection, lessons = parse_reflection_response(text)
        assert "slow tank" in reflection
        assert len(lessons) == 2
        assert "t=3000" in lessons[0]

    def test_parse_response_fallback(self):
        text = "Some unstructured analysis text without proper format."
        reflection, lessons = parse_reflection_response(text)
        assert reflection != ""  # falls back to raw text
        assert lessons == []

    def test_parse_response_empty(self):
        reflection, lessons = parse_reflection_response("")
        assert reflection == ""
        assert lessons == []

    def test_parse_response_none(self):
        reflection, lessons = parse_reflection_response(None)
        assert reflection == ""
        assert lessons == []
