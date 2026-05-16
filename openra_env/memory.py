"""Cross-episode memory for LLM agents.

Implements the Experience-Reflection-Consolidation loop (inspired by ERL):
  1. During each game, track key events (builds, combat, losses) with timestamps
  2. After each game, collect stats + events and generate a structured reflection via LLM
  3. Store reflections persistently in memory.json
  4. Before each new game, inject relevant memories into the system prompt

Usage:
    memory = GameMemory(Path("~/.openra-rl/memory/").expanduser())
    context = memory.get_context(max_entries=5)  # inject into prompt

    tracker = EventTracker()
    tracker.record("first_barracks", tick=150, detail="barr built")
    # ... during game loop ...

    memory.add_episode(result="win", stats={...}, reflection="...", lessons=[...],
                       events=tracker.summary())
    memory.save()
"""

import json
from datetime import datetime, timezone
from pathlib import Path


class EventTracker:
    """Tracks key in-game events with tick timestamps for post-game reflection.

    Records milestones (first build, first combat, building losses) so the
    reflection prompt can produce actionable advice like
    "t=5000 才出坦克，应该在 t=3000 前出".
    """

    def __init__(self):
        self.events: list[dict] = []
        self._milestones: set[str] = set()  # track "first X" deduplication
        # Running state for delta detection
        self._prev_units_killed = 0
        self._prev_units_lost = 0
        self._prev_buildings_killed = 0
        self._prev_buildings_lost = 0
        self._prev_building_types: set[str] = set()
        self._prev_unit_types: set[str] = set()

    def record(self, event_type: str, tick: int, detail: str = ""):
        """Record a timestamped event."""
        self.events.append({
            "type": event_type,
            "tick": tick,
            "detail": detail,
        })

    def record_milestone(self, milestone: str, tick: int, detail: str = ""):
        """Record a 'first time' event (deduplicated)."""
        if milestone not in self._milestones:
            self._milestones.add(milestone)
            self.record(milestone, tick, detail)

    def update_from_state(self, state: dict):
        """Extract events by diffing current game state against previous.

        Call this every turn with the get_game_state result.
        """
        tick = state.get("tick", 0)
        mil = state.get("military", {})
        eco = state.get("economy", {})

        # Track first building of each type
        building_types = set(state.get("building_types", []))
        new_buildings = building_types - self._prev_building_types
        for btype in new_buildings:
            self.record_milestone(f"first_{btype}", tick, f"First building: {btype}")
        self._prev_building_types = building_types

        # Track first unit of each type
        units = state.get("units_summary", [])
        unit_types = {u["type"] for u in units}
        new_unit_types = unit_types - self._prev_unit_types
        for utype in new_unit_types:
            self.record_milestone(f"first_{utype}", tick, f"First unit: {utype}")
        self._prev_unit_types = unit_types

        # Combat events: kills delta
        kills = mil.get("units_killed", 0)
        losses = mil.get("units_lost", 0)
        bkills = mil.get("buildings_killed", 0)
        blosses = mil.get("buildings_lost", 0)

        if kills > self._prev_units_killed:
            delta = kills - self._prev_units_killed
            if self._prev_units_killed == 0:
                self.record_milestone("first_kill", tick, f"First enemy kill (×{delta})")
            elif delta >= 3:
                self.record("major_kill", tick, f"Killed {delta} enemy units")
        self._prev_units_killed = kills

        if losses > self._prev_units_lost:
            delta = losses - self._prev_units_lost
            if self._prev_units_lost == 0:
                self.record_milestone("first_loss", tick, f"First unit lost (×{delta})")
            elif delta >= 3:
                self.record("major_loss", tick, f"Lost {delta} own units")
        self._prev_units_lost = losses

        if bkills > self._prev_buildings_killed:
            delta = bkills - self._prev_buildings_killed
            self.record("enemy_building_destroyed", tick, f"Destroyed {delta} enemy building(s)")
        self._prev_buildings_killed = bkills

        if blosses > self._prev_buildings_lost:
            delta = blosses - self._prev_buildings_lost
            self.record("building_lost", tick, f"Lost {delta} own building(s)")
        self._prev_buildings_lost = blosses

        # Economy milestones
        cash = eco.get("cash", 0)
        if cash <= 100 and tick > 500:
            self.record_milestone("low_cash", tick, f"Low cash (${cash})")

    def update_from_tool_result(self, tool_name: str, args: dict, result: dict, tick: int):
        """Extract events from tool call results."""
        if not isinstance(result, dict):
            return

        if tool_name in ("build_unit", "build_structure", "build_and_place"):
            if "error" not in result:
                item = args.get("unit_type") or args.get("building_type") or args.get("type", "?")
                self.record_milestone(f"first_build_{item}", tick, f"First build/train: {item}")

        elif tool_name in ("attack_move", "attack_target"):
            units_n = len(result.get("commanded_units", []))
            if units_n > 0:
                self.record_milestone("first_attack_order", tick, f"First attack order ({units_n} units)")

    def summary(self) -> list[dict]:
        """Return sorted event list for storage/prompt injection."""
        return sorted(self.events, key=lambda e: e["tick"])

    def format_timeline(self, max_events: int = 20) -> str:
        """Format events as a readable timeline string for reflection prompt.

        Args:
            max_events: Cap output to this many events. When exceeded,
                milestone events (first_*) are kept over non-milestone events.
        """
        if not self.events:
            return ""
        events = self.summary()
        if len(events) > max_events:
            milestones = [e for e in events if e["type"].startswith("first_")]
            others = [e for e in events if not e["type"].startswith("first_")]
            remaining = max_events - len(milestones)
            if remaining > 0:
                events = milestones + others[:remaining]
            else:
                events = milestones[:max_events]
            events.sort(key=lambda e: e["tick"])
        lines = ["Key event timeline:"]
        for ev in events:
            tick = ev["tick"]
            game_time = f"~{tick // 25}s"
            lines.append(f"  t={tick} ({game_time}): {ev['detail']}")
        return "\n".join(lines)


class GameMemory:
    """Cross-episode memory manager.

    Stores game results, reflections, and lessons learned across episodes.
    Provides context injection for LLM system prompts.
    """

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.memory_file = memory_dir / "memory.json"
        self.episodes: list[dict] = []
        self.load()

    def load(self):
        """Load memory from disk. Creates directory if needed."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if self.memory_file.exists():
            try:
                raw = self.memory_file.read_text(encoding="utf-8")
                data = json.loads(raw)
                if isinstance(data, dict):
                    self.episodes = data.get("episodes", [])
                else:
                    self.episodes = []
            except (json.JSONDecodeError, KeyError, OSError, TypeError, AttributeError):
                self.episodes = []

    def save(self):
        """Persist memory to disk."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "updated": datetime.now(timezone.utc).isoformat(),
            "total_episodes": len(self.episodes),
            "episodes": self.episodes,
        }
        self.memory_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_episode(
        self,
        result: str,
        ticks: int,
        faction: str,
        opponent: str,
        stats: dict,
        reflection: str,
        lessons: list[str],
        events: list[dict] | None = None,
    ):
        """Record one episode's outcome and reflection."""
        episode = {
            "id": len(self.episodes) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": result,
            "ticks": ticks,
            "faction": faction,
            "opponent": opponent,
            "stats": stats,
            "reflection": reflection,
            "lessons": lessons,
        }
        if events:
            episode["events"] = events
        self.episodes.append(episode)

    def get_context(self, max_entries: int = 5) -> str:
        """Build memory context string for system prompt injection.

        Selection strategy:
          - Most recent 3 episodes (freshest experience)
          - Up to 2 best wins (successful strategies worth repeating)
        Deduplicates if a win is also in recents.
        """
        if not self.episodes:
            return ""

        # Recent episodes
        recents = self.episodes[-3:]
        recent_ids = {e["id"] for e in recents}

        # Best wins: lowest tick count (fastest victory)
        wins = [e for e in self.episodes if e["result"] == "win"]
        wins.sort(key=lambda e: e.get("ticks", 99999))
        best_wins = [e for e in wins if e["id"] not in recent_ids][:2]

        selected = recents + best_wins
        selected = selected[:max_entries]

        if not selected:
            return ""

        parts = [f"You have played {len(self.episodes)} games. Key lessons:\n"]

        # Win/loss summary
        total_wins = sum(1 for e in self.episodes if e["result"] == "win")
        total_losses = sum(1 for e in self.episodes if e["result"] == "lose")
        parts.append(f"Record: {total_wins}W {total_losses}L\n")

        for ep in selected:
            tag = "WIN" if ep["result"] == "win" else "LOSS"
            stats = ep.get("stats", {})
            kd = f"K{stats.get('units_killed', 0)}/D{stats.get('units_lost', 0)}"
            parts.append(
                f"--- Game #{ep['id']} [{tag}] {ep.get('faction', '?')} vs {ep.get('opponent', '?')} "
                f"| {ep.get('ticks', '?')}ticks | {kd} ---"
            )
            if ep.get("reflection"):
                parts.append(f"Analysis: {ep['reflection']}")
            if ep.get("lessons"):
                for i, lesson in enumerate(ep["lessons"], 1):
                    parts.append(f"  Lesson {i}: {lesson}")
            parts.append("")

        # Aggregate lessons from recent losses
        recent_losses = [e for e in self.episodes[-5:] if e["result"] == "lose"]
        if recent_losses:
            all_lessons = []
            for e in recent_losses:
                all_lessons.extend(e.get("lessons", []))
            if all_lessons:
                seen = set()
                unique = []
                for lesson in all_lessons:
                    if lesson not in seen:
                        seen.add(lesson)
                        unique.append(lesson)
                parts.append("Warning — recurring issues from recent losses:")
                for lesson in unique[:5]:
                    parts.append(f"  - {lesson}")

        return "\n".join(parts)

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def win_rate(self) -> float:
        if not self.episodes:
            return 0.0
        wins = sum(1 for e in self.episodes if e["result"] == "win")
        return wins / len(self.episodes)


def build_reflection_prompt(
    result: str,
    ticks: int,
    faction: str,
    opponent: str,
    stats: dict,
    planning_strategy: str = "",
    event_timeline: str = "",
) -> str:
    """Build the prompt that asks the LLM to reflect on a completed game.

    Args:
        event_timeline: Formatted string from EventTracker.format_timeline()
            containing timestamped in-game events for actionable analysis.
    """
    kd_ratio = stats.get("kills_cost", 0) / max(stats.get("deaths_cost", 1), 1)

    timeline_section = ""
    if event_timeline:
        timeline_section = f"""
{event_timeline}
"""

    return f"""Game over. Analyze this match.

Result: {result.upper()}
Faction: {faction} vs {opponent}
Duration: {ticks} ticks
Kills: {stats.get('units_killed', 0)} units (value ${stats.get('kills_cost', 0)})
Losses: {stats.get('units_lost', 0)} units (value ${stats.get('deaths_cost', 0)})
K/D value ratio: {kd_ratio:.2f}
Buildings destroyed: {stats.get('buildings_killed', 0)}
Buildings lost: {stats.get('buildings_lost', 0)}
Army value: ${stats.get('army_value', 0)}
Cash remaining: ${stats.get('cash_remaining', 0)}
{f'Pre-game strategy: {planning_strategy}' if planning_strategy else ''}{timeline_section}
Answer:
1. Using the timeline above, analyze key decision points and timing issues (e.g. "tanks didn't come until t=5000, too late")
2. List 1-3 specific actionable lessons with concrete timing targets (e.g. "must have tanks before t=3000")

Required format:
Analysis: <your analysis, referencing specific tick timestamps>
Lesson1: <specific lesson with timing target>
Lesson2: <specific lesson with timing target>
Lesson3: <specific lesson (optional)>"""


def parse_reflection_response(text: str | None) -> tuple[str, list[str]]:
    """Parse the LLM's reflection response into reflection text and lessons.

    Returns:
        (reflection_text, lessons_list)
    """
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    reflection = ""
    lessons = []

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("analysis:"):
            reflection = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("lesson"):
            parts = line.split(":", 1)
            if len(parts) > 1:
                lessons.append(parts[1].strip())

    # Fallback: if parsing failed, use the whole text as reflection
    if not reflection and text.strip():
        reflection = text.strip()[:300]

    return reflection, lessons
