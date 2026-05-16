"""Named scenario benchmark presets for OpenRA-RL."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkPreset:
    """Configuration overrides for a named scenario benchmark."""

    id: str
    name: str
    map_name: str | None = None
    bot_type: str | None = None
    agent_type: str = "LLM"
    max_time_s: int | None = None
    description: str = ""

    def config_overrides(self) -> dict:
        overrides: dict = {"agent": {"agent_type": self.agent_type}}
        if self.max_time_s is not None:
            overrides["agent"]["max_time_s"] = self.max_time_s
        if self.map_name:
            overrides["game"] = {"map_name": self.map_name}
        if self.bot_type:
            overrides["opponent"] = {"bot_type": self.bot_type}
        return overrides

    def reset_options(self) -> dict[str, str]:
        options: dict[str, str] = {}
        if self.map_name:
            options["map_name"] = self.map_name
        if self.bot_type:
            options["bot_type"] = self.bot_type
        return options


BENCHMARKS: dict[str, BenchmarkPreset] = {
    "backwater-hanxin": BenchmarkPreset(
        id="backwater-hanxin",
        name="Backwater Battle - Han Xin",
        map_name="backwater-battle-hanxin",
        bot_type="beginner",
        max_time_s=300,
        description=(
            "LLM controls Han in a five-minute lure-and-pincer scenario: "
            "a river-backed bait force pulls Zhao into the pass while hidden "
            "rear raiders strike the emptied Zhao camp."
        ),
    ),
}


def get_benchmark(benchmark_id: str) -> BenchmarkPreset:
    try:
        return BENCHMARKS[benchmark_id]
    except KeyError as e:
        valid = ", ".join(sorted(BENCHMARKS))
        raise ValueError(f"Unknown benchmark '{benchmark_id}'. Valid options: {valid}") from e


def benchmark_ids() -> list[str]:
    return sorted(BENCHMARKS)
