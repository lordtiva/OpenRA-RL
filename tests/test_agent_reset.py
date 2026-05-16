"""Tests for LLM agent environment reset configuration."""

import asyncio
import tempfile
from unittest.mock import AsyncMock, patch


from openra_env.config import OpenRARLConfig


def test_parse_tool_call_handles_missing_function():
    from openra_env.agent import _parse_tool_call

    fn_name, fn_args, tool_call_id, error = _parse_tool_call({
        "id": "bad-call",
        "type": "function",
        "function": None,
    })

    assert fn_name is None
    assert fn_args == {}
    assert tool_call_id == "bad-call"
    assert "missing function.name" in error["error"]


def test_parse_tool_call_unwraps_empty_key_arguments():
    from openra_env.agent import _parse_tool_call

    fn_name, fn_args, _, error = _parse_tool_call({
        "id": "map-analysis",
        "type": "function",
        "function": {
            "name": "get_map_analysis",
            "arguments": "{\"\": {}}",
        },
    })

    assert error is None
    assert fn_name == "get_map_analysis"
    assert fn_args == {}


def test_assistant_choice_normalizes_none_content():
    from openra_env.agent import _assistant_choice

    _, message = _assistant_choice({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": None,
            }
        }]
    })

    assert message["content"] == ""
    assert message["tool_calls"] == []


def test_plain_json_actions_recovered_as_batch_tool_call():
    from openra_env.agent import _parse_tool_call, _tool_calls_from_plain_json_actions

    tool_calls = _tool_calls_from_plain_json_actions(
        '{"actions":[{"tool":"attack_move","unit_ids":"all_combat","target_x":49,"target_y":27}]}'
    )

    assert len(tool_calls) == 1
    fn_name, fn_args, _, error = _parse_tool_call(tool_calls[0])
    assert error is None
    assert fn_name == "batch"
    assert fn_args == {
        "actions": [{
            "tool": "attack_move",
            "unit_ids": "all_combat",
            "target_x": 49,
            "target_y": 27,
        }]
    }


def test_recovered_tool_calls_attached_to_history():
    from openra_env.agent import _attach_recovered_tool_calls_to_last_message

    tool_calls = [{
        "id": "plain-json-actions",
        "type": "function",
        "function": {"name": "batch", "arguments": "{\"actions\": []}"},
    }]
    messages = [{"role": "assistant", "content": "{\"actions\": []}"}]
    trace = [{"role": "assistant", "content": "{\"actions\": []}"}]

    _attach_recovered_tool_calls_to_last_message(messages, trace, tool_calls)

    assert messages[-1]["tool_calls"] == tool_calls
    assert trace[-1]["tool_calls"] == tool_calls


def test_repair_batch_args_recovers_actions_from_content():
    from openra_env.agent import _repair_batch_args

    args, error = _repair_batch_args(
        "batch",
        {},
        '{"actions":[{"tool":"build_unit","unit_type":"1tnk","count":1}]}',
    )

    assert error is None
    assert args == {"actions": [{"tool": "build_unit", "unit_type": "1tnk", "count": 1}]}


def test_repair_batch_args_rejects_empty_batch():
    from openra_env.agent import _repair_batch_args

    args, error = _repair_batch_args("batch", {}, "")

    assert args == {}
    assert "missing required 'actions' list" in error["error"]
    assert "expected_shape" in error


def test_prevalidate_assign_group_rejects_empty_args():
    from openra_env.agent import _prevalidate_tool_args

    error = _prevalidate_tool_args("assign_group", {})

    assert "missing 'group_name' and 'unit_ids'" in error["error"]
    assert error["expected_shape"]["group_name"] == "raiders"


def test_should_auto_advance_for_action_tool():
    from openra_env.agent import _should_auto_advance

    assert _should_auto_advance("attack_move", {"tick": 10, "done": False})
    assert _should_auto_advance("batch", {"tick": 10, "done": False})
    assert not _should_auto_advance("get_map_analysis", {"tick": 10, "done": False})
    assert not _should_auto_advance("attack_move", {"error": "bad"})


def test_should_advance_after_query_only_turn():
    from openra_env.agent import _should_advance_after_turn

    assert _should_advance_after_turn([{"function": {"name": "get_map_analysis"}}], False, False)
    assert not _should_advance_after_turn([{"function": {"name": "attack_move"}}], True, False)
    assert not _should_advance_after_turn([], False, False)
    assert not _should_advance_after_turn([{"function": {"name": "get_map_analysis"}}], False, True)


def test_run_agent_resets_with_configured_map_and_bot():
    from openra_env import agent

    config = OpenRARLConfig()
    config.game.map_name = "backwater-battle-hanxin"
    config.opponent.bot_type = "beginner"
    config.llm.base_url = "http://localhost:11434/v1/chat/completions"
    config.llm.api_key = ""
    config.llm.model = "local-model"
    config.agent.max_turns = 1
    config.agent.max_time_s = 0
    config.agent.bench_upload = False
    config.agent.memory_enabled = False

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.reset = AsyncMock()
    client.list_tools = AsyncMock(return_value=[])
    game_state = {
        "done": False,
        "result": "",
        "tick": 0,
        "map": {"map_name": "backwater-battle-hanxin"},
        "units_summary": [],
        "faction": "england",
        "military": {},
        "economy": {},
    }

    async def call_tool(name, **kwargs):
        if name == "get_planning_status":
            return {"planning_enabled": False}
        if name == "get_game_state":
            return game_state
        if name == "surrender":
            return {"done": True, "result": "loss", "tick": 1}
        return {}

    client.call_tool = AsyncMock(side_effect=call_tool)

    with tempfile.TemporaryDirectory() as temp_home:
        with (
            patch.dict("os.environ", {"USERPROFILE": temp_home, "HOME": temp_home}),
            patch("openra_env.agent.OpenRAMCPClient", return_value=client),
            patch("openra_env.agent.load_system_prompt", return_value="system"),
            patch("openra_env.agent.compose_pregame_briefing", return_value="briefing"),
            patch(
                "openra_env.agent.chat_completion",
                AsyncMock(return_value={
                    "choices": [{
                        "message": {"role": "assistant", "content": "done", "tool_calls": []},
                        "finish_reason": "stop",
                    }]
                }),
            ),
        ):
            asyncio.run(agent.run_agent(config))

    client.reset.assert_awaited_once_with(
        map_name="backwater-battle-hanxin",
        bot_type="beginner",
    )


def test_run_agent_handles_malformed_tool_call():
    from openra_env import agent

    config = OpenRARLConfig()
    config.game.map_name = "backwater-battle-hanxin"
    config.opponent.bot_type = "beginner"
    config.llm.base_url = "http://localhost:11434/v1/chat/completions"
    config.llm.api_key = ""
    config.llm.model = "local-model"
    config.agent.max_turns = 1
    config.agent.max_time_s = 0
    config.agent.bench_upload = False
    config.agent.memory_enabled = False

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.reset = AsyncMock()
    client.list_tools = AsyncMock(return_value=[])
    game_state = {
        "done": False,
        "result": "",
        "tick": 0,
        "map": {"map_name": "backwater-battle-hanxin"},
        "units_summary": [],
        "faction": "england",
        "military": {},
        "economy": {},
    }

    async def call_tool(name, **kwargs):
        if name == "get_planning_status":
            return {"planning_enabled": False}
        if name == "get_game_state":
            return game_state
        if name == "surrender":
            return {"done": True, "result": "loss", "tick": 1}
        return {}

    client.call_tool = AsyncMock(side_effect=call_tool)

    with tempfile.TemporaryDirectory() as temp_home:
        with (
            patch.dict("os.environ", {"USERPROFILE": temp_home, "HOME": temp_home}),
            patch("openra_env.agent.OpenRAMCPClient", return_value=client),
            patch("openra_env.agent.load_system_prompt", return_value="system"),
            patch("openra_env.agent.compose_pregame_briefing", return_value="briefing"),
            patch(
                "openra_env.agent.chat_completion",
                AsyncMock(return_value={
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{"id": "bad-call", "type": "function", "function": None}],
                        },
                        "finish_reason": "tool_calls",
                    }]
                }),
            ),
        ):
            asyncio.run(agent.run_agent(config))

    called_tools = [call.args[0] for call in client.call_tool.await_args_list if call.args]
    assert "surrender" in called_tools


def test_run_agent_handles_malformed_planning_tool_call():
    from openra_env import agent

    config = OpenRARLConfig()
    config.game.map_name = "backwater-battle-hanxin"
    config.opponent.bot_type = "beginner"
    config.llm.base_url = "http://localhost:11434/v1/chat/completions"
    config.llm.api_key = ""
    config.llm.model = "local-model"
    config.planning.max_turns = 1
    config.agent.max_turns = 1
    config.agent.max_time_s = 0
    config.agent.bench_upload = False
    config.agent.memory_enabled = False

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.reset = AsyncMock()
    client.list_tools = AsyncMock(return_value=[])
    game_state = {
        "done": False,
        "result": "",
        "tick": 0,
        "map": {"map_name": "backwater-battle-hanxin", "width": 64, "height": 64},
        "base_position": {"x": 16, "y": 43},
        "enemy_estimated_position": {"x": 50, "y": 24},
        "your_faction": "allies",
        "your_side": "Han",
        "units_summary": [],
        "faction": "england",
        "military": {},
        "economy": {},
    }

    async def call_tool(name, **kwargs):
        if name == "get_planning_status":
            return {"planning_enabled": True, "planning_active": False}
        if name == "start_planning_phase":
            return {
                "planning_active": True,
                "max_turns": 1,
                "map": {"map_name": "backwater-battle-hanxin", "width": 64, "height": 64},
                "base_position": {"x": 16, "y": 43},
                "enemy_estimated_position": {"x": 50, "y": 24},
                "your_faction": "allies",
                "your_side": "Han",
                "opponent_summary": "Beginner Zhao AI",
            }
        if name == "end_planning_phase":
            return {"planning_complete": True, "strategy": "forced"}
        if name == "get_game_state":
            return game_state
        if name == "surrender":
            return {"done": True, "result": "loss", "tick": 1}
        return {}

    client.call_tool = AsyncMock(side_effect=call_tool)
    responses = [
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "bad-plan", "type": "function", "function": None}],
                },
                "finish_reason": "tool_calls",
            }]
        },
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "end-plan",
                        "type": "function",
                        "function": {
                            "name": "end_planning_phase",
                            "arguments": "{\"strategy\":\"forced\"}",
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }]
        },
        {
            "choices": [{
                "message": {"role": "assistant", "content": "done", "tool_calls": []},
                "finish_reason": "stop",
            }]
        },
    ]

    with tempfile.TemporaryDirectory() as temp_home:
        with (
            patch.dict("os.environ", {"USERPROFILE": temp_home, "HOME": temp_home}),
            patch("openra_env.agent.OpenRAMCPClient", return_value=client),
            patch("openra_env.agent.load_system_prompt", return_value="system"),
            patch("openra_env.agent.compose_pregame_briefing", return_value="briefing"),
            patch("openra_env.agent.chat_completion", AsyncMock(side_effect=responses)),
        ):
            asyncio.run(agent.run_agent(config))

    called_tools = [call.args[0] for call in client.call_tool.await_args_list if call.args]
    assert "surrender" in called_tools
