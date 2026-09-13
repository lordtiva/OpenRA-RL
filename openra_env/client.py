"""OpenRA-RL environment client.

Provides the EnvClient subclass for connecting to the OpenRA-RL
environment server over WebSocket.

Sobre advance():
  El modo macro (rollout v4) avanza el bloque via el tool MCP `advance`
  del server (openra_env/server/openra_environment.py:1427), hablando
  JSON-RPC MCP sobre LA MISMA conexion /ws que reset/step (cada conexion
  es su propia sesion de juego). Devuelve el resumen con
  interrupted/interrupt_reason/actual_ticks_advanced.
"""

import asyncio
import os
from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from websockets.asyncio.client import connect as ws_connect

from openra_env.mcp_ws_client import OpenRAMCPClient


def _fast_advance_deadline_s() -> float:
    """Match C# / bridge_client OPENRA_RL_FAST_ADVANCE_DEADLINE_S (default 90)."""
    raw = os.environ.get("OPENRA_RL_FAST_ADVANCE_DEADLINE_S", "")
    try:
        val = float(raw) if str(raw).strip() else 90.0
    except (TypeError, ValueError):
        val = 90.0
    return max(5.0, val)


def default_op_message_timeout_s() -> float:
    """Per-advance/step WS recv timeout: deadline + slack, fail-fast on hung call.

    Episode wall-time is many advances; a single stuck dialogue must not sit
    for max_steps*3 (~3000s).
    """
    return _fast_advance_deadline_s() + 20.0

from openra_env.models import (
    BuildingInfoModel,
    EconomyInfo,
    MapInfoModel,
    MilitaryInfo,
    OpenRAAction,
    OpenRAObservation,
    OpenRAState,
    ProductionInfoModel,
    UnitInfoModel,
)


class OpenRAEnv(EnvClient[OpenRAAction, OpenRAObservation, OpenRAState]):
    """WebSocket client for the OpenRA-RL environment.

    Usage:
        async with OpenRAEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            while not result.done:
                action = OpenRAAction(commands=[...])
                result = await env.step(action)
    """

    async def connect(self) -> "OpenRAEnv":
        """Connect with ping keepalive disabled.

        OpenRA operations (especially reset) can take 60-120+ seconds
        with software rendering. The default websockets ping_interval=20s
        would kill the connection before the server responds.
        """
        if self._ws is not None:
            return self

        ws_url_lower = self._ws_url.lower()
        is_localhost = "localhost" in ws_url_lower or "127.0.0.1" in ws_url_lower

        old_no_proxy = os.environ.get("NO_PROXY")
        if is_localhost:
            current_no_proxy = old_no_proxy or ""
            if "localhost" not in current_no_proxy.lower():
                os.environ["NO_PROXY"] = (
                    f"{current_no_proxy},localhost,127.0.0.1"
                    if current_no_proxy
                    else "localhost,127.0.0.1"
                )

        try:
            self._ws = await ws_connect(
                self._ws_url,
                open_timeout=self._connect_timeout,
                max_size=50*1024*1024,  # 50MB
                ping_interval=None,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self._ws_url}: {e}") from e
        finally:
            if is_localhost:
                if old_no_proxy is None:
                    os.environ.pop("NO_PROXY", None)
                else:
                    os.environ["NO_PROXY"] = old_no_proxy

        return self

    def _step_payload(self, action: OpenRAAction) -> Dict[str, Any]:
        """Convert action to JSON for WebSocket transport."""
        return action.model_dump()

    def _parse_result(self, data: Dict[str, Any]) -> StepResult[OpenRAObservation]:
        """Parse server response into StepResult."""
        obs_data = data.get("observation", data)

        meta = dict(obs_data.get("metadata") or {})
        # peer may also sit at top-level of observation payload
        if "peer" in obs_data and "peer" not in meta:
            meta["peer"] = obs_data["peer"]
        observation = OpenRAObservation(
            tick=obs_data.get("tick", 0),
            economy=EconomyInfo(**obs_data.get("economy", {})),
            military=MilitaryInfo(**obs_data.get("military", {})),
            units=[UnitInfoModel(**u) for u in obs_data.get("units", [])],
            buildings=[BuildingInfoModel(**b) for b in obs_data.get("buildings", [])],
            production=[ProductionInfoModel(**p) for p in obs_data.get("production", [])],
            visible_enemies=[UnitInfoModel(**u) for u in obs_data.get("visible_enemies", [])],
            visible_enemy_buildings=[BuildingInfoModel(**b) for b in obs_data.get("visible_enemy_buildings", [])],
            map_info=MapInfoModel(**obs_data.get("map_info", {})),
            available_production=obs_data.get("available_production", []),
            ready_support_powers=obs_data.get("ready_support_powers", []),
            player_faction=obs_data.get("player_faction", "") or meta.get("player_faction", ""),
            enemy_faction=obs_data.get("enemy_faction", "") or meta.get("enemy_faction", ""),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward"),
            result=obs_data.get("result", ""),
            spatial_map=obs_data.get("spatial_map", ""),
            spatial_channels=obs_data.get("spatial_channels", 0),
            peer=obs_data.get("peer") or meta.get("peer"),
            metadata=meta,
        )

        return StepResult(
            observation=observation,
            reward=data.get("reward", obs_data.get("reward")),
            done=data.get("done", obs_data.get("done", False)),
        )

    async def _force_drop_ws(self, join_s: float = 1.0) -> None:
        """Clear _ws and best-effort abort so a wedged send can unblock.

        asyncio.wait_for cancel alone is not enough: if ws.send ignores
        CancelledError, wait_for hangs forever. Drop the socket first.
        Bound close/abort to ~0.5-1s; never wait unbounded.
        """
        ws = getattr(self, "_ws", None)
        self._ws = None
        if ws is None:
            return
        # Abort transport first — close() can also hang on a wedged peer.
        for attr in ("transport", "_transport"):
            tr = getattr(ws, attr, None)
            if tr is not None and hasattr(tr, "abort"):
                try:
                    tr.abort()
                except Exception:
                    pass
                break
        try:
            await asyncio.wait_for(ws.close(), timeout=min(0.5, float(join_s)))
        except Exception:
            pass

    async def _send_and_receive_op(self, message: dict, timeout_s: float | None = None):
        """Fail-fast send+recv. Recv-only timeout is not enough.

        EnvClient._receive times out _ws.recv, but a wedged server that is
        not reading (stuck in to_thread FastAdvance) makes _ws.send hang
        forever. Use asyncio.wait + force-drop socket on timeout so a
        cancel-ignoring send cannot hang wait_for forever.
        """
        op_t = float(timeout_s if timeout_s is not None else default_op_message_timeout_s())
        configured = float(getattr(self, "_message_timeout", op_t) or op_t)
        use_t = min(configured, op_t) if configured > 0 else op_t
        old = self._message_timeout
        self._message_timeout = use_t
        task = asyncio.create_task(self._send_and_receive(message))
        try:
            done, _pending = await asyncio.wait({task}, timeout=use_t)
            if task in done:
                return task.result()
            # Timeout: force-drop so wedged send unblocks, then cancel + <=1s join.
            await self._force_drop_ws(join_s=1.0)
            task.cancel()
            try:
                await asyncio.wait({task}, timeout=1.0)
            except Exception:
                pass
            raise TimeoutError(
                f"WS op timed out after {use_t:.0f}s "
                f"(type={message.get('type')}; socket force-dropped)"
            )
        finally:
            self._message_timeout = old

    async def reset(self, **kwargs):
        """Reset with the same per-op WS cap as advance (CreateSession <=60s)."""
        message = {"type": "reset", "data": kwargs}
        response = await self._send_and_receive_op(message)
        return self._parse_result(response.get("data", {}))

    async def close(self) -> None:
        """Disconnect in <=5s so a wedged server cannot hold gather."""
        try:
            await asyncio.wait_for(self.disconnect(), timeout=5.0)
        except Exception:
            self._ws = None
        provider = getattr(self, "_provider", None)
        if provider is None:
            return
        try:
            if hasattr(provider, "stop_container"):
                provider.stop_container()
            elif hasattr(provider, "stop"):
                provider.stop()
        except Exception:
            pass

    async def step(self, action, **kwargs):
        """Step with per-op WS timeout (hung FastAdvance fails ~deadline+slack)."""
        message = {
            "type": "step",
            "data": self._step_payload(action),
        }
        response = await self._send_and_receive_op(message)
        return self._parse_result(response.get("data", {}))

    async def advance(self, ticks: int) -> dict:
        """Avanza hasta N ticks (clamp server: 50/llamada) vía el tool MCP
        `advance` sobre la MISMA sesion /ws (reset/step/advance comparten
        conexion). Devuelve el resumen con interrupted/interrupt_reason/
        actual_ticks_advanced que el rollout usa para cerrar el bloque macro.
        Se habla JSON-RPC MCP ({"type":"mcp",...}) igual que mcp_ws_client.

        Per-call WS timeout is capped near OPENRA_RL_FAST_ADVANCE_DEADLINE_S
        (+20s slack), not the episode-scaled message_timeout (~3000s).
        """
        self._rpc_advance = getattr(self, "_rpc_advance", 0) + 1
        rpc_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "advance", "arguments": {"ticks": int(ticks)}},
            "id": f"adv{self._rpc_advance}",
        }
        response = await self._send_and_receive_op({"type": "mcp", "data": rpc_request})
        data = response.get("data", {})
        result = data.get("result", {})
        return (OpenRAMCPClient._unwrap_mcp_result(result)
                if isinstance(result, dict) else result)

    def _parse_state(self, data: Dict[str, Any]) -> OpenRAState:
        """Parse state response into OpenRAState."""
        return OpenRAState(**data)