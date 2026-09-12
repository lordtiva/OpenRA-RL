"""gRPC bridge client for communicating with the OpenRA ExternalBotBridge.

This client connects to the gRPC server running inside the OpenRA process
using synchronous unary RPCs. Each environment runs in its own thread,
so sync gRPC calls naturally block that thread without affecting others.

Protocol:
  - Unary RPC (FastAdvance): send commands + advance N ticks, get observation back
  - Unary RPC (GetState): query current game state on demand
  - Unary RPC (CreateSession/DestroySession): session lifecycle (multi-session mode)
"""

import base64
import logging
import os
import time
from typing import Optional

import grpc

from openra_env.generated import rl_bridge_pb2, rl_bridge_pb2_grpc

logger = logging.getLogger(__name__)

# Align with C# RLSessionManager.FastAdvanceDeadlineSeconds /
# OPENRA_RL_FAST_ADVANCE_DEADLINE_S (docker-compose default 90).
DEFAULT_FAST_ADVANCE_DEADLINE_S = 90.0


def fast_advance_deadline_s() -> float:
    """C# FastAdvance hard deadline (seconds), from env or default 90."""
    raw = os.environ.get("OPENRA_RL_FAST_ADVANCE_DEADLINE_S", "")
    try:
        val = float(raw) if str(raw).strip() else DEFAULT_FAST_ADVANCE_DEADLINE_S
    except (TypeError, ValueError):
        val = DEFAULT_FAST_ADVANCE_DEADLINE_S
    return max(5.0, val)


def fast_advance_grpc_timeout_s(ticks: int, deadline_s: float | None = None) -> float:
    """gRPC timeout for FastAdvance: hard-capped to the C# deadline.

    Previously ``max(90, ticks+30)`` grew unbounded past the server deadline,
    so a wedged advance could outlive C# abort while the WS layer waited
    even longer. Always use the deadline (env/default); ticks do not inflate it.
    """
    d = float(deadline_s) if deadline_s is not None else fast_advance_deadline_s()
    _ = ticks  # ticks kept for call-site clarity; must not inflate past deadline
    return max(5.0, float(d))


class BridgeClient:
    """Synchronous gRPC client for the OpenRA RL Bridge.

    Uses unary RPCs only (FastAdvance, GetState, CreateSession, DestroySession).
    No streaming, no async, no shared poller — scales to hundreds of concurrent connections.

    In multi-session mode, a single gRPC channel is shared across all environments.
    Each environment uses its own session_id to route RPCs to the correct game session.
    """

    def __init__(self, host: str = "localhost", port: int = 9999, timeout_s: float = 30.0,
                 session_id: str = "", shared_channel: Optional[grpc.Channel] = None):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.session_id = session_id
        self._shared_channel = shared_channel
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[rl_bridge_pb2_grpc.RLBridgeStub] = None
        self._connected = False

    def connect(self) -> None:
        """Establish gRPC channel."""
        if self._shared_channel is not None:
            # Multi-session mode: use shared channel
            self._channel = self._shared_channel
        else:
            # Single-session mode: create own channel
            target = f"{self.host}:{self.port}"
            self._channel = grpc.insecure_channel(
                target,
                options=[
                    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                    ("grpc.max_send_message_length", 16 * 1024 * 1024),
                    ("grpc.keepalive_time_ms", 10000),
                    ("grpc.keepalive_timeout_ms", 5000),
                ],
            )
        self._stub = rl_bridge_pb2_grpc.RLBridgeStub(self._channel)
        self._connected = True
        logger.info(f"Connected to OpenRA bridge at {self.host}:{self.port}")

    def wait_for_ready(self, max_retries: int = 30, retry_interval: float = 1.0,
                       deadline_s: float | None = None) -> bool:
        """Wait for the gRPC server / session to become available.

        Wall-clock `deadline_s` (default FastAdvance deadline) is the hard
        cap. GetState used to use timeout_s=30; 40 retries x 30s = 20 min
        with no WS progress — that is the 10 min collect hang if the client
        is stuck in reset send/recv or the server executor.

        Phases accepted as "ready": playing / paused / ready.
        """
        if not self._connected:
            self.connect()
        if deadline_s is not None:
            wall = max(0.2, float(deadline_s))
        else:
            wall = max(5.0, fast_advance_deadline_s())
        t0 = time.monotonic()
        probe_t = min(5.0, wall)
        for attempt in range(max_retries):
            left = wall - (time.monotonic() - t0)
            if left <= 0:
                logger.error(f"Bridge wait_for_ready wall {wall:.0f}s exceeded")
                return False
            try:
                state = self.get_state(timeout_s=min(probe_t, left))
                phase = getattr(state, "phase", "")
                if phase in ("playing", "paused", "ready"):
                    logger.info(f"Bridge ready after {attempt + 1} attempts, phase={phase}")
                    return True
                if phase == "error":
                    logger.error(
                        f"Session in error phase after {attempt + 1} attempts — aborting wait"
                    )
                    return False
                logger.debug(f"Bridge not ready (attempt {attempt + 1}), phase={phase!r}")
            except grpc.RpcError as e:
                if attempt >= max_retries - 1:
                    logger.error(f"Bridge failed to become ready after {max_retries} attempts")
                    return False
                logger.debug(f"Bridge not ready (attempt {attempt + 1}): {e.code()}")
            except Exception as e:
                if attempt >= max_retries - 1:
                    return False
                logger.debug(f"Connection attempt {attempt + 1} failed: {e}")
            left = wall - (time.monotonic() - t0)
            if left <= 0:
                logger.error(f"Bridge wait_for_ready wall {wall:.0f}s exceeded")
                return False
            time.sleep(min(float(retry_interval), left))
        return False

    @property
    def session_started(self) -> bool:
        """Always False — no streaming session in sync mode."""
        return False

    def fast_advance_unary(
        self, ticks: int, commands=None,
        check_events_every: int = 0,
        enabled_interrupts: list[str] | None = None,
        peer_commands=None,
        peer_slot: str = "Multi0",
    ) -> rl_bridge_pb2.GameObservation:
        """Advance N ticks via unary RPC.

        Works reliably on all platforms including aarch64 where
        gRPC bidirectional streaming has transport issues.

        Args:
            ticks: Number of game ticks to advance.
            commands: Optional list of proto Command objects (primary / Multi1).
            check_events_every: Check interrupt signals every N ticks (0=disabled).
            enabled_interrupts: Signal names to check (e.g. ["enemy_spotted"]).
            peer_commands: Optional proto Commands for the other rl-agent (RL-vs-RL).
            peer_slot: Player slot for peer_commands (default Multi0).
        """
        if not self._connected:
            self.connect()

        request = rl_bridge_pb2.FastAdvanceRequest(ticks=ticks, session_id=self.session_id)
        if check_events_every > 0:
            logger.info("FastAdvance: ticks=%d, check_events_every=%d, interrupts=%s",
                        ticks, check_events_every, enabled_interrupts)
        if commands:
            request.commands.extend(commands)
        if check_events_every > 0:
            request.check_events_every = check_events_every
        if enabled_interrupts:
            request.enabled_interrupts.extend(enabled_interrupts)
        if peer_commands:
            request.peer_commands.extend(peer_commands)
            request.peer_slot = peer_slot or "Multi0"

        # Hard-cap to C# OPENRA_RL_FAST_ADVANCE_DEADLINE_S (default 90).
        # Do NOT use ticks+30 unbounded — that outgrew the server deadline and
        # left the client/WS layer waiting while C# had already aborted.
        timeout = fast_advance_grpc_timeout_s(ticks)
        try:
            return self._stub.FastAdvance(request, timeout=timeout)
        except grpc.RpcError as e:
            code = e.code()
            if code in (grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.ABORTED):
                details = e.details() or "hung or poisoned session"
                raise RuntimeError(
                    f"FastAdvance {code.name}: session={self.session_id!r} "
                    f"ticks={ticks} timeout_s={timeout:.0f} — {details}; "
                    f"destroy session and CreateSession required"
                ) from e
            raise

    def get_observation(self, player_slot: str = "") -> rl_bridge_pb2.GameObservation:
        """Snapshot observation for a player slot (RL-vs-RL peer after FastAdvance)."""
        if not self._connected or self._stub is None:
            raise RuntimeError("Not connected. Call connect() first.")
        request = rl_bridge_pb2.ObservationRequest(
            session_id=self.session_id,
            player_slot=player_slot or "",
        )
        return self._stub.GetObservation(request, timeout=self.timeout_s)

    def get_state(self, timeout_s: float | None = None) -> rl_bridge_pb2.GameState:
        """Query current game state via unary RPC."""
        if not self._connected or self._stub is None:
            raise RuntimeError("Not connected. Call connect() first.")
        request = rl_bridge_pb2.StateRequest(session_id=self.session_id)
        t = float(self.timeout_s if timeout_s is None else timeout_s)
        return self._stub.GetState(request, timeout=max(0.5, t))

    def create_session(self, map_name: str, bots: str, seed: int = 0,
                       player_faction: str = "", enemy_faction: str = "") -> str:
        """Create a new game session (multi-session mode).

        Returns the session_id assigned by the server.
        player_faction empty → C# default RandomAllies on the rl-agent slot.
        """
        if not self._connected or self._stub is None:
            self.connect()

        kwargs = dict(map_name=map_name, bots=bots, seed=seed)
        # Proto3 fields added 2026-09-09; tolerate older generated stubs.
        fields = rl_bridge_pb2.CreateSessionRequest.DESCRIPTOR.fields_by_name
        if "player_faction" in fields:
            kwargs["player_faction"] = player_faction or ""
        if "enemy_faction" in fields:
            kwargs["enemy_faction"] = enemy_faction or ""
        request = rl_bridge_pb2.CreateSessionRequest(**kwargs)
        response = self._stub.CreateSession(request, timeout=60.0, wait_for_ready=True)
        self.session_id = response.session_id
        logger.info(f"Created session {self.session_id} (map={map_name})")
        return response.session_id

    def destroy_session(self, session_id: str = "") -> None:
        """Destroy a game session (multi-session mode)."""
        if not self._connected or self._stub is None:
            return

        sid = session_id or self.session_id
        if not sid:
            return

        try:
            request = rl_bridge_pb2.DestroySessionRequest(session_id=sid)
            self._stub.DestroySession(request, timeout=30.0)
            logger.info(f"Destroyed session {sid}")
        except grpc.RpcError as e:
            logger.warning(f"Failed to destroy session {sid}: {e.code()}")

        if sid == self.session_id:
            self.session_id = ""

    def close(self) -> None:
        """Close the gRPC channel."""
        # Don't close shared channels — they're managed by the caller
        if self._channel is not None and self._shared_channel is None:
            self._channel.close()
            self._channel = None

        self._stub = None
        self._connected = False
        logger.info("Bridge connection closed")

    @property
    def is_connected(self) -> bool:
        return self._connected


def observation_to_dict(obs: rl_bridge_pb2.GameObservation) -> dict:
    """Convert a protobuf GameObservation to a plain dict for the OpenEnv layer."""
    return {
        "tick": obs.tick,
        "economy": {
            "cash": obs.economy.cash,
            "ore": obs.economy.ore,
            "power_provided": obs.economy.power_provided,
            "power_drained": obs.economy.power_drained,
            "resource_capacity": obs.economy.resource_capacity,
            "harvester_count": obs.economy.harvester_count,
        },
        "military": {
            "units_killed": obs.military.units_killed,
            "units_lost": obs.military.units_lost,
            "buildings_killed": obs.military.buildings_killed,
            "buildings_lost": obs.military.buildings_lost,
            "army_value": obs.military.army_value,
            "active_unit_count": obs.military.active_unit_count,
            "kills_cost": obs.military.kills_cost,
            "deaths_cost": obs.military.deaths_cost,
            "assets_value": obs.military.assets_value,
            "experience": obs.military.experience,
            "order_count": obs.military.order_count,
        },
        "units": [
            {
                "actor_id": u.actor_id,
                "type": u.type,
                "pos_x": u.pos_x,
                "pos_y": u.pos_y,
                "cell_x": u.cell_x,
                "cell_y": u.cell_y,
                "hp_percent": u.hp_percent,
                "is_idle": u.is_idle,
                "current_activity": u.current_activity,
                "owner": u.owner,
                "can_attack": u.can_attack,
                "facing": u.facing,
                "experience_level": u.experience_level,
                "stance": u.stance,
                "speed": u.speed,
                "attack_range": u.attack_range,
                "passenger_count": u.passenger_count,
                "is_building": u.is_building,
            }
            for u in obs.units
        ],
        "buildings": [
            {
                "actor_id": b.actor_id,
                "type": b.type,
                "pos_x": b.pos_x,
                "pos_y": b.pos_y,
                "hp_percent": b.hp_percent,
                "owner": b.owner,
                "is_producing": b.is_producing,
                "production_progress": b.production_progress,
                "producing_item": b.producing_item,
                "is_powered": b.is_powered,
                "is_repairing": b.is_repairing,
                "sell_value": b.sell_value,
                "rally_x": b.rally_x,
                "rally_y": b.rally_y,
                "power_amount": b.power_amount,
                "can_produce": list(b.can_produce),
                "cell_x": b.cell_x,
                "cell_y": b.cell_y,
            }
            for b in obs.buildings
        ],
        "production": [
            {
                "queue_type": p.queue_type,
                "item": p.item,
                "progress": p.progress,
                "remaining_ticks": p.remaining_ticks,
                "remaining_cost": p.remaining_cost,
                "paused": p.paused,
            }
            for p in obs.production
        ],
        "visible_enemies": [
            {
                "actor_id": u.actor_id,
                "type": u.type,
                "pos_x": u.pos_x,
                "pos_y": u.pos_y,
                "cell_x": u.cell_x,
                "cell_y": u.cell_y,
                "hp_percent": u.hp_percent,
                "is_idle": u.is_idle,
                "current_activity": u.current_activity,
                "owner": u.owner,
                "can_attack": u.can_attack,
                "facing": u.facing,
                "experience_level": u.experience_level,
                "stance": u.stance,
                "speed": u.speed,
                "attack_range": u.attack_range,
                "passenger_count": u.passenger_count,
                "is_building": u.is_building,
            }
            for u in obs.visible_enemies
        ],
        "visible_enemy_buildings": [
            {
                "actor_id": b.actor_id,
                "type": b.type,
                "pos_x": b.pos_x,
                "pos_y": b.pos_y,
                "hp_percent": b.hp_percent,
                "owner": b.owner,
                "is_producing": b.is_producing,
                "production_progress": b.production_progress,
                "producing_item": b.producing_item,
                "is_powered": b.is_powered,
                "is_repairing": b.is_repairing,
                "sell_value": b.sell_value,
                "rally_x": b.rally_x,
                "rally_y": b.rally_y,
                "power_amount": b.power_amount,
                "can_produce": list(b.can_produce),
                "cell_x": b.cell_x,
                "cell_y": b.cell_y,
            }
            for b in obs.visible_enemy_buildings
        ],
        "map_info": {
            "width": obs.map_info.width,
            "height": obs.map_info.height,
            "map_name": obs.map_info.map_name,
        },
        "available_production": list(obs.available_production),
        "ready_support_powers": list(getattr(obs, "ready_support_powers", None) or []),
        "done": obs.done,
        "reward": obs.reward,
        "result": obs.result,
        "spatial_map": base64.b64encode(bytes(obs.spatial_map)).decode("ascii"),
        "spatial_channels": obs.spatial_channels,
        "explored_percent": obs.explored_percent,
        # Server-side interrupt detection fields
                "interrupted": obs.interrupted,
                "interrupt_reason": obs.interrupt_reason,
                "actual_ticks_advanced": obs.actual_ticks_advanced,
                # Resumen espectador por bando (economy_race / supremacía)
                                "global_summary": {
                                                    "own": {
                                                        "cash": obs.global_summary.own.cash,
                                                        "unit_value": obs.global_summary.own.unit_value,
                                                        "building_value": obs.global_summary.own.building_value,
                                                        "n_buildings": obs.global_summary.own.n_buildings,
                                                        "earned": obs.global_summary.own.earned,
                                                    },
                                                    "enemy": {
                                                        "cash": obs.global_summary.enemy.cash,
                                                        "unit_value": obs.global_summary.enemy.unit_value,
                                                        "building_value": obs.global_summary.enemy.building_value,
                                                        "n_buildings": obs.global_summary.enemy.n_buildings,
                                                        "earned": obs.global_summary.enemy.earned,
                                                    },
                                                },
                            }


def commands_to_proto(commands: list[dict]) -> rl_bridge_pb2.AgentAction:
    """Convert a list of command dicts to a protobuf AgentAction."""
    action_type_map = {
        "no_op": rl_bridge_pb2.NO_OP,
        "move": rl_bridge_pb2.MOVE,
        "attack_move": rl_bridge_pb2.ATTACK_MOVE,
        "attack": rl_bridge_pb2.ATTACK,
        "stop": rl_bridge_pb2.STOP,
        "harvest": rl_bridge_pb2.HARVEST,
        "build": rl_bridge_pb2.BUILD,
        "train": rl_bridge_pb2.TRAIN,
        "deploy": rl_bridge_pb2.DEPLOY,
        "sell": rl_bridge_pb2.SELL,
        "repair": rl_bridge_pb2.REPAIR,
        "place_building": rl_bridge_pb2.PLACE_BUILDING,
        "cancel_production": rl_bridge_pb2.CANCEL_PRODUCTION,
        "set_rally_point": rl_bridge_pb2.SET_RALLY_POINT,
        "guard": rl_bridge_pb2.GUARD,
        "set_stance": rl_bridge_pb2.SET_STANCE,
        "enter_transport": rl_bridge_pb2.ENTER_TRANSPORT,
        "unload": rl_bridge_pb2.UNLOAD,
        "power_down": rl_bridge_pb2.POWER_DOWN,
        "set_primary": rl_bridge_pb2.SET_PRIMARY,
        "surrender": rl_bridge_pb2.SURRENDER,
        "patrol": rl_bridge_pb2.PATROL,
        "fast_advance": rl_bridge_pb2.FAST_ADVANCE,
        "army_attack_move": rl_bridge_pb2.ARMY_ATTACK_MOVE,
        "support_power": getattr(rl_bridge_pb2, "SUPPORT_POWER", rl_bridge_pb2.NO_OP),
    }

    proto_commands = []
    for cmd in commands:
        action_str = cmd.get("action", "no_op")
        proto_cmd = rl_bridge_pb2.Command(
            action=action_type_map.get(action_str, rl_bridge_pb2.NO_OP),
            actor_id=cmd.get("actor_id", 0),
            target_actor_id=cmd.get("target_actor_id", 0),
            target_x=cmd.get("target_x", 0),
            target_y=cmd.get("target_y", 0),
            item_type=cmd.get("item_type", ""),
            queued=cmd.get("queued", False),
            ticks=cmd.get("ticks", 0),
        )
        proto_commands.append(proto_cmd)

    return rl_bridge_pb2.AgentAction(commands=proto_commands)
