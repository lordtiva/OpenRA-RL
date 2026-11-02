"""play_skirmish.py — Real-time PPO / Commander sidecar for OpenRA GUI Skirmish.

When a player selects "PPO Agent" in the skirmish lobby, OpenRA launches
the ExternalBotBridge on port 10001 (GUI port) and spawns this script as a sidecar.
The engine ticks at 25 TPS in real-time. FastAdvance(ticks=0) injects commander decisions
and retrieves game observations without advancing or freezing the game clock.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Handle PyInstaller onefile directory or standard root
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    BUNDLE_DIR = Path(sys._MEIPASS)
    ROOT_DIR = Path(sys.executable).resolve().parent
else:
    BUNDLE_DIR = Path(__file__).resolve().parent.parent
    ROOT_DIR = BUNDLE_DIR

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import grpc

from ra_env import Bridge, StepResult
from ra_env._proto import rl_bridge_pb2 as pb
from ra_rl.policies import make_policy

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [skirmish] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("skirmish")


def find_default_checkpoint() -> str | None:
    """Backward-compatible alias for finding model checkpoint path (.pt)."""
    pt_candidates = [
        os.environ.get("OPENRA_RL_CKPT"),
        ROOT_DIR / "rl_agent" / "checkpoint_best.pt",
        ROOT_DIR / "rl_agent" / "model.pt",
        ROOT_DIR / "runs" / "v4" / "selfplay" / "checkpoint_best.pt",
        ROOT_DIR / "runs" / "v4" / "selfplay" / "checkpoint_latest.pt",
        ROOT_DIR / "runs" / "v4" / "bc_v4" / "model.pt",
        ROOT_DIR / "runs" / "phase4" / "selfplay" / "checkpoint_best.pt",
    ]
    for c in pt_candidates:
        if c:
            p = Path(c)
            if p.is_file():
                return str(p.resolve())
    return None


def find_default_model() -> str | None:
    """Search for the best available model (ONNX preferred for low overhead, or PyTorch .pt)."""
    # 1. Check for ONNX models first (lightweight, zero-torch runtime)
    onnx_candidates = [
        os.environ.get("OPENRA_RL_ONNX"),
        BUNDLE_DIR / "onnx_models",
        ROOT_DIR / "onnx_models",
        ROOT_DIR / "rl_agent" / "onnx_models",
        ROOT_DIR.parent / "onnx_models",
    ]
    for c in onnx_candidates:
        if c:
            p = Path(c)
            if p.is_dir() and (p / "commander_item.onnx").is_file():
                return f"onnx:{p.resolve()}"

    ckpt = find_default_checkpoint()
    if ckpt:
        return f"net:{ckpt}"
    return None


def wait_for_match(bridge: Bridge, session_id: str = "", slot: str = "", timeout_s: float = 60.0) -> pb.GameState | None:
    """Wait for OpenRA bridge to enter the 'playing' phase."""
    tag = f"[{slot}] " if slot else ""
    t0 = time.monotonic()
    log.info(f"{tag}Connecting to OpenRA bridge at {bridge.address}" + (f" (session='{session_id}')" if session_id else "") + "...")
    attempt = 0
    while True:
        try:
            st = bridge.get_state(session_id, timeout_s=1.0)
            phase = getattr(st, "phase", "")
            if phase == "playing":
                log.info(f"{tag}Match active! Episode: {st.episode_id}, Tick: {st.tick}")
                return st
            if phase == "game_over":
                log.warning(f"{tag}Match ended before initialization.")
                return None
            if attempt % 10 == 0:
                log.info(f"{tag}Waiting for match start (current phase: '{phase}')...")
        except (grpc.RpcError, Exception) as e:
            if attempt % 10 == 0:
                log.debug(f"{tag}Bridge not yet ready: {e}")

        attempt += 1
        if timeout_s > 0 and (time.monotonic() - t0) >= timeout_s:
            log.error(f"{tag}Timeout ({timeout_s}s) waiting for match start.")
            return None
        time.sleep(0.5)


def run_skirmish_loop(
    bridge: Bridge,
    policy,
    session_id: str = "",
    slot: str = "",
    step_delay: float = 0.03,
) -> str:
    """Run real-time decision loop with OpenRA at ticks=0."""
    pending_decisions: list[pb.Decision] = []
    policy.reset(0)
    tag = f"[{slot}] " if slot else ""
    log.info(f"{tag}Agent started real-time decision loop.")

    total_decisions = 0
    last_log_tick = -1

    while True:
        try:
            obs = bridge.fast_advance(
                session_id=session_id,
                ticks=0,
                decisions=pending_decisions,
            )
        except grpc.RpcError as rpc_err:
            log.info(f"{tag}gRPC stream ended or disconnected: {rpc_err.code().name}")
            return "disconnected"
        except Exception as e:
            log.error(f"{tag}Error during FastAdvance: {e}")
            return "error"

        if obs.done:
            result = obs.result or "done"
            log.info(f"{tag}Game Over! Outcome: {result} at tick {obs.tick} (Total decisions: {total_decisions})")
            return result

        step = StepResult(
            obs=obs,
            tick=obs.tick,
            done=False,
            truncated=False,
            result="",
            pending=list(obs.pending_decisions),
            teacher=[],
        )

        if step.pending:
            pending_decisions = policy.decide(step)
            total_decisions += len(pending_decisions)
            for req, d in zip(step.pending, pending_decisions):
                choice_str = d.choice if d.choice else (f"target[{d.target_index}]" if d.target_index >= 0 else "hold")
                log.info(f"{tag}Tick {obs.tick}: Decision for {req.kind} ({req.queue}): {choice_str}")
        else:
            pending_decisions = []

        if obs.tick - last_log_tick >= 500:
            last_log_tick = obs.tick
            units_count = len(obs.units)
            cash = getattr(obs.economy, "cash", 0)
            log.info(
                f"{tag}Tick {obs.tick:5d} | Cash: ${cash:5d} | Units: {units_count:3d} | Decisions: {total_decisions:4d}"
            )

        if step_delay > 0:
            time.sleep(step_delay)


def main():
    parser = argparse.ArgumentParser(description="OpenRA RL Skirmish Sidecar Agent")
    parser.add_argument("--attach", action="store_true", default=True, help="Attach to running OpenRA instance")
    parser.add_argument("--host", default="127.0.0.1", help="gRPC host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=10001, help="gRPC port (default: 10001 for GUI skirmish)")
    parser.add_argument("--ckpt", default=None, help="Path to checkpoint model (.pt) or ONNX model dir")
    parser.add_argument("--policy", choices=["onnx", "net", "scripted"], default=None, help="Force policy type")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--wait-timeout", type=float, default=60.0, help="Seconds to wait for game start")
    parser.add_argument("--step-delay", type=float, default=0.03, help="Delay between real-time ticks in seconds")
    parser.add_argument("--session", default="", help="Specific session ID to connect to")
    parser.add_argument("--slot", default="", help="Player slot (e.g. Multi0, Multi1)")

    args = parser.parse_args()

    # Determine policy
    model_spec = None
    if args.ckpt:
        p = Path(args.ckpt)
        if p.is_dir() or str(p).endswith(".onnx"):
            model_spec = f"onnx:{args.ckpt}"
        else:
            model_spec = f"net:{args.ckpt}"
    else:
        model_spec = find_default_model()

    if args.policy == "scripted" or model_spec is None:
        log.info("Using scripted teacher policy.")
        policy = make_policy("scripted")
    elif args.policy == "onnx" or model_spec.startswith("onnx:"):
        log.info(f"Loading ONNX policy: {model_spec}...")
        policy = make_policy(model_spec)
    else:
        log.info(f"Loading PyTorch neural policy: {model_spec} (T={args.temperature})...")
        policy = make_policy(model_spec)

    bridge = Bridge(address=f"{args.host}:{args.port}")
    state = wait_for_match(bridge, session_id=args.session, slot=args.slot, timeout_s=args.wait_timeout)
    if state is None:
        sys.exit(1)

    result = run_skirmish_loop(bridge, policy, session_id=args.session, slot=args.slot, step_delay=args.step_delay)
    log.info(f"Skirmish sidecar finished with result: {result}")
    sys.exit(0)


if __name__ == "__main__":
    main()
