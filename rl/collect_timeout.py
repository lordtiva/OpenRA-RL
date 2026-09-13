"""Fail-fast collect I/O: a hung FastAdvance must not stall the iter.

Root cause (2026-09-12 Phase C): EnvClient only times out `_ws.recv()`.
If the server WS handler is stuck in `to_thread(FastAdvance)` / session
executor it is not reading, so the *next* `_ws.send` (step/reset/close)
blocks forever. `asyncio.TimeoutError` is often empty-string so the
worker did not treat it as DEADLINE, retried the same socket, and
`asyncio.gather` waited. Heartbeat is only written on decision start,
so hb_age grew past 10 min while GPU~0 / markers=0.
"""
from __future__ import annotations

import asyncio
from typing import Any

from rl.collect_heartbeat import write_heartbeat


def is_collect_timeout(exc: BaseException | None) -> bool:
    """True for WS/gRPC deadline, abort, or asyncio TimeoutError (often msg="").

    On Python 3.10 asyncio.TimeoutError is NOT a subclass of builtin
    TimeoutError — that hid hung-recv from the worker except.
    """
    if exc is None:
        return False
    if isinstance(exc, TimeoutError) or isinstance(exc, asyncio.TimeoutError):
        return True
    if "TimeoutError" in type(exc).__name__:
        return True
    msg = str(exc)
    low = msg.lower()
    return (
        "DEADLINE" in msg or "Deadline" in msg
        or "ABORTED" in msg or "Aborted" in msg
        or "poisoned" in low
        or "CreateSession required" in msg
        or "timed out" in low
        or "timeout" in low
    )


def note_collect_timeout(
    heartbeat_path,
    *,
    error: str = "op_timeout",
    **extra: Any,
) -> None:
    """Force heartbeat so auto_train sees death/life (never silent stall)."""
    try:
        write_heartbeat(
            heartbeat_path,
            phase="collect_timeout",
            force=True,
            error=error,
            **extra,
        )
    except Exception:
        pass


async def await_env_op(
    coro,
    *,
    timeout_s: float,
    heartbeat_path=None,
    what: str = "env_op",
    error: str = "op_timeout",
    **hb: Any,
):
    """Bounded wait around send+recv+close. Always annotate TimeoutError.

    Uses asyncio.wait + cancel + <=1s join — never unbounded wait_for on a
    cancel-ignoring coro (wedged ws.send can ignore CancelledError).
    """
    task = asyncio.ensure_future(coro)
    done, _pending = await asyncio.wait({task}, timeout=float(timeout_s))
    if task in done:
        return task.result()
    note_collect_timeout(heartbeat_path, error=error, **hb)
    task.cancel()
    try:
        await asyncio.wait({task}, timeout=1.0)
    except Exception:
        pass
    raise TimeoutError(
        f"{what} timed out after {float(timeout_s):.0f}s"
    )


async def close_env_now(env, timeout_s: float = 5.0) -> None:
    """Drop a wedged WS in <=timeout_s so gather can continue."""
    if env is None:
        return
    try:
        await asyncio.wait_for(env.close(), timeout=float(timeout_s))
    except Exception:
        try:
            env._ws = None
        except Exception:
            pass
