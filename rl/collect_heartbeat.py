"""Lightweight collect/update heartbeat for auto_train hang detection (Phase 1.5).

Train/rollout writes ckpt-dir/collect_heartbeat.json so stalls are visible even
when metrics.jsonl is idle mid-iter. auto_train treats an advancing heartbeat as
progress (healthy collect) and uses it in hang classification logs.
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile
import threading
import time
from typing import Any

HEARTBEAT_NAME = "collect_heartbeat.json"
DEFAULT_MIN_INTERVAL_S = 15.0

_lock = threading.Lock()
_last_write_mono = 0.0


def heartbeat_path(ckpt_dir: str | os.PathLike | None) -> pathlib.Path | None:
    if not ckpt_dir:
        return None
    return pathlib.Path(ckpt_dir) / HEARTBEAT_NAME


def write_heartbeat(
    ckpt_dir_or_path: str | os.PathLike | None,
    *,
    tick: int | None = None,
    step: int | None = None,
    phase: str = "collect",
    session: str | None = None,
    iter: int | None = None,
    worker: int | None = None,
    force: bool = False,
    min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
    **extra: Any,
) -> bool:
    """Atomically write heartbeat JSON under ckpt-dir (or exact .json path).

    Throttled to min_interval_s unless force=True. Returns True if written.
    """
    global _last_write_mono
    if not ckpt_dir_or_path:
        return False
    p = pathlib.Path(ckpt_dir_or_path)
    if p.suffix.lower() != ".json":
        p = p / HEARTBEAT_NAME
    now = time.time()
    mono = time.monotonic()
    with _lock:
        # Throttle on monotonic clock (wall clock jumps must not freeze hb).
        if not force and (mono - _last_write_mono) < float(min_interval_s):
            return False
        payload: dict[str, Any] = {
            "ts": now,
            "wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "phase": str(phase or "collect"),
        }
        if tick is not None:
            payload["tick"] = int(tick)
        if step is not None:
            payload["step"] = int(step)
        if session is not None:
            payload["session"] = str(session)
        if iter is not None:
            payload["iter"] = int(iter)
        if worker is not None:
            payload["worker"] = int(worker)
        for k, v in extra.items():
            if v is not None:
                payload[k] = v
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                prefix="hb_", suffix=".json", dir=str(p.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, separators=(",", ":"))
                    f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, p)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            _last_write_mono = mono
            return True
        except OSError:
            return False


def read_heartbeat(ckpt_dir_or_path: str | os.PathLike | None) -> dict | None:
    """Read heartbeat JSON; None if missing/corrupt."""
    if not ckpt_dir_or_path:
        return None
    p = pathlib.Path(ckpt_dir_or_path)
    if p.suffix.lower() != ".json":
        p = p / HEARTBEAT_NAME
    try:
        raw = p.read_text(encoding="utf-8")
        row = json.loads(raw)
        return row if isinstance(row, dict) else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def heartbeat_advanced(prev: dict | None, cur: dict | None) -> bool:
    """True if cur is newer progress than prev (ts, tick, or step)."""
    if not cur or not isinstance(cur, dict):
        return False
    if not prev:
        return True
    try:
        if float(cur.get("ts") or 0) > float(prev.get("ts") or 0) + 0.5:
            return True
    except (TypeError, ValueError):
        pass
    for key in ("tick", "step", "iter"):
        try:
            if int(cur.get(key) or 0) > int(prev.get(key) or 0):
                return True
        except (TypeError, ValueError):
            continue
    return False


def heartbeat_age_s(hb: dict | None, now: float | None = None) -> float | None:
    if not hb:
        return None
    try:
        return float(now if now is not None else time.time()) - float(hb.get("ts") or 0)
    except (TypeError, ValueError):
        return None

def gather_stall_should_cancel(age: float | None, slice_s: float) -> bool:
    """True iff gather wait-timeout should cancel pending workers.

    Fresh heartbeat (age is not None and age < slice_s) means workers are
    alive during long episodes — never cancel on completion-stall streak
    alone. Cancel only when hb is missing (age is None) or stale
    (age >= slice_s).
    """
    if age is not None and age < float(slice_s):
        return False
    return True


# After a gather-stall cancel, refill shortfall episodes in the same iter
# instead of accepting a partial batch (2/3 of 4). Cap rounds to avoid loops.
MAX_GATHER_REFILL_ROUNDS = 2


def episodes_shortfall(have: int, target: int) -> int:
    """How many more episodes needed to reach target (0 if already enough)."""
    return max(0, int(target) - int(have))


def should_refill(
    need: int,
    rounds_done: int,
    max_rounds: int = MAX_GATHER_REFILL_ROUNDS,
) -> bool:
    """True if we should spawn another refill round after a gather stall."""
    return int(need) > 0 and int(rounds_done) < int(max_rounds)


def note_gather_wait(
    ckpt_dir_or_path: str | os.PathLike | None,
    *,
    phase: str = "collect_wait",
    iter: int | None = None,
    pending: int | None = None,
    **extra: Any,
) -> bool:
    """Force-write hb on gather wait-timeout (empty asyncio.wait done).

    Call AFTER reading worker hb age for cancel decisions, so auto_train
    sees supervisor progress (phase=collect_wait / bc_wait) even when
    workers are wedged and no longer advancing tick/step.
    """
    return write_heartbeat(
        ckpt_dir_or_path,
        phase=phase,
        force=True,
        iter=iter,
        pending=pending,
        **extra,
    )


def note_gather_refill(
    ckpt_dir_or_path: str | os.PathLike | None,
    *,
    phase: str = "collect_refill",
    iter: int | None = None,
    pending: int | None = None,
    refill_round: int | None = None,
    **extra: Any,
) -> bool:
    """Force-write hb when stall cancel + refill starts (ts must advance)."""
    if refill_round is not None:
        extra = {**extra, "refill_round": int(refill_round)}
    return write_heartbeat(
        ckpt_dir_or_path,
        phase=phase,
        force=True,
        iter=iter,
        pending=pending,
        **extra,
    )

