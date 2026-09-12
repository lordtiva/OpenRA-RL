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
