"""Portable shared/exclusive lock for metrics.jsonl (Unix fcntl / Windows msvcrt).

Degrades to a no-op if the lock backend is unavailable — never blocks training.
Sidecar file: ``<metrics_path>.lock``.
"""
from __future__ import annotations

import contextlib
import os
import time
from typing import Iterator

__all__ = ["metrics_lock"]


@contextlib.contextmanager
def metrics_lock(path, exclusive: bool = True, timeout_s: float = 5.0) -> Iterator[None]:
    """Acquire a lock around a metrics.jsonl read (shared) or append (exclusive).

    ``path`` is the metrics file (not the lock). Safe to call when path is None
    or empty — yields immediately.
    """
    if not path:
        yield
        return
    lock_path = f"{path}.lock"
    fh = None
    locked = False
    try:
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        fh = open(lock_path, "a+b")
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        if os.name == "nt":
            locked = _lock_msvcrt(fh, exclusive=exclusive, deadline=deadline)
        else:
            locked = _lock_fcntl(fh, exclusive=exclusive, deadline=deadline)
    except Exception:
        # Degrade: proceed without lock rather than break train/auto_train.
        fh = None
    try:
        yield
    finally:
        if fh is not None:
            try:
                if locked:
                    _unlock(fh)
            except Exception:
                pass
            try:
                fh.close()
            except Exception:
                pass


def _lock_fcntl(fh, *, exclusive: bool, deadline: float) -> bool:
    import fcntl
    flag = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    while True:
        try:
            fcntl.flock(fh.fileno(), flag | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if time.monotonic() >= deadline:
                # Last try blocking briefly, then give up (caller still runs).
                try:
                    fcntl.flock(fh.fileno(), flag)
                    return True
                except Exception:
                    return False
            time.sleep(0.02)


def _lock_msvcrt(fh, *, exclusive: bool, deadline: float) -> bool:
    """Byte-range lock via msvcrt (exclusive only; shared falls back to exclusive).

    Windows msvcrt has no shared lock; readers also take exclusive on the
    1-byte sidecar so they still serialize against writers safely.
    """
    import msvcrt
    fh.seek(0)
    if fh.read(1) == b"":
        fh.write(b"\0")
        fh.flush()
    while True:
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            if time.monotonic() >= deadline:
                # Blocking fallback (LK_LOCK waits up to ~10s per call).
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                    return True
                except OSError:
                    return False
            time.sleep(0.02)


def _unlock(fh) -> None:
    if os.name == "nt":
        import msvcrt
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        import fcntl
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
