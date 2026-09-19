# -*- coding: utf-8 -*-
"""Kenyon-style sparse hashing for SIL/BC novelty (Fly-inspired, no connectome data).

Fixed random projection + top-k → fingerprint. Used to dedupe EliteBuffer /
TeacherWinBuffer samples so even-pick does not flood SIL with identical march frames.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

# Default: scalars(41) + type one-hot(~40) padded/truncated to FEAT_DIM.
FEAT_DIM = 96
KENYON_DIM = 2048
KENYON_K = 32


def _as_1d_np(x) -> np.ndarray:
    if x is None:
        return np.zeros(0, dtype=np.float32)
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    return arr


def step_feature_vec(step: dict, feat_dim: int = FEAT_DIM) -> np.ndarray:
    """Cheap tactical fingerprint input from a stored rollout step."""
    batch = step.get("batch") or {}
    sc = _as_1d_np(batch.get("scalars"))
    parts = [sc] if sc.size else [np.zeros(41, dtype=np.float32)]
    # Action type one-hot (stable across resumes).
    act = step.get("action") or {}
    t = act.get("type")
    if torch.is_tensor(t):
        t = int(t.reshape(-1)[0].item())
    else:
        try:
            t = int(t) if t is not None else 0
        except (TypeError, ValueError):
            t = 0
    n_types = 40
    oh = np.zeros(n_types, dtype=np.float32)
    if 0 <= t < n_types:
        oh[t] = 1.0
    parts.append(oh)
    # Cell normalized if present (coarse spatial cue).
    cell = act.get("cell")
    if torch.is_tensor(cell):
        c = float(cell.reshape(-1)[0].item())
        parts.append(np.array([math.tanh(c / 2048.0)], dtype=np.float32))
    vec = np.concatenate(parts, axis=0)
    if vec.size >= feat_dim:
        return vec[:feat_dim].astype(np.float32)
    out = np.zeros(feat_dim, dtype=np.float32)
    out[: vec.size] = vec
    return out


class KenyonHasher:
    """Fixed random proj + k-WTA → frozenset of active indices."""

    def __init__(self, in_dim: int = FEAT_DIM, hidden: int = KENYON_DIM,
                 k: int = KENYON_K, seed: int = 20260919):
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.k = int(k)
        rng = np.random.RandomState(int(seed))
        self.W = (rng.randn(self.in_dim, self.hidden).astype(np.float32)
                  * (1.0 / math.sqrt(self.in_dim)))

    def fingerprint(self, vec: np.ndarray) -> frozenset:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.size != self.in_dim:
            tmp = np.zeros(self.in_dim, dtype=np.float32)
            n = min(v.size, self.in_dim)
            tmp[:n] = v[:n]
            v = tmp
        act = v @ self.W
        k = min(self.k, self.hidden)
        idx = np.argpartition(act, -k)[-k:]
        return frozenset(int(i) for i in idx.tolist())

    def fingerprint_step(self, step: dict) -> frozenset:
        return self.fingerprint(step_feature_vec(step, self.in_dim))


_HASHER: KenyonHasher | None = None


def get_hasher() -> KenyonHasher:
    global _HASHER
    if _HASHER is None:
        _HASHER = KenyonHasher()
    return _HASHER


def hamming_fp(a: frozenset, b: frozenset) -> int:
    return len(a.symmetric_difference(b))


def novelty_pick(group: list, cap: int, hasher: KenyonHasher | None = None,
                 min_hamming: int = 12) -> list:
    """Pick up to cap steps preferring novel Kenyon fingerprints.

    Greedy: walk chronological order, keep a step if its fingerprint is at
    least min_hamming away from all already kept (or buffer empty). If that
    yields fewer than cap, fill remainder with even spacing among leftovers.
    """
    if cap <= 0 or not group:
        return []
    if len(group) <= cap:
        return list(group)
    hasher = hasher or get_hasher()
    fps = [hasher.fingerprint_step(s) for s in group]
    kept_i: list[int] = []
    kept_fp: list[frozenset] = []
    leftover: list[int] = []
    for i, fp in enumerate(fps):
        if len(kept_i) >= cap:
            leftover.append(i)
            continue
        if not kept_fp:
            kept_i.append(i)
            kept_fp.append(fp)
            continue
        if min(hamming_fp(fp, p) for p in kept_fp) >= int(min_hamming):
            kept_i.append(i)
            kept_fp.append(fp)
        else:
            leftover.append(i)
    if len(kept_i) < cap and leftover:
        need = cap - len(kept_i)
        # even-pick indices from leftover
        if need >= len(leftover):
            kept_i.extend(leftover)
        else:
            picks = [leftover[round(j * (len(leftover) - 1) / (need - 1))]
                     for j in range(need)] if need > 1 else [leftover[0]]
            kept_i.extend(picks)
    kept_i = sorted(set(kept_i))[:cap]
    return [group[i] for i in kept_i]


def diversity_stats(steps: list, hasher: KenyonHasher | None = None) -> dict:
    """Measurement bundle for A/B vs even-pick."""
    hasher = hasher or get_hasher()
    n = len(steps or [])
    if n <= 0:
        return {
            "n_steps": 0,
            "n_unique_hash": 0,
            "unique_ratio": 0.0,
            "mean_pairwise_hamming": 0.0,
        }
    fps = [hasher.fingerprint_step(s) for s in steps]
    uniq = set(fps)
    # Sample pairwise Hamming (cap 200 pairs) for a cheap diversity pulse.
    pair_h = []
    lim = min(n, 40)
    for i in range(lim):
        for j in range(i + 1, lim):
            pair_h.append(hamming_fp(fps[i], fps[j]))
    mean_h = float(sum(pair_h) / len(pair_h)) if pair_h else 0.0
    return {
        "n_steps": n,
        "n_unique_hash": len(uniq),
        "unique_ratio": float(len(uniq) / max(n, 1)),
        "mean_pairwise_hamming": mean_h,
    }
