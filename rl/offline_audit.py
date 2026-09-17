#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pragmatic offline validation suite for OpenRA-RL (no Docker / no OpenRA).

Loads a checkpoint + recorded observations (elite.pt / teacher_wins) and
metrics.jsonl, then runs static + counterfactual audits that catch known
bug classes without launching a live game.

Usage (from repo root, venv activated)::

    python rl/offline_audit.py --ckpt rl/ckpts_v2/best.pt --n 500
    python rl/offline_audit.py --ckpt rl/ckpts_v2/latest.pt --n 200 --strict

Exit code is non-zero only on hard FAIL when ``--strict`` is set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

# Allow `python rl/offline_audit.py` and `python -m rl.offline_audit`
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.action_adapter import Vocab
from rl.imitation import (
    EliteBuffer,
    TAPE_SCHEMA,
    TAPE_SCHEMA_EXPAND,
    sample_type_name,
)
from rl.network import ACTION_TYPES, AlphaLiteNet, TYPES_USE_CELL
from rl.obs_encoding import SCALAR_DIM
from rl.onboard import load_metrics_rows
from rl.roles import (
    IDENTITY_ITEMS,
    ROLE_OF_ITEM,
    ROLE_POWER_INFRA,
    ROLE_REFINERY,
    cheapest_of,
    concretos_de,
    role_of,
)
from rl.trainer import load_checkpoint

# Scalar layout (obs_encoding.SCALAR_DIM) - keep in sync.
_IDX_CASH = 0
_IDX_POWER_RATIO = 3
_IDX_LOW_POWER = 4
_IDX_HAS_REFINERY = 16
_IDX_CAN_AFFORD_PROC = 17
_IDX_POWER_BALANCE = 33

# Typical a_short / onboard timeout wall (metrics ticks).
_DEFAULT_MAX_TICKS = 53000

_PASS, _WARN, _FAIL, _SKIP = "PASS", "WARN", "FAIL", "SKIP"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve(p: str | Path | None, default: Path | None = None) -> Path | None:
    if p is None:
        return default
    path = Path(p)
    if path.exists():
        return path
    alt = _repo_root() / path
    if alt.exists():
        return alt
    return path  # may not exist; caller decides


def _pick_device(req: str) -> str:
    if req != "auto":
        return req
    return "cuda" if torch.cuda.is_available() else "cpu"


def _as_int(t) -> int:
    if torch.is_tensor(t):
        return int(t.reshape(-1)[0].item())
    try:
        return int(t)
    except (TypeError, ValueError):
        return 0


def _as_float(t, default: float = 0.0) -> float:
    if torch.is_tensor(t):
        return float(t.reshape(-1)[0].item())
    try:
        return float(t)
    except (TypeError, ValueError):
        return default


def _scalar_at(batch: dict, idx: int, default: float = 0.0) -> float:
    sc = batch.get("scalars")
    if not torch.is_tensor(sc) or sc.numel() <= idx:
        return default
    flat = sc.reshape(-1)
    return float(flat[idx].item())


def _status_line(name: str, status: str, detail: str) -> str:
    return f"[{status:4}] {name}: {detail}"


def _verdict(status: str, *, ok_if: bool, warn_if: bool = False,
             fail_msg: str = "", warn_msg: str = "",
             pass_msg: str = "ok") -> tuple[str, str]:
    if ok_if and not warn_if:
        return _PASS, pass_msg
    if warn_if and not (not ok_if):
        # warn takes precedence when still "ok-ish"
        return _WARN, warn_msg or pass_msg
    if not ok_if:
        return _FAIL, fail_msg or "failed"
    return _WARN, warn_msg or pass_msg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_obs_steps(
    elite_path: Path | None,
    teacher_dir: Path | None,
    n: int,
    *,
    prefer_elite: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    """Load up to ``n`` CPU step dicts from elite and/or teacher_wins.

    Prefer elite (compact SIL ring). Teacher episodes are huge (~200MB each);
    we stream only as many ep_*.pt files as needed to fill the quota.
    """
    meta: dict[str, Any] = {
        "elite_path": str(elite_path) if elite_path else None,
        "teacher_dir": str(teacher_dir) if teacher_dir else None,
        "elite_steps": 0,
        "teacher_steps": 0,
        "teacher_eps_loaded": 0,
        "sources": [],
    }
    steps: list[dict] = []

    if prefer_elite and elite_path and elite_path.is_file():
        buf = EliteBuffer()
        n_elite = buf.load(elite_path)
        meta["elite_steps"] = int(n_elite)
        if n_elite > 0:
            got = buf.sample_recent(max_steps=n) if n < n_elite else buf.snapshot()
            steps.extend(got)
            meta["sources"].append(f"elite:{len(got)}")

    need = max(0, int(n) - len(steps))
    if need > 0 and teacher_dir and teacher_dir.is_dir():
        tw_steps, tw_meta = _load_teacher_steps(teacher_dir, need)
        steps.extend(tw_steps)
        meta["teacher_steps"] = tw_meta.get("n_steps", 0)
        meta["teacher_eps_loaded"] = tw_meta.get("n_eps", 0)
        meta["teacher_schema"] = tw_meta.get("schema")
        if tw_steps:
            meta["sources"].append(f"teacher:{len(tw_steps)}")

    meta["n_loaded"] = len(steps)
    return steps[:n], meta


def _load_teacher_steps(teacher_dir: Path, n: int) -> tuple[list[dict], dict]:
    """Partial load of teacher_wins without hydrating the full ring."""
    man_path = teacher_dir / "manifest.json"
    meta: dict[str, Any] = {"n_steps": 0, "n_eps": 0, "schema": None}
    if not man_path.is_file():
        return [], meta
    try:
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], meta
    schema = str(manifest.get("schema") or "")
    meta["schema"] = schema
    # Accept both onboard schemas; refuse unknown (corrupt / wrong run).
    if schema and schema not in (TAPE_SCHEMA, TAPE_SCHEMA_EXPAND):
        meta["skip_reason"] = f"schema mismatch: {schema!r}"
        return [], meta

    entries = list(manifest.get("episodes") or [])
    if not entries:
        return [], meta

    # Even-pick episode files so we do not only read the first few.
    n_eps = len(entries)
    # Load at most ceil(n / avg_steps) eps; avg ~800 - cap 8 files for speed.
    max_files = min(n_eps, max(1, min(8, (int(n) + 799) // 800)))
    if max_files <= 1:
        pick = entries[:1]
    elif n_eps <= max_files:
        pick = entries
    else:
        pick = [
            entries[round(i * (n_eps - 1) / (max_files - 1))]
            for i in range(max_files)
        ]

    out: list[dict] = []
    for entry in pick:
        if len(out) >= n:
            break
        fname = str(entry.get("file") or "")
        if not fname:
            eid = int(entry.get("id") or 0)
            fname = f"ep_{eid:04d}.pt"
        fpath = teacher_dir / fname
        if not fpath.is_file():
            continue
        try:
            blob = torch.load(fpath, map_location="cpu", weights_only=False)
        except TypeError:
            try:
                blob = torch.load(fpath, map_location="cpu")
            except Exception:
                continue
        except Exception:
            continue
        ep_steps = list(blob.get("steps") or [])
        if not ep_steps:
            continue
        meta["n_eps"] += 1
        # Even-pick within episode to diversify.
        remain = n - len(out)
        if len(ep_steps) <= remain:
            out.extend(ep_steps)
        else:
            # reuse EliteBuffer-style even pick
            idxs = [
                round(i * (len(ep_steps) - 1) / (remain - 1))
                for i in range(remain)
            ] if remain > 1 else [0]
            out.extend(ep_steps[i] for i in idxs)

    meta["n_steps"] = len(out)
    return out, meta


def _vocab_id_to_name(vocab: Vocab) -> dict[int, str]:
    return {int(v): str(k) for k, v in (vocab.type_to_id or {}).items()}


def _decode_item_label(step: dict, vocab_inv: dict[int, str]) -> str:
    """Resolve action item_slot via batch item_indices -> vocab role/item name."""
    act = step.get("action") or {}
    batch = step.get("batch") or {}
    slot = _as_int(act.get("item_slot"))
    idx = batch.get("item_indices")
    if not torch.is_tensor(idx):
        return ""
    flat = idx.reshape(-1)
    if slot < 0 or slot >= flat.numel():
        return ""
    vid = int(flat[slot].item())
    return vocab_inv.get(vid, f"id{vid}")


def _stack_batches(steps: list[dict], keys: list[str] | None = None) -> dict:
    """Cat leading batch-dim of stored step batches (each is [1, ...])."""
    if not steps:
        return {}
    b0 = steps[0].get("batch") or {}
    keys = keys or [k for k, v in b0.items() if torch.is_tensor(v)]
    out = {}
    for k in keys:
        tensors = []
        for s in steps:
            v = (s.get("batch") or {}).get(k)
            if torch.is_tensor(v):
                tensors.append(v if v.dim() >= 1 else v.unsqueeze(0))
        if tensors:
            out[k] = torch.cat(tensors, dim=0)
    return out


def _stack_actions(steps: list[dict]) -> dict:
    keys = ("type", "unit_slot", "cell_flat", "item_slot", "had_item")
    out = {}
    for k in keys:
        tensors = []
        for s in steps:
            v = (s.get("action") or {}).get(k)
            if torch.is_tensor(v):
                tensors.append(v.reshape(-1))
            elif k == "had_item" and v is not None:
                tensors.append(torch.tensor([bool(v)], dtype=torch.bool))
            elif v is not None:
                tensors.append(torch.tensor([v]))
        if tensors:
            out[k] = torch.cat(tensors, dim=0)
    return out


def _stack_hidden(steps: list[dict], hidden_dim: int, device: str):
    hs = []
    for s in steps:
        h = s.get("h_in")
        if torch.is_tensor(h):
            hs.append(h.reshape(1, -1))
        else:
            hs.append(torch.zeros(1, hidden_dim))
    return torch.cat(hs, dim=0).to(device)


# ---------------------------------------------------------------------------
# Check 1: Role -> concrete mapping
# ---------------------------------------------------------------------------

def check_role_mapping(
    steps: list[dict],
    vocab: Vocab,
) -> dict[str, Any]:
    """Static cheapest_of sanity + trajectory role->concrete rates."""
    result: dict[str, Any] = {
        "name": "role_concrete_mapping",
        "status": _SKIP,
        "detail": "",
        "numbers": {},
        "alerts": [],
    }

    # --- Static (always runnable) ---
    static = {
        "cheapest_refinery_proc_silo": cheapest_of(["proc", "silo"]),
        "cheapest_power": cheapest_of(["powr", "apwr"]),
        "cheapest_barracks": cheapest_of(["barr", "tent", "kenn"]),
        "role_of_proc": role_of("proc"),
        "role_of_silo": role_of("silo"),
        "silo_is_identity": "silo" in IDENTITY_ITEMS,
        "proc_role": ROLE_OF_ITEM.get("proc"),
        "silo_role": ROLE_OF_ITEM.get("silo"),
    }
    result["numbers"]["static"] = static
    alerts = []
    if static["cheapest_refinery_proc_silo"] != "proc":
        alerts.append(
            f"FAIL static: cheapest_of(proc,silo)={static['cheapest_refinery_proc_silo']!r} "
            f"(expected 'proc' - silo must not win)"
        )
    if static["role_of_proc"] != ROLE_REFINERY or static["role_of_silo"] != ROLE_REFINERY:
        alerts.append(
            f"WARN static: proc/silo roles = "
            f"{static['role_of_proc']}/{static['role_of_silo']} "
            f"(expected both {ROLE_REFINERY!r})"
        )
    if not static["silo_is_identity"]:
        alerts.append(
            "WARN static: silo not in IDENTITY_ITEMS - cheapest_of(refinery) "
            "could emit silo again if silo rejoins the bucket"
        )

    # --- Trajectory decode (if we have steps + vocab) ---
    vocab_inv = _vocab_id_to_name(vocab) if vocab is not None else {}
    role_counts: Counter = Counter()
    concrete_counts: Counter = Counter()
    silo_after_proc = 0
    refinery_when_has = 0
    power_spam = 0
    build_n = 0
    n_decoded = 0

    for s in steps:
        tname = sample_type_name(s)
        if tname not in ("build", "train", "place_building"):
            continue
        build_n += 1
        label = _decode_item_label(s, vocab_inv) if vocab_inv else ""
        if not label:
            continue
        n_decoded += 1
        # label is a role (or identity item name)
        rol = label if label in IDENTITY_ITEMS else role_of(label)
        if label in (ROLE_REFINERY, "silo", "proc") or rol == ROLE_REFINERY:
            rol = ROLE_REFINERY if label != "silo" else "silo"
        role_counts[(tname, label)] += 1

        # What concrete would the adapter emit?
        if label == "silo":
            concrete = "silo"
        elif label == "proc":
            concrete = "proc"
        elif label == ROLE_REFINERY:
            concrete = cheapest_of(concretos_de(ROLE_REFINERY) or ["proc", "silo"])
        elif label == ROLE_POWER_INFRA or rol == ROLE_POWER_INFRA:
            concrete = cheapest_of(concretos_de(ROLE_POWER_INFRA) or ["powr", "apwr"])
        else:
            items = concretos_de(label) or concretos_de(rol)
            if label in IDENTITY_ITEMS:
                concrete = label
            else:
                concrete = cheapest_of(items) if items else label
        concrete_counts[concrete] += 1

        batch = s.get("batch") or {}
        has_ref = _scalar_at(batch, _IDX_HAS_REFINERY)
        low_pwr = _scalar_at(batch, _IDX_LOW_POWER)
        pbal = _scalar_at(batch, _IDX_POWER_BALANCE)

        if tname in ("build", "place_building"):
            if has_ref >= 0.5 and (label == "silo" or concrete == "silo"):
                silo_after_proc += 1
            if has_ref >= 0.5 and label in (ROLE_REFINERY, "refinery", "proc", "silo"):
                refinery_when_has += 1
            if (
                label in (ROLE_POWER_INFRA, "power", "powr", "apwr")
                and low_pwr < 0.5
                and pbal > 0.15
            ):
                power_spam += 1

    traj = {
        "build_train_place_n": build_n,
        "decoded_n": n_decoded,
        "role_top": role_counts.most_common(20),
        "concrete_top": concrete_counts.most_common(20),
        "silo_count": int(concrete_counts.get("silo", 0)),
        "proc_count": int(concrete_counts.get("proc", 0)),
        "silo_after_proc": silo_after_proc,
        "refinery_role_when_has_refinery": refinery_when_has,
        "power_spam_approx": power_spam,
    }
    silo_n = traj["silo_count"]
    proc_n = traj["proc_count"]
    if proc_n > 0:
        traj["silo_proc_ratio"] = round(silo_n / proc_n, 3)
    elif silo_n > 0:
        traj["silo_proc_ratio"] = float("inf")
        alerts.append(
            f"WARN traj: {silo_n} silo concrete(s) and 0 proc in sampled builds"
        )
    else:
        traj["silo_proc_ratio"] = None

    if silo_after_proc > 0:
        alerts.append(
            f"WARN traj: silo chosen {silo_after_proc}x while has_refinery=1"
        )
    if power_spam >= 5:
        alerts.append(
            f"WARN traj: power build/place {power_spam}x with surplus power "
            f"(power_balance>0.15, low_power=0)"
        )

    result["numbers"]["trajectory"] = traj
    result["alerts"] = alerts

    hard_fail = any(a.startswith("FAIL") for a in alerts)
    if hard_fail:
        result["status"] = _FAIL
    elif alerts:
        result["status"] = _WARN
    else:
        result["status"] = _PASS

    bits = [
        f"cheapest(proc,silo)={static['cheapest_refinery_proc_silo']}",
        f"decoded={n_decoded}/{build_n}",
    ]
    if traj.get("silo_proc_ratio") is not None:
        bits.append(f"silo:proc={traj['silo_proc_ratio']}")
    if silo_after_proc:
        bits.append(f"silo@has_ref={silo_after_proc}")
    if power_spam:
        bits.append(f"power_spam~{power_spam}")
    result["detail"] = "; ".join(bits)
    return result


# ---------------------------------------------------------------------------
# Check 2: Action legality / water / self-base spam
# ---------------------------------------------------------------------------

def check_action_legality(
    steps: list[dict],
    metrics_rows: list[dict],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": "action_legality",
        "status": _SKIP,
        "detail": "skipped: no data",
        "numbers": {},
        "alerts": [],
    }
    alerts = []
    numbers: dict[str, Any] = {}

    # --- From stored spatial Ch3 (passable) ---
    cell_n = 0
    impassable = 0
    masked_illegal = 0  # cell_mask False at chosen cell
    type_ctr: Counter = Counter()
    combat_n = 0

    for s in steps:
        tname = sample_type_name(s)
        type_ctr[tname] += 1
        if tname not in TYPES_USE_CELL:
            continue
        combat_n += 1
        batch = s.get("batch") or {}
        act = s.get("action") or {}
        sp = batch.get("spatial")
        cf = _as_int(act.get("cell_flat"))
        if not torch.is_tensor(sp):
            continue
        # spatial [1,9,H,W] or [9,H,W]
        if sp.dim() == 4:
            sp = sp[0]
        if sp.dim() != 3 or sp.shape[0] < 4:
            continue
        _c, h, w = int(sp.shape[0]), int(sp.shape[1]), int(sp.shape[2])
        if w <= 0:
            continue
        y, x = divmod(cf, w)
        cell_n += 1
        if not (0 <= y < h and 0 <= x < w):
            impassable += 1
            continue
        passv = float(sp[3, y, x].item())
        if passv < 0.5:
            impassable += 1
        cm = batch.get("cell_mask")
        if torch.is_tensor(cm):
            flat = cm.reshape(-1)
            if 0 <= cf < flat.numel() and not bool(flat[cf].item()):
                masked_illegal += 1

    if cell_n > 0:
        rate = impassable / cell_n
        mask_rate = masked_illegal / cell_n
        numbers["cell_actions"] = cell_n
        numbers["impassable_or_oob"] = impassable
        numbers["impassable_rate"] = round(rate, 4)
        numbers["cell_mask_false"] = masked_illegal
        numbers["cell_mask_false_rate"] = round(mask_rate, 4)
        if rate > 0.15:
            alerts.append(
                f"FAIL cells: {impassable}/{cell_n} "
                f"({rate:.1%}) impassable/OOB (spatial Ch3)"
            )
        elif rate > 0.05:
            alerts.append(
                f"WARN cells: {impassable}/{cell_n} ({rate:.1%}) impassable/OOB"
            )
        if mask_rate > 0.10:
            alerts.append(
                f"WARN cells: {masked_illegal}/{cell_n} "
                f"({mask_rate:.1%}) violate stored cell_mask"
            )
    else:
        numbers["cell_actions"] = 0

    # --- From metrics action_hist (aggregate tail) ---
    hist_agg: Counter = Counter()
    n_rows_hist = 0
    for row in metrics_rows[-40:]:
        ah = row.get("action_hist") or {}
        if isinstance(ah, dict) and ah:
            n_rows_hist += 1
            for k, v in ah.items():
                try:
                    hist_agg[str(k)] += int(v)
                except (TypeError, ValueError):
                    continue
    if hist_agg:
        total = sum(hist_agg.values()) or 1
        top = hist_agg.most_common(12)
        numbers["metrics_action_hist_top"] = top
        numbers["metrics_action_hist_rows"] = n_rows_hist
        noop_frac = hist_agg.get("no_op", 0) / total
        army_frac = hist_agg.get("army_attack_move", 0) / total
        build_frac = (hist_agg.get("build", 0) + hist_agg.get("place_building", 0)) / total
        numbers["noop_frac"] = round(noop_frac, 4)
        numbers["army_attack_move_frac"] = round(army_frac, 4)
        numbers["build_place_frac"] = round(build_frac, 4)
        if noop_frac > 0.55:
            alerts.append(f"WARN hist: no_op dominates ({noop_frac:.1%} of last ~40 iters)")
        if army_frac > 0.70:
            alerts.append(
                f"WARN hist: army_attack_move very dominant ({army_frac:.1%}) "
                f"- possible self-base / push spam"
            )
        if build_frac < 0.005 and n_rows_hist >= 5:
            alerts.append(
                f"WARN hist: build+place only {build_frac:.2%} - economy head starved?"
            )

    if steps:
        numbers["step_type_top"] = type_ctr.most_common(12)

    if not steps and not hist_agg:
        result["status"] = _SKIP
        result["detail"] = "skipped: no steps and no action_hist in metrics"
        return result

    hard_fail = any(a.startswith("FAIL") for a in alerts)
    result["status"] = _FAIL if hard_fail else (_WARN if alerts else _PASS)
    result["alerts"] = alerts
    result["numbers"] = numbers
    bits = []
    if cell_n:
        bits.append(
            f"impassable={impassable}/{cell_n} "
            f"({numbers.get('impassable_rate', 0):.1%})"
        )
    if hist_agg:
        bits.append(
            f"noop={numbers.get('noop_frac', 0):.1%} "
            f"army_AM={numbers.get('army_attack_move_frac', 0):.1%}"
        )
    result["detail"] = "; ".join(bits) if bits else "ok"
    return result


# ---------------------------------------------------------------------------
# Check 3: Reward scale audit
# ---------------------------------------------------------------------------

def check_reward_scale(
    steps: list[dict],
    metrics_rows: list[dict],
    race_path: Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": "reward_scale",
        "status": _SKIP,
        "detail": "skipped: no data",
        "numbers": {},
        "alerts": [],
    }
    alerts = []
    numbers: dict[str, Any] = {}

    # Per-step rewards from buffers (not full shaped components).
    if steps:
        rews = []
        for s in steps:
            r = s.get("reward")
            if torch.is_tensor(r):
                rews.append(float(r.reshape(-1)[0].item()))
            elif r is not None:
                try:
                    rews.append(float(r))
                except (TypeError, ValueError):
                    pass
        if rews:
            numbers["step_reward"] = {
                "n": len(rews),
                "mean": round(sum(rews) / len(rews), 5),
                "min": round(min(rews), 5),
                "max": round(max(rews), 5),
                "abs_gt_20_frac": round(
                    sum(1 for x in rews if abs(x) > 20) / len(rews), 4
                ),
            }
            if numbers["step_reward"]["abs_gt_20_frac"] > 0.05:
                alerts.append(
                    f"WARN step rewards: "
                    f"{numbers['step_reward']['abs_gt_20_frac']:.1%} "
                    f"have |r|>20 (possible scale blow-up)"
                )

    # Aggregate reward_components from metrics tail.
    comp_sum: dict[str, float] = defaultdict(float)
    comp_n = 0
    timeout_vals = []
    garrison_vals = []
    mining_vals = []
    mean_ep_rews = []
    max_tick_hits = 0
    tick_samples = 0
    incomplete_n = 0
    outcome_n = 0

    tail = metrics_rows[-60:] if metrics_rows else []
    for row in tail:
        rc = row.get("reward_components")
        if isinstance(rc, dict) and rc:
            comp_n += 1
            for k, v in rc.items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                comp_sum[k] += fv
            if "timeout" in rc:
                try:
                    timeout_vals.append(float(rc["timeout"]))
                except (TypeError, ValueError):
                    pass
            if "garrison" in rc:
                try:
                    garrison_vals.append(float(rc["garrison"]))
                except (TypeError, ValueError):
                    pass
            if "mining" in rc:
                try:
                    mining_vals.append(float(rc["mining"]))
                except (TypeError, ValueError):
                    pass
        mer = row.get("mean_episode_reward")
        if mer is not None:
            try:
                mean_ep_rews.append(float(mer))
            except (TypeError, ValueError):
                pass
        ticks = row.get("ticks") or []
        if isinstance(ticks, list):
            for t in ticks:
                try:
                    ti = int(t)
                except (TypeError, ValueError):
                    continue
                tick_samples += 1
                if ti >= _DEFAULT_MAX_TICKS - 1:
                    max_tick_hits += 1
        outcomes = row.get("outcomes") or []
        if isinstance(outcomes, list):
            for o in outcomes:
                outcome_n += 1
                if str(o) == "incomplete":
                    incomplete_n += 1

    if comp_n:
        means = {k: round(v / comp_n, 4) for k, v in sorted(comp_sum.items())}
        numbers["reward_components_mean_tail"] = means
        numbers["reward_components_rows"] = comp_n
        # Magnitude ranking
        ranked = sorted(means.items(), key=lambda kv: -abs(kv[1]))
        numbers["reward_components_top_abs"] = ranked[:8]
        # Extremes
        if timeout_vals:
            numbers["timeout_component"] = {
                "mean": round(sum(timeout_vals) / len(timeout_vals), 4),
                "min": round(min(timeout_vals), 4),
                "max": round(max(timeout_vals), 4),
            }
            if min(timeout_vals) < -20:
                alerts.append(
                    f"WARN timeout component min={min(timeout_vals):.2f} "
                    f"(very punitive; check eradicate_v4 w_timeout)"
                )
        if garrison_vals:
            numbers["garrison_component"] = {
                "mean": round(sum(garrison_vals) / len(garrison_vals), 4),
                "min": round(min(garrison_vals), 4),
                "max": round(max(garrison_vals), 4),
            }
            if max(garrison_vals) > 5:
                alerts.append(
                    f"WARN garrison component max={max(garrison_vals):.2f} "
                    f"(turtle may out-earn push)"
                )
        if mining_vals:
            numbers["mining_component"] = {
                "mean": round(sum(mining_vals) / len(mining_vals), 4),
                "min": round(min(mining_vals), 4),
                "max": round(max(mining_vals), 4),
            }

    if mean_ep_rews:
        numbers["mean_episode_reward_tail"] = {
            "n": len(mean_ep_rews),
            "mean": round(sum(mean_ep_rews) / len(mean_ep_rews), 4),
            "min": round(min(mean_ep_rews), 4),
            "max": round(max(mean_ep_rews), 4),
        }

    if tick_samples:
        frac_max = max_tick_hits / tick_samples
        numbers["max_tick_hits"] = {
            "hits": max_tick_hits,
            "n": tick_samples,
            "frac": round(frac_max, 4),
            "max_ticks_assumed": _DEFAULT_MAX_TICKS,
        }
        if frac_max > 0.35:
            alerts.append(
                f"WARN {frac_max:.1%} of episode ticks hit ~{_DEFAULT_MAX_TICKS} "
                f"(timeout / incomplete heavy)"
            )
    if outcome_n:
        numbers["incomplete_frac_tail"] = round(incomplete_n / outcome_n, 4)
        if incomplete_n / outcome_n > 0.35:
            alerts.append(
                f"WARN incomplete outcomes {incomplete_n}/{outcome_n} "
                f"({incomplete_n / outcome_n:.1%}) in metrics tail"
            )

    # economy_race.jsonl - light sniff (file can be huge)
    if race_path and race_path.is_file():
        try:
            # Read last ~200KB only
            size = race_path.stat().st_size
            with race_path.open("rb") as f:
                if size > 200_000:
                    f.seek(-200_000, os.SEEK_END)
                    f.readline()  # sync to next line
                tail_bytes = f.read()
            lines = tail_bytes.decode("utf-8", errors="ignore").splitlines()
            race_n = 0
            harvest_edges = []
            for line in lines[-80:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                race_n += 1
                # metrics embeds economy_race; file rows are per-episode series
                # Prefer harvest deltas if present on the row.
                ow = row.get("own_earned") or row.get("own_harvest_per_1k")
                if isinstance(ow, list) and len(ow) >= 2:
                    try:
                        harvest_edges.append(float(ow[-1]) - float(ow[0]))
                    except (TypeError, ValueError):
                        pass
            numbers["economy_race_tail_rows"] = race_n
            if harvest_edges:
                numbers["economy_race_earned_delta_mean"] = round(
                    sum(harvest_edges) / len(harvest_edges), 2
                )
        except OSError as e:
            numbers["economy_race_error"] = str(e)

    if not numbers:
        result["status"] = _SKIP
        result["detail"] = "skipped: no reward data in steps/metrics"
        return result

    hard_fail = any(a.startswith("FAIL") for a in alerts)
    result["status"] = _FAIL if hard_fail else (_WARN if alerts else _PASS)
    result["alerts"] = alerts
    result["numbers"] = numbers
    bits = []
    if "reward_components_top_abs" in numbers:
        top3 = numbers["reward_components_top_abs"][:3]
        bits.append("top|" + ",".join(f"{k}={v}" for k, v in top3))
    if "max_tick_hits" in numbers:
        bits.append(
            f"timeout_tick_frac={numbers['max_tick_hits']['frac']:.1%}"
        )
    if "mean_episode_reward_tail" in numbers:
        bits.append(
            f"mean_ep_r={numbers['mean_episode_reward_tail']['mean']}"
        )
    result["detail"] = "; ".join(bits) if bits else "ok"
    return result


# ---------------------------------------------------------------------------
# Check 4: Policy sanity forward
# ---------------------------------------------------------------------------

@torch.no_grad()
def check_policy_forward(
    net: AlphaLiteNet,
    steps: list[dict],
    device: str,
    *,
    max_n: int = 500,
    mb: int = 16,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": "policy_forward",
        "status": _SKIP,
        "detail": "skipped: no data",
        "numbers": {},
        "alerts": [],
    }
    if net is None:
        result["detail"] = "skipped: no checkpoint loaded"
        return result
    if not steps:
        result["detail"] = "skipped: no observation steps for forward"
        return result

    alerts = []
    use = steps[:max_n]
    net.eval()
    type_ctr: Counter = Counter()
    entropies = []
    values = []
    nan_batches = 0
    inf_batches = 0
    n_ok = 0
    errors = []

    # Process in mini-batches (stored tensors already have leading 1).
    for i in range(0, len(use), mb):
        chunk = use[i : i + mb]
        try:
            batch = _stack_batches(chunk)
            actions = _stack_actions(chunk)
            if not batch or "spatial" not in batch or "type" not in actions:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            actions = {k: v.to(device) for k, v in actions.items()}
            # Ensure had_item exists
            if "had_item" not in actions:
                actions["had_item"] = torch.ones(
                    actions["type"].shape[0], dtype=torch.bool, device=device
                )
            h = _stack_hidden(chunk, net.core.hidden_size
                              if hasattr(net.core, "hidden_size")
                              else batch["scalars"].shape[0], device)
            # AlphaLiteNet GRU: hidden [B, H]
            if h.shape[-1] != net.core.weight_hh.shape[1]:
                # Fallback zeros with correct dim
                hdim = net.core.weight_hh.shape[1]
                h = torch.zeros(len(chunk), hdim, device=device)

            lp, ent, val, *_ = net.evaluate_actions(batch, h, actions)
            if not torch.isfinite(lp).all() or not torch.isfinite(ent).all() \
                    or not torch.isfinite(val).all():
                if (torch.isinf(lp).any() or torch.isinf(ent).any()
                        or torch.isinf(val).any()):
                    inf_batches += 1
                if (torch.isnan(lp).any() or torch.isnan(ent).any()
                        or torch.isnan(val).any()):
                    nan_batches += 1
            else:
                n_ok += int(lp.numel())
            entropies.extend(ent.detach().cpu().tolist())
            values.extend(val.detach().cpu().tolist())

            # Dominant modes from *policy sample* (act), not stored labels -
            # cheaper: use type head argmax via act(T=0) on same batch.
            out = net.act(batch, h, temperature=0.0)
            t = out["type"].detach().cpu().tolist()
            for ti in t:
                ti = int(ti)
                name = ACTION_TYPES[ti] if 0 <= ti < len(ACTION_TYPES) else str(ti)
                type_ctr[name] += 1
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
            if len(errors) >= 3:
                break

    numbers: dict[str, Any] = {
        "n_steps": len(use),
        "n_ok": n_ok,
        "nan_batches": nan_batches,
        "inf_batches": inf_batches,
        "action_type_hist": type_ctr.most_common(15),
        "errors": errors[:5],
    }
    if entropies:
        numbers["entropy"] = {
            "mean": round(sum(entropies) / len(entropies), 4),
            "min": round(min(entropies), 4),
            "max": round(max(entropies), 4),
        }
    if values:
        numbers["value"] = {
            "mean": round(sum(values) / len(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    if nan_batches or inf_batches:
        alerts.append(
            f"FAIL NaN/Inf in forward: nan_batches={nan_batches} "
            f"inf_batches={inf_batches}"
        )
    if errors and n_ok == 0:
        alerts.append(f"FAIL forward crashed: {errors[0]}")
        result["status"] = _FAIL
        result["alerts"] = alerts
        result["numbers"] = numbers
        result["detail"] = errors[0]
        return result

    if type_ctr:
        top_name, top_n = type_ctr.most_common(1)[0]
        top_frac = top_n / max(sum(type_ctr.values()), 1)
        numbers["dominant_action"] = {"type": top_name, "frac": round(top_frac, 4)}
        if top_frac > 0.85:
            alerts.append(
                f"WARN policy mode collapse: {top_name}={top_frac:.1%} of greedy acts"
            )
        if top_name == "no_op" and top_frac > 0.5:
            alerts.append(
                f"WARN greedy policy prefers no_op ({top_frac:.1%})"
            )

    if entropies:
        mean_e = sum(entropies) / len(entropies)
        if mean_e < 0.3:
            alerts.append(f"WARN very low mean entropy={mean_e:.3f}")
        if mean_e > 4.5:
            alerts.append(f"WARN very high mean entropy={mean_e:.3f} (near uniform?)")

    if n_ok == 0 and not errors:
        result["status"] = _SKIP
        result["detail"] = "skipped: could not run forward on steps"
        result["numbers"] = numbers
        return result

    hard_fail = any(a.startswith("FAIL") for a in alerts)
    result["status"] = _FAIL if hard_fail else (_WARN if alerts else _PASS)
    result["alerts"] = alerts
    result["numbers"] = numbers
    bits = [f"forward_ok={n_ok}/{len(use)}"]
    if entropies:
        bits.append(f"H={numbers['entropy']['mean']}")
    if type_ctr:
        bits.append(
            f"greedy_top={numbers['dominant_action']['type']}"
            f"@{numbers['dominant_action']['frac']:.0%}"
        )
    result["detail"] = "; ".join(bits)
    return result


# ---------------------------------------------------------------------------
# Check 5: Metrics regression sniff
# ---------------------------------------------------------------------------

def check_metrics_regression(rows: list[dict]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": "metrics_regression",
        "status": _SKIP,
        "detail": "skipped: no metrics.jsonl",
        "numbers": {},
        "alerts": [],
    }
    if not rows:
        return result

    # Drop era_reset marker rows for trend math
    data = [r for r in rows if isinstance(r.get("iter"), int) and not r.get("era_reset")]
    if not data:
        result["detail"] = "skipped: no numeric iter rows"
        return result

    alerts = []
    tail = data[-40:]
    last = data[-1]
    numbers: dict[str, Any] = {
        "n_rows": len(data),
        "last_iter": int(last.get("iter") or 0),
        "last_wr20": last.get("winrate_rolling20"),
        "last_winrate": last.get("winrate"),
        "last_heuristic_p": last.get("heuristic_p"),
        "last_kl": last.get("kl"),
        "last_clip_frac": last.get("clip_frac"),
        "last_entropy": last.get("entropy"),
        "onboard_phase": last.get("onboard_phase"),
        "bot_type": last.get("bot_type"),
    }

    # wr20 series
    wr20_series = []
    for r in tail:
        v = r.get("winrate_rolling20")
        if v is None:
            continue
        try:
            wr20_series.append((int(r["iter"]), float(v)))
        except (TypeError, ValueError):
            continue
    numbers["wr20_tail"] = wr20_series[-15:]

    # Trend: compare mean first half vs second half of wr20_tail
    if len(wr20_series) >= 8:
        vals = [v for _, v in wr20_series]
        mid = len(vals) // 2
        m0 = sum(vals[:mid]) / mid
        m1 = sum(vals[mid:]) / (len(vals) - mid)
        numbers["wr20_trend"] = {
            "early_mean": round(m0, 4),
            "late_mean": round(m1, 4),
            "delta": round(m1 - m0, 4),
        }
        if m1 + 1e-9 < m0 - 0.02:
            alerts.append(
                f"WARN wr20 declining: early={m0:.3f} -> late={m1:.3f} "
                f"(d={m1 - m0:+.3f}) over last {len(vals)} pts"
            )
        if m1 <= 0.0 and m0 <= 0.0:
            alerts.append("WARN wr20 stuck at 0.0 across metrics tail")

    # Consecutive zero-win iters (iter_winrate == 0 or no wins in outcomes)
    zero_streak = 0
    for r in reversed(data):
        iwr = r.get("iter_winrate")
        try:
            iwr_f = float(iwr) if iwr is not None else None
        except (TypeError, ValueError):
            iwr_f = None
        if iwr_f is not None:
            if iwr_f <= 0.0:
                zero_streak += 1
                continue
            break
        # fallback: outcomes
        outs = r.get("outcomes") or []
        if outs and not any(str(o).startswith("win") for o in outs):
            zero_streak += 1
            continue
        break
    numbers["consecutive_zero_win_iters"] = zero_streak
    if zero_streak >= 20:
        alerts.append(
            f"FAIL {zero_streak} consecutive zero-win iters "
            f"(iter_winrate=0 streak)"
        )
    elif zero_streak >= 10:
        alerts.append(
            f"WARN {zero_streak} consecutive zero-win iters"
        )

    # KL / clip alerts (tail means)
    kl_vals, clip_vals = [], []
    for r in tail:
        try:
            if r.get("kl") is not None:
                kl_vals.append(float(r["kl"]))
            if r.get("clip_frac") is not None:
                clip_vals.append(float(r["clip_frac"]))
        except (TypeError, ValueError):
            pass
    if kl_vals:
        numbers["kl_tail_mean"] = round(sum(kl_vals) / len(kl_vals), 5)
        numbers["kl_tail_max"] = round(max(kl_vals), 5)
        if numbers["kl_tail_mean"] >= 0.05:
            alerts.append(
                f"WARN KL elevated: mean={numbers['kl_tail_mean']} "
                f"max={numbers['kl_tail_max']} (hyper early_stop band ~0.05)"
            )
        elif numbers["kl_tail_mean"] >= 0.03:
            alerts.append(
                f"WARN KL soft-high: mean={numbers['kl_tail_mean']} "
                f"(>=0.03 lr_down band)"
            )
    if clip_vals:
        numbers["clip_frac_tail_mean"] = round(sum(clip_vals) / len(clip_vals), 5)
        if numbers["clip_frac_tail_mean"] >= 0.6:
            alerts.append(
                f"WARN clip_frac high: mean={numbers['clip_frac_tail_mean']}"
            )

    # heuristic_p presence
    hp = last.get("heuristic_p")
    if hp is not None:
        try:
            numbers["heuristic_p"] = float(hp)
            if float(hp) > 0.5 and str(last.get("onboard_phase") or "") in ("C", "S"):
                alerts.append(
                    f"WARN heuristic_p={hp} still high in phase "
                    f"{last.get('onboard_phase')}"
                )
        except (TypeError, ValueError):
            pass

    hard_fail = any(a.startswith("FAIL") for a in alerts)
    result["status"] = _FAIL if hard_fail else (_WARN if alerts else _PASS)
    result["alerts"] = alerts
    result["numbers"] = numbers
    bits = [
        f"iter={numbers['last_iter']}",
        f"wr20={numbers.get('last_wr20')}",
        f"zero_win_streak={zero_streak}",
    ]
    if "kl_tail_mean" in numbers:
        bits.append(f"kl={numbers['kl_tail_mean']}")
    if "wr20_trend" in numbers:
        bits.append(f"wr20_delta={numbers['wr20_trend']['delta']:+.3f}")
    result["detail"] = "; ".join(str(b) for b in bits)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Offline audit for OpenRA-RL checkpoints (no Docker)."
    )
    ap.add_argument(
        "--ckpt",
        default="rl/ckpts_v2/best.pt",
        help="Policy checkpoint (.pt). Default: rl/ckpts_v2/best.pt",
    )
    ap.add_argument(
        "--n", type=int, default=500,
        help="Max observation steps to load/forward (default 500)",
    )
    ap.add_argument(
        "--teacher-wins",
        default="rl/ckpts_v2/teacher_wins",
        help="teacher_wins/ directory (manifest + ep_XXXX.pt)",
    )
    ap.add_argument(
        "--elite",
        default="rl/ckpts_v2/elite.pt",
        help="elite.pt SIL buffer path",
    )
    ap.add_argument(
        "--metrics",
        default="rl/ckpts_v2/metrics.jsonl",
        help="metrics.jsonl path",
    )
    ap.add_argument(
        "--economy-race",
        default="rl/ckpts_v2/economy_race.jsonl",
        help="economy_race.jsonl path (optional sniff)",
    )
    ap.add_argument(
        "--out",
        default="rl/ckpts_v2/offline_audit_report.json",
        help="JSON report output path",
    )
    ap.add_argument(
        "--device", default="auto",
        help="cpu|cuda|auto (default auto)",
    )
    ap.add_argument(
        "--mb", type=int, default=16,
        help="Mini-batch size for policy forward (default 16)",
    )
    ap.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero if any check is FAIL",
    )
    ap.add_argument(
        "--no-teacher", action="store_true",
        help="Do not load teacher_wins (elite + metrics only)",
    )
    ap.add_argument(
        "--no-elite", action="store_true",
        help="Do not load elite.pt",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    t0 = time.time()

    ckpt_path = _resolve(args.ckpt)
    elite_path = None if args.no_elite else _resolve(args.elite)
    teacher_dir = None if args.no_teacher else _resolve(args.teacher_wins)
    metrics_path = _resolve(args.metrics)
    race_path = _resolve(args.economy_race)
    out_path = _resolve(args.out) or Path(args.out)

    print("=== OpenRA-RL offline audit ===")
    print(f"ckpt:     {ckpt_path}  exists={ckpt_path.is_file() if ckpt_path else False}")
    print(f"elite:    {elite_path}  exists={bool(elite_path and elite_path.is_file())}")
    print(f"teacher:  {teacher_dir}  exists={bool(teacher_dir and teacher_dir.is_dir())}")
    print(f"metrics:  {metrics_path}  exists={bool(metrics_path and metrics_path.is_file())}")
    print(f"n={args.n} device={args.device} strict={args.strict}")
    print()

    device = _pick_device(args.device)
    report: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ckpt": str(ckpt_path) if ckpt_path else None,
        "device": device,
        "n_requested": int(args.n),
        "checks": {},
        "summary": {},
        "limitations": [
            "No Docker / OpenRA - cannot verify live place legality or fog.",
            "Water/illegal cells approximated via stored spatial Ch3 + cell_mask.",
            "Role->concrete uses vocab+cheapest_of; faction-specific available "
            "lists are not reconstructed from raw obs (only from stored "
            "item_indices).",
            "Reward components come from metrics.jsonl aggregates, not a full "
            "re-shape of transitions (step rewards only when present on tapes).",
            "Teacher episodes are partially loaded (<=8 files) to avoid multi-GB RAM.",
        ],
    }

    # --- Load policy ---
    net = None
    vocab = Vocab()
    ckpt_iter = None
    if ckpt_path and ckpt_path.is_file():
        print(f"[load] checkpoint {ckpt_path} ...")
        net = AlphaLiteNet()
        ckpt_iter = load_checkpoint(str(ckpt_path), net, vocab=vocab)
        if not vocab.type_to_id:
            vocab.seed_roles()
        net.to(device)
        net.eval()
        n_params = sum(p.numel() for p in net.parameters())
        print(f"[load] iter={ckpt_iter} vocab={len(vocab.type_to_id)} "
              f"params={n_params / 1e6:.2f}M device={device}")
        report["ckpt_iter"] = ckpt_iter
        report["vocab_size"] = len(vocab.type_to_id)
    else:
        print("[load] WARN: checkpoint missing - policy_forward will SKIP; "
              "role decode uses seed_roles vocab")
        vocab.seed_roles()
        report["alerts_global"] = ["checkpoint missing"]

    # --- Load obs / metrics ---
    print("[load] observation buffers ...")
    steps, obs_meta = load_obs_steps(
        elite_path if elite_path and elite_path.is_file() else None,
        teacher_dir if teacher_dir and teacher_dir.is_dir() else None,
        int(args.n),
    )
    report["obs_meta"] = obs_meta
    print(f"[load] steps={len(steps)} sources={obs_meta.get('sources')} "
          f"elite={obs_meta.get('elite_steps')} "
          f"teacher_eps={obs_meta.get('teacher_eps_loaded')}")

    metrics_rows = []
    if metrics_path and metrics_path.is_file():
        metrics_rows = load_metrics_rows(metrics_path)
        print(f"[load] metrics rows={len(metrics_rows)}")
    else:
        print("[load] WARN: metrics.jsonl missing")

    # --- Run checks ---
    checks = []
    print("\n--- checks ---")
    c1 = check_role_mapping(steps, vocab)
    checks.append(c1)
    print(_status_line(c1["name"], c1["status"], c1["detail"]))
    for a in c1.get("alerts") or []:
        print(f"         ! {a}")

    c2 = check_action_legality(steps, metrics_rows)
    checks.append(c2)
    print(_status_line(c2["name"], c2["status"], c2["detail"]))
    for a in c2.get("alerts") or []:
        print(f"         ! {a}")

    c3 = check_reward_scale(steps, metrics_rows, race_path)
    checks.append(c3)
    print(_status_line(c3["name"], c3["status"], c3["detail"]))
    for a in c3.get("alerts") or []:
        print(f"         ! {a}")

    c4 = check_policy_forward(net, steps, device, max_n=int(args.n), mb=int(args.mb))
    checks.append(c4)
    print(_status_line(c4["name"], c4["status"], c4["detail"]))
    for a in c4.get("alerts") or []:
        print(f"         ! {a}")

    c5 = check_metrics_regression(metrics_rows)
    checks.append(c5)
    print(_status_line(c5["name"], c5["status"], c5["detail"]))
    for a in c5.get("alerts") or []:
        print(f"         ! {a}")

    for c in checks:
        report["checks"][c["name"]] = c

    statuses = [c["status"] for c in checks]
    n_fail = statuses.count(_FAIL)
    n_warn = statuses.count(_WARN)
    n_pass = statuses.count(_PASS)
    n_skip = statuses.count(_SKIP)
    report["summary"] = {
        "PASS": n_pass,
        "WARN": n_warn,
        "FAIL": n_fail,
        "SKIP": n_skip,
        "elapsed_s": round(time.time() - t0, 2),
        "SCALAR_DIM": SCALAR_DIM,
    }

    print("\n=== summary ===")
    print(
        f"PASS={n_pass} WARN={n_warn} FAIL={n_fail} SKIP={n_skip} "
        f"elapsed={report['summary']['elapsed_s']}s"
    )

    # Write JSON (convert non-JSON bits)
    def _jsonable(o):
        if isinstance(o, dict):
            return {str(k): _jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonable(x) for x in o]
        if isinstance(o, float):
            if math.isnan(o) or math.isinf(o):
                return str(o)
            return o
        if isinstance(o, (str, int, bool)) or o is None:
            return o
        return str(o)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_jsonable(report), indent=2), encoding="utf-8"
    )
    print(f"report -> {out_path}")

    if args.strict and n_fail > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
