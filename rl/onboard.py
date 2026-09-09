"""Onboarding curriculum A → B → C (no .pt required).

Phase A: SFT from ScriptedTeacher vs beginner (no PPO).
Phase B: PPO+SIL+teacher BC vs beginner until wr20 holds.
Phase C: PPO+SIL vs easy until wr20 holds (no teacher: not an expert vs easy).

auto_train owns promotion (kill + relaunch). train.py only sees the flags
of the current phase. State lives in rl/ckpts/curriculum.json so a crash
does not restart SFT.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from rl.imitation import BC_WIN_CAP, BC_WIN_PREFER_TICKS

PHASES = ("A", "B", "C", "done")

DEFAULTS = {
    "sft_iters": 20,
    "a_promote_wr20": 0.25,
    "promote_wr20": 0.50,
    "done_wr20": 0.45,
    "streak": 10,
    "min_iters": 20,
    "bc_games": 4,
    "bc_epochs": 6,
    "a_macro_ticks": 40,
    "a_max_steps": 1800,
    "a_k_skip": 4,
    "a_eval_games": 4,
    "a_rush": 8,
    "bc_iters": 10000,
    # Phase B mixed BC (not A's 4 games / 6 epochs — that is SFT).
    "b_bc_games": 2,
    "b_bc_epochs": 2,
    "b_bc_warmup": 40,
    "b_bc_lambda_end": 0.10,
    "b_bc_start_iter": 0,  # 0 = use phase_started_iter
    "bc_win_cap": BC_WIN_CAP,
    "bc_win_prefer_ticks": BC_WIN_PREFER_TICKS,
    # C: linear P(easy) from c_mix_start → 1.0 over c_mix_warmup iters.
    # No BC vs easy (teacher is not an expert there). SIL + mix is the ancla.
    "c_mix_from": "beginner",
    "c_mix_warmup": 40,
    "c_mix_start": 0.25,
}

# Flags that TRAIN_ARGS may set and that a phase must replace or drop.
# store_true flags have no value; the rest consume the next argv token.
_STORE_TRUE = {
    "--pfsp", "--pfsp-rl", "--bc", "--bc-only", "--sil",
    "--reset-opt", "--bc-keep-incomplete", "--bc-replay",
    "--bc-collect-only", "--no-amp",
}
_STRIP = {
    "--bot-type", "--pfsp", "--pfsp-rl", "--pfsp-pool", "--pfsp-anchor-prob",
    "--bc", "--bc-only", "--bc-warmup", "--bc-start-iter", "--bc-teacher-bot",
    "--bc-games", "--bc-epochs", "--bc-keep-incomplete",
    "--bc-macro-ticks", "--bc-max-steps", "--bc-lambda-end",
    "--bc-rush", "--bc-replay", "--bc-collect-only", "--bc-collect-target",
    "--bc-win-cap", "--bc-win-prefer-ticks",
    "--eval-games",
    "--sil", "--lambda-sil",
    "--iters", "--reset-opt", "--onboard-phase",
    "--lr", "--adv-mode",
    "--mix-from", "--mix-warmup", "--mix-start", "--mix-start-iter",
    "--amp-init-scale", "--no-amp",
}


def strip_flags(argv: list[str], names: set[str] | None = None) -> list[str]:
    """Drop flags (and their values) in `names`. Preserves everything else."""
    drop = names if names is not None else set(_STRIP)
    out: list[str] = []
    i = 0
    n = len(argv)
    while i < n:
        a = argv[i]
        key = a.split("=", 1)[0]
        if key in drop:
            if "=" in a or key in _STORE_TRUE:
                i += 1
                continue
            i += 2
            continue
        out.append(a)
        i += 1
    return out


def _cfg(raw: dict | None) -> dict:
    out = dict(DEFAULTS)
    if raw:
        for k, v in raw.items():
            if k in DEFAULTS or k in (
                "phase", "a_launched", "c_reset_opt_done", "phase_started_iter",
            ):
                out[k] = v
    return out


def load_curriculum(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    phase = str(raw.get("phase") or "")
    if phase not in PHASES:
        return None
    cfg = _cfg(raw)
    cfg["phase"] = phase
    cfg["a_launched"] = bool(raw.get("a_launched"))
    cfg["c_reset_opt_done"] = bool(raw.get("c_reset_opt_done"))
    cfg["phase_started_iter"] = int(raw.get("phase_started_iter") or 0)
    return cfg


def save_curriculum(path: str | Path, cfg: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": cfg.get("phase", "A"),
        "sft_iters": int(cfg.get("sft_iters", DEFAULTS["sft_iters"])),
        "a_promote_wr20": float(
            cfg.get("a_promote_wr20", DEFAULTS["a_promote_wr20"])),
        "promote_wr20": float(cfg.get("promote_wr20", DEFAULTS["promote_wr20"])),
        "done_wr20": float(cfg.get("done_wr20", DEFAULTS["done_wr20"])),
        "streak": int(cfg.get("streak", DEFAULTS["streak"])),
        "min_iters": int(cfg.get("min_iters", DEFAULTS["min_iters"])),
        "bc_games": int(cfg.get("bc_games", DEFAULTS["bc_games"])),
        "bc_epochs": int(cfg.get("bc_epochs", DEFAULTS["bc_epochs"])),
        "a_macro_ticks": int(cfg.get("a_macro_ticks", DEFAULTS["a_macro_ticks"])),
        "a_max_steps": int(cfg.get("a_max_steps", DEFAULTS["a_max_steps"])),
        "a_k_skip": int(cfg.get("a_k_skip", DEFAULTS["a_k_skip"])),
        "a_eval_games": int(cfg.get("a_eval_games", DEFAULTS["a_eval_games"])),
        "a_rush": int(cfg.get("a_rush", DEFAULTS["a_rush"])),
        "bc_iters": int(cfg.get("bc_iters", DEFAULTS["bc_iters"])),
        "b_bc_games": int(cfg.get("b_bc_games", DEFAULTS["b_bc_games"])),
        "b_bc_epochs": int(cfg.get("b_bc_epochs", DEFAULTS["b_bc_epochs"])),
        "b_bc_warmup": int(cfg.get("b_bc_warmup", DEFAULTS["b_bc_warmup"])),
        "b_bc_lambda_end": float(
            cfg.get("b_bc_lambda_end", DEFAULTS["b_bc_lambda_end"])),
        "b_bc_start_iter": int(cfg.get("b_bc_start_iter") or 0),
        "bc_win_cap": int(cfg.get("bc_win_cap", DEFAULTS["bc_win_cap"])),
        "bc_win_prefer_ticks": int(
            cfg.get("bc_win_prefer_ticks", DEFAULTS["bc_win_prefer_ticks"])),
        "c_mix_from": str(cfg.get("c_mix_from") or DEFAULTS["c_mix_from"]),
        "c_mix_warmup": int(cfg.get("c_mix_warmup", DEFAULTS["c_mix_warmup"])),
        "c_mix_start": float(cfg.get("c_mix_start", DEFAULTS["c_mix_start"])),
        "a_launched": bool(cfg.get("a_launched")),
        "c_reset_opt_done": bool(cfg.get("c_reset_opt_done")),
        "phase_started_iter": int(cfg.get("phase_started_iter") or 0),
    }
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def new_curriculum(overrides: dict | None = None) -> dict:
    cfg = _cfg(overrides)
    cfg["phase"] = "A"
    cfg["a_launched"] = False
    cfg["c_reset_opt_done"] = False
    cfg["phase_started_iter"] = 0
    return cfg


def mix_target_prob(it: int, start_iter: int, warmup: int,
                    start_p: float) -> float:
    """Linear P(target bot) from start_p at start_iter to 1.0 over warmup.

    it == start_iter → start_p. it >= start_iter + warmup → 1.0.
    warmup<=0 → always 1.0 (no mix).
    """
    start_p = min(1.0, max(0.0, float(start_p)))
    warmup = int(warmup or 0)
    if warmup <= 0:
        return 1.0
    t = (int(it) - int(start_iter or 0)) / float(warmup)
    t = min(1.0, max(0.0, t))
    return start_p + (1.0 - start_p) * t


def snapshot_phase_best(ckpt_dir: str | Path, phase: str) -> Path | None:
    """Copy best.pt/json aside before a harder bot can overwrite them."""
    d = Path(ckpt_dir)
    src = d / "best.pt"
    if not src.exists():
        return None
    tag = str(phase or "B")
    dst = d / f"best_{tag}.pt"
    shutil.copy2(src, dst)
    js = d / "best.json"
    if js.exists():
        shutil.copy2(js, d / f"best_{tag}.json")
    return dst


def phase_flags(phase: str, cfg: dict) -> list[str]:
    """Argv overlay for one phase. Appended after stripped TRAIN_ARGS."""
    cfg = _cfg(cfg)
    if phase == "A":
        return [
            "--bot-type", "beginner",
            "--bc", "--bc-only",
            "--bc-teacher-bot", "beginner",
            "--bc-games", str(int(cfg["bc_games"])),
            "--bc-epochs", str(int(cfg["bc_epochs"])),
            "--bc-rush", str(int(cfg.get("a_rush") or DEFAULTS["a_rush"])),
            "--bc-start-iter", "1",
            "--bc-warmup", "1",
            "--macro-ticks", str(int(cfg["a_macro_ticks"])),
            "--max-steps", str(int(cfg["a_max_steps"])),
            "--k-skip", str(int(cfg["a_k_skip"])),
            "--eval-games", str(int(cfg.get("a_eval_games") or DEFAULTS["a_eval_games"])),
            "--bc-win-cap", str(int(cfg.get("bc_win_cap") or DEFAULTS["bc_win_cap"])),
            "--bc-win-prefer-ticks", str(int(
                cfg.get("bc_win_prefer_ticks") or DEFAULTS["bc_win_prefer_ticks"])),
            # Phase A: dense QSA/XF (topk=0 = off). Sparse topk masks enemy
            # push cells and starves attack BC (nll>=18). B/C keep TRAIN_ARGS.
            "--qsa-topk", "0",
            "--xf-topk", "0",
            "--lr", "1.0e-4",
            "--adv-mode", "episode",
            "--iters", str(int(cfg["bc_iters"])),
            "--onboard-phase", "A",
        ]
    if phase == "B":
        bc_start = int(cfg.get("b_bc_start_iter") or 0)
        if bc_start <= 0:
            bc_start = int(cfg.get("phase_started_iter") or 0) or 1
        return [
            "--bot-type", "beginner",
            "--bc",
            "--bc-teacher-bot", "beginner",
            # wins-only BC (no --bc-keep-incomplete): incompletes = timeout turtle
            "--bc-games", str(int(cfg.get("b_bc_games") or DEFAULTS["b_bc_games"])),
            "--bc-epochs", str(int(cfg.get("b_bc_epochs") or DEFAULTS["b_bc_epochs"])),
            "--bc-rush", str(int(cfg.get("a_rush") or DEFAULTS["a_rush"])),
            "--bc-warmup", str(int(cfg.get("b_bc_warmup") or DEFAULTS["b_bc_warmup"])),
            "--bc-lambda-end", "{:.2f}".format(float(
                cfg.get("b_bc_lambda_end", DEFAULTS["b_bc_lambda_end"]))),
            "--bc-start-iter", str(bc_start),
            "--bc-macro-ticks", str(int(cfg.get("a_macro_ticks") or DEFAULTS["a_macro_ticks"])),
            "--bc-max-steps", str(int(cfg.get("a_max_steps") or DEFAULTS["a_max_steps"])),
            "--bc-win-cap", str(int(cfg.get("bc_win_cap") or DEFAULTS["bc_win_cap"])),
            "--bc-win-prefer-ticks", str(int(
                cfg.get("bc_win_prefer_ticks") or DEFAULTS["bc_win_prefer_ticks"])),
            # 5e-5 still clip 0.55–0.96 in B (target ~0.20–0.30).
            "--lr", "2.0e-5",
            "--adv-mode", "global",
            "--sil", "--lambda-sil", "0.5",
            "--iters", str(int(cfg["bc_iters"])),
            "--onboard-phase", "B",
        ]
    if phase == "C":
        mix_start_iter = int(cfg.get("phase_started_iter") or 0) + 1
        return [
            "--bot-type", "easy",
            "--sil", "--lambda-sil", "0.5",
            # Same knobs as B: 1e-4 + Adam fresco + episode adv killed C.
            "--lr", "2.0e-5",
            "--adv-mode", "global",
            # C must not use AMP: skip_frac=1.0 killed learning.
            "--no-amp",
            "--mix-from", str(cfg.get("c_mix_from") or DEFAULTS["c_mix_from"]),
            "--mix-warmup", str(int(
                cfg.get("c_mix_warmup", DEFAULTS["c_mix_warmup"]))),
            "--mix-start", "{:.2f}".format(float(
                cfg.get("c_mix_start", DEFAULTS["c_mix_start"]))),
            "--mix-start-iter", str(max(1, mix_start_iter)),
            "--iters", str(int(cfg["bc_iters"])),
            "--onboard-phase", "C",
        ]
    return []


def build_train_argv(base: list[str], phase: str, cfg: dict) -> list[str]:
    """TRAIN_ARGS minus opponent/imitation flags, plus phase overlay.

    Last flag wins for duplicated keys that we re-add (macro-ticks, etc.).
    """
    return strip_flags(list(base)) + phase_flags(phase, cfg)


def should_resume(cfg: dict | None) -> bool:
    """First Phase A launch is tabula rasa; everything after resumes latest."""
    if cfg is None:
        return True
    if cfg.get("phase") == "A" and not cfg.get("a_launched"):
        return False
    return True


def outcomes_from_rows(rows: list[dict], bot_type: str,
                       onboard_phase: str | None = None) -> list[str]:
    """Flatten per-episode results for rows of this bot_type (skip era_reset).

    Mix rows (C ramp) are tagged bot_type=easy but opponent_bots lists
    the real rival per episode. Prefer that so beginner wins don't count
    as wr vs easy.
    """
    out: list[str] = []
    want = str(bot_type or "")
    for r in rows or []:
        if r.get("era_reset"):
            continue
        if not isinstance(r.get("iter"), int):
            continue
        if onboard_phase and r.get("onboard_phase") != onboard_phase:
            continue
        results = list(r.get("outcomes") or [])
        bots = r.get("opponent_bots")
        if isinstance(bots, list) and len(bots) == len(results):
            for res, b in zip(results, bots):
                if want and str(b or "") != want:
                    continue
                out.append(str(res))
            continue
        if want and str(r.get("bot_type") or "") != want:
            continue
        for res in results:
            out.append(str(res))
    return out


def wr20(results: list[str]) -> float:
    if not results:
        return 0.0
    window = results[-20:]
    return sum(1 for r in window if r.startswith("win")) / len(window)


def wr20_streak(results: list[str], threshold: float, streak: int,
                games_per_iter: int = 4) -> int:
    """How many trailing *iters* have wr20 >= threshold.

    wr20 is computed on the last 20 episodes at the end of each iter.
    `games_per_iter` is the number of outcomes appended per metrics row
    (train --episodes, default 4).
    """
    if streak <= 0 or not results:
        return 0
    gpi = max(1, int(games_per_iter))
    n_iters = len(results) // gpi
    held = 0
    for i in range(1, n_iters + 1):
        end = i * gpi
        w = wr20(results[:end])
        if w >= float(threshold) and end >= 20:
            held += 1
        else:
            held = 0
    return held


def phase_bot(phase: str) -> str:
    if phase == "C":
        return "easy"
    return "beginner"


def iters_in_phase(rows: list[dict], bot_type: str, started_iter: int,
                   onboard_phase: str | None = None) -> int:
    n = 0
    want = str(bot_type or "")
    start = int(started_iter or 0)
    for r in rows or []:
        if r.get("era_reset") or not isinstance(r.get("iter"), int):
            continue
        if want and str(r.get("bot_type") or "") != want:
            continue
        if onboard_phase and r.get("onboard_phase") != onboard_phase:
            continue
        if int(r["iter"]) > start:
            n += 1
    return n


def drought_since_iter(cfg: dict | None, last_restore_iter: int = 0) -> int:
    """Ignore previous-phase wr20 when deciding C (or B) drought restore."""
    restore = int(last_restore_iter or 0)
    if not cfg:
        return restore
    started = int(cfg.get("phase_started_iter") or 0)
    return max(restore, started)


def drought_blocked_until_min_iters(cfg: dict | None, rows: list[dict]) -> bool:
    """C: wr vs easy starts at 0 — that is not a drought. Wait min_iters."""
    if not cfg or cfg.get("phase") != "C":
        return False
    min_it = int(cfg.get("min_iters") or DEFAULTS["min_iters"])
    n_phase = iters_in_phase(
        rows, phase_bot("C"), int(cfg.get("phase_started_iter") or 0),
        onboard_phase="C")
    return n_phase < min_it


def should_promote(cfg: dict, rows: list[dict], last_iter: int,
                   games_per_iter: int = 4) -> str | None:
    """Return the next phase name, or None if the current one continues."""
    phase = cfg.get("phase")
    if phase == "A":
        # Use A-tagged rows only. A dirty metrics.jsonl from another run
        # (iter 400) must not skip SFT. Easy gate: sft_iters floor + current
        # student wr20 >= a_promote_wr20 (no streak). B/C keep hard bar.
        a_iters = [
            int(r["iter"]) for r in (rows or [])
            if isinstance(r.get("iter"), int)
            and r.get("onboard_phase") == "A"
        ]
        if not a_iters or max(a_iters) < int(
                cfg.get("sft_iters") or DEFAULTS["sft_iters"]):
            return None
        results = outcomes_from_rows(rows, "beginner", onboard_phase="A")
        if not results:
            return None
        thr = float(cfg.get("a_promote_wr20")
                    if cfg.get("a_promote_wr20") is not None
                    else DEFAULTS["a_promote_wr20"])
        if wr20(results) >= thr:
            return "B"
        return None
    if phase not in ("B", "C"):
        return None
    bot = phase_bot(phase)
    results = outcomes_from_rows(rows, bot, onboard_phase=phase)
    thr = float(cfg["promote_wr20"] if phase == "B" else cfg["done_wr20"])
    need = int(cfg.get("streak") or DEFAULTS["streak"])
    min_it = int(cfg.get("min_iters") or DEFAULTS["min_iters"])
    held = wr20_streak(results, thr, need, games_per_iter=games_per_iter)
    n_phase = iters_in_phase(
        rows, bot, int(cfg.get("phase_started_iter") or 0),
        onboard_phase=phase)
    if held >= need and n_phase >= min_it:
        return "C" if phase == "B" else "done"
    return None


def era_reset_row(note: str, bot_type: str) -> dict:
    """Sentinel that zeros train.py era wins/total hydration."""
    return {
        "note": note,
        "wins": 0,
        "total": 0,
        "era_reset": True,
        "bot_type": bot_type,
    }


def append_era_reset(metrics_path: str | Path, note: str, bot_type: str) -> None:
    p = Path(metrics_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(era_reset_row(note, bot_type)) + "\n")


def truncate_jsonl(path: str | Path, keep_iter: int) -> int:
    """Keep lines with no iter, or iter <= keep_iter. Returns kept count."""
    p = Path(path)
    if not p.exists():
        return 0
    kept: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        it = row.get("iter") if isinstance(row, dict) else None
        if not isinstance(it, int) or it <= int(keep_iter):
            kept.append(s)
    tmp = p.with_suffix(p.suffix + ".tmp")
    text = ("\n".join(kept) + "\n") if kept else ""
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)
    return len(kept)


def resolve_rewind_ckpt(ckpt_dir: str | Path, keep_iter: int) -> Path:
    """best.pt if best.json.iter matches, else iterNNNN.pt."""
    d = Path(ckpt_dir)
    keep_iter = int(keep_iter)
    best_json = d / "best.json"
    best_pt = d / "best.pt"
    if best_json.exists() and best_pt.exists():
        try:
            meta = json.loads(best_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
        if isinstance(meta, dict) and int(meta.get("iter") or -1) == keep_iter:
            return best_pt
    decade = d / f"iter{keep_iter:04d}.pt"
    if decade.exists():
        return decade
    raise FileNotFoundError(
        f"no ckpt for iter {keep_iter}: need best.pt@that iter or {decade.name}")


def rewind_onboard(ckpt_dir: str | Path, keep_iter: int,
                   cfg: dict | None = None) -> tuple[dict, dict]:
    """Restore latest.pt to keep_iter and drop later metrics/race.

    Does not restart λ_bc at 1.0 (pinning keep_iter did that and clip_frac
    exploded 0.7→0.96). Warmup origin stays phase B start so λ is at the
    floor on the next launch.

    Also copies src → best.pt (C used to leave a 0-win easy best that
    collapse would restore). Allowed from B or C; keep_iter after A
    returns to B.
    """
    d = Path(ckpt_dir)
    keep_iter = int(keep_iter)
    src = resolve_rewind_ckpt(d, keep_iter)
    shutil.copy2(src, d / "latest.pt")
    best_pt = d / "best.pt"
    if src.resolve() != best_pt.resolve():
        shutil.copy2(src, best_pt)
    n_m = truncate_jsonl(d / "metrics.jsonl", keep_iter)
    n_r = truncate_jsonl(d / "economy_race.jsonl", keep_iter)
    _write_rewind_best_json(d, keep_iter)
    elite_p = d / "elite.pt"
    if elite_p.exists():
        try:
            elite_p.unlink()
        except OSError:
            pass
    if cfg is None:
        cfg = load_curriculum(d / "curriculum.json") or new_curriculum()
    sft = int(cfg.get("sft_iters") or DEFAULTS["sft_iters"])
    cfg["a_launched"] = True
    if keep_iter <= sft:
        # Back to A (SFT seed). Don't pin BC start.
        cfg["phase"] = "A"
        cfg["b_bc_start_iter"] = 0
        cfg["phase_started_iter"] = 0
    else:
        if cfg.get("phase") not in ("B", "C"):
            raise ValueError(
                f"rewind > sft_iters solo en fase B o C "
                f"(curriculum phase={cfg.get('phase')!r})")
        warmup = int(cfg.get("b_bc_warmup") or DEFAULTS["b_bc_warmup"])
        from_c = cfg.get("phase") == "C"
        origin = int(cfg.get("b_bc_start_iter") or 0)
        if origin <= 0:
            origin = int(cfg.get("phase_started_iter") or 0)
        cfg["phase"] = "B"
        cfg["c_reset_opt_done"] = False
        if from_c:
            # Undo a bad C: λ stays at the B floor; min_iters restarts
            # from keep_iter so wr20 leftover from B no re-promueve ya.
            if origin > 0:
                cfg["b_bc_start_iter"] = origin
            else:
                cfg["b_bc_start_iter"] = max(1, keep_iter + 1 - warmup)
            cfg["phase_started_iter"] = keep_iter
        elif origin > 0 and origin <= keep_iter:
            cfg["b_bc_start_iter"] = origin
        else:
            cfg["phase_started_iter"] = keep_iter
            cfg["b_bc_start_iter"] = max(1, keep_iter + 1 - warmup)
    save_curriculum(d / "curriculum.json", cfg)
    info = {"src": src.name, "metrics_kept": n_m, "race_kept": n_r,
            "phase": cfg["phase"]}
    return cfg, info


def _write_rewind_best_json(ckpt_dir: Path, keep_iter: int) -> None:
    """best.json matching the restored weights (bot_type of that iter)."""
    d = Path(ckpt_dir)
    row = None
    metrics = d / "metrics.jsonl"
    if metrics.exists():
        try:
            for line in metrics.read_text(encoding="utf-8").splitlines():
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(j, dict) and j.get("iter") == keep_iter:
                    row = j
        except OSError:
            row = None
    payload = {
        "iter": keep_iter,
        "reason": "onboard_rewind",
        "bot_type": (row or {}).get("bot_type") or "beginner",
        "winrate": (row or {}).get("winrate"),
        "iter_winrate": (row or {}).get("iter_winrate"),
        "winrate_rolling20": (row or {}).get("winrate_rolling20"),
        "outcomes": (row or {}).get("outcomes"),
        "ticks": (row or {}).get("ticks"),
    }
    (d / "best.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
