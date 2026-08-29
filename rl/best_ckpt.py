"""Select and copy rl/ckpts/best.pt without using mean_episode_reward.

Called from train.py after each PPO iter writes latest.pt. On Windows we
COPY the file (os.replace of a temp copy) — never symlink, because live
and dashboard open the path as a regular file.

Selection (strictly better only, so the pointer does not flap):
  1) higher per-iter winrate (then winrate_rolling20 if both present)
  2) if tied — including all-zero wins — higher viability

Viability is NOT raw mean_episode_reward (garrison / mining-farm collapse
can inflate that). It prefers razing, some mining / first_ore, and own
buildings still standing, and heavily penalizes deploy-then-no_op:
own buildings ~0 AND attack_move collapsed AND several quick ~9k-tick
loses must NOT beat an iter that built a base and razed.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

# Quick-wipe / empty-ore death on a_short is ~8-9k ticks.
COLLAPSE_TICK_HI = 10500
# Typical beginner base size on a_short; fewer remaining => razed.
ENEMY_BASE_NOM = 6.0


def iter_winrate(row: dict) -> float:
    """Per-iteration winrate. Prefer explicit field, else outcomes, else era wr."""
    if row.get("iter_winrate") is not None:
        return float(row["iter_winrate"])
    outs = row.get("outcomes") or []
    if outs:
        return sum(1 for r in outs if str(r).startswith("win")) / len(outs)
    return float(row.get("winrate") or 0.0)


def viability_breakdown(row: dict) -> dict:
    """Exact viability formula used to break winrate ties.

    score =
        4.0 * raze
      + 2.0 * max(0, 6 - enemy_buildings)     # enemy buildings destroyed proxy
      + 3.0 * mined_any                       # 1 if mining or first_ore or harvest
      + 1.5 * mining
      + 2.0 * first_ore
      + 1.0 * min(own_buildings, 8)
      + 0.002 * own_harvest_per_1k
      - 20.0 if collapse else 0

    collapse iff:
        own_buildings < 0.5
        AND (attack_move + army_attack_move) / n_actions < 0.02
        AND count(lose with ticks < 10500) >= 3
        AND mean_ticks < 10500
    """
    rc = row.get("reward_components") or {}
    raze = float(rc.get("raze", 0) or 0)
    mining = float(rc.get("mining", 0) or 0)
    first_ore = float(rc.get("first_ore", 0) or 0)

    n_b = row.get("n_buildings") or {}
    own_b = float(n_b.get("own", 0) or 0)
    enemy_b = float(n_b.get("enemy", 0) or 0)
    enemy_destroyed = max(0.0, ENEMY_BASE_NOM - enemy_b)

    eco = row.get("economy_race") or {}
    harvest_1k = float(eco.get("own_harvest_per_1k", 0) or 0)
    mined_any = 1.0 if (mining > 0.0 or first_ore > 0.0 or harvest_1k > 0.0) else 0.0

    hist = row.get("action_hist") or {}
    n_act = float(sum(int(v) for v in hist.values()) or 0)
    atk = float(hist.get("attack_move", 0) or 0) + float(
        hist.get("army_attack_move", 0) or 0)
    atk_frac = (atk / n_act) if n_act > 0 else 0.0

    ticks = list(row.get("ticks") or [])
    outcomes = list(row.get("outcomes") or [])
    mean_ticks = (sum(float(t) for t in ticks) / len(ticks)) if ticks else 0.0
    n_quick_lose = 0
    for i, r in enumerate(outcomes):
        t = float(ticks[i]) if i < len(ticks) else mean_ticks
        if str(r).startswith("lose") and t < COLLAPSE_TICK_HI:
            n_quick_lose += 1

    collapse = (
        own_b < 0.5
        and atk_frac < 0.02
        and n_quick_lose >= 3
        and mean_ticks < COLLAPSE_TICK_HI
    )
    collapse_pen = 20.0 if collapse else 0.0

    score = (
        4.0 * raze
        + 2.0 * enemy_destroyed
        + 3.0 * mined_any
        + 1.5 * mining
        + 2.0 * first_ore
        + 1.0 * min(own_b, 8.0)
        + 0.002 * harvest_1k
        - collapse_pen
    )
    return {
        "score": round(float(score), 6),
        "raze": raze,
        "enemy_destroyed": enemy_destroyed,
        "mined_any": mined_any,
        "mining": mining,
        "first_ore": first_ore,
        "own_buildings": own_b,
        "enemy_buildings": enemy_b,
        "harvest_1k": harvest_1k,
        "atk_frac": round(atk_frac, 4),
        "mean_ticks": round(mean_ticks, 1),
        "n_quick_lose": n_quick_lose,
        "collapse": bool(collapse),
        "collapse_pen": collapse_pen,
    }


def viability_score(row: dict) -> float:
    return float(viability_breakdown(row)["score"])


def attack_spam_frac(row: dict) -> float:
    """Fraction of actions that are combat-move (attack / attack_move / army)."""
    hist = row.get("action_hist") or {}
    n_act = float(sum(int(v) for v in hist.values()) or 0)
    atk = (float(hist.get("attack_move", 0) or 0)
           + float(hist.get("army_attack_move", 0) or 0)
           + float(hist.get("attack", 0) or 0))
    return (atk / n_act) if n_act > 0 else 0.0


def is_attack_spam_collapse(row: dict) -> bool:
    """True when the policy is the 275-309 mode: no base, 0 rolling wr, >90% attack."""
    n_b = row.get("n_buildings") or {}
    own_b = float(n_b.get("own", 0) or 0)
    r20 = float(row.get("winrate_rolling20") or 0.0)
    return own_b < 0.5 and r20 <= 0.0 and attack_spam_frac(row) > 0.90


def action_frac(row: dict, keys) -> float:
    hist = row.get("action_hist") or {}
    n_act = float(sum(int(v) for v in hist.values()) or 0)
    if n_act <= 0:
        return 0.0
    return sum(float(hist.get(k, 0) or 0) for k in keys) / n_act


def dead_policy_reason(row: dict) -> str | None:
    """Why this metrics row is a dead policy, or None if it still plays.

    Covers the two observed collapse modes plus a generic entropy crash:
      - attack-spam (Run 7): no base, wr20=0, >90% combat
      - deploy-noop (Run 8): viability collapse, or wr20=0 and >80% no_op
      - entropy crash: wr20=0, no base, H<0.15
    """
    if is_attack_spam_collapse(row):
        return "attack-spam"
    if viability_breakdown(row).get("collapse"):
        return "deploy-noop"
    r20 = float(row.get("winrate_rolling20") or 0.0)
    if r20 <= 0.0 and action_frac(row, ("no_op",)) > 0.80:
        return "no_op-spam"
    n_b = row.get("n_buildings") or {}
    own_b = float(n_b.get("own", 0) or 0)
    h = row.get("entropy")
    if (r20 <= 0.0 and own_b < 0.5 and h is not None
            and float(h) < 0.15):
        return "entropy-crash"
    return None


def is_dead_policy(row: dict) -> bool:
    return dead_policy_reason(row) is not None


def batch_is_dead(outcomes) -> bool:
    """True when this PPO batch is 80%+ no_op — skip the update, don't drive H to 0."""
    hist = {}
    for o in outcomes or []:
        for k, v in (o.get("action_hist") or {}).items():
            hist[k] = hist.get(k, 0) + int(v or 0)
    n = float(sum(hist.values()) or 0)
    if n <= 0:
        return False
    return hist.get("no_op", 0) / n > 0.80


def is_strictly_better(cand: dict, best: dict) -> tuple[bool, str]:
    """True only when cand beats best. Equal scores do not replace (no flap)."""
    wr_c, wr_b = iter_winrate(cand), iter_winrate(best)
    if wr_c > wr_b:
        return True, "higher_iter_winrate"
    if wr_c < wr_b:
        return False, "lower_iter_winrate"
    # Era winrate before rolling20: first win sets rolling20=0.25 on a
    # 4-ep window and would freeze best.pt there all night.
    era_c = float(cand.get("winrate") or 0.0)
    era_b = float(best.get("winrate") or 0.0)
    if era_c > era_b:
        return True, "higher_era_winrate"
    if era_c < era_b:
        return False, "lower_era_winrate"
    r20_c, r20_b = cand.get("winrate_rolling20"), best.get("winrate_rolling20")
    if r20_c is not None and r20_b is not None:
        r20_c, r20_b = float(r20_c), float(r20_b)
        if r20_c > r20_b:
            return True, "higher_winrate_rolling20"
        if r20_c < r20_b:
            return False, "lower_winrate_rolling20"
    v_c, v_b = viability_score(cand), viability_score(best)
    if v_c > v_b:
        return True, "higher_viability"
    return False, "not_strictly_better"


def maybe_update_best(ckpt_dir, metrics_row: dict,
                      latest_path: str | None = None) -> bool:
    """Copy latest.pt -> best.pt + write best.json if this iter is strictly better.

    Returns True if best.pt was written/replaced.
    """
    ckpt_dir = Path(ckpt_dir)
    latest = Path(latest_path) if latest_path else ckpt_dir / "latest.pt"
    best_pt = ckpt_dir / "best.pt"
    best_json = ckpt_dir / "best.json"
    if not latest.exists():
        return False

    current = None
    if best_json.exists():
        try:
            current = json.loads(best_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None

    if current is None:
        better, reason = True, "first"
    else:
        better, reason = is_strictly_better(metrics_row, current)
    if not better:
        return False

    tmp = ckpt_dir / "best.pt.tmp"
    try:
        shutil.copy2(latest, tmp)
        os.replace(tmp, best_pt)
    except OSError as e:
        print(f"[best.pt] copy failed: {e}", flush=True)
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        return False

    vig = viability_breakdown(metrics_row)
    payload = {
        "iter": metrics_row.get("iter"),
        "reason": reason,
        "winrate": metrics_row.get("winrate"),
        "iter_winrate": iter_winrate(metrics_row),
        "winrate_rolling20": metrics_row.get("winrate_rolling20"),
        "viability": vig["score"],
        "scores": vig,
        "mean_episode_reward": metrics_row.get("mean_episode_reward"),
        # Persist the fields needed to re-compare (do not flap on equal).
        "reward_components": metrics_row.get("reward_components"),
        "n_buildings": metrics_row.get("n_buildings"),
        "action_hist": metrics_row.get("action_hist"),
        "economy_race": metrics_row.get("economy_race"),
        "outcomes": metrics_row.get("outcomes"),
        "ticks": metrics_row.get("ticks"),
    }
    best_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[best.pt] updated iter={payload['iter']} reason={reason} "
        f"iter_wr={payload['iter_winrate']:.3f} vig={vig['score']:.3f}",
        flush=True,
    )
    return True