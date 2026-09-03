"""Poor-man's PFSP over OpenRA *scripted bots* (not policy-vs-policy).

True checkpoint self-play needs RL-vs-RL in the bridge (Multi0 still ai.yaml).
Until that lands, this module mixes bot difficulties with the same sampling
idea: 50% vs an anchor bot, 50% vs a pool prioritized to who beats you.

Persists win/lose counts in rl/ckpts/pfsp_stats.json so auto_train relaunches
keep the league memory. Also rotates a frozen snapshot `prev20.pt` every
N iters — ready for real SP later, unused as an opponent today.
"""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

VALID_BOTS = ("beginner", "easy", "medium", "hard", "brutal", "dummy")

# Prioritize opponents with low winrate. eps keeps everyone in the mix.
_EPS = 0.05
_PRIOR_GAMES = 2.0  # Laplace: pretend PRIOR wins and PRIOR losses at start


class BotPFSP:
    """Sample bot_type per episode; track wr; rotate prev20.pt."""

    def __init__(
        self,
        ckpt_dir: str | Path,
        anchor: str = "easy",
        pool: list[str] | None = None,
        anchor_prob: float = 0.5,
        stats_name: str = "pfsp_stats.json",
        prev20_every: int = 20,
        rng: random.Random | None = None,
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.anchor = str(anchor or "easy")
        raw_pool = list(pool) if pool else ["beginner", "easy", "medium"]
        self.pool = []
        for b in raw_pool:
            b = str(b).strip().lower()
            if b in VALID_BOTS and b not in self.pool:
                self.pool.append(b)
        if self.anchor in VALID_BOTS and self.anchor not in self.pool:
            self.pool.insert(0, self.anchor)
        if not self.pool:
            self.pool = [self.anchor]
        self.anchor_prob = float(anchor_prob)
        self.stats_path = self.ckpt_dir / stats_name
        self.prev20_every = int(prev20_every)
        self.rng = rng or random.Random()
        self.stats = {b: {"wins": 0, "games": 0} for b in self.pool}
        self._load()

    def _load(self) -> None:
        if not self.stats_path.exists():
            return
        try:
            raw = json.loads(self.stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        bots = raw.get("bots") or {}
        for b, row in bots.items():
            if b not in VALID_BOTS:
                continue
            if b not in self.stats:
                self.stats[b] = {"wins": 0, "games": 0}
            self.stats[b]["wins"] = int(row.get("wins", 0) or 0)
            self.stats[b]["games"] = int(row.get("games", 0) or 0)
            if b not in self.pool:
                self.pool.append(b)

    def save(self) -> None:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "anchor": self.anchor,
            "pool": list(self.pool),
            "anchor_prob": self.anchor_prob,
            "bots": self.stats,
            "summary": self.summary(),
        }
        tmp = self.stats_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.stats_path)

    def winrate(self, bot: str) -> float:
        row = self.stats.get(bot) or {"wins": 0, "games": 0}
        w = float(row["wins"]) + _PRIOR_GAMES
        g = float(row["games"]) + 2.0 * _PRIOR_GAMES
        return w / g

    def priority(self, bot: str) -> float:
        """Higher when we lose more often to this bot."""
        return max(_EPS, 1.0 - self.winrate(bot))

    def sample(self) -> str:
        """50% anchor, else PFSP over the pool (may include the anchor)."""
        if self.rng.random() < self.anchor_prob:
            return self.anchor
        weights = [self.priority(b) for b in self.pool]
        total = sum(weights) or 1.0
        r = self.rng.random() * total
        acc = 0.0
        for b, w in zip(self.pool, weights):
            acc += w
            if r <= acc:
                return b
        return self.pool[-1]

    def record(self, bot: str, result: str) -> None:
        bot = str(bot or self.anchor)
        if bot not in self.stats:
            self.stats[bot] = {"wins": 0, "games": 0}
            if bot in VALID_BOTS and bot not in self.pool:
                self.pool.append(bot)
        self.stats[bot]["games"] += 1
        if str(result).startswith("win"):
            self.stats[bot]["wins"] += 1

    def record_many(self, outcomes: list[dict]) -> None:
        for o in outcomes or []:
            bt = o.get("bot_type") or self.anchor
            self.record(bt, o.get("result", ""))
        self.save()

    def summary(self) -> dict:
        out = {}
        for b, row in self.stats.items():
            g = int(row["games"])
            w = int(row["wins"])
            out[b] = {
                "wins": w,
                "games": g,
                "wr": round((w / g) if g else self.winrate(b), 3),
                "priority": round(self.priority(b), 3),
            }
        return out

    def maybe_rotate_prev20(self, iteration: int, latest_path: str | Path) -> bool:
        """Every prev20_every iters, copy latest.pt -> prev20.pt (frozen ring)."""
        if self.prev20_every <= 0:
            return False
        if int(iteration) % self.prev20_every != 0:
            return False
        latest = Path(latest_path)
        if not latest.exists():
            return False
        dest = self.ckpt_dir / "prev20.pt"
        tmp = self.ckpt_dir / "prev20.pt.tmp"
        try:
            shutil.copy2(latest, tmp)
            os.replace(tmp, dest)
            return True
        except OSError:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            return False


def parse_pool(s: str | None) -> list[str]:
    if not s:
        return ["beginner", "easy", "medium"]
    return [p.strip().lower() for p in str(s).split(",") if p.strip()]
