# -*- coding: utf-8 -*-
"""Unit tests for BotPFSP (scripted-bot league, not RL-vs-RL)."""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.pfsp import BotPFSP, parse_pool

ok = True


def check(name, cond):
    global ok
    print(f"  [{'OK' if cond else 'FALLA'}] {name}")
    ok = ok and bool(cond)


print("=== pfsp bots ===")
check("parse_pool default", parse_pool(None) == ["beginner", "easy", "medium"])
check("parse_pool custom", parse_pool("easy, hard") == ["easy", "hard"])

with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    league = BotPFSP(d, anchor="easy", pool=["beginner", "easy", "medium"],
                     anchor_prob=0.5, rng=random.Random(0))
    # Force losses vs medium -> high priority
    for _ in range(20):
        league.record("medium", "lose")
    for _ in range(20):
        league.record("beginner", "win")
    check("medium priority > beginner",
          league.priority("medium") > league.priority("beginner"))
    samples = [league.sample() for _ in range(400)]
    n_easy = sum(1 for s in samples if s == "easy")
    # Anchor coin 50% plus easy also in the PFSP pool => easy share >= ~50%.
    check("easy at least half", n_easy >= 180)
    import random as _rnd
    always = BotPFSP(d, anchor="easy", pool=["beginner", "easy"],
                     anchor_prob=1.0, rng=_rnd.Random(1))
    check("anchor_prob 1 always easy",
          all(always.sample() == "easy" for _ in range(40)))
    # Among non-anchor draws, medium should dominate
    non = [s for s in samples if s != "easy"]
    n_med = sum(1 for s in non if s == "medium")
    n_beg = sum(1 for s in non if s == "beginner")
    check("PFSP favors medium over beginner", n_med > n_beg)
    league.save()
    check("stats file written", (d / "pfsp_stats.json").exists())
    league2 = BotPFSP(d, anchor="easy", pool=["beginner", "easy", "medium"],
                      rng=random.Random(1))
    check("stats reloaded", league2.stats["medium"]["games"] == 20)
    # prev20 rotate
    latest = d / "latest.pt"
    latest.write_bytes(b"CKPT")
    check("prev20 at iter 20", league.maybe_rotate_prev20(20, latest) is True)
    check("prev20 bytes", (d / "prev20.pt").read_bytes() == b"CKPT")
    check("prev20 skip odd", league.maybe_rotate_prev20(21, latest) is False)

print("\n" + ("TODOS LOS TESTS OK" if ok else "HAY FALLAS"))
sys.exit(0 if ok else 1)
