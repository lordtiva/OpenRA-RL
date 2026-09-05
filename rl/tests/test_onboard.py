# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.onboard import (
    build_train_argv,
    era_reset_row,
    new_curriculum,
    outcomes_from_rows,
    phase_flags,
    rewind_onboard,
    should_promote,
    should_resume,
    strip_flags,
    wr20,
    wr20_streak,
)

ok = True


def check(name, cond):
    global ok
    print(f"  [{'OK' if cond else 'FALLA'}] {name}")
    ok = ok and bool(cond)


print("=== onboard curriculum ===")

base = [
    "python", "-m", "rl.train",
    "--url", "http://localhost:8000",
    "--iters", "400",
    "--bot-type", "easy",
    "--pfsp", "--pfsp-rl", "--pfsp-pool", "rl",
    "--sil", "--lambda-sil", "0.5",
    "--macro-ticks", "50",
    "--max-steps", "1000",
    "--auto-support", "--no-war-nudge",
]
stripped = strip_flags(base)
check("strip saca --pfsp", "--pfsp" not in stripped)
check("strip saca --bot-type y valor", "--bot-type" not in stripped and "easy" not in stripped)
check("strip conserva --auto-support", "--auto-support" in stripped)
check("strip conserva --macro-ticks (A lo pisa)", "--macro-ticks" in stripped)

cfg = new_curriculum()
check("new curriculum fase A", cfg["phase"] == "A")
check("primer A no resume", should_resume(cfg) is False)
cfg["a_launched"] = True
check("A ya lanzada resume", should_resume(cfg) is True)
cfg["phase"] = "B"
check("B resume", should_resume(cfg) is True)

fa = phase_flags("A", cfg)
check("A es bc-only", "--bc-only" in fa and "--bc" in fa)
check("A teacher beginner", fa[fa.index("--bc-teacher-bot") + 1] == "beginner")
check("A 4 games paralelo", fa[fa.index("--bc-games") + 1] == "4")
check("A eval alumno 4", fa[fa.index("--eval-games") + 1] == "4")
check("A iters largo (wr gate, no sft_iters)",
      fa[fa.index("--iters") + 1] == "10000")
check("A macro 20", fa[fa.index("--macro-ticks") + 1] == "20")
check("A no sil", "--sil" not in fa)
fb = phase_flags("B", cfg)
check("B sil beginner", "--sil" in fb and fb[fb.index("--bot-type") + 1] == "beginner")
check("B bc mezclado no bc-only", "--bc" in fb and "--bc-only" not in fb)
check("B teacher beginner", fb[fb.index("--bc-teacher-bot") + 1] == "beginner")
check("B bc-games 4", fb[fb.index("--bc-games") + 1] == "4")
check("B bc-warmup 80", fb[fb.index("--bc-warmup") + 1] == "80")
check("B keep incomplete", "--bc-keep-incomplete" in fb)
check("B lambda piso 0.25", fb[fb.index("--bc-lambda-end") + 1] == "0.25")
check("B teacher macro 20", fb[fb.index("--bc-macro-ticks") + 1] == "20")
check("B teacher max-steps 2800", fb[fb.index("--bc-max-steps") + 1] == "2800")
cfg_b_start = dict(cfg)
cfg_b_start["phase"] = "B"
cfg_b_start["phase_started_iter"] = 20
fb20 = phase_flags("B", cfg_b_start)
check("B bc-start-iter = phase_started",
      fb20[fb20.index("--bc-start-iter") + 1] == "20")
cfg_b_start["b_bc_start_iter"] = 24
fb24 = phase_flags("B", cfg_b_start)
check("B pin b_bc_start_iter",
      fb24[fb24.index("--bc-start-iter") + 1] == "24")
fc = phase_flags("C", cfg)
check("C easy + reset-opt", fc[fc.index("--bot-type") + 1] == "easy" and "--reset-opt" in fc)
cfg["c_reset_opt_done"] = True
fc2 = phase_flags("C", cfg)
check("C relaunch sin reset-opt", "--reset-opt" not in fc2)

cmd = build_train_argv(base, "A", new_curriculum())
check("argv A last bot-type beginner",
      cmd[cmd.index("--bot-type") + 1] == "beginner"
      or cmd[[i for i, a in enumerate(cmd) if a == "--bot-type"][-1] + 1] == "beginner")
check("argv A last --bc-only", "--bc-only" in cmd)
check("argv A sin pfsp", "--pfsp" not in cmd)

rows = [
    {"iter": i, "bot_type": "beginner", "onboard_phase": "B",
     "outcomes": ["win", "win", "lose", "win"]}
    for i in range(1, 25)
]
check("wr20 0.75", abs(wr20(outcomes_from_rows(rows, "beginner")) - 0.75) < 1e-9)
check("streak beginner 0.50", wr20_streak(
    outcomes_from_rows(rows, "beginner"), 0.50, 10) >= 10)
rows_a = [
    {"iter": i, "bot_type": "beginner", "onboard_phase": "A", "outcomes": []}
    for i in range(1, 21)
]
check("no promove A antes de sft_iters",
      should_promote(new_curriculum(), rows_a[:5], last_iter=5) is None)
check("iter 400 sucio no salta A",
      should_promote(new_curriculum(),
                     [{"iter": 400, "bot_type": "easy", "outcomes": ["win"]}],
                     last_iter=400) is None)
cfg_a = new_curriculum()
check("sft_iters sin wr alumno no promove",
      should_promote(cfg_a, rows_a, last_iter=20) is None)
rows_a_win = [
    {"iter": i, "bot_type": "beginner", "onboard_phase": "A",
     "outcomes": ["win", "win", "win", "win"]}
    for i in range(1, 21)
]
check("promove A->B con sft + wr alumno",
      should_promote(cfg_a, rows_a_win, last_iter=20) == "B")

cfg_b = new_curriculum()
cfg_b["phase"] = "B"
cfg_b["phase_started_iter"] = 0
check("promove B->C con wr20 alto y min iters",
      should_promote(cfg_b, rows, last_iter=24) == "C")

easy_rows = [
    {"iter": 30 + i, "bot_type": "easy", "onboard_phase": "C",
     "outcomes": ["win", "win", "incomplete", "win"]}
    for i in range(25)
]
cfg_c = new_curriculum()
cfg_c["phase"] = "C"
cfg_c["phase_started_iter"] = 29
check("promove C->done",
      should_promote(cfg_c, easy_rows, last_iter=54) == "done")

mix = rows + easy_rows
check("wr20 filtra bot_type", wr20(outcomes_from_rows(mix, "easy")) > 0.6)
check("era_reset no entra a wr20",
      wr20(outcomes_from_rows(
          [{"era_reset": True, "wins": 0, "total": 0, "bot_type": "easy"}] + easy_rows,
          "easy")) == wr20(outcomes_from_rows(easy_rows, "easy")))

sent = era_reset_row("phase C", "easy")
check("sentinel total 0", sent["total"] == 0 and sent["era_reset"] is True)

# No promover en un 4/4 suelto
short = [{"iter": 1, "bot_type": "beginner", "onboard_phase": "B",
          "outcomes": ["win", "win", "win", "win"]}]
cfg_b2 = new_curriculum()
cfg_b2["phase"] = "B"
cfg_b2["phase_started_iter"] = 0
check("un 4/4 no promociona",
      should_promote(cfg_b2, short, last_iter=1) is None)

cmd_b = build_train_argv(base, "B", cfg_b_start)
check("argv B last teacher beginner",
      cmd_b[[i for i, a in enumerate(cmd_b) if a == "--bc-teacher-bot"][-1] + 1]
      == "beginner")
check("argv B sin pfsp", "--pfsp" not in cmd_b)
check("argv C no bc", "--bc" not in phase_flags("C", cfg))

# rewind: latest <- best@24, drop metrics > 24
import tempfile
td = Path(tempfile.mkdtemp())
(td / "best.pt").write_bytes(b"BEST24")
(td / "latest.pt").write_bytes(b"WIPE187")
(td / "best.json").write_text(json.dumps({"iter": 24}), encoding="utf-8")
(td / "metrics.jsonl").write_text(
    '{"note": "era A", "era_reset": true, "wins": 0, "total": 0}\n'
    '{"iter": 20, "onboard_phase": "A", "outcomes": []}\n'
    '{"iter": 24, "onboard_phase": "B", "outcomes": ["win"]}\n'
    '{"iter": 185, "onboard_phase": "B", "outcomes": ["lose"]}\n',
    encoding="utf-8")
(td / "economy_race.jsonl").write_text(
    '{"iter": 24, "result": "win"}\n'
    '{"iter": 185, "result": "lose"}\n',
    encoding="utf-8")
cfg_rw = new_curriculum()
cfg_rw["phase"] = "B"
cfg_rw["a_launched"] = True
cfg_rw["phase_started_iter"] = 20
cfg_out, info = rewind_onboard(td, 24, cfg=cfg_rw)
check("rewind latest = best", (td / "latest.pt").read_bytes() == b"BEST24")
check("rewind no toca best", (td / "best.pt").read_bytes() == b"BEST24")
kept_m = (td / "metrics.jsonl").read_text(encoding="utf-8")
check("rewind trunca metrics >24", '"iter": 185' not in kept_m and '"iter": 24' in kept_m)
check("rewind conserva era_reset", "era_reset" in kept_m)
check("rewind trunca race >24",
      '"iter": 185' not in (td / "economy_race.jsonl").read_text(encoding="utf-8"))
check("rewind pin bc start 24", cfg_out["b_bc_start_iter"] == 24)
check("rewind no mueve phase_started", cfg_out["phase_started_iter"] == 20)
check("rewind src best.pt", info["src"] == "best.pt")
cfg_a_rw = new_curriculum()
try:
    rewind_onboard(td, 24, cfg=cfg_a_rw)
    check("rewind 24 rechaza fase A", False)
except ValueError:
    check("rewind 24 rechaza fase A", True)
(td / "iter0020.pt").write_bytes(b"SFT20")
cfg_back = new_curriculum()
cfg_back["phase"] = "B"
cfg_back["a_launched"] = True
cfg_back["phase_started_iter"] = 20
out20, inf20 = rewind_onboard(td, 20, cfg=cfg_back)
check("rewind 20 vuelve a A", out20["phase"] == "A")
check("rewind 20 latest = iter0020",
      (td / "latest.pt").read_bytes() == b"SFT20")
check("rewind 20 src decade", inf20["src"] == "iter0020.pt")

print("\n" + ("TODOS LOS TESTS OK" if ok else "HAY FALLAS"))
sys.exit(0 if ok else 1)
