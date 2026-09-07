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
check("A rush 8", fa[fa.index("--bc-rush") + 1] == "8")
check("A win-cap 64000", fa[fa.index("--bc-win-cap") + 1] == "64000")
check("A prefer-ticks 20000", fa[fa.index("--bc-win-prefer-ticks") + 1] == "20000")
check("A iters largo (wr gate, no sft_iters)",
      fa[fa.index("--iters") + 1] == "10000")
check("A macro 40", fa[fa.index("--macro-ticks") + 1] == "40")
check("A max-steps 1800", fa[fa.index("--max-steps") + 1] == "1800")
check("A no sil", "--sil" not in fa)
check("A qsa-topk dense/off", fa[fa.index("--qsa-topk") + 1] == "0")
check("A xf-topk dense/off", fa[fa.index("--xf-topk") + 1] == "0")
# Last flag wins over TRAIN_ARGS sparse defaults when building argv.
cmd_a_dense = build_train_argv(
    base + ["--qsa-topk", "8", "--xf-topk", "16"], "A", new_curriculum())
_qsa_idxs = [i for i, a in enumerate(cmd_a_dense) if a == "--qsa-topk"]
_xf_idxs = [i for i, a in enumerate(cmd_a_dense) if a == "--xf-topk"]
check("argv A last qsa-topk 0",
      cmd_a_dense[_qsa_idxs[-1] + 1] == "0")
check("argv A last xf-topk 0",
      cmd_a_dense[_xf_idxs[-1] + 1] == "0")
fb = phase_flags("B", cfg)
check("B sil beginner", "--sil" in fb and fb[fb.index("--bot-type") + 1] == "beginner")
check("B bc mezclado no bc-only", "--bc" in fb and "--bc-only" not in fb)
check("B teacher beginner", fb[fb.index("--bc-teacher-bot") + 1] == "beginner")
check("B bc-games 2", fb[fb.index("--bc-games") + 1] == "2")
check("B bc-warmup 80", fb[fb.index("--bc-warmup") + 1] == "80")
check("B wins-only (no keep incomplete)", "--bc-keep-incomplete" not in fb)
check("B lambda piso 0.25", fb[fb.index("--bc-lambda-end") + 1] == "0.25")
check("B teacher macro 40", fb[fb.index("--bc-macro-ticks") + 1] == "40")
check("B teacher max-steps 1800", fb[fb.index("--bc-max-steps") + 1] == "1800")
check("B rush 8", fb[fb.index("--bc-rush") + 1] == "8")
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


print("=== collect-only wiring ===")
from rl import auto_train as at

pa = at.parse_auto_args([
    "--onboard", "--onboard-collect-only", "--onboard-collect-target", "40",
])
check("parse collect-only flag", pa.onboard_collect_only is True)
check("parse collect target 40", int(pa.onboard_collect_target) == 40)
check("parse still onboard", pa.onboard is True)

extras = at.collect_only_train_extras(40)
check("extras has --bc-collect-only", "--bc-collect-only" in extras)
check("extras has target 40",
      extras[extras.index("--bc-collect-target") + 1] == "40")
check("extras NO --bc-replay", "--bc-replay" not in extras)

check("would_pass_bc_replay False when collect-only",
      at.would_pass_bc_replay(
          replay_tapes=True,
          onboard={"phase": "A"},
          collect_only=True) is False)
check("would_pass_bc_replay True when replay+A without collect-only",
      at.would_pass_bc_replay(
          replay_tapes=True,
          onboard={"phase": "A"},
          collect_only=False) is True)
check("would_pass_bc_replay False when no tapes",
      at.would_pass_bc_replay(
          replay_tapes=False,
          onboard={"phase": "A"},
          collect_only=False) is False)

check("would_pass_bc_replay False when phase B even with tapes",
      at.would_pass_bc_replay(
          replay_tapes=True,
          onboard={"phase": "B"},
          collect_only=False) is False)

print("=== resume replay wiring ===")
import json
import tempfile
from unittest import mock

td = Path(tempfile.mkdtemp())
(td / "teacher_wins").mkdir()
man = {
    "schema": "eco_and_combat_mental_v4",
    "episodes": [{"file": "ep_0000.pt"} for _ in range(40)],
}
(td / "teacher_wins" / "manifest.json").write_text(
    json.dumps(man), encoding="utf-8")
(td / "curriculum.json").write_text(
    json.dumps({
        "phase": "A", "sft_iters": 20, "promote_wr20": 0.5,
        "done_wr20": 0.45, "streak": 10, "min_iters": 20,
        "bc_games": 4, "a_eval_games": 4, "a_rush": 8,
        "a_launched": True, "c_reset_opt_done": False,
        "phase_started_iter": 0,
    }), encoding="utf-8")

orig_ckpt = at.CKPT_DIR
orig_cur = at.CURRICULUM
orig_met = at.METRICS
orig_rel = at.CKPT_DIR_REL
try:
    at.CKPT_DIR = td
    at.CURRICULUM = td / "curriculum.json"
    at.METRICS = td / "metrics.jsonl"
    args = at.parse_auto_args(["--onboard", "--no-collapse"])
    at._replay_tapes = False
    at._collect_only = False
    at._onboard = None
    with mock.patch.object(at.ob, "save_curriculum"):
        at._init_onboard(args)
    check("resume phase A sets _replay_tapes", at._replay_tapes is True)
    check("resume would_pass_bc_replay True",
          at.would_pass_bc_replay() is True)

    # --onboard-collect must NOT set replay
    at._replay_tapes = False
    at._collect_only = False
    at._onboard = None
    args_c = at.parse_auto_args(["--onboard", "--onboard-collect"])
    with mock.patch.object(at.ob, "save_curriculum"):
        at._init_onboard(args_c)
    check("resume+collect keeps _replay_tapes False",
          at._replay_tapes is False)
    check("resume+collect would_pass False",
          at.would_pass_bc_replay() is False)

    # Phase B resume: do not force bc-replay
    (td / "curriculum.json").write_text(
        json.dumps({
            "phase": "B", "sft_iters": 20, "promote_wr20": 0.5,
            "done_wr20": 0.45, "streak": 10, "min_iters": 20,
            "bc_games": 4, "a_eval_games": 4, "a_rush": 8,
            "a_launched": True, "c_reset_opt_done": False,
            "phase_started_iter": 20,
        }), encoding="utf-8")
    at._replay_tapes = False
    at._collect_only = False
    at._onboard = None
    args_b = at.parse_auto_args(["--onboard"])
    with mock.patch.object(at.ob, "save_curriculum"):
        at._init_onboard(args_b)
    check("resume phase B leaves _replay_tapes False",
          at._replay_tapes is False)
    check("resume phase B would_pass False",
          at.would_pass_bc_replay() is False)
finally:
    at.CKPT_DIR = orig_ckpt
    at.CURRICULUM = orig_cur
    at.METRICS = orig_met
    at.CKPT_DIR_REL = orig_rel


# strip keeps phase A clean if collect flags leaked into base
base_co = base + ["--bc-collect-only", "--bc-collect-target", "40"]
stripped_co = strip_flags(base_co)
check("strip saca --bc-collect-only", "--bc-collect-only" not in stripped_co)
check("strip saca --bc-collect-target", "--bc-collect-target" not in stripped_co)


print("\n" + ("TODOS LOS TESTS OK" if ok else "HAY FALLAS"))
sys.exit(0 if ok else 1)
