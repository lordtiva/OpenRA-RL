# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.onboard import (
    apply_earned_promotions,
    build_train_argv,
    drought_blocked_until_min_iters,
    drought_since_iter,
    era_reset_row,
    mix_target_prob,
    new_curriculum,
    outcomes_from_rows,
    phase_flags,
    rewind_onboard,
    should_promote,
    should_resume,
    snapshot_phase_best,
    strip_flags,
    teacher_mode_for_phase,
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
check("default a_promote_wr20 0.25", abs(cfg["a_promote_wr20"] - 0.25) < 1e-9)
check("default done_wr20 0.50 vs hard", abs(cfg["done_wr20"] - 0.50) < 1e-9)
check("default c_promote 0.45", abs(cfg["c_promote_wr20"] - 0.45) < 1e-9)
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
check("A teacher-mode rush", fa[fa.index("--bc-teacher-mode") + 1] == "rush")
check("A win-cap 64000", fa[fa.index("--bc-win-cap") + 1] == "64000")
check("A ep-cap 40", fa[fa.index("--bc-win-ep-cap") + 1] == "40")
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
check("B bc-warmup 40", fb[fb.index("--bc-warmup") + 1] == "40")
check("B wins-only (no keep incomplete)", "--bc-keep-incomplete" not in fb)
check("B lambda piso 0.10", fb[fb.index("--bc-lambda-end") + 1] == "0.10")
check("B lr 2e-5", fb[fb.index("--lr") + 1] == "2.0e-5")
check("B adv-mode global", fb[fb.index("--adv-mode") + 1] == "global")
check("B no-amp", "--no-amp" in fb and "--amp-init-scale" not in fb)
check("B teacher macro 40", fb[fb.index("--bc-macro-ticks") + 1] == "40")
check("B teacher max-steps 1800", fb[fb.index("--bc-max-steps") + 1] == "1800")
check("B rush 8", fb[fb.index("--bc-rush") + 1] == "8")
check("B teacher-mode rush", fb[fb.index("--bc-teacher-mode") + 1] == "rush")
check("B ep-cap 40", fb[fb.index("--bc-win-ep-cap") + 1] == "40")
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
check("A lr 1e-4", fa[fa.index("--lr") + 1] == "1.0e-4")
check("A adv-mode episode", fa[fa.index("--adv-mode") + 1] == "episode")
check("C easy sin reset-opt", fc[fc.index("--bot-type") + 1] == "easy"
      and "--reset-opt" not in fc)
check("C lr 2e-5 (igual B)", fc[fc.index("--lr") + 1] == "2.0e-5")
check("C adv-mode global", fc[fc.index("--adv-mode") + 1] == "global")
check("C no-amp", "--no-amp" in fc and "--amp-init-scale" not in fc)
check("C mix-from beginner", fc[fc.index("--mix-from") + 1] == "beginner")
check("C mix-warmup 40", fc[fc.index("--mix-warmup") + 1] == "40")
check("C mix-start 0.25", fc[fc.index("--mix-start") + 1] == "0.25")
check("C sil + expand BC", "--sil" in fc and "--bc" in fc
      and "--bc-only" not in fc)
check("C teacher-mode expand", fc[fc.index("--bc-teacher-mode") + 1] == "expand")
check("C teacher-bot easy", fc[fc.index("--bc-teacher-bot") + 1] == "easy")
check("C ep-cap 40", fc[fc.index("--bc-win-ep-cap") + 1] == "40")
fd = phase_flags("D", cfg)
check("D bot medium", fd[fd.index("--bot-type") + 1] == "medium")
check("D mix-from easy", fd[fd.index("--mix-from") + 1] == "easy")
check("D expand BC", "--bc" in fd and fd[fd.index("--bc-teacher-mode") + 1] == "expand")
fe = phase_flags("E", cfg)
check("E bot hard", fe[fe.index("--bot-type") + 1] == "hard")
check("E mix-from medium", fe[fe.index("--mix-from") + 1] == "medium")
check("E expand BC", "--bc" in fe and fe[fe.index("--bc-teacher-mode") + 1] == "expand")
check("teacher_mode A/B rush CDE expand",
      teacher_mode_for_phase("B") == "rush"
      and teacher_mode_for_phase("C") == "expand")
cfg["c_reset_opt_done"] = True
fc2 = phase_flags("C", cfg)
check("C nunca reset-opt", "--reset-opt" not in fc2)
cfg_c_start = dict(cfg)
cfg_c_start["phase_started_iter"] = 132
fc132 = phase_flags("C", cfg_c_start)
check("C mix-start-iter = phase_started+1",
      fc132[fc132.index("--mix-start-iter") + 1] == "133")

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

# A easy gate: wr20>=0.25 once (no streak / no promote_wr20 0.50 bar)
rows_a_025 = [
    {"iter": i, "bot_type": "beginner", "onboard_phase": "A",
     "outcomes": ["lose", "lose", "lose", "win"]}
    for i in range(1, 21)
]
check("A wr20=0.25 promove sin streak",
      abs(wr20(outcomes_from_rows(rows_a_025, "beginner", "A")) - 0.25) < 1e-9
      and should_promote(cfg_a, rows_a_025, last_iter=20) == "B")
# 1 win in last 20 eps via last iter only — build 19 lose-iters + one mixed
rows_a_024 = (
    [{"iter": i, "bot_type": "beginner", "onboard_phase": "A",
      "outcomes": ["lose", "lose", "lose", "lose"]} for i in range(1, 20)]
    + [{"iter": 20, "bot_type": "beginner", "onboard_phase": "A",
        "outcomes": ["lose", "lose", "lose", "win"]}]
)
check("A wr20<0.25 no promove",
      wr20(outcomes_from_rows(rows_a_024, "beginner", "A")) < 0.25
      and should_promote(cfg_a, rows_a_024, last_iter=20) is None)
# Even if promote_wr20 were still 0.50 with streak, A uses a_promote only
cfg_a_hard = new_curriculum()
cfg_a_hard["promote_wr20"] = 0.50
cfg_a_hard["streak"] = 10
check("A ignora promote_wr20/streak (solo a_promote)",
      should_promote(cfg_a_hard, rows_a_025, last_iter=20) == "B")

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
check("promove C->D",
      should_promote(cfg_c, easy_rows, last_iter=54) == "D")
med_rows = [
    {"iter": 60 + i, "bot_type": "medium", "onboard_phase": "D",
     "outcomes": ["win", "win", "incomplete", "win"]}
    for i in range(25)
]
cfg_d = new_curriculum()
cfg_d["phase"] = "D"
cfg_d["phase_started_iter"] = 59
check("promove D->E",
      should_promote(cfg_d, med_rows, last_iter=84) == "E")
hard_rows = [
    {"iter": 90 + i, "bot_type": "hard", "onboard_phase": "E",
     "outcomes": ["win", "win", "win", "incomplete"]}
    for i in range(25)
]
cfg_e = new_curriculum()
cfg_e["phase"] = "E"
cfg_e["phase_started_iter"] = 89
check("promove E->done",
      should_promote(cfg_e, hard_rows, last_iter=114) == "done")
hard_low = [
    {"iter": 90 + i, "bot_type": "hard", "onboard_phase": "E",
     "outcomes": ["win", "lose", "lose", "lose"]}
    for i in range(25)
]
check("E wr20 0.25 no done",
      should_promote(cfg_e, hard_low, last_iter=114) is None)

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
check("argv B no-amp", "--no-amp" in cmd_b)
check("argv C last teacher-mode expand",
      phase_flags("C", cfg)[phase_flags("C", cfg).index("--bc-teacher-mode") + 1]
      == "expand")

print("=== mix ramp / drought gate ===")
check("mix start_iter p=start",
      abs(mix_target_prob(133, 133, 40, 0.25) - 0.25) < 1e-9)
check("mix mid",
      abs(mix_target_prob(153, 133, 40, 0.25) - 0.625) < 1e-9)
check("mix end",
      abs(mix_target_prob(173, 133, 40, 0.25) - 1.0) < 1e-9)
check("mix after warmup stays 1",
      abs(mix_target_prob(200, 133, 40, 0.25) - 1.0) < 1e-9)
check("mix warmup 0 = 100% target",
      abs(mix_target_prob(1, 1, 0, 0.25) - 1.0) < 1e-9)
cfg_d = new_curriculum()
cfg_d["phase"] = "C"
cfg_d["phase_started_iter"] = 132
check("drought since = phase start",
      drought_since_iter(cfg_d, 0) == 132)
check("drought since max(restore, start)",
      drought_since_iter(cfg_d, 140) == 140)
c_early = [{"iter": 132 + i, "bot_type": "easy", "onboard_phase": "C",
            "outcomes": ["incomplete"] * 4} for i in range(1, 6)]
check("C drought blocked < min_iters",
      drought_blocked_until_min_iters(cfg_d, c_early) is True)
c_late = [{"iter": 132 + i, "bot_type": "easy", "onboard_phase": "C",
           "outcomes": ["incomplete"] * 4} for i in range(1, 22)]
check("C drought allowed after min_iters",
      drought_blocked_until_min_iters(cfg_d, c_late) is False)
check("B drought never blocked by this gate",
      drought_blocked_until_min_iters({"phase": "B"}, c_early) is False)
mix_row = {
    "iter": 140, "bot_type": "easy", "onboard_phase": "C",
    "outcomes": ["win", "win", "win", "incomplete"],
    "opponent_bots": ["beginner", "easy", "beginner", "easy"],
}
check("mix wr20 cuenta solo easy",
      abs(wr20(outcomes_from_rows([mix_row], "easy", "C")) - 0.5) < 1e-9)
check("mix beginner wins no inflan C",
      abs(wr20(outcomes_from_rows([mix_row], "beginner", "C")) - 1.0) < 1e-9)

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
check("rewind no reinicia lambda a keep_iter",
      cfg_out["b_bc_start_iter"] == 20)
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

# C → B rewind (undo a bad promote). Restore best + latest from decade.
td2 = Path(tempfile.mkdtemp())
(td2 / "best.pt").write_bytes(b"EASY0")
(td2 / "latest.pt").write_bytes(b"EASYLATE")
(td2 / "iter0130.pt").write_bytes(b"B130")
(td2 / "best.json").write_text(
    json.dumps({"iter": 153, "bot_type": "easy", "iter_winrate": 0.0}),
    encoding="utf-8")
(td2 / "best_B.pt").write_bytes(b"BBEST")
(td2 / "metrics.jsonl").write_text(
    '{"iter": 130, "onboard_phase": "B", "bot_type": "beginner", '
    '"outcomes": ["win","win","win","incomplete"], "iter_winrate": 0.75}\n'
    '{"iter": 153, "onboard_phase": "C", "bot_type": "easy"}\n',
    encoding="utf-8")
(td2 / "economy_race.jsonl").write_text(
    '{"iter": 130}\n{"iter": 153}\n', encoding="utf-8")
(td2 / "elite.pt").write_bytes(b"ELITE")
cfg_c_rw = new_curriculum()
cfg_c_rw["phase"] = "C"
cfg_c_rw["a_launched"] = True
cfg_c_rw["phase_started_iter"] = 132
cfg_c_rw["b_bc_start_iter"] = 36
out_c, inf_c = rewind_onboard(td2, 130, cfg=cfg_c_rw)
check("rewind C->B phase B", out_c["phase"] == "B")
check("rewind C latest = decade", (td2 / "latest.pt").read_bytes() == b"B130")
check("rewind C restaura best", (td2 / "best.pt").read_bytes() == b"B130")
check("rewind C borra elite", not (td2 / "elite.pt").exists())
check("rewind C lambda origin B", out_c["b_bc_start_iter"] == 36)
check("rewind C phase_started = keep (min_iters)",
      int(out_c["phase_started_iter"]) == 130)
check("rewind C trunca metrics 153",
      '"iter": 153' not in (td2 / "metrics.jsonl").read_text(encoding="utf-8"))
best_meta = json.loads((td2 / "best.json").read_text(encoding="utf-8"))
check("rewind C best.json iter 130",
      best_meta.get("iter") == 130
      and best_meta.get("reason") == "onboard_rewind")
check("snapshot_phase_best copia",
      snapshot_phase_best(td2, "B") is not None
      and (td2 / "best_B.pt").exists())

# Rewind to sft_iters with A wr20 already >= threshold → land in B
td3 = Path(tempfile.mkdtemp())
(td3 / "iter0020.pt").write_bytes(b"SFT20W")
(td3 / "latest.pt").write_bytes(b"LATE")
(td3 / "best.pt").write_bytes(b"BEST")
(td3 / "best.json").write_text(json.dumps({"iter": 40}), encoding="utf-8")
# 5 iters x 4 games = 20 outcomes, 8 wins → wr20=0.40 >= 0.25
a_rows = []
for i in range(1, 21):
    # pad early iters with losses so last-20 window is controlled
    if i < 16:
        outs = ["lose", "lose", "lose", "lose"]
    else:
        outs = ["win", "win", "lose", "incomplete"]  # 2/4
    a_rows.append(
        json.dumps({"iter": i, "bot_type": "beginner", "onboard_phase": "A",
                    "outcomes": outs}))
# last 20 games = iters 16..20 = 5*4: each 2 wins → 10/20 = 0.50
(td3 / "metrics.jsonl").write_text(
    "\n".join(a_rows) + "\n"
    + json.dumps({"iter": 40, "onboard_phase": "B", "bot_type": "beginner",
                  "outcomes": ["lose"]}) + "\n",
    encoding="utf-8")
(td3 / "economy_race.jsonl").write_text(
    '{"iter": 20}\n{"iter": 40}\n', encoding="utf-8")
cfg_a2b = new_curriculum()
cfg_a2b["phase"] = "B"
cfg_a2b["a_launched"] = True
cfg_a2b["phase_started_iter"] = 20
cfg_a2b["sft_iters"] = 20
cfg_a2b["a_promote_wr20"] = 0.25
out_a2b, inf_a2b = rewind_onboard(td3, 20, cfg=cfg_a2b)
check("rewind-20 con wr20>=a_promote landa en B", out_a2b["phase"] == "B")
check("rewind-20 earned [B]", inf_a2b.get("earned_promotions") == ["B"])
check("rewind-20 phase_started = keep",
      int(out_a2b["phase_started_iter"]) == 20)
check("rewind-20 trunca B posterior",
      '"iter": 40' not in (td3 / "metrics.jsonl").read_text(encoding="utf-8"))

# Same metrics, empty wr → stays A (no invent wins)
td3b = Path(tempfile.mkdtemp())
(td3b / "iter0020.pt").write_bytes(b"SFT20")
(td3b / "latest.pt").write_bytes(b"L")
(td3b / "best.pt").write_bytes(b"B")
(td3b / "best.json").write_text(json.dumps({"iter": 20}), encoding="utf-8")
empty_a = "\n".join(
    json.dumps({"iter": i, "bot_type": "beginner", "onboard_phase": "A",
                "outcomes": []}) for i in range(1, 21)) + "\n"
(td3b / "metrics.jsonl").write_text(empty_a, encoding="utf-8")
cfg_stay = new_curriculum()
cfg_stay["phase"] = "B"
cfg_stay["a_launched"] = True
out_stay, inf_stay = rewind_onboard(td3b, 20, cfg=cfg_stay)
check("rewind-20 sin outcomes sigue A", out_stay["phase"] == "A")
check("rewind-20 sin outcomes no earned",
      inf_stay.get("earned_promotions") == [])

# B→C: kept B history already has streak+wr20+min_iters → promote to C
# (also covers resume-style apply_earned_promotions)
b_lines = []
for i in range(21, 51):  # 30 B iters > min_iters 20
    b_lines.append(json.dumps({
        "iter": i, "bot_type": "beginner", "onboard_phase": "B",
        "outcomes": ["win", "win", "win", "incomplete"],  # wr high
    }))
rows_b = [json.loads(s) for s in b_lines]
cfg_b2c = new_curriculum()
cfg_b2c["phase"] = "B"
cfg_b2c["phase_started_iter"] = 50  # pinned like post-rewind; history still counts
cfg_b2c["promote_wr20"] = 0.50
cfg_b2c["streak"] = 10
cfg_b2c["min_iters"] = 20
earned_b2c = apply_earned_promotions(cfg_b2c, rows_b, last_iter=50)
check("apply_earned B->C con streak historico", earned_b2c == ["C"])
check("apply_earned deja phase C", cfg_b2c["phase"] == "C")

# Rewind C→B when B streak already met on kept rows → re-promote to C
td4 = Path(tempfile.mkdtemp())
(td4 / "iter0050.pt").write_bytes(b"B50")
(td4 / "latest.pt").write_bytes(b"CLATE")
(td4 / "best.pt").write_bytes(b"CBEST")
(td4 / "best.json").write_text(json.dumps({"iter": 60}), encoding="utf-8")
(td4 / "metrics.jsonl").write_text(
    "\n".join(b_lines) + "\n"
    + json.dumps({"iter": 60, "onboard_phase": "C", "bot_type": "easy",
                  "outcomes": ["lose", "lose", "lose", "lose"]}) + "\n",
    encoding="utf-8")
(td4 / "economy_race.jsonl").write_text(
    '{"iter": 50}\n{"iter": 60}\n', encoding="utf-8")
cfg_c2c = new_curriculum()
cfg_c2c["phase"] = "C"
cfg_c2c["a_launched"] = True
cfg_c2c["phase_started_iter"] = 51
cfg_c2c["b_bc_start_iter"] = 21
cfg_c2c["promote_wr20"] = 0.50
cfg_c2c["streak"] = 10
cfg_c2c["min_iters"] = 20
out_c2c, inf_c2c = rewind_onboard(td4, 50, cfg=cfg_c2c)
check("rewind C con B-streak met landa en C", out_c2c["phase"] == "C")
check("rewind C earned includes C", "C" in (inf_c2c.get("earned_promotions") or []))


print("=== collapse A vs B ===")
from rl import auto_train as at

check("A never collapse even if flag on",
      at.collapse_active("A", True) is False)
check("B collapse follows flag on",
      at.collapse_active("B", True) is True)
check("B collapse follows flag off",
      at.collapse_active("B", False) is False)
check("C collapse follows flag on",
      at.collapse_active("C", True) is True)
check("C collapse follows flag off",
      at.collapse_active("C", False) is False)
check("no onboard follows flag on",
      at.collapse_active(None, True) is True)


print("=== collect-only wiring ===")

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

check("would_pass_bc_replay True when replay+B without collect-only",
      at.would_pass_bc_replay(
          replay_tapes=True,
          onboard={"phase": "B"},
          collect_only=False) is True)
check("would_pass_bc_replay True when replay+C expand",
      at.would_pass_bc_replay(
          replay_tapes=True,
          onboard={"phase": "C"},
          collect_only=False) is True)

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

    # Phase B resume: reuse tapes + --bc-replay (same as A offline path)
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
    check("resume phase B sets _replay_tapes",
          at._replay_tapes is True)
    check("resume phase B would_pass True",
          at.would_pass_bc_replay() is True)

    at._replay_tapes = False
    at._collect_only = False
    check("arm_replay B with v4 tapes",
          at.arm_replay_for_phase("B") is True)
    check("arm_replay B armed _replay_tapes", at._replay_tapes is True)
    at._replay_tapes = False
    check("arm_replay C with rush tapes False",
          at.arm_replay_for_phase("C") is False)
    check("arm_replay C did not arm", at._replay_tapes is False)

    at._onboard = {
        "phase": "A", "sft_iters": 20, "promote_wr20": 0.5,
        "done_wr20": 0.45, "streak": 10, "min_iters": 20,
        "bc_games": 4, "a_eval_games": 4, "a_rush": 8,
        "a_launched": True, "c_reset_opt_done": False,
        "phase_started_iter": 0,
    }
    at._replay_tapes = False
    at._collect_only = False
    with mock.patch.object(at.ob, "should_promote", return_value="B"), \
            mock.patch.object(at, "last_metrics_rows", return_value=[]), \
            mock.patch.object(at.ob, "save_curriculum"), \
            mock.patch.object(at.ob, "append_era_reset"):
        nxt_ab = at.try_promote(20)
    check("try_promote A->B", nxt_ab == "B")
    check("try_promote A->B arms replay", at._replay_tapes is True)

    # Phase C resume with RUSH tapes: schema mismatch, re-collect.
    (td / "curriculum.json").write_text(
        json.dumps({
            "phase": "C", "sft_iters": 20, "promote_wr20": 0.5,
            "done_wr20": 0.50, "streak": 10, "min_iters": 20,
            "bc_games": 4, "a_eval_games": 4, "a_rush": 8,
            "a_launched": True, "c_reset_opt_done": False,
            "phase_started_iter": 40,
        }), encoding="utf-8")
    at._replay_tapes = False
    at._collect_only = False
    at._onboard = None
    args_c2 = at.parse_auto_args(["--onboard"])
    with mock.patch.object(at.ob, "save_curriculum"):
        at._init_onboard(args_c2)
    check("resume phase C + rush tapes leaves _replay_tapes False",
          at._replay_tapes is False)
    check("resume phase C + rush tapes would_pass False",
          at.would_pass_bc_replay() is False)

    # Phase C resume with EXPAND tapes: replay.
    (td / "teacher_wins" / "manifest.json").write_text(
        json.dumps({
            "schema": "eco_and_combat_expand_v2",
            "episodes": [{"file": "ep_0000.pt"} for _ in range(10)],
        }), encoding="utf-8")
    at._replay_tapes = False
    at._collect_only = False
    at._onboard = None
    args_c3 = at.parse_auto_args(["--onboard"])
    with mock.patch.object(at.ob, "save_curriculum"):
        at._init_onboard(args_c3)
    check("resume phase C + expand tapes sets _replay_tapes",
          at._replay_tapes is True)
    check("resume phase C + expand tapes would_pass True",
          at.would_pass_bc_replay() is True)
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
