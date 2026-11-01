r"""End-to-end resumable orchestrator for Commander v4 + Hybrid Self-Play (Eje B).

    # Run the full pipeline (Fase 4.1 BC/DAgger -> Fase 4.2 Warm-Start League -> Fase 4.3 Hybrid Self-Play -> Final Eval):
    .\.venv\Scripts\python.exe scripts/run_v4_pipeline.py --stage all

    # Or run individual stages:
    .\.venv\Scripts\python.exe scripts/run_v4_pipeline.py --stage phase4_1
    .\.venv\Scripts\python.exe scripts/run_v4_pipeline.py --stage phase4_2
    .\.venv\Scripts\python.exe scripts/run_v4_pipeline.py --stage phase4_3
    .\.venv\Scripts\python.exe scripts/run_v4_pipeline.py --stage eval
"""

import argparse
from datetime import datetime, timedelta
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ra_rl.eval import DEFAULT_DAEMONS


def format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 0:
        s = 0
    hrs = s // 3600
    rem = s % 3600
    mins = rem // 60
    secs = rem % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def format_clock(dt: datetime) -> str:
    now = datetime.now()
    if dt.date() == now.date():
        return dt.strftime("%H:%M:%S")
    return dt.strftime("%d/%m %H:%M:%S")


def format_bar(current: int, total: int, width: int = 15) -> str:
    if total <= 0:
        return "-" * width
    pct = min(1.0, max(0.0, current / total))
    filled = int(round(pct * width))
    return "=" * filled + "-" * (width - filled)


class StageProgressMonitor:
    def __init__(self, title: str, total: int, unit: str = "partidas", initial: int = 0):
        self.title = title
        self.total = max(1, total)
        self.unit = unit
        self.initial = initial
        self.current = initial
        self.t0 = time.perf_counter()
        self.last_render_time = 0.0
        self.last_len = 0
        self.is_tty = sys.stdout.isatty()
        self.extra_status = ""
        self.last_reported_pct = -1

    def update(self, current: int | None = None, delta: int = 0, extra: str = "") -> None:
        if current is not None:
            self.current = current
        else:
            self.current += delta
        if extra:
            self.extra_status = extra
        self.render()

    def render(self, force: bool = False) -> None:
        now_perf = time.perf_counter()
        if not force and (now_perf - self.last_render_time < 0.1):
            return
        self.last_render_time = now_perf

        elapsed = now_perf - self.t0
        completed_this_run = max(0, self.current - self.initial)
        pct = min(100.0, max(0.0, (self.current / self.total) * 100.0))

        if completed_this_run > 0 and self.current < self.total:
            rate = completed_this_run / elapsed
            rem_items = self.total - self.current
            eta_s = rem_items / rate if rate > 0 else 0
            eta_str = format_duration(eta_s)
            finish_dt = datetime.now() + timedelta(seconds=eta_s)
            clock_str = format_clock(finish_dt)
        elif self.current >= self.total:
            eta_str = "00:00"
            clock_str = format_clock(datetime.now())
        else:
            eta_str = "--:--"
            clock_str = "--:--:--"

        elapsed_str = format_duration(elapsed)
        bar = format_bar(self.current, self.total, width=15)
        extra_info = f" | {self.extra_status}" if self.extra_status else ""

        line = (
            f"[{self.title}] [{bar}] {self.current}/{self.total} {self.unit} ({pct:5.1f}%) | "
            f"Transcurrido: {elapsed_str} | Restante: {eta_str} | Fin Est.: {clock_str}{extra_info}"
        )

        if self.is_tty:
            pad = max(0, self.last_len - len(line))
            sys.stdout.write("\r" + line + (" " * pad))
            sys.stdout.flush()
            self.last_len = max(len(line), self.last_len)
        else:
            current_pct_step = int(pct // 10) * 10
            if force or current_pct_step > self.last_reported_pct:
                self.last_reported_pct = current_pct_step
                sys.stdout.write(line + "\n")
                sys.stdout.flush()

    def finish(self, summary: str = "") -> None:
        self.current = self.total
        self.render(force=True)
        elapsed = format_duration(time.perf_counter() - self.t0)
        clock_str = format_clock(datetime.now())
        extra = f" | {summary}" if summary else ""
        done_line = f"✔ [{self.title}] {self.total}/{self.total} {self.unit} (100.0%) en {elapsed} ({clock_str}){extra}"
        pad = max(0, self.last_len - len(done_line))
        if self.is_tty:
            sys.stdout.write("\r" + done_line + (" " * pad) + "\n")
        else:
            sys.stdout.write(done_line + "\n")
        sys.stdout.flush()


def run_cmd_monitored(
    cmd: list[str],
    title: str,
    total: int,
    unit: str,
    initial: int = 0,
    kind: str = "generic",  # "record" | "bc" | "league" | "selfplay" | "eval"
) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    monitor = StageProgressMonitor(title, total=total, unit=unit, initial=initial)
    monitor.render(force=True)

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    wins = 0
    games_seen = 0

    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue

            parsed = None
            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed = json.loads(line)
                except Exception:
                    pass

            if parsed is not None:
                if kind in ("record", "eval"):
                    if "seed" in parsed:
                        games_seen += 1
                        cur = initial + games_seen
                        res = parsed.get("result", parsed.get("outcome", ""))
                        if res in ("win", "win_early"):
                            wins += 1
                        wr_pct = (wins / games_seen) * 100.0
                        bot_name = parsed.get("bot", "")
                        extra = f"WR: {wr_pct:4.1f}%" + (f" ({bot_name})" if bot_name else "")
                        monitor.update(current=min(total, cur), extra=extra)
                    elif "summary" in parsed:
                        s = parsed["summary"]
                        wr = float(s.get("win_rate", 0.0))
                        monitor.extra_status = f"WR Final: {wr*100:.1f}%"
                elif kind == "bc":
                    if "epoch" in parsed:
                        ep = int(parsed["epoch"])
                        vp = parsed.get("val_prod_loss", 0.0)
                        va = parsed.get("val_army_loss", 0.0)
                        extra = f"val_prod: {vp:.4f}, val_army: {va:.4f}"
                        monitor.update(current=ep, extra=extra)
                elif kind == "league":
                    if "iteration" in parsed:
                        it = int(parsed["iteration"])
                        wr = float(parsed.get("win_rate", 0.0))
                        extra = f"Rollout WR: {wr*100:4.1f}%"
                        if "eval" in parsed and "harmonic_score" in parsed["eval"]:
                            h = float(parsed["eval"]["harmonic_score"])
                            extra += f" | H-Score: {h:.3f}"
                        monitor.update(current=it, extra=extra)
                elif kind == "selfplay":
                    if "iteration" in parsed:
                        it = int(parsed["iteration"])
                        elo = float(parsed.get("elo", 0.0))
                        wr = float(parsed.get("win_rate", 0.0))
                        extra = f"Elo: {elo:6.1f} | Rollout WR: {wr*100:4.1f}%"
                        if "eval" in parsed and "composite_score" in parsed["eval"]:
                            c = float(parsed["eval"]["composite_score"])
                            extra += f" | Composite: {c:.3f}"
                        monitor.update(current=it, extra=extra)
            else:
                # Non-json line from process (warning, error, status notice)
                low = line.lower()
                if any(k in low for k in ("error", "traceback", "exception", "failed", "warning", "[gate]")):
                    if monitor.is_tty:
                        sys.stdout.write("\r" + (" " * monitor.last_len) + "\r")
                    print(line, flush=True)
                    monitor.render(force=True)

        ret = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        if monitor.is_tty:
            sys.stdout.write("\n")
        raise

    if ret != 0:
        if monitor.is_tty:
            sys.stdout.write("\n")
        raise SystemExit(f"Command failed with exit code {ret}: {' '.join(cmd)}")

    monitor.finish(monitor.extra_status)


def eval_gate(
    daemons: str,
    policy: str,
    bot: str,
    games: int,
    min_wr: float,
    out_jsonl: Path,
    seed_base: int = 20000,
    player_faction: str = "Random",
    opponent_policy: str = "",
    title: str = "",
) -> float:
    label = f"vs {opponent_policy}" if opponent_policy else f"vs {bot}"
    display_title = title or f"Gate {policy} {label}"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl.exists():
        try:
            lines = [json.loads(x) for x in out_jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
            if lines and "summary" in lines[-1]:
                summary = lines[-1]["summary"]
                n_played = int(summary.get("games", 0))
                if n_played == games:
                    wr = float(summary["win_rate"])
                    errs = int(summary.get("outcomes", {}).get("error", 0))
                    if wr >= min_wr:
                        print(
                            f"[RESUME] Gate {policy} ({player_faction}) {label} already verified: "
                            f"WR={wr*100:.1f}% (min {min_wr*100:.1f}%), errors={errs}",
                            flush=True,
                        )
                        return wr
        except Exception:
            pass
        out_jsonl.unlink(missing_ok=True)

    cmd = [
        sys.executable, "-m", "ra_rl.eval",
        "--daemons", daemons,
        "--policy", policy,
        "--player-faction", player_faction,
        "--games", str(games),
        "--seed-base", str(seed_base),
        "--max-ticks", "54000",
        "--pre-recycle",
        "--out", str(out_jsonl),
    ]
    if opponent_policy:
        cmd.extend(["--opponent-policy", opponent_policy])
    else:
        cmd.extend(["--bot", bot])

    run_cmd_monitored(cmd, title=display_title, total=games, unit="partidas", initial=0, kind="eval")

    lines = [json.loads(x) for x in out_jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
    summary = lines[-1]["summary"]
    wr = float(summary["win_rate"])
    errs = int(summary["outcomes"].get("error", 0))
    print(f"[GATE RESULT] policy={policy} ({player_faction}) {label}: WR={wr*100:.1f}% (min {min_wr*100:.1f}%), errors={errs}", flush=True)
    if wr < min_wr:
        raise SystemExit(f"Gate failed for {policy} {label}: {wr:.3f} < {min_wr:.3f}")
    return wr


def count_npz_files(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    return len(list(dir_path.glob("g*.npz")))


def get_completed_iterations(metrics_path: Path) -> int:
    if not metrics_path.exists():
        return 0
    max_it = 0
    try:
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            max_it = max(max_it, int(rec.get("iteration", 0)))
    except Exception:
        pass
    return max_it


def run_phase4_1(args) -> None:
    bc_final = Path("runs/v4/bc_v4/model.pt")
    gates_dir = Path("runs/v4/bc_v4/gates")

    # If final model and all gate files exist and pass, skip Phase 4.1 entirely
    if bc_final.exists() and (gates_dir / "turtle.jsonl").exists():
        try:
            w_h = eval_gate(args.daemons, f"net:{bc_final}", "hard", args.gate_games, 0.50, gates_dir / "hard.jsonl", player_faction=args.player_faction, title="Gate 4.1 vs hard")
            w_b = eval_gate(args.daemons, f"net:{bc_final}", "brutal", args.gate_games, 0.40, gates_dir / "brutal.jsonl", player_faction=args.player_faction, title="Gate 4.1 vs brutal")
            w_t = eval_gate(args.daemons, f"net:{bc_final}", "turtle", args.gate_games, 0.50, gates_dir / "turtle.jsonl", player_faction=args.player_faction, title="Gate 4.1 vs turtle")
            print(
                f"[RESUME] Fase 4.1 already complete and gates verified (hard={w_h*100:.1f}%, brutal={w_b*100:.1f}%, turtle={w_t*100:.1f}%); skipping Phase 4.1.",
                flush=True,
            )
            return
        except SystemExit:
            pass

    # 1. Bilateral Allies+Soviet demonstrations (75% vs hard/brutal/turtle + 25% RL-vs-RL mirror)
    scripted_dir = Path("data/scripted_v4")
    n_scripted = count_npz_files(scripted_dir)
    if n_scripted >= args.bc_games:
        print(f"[RESUME] {scripted_dir} already complete ({n_scripted}/{args.bc_games} games); skipping recording.", flush=True)
    else:
        cmd_rec = [
            sys.executable, "-m", "ra_rl.record",
            "--teacher", "scripted",
            "--bots", "hard,brutal,turtle",
            "--player-faction", args.player_faction,
            "--selfplay-prob", "0.25",
            "--eps", "0.15",
            "--eps-army", "0.10",
            "--games", str(args.bc_games),
            "--seed-base", "1",
            "--daemons", args.daemons,
            "--recycle-every", "50",
            "--out", str(scripted_dir),
        ]
        run_cmd_monitored(cmd_rec, title="Fase 4.1 (1/5) Record Scripted", total=args.bc_games, unit="partidas", initial=n_scripted, kind="record")

    # 2. Train initial bilateral multi-squad BC policy (bc_s0)
    bc_s0 = Path("runs/v4/bc_s0/model.pt")
    if bc_s0.exists():
        print(f"[RESUME] {bc_s0} already exists; skipping bc_s0 training.", flush=True)
    else:
        cmd_bc0 = [
            sys.executable, "-m", "ra_rl.bc",
            "--data", "data/scripted_v4",
            "--out", "runs/v4/bc_s0",
            "--epochs", "25",
        ]
        run_cmd_monitored(cmd_bc0, title="Fase 4.1 (2/5) Train BC s0", total=25, unit="épocas", initial=0, kind="bc")

    # 3. Record 1 round of bilateral DAgger (75% bots + 25% mirror)
    dagger_dir = Path("data/dagger1_v4")
    n_dagger = count_npz_files(dagger_dir)
    if n_dagger >= args.dagger_games:
        print(f"[RESUME] {dagger_dir} already complete ({n_dagger}/{args.dagger_games} games); skipping DAgger recording.", flush=True)
    else:
        cmd_dag = [
            sys.executable, "-m", "ra_rl.record",
            "--teacher", "scripted",
            "--executor", "net:runs/v4/bc_s0/model.pt",
            "--bots", "hard,brutal,turtle",
            "--player-faction", args.player_faction,
            "--selfplay-prob", "0.25",
            "--games", str(args.dagger_games),
            "--seed-base", "30001",
            "--daemons", args.daemons,
            "--recycle-every", "50",
            "--out", str(dagger_dir),
        ]
        run_cmd_monitored(cmd_dag, title="Fase 4.1 (3/5) Record DAgger", total=args.dagger_games, unit="partidas", initial=n_dagger, kind="record")

    # 4. Train final supervised v4 anchor model (bc_v4)
    if bc_final.exists():
        print(f"[RESUME] {bc_final} already exists; skipping bc_v4 training.", flush=True)
    else:
        cmd_bc_v4 = [
            sys.executable, "-m", "ra_rl.bc",
            "--data", "data/scripted_v4,data/dagger1_v4",
            "--out", "runs/v4/bc_v4",
            "--epochs", "25",
        ]
        run_cmd_monitored(cmd_bc_v4, title="Fase 4.1 (4/5) Train BC v4 (Anchor)", total=25, unit="épocas", initial=0, kind="bc")

    # 5. Verify bilateral supervised gates on selection seeds (20000+)
    eval_gate(args.daemons, f"net:{bc_final}", "hard", args.gate_games, 0.50, gates_dir / "hard.jsonl", player_faction=args.player_faction, title="Gate 4.1 vs hard")
    eval_gate(args.daemons, f"net:{bc_final}", "brutal", args.gate_games, 0.40, gates_dir / "brutal.jsonl", player_faction=args.player_faction, title="Gate 4.1 vs brutal")
    eval_gate(args.daemons, f"net:{bc_final}", "turtle", args.gate_games, 0.50, gates_dir / "turtle.jsonl", player_faction=args.player_faction, title="Gate 4.1 vs turtle")


def run_phase4_2(args) -> None:
    bc_ckpt = Path("runs/v4/bc_v4/model.pt")
    if not bc_ckpt.exists():
        raise SystemExit(f"Missing Fase 4.1 anchor checkpoint: {bc_ckpt}")

    out_dir = Path("runs/v4/league_warmstart")
    completed_iters = get_completed_iterations(out_dir / "metrics.jsonl")
    if completed_iters >= args.warmstart_iters and (out_dir / "checkpoint_best.pt").exists():
        print(f"[RESUME] Fase 4.2 at {out_dir} already completed {completed_iters}/{args.warmstart_iters} iterations; skipping.", flush=True)
        return

    cmd = [
        sys.executable, "-m", "ra_rl.league",
        "--init", str(bc_ckpt),
        "--ref", str(bc_ckpt),
        "--player-faction", args.player_faction,
        "--out", str(out_dir),
        "--iterations", str(args.warmstart_iters),
        "--games-per-iter", str(args.games_per_iter),
        "--critic-warmup-iters", "6",
        "--kl-coef", "0.04",
        "--daemons", args.daemons,
    ]
    if completed_iters == 0 and not (out_dir / "checkpoint_latest.pt").exists():
        cmd.append("--reset-critic")

    run_cmd_monitored(
        cmd,
        title="Fase 4.2 Warm-Start League",
        total=args.warmstart_iters,
        unit="iteraciones",
        initial=completed_iters,
        kind="league",
    )


def run_phase4_3(args) -> None:
    bc_ckpt = Path("runs/v4/bc_v4/model.pt")
    warm_best = Path("runs/v4/league_warmstart/checkpoint_best.pt")
    warm_latest = Path("runs/v4/league_warmstart/checkpoint_latest.pt")
    out_dir = Path("runs/v4/selfplay")

    completed_iters = get_completed_iterations(out_dir / "metrics.jsonl")
    if completed_iters >= args.selfplay_iters and (out_dir / "checkpoint_best.pt").exists():
        print(f"[RESUME] Fase 4.3 at {out_dir} already completed {completed_iters}/{args.selfplay_iters} iterations; skipping.", flush=True)
        return

    latest_sp_ckpt = out_dir / "checkpoint_latest.pt"
    if latest_sp_ckpt.exists() and completed_iters > 0:
        init_ckpt = latest_sp_ckpt
    else:
        init_ckpt = warm_best if warm_best.exists() else (warm_latest if warm_latest.exists() else bc_ckpt)

    if not init_ckpt.exists():
        raise SystemExit(f"Missing starting checkpoint for Fase 4.3: {init_ckpt}")

    cmd = [
        sys.executable, "-m", "ra_rl.selfplay",
        "--init", str(init_ckpt),
        "--ref", str(bc_ckpt if bc_ckpt.exists() else init_ckpt),
        "--player-faction", args.player_faction,
        "--maps", args.maps,
        "--out", str(out_dir),
        "--iterations", str(args.selfplay_iters),
        "--games-per-iter", str(args.games_per_iter),
        "--snapshot-every", str(args.snapshot_every),
        "--daemons", args.daemons,
    ]
    if init_ckpt == bc_ckpt and completed_iters == 0 and not latest_sp_ckpt.exists():
        cmd.append("--reset-critic")

    run_cmd_monitored(
        cmd,
        title="Fase 4.3 Hybrid Self-Play",
        total=args.selfplay_iters,
        unit="iteraciones",
        initial=completed_iters,
        kind="selfplay",
    )


def run_final_eval(args) -> None:
    sp_best = Path("runs/v4/selfplay/checkpoint_best.pt")
    sp_latest = Path("runs/v4/selfplay/checkpoint_latest.pt")
    champ = sp_best if sp_best.exists() else sp_latest
    if not champ.exists():
        raise SystemExit(f"Missing self-play checkpoint: {champ}")

    eval_dir = Path("runs/v4/eval_test_10000")
    spec = f"net:{champ}"
    # Official 100-seed test benchmark (10000-10099): both bilateral Random and Allied-parity
    for fac_tag, fac in (("random", "Random"), ("allies", "RandomAllies")):
        for bot in ("hard", "brutal", "turtle"):
            eval_gate(
                args.daemons, spec, bot, 100, 0.0,
                eval_dir / f"{fac_tag}_{bot}.jsonl",
                seed_base=10000, player_faction=fac,
                title=f"Eval Final {fac_tag}_{bot}",
            )

    bc_ckpt = Path("runs/v4/bc_v4/model.pt")
    if bc_ckpt.exists():
        eval_gate(
            args.daemons, spec, "", 100, 0.0,
            eval_dir / "head2head_vs_bc_v4.jsonl",
            seed_base=10000, player_faction="Random",
            opponent_policy=f"net:{bc_ckpt}",
            title="Eval Final vs bc_v4",
        )
    eval_gate(
        args.daemons, spec, "", 100, 0.0,
        eval_dir / "head2head_vs_scripted_v4.jsonl",
        seed_base=10000, player_faction="Random",
        opponent_policy="scripted",
        title="Eval Final vs scripted_v4",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("all", "phase4_1", "phase4_2", "phase4_3", "eval"), default="all")
    ap.add_argument("--daemons", default=DEFAULT_DAEMONS)
    ap.add_argument("--player-faction", default="Random")
    ap.add_argument("--maps", default="singles.oramap")
    ap.add_argument("--bc-games", type=int, default=1800)
    ap.add_argument("--dagger-games", type=int, default=1200)
    ap.add_argument("--gate-games", type=int, default=60)
    ap.add_argument("--warmstart-iters", type=int, default=30)
    ap.add_argument("--selfplay-iters", type=int, default=80)
    ap.add_argument("--snapshot-every", type=int, default=8)
    ap.add_argument("--games-per-iter", type=int, default=48)
    args = ap.parse_args()

    if args.stage in ("all", "phase4_1"):
        run_phase4_1(args)
    if args.stage in ("all", "phase4_2"):
        run_phase4_2(args)
    if args.stage in ("all", "phase4_3"):
        run_phase4_3(args)
    if args.stage in ("all", "eval"):
        run_final_eval(args)


if __name__ == "__main__":
    main()
