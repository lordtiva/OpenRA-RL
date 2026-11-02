r"""run_v5_pipeline.py — Master orchestrator for OpenRA-RL v5 (Autonomous General & Creative Meta).

Stages:
  warmstart / phase5_0 : Transfer v4 Champion weights (27 tensors) to v5 CommanderNet
  record / phase5_1    : Record dual-map DAgger dataset (singles.oramap + doubles.oramap) -> data/v5_bc
  bc / phase5_1        : Train bc_v5 with label smoothing (eps=0.04) -> runs/v5/bc_v5
  ppo / phase5_2       : Warm-start PPO training with Clipped Reverse KL -> runs/v5/ppo
  selfplay / phase5_3  : Hybrid Self-Play League (Mirror, PFSP, Bots, Anchor, Exploiters) -> runs/v5/selfplay
  eval / phase5_4      : Comprehensive evaluation tournament (100 H2H vs v4, bots, tech diversity)

Usage:
  .\.venv\Scripts\python.exe scripts/run_v5_pipeline.py --stage all
  .\.venv\Scripts\python.exe scripts/run_v5_pipeline.py --stage warmstart
  .\.venv\Scripts\python.exe scripts/run_v5_pipeline.py --stage record
  .\.venv\Scripts\python.exe scripts/run_v5_pipeline.py --stage bc
  .\.venv\Scripts\python.exe scripts/run_v5_pipeline.py --stage selfplay
  .\.venv\Scripts\python.exe scripts/run_v5_pipeline.py --stage eval
"""

import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from ra_rl.eval import DEFAULT_DAEMONS
from ra_rl.features import PRIV_DIM
from ra_rl.model import CommanderNet


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
        return "░" * width
    pct = min(1.0, max(0.0, current / total))
    filled = int(round(pct * width))
    return "█" * filled + "░" * (width - filled)


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
    kind: str = "generic",  # "record" | "bc" | "league" | "selfplay" | "ppo" | "eval"
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
                        wr_pct = (wins / games_seen) * 100.0 if games_seen > 0 else 0.0
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
                elif kind in ("league", "selfplay", "ppo"):
                    if "iteration" in parsed:
                        it = int(parsed["iteration"])
                        elo = float(parsed.get("elo", 0.0))
                        wr = float(parsed.get("win_rate", 0.0))
                        extra = f"Elo: {elo:6.1f} | Rollout WR: {wr*100:4.1f}%" if elo > 0 else f"Rollout WR: {wr*100:4.1f}%"
                        if "eval" in parsed and isinstance(parsed["eval"], dict):
                            if "composite_score" in parsed["eval"]:
                                c = float(parsed["eval"]["composite_score"])
                                extra += f" | Composite: {c:.3f}"
                            elif "harmonic_score" in parsed["eval"]:
                                h = float(parsed["eval"]["harmonic_score"])
                                extra += f" | H-Score: {h:.3f}"
                            elif "win_rate" in parsed["eval"]:
                                ewr = float(parsed["eval"]["win_rate"])
                                extra += f" | Eval WR: {ewr*100:4.1f}%"
                        elif "eval_win_rate" in parsed:
                            ewr = float(parsed["eval_win_rate"])
                            extra += f" | Eval WR: {ewr*100:4.1f}%"
                        monitor.update(current=it, extra=extra)
            else:
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


def get_completed_epochs(metrics_path: Path) -> int:
    if not metrics_path.exists():
        return 0
    max_ep = 0
    try:
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            max_ep = max(max_ep, int(rec.get("epoch", 0)))
    except Exception:
        pass
    return max_ep


def check_eval_complete(jsonl_path: Path, expected_games: int) -> dict | None:
    if not jsonl_path.exists():
        return None
    try:
        lines = [json.loads(x) for x in jsonl_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if lines and "summary" in lines[-1]:
            summary = lines[-1]["summary"]
            if int(summary.get("games", 0)) == expected_games:
                return summary
    except Exception:
        pass
    return None


def warmstart_v4_to_v5(v4_path: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[WARMSTART] Loading v4 champion from: {v4_path}")
    v4_ckpt = torch.load(v4_path, map_location="cpu", weights_only=False)
    state_v4 = v4_ckpt["state"]

    v5_net = CommanderNet(hidden=256, cand_hidden=128, priv_dim=PRIV_DIM)
    state_v5 = v5_net.state_dict()

    matched = 0
    for k, v in state_v4.items():
        if k in state_v5:
            if v.shape == state_v5[k].shape:
                state_v5[k].copy_(v)
                matched += 1
            elif k == "trunk.0.weight":
                # Copy first 304 feature dimensions, zero-init the new 305th feature (ally_building)
                state_v5[k][:, :v.shape[1]].copy_(v)
                state_v5[k][:, v.shape[1]:].zero_()
                matched += 1
                print(f"  [+] Transferred {k}: {v.shape} -> {state_v5[k].shape} (zero-padded new feature)")
            elif k == "critic.0.weight":
                state_v5[k][:, :v.shape[1]].copy_(v)
                state_v5[k][:, v.shape[1]:].zero_()
                matched += 1
                print(f"  [+] Transferred {k}: {v.shape} -> {state_v5[k].shape} (zero-padded new feature)")

    v5_net.load_state_dict(state_v5)
    v5_net.save(str(out_path), source=str(v4_path), transferred_tensors=matched)
    print(f"[WARMSTART] Successfully transferred {matched} weight tensors to: {out_path}")
    return out_path


def eval_tournament(
    daemons: str,
    v5_policy: str,
    v4_policy: str,
    out_dir: Path,
    games_h2h: int = 100,
    games_bot: int = 40,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": time.time(), "v5_policy": v5_policy, "v4_policy": v4_policy}

    # 1. H2H Tournament vs v4 Champion across dual maps
    h2h_jsonl = out_dir / "h2h_v5_vs_v4.jsonl"
    existing_h2h = check_eval_complete(h2h_jsonl, games_h2h)
    if existing_h2h is not None:
        h2h_wr = float(existing_h2h["win_rate"])
        print(f"[RESUME] H2H Tournament already completed ({games_h2h} partidas): WR={h2h_wr * 100:.1f}%")
        report["h2h_win_rate"] = h2h_wr
        report["h2h_summary"] = existing_h2h
    else:
        if h2h_jsonl.exists():
            h2h_jsonl.unlink()

        cmd_h2h = [
            sys.executable, "-m", "ra_rl.eval",
            "--daemons", daemons,
            "--policy", v5_policy,
            "--peer-policy", v4_policy,
            "--games", str(games_h2h),
            "--map", "singles.oramap,doubles.oramap",
            "--seed-base", "50000",
            "--max-ticks", "54000",
            "--pre-recycle",
            "--out", str(h2h_jsonl),
        ]
        run_cmd_monitored(
            cmd_h2h,
            title=f"Eval H2H v5 vs v4 ({games_h2h} partidas)",
            total=games_h2h,
            unit="partidas",
            kind="eval",
        )

        h2h_lines = [json.loads(x) for x in h2h_jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
        h2h_summary = h2h_lines[-1]["summary"]
        h2h_wr = float(h2h_summary["win_rate"])
        report["h2h_win_rate"] = h2h_wr
        report["h2h_summary"] = h2h_summary
        print(f"\n>>> H2H Result: v5 Win Rate vs v4 Champion = {h2h_wr * 100:.1f}%")

    # 2. Evaluation against C# bots on dual map pool
    bot_results = {}
    for bot in ("brutal", "hard", "turtle"):
        bot_jsonl = out_dir / f"eval_vs_{bot}.jsonl"
        existing_bot = check_eval_complete(bot_jsonl, games_bot)
        if existing_bot is not None:
            b_wr = float(existing_bot["win_rate"])
            print(f"[RESUME] Eval vs {bot} already completed ({games_bot} partidas): WR={b_wr * 100:.1f}%")
            bot_results[bot] = {"win_rate": b_wr, "outcomes": existing_bot["outcomes"]}
        else:
            if bot_jsonl.exists():
                bot_jsonl.unlink()
            cmd_bot = [
                sys.executable, "-m", "ra_rl.eval",
                "--daemons", daemons,
                "--policy", v5_policy,
                "--bot", bot,
                "--games", str(games_bot),
                "--map", "singles.oramap,doubles.oramap",
                "--seed-base", "55000",
                "--max-ticks", "54000",
                "--pre-recycle",
                "--out", str(bot_jsonl),
            ]
            run_cmd_monitored(
                cmd_bot,
                title=f"Eval v5 vs {bot} ({games_bot} partidas)",
                total=games_bot,
                unit="partidas",
                kind="eval",
            )
            lines = [json.loads(x) for x in bot_jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
            b_summary = lines[-1]["summary"]
            b_wr = float(b_summary["win_rate"])
            bot_results[bot] = {"win_rate": b_wr, "outcomes": b_summary["outcomes"]}
            print(f">>> v5 vs {bot}: Win Rate = {b_wr * 100:.1f}%")

    report["bots"] = bot_results

    # Summary report file
    summary_path = out_dir / "v5_eval_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[REPORT] Tournament summary saved to: {summary_path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--stage",
        choices=[
            "all",
            "warmstart", "phase5_0",
            "record", "phase5_1_record",
            "bc", "phase5_1_train", "phase5_1",
            "ppo", "phase5_2",
            "selfplay", "phase5_3",
            "eval", "phase5_4",
        ],
        default="all",
        help="pipeline stage to execute",
    )
    ap.add_argument("--daemons", default=DEFAULT_DAEMONS, help="comma-separated daemon addresses")
    ap.add_argument("--v4-ckpt", default="runs/v4/selfplay/checkpoint_best.pt", help="path to v4 champion checkpoint")
    ap.add_argument("--bc-games", type=int, default=1200, help="games to record for v5 BC DAgger dataset")
    ap.add_argument("--bc-epochs", type=int, default=25, help="training epochs for bc_v5")
    ap.add_argument("--ppo-iters", type=int, default=20, help="iterations of single-agent PPO warm-start")
    ap.add_argument("--selfplay-iters", type=int, default=80, help="iterations of self-play league")
    ap.add_argument("--out-root", default="runs/v5", help="root directory for v5 artifacts")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    v4_path = Path(args.v4_ckpt)
    warmstart_path = out_root / "warmstart" / "model.pt"
    bc_dir = out_root / "bc_v5"
    selfplay_dir = out_root / "selfplay"

    # Stage 0: Warmstart
    if args.stage in ("all", "warmstart", "phase5_0"):
        if warmstart_path.exists():
            print(f"[RESUME] Warmstart model at {warmstart_path} already exists; skipping transfer.", flush=True)
        else:
            if not v4_path.exists():
                raise SystemExit(f"v4 checkpoint not found: {v4_path}")
            warmstart_v4_to_v5(v4_path, warmstart_path)
        if args.stage in ("warmstart", "phase5_0"):
            return

    # Stage 1: Record Dual-Map DAgger dataset
    bc_data_dir = Path("data/v5_bc")
    if args.stage in ("all", "record", "phase5_1", "phase5_1_record"):
        if not warmstart_path.exists():
            warmstart_v4_to_v5(v4_path, warmstart_path)
        bc_data_dir.mkdir(parents=True, exist_ok=True)
        completed_games = count_npz_files(bc_data_dir)
        if completed_games >= args.bc_games:
            print(f"[RESUME] Dataset at {bc_data_dir} already has {completed_games}/{args.bc_games} games; skipping recording.", flush=True)
        else:
            print(f"\n[RECORD] Collecting {args.bc_games} dual-map DAgger games ({completed_games} already present)...")
            cmd_record = [
                sys.executable, "-m", "ra_rl.record",
                "--teacher", "scripted",
                "--executor", f"net:{warmstart_path}",
                "--bots", "hard,brutal,turtle",
                "--map", "singles.oramap,doubles.oramap",
                "--player-faction", "Random",
                "--selfplay-prob", "0.25",
                "--eps", "0.12",
                "--eps-army", "0.08",
                "--games", str(args.bc_games),
                "--seed-base", "1000",
                "--daemons", args.daemons,
                "--recycle-every", "50",
                "--out", str(bc_data_dir),
            ]
            run_cmd_monitored(
                cmd_record,
                title="Fase 5.1 Record DAgger",
                total=args.bc_games,
                unit="partidas",
                initial=completed_games,
                kind="record",
            )
        if args.stage in ("record", "phase5_1_record"):
            return

    # Stage 2: Train BC with Label Smoothing (eps=0.04)
    if args.stage in ("all", "bc", "phase5_1", "phase5_1_train"):
        bc_model = bc_dir / "model.pt"
        completed_epochs = get_completed_epochs(bc_dir / "metrics.jsonl")
        if bc_model.exists() and completed_epochs >= args.bc_epochs:
            print(f"[RESUME] BC model at {bc_model} already completed {completed_epochs}/{args.bc_epochs} epochs; skipping BC training.", flush=True)
        else:
            print(f"\n[BC] Training bc_v5 with label smoothing eps=0.04...")
            cmd_bc = [
                sys.executable, "-m", "ra_rl.bc",
                "--data", str(bc_data_dir),
                "--init", str(warmstart_path),
                "--label-smoothing", "0.04",
                "--epochs", str(args.bc_epochs),
                "--out", str(bc_dir),
            ]
            run_cmd_monitored(
                cmd_bc,
                title="Fase 5.1 BC Training",
                total=args.bc_epochs,
                unit="épocas",
                kind="bc",
            )
        if args.stage in ("bc", "phase5_1", "phase5_1_train"):
            return

    # Stage 2.5: Warm-Start PPO (Optional if stage is ppo or phase5_2)
    if args.stage in ("ppo", "phase5_2"):
        ppo_dir = out_root / "ppo"
        bc_model = bc_dir / "model.pt"
        init_model = bc_model if bc_model.exists() else warmstart_path
        completed_iters = get_completed_iterations(ppo_dir / "metrics.jsonl")
        if completed_iters >= args.ppo_iters and (ppo_dir / "checkpoint_best.pt").exists():
            print(f"[RESUME] PPO at {ppo_dir} already completed {completed_iters}/{args.ppo_iters} iters; skipping.", flush=True)
        else:
            print(f"\n[PPO] Launching Warm-Start PPO ({args.ppo_iters} iterations)...")
            cmd_ppo = [
                sys.executable, "-m", "ra_rl.ppo",
                "--init", str(init_model),
                "--ref", str(init_model),
                "--daemons", args.daemons,
                "--map", "singles.oramap",
                "--iterations", str(args.ppo_iters),
                "--games-per-iter", "48",
                "--recycle-every", "60",
                "--out", str(ppo_dir),
            ]
            run_cmd_monitored(
                cmd_ppo,
                title="Fase 5.2 Warm-Start PPO",
                total=args.ppo_iters,
                unit="iteraciones",
                initial=completed_iters,
                kind="ppo",
            )
        return

    # Stage 3: Self-Play League with PFSP & Discovery Exploiters
    if args.stage in ("all", "selfplay", "phase5_3"):
        bc_model = bc_dir / "model.pt"
        init_model = bc_model if bc_model.exists() else warmstart_path
        completed_iters = get_completed_iterations(selfplay_dir / "metrics.jsonl")
        latest_sp_ckpt = selfplay_dir / "checkpoint_latest.pt"
        if completed_iters >= args.selfplay_iters and (selfplay_dir / "checkpoint_best.pt").exists():
            print(f"[RESUME] Selfplay at {selfplay_dir} already completed {completed_iters}/{args.selfplay_iters} iters; skipping.", flush=True)
        else:
            cur_init = latest_sp_ckpt if (latest_sp_ckpt.exists() and completed_iters > 0) else init_model
            print(f"\n[SELFPLAY] Launching v5 League with Discovery Exploiters ({args.selfplay_iters} iterations)...")
            if completed_iters > 0:
                print(f"[RESUME] Resuming from iteration {completed_iters + 1}/{args.selfplay_iters} with init={cur_init}")
            cmd_sp = [
                sys.executable, "-m", "ra_rl.selfplay",
                "--init", str(cur_init),
                "--ref", str(init_model),
                "--daemons", args.daemons,
                "--maps", "singles.oramap,doubles.oramap",
                "--iterations", str(args.selfplay_iters),
                "--games-per-iter", "48",
                "--recycle-every", "60",
                "--out", str(selfplay_dir),
            ]
            run_cmd_monitored(
                cmd_sp,
                title="Fase 5.3 Self-Play League",
                total=args.selfplay_iters,
                unit="iteraciones",
                initial=completed_iters,
                kind="selfplay",
            )
        if args.stage in ("selfplay", "phase5_3"):
            return

    # Stage 4: Evaluation Tournament & Verification
    if args.stage in ("all", "eval", "phase5_4"):
        v5_best = selfplay_dir / "checkpoint_best.pt"
        if not v5_best.exists():
            v5_best = bc_dir / "model.pt" if (bc_dir / "model.pt").exists() else warmstart_path
        eval_tournament(
            daemons=args.daemons,
            v5_policy=f"net:{v5_best}",
            v4_policy=f"net:{v4_path}",
            out_dir=out_root / "tournament",
            games_h2h=100,
            games_bot=40,
        )


if __name__ == "__main__":
    main()
