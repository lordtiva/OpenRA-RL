r"""run_v5_pipeline.py — Master orchestrator for OpenRA-RL v5 (Autonomous General & Creative Meta).

Stages:
  phase5_0_warmstart : Transfer v4 Champion weights (27 tensors) to v5 CommanderNet
  phase5_1_record    : Record dual-map DAgger dataset (singles.oramap + doubles.oramap) -> data/v5_bc
  phase5_1_train     : Train bc_v5 with label smoothing (eps=0.04) -> runs/v5/bc_v5
  phase5_2_ppo       : PPO training with Clipped Reverse KL & Discovery Exploiters -> runs/v5/ppo
  phase5_3_selfplay  : Hybrid Self-Play League (Mirror, PFSP, Bots, Anchor, Exploiters) -> runs/v5/selfplay
  phase5_4_eval      : Comprehensive evaluation tournament (100 H2H vs v4, bots, tech diversity)

Usage:
  .\.venv\Scripts\python.exe scripts/run_v5_pipeline.py --stage all
  .\.venv\Scripts\python.exe scripts/run_v5_pipeline.py --stage warmstart
  .\.venv\Scripts\python.exe scripts/run_v5_pipeline.py --stage eval
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from ra_rl.eval import DEFAULT_DAEMONS
from ra_rl.features import PRIV_DIM
from ra_rl.model import CommanderNet


def run_cmd(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")


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
    print(f"\n=======================================================")
    print(f"  H2H Tournament: v5 vs v4 Champion ({games_h2h} dual-map games)")
    print(f"=======================================================")
    h2h_jsonl = out_dir / "h2h_v5_vs_v4.jsonl"
    if h2h_jsonl.exists():
        h2h_jsonl.unlink()

    run_cmd([
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
    ])

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
        if bot_jsonl.exists():
            bot_jsonl.unlink()
        run_cmd([
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
        ])
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
        choices=["all", "warmstart", "record", "bc", "ppo", "selfplay", "eval"],
        default="all",
        help="pipeline stage to execute",
    )
    ap.add_argument("--daemons", default=DEFAULT_DAEMONS, help="comma-separated daemon addresses")
    ap.add_argument("--v4-ckpt", default="runs/v4/selfplay/checkpoint_best.pt", help="path to v4 champion checkpoint")
    ap.add_argument("--bc-games", type=int, default=1200, help="games to record for v5 BC DAgger dataset")
    ap.add_argument("--bc-epochs", type=int, default=25, help="training epochs for bc_v5")
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
    if args.stage in ("all", "warmstart"):
        if not v4_path.exists():
            raise SystemExit(f"v4 checkpoint not found: {v4_path}")
        warmstart_v4_to_v5(v4_path, warmstart_path)
        if args.stage == "warmstart":
            return

    # Stage 1: Record Dual-Map DAgger dataset
    bc_data_dir = Path("data/v5_bc")
    if args.stage in ("all", "record"):
        if not warmstart_path.exists():
            warmstart_v4_to_v5(v4_path, warmstart_path)
        bc_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[RECORD] Collecting {args.bc_games} dual-map DAgger games...")
        run_cmd([
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
        ])
        if args.stage == "record":
            return

    # Stage 2: Train BC with Label Smoothing (eps=0.04)
    if args.stage in ("all", "bc"):
        print(f"\n[BC] Training bc_v5 with label smoothing eps=0.04...")
        run_cmd([
            sys.executable, "-m", "ra_rl.bc",
            "--data", str(bc_data_dir),
            "--init", str(warmstart_path),
            "--label-smoothing", "0.04",
            "--epochs", str(args.bc_epochs),
            "--out", str(bc_dir),
        ])
        if args.stage == "bc":
            return

    # Stage 3: Self-Play League with PFSP & Discovery Exploiters
    if args.stage in ("all", "selfplay"):
        bc_model = bc_dir / "model.pt"
        init_model = bc_model if bc_model.exists() else warmstart_path
        print(f"\n[SELFPLAY] Launching v5 League with Discovery Exploiters...")
        run_cmd([
            sys.executable, "-m", "ra_rl.selfplay",
            "--init", str(init_model),
            "--ref", str(init_model),
            "--daemons", args.daemons,
            "--maps", "singles.oramap,doubles.oramap",
            "--iterations", str(args.selfplay_iters),
            "--games-per-iter", "48",
            "--recycle-every", "60",
            "--out", str(selfplay_dir),
        ])
        if args.stage == "selfplay":
            return

    # Stage 4: Evaluation Tournament & Verification
    if args.stage in ("all", "eval"):
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
