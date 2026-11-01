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
import json
import subprocess
import sys
from pathlib import Path

from ra_rl.eval import DEFAULT_DAEMONS


def run_cmd(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")


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
) -> float:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl.exists():
        out_jsonl.unlink()
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
    run_cmd(cmd)
    lines = [json.loads(x) for x in out_jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
    summary = lines[-1]["summary"]
    wr = float(summary["win_rate"])
    errs = int(summary["outcomes"].get("error", 0))
    label = f"vs {opponent_policy}" if opponent_policy else f"vs {bot}"
    print(f"[GATE] policy={policy} ({player_faction}) {label}: WR={wr*100:.1f}% (min {min_wr*100:.1f}%), errors={errs}", flush=True)
    if wr < min_wr:
        raise SystemExit(f"Gate failed for {policy} {label}: {wr:.3f} < {min_wr:.3f}")
    return wr


def run_phase4_1(args) -> None:
    # 1. Bilateral Allies+Soviet demonstrations (75% vs hard/brutal/turtle + 25% RL-vs-RL mirror)
    run_cmd([
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
        "--out", "data/scripted_v4",
    ])

    # 2. Train initial bilateral multi-squad BC policy (bc_s0)
    run_cmd([
        sys.executable, "-m", "ra_rl.bc",
        "--data", "data/scripted_v4",
        "--out", "runs/v4/bc_s0",
        "--epochs", "25",
    ])

    # 3. Record 1 round of bilateral DAgger (75% bots + 25% mirror)
    run_cmd([
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
        "--out", "data/dagger1_v4",
    ])

    # 4. Train final supervised v4 anchor model (bc_v4)
    run_cmd([
        sys.executable, "-m", "ra_rl.bc",
        "--data", "data/scripted_v4,data/dagger1_v4",
        "--out", "runs/v4/bc_v4",
        "--epochs", "25",
    ])

    # 5. Verify bilateral supervised gates on selection seeds (20000+)
    gates_dir = Path("runs/v4/bc_v4/gates")
    eval_gate(args.daemons, "net:runs/v4/bc_v4/model.pt", "hard", args.gate_games, 0.50, gates_dir / "hard.jsonl", player_faction=args.player_faction)
    eval_gate(args.daemons, "net:runs/v4/bc_v4/model.pt", "brutal", args.gate_games, 0.40, gates_dir / "brutal.jsonl", player_faction=args.player_faction)
    eval_gate(args.daemons, "net:runs/v4/bc_v4/model.pt", "turtle", args.gate_games, 0.50, gates_dir / "turtle.jsonl", player_faction=args.player_faction)


def run_phase4_2(args) -> None:
    bc_ckpt = Path("runs/v4/bc_v4/model.pt")
    if not bc_ckpt.exists():
        raise SystemExit(f"Missing Fase 4.1 anchor checkpoint: {bc_ckpt}")

    run_cmd([
        sys.executable, "-m", "ra_rl.league",
        "--init", str(bc_ckpt),
        "--ref", str(bc_ckpt),
        "--reset-critic",
        "--player-faction", args.player_faction,
        "--out", "runs/v4/league_warmstart",
        "--iterations", str(args.warmstart_iters),
        "--games-per-iter", str(args.games_per_iter),
        "--critic-warmup-iters", "6",
        "--kl-coef", "0.04",
        "--daemons", args.daemons,
    ])


def run_phase4_3(args) -> None:
    bc_ckpt = Path("runs/v4/bc_v4/model.pt")
    warm_best = Path("runs/v4/league_warmstart/checkpoint_best.pt")
    warm_latest = Path("runs/v4/league_warmstart/checkpoint_latest.pt")
    init_ckpt = warm_best if warm_best.exists() else (warm_latest if warm_latest.exists() else bc_ckpt)
    if not init_ckpt.exists():
        raise SystemExit(f"Missing starting checkpoint for Fase 4.3: {init_ckpt}")

    cmd = [
        sys.executable, "-m", "ra_rl.selfplay",
        "--init", str(init_ckpt),
        "--ref", str(bc_ckpt if bc_ckpt.exists() else init_ckpt),
        "--player-faction", args.player_faction,
        "--maps", args.maps,
        "--out", "runs/v4/selfplay",
        "--iterations", str(args.selfplay_iters),
        "--games-per-iter", str(args.games_per_iter),
        "--snapshot-every", str(args.snapshot_every),
        "--daemons", args.daemons,
    ]
    if init_ckpt == bc_ckpt:
        cmd.append("--reset-critic")
    run_cmd(cmd)


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
            )

    bc_ckpt = Path("runs/v4/bc_v4/model.pt")
    if bc_ckpt.exists():
        eval_gate(
            args.daemons, spec, "", 100, 0.0,
            eval_dir / "head2head_vs_bc_v4.jsonl",
            seed_base=10000, player_faction="Random",
            opponent_policy=f"net:{bc_ckpt}",
        )
    eval_gate(
        args.daemons, spec, "", 100, 0.0,
        eval_dir / "head2head_vs_scripted_v4.jsonl",
        seed_base=10000, player_faction="Random",
        opponent_policy="scripted",
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
