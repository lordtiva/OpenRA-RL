#!/usr/bin/env python3
"""Archive the live ckpt dir into rl/ckpts/<name>/ and reset era metrics.

Call AFTER Ctrl+C on auto_train. Copies metrics/race/live/best/latest/iter*.pt
into the run folder, moves decade snapshots out of the root (latest.pt and
best.pt stay), then truncates metrics.jsonl + economy_race.jsonl so the next
regime hydrates wins/total at 0 (see train.py era counters).

  .\\.venv\\Scripts\\python.exe rl/archive_run.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "rl" / "ckpts"
DEFAULT_NAME = "Run 13 (a_short reassault sil-bomb 900-923)"

KEEP_ROOT = {"best.pt", "best.json", "latest.pt"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT)
    ap.add_argument(
        "--restore-best", action="store_true",
        help="Tras archivar, copia best.pt -> latest.pt (el latest del run "
             "archivado se queda en la carpeta; resume usa el best).")
    args = ap.parse_args()
    src: Path = args.ckpt_dir
    dest = src / args.name
    if dest.exists():
        print(f"FAIL: ya existe {dest}", file=sys.stderr)
        return 1
    dest.mkdir(parents=True)

    copied = []
    for name in (
        "metrics.jsonl",
        "economy_race.jsonl",
        "live_games.jsonl",
        "best.json",
        "best.pt",
        "latest.pt",
    ):
        p = src / name
        if p.exists() and p.is_file():
            shutil.copy2(p, dest / name)
            copied.append(name)

    moved = []
    for p in sorted(src.glob("iter*.pt")):
        shutil.copy2(p, dest / p.name)
        if p.name not in KEEP_ROOT:
            p.unlink()
            moved.append(p.name)
    for p in sorted(src.glob("aborted_*.pt")):
        shutil.copy2(p, dest / p.name)
        p.unlink()
        moved.append(p.name)

    # New era: do not hydrate beginner wins into easy wr.
    note = (
        '{"note": "era reset after archive", '
        f'"archive": "{args.name}"}}\n'
    )
    (src / "metrics.jsonl").write_text(note, encoding="utf-8")
    race = src / "economy_race.jsonl"
    if race.exists():
        race.write_text("", encoding="utf-8")
    live = src / "live_games.jsonl"
    if live.exists():
        live.write_text("", encoding="utf-8")

    print(f"OK {dest}")
    print(f"  copied {len(copied)} logs/ckpts: {copied}")
    print(f"  moved {len(moved)} snapshots out of root")
    print("  metrics.jsonl + race + live_games reset (era nueva)")
    print("  latest.pt / best.pt siguen en la raiz para resume")
    if args.restore_best:
        best = src / "best.pt"
        latest = src / "latest.pt"
        if not best.exists():
            print("  restore-best SKIP: no hay best.pt", file=sys.stderr)
        else:
            shutil.copy2(best, latest)
            print("  restore: best.pt -> latest.pt (pesos del best, no el latest archivado)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
