"""Unit tests for v5 pipeline orchestrator and progress monitoring telemetry."""

from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import sys
import time

from scripts.run_v5_pipeline import (
    StageProgressMonitor,
    count_npz_files,
    format_bar,
    format_clock,
    format_duration,
    get_completed_iterations,
)


def test_format_duration():
    assert format_duration(0) == "00:00"
    assert format_duration(-10) == "00:00"
    assert format_duration(45) == "00:45"
    assert format_duration(65) == "01:05"
    assert format_duration(3600) == "01:00:00"
    assert format_duration(3665) == "01:01:05"
    assert format_duration(7325) == "02:02:05"


def test_format_clock():
    now = datetime.now()
    today_dt = now.replace(hour=14, minute=30, second=15)
    assert format_clock(today_dt) == "14:30:15"

    tomorrow_dt = now + timedelta(days=1)
    tomorrow_fmt = format_clock(tomorrow_dt)
    assert "/" in tomorrow_fmt and ":" in tomorrow_fmt


def test_format_bar():
    assert format_bar(0, 10, width=10) == "░░░░░░░░░░"
    assert format_bar(5, 10, width=10) == "█████░░░░░"
    assert format_bar(10, 10, width=10) == "██████████"
    assert format_bar(15, 10, width=10) == "██████████"
    assert format_bar(0, 0, width=10) == "░░░░░░░░░░"


def test_stage_progress_monitor_eta_calculation():
    # Simulate non-tty monitor
    monitor = StageProgressMonitor("Test Stage", total=100, unit="items", initial=0)
    monitor.is_tty = False

    # Mock time
    monitor.t0 = time.perf_counter() - 10.0  # 10s elapsed
    monitor.update(current=20, extra="loss: 0.1234")

    assert monitor.current == 20
    assert monitor.extra_status == "loss: 0.1234"

    # Finish monitor
    buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        monitor.finish("Completed successfully")
    finally:
        sys.stdout = old_stdout

    output = buf.getvalue()
    assert "✔ [Test Stage] 100/100 items (100.0%)" in output
    assert "Completed successfully" in output


def test_stage_progress_monitor_resume():
    # Started at initial=50 out of 100
    monitor = StageProgressMonitor("Resume Stage", total=100, unit="games", initial=50)
    monitor.is_tty = False

    # Simulate 5 items completed in 5 seconds (1 item/s)
    monitor.t0 = time.perf_counter() - 5.0
    monitor.update(current=55)

    assert monitor.current == 55


def test_count_npz_files(tmp_path: Path):
    d = tmp_path / "data"
    assert count_npz_files(d) == 0
    d.mkdir()
    (d / "g000001.npz").touch()
    (d / "g000002.npz").touch()
    (d / "other.txt").touch()
    assert count_npz_files(d) == 2


def test_get_completed_iterations(tmp_path: Path):
    m = tmp_path / "metrics.jsonl"
    assert get_completed_iterations(m) == 0

    lines = [
        json.dumps({"iteration": 1, "elo": 1200.0}),
        json.dumps({"iteration": 2, "elo": 1250.0}),
        json.dumps({"iteration": 5, "elo": 1350.0}),
    ]
    m.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert get_completed_iterations(m) == 5
