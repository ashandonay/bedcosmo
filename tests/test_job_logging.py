"""Tests for the universal jobs log (scripts/job_logging.sh) and per-run job log locations."""

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HELPER = REPO / "scripts" / "job_logging.sh"
ENTRY = re.compile(r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\] (\w+)\s+(.*)$")


def _bash(script, scratch, **env_overrides):
    env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
    env.update(SCRATCH=str(scratch), BED_JOB_SUMMARY="train num_visits/bb run=abc", **env_overrides)
    return subprocess.run(
        ["bash", "-c", f'source "{HELPER}"\n{script}'], env=env, capture_output=True, text=True
    )


def _entries(scratch):
    """Parse jobs.log into [(event, summary, [detail lines])]."""
    entries = []
    for line in (scratch / "bedcosmo" / "jobs.log").read_text().splitlines():
        m = ENTRY.match(line)
        if m:
            entries.append((m.group(1), m.group(2), []))
        else:
            assert line.startswith("    "), line
            entries[-1][2].append(line.strip())
    return entries


def test_append_writes_entry_with_details(tmp_path):
    _bash(
        'jobs_log_append QUEUED "train x run=1 job=7" "note: try bins16" "log:  /a/b.log"', tmp_path
    )
    _bash('jobs_log_append QUEUED "eval x run=1 job=8"', tmp_path)
    assert _entries(tmp_path) == [
        ("QUEUED", "train x run=1 job=7", ["note: try bins16", "log:  /a/b.log"]),
        ("QUEUED", "eval x run=1 job=8", []),
    ]


@pytest.mark.parametrize(
    "body, status, code",
    [
        ("true", "COMPLETED", 0),
        ("exit 3", "FAILED", 3),
        ('JOB_SKIPPED="train failed"; exit 0', "SKIPPED", 0),
    ],
)
def test_track_records_start_and_end_state(tmp_path, body, status, code):
    result = _bash(f'jobs_log_track 42 "log:  /x.log"\n{body}', tmp_path)
    assert result.returncode == code
    (start, start_summary, start_details), (end, end_summary, end_details) = _entries(tmp_path)
    assert (start, start_summary, start_details) == (
        "STARTED",
        "train num_visits/bb run=abc job=42",
        ["log:  /x.log"],
    )
    assert end == status
    assert end_summary == f"train num_visits/bb run=abc job=42 exit={code} elapsed=0s"
    assert end_details == (["reason: train failed"] if status == "SKIPPED" else [])


def test_track_names_node_under_slurm(tmp_path):
    _bash("jobs_log_track 42", tmp_path, SLURM_JOB_ID="42")
    (_, start_summary, start_details), _ = _entries(tmp_path)
    assert start_summary == f"train num_visits/bb run=abc job=42 node={socket.gethostname()}"
    assert start_details == []


@pytest.mark.parametrize("seconds, expected", [(45, "45s"), (754, "12m"), (15480, "4h18m")])
def test_duration_format(tmp_path, seconds, expected):
    result = _bash(f"JOB_START=$(( $(date +%s) - {seconds} )); _jobs_log_duration", tmp_path)
    assert result.stdout.strip() == expected


def test_track_records_stopped_on_sigterm(tmp_path):
    # SLURM sends SIGTERM to the batch script and its step; the step exits and the
    # script's EXIT trap records STOPPED.
    proc = subprocess.Popen(
        ["bash", "-c", f'source "{HELPER}"\njobs_log_track 9\nsleep 30 & wait $!'],
        env={**os.environ, "SCRATCH": str(tmp_path), "BED_JOB_SUMMARY": "grid x out=y"},
    )
    log = tmp_path / "bedcosmo" / "jobs.log"
    for _ in range(100):
        if log.exists():
            break
        time.sleep(0.05)
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)
    assert [e[0] for e in _entries(tmp_path)] == ["STARTED", "STOPPED"]


def test_mlflow_run_dir(tmp_path):
    run_dir = tmp_path / "bedcosmo" / "num_visits" / "mlruns" / "5" / "abc123"
    run_dir.mkdir(parents=True)
    found = _bash("mlflow_run_dir num_visits abc123", tmp_path)
    assert found.returncode == 0
    assert found.stdout.strip() == str(run_dir)
    missing = _bash("mlflow_run_dir num_visits nope", tmp_path)
    assert missing.returncode != 0
    assert "MLflow run nope not found" in missing.stderr


def test_create_run_for_restart_skips_snapshot(tmp_path):
    # Restart pre-creates its run only so the job log location is known at submission;
    # the job copies config from the source run, so nothing is snapshotted here.
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "create_run.py"),
            "--cosmo-exp",
            "num_tracers",
            "--restart-id",
            "src123",
            "--restart-step",
            "100",
            "--mlflow-exp",
            "debug",
        ],
        env={**os.environ, "SCRATCH": str(tmp_path)},
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    run_path = Path(re.search(r"^RUN_PATH=(.*)$", result.stdout, re.M).group(1))
    assert run_path.name == result.stdout.strip().splitlines()[-1]
    assert list((run_path / "artifacts").iterdir()) == []


def test_grid_calc_rejects_out_dir_with_run_id(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bedcosmo.grid_calc",
            "num_visits",
            "--run-id",
            "abc",
            "--out-dir",
            str(tmp_path),
        ],
        env={**os.environ, "SCRATCH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--out-dir is for standalone runs" in result.stderr
