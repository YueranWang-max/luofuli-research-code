"""Resilient supervisor for the frozen R3 study.

The scientific driver remains unchanged.  This operational wrapper tolerates
sleep/resume, transient CUDA/console failures, and stale locks left by a dead
worker, then restarts from the driver's durable per-batch artifacts.
"""
from __future__ import annotations

import subprocess
import sys
import time
import json
import os
import re
import ctypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLL_SECONDS = 30
MAX_USED_MIB = 1024
REQUIRED_CLEAR_POLLS = 3
RETRY_SECONDS = 60
LOCK = ROOT / "artifacts" / ".gpu_lock"
EVENTS = ROOT / "artifacts" / "v3" / "supervisor_events.jsonl"


def used_mib() -> int | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return None
    try:
        return max(int(line.strip()) for line in result.stdout.splitlines() if line.strip())
    except ValueError:
        return None


def record(event: str, **details) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    row = {"event": event, "unix": time.time(), **details}
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def pid_is_alive(pid: int) -> bool:
    # Avoid localized tasklist output and its console encoding after resume.
    process = ctypes.windll.kernel32.OpenProcess(0x00100000, False, pid)
    if not process:
        return False
    try:
        code = ctypes.c_ulong()
        if not ctypes.windll.kernel32.GetExitCodeProcess(process, ctypes.byref(code)):
            return False
        return code.value == 259  # STILL_ACTIVE
    finally:
        ctypes.windll.kernel32.CloseHandle(process)


def clear_dead_worker_lock() -> None:
    if not LOCK.exists():
        return
    text = LOCK.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"pid=(\d+)", text)
    if match and pid_is_alive(int(match.group(1))):
        return
    archived = LOCK.with_name(f".gpu_lock.stale.{time.time_ns()}.txt")
    LOCK.replace(archived)
    record("stale_lock_archived", original=text, archive=str(archived))


def wait_for_gpu() -> None:
    clear = 0
    while clear < REQUIRED_CLEAR_POLLS:
        memory = used_mib()
        clear = clear + 1 if memory is not None and memory <= MAX_USED_MIB else 0
        print(
            f"GPU gate: used_mib={memory}, clear_polls={clear}/{REQUIRED_CLEAR_POLLS}",
            flush=True,
        )
        if clear < REQUIRED_CLEAR_POLLS:
            time.sleep(POLL_SECONDS)


def main() -> None:
    while True:
        clear_dead_worker_lock()
        wait_for_gpu()
        result = subprocess.run(
            [sys.executable, "-u", "-X", "utf8", str(ROOT / "paper" / "run_v3_study.py")],
            cwd=ROOT,
            check=False,
        )
        record("study_exit", returncode=result.returncode)
        if result.returncode == 0:
            return
        clear_dead_worker_lock()
        time.sleep(RETRY_SECONDS)


if __name__ == "__main__":
    main()
