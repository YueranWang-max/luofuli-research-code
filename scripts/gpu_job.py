"""GPU job wrapper: lock + idle-wait + health-guard for one-GPU machines.

Usage:
  python -X utf8 scripts/gpu_job.py --name e0_256 -- <full command...>

Behavior:
  1. waits until nvidia-smi reports < idle_mb VRAM used (default 2000) for
     `settle_s` consecutive seconds (default 180) — file-event triggers are
     NOT used because they race with the next matrix job's startup
  2. takes artifacts/.gpu_lock atomically (skip-lock-overridden by --force)
  3. runs the command; releases the lock on exit
  4. after `--check-delay` seconds (default 600) prints a one-line health
     summary (VRAM, child alive) so a delayed post-launch check is easy
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "artifacts" / ".gpu_lock"


def vram_used_mb() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return 1 << 30  # unknown: treat as busy


def acquire_lock(name: str, force: bool) -> bool:
    if LOCK.exists():
        if force:
            LOCK.unlink()
        else:
            holder = LOCK.read_text(encoding="utf-8").strip()
            print(f"[gpu_job] lock held by: {holder}", flush=True)
            return False
    tmp = LOCK.with_suffix(".tmp")
    tmp.write_text(f"{name} pid={os.getpid()} t={time.strftime('%H:%M:%S')}", encoding="utf-8")
    try:
        os.replace(tmp, LOCK)  # atomic on same volume
    except OSError:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--idle-mb", type=int, default=2000)
    ap.add_argument("--settle-s", type=int, default=180)
    ap.add_argument("--check-delay", type=int, default=600)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = [c for c in args.cmd if c != "--"]

    settle = 0
    while settle < args.settle_s:
        mb = vram_used_mb()
        if mb < args.idle_mb:
            settle += 30
        else:
            settle = 0
            print(f"[gpu_job] busy ({mb} MB), waiting...", flush=True)
        time.sleep(30)
    if not acquire_lock(args.name, args.force):
        sys.exit(3)
    print(f"[gpu_job] lock acquired for {args.name}, launching", flush=True)
    try:
        proc = subprocess.Popen(cmd, cwd=str(ROOT))
        t0 = time.time()
        checked = False
        while proc.poll() is None:
            time.sleep(10)
            if not checked and time.time() - t0 > args.check_delay:
                checked = True
                print(
                    f"[gpu_job] health@{args.check_delay}s: vram={vram_used_mb()}MB "
                    f"child_alive={proc.poll() is None}",
                    flush=True,
                )
        rc = proc.returncode
    finally:
        LOCK.unlink(missing_ok=True)
    print(f"[gpu_job] done rc={rc}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
