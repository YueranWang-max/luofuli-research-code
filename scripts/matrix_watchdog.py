"""Matrix watchdog: survives machine sleep, self-heals the v2.4 pipeline.

Every 5 minutes:
  - if the matrix is incomplete AND no matrix/training process is alive for
    two consecutive checks -> relaunch run_matrix_v2.py (resumes finished jobs)
  - if the matrix is complete AND the E0@256 baseline eval is missing ->
    run it (GPU is free at that point; guarded by an idle check anyway)

All actions are logged to artifacts/logs/watchdog.log (append).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
LOGS = ART / "logs"
STATUS = LOGS / "matrix_v2_mathhard_status.json"
E0_256 = ART / "bench_eval_e0_math500_256.json"
PY = sys.executable

ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
       "BENCH": "mathhard", "EVAL_BENCH": "math500",
       "ARMS": "e6,e5,e3,e2", "SEEDS": "42,123,2026", "STEPS": "100"}
ALL_JOBS = [f"mathhard_{a}_s{s}" for a in ("e2", "e3", "e5", "e6") for s in (42, 123, 2026)]


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (LOGS / "watchdog.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def matrix_procs_alive() -> bool:
    ps = ("(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Select-Object -ExpandProperty CommandLine) -join \"`n\"")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return True
    return any(k in out for k in ("run_matrix_v2", "run_grpo", "eval_pass_at_1"))


def matrix_complete() -> bool:
    if not STATUS.exists():
        return False
    try:
        d = json.loads(STATUS.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    jobs = d.get("jobs", {})
    return all(jobs.get(j, {}).get("state") in ("done", "skipped") for j in ALL_JOBS)


def vram_idle() -> bool:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()[0]
        return int(out) < 2000
    except Exception:
        return False


def run_e0_256() -> None:
    idle_seen = 0
    while idle_seen < 3:  # 3 consecutive minutes idle
        if vram_idle():
            idle_seen += 1
        else:
            idle_seen = 0
        time.sleep(60)
    log("launching E0@256 baseline eval")
    rc = subprocess.run(
        [PY, "-X", "utf8", "src/evaluation/eval_pass_at_1.py",
         "--model", "models/Qwen3-4B-Instruct",
         "--data", "artifacts/benchmarks/math500/test.jsonl",
         "--max-new-tokens", "256",
         "--out", str(E0_256),
         "--dump-rows", str(ART / "bench_rows" / "e0_math500_256.jsonl")],
        cwd=str(ROOT), env={**ENV},
    ).returncode
    log(f"E0@256 eval rc={rc}")


def main() -> None:
    log("watchdog started")
    miss_streak = 0
    while True:
        try:
            if matrix_complete():
                if not E0_256.exists():
                    log("matrix complete; ensuring E0@256 baseline")
                    run_e0_256()
                log("pipeline fully complete; watchdog exiting")
                return
            alive = matrix_procs_alive()
            miss_streak = miss_streak + 1 if not alive else 0
            if miss_streak >= 2:
                log("matrix dead and incomplete -> relaunching")
                subprocess.run([PY, "-X", "utf8", "scripts/run_matrix_v2.py"],
                               cwd=str(ROOT), env=ENV)
                miss_streak = 0
        except Exception as e:
            log(f"watchdog error: {e!r}")
        time.sleep(300)


if __name__ == "__main__":
    main()
