"""Sleep-robust orchestrator: finish v2 matrix, then pivot to v2.2 (KL-anchored).

Stages (all idempotent, safe to re-run after any interruption):
  1. wait until no matrix/training process is alive; if jobs are incomplete,
     re-run run_matrix_v2.py to finish them (it skips completed jobs)
  2. verify the KL-drift evidence (all three no-KL E2 seeds below baseline-1pp);
     if not confirmed, stop and leave a marker for manual review
  3. rename no-KL E2 artifacts to e2nokl_* (ablation arm)
  4. patch configs/v2/*.yaml with train.beta: 0.05
  5. run the full v2.2 matrix (12 jobs)

Marker files under artifacts/logs/ record progress so re-runs skip stages.
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
PY = sys.executable
STATUS = LOGS / "matrix_v2_mathtrain_status.json"
MARKER = LOGS / "pivot_v22_state.txt"
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
       "BENCH": "mathtrain", "EVAL_BENCH": "math500",
       "ARMS": "e2,e3,e5,e6", "SEEDS": "42,123,2026"}
ALL_JOBS = [f"mathtrain_{a}_s{s}" for a in ("e2", "e3", "e5", "e6") for s in (42, 123, 2026)]


def marker() -> str:
    return MARKER.read_text(encoding="utf-8").strip() if MARKER.exists() else ""


def set_marker(stage: str) -> None:
    MARKER.write_text(stage, encoding="utf-8")
    print(f"[pivot] stage -> {stage}", flush=True)


def matrix_procs_alive() -> bool:
    ps = ("(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Select-Object -ExpandProperty CommandLine) -join \"`n\"")
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:
        return True  # cannot determine: assume alive to be safe
    return any(k in out for k in ("run_matrix_v2", "run_grpo", "eval_pass_at_1"))


def matrix_complete() -> bool:
    if not STATUS.exists():
        return False
    d = json.loads(STATUS.read_text(encoding="utf-8"))
    jobs = d.get("jobs", {})
    return all(jobs.get(j, {}).get("state") in ("done", "skipped") for j in ALL_JOBS)


def run_matrix() -> int:
    print("[pivot] running matrix (resumes finished jobs)", flush=True)
    proc = subprocess.run(
        [PY, "-X", "utf8", "scripts/run_matrix_v2.py"],
        cwd=str(ROOT), env=ENV,
    )
    return proc.returncode


def wait_and_complete_matrix() -> None:
    retries = 0
    while not matrix_complete():
        if matrix_procs_alive() and status_fresh(max_age_h=6.0):
            print("[pivot] matrix processes alive, waiting 120s", flush=True)
            time.sleep(120)
            continue
        if retries >= 3:
            print("[pivot] matrix kept failing; moving on for manual review", flush=True)
            return
        retries += 1
        rc = run_matrix()
        if rc != 0:
            print(f"[pivot] matrix run rc={rc}, retry {retries}/3 in 5 min", flush=True)
            time.sleep(300)
    print("[pivot] v2 matrix complete", flush=True)


def status_fresh(max_age_h: float) -> bool:
    if not STATUS.exists():
        return True
    age_h = (time.time() - STATUS.stat().st_mtime) / 3600
    return age_h < max_age_h


def e2_scores() -> dict:
    out = {}
    for s in (42, 123, 2026):
        p = ART / f"bench_eval_mathtrain_e2_s{s}.json"
        if p.exists():
            out[s] = json.loads(p.read_text(encoding="utf-8"))["pass_at_1"]
    return out


def rename_e2_to_nokl() -> None:
    for s in (42, 123, 2026):
        pairs = [
            (ART / f"bench_eval_mathtrain_e2_s{s}.json",
             ART / f"bench_eval_mathtrain_e2nokl_s{s}.json"),
            (ART / "bench_rows" / f"mathtrain_e2_s{s}.jsonl",
             ART / "bench_rows" / f"mathtrain_e2nokl_s{s}.jsonl"),
            (ART / "bench_reward_stats" / f"mathtrain_e2_s{s}.jsonl",
             ART / "bench_reward_stats" / f"mathtrain_e2nokl_s{s}.jsonl"),
        ]
        for src, dst in pairs:
            if src.exists():
                src.rename(dst)
                print(f"[pivot] {src.name} -> {dst.name}", flush=True)


def patch_configs() -> None:
    for arm in ("e2", "e3", "e5", "e6"):
        p = ROOT / "configs" / "v2" / f"{arm}.yaml"
        text = p.read_text(encoding="utf-8")
        if "beta:" in text:
            continue
        text = text.replace("train:\n", "train:\n  beta: 0.05\n", 1)
        p.write_text(text, encoding="utf-8")
        print(f"[pivot] {p.name}: beta=0.05", flush=True)


def main() -> None:
    stage = marker()
    if stage != "done_v22_matrix":
        if stage == "":
            set_marker("completing_v2")
        wait_and_complete_matrix()
    if marker() in ("", "completing_v2"):
        scores = e2_scores()
        print(f"[pivot] no-KL E2 scores: {scores}", flush=True)
        if len(scores) == 3 and all(v < 0.655 for v in scores.values()):
            set_marker("renaming")
            rename_e2_to_nokl()
            set_marker("patching")
            patch_configs()
            set_marker("running_v22")
            run_matrix()
            set_marker("done_v22_matrix")
        else:
            (LOGS / "pivot_v22_HALT.txt").write_text(
                f"E2 degradation not confirmed by all seeds: {scores}. "
                "Manual review needed before KL pivot.\n", encoding="utf-8")
            print("[pivot] HALTED: degradation not confirmed, see pivot_v22_HALT.txt", flush=True)
            return
    elif stage in ("renaming",):
        rename_e2_to_nokl()
        set_marker("patching")
        patch_configs()
        set_marker("running_v22")
        run_matrix()
        set_marker("done_v22_matrix")
    elif stage in ("patching",):
        patch_configs()
        set_marker("running_v22")
        run_matrix()
        set_marker("done_v22_matrix")
    elif stage in ("running_v22",):
        run_matrix()
        set_marker("done_v22_matrix")
    print("[pivot] all stages complete", flush=True)


if __name__ == "__main__":
    main()
