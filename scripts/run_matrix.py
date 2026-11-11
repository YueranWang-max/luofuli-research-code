"""Sequential, resumable experiment-matrix runner (single GPU).

Protocol matches the existing E2/E5 runs: 60 GRPO steps, greedy 96-token eval
on artifacts/datasets/test.jsonl. Each job writes:
  - checkpoint: artifacts/checkpoints/{arm}_s{seed}
  - eval:       artifacts/eval_{arm}_s{seed}.json
A job is skipped when its eval json already exists, so the script can be
re-run after interruption without redoing finished arms.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "artifacts" / "checkpoints"
LOGS = ROOT / "artifacts" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

PY = sys.executable
STEPS = "60"

CONFIGS = {
    "e2": "configs/e2_binary_grpo.yaml",
    "e3": "configs/e3_length_penalty.yaml",
    "e4": "configs/e4_tiny_grs.yaml",
    "e5": "configs/e5_tiny_gar.yaml",
    "e6": "configs/e6_budget_gar.yaml",
    "e7l": "configs/e7_uncertainty_gar_local.yaml",
}


def jobs() -> list[dict]:
    out = [
        # canonical E0 baseline with the current scorer (base model, no training)
        {
            "name": "e0_canon",
            "eval_only": True,
            "model": "models/Qwen3-4B-Instruct",
        }
    ]
    # main matrix, seed 42 first so headline results land early
    for arm in ("e6", "e3", "e4", "e7l"):
        out.append({"name": f"{arm}_s42", "arm": arm, "seed": 42})
    # legacy s42 artifacts (keep names so existing evals stay valid)
    out.append({"name": "e2_s42", "arm": "e2", "seed": 42, "ckpt": "e2_hard",
                "eval_src": "eval_e2_hard.json"})
    out.append({"name": "e5_s42", "arm": "e5", "seed": 42, "ckpt": "e5_hard",
                "eval_src": "eval_e5_hard.json"})
    # seed replication for the must-do arms
    for arm in ("e2", "e3", "e5", "e6"):
        for seed in (123, 2026):
            out.append({"name": f"{arm}_s{seed}", "arm": arm, "seed": seed})
    return out


def run(cmd: list[str], log_name: str) -> int:
    log_path = LOGS / log_name
    print(f"[run] {log_name}: {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
    print(f"[done] {log_name} rc={proc.returncode}", flush=True)
    return proc.returncode


def main() -> None:
    status_path = LOGS / "matrix_status.json"
    rows_dir = ROOT / "artifacts" / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    status: dict = {"jobs": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    for job in jobs():
        name = job["name"]
        eval_out = ROOT / "artifacts" / f"eval_{name}.json"
        rows_out = rows_dir / f"{name}.jsonl"
        if eval_out.exists():
            print(f"[skip] {name}: {eval_out.name} exists", flush=True)
            status["jobs"][name] = {"state": "skipped"}
            continue

        t0 = time.time()
        rc_all = 0
        if job.get("eval_only"):
            rc = run(
                [PY, "-X", "utf8", "src/evaluation/eval_pass_at_1.py",
                 "--model", job["model"], "--out", str(eval_out),
                 "--dump-rows", str(rows_out)],
                f"matrix_{name}_eval.log",
            )
            rc_all |= rc
        else:
            arm, seed = job["arm"], job["seed"]
            ckpt = job.get("ckpt", f"{arm}_s{seed}")
            src = job.get("eval_src")
            if src and (ROOT / "artifacts" / src).exists():
                # reuse the existing s42 eval artifact under the canonical name
                eval_out.write_text(
                    (ROOT / "artifacts" / src).read_text(encoding="utf-8"), encoding="utf-8"
                )
                status["jobs"][name] = {"state": "reused", "elapsed_s": 0}
                continue
            rc = run(
                [PY, "-X", "utf8", "src/train/run_grpo.py",
                 "--config", CONFIGS[arm], "--seed", str(seed),
                 "--max-steps", STEPS, "--output", f"artifacts/checkpoints/{ckpt}"],
                f"matrix_{name}_train.log",
            )
            rc_all |= rc
            if rc == 0:
                rc = run(
                    [PY, "-X", "utf8", "src/evaluation/eval_pass_at_1.py",
                     "--model", f"artifacts/checkpoints/{ckpt}", "--out", str(eval_out),
                     "--dump-rows", str(rows_out)],
                    f"matrix_{name}_eval.log",
                )
                rc_all |= rc
        status["jobs"][name] = {
            "state": "done" if rc_all == 0 else f"failed_rc{rc_all}",
            "elapsed_s": round(time.time() - t0, 1),
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"[matrix] complete: {status_path}", flush=True)


if __name__ == "__main__":
    main()
