"""Run E2/E5 comparison and write results to artifacts/logs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "artifacts" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

PY = sys.executable


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
    steps = sys.argv[1] if len(sys.argv) > 1 else "30"
    summary: dict = {"steps": steps, "runs": []}

    jobs = [
        (
            [PY, "-X", "utf8", "src/train/run_grpo.py",
             "--config", "configs/e2_binary_grpo.yaml",
             "--max-steps", steps, "--output", "artifacts/checkpoints/e2_hard"],
            "e2_train.log",
        ),
        (
            [PY, "-X", "utf8", "src/evaluation/eval_pass_at_1.py",
             "--model", "artifacts/checkpoints/e2_hard",
             "--out", "artifacts/eval_e2_hard.json"],
            "e2_eval.log",
        ),
        (
            [PY, "-X", "utf8", "src/train/run_grpo.py",
             "--config", "configs/e5_tiny_gar.yaml",
             "--max-steps", steps, "--output", "artifacts/checkpoints/e5_hard"],
            "e5_train.log",
        ),
        (
            [PY, "-X", "utf8", "src/evaluation/eval_pass_at_1.py",
             "--model", "artifacts/checkpoints/e5_hard",
             "--out", "artifacts/eval_e5_hard.json"],
            "e5_eval.log",
        ),
    ]

    t0 = time.time()
    for cmd, name in jobs:
        rc = run(cmd, name)
        summary["runs"].append({"log": name, "rc": rc, "elapsed_s": round(time.time() - t0, 1)})

    for tag in ("e2_hard", "e5_hard"):
        p = ROOT / "artifacts" / f"eval_{tag}.json"
        if p.exists():
            summary[tag] = json.loads(p.read_text(encoding="utf-8"))

    out = LOGS / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[summary] {out}", flush=True)


if __name__ == "__main__":
    main()
