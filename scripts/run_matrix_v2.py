"""v2 experiment matrix on real benchmarks (resumable, single GPU).

Protocol: Qwen3-4B + LoRA, GRPO group=4, 200 steps, 256-token completions.
Per arm/seed writes:
  - checkpoint:   artifacts/bench_ckpt/{bench}_{arm}_s{seed}
  - eval:         artifacts/bench_eval_{bench}_{arm}_s{seed}.json
  - per-task rows: artifacts/bench_rows/{bench}_{arm}_s{seed}.jsonl
  - reward stats: artifacts/bench_reward_stats/{bench}_{arm}_s{seed}.jsonl
Set BENCH env var (default gsm8k). Skips finished jobs (eval json exists).
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

BENCH = os.environ.get("BENCH", "gsm8k")          # training pool
EVAL_BENCH = os.environ.get("EVAL_BENCH", BENCH)  # eval split (may differ)
STEPS = os.environ.get("STEPS", "200")
ARMS = tuple(os.environ.get("ARMS", "e2,e3,e5,e6").split(","))
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "42,123,2026").split(","))


def run(cmd: list[str], log_name: str, env_extra: dict | None = None) -> int:
    log_path = LOGS / log_name
    print(f"[run] {log_name}: {' '.join(cmd)}", flush=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", **(env_extra or {})}
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, env=env)
    print(f"[done] {log_name} rc={proc.returncode}", flush=True)
    return proc.returncode


def main() -> None:
    data = f"artifacts/benchmarks/{BENCH}/train.jsonl"
    test = f"artifacts/benchmarks/{EVAL_BENCH}/test.jsonl"
    status_path = LOGS / f"matrix_v2_{BENCH}_status.json"
    (ART / "bench_reward_stats").mkdir(parents=True, exist_ok=True)
    status: dict = {"jobs": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "bench": BENCH}

    for arm in ARMS:
        for seed in SEEDS:
            name = f"{BENCH}_{arm}_s{seed}"
            eval_out = ART / f"bench_eval_{name}.json"
            if eval_out.exists():
                print(f"[skip] {name}", flush=True)
                status["jobs"][name] = {"state": "skipped"}
                continue
            t0 = time.time()
            ckpt = f"artifacts/bench_ckpt/{name}"
            rc = run(
                [PY, "-X", "utf8", "src/train/run_grpo.py",
                 "--config", f"configs/v2/{arm}.yaml", "--data", data,
                 "--seed", str(seed), "--max-steps", STEPS, "--output", ckpt],
                f"v2_{name}_train.log",
                env_extra={"REWARD_STATS_PATH": str(ART / "bench_reward_stats" / f"{name}.jsonl")},
            )
            if rc == 0:
                rc = run(
                    [PY, "-X", "utf8", "src/evaluation/eval_pass_at_1.py",
                     "--model", ckpt, "--data", test, "--max-new-tokens", "256",
                     "--out", str(eval_out),
                     "--dump-rows", str(ART / "bench_rows" / f"{name}.jsonl")],
                    f"v2_{name}_eval.log",
                )
            status["jobs"][name] = {
                "state": "done" if rc == 0 else f"failed_rc{rc}",
                "elapsed_s": round(time.time() - t0, 1),
            }
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"[v2 complete] {status_path}", flush=True)


if __name__ == "__main__":
    main()
