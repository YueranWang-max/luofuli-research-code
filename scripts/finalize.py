"""Idempotent end-to-end finalization (sleep/crash safe).

Safe to re-run at any time; each stage checks its own outputs first:
  1. re-evaluate legacy checkpoints e2_hard/e5_hard with the current scorer
     (only when their per-task rows are missing, e.g. after a sleep interrupt)
  2. rebuild master table + statistics
  3. generate figures
  4. offline multi-judge gate validation (skipped unless Ollama is reachable)

Usage: python -X utf8 scripts/finalize.py [--with-judge]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
PY = sys.executable
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

# legacy checkpoint -> canonical (arm, seed) whose eval/rows must exist
REEVALS = [
    ("artifacts/checkpoints/e2_hard", "e2_s42"),
    ("artifacts/checkpoints/e5_hard", "e5_s42"),
]


def sh(cmd: list[str], log: Path) -> int:
    print(f"[run] {cmd}", flush=True)
    with log.open("w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, env=ENV).returncode
    print(f"[done] rc={rc} log={log.name}", flush=True)
    return rc


def ollama_up(timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-judge", action="store_true", help="Also run Ollama gate validation")
    args = ap.parse_args()

    # 1. ensure re-evaluated s42 rows exist
    for ckpt, name in REEVALS:
        rows = ART / "rows" / f"{name}.jsonl"
        if rows.exists() and rows.stat().st_size > 1000:
            print(f"[ok] {name}: rows already present")
            continue
        rc = sh(
            [PY, "-X", "utf8", "src/evaluation/eval_pass_at_1.py",
             "--model", ckpt,
             "--out", f"artifacts/eval_{name}.json",
             "--dump-rows", f"artifacts/rows/{name}.jsonl"],
            ART / "logs" / f"finalize_{name}_eval.log",
        )
        if rc != 0:
            print(f"[warn] {name} eval rc={rc}; continuing (table uses existing artifacts)")

    # 2. master table
    sh([PY, "-X", "utf8", "scripts/build_master_table.py"], ART / "logs" / "finalize_table.log")

    # 3. figures
    sh([PY, "-X", "utf8", "scripts/make_figures.py"], ART / "logs" / "finalize_figures.log")

    # 4. judge gate validation (optional; needs Ollama + free GPU)
    if args.with_judge or ollama_up():
        if ollama_up():
            sh([PY, "-X", "utf8", "scripts/judge_gate_demo.py"], ART / "logs" / "finalize_judge.log")
        else:
            print("[skip] judge gate: Ollama not reachable")

    # final one-line status
    mt = ART / "master_table.json"
    if mt.exists():
        t = json.loads(mt.read_text(encoding="utf-8"))
        arms = {a: f"{d['pass_at_1_mean']:.3f}" for a, d in t["arms"].items()}
        print(f"[final] arms: {arms}", flush=True)
    print("[finalize] complete — see RESULTS.md; hand-update remaining prose if needed")


if __name__ == "__main__":
    main()
