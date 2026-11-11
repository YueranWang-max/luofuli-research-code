"""Evaluate Ollama teachers on the MATH-500 eval set under the exact v2.1 protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from environments.math_env import MathVerifier  # noqa: E402
from gen_teacher_data import ollama_generate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", default="qwen3:8b,qwen2.5:7b")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "bench_eval_teachers.json"))
    args = ap.parse_args()

    rows = [
        json.loads(l)
        for l in (ROOT / "artifacts" / "benchmarks" / "math500" / "test.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    verifier = MathVerifier()
    result = {}
    for teacher in args.teachers.split(","):
        passes = []
        for i, row in enumerate(rows):
            try:
                text = ollama_generate(teacher, row["prompt"], temperature=0.0, num_predict=384)
            except Exception as e:
                print(f"[warn] {teacher} #{i}: {e}", flush=True)
                text = ""
            res = verifier.verify(text, row["reference_answer"])
            passes.append(res.passed)
            if i % 25 == 0:
                print(f"[{teacher} {i}/{len(rows)}] pass={sum(passes)}/{len(passes)}", flush=True)
        result[teacher] = {
            "pass_at_1": sum(passes) / len(passes),
            "n": len(passes),
        }
        print(f"[{teacher}] FINAL pass@1 = {result[teacher]['pass_at_1']:.3f}", flush=True)

    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
