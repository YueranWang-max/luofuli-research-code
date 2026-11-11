"""Build the MATH RL training pool from hendrycks MATH train parquets.

Keeps L3-L5 problems whose boxed answer is a plain number (machine-verifiable),
deduplicates against the MATH-500 eval subset by problem text, and writes
artifacts/benchmarks/mathtrain/{train,val}.jsonl.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / "benchmarks" / "raw" / "math_train"
OUT = ROOT / "artifacts" / "benchmarks" / ("mathhard" if "--l45" in sys.argv else "mathtrain")
OUT.mkdir(parents=True, exist_ok=True)

MATH_PROMPT = (
    "{q}\n\nSolve this problem with at most three short steps (under 120 words "
    "total), then end your reply with exactly 'Answer: <final number>'. "
    "The final answer must be a plain number (no units, no commas)."
)


def boxed_answer(sol: str) -> str:
    m = re.findall(r"\\boxed\{([^{}]+)\}", sol)
    return m[-1].strip() if m else ""


def main() -> None:
    hard_only = "--l45" in sys.argv
    if hard_only:
        sys.argv.remove("--l45")
        keep = ("Level 4", "Level 5")
    else:
        keep = ("Level 1", "Level 2", "Level 3", "Level 4", "Level 5")
    rows = []
    for p in sorted(RAW.glob("*_train.parquet")):
        subj = p.name.split("_train")[0]
        for ex in pd.read_parquet(p).to_dict("records"):
            ans = boxed_answer(str(ex.get("solution", "")))
            level = str(ex.get("level", ""))
            if level not in keep:
                continue
            if not re.fullmatch(r"-?\d+(\.\d+)?", ans):
                continue
            rows.append({
                "problem": str(ex["problem"]).strip(),
                "answer": ans,
                "level": level,
                "subject": subj,
            })

    # dedup against MATH-500 eval problems
    eval_texts = {
        json.loads(l)["prompt"].split("\n\nReason briefly")[0]
        for l in (ROOT / "artifacts" / "benchmarks" / "math500" / "test.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()
    }
    before = len(rows)
    rows = [r for r in rows if r["problem"] not in eval_texts]
    print(f"rows L3-L5 numeric: {before}, after eval-dedup: {len(rows)}")

    rng = random.Random(42)
    rng.shuffle(rows)
    n_train = min(1500, int(len(rows) * 0.9))
    splits = {
        "train": rows[:n_train],
        "val": rows[n_train:n_train + 100],
    }
    for split, items in splits.items():
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for i, r in enumerate(items):
                f.write(json.dumps({
                    "task_id": f"mathtr_{split[:2]}_{i}",
                    "prompt": MATH_PROMPT.format(q=r["problem"]),
                    "reference_answer": r["answer"],
                    "public_tests": [],
                    "hidden_tests": [],
                    "difficulty": r["level"].replace("Level ", "L"),
                    "domain": "math",
                    "meta": {"source": "hendrycks-MATH-train", "subject": r["subject"]},
                }, ensure_ascii=False) + "\n")
        print(f"[mathtrain] {split}: {len(items)}")


if __name__ == "__main__":
    main()
