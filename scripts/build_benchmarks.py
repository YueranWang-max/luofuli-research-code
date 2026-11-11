"""Build real-benchmark datasets (GSM8K / MBPP-sanitized / MATH-500) into our schema.

Writes artifacts/benchmarks/{name}/{train,val,test}.jsonl with the unified row:
  task_id, prompt, reference_answer | public/hidden tests, domain, difficulty

Math rows: brief-CoT prompt, final answer extraction via 'Answer: X'.
Code rows (MBPP): function-completion prompt; test_list split 1 public / 2 hidden.
Downloads via HF datasets (cached under HF_HOME default).
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)

MATH_PROMPT = (
    "{q}\n\nSolve this problem with at most three short steps (under 120 words "
    "total), then end your reply with exactly 'Answer: <final number>'. "
    "The final answer must be a plain number (no units, no commas)."
)
CODE_PROMPT = (
    "{q}\n\nWrite a complete Python function that solves this task. "
    "The function must be named `{fname}` (the tests call it by this name). "
    "Output only the function definition in a ```python code block, nothing else."
)


def gsm8k() -> None:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")
    rng = random.Random(42)

    def fmt(split: str, n: int, tag: str) -> list[dict]:
        rows = []
        for ex in ds[split]:
            ans = ex["answer"].split("####")[-1].strip().replace(",", "")
            q = ex["question"].strip()
            rows.append({"q": q, "a": ans})
        rng.shuffle(rows)
        rows = rows[:n]
        out = []
        for i, r in enumerate(rows):
            out.append({
                "task_id": f"gsm8k_{tag}_{i}",
                "prompt": MATH_PROMPT.format(q=r["q"]),
                "reference_answer": r["a"],
                "public_tests": [],
                "hidden_tests": [],
                "difficulty": "math_word",
                "domain": "math",
                "meta": {"source": "openai/gsm8k-main", "split": split},
            })
        return out

    splits = {"train": fmt("train", 1500, "tr"), "val": fmt("test", 100, "va"), "test": fmt("test", 500, "te")}
    _write("gsm8k", splits)


def mbpp() -> None:
    import pandas as pd

    raw = ROOT / "artifacts" / "benchmarks" / "raw"
    rng = random.Random(42)

    def to_rows(rows: list[dict], tag: str) -> list[dict]:
        out = []
        for ex in rows:
            raw_tests = ex.get("test_list")
            tests = [str(t) for t in (list(raw_tests) if raw_tests is not None else []) if str(t).strip()]
            if len(tests) < 2:
                continue
            raw_imports = ex.get("test_imports")
            imports = [str(i) for i in (list(raw_imports) if raw_imports is not None else [])]
            prefix = ("\n".join(imports) + "\n") if imports else ""
            # expected function name from the first public assert, so model
            # definitions match the hidden-test call sites
            m = re.search(r"(\w+)\s*\(", tests[0])
            fname = m.group(1) if m else "solve"
            out.append({
                "task_id": f"mbpp_{tag}_{ex['task_id']}",
                "prompt": CODE_PROMPT.format(q=str(ex["prompt"]).strip(), fname=fname),
                "reference_answer": "",
                "public_tests": [prefix + tests[0]],
                "hidden_tests": [prefix + t for t in tests[1:]],
                "difficulty": "code_func",
                "domain": "code",
                "meta": {"source": "mbpp-sanitized"},
            })
        return out

    train_pool = pd.concat([
        pd.read_parquet(raw / "mbpp_sanitized_train.parquet"),
        pd.read_parquet(raw / "mbpp_sanitized_validation.parquet"),
    ]).to_dict("records")
    rng.shuffle(train_pool)
    # single shuffle of the test split so val/test never overlap
    test_pool = pd.read_parquet(raw / "mbpp_sanitized_test.parquet").to_dict("records")
    rng.shuffle(test_pool)
    test_rows, val_rows = test_pool[:150], test_pool[150:210]

    splits = {
        "train": to_rows(train_pool, "tr"),
        "val": to_rows(val_rows, "va"),
        "test": to_rows(test_rows, "te"),
    }
    _write("mbpp", splits)


def math500() -> None:
    import re as _re

    raw = ROOT / "artifacts" / "benchmarks" / "raw" / "math500_test.jsonl"
    rows = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines() if l.strip()]
    # keep hard levels with machine-verifiable numeric answers only
    rows = [
        ex for ex in rows
        if str(ex.get("level")) in ("4", "5")
        and _re.fullmatch(r"-?\d+(\.\d+)?", str(ex["answer"]).strip())
    ]
    rng = random.Random(42)
    rng.shuffle(rows)
    test, pool = rows[:200], rows[200:]
    splits = {
        "test": [{
            "task_id": f"math500_te_{i}",
            "prompt": MATH_PROMPT.format(q=ex["problem"].strip()),
            "reference_answer": str(ex["answer"]).strip(),
            "public_tests": [],
            "hidden_tests": [],
            "difficulty": f"L{ex['level']}",
            "domain": "math",
            "meta": {"source": "MATH-500", "level": ex["level"], "subject": ex.get("subject", "")},
        } for i, ex in enumerate(test)],
        "train": [{
            "task_id": f"math500_tr_{i}",
            "prompt": MATH_PROMPT.format(q=ex["problem"].strip()),
            "reference_answer": str(ex["answer"]).strip(),
            "public_tests": [],
            "hidden_tests": [],
            "difficulty": f"L{ex['level']}",
            "domain": "math",
            "meta": {"source": "MATH-500", "level": ex["level"]},
        } for i, ex in enumerate(pool)],
    }
    _write("math500", splits)


def _write(name: str, splits: dict[str, list[dict]]) -> None:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with (d / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{name}] {split}: {len(rows)}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "gsm8k"):
        gsm8k()
    if which in ("all", "mbpp"):
        mbpp()
    if which in ("all", "math500"):
        math500()
