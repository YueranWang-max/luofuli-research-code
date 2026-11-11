"""Failure analysis: locate and explain the code-domain collapse after RL.

Compares per-task completions between a reference arm (default E0 base) and a
trained arm on the code domain, and summarises how the answers changed.
Writes artifacts/failure_analysis_<ref>_vs_<arm>.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "artifacts" / "rows"


def load(name: str) -> dict[str, dict]:
    p = ROWS / f"{name}.jsonl"
    return {
        r["task_id"]: r
        for r in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
    }


def classify(text: str) -> str:
    t = text.strip()
    if not t:
        return "empty"
    if any(ch.isdigit() for ch in t) and not any(op in t for op in ("*", "+", "-", "/", "**", "min", "max", "not", "x", "a", "b")):
        return "bare_number"
    if any(op in t for op in ("*", "+", "-", "/", "**")) or any(k in t for k in ("min", "max", "not")):
        return "expression"
    return "other"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default="e0_canon")
    p.add_argument("--arm", default="e6_s42")
    args = p.parse_args()

    ref, arm = load(args.ref), load(args.arm)
    common = sorted(set(ref) & set(arm))
    code = [t for t in common if ref[t]["domain"] == "code"]
    math = [t for t in common if ref[t]["domain"] == "math"]

    report = {
        "ref": args.ref,
        "arm": args.arm,
        "n_common": len(common),
        "math": {
            "ref_pass": sum(ref[t]["passed"] for t in math) / max(len(math), 1),
            "arm_pass": sum(arm[t]["passed"] for t in math) / max(len(math), 1),
            "gained": sum(1 for t in math if not ref[t]["passed"] and arm[t]["passed"]),
            "lost": sum(1 for t in math if ref[t]["passed"] and not arm[t]["passed"]),
        },
        "code": {
            "ref_pass": sum(ref[t]["passed"] for t in code) / max(len(code), 1),
            "arm_pass": sum(arm[t]["passed"] for t in code) / max(len(code), 1),
            "ref_answer_kinds": dict(Counter(classify(ref[t]["completion"]) for t in code)),
            "arm_answer_kinds": dict(Counter(classify(arm[t]["completion"]) for t in code)),
        },
        "code_examples": [
            {
                "task_id": t,
                "ref_completion": ref[t]["completion"][:120],
                "arm_completion": arm[t]["completion"][:120],
                "ref_passed": ref[t]["passed"],
                "arm_passed": arm[t]["passed"],
            }
            for t in code[:8]
        ],
    }
    out = ROOT / "artifacts" / f"failure_analysis_{args.ref}_vs_{args.arm}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "code_examples"}, indent=2, ensure_ascii=False))
    for e in report["code_examples"]:
        print(f"  {e['task_id']}: ref={e['ref_completion']!r} ({e['ref_passed']}) -> arm={e['arm_completion']!r} ({e['arm_passed']})")
    print(f"[report] {out}")


if __name__ == "__main__":
    main()
