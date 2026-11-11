"""Offline validation of the multi-judge uncertainty gate (E7 mechanism).

Samples completions from an eval rows dump (artifacts/rows/*.jsonl), scores
them with the real local Ollama judges, and reports whether judge disagreement
concentrates on wrong/low-quality answers — i.e. whether the gate suppresses
the right trajectories. Uses GPU via Ollama: run when no training job is active.

Writes artifacts/judge_gate_report.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rewards.ollama_judge import multi_judge  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default=str(ART_ROWS := ROOT / "artifacts" / "rows" / "e0_canon.jsonl"))
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--out", default=str(ROOT / "artifacts" / "judge_gate_report.json"))
    args = p.parse_args()

    rows = load_rows(Path(args.rows))
    rng = random.Random(42)
    # half passed / half failed, to measure discrimination
    passed = [r for r in rows if r["passed"]]
    failed = [r for r in rows if not r["passed"]]
    sample = rng.sample(passed, min(args.n // 2, len(passed))) + rng.sample(
        failed, min(args.n // 2, len(failed))
    )
    rng.shuffle(sample)

    task_lookup = {
        r["task_id"]: r
        for r in (
            json.loads(l)
            for l in (ROOT / "artifacts" / "datasets" / "test.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        )
    }

    out_rows = []
    for r in sample:
        task = task_lookup.get(r["task_id"], {})
        res = multi_judge(task.get("prompt", r["task_id"]), r["completion"])
        scores = [(s.judge, s.quality, s.ok) for s in res.scores]
        out_rows.append(
            {
                "task_id": r["task_id"],
                "domain": r["domain"],
                "passed": r["passed"],
                "tokens": r["tokens"],
                "judge_scores": scores,
                "mean_quality": res.mean_quality,
                "uncertainty": res.uncertainty,
                "gate_suppresses": res.uncertainty >= 0.6,
            }
        )
        print(
            f"  {r['task_id']:<20} pass={r['passed']!s:<5} mean_q={res.mean_quality:.2f} "
            f"unc={res.uncertainty:.2f} gate_off={res.uncertainty >= 0.6}"
        )

    def _avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    ok_rows = [r for r in out_rows if all(s[2] for s in r["judge_scores"])]
    summary = {
        "n": len(out_rows),
        "judge_fully_ok_rows": len(ok_rows),
        "mean_quality_on_passed": _avg([r["mean_quality"] for r in out_rows if r["passed"]]),
        "mean_quality_on_failed": _avg([r["mean_quality"] for r in out_rows if not r["passed"]]),
        "mean_uncertainty_on_passed": _avg([r["uncertainty"] for r in out_rows if r["passed"]]),
        "mean_uncertainty_on_failed": _avg([r["uncertainty"] for r in out_rows if not r["passed"]]),
        "gate_suppression_rate_on_passed": (
            sum(1 for r in out_rows if r["passed"] and r["gate_suppresses"]) /
            max(sum(1 for r in out_rows if r["passed"]), 1)
        ),
        "gate_suppression_rate_on_failed": (
            sum(1 for r in out_rows if not r["passed"] and r["gate_suppresses"]) /
            max(sum(1 for r in out_rows if not r["passed"]), 1)
        ),
        "rows": out_rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    core = {k: v for k, v in summary.items() if k != "rows"}
    print(json.dumps(core, indent=2, ensure_ascii=False))
    print(f"[report] {args.out}")


if __name__ == "__main__":
    main()
