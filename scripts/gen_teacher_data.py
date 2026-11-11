"""MOPD-lite stage 1: multi-teacher verified distillation data.

For each MATH train problem, sample k completions from each local Ollama
teacher, verify against the ground-truth answer, and keep the shortest correct
solution per (problem, teacher). Output: artifacts/distill/sft_data.jsonl with
{prompt, completion} pairs matching the v2.1 brief-CoT format.

This is our offline, budget simplification of MiMo-V2-Flash's MOPD
(multi-teacher on-policy distillation): multi-teacher diversity is preserved,
verification replaces token-level KL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from environments.math_env import MathVerifier  # noqa: E402

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
TEACHERS = ["qwen3:8b", "qwen2.5:7b"]


def ollama_generate(model: str, prompt: str, temperature: float, num_predict: int, timeout: float = 180.0) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict, "top_p": 0.95},
        "think": False,
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "artifacts" / "benchmarks" / "mathtrain" / "train.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "distill" / "sft_data.jsonl"))
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--teachers", default=",".join(TEACHERS))
    ap.add_argument("--num-predict", type=int, default=320)
    args = ap.parse_args()

    teachers = args.teachers.split(",")
    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    verifier = MathVerifier()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # resume: skip task_ids already present in the partial output
    done: set[str] = set()
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["task_id"])
        print(f"[resume] {len(done)} task_ids already done", flush=True)
        rows = [r for r in rows if r["task_id"] not in done]

    n_kept = n_tried = 0
    stats = {t: {"tried": 0, "correct": 0} for t in teachers}
    with out_path.open("a", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            best: dict[str, str] = {}
            for teacher in teachers:
                for _ in range(args.k):
                    try:
                        text = ollama_generate(
                            teacher, row["prompt"],
                            temperature=0.8, num_predict=args.num_predict,
                        )
                    except Exception as e:
                        print(f"[warn] {teacher} failed: {e}", flush=True)
                        break
                    n_tried += 1
                    stats[teacher]["tried"] += 1
                    res = verifier.verify(text, row["reference_answer"])
                    if res.passed and "Answer:" in text:
                        stats[teacher]["correct"] += 1
                        # keep the shortest correct answer per teacher (brevity style)
                        if teacher not in best or len(text) < len(best[teacher]):
                            best[teacher] = text.strip()
                        break  # one verified sample per teacher per problem is enough
            for teacher, text in best.items():
                fout.write(json.dumps({
                    "prompt": row["prompt"],
                    "completion": text,
                    "task_id": row["task_id"],
                    "teacher": teacher,
                }, ensure_ascii=False) + "\n")
                n_kept += 1
                fout.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(rows)}] kept so far: {n_kept} ({time.strftime('%H:%M:%S')})", flush=True)

    summary = {
        "n_problems": len(rows),
        "n_generations_tried": n_tried,
        "n_sft_pairs": n_kept,
        "per_teacher": stats,
        "coverage_problems": n_kept / max(len(rows), 1),
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
