"""Evaluate Pass@1 and token efficiency on a held-out split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from train.run_grpo import score_completion


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models" / "Qwen2.5-1.5B-Instruct"))
    p.add_argument("--data", default=str(ROOT / "artifacts" / "datasets" / "test.jsonl"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--out", default="")
    p.add_argument("--dump-rows", default="", help="Optional path for per-task pass/fail rows (jsonl)")
    args = p.parse_args()

    rows = load_jsonl(Path(args.data))
    if args.limit:
        rows = rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    import json as _json

    model_dir = Path(args.model)
    adapter_cfg = model_dir / "adapter_config.json"
    if adapter_cfg.exists():
        base_name = _json.loads(adapter_cfg.read_text(encoding="utf-8")).get(
            "base_model_name_or_path", "models/Qwen2.5-1.5B-Instruct"
        )
        if not Path(base_name).exists():
            local_base = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
            if local_base.exists():
                base_name = str(local_base)
        base = AutoModelForCausalLM.from_pretrained(
            base_name, dtype=torch.bfloat16, device_map="auto", use_safetensors=True
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, str(model_dir))
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto", use_safetensors=True
        )
    model.eval()

    n = passed = total_tokens = pass_tokens = pass_n = 0
    per_domain: dict[str, dict] = {}
    task_rows: list[dict] = []

    for row in rows:
        messages = [{"role": "user", "content": row["prompt"]}]
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True)
        g = score_completion(row, text)
        n += 1
        total_tokens += g.generated_tokens
        if g.passed:
            passed += 1
            pass_n += 1
            pass_tokens += g.generated_tokens
        d = row.get("domain", "other")
        slot = per_domain.setdefault(d, {"n": 0, "pass": 0, "tokens": 0})
        slot["n"] += 1
        slot["pass"] += int(g.passed)
        slot["tokens"] += g.generated_tokens
        task_rows.append(
            {
                "task_id": row.get("task_id", ""),
                "domain": d,
                "difficulty": row.get("difficulty", ""),
                "passed": bool(g.passed),
                "tokens": g.generated_tokens,
                "completion": text,
            }
        )

    result = {
        "n": n,
        "pass_at_1": passed / max(n, 1),
        "avg_tokens_all": total_tokens / max(n, 1),
        "avg_tokens_success": pass_tokens / max(pass_n, 1),
        "per_domain": {
            k: {
                "pass_at_1": v["pass"] / max(v["n"], 1),
                "avg_tokens": v["tokens"] / max(v["n"], 1),
                "n": v["n"],
            }
            for k, v in per_domain.items()
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.dump_rows:
        with Path(args.dump_rows).open("w", encoding="utf-8") as f:
            for r in task_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
