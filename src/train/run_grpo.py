"""TRL GRPO training entry for single-GPU verifiable math/code experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from environments import CodeVerifier, MathVerifier
from rewards import (
    GraderOutputs,
    GroupwiseConfig,
    QualityConfig,
    RewardMode,
    RewardPipeline,
    RewardPipelineConfig,
    TrajectoryRecord,
)


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_pipeline(cfg: dict) -> RewardPipeline:
    r = cfg.get("reward", {})
    return RewardPipeline(
        RewardPipelineConfig(
            mode=RewardMode(r.get("mode", "binary")),
            quality=QualityConfig(),
            groupwise=GroupwiseConfig(
                lambda_gar=float(r.get("lambda_gar", 0.2)),
                lambda_0=float(r.get("lambda_0", 0.3)),
            ),
            length_penalty_coef=float(r.get("length_penalty_coef", 0.0005)),
            length_target_tokens=int(r.get("length_target_tokens", 256)),
        )
    )


def length_quality(n_tokens: int, knee: float = 24.0) -> float:
    return float(knee / (knee + max(n_tokens, 1)))


def wrap_fill_in(row: dict, completion: str) -> str:
    """Normalize a completion into runnable code.

    Handles full fenced blocks (```python ... ```), truncated fences, and
    bare fill-in expressions (wrapped with the task's function signature).
    """
    import re as _re

    text = completion.strip()
    if "```" in text:
        m = _re.search(r"```(?:python)?\s*\n(.*?)```", text, _re.S)
        if m:
            text = m.group(1).strip()
        else:
            # fence opened but never closed (truncated generation)
            parts = _re.split(r"```(?:python)?", text, 1)
            text = parts[1].strip() if len(parts) > 1 else text
    expr = text.rstrip("`").strip()
    if "Answer:" in expr:
        expr = expr.split("Answer:", 1)[1].strip()
    expr = expr.strip().rstrip(";").strip()
    if expr.startswith("return "):
        expr = expr[len("return ") :].strip()
    if "\n" not in expr and "def " not in expr:
        prompt = row.get("prompt", "")
        # Match the def that contains the blank marker, not the few-shot example.
        m = _re.search(r"def\s+(\w+)\s*\(([^)]*)\)\s*:\s*\n\s*return\s+___", prompt)
        if m:
            fname, args = m.group(1), m.group(2)
            return f"def {fname}({args}):\n    return {expr}\n"
        # Fallback: last def in the prompt (the real task, not the example).
        all_defs = _re.findall(r"def\s+(\w+)\s*\(([^)]*)\)\s*:", prompt)
        if all_defs:
            fname, args = all_defs[-1]
            return f"def {fname}({args}):\n    return {expr}\n"
    return expr


JUDGE_BLEND_CFG = {"enabled": False, "weight": 0.5}


def score_completion(row: dict, completion: str) -> GraderOutputs:
    n_tok = max(len(completion.split()), 1)
    g = GraderOutputs(generated_tokens=n_tok)
    if row.get("domain") == "code":
        code_to_run = wrap_fill_in(row, completion)
        res = CodeVerifier().verify(
            code_to_run,
            row.get("public_tests") or [],
            row.get("hidden_tests") or [],
        )
        g.passed = res.passed
        g.timed_out = res.timed_out
        g.reward_hacking = res.reward_hacking
        g.hidden_passed = res.hidden_passed == res.hidden_total and res.hidden_total > 0
        if res.passed:
            q = length_quality(n_tok, knee=48.0)
            g.coverage = 1.0
            g.minimality = q
            g.efficiency = q
            g.scope = 1.0
            g.style = 0.8
            g.complexity = q
    else:
        res = MathVerifier().verify(completion, row.get("reference_answer") or "")
        g.passed = res.passed
        if res.passed:
            q = length_quality(n_tok)
            g.coverage = 1.0
            g.minimality = q
            g.efficiency = q
            g.scope = 1.0
            g.style = 0.8
            g.complexity = q

    if JUDGE_BLEND_CFG.get("enabled"):
        try:
            from rewards.ollama_judge import multi_judge

            res = multi_judge(row.get("prompt", ""), completion)
            for s in res.scores:
                if s.ok:
                    g.ranker_scores[s.judge] = s.quality
        except Exception:
            g.ranker_scores["judge_fail"] = 0.0
    return g


def make_reward_fn(pipeline: RewardPipeline, task_lookup: dict[str, dict]):
    stats_path = os.environ.get("REWARD_STATS_PATH", "")

    def reward_fn(prompts, completions, **kwargs):
        # group by task id when provided; otherwise score independently
        ids = kwargs.get("task_id") or [f"p{i}" for i in range(len(prompts))]
        groups: dict[str, list[int]] = {}
        for i, tid in enumerate(ids):
            groups.setdefault(str(tid), []).append(i)

        rewards = [0.0] * len(completions)
        n_success_groups = n_redist_groups = n_success = 0
        for tid, idxs in groups.items():
            row = task_lookup.get(tid, {})
            records = []
            for i in idxs:
                text = completions[i]
                if isinstance(text, list):
                    text = text[0].get("content", "") if text else ""
                records.append(
                    TrajectoryRecord(
                        trajectory_id=str(i),
                        task_id=str(tid),
                        group_id=str(tid),
                        seed=0,
                        graders=score_completion(row, text),
                    )
                )
            pipeline.score_group(records)
            n_success += sum(r.graders.passed for r in records)
            if any(r.graders.passed for r in records):
                n_success_groups += 1
            if any(r.gar_delta != 0.0 for r in records):
                n_redist_groups += 1
            for i, rec in zip(idxs, records):
                rewards[i] = rec.final_reward
        if stats_path:
            with open(stats_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "t": round(time.time(), 3),
                    "n_groups": len(groups),
                    "n_success_groups": n_success_groups,
                    "n_redist_groups": n_redist_groups,
                    "n_rollouts": len(completions),
                    "n_success": n_success,
                }) + "\n")
        return rewards

    return reward_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-GPU GRPO training")
    parser.add_argument("--config", default=str(ROOT / "configs" / "e2_binary_grpo.yaml"))
    parser.add_argument("--data", default=str(ROOT / "artifacts" / "datasets" / "train.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "checkpoints" / "e2"))
    parser.add_argument("--max-steps", type=int, default=0, help="Override train.max_steps")
    parser.add_argument("--seed", type=int, default=0, help="Override train.seed (0 = keep config)")
    parser.add_argument("--smoke", action="store_true", help="Tiny run to verify wiring")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = load_jsonl(Path(args.data))
    _r = cfg.get("reward", {})
    JUDGE_BLEND_CFG["enabled"] = bool(_r.get("judge_enabled", False))
    JUDGE_BLEND_CFG["weight"] = float(_r.get("judge_weight", 0.5))
    pipeline = build_pipeline(cfg)
    task_lookup = {r["task_id"]: r for r in rows}

    prompts = [r["prompt"] for r in rows]
    print(f"[data] prompts={len(prompts)} mode={cfg['reward']['mode']}")

    if args.smoke:
        demo_rows = rows[:3]
        completions = ["The answer is 3", "42", "def add(a,b):\n    return a+b\n"]
        task_ids = [r["task_id"] for r in demo_rows]
        fn = make_reward_fn(pipeline, task_lookup)
        out = fn([r["prompt"] for r in demo_rows], completions, task_id=task_ids)
        for r, s in zip(demo_rows, out):
            print(f"  {r['task_id']}: reward={s}")
        print("[smoke] reward_fn ok")
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    model_name = cfg["model_name"]
    # Resolve relative paths against project root; respect config, do not override.
    if not Path(model_name).is_absolute() and (ROOT / model_name).exists():
        model_name = str(ROOT / model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if cfg.get("compute_dtype") == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto",
        use_safetensors=True,
    )

    lora_cfg = LoraConfig(
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"].get("dropout", 0.0)),
        target_modules=list(cfg["lora"]["target_modules"]),
        task_type="CAUSAL_LM",
    )

    g = cfg["generation"]
    t = cfg["train"]
    max_steps = args.max_steps or int(t.get("max_steps", 300))
    seed = args.seed or int(t.get("seed", 42))

    train_ds = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": r["prompt"]}],
                "task_id": r["task_id"],
            }
            for r in rows
        ]
    )

    grpo_args = GRPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=int(t.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 8)),
        learning_rate=float(t.get("learning_rate", 1e-5)),
        max_steps=max_steps,
        max_completion_length=int(g.get("completion_max_tokens", 256)),
        num_generations=int(g.get("num_generations", 2)),
        temperature=float(g.get("temperature", 0.8)),
        top_p=float(g.get("top_p", 0.95)),
        epsilon_high=float(g.get("epsilon_high", 0.2)),
        gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
        beta=float(t.get("beta", 0.0)),
        save_steps=int(t.get("save_steps", 100)),
        logging_steps=1,
        bf16=cfg.get("compute_dtype") == "bfloat16",
        seed=seed,
        mask_truncated_completions=True,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward_fn(pipeline, task_lookup),
        args=grpo_args,
        train_dataset=train_ds,
        peft_config=lora_cfg,
    )
    trainer.train()
    trainer.save_model(args.output)
    print(f"[done] saved to {args.output}")


if __name__ == "__main__":
    main()
