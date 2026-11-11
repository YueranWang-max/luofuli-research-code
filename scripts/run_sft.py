"""MOPD-lite stage 2: SFT the 4B student on verified teacher solutions.

TRL SFTTrainer with completion-only loss, LoRA matching the RL setup so the
distilled checkpoint is a drop-in init for the v2 GRPO matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "artifacts" / "distill" / "sft_data.jsonl"))
    ap.add_argument("--base", default="models/Qwen3-4B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "bench_ckpt" / "distilled_init"))
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[data] {len(rows)} teacher-verified pairs")
    ds = Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "completion": [{"role": "assistant", "content": r["completion"]}],
        }
        for r in rows
    ])

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="auto", use_safetensors=True
    )

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    cfg = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=args.epochs,
        max_length=1024,
        completion_only_loss=True,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=42,
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"[done] saved to {args.out}")


if __name__ == "__main__":
    main()
