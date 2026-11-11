# External-family validation protocol: MiniCPM5-2B

**Status:** planned and outcome-independent; separate from the frozen R3 confirmatory study.  
**Decision date:** 21 October 2026, before any R3 confirmation-set evaluation (0/20 evaluation endpoints complete).  
**Purpose:** test whether the central reward-design comparison transfers beyond the Qwen family without changing the R3 estimand, multiplicity family, or frozen artifacts.

## Scientific question

Does `centered_budget` change strict contract-valid mathematical accuracy and token cost relative to the binary-reward control when the same GRPO protocol is applied to an independently developed compact model family?

MiniCPM5-2B is selected because it is an open-weight dense model from OpenBMB, has a compact parameter scale relevant to consumer hardware, and uses a standard causal-language-model interface. Model recency is not treated as evidence of superiority. The extension addresses model-family transfer; it does not estimate a pure architecture, parameter-count, or release-date effect.

## Fixed comparison set

- Model: `openbmb/MiniCPM5-2B`, BF16 final post-trained checkpoint.
- Arms: `binary` and `centered_budget` only.
- Seeds: 42, 123, and 2026.
- Training endpoints: 6.
- Evaluation endpoints: the untrained base plus all 6 adapters, for 7 endpoints.
- Training data, group size, LoRA target policy, optimizer, number of steps, completion limit, reward implementation, strict scoring contract, and logging fields match R3 unless an architecture incompatibility requires an amendment before any formal run.
- Confirmation data: the frozen R3 MATH and GSM8K sets, 300 items each.
- Prompt and decoding: compact prompt, greedy decoding, 1,024-token maximum, with 256/512 prefixes used only after the same prefix-consistency gate.

`grs_only` is not repeated in this extension. It remains a mechanism comparison in R3; the external-family extension tests the paper's central method-versus-control claim with the smallest sufficient additional matrix.

## Environment and identity gate

MiniCPM5-2B currently requires a newer Transformers stack than frozen R3 (`transformers==4.57.6`). It must therefore run in an isolated environment. The R3 environment, manifests, checkpoints, and dependency versions must remain unchanged.

Before formal execution:

1. Pin the exact upstream revision and record the model-card URL, license, configuration, tokenizer/chat template, generation configuration, and hashes of every weight shard.
2. Record Python, CUDA, PyTorch, Transformers, TRL, PEFT, Accelerate, Datasets, and NumPy versions.
3. Verify LoRA target discovery, padding/EOS behavior, chat-template thinking mode, response extraction, strict answer parsing, checkpoint save/resume, and deterministic greedy prefix consistency.
4. Run one two-step engineering smoke outside the formal evidence directories. The smoke tests interfaces and memory only; its accuracy and reward values are not publication evidence.
5. Freeze the extension manifest before the first formal training endpoint.

An engineering incompatibility may change implementation details only through a dated amendment made before formal outcome inspection. It may not selectively change one arm or seed.

## Estimands and analysis

The primary extension estimand is the paired difference `centered_budget - binary` in strict contract-valid accuracy, separately for MATH and GSM8K at the 1,024-token endpoint. Token cost per correct result, EOS/truncation, valid-number accuracy, output-format failure, reward dispersion, and shorter prefixes are secondary outcomes.

- Pair by item and seed.
- Report all three seed-specific effects and their mean; do not select the best seed.
- Use the R3 paired item-level procedure conditional on the observed seeds, with an explicitly labeled crossed seed-item bootstrap sensitivity interval.
- Correct the two primary dataset tests with Holm within this external extension.
- Do not pool MiniCPM and Qwen items or seeds as independent replications.
- Analyze the extension regardless of direction. Negative, null, failed, and missing endpoints remain visible.

## Interpretation boundary

Agreement with both Qwen models supports transfer to one additional independent model family under this bounded LoRA/100-step setting. Disagreement identifies a model-family boundary and narrows the paper's conclusion. Neither outcome establishes universal robustness. The extension remains secondary and cannot retroactively alter R3's primary hypotheses or replace a missing R3 endpoint.

## Execution order

1. Complete all frozen R3 training and evaluation endpoints.
2. Complete R3 integrity checks without using their outcome to cancel this extension.
3. Prepare and freeze the isolated MiniCPM environment and manifest.
4. Run the engineering smoke.
5. Run all six formal training endpoints in a fixed seeded order.
6. Run all seven evaluation endpoints.
7. Audit completeness, reconcile configurations and hashes, and then integrate the extension into the manuscript as secondary external validation.

