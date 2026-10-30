# Verified reasoning experiment for Ornith

This experiment targets general reasoning and problem solving as well as coding. It is a data and evaluation pipeline, not a trained checkpoint or a quality claim. The target metric is correct, checkable tasks per wall-clock hour under this host's CPU and memory limits. Report scores separately for quantitative reasoning, logic, planning, and coding so an improvement in one cannot hide a regression in another.

## Method

| Ingredient | Strength | Limit |
| --- | --- | --- |
| LoRA / QLoRA | Adapts a frozen base with a small set of trainable weights and curated examples. | Training still needs activations, compute, and a trainable checkpoint. The local IQ2_M GGUF is an inference artifact, not an NF4 QLoRA checkpoint. |
| GGUF quantization | Fits inference into smaller memory budgets. | Changing metadata does not teach reasoning; low-bit weights may lose information. IQ3_M already triggered this host's pressure guard. |
| Short active context with retrieval | Lets the model inspect relevant evidence without loading an entire corpus into attention. | A missed retrieval or misleading source can damage the answer. |
| Verified feedback | Rejects wrong answers and allows a targeted retry or a curated training example. | A checker validates only what it actually tests. A plausible explanation can still be wrong. |
| Small, diverse reasoning data | Focuses limited training compute on distinct skills and failure modes. | Published gains on other models and benchmarks do not imply a gain for Ornith. |

The synthesis is **verified reasoning distillation**: curate difficult, diverse questions; check their answer or action sequence with a deterministic verifier; have the coding agent audit the short explanation against the problem and checker result; and train a text-only adapter on the accepted training split. At inference time, use a bounded context and verified feedback when a task has an executable checker. This avoids spending extra model calls on unverified self-critique. No human review gate is required for the experiment.

The mixed compiler supports four domains:

- **Quantitative:** exact rational arithmetic from a restricted expression tree; no floating-point tolerance or unrestricted `eval`.
- **Logic:** truth-table entailment of small Boolean premises, including counterexamples to invalid conclusions.
- **Planning:** action preconditions and effects are simulated; the proposed sequence must reach the goal. This checks validity, not optimality.
- **Coding:** the original file must fail an approved isolated test and the exact edit must pass it.

`scripts/prepare_verified_tasks.py` writes conversational prompt/completion JSONL and a manifest with the domain, verifier, family group, split, and content hash. The code-repair path runs inside Bubblewrap with a 512 MiB process limit and a 10-second timeout per invocation. Quantitative, logic, and planning checkers run on small, bounded structured inputs. The explanation text is *not* machine-verified; the manifest records that distinction so the agent can audit it separately. No agent audit log is exported automatically.

Ornith's published Qwen3.5 configuration has 32 text layers, full attention every fourth layer, linear attention in the others, and a separate vision branch. A text-only adapter should target inspected text-module names, not blindly select every linear layer. Keep the original tokenizer and chat template when comparing a trained adapter with a GGUF export.

## Seed and protocol

```bash
python3 scripts/prepare_verified_tasks.py \
  --input experiments/verified-reasoning/seed.jsonl \
  --output-dir artifacts/verified-reasoning-seed
```

The seed has **five training and four held-out examples** across the four domains. It exercises the compiler; it is much too small or simple to measure a reasoning gain. Family groups cannot cross splits, and exact duplicate problems across splits are rejected. The coding agent must also check semantic overlap, provenance, licensing, and the explanations before actual training.

1. Freeze a substantial unseen evaluation set with distinct families in each domain. Include tasks that require multiple steps, distractors, and incomplete or conflicting evidence. Report answer accuracy, valid-plan rate, verified code repair rate, format errors, timeouts, tokens, wall time, and memory pressure by domain.
2. Benchmark the current Ornith GGUF and agent under fixed CPU, RAM, swap, context, thinking mode, and generation limits. The existing 1/3 short coding quiz and two repair fixtures are only exploratory baselines.
3. Change one variable at a time: retrieval/prompt method on the same GGUF, then a reviewed LoRA or QLoRA adapter trained on separate GPU hardware, then export and quantization. Compare base and adapter before quantization to separate training effects from low-bit damage.
4. Keep a candidate only if improvements repeat on unseen families without unacceptable losses in another domain, instruction following, latency, or host responsiveness. A million-token positional setting does not substitute for this evaluation and cannot provide literal million-token attention on this host.

The published Ornith card describes a dense 9B reasoning model and about 19 GB of BF16 weights. Capped local stress tests now cover both thinking and non-thinking modes; their IQ2_M results still do not isolate quantization as the cause of mistakes. No Ornith weights or GGUF metadata were changed by this experiment.

The first live multi-domain and feedback-loop results are in [the reasoning stress report](ornith-reasoning-stress.md). It records both thinking and non-thinking runs, so the original non-thinking result should no longer be read as the only local evidence.

## References

- [Ornith-1.5-9B model card](https://huggingface.co/ornith-ai/Ornith-1.5-9B) and [configuration](https://huggingface.co/ornith-ai/Ornith-1.5-9B/blob/main/config.json)
- [LoRA](https://arxiv.org/abs/2106.09685) and [QLoRA](https://arxiv.org/abs/2305.14314)
- [TRL supervised fine-tuning dataset formats](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Small Language Models Need Strong Verifiers to Self-Correct Reasoning](https://arxiv.org/abs/2404.17140)
- [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)
- [llama.cpp LoRA export](https://github.com/ggml-org/llama.cpp/blob/master/tools/export-lora/README.md)
