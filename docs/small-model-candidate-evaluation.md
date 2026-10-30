# Small-model candidate evaluation

This is an independent model selection pass. The current VibeThinker/DeepSeek
router and model weights were not changed. These seven known project cases are
diagnostic, not a general intelligence benchmark.

## Conditions

Amazon Linux 2023, two Xeon Platinum 8488C vCPUs, 7.6 GiB RAM, 19 GiB swap.
One model at a time, `CPUQuota=50%`, `MemoryHigh=4G`, `MemoryMax=6G`,
`MemorySwapMax=19G`, one generation thread, 2,048 context tokens, and a
deadline. The seven reasoning cases used one attempt, temperature zero, the
same JSON Schema, and exact arithmetic, truth-table, or plan checkers. Qwen3
used `enable_thinking=false`. Repair used two bounded proposals, staged tests,
and a fresh temporary SQLite store for each run.

The seven cases cover arithmetic (3), logic (2), and full action plans (2).
The separate role probe checks three objective restatements against the
router's deterministic gate and two workflows one action at a time. The
workflow checker calculates the shortest solution independently. All prompts
come from existing verified cases or the `tenant_reports` fixture. Valid JSON
format does not imply a correct answer.

## Independent results

| Model, Q4_K_M | Exact reasoning | Valid format | Sum request time | Mean decode tokens/s | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 4/7 | 7/7 | 59.77 s | 2.72 | 2,465 MiB |
| Granite 4.0 H-1B | 4/7 | 7/7 | 72.84 s | 3.12 | 3,033 MiB |
| Gemma 3 1B IT | 1/7 | 7/7 | 83.69 s | 4.22 | 1,560 MiB |
| DeepSeek R1 Distill Qwen 1.5B | 4/7 | 7/7 | 89.68 s | 2.94 | 2,084 MiB |
| VibeThinker 1.5B | 1/7 | 7/7 | 133.61 s | 2.74 | 2,084 MiB |

Gemma generated fastest but failed six answers. Qwen generated 73 completion
tokens across these requests versus DeepSeek's 179, so its shorter request time
does not show a higher raw token rate. The Qwen timing above comes from a clean
sequential run; a diagnostic run overlapped a failed Youtu load and was excluded.

| Model | Objective fidelity | Shortest valid workflows | Repair demo | Tenant isolation |
| --- | ---: | ---: | --- | --- |
| Qwen3-1.7B | 1/3 | 2/2 | pass, 46.49 s | fail, 118.61 s |
| Granite 4.0 H-1B | 1/3 | 1/2 | pass, 45.10 s | fail, 115.42 s |
| Gemma 3 1B IT | 1/3 | 2/2 | pass, 38.42 s | fail, 105.04 s |
| DeepSeek R1 Distill Qwen 1.5B | 1/3 | 2/2 | pass, 49.01 s | fail, 110.12 s |
| VibeThinker 1.5B | 1/3 | 2/2 | pass, 87.63 s | fail, 146.42 s |

All successful stepwise workflows used zero extra actions. Qwen solved both
stepwise workflows but missed one full-plan case. Granite solved a full-plan
case yet chose a dead-end `restart` branch one action at a time. The objective
gate originally accepted `E is true` as a definition of done for `Must E be
true?`; it now requires the question to be resolved. Qwen's saved role artifact
was rescored against that corrected gate. No model exceeded 1/3 objective
fidelity, the main blocker to promoting a new planner.

Qwen3 thinking mode was probed on two non-thinking misses with a 96-token
reasoning budget and 256-token output ceiling. It still missed the committee
count (42.82 s versus 6.10 s) and workflow (45.88 s versus 8.50 s). These two
probes do not establish thinking-mode quality generally, but do not justify
enabling it for this capped fast path.

## Youtu runtime status

[Tencent's model card](https://huggingface.co/tencent/Youtu-LLM-2B) describes
Youtu-LLM-2B as a 1.96B MLA model. Its official GGUF is on that repository's
`GGUF` branch; the separate `tencent/Youtu-LLM-2B-GGUF` URL points elsewhere.
Publisher Q8_0 and F16 files loaded far enough to reach the model graph, but
the current vendored llama.cpp CPU backend rejected an MLA `MUL_MAT` node
(`fattn_mla-0` for Q4_K_M and `q_nope_absorbed-0` for F16). The capped server
never became ready, so both load artifacts under `artifacts/youtu-*` have zero
cases. This is **runtime incompatibility**, not a zero intelligence score.
Youtu needs a CPU backend that executes this graph within the same envelope.

## Model provenance

| Ignored local GGUF | Pinned source revision | SHA-256 |
| --- | --- | --- |
| `models/qwen3-1.7b-q4_k_m.gguf` | [Qwen3-1.7B BF16](https://huggingface.co/Qwen/Qwen3-1.7B), `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; converted to BF16 GGUF with vendored `convert_hf_to_gguf.py`, then `llama-quantize Q4_K_M` | `9f935d53513afc054b8a82d86fb3d9f4b9f4855a8a6d2dfc0e66cadfa2e87488` |
| `models/granite-4.0-h-1b-q4_k_m.gguf` | [IBM official GGUF](https://huggingface.co/ibm-granite/granite-4.0-h-1b-GGUF), `c2cb1972f511add21f3bae244990b8ff3a3ffb23` | `da3d737121a96f3c9a316685212376257a7f167b74380855666dd488d6af3bcb` |
| `models/gemma-3-1b-it-q4_k_m.gguf` | [ggml-org GGUF](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF), `f9c28bcd85737ffc5aef028638d3341d49869c27` | `8ccc5cd1f1b3602548715ae25a66ed73fd5dc68a210412eea643eb20eb75a135` |
| `models/youtu-llm-2b-q4_k_m.gguf` | [Tencent F16 GGUF](https://huggingface.co/tencent/Youtu-LLM-2B/tree/GGUF), `4465ef1752434c10be07a245a35e994dea2f82c6`; `llama-quantize Q4_K_M` | `47254a836943faf1637243699762de557bdf0c156429a7463f9535ebf55cef09` |

Qwen's official GGUF repository offers Q8_0; we converted the pinned BF16
checkpoint directly to Q4_K_M to avoid quantizing an already quantized file.
The GGUFs are not in Git. Reconstruct them from the pinned sources, conversion
steps above, and verify hashes before repeating the benchmark.

## Next decision

Qwen3 is the best replacement **candidate** for VibeThinker on these cases:
it matches DeepSeek's 4/7, has lower total request latency, and solves both
stepwise workflows. It has not earned automatic promotion because objective
fidelity was 1/3 and tenant repair failed. Granite increased decode rate but
also RSS without improving checked outcomes. Gemma is a low-memory action
control, not a reasoning replacement on this set. Keep the router unchanged
while building a larger unseen objective-fidelity and repair set. Then test
reasoning preservation across model handoffs with a deterministic verifier.
Invariant recognition, evidence discipline, and handoff preservation are not
separately scored by these fixtures yet.

Re-run with `scripts/run_ornith_stress.sh`, `scripts/run_candidate_roles.sh`,
and `scripts/run_standalone_agent.sh` using the paths above. Raw results are
in `artifacts/qwen3-*`, `artifacts/granite-*`, `artifacts/gemma-*`, and the
corresponding baseline artifacts.
