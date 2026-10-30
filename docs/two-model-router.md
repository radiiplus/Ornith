# Two-model router experiment

The current experiment assigns objective restatement and planning to
VibeThinker-1.5B Q4_K_M, and bounded action selection and repair proposals to
DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M. The router changes control flow. A
separate, disabled VibeThinker LoRA experiment is documented in
[the training pilot](lora-pilot.md). The old Ornith IQ2_M results remain a
historical baseline.

## Protocol

1. The reasoning model restates the original objective with subgoals,
   assumptions, and a definition of done. The gate checks its JSON shape and
   overlap with key terms in the original request. After one failed correction,
   execution stops and the task is marked failed.
2. The reasoning model returns one to three ordered steps. Every step has an
   objective, a `done_when` statement, and a machine-checkable kind:
   `tests_pass`, `evidence_found`, or `checked_answer`. JSON Schema constrains
   each handoff; the Python validator also checks lengths, allowed values, and
   approved paths.
3. The execution model chooses one action at a time: `inspect`, `test`,
   `repair`, `answer`, or `advance` for a supplied action graph. It cannot
   supply arbitrary shell commands. Reads are
   limited to approved files. Tests use the caller's approved command through
   Bubblewrap. Repair delegates to the existing staged `EngineeringAgent`,
   which checks exact edits and reruns approved tests.
   For verified workflow tasks, the router exposes currently available graph
   actions, applies the model's chosen action, and records the resulting
   state. It prefers actions that preserve already achieved goals when other
   progress is possible. The original independent plan checker still judges
   the complete action history.
4. A step passes only with actual file evidence, an approved test result, a
   verified staged repair, or an external deterministic answer checker, as
   specified by the step. Failed steps get bounded retries. After exhaustion,
   the reasoning model can revise the remaining plan once, recording the
   broken assumption, change, and evidence in SQLite. All requests, outcomes,
   revisions, and conclusions are logged in the existing `actions` table.
5. The conclusion names the original objective and reports completed steps or
   the unresolved evidence. A successful status does not mean the model's
   rationale was independently verified beyond the stated checker.

`parked_observations` is a new SQLite table for a short observation, its
context, a relevance tag, and a reason to revisit it. New tasks retrieve up
to five overlapping notes for planning. Model-generated notes are not treated
as verified facts. The default `corpus/hunt` directory is built from the
repository's verified fixtures; `--corpus-dir` can replace it. The router scans `areas*` and
`*representation*` files for keyword overlaps before research. Unmatched
objectives search the existing research library; `--research-seed-url` enables
a bounded crawl through the same research code used by `research_cli`.
The local corpus is deliberately small. It provides tested patterns, not
separate unseen vulnerability-analysis tasks. The tenant-isolation fixture is
the executable security-relevant case in the current comparison.

## Runtime

The production runner loads only one model process at a time. Switching roles
stops the current llama.cpp server before loading the next, and records model
load and request times. A systemd scope caps the whole run at one CPU (100%
quota) on this two-vCPU host,
4 GiB soft/6 GiB hard RAM, and 19 GiB swap, with a 30-minute deadline. The
host-pressure monitor can stop the model earlier. Model files are pinned by
revision and SHA-256 in `scripts/bootstrap_new_instance.sh` and excluded from
Git. The router uses 2048-token context, 192-token default outputs, 384 for
planning, and 320 for plan revision. These are experimental settings, not
measured optima. Exact-answer plans have one step and one answer action; the
router keeps the original objective as the execution scope and logs any
model-generated narrowing as `plan_scope_correction`. Verifier-specific output
instructions distinguish arithmetic, logic, and action-order answers.
`--prompt-style role-aware` keeps an explicit system message for VibeThinker
but places DeepSeek's JSON instruction in its user message. This is an
optional comparison informed by the embedded chat templates: VibeThinker's
GGUF inserts a different system instruction when none is supplied. The
one-thread choice is
based on the local 128-token throughput comparison in `artifacts/speed-*.json`;
two threads under the same one-core quota were slower for both models.

```bash
bash scripts/run_dual_pipeline.sh --suite
bash scripts/run_dual_pipeline.sh --objective 'Fix add so the test passes' \
  --repo examples/repair-demo --allow-path calc.py \
  --test 'python3 -m unittest discover -s tests'
```

The suite uses four verified challenge rows, three held-out reasoning seed
rows, the existing `examples/repair-demo`, and the tested `tenant_reports`
fixture. Standalone baselines use `run_ornith_stress.sh` with `--model-path`,
`--model-name`, `--constrained`, and `--include-seed`; repair and investigation
use their corresponding evaluators. Results are saved under `artifacts/`.
Scores from these fixtures are exploratory because the seed is tiny and some
cases have appeared in earlier experiments. Compare pass rates and latency on
these exact cases, then test unseen cases before claiming a general gain.

## Measured standalone baseline

All seven reasoning cases used Q4_K_M, the same exact arithmetic, truth-table,
or plan simulator checker, one generation thread, and a 50% CPU quota. The
corrected JSON Schema request was used; each case had one attempt. The figures
below are local measurements, not published model-card claims.

| Model | Exact passes | Sum of request time | Median request | Mean decode rate | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| VibeThinker-1.5B | 1/7 | 133.61 s | 18.61 s | 2.74 tokens/s | 2083.8 MiB |
| DeepSeek-R1-Distill-Qwen-1.5B | 4/7 | 89.68 s | 10.29 s | 2.94 tokens/s | 2083.9 MiB |

Artifacts: `artifacts/vibethinker-standalone-constrained-v2.json` and
`artifacts/deepseek-standalone-constrained-v2.json`. A prior unconstrained
160-token run scored 0/4 for each model because reasoning text exhausted the
budget before a checkable answer. The server-side JSON Schema had originally
been sent in the wrong shape; those pre-fix constrained attempts are invalid
as a comparison and are excluded from the table.

The actual `examples/repair-demo` baseline at 50% CPU passed for VibeThinker
in 87.63 s after two proposals. DeepSeek initially failed in 87.84 s because
it invented edit paths. Constraining the schema path to the approved file made
the same case pass in 49.01 s. The `tenant_reports` fixture failed for both:
VibeThinker took 146.42 s and DeepSeek 110.12 s. Its exact-edit safety check
correctly rejected invalid source fragments and wrong-file edits. See the
corresponding `artifacts/*standalone-repair-demo*.json` and
`artifacts/*standalone-tenant-reports.json` records.

For decode speed, `scripts/run_small_model_speed.sh` generated the same 128-token
sample in each condition under the same memory limits. This is a speed probe,
not a quality test:

| Model | 50% CPU, 1 thread | 100% CPU, 1 thread | 100% CPU, 2 threads |
| --- | ---: | ---: | ---: |
| VibeThinker | 3.03 tokens/s | 6.75 tokens/s | 3.89 tokens/s |
| DeepSeek | 3.06 tokens/s | 6.64 tokens/s | 3.84 tokens/s |

The one-core quota left the second vCPU for the host. Each speed probe peaked
near 2084 MiB RSS and stayed within the memory guard. Two threads under one
core increased contention and reduced decode speed. Ornith's earlier local
generation was about 0.5 tokens/s at a 50% quota and about 4 GiB peak RSS;
its published stress cases and these seven cases are not an identical score
set, so only the hardware and latency scale are comparable.

## Model source and reasoning data

Both checkpoints descend from Qwen2.5-Math-1.5B. The
[VibeThinker model card](https://huggingface.co/WeiboAI/VibeThinker-1.5B)
describes diversity-oriented supervised fine-tuning followed by reinforcement
learning, recommends long sampled outputs for competitive math and code, and
warns that other tasks are outside its intended experimental use. Its
[source repository](https://github.com/WeiboAI/VibeThinker) publishes evaluation
programs and reports, but does not contain a released full training corpus or
end-to-end training implementation. The
[DeepSeek model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
says its small checkpoint was fine-tuned from DeepSeek-R1-generated reasoning
data; it recommends user-only prompts and active reasoning. That training data
is not published with the checkpoint. The current fast path disables exposed
reasoning and uses deterministic constrained JSON, so its host scores should
not be treated as a reproduction of either author's published benchmarks.
The [VibeThinker technical report](https://arxiv.org/html/2511.06221v1)
states that training used mostly public datasets plus some proprietary
synthetic data, and reports up to 40,000 output tokens and repeated sampling
for its evaluations. The optional role-aware prompt mode tests one part of
the recommended framing without altering weights or silently changing the
baseline.

## Combined fast-path baseline

The first complete nine-case run used the default system-message prompt style,
one CPU core, Q4_K_M GGUF models, one generation thread, and fresh SQLite
state. It passed 1/9 cases (`repair-demo`), generated 4,594 tokens in 880.32 s,
and achieved 5.22 effective tokens/s including model swaps, prompt processing,
and tests. Peak process RSS was 2083.9 MiB. Three plan revisions fired.
The seven reasoning cases passed 0/7 and `tenant_reports` failed its staged
repair checks. The full trace and per-case times are in
`artifacts/dual-model-combined-fast-final.json`.
This run also exposed an evaluation handoff defect: logic premises and the
action graph for plan tasks were omitted from the combined prompts. The
result is a diagnostic baseline, not a fair final score for those task types.
The corrected router supplies those inputs while keeping arithmetic expected
expressions hidden from the models. The six-fact implication case then passed
under both prompt styles (74.61 s default; 74.95 s role-aware).

## Corrected combined comparison

The complete corrected nine-case run used the same case set, fresh SQLite
state, one-core quota, and one-thread Q4 models. It passed 4/9 in 1,077.17 s:
the six-fact implication, both action workflows, and `repair-demo`. It
generated 5,516 tokens, or 5.12 effective tokens/s including prompts, model
swaps, and staged tests; peak model RSS was 2,084.7 MiB. Two plan revisions
fired. The seven reasoning cases passed 3/7. `tenant_reports` still failed its
staged repair check. See `artifacts/dual-model-combined-final.json` for each
case's latency, token count, and outcome.

| Evaluation | Exact reasoning | Repair demo | Tenant isolation | CPU quota |
| --- | ---: | --- | --- | ---: |
| VibeThinker alone | 1/7 | passed, 87.63 s | failed | 50% |
| DeepSeek alone | 4/7 | passed, 49.01 s | failed | 50% |
| Corrected router | 3/7 | passed, 86.91 s | failed | 100% |

The combined workflow has extra objective, plan, and verification calls. Its
effective rate is not a decode-rate comparison with the single-model runs;
the one-prompt decode probe at 100% CPU measured 6.75 tokens/s for VibeThinker
and 6.64 for DeepSeek. The combined suite is more capable than its first
diagnostic run but remains slower end to end and less accurate on these
reasoning fixtures than DeepSeek alone. Ornith's earlier stress tasks differ,
so its roughly 0.5 tokens/s at 50% CPU is a speed reference, not a comparable
pass rate.

The corrected run exposed two remaining model-level failures: VibeThinker
restated the converse-logic question as an unsupported requirement, which the
objective gate rejected, and DeepSeek proposed invalid source fragments for
tenant repair, which staging rejected. One counting-case plan revision hit
its 320-token ceiling and yielded incomplete JSON. The revision schema now
limits its size and treats malformed revision output as unresolved work; a
focused unit test covers this fix. A targeted post-fix counting rerun reached
`needs_review` after one valid revision in 119.54 s, without a correct answer
(`artifacts/dual-committee-revision-fix-v2.json`). The saved full-run artifact
predates that last revision safeguard, but its pass count is unaffected.

The standalone exact-answer baselines above used a 50% CPU quota and passed
1/7 for VibeThinker and 4/7 for DeepSeek. The combined run used a 100% quota,
so its latency and token-rate differences cannot be credited solely to the
router. The same-prompt speed probe at 100% measured 6.75 and 6.64 decode
tokens/s for the models independently. The combined architecture is currently
slower end to end and less accurate on these verified reasoning tasks. Its
most visible failures were an overstrict question gate, missing verifier
inputs, VibeThinker narrowing the requested scope, DeepSeek giving prose or
wrong values to exact checkers, and invalid staged source edits. This
baseline remains unchanged as evidence.

## Structural experiment boundary

The local GGUF files are dense Qwen2 checkpoints with 28 blocks, 1536-wide
embeddings, and 339 tensors each, mostly Q4_K. The compiled llama.cpp CPU
backend already uses `-march=native` for the host's AVX-512 processor.
Its vendored `examples/training/README.md` describes full fine-tuning as a
work-in-progress FP32 path requiring roughly 24 GiB for a 1B model; it cannot
train these Q4 files within the 6 GiB hard cap. Removing arbitrary quantized
weights would still run the dense kernels and would have no principled
accuracy target. Controlled inference probes and single-layer ablations can
measure whether there is useful headroom before considering a separately
trained structured-pruning checkpoint.

A reversible forward-pass ablation skipped Qwen2 block 14 in DeepSeek. On the
128-token speed probe it reached 6.81 tokens/s versus 6.64 baseline, but its
output changed. On the same seven exact-checker cases, it passed 2/7 versus
the unmodified checkpoint's 4/7. The hook was removed and the normal binary
rebuilt; a post-restore probe reproduced the original output SHA-256. See
`artifacts/speed-execution-100pct-1t-ablate-layer14.json`,
`artifacts/deepseek-layer14-constrained.json`, and
`artifacts/speed-execution-100pct-1t-restored.json`. N-gram speculation kept
the output hash but slowed decode to 5.91 tokens/s for DeepSeek and 6.49 for
VibeThinker; disabling flash attention slowed DeepSeek to 6.49 tokens/s and
changed its output. None of these changes is enabled by default.

## Current limits

The objective gate is lexical; a correct paraphrase may be rejected, and a
copy of the original wording can pass without deeper understanding. A
`tests_pass` step establishes only that the supplied tests pass, not that the
full security property holds. Model-proposed `evidence_found` patterns can be
weak, so inspect the saved evidence. The router does not modify weights,
resume an interrupted generation, or execute unrestricted terminal commands.
