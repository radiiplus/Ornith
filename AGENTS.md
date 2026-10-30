# Project continuation

Read `docs/new-instance-handoff.md`, `docs/two-model-router.md`, and `README.md`
before changing the agent or running a model. The current experiment uses
VibeThinker-1.5B Q4_K_M for planning and DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M
for execution. Ornith-1.5-9B IQ2_M is a historical baseline. Pinned GGUFs and
the patched llama.cpp checkout are restored by `scripts/bootstrap_new_instance.sh`.

The current tests and benchmark artifacts are the evidence baseline. Keep model loads
inside a CPU, memory, swap, and deadline scope; do not run two large model copies at
once. On this two-vCPU host, one thread under a 100% one-core quota measured
best; remeasure on a larger host. Read `corpus/hunt/` before treating a pattern
as novel. Preserve the distinction between emitted reasoning text and hidden
model state.

The repository is public. Do not commit tokens, private data, model weights, build
output, or SQLite audit databases. New agent sessions should continue from the handoff
and rerun the relevant tests; chat history is not embedded in Git.
