# Project continuation

Read `docs/new-instance-handoff.md` and `README.md` before changing the agent or
running a model. Ornith-1.5-9B IQ2_M is the target model. The GGUF and the pinned,
locally patched llama.cpp checkout are restored by `scripts/bootstrap_new_instance.sh`.

The current tests and benchmark artifacts are the evidence baseline. Keep model loads
inside a CPU, memory, swap, and deadline scope; do not run two large model copies at
once. On a larger host, measure 4, 6, 8, and 12 generation threads before choosing a
quota. Preserve the distinction between emitted reasoning text and hidden model state.

The repository is public. Do not commit tokens, private data, model weights, build
output, or SQLite audit databases. New agent sessions should continue from the handoff
and rerun the relevant tests; chat history is not embedded in Git.
