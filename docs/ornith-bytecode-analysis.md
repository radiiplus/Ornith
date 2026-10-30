# Ornith bytecode blind-spot probe

The fixture at `examples/bytecode-boundary-demo/report_reader.py` appears to check that
a resolved report path remains inside a workspace. Its check is an `assert`. Python
removes assertions under `-O`, so the compiled function still calls `read_text` but no
longer calls `is_relative_to`. A local subprocess check confirms the effect: normal
execution rejects `../outside.txt`, while optimized execution reads the marker from a
file outside the workspace. The fixture uses temporary files and no network access.

`scripts/evaluate_ornith_bytecode.py` compares three prompts against the same checked
fixture. It asks Ornith to classify the failure and explain it using source alone,
source plus optimized bytecode, or a deterministic disassembler comparison of normal
and optimized bytecode. The comparison reports the removed `is_relative_to` method and
the remaining `read_text` method. The model does not receive the subprocess result.

| Prompt | Observed classification | Checked result |
| --- | --- | --- |
| Source only | Unsafe, but attributed to generic path traversal/injection | Failed |
| Source and optimized bytecode | Same incorrect cause | Failed |
| Source, optimized bytecode, and method delta | Assert stripped under `python -O`; workspace escape follows | Passed |

The original responses and timings are in `artifacts/ornith-bytecode-probe.json` and
`artifacts/ornith-bytecode-guided.json`. The latter was initially marked failed because
the scorer omitted the word "stripped" from its evidence synonyms. The raw responses
were preserved, and `scripts/rescore_ornith_bytecode.py` wrote separate `-rescored.json`
artifacts. The guided pass took 157 seconds. All runs used Ornith-1.5-9B IQ2_M within
the existing CPU, RAM, swap, and deadline envelope.

## Evidence And Feedback

The `--mode feedback` probe asks for the effective bytecode guard and predicts normal
and `python -O` behavior before grading the cause. A correct first answer stops the
loop. Otherwise the model receives one message with the independently observed
execution outcomes and the disassembler comparison. The artifact stores first-pass
and feedback-assisted grades separately, including a format-validity flag. The model
cannot mark its own answer correct.

On the original assertion case, the first answer described the optimization issue but
was truncated JSON, so it failed. The revision gave the correct cause and both observed
outcomes in nonconforming JSON; semantic scoring passed and format scoring failed.
The saved responses are in `artifacts/ornith-bytecode-feedback.json`, with the corrected
scorer result in `artifacts/ornith-bytecode-feedback-rescored.json`.

A held-out fixture uses `if __debug__` instead of `assert`. Its guard instructions are
still present in optimized bytecode, but a constant-false jump makes them unreachable.
The tool now reports unreachable methods as well as removed methods. Ornith gave a
contradictory first answer and an incorrect revision even after feedback. The checked
subprocess behavior and bytecode control-flow observation still identify the flaw;
`artifacts/ornith-bytecode-debug-feedback-rescored.json` preserves the model failure
alongside that tool-verified finding. A safe runtime guard is also a local negative
control for the disassembler comparison. This is a two-case ladder, not a broad
generalization result.

Run the local checks with `PYTHONPATH=src python3 -m unittest
tests.test_ornith_bytecode_probe -v`. Run the full model comparison with `bash
scripts/run_ornith_bytecode_probe.sh`, or the guided pass alone with `bash
scripts/run_ornith_bytecode_probe.sh --mode delta --output
artifacts/ornith-bytecode-guided.json`.
Run the correction loop with `bash scripts/run_ornith_bytecode_probe.sh --mode feedback
--output artifacts/ornith-bytecode-feedback.json`; add `--case debug` for the held-out
variant. `scripts/rescore_ornith_bytecode.py` can replay either saved response set
without loading Ornith.

This is one intentionally vulnerable Python case. A robust static analyzer can flag
security checks written as assertions, so the result does not show that bytecode is
uniquely capable of finding this flaw. It shows that raw disassembly did not help this
Ornith configuration on its own, while a deterministic bytecode comparison supplied
the missing evidence for a correct explanation.
