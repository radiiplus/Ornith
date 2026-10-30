# Area: assertions in optimized Python

Pattern: Python removes `assert` statements when optimization is enabled, so
an assertion used as an access or validation boundary may disappear under
`python -O`. Relevant keywords: optimized, bytecode, assert, boundary,
validation, authorization.

Verified local fixture: `examples/bytecode-boundary-demo/` and the executable
checker in `scripts/evaluate_ornith_bytecode.py`. The probe compares normal
and optimized bytecode, then checks the observed behavior. See
`docs/ornith-bytecode-analysis.md` for the saved local results and their
limits. Reproduce the checker before applying this pattern to a new source.

This note does not imply that every removed assertion was a security check.
