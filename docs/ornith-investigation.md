# Multi-file Ornith investigation probe

This probe tests whether Ornith can investigate a regression across several files, use observations to choose its next action, and propose a repair that passes tests. A successful answer on one fixture does not establish general reasoning ability.

## Cases

| Case | Files | Failing behavior | Held-back cause |
| --- | ---: | --- | --- |
| `tenant_reports` | 7 | Switching tenants returns a previously cached report | The service cache key omits tenant identity |
| `discount_shipping` | 7 | A coupon can incorrectly retain free shipping | The order passes the original subtotal to the shipping threshold |

Each case has a plausible neighboring module. The evaluator shows the objective and a failing test, then permits five model-selected actions by default (configurable from two to eight). `list` enumerates fixture files, `read` opens one allowed file, `search` finds literal text in those files, and `test` reruns the approved test command. No generated shell commands are executed. Each model turn records its hypothesis (`why`), expected observation (`expect`), chosen action, actual observation, and time.

The evaluator checks the baseline failure and a held-back oracle repair before loading the model. A final answer proposes an exact bounded replacement in a file the model inspected. The replacement is tested in an isolated copy; the model never sees the oracle edit or the correct-file flag. The final grade requires the correct file and passing tests. The case tests cover additional inputs so a single hardcoded output is insufficient.

If the proposed edit fails, the isolated test result returns as an observation and the model may revise it within the step limit. The trace reports whether the model ran tests, inspected the true file, moved from a plausible neighboring file to the cause, repeated an identical action, recorded predictions, and recovered after a failed edit. Those are observable process measures, not proof that the model internally reasoned correctly. Inspect the saved sequence for whether it changed hypotheses after contrary evidence and questioned its initial search direction.

Run each case inside the resource envelope:

```bash
bash scripts/run_ornith_investigation.sh --case tenant_reports
bash scripts/run_ornith_investigation.sh --case discount_shipping
```

The launcher caps CPU at 50%, sets 4 GiB soft and 6 GiB hard RAM limits, 19 GiB swap, and a 30-minute deadline. The evaluator refuses to load Ornith outside that scope. Results are written incrementally to `artifacts/ornith-investigation-<case>.json`, so a timeout still leaves the completed trajectory. A timeout or missing final edit is a failed case, not a successful diagnosis.

Use `--resume` with the same case and output path to continue a saved incomplete trace in a fresh model process. The evaluator rechecks the fixture and preserves the prior interruption in the artifact. This allows the bounded run to continue after a memory-guard stop without repeating completed model actions.

## Results

Results are recorded in the artifacts after each bounded run. Compare both the final test result and the ordered actions; do not infer broad reasoning ability from a single pass.
