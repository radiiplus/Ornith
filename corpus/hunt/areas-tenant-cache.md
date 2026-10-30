# Area: tenant-scoped cache identity

Pattern: tenant reports can cross an isolation boundary when authorization is
scoped by tenant but a cached report is keyed only by user. Relevant keywords:
tenant, report, cache, key, authorization, isolation.

Verified local fixture: `examples/investigation/tenant_reports/`.
`reporting/service.py` checks `allowed(user, tenant)` but currently caches with
`key = user['id']`. `tests/test_reports.py` checks tenant switching, separation,
and rejection of unauthorized tenants. The oracle repair in
`scripts/evaluate_ornith_investigation.py` uses `(user['id'], tenant)` and is
tested in an isolated copy. Run the fixture tests with:

```bash
cd examples/investigation/tenant_reports
python3 -m unittest discover -s tests -v
```

The source fixture intentionally fails before repair. Apply no finding to a
new project until its own authorization and cache-key paths are inspected.
