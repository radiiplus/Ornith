# Representation-confusion method

A representation-confusion candidate exists when two parts of a workflow use
different values for what should be the same decision boundary. Search for the
value at the input, authorization or policy check, cache key, transformation,
and final output. State exactly which representation each stage uses. A match
is only a hypothesis until an approved test or executable checker reproduces
the mismatch and a minimal change makes that test pass.

The local verified examples below establish narrow patterns, not a general
vulnerability claim about another codebase:

- Tenant reports: authorization uses tenant identity but the cache key uses
  only user identity. See `areas-tenant-cache.md`.
- Discounted checkout: the delivery threshold uses the original subtotal
  while the charged amount uses the discounted subtotal. See
  `areas-discount-shipping.md`.
- Optimized Python: an assertion used as a boundary check is removed under
  `python -O`. See `areas-optimized-assertions.md`.

For a new objective, compare these patterns with the actual source and tests.
If no local pattern matches, consult the research library or explicitly
supplied sources. Do not treat a keyword match as proof of the cause.
