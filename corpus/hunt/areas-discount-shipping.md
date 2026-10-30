# Area: pre- and post-discount totals

Pattern: checkout logic can make inconsistent shipping decisions when the fee
threshold uses a pre-discount subtotal while tax and final payment use a
discounted subtotal. Relevant keywords: discount, shipping, delivery,
threshold, subtotal, fee.

Verified local fixture: `examples/investigation/discount_shipping/`.
`checkout/order.py` passes the original subtotal to `fee` after calculating
the discounted value. `tests/test_order.py` covers orders on both sides of the
threshold and an order with no coupon. The oracle in
`scripts/evaluate_ornith_investigation.py` passes `discounted` to `fee` and is
checked in an isolated copy. Run:

```bash
cd examples/investigation/discount_shipping
python3 -m unittest discover -s tests -v
```

This is a verified regression fixture, not evidence that an unrelated
checkout has the same flaw.
