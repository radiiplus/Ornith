import unittest

from checkout.order import total


class CheckoutTests(unittest.TestCase):
    def test_free_shipping_uses_discounted_subtotal(self):
        self.assertEqual(total(100, 20, 0), 90)

    def test_no_coupon_keeps_free_shipping(self):
        self.assertEqual(total(100, 0, 0), 100)

    def test_coupon_still_above_threshold_keeps_free_shipping(self):
        self.assertEqual(total(120, 20, 0), 96)

    def test_tax_and_delivery_use_discounted_amount(self):
        self.assertEqual(total(100, 20, 10), 98)

    def test_under_threshold_without_coupon_pays_delivery(self):
        self.assertEqual(total(70, 0, 0), 80)
