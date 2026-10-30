from .discount import apply
from .shipping import fee
from .tax import charge


def total(subtotal, coupon_percent, tax_percent):
    discounted = apply(subtotal, coupon_percent)
    delivery = fee(subtotal)
    return discounted + delivery + charge(discounted, tax_percent)
