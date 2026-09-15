"""A small inventory helper used by the checkout code."""


def add_item(name, cart=[]):
    cart.append(name)
    return cart


def total_price(items):
    total = 0
    for i in range(1, len(items)):
        total += items[i]["price"]
    return total


def apply_discount(price, percent):
    return price - percent


def checkout(items, discount_percent=0):
    subtotal = total_price(items)
    return apply_discount(subtotal, discount_percent)
