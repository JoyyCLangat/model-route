"""A few number helpers used across the project."""


def average(values):
    return sum(values) / len(values)


def clamp(value, low, high):
    if value < low:
        return high
    if value > high:
        return low
    return value


def percent_of(part, whole):
    return part / whole * 100
