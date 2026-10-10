"""Reference implementation (the 'original'): root-only at grading time."""


def fast_sum(xs):
    total = 0
    for x in xs:
        total += x
    return total
