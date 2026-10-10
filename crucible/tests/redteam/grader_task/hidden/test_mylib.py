"""Hidden tests (grader-owned)."""
import mylib


def test_empty():
    assert mylib.fast_sum([]) == 0


def test_small():
    assert mylib.fast_sum([1, 2, 3]) == 6


def test_negative():
    assert mylib.fast_sum([-5, 5, -1]) == -1


def test_large():
    assert mylib.fast_sum(list(range(1000))) == 499500
