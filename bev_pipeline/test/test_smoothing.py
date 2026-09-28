"""Tests for mode-filter smoothing."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bev_pipeline.smoothing import mode_filter, ModeSmoother   # noqa: E402


def test_empty_returns_default():
    assert mode_filter([], 5, default=-7) == -7


def test_basic_mode():
    assert mode_filter([1, 1, 2, 1, 3], 5) == 1


def test_window_trims():
    # only last 3 considered: [2,2,3] -> 2
    assert mode_filter([1, 1, 1, 2, 2, 3], 3) == 2


def test_tie_break_most_recent():
    # 1 and 2 tie; most recent among tied is 2
    assert mode_filter([1, 2, 1, 2], 4) == 2


def test_smoother_stateful():
    s = ModeSmoother(window=3, default=0)
    assert s.push(5) == 5
    assert s.push(5) == 5
    s.push(9)
    # buffer [5,9,?]; push another 9 -> [5,9,9] wait window=3 keeps last 3
    assert s.push(9) == 9
    s.reset()
    assert s.push(2) == 2
