"""Mode-filter temporal smoothing (consolidated from the legacy inference node).

The legacy node smoothed predictions inline with an untested buffer; this pulls
that logic into a pure, tested function so both the node and offline evaluation
use identical behavior.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence


def mode_filter(history: Sequence[int], window: int, default: int = 0) -> int:
    """Return the most common value among the last ``window`` items of history.

    Ties are broken toward the value that appeared most recently among the tied
    candidates (stable, predictable). An empty history returns ``default``.
    """
    if not history:
        return default
    w = history[-window:] if window and window > 0 else list(history)
    counts = Counter(w)
    top = max(counts.values())
    tied = {v for v, c in counts.items() if c == top}
    if len(tied) == 1:
        return int(next(iter(tied)))
    # tie-break: most recent among tied
    for v in reversed(w):
        if v in tied:
            return int(v)
    return default


class ModeSmoother:
    """Stateful rolling-window mode smoother for the live node."""

    def __init__(self, window: int = 10, default: int = 0):
        self.window = window
        self.default = default
        self._buf: list = []

    def push(self, value: int) -> int:
        self._buf.append(int(value))
        if len(self._buf) > self.window:
            self._buf = self._buf[-self.window:]
        return mode_filter(self._buf, self.window, self.default)

    def reset(self):
        self._buf = []
