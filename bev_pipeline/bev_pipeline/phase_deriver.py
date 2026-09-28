"""Map-free, sensor-grounded phase & progress from the cluster stream.

Turns a per-frame cluster stream (what the classifier publishes on
``/predicted_cluster``) into approach / in / exit phases and topological plan
progress — WITHOUT metric grounding or a map. Progress is the ORDERED SEQUENCE
of sensed region-transitions matched against a plan, not distance-to-go.

Phases:
  in       — the contiguous run of a REGION cluster (robustly sensed: current
             cluster IS the region).
  exit     — the falling edge (region -> transit), robustly sensed.
  approach — soft, from the FORWARD sensing horizon: either ``ahead_open`` fires
             (region geometry opening ahead while still bounded here) or a
             ``Detect`` cue is seen ahead. Falls back to a short stream-adjacency
             window (topological, NOT odometry distance) when no forward signal
             is available. Approach is preparatory, never required for progress.

Plan matching applies the subsumption hierarchy / DEFAULTING: an un-targeted
region (right type but the required cue absent) is passed through as transit
("default down to road"), so the plan only advances at the cued target.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set

REGION_DEFAULT = ("junction", "passage")
TRANSIT_DEFAULT = ("path", "open_space", "along_edge")


@dataclass
class Run:
    start: int
    end: int              # inclusive
    rtype: str
    cues: Set[str] = field(default_factory=set)   # cues detected anywhere in the run
    matched_step: Optional[int] = None            # plan step index, or None if defaulted


def segment_runs(stream: Sequence[str], region_types=REGION_DEFAULT,
                 min_len: int = 1, max_gap: int = 1) -> List[Run]:
    """Contiguous runs of a region cluster, debounced (drop < min_len, bridge <= max_gap)."""
    runs: List[Run] = []
    i, n = 0, len(stream)
    while i < n:
        if stream[i] in region_types:
            rt = stream[i]
            j = i
            gap = 0
            last = i
            while j + 1 < n:
                if stream[j + 1] == rt:
                    j += 1; last = j; gap = 0
                elif stream[j + 1] in region_types and stream[j + 1] != rt:
                    break
                else:
                    gap += 1
                    if gap > max_gap:
                        break
                    j += 1
            if last - i + 1 >= min_len:
                runs.append(Run(i, last, rt))
            i = last + 1
        else:
            i += 1
    return runs


def derive_phases(stream: Sequence[str], region_types=REGION_DEFAULT,
                  transit_types=TRANSIT_DEFAULT, ahead_open: Optional[Sequence] = None,
                  detects: Optional[Sequence[Set[str]]] = None,
                  approach_win: int = 2, exit_win: int = 2,
                  min_len: int = 1, max_gap: int = 1):
    """Return per-frame (region_type|None, phase|None) and the list of Runs.

    ``ahead_open[i]`` truthy => region sensed opening ahead at frame i (grounds
    approach). ``detects[i]`` = set of cues detected at frame i (a cue ahead also
    grounds approach). Without either, approach is a short adjacency window.
    """
    n = len(stream)
    runs = segment_runs(stream, region_types, min_len, max_gap)
    phase = [None] * n
    rtype = [None] * n
    for r in runs:
        for k in range(r.start, r.end + 1):
            phase[k] = "in"; rtype[k] = r.rtype
            if detects is not None:
                r.cues |= set(detects[k])
        # approach: transit frames before onset
        k = r.start - 1
        steps = 0
        while k >= 0 and stream[k] in transit_types and steps < max(approach_win, 1) * 4:
            grounded = ((ahead_open is not None and ahead_open[k]) or
                        (detects is not None and detects[k]))
            within_win = (r.start - k) <= approach_win
            if grounded or within_win:
                if phase[k] is None:
                    phase[k] = "approach"; rtype[k] = r.rtype
                k -= 1; steps += 1
            else:
                break
        # exit: transit frames after
        for k in range(r.end + 1, min(n, r.end + 1 + exit_win)):
            if stream[k] in transit_types and phase[k] is None:
                phase[k] = "exit"; rtype[k] = r.rtype
            else:
                break
    return list(zip(rtype, phase)), runs


def match_plan(runs: List[Run], plan: List[dict]) -> List[Run]:
    """Topologically match region runs to an ordered plan, applying DEFAULTING.

    plan step = {"type": <region>, "cue": <optional str>}. A run satisfies the
    current step iff its rtype matches AND (no cue required OR the cue is in the
    run's detected cues). Satisfied -> advance the plan pointer (run.matched_step
    set). Unsatisfied -> defaulted (matched_step stays None; treated as transit).
    """
    ptr = 0
    for r in runs:
        if ptr >= len(plan):
            break
        step = plan[ptr]
        type_ok = (r.rtype == step["type"])
        cue_ok = ("cue" not in step or step["cue"] is None or step["cue"] in r.cues)
        if type_ok and cue_ok:
            r.matched_step = ptr
            ptr += 1
    return runs


def smooth_stream(stream: Sequence[str], window: int = 5) -> List[str]:
    """Rolling mode-filter on a symbolic cluster stream (kills per-frame flicker
    before edges are extracted). Pure temporal; no metric grounding."""
    from collections import Counter
    out: List[str] = []
    buf: List[str] = []
    for s in stream:
        buf.append(s)
        if len(buf) > window:
            buf = buf[-window:]
        c = Counter(buf); top = max(c.values())
        tied = {v for v, n in c.items() if n == top}
        if len(tied) == 1:
            out.append(next(iter(tied)))
        else:                                   # tie -> most recent among tied
            out.append(next(v for v in reversed(buf) if v in tied))
    return out


def phase_edges(stream: Sequence[str], region_types=REGION_DEFAULT,
                min_len: int = 2, max_gap: int = 1) -> List[dict]:
    """Discrete ENTER/EXIT edge events from a (smoothed) cluster stream — the
    signal the brain acts on. exit == the region->transit falling edge == "region
    no longer sensed => traversal complete". min_len debounces blips; max_gap
    bridges dropouts. Returns [{frame, kind: enter|exit, type}] in time order."""
    runs = segment_runs(stream, region_types, min_len=min_len, max_gap=max_gap)
    ev: List[dict] = []
    for r in runs:
        ev.append({"frame": r.start, "kind": "enter", "type": r.rtype})
        ev.append({"frame": r.end, "kind": "exit", "type": r.rtype})
    ev.sort(key=lambda e: (e["frame"], 0 if e["kind"] == "enter" else 1))
    return ev


def progress_summary(runs: List[Run], plan: List[dict]) -> dict:
    matched = [r for r in runs if r.matched_step is not None]
    defaulted = [r for r in runs if r.matched_step is None]
    return {
        "plan_steps": len(plan),
        "steps_reached": len(matched),
        "complete": len(matched) >= len(plan),
        "regions_seen": len(runs),
        "regions_defaulted": len(defaulted),
        "matched": [(r.start, r.end, r.rtype, r.matched_step) for r in matched],
        "defaulted": [(r.start, r.end, r.rtype) for r in defaulted],
    }
