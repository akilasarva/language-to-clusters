"""The two cue-timing concerns, as worlds you can run rather than arguments.

THE TWO CONCERNS:

  (1) "turn at the intersection with the cone" — the map has a cone but no intersection,
      then later an intersection with no cone. The earlier cone sighting MUST be dropped,
      or the robot turns early at the intersection that has no cone.

  (2) the query only happens once the robot is AT the intersection, but the cue was
      visible BEFORE it — by arrival the cue may be out of frame. The sighting should be
      counted and then removed if the joint requirement is not met within some
      time/distance.

THEY ARE THE SAME KNOB IN OPPOSITE DIRECTIONS. Both are about how long a cone sighting
stays valid while the robot waits for the OTHER half of the conjunction (being at a
junction):

    window too SHORT -> concern (2) fails: the cue is forgotten before arrival
    window too LONG  -> concern (1) fails: a stale cone fires at the wrong junction

So there is no "fix concern 1" and "fix concern 2" separately. There is one parameter,
and this file checks whether ANY value of it satisfies both — and on what margin. That
question is why these are worlds and a sweep rather than two assertions.

WHY OFFLINE. Driven runs are stochastic (the same plan and seed diverge in position over
time) and junction turns do not always complete, so a driven pair cannot separate "the
cue logic counted wrong" from "the controller missed the turn". These worlds are
deterministic and take milliseconds, so they say whether the logic is right; CARLA then
says whether it is sufficient.

Distances are metres of travel, which is what StepProgress already uses for its spatial
de-bounce (`resight_metres`). Town05 junction regions are roughly 35-50 m across, and a
prop can sit ~28 m from where the robot enters the region; the worlds below are scaled
to that geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from carla_gt_bridge import cue_answers as CA


@dataclass(frozen=True)
class Tick:
    """One poll: where the robot is, what the camera shows, how far it has come."""
    label: str            # "path" | "junction" -- from the region classifier
    cone_visible: bool    # what a VLM answers about the IMAGE alone
    travelled: float      # cumulative metres


# --------------------------------------------------------------------------- #
# The three policies, so "is this a problem?" is answered by comparison
# --------------------------------------------------------------------------- #

def _fire_tick(world, policy, *, window_m=12.0, needed=1):
    """Index of the tick at which the step would fire, or None if it never does.

    `policy`:
      "vlm_today"  what cue_source=vlm does NOW: the VLM is asked only about the object
                   ("can you see a traffic cone"), with no place predicate ANDed in.
      "gt_today"   what gt_cue_node does NOW: `cone AND at_intersection`, evaluated per
                   tick with no memory -- a sighting that is not simultaneous is lost.
      "windowed"   the proposal: the conjunction may be satisfied by a cone seen up to
                   `window_m` metres ago, and the evidence expires after that.
    """
    seen_at = None                      # travelled when the cone was last seen
    for i, t in enumerate(world):
        at_junction = t.label == "junction"
        if policy == "vlm_today":
            found = t.cone_visible
        elif policy == "gt_today":
            found = t.cone_visible and at_junction
        elif policy == "windowed":
            if t.cone_visible:
                seen_at = t.travelled
            fresh = (seen_at is not None
                     and t.travelled - seen_at <= window_m)
            found = fresh and at_junction
        else:                                          # pragma: no cover
            raise ValueError(policy)
        # Route through the REAL vocabulary so this cannot drift from the runtime:
        # a test that reimplements the matching proves only that it can be reimplemented.
        answered = CA.resolve("a traffic cone is in the intersection",
                              at_junction=at_junction, cone=found)
        if answered:
            return i
    return None


# --------------------------------------------------------------------------- #
# The worlds
# --------------------------------------------------------------------------- #
# Concern (1): cone with no junction, THEN a junction with no cone, THEN both.
# The middle segment is the trap: that junction must NOT be turned at.
CONCERN_1 = (
    [Tick("path", True, d) for d in (0, 4, 8, 12)]          # lone cone on a path
    + [Tick("path", False, d) for d in (16, 20, 24, 28)]    # cone behind us
    + [Tick("junction", False, d) for d in (32, 36, 40)]    # DECOY junction, no cone
    + [Tick("path", False, d) for d in (44, 48, 52, 56)]
    + [Tick("junction", True, d) for d in (60, 64, 68)]     # the real one
)
# DERIVED, NOT HAND-COUNTED. A hand-counted index that is off by a few ticks produces a
# failure that looks like "no window satisfies both worlds", i.e. a finding about the
# system rather than a bug in the test. Indices that describe a literal must be read off
# the literal.
_C1_JUNCTION_TICKS = [i for i, t in enumerate(CONCERN_1) if t.label == "junction"]
CONCERN_1_DECOY = [i for i in _C1_JUNCTION_TICKS if not CONCERN_1[i].cone_visible]
CONCERN_1_TARGET = min(i for i in _C1_JUNCTION_TICKS if CONCERN_1[i].cone_visible)

# Concern (2): the cone is seen on approach and is OUT OF FRAME by arrival -- which is
# what a ~9 m verge offset does (such a prop is >100 deg off-axis at 5 m).
CONCERN_2 = (
    [Tick("path", False, d) for d in (0, 4)]
    + [Tick("path", True, d) for d in (8, 12, 16)]          # visible on approach
    + [Tick("junction", False, d) for d in (20, 24, 28)]    # arrived; out of frame
)
CONCERN_2_TARGET = min(i for i, t in enumerate(CONCERN_2) if t.label == "junction")

# THE BAND, derived from the two worlds' geometry so the sweep cannot assert a stale
# number. Concern (2) needs the window to reach back from arrival to the last sighting;
# concern (1) needs it NOT to reach from the lone cone forward to the decoy junction.
NEED_AT_LEAST = (min(t.travelled for t in CONCERN_2 if t.label == "junction")
                 - max(t.travelled for t in CONCERN_2 if t.cone_visible))       # 4 m
MUST_BE_UNDER = (min(t.travelled for t in CONCERN_1 if t.label == "junction")
                 - max(t.travelled for t in CONCERN_1[:8] if t.cone_visible))   # 20 m


def test_concern_1_the_vlm_path_as_it_stands_turns_at_the_wrong_junction():
    """The object-only question fires on the LONE cone, before any junction exists.

    This is concern (1). It is latent only while brain polls cues in CHECKING_CUE/DECIDING,
    both entered at the goal region -- so the robot never looks while passing a lone cone.
    It becomes real the moment cues are polled during NAVIGATING, which is exactly what
    concern (2) requires.
    """
    fired = _fire_tick(CONCERN_1, "vlm_today")
    assert fired is not None and fired < 8, (
        "expected the object-only cue to fire on the lone path cone (tick <8), "
        f"got {fired}")


def test_concern_1_the_conjunction_alone_already_fixes_the_early_turn():
    """`cone AND at_junction` never fires at the decoy, with or without memory.

    So concern (1) does NOT need the window -- it needs the conjunction, which
    gt_cue_node already does and the VLM path does not. Worth separating: the window is
    concern (2)'s requirement, and it is what puts concern (1) back at risk.
    """
    for policy in ("gt_today", "windowed"):
        fired = _fire_tick(CONCERN_1, policy)
        assert fired not in CONCERN_1_DECOY, (
            f"{policy} turned at the DECOY junction (tick {fired}) -- the cone was 20 m "
            f"behind")
        assert fired == CONCERN_1_TARGET, (
            f"{policy} should fire at the real junction (tick {CONCERN_1_TARGET}), "
            f"got {fired}")


def test_concern_2_todays_conjunction_never_fires_when_the_cue_leaves_frame():
    """The cost of requiring the two halves SIMULTANEOUSLY.

    The cone was plainly visible 8 m earlier; by arrival it is off-axis and the VLM
    answers NO -- correctly, about the image. With no memory the step never completes and
    times out, which presents as a planning or control failure rather than a cue one.
    """
    assert _fire_tick(CONCERN_2, "gt_today") is None
    assert _fire_tick(CONCERN_2, "vlm_today") is not None, (
        "sanity: the object-only policy does see it, just at the wrong place")


def test_concern_2_a_window_recovers_it():
    fired = _fire_tick(CONCERN_2, "windowed", window_m=float(NEED_AT_LEAST))
    assert fired == CONCERN_2_TARGET, (
        f"expected the junction tick {CONCERN_2_TARGET}, got {fired}")


@pytest.mark.parametrize("window_m", [0.0, 4.0, 8.0, 12.0, 20.0, 30.0, 60.0])
def test_the_window_has_a_working_range_and_both_ends_are_real(window_m):
    """THE GAUGE. One knob, two failure directions -- sweep it and see the band.

    Concern (2) needs the window to reach back from the junction to where the cone was
    last visible (12 m in CONCERN_2). Concern (1) needs it NOT to reach from the lone
    cone forward to the decoy junction (20 m in CONCERN_1). So the band is 12-20 m, and
    it exists only because those two distances are ordered that way -- it is a property
    of the ROUTE, not a universal constant. A cone that goes out of frame further back,
    or a decoy junction closer behind, closes it.
    """
    ok_2 = _fire_tick(CONCERN_2, "windowed", window_m=window_m) == CONCERN_2_TARGET
    fired_1 = _fire_tick(CONCERN_1, "windowed", window_m=window_m)
    ok_1 = fired_1 == CONCERN_1_TARGET
    if window_m < NEED_AT_LEAST:
        assert not ok_2, f"window {window_m} m should be too short for concern 2"
    if window_m >= MUST_BE_UNDER:
        assert not ok_1, (
            f"window {window_m} m should be long enough to fire at the DECOY "
            f"(fired at tick {fired_1})")
    if NEED_AT_LEAST <= window_m < MUST_BE_UNDER:
        assert ok_1 and ok_2, f"window {window_m} m should satisfy BOTH"


def test_the_working_band_is_narrow_enough_to_be_worth_stating():
    """Report the band rather than asserting a magic number somewhere else."""
    good = [w for w in range(0, 61)
            if _fire_tick(CONCERN_2, "windowed", window_m=float(w)) == CONCERN_2_TARGET
            and _fire_tick(CONCERN_1, "windowed", window_m=float(w)) == CONCERN_1_TARGET]
    assert good, "no window satisfies both worlds -- the conjunction needs a richer cue"
    assert (min(good), max(good)) == (NEED_AT_LEAST, MUST_BE_UNDER - 1), (
        f"working band {min(good)}-{max(good)} m does not match the geometry "
        f"{NEED_AT_LEAST}-{MUST_BE_UNDER - 1} m; the policy changed")
    # StepProgress.resight_metres is 12.0 -- the only distance constant in the cue
    # machinery, and it lands INSIDE this band (4-19 m), which is why reusing it as the
    # window is defensible. But the margin is asymmetric: 8 m of slack below, 7 m above,
    # on worlds whose spacing was chosen here. On a route where the decoy junction is
    # closer behind the lone cone than 12 m, 12.0 fires early.
    assert min(good) <= 12.0 < max(good), (
        "resight_metres=12.0 has fallen outside the working band -- do not reuse it")
