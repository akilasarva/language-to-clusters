#!/usr/bin/env python3
"""Three different questions that "did the plan pass?" was collapsing into one.

Validity is the weakest of the three questions below, and the only one that can be gamed
by doing less: a linear plan that drops a mission's contingency passes every validator,
so a metric that scores validity alone can rank the plan that discards the mission highest.

  1. VALID       Is it a well-formed plan? Schema, legal transitions, answerable cues,
                 no dead steps. Purely structural: it never reads the English.

  2. GROUNDED    Do the names in it refer to anything real? Every mode must resolve to
                 cluster ids in the environment's taxonomy. A plan naming `grass` where
                 the map has `Space: Open` is valid and ungrounded -- it will not
                 materialize, and if it did the monitor's accept set would be empty.

  3. FAITHFUL    Does it say what the MISSION said? A contingency kept, a prohibition
                 carried, an ordinal preserved. This is the only one that reads the
                 English, and the only one that can catch a plan that quietly does
                 something simpler than it was asked to.

The three are independent, and the interesting cells are the disagreements. A plan can
be valid and unfaithful (the linear arm on every branching mission), faithful and
ungrounded (the untaxed arm inventing sensible names for a vocabulary it was never
given), or valid and grounded and still unfaithful.

WHAT THE EXPECTATIONS ARE, AND ARE NOT. `expectations_from(text)` reads the English for
surface markers -- if/unless, never/without, second/2nd. That is a mechanical PROXY for
an answer key, deliberately conservative: it only asserts an axis when the mission
plainly states one, so it under-claims rather than over-claims. It is not a substitute
for the reviewed key, and a mission whose contingency is implied rather than stated will
be scored as having none. Treat a FAITHFUL score as a lower bound.
"""
from __future__ import annotations

import re
from typing import Any

__all__ = ["expectations_from", "grounded", "faithful", "score_plan"]


# --------------------------------------------------------------------------- #
# 3. FAITHFUL — what the English asked for                                    #
# --------------------------------------------------------------------------- #

#: Surface markers, matched on word boundaries so "unless" does not fire on "sunless"
#: and "second" does not fire on "seconds". Conservative by construction.
_CONTINGENCY = re.compile(
    r"\b(if|unless|in case|otherwise|else|either way|whichever|"
    r"should the|in the event)\b", re.I)
_PROHIBITION = re.compile(
    r"\b(never|without (?:ever )?(?:going|entering|crossing|passing)|"
    r"do not enter|don't enter|avoid|stay (?:on|off)|keep to|"
    r"the (?:whole|entire) way)\b", re.I)
_ORDINAL = re.compile(
    r"\b(second|2nd|third|3rd|fourth|4th|two more|second-to-last)\b", re.I)


def expectations_from(text: str) -> dict[str, bool]:
    """Which axes this mission plainly states. See the module docstring on proxies."""
    return {
        "contingency": bool(_CONTINGENCY.search(text)),
        "prohibition": bool(_PROHIBITION.search(text)),
        "ordinal": bool(_ORDINAL.search(text)),
    }


def _walk(steps):
    for st in steps or []:
        yield st
        for b in (st.get("branches") or []):
            yield from _walk(b.get("sub_plan"))


def faithful(plan: dict, stl: str, text: str, *,
             leaf_paths: int = 1, discriminates: bool = False,
             captured: list | None = None) -> dict[str, Any]:
    """Per-axis: did the plan keep what the mission stated?

    `leaf_paths` / `discriminates` / `captured` are passed in rather than recomputed
    because the caller already has them and two of the three need a taxonomy to
    materialize against.
    """
    want = expectations_from(text)
    got: dict[str, bool] = {}

    if want["contingency"]:
        # BOTH: branches that exist but which no cue selects between are not a
        # contingency, they are decoration.
        got["contingency"] = leaf_paths > 1 and discriminates
    if want["prohibition"]:
        got["prohibition"] = bool(captured)
    if want["ordinal"]:
        got["ordinal"] = any(st.get("cue_ordinal") for st in _walk(plan.get("steps")))

    applicable = sorted(k for k, v in want.items() if v)
    kept = sorted(k for k in applicable if got.get(k))
    return {
        "expected": applicable,
        "kept": kept,
        # A mission stating nothing measurable is vacuously faithful. `n_expected`
        # is what distinguishes that from a mission whose demands were all met --
        # the same declared-vs-satisfied distinction ConstraintMonitor draws.
        "faithful": len(kept) == len(applicable),
        "n_expected": len(applicable),
        "per_axis": {k: bool(got.get(k)) for k in applicable},
    }


# --------------------------------------------------------------------------- #
# 2. GROUNDED — do the names refer to anything real?                          #
# --------------------------------------------------------------------------- #

def grounded(plan: dict, taxonomy) -> dict[str, Any]:
    """Fraction of the modes this plan names that resolve in the taxonomy.

    A rate rather than a boolean, because that is the measurement the ungrounded arms
    exist to produce: the interesting question is not "did it fail" but "how much of
    an invented vocabulary happens to land on the real one". Compared through
    `canonical_mode`, so `path` counts against a map spelling it `Path` -- an exact
    match would report drift that is only spelling.
    """
    named: list[str] = []
    for st in _walk(plan.get("steps")):
        for f in ("start_mode", "goal_mode", "hold_mode"):
            m = st.get(f)
            if m and m not in named:
                named.append(m)
    for f in ("forbid_modes", "require_modes"):
        for m in (plan.get(f) or []):
            if m not in named:
                named.append(m)
    if not named:
        return {"grounded": True, "rate": 1.0, "named": [], "unresolvable": []}
    bad = [m for m in named if taxonomy.canonical_mode(m) is None]
    return {
        "grounded": not bad,
        "rate": 1.0 - len(bad) / len(named),
        "named": named,
        "unresolvable": bad,
    }


# --------------------------------------------------------------------------- #
# The three together                                                          #
# --------------------------------------------------------------------------- #

def score_plan(plan: dict, stl: str, text: str, taxonomy, *,
               valid: bool, leaf_paths: int = 1, discriminates: bool = False,
               captured: list | None = None) -> dict[str, Any]:
    """VALID / GROUNDED / FAITHFUL, reported separately and never collapsed.

    Deliberately returns no single number: any weighting of these is a claim about
    which failure matters most, and it should be made explicitly by the caller.
    """
    g = grounded(plan, taxonomy)
    f = faithful(plan, stl, text, leaf_paths=leaf_paths,
                 discriminates=discriminates, captured=captured)
    return {
        "valid": bool(valid),
        "grounded": g["grounded"], "grounded_rate": round(g["rate"], 4),
        "unresolvable_modes": g["unresolvable"],
        "faithful": f["faithful"], "faithful_axes": f["per_axis"],
        "n_expected": f["n_expected"],
        # The cell that makes the point: a well-formed plan that does something
        # simpler than it was asked to. Nothing we had could see this.
        "valid_but_unfaithful": bool(valid) and f["n_expected"] > 0 and not f["faithful"],
    }
