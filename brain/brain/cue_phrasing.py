"""Turn a plan's cue token into a question a VLM can actually answer.

THE PROBLEM. `transition_cue` is passed to the VLM verbatim inside
``Is '{cue}' visible in this image?``. When the generator writes a bare landmark that
reads fine — ``Is 'a traffic cone' visible`` — but the generator also writes formula
tokens, and then the model is asked

    Is 'Detect(BusStandLeft)' visible in this image?

which requires it to parse our notation before it can look at the picture. Worse, the
spatial half of that token — the robot's LEFT — is the part the mission turns on, and
buried in CamelCase it is the part most likely to be ignored.

This module is the normalisation step in between. It is ROS-free and pure so it can be
tested directly; `brain_controller` calls it and nothing else changes.

WHAT IT DOES NOT DO. It does not invent perception. A spatial cue only works because the
VLM can see left from right in a forward image; phrasing it well is necessary, not
sufficient. And it deliberately refuses maneuver macros — ``Bearing(Left) completed`` is a
statement about what the robot DID, not about what is in frame, and asking a camera to
confirm it is a category error that would answer NO forever.
"""
from __future__ import annotations

import re

__all__ = ["cue_question", "is_perceptual"]

#: Trailing words that describe WHERE, not WHAT. Split off and re-attached as a phrase so
#: the spatial constraint survives into the question instead of hiding inside CamelCase.
_SPATIAL = {
    "left": "on your left", "onleft": "on your left", "toleft": "on your left",
    "right": "on your right", "onright": "on your right", "toright": "on your right",
    "ahead": "ahead of you", "infront": "ahead of you", "front": "ahead of you",
    "behind": "behind you", "above": "above you", "overhead": "above you",
    "either side": "on both sides of you", "bothsides": "on both sides of you",
}

#: Tokens that name a maneuver or a plan state rather than something visible. Asking a
#: camera about these is a category error; the executor knows the answer already.
_NOT_PERCEPTUAL = re.compile(r"\b(bearing|completed|traverse|arrived|advance)\b", re.I)

_WRAPPER = re.compile(r"^\s*(?:\\?text\{)?(Detect|Sees?|Observe)\s*\(\s*(.+?)\s*\)\s*$",
                      re.I | re.DOTALL)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_VOWEL = re.compile(r"^[aeiou]", re.I)


def is_perceptual(cue: str) -> bool:
    """False for maneuver macros — things a camera cannot confirm."""
    return not _NOT_PERCEPTUAL.search(cue or "")


def _words(token: str) -> str:
    t = re.sub(r"[\\{}$]", "", token or "")
    t = _CAMEL.sub(" ", t)
    t = re.sub(r"[_\-]+", " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def cue_question(cue: str) -> str:
    """The noun phrase to ask about, with any spatial part spelled out.

    Plain English is passed through untouched — the generator often writes a perfectly
    good phrase already, and rewriting it would be a second chance to get it wrong.
    """
    raw = (cue or "").strip()
    if not raw:
        return ""
    m = _WRAPPER.match(raw)
    if not m:
        # already prose; only strip a stray macro wrapper if one slipped in
        return raw
    body = _words(m.group(2))
    if not body:
        return raw

    # peel a trailing spatial word off the landmark
    where = ""
    parts = body.split()
    for n in (2, 1):                       # "both sides" before "sides"
        if len(parts) > n:
            tail = "".join(parts[-n:])
            if tail in _SPATIAL:
                where = _SPATIAL[tail]
                parts = parts[:-n]
                break
    body = " ".join(parts)
    if not body:
        return raw

    article = "" if body.split()[0] in ("the", "a", "an") else ("an " if _VOWEL.match(body) else "a ")
    return f"{article}{body}" + (f" {where}" if where else "")


def scoped_cue_question(cue: str) -> str:
    """The cue as a question SCOPED TO THE NEXT INTERSECTION, for the approach query.

    Asked ~15 m before a junction, "is there an X in the NEAREST intersection ahead?"
    discriminates a cue at THIS junction from one at the next. The unscoped "can you see
    an X" cannot: it gives the same answer for a cone lying on the road, a cone at this
    junction, and a cone at the one after.

    This is deliberately the weakest form that suffices: ordinal ("at which intersection,
    1 or 2?") and relational ("BEFORE, AT or AFTER?") phrasings do not reliably attribute
    a prop to the second junction, even a large one plainly in frame. Per-frame lookahead
    is therefore not assumed, so anything beyond "this junction, yes or no" is built by
    remembering junctions as they are passed -- see JunctionCueLedger.

    Returns "" when the cue names no object (a place-only or bearing cue), which is the
    caller's signal that there is nothing to ask on approach.
    """
    raw = (cue or "").strip()
    # A Bearing(...) cue is answered from the TRAJECTORY, not the camera -- it must never
    # become a visual question. Without this it produced "Is there Bearing(Right)
    # completed in the NEAREST intersection ahead", which a model will cheerfully answer.
    if re.search(r"\bbearing\s*\(", raw, re.I):
        return ""
    phrase = cue_question(cue)
    if not phrase:
        return ""
    # STRIP A PLACE CLAUSE THE CUE ALREADY CARRIES. The generator often writes
    # "a traffic cone is in the intersection"; scoping that verbatim yields
    # "Is there a traffic cone is in the intersection in the NEAREST intersection ahead",
    # which is ungrammatical and asks two questions at once. Keep the object, drop the
    # place -- this function supplies the place.
    phrase = re.sub(r"\s+(is|are)\s+(present\s+)?(in|at|on)\s+the\s+"
                    r"(nearest\s+|next\s+)?(intersection|junction|crossroads?|four-way)\b.*$",
                    "", phrase, flags=re.I).strip()
    phrase = re.sub(r"\s+(in|at)\s+the\s+(nearest\s+|next\s+)?"
                    r"(intersection|junction|crossroads?|four-way)\b.*$",
                    "", phrase, flags=re.I).strip()
    phrase = re.sub(r"\s+(is|are)\s+(present|visible)\s*$", "", phrase, flags=re.I).strip()
    if not phrase:
        return ""
    low = phrase.lower()
    # A place-only cue has nothing to scope: "are you at an intersection" asked ABOUT the
    # next intersection is a tautology, and asking it would latch a yes at every junction.
    if any(w in low for w in ("intersection", "junction", "crossroad", "four-way")) \
            and not any(w in low for w in ("cone", "bench", "sign", "light", "bus",
                                           "fountain", "kiosk", "barrier", "bin",
                                           "container", "machine", "bale", "mailbox",
                                           "advertis", "billboard")):
        return ""
    # TWO SHAPES, because a cue is not always a noun phrase. `blocked` is free prose by
    # design ("the way ahead is blocked"), and forcing it into "Is there ..." produces
    # "Is there the way ahead is blocked in the NEAREST intersection", which a model will
    # answer anyway -- wrongly, and with no sign that the question was malformed.
    if re.search(r"\b(is|are|has|have|can)\b", low):
        return (f"At the NEAREST intersection ahead of the robot, {phrase}? "
                f"Answer NO if this is true of an intersection further away, or of "
                f"somewhere that is not an intersection.")
    # bare noun -> give it an article so the question reads as English
    if not re.match(r"^(a|an|the)\b", low):
        phrase = ("an " if low[0] in "aeiou" else "a ") + phrase
    return (f"Is there {phrase} in the NEAREST intersection ahead of the robot? "
            f"Answer NO if {phrase} is visible but belongs to an intersection "
            f"further away, or is not at an intersection at all.")
