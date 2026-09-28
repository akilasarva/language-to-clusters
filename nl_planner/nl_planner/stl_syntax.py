"""Fast programmatic pre-check for STL formulas.

Why this exists
---------------
The LLM-based ``verify_syntax`` agent reads our prompt and decides whether a
formula is well-formed. In practice, the LLM gets the easy cases right but
disagrees with itself across runs on edge cases, and it costs a round-trip per
attempt. This module is a deterministic regex-based check that runs **before**
the LLM verifier and catches the common failure modes of the generator:

1. Unbalanced ``()``/``{}``.
2. Predicate arguments that are neither bare CamelCase nor a ``\\text{...}``
   wrapper (e.g. ``Detect(Open Space)``).
3. Invented ``\\Phi_{...}`` macros (e.g. ``\\Phi_{GoForward}``).
4. ASCII boolean / temporal operators (``&&``, ``||``, bare ``F``).
5. A predicate argument that names a semantic MODE of the current environment
   (e.g. ``Detect(Path)`` where ``path`` is a taxonomy mode). Only checked when
   the caller passes ``modes=``; see ``_mode_leak`` for why.

When it rejects, it returns a SPECIFIC error string that the generator's
retry feedback can use verbatim — the LLM gets a fixed-instruction-style
diff rather than another vague "syntax fail" message.

This is **not** a complete STL grammar — anything not on the bullet list
above is delegated to the LLM verifier downstream.
"""
from __future__ import annotations

import re
from typing import Iterable


# --------------------------------------------------------------------------- #
# Allow-lists (kept in lockstep with prompts/verify_syntax.md)                #
# --------------------------------------------------------------------------- #

# Canonical macro spellings. Either form of underscore (`_` or `\_`) is OK.
#: Car-on-road macros (CARLA). The originals.
_ROAD_MACROS: frozenset[str] = frozenset({
    "Int_Pass",
    "Int_Turn",
    "Bridge",
    "Build_Past",
    "Road",
    "LongRoad",
})

#: PEDESTRIAN macros (the real robot). Same shapes, domain-correct names — a
#: footpath fork is a junction, not an intersection, and a pedestrian follows an
#: edge rather than "passing a building". Which subset the generator is told to use
#: is injected per environment; the validator accepts both so one grammar serves
#: both domains and a CARLA plan does not fail validation on a pedestrian rig.
_PED_MACROS: frozenset[str] = frozenset({
    "Junc_Pass",     # straight through a fork
    "Junc_Turn",     # turn at a fork
    "Passage",       # through a flanked/narrow way (alley, bridge deck, corridor)
    "Edge_Follow",   # travel alongside one structure
    "Open_Space",    # cross an open area (plaza/lawn/field)
    "Path",          # stay on the walkway (safety/traverse constraint)
    "LongPath",      # the longer alternative route
})

_ALLOWED_MACRO_BODIES: frozenset[str] = _ROAD_MACROS | _PED_MACROS

_FORM_A_ARG = re.compile(r"^[A-Z][A-Za-z0-9]*$")
# Form B: \text{<phrase>} — letters/digits/spaces/commas/dots/dashes/colons/slashes.
# We deliberately exclude `\` so LaTeX commands cannot appear inside.
_FORM_B_ARG = re.compile(r"^\\text\{[A-Za-z0-9 ,\.\-:/]+\}$")

# Macro detection: `\Phi_{Body}` or `\Phi_{Body\_With\_Escaped\_Underscore}`.
# MATCH TO THE CLOSING BRACE, not just a clean character class. For a body such as
# `\Phi_{Junc\_Turn,\text{ordinal}=2}`, a pattern like `[A-Za-z0-9_\\]+` does not match
# a body containing a comma or braces, so the macro would never be SEEN, the CLOSED
# allow-list would never reject it, and the formula would pass `quick_syntax_check` and
# `validate_stl_modes` and then raise ParseError inside `compile_monitors` -- valid plan,
# no monitor. Matching greedily to `}` (allowing one level of nested braces, which is what
# `\text{...}` inside a body looks like) puts an invented body in front of the allow-list
# check, where it is rejected at the point it is written.
_MACRO_RE = re.compile(r"\\Phi_\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

# Predicate detection: `\text{Detect}(...)` or `Detect(...)` (likewise Bearing).
# Captures: 1=name, 2=arg.
_PREDICATE_RE = re.compile(
    r"(?:\\text\{(?P<n1>Detect|Bearing)\}|(?P<n2>Detect|Bearing))"
    r"\((?P<arg>[^()]*)\)"
)

# Forbidden ASCII operators. We match them as standalone tokens so we don't
# false-positive on letters inside CamelCase identifiers (e.g. `F` in
# `EndOfBridge` is fine; a lone `F` between tokens is not).
_FORBIDDEN_ASCII = [
    (re.compile(r"(?<![A-Za-z\\])(&&)(?!\w)"),       "&&"),
    (re.compile(r"(?<![A-Za-z\\])(\|\|)(?!\w)"),     "||"),
    (re.compile(r"(?<![A-Za-z\\])(!)(?!=)"),         "!"),
    # Bare standalone temporal letters — must be space-delimited so we don't
    # match the F in `EndOfBridge`.
    (re.compile(r"(?<![A-Za-z\\{])([UFGX])(?![A-Za-z}])"), "<bare temporal>"),
]


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def quick_syntax_check(
    stl: str,
    modes: Iterable[str] | None = None,
) -> tuple[bool, str | None]:
    """Run the deterministic pre-check.

    Args:
        stl: the formula to check.
        modes: the semantic-mode vocabulary of the CURRENT environment (e.g.
            ``taxonomy.modes_for_prompt()``). When given, a predicate argument
            that collapses onto one of these mode names is rejected — see
            ``_mode_leak``. Optional and defaults to no mode check, because the
            vocabulary is per-environment: ``Detect(OpenSpace)`` is a leak on a
            rig whose taxonomy has an ``Open Space`` mode and a perfectly good
            landmark predicate on one that does not, and ``generator.md``
            teaches it as the correct Form A rewrite.

    Returns ``(True, None)`` if no obvious violation is found (the LLM
    verifier still has a chance to reject). Returns ``(False, error)``
    with a specific, actionable error message suitable for retry feedback.
    """
    if not stl or not stl.strip():
        return False, "Rule 0: empty STL formula."

    body = _strip_display_math(stl)

    # --- Rule 1: balanced brackets ---
    if body.count("(") != body.count(")"):
        return False, (
            f"Rule 1 (balanced brackets): unmatched parenthesis — "
            f"{body.count('(')} `(` vs {body.count(')')} `)`. "
            "Recount and ensure every `(` has a matching `)`."
        )
    if body.count("{") != body.count("}"):
        return False, (
            f"Rule 1 (balanced brackets): unmatched curly brace — "
            f"{body.count('{')} `{{` vs {body.count('}')} `}}`."
        )
    if body.count("[") != body.count("]"):
        return False, (
            f"Rule 1 (balanced brackets): unmatched square bracket — "
            f"{body.count('[')} `[` vs {body.count(']')} `]`."
        )

    # --- Rule 5: no ASCII boolean ops ---
    for pat, label in _FORBIDDEN_ASCII:
        m = pat.search(body)
        if m:
            return False, (
                f"Rule 5 (boolean/temporal operators): forbidden ASCII token "
                f"`{m.group(0)}` near `...{_context(body, m.start())}...`. "
                "Use `\\land`, `\\lor`, `\\lnot`, `\\mathbf{U}`, `\\mathbf{F}` "
                "instead."
            )

    # --- Rule 2: macros are from the closed allow-list ---
    for m in _MACRO_RE.finditer(body):
        normalized = m.group(1).replace("\\_", "_")
        if normalized not in _ALLOWED_MACRO_BODIES:
            allowed = sorted(_ALLOWED_MACRO_BODIES)
            backslash = "\\"
            allowed_str = ", ".join(
                backslash + "Phi_{" + a + "}" for a in allowed
            )
            return False, (
                f"Rule 2 (allowed macros): unknown macro `{m.group(0)}`. "
                f"The macro allow-list is CLOSED: only "
                f"{allowed_str} are accepted. "
                "If your route doesn't match one of these, describe it with "
                "Detect(...) / Bearing(...) predicates plus the JSON `steps` "
                "topology — do not invent new macros."
            )

    # --- Rule 3: every Detect/Bearing argument is Form A or Form B ---
    mode_index = _collapse_modes(modes)
    for m in _PREDICATE_RE.finditer(body):
        name = m.group("n1") or m.group("n2")
        arg = m.group("arg").strip()
        if not arg:
            return False, (
                f"Rule 3 (predicate arguments): `{name}()` has an empty "
                "argument."
            )
        if _FORM_A_ARG.match(arg) or _FORM_B_ARG.match(arg):
            # Well-FORMED, but is it well-CHOSEN? A correctly spelled mode name
            # is the failure this catches; see _mode_leak.
            leaked = _mode_leak(arg, mode_index)
            if leaked is not None:
                return False, _mode_leak_error(name, m.group(0), arg, leaked, mode_index)
            continue
        suggestion = _suggest_fix(arg)
        return False, (
            f"Rule 3 (predicate arguments): argument `{arg}` (in "
            f"`{m.group(0)}`) is neither Form A (CamelCase identifier) "
            f"nor Form B (\\text{{phrase}}). "
            f"Rewrite as Form A `{suggestion}` or Form B "
            f"`\\text{{{arg}}}`. Mode names with spaces / colons / slashes "
            "must NEVER appear bare inside Detect(...) or Bearing(...) — "
            "they belong in the JSON `start_mode`/`goal_mode` fields, not "
            "in STL predicate arguments."
        )

    return True, None


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _strip_display_math(stl: str) -> str:
    """Drop a single matching outer `$$...$$` or `$...$` wrapper."""
    s = stl.strip()
    if s.startswith("$$") and s.endswith("$$") and len(s) >= 4:
        return s[2:-2]
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        return s[1:-1]
    return s


def _collapse(s: str) -> str:
    """Alphanumerics only, lower-cased. ``"Intersection: In"`` -> ``"intersectionin"``."""
    return re.sub(r"[^A-Za-z0-9]+", "", s).lower()


def _collapse_modes(modes: Iterable[str] | None) -> dict[str, str]:
    """``{collapsed: original}`` for the environment's EXTENT modes; ``{}`` if none.

    Only extent modes — the ones you travel ALONG — are indexed. Rejecting every
    mode name would kill ``Detect(Junction)``, but ``junction`` is a discrete place
    you approach and ``cue_oracle`` answers it; rejecting it would burn every retry
    on a formula that would have driven.

    You are ALWAYS inside an extent mode, so detecting one can never mark a
    transition. That is the whole principle, and it is the same ``EXTENT_MODES``
    distinction ``validate_plan_transitions`` already turns on.
    """
    if not modes:
        return {}
    from .taxonomy import is_extent_mode
    return {_collapse(m): m for m in modes if _collapse(m) and is_extent_mode(m)}


def _mode_leak(arg: str, mode_index: dict[str, str]) -> str | None:
    """Return the EXTENT mode ``arg`` names, or None.

    A ``Detect(...)`` / ``Bearing(...)`` argument is answered by a perception
    query about the WORLD — "is there a cone", "is there a stop sign". An extent
    mode is the medium you are travelling through, so the answer is always yes
    and never changes, and a step cued on one can never advance (e.g.
    ``Detect(Path)`` against the ``{junction, path}`` CARLA taxonomy). Rule 3
    cannot see it — ``Path`` is valid Form A CamelCase.

    Only extent modes; see ``_collapse_modes`` for why ``Detect(Junction)`` must
    stay legal even though ``junction`` is a mode.

    Matching is on the COLLAPSED form (alphanumerics, lower-cased) so it catches
    the mode however it is spelled: Form A ``Detect(OpenSpace)`` and Form B
    ``Detect(\\text{Open Space})`` both collapse onto the mode ``Open Space``.

    The match is EXACT, never prefix or substring. ``Detect(Intersection)`` must
    stay legal against a taxonomy holding ``Intersection: In`` — it is a live cue
    in the hand-written missions, and the oracle answers it. "Intersection" names
    a thing you can see; "Intersection: In" names a state you are in.
    """
    if not mode_index:
        return None
    inner = arg
    if inner.startswith("\\text{") and inner.endswith("}"):
        inner = inner[len("\\text{"):-1]
    return mode_index.get(_collapse(inner))


def _mode_leak_error(
    name: str, predicate: str, arg: str, leaked: str, mode_index: dict[str, str]
) -> str:
    """Retry feedback for a leaked mode name.

    Deliberately NOT phrased as a re-spelling. ``_suggest_fix("Path")`` returns
    ``"Path"`` — routing this through the Rule 3 message would tell the generator
    to rewrite `Path` as `Path`, it would re-emit the same formula, and three
    attempts would burn on a fixed point. The mode is spelled fine; it is the
    wrong KIND of thing, so the fix has to name a different construct.
    """
    return (
        f"Rule 3b (you cannot detect the medium you are in): argument `{arg}` in "
        f"`{predicate}` names {leaked!r}, a mode you travel ALONG. `{name}(...)` "
        "asks a perception question about the world, and the answer here is "
        "always yes and never changes — so a step cued on it can never advance. "
        "This is NOT a spelling problem: rewriting the argument will not fix it. "
        "Do ONE of these instead:\n"
        f"  (a) Put {leaked!r} in the step's `start_mode` / `goal_mode` (that is "
        "where modes belong) and drop the predicate.\n"
        "  (b) If the step should advance on reaching that mode with nothing to "
        "look for, give it `trigger: \"traverse\"` and NO transition cue — the "
        "cluster is the evidence.\n"
        "  (c) If there really is something to perceive, name the visible OBJECT "
        "or the discrete PLACE you are approaching instead — Detect(Cone), "
        "Detect(StopSign), Detect(Junction).\n"
        f"Travelled-along modes in this environment (never valid as predicate "
        f"arguments): {sorted(mode_index.values())}. Modes you PASS THROUGH, such "
        "as a junction, are fine — you approach those, so they can be detected."
    )


def _suggest_fix(arg: str) -> str:
    """Best-effort CamelCase suggestion for a malformed predicate argument."""
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", arg)
    parts = [p for p in cleaned.split() if p]
    if not parts:
        return "Landmark"
    return "".join(p[0].upper() + p[1:] for p in parts)


def _context(s: str, idx: int, span: int = 12) -> str:
    """Return a small substring centred on ``idx`` for error messages."""
    lo = max(0, idx - span)
    hi = min(len(s), idx + span)
    return s[lo:hi].replace("\n", " ")


__all__ = ["quick_syntax_check"]
