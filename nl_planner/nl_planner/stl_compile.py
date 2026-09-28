"""Compile an STL formula into RUNTIME MONITORS, instead of discarding it.

WHY THIS EXISTS. Otherwise the formula is generated, syntax-checked, and thrown away:
`branch_materializer` never references it, so execution runs entirely off the JSON plan
tree, and a representation nothing reads cannot help.

Most generated formulas are chains of `\\land`, `\\mathbf{F}`, `\\Phi_{...}` and negated
`Detect(...)`: "reach this mode, then see this thing, and here is the else-branch" —
which is *precisely* what the JSON plan already says. That part is a REDUNDANT ENCODING,
isomorphic to the tree, and adds no information.

WHAT IS NOT REDUNDANT, and is the whole point of compiling:

    Phi_X U psi     "stay in X until psi" — the JSON records the DESTINATION of a step
                    and never that a mode had to HOLD along the way. `path -> junction`
                    and "path -> junction without leaving the path" are the same tree.
    G(~Phi_X)       "never enter X" — a plan-level invariant with no field in the tree.

Both are strictly more than the plan can express. Both now have monitors:
`trigger_policy.InvariantMonitor` and `ConstraintMonitor`. This module is the missing
half — the bridge from the written formula to those monitors.

WHAT THIS IS NOT. Not a model checker, and not an evaluator over a trace. It is a SHAPE
RECOGNISER: it parses the formula and extracts the sub-formulas that correspond to a
monitor we can actually run. Everything it cannot place is reported rather than ignored,
so `coverage` is an honest measurement of how much of the formula does work — and the
share that is merely redundant with the plan is reported separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# --------------------------------------------------------------------------- #
# Macro -> semantic mode                                                       #
# --------------------------------------------------------------------------- #

#: A `\Phi_{...}` macro names a BEHAVIOUR; a monitor needs the MODE it happens in.
#: Traversal macros (Junc_Pass, Junc_Turn, Int_Pass, Int_Turn) are movements THROUGH a
#: junction, so they map to `junction`; the rest name the medium you are travelling in.
#: Kept here rather than in the taxonomy because it is a property of the STL vocabulary,
#: which is fixed, not of any one environment's cluster map, which is not.
#:
#: But the VALUES are pedestrian spellings: on a car map spelling its modes `Road: On`,
#: `\Phi_{Road}` would resolve to `path`, which that taxonomy does not have, and the
#: constraint would land in `unapplied`. So `_mode_of` takes an optional taxonomy and
#: tries the macro's OWN name first, so `\Phi_{Passage}` resolves directly against a map
#: spelling it `Passage` without needing an entry here.
MACRO_MODE: dict[str, str] = {
    "Path": "path", "LongPath": "path", "Road": "path", "LongRoad": "path",
    "Passage": "passage", "Bridge": "passage",
    "Edge_Follow": "along_edge", "Build_Past": "along_edge",
    "Open_Space": "open_space",
    "Junc_Pass": "junction", "Junc_Turn": "junction",
    "Int_Pass": "junction", "Int_Turn": "junction",
}


# --------------------------------------------------------------------------- #
# AST                                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class Node:
    kind: str                      # macro | pred | not | F | G | U | and | or
    name: str = ""                 # macro body, or predicate name
    arg: str = ""                  # predicate argument
    kids: list["Node"] = field(default_factory=list)

    def walk(self) -> Iterable["Node"]:
        yield self
        for k in self.kids:
            yield from k.walk()

    def __repr__(self) -> str:      # readable failures beat pretty ones
        if self.kind == "macro":
            return f"Phi[{self.name}]"
        if self.kind == "pred":
            return f"{self.name}({self.arg})"
        if self.kind in ("not", "F", "G"):
            return f"{self.kind}({self.kids[0]!r})"
        return f"({f' {self.kind} '.join(repr(k) for k in self.kids)})"


class ParseError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Tokeniser                                                                    #
# --------------------------------------------------------------------------- #

_STRIP = [
    (re.compile(r"^\s*\$\$?|\$\$?\s*$"), ""),          # display math wrappers
    (re.compile(r"\\(?:bigg?|Bigg?|left|right)\s*"), ""),
    (re.compile(r"\\text\{([^{}]*)\}"), r"\1"),        # \text{X} -> X
    (re.compile(r"\\[,;:! ]"), " "),                   # LaTeX spacing
    (re.compile(r"\\_"), "_"),                         # escaped underscore
]

_TOKEN = re.compile(r"""
      (?P<macro>\\Phi_\{(?P<mbody>[A-Za-z0-9_]+)\})
    | (?P<pred>(?P<pname>Detect|Bearing)\s*\(\s*(?P<parg>[^()]*?)\s*\))
    | (?P<op>\\land|\\wedge|\\lor|\\vee|\\lnot|\\neg|\\Box|\\square|\\Diamond)
    | (?P<temporal>\\mathbf\{(?P<tname>[FGU])\})
    | (?P<lp>\()
    | (?P<rp>\))
""", re.X)


def tokenize(stl: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Return (tokens, junk). Junk is REPORTED, never parsed.

    Trailing prose is normal in this dialect — `Bearing(Right) completed` is how the
    generator spells a topology cue, and the same wording appears in the plan's
    `transition_cue`. Feeding "completed" to the grammar as a term would make such
    formulas fail with a bogus "missing )", understating compile coverage. It is dropped
    from the stream and returned separately so `coverage` can still say what was ignored.
    """
    s = stl
    for pat, rep in _STRIP:
        s = pat.sub(rep, s)
    out: list[tuple[str, str, str]] = []
    junk: list[str] = []
    i = 0
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        m = _TOKEN.match(s, i)
        if not m:
            # Unrecognised text is skipped, but recorded so coverage can see it. A
            # tokenizer that silently ate the interesting half would make every formula
            # look fully compiled.
            j = i
            while j < len(s) and not s[j].isspace() and not _TOKEN.match(s, j):
                j += 1
            junk.append(s[i:j])
            i = max(j, i + 1)
            continue
        g = m.groupdict()
        if g["macro"]:
            out.append(("macro", g["mbody"], ""))
        elif g["pred"]:
            out.append(("pred", g["pname"], g["parg"]))
        elif g["temporal"]:
            out.append(("temporal", g["tname"], ""))
        elif g["op"]:
            op = g["op"]
            name = {"\\land": "and", "\\wedge": "and", "\\lor": "or", "\\vee": "or",
                    "\\lnot": "not", "\\neg": "not", "\\Box": "G", "\\square": "G",
                    "\\Diamond": "F"}[op]
            out.append(("temporal" if name in ("F", "G") else name, name, ""))
        elif g["lp"]:
            out.append(("lp", "(", ""))
        else:
            out.append(("rp", ")", ""))
        i = m.end()
    return out, junk


# --------------------------------------------------------------------------- #
# Parser — precedence: or < and < U < unary(not/F/G) < atom                     #
# --------------------------------------------------------------------------- #

class _P:
    def __init__(self, toks): self.t, self.i = toks, 0

    def peek(self): return self.t[self.i] if self.i < len(self.t) else ("eof", "", "")

    def eat(self):
        tok = self.peek()
        self.i += 1
        return tok

    def parse(self) -> Node:
        n = self.or_()
        if self.peek()[0] not in ("eof", "rp"):
            # Trailing tokens mean the grammar did not describe this formula. Say so
            # rather than returning a partial tree that looks complete.
            raise ParseError(f"unconsumed input at token {self.i}: {self.peek()}")
        return n

    def or_(self) -> Node:
        kids = [self.and_()]
        while self.peek()[0] == "or":
            self.eat()
            kids.append(self.and_())
        return kids[0] if len(kids) == 1 else Node("or", kids=kids)

    def and_(self) -> Node:
        kids = [self.until_()]
        while self.peek()[0] == "and":
            self.eat()
            kids.append(self.until_())
        return kids[0] if len(kids) == 1 else Node("and", kids=kids)

    def until_(self) -> Node:
        left = self.unary()
        if self.peek()[:2] == ("temporal", "U"):
            self.eat()
            return Node("U", kids=[left, self.unary()])
        return left

    def unary(self) -> Node:
        k, v, _ = self.peek()
        if k == "not":
            self.eat()
            return Node("not", kids=[self.unary()])
        if k == "temporal" and v in ("F", "G"):
            self.eat()
            return Node(v, kids=[self.unary()])
        return self.atom()

    def atom(self) -> Node:
        k, v, a = self.eat()
        if k == "macro":
            return Node("macro", name=v)
        if k == "pred":
            return Node("pred", name=v, arg=a)
        if k == "lp":
            n = self.or_()
            if self.peek()[0] != "rp":
                raise ParseError("missing )")
            self.eat()
            return n
        raise ParseError(f"unexpected {k!r} {v!r}")


def parse(stl: str) -> tuple[Node, list[str]]:
    """Return (ast, junk). Junk is the prose the grammar ignored."""
    toks, junk = tokenize(stl)
    if not toks:
        raise ParseError("empty formula")
    return _P(toks).parse(), junk


# --------------------------------------------------------------------------- #
# Compile to monitors                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class MonitorSpec:
    """What the formula asks to be MONITORED, over and above the plan tree."""

    #: `Phi_X U psi` -> hold X for the duration, terminate on psi.
    holds: list[dict] = field(default_factory=list)
    #: `G(~Phi_X)` -> never enter X.
    forbid_modes: list[str] = field(default_factory=list)
    #: `G(Phi_X)` -> never LEAVE X. The positive invariant.
    require_modes: list[str] = field(default_factory=list)
    #: `F Detect(x)` / `F Phi_X` — reach-goals. REDUNDANT with the plan tree; counted
    #: so the redundancy can be reported rather than mistaken for compiled work.
    reach: list[str] = field(default_factory=list)
    #: Sub-formulas the recogniser could not place.
    unhandled: list[str] = field(default_factory=list)

    #: `G(~Detect(X))` -- a forbidden PREDICATE rather than a mode. Recorded, not
    #: enforced: brain forbids CLUSTERS and there is no runtime for a predicate ban.
    #: Kept visible so these discards are reported rather than silently dropped.
    forbid_predicates: list[str] = field(default_factory=list)
    #: `G(Phi_A or Phi_B)` -- always in the UNION. Not distributable: alternating between
    #: A and B satisfies it and would fail G(A) and G(B) separately.
    require_any: list[list] = field(default_factory=list)
    #: `~Phi_X U psi` -- stay OUT of X until psi. The scoped mirror of hold_mode.
    #: InvariantMonitor holds a mode IN; enforcing this needs a scoped ConstraintMonitor
    #: that does not exist yet, so this is recorded and not yet executable.
    forbid_until: list[dict] = field(default_factory=list)

    @property
    def novel(self) -> int:
        """Monitors expressing something the JSON plan CANNOT."""
        return len(self.holds) + len(self.forbid_modes) + len(self.require_modes)

    def summary(self) -> dict:
        return {"holds": self.holds, "forbid_modes": self.forbid_modes,
                "require_modes": self.require_modes,
                "n_reach_redundant": len(self.reach),
                "n_novel": self.novel, "unhandled": self.unhandled}


def _mode_of(macro: str, taxonomy=None) -> str | None:
    """Semantic mode for a `\\Phi_{...}` macro, in the taxonomy's own spelling.

    Resolution order, and the order matters:

    1. **The macro's own name**, against the supplied taxonomy. `\\Phi_{Passage}` on a map
       that spells it `Passage` needs no translation table at all, and this is what lets a
       formula constrain an environment `MACRO_MODE` was never updated for.
    2. **The `MACRO_MODE` translation**, also against the taxonomy. This is what carries
       the genuine synonyms — `\\Phi_{Junc\\_Turn}` is a movement *through* a junction, not
       a mode named "Junc_Turn", and no amount of string matching would find that.
    3. `None` — and a `None` here is honest: it becomes an unapplied constraint that
       `validate_stl_modes` reports, rather than a monitor over a mode that cannot resolve.

    With ``taxonomy=None`` only ``MACRO_MODE`` is used (pedestrian spellings).
    """
    raw = macro.replace("\\_", "_")
    translated = MACRO_MODE.get(raw)
    if taxonomy is None:
        return translated
    for candidate in (raw, translated):
        if candidate:
            canon = taxonomy.canonical_mode(candidate)
            if canon is not None:
                return canon
    return None


def _describe(n: Node) -> str:
    return repr(n)


def compile_monitors(node: Node, taxonomy=None) -> MonitorSpec:
    """Recognise the shapes that correspond to a monitor we can actually run.

    Pass ``taxonomy`` to resolve macros into THAT environment's mode spellings; without
    it the pedestrian ``MACRO_MODE`` values are used verbatim, which cannot resolve car-map
    spellings.
    """
    spec = MonitorSpec()

    def _visit_G(inner: Node) -> None:
        """Everything under a Globally.

        WHY THIS IS NOT JUST TWO CASES. Accepting only `G(Phi_X)` and `G(~Phi_X)` would
        leave most generated invariants uncompiled: they commonly carry conjunctions of
        several terms and negations.

        Three identities do most of the work, and all three are sound:

            G(A and B)  ==  G(A) and G(B)         distribute, recurse
            G(A or B)   ==  "always in A or B"    a UNION requirement, not distributable
            G(~(A or B)) == G(~A) and G(~B)       de Morgan, then distribute

        `G(A or B)` is the one that cannot distribute -- "always on a path or at a
        junction" is satisfiable by alternating between them, which G(A) and G(B) would
        both reject. It is a single requirement over the union, which is exactly what
        require_modes already means when its entries share an axis.
        """
        # G(~X) -- de Morgan first so a negated disjunction becomes forbids
        if inner.kind == "not":
            body = inner.kids[0]
            if body.kind == "macro":
                mode = _mode_of(body.name, taxonomy)
                if mode:
                    spec.forbid_modes.append(mode)
                    return
            if body.kind == "or":                       # ~(A or B) == ~A and ~B
                for k in body.kids:
                    _visit_G(Node("not", kids=[k]))
                return
            if body.kind == "and":                      # ~(A and B) -- NOT distributable
                spec.unhandled.append(
                    f"G(~(A and B)) is a disjunction of forbids and needs a monitor that "
                    f"can express 'not both': {body!r}")
                return
            if body.kind == "pred":
                # A forbidden PREDICATE, not a mode. Recorded rather than dropped: brain
                # forbids CLUSTERS, so there is nothing to enforce this with today, and
                # it must not be silently discarded.
                spec.forbid_predicates.append(f"{body.name}:{body.arg}")
                return
            spec.unhandled.append(f"G over an unrecognised negation: {body!r}")
            return
        # G(Phi_X) -- never LEAVE X. Distinct from a step's `hold_mode`, which is scoped
        # to one step; this binds the whole mission including every branch.
        if inner.kind == "macro":
            mode = _mode_of(inner.name, taxonomy)
            if mode:
                spec.require_modes.append(mode)
                return
        if inner.kind == "and":                         # G(A and B) == G(A) and G(B)
            for k in inner.kids:
                _visit_G(k)
            return
        if inner.kind == "or":
            # A single requirement over the UNION. Only expressible when every disjunct
            # is a mode -- "always on a path or at a junction". Mixed macro/predicate
            # disjunctions have no runtime form.
            modes = [_mode_of(k.name, taxonomy) for k in inner.kids if k.kind == "macro"]
            if len(modes) == len(inner.kids) and all(modes):
                spec.require_any.append(list(modes))
                return
            spec.unhandled.append(f"G over a mixed disjunction: {inner!r}")
            return
        if inner.kind == "U":
            # G(A U B) -- the Until already binds globally; hand it to the U handler
            visit(inner)
            return
        spec.unhandled.append(f"G over an unrecognised body: {inner!r}")
        visit(inner, under_G=True)

    def visit(n: Node, under_G: bool = False) -> None:
        if n.kind == "U":
            left, right = n.kids
            mode = _mode_of(left.name, taxonomy) if left.kind == "macro" else None
            if mode:
                spec.holds.append({"hold_mode": mode, "until": _describe(right)})
            elif left.kind == "not" and left.kids[0].kind == "macro":
                # ~Phi_X U psi -- "stay OUT of X until psi". The scoped mirror of
                # hold_mode, and the shape "don't turn off the road until you pass the
                # cone" produces. Recorded separately because brain's InvariantMonitor
                # holds a mode IN, not OUT; enforcing this needs a scoped
                # ConstraintMonitor that does not exist yet.
                m = _mode_of(left.kids[0].name, taxonomy)
                if m:
                    spec.forbid_until.append({"forbid_mode": m, "until": _describe(right)})
                else:
                    spec.unhandled.append(f"U over an unresolvable negated mode: {left!r}")
            else:
                spec.unhandled.append(f"U with non-mode left operand: {left!r}")
            visit(right)
            return
        if n.kind == "G":
            _visit_G(n.kids[0])
            return
        if n.kind == "F":
            inner = n.kids[0]
            if inner.kind == "macro":
                spec.reach.append(f"mode:{_mode_of(inner.name, taxonomy) or inner.name}")
            elif inner.kind == "pred":
                spec.reach.append(f"{inner.name}:{inner.arg}")
            visit(inner)
            return
        for k in n.kids:
            visit(k, under_G)

    visit(node)
    # dedupe, order-stable
    seen: set = set()
    spec.forbid_modes = [m for m in spec.forbid_modes
                         if not (m in seen or seen.add(m))]
    seen2: set = set()
    spec.require_modes = [m for m in spec.require_modes
                          if not (m in seen2 or seen2.add(m))]
    return spec


def coverage(stl: str) -> dict:
    """How much of this formula becomes a monitor, and how much merely restates the plan.

    Returns counts rather than a single score on purpose. One number would hide the
    distinction that matters: a formula can be 100% "understood" and still contribute
    nothing, because everything it says is already in the tree.
    """
    try:
        ast, junk = parse(stl)
    except ParseError as exc:
        return {"parsed": False, "error": str(exc), "n_novel": 0,
                "n_reach_redundant": 0, "unhandled": [str(exc)], "junk": []}
    spec = compile_monitors(ast)
    n_nodes = sum(1 for _ in ast.walk())
    return {"parsed": True, "n_nodes": n_nodes, "junk": junk, **spec.summary()}


__all__ = ["MACRO_MODE", "MonitorSpec", "Node", "ParseError", "compile_monitors",
           "coverage", "parse", "tokenize"]


def validate_stl_modes(stl: str, taxonomy) -> list[str]:
    """Return every mode the FORMULA names that the taxonomy does not have.

    The formula-side counterpart of ``validate_plan_modes``. A constraint naming
    a mode outside the taxonomy
    compiles to a monitor whose accept set resolves to nothing, so it can never
    fire — an unenforceable constraint that is indistinguishable, at runtime,
    from having stated no constraint at all. It is strictly worse than a
    rejected plan, because it looks like the rule is being enforced.

    Without this check, formula constraints naming e.g. ``passage`` against a
    taxonomy holding only ``junction`` and ``path`` would be counted as captured
    yet be dead at runtime, while the plan's mode-validated ``forbid_modes``
    field cannot make the same mistake -- an asymmetry between the two outlets.
    """
    try:
        ast, _ = parse(stl)
        spec = compile_monitors(ast)
    except Exception:                                              # noqa: BLE001
        return []                                                  # syntax gate's job
    named = set(spec.forbid_modes) | set(spec.require_modes)
    named |= {h["hold_mode"] for h in spec.holds if h.get("hold_mode")}

    # Delegate the spelling rule to the taxonomy so this gate and `resolve` cannot
    # disagree (otherwise a formula naming `path` against a `Path` taxonomy could
    # pass the gate and then crash during materialization).
    return sorted(m for m in named if m and taxonomy.canonical_mode(m) is None)
