"""Check that every path the plan tree can take satisfies the formula: L(T) subseteq L(phi).

WHY THIS EXISTS. The argument for carrying two representations is that the tree is
the strategy and the formula is the specification, and that the pair is sound because the
one is checked against the other. `to_brain_tree` MERGES the two -- it unions the formula's
`G`/`U` constraints into the tree and the plan always wins on conflict -- which means a plan
stating a WRONG constraint is never contradicted, only supplemented. Merging is not
verification; this module is the verification.

WHY IT IS TRACTABLE HERE, when LTL model checking in general is not. Two properties of these
plans make it a direct evaluation rather than an automaton construction:

  * the tree is finite and SMALL -- typically one to a few leaf paths per mission.
    Enumerating every path is cheap.
  * each path is a FINITE word over modes. LTL_f over a finite trace evaluates by recursion
    on the formula with no Buchi automaton, no determinisation, no doubly-exponential step.

So this enumerates the tree's leaf paths, turns each into a word, and evaluates the formula on it.

THE WORD, and the one modelling choice worth arguing about. Position `i` is plan step `i`,
and the atom `Phi_X` holds there iff that step's GOAL mode is X -- a step is named by where
it ends. Position 0 is prepended for the first step's START mode, because a plan whose first
step is `path -> junction` does occupy `path` before it occupies `junction`, and without the
prepend `G Phi_path` would be false for every plan that ever reaches a junction. Predicates
(`Detect`, `Bearing`) hold at the position of the step whose `transition_cue` names them, and
at a branching step also from the branch's own cue.

WHAT A FAILURE MEANS. A rejected leaf is a path the plan permits and the formula forbids.
That is a genuine disagreement between the two artifacts about the same instruction, and it
is the thing neither representation can detect alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .stl_compile import MACRO_MODE, Node, ParseError, parse

__all__ = ["Position", "LeafPath", "Verdict", "Report",
           "leaf_paths", "word_of", "words_of", "holds", "check_tree", "check"]


# --------------------------------------------------------------------------- #
# The word                                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class Position:
    """One step of a leaf path, as the formula sees it."""

    mode: str
    cues: frozenset[str] = frozenset()      # e.g. {"detect:cone", "bearing:left"}
    label: str = ""                         # for readable failures
    #: Produced by ordinal expansion, and therefore NEVER stutter-collapsed. The two
    #: mechanisms genuinely conflict: junction splitting needs consecutive same-mode
    #: positions merged, while an ordinal alternation IS a run of same-mode positions
    #: whose whole meaning is that they are distinct. Merging them collapses
    #: "seen, not-seen, seen" back to "seen" and the count disappears.
    atomic: bool = False

    def __repr__(self) -> str:              # noqa: D105
        c = ("+" + ",".join(sorted(self.cues))) if self.cues else ""
        return f"{self.mode}{c}"


@dataclass
class LeafPath:
    """One root-to-leaf path, with the branch cue chosen at each branching step.

    `extra_cues[i]` are atoms contributed by the BRANCH taken at step `i`, not by that
    step's own `transition_cue`. Without them a decision formula is unsatisfiable by
    construction: the generator writes `Detect(Cone)` for the decision, and that atom
    lives on the branch (`vlm_cue`), never on the step.
    """

    steps: list[Mapping[str, Any]]
    choices: list[str] = field(default_factory=list)
    extra_cues: dict[int, frozenset[str]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return " / ".join(self.choices) if self.choices else "(single path)"


def _parse_cue_atoms(raw: str) -> set[str]:
    """Cue text -> predicate atoms, via the SAME vocabulary the runtime resolves cues with.

    Two spellings both have to work, because both occur in real plans:
      `"Detect(Cone) is visible"`          -> {"detect:cone"}   (explicit call)
      `"traffic cone present in intersection"` -> {"detect:cone"}   (prose)
    Branch cues are frequently prose, and reading only the explicit form would make every
    such decision unsatisfiable, turning agreeing plan/formula pairs into reported
    violations. `families_matching` is the
    drift-tested vocabulary shared with `carla_gt_bridge.cue_answers`, so resolving through
    it means this check and the robot answer the same cue the same way.
    """
    out: set[str] = set()
    low = str(raw or "").lower()
    if low and low != "default":
        try:
            from .cue_vocab import CARLA_GT_VOCAB, families_matching
            for fam in families_matching(low, CARLA_GT_VOCAB):
                out.add(f"detect:{fam}")
        except Exception:                                          # noqa: BLE001
            pass
    for kind in ("detect", "bearing"):
        idx = 0
        while True:
            i = low.find(kind + "(", idx)
            if i < 0:
                break
            j = low.find(")", i)
            if j < 0:
                break
            out.add(f"{kind}:{low[i + len(kind) + 1:j].strip()}")
            idx = j + 1
    return out


def _cues_of(step: Mapping[str, Any]) -> set[str]:
    """Predicate atoms this step's own `transition_cue` makes true."""
    return _parse_cue_atoms(str(step.get("transition_cue") or ""))


def leaf_paths(tree: Mapping[str, Any]) -> list[LeafPath]:
    """Every root-to-leaf path. A branching step contributes its own position, then one
    path per branch; a plan with no branches yields exactly one path."""
    def walk(steps: Sequence[Mapping[str, Any]], prefix: list[Mapping[str, Any]],
             choices: list[str], extra: dict[int, frozenset[str]]) -> list[LeafPath]:
        out: list[LeafPath] = []
        for k, s in enumerate(steps):
            prefix = prefix + [s]
            br = s.get("branches")
            if br:
                rest = list(steps[k + 1:])
                here = len(prefix) - 1          # index of the branching step in the word
                for b in br:
                    cue = str(b.get("vlm_cue") or "")
                    sub = list(b.get("sub_plan") or [])
                    e = dict(extra)
                    got = _parse_cue_atoms(cue)
                    if got:
                        e[here] = frozenset(got)
                    out += walk(sub + rest, prefix, choices + [cue or "default"], e)
                return out
        return [LeafPath(prefix, choices, extra)]

    return walk(list(tree.get("steps") or []), [], [], {})


def word_of(path: LeafPath, include_start: bool = False) -> list[Position]:
    """Finite word for one leaf path.

    `include_start` prepends the first step's START mode. Both anchorings are needed --
    see `words_of` for why neither alone is correct.
    """
    # NO PREPENDED START POSITION. Position i is step i's GOAL state, because that is how
    # the generator indexes the formula: it writes `Detect(Junction) and F(Bearing(Left))`
    # for a plan whose first step is `path -> junction`, i.e. the junction is true at the
    # first position, not the second. Prepending the initial `path` state makes every such
    # formula false at position 0 -- violations that are an artifact of the model, not a
    # disagreement between the artifacts.
    # ORDINAL EXPANSION. The two artifacts encode "the Nth X" completely differently:
    # the plan sets `cue_ordinal: N` on ONE step, while the formula writes the count as an
    # ALTERNATION -- seen, not-seen, seen -- across N positions. Emitting one position for
    # an ordinal step therefore fails every ordinal formula by construction.
    #
    # A step with `cue_ordinal: N` is expanded to N sightings separated by N-1 gaps. The
    # gap positions carry the step's mode with the cue ABSENT, which is what "you stopped
    # seeing it and then saw another" means and what makes `not Detect(x)` satisfiable
    # between them.
    def _expand(s, cues):
        n = s.get("cue_ordinal")
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = 0
        mode = str(s.get("goal_mode") or "")
        label = str(s.get("description") or "")[:48]
        if n < 2 or not cues:
            return [Position(mode, frozenset(cues), label)]
        out = []
        for i in range(n):
            if i:
                out.append(Position(mode, frozenset(), f"{label} (gap {i})", atomic=True))
            out.append(Position(mode, frozenset(cues), f"{label} (#{i + 1})",
                                atomic=True))
        return out

    raw: list[Position] = []
    if include_start and path.steps:
        start = str(path.steps[0].get("start_mode") or "")
        if start:
            raw.append(Position(start, frozenset(), "initial state"))
    for i, s in enumerate(path.steps):
        cues = set(_cues_of(s)) | set(path.extra_cues.get(i, ()))
        raw.extend(_expand(s, cues))

    # STUTTER-COLLAPSE, and it is sound rather than convenient. These formulas use only
    # F, G, U, not, and, or -- there is no NEXT operator anywhere in the grammar -- and
    # X-free LTL is invariant under stuttering, so merging consecutive positions that
    # share a mode cannot change the truth of any formula the parser accepts.
    #
    # It matters because the two artifacts index the same instant differently. The plan
    # splits "come up to the crossroads" and "decide there" into `path -> junction` then
    # `junction -> junction`, so the decision cue lands one position AFTER the junction
    # the formula pairs it with, and `Detect(Junction) and Detect(Cone)` is false at every
    # position despite plan and formula agreeing completely.
    word: list[Position] = []
    for p in raw:
        if word and word[-1].mode == p.mode and not p.atomic and not word[-1].atomic:
            word[-1] = Position(p.mode, word[-1].cues | p.cues,
                                word[-1].label or p.label)
        else:
            word.append(p)
    return word


def words_of(path: LeafPath) -> list[tuple[str, list[Position]]]:
    """Both anchorings of the same path, because the generator uses both conventions.

    THE AMBIGUITY IS IN THE CORPUS, NOT IN THIS FILE, and it is worth stating rather than
    resolving by fiat. Position 0 can mean two things and real formulas mean each:

      * "after step 0 completes" -- the depth-3 formula opens
        `Detect(Junction) and (Detect(Cone) and F(...))` for a plan whose first step is
        `path -> junction`, so the junction must hold at position 0.
      * "before step 0 executes" -- `Phi_{Path} and G(not Phi_{Junc} or Detect(Cone))`
        for the same shape of plan asserts the vehicle STARTS on the path.

    Picking one manufactures violations in every plan that used the other. A path is
    contained when it satisfies the formula under EITHER anchoring, and the report says
    which. A violation then means the plan disagrees with its formula under both readings.
    """
    return [("goal-anchored", word_of(path, include_start=False)),
            ("start-anchored", word_of(path, include_start=True))]


# --------------------------------------------------------------------------- #
# LTL_f evaluation over a finite word                                          #
# --------------------------------------------------------------------------- #

def _mode_for_macro(macro: str) -> str:
    """`Phi_{Road}` -> `path`. Falls back to the lowercased macro so an unknown symbol
    compares against a mode name rather than silently evaluating to False."""
    return MACRO_MODE.get(macro, macro.lower())


#: Modes are reported by the CLUSTER classifier, never by the landmark cue oracle, but
#: generators write `Detect(Junction)` for "you are at an intersection" because the English
#: says "intersection". Treating that as a landmark makes every such formula unsatisfiable.
#: It is the same distinction the runtime makes: the geometric cluster drives progress, the
#: visual cue drives decisions.
_MODE_WORDS = {"junction", "intersection", "crossroads", "crossroad", "roundabout",
               "path", "road", "street", "passage", "open_space", "along_edge"}
_MODE_ALIAS = {"intersection": "junction", "crossroads": "junction",
               "crossroad": "junction", "roundabout": "junction",
               "road": "path", "street": "path"}


def _atom(node: Node, pos: Position) -> bool:
    if node.kind == "macro":
        return pos.mode == _mode_for_macro(node.name)
    if node.kind == "pred":
        arg = node.arg.strip().lower()
        if node.name.lower() == "detect":
            if arg in _MODE_WORDS:
                return pos.mode == _MODE_ALIAS.get(arg, arg)
            # Resolve the formula's spelling to a FAMILY too, so `Detect(TrafficCone)` in
            # the formula matches `detect:cone` from a prose branch cue. Both sides go
            # through the one vocabulary; neither gets a private spelling.
            try:
                from .cue_vocab import CARLA_GT_VOCAB, families_matching
                fams = families_matching(arg, CARLA_GT_VOCAB)
            except Exception:                                      # noqa: BLE001
                fams = []
            if fams:
                return any(f"detect:{f}" in pos.cues for f in fams)
        return f"{node.name.lower()}:{arg}" in pos.cues
    raise ValueError(f"not an atom: {node.kind}")


def holds(node: Node, word: Sequence[Position], i: int = 0) -> bool:
    """Standard LTL_f semantics on a finite word. `i` past the end is vacuously False."""
    if i >= len(word):
        return False
    k = node.kind
    if k in ("macro", "pred"):
        return _atom(node, word[i])
    if k == "not":
        return not holds(node.kids[0], word, i)
    if k == "and":
        return all(holds(c, word, i) for c in node.kids)
    if k == "or":
        return any(holds(c, word, i) for c in node.kids)
    if k == "F":
        return any(holds(node.kids[0], word, j) for j in range(i, len(word)))
    if k == "G":
        return all(holds(node.kids[0], word, j) for j in range(i, len(word)))
    if k == "U":
        lhs, rhs = node.kids[0], node.kids[1]
        for j in range(i, len(word)):
            if holds(rhs, word, j):
                return all(holds(lhs, word, m) for m in range(i, j))
        return False
    raise ValueError(f"unhandled node kind {k!r}")


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class Verdict:
    path: str
    word: list[Position]
    satisfied: bool
    anchoring: str = ""          # which reading satisfied it, "" when none did


@dataclass
class Report:
    parsed: bool
    contained: bool | None                  # None when the formula did not parse
    verdicts: list[Verdict] = field(default_factory=list)
    error: str = ""
    ignored_prose: list[str] = field(default_factory=list)

    @property
    def n_paths(self) -> int:
        return len(self.verdicts)

    @property
    def n_violating(self) -> int:
        return sum(1 for v in self.verdicts if not v.satisfied)

    def summary(self) -> str:
        if not self.parsed:
            return f"formula did not parse: {self.error}"
        if self.contained:
            return f"CONTAINED: all {self.n_paths} leaf path(s) satisfy the formula"
        bad = [v for v in self.verdicts if not v.satisfied]
        lines = [f"NOT CONTAINED: {len(bad)} of {self.n_paths} leaf path(s) violate it"]
        for v in bad:
            lines.append(f"    {v.path}")
            lines.append(f"      word: {' -> '.join(repr(p) for p in v.word)}")
        return "\n".join(lines)


def check_tree(tree: Mapping[str, Any], stl: str | None = None) -> Report:
    """Check a materialised brain tree against its formula."""
    formula = stl if stl is not None else tree.get("stl_formula")
    if not formula:
        return Report(parsed=False, contained=None, error="no formula on this plan")
    try:
        ast, junk = parse(formula)
    except ParseError as exc:
        return Report(parsed=False, contained=None, error=str(exc))
    verdicts = []
    for p in leaf_paths(tree):
        chosen, ok = None, False
        for name, w in words_of(p):
            if holds(ast, w, 0):
                chosen, ok = (name, w), True
                break
        if ok:
            verdicts.append(Verdict(p.name, chosen[1], True, chosen[0]))
        else:
            verdicts.append(Verdict(p.name, word_of(p), False, ""))
    return Report(parsed=True, contained=all(v.satisfied for v in verdicts),
                  verdicts=verdicts, ignored_prose=junk)


def check(tree: Mapping[str, Any], stl: str | None = None) -> bool | None:
    """Convenience: True / False / None (unparseable)."""
    return check_tree(tree, stl).contained
