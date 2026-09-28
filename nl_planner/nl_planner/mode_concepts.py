"""One concept per idea, however the cluster map of the day spells it.

THE PROBLEM THIS SOLVES. The cluster maps use many distinct mode spellings, and only a
few pairs (`Path`/`path`, `Junction`/`junction`) collapse under the case-and-punctuation
rule. The rest differ by RENAME: the same idea is written `Open Space` in one map and
`Space: Open` in another, `Along Wall` vs `Wall: Along`, `Road: On` vs `On: Path`.
Without a shared layer:

  - an answer key voted across maps counts `requir:Path` and `requir:Road: On` as two
    different answers to the same question;
  - a formula naming a mode from one environment silently fails to apply in another;
  - `MACRO_MODE` would have to hardcode one map's spellings.

The fix is a concept layer, not more aliases in each map: aliases-per-map is O(maps^2) to
keep in sync and drifts again whenever a map is added.

MATCHING IS ORDER-INSENSITIVE OVER TOKENS, plus a small synonym table. `Open Space` and
`Space: Open` are the same token set; `Road: On` and `On: Path` are not, so `road` and
`path` are declared synonyms explicitly. Nothing is inferred from string similarity --
every merge below is a claim about the world that someone can check and disagree with.

RESOLUTION IS DETERMINISTIC, NOT A COIN FLIP. Several modes in one map can share a
concept -- `full_campus_1hz` has Approach:, In: and Exit: Junction, all of them junctions.
`ClusterTaxonomy.canonical_mode` breaks the tie by the order spellings are listed under
their concept below, core first, so `junction` resolves to `In: Junction` there, the same
way every time. Ordering the lists is therefore a real decision and not cosmetic.

WHAT PERCEPTION CAN ACTUALLY SEE is a separate question from what the vocabulary SHOULD
be, and the two are recorded separately (see `SENSOR`, `JUNCTION_IS_NOT_PERCEIVED`). Some
concepts are not reliably recoverable from real sensor data. That is a limitation of the
stack, not a reason to collapse the vocabulary -- the language layer still needs to say
"junction" even where the classifier often reads it as path.
"""
from __future__ import annotations

import re

__all__ = ["concept_of", "spellings_of", "CONCEPTS", "SYNONYMS", "SETTLED",
           "UNSETTLED", "SENSOR", "AXIS", "RANK", "JUNCTION_PHASE_RECALL",
           "JUNCTION_IS_NOT_PERCEIVED", "LABELS_OVERCALL"]


#: Tokens that mean the same thing across map generations. Kept tiny and explicit --
#: each line is a claim that two words name one idea in THIS domain.
SYNONYMS: dict[str, str] = {
    "intersection": "junction",
    # A PLACE NOUN NAMES A CLUSTER MACRO, and must resolve to one. The generator writes
    # the instruction's own word, so "come up to the crossroads" arrives as `crossroads`.
    # Without a mapping it is not a mode, so the step cannot target a cluster and the
    # topology channel -- which is what advances the plan and gives the MPC its goal --
    # has nothing to work with (e.g. `Detect(Crossroads)` would leave brain stuck at
    # step 0).
    #
    # Single tokens only: `_tokens` splits before substituting, so a hyphenated form like
    # "four-way" becomes {four, way} and a mapping on "way" would corrupt unrelated modes.
    # Hyphenated place words are handled on the CUE side instead.
    "crossroads": "junction",
    "crossroad": "junction",
    "roundabout": "junction",
    "road": "path",
    "street": "path",
    "avenue": "path",
    "boulevard": "path",
    "thoroughfare": "path",
    "walkway": "path",
    "on": "",            # positional filler in `Road: On`, `On: Path`
    "in": "",            # ditto in `Intersection: In`, `In: Junction`
}

#: THE DIVISION. Two orthogonal axes, five concepts.
#:
#: The vocabulary is not a flat list -- it is ENCLOSURE crossed with GROUND TOPOLOGY, and
#: the two axes are read by different sensors, which is why a routed design (each sensor
#: answers its own axis) works better than a single 5-way classifier.
#:
#:   ENCLOSURE -- how many sides are closed. Owned by the CAMERA/VLM.
#:       both sides   -> passage        corridor, bridge, covered walkway, indoors
#:       one side     -> along_edge     alongside a wall or building
#:       neither      -> open_space     lawn, plaza
#:
#:   GROUND TOPOLOGY -- what the traversable surface does. Owned by 360-degree LIDAR.
#:       ways branch  -> junction
#:       a route      -> path
#:
#: `along_edge` is a FIRST-CLASS member of the enclosure axis, not a variety of path. It
#: is the one-sided case and it sits between passage and open_space; folding it into path
#: destroys the ordering the axis is built on. A classifier frequently confusing
#: along_edge with path shows the classifier cannot separate them, not that they are the
#: same thing.
#:
#: ORDER MATTERS. When several modes in one map match a concept, the FIRST listed
#: spelling wins, so `full_campus_1hz` (Approach:/In:/Exit: Junction) resolves `junction`
#: to `In: Junction` deterministically rather than on dict order.
CONCEPTS: dict[str, tuple[str, ...]] = {
    # -- ground topology (LiDAR) --------------------------------------------- #
    "path": ("Path", "path", "Road: On", "On: Path", "LongPath", "Road", "LongRoad",
             # transitional: on a route leading to or from the structure, not inside it
             "Bridge Approach", "Bridge Exit",
             "Building Approach", "Building Exit"),
    "junction": ("Junction", "junction", "Intersection: In", "In: Junction",
                 # phases: refinements of the same place, ordered core-first
                 "Approach: Junction", "Intersection: Approach/Enter",
                 "Exit: Junction", "Intersection: Exit"),
    # -- enclosure (camera/VLM) ---------------------------------------------- #
    "passage": ("Passage", "passage", "Covered", "Corridor",
                "On Bridge", "In Building"),          # closed on BOTH sides
    "along_edge": ("Along Wall", "Wall: Along", "Edge: Along", "along_edge"),  # ONE side
    "open_space": ("Open Space", "Space: Open", "open_space"),                 # NEITHER
    # -- surface (a SEPARATE axis: what you are standing on, not where it leads) ------ #
    # A sidewalk is topologically a path; what distinguishes it is the surface. Modelling
    # it as another topology mode would make "require sidewalk" and "require path" merge
    # by union into "sidewalk OR road" -- satisfied by driving the whole route on the
    # carriageway, the mission's exact opposite. See AXIS below.
    "sidewalk": ("Sidewalk", "sidewalk", "Pavement", "Footpath"),
    # NOT called `road`: `road` is already a SYNONYM for `path` on the topology axis, so
    # its token set collapses and it cannot name a surface. `asphalt` is the surface with
    # no topology overtones.
    "asphalt": ("Asphalt", "asphalt", "Carriageway", "Paved"),
    # Grass is a SURFACE, not the enclosure concept `open_space`. Conflating them
    # ("lawn, plaza") is wrong: a paved plaza is open_space and is not grass, so "never
    # cross the grass"
    # mapped onto `forbid open_space` forbids the wrong set. Separate axes, so a
    # region can be open AND grass, or open AND paved, and the constraints intersect.
    "grass": ("Grass", "grass", "Terrain", "Lawn", "Vegetation"),
    "other": ("Other", "other"),
}

#: The design decisions, with the rationale for each ("why is a bridge a passage",
#: "why is along_edge separate").
SETTLED: tuple[tuple[str, str], ...] = (
    ("Corridor, Covered, On Bridge, In Building -> passage",
     "Closed on both sides. Also resolves a contradiction: meadow_1hz has Corridor and "
     "no path mode while full_campus_1hz has BOTH Corridor and On: Path, which cannot "
     "both hold if Corridor is a path. As a passage it is consistent with each."),
    ("along_edge stays its OWN concept",
     "It is the one-sided case on the enclosure axis, between passage (two) and "
     "open_space (zero). The camera can see a building on "
     "one side. Merging it into path would collapse a three-valued axis into two."),
    ("Bridge/Building Approach and Exit -> path",
     "Transitional: on a route leading to or from the structure, not inside it, so no "
     "lateral enclosure. Only CARLA maps use these and their clusters are ground truth."),
    ("junction phases -> junction",
     "A mission says 'at the intersection', never 'at the approach phase'. The phase is "
     "a map-level refinement; ordering keeps resolution deterministic."),
)

#: WHICH SENSOR DECIDES WHAT -- the routing. In the ROUTED design each sensor is asked
#: only the question it is good at (not the whole 5-way task):
#:
#:     LIDAR / geometry  ->  ENCLOSURE + open_space        (structure)
#:     CAMERA / VLM      ->  path | junction | along_edge  (within-open)
#:
#: `along_edge` sits on the enclosure axis but is DECIDED BY CAMERA in the final
#: pipeline: stage-1 geometry only claims it when the RF is >= 0.75 confident,
#: otherwise it falls through to the camera. So it is an
#: enclosure concept with a camera decision rule -- which is exactly why it must stay
#: its own concept rather than folding into path.
SENSOR: dict[str, str] = {
    "passage": "geom",       # enclosure: closed both sides
    "open_space": "geom",    # enclosure: closed neither side
    "along_edge": "camera",  # enclosure, but camera-decided below the 0.75 geom floor
    "path": "camera",        # within-open
    "junction": "camera",    # within-open -- but see JUNCTION_IS_NOT_PERCEIVED
}

#: Which axis each concept belongs to. Distinct from which sensor decides it: the axis
#: is what the concept MEANS, the sensor is how it is recovered, and along_edge is the
#: case where those two answers differ.
AXIS: dict[str, str] = {
    "passage": "enclosure", "along_edge": "enclosure", "open_space": "enclosure",
    "path": "topology", "junction": "topology",
    # Surface is a third, independent axis. A place has a topology AND an enclosure AND a
    # surface simultaneously, which is why constraints naming different axes must
    # INTERSECT rather than union.
    "sidewalk": "surface", "asphalt": "surface", "grass": "surface",
}

#: JUNCTION HANDLING, which is architectural rather than a tuning choice.
#:
#: Junction is NOT reliably per-frame perceivable on the real sensor data, across BEV
#: and forward-camera representations alike. Local top-down geometry does not match the
#: labels, because the labels encode ROUTE CONTEXT: a frame labelled junction is often a
#: single corridor with the branch beyond ~18m of view.
#:
#: CONSEQUENCE: junction is handled OPEN-LOOP. The plan executes the turn (heading from
#: the plan/MPC, no junction percept required) and the EXIT edge -- back on a path,
#: which is reliably perceived -- confirms traversal. Junction = plan-executed + path-confirmed,
#: NOT perceived. This is consistent with the map-free design and it is why
#: `invariant_policy` defaults to off for anything keyed on junction membership.
JUNCTION_IS_NOT_PERCEIVED = True

#: Per-phase junction recall (small sample; indicative only). Approach/in is recovered
#: far better than exit: the camera sees a junction coming, while exit is camera-blind
#: and recovered from the cluster->path edge. So the junction PHASES are load-bearing for
#: perception even though the language layer only ever says "junction" -- so phases may
#: resolve to `junction` for the LLM, and must NOT be collapsed in the accept sets.
JUNCTION_PHASE_RECALL: dict[str, float] = {"approach_in": 0.64, "exit": 0.12}

#: A known label-quality caveat: on the deployment bags junction and open_space are
#: OVER-labelled -- disagreements are mostly junction->path and open_space->path, where
#: the model's `path` is often the correct reading.
LABELS_OVERCALL = ("junction", "open_space")

#: In CARLA none of this bites: clusters come from ground truth.

#: Empty. Kept so the name still resolves for callers.
UNSETTLED: tuple = ()

_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def _tokens(mode: str) -> frozenset[str]:
    """Order-insensitive token set, synonyms applied, positional fillers dropped."""
    out = set()
    for raw in _SPLIT.split(mode or ""):
        if not raw:
            continue
        t = SYNONYMS.get(raw.lower(), raw.lower())
        if t:
            out.add(t)
    return frozenset(out)


_BY_TOKENS: dict[frozenset[str], str] = {}
#: spelling -> its rank within its concept, so the taxonomy can break a tie the same way
#: every time instead of on dict order.
RANK: dict[str, int] = {}
for _concept, _spellings in CONCEPTS.items():
    for _i, _s in enumerate(_spellings):
        _BY_TOKENS.setdefault(_tokens(_s), _concept)
        RANK.setdefault(_s, _i)


def concept_of(mode: str | None) -> str | None:
    """The concept ``mode`` names, or None if it is not one we know.

    None is a real answer, not a failure to try: an unknown mode should surface as
    unresolvable rather than be guessed into the nearest concept.
    """
    if not mode:
        return None
    return _BY_TOKENS.get(_tokens(mode))


def spellings_of(concept: str) -> tuple[str, ...]:
    return CONCEPTS.get(concept, ())
