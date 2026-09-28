"""The ONE place that says which cues exist and what answers them.

WHY THIS FILE EXISTS. Two callers answer cues: the ROS node `nodes/gt_cue_node.py` and
its offline twin `scripts/missions.py::cue_oracle`. If their matching rules differ, the
offline harness certifies plans the robot then executes differently (e.g. for
`a traffic cone is in the intersection`, one answers the CONE question and the other the
JUNCTION one, which is true at every junction, so the branch fires unconditionally), and
nothing downstream can tell. Two implementations that agree only by inspection drift.

So: the vocabulary, the ORDER, and the matching rule live here, and both callers delegate.

THE ORDER IS PART OF THE SEMANTICS, not presentation. brain resolves a cue by taking the
first published key that matches, so an object key MUST precede a place key or a cue
naming both is answered about the place.

THE MATCHING RULE IS BRAIN'S, deliberately -- bidirectional substring, so a plan saying
`intersection` matches a publisher keyed on `Detect(Intersection)`. Reproducing brain's
rule rather than a stricter one is the point: a twin that matched differently would pass
cues the robot then fails on.
"""
from __future__ import annotations

__all__ = ["answer_keys", "answers_for_world", "resolve",
           "LANDMARK_SPELLINGS", "LANDMARK_ALIAS"]


#: landmark family -> the cue spellings that name it, MOST SPECIFIC FIRST.
#:
#: Adding a family here is the whole job of supporting a new landmark: the node fills in
#: the boolean, and both callers pick the vocabulary up for free. A missing family makes
#: a branch unanswerable: brain logs "no branch answer yet" and the vehicle sits at the
#: junction even though the plan itself is correct.
#:
#: CONTAINMENT IS THE CONSTRAINT, not taste. brain takes the FIRST published key whose
#: text matches the cue in either direction, so if one family's spelling were a substring
#: of another's, a cue could be answered about the wrong thing. Checked by
#: `test_no_cross_family_containment`. That is why there is no bare "stop" (it sits inside
#: both "stop sign" and "bus stop") and no bare "sign" (inside "stop sign").
LANDMARK_SPELLINGS: dict[str, tuple[str, ...]] = {
    # things placed to make a boolean true
    "cone":        ("Detect(TrafficCone)", "traffic cone", "cone"),
    # "blocked" is not a synonym for convenience: in this deployment the only thing that
    # obstructs a junction IS a cone, and a mission that says "blocked" was unanswerable.
    "blocked":     ("Detect(BlockedAhead)", "Detect(Blocked)",
                    "the way ahead is blocked", "way ahead is blocked",
                    "blocked ahead", "blocked"),
    "bench":       ("Detect(Bench)", "bench"),
    "bus_shelter": ("Detect(BusShelter)", "Detect(BusStop)", "bus shelter",
                    "bus stand", "bus stop"),
    # SPAWNABLE PROPS. Every blueprint named in the comments is present in CARLA 0.9.14's
    # library; the list lives in test/fixtures/carla_static_props.json and the families
    # are checked against it by test_every_family_substring_matches_a_real_blueprint.
    # A misspelled blueprint name yields a SpawnObject id with no error and no prop.
    #
    # Chosen for VISUAL SEPARABILITY, not coverage: a VLM asked "is there a bench" about a
    # picnic table is being asked one question, so table/bench-likes are deliberately ONE
    # family. Ten well-separated families beat twenty with three bench-likes in them.
    #
    # CONTAINMENT, again: nothing here may contain "cone", "blocked", "bench", "signal" or
    # "light" -- the last two are bare traffic_light spellings, so "streetlight" and
    # "gardenlamp" are deliberately absent even though both blueprints exist.
    "fountain":    ("Detect(Fountain)", "fountain"),                # fountain, streetfountain
    "kiosk":       ("Detect(Kiosk)", "kiosk", "food cart", "newsstand"),   # kiosk_01, foodcart
    "vending_machine": ("Detect(VendingMachine)", "vending machine"),      # vendingmachine, atm
    "trash_can":   ("Detect(TrashCan)", "trash can", "garbage can",
                    "rubbish bin", "wheelie bin"),                   # trashcan01..05
    "recycling_container": ("Detect(RecyclingContainer)", "recycling container",
                            "bottle bank"),      # clothcontainer, glasscontainer, container
    "barrier":     ("Detect(Barrier)", "road barrier", "barrier"),   # streetbarrier, chainbarrier
    "construction_sign": ("Detect(ConstructionSign)", "construction sign",
                          "road works sign", "warning sign"),  # trafficwarning, warning*
    "advertisement": ("Detect(Advertisement)", "advertisement", "billboard"),
    "haybale":     ("Detect(HayBale)", "hay bale", "haybale"),       # haybale, haybalelb
    "mailbox":     ("Detect(Mailbox)", "mailbox", "post box"),
    # map furniture -- present in the town whether or not we spawn anything
    "stop_sign":   ("Detect(StopSign)", "stop sign", "stopsign"),
    "traffic_light": ("Detect(TrafficLight)", "Detect(Stoplight)", "traffic light",
                      "stoplight", "traffic signal", "signal", "light"),
    # MAP STRUCTURES. Region-scoped from landmarks.<town>.json, extracted by
    # scripts/extract_structures.py: an overpass is a ground road passing under an elevated
    # lane; a car park is a region holding parking-barrier objects. No bare "park" or "pass".
    "overpass":    ("Detect(Overpass)", "Detect(Underpass)", "overpass", "underpass", "flyover"),
    "car_park":    ("Detect(CarPark)", "Detect(ParkingLot)", "car park", "carpark",
                    "parking lot", "parking garage"),
}

#: Which families answer the same underlying boolean. `blocked` is the only alias, and it
#: is here rather than in the node so the node does not need to know the fiction.
LANDMARK_ALIAS = {"blocked": "cone"}


def answer_keys(at_junction: bool, cone: bool,
                cone_ahead: bool | None = None,
                landmarks: dict[str, bool] | None = None) -> dict[str, bool]:
    """Every cue spelling this deployment answers, ORDERED most specific first.

    No key may CONTAIN another family's key. A key such as "a traffic cone is in the
    intersection" would, under brain's bidirectional match, also match the bare cue
    `intersection` and answer it with the cone family. `cone` already covers any cue that
    mentions one.
    """
    keys: dict[str, bool] = {}
    if cone_ahead is not None:
        # LOOKAHEAD, and it must come FIRST. These spellings all contain "cone", so with
        # the bare `cone` key ahead of them a lookahead cue would match the here-and-now
        # answer and the whole distinction would vanish silently.
        #
        # Why a lookahead key exists at all: "turn right at the intersection BEFORE the
        # one with the cone" asks the robot to act at a point defined by something it has
        # not reached. That reads like a plan-architecture problem -- the plan advances
        # monotonically and cannot un-advance -- but it is really a CUE problem. The
        # camera sees considerably further than the LiDAR clustering does, so "is there a
        # cone at the next intersection?" is a question a forward-looking sensor can
        # answer from where the robot already is. This key is the ground-truth stand-in
        # for that question, so the PLAN side can be tested before the perception side
        # exists.
        keys.update({
            "cone at the next intersection": cone_ahead,
            "cone at the next junction": cone_ahead,
            "Detect(ConeAtNextIntersection)": cone_ahead,
            "Detect(ConeAtNextJunction)": cone_ahead,
            "cone ahead": cone_ahead,
        })
    # LANDMARKS FIRST, PLACE LAST. An object key must precede a place key or a cue
    # naming both ("a cone in the intersection") is answered about the intersection,
    # which is true at every junction, so the branch fires unconditionally.
    lm = dict(landmarks or {})
    lm.setdefault("cone", cone)
    for fam, spellings in LANDMARK_SPELLINGS.items():
        val = lm.get(LANDMARK_ALIAS.get(fam, fam))
        if val is None:
            continue          # this deployment cannot see it; leave the cue unanswerable
        for spelling in spellings:
            keys[spelling] = bool(val)
    keys.update({
        # then place. SYNONYMS MATTER: the generator writes the word the INSTRUCTION
        # used, and "crossroads" is what a British route description calls an
        # intersection. A missing spelling (e.g. `Detect(Crossroads)`) leaves brain stuck
        # at step 0. Rejecting the word would treat a vocabulary gap as a plan error;
        # answering it is the correct fix.
        "Detect(FourWayIntersection)": at_junction,
        "Detect(Crossroads)": at_junction,
        "Detect(Crossroad)": at_junction,
        "Detect(Intersection)": at_junction,
        "Detect(Junction)": at_junction,
        "four-way intersection": at_junction,
        "crossroads": at_junction,
        "crossroad": at_junction,
        "four-way": at_junction,
        "intersection": at_junction,
        "junction": at_junction,
    })
    return keys


def answers_for_world(at_junction: bool, cone: bool,
                      place_first: bool = False,
                      cone_ahead: bool | None = None,
                      landmarks: dict[str, bool] | None = None) -> dict[str, bool]:
    """What the publisher should put on the wire.

    ``place_first`` reproduces the ordering that answered an ambiguous cue with the
    place predicate. It exists so "the order decides the branch" can be TESTED rather
    than argued: with it on, a matched pair takes the same branch both times.
    """
    a = answer_keys(at_junction, cone, cone_ahead, landmarks)
    if place_first:
        # DERIVED, not hardcoded: a hardcoded object-key list silently drops any newly
        # added spelling into `head`, ahead of the place keys, and the reproduction stops
        # reproducing. The place keys are the tail of answer_keys by construction, so ask
        # for them by name and treat every landmark spelling as the object group.
        place = ["Detect(FourWayIntersection)", "Detect(Crossroads)",
                 "Detect(Crossroad)", "Detect(Intersection)", "Detect(Junction)",
                 "four-way intersection", "crossroads", "crossroad", "four-way",
                 "intersection", "junction"]
        objects = [k for fam, spell in LANDMARK_SPELLINGS.items() for k in spell
                   if k in a]
        order = place + objects
        # lookahead keys keep their place at the FRONT even under place_first -- the
        # ordering experiment is about object-vs-place, and burying a lookahead key
        # behind `cone` would make it unreachable rather than merely lower priority
        head = [k for k in a if k not in order]
        return {k: a[k] for k in head + [k for k in order if k in a]}
    return a


def resolve(cue: str, at_junction: bool, cone: bool,
            place_first: bool = False,
            cone_ahead: bool | None = None,
            landmarks: dict[str, bool] | None = None) -> bool | None:
    """Answer ``cue``, or None if nothing here answers it.

    None is a real answer and must not be collapsed to False: an unanswerable cue makes
    a step TIME OUT rather than fail, and reporting False would make it look like a
    confident negative instead of a plan naming something that does not exist.
    """
    low = (cue or "").lower().replace("_", " ").replace("-", " ")
    # SEPARATOR-NORMALISE THE CUE SIDE ONLY. Without this, `Detect(stop_sign)` is
    # rejected even though `stop_sign` is answerable, because the spellings are
    # 'stop sign' and 'stopsign' and an underscore matches neither (`traffic_cone`
    # would pass only by accident, since it contains 'cone'). Only the cue is normalised
    # -- no vocabulary key contains a separator -- so nothing that already matched
    # can change family. The same three characters must be applied in all THREE
    # implementations of this rule or the validator and the runtime drift apart.
    if not low:
        return None
    for k, v in answers_for_world(at_junction, cone, place_first,
                                  cone_ahead, landmarks).items():
        if k.lower() in low or low in k.lower():
            return v
    return None
