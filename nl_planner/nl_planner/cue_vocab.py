"""Which cues a deployment can actually ANSWER, and whether a plan's cue is one of them.

WHY THIS EXISTS. The taxonomy check validates that every MODE a plan names resolves; this
does the same for CUES. Two failure modes pass every structural validator, the STL syntax
gate, and the offline simulator:

  1. UNANSWERABLE. E.g. `Detect(Junction)`: the offline oracle matches "intersection" OR
     "junction" and answers it, but if `gt_cue_node` publishes no key containing
     "junction", in CARLA the plan deadlocks in CHECKING_CUE -- step 0 never completes
     and the cue times out with no sightings while the vehicle sits in the junction.

  2. AMBIGUOUS. E.g. `traffic cone is present in the intersection` as a BRANCH cue
     matches both the cone predicate and the junction predicate. brain takes the first
     match, gets the junction answer -- true at every junction -- and the branch fires
     unconditionally, whether or not the cone is there.

Both are the same defect: a plan naming something the runtime cannot answer, with nothing
checking. This turns them into RETRYABLE generation errors, which is what the mode
validators already do for modes.

SCOPE. A vocabulary is a property of the deployment, not of the language. Under
`cue_source=vlm` a model answers free text and nothing here applies -- pass `vocab=None`
and the check is skipped rather than inventing a constraint the runtime does not have.
Under `cue_source=topic` the publisher's keys ARE the vocabulary and a cue outside it
cannot be answered by anything.
"""
from __future__ import annotations

import os
import re
from typing import Iterable

__all__ = ["CARLA_GT_VOCAB", "families_matching", "check_cue", "validate_plan_cues",
           "vocab_for_deployment"]

#: family -> the spellings a publisher answers it by. Mirrors
#: `carla_gt_bridge/nodes/gt_cue_node.py`; a test asserts the two do not drift, because
#: this file being wrong in the same direction as the node would hide exactly the bug it
#: exists to catch.
CARLA_GT_VOCAB: dict[str, tuple[str, ...]] = {
    # Deliberately NO long descriptive key. A key containing another family's key makes
    # that family ambiguous under the runtime's bidirectional substring match: with
    # "a traffic cone is in the intersection" published, the bare cue `intersection`
    # would match BOTH families. `cone` covers every cue that mentions one.
    "cone": ("Detect(TrafficCone)", "traffic cone", "cone"),
    # `blocked` answers to the cone: in this deployment nothing else obstructs a junction.
    # Without it, a cue such as `vlm_cue: "way ahead is blocked"` matches no published key
    # and brain waits at the junction indefinitely even though the plan is correct.
    "blocked": ("Detect(BlockedAhead)", "Detect(Blocked)", "the way ahead is blocked",
                "way ahead is blocked", "blocked ahead", "blocked"),
    "bench": ("Detect(Bench)", "bench"),
    "bus_shelter": ("Detect(BusShelter)", "Detect(BusStop)", "bus shelter",
                    "bus stand", "bus stop"),
    # SPAWNABLE PROPS. Mirrors cue_answers.LANDMARK_SPELLINGS exactly;
    # preflight [5] fails on any drift. This file is the VALIDATOR, so a family missing
    # here is not merely unvalidated -- `check_cue` reports it unanswerable and the
    # generator retries, so a plan that names a prop that can actually be spawned and detected is
    # rejected before it ever runs -- which looks like the LLM failing to follow the
    # schema rather than a validator bug.
    "fountain": ("Detect(Fountain)", "fountain"),
    "kiosk": ("Detect(Kiosk)", "kiosk", "food cart", "newsstand"),
    "vending_machine": ("Detect(VendingMachine)", "vending machine"),
    "trash_can": ("Detect(TrashCan)", "trash can", "garbage can",
                  "rubbish bin", "wheelie bin"),
    "recycling_container": ("Detect(RecyclingContainer)", "recycling container",
                            "bottle bank"),
    "barrier": ("Detect(Barrier)", "road barrier", "barrier"),
    "construction_sign": ("Detect(ConstructionSign)", "construction sign",
                          "road works sign", "warning sign"),
    "advertisement": ("Detect(Advertisement)", "advertisement", "billboard"),
    "haybale": ("Detect(HayBale)", "hay bale", "haybale"),
    "mailbox": ("Detect(Mailbox)", "mailbox", "post box"),
    # No bare "stop" (it sits inside both "stop sign" and "bus stop") and no bare "sign"
    # (inside "stop sign") -- either would make two families answer one cue.
    "stop_sign": ("Detect(StopSign)", "stop sign", "stopsign"),
    "traffic_light": ("Detect(TrafficLight)", "Detect(Stoplight)", "traffic light",
                      "stoplight", "traffic signal", "signal", "light"),
    # Map structures -- mirrors cue_answers.LANDMARK_SPELLINGS.
    "overpass": ("Detect(Overpass)", "Detect(Underpass)", "overpass", "underpass", "flyover"),
    "car_park": ("Detect(CarPark)", "Detect(ParkingLot)", "car park", "carpark",
                 "parking lot", "parking garage"),
    # Mirrors the runtime exactly (a test asserts no drift). "crossroads" is here
    # because the generator writes the instruction's own word, and Detect(Crossroads)
    # must be answerable.
    "junction": ("Detect(FourWayIntersection)", "Detect(Crossroads)", "Detect(Crossroad)",
                 "Detect(Intersection)", "Detect(Junction)", "four-way intersection",
                 "crossroads", "crossroad", "four-way", "intersection", "junction"),
}

#: Cues resolved from the TRAJECTORY rather than from a publisher. `Bearing(Right)` is
#: answered by the heading change since the step began, so it needs no vocabulary entry
#: and must not be reported unanswerable.
_TRAJECTORY_CUE = re.compile(r"\bbearing\s*\(", re.I)


def families_matching(cue: str, vocab: dict[str, tuple[str, ...]]) -> list[str]:
    """Every predicate family this cue could be answered by, by the runtime's own rule.

    The matching is brain's: a published key that is a substring of the cue, or the cue a
    substring of the key. Reproducing the runtime's rule rather than a stricter one is
    the point -- a validator that matched differently would pass cues the robot then
    fails on.
    """
    low = (cue or "").lower().replace("_", " ").replace("-", " ")
    # SEPARATOR-NORMALISE THE CUE SIDE ONLY. Without it `Detect(stop_sign)` is
    # rejected even though `stop_sign` is answerable, because the spellings are
    # 'stop sign' and 'stopsign' and an underscore matches neither (`traffic_cone`
    # would pass only by accident, since it contains 'cone'). Only the cue is normalised
    # -- no vocabulary key contains a separator -- so nothing that already matched
    # can change family. The same three characters must be applied in all THREE
    # implementations of this rule or the validator and the runtime drift apart.
    if not low:
        return []
    out = []
    for fam, keys in vocab.items():
        if any(k.lower() in low or low in k.lower() for k in keys):
            out.append(fam)
    return out


def check_cue(cue: str, vocab: dict[str, tuple[str, ...]] | None) -> tuple[bool, str]:
    """(ok, reason). ``vocab=None`` means a VLM answers free text -- nothing to check."""
    if vocab is None or not cue or _TRAJECTORY_CUE.search(cue):
        return True, ""
    fams = families_matching(cue, vocab)
    if not fams:
        return False, (
            f"cue {cue!r} matches no answerable predicate "
            f"(this deployment answers: {sorted(vocab)}). It will time out, not fail.")
    if len(fams) > 1:
        return False, (
            f"cue {cue!r} matches {len(fams)} predicates {fams} — the runtime takes one "
            f"of them and a branch on it fires on the wrong question. Name ONE.")
    return True, ""


def _walk(steps: Iterable):
    for st in steps or []:
        yield st
        for b in (st.get("branches") or []):
            yield from _walk(b.get("sub_plan"))


def validate_plan_cues(plan, vocab: dict[str, tuple[str, ...]] | None) -> list[str]:
    """Every cue in the plan the deployment cannot answer unambiguously. [] = fine.

    Branch `vlm_cue`s are checked as well as step `transition_cue`s, and they are where
    ambiguity actually bites: a step cue that matches the wrong predicate merely advances
    early, while a BRANCH cue that does decides which way the robot goes.
    """
    steps = plan.get("steps") if isinstance(plan, dict) else getattr(plan, "steps", None)
    if hasattr(steps, "__iter__") and steps is not None and not isinstance(steps, list):
        steps = list(steps)
    if steps and not isinstance(steps[0], dict):
        steps = [s.model_dump() if hasattr(s, "model_dump") else s for s in steps]

    bad: list[str] = []
    for st in _walk(steps):
        ok, why = check_cue(st.get("transition_cue"), vocab)
        if not ok:
            bad.append(f"step {st.get('step')}: {why}")
        for b in (st.get("branches") or []):
            vc = b.get("vlm_cue")
            if not vc or str(vc).strip().lower() == "default":
                continue
            ok, why = check_cue(vc, vocab)
            if not ok:
                bad.append(f"step {st.get('step')} branch: {why}")
    return bad


def vocab_for_deployment(source: str | None = None) -> dict[str, tuple[str, ...]] | None:
    """The answerable vocabulary for the deployment in force, or None for free text.

    THE ONE PLACE THE RULE LIVES. Both `stl_ablation.run_one` and
    `pipeline.generate_plan` (which `planner_node.py` runs) resolve through here, so the
    harness and the robot cannot drift apart.

    The rule follows the SENSOR, not the language, which is `check_cue`'s own scoping:

      cue_source=topic  the publisher's keys ARE the answerable set (default). A cue
                        outside it can be answered by nothing -- it times out silently.
      cue_source=vlm    a model answers free text; there is no closed set, and enforcing
                        one would reject a perfectly answerable open-set cue.

    `CUE_SOURCE` is the existing deployment-wide convention -- `drive_english.py`,
    `preflight.py` and `stl_ablation.py` all read that same variable -- so this introduces
    no new configuration surface. Pass `source` explicitly to override it.

    NOT A CAMPUS VOCABULARY. `CARLA_GT_VOCAB` covers CARLA prop families only. Anything
    running off CARLA must pass its own dict (a scene graph's object labels) or `None`;
    e.g. most Touchdown cues name referents CARLA has no family for.
    """
    src = (source if source is not None
           else os.environ.get("CUE_SOURCE", "topic")).strip().lower()
    return None if src == "vlm" else CARLA_GT_VOCAB
