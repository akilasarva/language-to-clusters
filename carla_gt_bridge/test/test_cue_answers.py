"""One implementation of "answer this cue", and proof that both callers use it.

If the node and its offline twin had different matching rules, the offline harness would
certify plans the robot then executes differently, and nothing downstream could tell.
These tests are about keeping them one thing, not about the answers themselves.
"""
import os

import pytest

from carla_gt_bridge import cue_answers as CA

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_object_keys_precede_place_keys():
    """Order is semantics: brain takes the FIRST matching key.

    With the place keys first, a matched pair takes the same branch both times -- the cue
    is answered "am I at a junction", true at the decision point every time.
    """
    keys = list(CA.answer_keys(at_junction=True, cone=False))
    assert keys.index("cone") < keys.index("intersection")
    assert keys.index("Detect(TrafficCone)") < keys.index("Detect(Intersection)")


def test_no_key_contains_another_familys_key():
    """The self-poisoning failure. brain's match is bidirectional, so a key that CONTAINS
    another family's key makes that family ambiguous: with "a traffic cone is in the
    intersection" published, the bare cue `intersection` would match the cone family."""
    ks = CA.answer_keys(True, True)
    cone_keys = {k for k in ks if "cone" in k.lower()}
    place_keys = set(ks) - cone_keys
    for c in cone_keys:
        for p in place_keys:
            assert p.lower() not in c.lower(), f"{c!r} contains the place key {p!r}"
            assert c.lower() not in p.lower(), f"{p!r} contains the cone key {c!r}"


@pytest.mark.parametrize("at_junction,cone", [(True, True), (True, False),
                                              (False, True), (False, False)])
def test_the_node_and_the_offline_twin_agree_on_every_cue_and_world(at_junction, cone):
    """The node and the offline twin must answer every cue identically."""
    import sys
    sys.path.insert(0, os.path.join(WS, "carla_gt_bridge", "scripts"))
    from missions import cue_oracle
    label = "junction" if at_junction else "path"
    # BOTH callers gate "is there a cone" on being at a junction BEFORE consulting the
    # vocabulary -- the node inside `_cone_ahead()`, the twin in its bool branch. That
    # gate is caller-specific (actor list vs region set) and stays with the caller;
    # `resolve` takes the already-resolved world fact. Compare like with like.
    gated = cone and at_junction
    for cue in list(CA.answer_keys(True, True)) + [
            "a traffic cone is in the intersection", "traffic cone in the intersection",
            "Detect(Junction)", "intersection"]:
        node = CA.resolve(cue, at_junction=at_junction, cone=gated)
        twin = cue_oracle(cue, label, cone_present=cone)
        assert bool(node) == twin, (
            f"{cue!r} at_junction={at_junction} cone={cone}: node={node} twin={twin}")


def test_an_unanswerable_cue_is_None_not_False():
    """None and False are different claims. An unanswerable cue makes a step TIME OUT;
    reporting False would make it look like a confident negative."""
    assert CA.resolve("Detect(Bench)", at_junction=True, cone=True) is None
    assert CA.resolve("Detect(TrafficCone)", at_junction=True, cone=False) is False


def test_place_first_reproduces_the_bug_on_demand():
    """Kept switchable so "the order decides the branch" stays testable: cone-first
    discriminates, place-first answers the place predicate for an ambiguous cue."""
    amb = "traffic cone is present in the intersection"
    assert CA.resolve(amb, at_junction=True, cone=False) is False          # cone answer
    assert CA.resolve(amb, at_junction=True, cone=False,
                      place_first=True) is True                            # place answer


def test_the_node_does_not_build_its_own_dict_any_more():
    src = open(os.path.join(WS, "carla_gt_bridge", "carla_gt_bridge", "nodes",
                            "gt_cue_node.py")).read()
    assert "cue_answers.answers_for_world" in src
    assert '"Detect(Intersection)":' not in src, "node is rebuilding the vocabulary"


# --------------------------------------------------------------------------- #
# Landmark families vs the REAL blueprint library
#
# Every check here guards a way a landmark family fails SILENTLY: the prop spawns, the
# actor list shows it, and its cue answers False at the region it is standing in, so the
# run completes and reads as "the plan ignored the world".
# --------------------------------------------------------------------------- #

import ast
import json
import pathlib
import re

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "carla_static_props.json"


def _actor_types():
    """LANDMARK_ACTOR_TYPES without importing the node (which needs rclpy)."""
    src = (pathlib.Path(__file__).parents[1] / "carla_gt_bridge" / "nodes"
           / "gt_cue_node.py").read_text()
    m = re.search(r"LANDMARK_ACTOR_TYPES: dict\[str, tuple\[str, \.\.\.\]\] = (\{.*?\n\})",
                  src, re.S)
    assert m, "LANDMARK_ACTOR_TYPES literal not found -- did its annotation change?"
    return ast.literal_eval(m.group(1))


def _blueprints():
    return [p["id"] for p in json.loads(FIXTURE.read_text())["props"]]


def test_no_blueprint_matches_two_families():
    """`next()` over the dict resolves a double match SILENTLY, and by dict ORDER.

    So a collision is not merely wrong, it is wrong in a way that changes when someone
    reorders the table. `static.prop.busstop` contains "stop", which is why stop_sign is
    keyed on "stopsign"/"traffic.stop" and never on a bare "stop".
    """
    T = _actor_types()
    for b in _blueprints():
        hit = [f for f, subs in T.items() if any(s in b for s in subs)]
        assert len(hit) <= 1, f"{b} matches families {hit}"


def test_every_family_substring_matches_a_real_blueprint():
    """A substring matching nothing is a family that can never be spawned or answered.

    stop_sign and traffic_light are exempt: they are map furniture (`traffic.*`), present
    whether or not we spawn anything, and so absent from a static.prop.* listing. overpass and
    car_park are exempt for the same reason one level up: they are map STRUCTURES, answered
    only from region lists in config/landmarks.*.json, never spawned.
    """
    T = _actor_types()
    bps = _blueprints()
    for fam, subs in T.items():
        if fam in ("stop_sign", "traffic_light", "overpass", "car_park"):
            continue
        assert any(any(s in b for s in subs) for b in bps), \
            f"family {fam!r} {subs} matches no blueprint in CARLA's library"


def test_static_prop_mesh_is_excluded_from_the_fixture():
    """It reports infinite extent, so it passes every `>= threshold` size filter."""
    blob = json.loads(FIXTURE.read_text())
    assert "static.prop.mesh" in blob["excluded"]
    assert "static.prop.mesh" not in _blueprints()


def test_every_actor_family_has_cue_spellings_and_vice_versa():
    """The two halves are useless apart.

    A family the node can DETECT but the vocabulary cannot NAME is unreachable from a
    plan; a family the vocabulary names but the node cannot detect answers None forever
    and times the step out. `blocked` is the documented exception -- an alias onto cone
    with no actor type of its own.
    """
    T = _actor_types()
    spoken = set(CA.LANDMARK_SPELLINGS) - set(CA.LANDMARK_ALIAS) - {"cone"}
    assert set(T) == spoken, (
        f"only in the node: {sorted(set(T) - spoken)}; "
        f"only in the vocabulary: {sorted(spoken - set(T))}")


def test_no_cue_spelling_contains_another_familys_spelling():
    """Generalises test_no_key_contains_another_familys_key to EVERY pair.

    With many families the exposure is quadratic, and the failure is the same: brain's match is
    bidirectional and takes the first hit, so a contained spelling answers about the
    wrong object. This is why there is no "streetlight" family -- "light" is a bare
    traffic_light spelling.
    """
    S = CA.LANDMARK_SPELLINGS
    fams = list(S)
    for i, a in enumerate(fams):
        for b in fams[i + 1:]:
            if CA.LANDMARK_ALIAS.get(a) == b or CA.LANDMARK_ALIAS.get(b) == a:
                continue
            for ka in S[a]:
                for kb in S[b]:
                    assert ka.lower() not in kb.lower(), f"{b}:{kb!r} contains {a}:{ka!r}"
                    assert kb.lower() not in ka.lower(), f"{a}:{ka!r} contains {b}:{kb!r}"


def test_a_new_family_is_answerable_end_to_end():
    """Name a family in a plan, get an answer."""
    for fam, cue in [("fountain", "Detect(Fountain)"),
                     ("trash_can", "trash can"),
                     ("barrier", "a road barrier is ahead"),
                     ("kiosk", "Detect(Kiosk)")]:
        assert CA.resolve(cue, at_junction=True, cone=False,
                          landmarks={fam: True}) is True, f"{cue} unanswered"
        assert CA.resolve(cue, at_junction=True, cone=False,
                          landmarks={fam: False}) is False, f"{cue} wrong polarity"
        # absent from the world -> UNANSWERABLE, not a confident False
        assert CA.resolve(cue, at_junction=True, cone=False) is None, \
            f"{cue} answered without the family present"
