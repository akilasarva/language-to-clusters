import os
"""Many cluster maps and spellings, one concept per idea."""
import pytest
from nl_planner.mode_concepts import AXIS, UNSETTLED, concept_of
from nl_planner.taxonomy import load_taxonomy

W = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) + "/"
COMBINED = W + "bev_pipeline/config/cluster_map.combined.yaml"
CAMPUS = W + "bev_pipeline/config/cluster_map.full_campus.yaml"
CAMPUS1 = W + "bev_pipeline/config/cluster_map.full_campus_1hz.yaml"
CARLA = W + "carla_gt_bridge/config/cluster_map.carla_town01.yaml"


@pytest.mark.parametrize("spellings,concept", [
    (("Path", "path", "Road: On", "On: Path"), "path"),
    (("Junction", "junction", "Intersection: In", "In: Junction"), "junction"),
    (("Open Space", "Space: Open", "open_space"), "open_space"),
    (("Along Wall", "Wall: Along", "Edge: Along"), "along_edge"),
    (("Corridor", "Covered", "On Bridge", "In Building"), "passage"),
])
def test_renames_across_map_generations_are_one_concept(spellings, concept):
    """The drift is word ORDER, not case: `Open Space` -> `Space: Open`."""
    assert {concept_of(s) for s in spellings} == {concept}


def test_one_name_resolves_into_each_maps_own_vocabulary():
    """What the answer key needs: ask for `path`, get whatever THIS map calls it."""
    assert load_taxonomy(COMBINED).canonical_mode("path") == "Path"
    assert load_taxonomy(CAMPUS).canonical_mode("path") == "Road: On"
    assert load_taxonomy(CAMPUS1).canonical_mode("path") == "On: Path"
    assert load_taxonomy(CARLA).canonical_mode("path") == "path"
    # and the reverse direction, which is what a formula written for one map needs
    assert load_taxonomy(COMBINED).canonical_mode("Road: On") == "Path"


def test_an_absent_concept_still_returns_none():
    """carla_town01 is {junction, path}. Concepts must not invent what is not there."""
    tax = load_taxonomy(CARLA)
    assert tax.canonical_mode("open_space") is None
    assert tax.canonical_mode("Passage") is None
    assert tax.canonical_mode("On Bridge") is None
    # `Along Wall` is along_edge, which carla_town01 does not have either -- an
    # enclosure concept must not fall back onto a topology one
    assert tax.canonical_mode("Along Wall") is None
    # but a mode that IS one of this map's concepts under another map's spelling
    # resolves -- that is the point of the layer, not a leak in it
    assert tax.canonical_mode("Road: On") == "path"
    assert tax.canonical_mode("Intersection: In") == "junction"


def test_several_modes_under_one_concept_resolve_deterministically():
    """full_campus_1hz has Approach:, In: and Exit: Junction -- all junctions.

    The tie is broken by the order the spellings are listed under their concept, core
    first, so this is the same answer every time rather than dict order. That is why
    the CONCEPTS lists are hand-ordered and not alphabetical.
    """
    tax = load_taxonomy(CAMPUS1)
    assert {"Approach: Junction", "In: Junction", "Exit: Junction"} <= set(tax.modes)
    assert tax.canonical_mode("junction") == "In: Junction"
    assert tax.canonical_mode("junction") == tax.canonical_mode("Intersection: In")


@pytest.mark.parametrize("narrow", ["Corridor", "Covered", "On Bridge", "In Building"])
def test_narrow_and_enclosed_things_are_passages(narrow):
    """The settled division: a bridge and a corridor are passages, not paths.

    It also resolves a contradiction: meadow_1hz has Corridor and no
    path mode while full_campus_1hz has BOTH Corridor and On: Path -- which could not
    both be true if Corridor were a path.
    """
    assert concept_of(narrow) == "passage"


@pytest.mark.parametrize("edge", ["Along Wall", "Wall: Along", "Edge: Along"])
def test_along_edge_is_its_own_concept_on_the_enclosure_axis(edge):
    """The one-sided case, between passage (two sides) and open_space (zero).

    A classifier confusing along_edge with path says the classifier cannot separate two
    things, not that they are the same thing. A forward camera CAN see a building on
    one side, and merging it would collapse a three-valued axis into two.
    """
    from nl_planner.mode_concepts import AXIS
    assert concept_of(edge) == "along_edge"
    assert AXIS["along_edge"] == "enclosure"
    assert concept_of(edge) != concept_of("Path")


def test_every_spelling_in_every_map_lands_on_a_concept():
    """Every spelling across the maps has a concept; an unclassified one is drift."""
    import glob
    unclassified = []
    for f in glob.glob(W + "*/config/cluster_map*.yaml"):
        try:
            tax = load_taxonomy(f)
        except Exception:                                          # noqa: BLE001
            continue
        unclassified += [m for m in tax.modes_for_prompt() if concept_of(m) is None]
    assert not unclassified, f"unclassified modes: {sorted(set(unclassified))}"


def test_the_four_divisions_resolve_in_the_maps_that_have_them():
    assert load_taxonomy(COMBINED).canonical_mode("passage") == "Passage"
    assert load_taxonomy(CAMPUS1).canonical_mode("passage") == "Covered"
    assert load_taxonomy(COMBINED).canonical_mode("open_space") == "Space: Open"
    assert load_taxonomy(CAMPUS).canonical_mode("open_space") == "Open Space"


def test_unknown_modes_are_not_guessed_into_the_nearest_concept():
    # Tokens with no concept at all (`grass` is a real surface mode, tested below).
    # An unknown mode surfaces as unresolvable, never as the nearest neighbour.
    assert concept_of("escalator") is None
    assert concept_of("riverbank") is None
    assert concept_of("") is None


def test_grass_is_a_surface_and_does_not_resolve_where_the_map_lacks_it():
    """Grass is on the SURFACE axis, not the enclosure concept `open_space`.

    Conflating them ("lawn, plaza") would make "never cross the grass" compile to
    `forbid open_space` -- which forbids paved plazas and permits a grass verge, the
    mission's opposite. Separating them lets a
    region be open AND grass, with the two constraints intersecting.
    """
    assert concept_of("grass") == "grass"
    assert AXIS["grass"] == "surface"
    assert AXIS["grass"] != AXIS["open_space"]
    # ...but only where the map actually declares it.
    assert load_taxonomy(COMBINED).canonical_mode("grass") is None
    terrain = W + "carla_gt_bridge/config/cluster_map.carla_town05_terrain.yaml"
    assert load_taxonomy(terrain).canonical_mode("grass") == "grass"


def test_the_settled_sensor_routing():
    """LiDAR decides STRUCTURE; the camera decides everything within-open.

    Routed design: each sensor is asked only the question it is good at, rather than
    the whole 5-way task -- geometry for enclosure+open_space, camera for
    path/junction/along_edge.
    """
    from nl_planner.mode_concepts import AXIS, SENSOR
    assert SENSOR["passage"] == "geom" and SENSOR["open_space"] == "geom"
    assert SENSOR["path"] == "camera" and SENSOR["junction"] == "camera"
    # along_edge is the case where axis and deciding sensor disagree: an enclosure
    # concept, but geometry only claims it above a 0.75 RF confidence floor and it
    # otherwise falls through to the camera
    assert AXIS["along_edge"] == "enclosure" and SENSOR["along_edge"] == "camera"
    assert not UNSETTLED, "all merges settled 19 Aug 2026"


def test_junction_is_plan_executed_not_perceived():
    """Junction is handled open-loop, by design.

    Junction labels encode ROUTE CONTEXT that local per-frame perception cannot carry,
    so the plan executes the turn and the cluster->path exit edge confirms traversal.
    """
    from nl_planner.mode_concepts import (JUNCTION_IS_NOT_PERCEIVED,
                                          JUNCTION_PHASE_RECALL, LABELS_OVERCALL)
    assert JUNCTION_IS_NOT_PERCEIVED
    # and a single junction recall hides a real per-phase split
    assert JUNCTION_PHASE_RECALL["approach_in"] > 0.6   # camera sees it coming
    assert JUNCTION_PHASE_RECALL["exit"] < 0.2          # camera-blind, topology edge
    assert "junction" in LABELS_OVERCALL


def test_meadow_has_no_path_and_that_is_correct():
    """A lawn environment: Corridor and Covered are passages, Edge: Along is one-sided.

    Pinned because it looks like a bug and is not -- meadow_1hz genuinely has no
    path-topology mode, which is why `canonical_mode("path")` must return None there
    rather than reaching for the nearest enclosure mode.
    """
    tax = load_taxonomy(W + "bev_pipeline/config/cluster_map.meadow_1hz.yaml")
    assert tax.canonical_mode("path") is None
    assert tax.canonical_mode("passage") == "Covered"
    assert tax.canonical_mode("along_edge") == "Edge: Along"
