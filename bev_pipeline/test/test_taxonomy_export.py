"""Tests for taxonomy export — the key nl_planner compatibility guarantee.

Validates the generated cluster_map.<env>.yaml against the REAL
``nl_planner.taxonomy.load_taxonomy`` (imported if nl_planner is on the path,
else the loader vendored from origin/english_to_stl at test-authoring time).
"""

import importlib.util
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bev_pipeline.taxonomy_export import (build_cluster_map, write_cluster_map,   # noqa: E402
                                          build_hierarchical_cluster_map,
                                          write_hierarchical_cluster_map,
                                          prettify_label)


def _load_taxonomy_module():
    try:
        import nl_planner.taxonomy as t   # real package if available
        return t
    except Exception:                     # noqa: BLE001
        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(
            "_vendored_taxonomy", os.path.join(here, "_vendored_taxonomy.py"))
        mod = importlib.util.module_from_spec(spec)
        # Register BEFORE exec so dataclass field-type resolution can find the
        # module via cls.__module__ (else it NPEs on frozen dataclasses).
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod


LABELS = ["open_road", "approach_building", "along_building", "on_intersection"]


def test_prettify():
    assert prettify_label("open_road") == "Open Road"
    assert prettify_label("along_building") == "Building: Along"
    assert prettify_label("on_intersection") == "Intersection: On"


def test_planner_mode_mapping_merges_ids():
    from bev_pipeline.taxonomy_export import build_cluster_map
    # two labels sharing a planner mode should merge into one id list
    pm = {"path_on": "path", "corridor": "path", "open_space": "open_space"}
    doc = build_cluster_map(["open_space", "path_on", "corridor"], "e", planner_mode_map=pm)
    assert doc["modes"]["path"] == [1, 2]
    assert doc["modes"]["open_space"] == [0]


def test_build_cluster_map_structure():
    doc = build_cluster_map(LABELS, "full_campus", source="unit-test")
    assert doc["environment"] == "full_campus"
    assert doc["source"] == "unit-test"
    # each label -> one mode -> its index id
    assert doc["modes"]["Open Road"] == [0]
    assert doc["modes"]["Building: Along"] == [2]


def test_roundtrip_loads_via_nl_planner_loader():
    tax = _load_taxonomy_module()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cluster_map.full_campus.yaml")
        write_cluster_map(LABELS, "full_campus", path, source="unit-test")
        loaded = tax.load_taxonomy(path)      # must not raise

    assert loaded.environment == "full_campus"
    # canonical id lookups work and match our indices
    assert loaded.canonical_id("Open Road") == 0
    assert loaded.canonical_id("Building: Along") == 2
    # inverse lookup
    assert loaded.label_for_id(3) == "Intersection: On"
    # every generated mode is resolvable
    for mode in loaded.modes_for_prompt():
        assert len(loaded.resolve(mode)) >= 1


# --------------------------------------------------------------------------- #
# Hierarchical exporter — the subsumption lattice + degradation policy         #
# --------------------------------------------------------------------------- #

#: The geometric cluster vocabulary from config/nav_modes.yaml, in id order.
NAV_LABELS = ["open_space", "path", "along_edge", "passage", "junction", "other"]
OPEN_SPACE, PATH, ALONG_EDGE, PASSAGE, JUNCTION, OTHER = range(6)


def test_hierarchical_subsumption_is_upward():
    """Specialized clusters also satisfy their coarse parent mode."""
    doc = build_hierarchical_cluster_map(NAV_LABELS, "livox1")
    road = doc["modes"]["path"]
    # along_edge / passage / junction all ARE on a road.
    assert set(road) == {PATH, ALONG_EDGE, PASSAGE, JUNCTION}
    # ...but the fine modes stay narrow: being on a path is not being along a wall.
    assert doc["modes"]["along_edge"] == [ALONG_EDGE]
    assert doc["modes"]["junction"] == [JUNCTION]
    assert doc["modes"]["passage"] == [PASSAGE]


def test_hierarchical_ids_are_preferred_first():
    """canonical_id() must keep returning the mode's OWN (finest) id."""
    tax = _load_taxonomy_module()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cluster_map.livox1.yaml")
        write_hierarchical_cluster_map(NAV_LABELS, "livox1", path)
        loaded = tax.load_taxonomy(path)

    # "path" carries four ids; the first must be `path`, not a subsumed child.
    assert loaded.canonical_id("path") == PATH
    assert loaded.canonical_id("along_edge") == ALONG_EDGE
    assert loaded.canonical_id("junction") == JUNCTION


def test_degradation_targets_only_the_coarse_modes_own_id():
    """`along_edge` degrades to `path` — NOT to junction/passage.

    Regression guard: degrading to every id in the coarse mode's list would
    silently let `junction` satisfy an `along_edge` step.
    Degradation must target the coarse mode's OWN id only.
    """
    doc = build_hierarchical_cluster_map(NAV_LABELS, "livox1")
    meta = doc["mode_meta"]

    assert meta["along_edge"]["accept_degraded"] == [PATH]
    assert meta["passage"]["accept_degraded"] == [PATH]
    assert meta["junction"]["accept_degraded"] == [PATH]
    # open_space is not a specialization of path -> no degradation entry.
    assert "accept_degraded" not in meta["open_space"]
    # a coarse mode does not degrade to itself
    assert "accept_degraded" not in meta["path"]


def test_perception_backed_defaults_and_override():
    """Same plan, different binding: junction is a percept in CARLA, not on campus."""
    carla = build_hierarchical_cluster_map(NAV_LABELS, "carla_intersection_right")
    assert carla["mode_meta"]["junction"]["perception_backed"] is True

    campus = build_hierarchical_cluster_map(
        NAV_LABELS, "livox1", perception_backed={"junction": False}
    )
    assert campus["mode_meta"]["junction"]["perception_backed"] is False
    # the mode is still present and resolvable — it is a plan construct, not absent
    assert campus["modes"]["junction"] == [JUNCTION]


def test_hierarchical_roundtrips_through_nl_planner_loader():
    """mode_meta is additive: the real loader must still accept the file."""
    tax = _load_taxonomy_module()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cluster_map.livox1.yaml")
        write_hierarchical_cluster_map(
            NAV_LABELS, "livox1", path, source="unit-test",
            perception_backed={"junction": False},
        )
        loaded = tax.load_taxonomy(path)      # must not raise on the extra key

    assert loaded.environment == "livox1"
    for mode in loaded.modes_for_prompt():
        assert len(loaded.resolve(mode)) >= 1


def test_unknown_labels_fall_back_to_prettified_standalone_mode():
    doc = build_hierarchical_cluster_map(["path", "bollard_field"], "e")
    assert doc["modes"]["path"] == [0]
    assert doc["modes"]["Field: Bollard"] == [1]


def test_road_hierarchy_still_available_for_carla():
    """The car-on-road binding must survive the pedestrian switch."""
    from bev_pipeline.taxonomy_export import (ROAD_HIERARCHY, ROAD_DEGRADE_TO,
                                              build_hierarchical_cluster_map)
    doc = build_hierarchical_cluster_map(
        NAV_LABELS, "carla_intersection_right",
        hierarchy=ROAD_HIERARCHY, degrade_to=ROAD_DEGRADE_TO)
    assert set(doc["modes"]["Road: On"]) == {PATH, ALONG_EDGE, PASSAGE, JUNCTION}
    assert doc["modes"]["Along Wall"] == [ALONG_EDGE]
    assert doc["mode_meta"]["Along Wall"]["accept_degraded"] == [PATH]


def test_pedestrian_names_equal_cluster_names():
    """One vocabulary in the pedestrian domain — no translation layer to drift."""
    from bev_pipeline.taxonomy_export import HIERARCHY
    for cluster, (planner, _) in HIERARCHY.items():
        assert cluster == planner, (
            f"pedestrian planner name {planner!r} differs from cluster {cluster!r}"
        )
