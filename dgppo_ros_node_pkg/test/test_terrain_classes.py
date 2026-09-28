"""Tests for the one terrain vocabulary.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
         dgppo_ros_node_pkg/test/test_terrain_classes.py

Several tests here encode label-mapping bugs present in the legacy `terrain_analysis/`
nodes; the module exists to keep those bugs from reaching a controller.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "carla_gt_bridge"))

from carla_gt_bridge.terrain_classes import (CARLA_TAG_TO_CLASS,  # noqa: E402
                                             CITYSCAPES_TO_CLASS, DEFAULT_COSTS, GRASS,
                                             NAMES, OTHER, ROAD, SIDEWALK, UNOBSERVED,
                                             check_forbid, classes_from_names,
                                             from_carla_tags, from_cityscapes)


def test_nothing_unrecognised_ever_reads_as_road():
    """`terrain_analysis/terrain_analysis/segformer_node.py` is the counterexample:

        seg_label_map = np.zeros((h, w), dtype=np.uint8)     # every pixel is now its Road=0
        seg_label_map[pred == 0] = 0
        seg_label_map[pred == 8] = 1
        seg_label_map[pred == 1] = 2

    so building, sky, car, pole, fence and vegetation all come out road. Reduced to one
    whole-frame label over a bottom-centre crop that is survivable; used to score a rollout
    aimed at a wall it is not.
    """
    cs = np.array([[2, 10, 13, 5, 11, 17, 255]])          # building, sky, car, pole, ...
    assert (from_cityscapes(cs) != ROAD).all()
    assert (from_cityscapes(cs) == OTHER).all()
    carla = np.array([[3, 11, 14, 6, 5, 26, 99]])         # Buildings, Sky, Car, Poles, ...
    assert (from_carla_tags(carla) != ROAD).all()
    assert (from_carla_tags(carla) == OTHER).all()


def test_zero_means_saw_nothing_not_road():
    """The inversion that makes the bug above harmless rather than silent: an array that was
    never written to reads as UNOBSERVED, which is dropped from the score and reported.
    """
    assert UNOBSERVED == 0
    assert NAMES[int(np.zeros(1, dtype=np.uint8)[0])] == "unobserved"
    assert ROAD != 0


def test_vegetation_is_not_grass_and_terrain_is():
    """Cityscapes trainId 8 is VEGETATION -- tree canopy -- and 9 is TERRAIN, the grass and
    soil. `segformer_node.py` maps 8 to grass, so it calls the canopy "grass" and leaves
    the actual lawn unmapped, which under its zero-default then reads as road. Both halves
    are wrong and both are inverted here.

    CARLA numbers them the other way round (9 Vegetation, 10 Terrain), which is exactly the
    kind of near-miss that makes a hand-typed table fail quietly.
    """
    assert CITYSCAPES_TO_CLASS[9] == GRASS
    assert from_cityscapes(np.array([[8]]))[0, 0] == OTHER
    assert CARLA_TAG_TO_CLASS[10] == GRASS
    assert from_carla_tags(np.array([[9]]))[0, 0] == OTHER


def test_road_markings_are_road():
    """CARLA tags lane markings separately from the carriageway (`RoadLines`, 24). Left
    unmapped, every marking in the image reads OTHER and the middle of a road scores as
    costly terrain -- a penalty that would look like a segmentation problem.
    """
    assert from_carla_tags(np.array([[24]]))[0, 0] == ROAD
    assert from_carla_tags(np.array([[1]]))[0, 0] == ROAD


def test_the_carla_table_is_derived_from_the_verified_tag_names():
    """Not typed in. `frame_labels.SEMANTIC_TAGS` was checked against a real bag, and CARLA
    has renumbered these between versions -- the host's importable `carla` is 0.10.0 while
    the server this drives is `carlasim/carla:0.9.14`, so the numbers must never come from
    an import here. A rename must raise at import, which `_assert_names_exist` does.
    """
    from carla_gt_bridge.frame_labels import SEMANTIC_TAGS
    for tag in CARLA_TAG_TO_CLASS:
        assert tag in SEMANTIC_TAGS
    assert {SEMANTIC_TAGS[t] for t in CARLA_TAG_TO_CLASS} == {
        "Roads", "RoadLines", "Sidewalks", "Terrain"}


def test_unobserved_is_priced_nowhere_and_forbidden_never():
    """It is DROPPED from the weighted mean, not given a cost, so a cost entry for it would
    be dead code that reads as a decision. And a forbid set containing it would let a stale
    or dead camera veto every candidate, leaving the vehicle circling indefinitely.
    """
    assert UNOBSERVED not in DEFAULT_COSTS
    with pytest.raises(ValueError, match="never be forbidden"):
        check_forbid({GRASS, UNOBSERVED})
    check_forbid({GRASS, SIDEWALK})              # must not raise


def test_an_unknown_forbid_name_raises_rather_than_being_dropped():
    """A prohibition that silently loses a term completes the run and prohibits nothing --
    the same shape as `forbid_modes` naming a mode that does not exist.
    """
    with pytest.raises(ValueError, match="unknown terrain class"):
        classes_from_names(["grass", "gravel"])
    assert classes_from_names(["grass"]) == frozenset({GRASS})


def test_the_ids_fit_in_the_wire_they_travel_on():
    """The mask is published as `mono8`. A negative id does not survive that -- `-1` arrives
    as 255 -- which is why UNOBSERVED is 0 and not -1.
    """
    ids = list(NAMES)
    assert min(ids) >= 0 and max(ids) <= 255
    assert np.array(ids, dtype=np.uint8).tolist() == ids
