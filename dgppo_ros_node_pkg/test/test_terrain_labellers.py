"""Tests for the swappable terrain labellers.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \\
         dgppo_ros_node_pkg/test/test_terrain_labellers.py

`segformer` is not exercised here -- it needs torch, transformers and a downloaded
checkpoint, and a test that silently skips is worse than no test. It should be scored
against the CARLA semantic oracle on real frames instead.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "carla_gt_bridge"))

from carla_gt_bridge.terrain_classes import (GRASS, OTHER, ROAD,  # noqa: E402
                                             SIDEWALK, UNOBSERVED)
from carla_gt_bridge.terrain_labellers import (LABELLERS, agreement,  # noqa: E402
                                               make_labeller)


def test_an_unknown_labeller_raises_rather_than_falling_back():
    """A silent fallback would report whichever labeller it landed on under the name of the
    one asked for, so two configurations could quietly run the same model.
    """
    with pytest.raises(ValueError, match="unknown labeller"):
        make_labeller("deeplab")
    assert set(LABELLERS) == {"carla_semantic", "hsv", "segformer"}


def test_the_carla_oracle_reads_the_tag_from_the_red_channel():
    """CARLA packs the semantic tag into RED. Handed a BGRA frame the labeller must take
    channel 2, not channel 0 -- reading blue gives zeros, i.e. a frame that is entirely
    "unlabeled" and therefore entirely OTHER, which looks like a segmentation failure.
    """
    lab = make_labeller("carla_semantic")
    bgra = np.zeros((4, 4, 4), dtype=np.uint8)
    bgra[:, :, 2] = 1                                  # Roads
    assert (lab(bgra) == ROAD).all()
    assert (lab(np.full((4, 4), 10, dtype=np.uint8)) == GRASS).all()   # Terrain, 2-D input


def test_hsv_labels_the_three_colours_it_was_written_for():
    """A positive control, so the tests below about its DEFAULT cannot pass vacuously."""
    import cv2
    lab = make_labeller("hsv")
    hsv = np.zeros((3, 1, 3), dtype=np.uint8)
    hsv[0, 0] = (60, 200, 200)        # green
    hsv[1, 0] = (0, 0, 30)            # near-black, low saturation
    hsv[2, 0] = (0, 0, 150)           # light grey
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    got = lab(bgr).ravel()
    assert got[0] == GRASS and got[1] == ROAD and got[2] == SIDEWALK


def test_hsv_does_not_call_everything_it_cannot_place_road():
    """The zero-default bug this layer guards against; it is worse in HSV than in SegFormer.

    `terrain_analysis/terrain_segmenter.py` builds `np.zeros(...)` and paints three
    ranges, so every pixel matching none keeps id 0 -- which is ROAD in that vocabulary. The
    three ranges do not cover HSV space, so that is not an edge case: a saturated red car, a
    blue sky and a bright wall are all outside them. SegFormer at least assigns every pixel
    SOME Cityscapes id; HSV genuinely has nowhere to put these.
    """
    import cv2
    lab = make_labeller("hsv")
    hsv = np.zeros((3, 1, 3), dtype=np.uint8)
    hsv[0, 0] = (0, 255, 255)         # saturated red -- a car
    hsv[1, 0] = (110, 180, 240)       # bright saturated blue -- sky
    hsv[2, 0] = (20, 200, 255)        # bright orange -- a cone
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    got = lab(bgr).ravel()
    assert (got != ROAD).all()
    assert (got == UNOBSERVED).all()


def test_agreement_reports_the_two_confusions_a_prohibition_depends_on():
    """Overall accuracy hides the only errors that matter. Here a labeller is right about
    93% of the frame -- sky and buildings are most of it -- while calling a third of the lawn
    asphalt, which is precisely the failure that makes "never cross the grass" unenforceable.
    """
    truth = np.full((10, 10), OTHER, dtype=np.int16)
    truth[:3] = ROAD
    truth[3:6] = GRASS
    pred = truth.copy()
    pred[3:4] = ROAD                                    # 1 of 3 grass rows called road
    m = agreement(pred, truth)
    # Exactly 10 of 100 pixels wrong. Stated as the number rather than a threshold: the point
    # of the test is the GAP between this and `grass_called_road`, and a bound would let the
    # gap shrink without failing.
    assert m["overall_accuracy"] == pytest.approx(0.90)
    assert m["grass_called_road"] == pytest.approx(1 / 3)
    assert m["road_called_grass"] == pytest.approx(0.0)
    assert m["grass_recall"] == pytest.approx(2 / 3)


def test_agreement_counts_declining_to_label_as_a_cost():
    """`unobserved` in the PREDICTION is not free -- it is what the graded term drops and
    what `blind_arcs` counts, so a labeller that abstains everywhere must not score well.
    """
    truth = np.full((4, 4), ROAD, dtype=np.int16)
    m = agreement(np.full((4, 4), UNOBSERVED, dtype=np.int16), truth)
    assert m["pred_unobserved_frac"] == pytest.approx(1.0)
    assert m["overall_accuracy"] == pytest.approx(0.0)
    assert m["road_recall"] == pytest.approx(0.0)


def test_agreement_refuses_mismatched_shapes():
    """Comparing a resized mask against a full-resolution oracle would score the resize."""
    with pytest.raises(ValueError, match="shape mismatch"):
        agreement(np.zeros((4, 4), np.int16), np.zeros((8, 8), np.int16))
