"""Tests for the multi-channel polar feature.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest clustering/test/test_multichannel_features.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "clustering"))

from multichannel_features import (ChannelSpec, PolarFeatureConfig,  # noqa: E402
                                   channel_coverage, feature_vector, polar_channels)

H = 2.5   # sensor height used across these tests; ground therefore sits at z = -2.5


def _ring(r, z, n=360):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(a), r * np.sin(a), np.full(n, z)], axis=1)


def test_a_config_without_a_sensor_height_is_refused():
    """The bands are robot-relative, so they are meaningless without it. A default would
    silently compute a plausible feature over the wrong slice."""
    with pytest.raises(TypeError):
        PolarFeatureConfig()          # type: ignore[call-arg]


def test_min_and_max_reducers_answer_different_questions():
    """Two concentric rings of ground: `min` reports the near one, `max` the far one.
    The near ground return is the blind-zone edge and barely moves, so `max` is the
    informative reducer for ground extent."""
    pts = np.vstack([_ring(4.0, -H + 0.05), _ring(18.0, -H + 0.05)])
    cfg = PolarFeatureConfig(sensor_height=H, channels=[
        ChannelSpec("near", -H - 0.1, -H + 0.3, "min"),
        ChannelSpec("far", -H - 0.1, -H + 0.3, "max")])
    ch, _ = polar_channels(pts, cfg)
    assert np.allclose(ch[0], 4.0, atol=0.3), ch[0][:5]
    assert np.allclose(ch[1], 18.0, atol=0.3), ch[1][:5]


def test_ground_extent_sees_a_junction_as_more_open_than_a_corridor():
    """A corridor is drivable along one axis; a crossroads along two. Ground extent must be
    larger overall for the crossroads. This says the CHANNEL responds to the geometry as
    intended -- NOT that it separates real classes, which needs ground truth.
    """
    def arm(dx, dy, n=400):
        t = np.linspace(2, 22, n)
        return np.stack([t * dx, t * dy, np.full(n, -H + 0.05)], axis=1)

    cfg = PolarFeatureConfig(sensor_height=H, channels=[
        ChannelSpec("g", -H - 0.1, -H + 0.3, "max")])
    corridor = np.vstack([arm(1, 0), arm(-1, 0)])
    cross = np.vstack([arm(1, 0), arm(-1, 0), arm(0, 1), arm(0, -1)])
    g_cor, _ = polar_channels(corridor, cfg)
    g_cro, _ = polar_channels(cross, cfg)
    assert g_cro[0].sum() > g_cor[0].sum(), "a crossroads must read as more open"


def test_coverage_is_reported_and_an_empty_band_reads_zero():
    """A band with no returns must report coverage 0.0, not a confident vector of fill
    values."""
    pts = _ring(8.0, -H + 0.05)
    cfg = PolarFeatureConfig(sensor_height=H, channels=[
        ChannelSpec("ground", -H - 0.1, -H + 0.3, "max"),
        ChannelSpec("high", -H + 5.0, -H + 9.0, "min")])
    cov = channel_coverage(pts, cfg)
    assert cov["ground"] > 0.9
    assert cov["high"] == 0.0


def test_a_filled_sector_is_not_reported_as_measured():
    """A filled sector and a measured one at the same value are different observations.
    Without the mask, a saturated sector contributes zero frame-to-frame change no
    matter what the world did."""
    half = _ring(9.0, -H + 0.05)
    half = half[half[:, 0] > 0]                      # returns on one side only
    cfg = PolarFeatureConfig(sensor_height=H, channels=[
        ChannelSpec("g", -H - 0.1, -H + 0.3, "max")])
    ch, valid = polar_channels(half, cfg)
    assert 0.2 < valid[0].mean() < 0.8, valid[0].mean()
    # Against the channel's OWN fill, not a hardcoded max_range: an extent channel fills
    # with 0 ("no surface seen") and an obstacle channel with max_range ("nothing out to
    # here").
    assert np.allclose(ch[0][~valid[0]], cfg.channels[0].fill_value(cfg.max_range))


def test_count_normalisation_does_not_leak_the_sensor_point_budget():
    """Density must be RELATIVE: doubling the returns without changing the geometry must
    not change the feature, or a denser sensor reads as a different place.

    Tested as the invariant itself rather than per-element equality. Resampling a ring at
    a different density moves points across sector boundaries, so individual sectors
    legitimately differ by one return (pure quantisation). What must hold exactly is that the channel sums to the same
    total (share-of-returns is scale-free), and that the SHAPE is preserved.
    """
    cfg = PolarFeatureConfig(sensor_height=H, channels=[
        ChannelSpec("c", -H - 0.1, -H + 0.3, "count")])
    a = feature_vector(_ring(7.0, -H + 0.05, n=360), cfg)
    b = feature_vector(_ring(7.0, -H + 0.05, n=720), cfg)
    assert np.isclose(a.sum(), b.sum()), (a.sum(), b.sum())      # exact scale invariance
    assert np.corrcoef(a, b)[0, 1] > 0.95                        # same shape
    # and the giveaway a leak would produce: a factor tracking the point count
    assert not np.isclose(b.sum(), 2 * a.sum())


def test_the_feature_is_channel_major_and_correctly_sized():
    cfg = PolarFeatureConfig(sensor_height=H, num_sectors=36)
    v = feature_vector(_ring(6.0, -H + 0.05), cfg)
    assert v.shape == (len(cfg.channels) * 36,)
    assert np.isfinite(v).all()
