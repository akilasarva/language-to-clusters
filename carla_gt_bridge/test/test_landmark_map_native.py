"""A map-native landmark must not be answered False just because a prop was spawned.

`_landmark_near` must not conclude "genuinely absent from this world" for EVERY family
once anything has been spawned: that would skip the actor-list lookup where the town's own
traffic lights are, so a mission that spawns a cone could never fire Detect(TrafficLight).

`_landmark_near` is exercised through the class rather than a live node: instantiating
GtCueNode needs rclpy, a region table and a running graph, none of which this rule depends
on. The stub carries exactly the attributes the method reads.
"""
import math
import pytest

from carla_gt_bridge.nodes.gt_cue_node import GtCueNode


class _Param:
    def __init__(self, value):
        self.value = value


class _Stub:
    """Minimal stand-in carrying only what `_landmark_near` touches."""

    def __init__(self, spawned, region_family, cluster=None, landmark_ids=None,
                 object_xy=None, pose=(0.0, 0.0), saw_actor_list=True, radius=8.0,
                 native_regions=None):
        self._spawned = list(spawned)
        self._region_family = dict(region_family)
        self._cluster = cluster
        self._landmark_ids = landmark_ids or {}
        self._object_xy = object_xy or {}
        self._pose = pose
        self._saw_actor_list = saw_actor_list
        self._radius = radius
        self._native_regions = native_regions or {}

    def _spawned_regions(self):
        return self._spawned

    def get_parameter(self, name):
        assert name == "cone_radius_m"
        return _Param(self._radius)


near = GtCueNode._landmark_near
# what the deployment can actually PLACE: cones.town05.json + props.town05.json
PLACEABLE = {60: "cone", 49: "bench"}


def test_a_spawnable_family_that_was_not_spawned_is_still_a_real_absence():
    """Benches are placeable, none was placed here, so False."""
    s = _Stub(spawned=[60], region_family=PLACEABLE)
    assert near(s, "bench") is False


def test_the_spawned_family_is_answered_by_region():
    s = _Stub(spawned=[60], region_family=PLACEABLE, cluster=60)
    assert near(s, "cone") is True
    s2 = _Stub(spawned=[60], region_family=PLACEABLE, cluster=7)
    assert near(s2, "cone") is False


def test_a_map_native_family_falls_through_to_the_actor_list_although_a_prop_was_spawned():
    """traffic_light is in no prop table, so spawning a cone says nothing
    about whether the town has a light next to us -- the actor list does."""
    s = _Stub(spawned=[60], region_family=PLACEABLE, cluster=60,
              landmark_ids={"traffic_light": {11}}, object_xy={11: (3.0, 0.0)})
    assert near(s, "traffic_light") is True


def test_a_map_native_family_is_false_when_its_actors_are_far_away():
    s = _Stub(spawned=[60], region_family=PLACEABLE, cluster=60,
              landmark_ids={"traffic_light": {11}}, object_xy={11: (500.0, 0.0)})
    assert near(s, "traffic_light") is False


def test_unanswerable_stays_none_before_the_actor_list_arrives():
    """None must not collapse to False: a branch would take its default on a
    confident-looking negative."""
    s = _Stub(spawned=[60], region_family=PLACEABLE, cluster=60, saw_actor_list=False)
    assert near(s, "traffic_light") is None


def test_nothing_spawned_still_uses_the_actor_list():
    s = _Stub(spawned=[], region_family=PLACEABLE,
              landmark_ids={"traffic_light": {11}}, object_xy={11: (2.0, 1.0)})
    assert near(s, "traffic_light") is True


# ---- the region-scoped table for map-native landmarks -------------------------------- #
# /carla/objects publishes no pose for traffic lights (the actor list knows they exist,
# but none is located), so proximity cannot answer them at all. The table built by
# scripts/extract_traffic_lights.py is the only path.
LIGHTS = {"traffic_light": {53, 60, 63}}


def test_a_map_native_landmark_is_answered_by_region_even_with_a_cone_spawned():
    s = _Stub(spawned=[60], region_family=PLACEABLE, cluster=63, native_regions=LIGHTS)
    assert near(s, "traffic_light") is True


def test_a_map_native_landmark_is_false_in_a_region_without_one():
    s = _Stub(spawned=[60], region_family=PLACEABLE, cluster=4, native_regions=LIGHTS)
    assert near(s, "traffic_light") is False


def test_the_region_table_beats_the_empty_pose_topic():
    """The actor list knows the id and /carla/objects has no pose for it; without the
    table this would return False and the branch could never fire."""
    s = _Stub(spawned=[60], region_family=PLACEABLE, cluster=53,
              landmark_ids={"traffic_light": {97}}, object_xy={},   # 0 located
              native_regions=LIGHTS)
    assert near(s, "traffic_light") is True


def test_unknown_cluster_is_unanswerable_not_false():
    s = _Stub(spawned=[60], region_family=PLACEABLE, cluster=None, native_regions=LIGHTS)
    assert near(s, "traffic_light") is None


def test_a_spawnable_family_is_unaffected_by_the_native_table():
    s = _Stub(spawned=[60], region_family=PLACEABLE, cluster=60, native_regions=LIGHTS)
    assert near(s, "cone") is True
    assert near(s, "bench") is False
