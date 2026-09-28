"""Tests for offline CARLA map segmentation.

Split into two tiers:

* **Synthetic** — an inline `.xodr` document, so the geometry maths and the
  region/adjacency logic are testable with no CARLA, no Docker, no map file.
* **Town01** — skipped unless the real map is present. Pins the numbers the
  design depends on (26 path / 12 junction regions, T-junctions, the mission
  corridor). Extract it with:

      docker run --rm --entrypoint /bin/cat carlasim/carla:0.9.14 \
        /home/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town01.xodr \
        > carla_gt_bridge/config/Town01.xodr
"""

from __future__ import annotations

import math
import os
import sys

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from carla_gt_bridge.opendrive import (UnsupportedGeometry, GeomSegment,  # noqa: E402
                                       load, parse)
from carla_gt_bridge.segmenter import segment                            # noqa: E402


# --------------------------------------------------------------------------- #
# geometry — no map file needed                                               #
# --------------------------------------------------------------------------- #

def test_line_segment_walks_along_its_heading():
    seg = GeomSegment(s=0, x=10.0, y=5.0, hdg=0.0, length=10.0)
    assert seg.point_at(0) == pytest.approx((10.0, 5.0, 0.0))
    assert seg.point_at(10)[:2] == pytest.approx((20.0, 5.0))
    # heading north
    seg = GeomSegment(s=0, x=0.0, y=0.0, hdg=math.pi / 2, length=4.0)
    assert seg.point_at(4)[:2] == pytest.approx((0.0, 4.0), abs=1e-9)


def test_arc_quarter_circle_geometry():
    """A curvature-k arc of length (pi/2)/k is a quarter circle."""
    k = 0.05                      # radius 20 m
    L = (math.pi / 2) / k
    seg = GeomSegment(s=0, x=0.0, y=0.0, hdg=0.0, length=L, curvature=k)
    x, y, hdg = seg.point_at(L)
    # starting east, curving left (+k): ends at (r, r) heading north
    assert (x, y) == pytest.approx((20.0, 20.0), abs=1e-6)
    assert hdg == pytest.approx(math.pi / 2, abs=1e-9)


def test_arc_polyline_length_matches_declared_length():
    """The maths must be exact, not approximate — centroids depend on it.

    Goes through `Road.sample()` rather than hand-stepping `point_at`, because
    `sample()` is the real code path and it always includes the segment endpoint.
    A hand-rolled loop that stops at the last whole step silently drops the final
    partial metre and looks like a geometry error when it is a sampling error.
    """
    from carla_gt_bridge.opendrive import Road

    seg = GeomSegment(s=0, x=3.0, y=-7.0, hdg=0.4, length=157.55, curvature=-0.012)
    road = Road(road_id=0, length=seg.length, junction=-1, geometry=[seg])
    pts = road.sample(0.5)
    L = sum(math.dist(pts[i][:2], pts[i + 1][:2]) for i in range(len(pts) - 1))
    # chord-vs-arc error at 0.5 m spacing on an r=83 m arc is well under a mm
    assert L == pytest.approx(seg.length, rel=1e-5)
    # and the endpoint is actually reached
    assert math.dist(pts[-1][:2], seg.point_at(seg.length)[:2]) < 1e-9


_SYNTH = """<?xml version="1.0"?>
<OpenDRIVE>
  <header name="Synth"/>
  <road id="1" length="100.0" junction="-1">
    <link><successor elementType="junction" elementId="900"/></link>
    <planView><geometry s="0" x="0" y="0" hdg="0" length="100.0"><line/></geometry></planView>
  </road>
  <road id="2" length="100.0" junction="-1">
    <link><predecessor elementType="junction" elementId="900"/></link>
    <planView><geometry s="0" x="110" y="0" hdg="0" length="100.0"><line/></geometry></planView>
  </road>
  <road id="3" length="50.0" junction="-1">
    <link><predecessor elementType="junction" elementId="900"/></link>
    <planView><geometry s="0" x="105" y="-10" hdg="-1.5707963" length="50.0"><line/></geometry></planView>
  </road>
  <road id="50" length="10.0" junction="900">
    <planView><geometry s="0" x="100" y="0" hdg="0" length="10.0"><line/></geometry></planView>
  </road>
  <junction id="900">
    <connection id="0" incomingRoad="1" connectingRoad="50"/>
    <connection id="1" incomingRoad="2" connectingRoad="50"/>
    <connection id="2" incomingRoad="3" connectingRoad="50"/>
  </junction>
</OpenDRIVE>"""


def test_synthetic_segmentation_shape():
    rm = segment(parse(_SYNTH))
    kinds = sorted(r.kind for r in rm.regions.values())
    assert kinds == ["junction", "path", "path", "path"]
    assert rm.modes() == {"path": [0, 1, 2], "junction": [3]}
    # all three roads link to the junction, so the junction has degree 3
    deg = {}
    for a, b in rm.adjacency:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    assert deg[3] == 3
    assert all(deg[r] == 1 for r in (0, 1, 2))


def test_adjacency_comes_from_road_links_not_junction_connections():
    """Regression: a road linking road->road must still get an edge.

    Deriving adjacency from the junction's own `incomingRoad` list leaves
    map-edge stubs isolated (Town01 has six), because they link to another *road*
    and so never appear as an incomingRoad anywhere.
    """
    doc = _SYNTH.replace(
        '<road id="3" length="50.0" junction="-1">\n'
        '    <link><predecessor elementType="junction" elementId="900"/></link>',
        '<road id="3" length="50.0" junction="-1">\n'
        '    <link><predecessor elementType="road" elementId="2"/></link>')
    rm = segment(parse(doc))
    deg = {}
    for a, b in rm.adjacency:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    assert all(deg.get(r, 0) > 0 for r in rm.regions), "no region may be isolated"


def test_bearing_map_is_degrees_and_both_directions():
    """Degrees, matching nl_planner.taxonomy — the other on-disk convention in
    this repo uses radians, which silently scales every heading by ~57x."""
    rm = segment(parse(_SYNTH))
    bm = rm.bearing_map()
    assert "0-3" in bm and "3-0" in bm
    # region 0 is a straight run east of the junction -> bearing near 0 deg
    assert abs(bm["0-3"]) < 45.0
    assert all(-180.0 <= v <= 180.0 for v in bm.values())
    # opposite directions differ by ~180
    assert abs(abs(bm["0-3"] - bm["3-0"]) - 180.0) < 1e-6


def test_exotic_geometry_raises_rather_than_approximating():
    doc = _SYNTH.replace('<line/>', '<spiral curvStart="0" curvEnd="0.1"/>', 1)
    with pytest.raises(UnsupportedGeometry, match="spiral"):
        parse(doc)


# --------------------------------------------------------------------------- #
# passage derivation — bridges, and why the <bridge> record is not enough      #
# --------------------------------------------------------------------------- #

def _bridge_road(rid, length, span_s, span_len, n_lanes, frag=False):
    """A road with a bridge deck and `n_lanes` driving lanes of 3.5 m each."""
    if frag:
        # how CARLA actually exports a deck: abutting fragments, some 1 cm long
        parts, s = [], span_s
        for L in (0.01, 0.24, span_len - 0.25):
            parts.append(f'<bridge s="{s}" length="{L}" type="concrete"/>')
            s += L
        bridges = "".join(parts)
    else:
        bridges = f'<bridge s="{span_s}" length="{span_len}" type="concrete"/>'
    lanes = "".join('<lane type="driving"><width a="3.5"/></lane>'
                    for _ in range(n_lanes))
    return f"""
  <road id="{rid}" length="{length}" junction="-1">
    <link><successor elementType="junction" elementId="900"/></link>
    <planView><geometry s="0" x="0" y="{200 + rid}" hdg="0" length="{length}">
      <line/></geometry></planView>
    <lanes><laneSection s="0"><right>
      <lane type="shoulder"><width a="1.0"/></lane>{lanes}
      <lane type="sidewalk"><width a="4.0"/></lane>
    </right></laneSection></lanes>
    <objects>{bridges}</objects>
  </road>"""


def _with_roads(*extra):
    return _SYNTH.replace("</OpenDRIVE>", "".join(extra) + "\n</OpenDRIVE>")


def test_narrow_bridge_becomes_passage_wide_one_stays_path():
    """The width test, which is the whole point.

    A 2-lane deck is `passage` (structure close on both sides); a 6-lane elevated
    freeway is `path`, because from the middle of 21 m nothing is close. Without
    this, CARLA's `passage` would mean something different from a `passage` in the
    pedestrian bags and no plan would transfer between them.
    """
    from carla_gt_bridge.segmenter import bridge_passages

    m = parse(_with_roads(_bridge_road(10, 46.7, 5.0, 36.4, n_lanes=2),
                          _bridge_road(11, 814.8, 0.0, 814.8, n_lanes=6)))
    assert m.roads[10].driving_width == pytest.approx(7.0)
    assert m.roads[11].driving_width == pytest.approx(21.0)
    assert bridge_passages(m) == {"road:10": "passage"}

    labels = {r.source: r.label for r in segment(m).regions.values()}
    assert labels["road:10"] == "passage"
    assert labels["road:11"] == "path"


def test_short_bridge_is_not_a_passage_even_if_narrow():
    """An expansion joint is not something you traverse."""
    from carla_gt_bridge.segmenter import bridge_passages

    m = parse(_with_roads(_bridge_road(12, 40.0, 0.0, 2.0, n_lanes=2)))
    assert bridge_passages(m) == {}


def test_exported_bridge_fragments_merge_into_one_deck():
    """Regression: CARLA splits one deck into abutting spans, some 1 cm long.

    Reading `<bridge>` records individually makes the longest span 1 cm and the
    deck vanishes; summing them without merging would instead accumulate across
    genuine gaps. Merge at parse time, then measure the longest run.
    """
    m = parse(_with_roads(_bridge_road(13, 46.7, 5.0, 36.4, n_lanes=2, frag=True)))
    assert len(m.roads[13].bridge_spans) == 1
    assert m.roads[13].longest_bridge == pytest.approx(36.4, abs=1e-6)
    assert segment(m).regions[
        {r.source: r.rid for r in segment(m).regions.values()}["road:13"]
    ].label == "passage"


def test_explicit_tags_override_derived_ones():
    """`along_edge` can only ever be hand-tagged, so the override must hold."""
    m = parse(_with_roads(_bridge_road(14, 46.7, 5.0, 36.4, n_lanes=2)))
    rm = segment(m, tags={"road:14": "along_edge"})
    labels = {r.source: r.label for r in rm.regions.values()}
    assert labels["road:14"] == "along_edge"


def test_auto_passage_is_a_noop_on_a_town_without_bridges():
    before = segment(parse(_SYNTH), auto_passage=False).modes()
    after = segment(parse(_SYNTH), auto_passage=True).modes()
    assert before == after


def test_narrow_roads_are_reported_as_candidates_not_labelled_passage():
    """Narrowness is necessary for `passage` but not sufficient.

    The rule is "structure close on BOTH sides". A 5.8 m service road across an open
    field is a `path`, and whether anything flanks it is invisible offline — buildings
    are Unreal level geometry and appear nowhere in the `.xodr`. A bridge is the one case
    where flanking is implied (a deck has parapets), which is why bridges auto-label and
    bare narrowness does not.
    """
    from carla_gt_bridge.segmenter import bridge_passages, narrow_candidates

    # a narrow road with NO bridge record
    doc = _with_roads(_bridge_road(20, 40.0, 0.0, 0.0, n_lanes=1))
    doc = doc.replace('<bridge s="0.0" length="0.0" type="concrete"/>', '')
    m = parse(doc)
    assert m.roads[20].driving_width == pytest.approx(3.5)
    cands = [c["road_id"] for c in narrow_candidates(m)]
    assert 20 in cands, "a 3.5 m single-lane road should be reported"
    assert bridge_passages(m) == {}, "but must NOT be auto-labelled a passage"
    assert segment(m).regions[
        {r.source: r.rid for r in segment(m).regions.values()}["road:20"]
    ].label == "path"


def test_sidewalks_disqualify_a_narrow_carriageway():
    """A 2-lane street with a footway each side is a street, not an alley.

    Town01's roads are 8.0 m of carriageway — narrower than Town05's freeway — but 16.6 m
    once sidewalks are counted. Judging on carriageway alone would call every ordinary
    street a passage.
    """
    from carla_gt_bridge.segmenter import narrow_candidates

    m = parse(_with_roads(_bridge_road(21, 40.0, 0.0, 0.0, n_lanes=2)))
    assert m.roads[21].driving_width == pytest.approx(7.0)
    assert m.roads[21].total_width == pytest.approx(12.0)   # + 1 m shoulder + 4 m footway
    assert [c["road_id"] for c in narrow_candidates(m)] == []


def test_nondrivable_roads_are_surfaced():
    """A `path` region with no drivable lane would send the vehicle off-road.

    Town03 has five, one of them 967 m long: their lanes are typed `none`, so the geometry
    exists but no vehicle may use it. They still pass the `junction == -1` test that makes
    a `path` region.
    """
    from carla_gt_bridge.segmenter import nondrivable_regions

    doc = _with_roads(_bridge_road(22, 60.0, 0.0, 0.0, n_lanes=1))
    doc = doc.replace('<lane type="driving"><width a="3.5"/></lane>',
                      '<lane type="none"><width a="3.5"/></lane>')
    m = parse(doc)
    assert m.roads[22].driving_width == 0.0
    assert m.roads[22].total_width > 0.0
    assert nondrivable_regions(m) == [22]


def test_widths_take_the_widest_lane_section():
    """A road that gains a turn lane part-way must not be judged on its first section."""
    two = '<lane type="driving"><width a="3.5"/></lane>' * 2
    doc = _with_roads(f"""
  <road id="23" length="80.0" junction="-1">
    <link><successor elementType="junction" elementId="900"/></link>
    <planView><geometry s="0" x="0" y="500" hdg="0" length="80.0"><line/></geometry></planView>
    <lanes>
      <laneSection s="0"><right>{two}</right></laneSection>
      <laneSection s="40"><right>{two * 2}</right></laneSection>
    </lanes>
  </road>""")
    m = parse(doc)
    assert m.roads[23].driving_width == pytest.approx(14.0)


def test_corridor_rejects_unknown_ids():
    rm = segment(parse(_SYNTH))
    with pytest.raises(KeyError):
        rm.corridor([0, 999])


def test_corridor_keeps_only_internal_adjacency():
    rm = segment(parse(_SYNTH))
    sub = rm.corridor([0, 3])
    assert sorted(sub.regions) == [0, 3]
    assert sub.adjacency == {(0, 3)}


# --------------------------------------------------------------------------- #
# Town01 — the numbers the design depends on                                  #
# --------------------------------------------------------------------------- #

def _town01():
    for cand in (os.path.join(PKG, "config", "Town01.xodr"),
                 os.environ.get("TOWN01_XODR", "")):
        if cand and os.path.exists(cand):
            return load(cand)
    pytest.skip("Town01.xodr not present — see this file's docstring")


def test_town01_region_counts():
    """Pins the segmentation. Cross-checked against the raw .xodr road table."""
    rm = segment(_town01())
    modes = rm.modes()
    assert len(modes["path"]) == 26
    assert len(modes["junction"]) == 12
    assert len(rm.regions) == 38


def test_town01_junctions_are_all_t_junctions():
    """All 12 are 3-approach T-junctions — NOT 4-way crossroads.

    The 8 connections each junction lists do not imply 4-way. It matters: at a
    T-junction "go straight or turn right" only both exist from certain approaches,
    so a branch mission cannot be sited arbitrarily.
    """
    rm = segment(_town01())
    deg = {}
    for a, b in rm.adjacency:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    for r in rm.regions.values():
        if r.kind == "junction":
            assert deg[r.rid] == 3, f"junction region {r.rid} has degree {deg[r.rid]}"


def test_town01_no_isolated_regions_and_streets_join_at_both_ends():
    rm = segment(_town01())
    deg = {}
    for a, b in rm.adjacency:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    assert all(deg.get(r, 0) > 0 for r in rm.regions), "no region may be isolated"
    for r in rm.regions.values():
        if r.kind == "path":
            assert deg[r.rid] == 2, f"path region {r.rid} degree {deg[r.rid]}"


def test_town01_mission_corridor_is_connected():
    """The cone-mission corridor: 3 -> j29 -> 2 -> j27 -> {1 straight | 25 right}."""
    rm = segment(_town01())
    sub = rm.corridor([3, 29, 2, 27, 1, 25])
    assert len(sub.regions) == 6
    for pair in ((3, 29), (2, 29), (2, 27), (1, 27), (25, 27)):
        a, b = min(pair), max(pair)
        assert (a, b) in sub.adjacency, f"expected {a}-{b} adjacent"


def test_town01_decision_junction_offers_straight_and_right():
    """At junction region 27 approached from region 2, both maneuvers must exist,
    or the cone branch mission has nothing to choose between."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "map_regions", os.path.join(PKG, "scripts", "map_regions.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    m = _town01()
    rm = segment(m)
    rows = [r for r in mod.maneuvers(m, rm) if r[1] == 27 and r[2] == 2]
    assert rows, "no approach from region 2 into junction region 27"
    kinds = {k for k, _t, _r in rows[0][3]}
    assert {"straight", "right"} <= kinds
    exits = {k: rid for k, _t, rid in rows[0][3]}
    assert exits["straight"] == 1
    assert exits["right"] == 25


def test_town01_has_no_bridge_and_no_narrow_road_at_all():
    """Town01's road network is perfectly uniform and flat.

    All 26 roads are identical — 8.0 m carriageway (2 x 4 m), 16.6 m total with a shoulder
    and a sidewalk each side — with zero elevation change anywhere in the town and no
    `<bridge>` record. So there is no bridge deck and no alley in Town01's *road network*,
    whatever the scenery looks like: any bridge-like structure is flat, unannotated Unreal
    geometry, and it would fail the width test regardless.
    """
    from carla_gt_bridge.segmenter import (bridge_passages, narrow_candidates,
                                           nondrivable_regions)

    m = _town01()
    widths = {(round(r.driving_width, 2), round(r.total_width, 2))
              for r in m.path_roads}
    assert widths == {(8.0, 16.6)}, f"expected one uniform width, got {widths}"
    assert all(r.longest_bridge == 0.0 for r in m.roads.values())
    assert narrow_candidates(m) == []
    assert bridge_passages(m) == {}
    assert nondrivable_regions(m) == []


def test_town01_waypoint_table_shape():
    rm = segment(_town01())
    wp = rm.waypoints()
    assert wp.ndim == 2 and wp.shape[1] == 3
    assert len(wp) > 2000
    assert set(wp[:, 2].astype(int)) == set(rm.regions)
