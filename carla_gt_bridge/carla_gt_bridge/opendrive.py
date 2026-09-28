"""Minimal OpenDRIVE (.xodr) reader — enough to sample CARLA road reference lines.

Why this exists instead of calling CARLA
----------------------------------------
The region segmentation (which road/junction becomes which cluster id) only needs
the road network's geometry, and every town's `.xodr` ships inside the CARLA
Docker image. Parsing it directly means the whole segmenter — plus its tests and
the region plot — runs **offline: no CARLA server, no GPU, deterministically**,
and region geometry is derived from the map rather than hand-guessed.

All 16 shipped towns use **only `line` and `arc`** primitives, no clothoids or
polynomials, so
the ~40 lines of maths below are complete for the whole town set — which is what
makes changing towns a config change rather than a porting job. `spiral` / `poly3`
/ `paramPoly3` are still detected and raise rather than being silently
approximated: a wrong road shape would poison every centroid downstream, so
failing loudly is the honest behaviour if a custom map ever uses one.

Coordinate frame
----------------
Raw OpenDRIVE metres: x east, **y north**, headings counter-clockwise — the ordinary
right-handed frame, which is also what ROS and `carla_ros_bridge` use. It is NOT
CARLA's own frame; CARLA negates y (verified: Town01's `.xodr` spans y in
[-328.6, 0.0] while a running server reports y ~ +273..+330 for the same town).
**No sign flips and no offsets are applied here.** Every conversion belongs in
exactly one place (see `frames.py`, which carries the table of which data is in
which frame); scattered conversions are how a mirrored-target bug happens.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


class UnsupportedGeometry(NotImplementedError):
    """An OpenDRIVE primitive this reader cannot evaluate exactly."""


@dataclass(frozen=True)
class GeomSegment:
    """One `<geometry>` element of a road's reference line (planView)."""

    s: float          # arc length along the road where this segment starts
    x: float          # start point
    y: float
    hdg: float        # start heading, radians
    length: float
    curvature: float = 0.0   # 0.0 => straight line

    def point_at(self, ds: float) -> tuple[float, float, float]:
        """Return (x, y, heading) at ``ds`` metres into this segment."""
        if abs(self.curvature) < 1e-12:
            return (self.x + ds * math.cos(self.hdg),
                    self.y + ds * math.sin(self.hdg),
                    self.hdg)
        # Arc of constant curvature k: the reference line is a circle of radius
        # 1/k, entered at (x, y) with heading hdg.
        k = self.curvature
        h1 = self.hdg + k * ds
        return (self.x + (math.sin(h1) - math.sin(self.hdg)) / k,
                self.y - (math.cos(h1) - math.cos(self.hdg)) / k,
                h1)


@dataclass
class Road:
    road_id: int
    length: float
    #: OpenDRIVE junction attribute: -1 for a normal road, else the junction id
    #: this road is a *connecting road inside*. This is the field that splits the
    #: network into `path` regions (-1) and `junction` regions (>= 0).
    junction: int
    geometry: list[GeomSegment] = field(default_factory=list)
    #: `<link>` neighbours as (element_type, element_id), element_type in
    #: {"road", "junction"}. This is the CANONICAL road topology and the correct
    #: source of region adjacency.
    #:
    #: Using the junction's own `<connection incomingRoad=...>` list instead
    #: leaves map-edge stubs isolated, because their links point at another
    #: *road*, not a junction, so they never appear as an incomingRoad.
    links: list[tuple[str, int]] = field(default_factory=list)
    #: Merged `<objects><bridge s= length=>` spans as (s_start, s_end) metres.
    #: OpenDRIVE *does* carry a bridge record and CARLA populates it (Town04/05/07),
    #: so bridges are derivable rather than hand-guessed — but see
    #: `segmenter.bridge_passages` for why the record alone is not enough.
    bridge_spans: list[tuple[float, float]] = field(default_factory=list)
    #: Summed width of `type="driving"` lanes, metres (widest lane section).
    #: This is the *narrowness* measurement the `passage` label depends on:
    #: `passage` means structure close on BOTH sides, which a 6-lane deck is not.
    driving_width: float = 0.0
    #: Summed width of ALL lanes including shoulders and sidewalks, metres. A road with a
    #: sidewalk each side is a street, not an alley, however narrow its carriageway is —
    #: so both numbers are needed to talk about shape.
    total_width: float = 0.0

    @property
    def in_junction(self) -> bool:
        return self.junction >= 0

    @property
    def bridge_length(self) -> float:
        """Total road length carried on a bridge, metres."""
        return sum(b - a for a, b in self.bridge_spans)

    @property
    def longest_bridge(self) -> float:
        """Longest single contiguous bridge span, metres.

        Prefer this over :attr:`bridge_length` when deciding whether a road *is* a
        bridge: CARLA's exporter emits the deck as a run of adjacent fragments, some
        as short as 1 cm, so the count of spans is meaningless and their sum can
        accumulate across gaps. Merging happens at parse time; this reads the
        longest merged run.
        """
        return max((b - a for a, b in self.bridge_spans), default=0.0)

    def sample(self, step: float = 2.0) -> list[tuple[float, float, float]]:
        """Sample the reference line every ``step`` metres.

        Reference line, not lane centres: a lane sits ~1.75 m to one side, which
        is far below the tens-of-metres separation between regions, so it cannot
        change which region a point belongs to. Skipping lane-width maths keeps
        this exact rather than approximately right.
        """
        out: list[tuple[float, float, float]] = []
        for seg in self.geometry:
            if seg.length <= 0:
                continue
            n = max(1, int(math.ceil(seg.length / step)))
            for i in range(n + 1):
                ds = min(seg.length, i * step)
                out.append(seg.point_at(ds))
                if ds >= seg.length:
                    break
        return out


@dataclass
class Junction:
    junction_id: int
    #: (incoming_road_id, connecting_road_id) pairs
    connections: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class OpenDriveMap:
    name: str
    roads: dict[int, Road]
    junctions: dict[int, Junction]

    @property
    def path_roads(self) -> list[Road]:
        """Roads outside any junction — these become `path` regions."""
        return [r for r in self.roads.values() if not r.in_junction]

    @property
    def junction_roads(self) -> dict[int, list[Road]]:
        """junction_id -> the connecting roads inside it."""
        out: dict[int, list[Road]] = {}
        for r in self.roads.values():
            if r.in_junction:
                out.setdefault(r.junction, []).append(r)
        return out


_SUPPORTED = {"line", "arc"}

#: Two adjacent bridge fragments this close together are one deck. CARLA exports a
#: single physical bridge as a run of abutting spans (down to 1 cm long), so without
#: merging, `longest_bridge` reports a fragment instead of the deck.
_BRIDGE_JOIN_M = 0.5


def _merge_spans(spans: list[tuple[float, float]],
                 join: float = _BRIDGE_JOIN_M) -> list[tuple[float, float]]:
    """Merge overlapping / abutting (start, end) intervals."""
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + join:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _section_widths(ls: ET.Element) -> tuple[float, float]:
    """``(driving_width, total_width)`` of one `<laneSection>`, both sides summed."""
    drive = total = 0.0
    for side in ("left", "right"):
        el = ls.find(side)
        if el is None:
            continue
        for lane in el.findall("lane"):
            w = lane.find("width")
            if w is None:
                continue
            a = float(w.get("a", 0.0))
            total += a
            if lane.get("type") == "driving":
                drive += a
    return drive, total


def _road_widths(road_el: ET.Element) -> tuple[float, float]:
    """Widest ``(driving, total)`` across all of a road's lane sections.

    Widest rather than first: a CARLA road that gains a turn lane part-way along has
    several sections, and taking only the first under-reports it. The passage test needs
    the order of magnitude (7 m vs 21 m), so one number per road is enough — but it should
    be the generous one, since a road is only "narrow" if it is narrow everywhere.

    ``driving`` counts only ``type="driving"`` lanes. That distinction matters: Town03 has
    five long "roads" whose lanes are typed ``none`` — 3.5 m of geometry each side that no
    vehicle may use. They come out with driving width 0.0, which is correct and is how
    :func:`segmenter.nondrivable_regions` finds them.
    """
    best = (0.0, 0.0)
    for ls in road_el.findall("lanes/laneSection"):
        d, t = _section_widths(ls)
        if d > best[0] or (d == best[0] and t > best[1]):
            best = (d, t)
    return best


def parse(xodr_text: str, name: str = "") -> OpenDriveMap:
    """Parse an .xodr document. Raises UnsupportedGeometry on exotic primitives."""
    root = ET.fromstring(xodr_text)

    if not name:
        header = root.find("header")
        if header is not None:
            name = header.get("name") or ""

    roads: dict[int, Road] = {}
    for r in root.findall("road"):
        rid = int(r.get("id"))
        road = Road(road_id=rid,
                    length=float(r.get("length", 0.0)),
                    junction=int(r.get("junction", -1)))
        link = r.find("link")
        for tag in ("predecessor", "successor"):
            e = link.find(tag) if link is not None else None
            if e is None:
                continue
            etype, eid = e.get("elementType"), e.get("elementId")
            if etype in ("road", "junction") and eid is not None:
                road.links.append((etype, int(eid)))
        pv = r.find("planView")
        for g in (pv.findall("geometry") if pv is not None else []):
            kinds = [c.tag for c in g]
            exotic = [k for k in kinds if k not in _SUPPORTED]
            if exotic:
                raise UnsupportedGeometry(
                    f"road {rid}: geometry primitive(s) {exotic} are not "
                    f"implemented. Only {sorted(_SUPPORTED)} are exact. Refusing "
                    f"to approximate — a wrong road shape corrupts every centroid "
                    f"derived from it."
                )
            arc = g.find("arc")
            road.geometry.append(GeomSegment(
                s=float(g.get("s", 0.0)),
                x=float(g.get("x")), y=float(g.get("y")),
                hdg=float(g.get("hdg")),
                length=float(g.get("length", 0.0)),
                curvature=float(arc.get("curvature")) if arc is not None else 0.0,
            ))
        road.bridge_spans = _merge_spans(
            [(float(b.get("s", 0.0)),
              float(b.get("s", 0.0)) + float(b.get("length", 0.0)))
             for b in r.iter("bridge")])
        road.driving_width, road.total_width = _road_widths(r)
        roads[rid] = road

    junctions: dict[int, Junction] = {}
    for j in root.findall("junction"):
        jid = int(j.get("id"))
        jn = Junction(junction_id=jid)
        for c in j.findall("connection"):
            jn.connections.append((int(c.get("incomingRoad")),
                                   int(c.get("connectingRoad"))))
        junctions[jid] = jn

    return OpenDriveMap(name=name, roads=roads, junctions=junctions)


def load(path: str) -> OpenDriveMap:
    with open(path, "r", encoding="utf-8") as f:
        return parse(f.read(), name=path.split("/")[-1].replace(".xodr", ""))


__all__ = ["GeomSegment", "Junction", "OpenDriveMap", "Road",
           "UnsupportedGeometry", "load", "parse"]
