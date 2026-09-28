"""Turn a CARLA road network into cluster regions, centroids and bearings.

The segmentation is CARLA's own ground truth, not a clustering algorithm:

* **`junction` region** — one per OpenDRIVE `<junction>`. Its id is CARLA's own
  stable, globally-unique id for that *physical* intersection, which is what
  makes "the **second** intersection" expressible with no custom logic: two
  distinct intersections along a route simply have two distinct ids.
* **`path` region** — one per road with `junction == -1`, i.e. a street segment
  between intersections. Grouped by road, so both directions of travel share one
  region: the cluster vocabulary describes the *shape* of navigable space, not
  lanes.
* **`passage` region** ← a road carrying a long enough bridge deck that is also
  *narrow*. Derived, not hand-guessed: OpenDRIVE does carry a `<bridge>` record and
  CARLA populates it. But the record alone is not the label — see
  :func:`bridge_passages`, which applies the vocabulary's own width test.
* **`along_edge` / `open_space`** — **not derivable offline.** `along_edge` means a
  structure close on exactly one side, and buildings are Unreal level geometry: they
  appear nowhere in the `.xodr` (its only objects are road furniture — curbs,
  crosswalks, stop lines, signs) and they are not actors, so the bridge's actor list
  will not report them either. These stay hand-tagged via ``tags``.

The labels follow `bev_pipeline/config/nav_modes.yaml`'s decision rule verbatim, so
one vocabulary covers both the pedestrian bags and the car: structure close on BOTH
sides is `passage`, on ONE side is `along_edge`, 3+ open directions is `junction`,
and a channeled way with neither side close is `path` — which is what a road is.

Everything is in **planar** metres: +x east, +y north, headings counter-clockwise —
the OpenDRIVE convention, which is also the ROS convention the `carla_ros_bridge`
publishes in. It is NOT CARLA's own frame: CARLA negates y (verified — Town01's
`.xodr` spans y in [-328.6, 0.0] while a running server reports y ~ +273..+330 for the
same town). No sign flips or offsets are applied here; see `frames.py` for the table of
which data is in which frame, and `opendrive.py` for why that discipline matters.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .opendrive import OpenDriveMap

#: Cluster vocabulary, shared with bev_pipeline/config/nav_modes.yaml. The
#: pedestrian names are used for the car too: the vocabulary is geometric, so it
#: is agent-agnostic — a driving lane is a `path`, an intersection is a `junction`.
#: `approach` stays a PLAIN label here, like every other. The pretty names the robot
#: taxonomy uses -- `Intersection: Approach/Enter`, `Intersection: In` -- are applied by
#: `taxonomy_export.ROAD_HIERARCHY`, which is keyed by exactly these plain labels and
#: also carries the subsumption. Emitting the pretty strings from the segmenter would
#: skip that lattice: `path` would cover only some regions and no `junction` mode would
#: exist, so any step naming `junction` would fail to ground.
LABELS = ("open_space", "path", "along_edge", "passage", "junction", "approach", "other")


@dataclass
class Region:
    rid: int                     # cluster id published on /predicted_cluster
    kind: str                    # "junction" | "path"
    label: str                   # a LABELS entry (tag map may override)
    source: str                  # "junction:7" or "road:12" — provenance
    points: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def centroid(self) -> tuple[float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (sum(xs) / len(xs), sum(ys) / len(ys))


@dataclass
class RegionMap:
    town: str
    regions: dict[int, Region]
    #: undirected adjacency between region ids, from the junction connection table
    adjacency: set[tuple[int, int]]

    # -- lookups ---------------------------------------------------------- #

    def centroids(self) -> dict[str, list[float]]:
        """`{"<rid>": [x, y, 0.0]}` in planar metres (+y north), for the cluster map YAML."""
        return {str(r.rid): [*r.centroid, 0.0] for r in self.regions.values()}

    def bearing_map(self) -> dict[str, float]:
        """`{"a-b": degrees}` for every adjacent ordered pair.

        DEGREES, matching `nl_planner.taxonomy`'s own docstring. The other on-disk
        convention in this repo stores radians, which silently multiplies every
        heading by ~57x if the two are crossed.
        """
        out: dict[str, float] = {}
        for a, b in self.adjacency:
            for src, dst in ((a, b), (b, a)):
                ax, ay = self.regions[src].centroid
                bx, by = self.regions[dst].centroid
                out[f"{src}-{dst}"] = math.degrees(math.atan2(by - ay, bx - ax))
        return out

    def modes(self) -> dict[str, list[int]]:
        """`{label: [rid, ...]}` — the `modes` block of a cluster_map YAML."""
        out: dict[str, list[int]] = {}
        for r in sorted(self.regions.values(), key=lambda r: r.rid):
            out.setdefault(r.label, []).append(r.rid)
        return out

    def waypoints(self):
        """(N,3) float array of x, y, rid — the runtime KD-tree lookup table."""
        import numpy as np

        rows = [(p[0], p[1], float(r.rid))
                for r in self.regions.values() for p in r.points]
        return np.asarray(rows, dtype=float)

    # -- scoping ---------------------------------------------------------- #

    def corridor(self, rids: list[int]) -> "RegionMap":
        """Scope to an explicit, ordered region sequence — the mission's corridor.

        Prefer this over :meth:`sub_route` for missions. BFS-by-hops expands in
        *every* direction, so on Town01 a 2-hop neighbourhood of a top-edge street
        also pulls in regions on the far side of the map; a mission travels one
        way through a specific chain of regions.
        """
        keep = set(rids)
        missing = keep - set(self.regions)
        if missing:
            raise KeyError(f"unknown region ids: {sorted(missing)}")
        return RegionMap(
            town=self.town,
            regions={k: v for k, v in self.regions.items() if k in keep},
            adjacency={(a, b) for (a, b) in self.adjacency
                       if a in keep and b in keep},
        )

    def sub_route(self, start_rid: int, hops: int) -> "RegionMap":
        """Breadth-first sub-map ``hops`` regions out from ``start_rid``.

        Needed because a whole town has ~38 regions, and the MPC scores rollouts
        by nearest centroid — a test that degrades as centroids crowd together.
        A mission only needs its own corridor (5-15 regions), so scope to it.
        """
        keep = {start_rid}
        frontier = deque([(start_rid, 0)])
        while frontier:
            rid, d = frontier.popleft()
            if d >= hops:
                continue
            for a, b in self.adjacency:
                nxt = b if a == rid else (a if b == rid else None)
                if nxt is not None and nxt not in keep:
                    keep.add(nxt)
                    frontier.append((nxt, d + 1))
        return RegionMap(
            town=self.town,
            regions={k: v for k, v in self.regions.items() if k in keep},
            adjacency={(a, b) for (a, b) in self.adjacency
                       if a in keep and b in keep},
        )


#: A `passage` must be narrow. Measured as summed DRIVING-lane width, so shoulders
#: and sidewalks do not inflate it: 2 lanes + shoulders ~= 9 m of clearance, which is
#: "close on both sides"; 6 lanes = 21 m, from the middle of which nothing is close.
#: The value sits in the wide empty gap between CARLA's two populations (7 m vs 21/28 m).
MAX_PASSAGE_WIDTH_M = 10.0

#: A `passage` must be long enough to *traverse* — a deck you cross, not a expansion
#: joint. Also discards CARLA's export fragments (some are 1 cm long).
MIN_PASSAGE_SPAN_M = 15.0


def bridge_passages(m: OpenDriveMap,
                    max_width: float = MAX_PASSAGE_WIDTH_M,
                    min_span: float = MIN_PASSAGE_SPAN_M) -> dict[str, str]:
    """Roads whose bridge deck qualifies as a `passage`, as a ``tags`` mapping.

    Two conditions, both from the vocabulary's own decision rule:

    1. a contiguous bridge span of at least ``min_span`` metres, and
    2. driving width at most ``max_width`` metres — *narrow*.

    Condition 2 is the one that matters, and it is why `<bridge>` cannot be trusted
    on its own. Across the towns:

    ===========  ======  =========  ==============  ==============
    road         length  longest    driving width   verdict
    ===========  ======  =========  ==============  ==============
    Town05 r37   814.8   814.8 m    21.0 m (6 lane)  `path` — freeway deck
    Town05 r36    12.2    12.2 m    21.0 m (6 lane)  `path` — too wide AND too short
    Town04 r39   133.9    49.9 m    28.0 m (8 lane)  `path` — freeway overpass
    Town07 r45    46.7    36.4 m     7.0 m (2 lane)  **`passage`**
    ===========  ======  =========  ==============  ==============

    So CARLA contains exactly one bridge that is a `passage` under this rule, and
    it is in rural Town07. Calling a 21 m elevated freeway a `passage` would break
    the property the whole vocabulary exists to preserve — that a CARLA
    `passage` means the same shape as a `passage` in the pedestrian bags (an alley,
    a footbridge, a corridor), so a plan written for one transfers to the other.

    Junction-internal roads are excluded: they are already `junction` regions, and
    those *are* branch points regardless of what carries them.
    """
    out: dict[str, str] = {}
    for road in m.path_roads:
        if (road.longest_bridge >= min_span
                and 0.0 < road.driving_width <= max_width):
            out[f"road:{road.road_id}"] = "passage"
    return out


#: A road this narrow OVERALL — carriageway plus shoulders plus any footway — is an alley
#: or lane rather than a street. Town06's service roads are 5.5-5.8 m total with a single
#: 3.2-3.5 m lane and no sidewalk; Town01's streets are 16.6 m. Nothing sits in between.
MAX_NARROW_TOTAL_M = 10.0


def narrow_candidates(m: OpenDriveMap, max_total: float = MAX_NARROW_TOTAL_M,
                      min_length: float = 15.0) -> list[dict]:
    """Roads narrow enough to be a `passage` **if** something flanks them.

    Reported, NOT auto-labelled — and that distinction is the whole point. Narrowness is
    necessary for `passage` but not sufficient: the rule is "structure close on BOTH
    sides", and a 5.8 m service road across an open field is a `path`. Whether anything
    flanks it is invisible offline, because buildings and walls are Unreal level geometry
    and appear nowhere in the `.xodr`.

    A bridge is the one case where the flanking IS implied — a deck has parapets both
    sides — which is why :func:`bridge_passages` auto-labels and this does not.

    Across the town set (roads over 15 m, total width <= 10 m):

    ==========  =====  ==========================================================
    town        count  character
    ==========  =====  ==========================================================
    Town01/02       0  every road identical: 8.0 m carriageway, 16.6 m with
                       sidewalks, perfectly flat. No alleys, no bridges at all.
    Town06         13  **narrowest in CARLA** — single 3.2-3.5 m lane, 5.5-5.8 m
                       total, no sidewalk. Service roads / alleys.
    Town07         19  rural lanes, 6.2-7.0 m carriageway, 7.2-9.0 m total.
    Town04          5  7.0 m carriageway, 8.0-8.6 m total.
    Town03          5  9.3 m total but ZERO drivable lanes - see
                       :func:`nondrivable_regions`.
    ==========  =====  ==========================================================

    So "CARLA has one `passage`" is true of *bridge-flagged* roads only. By shape there are
    ~40 candidates, and the narrowest are in Town06, which has no bridge records at all.
    """
    out = []
    for road in m.path_roads:
        # total_width == 0 means the road carries no <lanes> data at all — unmeasured, not
        # narrow. Treating "no information" as "narrow" would report every stub road.
        if (road.length < min_length or road.total_width <= 0.0
                or road.total_width > max_total):
            continue
        out.append({
            "source": f"road:{road.road_id}",
            "road_id": road.road_id,
            "length": round(road.length, 1),
            "driving_width": road.driving_width,
            "total_width": road.total_width,
            "has_bridge": road.longest_bridge > 0.0,
            "drivable": road.driving_width > 0.0,
        })
    return sorted(out, key=lambda d: d["total_width"])


def nondrivable_regions(m: OpenDriveMap) -> list[int]:
    """Road ids that become `path` regions but have NO drivable lane.

    Town03 has five, one of them 967 m long: their lanes are typed ``none``, so the
    geometry exists but no vehicle may use it. They still satisfy the `junction == -1`
    test that creates a `path` region, so a plan step could ground onto one and the MPC
    would steer at a centroid the vehicle cannot legally reach. Surfaced rather than
    silently dropped, because dropping regions changes ids and any hand-picked corridor
    with them.
    """
    # total_width > 0 required: a road with no <lanes> block is unmeasured rather than
    # known to be undrivable, and flagging it would bury the five real Town03 cases.
    return sorted(r.road_id for r in m.path_roads
                  if r.driving_width <= 0.0 < r.total_width)


def _lane_samples(pts, driving_width: float, lanes_per_side: int = 1):
    """Reference-line samples plus LANE-CENTRE samples offset either side.

    WHY. `OpenDriveSegment.sample` returns the ROAD REFERENCE LINE, and its docstring
    justifies that with "a lane sits ~1.75 m to one side, which is far below the
    tens-of-metres separation between regions, so it cannot change which region a point
    belongs to". That is correct for REGION CLASSIFICATION and wrong for the DRIVABLE
    SURFACE, and the same table is used for both: `sampling_mpc.RoadSurface` measures
    `excess` as distance-to-nearest-waypoint minus a global `road_half_width = 7.0`.

    Example, Town05 road 3: 14.0 m of driving width around a single reference line, so
    its outer lane centres are ~5 m out and its edge is 7.0 m out -- exactly the global
    threshold. Where that road meets the junction roads at either end, the
    nearest-waypoint test flips onto the far carriageway, leaving a ~35 m stretch with NO
    reference-line sample on one side. A vehicle there can be beyond the surface with
    nothing to steer back to. Lane-centre samples close that hole.

    OFF BY DEFAULT (`lanes_per_side=0`) so an existing region table regenerates
    byte-identical, the same contract `approach_m` keeps.
    """
    if lanes_per_side <= 0 or driving_width <= 0:
        return pts
    half = driving_width / 2.0
    # Lane centres, not edges: n bands per side, sampled at each band's middle.
    offsets = [(-1) ** k * half * (i + 0.5) / lanes_per_side
               for i in range(lanes_per_side) for k in (0, 1)]
    out = list(pts)
    for x, y, h in pts:
        nx, ny = -math.sin(h), math.cos(h)      # left-hand normal to the heading
        for d in offsets:
            out.append((x + d * nx, y + d * ny, h))
    return out


def _junction_end_indices(pts, jpts, approach_m: float, step: float):
    """Which samples of a road are its junction-adjacent ends.

    Returns (head_n, tail_n): how many samples to carve from each end. A road may touch
    a junction at one end, both, or neither.

    The end is found GEOMETRICALLY rather than from `<link>`, because the parser records
    links as a flat (elementType, elementId) list and does not keep whether the element
    was a `<predecessor>` or a `<successor>` -- so the topology says a road meets a
    junction but not at which end. Comparing the two endpoints to the junction's own
    samples answers it without touching the parser.
    """
    if not pts or not jpts or approach_m <= 0:
        return 0, 0
    n = max(1, int(round(approach_m / max(step, 1e-6))))
    if len(pts) < 2 * n + 2:          # too short to carve without erasing the middle
        return 0, 0

    def d2(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    head = min(d2(pts[0], q) for q in jpts)
    tail = min(d2(pts[-1], q) for q in jpts)
    # Only the nearer end touches THIS junction; the far end may touch another, which is
    # handled by the caller's second pass over the road's other junction links.
    return (n, 0) if head <= tail else (0, n)


def segment(m: OpenDriveMap, step: float = 2.0,
            tags: dict[str, str] | None = None,
            auto_passage: bool = True,
            approach_m: float = 0.0,
            lanes_per_side: int = 0) -> RegionMap:
    """Build the region map.

    ``tags`` maps a region ``source`` string (e.g. ``"road:12"``) to a label, for the
    shapes OpenDRIVE cannot express — ``{"road:12": "along_edge"}``. Explicit tags
    always win over derived ones.

    ``auto_passage`` applies :func:`bridge_passages`. On by default so the artifact
    producers cannot forget it; it is a no-op on every town without a narrow bridge
    (all of them except Town07), so it changes no existing output.

    ``approach_m`` carves the junction-adjacent ends of each road into their own
    ``approach`` regions, giving the three-phase decomposition the robot taxonomy already
    uses (`cluster_map.livox1.yaml`: Intersection Approach/Enter, In, Exit) and that
    `generator.md` defines the Int_Turn macro as -- Road: On -> Approach/Enter -> In ->
    Road: On. Without it a region table has only `path` and `junction`, and cannot
    distinguish the phases.

    ONE LABEL, NOT TWO, and the reason is that geometry cannot supply the second.
    The same stretch of road is an APPROACH when driven toward the junction and an EXIT
    when driven away from it; a static region carries no direction. Direction comes from
    the plan step (`start_mode -> goal_mode`), so the segmenter emits the direction-free
    fact -- this road stretch adjoins a junction -- and leaves the naming to execution.

    Defaults to 0.0, i.e. OFF, so every existing region table regenerates byte-identical.
    """
    tags = {**(bridge_passages(m) if auto_passage else {}), **(tags or {})}
    regions: dict[int, Region] = {}
    by_source: dict[str, int] = {}
    rid = 0

    # Junction samples are needed before the roads when splitting, to find which end of
    # each road touches which junction.
    jpts_by_id = {jid: [p for r in roads for p in r.sample(step)]
                  for jid, roads in m.junction_roads.items()} if approach_m > 0 else {}

    # path regions first, so ids are stable and readable (paths, then junctions)
    approach_pending: list[tuple[str, list]] = []
    for road in sorted(m.path_roads, key=lambda r: r.road_id):
        src = f"road:{road.road_id}"
        pts = road.sample(step)
        head_n = tail_n = 0
        if approach_m > 0:
            for etype, eid in road.links:
                if etype != "junction" or eid not in jpts_by_id:
                    continue
                h, t = _junction_end_indices(pts, jpts_by_id[eid], approach_m, step)
                head_n, tail_n = max(head_n, h), max(tail_n, t)
        if head_n or tail_n:
            if head_n:
                approach_pending.append((f"{src}:approach_head", pts[:head_n]))
            if tail_n:
                approach_pending.append((f"{src}:approach_tail", pts[len(pts) - tail_n:]))
            pts = pts[head_n:len(pts) - tail_n]

        regions[rid] = Region(
            rid=rid, kind="path", label=tags.get(src, "path"), source=src,
            points=_lane_samples(pts, getattr(road, "driving_width", 0.0), lanes_per_side))
        by_source[src] = rid
        rid += 1

    for asrc, apts in approach_pending:
        regions[rid] = Region(rid=rid, kind="approach",
                              label=tags.get(asrc, "approach"),
                              source=asrc, points=apts)
        by_source[asrc] = rid
        rid += 1

    for jid, roads in sorted(m.junction_roads.items()):
        src = f"junction:{jid}"
        pts = [q for r in roads
               for q in _lane_samples(r.sample(step),
                                      getattr(r, "driving_width", 0.0), lanes_per_side)]
        regions[rid] = Region(rid=rid, kind="junction",
                              label=tags.get(src, "junction"), source=src,
                              points=pts)
        by_source[src] = rid
        rid += 1

    # Adjacency comes from each road's own `<link>` predecessor/successor, which
    # is the canonical topology. Deriving it from the junction's `incomingRoad`
    # list instead leaves map-edge stub roads isolated, because those link
    # road->road and so never appear as an incomingRoad anywhere.
    adjacency: set[tuple[int, int]] = set()
    for road in m.path_roads:
        a = by_source[f"road:{road.road_id}"]
        # An approach sits BETWEEN its road and the junction, so it inherits the road's
        # junction links and gains an edge to the road's remaining middle. Without this
        # the carved stretch would be an island and every route through it would fail.
        mine = [by_source[k] for k in (f"road:{road.road_id}:approach_head",
                                       f"road:{road.road_id}:approach_tail")
                if k in by_source]
        for ap in mine:
            adjacency.add((min(a, ap), max(a, ap)))
        for etype, eid in road.links:
            b = by_source.get(f"{'junction' if etype == 'junction' else 'road'}:{eid}")
            if b is None or b == a:
                continue
            if mine and etype == "junction":
                for ap in mine:
                    adjacency.add((min(ap, b), max(ap, b)))
            else:
                adjacency.add((min(a, b), max(a, b)))

    return RegionMap(town=m.name, regions=regions, adjacency=adjacency)


__all__ = ["LABELS", "MAX_NARROW_TOTAL_M", "MAX_PASSAGE_WIDTH_M",
           "MIN_PASSAGE_SPAN_M", "Region", "RegionMap", "bridge_passages",
           "narrow_candidates", "nondrivable_regions", "segment"]
