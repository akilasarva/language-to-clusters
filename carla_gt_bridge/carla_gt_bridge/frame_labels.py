"""Per-frame ground-truth labels for LiDAR scans, and the trajectory relations over them.

This replaces the interactive step in `clustering/cluster_training.py`, where a human
looks at sample scans from each HDBSCAN cluster and types a label. Everything here is
derived: from the town `.xodr` (topology) and from semantic LiDAR `ObjTag` (enclosure).

No ROS, no CARLA, no bag reading — those live in `scripts/label_frames.py`. This module
is pure so the geometry can be tested against hand-built cases, which is the only way the
sign conventions below stay honest.

TWO AXES, NOT ONE ELEVEN-WAY LABEL
----------------------------------
The requested vocabulary mixes two independent questions, and a frame answers both:

    topology   path | approach | junction            <- .xodr regions + odometry
    enclosure  open_space | along_edge | passage        <- semantic LiDAR ObjTag
    relation   none | along | past | around             <- the same, over a window

So "am I turning at an intersection, or turning around a building?" is not an ambiguity
to resolve — it is one frame that is `junction` on the topology axis and
`along_edge`/`around` on the enclosure axis, and both are true. `ClusterTaxonomy` already
carries an `axes` field for exactly this, and a union across axes is the wrong merge.

Residual uncertainty WITHIN an axis is what the STRICT / DEGRADED split is for
(`nl_planner.taxonomy.accept_clusters(mode, degraded=True)`). A cluster whose frames sit
near the around/along heading threshold gets the dominant label as its strict id and the
runner-up in `accept_degraded`, so a missed fine detection does not stall a step. See
:func:`assign_cluster_labels`, which emits exactly the `modes` + `mode_meta` blocks a
`cluster_map.<env>.yaml` wants.

WHY WORLD-FRAME BLOBS AND NOT ``ObjIdx``
----------------------------------------
The obvious instance tracker is the semantic LiDAR's own ``ObjIdx``, which would say
"the same building stayed on your right through the turn". But **ObjIdx is 0 for every
static structure point** — buildings, walls, fences and the bridge deck alike. It
identifies actors, and scenery is not an actor. So instances are recovered here instead by projecting structure points into
the world with the GT pose and hashing them onto a grid: a persistent building is a
persistent set of world cells. That is exact, because the pose is ground truth.

FRAME CONVENTIONS, WHICH ARE NOT THE SAME FOR THE TWO INPUTS
------------------------------------------------------------
* odometry from the ros-bridge is already PLANAR (+y north / +y left, radians CCW) and
  needs no conversion — see :mod:`carla_gt_bridge.frames`.
* the LiDAR point cloud is CARLA's own LEFT-handed frame (+y right), so every point read
  here has y negated once, on entry, in :func:`structure_sides`. This is the same
  correction `nodes/lidar_ranges_node.py` applies.

After that single negation everything below is planar: **lateral > 0 is LEFT**.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = [
    "AROUND_TURN_DEG", "ENCLOSURE_LABELS", "NEAR_STRUCTURE_M", "STRUCTURE_TAGS",
    "SEMANTIC_TAGS", "TOPOLOGY_LABELS", "FrameRow", "StructureMap",
    "assign_cluster_labels", "enclosure_relations", "junction_phases",
    "ROAD_LEVEL_BAND", "smooth_enclosure", "structure_sides", "trajectory_breaks",
]

#: CARLA 0.9.14 ``CityObjectLabel``, as it arrives in the semantic LiDAR's ``ObjTag``.
#: Only the entries observed in recorded CARLA bags are named; the rest pass through as
#: ints.
SEMANTIC_TAGS = {
    0: "Unlabeled", 1: "Roads", 2: "Sidewalks", 3: "Buildings", 4: "Walls",
    5: "Fences", 6: "Poles", 7: "TrafficLight", 8: "TrafficSigns", 9: "Vegetation",
    10: "Terrain", 11: "Sky", 12: "Pedestrians", 14: "Car", 20: "Static",
    21: "Dynamic", 22: "Other", 23: "Water", 24: "RoadLines", 25: "Ground",
    26: "Bridge", 27: "RailTrack", 28: "GuardRail",
}

#: Tags that make a space ENCLOSED — what the LiDAR can actually feel as structure.
#: Vegetation is deliberately excluded: a hedge returns like a wall but is not one, and
#: the enclosure axis is about geometry, not foliage.
#: Buildings and Walls are merged into one "building" concept because CARLA splits a
#: single city block across both tags.
STRUCTURE_TAGS = {3: "building", 4: "building", 5: "fence",
                  26: "bridge", 28: "guardrail"}

TOPOLOGY_LABELS = ("path", "approach", "junction", "exit", "off_network")

#: The SETTLED enclosure axis, from `nl_planner.mode_concepts`: how many sides are
#: closed. These exact spellings are what `ClusterTaxonomy` grounds against and what
#: `taxonomy.ALIASES` maps the human labels onto ("Along Wall" -> along_edge, "On Bridge"
#: -> passage). Labels outside this vocabulary (e.g. `open`, `on_bridge`) would produce a
#: cluster map whose every mode fails to resolve.
ENCLOSURE_LABELS = ("open_space", "along_edge", "passage")

#: The TRAJECTORY refinement, orthogonal to the axis above and the thing a single scan
#: cannot answer. `along_edge` + `past` is "Past Building"; `along_edge` + `around` is
#: "Around Corner"; `along_edge` + `along` is "Along Wall". Kept as its own column rather
#: than folded into the label, because the axis value must stay groundable on its own.
RELATIONS = ("none", "along", "past", "around")

#: Fallback when the local road width is unknown. Prefer the width-relative rule below.
NEAR_STRUCTURE_M = 14.0

#: Enclosure is a claim about the SPACE YOU OCCUPY, so a width-relative threshold is the
#: principled alternative to a fixed 14 m: a fixed range cannot mean the same thing on
#: streets of different widths (Town01's nearest structure is much closer than
#: Town10HD's), so a boundary fitted at one town's setback lands in the wrong place at
#: another.
#:
#: The factor is derived, not tuned: Town01's total road width is 16.6 m and
#: 0.85 x 16.6 = 14.1, i.e. the fixed 14 m constant is this rule at Town01's width.
#: Applied elsewhere:
#:
#:     Town01   total 16.6 m  ->  14.1 m
#:     Town07   total 10.6 m  ->   9.0 m
#:     Town05   total 22.3 m  ->  18.9 m
#:     Town10HD total 31.3 m  ->  26.6 m
#:
#: OFF BY DEFAULT (`label_frames --width-scaled`). In a wide, dense downtown the scaled
#: threshold is large enough that something is always on both sides, so `passage`
#: becomes trivially true and `open_space` all but disappears -- a degenerate label.
#: `total_width` counts wide verges and sidewalks that are not the space the vehicle
#: occupies, so it is the wrong quantity to scale by. An untested alternative is to
#: normalise the FEATURE (the scan, `ranges / max_lidar_range`) by local road width
#: instead, leaving the label at the fixed threshold.
WIDTH_FACTOR = 0.85

#: Cumulative heading change that turns "along" into "around". A CARLA block corner is
#: 90 degrees; 60 admits a rounded corner without firing on lane-keeping wobble.
AROUND_TURN_DEG = 60.0

#: A structure must be seen this many frames running to count as persistent, which is
#: what separates "drove around THIS building" from "buildings were nearby".
MIN_PERSIST_FRAMES = 3


# --------------------------------------------------------------------------- #
# Per-frame facts                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class FrameRow:
    """One LiDAR frame, with everything ground truth can say about it.

    ``topology`` and ``enclosure`` are the two axis labels; the remaining fields are the
    measurements they were derived from, kept so a disagreement can be diagnosed without
    re-reading the bag.
    """

    t_ns: int
    #: The LiDAR message's own HEADER stamp. `t_ns` is the bag RECEIVE time, and the two
    #: differ; `bag_to_pcd.py` names each PCD from the header stamp, so this is the
    #: column that joins a trained cluster back to its frame's ground truth. Joining on
    #: the receive time matches no scans.
    stamp_ns: int = 0
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0                      # planar radians, CCW, +y north
    speed: float = 0.0

    # -- topology axis, from the .xodr region table --
    rid: int = -1
    region_label: str = "off_network"
    region_dist_m: float = float("inf")
    topology: str = "off_network"         # path | approach | junction | off_network
    #: Per-frame enclosure threshold, WIDTH_FACTOR x the local road's total width.
    #: 0 means "unknown, fall back to NEAR_STRUCTURE_M".
    near_m: float = 0.0

    # -- enclosure axis, from semantic LiDAR --
    # lateral > 0 is LEFT, after the single y negation on entry.
    building_left_m: float = float("inf")
    building_right_m: float = float("inf")
    #: forward offset of that nearest return, SIGN INTACT: negative is behind the beam.
    #: This is what makes `past_building` a statement about the scan rather than about
    #: the last few seconds of history.
    building_fwd_left: float = 0.0
    building_fwd_right: float = 0.0
    building_blob_left: int = -1
    building_blob_right: int = -1
    bridge_left_m: float = float("inf")
    bridge_right_m: float = float("inf")
    bridge_m: float = float("inf")
    #: settled enclosure axis: open_space | along_edge | passage
    enclosure: str = "open_space"
    #: trajectory refinement: none | along | past | around
    relation: str = "none"
    #: within-axis runner-up, for `mode_meta.accept_degraded`
    enclosure_degraded: str = ""

    n_points: int = 0
    off_network: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


#: Sensor-frame z band for the enclosure GT. The collection rig sits 2.0 m up, so this
#: keeps returns 0.4 .. 1.5 m ABOVE THE ROAD -- kerbs, walls and building bases.
#:
#: A higher band (1.0 .. 3.5 m above the road) catches upper facades at range while
#: missing the wall beside the car; a wider band adds noise rather than evidence. This is
#: also the band used for the cluster model's 1-D scan input, so the ground truth and the
#: model input describe the same slice of the world.
ROAD_LEVEL_BAND = (-1.6, -0.5)


def structure_sides(xyz: np.ndarray, tags: np.ndarray, *,
                    z_band: tuple[float, float] = ROAD_LEVEL_BAND,
                    max_range_m: float = 40.0) -> dict:
    """Nearest structure of each class, per side, from ONE semantic LiDAR frame.

    ``xyz`` is (N, 3) in the **CARLA sensor frame** (left-handed, +y right) exactly as it
    arrives on the wire; y is negated here, once, and nowhere else.

    ``z_band`` keeps only returns at body height. The default matches
    ``lidar_ranges_node``'s settings: below -1.0 (sensor sits ~2 m up) is road
    and kerb, which would otherwise fill every bin; above 1.5 is gantries and overhanging
    foliage the vehicle drives under, and treating those as structure makes a bridge
    impassable — which matters here, because the bridge deck is the one `passage`.

    Returns ``{"<class>_left": (range_m, lateral_m, forward_m), ...}`` plus
    ``"points"``, the (M, 2) body-frame xy of every kept structure point, for
    :class:`StructureMap`.

    ``forward_m`` is carried rather than recomputed by the caller as
    ``sqrt(range^2 - lateral^2)``: that loses the SIGN, so a wall the vehicle has
    already passed would be placed the same distance ahead of it, and the blob lookup
    would land on empty road.
    """
    out: dict = {"points": np.zeros((0, 2)), "point_tags": np.zeros(0, dtype=int)}
    if len(xyz) == 0:
        return out
    x = np.asarray(xyz[:, 0], dtype=float)
    y = -np.asarray(xyz[:, 1], dtype=float)      # the one negation; now +y is LEFT
    z = np.asarray(xyz[:, 2], dtype=float)
    tags = np.asarray(tags, dtype=int)

    rng = np.hypot(x, y)
    band = (z > z_band[0]) & (z < z_band[1]) & (rng <= max_range_m) & (rng > 0.1)
    keep_any = band & np.isin(tags, list(STRUCTURE_TAGS))
    out["points"] = np.column_stack([x[keep_any], y[keep_any]])
    out["point_tags"] = tags[keep_any]

    for tag, name in STRUCTURE_TAGS.items():
        sel = band & (tags == tag)
        if not sel.any():
            continue
        for side, side_sel in (("left", y > 0), ("right", y < 0)):
            s = sel & side_sel
            if not s.any():
                continue
            j = int(np.argmin(rng[s]))
            key = f"{name}_{side}"
            cand = (float(rng[s][j]), float(y[s][j]), float(x[s][j]))
            # `building` merges two tags, so keep the nearer of the two.
            if key not in out or cand[0] < out[key][0]:
                out[key] = cand
    return out


class StructureMap:
    """Persistent world-frame structure instances, standing in for a missing ``ObjIdx``.

    Structure points are projected into the world with the ground-truth pose and hashed
    onto a ``cell_m`` grid; connected occupied cells share a blob id. Two frames that see
    the same wall therefore report the same id, which is what lets
    :func:`enclosure_relations` say "the SAME building stayed on the inside of the turn"
    — the distinction between driving around one block and driving past three.

    Grid-and-merge rather than DBSCAN: it is O(points), incremental, and has one
    parameter. The blob boundary does not need to be exact, only stable.
    """

    def __init__(self, cell_m: float = 4.0) -> None:
        self.cell_m = float(cell_m)
        self._cell_blob: dict[tuple[int, int], int] = {}
        self._parent: dict[int, int] = {}
        self._next = 0

    # -- union-find over blob ids -- #
    def _find(self, a: int) -> int:
        while self._parent[a] != a:
            self._parent[a] = self._parent[self._parent[a]]
            a = self._parent[a]
        return a

    def _union(self, a: int, b: int) -> int:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)
        return min(ra, rb)

    def _new(self) -> int:
        self._parent[self._next] = self._next
        self._next += 1
        return self._next - 1

    def add(self, points_world: np.ndarray) -> None:
        """Fold one frame's world-frame structure points into the map."""
        if len(points_world) == 0:
            return
        cells = np.floor(np.asarray(points_world) / self.cell_m).astype(int)
        for cx, cy in {(int(a), int(b)) for a, b in cells}:
            neigh = [self._cell_blob[(cx + dx, cy + dy)]
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     if (cx + dx, cy + dy) in self._cell_blob]
            if not neigh:
                self._cell_blob[(cx, cy)] = self._new()
                continue
            root = self._find(neigh[0])
            for n in neigh[1:]:
                root = self._union(root, n)
            self._cell_blob[(cx, cy)] = root

    def blob_at(self, xw: float, yw: float) -> int:
        """Blob id covering a world point, or -1. Checks the 8-neighbourhood, so a
        point just outside an occupied cell still resolves to its blob."""
        cx, cy = int(math.floor(xw / self.cell_m)), int(math.floor(yw / self.cell_m))
        for dx in (0, -1, 1):
            for dy in (0, -1, 1):
                b = self._cell_blob.get((cx + dx, cy + dy))
                if b is not None:
                    return self._find(b)
        return -1

    @property
    def n_blobs(self) -> int:
        return len({self._find(b) for b in self._cell_blob.values()})


def body_to_world(px: float, py: float, x: float, y: float, yaw: float
                  ) -> tuple[float, float]:
    """Planar body point -> planar world point. Both frames are +y left, yaw CCW."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (x + c * px - s * py, y + s * px + c * py)


# --------------------------------------------------------------------------- #
# Topology axis                                                               #
# --------------------------------------------------------------------------- #

def junction_phases(rows: list[FrameRow], junction_centroids: dict[int, tuple[float, float]],
                    *, approach_label: str = "approach",
                    lookahead_m: float = 5.0) -> None:
    """Fill ``topology`` in place: path | approach | junction | off_network.

    THE PHASE IS NOT THE SIGN OF d(distance)/dt ALONE, and this is the trap the whole
    function exists to avoid. For a vehicle driving THROUGH a junction, the distance to
    the junction centroid stops falling at closest approach, which is in the MIDDLE of
    the junction, not at its boundary. Phasing purely on the derivative therefore
    mislabels the first half of the junction as `approach` and the second as `exit`,
    off by the width of the region — and the resulting confusion matrix looks like a
    perception weakness on `exit` rather than a labelling bug.

    So: ``junction`` comes from the REGION LABEL, which is exact, and the derivative is
    used only to split the direction-free `approach` region into approach vs exit. That
    split is direction-only information and the segmenter deliberately does not carry it
    (`segmenter.segment`: "the same stretch of road is an APPROACH when driven toward the
    junction and an EXIT when driven away from it").
    """
    for i, r in enumerate(rows):
        if r.off_network:
            r.topology = "off_network"
        elif r.region_label == "junction":
            r.topology = "junction"
        elif r.region_label != approach_label:
            r.topology = r.region_label
        else:
            # Nearest junction centroid, and whether we are closing on it.
            if not junction_centroids:
                r.topology = "approach"
                continue
            jx, jy = min(junction_centroids.values(),
                         key=lambda c: (c[0] - r.x) ** 2 + (c[1] - r.y) ** 2)
            d_now = math.hypot(jx - r.x, jy - r.y)
            # LOOK AHEAD A DISTANCE, NOT A FRAME COUNT. Frame spacing can be well under
            # 0.5 m, so a few-frame lookahead is ~1 m against a region whose own sampling
            # step is 2 m; the sign of such a small change is mostly noise and biases the
            # approach/exit split asymmetrically.
            j, travelled = i, 0.0
            while j + 1 < len(rows) and travelled < lookahead_m:
                travelled += math.hypot(rows[j + 1].x - rows[j].x,
                                        rows[j + 1].y - rows[j].y)
                j += 1
            d_next = math.hypot(jx - rows[j].x, jy - rows[j].y)
            r.topology = "approach" if d_next < d_now else "exit"


# --------------------------------------------------------------------------- #
# Enclosure axis — the trajectory relations                                   #
# --------------------------------------------------------------------------- #

def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def trajectory_breaks(rows: list[FrameRow], *, jump_m: float = 20.0) -> set:
    """Indices where the trajectory JUMPS rather than moves — teleports.

    `drive_collection.py` teleports the ego past an obstruction it cannot reverse out
    of, and the bridge can respawn one too. A jump is not motion, so a window spanning
    it would report `past` for a building the vehicle never drove past and `around` for
    a heading change that was an instant rotation.

    A plain DISPLACEMENT threshold, not a speed one: bag timestamps are receive times,
    so a pair of frames arriving 1 ms apart turns an ordinary 0.4 m step into 400 m/s and
    a speed test fires spuriously. At 20 Hz and 8 m/s a real step is ~0.4 m and even
    30 m/s gives 1.5 m, so 20 m is well clear of normal motion and needs no timing.

    Detected geometrically rather than read from the driver's teleport log: the log
    records wall time while frames carry bag time, and this also catches discontinuities
    the driver did not cause.
    """
    return {i for i in range(1, len(rows))
            if math.hypot(rows[i].x - rows[i - 1].x,
                          rows[i].y - rows[i - 1].y) > jump_m}


def enclosure_relations(rows: list[FrameRow], *,
                        breaks: set | None = None,
                        window_s: float = 4.0,
                        default_near_m: float = NEAR_STRUCTURE_M,
                        around_deg: float = AROUND_TURN_DEG,
                        min_persist: int = MIN_PERSIST_FRAMES) -> None:
    """Fill ``enclosure`` and ``enclosure_degraded`` in place, over a sliding window.

    These are TRAJECTORY relations: no single scan can say "past" or "around", because
    both are statements about how the scene changed while the vehicle moved. But the
    label is attached to a SCAN, and it is consumed by
    :func:`assign_cluster_labels`, which votes each HDBSCAN cluster from the ground truth
    of its member frames. So the window supplies the *relation* and the current frame
    must still supply the *evidence*.

    THAT SPLIT IS NOT A NICETY. A look-back-only rule labels frames `past_building` that
    have no building return in them at all; those frames get voted into whatever cluster
    is active and drag its purity down, which reads as "the autoencoder cannot see
    buildings" rather than as a labelling bug. Every branch below therefore requires a
    return in the CURRENT frame.

    ``passage``    structure within ``near_m`` on BOTH sides. Checked first for the
                   bridge case, because a deck has parapets either side; a bridge driven
                   PAST returns on one side only, often near the 14 m threshold, so a
                   one-sided min-range test gives knife-edge false positives on a bridge
                   the vehicle never drove on.
    ``along_edge`` structure within ``near_m`` on ONE side. The ``relation`` column then
                   says which of the three human labels it is:
                     ``along``   still beside it            ("Along Wall")
                     ``past``    now behind the beam        ("Past Building")
                     ``around``  persisted on the INSIDE of a >= ``around_deg`` turn
                                 ("Around Corner")
    ``open_space`` nothing within ``visible_m``. Also what an OFF-ROAD frame gets -- see
                   the note in the body.

    The runner-up goes in ``enclosure_degraded``, which becomes `accept_degraded` in the
    cluster map -- a frame one-sided at 14.1 m is `along_edge` strict and may accept
    `passage` on a landmark-triggered step.
    """
    if not rows:
        return
    if breaks is None:
        breaks = trajectory_breaks(rows)
    t = np.array([r.t_ns for r in rows], dtype=float) / 1e9
    # Start of the current continuous segment, so no window reaches across a teleport.
    seg_start, starts = 0, []
    for i in range(len(rows)):
        if i in breaks:
            seg_start = i
        starts.append(seg_start)
    for i, r in enumerate(rows):
        lo = max(int(np.searchsorted(t, t[i] - window_s)), starts[i])
        win = rows[lo:i + 1]
        # The frame's own width-scaled threshold when the labeller supplied one.
        near_m = r.near_m if r.near_m > 0 else default_near_m
        visible_m = 2 * near_m

        if r.bridge_left_m <= near_m and r.bridge_right_m <= near_m:
            r.enclosure, r.relation, r.enclosure_degraded = "passage", "along", ""
            continue

        turn = abs(sum(_wrap_pi(win[k + 1].yaw - win[k].yaw)
                       for k in range(len(win) - 1)))
        turn_deg = math.degrees(turn)
        # Sign of the net turn says which side is the INSIDE: a left turn (CCW,
        # positive) curls around whatever is on the left.
        net = sum(_wrap_pi(win[k + 1].yaw - win[k].yaw) for k in range(len(win) - 1))
        inside = "left" if net > 0 else "right"

        near = {"left": [], "right": []}
        for w in win:
            if w.building_left_m <= near_m and w.building_blob_left >= 0:
                near["left"].append(w.building_blob_left)
            if w.building_right_m <= near_m and w.building_blob_right >= 0:
                near["right"].append(w.building_blob_right)

        def persistent(side: str) -> int:
            """The blob seen on ``side`` for at least ``min_persist`` frames, else -1."""
            if not near[side]:
                return -1
            blob, n = Counter(near[side]).most_common(1)[0]
            return int(blob) if n >= min_persist else -1

        def now(side: str) -> tuple[float, float]:
            return ((r.building_left_m, r.building_fwd_left) if side == "left"
                    else (r.building_right_m, r.building_fwd_right))

        # No structure in THIS scan -> open_space, whatever the window remembers.
        # This is also the label an OFF-ROAD frame gets, and getting one is the point of
        # the two axes: a pose 30 m off the carriageway has no topology (`off_network`,
        # since no region owns it) but its enclosure is perfectly well defined, because
        # the LiDAR can see that nothing is close on either side. `open_space` is
        # otherwise unreachable -- the .xodr never emits it and a lane-following agent
        # never leaves the road.
        if min(r.building_left_m, r.building_right_m) > visible_m:
            r.enclosure, r.relation, r.enclosure_degraded = "open_space", "none", ""
            continue

        # Structure close on BOTH sides is `passage` whatever it is made of -- that is
        # the definition of the axis, and it is why a bridge deck and a walled alley are
        # the same concept.
        if (r.building_left_m <= near_m and r.building_right_m <= near_m):
            r.enclosure, r.relation, r.enclosure_degraded = "passage", "along", ""
            continue

        inside_blob = persistent(inside)
        if (turn_deg >= around_deg and inside_blob >= 0
                and now(inside)[0] <= near_m):
            r.enclosure, r.relation = "along_edge", "around"
            r.enclosure_degraded = "passage"
            continue

        side = min(("left", "right"), key=lambda s: now(s)[0])
        rng_now, fwd_now = now(side)
        was_near = persistent(side) >= 0 or any(
            (w.building_left_m if side == "left" else w.building_right_m) <= near_m
            for w in win)

        if rng_now <= near_m and fwd_now >= 0:
            r.enclosure, r.relation = "along_edge", "along"
            r.enclosure_degraded = "passage" if turn_deg >= around_deg / 2 else ""
        elif was_near and fwd_now < 0:
            r.enclosure, r.relation = "along_edge", "past"
            r.enclosure_degraded = ""
        elif rng_now <= near_m:
            r.enclosure, r.relation, r.enclosure_degraded = "along_edge", "along", ""
        else:
            r.enclosure, r.relation, r.enclosure_degraded = "open_space", "none", ""


def smooth_enclosure(rows: list[FrameRow], *, window_m: float = 8.0,
                     breaks: set | None = None) -> None:
    """Mode-filter the enclosure axis in place.

    The raw enclosure label is dominated by per-frame noise (median run length of about
    one frame, far more transitions than the topology axis over the same drive), and a
    cluster voting on it inherits that noise as impurity. Two causes, and a mode filter
    is the cheapest thing that answers both:

    * **dropout.** Many frames carry NO building return at all, interleaved with frames
      that do — sparse returns at range, the z-band edge, occlusion by another structure.
      Absence of a return for one frame is not absence of a building.
    * **threshold flicker.** Frames near the 14 m boundary flip with ordinary range
      jitter.

    Smoothing the GROUND TRUTH is deliberate: `live_cluster_inference_node` applies a
    10-frame mode filter to its own predictions, so an unsmoothed GT would be compared
    against a smoothed prediction and the disagreement would be mostly filter mismatch.

    THE WINDOW IS A DISTANCE, NOT A FRAME COUNT. Frame spacing is not constant: the
    bridge runs async and the server can free-run faster than real time, and spacing
    collapses whenever the vehicle slows. Smoothing over metres of travel ties the filter
    to the geometry it is smoothing, and it degrades gracefully when the vehicle stalls
    instead of averaging hundreds of duplicate frames.

    OFF BY DEFAULT. A majority filter can erase a rare class outright (`passage` occurs
    in short scattered runs that a majority vote always loses). Erasing a rare class is
    worse than the chatter it fixes, because the chatter is mostly honest sensing dropout
    that a majority vote over a cluster's frames already averages out, while a missing
    class cannot be recovered downstream. Enable it deliberately and check the class
    histogram before and after.

    ``breaks`` are teleport indices; the filter never looks across one.
    """
    if not rows or window_m <= 0:
        return
    if breaks is None:
        breaks = trajectory_breaks(rows)
    seg_start, starts = 0, []
    for i in range(len(rows)):
        if i in breaks:
            seg_start = i
        starts.append(seg_start)
    # cumulative travelled distance, reset at each break
    cum = [0.0] * len(rows)
    for i in range(1, len(rows)):
        d = math.hypot(rows[i].x - rows[i - 1].x, rows[i].y - rows[i - 1].y)
        cum[i] = cum[i - 1] + (0.0 if i in breaks else d)
    half = window_m / 2.0
    pairs = [(r.enclosure, r.relation) for r in rows]
    out = []
    for i in range(len(rows)):
        lo = i
        while lo > starts[i] and cum[i] - cum[lo - 1] <= half:
            lo -= 1
        hi = i + 1
        while hi < len(rows) and hi not in breaks and cum[hi] - cum[i] <= half:
            hi += 1
        out.append(Counter(pairs[lo:hi]).most_common(1)[0][0])
    for r, (e, rel) in zip(rows, out):
        r.enclosure, r.relation = e, rel


# --------------------------------------------------------------------------- #
# Cluster -> label, with the purity floor the human's "skip" key used to be   #
# --------------------------------------------------------------------------- #

@dataclass
class ClusterAssignment:
    cluster_id: int
    label: str
    purity: float
    n: int
    degraded: list[str] = field(default_factory=list)
    distribution: dict = field(default_factory=dict)


def assign_cluster_labels(cluster_ids, gt_labels, *,
                          min_purity: float = 0.6,
                          min_size: int = 8,
                          degraded_floor: float = 0.15
                          ) -> dict[int, ClusterAssignment]:
    """Majority-vote each HDBSCAN cluster onto one axis's GT labels.

    This is the function that replaces the human. It differs from a plain argmax in the
    two ways the human's behaviour differed:

    * ``min_purity`` / ``min_size`` — the operator could type ``s`` to skip a cluster
      whose samples did not agree. Without an equivalent, every mixed cluster is
      force-mapped to a plausible-looking wrong label, which is worse than no label
      because nothing downstream can tell the two apart. Those come back as
      ``"unlabeled"``.
    * ``degraded_floor`` — any other label holding at least this share becomes a
      DEGRADED accept id rather than being discarded. This is where the
      around-building-at-an-intersection case lands: the cluster is labelled by its
      majority and still accepts the runner-up on a landmark-triggered step.

    HDBSCAN's ``-1`` noise cluster is always ``"unlabeled"``: it is the model saying it
    does not recognise the frame, and that is a real answer worth preserving.
    """
    buckets: dict[int, list[str]] = defaultdict(list)
    for cid, lab in zip(cluster_ids, gt_labels):
        buckets[int(cid)].append(str(lab))

    out: dict[int, ClusterAssignment] = {}
    for cid, labs in sorted(buckets.items()):
        dist = Counter(labs)
        n = len(labs)
        top, top_n = dist.most_common(1)[0]
        purity = top_n / n if n else 0.0
        if cid == -1 or n < min_size or purity < min_purity:
            out[cid] = ClusterAssignment(cid, "unlabeled", purity, n,
                                         distribution=dict(dist))
            continue
        degraded = [lab for lab, c in dist.most_common()[1:]
                    if c / n >= degraded_floor]
        out[cid] = ClusterAssignment(cid, top, purity, n, degraded, dict(dist))
    return out


def to_cluster_map_blocks(assignments: dict[int, ClusterAssignment]) -> dict:
    """``{modes, mode_meta}`` — the two blocks of a ``cluster_map.<env>.yaml``.

    ``mode_meta[label].accept_degraded`` is populated so
    ``nl_planner.taxonomy.accept_clusters(label, degraded=True)`` returns the coarser
    ids, which is the mechanism that absorbs the labels ground truth cannot separate.
    """
    modes: dict[str, list[int]] = defaultdict(list)
    degraded: dict[str, set] = defaultdict(set)
    for cid, a in sorted(assignments.items()):
        if a.label == "unlabeled":
            continue
        modes[a.label].append(cid)
        for d in a.degraded:
            degraded[d].add(cid)
    meta = {lab: {"accept_degraded": sorted(ids - set(modes.get(lab, [])))}
            for lab, ids in degraded.items()
            if sorted(ids - set(modes.get(lab, [])))}
    return {"modes": {k: sorted(v) for k, v in sorted(modes.items())},
            "mode_meta": meta}
