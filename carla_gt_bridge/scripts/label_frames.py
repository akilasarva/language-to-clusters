#!/usr/bin/env python3
"""Bag -> one ground-truth-labelled row per LiDAR frame. Offline, no CARLA server.

    python3 scripts/label_frames.py --bag ~/carla_data/straight_line_terrain \
        --out <frames.csv>

The town is read from the bag's own ``/carla/map`` and matched against ``config/*.xodr``
by road-id set, so a bag identifies itself and cannot be scored against the wrong map.
A bag without either topic cannot identify its town (use --xodr).

Output columns are :class:`carla_gt_bridge.frame_labels.FrameRow`, with the two axis
labels (`topology`, `enclosure`) plus the measurements they came from. Feed it to
``label_clusters.py`` together with the HDBSCAN assignment to get a cluster map.

WHAT THIS NEEDS THE BAG TO CONTAIN
    /carla/ego_vehicle/lidar           the frames being labelled
    /carla/ego_vehicle/odometry        pose -> topology axis            (REQUIRED)
    /carla/ego_vehicle/semantic_lidar  ObjTag -> enclosure axis         (optional)
    /carla/world_info                  the town, for self-identification (or
      or /carla/map                    `/carla/map` on older bridges; --xodr overrides)

Without odometry there is NO join from a scan to a region and the bag cannot be
labelled at all. Run with --check to find out before processing.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import glob
import math
import os
import re
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from carla_gt_bridge.frame_labels import (WIDTH_FACTOR,  # noqa: E402
                                          FrameRow, StructureMap,
                                          body_to_world, enclosure_relations,
                                          junction_phases, smooth_enclosure,
                                          structure_sides, trajectory_breaks)
from carla_gt_bridge.region_lookup import (DEFAULT_MAX_DISTANCE_M,  # noqa: E402
                                           load_region_table)

LIDAR = "/carla/ego_vehicle/lidar"
SEMANTIC = "/carla/ego_vehicle/semantic_lidar"
ODOM = "/carla/ego_vehicle/odometry"
MAP = "/carla/map"
#: The OpenDRIVE arrives under ONE of two names depending on the bridge version, and the
#: two are not interchangeable. This workspace's `carla_ros_bridge` publishes
#: `/carla/world_info` (`carla_msgs/CarlaWorldInfo`, fields `map_name` + `opendrive`);
#: other bridge builds publish `/carla/map` (`std_msgs/String`) and no world_info.
#: Accept either.
WORLD_INFO = "/carla/world_info"


def _read_bag(bag: str, topics: set[str]):
    """Yield ``(topic, deserialised_msg, t_ns)``. ROS imports are local so ``--check``
    and the unit tests do not need a sourced workspace."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic in topics:
            yield topic, deserialize_message(data, get_message(types[topic])), t_ns


def bag_topics(bag: str) -> dict[str, str]:
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def identify_town(xodr_text: str, config_dir: str) -> tuple[str, float]:
    """Match the bag's embedded OpenDRIVE against the shipped towns by road-id set.

    Jaccard on road ids: ~1.0 for the right town and far lower for every other. NOTE it
    matches the road NETWORK: Unreal-side scenery edits leave road ids untouched, so this
    identifies the `.xodr`, not the dressing.
    """
    ids = set(re.findall(r'<road[^>]*id="(\d+)"', xodr_text))
    best, score = "", 0.0
    for f in sorted(glob.glob(os.path.join(config_dir, "Town*.xodr"))):
        other = set(re.findall(r'<road[^>]*id="(\d+)"', open(f).read()))
        j = len(ids & other) / max(1, len(ids | other))
        if j > score:
            best, score = f, j
    return best, score


def _yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


def _xyz_tags(msg):
    from sensor_msgs_py import point_cloud2
    a = point_cloud2.read_points(msg, skip_nans=True)
    names = a.dtype.names
    xyz = np.column_stack([np.asarray(a["x"]), np.asarray(a["y"]),
                           np.asarray(a["z"])]).astype(float)
    tags = (np.asarray(a["ObjTag"]).astype(int) if "ObjTag" in names
            else np.zeros(len(xyz), dtype=int))
    return xyz, tags


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--regions-npz", default="",
                    help="default: config/regions.<town>.approach.npz, else "
                         "regions.<town>.npz")
    ap.add_argument("--xodr", default="", help="override the town (bags with no "
                                               "/carla/map need this)")
    ap.add_argument("--max-distance-m", type=float, default=DEFAULT_MAX_DISTANCE_M)
    ap.add_argument("--window-s", type=float, default=4.0,
                    help="trajectory-relation window for past/around/along")
    ap.add_argument("--cell-m", type=float, default=4.0,
                    help="structure blob grid; the ObjIdx stand-in")
    ap.add_argument("--width-scaled", action="store_true",
                    help="scale the enclosure threshold by WIDTH_FACTOR x local road "
                         "total width instead of using a fixed NEAR_STRUCTURE_M. "
                         "MEASURED WORSE — see frame_labels.WIDTH_FACTOR.")
    ap.add_argument("--smooth-m", type=float, default=0.0,
                    help="mode-filter width for the enclosure axis, in METRES of "
                         "travel; 0 disables")
    ap.add_argument("--check", action="store_true",
                    help="report whether this bag CAN be labelled, then stop")
    args = ap.parse_args(argv)

    topics = bag_topics(args.bag)
    have = {k: (k in topics) for k in (LIDAR, ODOM, SEMANTIC, MAP, WORLD_INFO)}
    have["_map_any"] = have[MAP] or have[WORLD_INFO]
    print(f"bag: {args.bag}")
    for k, v in have.items():
        if not k.startswith("_"):
            print(f"  {'ok  ' if v else 'MISS'} {k}")
    if not have[ODOM]:
        print("\nCANNOT LABEL: no odometry, so there is no join from a scan to a pose "
              "and therefore none to a region. Re-collect with the topic set in "
              "the collection drive.")
        return 1
    if not have[SEMANTIC]:
        print("\nNote: no semantic_lidar — the topology axis will be labelled and the "
              "enclosure axis will be 'open' everywhere. That is a partial label set, "
              "not a wrong one.")
    if args.check:
        return 0

    # -- town + region table -------------------------------------------------- #
    cfg = os.path.join(PKG, "config")
    xodr_path, score = args.xodr, 1.0
    # A sidecar written by run_collection.sh straight from the CARLA API. Preferred over
    # the topic because `/carla/world_info` is a carla_msgs type that only the Humble
    # workspace can deserialize, and this labeller runs on the Jazzy host.
    sidecar = os.path.join(args.bag.rstrip("/"), "map.xodr")
    if not xodr_path and os.path.exists(sidecar):
        xodr_path, score = identify_town(open(sidecar).read(), cfg)
        print(f"town: {os.path.basename(xodr_path)} from the bag's own map.xodr "
              f"(road-id jaccard {score:.3f})")
    if not xodr_path:
        if not have["_map_any"]:
            print("neither /carla/world_info nor /carla/map in the bag, and no --xodr: "
                  "cannot identify the town")
            return 1
        if have[WORLD_INFO]:
            msg = next(m for t, m, _ in _read_bag(args.bag, {WORLD_INFO}))
            text = msg.opendrive
            print(f"  /carla/world_info says map_name={msg.map_name!r}")
        else:
            text = next(m.data for t, m, _ in _read_bag(args.bag, {MAP}))
        xodr_path, score = identify_town(text, cfg)
        print(f"town: {os.path.basename(xodr_path)} (road-id jaccard {score:.3f})")
        if score < 0.99:
            print("  WARNING: not an exact match — scoring against the wrong map "
                  "silently produces plausible labels. Pass --xodr explicitly.")
    town = os.path.basename(xodr_path).replace(".xodr", "").lower()

    npz = args.regions_npz
    if not npz:
        for cand in (f"regions.{town}.approach.npz", f"regions.{town}.npz"):
            if os.path.exists(os.path.join(cfg, cand)):
                npz = os.path.join(cfg, cand)
                break
    if not npz or not os.path.exists(npz):
        print(f"no region table for {town}. Generate one:\n"
              f"  python3 scripts/map_regions.py --xodr {xodr_path} --approach-m 15 "
              f"--suffix approach")
        return 1
    table = load_region_table(npz)
    print(f"regions: {os.path.basename(npz)} ({len(table.region_ids)} regions)")
    if "approach" not in set(table.labels.tolist()):
        print("  NOTE: this table has no `approach` regions, so the junction phase "
              "collapses to in/none. Regenerate with --approach-m 15 for the "
              "three-phase decomposition.")

    jc = {rid: table.centroid_of(rid) for rid in table.region_ids
          if table.label_of(rid) == "junction"}

    # PER-REGION ENCLOSURE THRESHOLD, from the road's own width.
    # A fixed 14 m does not transfer: it is 0.85 x Town01's 16.6 m total width, and
    # applying it to Town10HD (31.3 m) calls buildings at 19 m "open". See WIDTH_FACTOR.
    # A junction inherits the widest road that meets it, because a junction's extent is
    # set by the roads it joins.
    rid_near: dict[int, float] = {}
    if not args.width_scaled:
        print("enclosure threshold: FIXED "
              f"{__import__('carla_gt_bridge.frame_labels', fromlist=['x']).NEAR_STRUCTURE_M} m "
              "(--width-scaled to scale it by road width; see the note in frame_labels)")
    try:
        if not args.width_scaled:
            raise RuntimeError("width scaling disabled")
        from carla_gt_bridge.opendrive import load as _load_xodr
        from carla_gt_bridge.segmenter import segment as _segment
        _m = _load_xodr(xodr_path)
        _rm = _segment(_m, step=2.0, approach_m=15.0)
        _w = {r.road_id: r.total_width for r in _m.path_roads}
        _default = float(np.median(list(_w.values()))) if _w else 0.0
        for _r in _rm.regions.values():
            src = _r.source
            if src.startswith("road:"):
                rid_near[_r.rid] = WIDTH_FACTOR * _w.get(
                    int(src.split(":")[1]), _default)
            else:                                  # junction:<id>
                jid = int(src.split(":")[1])
                widths = [_w[rd.road_id] for rd in _m.junction_roads.get(jid, [])
                          if rd.road_id in _w]
                rid_near[_r.rid] = WIDTH_FACTOR * (max(widths) if widths else _default)
        _vals = [v for v in rid_near.values() if v > 0]
        print(f"enclosure threshold: {WIDTH_FACTOR} x road total width -> "
              f"{min(_vals):.1f}-{max(_vals):.1f} m across {len(rid_near)} regions "
              f"(median {np.median(_vals):.1f})")
    except Exception as e:
        if args.width_scaled:
            print(f"could not derive per-region widths ({e}); using the fixed "
                  f"threshold")

    # -- pass A: odometry + LiDAR stamps only ---------------------------------- #
    # STREAMING, NOT ACCUMULATING. Each semantic cloud is ~30k points, so keeping them
    # all costs frames x 30k x 32 B -- tens of GB on a long drive (e.g. Town07), which
    # exhausts RAM and ends in swap thrash or an OOM kill rather than a Python error.
    # Each cloud is consumed exactly once, so nothing needs to be kept: the per-frame
    # result is ~10 floats.
    odom: list[tuple[int, float, float, float, float]] = []
    lidar_t: list[int] = []
    lidar_hdr: list[int] = []
    lidar_n: list[int] = []
    sem_t: list[int] = []
    for topic, msg, t_ns in _read_bag(args.bag, {LIDAR, ODOM}):
        if topic == ODOM:
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            v = msg.twist.twist.linear
            odom.append((t_ns, float(p.x), float(p.y), _yaw_of(q),
                         math.hypot(v.x, v.y)))
        else:
            xyz, _ = _xyz_tags(msg)
            lidar_t.append(t_ns)
            lidar_hdr.append(msg.header.stamp.sec * 10 ** 9 + msg.header.stamp.nanosec)
            lidar_n.append(len(xyz))
    if not lidar_t:
        print("no lidar frames")
        return 1
    o_t = [o[0] for o in odom]

    def nearest(ts: list[int], t: int) -> int:
        i = bisect.bisect_left(ts, t)
        if i == 0:
            return 0
        if i >= len(ts):
            return len(ts) - 1
        return i if (ts[i] - t) < (t - ts[i - 1]) else i - 1

    # -- pass B: stream the semantic clouds, keep only the per-frame result ----- #
    smap = StructureMap(cell_m=args.cell_m)
    sem_res: list[dict] = []
    for topic, msg, t_ns in _read_bag(args.bag, {SEMANTIC}):
        xyz, tags = _xyz_tags(msg)
        oi = nearest(o_t, t_ns) if odom else 0
        _, ox, oy, oyaw, _spd = odom[oi]
        st = structure_sides(xyz, tags)
        pts = st["points"]
        if len(pts):
            smap.add(np.array([body_to_world(px, py, ox, oy, oyaw)
                               for px, py in pts]))
        keep = {"t": t_ns}
        for side in ("left", "right"):
            for name in ("building", "bridge"):
                k = f"{name}_{side}"
                if k in st:
                    rng, lat, fwd = st[k]
                    keep[k] = (rng, lat, fwd,
                               *body_to_world(fwd, lat, ox, oy, oyaw))
        sem_res.append(keep)
        sem_t.append(t_ns)
    print(f"frames: {len(lidar_t)} lidar, {len(odom)} odometry, "
          f"{len(sem_res)} semantic")

    # -- rows ------------------------------------------------------------------ #
    rows: list[FrameRow] = []
    skew_ms = []
    for k, t in enumerate(lidar_t):
        oi = nearest(o_t, t)
        _, x, y, yaw, spd = odom[oi]
        skew_ms.append(abs(o_t[oi] - t) / 1e6)
        rid, dist = table.nearest(x, y)
        off = dist > args.max_distance_m
        r = FrameRow(t_ns=t, stamp_ns=lidar_hdr[k], x=x, y=y, yaw=yaw, speed=spd,
                     rid=-1 if off else rid,
                     region_label="off_network" if off else table.label_of(rid),
                     region_dist_m=dist, off_network=off, n_points=lidar_n[k],
                     near_m=0.0 if off else rid_near.get(int(rid), 0.0))
        if sem_res:
            sr = sem_res[nearest(sem_t, t)]
            for key, rng_attr, fwd_attr, blob_attr in (
                    ("building_left", "building_left_m", "building_fwd_left",
                     "building_blob_left"),
                    ("building_right", "building_right_m", "building_fwd_right",
                     "building_blob_right")):
                if key in sr:
                    rng, lat, fwd, wx, wy = sr[key]
                    setattr(r, rng_attr, rng)
                    setattr(r, fwd_attr, fwd)
                    setattr(r, blob_attr, smap.blob_at(wx, wy))
            for key, attr in (("bridge_left", "bridge_left_m"),
                              ("bridge_right", "bridge_right_m")):
                if key in sr:
                    setattr(r, attr, sr[key][0])
            r.bridge_m = min(r.bridge_left_m, r.bridge_right_m)
        rows.append(r)

    junction_phases(rows, jc)
    breaks = trajectory_breaks(rows)
    enclosure_relations(rows, breaks=breaks, window_s=args.window_s)
    raw_tr = sum(1 for i in range(1, len(rows))
                 if rows[i].enclosure != rows[i - 1].enclosure)
    smooth_enclosure(rows, window_m=args.smooth_m, breaks=breaks)
    sm_tr = sum(1 for i in range(1, len(rows))
                if rows[i].enclosure != rows[i - 1].enclosure)

    # -- report --------------------------------------------------------------- #
    from collections import Counter
    print(f"\ntime skew lidar<->odometry: median {np.median(skew_ms):.1f} ms, "
          f"max {max(skew_ms):.1f} ms")
    print(f"pose->reference line: median {np.median([r.region_dist_m for r in rows]):.2f} m, "
          f"p95 {np.percentile([r.region_dist_m for r in rows], 95):.2f} m")
    print(f"structure blobs: {smap.n_blobs}")
    print(f"trajectory breaks (teleports): {len(breaks)}"
          + ("  — relation windows reset there" if breaks else ""))
    print(f"enclosure transitions: {raw_tr} raw -> {sm_tr} after the "
          f"{args.smooth_m:.0f} m mode filter")
    topo_c = Counter(r.topology for r in rows)
    encl_c = Counter(r.enclosure for r in rows)
    rel_c = Counter(r.relation for r in rows)
    print(f"topology : {dict(topo_c)}")
    print(f"enclosure: {dict(encl_c)}")
    print(f"relation : {dict(rel_c)}")
    # A CLASS SILENTLY GOING TO ZERO is the failure this line exists to catch: e.g. an
    # enclosure mode filter wider than a short `passage` run deletes that class outright
    # (a majority vote always loses short runs), and nothing else in the output says so.
    from carla_gt_bridge.frame_labels import (ENCLOSURE_LABELS, RELATIONS,
                                              TOPOLOGY_LABELS)
    for name, seen, expected in (("topology", topo_c, TOPOLOGY_LABELS),
                                 ("enclosure", encl_c, ENCLOSURE_LABELS),
                                 ("relation", rel_c, RELATIONS)):
        missing = [c for c in expected if not seen.get(c)]
        if missing:
            print(f"  NOTE: {name} classes with ZERO frames: {missing} — a class that "
                  f"is absent cannot be scored, and a smoothing or threshold change is "
                  f"the usual cause.")
    a, e = topo_c.get("approach", 0), topo_c.get("exit", 0)
    if a and e and not 0.5 <= e / a <= 2.0:
        print(f"  NOTE: approach/exit is {a}/{e} = {e / a:.2f}x. These should be near "
              f"balanced on a route that enters and leaves each junction once; a large "
              f"skew is a labelling artefact before it is a perception result.")

    out = args.out or os.path.join(os.path.dirname(args.bag.rstrip("/")),
                                   os.path.basename(args.bag.rstrip("/")) + ".labels.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].as_dict()))
        w.writeheader()
        for r in rows:
            w.writerow(r.as_dict())
    print(f"\nwrote {len(rows)} rows -> {out}")
    print("lateral sign convention: PLANAR, +y LEFT (the CARLA cloud's y is negated "
          "once on entry, in frame_labels.structure_sides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
