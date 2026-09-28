#!/usr/bin/env python3
"""Export a collection bag's LiDAR to the PCD folder `cluster_training.py` expects.

    python3 scripts/bag_to_pcd.py --bag ~/carla_data/collect_town01_... \
        --out ~/carla_data/town01_v2/town01_v2_pcds

Filenames are `<sec>-<nsec>.pcd`, which is what `cluster_training._pcd_sort_key` parses.
That is not cosmetic: the timestamp IS the join key back to `label_frames.py`'s CSV, so
each trained cluster can be scored against the ground truth of its own member frames.
Any other naming would train fine and be impossible to score.

Exports the RAW cloud, not the z-band slice. The band belongs to the training config so
it can be changed without re-exporting, and `get_ranges_from_points` applies it.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

LIDAR = "/carla/ego_vehicle/lidar"
# A dual-rig bag carries a second ray_cast on its own topic (see objects.dualfov.json),
# and both must be exportable from the SAME bag or the paired comparison is impossible.


def write_pcd(path: str, xyz: np.ndarray) -> None:
    """Minimal binary PCD writer — open3d reads it, but open3d is not needed to write it.

    open3d lives in the conda python and `rosbag2_py` lives in the ROS python, and the
    two cannot be imported into the same interpreter here. Rather than shuttle the cloud
    between two processes, write the format directly: it is a text header plus packed
    float32 xyz, which is exactly what `o3d.io.read_point_cloud` expects.
    """
    pts = np.asarray(xyz, dtype=np.float32)
    n = len(pts)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts.tobytes())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--topic", default=LIDAR,
                    help="which LiDAR topic to export (dual-rig bags carry two)")
    a = ap.parse_args(argv)

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from sensor_msgs_py import point_cloud2

    os.makedirs(a.out, exist_ok=True)
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=a.bag, storage_id="sqlite3"),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}

    n = written = 0
    while r.has_next():
        topic, data, t_ns = r.read_next()
        if topic != a.topic:
            continue
        n += 1
        if (n - 1) % a.stride:
            continue
        m = deserialize_message(data, get_message(types[topic]))
        arr = point_cloud2.read_points(m, skip_nans=True)
        xyz = np.column_stack([np.asarray(arr["x"]), np.asarray(arr["y"]),
                               np.asarray(arr["z"])]).astype(np.float64)
        # The message header stamp, not the bag receive time, so the PCD name matches
        # the `t_ns` column label_frames.py writes.
        sec = m.header.stamp.sec
        nsec = m.header.stamp.nanosec
        write_pcd(os.path.join(a.out, f"{sec}-{nsec:09d}.pcd"), xyz)
        written += 1
        if written % 500 == 0:
            print(f"  {written} written", flush=True)
        if a.limit and written >= a.limit:
            break
    print(f"{written} pcds (of {n} frames, stride {a.stride}) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
