"""Bag reading for the BEV pipeline.

Extends the ``band_tuner.py`` ``rosbag2_py.SequentialReader`` pattern to pull,
in one pass over a bag, the topics this pipeline needs:

  * ``sensor_msgs/PointCloud2``  — LiDAR, the primary geometry input.
  * ``nav_msgs/Odometry``        — robot pose (position + orientation).
  * ``sensor_msgs/Image``        — nearest camera frame (for the camera
                                   extractor + VLM auto-labeling).
  * ``sensor_msgs/Imu``          — optional gravity vector when odometry
                                   orientation is not trusted.

The public entry point is :func:`read_bag_frames`, a generator yielding
time-synchronized :class:`Frame` objects (one per LiDAR message, with the
nearest-in-time pose and image attached). A frame stride / sample rate keeps
memory bounded on multi-GB bags.

Pose sources
------------
Most bags publish a dedicated ``nav_msgs/Odometry`` topic; pass its name as
``odom_topic``. Bags without one (e.g. the Penn meadow bag) can fall back to a
GPS-derived local-ENU pose via :class:`GpsPoseSource` — see
:func:`build_pose_source`.

No ROS graph is required; everything is read straight from the bag.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Data structures                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class Pose:
    """A single robot pose in the bag's odometry/world frame."""

    t_ns: int
    position: np.ndarray          # (3,) float64  [x, y, z]
    orientation: np.ndarray       # (4,) float64  quaternion [x, y, z, w]

    def matrix(self) -> np.ndarray:
        """4x4 homogeneous transform world<-body for this pose."""
        return _pose_to_matrix(self.position, self.orientation)


@dataclass
class Frame:
    """One synchronized LiDAR frame plus the nearest pose / image."""

    t_ns: int
    points: np.ndarray                     # (N, 4) float32  [x, y, z, intensity]
    pose: Optional[Pose] = None
    image: Optional[np.ndarray] = None     # (H, W, 3) uint8 RGB, or None
    image_t_ns: Optional[int] = None
    frame_id: str = ""
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Quaternion / transform helpers                                              #
# --------------------------------------------------------------------------- #

def quat_to_yaw(quat_xyzw: np.ndarray) -> float:
    """Extract yaw (rotation about Z) from an xyzw quaternion."""
    x, y, z, w = [float(v) for v in quat_xyzw]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _pose_to_matrix(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from position + xyzw quaternion."""
    x, y, z, w = quat_xyzw
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        rot = np.eye(3)
    else:
        x, y, z, w = x / n, y / n, z / n, w / n
        rot = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ])
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = position
    return m


# --------------------------------------------------------------------------- #
# Low-level bag helpers                                                        #
# --------------------------------------------------------------------------- #

def _storage_id(bag_dir: str) -> str:
    """Guess the rosbag2 storage id ('mcap' or 'sqlite3') from metadata.yaml."""
    meta = os.path.join(bag_dir, "metadata.yaml")
    if os.path.exists(meta):
        with open(meta) as f:
            txt = f.read()
        if "mcap" in txt:
            return "mcap"
        if "sqlite3" in txt:
            return "sqlite3"
    # Fall back to file extension of any storage file present.
    for fn in os.listdir(bag_dir) if os.path.isdir(bag_dir) else []:
        if fn.endswith(".mcap"):
            return "mcap"
        if fn.endswith(".db3"):
            return "sqlite3"
    return "sqlite3"


def _resolve_bag_dir(bag_path: str) -> str:
    """rosbag2 wants the directory containing metadata.yaml, not a data file."""
    if os.path.isfile(bag_path) and (bag_path.endswith(".mcap") or bag_path.endswith(".db3")):
        return os.path.dirname(bag_path) or "."
    return bag_path


def _open_reader(bag_path: str, storage_id: Optional[str] = None):
    """Open a SequentialReader; returns (reader, type_map)."""
    try:
        import rosbag2_py
    except ImportError as e:  # pragma: no cover - env-dependent
        sys.exit(f"Missing ROS2 deps: {e}\nSource your ROS2 workspace first.")

    bag_dir = _resolve_bag_dir(bag_path)
    sid = storage_id or _storage_id(bag_dir)
    so = rosbag2_py.StorageOptions(uri=bag_dir, storage_id=sid)
    reader = rosbag2_py.SequentialReader()
    reader.open(so, rosbag2_py.ConverterOptions("", ""))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, type_map


def list_topics(bag_path: str, storage_id: Optional[str] = None) -> dict:
    """Return {topic_name: type_string} for a bag."""
    reader, type_map = _open_reader(bag_path, storage_id)
    del reader
    return type_map


def _points_from_cloud(msg) -> np.ndarray:
    """Extract an (N, 4) [x, y, z, intensity] float32 array from a PointCloud2."""
    from sensor_msgs_py.point_cloud2 import read_points

    field_names = [f.name for f in msg.fields]
    has_int = "intensity" in field_names
    want = ["x", "y", "z", "intensity"] if has_int else ["x", "y", "z"]
    raw = read_points(msg, field_names=want, skip_nans=True)
    if len(raw) == 0:
        return np.empty((0, 4), dtype=np.float32)
    if has_int:
        pts = np.column_stack([raw["x"], raw["y"], raw["z"], raw["intensity"]])
    else:
        xyz = np.column_stack([raw["x"], raw["y"], raw["z"]])
        pts = np.column_stack([xyz, np.zeros(len(xyz), dtype=np.float32)])
    return pts.astype(np.float32)


_PC2_DTYPES = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
               5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def _points_from_mcap_cloud(msg) -> np.ndarray:
    """Parse an (N,4) [x,y,z,intensity] array from an mcap-decoded PointCloud2.

    mcap_ros2 yields a dynamically-generated message class (not a real ROS
    PointCloud2), so sensor_msgs_py.read_points can't be used — parse the raw
    byte buffer via the field offsets/datatypes directly.
    """
    n = int(msg.width) * int(msg.height)
    if n == 0:
        return np.empty((0, 4), dtype=np.float32)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, int(msg.point_step))
    fields = {f.name: f for f in msg.fields}

    def col(name):
        f = fields[name]
        dt = _PC2_DTYPES.get(f.datatype, np.float32)
        sz = np.dtype(dt).itemsize
        return raw[:, f.offset:f.offset + sz].copy().view(dt).ravel().astype(np.float64)

    x, y, z = col("x"), col("y"), col("z")
    inten = col("intensity") if "intensity" in fields else np.zeros(n)
    pts = np.column_stack([x, y, z, inten]).astype(np.float32)
    m = np.isfinite(pts[:, :3]).all(axis=1)
    return pts[m]


def _image_to_rgb(msg) -> Optional[np.ndarray]:
    """Decode a sensor_msgs/Image to an (H, W, 3) uint8 RGB array."""
    enc = (msg.encoding or "").lower()
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    h, w = msg.height, msg.width
    try:
        if enc in ("rgb8", "bgr8"):
            img = buf.reshape(h, w, 3)
            if enc == "bgr8":
                img = img[:, :, ::-1]
        elif enc in ("rgba8", "bgra8"):
            img = buf.reshape(h, w, 4)[:, :, :3]
            if enc == "bgra8":
                img = img[:, :, ::-1]
        elif enc in ("mono8",):
            img = np.repeat(buf.reshape(h, w, 1), 3, axis=2)
        else:
            # Unknown encoding: best-effort assume 3-channel.
            img = buf.reshape(h, w, -1)[:, :, :3]
    except ValueError:
        return None
    return np.ascontiguousarray(img)


# --------------------------------------------------------------------------- #
# Pose sources                                                                 #
# --------------------------------------------------------------------------- #

class PoseSource:
    """Nearest-timestamp pose lookup backed by a sorted pose list."""

    def __init__(self, poses: Sequence[Pose]):
        self._poses = list(poses)
        self._ts = np.array([p.t_ns for p in self._poses], dtype=np.int64)

    def __len__(self) -> int:
        return len(self._poses)

    def nearest(self, t_ns: int, max_dt_ns: int = 200_000_000) -> Optional[Pose]:
        """Nearest pose within ``max_dt_ns`` (default 200 ms), else None."""
        if len(self._ts) == 0:
            return None
        idx = int(np.searchsorted(self._ts, t_ns))
        cands = []
        if idx < len(self._ts):
            cands.append(idx)
        if idx > 0:
            cands.append(idx - 1)
        best, best_dt = None, None
        for i in cands:
            dt = abs(int(self._ts[i]) - t_ns)
            if best_dt is None or dt < best_dt:
                best, best_dt = i, dt
        if best is None or best_dt > max_dt_ns:
            return None
        return self._poses[best]


def _collect_odometry(bag_path, odom_topic, storage_id=None) -> List[Pose]:
    """Read all Odometry messages on ``odom_topic`` into a Pose list."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader, type_map = _open_reader(bag_path, storage_id)
    if odom_topic not in type_map:
        raise KeyError(f"odom topic {odom_topic!r} not in bag; have {sorted(type_map)}")
    msg_type = get_message(type_map[odom_topic])
    try:
        import rosbag2_py
        reader.set_filter(rosbag2_py.StorageFilter(topics=[odom_topic]))
    except Exception:
        pass
    poses: List[Pose] = []
    while reader.has_next():
        name, data, ts = reader.read_next()
        if name != odom_topic:
            continue
        msg = deserialize_message(data, msg_type)
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        poses.append(Pose(
            t_ns=ts,
            position=np.array([p.x, p.y, p.z], dtype=np.float64),
            orientation=np.array([o.x, o.y, o.z, o.w], dtype=np.float64),
        ))
    return poses


def _gps_to_local_enu(lat, lon, alt, lat0, lon0, alt0) -> np.ndarray:
    """Equirectangular local-ENU approximation relative to (lat0, lon0, alt0)."""
    r_earth = 6_378_137.0
    lat_rad = math.radians(lat0)
    east = math.radians(lon - lon0) * r_earth * math.cos(lat_rad)
    north = math.radians(lat - lat0) * r_earth
    up = alt - alt0
    return np.array([east, north, up], dtype=np.float64)


def _collect_gps_poses(bag_path, gps_topic, storage_id=None) -> List[Pose]:
    """Read a NavSatFix / GPSFix topic into local-ENU Pose list.

    Orientation is derived from the heading between consecutive fixes (yaw
    only), which is enough for the submap's *relative* motion signal. Roll and
    pitch are assumed ~0, so gravity alignment on these bags is approximate.
    """
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader, type_map = _open_reader(bag_path, storage_id)
    if gps_topic not in type_map:
        raise KeyError(f"gps topic {gps_topic!r} not in bag; have {sorted(type_map)}")
    msg_type = get_message(type_map[gps_topic])
    try:
        import rosbag2_py
        reader.set_filter(rosbag2_py.StorageFilter(topics=[gps_topic]))
    except Exception:
        pass

    raw = []
    while reader.has_next():
        name, data, ts = reader.read_next()
        if name != gps_topic:
            continue
        msg = deserialize_message(data, msg_type)
        lat = getattr(msg, "latitude", None)
        lon = getattr(msg, "longitude", None)
        alt = getattr(msg, "altitude", 0.0)
        if lat is None or lon is None:
            continue
        if not (math.isfinite(lat) and math.isfinite(lon)):
            continue
        raw.append((ts, lat, lon, alt))
    if not raw:
        return []

    _, lat0, lon0, alt0 = raw[0]
    positions = [(_gps_to_local_enu(lat, lon, alt, lat0, lon0, alt0), ts)
                 for ts, lat, lon, alt in raw]

    poses: List[Pose] = []
    for i, (pos, ts) in enumerate(positions):
        # yaw from displacement to next fix (fallback to previous)
        if i + 1 < len(positions):
            nxt = positions[i + 1][0]
        elif i > 0:
            nxt = pos + (pos - positions[i - 1][0])
        else:
            nxt = pos
        dx, dy = nxt[0] - pos[0], nxt[1] - pos[1]
        yaw = math.atan2(dy, dx) if (dx * dx + dy * dy) > 1e-6 else 0.0
        quat = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
        poses.append(Pose(t_ns=ts, position=pos, orientation=quat))
    return poses


def build_pose_source(bag_path, odom_topic=None, gps_topic=None,
                      storage_id=None) -> PoseSource:
    """Build a PoseSource from odometry if available, else GPS fallback."""
    if odom_topic:
        poses = _collect_odometry(bag_path, odom_topic, storage_id)
        if poses:
            return PoseSource(poses)
    if gps_topic:
        poses = _collect_gps_poses(bag_path, gps_topic, storage_id)
        return PoseSource(poses)
    raise ValueError("no odom_topic (with data) or gps_topic given for pose source")


# --------------------------------------------------------------------------- #
# Image index (lazy nearest-image lookup)                                     #
# --------------------------------------------------------------------------- #

def _collect_image_timestamps(bag_path, image_topic, storage_id=None) -> np.ndarray:
    """Return a sorted int64 array of message timestamps for image_topic."""
    reader, type_map = _open_reader(bag_path, storage_id)
    if image_topic not in type_map:
        return np.array([], dtype=np.int64)
    try:
        import rosbag2_py
        reader.set_filter(rosbag2_py.StorageFilter(topics=[image_topic]))
    except Exception:
        pass
    ts_list = []
    while reader.has_next():
        name, _data, ts = reader.read_next()
        if name == image_topic:
            ts_list.append(ts)
    return np.array(sorted(ts_list), dtype=np.int64)


# --------------------------------------------------------------------------- #
# Main synchronized reader                                                     #
# --------------------------------------------------------------------------- #

def read_bag_frames(
    bag_path: str,
    lidar_topic: str,
    odom_topic: Optional[str] = None,
    image_topic: Optional[str] = None,
    gps_topic: Optional[str] = None,
    *,
    storage_id: Optional[str] = None,
    sample_hz: Optional[float] = None,
    max_frames: Optional[int] = None,
    with_images: bool = True,
    max_pose_dt_ns: int = 200_000_000,
) -> Iterator[Frame]:
    """Yield time-synchronized :class:`Frame` objects from a bag.

    One frame is produced per LiDAR message (optionally subsampled to
    ``sample_hz``), with the most-recent pose and — if ``with_images`` — the
    most-recent camera image attached.

    Implemented as a **single forward pass** over the bag: odometry poses are
    accumulated as they stream by, the latest image message is buffered
    (un-decoded), and only when a LiDAR frame is actually emitted do we
    deserialize the point cloud and decode the buffered image. Combined with
    ``max_frames``, this lets a short read touch only the head of a multi-GB
    bag instead of rescanning the whole file.

    Because messages arrive in time order, "most recent" pose/image lag the
    LiDAR stamp by at most one inter-message interval (odom ~100 Hz → <=10 ms;
    image ~24 Hz → <=40 ms), which is well within tolerance for this pipeline.
    GPS fallback (``gps_topic``, no ``odom_topic``) is handled by a pre-pass
    since heading needs neighbouring fixes; that path does scan the GPS topic
    once up front.
    """
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader, type_map = _open_reader(bag_path, storage_id)
    if lidar_topic not in type_map:
        raise KeyError(f"lidar topic {lidar_topic!r} not in bag; have {sorted(type_map)}")
    lidar_type = get_message(type_map[lidar_topic])

    have_odom = bool(odom_topic) and odom_topic in type_map
    odom_type = get_message(type_map[odom_topic]) if have_odom else None
    img_type = get_message(type_map[image_topic]) if (with_images and image_topic
                                                      and image_topic in type_map) else None

    # GPS fallback handled INLINE (single pass) when there's no odometry topic:
    # convert each fix to local ENU and derive yaw from the backward difference
    # to the previous fix. Avoids a full extra scan of the bag (critical for
    # large, unindexed mcaps where a pre-pass can take many minutes).
    have_gps = (not have_odom) and bool(gps_topic) and gps_topic in type_map
    gps_type = get_message(type_map[gps_topic]) if have_gps else None
    gps_ref = None                 # (lat0, lon0, alt0)
    last_gps_enu = None            # previous fix ENU position

    sample_dt_ns = int(1e9 / sample_hz) if sample_hz else 0
    last_emit_ns: Optional[int] = None
    n_emitted = 0

    last_pose: Optional[Pose] = None
    last_img_raw = None        # (ts, data) of most recent image message

    while reader.has_next():
        if max_frames is not None and n_emitted >= max_frames:
            break
        name, data, ts = reader.read_next()

        if have_odom and name == odom_topic:
            msg = deserialize_message(data, odom_type)
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            last_pose = Pose(
                t_ns=ts,
                position=np.array([p.x, p.y, p.z], dtype=np.float64),
                orientation=np.array([o.x, o.y, o.z, o.w], dtype=np.float64),
            )
            continue

        if have_gps and name == gps_topic:
            gmsg = deserialize_message(data, gps_type)
            lat = getattr(gmsg, "latitude", None)
            lon = getattr(gmsg, "longitude", None)
            alt = getattr(gmsg, "altitude", 0.0)
            if lat is not None and lon is not None and math.isfinite(lat) and math.isfinite(lon):
                if gps_ref is None:
                    gps_ref = (lat, lon, alt)
                enu = _gps_to_local_enu(lat, lon, alt, *gps_ref)
                if last_gps_enu is not None:
                    dx, dy = enu[0] - last_gps_enu[0], enu[1] - last_gps_enu[1]
                    yaw = math.atan2(dy, dx) if (dx * dx + dy * dy) > 1e-6 else (
                        quat_to_yaw(last_pose.orientation) if last_pose else 0.0)
                else:
                    yaw = 0.0
                quat = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
                last_pose = Pose(t_ns=ts, position=enu, orientation=quat)
                last_gps_enu = enu
            continue

        if img_type is not None and name == image_topic:
            last_img_raw = (ts, data)   # buffer only; decode lazily on emit
            continue

        if name != lidar_topic:
            continue

        if sample_dt_ns and last_emit_ns is not None and (ts - last_emit_ns) < sample_dt_ns:
            continue

        msg = deserialize_message(data, lidar_type)
        points = _points_from_cloud(msg)

        # last_pose is maintained inline for both odom and GPS paths.
        if last_pose is not None and abs(last_pose.t_ns - ts) <= max_pose_dt_ns:
            pose = last_pose
        else:
            pose = None

        image, image_t = None, None
        if img_type is not None and last_img_raw is not None:
            img_ts, img_data = last_img_raw
            if abs(img_ts - ts) <= max_pose_dt_ns:
                image = _image_to_rgb(deserialize_message(img_data, img_type))
                image_t = img_ts

        yield Frame(
            t_ns=ts,
            points=points,
            pose=pose,
            image=image,
            image_t_ns=image_t,
            frame_id=getattr(msg.header, "frame_id", ""),
        )
        last_emit_ns = ts
        n_emitted += 1


def _find_mcap_file(bag_path: str) -> str:
    """Return the .mcap file path for a bag dir or file."""
    if os.path.isfile(bag_path) and bag_path.endswith(".mcap"):
        return bag_path
    d = _resolve_bag_dir(bag_path)
    cands = [f for f in os.listdir(d) if f.endswith(".mcap")]
    if not cands:
        raise FileNotFoundError(f"no .mcap in {d}")
    return os.path.join(d, sorted(cands)[0])


def read_bag_frames_mcap_stream(
    bag_path: str,
    lidar_topic: str,
    odom_topic: Optional[str] = None,
    image_topic: Optional[str] = None,
    gps_topic: Optional[str] = None,
    *,
    sample_hz: Optional[float] = None,
    max_frames: Optional[int] = None,
    with_images: bool = True,
    max_pose_dt_ns: int = 200_000_000,
    **_ignored,
) -> Iterator[Frame]:
    """Single-pass reader for UNINDEXED mcap bags via mcap StreamReader.

    rosbag2's mcap reader wants a summary/index to read in timestamp order; on
    an unindexed multi-GB mcap it stalls (must scan the whole file first). This
    reads records linearly from the start (decompressing chunks), so frames
    stream immediately. Record log_time gives timestamps without decoding, so
    LiDAR clouds are only parsed at sample points.
    """
    from mcap.stream_reader import StreamReader
    from mcap.records import Schema, Channel, Message
    from mcap_ros2.decoder import DecoderFactory

    mcap_path = _find_mcap_file(bag_path)
    factory = DecoderFactory()
    schemas, channels, decoders = {}, {}, {}

    def decoder_for(ch):
        if ch.id not in decoders:
            decoders[ch.id] = factory.decoder_for(ch.message_encoding, schemas[ch.schema_id])
        return decoders[ch.id]

    sample_dt_ns = int(1e9 / sample_hz) if sample_hz else 0
    last_emit_ns: Optional[int] = None
    n_emitted = 0
    last_pose: Optional[Pose] = None
    last_img = None                      # (t_ns, decoded_image_msg)
    gps_ref = None
    last_gps_enu = None

    reader = StreamReader(open(mcap_path, "rb"))
    for rec in reader.records:
        if isinstance(rec, Schema):
            schemas[rec.id] = rec
            continue
        if isinstance(rec, Channel):
            channels[rec.id] = rec
            continue
        if not isinstance(rec, Message):
            continue
        ch = channels.get(rec.channel_id)
        if ch is None:
            continue
        topic = ch.topic
        ts = rec.log_time

        if odom_topic and topic == odom_topic:
            m = decoder_for(ch)(rec.data)
            p = m.pose.pose.position
            o = m.pose.pose.orientation
            last_pose = Pose(t_ns=ts, position=np.array([p.x, p.y, p.z], dtype=np.float64),
                             orientation=np.array([o.x, o.y, o.z, o.w], dtype=np.float64))
            continue

        if gps_topic and topic == gps_topic and not odom_topic:
            m = decoder_for(ch)(rec.data)
            lat, lon = getattr(m, "latitude", None), getattr(m, "longitude", None)
            alt = getattr(m, "altitude", 0.0)
            if lat is not None and lon is not None and math.isfinite(lat) and math.isfinite(lon):
                if gps_ref is None:
                    gps_ref = (lat, lon, alt)
                enu = _gps_to_local_enu(lat, lon, alt, *gps_ref)
                if last_gps_enu is not None:
                    dx, dy = enu[0] - last_gps_enu[0], enu[1] - last_gps_enu[1]
                    yaw = math.atan2(dy, dx) if (dx * dx + dy * dy) > 1e-6 else (
                        quat_to_yaw(last_pose.orientation) if last_pose else 0.0)
                else:
                    yaw = 0.0
                last_pose = Pose(t_ns=ts, position=enu,
                                 orientation=np.array([0.0, 0.0, math.sin(yaw / 2),
                                                       math.cos(yaw / 2)]))
                last_gps_enu = enu
            continue

        if with_images and image_topic and topic == image_topic:
            last_img = (ts, rec.data, ch)      # buffer raw; decode lazily on emit
            continue

        if topic != lidar_topic:
            continue
        if sample_dt_ns and last_emit_ns is not None and (ts - last_emit_ns) < sample_dt_ns:
            continue
        if max_frames is not None and n_emitted >= max_frames:
            break

        cloud = decoder_for(ch)(rec.data)
        points = _points_from_mcap_cloud(cloud)
        pose = last_pose if (last_pose is not None
                             and abs(last_pose.t_ns - ts) <= max_pose_dt_ns) else None
        image, image_t = None, None
        if with_images and last_img is not None and abs(last_img[0] - ts) <= max_pose_dt_ns:
            image = _image_to_rgb(decoder_for(last_img[2])(last_img[1]))
            image_t = last_img[0]
        yield Frame(t_ns=ts, points=points, pose=pose, image=image,
                    image_t_ns=image_t, frame_id=getattr(cloud.header, "frame_id", ""))
        last_emit_ns = ts
        n_emitted += 1


def count_frames(bag_path, lidar_topic, storage_id=None) -> int:
    """Cheap pass counting LiDAR messages (for progress bars / planning)."""
    reader, type_map = _open_reader(bag_path, storage_id)
    if lidar_topic not in type_map:
        return 0
    try:
        import rosbag2_py
        reader.set_filter(rosbag2_py.StorageFilter(topics=[lidar_topic]))
    except Exception:
        pass
    n = 0
    while reader.has_next():
        name, _d, _t = reader.read_next()
        if name == lidar_topic:
            n += 1
    return n
