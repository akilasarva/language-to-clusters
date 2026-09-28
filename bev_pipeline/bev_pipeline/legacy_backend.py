"""Offline replica of the legacy `clustering` backend (AE -> HDBSCAN).

Why this exists
---------------
To compare the legacy autoencoder+HDBSCAN classifier against the routed sensor
decision tree on the same frames. The legacy implementation only exists as a
live ROS node
(``clustering/clustering/live_cluster_inference_node.py``) which cannot be run on
cached frames, and which additionally cannot be launched outside its own source
tree because ``clustering/setup.py`` has no ``encoder_weights/`` install rule.

So this module reproduces the node's inference path exactly, against the cached
``points/frame_*.npy`` raw clouds, so both backends can be evaluated on identical
input.

Faithfulness
------------
The featurization is a port of ``lidar_processor.get_ranges_from_points`` and the
constants come from ``LiveClusterInferenceNode.__init__``:

    num_ranges 72 | range 0.5-8.0 m | z-slice [-0.5, 0.15] | embedding 16
    density filter r=0.30 m, min_neighbors=4 | mode-filter smoothing over 10

Two details matter for correctness:

1. **Raw, ground-INCLUDED, un-levelled points.** The node subscribes to the raw
   PointCloud2, so the z-slice is taken in the sensor frame with the ground still
   present. It must therefore be fed ``datasets/<env>/points/frame_*.npy``, NOT
   ``modality_points.npy`` (which is tilt-corrected and ground-removed by
   ``plane_level_and_remove``). Feeding the latter would silently evaluate a
   different algorithm.

2. **The un-tuned reassignment threshold.** ``distance_threshold_for_reassignment``
   is hardcoded to 6 in the node, next to the comment
   ``# <<< REPLACE THIS WITH YOUR VALUE``. The training script computes a proper
   95th-percentile value per environment, but the node does not read it.
   Reproduced as-is (and exposed as a parameter) because the point is to
   evaluate the backend as deployed, not an idealized version of it.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:      # pragma: no cover
    _HAS_SCIPY = False


# --------------------------------------------------------------------------- #
# Config — mirrors LiveClusterInferenceNode.__init__                          #
# --------------------------------------------------------------------------- #

@dataclass
class LegacyConfig:
    num_ranges: int = 72
    max_lidar_range: float = 8.0
    min_lidar_range: float = 0.5
    z_threshold_upper: float = 0.15
    z_threshold_lower: float = -0.5
    z_threshold_upper_2: float = 0.0
    z_threshold_lower_2: float = 0.0
    use_intensity: bool = False
    int_lower: float = 0.0
    int_upper: float = 255.0
    density_radius: float = 0.30
    min_neighbors: int = 4
    embedding_size: int = 16
    #: See the module docstring — deployed value, never calibrated.
    distance_threshold_for_reassignment: float = 6.0
    #: Mode filter over the last N predictions, as in _get_smoothed_label.
    smooth_window: int = 10

    def as_dict(self) -> dict:
        return {
            "num_ranges": self.num_ranges,
            "max_lidar_range": self.max_lidar_range,
            "min_lidar_range": self.min_lidar_range,
            "z_threshold_upper": self.z_threshold_upper,
            "z_threshold_lower": self.z_threshold_lower,
            "z_threshold_upper_2": self.z_threshold_upper_2,
            "z_threshold_lower_2": self.z_threshold_lower_2,
            "use_intensity": self.use_intensity,
            "int_lower": self.int_lower,
            "int_upper": self.int_upper,
            "density_radius": self.density_radius,
            "min_neighbors": self.min_neighbors,
        }


# --------------------------------------------------------------------------- #
# Featurization — port of lidar_processor.get_ranges_from_points              #
# --------------------------------------------------------------------------- #

def ranges_from_points(points: np.ndarray, cfg: LegacyConfig) -> np.ndarray:
    """72-sector minimum-range 'synthetic 2D scan'.

    Vectorized over sectors (the original loops 72 times recomputing every point
    angle); numerically identical because both take the per-sector minimum with a
    half-increment angular tolerance.
    """
    n = cfg.num_ranges
    fill = np.full(n, cfg.max_lidar_range, dtype=np.float64)
    if points is None or points.size == 0:
        return fill

    pts = np.asarray(points, dtype=np.float64)
    z = pts[:, 2]
    m1 = (z >= cfg.z_threshold_lower) & (z <= cfg.z_threshold_upper)
    m2 = (z >= cfg.z_threshold_lower_2) & (z <= cfg.z_threshold_upper_2)
    mask = m1 | m2
    if cfg.use_intensity and pts.shape[1] > 3:
        inten = pts[:, 3]
        mask &= (inten >= cfg.int_lower) & (inten <= cfg.int_upper)
    pts = pts[mask]
    if pts.size == 0:
        return fill

    dist = np.linalg.norm(pts[:, :2], axis=1)
    rmask = (dist >= cfg.min_lidar_range) & (dist <= cfg.max_lidar_range)
    pts, dist = pts[rmask], dist[rmask]
    if pts.size == 0:
        return fill

    # Spatial density filter — drop isolated returns (rain / multipath).
    if (_HAS_SCIPY and cfg.density_radius > 0 and cfg.min_neighbors > 0
            and len(pts) > cfg.min_neighbors):
        counts = cKDTree(pts[:, :2]).query_ball_point(
            pts[:, :2], r=cfg.density_radius, return_length=True)
        keep = (counts - 1) >= cfg.min_neighbors
        pts, dist = pts[keep], dist[keep]
    if pts.size == 0:
        return fill

    # Assign each point to its sector and take the per-sector minimum range.
    inc = 360.0 / n
    ang = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360.0
    sector = np.floor((ang + inc / 2.0) % 360.0 / inc).astype(int) % n
    out = fill.copy()
    np.minimum.at(out, sector, dist)
    return out


# --------------------------------------------------------------------------- #
# Encoder                                                                     #
# --------------------------------------------------------------------------- #

def _build_encoder(embedding_size: int):
    """The node's LidarEncoder: 3x Conv2d(1x3) + AdaptiveMaxPool + Linear."""
    import torch.nn as nn

    class LidarEncoder(nn.Module):
        def __init__(self, embedding_size: int) -> None:
            super().__init__()
            self.conv2d_1 = nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0, 1))
            self.conv2d_2 = nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 1))
            self.conv2d_3 = nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1))
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = nn.Linear(64, embedding_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv2d_1(x))
            x = self.relu(self.conv2d_2(x))
            x = self.relu(self.conv2d_3(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    return LidarEncoder(embedding_size)


# --------------------------------------------------------------------------- #
# The backend                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class LegacyBackend:
    weights_dir: str
    env: str
    cfg: LegacyConfig = field(default_factory=LegacyConfig)

    def __post_init__(self) -> None:
        import torch

        w = self.weights_dir
        e = self.env
        self._scaler = pickle.load(open(os.path.join(w, f"scaler_{e}.pkl"), "rb"))
        self._hdb = pickle.load(open(os.path.join(w, f"hdbscan_model_{e}.pkl"), "rb"))
        cpath = os.path.join(w, f"cluster_centroids_{e}.pkl")
        self._centroids = pickle.load(open(cpath, "rb")) if os.path.exists(cpath) else {}

        lpath = os.path.join(w, f"cluster_id_to_label_{e}.json")
        # bridge1_carla has trained weights but NO label JSON — it was never
        # labelled. Callers must handle an empty label map.
        self.id_to_label: dict[int, str] = {}
        if os.path.exists(lpath):
            self.id_to_label = {int(k): v for k, v in json.load(open(lpath)).items()}

        self._encoder = _build_encoder(self.cfg.embedding_size)
        sd = torch.load(os.path.join(w, f"lidar_encoder_autoencoder_{e}.pth"),
                        map_location="cpu", weights_only=False)
        # The checkpoint is the full autoencoder; keep only the encoder half.
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        self._encoder.load_state_dict(enc_sd)
        self._encoder.eval()

        self._history: list[int] = []

    # -- inference ---------------------------------------------------------- #

    def embed(self, points: np.ndarray) -> np.ndarray:
        import torch

        rng = ranges_from_points(points, self.cfg) / self.cfg.max_lidar_range
        x = torch.tensor(rng, dtype=torch.float32).reshape(1, 1, 1, -1)
        with torch.no_grad():
            return self._encoder(x).numpy()

    def raw_cluster(self, points: np.ndarray) -> int:
        """Cluster id before smoothing. -2 means 'new cluster / outlier'."""
        import hdbscan

        emb = self._scaler.transform(self.embed(points))
        labels, _ = hdbscan.approximate_predict(self._hdb, emb)
        cid = int(labels[0])
        if cid != -1:
            return cid
        return self._reassign(emb[0])

    def _reassign(self, emb: np.ndarray) -> int:
        """Nearest-centroid rescue for HDBSCAN noise, as in reassign_labels."""
        if not self._centroids:
            return -1
        best, best_d = -2, np.inf
        for cid, cen in self._centroids.items():
            d = float(np.linalg.norm(emb - np.asarray(cen, dtype=np.float64)))
            if d < best_d:
                best, best_d = int(cid), d
        if best_d <= self.cfg.distance_threshold_for_reassignment:
            return best
        return -2

    def predict_stream(self, frames: Iterable[np.ndarray],
                       smooth: bool = True) -> tuple[list[int], list[int]]:
        """Run a whole bag. Returns (raw_ids, smoothed_ids).

        Smoothing is the node's mode filter over the last ``smooth_window``
        predictions, which is why this is a stream method and not per-frame: the
        deployed behaviour is order-dependent.
        """
        from collections import Counter

        raw: list[int] = []
        sm: list[int] = []
        hist: list[int] = []
        for pts in frames:
            cid = self.raw_cluster(pts)
            raw.append(cid)
            hist.append(cid)
            if len(hist) > self.cfg.smooth_window:
                hist.pop(0)
            sm.append(Counter(hist).most_common(1)[0][0] if smooth else cid)
        return raw, sm

    def labels_for(self, ids: Iterable[int]) -> list[str]:
        return [self.id_to_label.get(int(i), "unknown") for i in ids]


# --------------------------------------------------------------------------- #
# Legacy label -> enclosure axis                                              #
# --------------------------------------------------------------------------- #

#: The legacy vocabulary mapped onto the ENCLOSURE axis (open / along_edge /
#: passage) — the only space in which the two backends are directly comparable.
#:
#: NOTE the legacy vocabulary has NO equivalent of `path`. "In Corridor" is a
#: both-sides-enclosed reading, so a plain open road or sidewalk (the most
#: common real class) cannot be expressed at all. This is a limitation of the
#: legacy backend, not a mapping choice.
LEGACY_TO_ENCLOSURE = {
    "Open Space":      "open",
    "In Intersection": "open",
    "Along Wall":      "along_edge",
    "In Corridor":     "passage",
    "Enter Corridor":  "passage",
}


def legacy_to_enclosure(label: str) -> str | None:
    return LEGACY_TO_ENCLOSURE.get(label)


__all__ = [
    "LegacyBackend",
    "LegacyConfig",
    "LEGACY_TO_ENCLOSURE",
    "legacy_to_enclosure",
    "ranges_from_points",
]
