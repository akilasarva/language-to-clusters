#!/usr/bin/env python3
"""A panel of filtered 2-D scans, one row per cluster CATEGORY. What the labels look like.

Two sources, because the interesting comparison is old-vs-new:

    # a historical weight set: rows are its human-assigned label categories
    python3 scripts/viz_cluster_panel.py --weights og_vae_ground_lidar \
        --pcds ~/ground_lidar/ground_lidar_pcds

    # a new GT-labelled corpus: rows are the derived categories
    python3 scripts/viz_cluster_panel.py --labels <frames.csv> \
        --pcds ~/carla_data/town07_v2/town07_v2_pcds --axis topology

Each cell is one frame drawn the way the clusterer sees it: grey = all points, orange =
the points surviving the z-band and range filter, blue = the 72-bin min-range scan that
becomes the feature vector. If two rows look alike, no model will separate them; if they
look different and the model still fails, the model is the problem.

Uses `clustering.lidar_processor.get_ranges_from_points` directly, so the blue outline is
the actual model input rather than a lookalike.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import math
import os
import re
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUSTERING = os.path.join(os.path.dirname(PKG), "clustering")
sys.path.insert(0, PKG)
sys.path.insert(0, CLUSTERING)
from clustering.lidar_processor import get_ranges_from_points  # noqa: E402

WEIGHTS_ROOT = os.path.join(CLUSTERING, "clustering", "encoder_weights")


def read_pcd(path: str) -> np.ndarray:
    """Binary or ASCII PCD -> (N,3). Avoids an open3d dependency in the ROS python."""
    with open(path, "rb") as f:
        head, line = [], b""
        while not line.startswith(b"DATA"):
            line = f.readline()
            head.append(line)
        txt = b"".join(head).decode("ascii", "replace")
        n = int(re.search(r"POINTS (\d+)", txt).group(1))
        fields = re.search(r"FIELDS ([^\n]+)", txt).group(1).split()
        if b"binary" in line:
            k = len(fields)
            a = np.frombuffer(f.read(n * 4 * k), dtype=np.float32).reshape(-1, k)
            return a[:, :3].astype(float)
        rows = [list(map(float, l.split()[:3])) for l in f.read().decode().split("\n")
                if l.strip()]
        return np.asarray(rows, dtype=float)


def from_weights(name: str) -> dict[str, list[str]]:
    """{category: [pcd filename, ...]} from a historical weight set."""
    d = os.path.join(WEIGHTS_ROOT, name)
    mapf = glob.glob(os.path.join(d, "train_cluster_to_filepaths_*.txt"))
    labf = glob.glob(os.path.join(d, "cluster_id_to_label_*.json"))
    if not mapf:
        raise SystemExit(f"no train_cluster_to_filepaths_*.txt in {d}")
    import json
    lab = json.load(open(labf[0])) if labf else {}
    out: dict[str, list[str]] = collections.defaultdict(list)
    for line in open(mapf[0]):
        m = re.match(r"Cluster (-?\d+): (.*)", line.strip())
        if not m:
            continue
        cid = m.group(1)
        cat = lab.get(cid, f"cluster {cid}")
        if cid == "-1":
            cat = f"NOISE (-1) [{lab.get('-1', 'unlabelled')}]"
        out[cat].extend(x.strip() for x in m.group(2).split(",") if x.strip())
    return out


def from_labels(csv_path: str, axis: str) -> dict[str, list[str]]:
    """{category: [pcd filename, ...]} from a label_frames CSV, via stamp_ns."""
    out: dict[str, list[str]] = collections.defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        s = int(r["stamp_ns"])
        out[r[axis]].append(f"{s // 10**9}-{s % 10**9:09d}.pcd")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcds", required=True)
    ap.add_argument("--weights", default="", help="historical weight-set name")
    ap.add_argument("--labels", default="", help="label_frames CSV")
    ap.add_argument("--axis", default="topology",
                    choices=("topology", "enclosure", "relation"))
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--max-range", type=float, default=25.0)
    ap.add_argument("--min-range", type=float, default=3.0)
    ap.add_argument("--z", type=float, nargs=2, default=(-1.6, -0.5))
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if not (a.weights or a.labels):
        ap.error("give --weights or --labels")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = from_weights(a.weights) if a.weights else from_labels(a.labels, a.axis)
    # Keep only frames that actually exist on disk. Town07's PCDs are exported at
    # stride 10 while its label CSV has every frame, so an unfiltered even sample
    # renders a grid of "missing" and hides the data it was meant to show.
    have = set(os.listdir(a.pcds))
    cats = {k: [f for f in v if f in have] for k, v in cats.items()}
    cats = {k: v for k, v in cats.items() if v}
    order = sorted(cats, key=lambda k: (-len(cats[k]), k))
    cfg = dict(num_ranges=72, max_lidar_range=a.max_range, min_lidar_range=a.min_range,
               z_threshold_lower=a.z[0], z_threshold_upper=a.z[1],
               z_threshold_lower_2=0.0, z_threshold_upper_2=0.0,
               use_intensity=False, density_radius=0.30, min_neighbors=4)

    fig, axes = plt.subplots(len(order), a.cols, dpi=120,
                             figsize=(2.05 * a.cols, 2.15 * len(order)),
                             squeeze=False)
    fig.patch.set_facecolor("white")
    R = a.max_range
    ang = np.arange(72) * (2 * math.pi / 72)
    for row, cat in enumerate(order):
        files = cats[cat]
        pick = [files[int(len(files) * (i + 0.5) / a.cols)] for i in range(a.cols)]
        for col, fn in enumerate(pick):
            ax = axes[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            p = os.path.join(a.pcds, fn)
            if not os.path.exists(p):
                ax.text(.5, .5, "missing", ha="center", va="center", fontsize=7)
                continue
            pts = read_pcd(p)
            keep = ((pts[:, 2] >= a.z[0]) & (pts[:, 2] <= a.z[1]))
            rng = np.hypot(pts[:, 0], pts[:, 1])
            inr = keep & (rng <= R) & (rng >= a.min_range)
            ax.scatter(pts[:, 1], pts[:, 0], s=.3, c="#E5E7EB", edgecolors="none")
            ax.scatter(pts[inr, 1], pts[inr, 0], s=1.0, c="#C2410C", edgecolors="none")
            r = get_ranges_from_points(pts, cfg)
            xs, ys = r * np.cos(ang), r * np.sin(ang)
            ax.plot(np.append(ys, ys[0]), np.append(xs, xs[0]), lw=.7, color="#2563EB")
            ax.set_xlim(R, -R)
            ax.set_ylim(-R, R)
            ax.set_aspect("equal")
            if col == 0:
                ax.set_ylabel(f"{cat}\n(n={len(files)})", fontsize=7, rotation=0,
                              ha="right", va="center", labelpad=6)
    src = a.weights or f"{os.path.basename(a.labels)}:{a.axis}"
    fig.suptitle(f"filtered 2-D scans by category — {src}\n"
                 f"grey = all points · orange = kept by z{tuple(a.z)} and "
                 f"{a.min_range}-{R} m · blue = the 72-bin feature", fontsize=10)
    out = a.out or os.path.join(PKG, "reports", "frame_labels",
                                f"panel_{a.weights or a.axis}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out)
    print(f"wrote {out}")
    for c in order:
        print(f"  {c:46s} {len(cats[c]):6d} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
