#!/usr/bin/env python3
"""Train the cluster model and label its clusters from GROUND TRUTH, not from a human.

    ~/miniconda3/bin/python scripts/train_and_autolabel.py \
        --pcds ~/carla_data/town01_v2/town01_v2_pcds \
        --labels <frames.csv> \
        --name town01_v2

This replaces the manual labelling step. `cluster_training.py` renders sample
scans for each HDBSCAN cluster and asks an operator to type a label; here the label comes
from `label_frames.py`'s per-frame ground truth, joined by the PCD's own timestamp.

WHAT IS REUSED AND WHY
`LidarEncoder`, `LidarDecoder`, `Autoencoder` and `LidarDataset` are IMPORTED from
`clustering.cluster_training`, not copied. The model the weights are fitted with has to
be the model the live node builds, and a copied twin drifts from the original. Only the
training loop
and the labelling are written here, because the labelling is the thing being changed.

THE JOIN
`bag_to_pcd.py` names each file `<sec>-<nsec>.pcd` from the message header stamp, which
is the same clock `label_frames.py` writes as `t_ns`. So a PCD maps to its GT row exactly,
with no nearest-time search and no drift. A mismatch shows up as a low join rate, which
is reported rather than silently tolerated.

TWO AXES, SCORED SEPARATELY
`topology` (path/approach/junction/exit) and `enclosure` (open_space/along_edge/passage)
are independent questions and one cluster map cannot be voted on both at once. Each gets
its own assignment and its own confusion matrix. LiDAR is expected to carry the enclosure
axis far better than junction phase, which is barely per-frame perceivable, so one blended
number would hide the difference.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "clustering"))

from carla_gt_bridge.frame_labels import (assign_cluster_labels,  # noqa: E402
                                          to_cluster_map_blocks)

AXES = ("topology", "enclosure")


def pcd_t_ns(path: str) -> int | None:
    m = re.match(r"(\d+)-(\d+)\.pcd$", os.path.basename(path))
    return int(m.group(1)) * 10 ** 9 + int(m.group(2)) if m else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcds", required=True)
    ap.add_argument("--labels", required=True, help="label_frames.py CSV")
    ap.add_argument("--name", required=True, help="weight-set name")
    ap.add_argument("--out-dir", default="")
    # Road-level band: 0.4 .. 1.5 m above the road for a 2 m mount. Chosen because it
    # fills far more of the 72 bins than the older training and inference bands -- see
    # ROAD_LEVEL_BAND in frame_labels.py.
    ap.add_argument("--z-lower", type=float, default=-1.6)
    ap.add_argument("--z-upper", type=float, default=-0.5)
    ap.add_argument("--max-range", type=float, default=25.0)
    ap.add_argument("--min-range", type=float, default=1.0)
    ap.add_argument("--num-ranges", type=int, default=72)
    ap.add_argument("--embedding", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-cluster-size", type=int, default=25)
    ap.add_argument("--min-purity", type=float, default=0.6)
    ap.add_argument("--min-size", type=int, default=8)
    a = ap.parse_args(argv)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import hdbscan
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics

    from clustering.cluster_training import (Autoencoder, LidarDataset, LidarDecoder,
                                             LidarEncoder)

    out_dir = a.out_dir or os.path.join(
        os.path.dirname(PKG), "clustering", "clustering", "encoder_weights", a.name)
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "num_ranges": a.num_ranges, "max_lidar_range": a.max_range,
        "min_lidar_range": a.min_range,
        "z_threshold_lower": a.z_lower, "z_threshold_upper": a.z_upper,
        "z_threshold_lower_2": 0.0, "z_threshold_upper_2": 0.0,
        "use_intensity": False, "min_intensity": 32.0, "max_intensity": 68.0,
        "embedding_size": a.embedding,
        "angle_increment_deg": 360.0 / a.num_ranges,
        "hdbscan_min_cluster_size": a.min_cluster_size,
        "hdbscan_cluster_selection_epsilon": 0.0,
        "density_radius": 0.30, "min_neighbors": 4,
        "training_data_name": a.name, "written_by": "train_and_autolabel.py",
    }
    # Written FIRST, so the config exists even if training is interrupted. This is the
    # file that makes a checkpoint reusable; without it the weights are `bridge1_carla`.
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"config -> {out_dir}/config.json")
    print(f"  z-band [{a.z_lower}, {a.z_upper}] (sensor frame; rig 2 m up => "
          f"{a.z_lower + 2:.1f}..{a.z_upper + 2:.1f} m above the road), "
          f"{a.num_ranges} bins / {a.max_range} m")

    ds = LidarDataset(a.pcds, config)
    if len(ds) == 0:
        print("no pcds")
        return 1
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = Autoencoder(LidarEncoder(a.embedding),
                     LidarDecoder(a.embedding, a.num_ranges)).to(dev)
    opt = optim.Adam(ae.parameters(), lr=a.lr)
    crit = nn.MSELoss()

    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4)
    print(f"training on {len(ds)} scans for {a.epochs} epochs ({dev})")
    for ep in range(a.epochs):
        ae.train()
        tot = 0.0
        for x in dl:
            x = x.to(dev)
            loss = crit(ae(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep + 1}/{a.epochs} loss {tot / len(dl):.5f}", flush=True)
    ae.eval()
    torch.save(ae.state_dict(),
               os.path.join(out_dir, f"lidar_encoder_autoencoder_{a.name}.pth"))

    # -- embed, scale, cluster ------------------------------------------------ #
    seq = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=4)
    embs = []
    with torch.no_grad():
        for x in seq:
            embs.append(ae.encoder(x.to(dev)).cpu().numpy())
    E = np.vstack(embs)
    scaler = StandardScaler()
    Es = scaler.fit_transform(E)
    clus = hdbscan.HDBSCAN(min_cluster_size=a.min_cluster_size,
                           cluster_selection_epsilon=0.0, prediction_data=True,
                           core_dist_n_jobs=-1)
    cid = clus.fit_predict(Es)
    uniq = sorted(set(cid.tolist()))
    print(f"\n{len(E)} embeddings -> {len([c for c in uniq if c >= 0])} clusters "
          f"+ {int((cid == -1).sum())} noise ({100 * (cid == -1).mean():.0f}%)")

    for obj, fn in ((scaler, f"scaler_{a.name}.pkl"),
                    (clus, f"hdbscan_model_{a.name}.pkl")):
        with open(os.path.join(out_dir, fn), "wb") as f:
            pickle.dump(obj, f)
    cents = {int(c): Es[cid == c].mean(axis=0) for c in uniq if c >= 0}
    with open(os.path.join(out_dir, f"cluster_centroids_{a.name}.pkl"), "wb") as f:
        pickle.dump(cents, f)

    # -- join to ground truth by timestamp ------------------------------------ #
    # Keyed on stamp_ns (the LiDAR header stamp), NOT t_ns (the bag receive time).
    # bag_to_pcd.py names each file from the header stamp; joining on the receive time
    # matches no scans.
    rows = list(csv.DictReader(open(a.labels)))
    if "stamp_ns" not in rows[0]:
        print("  the labels CSV has no stamp_ns column — regenerate it with the "
              "current label_frames.py, which records the LiDAR header stamp.")
        return 2
    gt = {int(r["stamp_ns"]): r for r in rows}
    stamps = [pcd_t_ns(p) for p in ds.pcd_files]
    joined = [(c, gt[t]) for c, t in zip(cid, stamps) if t in gt]
    print(f"joined {len(joined)}/{len(stamps)} scans to ground truth "
          f"({100 * len(joined) / max(len(stamps), 1):.0f}%)")
    if len(joined) < 0.5 * len(stamps):
        print("  REFUSING: fewer than half the scans matched a GT row. The PCD names "
              "and the CSV t_ns column are on different clocks; re-export with "
              "bag_to_pcd.py, which uses the message header stamp.")
        return 2

    report = {"config": config, "n_clusters": len([c for c in uniq if c >= 0]),
              "noise_frac": float((cid == -1).mean()), "axes": {}}
    for axis in AXES:
        ids = [c for c, _r in joined]
        truth = [r[axis] for _c, r in joined]
        assign = assign_cluster_labels(ids, truth, min_purity=a.min_purity,
                                       min_size=a.min_size)
        blocks = to_cluster_map_blocks(assign)
        labelled = {c: v for c, v in assign.items() if v.label != "unlabeled"}
        covered = sum(v.n for v in labelled.values())
        # frame-level accuracy of the derived map: a frame is correct when its cluster's
        # assigned label equals its own GT label. Unlabelled clusters count as wrong,
        # because downstream they ground nothing.
        correct = sum(1 for c, t in zip(ids, truth)
                      if c in labelled and labelled[c].label == t)
        nmi = metrics.normalized_mutual_info_score(truth, ids)
        ari = metrics.adjusted_rand_score(truth, ids)
        print(f"\n=== {axis}")
        print(f"  {len(labelled)}/{len([c for c in uniq if c >= 0])} clusters labelled "
              f"(purity >= {a.min_purity}), covering {100 * covered / len(ids):.0f}% "
              f"of frames")
        print(f"  frame accuracy {100 * correct / len(ids):5.1f}%   NMI {nmi:.3f}   "
              f"ARI {ari:.3f}")
        print(f"  modes: { {k: len(v) for k, v in blocks['modes'].items()} }")
        for c, v in sorted(labelled.items()):
            extra = f"  degraded={v.degraded}" if v.degraded else ""
            print(f"    cluster {c:3d}  n={v.n:5d}  {v.label:12s} "
                  f"purity {v.purity:.2f}{extra}")
        report["axes"][axis] = {
            "labelled_clusters": len(labelled),
            "frame_accuracy": correct / len(ids), "nmi": nmi, "ari": ari,
            "modes": blocks["modes"], "mode_meta": blocks["mode_meta"],
            "per_cluster": {str(c): {"label": v.label, "purity": v.purity, "n": v.n,
                                     "degraded": v.degraded,
                                     "distribution": v.distribution}
                            for c, v in sorted(assign.items())},
        }
        with open(os.path.join(out_dir,
                               f"cluster_id_to_label_{axis}_{a.name}.json"), "w") as f:
            json.dump({str(c): v.label for c, v in sorted(assign.items())}, f, indent=2)

    with open(os.path.join(out_dir, f"autolabel_report_{a.name}.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nartifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
