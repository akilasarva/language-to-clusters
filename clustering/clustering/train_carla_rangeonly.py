#!/usr/bin/env python3
"""Retrain the CARLA cluster encoder on RANGE features, and save the config with it.

WHY A SEPARATE DRIVER. `cluster_training.py`'s `main()` is configured for the real
hockfield bags (`use_intensity=True`, intensity clip [32, 68]); CARLA needs different
settings. This imports the model and dataset from that module so there is still ONE
implementation of the network and the features.

WHY RANGE AND NOT INTENSITY (on the CARLA PCDs):
  * CARLA emits intensity in roughly [0.8, 1.0]; the clip window is [32, 68], so no
    points fall inside it and every frame would become an all-zeros feature vector.
  * CARLA derives intensity from an exponential distance-attenuation model, not from
    material reflectance, so it is a monotone transform of range and carries no
    independent signal. Nothing is lost by dropping it.

WHY THE CONFIG IS WRITTEN OUT. Older `encoder_weights/*/` dirs hold only .pth/.pkl/scaler,
with no record of the z band, `num_ranges`, `max_lidar_range`, or range-vs-intensity.
The config is dumped as JSON BEFORE training, so it exists even if training is
interrupted.

`z_filter_mode: "abs"` records the band semantics of the original CARLA weights
(`np.abs(z)` in [lower, upper], a symmetric double band). Note: `cluster_training` now
filters on signed z, so check that the recorded mode matches the feature path in use
before comparing against those weights.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_training import (Autoencoder, LidarDataset,  # noqa: E402
                              LidarDecoder, LidarEncoder)

NAME = "bridge1_carla_rangeonly"
PCDS = os.path.expanduser("~/carla_data/bridge1_carla/bridge1_carla_pcds")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoder_weights", NAME)


def build_config() -> dict:
    cfg = {
        "training_data_name": NAME,
        "training_pcd_folder": PCDS,
        "embedding_size": 16,
        "num_ranges": 72,
        "angle_increment_deg": 360.0 / 72,
        "max_lidar_range": 25.0,
        "min_lidar_range": 1,
        # Band held at the values used for `bridge1_carla`, so the models are
        # comparable. With the 2.5 m sensor mount (ground plane at z = -2.5),
        # |z| in [0.1, 1.4] selects roughly 1.1-3.9 m above the road: STRUCTURE
        # height, with the ground excluded.
        "z_threshold_lower": 0.1,
        "z_threshold_upper": 1.4,
        "z_threshold_lower_2": 0,
        "z_threshold_upper_2": 0,
        #: How the band is applied. "abs" = np.abs(z) within [lower, upper], which is a
        #: SYMMETRIC double band. Inference must match this or it feeds the encoder a
        #: different slice than it learned.
        "z_filter_mode": "abs",
        "use_intensity": False,
        "num_epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "hdbscan_min_cluster_size": 8,
        "hdbscan_cluster_selection_epsilon": 0.0,
        "density_radius": 0.0,       # not applied in the training feature path
        "min_neighbors": 0,
        #: Properties of these PCDs, recorded for reference.
        "_measured_ground_plane_z": -2.5,
        "_measured_sensor_height_m": 2.5,
        "_measured_intensity_range": [0.8187, 0.9910],
        "_measured_corr_intensity_range": -0.9991,
    }
    cfg["model_save_path"] = f"{OUT}/lidar_encoder_autoencoder_{NAME}.pth"
    cfg["scaler_path"] = f"{OUT}/scaler_{NAME}.pkl"
    cfg["hdbscan_model_path"] = f"{OUT}/hdbscan_model_{NAME}.pkl"
    cfg["cluster_centroids_path"] = f"{OUT}/cluster_centroids_{NAME}.pkl"
    return cfg


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    cfg = build_config()
    # BEFORE training: the config must survive an interrupted run.
    with open(os.path.join(OUT, "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"config -> {OUT}/train_config.json")

    n_pcd = len([f for f in os.listdir(PCDS) if f.endswith(".pcd")])
    print(f"{n_pcd} PCDs in {PCDS}")
    if n_pcd == 0:
        sys.exit("no PCDs found")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    enc = LidarEncoder(embedding_size=cfg["embedding_size"])
    dec = LidarDecoder(embedding_size=cfg["embedding_size"], num_ranges=cfg["num_ranges"])
    ae = Autoencoder(enc, dec).to(dev)
    opt = optim.Adam(ae.parameters(), lr=cfg["learning_rate"])
    crit = nn.MSELoss()

    ds = LidarDataset(PCDS, cfg)
    print(f"dataset: {len(ds)} frames")
    # Guard: refuse to train on a degenerate (constant) feature.
    sample = torch.stack([ds[i] for i in range(0, min(len(ds), 40))])
    print(f"feature stats: min {sample.min():.4f} max {sample.max():.4f} "
          f"mean {sample.mean():.4f} std {sample.std():.4f}")
    if float(sample.std()) < 1e-6:
        sys.exit("FEATURE IS CONSTANT -- refusing to train (this is the intensity-clip bug)")

    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    for ep in range(cfg["num_epochs"]):
        ae.train(); tot = 0.0
        for b in dl:
            b = b.to(dev)
            opt.zero_grad()
            loss = crit(ae(b), b)
            loss.backward(); opt.step(); tot += float(loss) * len(b)
        if ep % 10 == 0 or ep == cfg["num_epochs"] - 1:
            print(f"  epoch {ep:3d}  loss {tot/len(ds):.6f}")
    torch.save(ae.state_dict(), cfg["model_save_path"])
    print(f"weights -> {cfg['model_save_path']}")


if __name__ == "__main__":
    main()
