# bev_pipeline

Behavioral-state ("cluster") classification for outdoor unstructured environments,
replacing the legacy planar-slice + autoencoder + HDBSCAN + manual-labeling
pipeline. Non-destructive: preserves the `/predicted_cluster` (Int16) contract
`nl_planner`/`brain_controller` consume, and adds a richer `/predicted_state`
(JSON) topic.

## Pipeline

```
bag → gravity-align → plane-fit ground removal → 3-5 frame submap accumulation
    → [BEV raster | 3D voxels | geometric features | raw points]
    → feature extractor → supervised classifier → /predicted_cluster + /predicted_state
```

Ground removal uses `plane_level_and_remove` (RANSAC dominant-plane → level →
split) rather than Patchwork++ alone: with a residual sensor-mount tilt,
Patchwork++ removes very little of the ground.

## Components

**Feature extractors** (`bev_pipeline/feature_extractors/`, pick via config):
- `geom_features` — hand-engineered geometric descriptors (free-space profile,
  clearances, overhead, height histogram). No training, interpretable,
  data-efficient.
- `bev_dinov2` / `bev_dino_mh` — frozen DINO(v2/v3) on a BEV projection.
  `bev_dino_mh` uses the `height_bands` encoding (empty=black, 3D-in-color)
  instead of the original `vico3d` projection.
- `camera_dinov2` — frozen DINO on the camera image.
- `point_encoder` — PointNet-style over the raw cloud (LiDAR-native, trainable).
- `bev_cnn` / `vol3d_cnn` — from-scratch 2D/3D CNNs (need more data).

**Label sources**:
- `geom` — deterministic geometry-grounded states: `open_space / path /
  along_edge / passage / junction` (see `geometry.classify_geometry`).
- `vlm` — gpt-4o auto-labels camera frames (`vlm_auto_labeler.py`, a
  human-labeling stand-in).
- `human` — hand labels from `tools/label_frames.py` (vocabulary in
  `config/nav_modes.yaml`).
- Geometry-derived trajectory labels via `trajectory_labeler` + hand-drawn
  landmark bboxes (`tools/landmark_bbox_editor.py`).

## Running

Dataset caching and training scripts are not included; the tools below operate
on an already-cached `datasets/<env>` directory.

```bash
# hand-label frames (needs a display)
python3 tools/label_frames.py --env-dir datasets/<name>

# draw landmark bboxes for the trajectory labeler (needs a display)
python3 tools/landmark_bbox_editor.py --env-dir datasets/<name>

# VLM audit of a label set (needs OPENAI_API_KEY)
python3 tools/qa_review_tool.py --env-dir datasets/<name> --label-set geom

# live node
ros2 launch bev_pipeline bev_inference.launch.py config:=config/bev_pipeline.yaml
```

DINOv3: weights are license-gated — accept Meta's license, download, and set
`DINO_WEIGHTS=/path/to/weights.pth`.

`pypatchworkpp` is pip-only and not installed by colcon; install it manually
(ground removal falls back to Open3D RANSAC without it).

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/ \
  --ignore=test/test_copyright.py --ignore=test/test_flake8.py \
  --ignore=test/test_pep257.py --ignore=test/test_xmllint.py
```
(The 4 ignored ament style-lint tests are colcon-managed.)
