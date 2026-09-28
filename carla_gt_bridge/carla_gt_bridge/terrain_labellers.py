"""Interchangeable ways to turn one camera frame into terrain class ids.

WHY A REGISTRY AND NOT A CHOICE
-------------------------------
There are several implementations of this in `terrain_analysis/`, written against real ZED
footage, and which of them is best on CARLA is an open question. So the labeller is a STRING,
every implementation returns the same `(H, W)` int16 array in the same vocabulary, and
`agreement()` below scores any of them against CARLA's ground-truth semantic camera on the
same frames.

    carla_semantic   CARLA's own tags. Ground truth, no model, no latency. The oracle,
                     in the same sense as `gt_cluster_node` and `gt_cue_node` -- it
                     isolates planning from perception, and it is NOT a perception claim.
    segformer        nvidia/segformer-b0-finetuned-cityscapes-1024-1024. The real thing.
    hsv              `cv2.inRange` colour thresholds. No model, no GPU, no lag.

NOTES
-----
1. **The `terrain_analysis/` implementations share a zero-default bug**, and it is worse in
   the HSV one. Both build `seg_label_map = np.zeros(...)` and then paint three classes, so
   anything matching none of them keeps id 0, which in their vocabulary is ROAD
   (`segformer_node.py`, `terrain_segmenter.py`). SegFormer at least assigns
   every pixel SOME Cityscapes id; the HSV ranges do not cover HSV space at all, so a
   saturated red car, a bright sky and a dark doorway all come out road. Here the default is
   UNOBSERVED (id 0) for "no rule matched" and OTHER for "matched, not drivable" -- a pixel
   nothing recognised is missing data, not asphalt.

2. **The HSV thresholds are not calibrated for CARLA.** They were tuned on the hamilton ZED
   bags, and CARLA's asphalt, concrete and grass are rendered, not photographed. They are
   carried over verbatim ON PURPOSE -- porting them with new numbers would make this a test
   of the retuning rather than of the method -- but they must be scored with `agreement()`
   before any result rests on them. Performance on other tasks (e.g. forward-view shape
   classification) says nothing about material labelling here.

3. **The LiDAR intensity classifier is not one of these.** `terrain_analysis/
   terrain_classifier.py` thresholds per-POINT intensity (<=30 road, <=60 sidewalk, else
   grass) and emits a coloured cloud, not an image. That is a different SOURCE SHAPE, not a
   different labeller: it yields terrain over space directly and needs no projection at all,
   which makes it attractive here. It does not belong behind this interface -- it belongs
   behind a sibling of `MaskTerrainSource` that does a nearest-neighbour lookup in the body
   plane. Two reasons it is not built: CARLA's LiDAR intensity is an
   attenuation model (range and incidence), not a material property, so those thresholds
   cannot transfer from the Livox bags they were tuned on; and `sensor.lidar.ray_cast`
   carries intensity while the SEMANTIC lidar carries the tag, so on the GT side the tag is
   strictly better information than a threshold over intensity.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from .terrain_classes import (GRASS, OTHER, ROAD, SIDEWALK, UNOBSERVED,
                              from_carla_tags, from_cityscapes)

__all__ = ["LABELLERS", "make_labeller", "carla_semantic_labeller", "hsv_labeller",
           "segformer_labeller", "agreement"]


# --------------------------------------------------------------------------- #
# carla_semantic -- the oracle                                                 #
# --------------------------------------------------------------------------- #

def carla_semantic_labeller() -> Callable[[np.ndarray], np.ndarray]:
    """Raw CARLA semantic tags -> class ids.

    Expects the RAW TAG PLANE, not the CityScapes palette. The ros-bridge's
    `SemanticSegmentationCamera.get_carla_image_data_array` calls
    `carla_image.convert(carla.ColorConverter.CityScapesPalette)` before publishing, which
    DESTROYS the tag: the wire carries colours. So whatever feeds this must read
    `image.raw_data[:, :, 2]` from its own CARLA client rather than subscribing to the
    bridge's topic -- the pattern `gt_obstacle_node.py` already uses.
    """
    def _label(tags: np.ndarray) -> np.ndarray:
        a = np.asarray(tags)
        if a.ndim == 3:                       # a BGRA frame: the tag lives in RED
            a = a[:, :, 2]
        return from_carla_tags(a)
    return _label


# --------------------------------------------------------------------------- #
# hsv -- no model, no GPU                                                      #
# --------------------------------------------------------------------------- #

#: Verbatim from `terrain_analysis/terrain_analysis/terrain_segmenter.py`. Tuned on the
#: hamilton ZED bags; see note 2 in the module docstring before trusting them on CARLA.
HSV_RANGES: Mapping[int, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    GRASS: ((35, 40, 40), (85, 255, 255)),
    ROAD: ((0, 0, 0), (180, 50, 70)),
    SIDEWALK: ((0, 0, 71), (180, 40, 210)),
}


def hsv_labeller(ranges: Mapping[int, tuple] | None = None
                 ) -> Callable[[np.ndarray], np.ndarray]:
    """BGR frame -> class ids by colour threshold.

    Assignment order is road, then sidewalk, then grass, matching the original so a
    comparison is against that method and not against a reordering of it. The ONE change is
    the default: a pixel in none of the three ranges is UNOBSERVED here and was ROAD there.
    """
    import cv2
    rng = dict(HSV_RANGES if ranges is None else ranges)

    def _label(bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(np.asarray(bgr), cv2.COLOR_BGR2HSV)
        # UNOBSERVED, not ROAD. See note 1 -- these three ranges do not cover HSV space, so
        # the default is what most of a real frame gets.
        out = np.full(hsv.shape[:2], UNOBSERVED, dtype=np.int16)
        for klass in (ROAD, SIDEWALK, GRASS):
            if klass not in rng:
                continue
            lo, hi = rng[klass]
            out[cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0] = klass
        return out
    return _label


# --------------------------------------------------------------------------- #
# segformer -- the real thing                                                  #
# --------------------------------------------------------------------------- #

#: Same checkpoint as `terrain_analysis/segformer_node.py`.
SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"


def segformer_labeller(model_name: str = SEGFORMER_MODEL, device: str | None = None
                       ) -> Callable[[np.ndarray], np.ndarray]:
    """RGB frame -> class ids via SegFormer-b0 fine-tuned on Cityscapes.

    Re-implemented rather than imported from an analysis script: such scripts tend to
    `os.chdir` and load datasets at module scope, and a ROS node importing one would break
    every relative path downstream (`log_dir`, `regions_npz`, the plan files).

    `torch` and `transformers` are imported HERE rather than at module scope so the GT and
    HSV labellers, and every unit test, run without them.
    """
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    proc = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(dev).eval()

    def _label(rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb)
        inp = proc(images=rgb, return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = model(**inp).logits
        up = torch.nn.functional.interpolate(
            logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
        return from_cityscapes(up.argmax(1).squeeze().cpu().numpy())
    return _label


LABELLERS: Mapping[str, Callable[..., Callable[[np.ndarray], np.ndarray]]] = {
    "carla_semantic": carla_semantic_labeller,
    "hsv": hsv_labeller,
    "segformer": segformer_labeller,
}


def make_labeller(name: str, **kw) -> Callable[[np.ndarray], np.ndarray]:
    """``name`` -> a ready labeller. Raises on an unknown name rather than defaulting.

    A silent fallback here would be the system-level version of the zero-default bug: the
    node would report whichever labeller it fell back to under the name of the one asked
    for, and both sides of a comparison could end up being the same model.
    """
    key = str(name).strip().lower()
    if key not in LABELLERS:
        raise ValueError(f"unknown labeller {name!r}; known: {sorted(LABELLERS)}")
    return LABELLERS[key](**kw)


# --------------------------------------------------------------------------- #
# Scoring one labeller against another                                         #
# --------------------------------------------------------------------------- #

def agreement(pred: np.ndarray, truth: np.ndarray, *,
              ignore_unobserved_truth: bool = True) -> dict:
    """Per-class recall, precision and IoU of `pred` against `truth`, plus the one number
    that actually decides whether a labeller is usable here.

    THE KEY NUMBER IS NOT ACCURACY, IT IS `road_called_grass` AND `grass_called_road`. A
    labeller that is 95% correct overall but calls 20% of the lawn asphalt cannot support a
    prohibition, while one that is 80% correct and never confuses those two can. Mean
    accuracy over classes would hide that behind sky and buildings, which occupy most of the
    frame and which nothing drives on.

    `unobserved` in PRED is counted -- declining to label is a real cost, it is what the
    graded term drops and what `blind_arcs` counts. `unobserved` in TRUTH is excluded by
    default, since the oracle has no such class and it would only appear from a mask edge.
    """
    p = np.asarray(pred).ravel()
    t = np.asarray(truth).ravel()
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs truth {truth.shape}")
    if ignore_unobserved_truth:
        keep = t != UNOBSERVED
        p, t = p[keep], t[keep]
    out = {"pixels": int(t.size), "overall_accuracy": float((p == t).mean()) if t.size else 0.0,
           "pred_unobserved_frac": float((p == UNOBSERVED).mean()) if t.size else 0.0}
    for klass, name in ((ROAD, "road"), (SIDEWALK, "sidewalk"), (GRASS, "grass"),
                        (OTHER, "other")):
        tp = int(((p == klass) & (t == klass)).sum())
        fp = int(((p == klass) & (t != klass)).sum())
        fn = int(((p != klass) & (t == klass)).sum())
        out[f"{name}_recall"] = tp / (tp + fn) if (tp + fn) else float("nan")
        out[f"{name}_precision"] = tp / (tp + fp) if (tp + fp) else float("nan")
        out[f"{name}_iou"] = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    # The two confusions a terrain prohibition actually rides on.
    road_t = (t == ROAD).sum()
    grass_t = (t == GRASS).sum()
    out["road_called_grass"] = float(((t == ROAD) & (p == GRASS)).sum() / road_t) \
        if road_t else float("nan")
    out["grass_called_road"] = float(((t == GRASS) & (p == ROAD)).sum() / grass_t) \
        if grass_t else float("nan")
    return out
