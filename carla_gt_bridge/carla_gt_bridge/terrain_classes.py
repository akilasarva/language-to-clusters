"""One terrain vocabulary, for every source that can produce one.

WHY THIS FILE EXISTS, AND WHAT IT IS GUARDING AGAINST
-----------------------------------------------------
`Road=0 / Grass=1 / Sidewalk=2` is copy-pasted into several files under
`terrain_analysis/` with no shared definition, and each copy builds its label map as::

    seg_label_map = np.zeros((h, w), dtype=np.uint8)    # <-- every pixel is now ROAD
    seg_label_map[pred == 0] = 0
    seg_label_map[pred == 8] = 1
    seg_label_map[pred == 1] = 2

so building, sky, car, pole and fence all read **road** (`segformer_node.py`). Reducing
that to one whole-frame label over a bottom-centre crop hides it. Scoring an MPC rollout does
not: a rollout aimed at a wall would read as clean asphalt. **An unrecognised class must map to
`OTHER`, never to `ROAD`**, and that is the single invariant this module exists to hold.

THE TWO MAPPINGS DISAGREE ABOUT VEGETATION, AND THE DISAGREEMENT IS INVERTED
---------------------------------------------------------------------------
    Cityscapes trainId   8 = vegetation (tree canopy)     9 = terrain (grass, soil)
    CARLA CityObjectLabel 9 = Vegetation                 10 = Terrain

`segformer_node.py` maps Cityscapes **8** to grass, i.e. it calls tree canopy "grass" and
leaves actual lawn unrecognised -- which, with the zero-default above, makes the lawn read as
ROAD. Both halves of that are fixed here.

`RoadLines` (CARLA 24) maps to ROAD. It is a separate tag from `Roads`, and left unmapped every
lane marking in the image would read as OTHER, i.e. the middle of a carriageway would score as
costly terrain.

TAG NUMBERS ARE DERIVED, NOT TYPED IN
-------------------------------------
The CARLA table is built by NAME against `frame_labels.SEMANTIC_TAGS`, which was verified
against recorded semantic LiDAR data. Typing the integers here would be a second copy of that
table, and CARLA has renumbered these between versions. `_assert_names_exist` fails loudly if a name ever stops resolving.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from .frame_labels import SEMANTIC_TAGS

__all__ = [
    "ROAD", "SIDEWALK", "GRASS", "OTHER", "UNOBSERVED",
    "NAMES", "BY_NAME", "CARLA_TAG_TO_CLASS", "CITYSCAPES_TO_CLASS", "DEFAULT_COSTS",
    "from_carla_tags", "from_cityscapes", "classes_from_names", "check_forbid",
]

#: Not a surface at all: no mask, out of frame, behind the camera, occluded, or the frame was
#: too old. **ZERO ON PURPOSE, and it is the most load-bearing choice in this file.**
#:
#: Two reasons, both concrete. (1) `np.zeros(...)` is how label maps typically get built,
#: and `terrain_analysis/segformer_node.py` is the counterexample -- it zero-fills and then
#: assigns only three ids, so building, sky, car and pole all come out as its `Road = 0`.
#: With zero meaning "nothing seen", the identical bug yields UNOBSERVED, which
#: is dropped from the score and reported, rather than driven on. (2) The mask travels as a
#: `mono8` image, and a negative id does not survive that wire -- `-1` arrives as 255.
UNOBSERVED = 0

#: Surface classes. Small, flat, and deliberately NOT the nav-mode vocabulary
#: (`path` / `junction` / `along_edge`) -- that axis is what the space DOES, this one is what
#: it is MADE OF, and the design keeps them separate on purpose.
ROAD = 1
SIDEWALK = 2
GRASS = 3
OTHER = 4

NAMES: Mapping[int, str] = {
    UNOBSERVED: "unobserved", ROAD: "road", SIDEWALK: "sidewalk",
    GRASS: "grass", OTHER: "other",
}
BY_NAME: Mapping[str, int] = {v: k for k, v in NAMES.items()}

#: CARLA tag NAME -> class. Names, not numbers; see the module docstring.
_CARLA_NAME_TO_CLASS = {
    "Roads": ROAD,
    "RoadLines": ROAD,
    "Sidewalks": SIDEWALK,
    "Terrain": GRASS,
}


def _assert_names_exist() -> None:
    """A rename in `SEMANTIC_TAGS` must break loudly, not silently drop a class.

    Without this, dropping "RoadLines" would leave lane markings classed OTHER and the middle
    of every road would score as costly terrain -- a change of behaviour with no error.
    """
    known = set(SEMANTIC_TAGS.values())
    missing = sorted(set(_CARLA_NAME_TO_CLASS) - known)
    if missing:
        raise RuntimeError(
            f"terrain_classes expects CARLA tag name(s) {missing} which frame_labels."
            f"SEMANTIC_TAGS no longer defines; the CARLA tag table has changed")


_assert_names_exist()

#: CARLA semantic tag id -> class. Every id NOT in here is OTHER.
CARLA_TAG_TO_CLASS: Mapping[int, int] = {
    tag: _CARLA_NAME_TO_CLASS[name]
    for tag, name in SEMANTIC_TAGS.items() if name in _CARLA_NAME_TO_CLASS
}

#: Cityscapes trainId -> class. Every id NOT in here is OTHER, INCLUDING 8 (vegetation):
#: tree canopy is not ground and is never driven on, so it is structure, not surface.
CITYSCAPES_TO_CLASS: Mapping[int, int] = {
    0: ROAD,        # road
    1: SIDEWALK,    # sidewalk
    9: GRASS,       # terrain -- grass and soil. NOT 8, which is vegetation.
}

#: Starting costs. Placeholders, stated as such: they are not tuned, and `TerrainMpcConfig`
#: carries no inherited weights for the same reason. SIDEWALK is cheap-but-not-free so that a
#: mission which does not forbid it still prefers the carriageway.
#: UNOBSERVED is deliberately ABSENT: it is dropped from the weighted mean, not priced.
#: `TerrainMpcConfig.unknown_cost` prices the one case that cannot be dropped -- an arc with
#: no observed sample at all.
DEFAULT_COSTS: Mapping[int, float] = {
    ROAD: 0.0, SIDEWALK: 0.25, GRASS: 1.0, OTHER: 1.0,
}


def _remap(src: np.ndarray, table: Mapping[int, int]) -> np.ndarray:
    """Vectorised lookup with OTHER as the default.

    `np.full(..., OTHER)` then assign, NEVER `np.zeros` then assign -- and the difference is
    not cosmetic now that zero means UNOBSERVED. A pixel the model DID classify, as a wall or
    a car, is evidence: it is `OTHER`, it costs, and it counts toward coverage. Defaulting it
    to UNOBSERVED would drop it from the score instead, so an arc aimed at a building would
    read as "nothing known here" rather than "not drivable". Unrecognised-but-seen and
    not-seen are different facts and only one of them is missing data.
    """
    out = np.full(src.shape, OTHER, dtype=np.int16)
    for raw, klass in table.items():
        out[src == raw] = klass
    return out


def from_carla_tags(tags: np.ndarray) -> np.ndarray:
    """CARLA semantic-camera tag ids -> class ids. ``(H, W)`` -> ``(H, W)`` int16."""
    return _remap(np.asarray(tags), CARLA_TAG_TO_CLASS)


def from_cityscapes(pred: np.ndarray) -> np.ndarray:
    """SegFormer/Cityscapes trainIds -> class ids. ``(H, W)`` -> ``(H, W)`` int16."""
    return _remap(np.asarray(pred), CITYSCAPES_TO_CLASS)


def classes_from_names(names: Iterable[str]) -> frozenset[int]:
    """``["grass", "sidewalk"]`` -> ``frozenset({GRASS, SIDEWALK})``.

    Raises on an unknown name rather than dropping it. A forbid set that silently loses a term
    is the failure this whole file is about: the run completes, reports a prohibition, and
    prohibits nothing.
    """
    out = set()
    for n in names:
        key = str(n).strip().lower()
        if not key:
            continue
        if key not in BY_NAME:
            raise ValueError(
                f"unknown terrain class {n!r}; known: {sorted(BY_NAME)}")
        out.add(BY_NAME[key])
    check_forbid(out)
    return frozenset(out)


def check_forbid(forbid: Iterable[int]) -> None:
    """Reject `UNOBSERVED` in a hard-veto set.

    A veto that can fire on every candidate hands control to the relax-everything fallback,
    which knows nothing about where the hazard is; an over-eager obstacle veto
    (`terrain_mpc.TerrainMpcConfig.obstacle_hard_m`) inside a junction can remove all arcs and
    leave the vehicle circling. `UNOBSERVED` is worse, because a dead
    camera or a stale frame makes EVERY point unobserved at once -- so the failure would be
    triggered by the sensor dropping out, which is precisely when the vehicle must keep driving.
    """
    if UNOBSERVED in set(int(c) for c in forbid):
        raise ValueError(
            "UNOBSERVED must never be forbidden: a dead or stale camera makes every point "
            "unobserved, which would veto all K candidates and engage the relax-everything "
            "fallback. Give it a COST instead (TerrainMpcConfig.unknown_cost).")
