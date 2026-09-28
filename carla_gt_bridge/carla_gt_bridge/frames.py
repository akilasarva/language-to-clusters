"""The ONE place CARLA's world frame is converted to the planner frame.

Why this file exists
--------------------
Geometry in this stack is stored in both conventions: some cluster maps store raw CARLA
``(x, y)`` centroids and bearings in **degrees**, while ``plans/bridge.json`` stores
``centroid[1] = -carla_y`` and bearings in **radians**. Crossing the two gives a
mirrored target and a 57x (degrees-as-radians) heading error, and neither failure
announces itself: the vehicle simply drives somewhere wrong.

So: every conversion goes through here, and it is round-trip tested.

The two frames
--------------
**CARLA world** (what ``carla.Location`` / ``carla.Rotation`` return from the Python API)
is left-handed: +x east, +y *south*, yaw in **degrees** increasing from +x toward +y —
i.e. clockwise seen from above.

**Planar** is the ordinary right-handed frame the unicycle model and every ROS convention
assume: +x east, +y *north*, yaw in **radians** counter-clockwise.

The conversion is therefore a pure mirror of y plus a sign flip on yaw::

    px = cx
    py = -cy
    pyaw = -radians(cyaw_deg)

Which of our data is in which frame — VERIFIED, not assumed
-----------------------------------------------------------
This is the part that is easy to get backwards, so it is checked against numbers
recorded from a running server rather than reasoned about:

===============================  =========  =========================================
source                           frame      evidence
===============================  =========  =========================================
``.xodr`` geometry, and so every  **planar** Town01's ``.xodr`` spans y in
centroid / bearing / waypoint                [-328.6, 0.0]; the live server reports
table from ``map_regions.py``                y ~ +273..+330 for the same town, so
                                             CARLA y = -(OpenDRIVE y).
``carla_ros_bridge`` odometry     **planar** The bridge converts to the ROS right-handed
and ``/carla/<role>/twist``                  convention on the way out and back.
``carla.Location`` via the        CARLA      left-handed, as documented above.
Python API (spawn points, the
old node's ``get_transform()``)
===============================  =========  =========================================

**So ROS-side data needs no conversion at all** — the OpenDRIVE frame and the ROS frame
are the same right-handed frame, and applying :func:`carla_xy_to_planar` to bridge
odometry would double-negate y and mirror the whole map. These functions are for data
that came through the CARLA Python API, which in this stack means the sim adapter and
nothing else.

That is also why ``segmenter.bearing_map`` values are counter-clockwise headings with
+y north: a **positive** turn there is a LEFT turn. In CARLA's own frame it would be a
right turn. This table is the single answer to the sign-of-y question.

**No origin offset and no scale.** The old node carried ``ORIG_X = 55.0`` /
``ORIG_Y = -210.0`` and a ``grid_scale``; both were DGPPO artifacts that mapped
metres into a normalised ``[0, 1.5]`` model space. The sampling MPC works in metres,
so they are gone — an offset that exists for no reason is one more thing to get wrong,
and it made every logged coordinate impossible to compare against a CARLA screenshot.

Because the mirror is an isometry, distances and *magnitudes* of angles are identical
in both frames; only the sense of rotation differs. That is what makes it safe to
convert a single angle rather than every point, when a conversion is needed at all.
"""

from __future__ import annotations

import math

__all__ = ["bearing_deg_to_planar_rad", "carla_to_planar", "planar_to_carla",
           "carla_xy_to_planar", "planar_xy_to_carla", "carla_yaw_to_planar",
           "planar_yaw_to_carla", "wrap_pi", "wrap_180"]


def wrap_pi(a: float) -> float:
    """Wrap radians to the half-open interval ``[-pi, pi)``."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def wrap_180(a: float) -> float:
    """Wrap degrees to the half-open interval ``[-180, 180)``.

    Half-open at the NEGATIVE end, matching ``map_regions._wrap180``: due west comes
    back as -180, never +180. Which end is closed does not matter physically, but two
    functions disagreeing about it does, so both ends of the repo use this one.
    """
    return (a + 180.0) % 360.0 - 180.0


def carla_xy_to_planar(cx: float, cy: float) -> tuple[float, float]:
    """CARLA world metres -> planar metres."""
    return (cx, -cy)


def planar_xy_to_carla(px: float, py: float) -> tuple[float, float]:
    """Planar metres -> CARLA world metres."""
    return (px, -py)


def carla_yaw_to_planar(cyaw_deg: float) -> float:
    """CARLA yaw (degrees, clockwise) -> planar yaw (radians, counter-clockwise)."""
    return wrap_pi(-math.radians(cyaw_deg))


def planar_yaw_to_carla(pyaw_rad: float) -> float:
    """Planar yaw (radians, CCW) -> CARLA yaw (degrees, CW)."""
    return wrap_180(-math.degrees(pyaw_rad))


def carla_to_planar(cx: float, cy: float,
                    cyaw_deg: float) -> tuple[float, float, float]:
    """Full pose: CARLA (x, y, yaw_deg) -> planar (x, y, yaw_rad)."""
    px, py = carla_xy_to_planar(cx, cy)
    return (px, py, carla_yaw_to_planar(cyaw_deg))


def planar_to_carla(px: float, py: float,
                    pyaw_rad: float) -> tuple[float, float, float]:
    """Full pose: planar (x, y, yaw_rad) -> CARLA (x, y, yaw_deg)."""
    cx, cy = planar_xy_to_carla(px, py)
    return (cx, cy, planar_yaw_to_carla(pyaw_rad))


def bearing_deg_to_planar_rad(bearing_deg: float) -> float:
    """A `bearing_map` entry (degrees, computed in the CARLA frame) -> planar radians.

    ``segmenter.RegionMap.bearing_map`` computes ``atan2(dy, dx)`` on **raw CARLA**
    centroids, so the resulting angle is a CARLA-frame heading and needs the same
    sign flip as a yaw. Kept as a named function rather than an inline `-radians(...)`
    because the inline version is exactly the step that gets lost when geometry is
    hand-converted between the two conventions.
    """
    return carla_yaw_to_planar(bearing_deg)
