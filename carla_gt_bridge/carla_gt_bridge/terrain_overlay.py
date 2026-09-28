"""Render the rollout fan onto a camera frame. ONE implementation, two callers.

An offline script can draw this from captured frames; the MPC node draws it live from the
mask it is actually steering on. Two implementations of one rule drift apart, so both callers
use this one.

THE ARC GEOMETRY IS PASSED IN, NOT COMPUTED HERE. `arc_rollout_k` lives in
`dgppo_ros_node_pkg`, which depends on this package and not the other way round; importing it
here would invert that, and re-deriving the closed form would be a second implementation of
the one thing in this system that must not have two. Callers hand over the dense points they
already have.
"""

from __future__ import annotations

import os

import numpy as np

from .camera_terrain import CameraModel
from .terrain_classes import DEFAULT_COSTS, GRASS, OTHER, ROAD, SIDEWALK, UNOBSERVED

__all__ = ["COST_MAX", "CLASS_BGR", "cost_bgr", "mask_to_bgr", "render_fan"]

#: BGR, for cv2. Chosen so the three surfaces stay distinguishable in greyscale too.
CLASS_BGR = {
    UNOBSERVED: (60, 60, 60),        # near-black: "nothing here", not a surface
    ROAD: (128, 64, 128),            # the Cityscapes road purple, so the panels read as one
    SIDEWALK: (232, 35, 244),
    GRASS: (60, 200, 60),
    OTHER: (70, 70, 70),
}


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    """Class ids -> a colour image, for the mask panel."""
    img = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for k, bgr in CLASS_BGR.items():
        img[mask == k] = bgr
    return img


COST_MAX = 1.0                   # the cost a fully-forbidden surface carries


def cost_bgr(c: float) -> tuple:
    """Terrain cost -> BGR, on a blue(cheap) -> red(costly) ramp.

    WHY NOT SHADES OF RED ALONE. A single-hue ramp varies only lightness, which the eye reads
    badly against a photograph whose own lightness varies wildly -- a dark red line on wet
    asphalt and a light red line on a sunlit kerb look like different costs when they are the
    same. A blue-to-red ramp varies HUE, which survives the background, and blue/orange is the
    one pair that stays distinguishable for the commonest colour-vision deficiencies.
    """
    import matplotlib
    r, g, b, _ = matplotlib.colormaps["coolwarm"](float(np.clip(c / COST_MAX, 0.0, 1.0)))
    return (int(b * 255), int(g * 255), int(r * 255))


def render_fan(mask, cam: CameraModel, dense, kappa, w_arc, best: int,
               touches, soft, out_png: str, rgb=None, title: str = "",
               max_arcs: int | None = None, forbid_classes=None) -> None:
    """Paint the fan onto the frame, encoding the scorer's own two quantities.

    TWO CHANNELS, BECAUSE THE SCORE HAS TWO INPUTS and a single colour hides one of them:

        COLOUR     the terrain cost AT THAT POINT -- what the ground there is.
        THICKNESS  the distance weight the scorer gives that arc length. Near stretches are
                   drawn thick and far ones thin, so a hazard that will actually be driven
                   over looks heavy and the same hazard at the end of the arc looks faint.
                   That is literally what `arc_weights` does to the number, made visible.

    Two things are deliberately NOT on the ramp, because they are different in kind rather
    than in degree:

        FORBIDDEN  a veto is not "very expensive". Vetoed arcs get a magenta dashed casing
                   and a cross at the FIRST forbidden point.
        UNOBSERVED not cheap, not costly: no evidence. Grey dashes, so a stretch the camera
                   could not describe never reads as clean road, and the CHOSEN-arc highlight
                   is broken there too -- a solid white line through ground nothing described
                   would contradict the convention the rest of the picture keeps.

    Everything is drawn over a black casing, so it stays legible on sky, asphalt and sunlit
    concrete alike, and the chosen arc carries a white casing plus a ring at its endpoint.
    """
    import cv2
    n_score, n_draw = w_arc.size, dense.shape[1]
    u, v, vis = cam.project_ground(dense)
    s_dense = np.linspace(0.0, 1.0, n_draw + 1)[1:]
    # Which scorer sample each drawn point belongs to, for the thickness channel.
    w_at = w_arc[np.clip((s_dense * n_score).astype(int), 0, n_score - 1)]
    wn = w_at / max(float(w_at.max()), 1e-9)

    rows = np.clip(np.rint(v).astype(int), 0, mask.shape[0] - 1)
    cols = np.clip(np.rint(u).astype(int), 0, mask.shape[1] - 1)
    klass_d = np.where(vis, mask[rows, cols], UNOBSERVED)
    cost_d = np.full(klass_d.shape, float(DEFAULT_COSTS.get(OTHER, 1.0)))
    for c, w in DEFAULT_COSTS.items():
        cost_d[klass_d == c] = float(w)

    # WHICH SAMPLES ARE ACTUALLY ILLEGAL. Inferring legality from `cost >= COST_MAX` is
    # wrong twice over: SIDEWALK costs 0.25, so with a sidewalk ban no sample would reach
    # the threshold; and grass and structure both cost 1.0, so a banned surface could not
    # be told from a merely expensive one. Cost and legality are different questions and
    # the picture has to be told both.
    bad_d = (np.isin(klass_d, np.fromiter(forbid_classes, dtype=int))
             if forbid_classes else np.zeros(klass_d.shape, dtype=bool))
    K = kappa.size
    # HOW MANY ARCS TO DRAW. None = all, which is what offline rendering wants. A LIVE run
    # cannot afford it: a full-fan render can take many seconds when CARLA and the bridge
    # share the machine, so most renders would be skipped. Drawing every n-th arc is the
    # cheap axis: the fan's shape reads the same at a third of the strokes, the chosen arc
    # is always included, and the chart beside it still summarises all K.
    draw_idx = list(range(K))
    if max_arcs and max_arcs < K:
        step = max(1, K // int(max_arcs))
        draw_idx = sorted(set(list(range(0, K, step)) + [int(best)]))
    panels = []
    # ONE PANEL WHEN THERE IS NO CAMERA IMAGE. Rendering the mask twice would invite the
    # reader to believe the "camera" panel shows the scene. The ground-truth labeller has
    # no RGB frame (it reads tags straight off a semantic camera), so this is the normal
    # case for a driven run, not an edge case.
    bases = ([mask_to_bgr(mask)] if rgb is None
             else [np.ascontiguousarray(rgb).copy(), mask_to_bgr(mask)])
    for base in bases:
        for i in sorted(draw_idx, key=lambda j: j == best):
            chosen = (i == best)
            forbidden = bool(touches[i])
            first_bad = None
            for j in range(n_draw - 1):
                if not (vis[i, j] and vis[i, j + 1]):
                    continue
                p0 = (int(round(u[i, j])), int(round(v[i, j])))
                p1 = (int(round(u[i, j + 1])), int(round(v[i, j + 1])))
                th = 1 + int(round(1.0 * wn[j])) + (2 if chosen else 0)
                if klass_d[i, j] == UNOBSERVED:
                    if j % 8 < 4:                       # dashed: no evidence here
                        cv2.line(base, p0, p1, (165, 165, 165), 1, cv2.LINE_AA)
                    continue
                # A CASING ONLY WHERE IT EARNS ITS INK. With dense sampling, a black casing
                # on every segment of every arc turns the lower half of the frame into a
                # solid mat that hides the photograph. The chosen arc keeps a white
                # casing because it has to be findable at a glance; the rest rely on the
                # ramp, which is saturated enough to hold against asphalt and concrete.
                if chosen:
                    cv2.line(base, p0, p1, (255, 255, 255), th + 4, cv2.LINE_AA)
                    cv2.line(base, p0, p1, (0, 0, 0), th + 1, cv2.LINE_AA)
                cv2.line(base, p0, p1, cost_bgr(cost_d[i, j]), th, cv2.LINE_AA)
                if first_bad is None and forbidden and cost_d[i, j] >= COST_MAX:
                    first_bad = p0
            if forbidden:
                # ONLY THE ILLEGAL SPAN. Dashing from j = 0 would paint an arc banned for
                # something 10 m away magenta right back to the bumper, over clean road,
                # which looks like a labelling fault in the mask. The arc's legal near
                # portion keeps its true cost colour, and magenta means "from here on, this
                # arc is illegal".
                bad_js = np.where(bad_d[i] & vis[i])[0]
                start = int(bad_js[0]) if bad_js.size else None
                if start is not None:
                    for j in range(start, n_draw - 1, 16):
                        if vis[i, j] and vis[i, j + 1] and klass_d[i, j] != UNOBSERVED:
                            cv2.line(base, (int(u[i, j]), int(v[i, j])),
                                     (int(u[i, j + 1]), int(v[i, j + 1])), (200, 0, 200), 2,
                                     cv2.LINE_AA)
                    first_bad = (int(round(u[i, start])), int(round(v[i, start])))
                if first_bad is not None:
                    cv2.drawMarker(base, first_bad, (200, 0, 200), cv2.MARKER_TILTED_CROSS,
                                   11, 2, cv2.LINE_AA)
            if chosen:
                seen = np.where(vis[i] & (klass_d[i] != UNOBSERVED))[0]
                if seen.size:
                    tip = (int(u[i, seen[-1]]), int(v[i, seen[-1]]))
                    cv2.circle(base, tip, 6, (0, 0, 0), -1)
                    cv2.circle(base, tip, 5, (255, 255, 255), -1)
                    cv2.putText(base, f"CHOSEN k={kappa[i]:+.3f}",
                                (min(tip[0] + 9, base.shape[1] - 190), max(tip[1] - 8, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(base, f"CHOSEN k={kappa[i]:+.3f}",
                                (min(tip[0] + 9, base.shape[1] - 190), max(tip[1] - 8, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1,
                                cv2.LINE_AA)
        hr = int(round(cam.horizon_row()))
        if 0 <= hr < base.shape[0]:
            cv2.line(base, (0, hr), (base.shape[1], hr), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(base, "predicted horizon", (8, max(hr - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        panels.append(base)

    h = panels[0].shape[0]
    chart = np.full((h, 460, 3), 24, dtype=np.uint8)
    cost = soft
    for i, _ in enumerate(kappa):
        x = int(30 + (i / max(K - 1, 1)) * 400)
        bar = int((h - 150) * min(cost[i] / COST_MAX, 1.0))
        cv2.line(chart, (x, h - 110), (x, h - 110 - bar), cost_bgr(cost[i]), 3)
        if touches[i]:
            cv2.line(chart, (x, h - 104), (x, h - 96), (200, 0, 200), 3)
    bx = int(30 + (best / max(K - 1, 1)) * 400)
    cv2.line(chart, (bx, 30), (bx, h - 110), (255, 255, 255), 1)
    cv2.putText(chart, "CHOSEN", (max(bx - 26, 4), 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(chart, "arc terrain cost vs curvature", (16, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    # Index 0 of the fan is kappa = -kappa_max. Positive kappa increases heading and body +y
    # is LEFT, so NEGATIVE curvature turns RIGHT: for the default fan, endpoint y = -5.53 m
    # at index 0 against +5.53 at index 64.
    cv2.putText(chart, "right turn", (16, h - 84), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(chart, "straight", (205, h - 84), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(chart, "left turn", (375, h - 84), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (170, 170, 170), 1, cv2.LINE_AA)
    for x in range(400):
        cv2.line(chart, (30 + x, h - 70), (30 + x, h - 52), cost_bgr(x / 400.0 * COST_MAX), 1)
    cv2.putText(chart, "road 0", (30, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(chart, "sidewalk .25", (118, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(chart, "grass / structure 1.0", (268, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(chart, "magenta = FORBIDDEN by the mission   grey dashes = unobserved",
                (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(chart, "thickness = distance weight", (16, h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    out = np.hstack(panels + [chart])
    cv2.putText(out, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                cv2.LINE_AA)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    cv2.imwrite(out_png, out)
