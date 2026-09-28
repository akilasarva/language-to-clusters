#!/usr/bin/env python3
"""Plan a data-collection drive that covers a town's regions and all three maneuvers.

    python3 scripts/plan_collection_routes.py --town Town01 --town Town07 \
        --out-dir reports/collection

Produces, per town:
    route.<town>.png    overhead: the commanded path over the ground-truth regions
    route.<town>.json   ordered region ids + target waypoints, PLANAR and CARLA frames

WHY A DERIVED TOUR AND NOT HAND-PICKED CORRIDORS
A hand-picked corridor is chosen to be drivable, not to be representative: a
region-sequence score over it cannot see a wrong turn, and a label corpus drawn from it
may contain only straight-through traversals, so the `exit` phase is never observed and
the confusion matrix blames perception. So the tour here is computed over the town's own
adjacency graph, and the script REPORTS its coverage: what fraction of regions it enters
and how many left / right / straight junction traversals it makes.

THE WALK. Greedy nearest-unvisited over the region graph: from the current region, BFS to
the closest region not yet visited, append that path, repeat. That is a covering walk
rather than a shortest tour — the optimal version is the Chinese-postman problem and the
difference is a few minutes of sim time.

Maneuver classification reuses the same rule as `map_regions.maneuvers`: the signed
turn between the bearing INTO a junction and the bearing OUT of it, `straight` under 45
degrees. Positive is LEFT, because the planar frame is +y north / counter-clockwise —
in CARLA's own left-handed frame the sign is inverted, which is why the JSON carries both.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from carla_gt_bridge.opendrive import load  # noqa: E402
from carla_gt_bridge.segmenter import segment  # noqa: E402

LABEL_COL = {"path": "#93C5FD", "approach": "#6EE7B7", "junction": "#FDBA74",
             "passage": "#C4B5FD", "along_edge": "#5EEAD4", "open_space": "#FCD34D"}


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def build_graph(rm):
    adj: dict[int, set[int]] = {r: set() for r in rm.regions}
    for a, b in rm.adjacency:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def bfs_path(adj, src, targets):
    """Shortest path from ``src`` to the nearest member of ``targets``."""
    if src in targets:
        return [src]
    prev = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in sorted(adj[u]):
            if v in prev:
                continue
            prev[v] = u
            if v in targets:
                path = [v]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            q.append(v)
    return []


def covering_walk(rm, start: int | None = None) -> list[int]:
    """Greedy nearest-unvisited walk over the region graph."""
    adj = build_graph(rm)
    reach = {r for r in rm.regions if adj[r]}
    if not reach:
        return []
    cur = start if start in reach else min(reach)
    walk = [cur]
    unvisited = reach - {cur}
    while unvisited:
        seg = bfs_path(adj, cur, unvisited)
        if not seg:
            break                       # remaining regions are in another component
        walk.extend(seg[1:])
        for r in seg:
            unvisited.discard(r)
        cur = seg[-1]
    return walk


def classify(rm, walk: list[int]) -> list[dict]:
    """Every junction traversal in the walk, with the maneuver it commands."""
    out = []
    for i in range(1, len(walk) - 1):
        rid = walk[i]
        if rm.regions[rid].label != "junction":
            continue
        ax, ay = rm.regions[walk[i - 1]].centroid
        jx, jy = rm.regions[rid].centroid
        bx, by = rm.regions[walk[i + 1]].centroid
        h_in = math.degrees(math.atan2(jy - ay, jx - ax))
        turn = _wrap180(math.degrees(math.atan2(by - jy, bx - jx)) - h_in)
        kind = ("straight" if abs(turn) < 45 else ("right" if turn < 0 else "left"))
        out.append({"step": i, "junction_rid": rid, "from_rid": walk[i - 1],
                    "to_rid": walk[i + 1], "turn_deg": round(turn, 1),
                    "maneuver": kind})
    return out


def route_length_m(rm, walk: list[int]) -> float:
    d = 0.0
    for a, b in zip(walk, walk[1:]):
        ax, ay = rm.regions[a].centroid
        bx, by = rm.regions[b].centroid
        d += math.hypot(bx - ax, by - ay)
    return d


def plot(rm, walk, turns, out_png: str, town: str, stats: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 11), dpi=130)
    fig.patch.set_facecolor("white")

    for r in rm.regions.values():
        xs = [p[0] for p in r.points]
        ys = [p[1] for p in r.points]
        ax.scatter(xs, ys, s=6, c=LABEL_COL.get(r.label, "#DDD"),
                   edgecolors="none", zorder=1)

    pts = [rm.regions[r].centroid for r in walk]
    # Colour the path by progress so the ORDER is readable, not just the shape.
    cmap = plt.get_cmap("viridis")
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(pts, pts[1:])):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", lw=1.6,
                                    color=cmap(i / max(len(pts) - 2, 1)),
                                    alpha=.9, shrinkA=0, shrinkB=0))

    mark = {"left": ("<", "#DC2626"), "right": (">", "#2563EB"),
            "straight": ("^", "#059669")}
    for t in turns:
        jx, jy = rm.regions[t["junction_rid"]].centroid
        m, c = mark[t["maneuver"]]
        ax.scatter([jx], [jy], marker=m, s=110, c=c, zorder=6,
                   edgecolors="white", linewidths=.7)

    # The rare labels are the REASON a town is in the collection, so mark them rather
    # than leaving them as one pale dot among 146 regions.
    for r in rm.regions.values():
        if r.label in ("passage", "along_edge", "open_space") and r.rid in set(walk):
            cx, cy = r.centroid
            ax.scatter([cx], [cy], s=420, marker="o", facecolors="none",
                       edgecolors="#7C3AED", lw=2.6, zorder=8)
            ax.annotate(f"{r.label} (r{r.rid})", (cx, cy), fontsize=9,
                        fontweight="bold", color="#5B21B6", zorder=9,
                        xytext=(12, 10), textcoords="offset points")

    sx, sy = pts[0]
    ex, ey = pts[-1]
    ax.scatter([sx], [sy], s=240, marker="o", facecolors="none", edgecolors="#111",
               lw=2.4, zorder=7)
    ax.annotate("START", (sx, sy), fontsize=9, fontweight="bold",
                xytext=(8, 8), textcoords="offset points", zorder=8)
    ax.scatter([ex], [ey], s=240, marker="s", facecolors="none", edgecolors="#111",
               lw=2.4, zorder=7)
    ax.annotate("END", (ex, ey), fontsize=9, fontweight="bold",
                xytext=(8, -14), textcoords="offset points", zorder=8)

    handles = [plt.Line2D([], [], marker=m, ls="", color=c, label=f"{k} turn")
               for k, (m, c) in mark.items()]
    handles += [plt.Line2D([], [], marker="o", ls="", color=col, label=lab)
                for lab, col in LABEL_COL.items()
                if any(r.label == lab for r in rm.regions.values())]
    ax.legend(handles=handles, fontsize=8, loc="best", framealpha=.92)

    ax.set_title(
        f"{town} — commanded data-collection route\n"
        f"{stats['regions_visited']}/{stats['regions_total']} regions "
        f"({stats['coverage_pct']:.0f}% coverage) · {stats['steps']} region steps · "
        f"~{stats['length_m'] / 1000:.2f} km · ~{stats['minutes']:.0f} min at 8 m/s\n"
        f"junction traversals: {stats['left']} left · {stats['right']} right · "
        f"{stats['straight']} straight  ·  all "
        f"{stats['junctions_traversed']} junctions entered\n"
        f"arrow colour = progress (dark to bright). Arrows join region CENTROIDS: the "
        f"commanded route is the region SEQUENCE, and the driven line comes from "
        f"CARLA's route planner between them.",
        fontsize=10)
    ax.set_xlabel("x (m), OpenDRIVE/ROS planar frame")
    ax.set_ylabel("y (m), +north — CARLA's own y is the negation")
    ax.set_aspect("equal")
    ax.grid(alpha=.15, lw=.5)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def plan_town(xodr: str, out_dir: str, approach_m: float = 15.0,
              speed_ms: float = 8.0) -> dict:
    m = load(xodr)
    rm = segment(m, step=2.0, approach_m=approach_m)
    walk = covering_walk(rm)
    turns = classify(rm, walk)
    length = route_length_m(rm, walk)
    counts = {k: sum(1 for t in turns if t["maneuver"] == k)
              for k in ("left", "right", "straight")}
    labels_hit = {}
    for r in set(walk):
        lab = rm.regions[r].label
        labels_hit[lab] = labels_hit.get(lab, 0) + 1

    stats = {
        "town": rm.town, "regions_total": len(rm.regions),
        "regions_visited": len(set(walk)),
        "coverage_pct": 100.0 * len(set(walk)) / max(len(rm.regions), 1),
        "steps": len(walk), "length_m": round(length, 1),
        "minutes": length / speed_ms / 60.0,
        "labels_visited": labels_hit,
        "junctions_traversed": len({t["junction_rid"] for t in turns}),
        "junctions_total": sum(1 for r in rm.regions.values()
                               if r.label == "junction"),
        **counts,
    }

    os.makedirs(out_dir, exist_ok=True)
    png = plot(rm, walk, turns, os.path.join(out_dir, f"route.{rm.town.lower()}.png"),
               rm.town, stats)

    # Targets for the drive. The Python API is LEFT-handed, the region table is planar;
    # carla_gt_bridge.frames is the authority and getting this backwards mirrors the
    # whole route, so both are written out and the consumer picks by frame, not by guess.
    targets = []
    for rid in walk:
        x, y = rm.regions[rid].centroid
        targets.append({"rid": rid, "label": rm.regions[rid].label,
                        "planar": [round(x, 2), round(y, 2)],
                        "carla": [round(x, 2), round(-y, 2)]})
    js = os.path.join(out_dir, f"route.{rm.town.lower()}.json")
    with open(js, "w") as f:
        json.dump({"stats": stats, "region_sequence": walk, "turns": turns,
                   "targets": targets,
                   "note": "planar = OpenDRIVE/ROS (+y north). carla = CARLA Python "
                           "API (+y south). Odometry off the ros-bridge is PLANAR."},
                  f, indent=2)
    stats["_png"], stats["_json"] = png, js
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--town", action="append", default=None,
                    help="repeatable; default Town01 and Town07")
    ap.add_argument("--out-dir", default=os.path.join(PKG, "reports", "collection"))
    ap.add_argument("--approach-m", type=float, default=15.0)
    ap.add_argument("--speed-ms", type=float, default=8.0)
    a = ap.parse_args(argv)
    towns = a.town or ["Town01", "Town07"]

    for t in towns:
        xodr = t if t.endswith(".xodr") else os.path.join(PKG, "config", f"{t}.xodr")
        if not os.path.exists(xodr):
            print(f"missing {xodr}")
            return 1
        s = plan_town(xodr, a.out_dir, a.approach_m, a.speed_ms)
        print(f"\n{s['town']}")
        print(f"  regions      {s['regions_visited']}/{s['regions_total']} "
              f"({s['coverage_pct']:.0f}%)  by label: {s['labels_visited']}")
        print(f"  route        {s['steps']} region steps, "
              f"{s['length_m'] / 1000:.2f} km, ~{s['minutes']:.0f} min at "
              f"{a.speed_ms:.0f} m/s")
        print(f"  maneuvers    {s['left']} left · {s['right']} right · "
              f"{s['straight']} straight")
        print(f"  junctions    {s['junctions_traversed']}/{s['junctions_total']} "
              f"entered")
        print(f"  {s['_png']}")
        print(f"  {s['_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
