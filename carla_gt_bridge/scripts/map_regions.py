#!/usr/bin/env python3
"""Derive cluster regions from a CARLA town and render them — fully offline.

No CARLA server, no GPU: the town's `.xodr` ships inside the Docker image, and
`carla_gt_bridge.opendrive` evaluates its geometry directly. That makes region
derivation deterministic and testable, which matters because uncalibrated
hand-guessed geometry is the failure mode this replaces.

Outputs (into --out-dir):
  cluster_map.<town>.yaml    modes + centroids + bearing_map, for nl_planner
  regions.<town>.npz         (N,3) x,y,rid waypoint table for the runtime KD-tree
  regions.<town>.png         top-down plot: what the ground truth actually is

Get the map first (one-off, needs docker but not a running server):
  docker run --rm --entrypoint /bin/cat carlasim/carla:0.9.14 \
    /home/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town01.xodr > Town01.xodr

Usage:
  python3 scripts/map_regions.py --xodr Town01.xodr
  python3 scripts/map_regions.py --xodr Town01.xodr --route 3 --hops 3
  python3 scripts/map_regions.py --xodr Town01.xodr --maneuvers
"""
from __future__ import annotations

import argparse
import math
import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from carla_gt_bridge.opendrive import load                      # noqa: E402
from carla_gt_bridge.segmenter import segment                   # noqa: E402


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def maneuvers(m, rm):
    """Per junction approach, classify each exit as straight / left / right.

    Needed to site a decision mission: at a T-junction (all 12 of Town01's are
    T-junctions, 3 approaches) "go straight OR turn right" only both exist from
    certain approaches, so the branch mission cannot be placed arbitrarily.
    """
    src2rid = {r.source: r.rid for r in rm.regions.values()}
    rows = []
    for jid, jn in m.junctions.items():
        j_rid = src2rid.get(f"junction:{jid}")
        if j_rid is None:
            continue
        jx, jy = rm.regions[j_rid].centroid
        approaches = sorted({rid for rid, _ in jn.connections})
        for a in approaches:
            a_rid = src2rid.get(f"road:{a}")
            if a_rid is None:
                continue
            ax, ay = rm.regions[a_rid].centroid
            h_in = math.degrees(math.atan2(jy - ay, jx - ax))
            exits = []
            for b in approaches:
                if b == a:
                    continue
                b_rid = src2rid.get(f"road:{b}")
                if b_rid is None:
                    continue
                bx, by = rm.regions[b_rid].centroid
                turn = _wrap180(math.degrees(math.atan2(by - jy, bx - jx)) - h_in)
                kind = ("straight" if abs(turn) < 45
                        else ("right" if turn < 0 else "left"))
                exits.append((kind, round(turn), b_rid))
            rows.append((jid, j_rid, a_rid, exits))
    return rows


def write_cluster_map(rm, out_dir: str, suffix: str = "") -> str:
    """Emit the taxonomy through bev_pipeline's HIERARCHICAL exporter.

    Not a flat `{label: [ids]}` map: routing through
    `write_hierarchical_cluster_map` is what gives the plan the subsumption
    lattice (`junction` also satisfies a `path` step) and the `mode_meta`
    degradation sets that the grounding layer depends on. A flat map would make
    the whole acceptance-set mechanism inert.

    Region ids are the real cluster ids, so the exporter is fed a labels list
    indexed BY region id — gaps included, since a corridor is a sparse subset of
    the town's ids.
    """
    ws = os.path.dirname(os.path.dirname(PKG))
    sys.path.insert(0, os.path.join(ws, "src", "bev_pipeline"))
    sys.path.insert(0, os.path.join(ws, "bev_pipeline"))
    from bev_pipeline.taxonomy_export import (ROAD_HIERARCHY,
                                              write_hierarchical_cluster_map)

    # When the junction is split into phases, use the ROAD vocabulary and extend it with
    # the approach phase. ROAD_HIERARCHY is keyed by the segmenter's plain labels and
    # supplies both the pretty name and the ancestor, so `Road: On` still subsumes every
    # region and a step naming the coarse mode keeps grounding.
    hierarchy = None
    if any(r.label == "approach" for r in rm.regions.values()):
        hierarchy = dict(ROAD_HIERARCHY)
        hierarchy["approach"] = ("Intersection: Approach/Enter", ["Road: On"])

    env = f"carla_{rm.town.lower()}"
    p = os.path.join(out_dir, f"cluster_map.{env}{suffix}.yaml")
    write_hierarchical_cluster_map(
        [], env, p,
        source=f"carla_gt_bridge/map_regions.py from {rm.town}.xodr",
        centroids=rm.centroids(),
        bearing_map={k: round(v, 2) for k, v in rm.bearing_map().items()},
        # region ids are sparse (a corridor is a subset of the town), so pass the
        # id->label mapping rather than a positional list.
        id_labels={r.rid: r.label for r in rm.regions.values()},
        hierarchy=hierarchy,
    )
    return p


def write_npz(rm, out_dir: str, suffix: str = "") -> str:
    import numpy as np

    wp = rm.waypoints()
    labels = {r.rid: r.label for r in rm.regions.values()}
    p = os.path.join(out_dir, f"regions.{rm.town.lower()}{suffix}.npz")
    np.savez_compressed(
        p, waypoints=wp,
        rids=np.array(sorted(rm.regions)),
        labels=np.array([labels[r] for r in sorted(rm.regions)]),
        centroids=np.array([rm.regions[r].centroid for r in sorted(rm.regions)]),
    )
    return p


def plot(rm, out_dir: str, highlight=None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    highlight = set(highlight or [])
    # Colour by cluster LABEL, not by id: the point of the picture is to show that
    # the ground truth separates streets from intersections. Label rather than
    # `kind` because a narrow bridge deck is kind `path` but label `passage`, and
    # the label is what the plan matches.
    col = {"path": "#2A62D0", "junction": "#C2410C", "passage": "#7C3AED",
           "along_edge": "#0F766E", "open_space": "#B45309", "other": "#6B7280"}
    fig, ax = plt.subplots(figsize=(11, 9), dpi=130)
    fig.patch.set_facecolor("white")

    for r in rm.regions.values():
        xs = [p[0] for p in r.points]
        ys = [p[1] for p in r.points]
        hot = r.rid in highlight
        ax.scatter(xs, ys, s=14 if hot else 5,
                   c=col.get(r.label, "#888"),
                   alpha=1.0 if hot else 0.35,
                   edgecolors="none", zorder=3 if hot else 2)
        cx, cy = r.centroid
        ax.annotate(str(r.rid), (cx, cy),
                    fontsize=9 if hot else 7,
                    fontweight="bold" if hot else "normal",
                    color="#111" if hot else "#666",
                    ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18",
                              fc="#FFE9B0" if hot else "white",
                              ec="#999" if hot else "#DDD", lw=.6, alpha=.95))

    for a, b in rm.adjacency:
        ax_, ay_ = rm.regions[a].centroid
        bx_, by_ = rm.regions[b].centroid
        ax.plot([ax_, bx_], [ay_, by_], lw=.8, color="#BBB", zorder=1)

    counts = {}
    for r in rm.regions.values():
        counts[r.label] = counts.get(r.label, 0) + 1
    ax.set_title(f"{rm.town} — ground-truth cluster regions\n"
                 + " · ".join(f"{n} {lab}" for lab, n in sorted(counts.items()))
                 + f" · {len(rm.adjacency)} adjacencies"
                 + (f" · highlighted route {sorted(highlight)}" if highlight else ""),
                 fontsize=11)
    ax.set_xlabel("x (m), OpenDRIVE/ROS frame")
    ax.set_ylabel("y (m), +north — CARLA's own y is negated")
    ax.set_aspect("equal")
    ax.grid(alpha=.15, lw=.5)
    fig.tight_layout()
    p = os.path.join(out_dir, f"regions.{rm.town.lower()}.png")
    fig.savefig(p)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xodr", required=True)
    ap.add_argument("--out-dir", default=os.path.join(PKG, "config"))
    ap.add_argument("--lanes-per-side", type=int, default=0,
                    help="sample LANE CENTRES this many per side as well as the road "
                         "reference line. The reference line alone is the drivable "
                         "surface the MPC measures against, and on a 14 m road its edge "
                         "sits exactly at the 7.0 m threshold -- which is where all 23 "
                         "CONTROL failures of the driven constraint campaign happened. "
                         "0 (default) regenerates the existing table byte-identically.")
    ap.add_argument("--approach-m", type=float, default=0.0,
                    help="carve this many metres of each junction-adjacent road end "
                         "into its own `approach` region, giving the three-phase "
                         "decomposition the robot taxonomy already uses. 0 = off, which "
                         "reproduces the existing tables exactly.")
    ap.add_argument("--suffix", default="",
                    help="append to the output filenames, so a variant table can sit "
                         "beside the canonical one instead of overwriting it")
    ap.add_argument("--step", type=float, default=2.0,
                    help="reference-line sample spacing, metres")
    ap.add_argument("--route", type=int, default=None,
                    help="scope to a BFS neighbourhood around this region id")
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--corridor", type=int, nargs="*", default=None,
                    help="scope to an EXPLICIT ordered region sequence — prefer "
                         "this for missions; --route expands in all directions")
    ap.add_argument("--highlight", type=int, nargs="*", default=None)
    ap.add_argument("--maneuvers", action="store_true",
                    help="list junction approaches offering straight AND right")
    ap.add_argument("--features", action="store_true",
                    help="audit the town's shape: bridges, narrow roads, non-drivable "
                         "roads, elevation. Run this before choosing a town.")
    args = ap.parse_args()

    m = load(args.xodr)
    rm_full = segment(m, step=args.step, approach_m=args.approach_m,
                      lanes_per_side=args.lanes_per_side)
    counts: dict = {}
    for r in rm_full.regions.values():
        counts[r.label] = counts.get(r.label, 0) + 1
    print(f"{rm_full.town}: {len(m.roads)} roads, {len(m.junctions)} junctions -> "
          f"{len(rm_full.regions)} regions "
          f"({', '.join(f'{n} {lab}' for lab, n in sorted(counts.items()))}), "
          f"{len(rm_full.adjacency)} adjacencies")

    if args.features:
        from carla_gt_bridge.segmenter import (bridge_passages, narrow_candidates,
                                               nondrivable_regions)
        print("\nshape audit — what this town actually contains:")
        bp = bridge_passages(m)
        bridged = [r for r in m.path_roads if r.longest_bridge > 0]
        print(f"  roads with a <bridge> record : {len(bridged)}")
        for r in sorted(bridged, key=lambda r: -r.longest_bridge):
            verdict = "PASSAGE" if f"road:{r.road_id}" in bp else "path (too wide/short)"
            print(f"      r{r.road_id:<5d} deck={r.longest_bridge:7.1f} m  "
                  f"carriageway={r.driving_width:5.1f} m  total={r.total_width:5.1f} m "
                  f"-> {verdict}")
        nc = narrow_candidates(m)
        print(f"  narrow roads (<=10 m TOTAL)  : {len(nc)}  "
              f"— candidates for a hand-tagged `passage`, not auto-labelled: "
              f"narrowness does not prove structure on both sides")
        for c in nc[:12]:
            flag = "  [bridge]" if c["has_bridge"] else ""
            flag += "  [NOT DRIVABLE]" if not c["drivable"] else ""
            print(f"      r{c['road_id']:<5d} len={c['length']:7.1f} m  "
                  f"carriageway={c['driving_width']:5.1f} m  "
                  f"total={c['total_width']:5.1f} m{flag}")
        if len(nc) > 12:
            print(f"      ... and {len(nc) - 12} more")
        nd = nondrivable_regions(m)
        print(f"  roads with NO drivable lane  : {len(nd)} {nd[:12]}"
              f"{' ...' if len(nd) > 12 else ''}")
        if nd:
            print("      these still become `path` regions, so a plan step could ground "
                  "onto one and steer the vehicle somewhere it may not drive")
        widths = sorted({(round(r.driving_width, 1), round(r.total_width, 1))
                         for r in m.path_roads})
        print(f"  distinct (carriageway, total) widths: {widths[:8]}"
              f"{' ...' if len(widths) > 8 else ''}")
        if len(widths) == 1:
            print("      a single width for the whole town — a uniform grid with no "
                  "alleys and no decks, whatever the scenery looks like")

    if args.maneuvers:
        print("\napproaches offering BOTH straight and right (decision-mission sites):")
        for jid, j_rid, a_rid, exits in maneuvers(m, rm_full):
            names = {e[0] for e in exits}
            if not {"straight", "right"} <= names:
                continue
            ex = "  ".join(f"{n}({t:+d})->r{r}" for n, t, r in exits)
            print(f"  junction {jid:>3} = region {j_rid:>2}   approach region "
                  f"{a_rid:>2}   {ex}")

    if args.corridor:
        rm = rm_full.corridor(args.corridor)
        print(f"\nscoped to corridor {args.corridor}: {len(rm.regions)} regions")
    elif args.route is not None:
        rm = rm_full.sub_route(args.route, args.hops)
        print(f"\nscoped to neighbourhood of region {args.route} ({args.hops} hops): "
              f"{len(rm.regions)} regions {sorted(rm.regions)}")
    else:
        rm = rm_full

    os.makedirs(args.out_dir, exist_ok=True)
    print()
    print(" ", write_cluster_map(rm, args.out_dir, args.suffix))
    print(" ", write_npz(rm, args.out_dir, args.suffix))
    hl = args.highlight or (sorted(rm.regions) if rm is not rm_full else None)
    print(" ", plot(rm_full, args.out_dir, highlight=hl))


if __name__ == "__main__":
    main()
