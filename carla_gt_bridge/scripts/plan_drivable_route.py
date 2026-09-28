#!/usr/bin/env python3
"""Plan a collection drive a CAR can actually make, and report what it covers.

    ~/miniconda3/bin/python scripts/plan_drivable_route.py --town Town01 --town Town07

Needs the `carla` module (conda python has it); no server, no GPU, no rendering.

WHY THIS REPLACES THE GRAPH WALK
`plan_collection_routes.py` walks the region ADJACENCY graph. That graph is undirected
and a car is not: two regions can be adjacent and ~18 m apart yet ~550 m apart by road
(Town01 regions 0 and 11 are opposite carriageways of one street, and the vehicle cannot
U-turn). A full region-graph walk resolved through the real planner comes out several
times longer than its centroid estimate, which at 20 Hz on two LiDARs means hundreds of
GB of bag and a drive far longer than planned.

So the tour is planned here with two changes, both of which cut the distance hard:

1. **Cost is driving distance**, from the same `GlobalRoutePlanner` the vehicle uses —
   not graph hops, not euclidean.
2. **Regions are credited when the car DRIVES THROUGH them**, not only when they are
   chosen as targets. Going down one street covers its `path`, both `approach`es and the
   `junction` at the end for free, so most regions never need a leg of their own. The
   coverage test is `RegionTable.nearest(x, y) <= cover_m`, which is the *same* test
   `label_frames.py` will use to assign the ground truth — so the coverage number here
   is a prediction of the corpus, not a proxy for it.

Greedy: from the current pose, take the nearest uncovered region by straight line as a
shortlist, trace each, drive to whichever is genuinely closest, mark everything the leg
passes through, repeat. It is not optimal — the optimal version is a directed rural
postman problem — but it is drivable, and it reports its own coverage so a thin corpus
is visible before the sim runs rather than after.

Outputs per town, into --out-dir:
    drivable.<town>.png     the route, lane level, coloured by CARLA's own RoadOption
    drivable.<town>.json    legs, targets (both frames), coverage and maneuver stats
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.expanduser("~/carla"))

import carla  # noqa: E402
from agents.navigation.global_route_planner import GlobalRoutePlanner  # noqa: E402

from carla_gt_bridge.frame_labels import FrameRow, junction_phases  # noqa: E402
from carla_gt_bridge.opendrive import load  # noqa: E402
from carla_gt_bridge.region_lookup import load_region_table  # noqa: E402
from carla_gt_bridge.segmenter import segment  # noqa: E402

LABEL_COL = {"path": "#93C5FD", "approach": "#6EE7B7", "junction": "#FDBA74",
             "passage": "#C4B5FD", "along_edge": "#5EEAD4", "open_space": "#FCD34D"}
OPT = {-1: "VOID", 1: "LEFT", 2: "RIGHT", 3: "STRAIGHT", 4: "LANEFOLLOW",
       5: "CHANGELANELEFT", 6: "CHANGELANERIGHT"}
OPT_COL = {"LEFT": "#DC2626", "RIGHT": "#2563EB", "STRAIGHT": "#059669",
           "LANEFOLLOW": "#334155", "CHANGELANELEFT": "#DB2777",
           "CHANGELANERIGHT": "#DB2777", "VOID": "#9CA3AF"}

#: A region counts as driven when a route waypoint is this close to one of its sampled
#: waypoints. Matches `label_frames.py`'s own lookup, so coverage predicts the corpus.
COVER_M = 6.0

PHASE_COL = {"path": "#2A62D0", "approach": "#0F766E", "junction": "#C2410C",
             "exit": "#7C3AED", "off_network": "#9CA3AF", "passage": "#DB2777"}


def predict_phases(pts, table, *, speed_ms=8.0, hz=10.0):
    """Resample the route at the LiDAR rate and derive the topology phase per frame.

    WHY THIS IS A SEPARATE PICTURE FROM THE REGION MAP. The region map has no `exit`
    colour and cannot have one: `segmenter.segment` emits `approach` as a DIRECTION-FREE
    label, because the same stretch of tarmac is an approach driven toward a junction and
    an exit driven away from it, and a static region carries no direction. `exit` is a
    property of the TRAJECTORY, so it appears only once a route is laid over the map --
    which is exactly what this does, and exactly what `label_frames.py` will do to the
    bag.

    Sampling at `speed_ms / hz` metres makes the counts a prediction of the CORPUS: how
    many frames of each phase the drive will actually yield. That matters most for
    `exit`, the rarest phase: if the route yields 40 exit frames out of 6000, the
    confusion matrix for that class is noise whatever the model does.
    """
    step = speed_ms / hz
    rows: list[FrameRow] = []
    acc, i = 0.0, 0
    labels = {r: table.label_of(r) for r in table.region_ids}
    jc = {r: table.centroid_of(r) for r in table.region_ids
          if labels[r] == "junction"}
    for (x0, y0, _o0), (x1, y1, _o1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-6:
            continue
        yaw = math.atan2(y1 - y0, x1 - x0)
        while acc < seg:
            f = acc / seg
            px, py = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
            rid, dist = table.nearest(px, py)
            off = dist > 12.0
            rows.append(FrameRow(t_ns=int(i * 1e9 / hz), x=px, y=py, yaw=yaw,
                                 speed=speed_ms, rid=-1 if off else int(rid),
                                 region_label="off_network" if off
                                 else labels[int(rid)],
                                 region_dist_m=dist, off_network=off))
            i += 1
            acc += step
        acc -= seg
    junction_phases(rows, jc)
    # `passage` is a label, not a phase, and it must not be swallowed by `path`: it is
    # the one region type Town07 is being driven for.
    for r in rows:
        if r.region_label == "passage":
            r.topology = "passage"
    return rows


def _leg_points(leg):
    """(x, y, option) in PLANAR metres. CARLA is left-handed; y is negated once here."""
    out = []
    for wp, opt in leg:
        t = wp.transform.location
        out.append((float(t.x), -float(t.y), OPT.get(int(opt), str(opt))))
    return out


def _len(pts):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


#: Labels so rare that a coverage tour clips them: a shortest-path leg only nicks the
#: corner of Town07's `passage` r45 (a 36.4 m deck), yielding fewer frames than one full
#: traversal, for the one class that town is being driven to obtain. HDBSCAN needs 8
#: members just to form a cluster and `assign_cluster_labels` needs a majority; a few
#: dozen frames scattered over several clusters is not a class, it is noise.
RARE_LABELS = ("passage", "along_edge", "open_space")


def rare_repeats(grp, table, rm, rare_rids, *, repeats: int, cover_m: float,
                 start: "carla.Location"):
    """Extra out-and-back traversals of the rare regions, appended to the tour.

    Routes between two regions on OPPOSITE sides of the rare one and alternates
    direction, so the corpus gets the region driven both ways — which matters because
    the enclosure relations (`past_`, `around_`) are direction-dependent, and a deck
    entered from one end only would train an asymmetry that is an artifact of the route.
    """
    legs, pts_all = [], []
    cur = start

    def connect(to_xy):
        """Drive from the current pose to a leg's first point; never jump."""
        nonlocal cur
        tx, ty = to_xy
        if math.hypot(tx - cur.x, ty + cur.y) < 1.0:
            return []
        try:
            return _leg_points(grp.trace_route(
                cur, carla.Location(x=tx, y=-ty, z=0.0)))
        except Exception:
            return []

    adj: dict[int, set[int]] = {r: set() for r in rm.regions}
    for a, b in rm.adjacency:
        adj[a].add(b)
        adj[b].add(a)

    for rid in rare_rids:
        nb = sorted(adj[rid])
        if len(nb) < 2:
            continue
        best_pair = None
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                a, b = nb[i], nb[j]
                ax, ay = rm.regions[a].centroid
                bx, by = rm.regions[b].centroid
                try:
                    leg = grp.trace_route(carla.Location(x=ax, y=-ay, z=0.0),
                                          carla.Location(x=bx, y=-by, z=0.0))
                except Exception:
                    continue
                pts = _leg_points(leg)
                # keep the pair only if the route really goes THROUGH the rare region
                n_in = sum(1 for x, y, _o in pts
                           if table.nearest(x, y)[0] == rid
                           and table.nearest(x, y)[1] <= cover_m)
                if n_in and (best_pair is None or n_in > best_pair[0]):
                    best_pair = (n_in, a, b, pts)
        if best_pair is None:
            continue
        _n, a, b, fwd = best_pair
        ax, ay = rm.regions[a].centroid
        bx, by = rm.regions[b].centroid
        try:
            rev = _leg_points(grp.trace_route(carla.Location(x=bx, y=-by, z=0.0),
                                              carla.Location(x=ax, y=-ay, z=0.0)))
        except Exception:
            rev = []
        for k in range(repeats):
            leg_pts = fwd if k % 2 == 0 else (rev or fwd)
            if not leg_pts:
                continue
            link = connect(leg_pts[0][:2])
            if link:
                pts_all.extend(link)
                cur = carla.Location(x=link[-1][0], y=-link[-1][1], z=0.0)
                legs.append({"target_rid": rid, "label": "connector",
                             "driven_m": round(_len(link), 1), "newly_covered": 0})
            pts_all.extend(leg_pts)
            cur = carla.Location(x=leg_pts[-1][0], y=-leg_pts[-1][1], z=0.0)
            legs.append({"target_rid": rid, "label": rm.regions[rid].label,
                         "driven_m": round(_len(leg_pts), 1),
                         "newly_covered": 0, "rare_repeat": k + 1,
                         "between": [a, b] if k % 2 == 0 else [b, a]})
    return pts_all, legs


def plan(town: str, xodr: str, table, rm, *, shortlist: int = 8,
         cover_m: float = COVER_M, max_legs: int = 400, repeats: int = 0):
    cmap = carla.Map(town, open(xodr).read())
    grp = GlobalRoutePlanner(cmap, 2.0)

    centroid = {r.rid: r.centroid for r in rm.regions.values()}
    uncovered = set(centroid)
    route: list[tuple[float, float, str]] = []
    legs: list[dict] = []

    def cover(pts):
        """Mark every region the polyline passes through, the labeller's own test."""
        hit = set()
        for x, y, _o in pts:
            rid, d = table.nearest(x, y)
            if d <= cover_m:
                hit.add(int(rid))
        return hit

    # Start on a real drivable lane near the first region, not at a centroid in mid-air.
    start_rid = min(uncovered)
    sx, sy = centroid[start_rid]
    cur = carla.Location(x=sx, y=-sy, z=0.0)
    uncovered -= cover([(sx, sy, "LANEFOLLOW")])

    while uncovered and len(legs) < max_legs:
        here = np.array([cur.x, -cur.y])
        cands = sorted(uncovered,
                       key=lambda r: (centroid[r][0] - here[0]) ** 2
                       + (centroid[r][1] - here[1]) ** 2)[:shortlist]
        best = None
        for rid in cands:
            cx, cy = centroid[rid]
            try:
                leg = grp.trace_route(cur, carla.Location(x=cx, y=-cy, z=0.0))
            except Exception:
                continue
            if not leg:
                continue
            pts = _leg_points(leg)
            d = _len(pts)
            if d < 1.0:                       # already there
                best = (0.0, rid, pts)
                break
            if best is None or d < best[0]:
                best = (d, rid, pts)
        if best is None:
            break                             # nothing reachable; stop honestly
        d, rid, pts = best
        got = cover(pts)
        # A leg that reaches its target but covers nothing new would loop forever.
        if not (got & uncovered) and rid not in uncovered:
            uncovered.discard(rid)
            continue
        route.extend(pts if not route else pts[1:])
        legs.append({"target_rid": rid, "label": rm.regions[rid].label,
                     "driven_m": round(d, 1), "newly_covered": len(got & uncovered)})
        uncovered -= got
        uncovered.discard(rid)
        lx, ly, _ = pts[-1]
        cur = carla.Location(x=lx, y=-ly, z=0.0)

    if repeats:
        rare = [r.rid for r in rm.regions.values() if r.label in RARE_LABELS]
        extra_pts, extra_legs = rare_repeats(grp, table, rm, rare,
                                             repeats=repeats, cover_m=cover_m,
                                             start=cur)
        route.extend(extra_pts)
        legs.extend(extra_legs)

    return route, legs, uncovered


def maneuvers(pts) -> dict:
    """Contiguous runs of each decision option — one run is one maneuver, not one point."""
    runs: dict[str, int] = {}
    prev = None
    for _x, _y, o in pts:
        if o != prev and o not in ("LANEFOLLOW", "VOID"):
            runs[o] = runs.get(o, 0) + 1
        prev = o
    return runs


def plot(rm, pts, out_png, town, stats, uncovered):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 11), dpi=130)
    fig.patch.set_facecolor("white")
    for r in rm.regions.values():
        ax.scatter([p[0] for p in r.points], [p[1] for p in r.points], s=6,
                   c=LABEL_COL.get(r.label, "#DDD"), edgecolors="none", zorder=1)
        if r.rid in uncovered:
            cx, cy = r.centroid
            ax.scatter([cx], [cy], s=150, marker="x", c="#111", lw=1.6, zorder=9)
        if r.label in ("passage", "along_edge", "open_space"):
            cx, cy = r.centroid
            ax.scatter([cx], [cy], s=420, marker="o", facecolors="none",
                       edgecolors="#7C3AED", lw=2.6, zorder=8)
            ax.annotate(f"{r.label} (r{r.rid})", (cx, cy), fontsize=9,
                        fontweight="bold", color="#5B21B6", zorder=10,
                        xytext=(12, 10), textcoords="offset points")

    for i in range(len(pts) - 1):
        o = pts[i][2]
        lane = o == "LANEFOLLOW"
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                lw=1.2 if lane else 4.2, solid_capstyle="round",
                color=OPT_COL.get(o, "#000"), alpha=.5 if lane else 1.0,
                zorder=3 if lane else 5)

    for p, m, lab in ((pts[0], "o", "START"), (pts[-1], "s", "END")):
        ax.scatter([p[0]], [p[1]], s=240, marker=m, facecolors="none",
                   edgecolors="#111", lw=2.4, zorder=11)
        ax.annotate(lab, p[:2], fontsize=9, fontweight="bold", zorder=11,
                    xytext=(8, 8), textcoords="offset points")

    handles = [plt.Line2D([], [], lw=3, color=OPT_COL[k], label=k)
               for k in ("LANEFOLLOW", "LEFT", "RIGHT", "STRAIGHT")]
    if uncovered:
        handles.append(plt.Line2D([], [], marker="x", ls="", color="#111",
                                  label=f"NOT covered ({len(uncovered)})"))
    handles += [plt.Line2D([], [], marker="o", ls="", color=c, label=lab)
                for lab, c in LABEL_COL.items()
                if any(r.label == lab for r in rm.regions.values())]
    ax.legend(handles=handles, fontsize=8, loc="best", framealpha=.92)

    mv = stats["maneuvers"]
    ax.set_title(
        f"{town} — drivable collection route (lane level, CARLA route planner)\n"
        f"{stats['covered']}/{stats['regions_total']} regions "
        f"({stats['coverage_pct']:.0f}%) · {stats['legs']} legs · "
        f"{stats['length_m'] / 1000:.2f} km · ~{stats['minutes']:.0f} min at 8 m/s · "
        f"~{stats['bag_gb']:.0f} GB at 10 Hz x 2 LiDAR\n"
        f"maneuvers: {mv.get('LEFT', 0)} left · {mv.get('RIGHT', 0)} right · "
        f"{mv.get('STRAIGHT', 0)} straight",
        fontsize=10)
    ax.set_xlabel("x (m), OpenDRIVE/ROS planar frame")
    ax.set_ylabel("y (m), +north — CARLA's own y is the negation")
    ax.set_aspect("equal")
    ax.grid(alpha=.15, lw=.5)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def plot_phases(rm, rows, out_png, town):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    counts = Counter(r.topology for r in rows)
    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(17, 10), dpi=130,
        gridspec_kw={"width_ratios": [3, 1], "wspace": .22})
    fig.patch.set_facecolor("white")

    for r in rm.regions.values():
        ax.scatter([p[0] for p in r.points], [p[1] for p in r.points], s=5,
                   c="#E5E7EB", edgecolors="none", zorder=1)
    for lab in ("path", "approach", "junction", "exit", "passage", "off_network"):
        sel = [r for r in rows if r.topology == lab]
        if not sel:
            continue
        ax.scatter([r.x for r in sel], [r.y for r in sel], s=9,
                   c=PHASE_COL[lab], edgecolors="none", zorder=3,
                   label=f"{lab} ({len(sel)})")
    ax.set_aspect("equal")
    ax.grid(alpha=.15, lw=.5)
    ax.legend(fontsize=9, loc="best", framealpha=.92)
    ax.set_xlabel("x (m), OpenDRIVE/ROS planar frame")
    ax.set_ylabel("y (m), +north")
    ax.set_title(f"{town} — DERIVED PHASE along the driven route\n"
                 f"the region map has no `exit` colour because `approach` is "
                 f"direction-free;\n`exit` exists only once a trajectory is laid "
                 f"over it", fontsize=11)

    order = [k for k in ("path", "approach", "junction", "exit", "passage",
                         "off_network") if counts.get(k)]
    vals = [counts[k] for k in order]
    bars = axb.barh(order[::-1], vals[::-1],
                    color=[PHASE_COL[k] for k in order[::-1]])
    tot = sum(vals)
    for b, v in zip(bars, vals[::-1]):
        axb.text(b.get_width() + tot * .01, b.get_y() + b.get_height() / 2,
                 f"{v}  ({100 * v / tot:.1f}%)", va="center", fontsize=9)
    axb.set_xlim(0, max(vals) * 1.35)
    axb.set_xlabel("frames at 10 Hz, 8 m/s")
    axb.set_title(f"predicted corpus: {tot} frames", fontsize=11)
    axb.grid(alpha=.15, axis="x", lw=.5)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png, counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--town", action="append", default=None)
    ap.add_argument("--out-dir", default=os.path.join(PKG, "reports", "collection"))
    ap.add_argument("--speed-ms", type=float, default=8.0)
    ap.add_argument("--cover-m", type=float, default=COVER_M)
    ap.add_argument("--shortlist", type=int, default=8)
    ap.add_argument("--rare-repeats", type=int, default=6,
                    help="extra alternating-direction traversals of passage / "
                         "along_edge / open_space regions; 0 disables")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    for town in (a.town or ["Town01", "Town07"]):
        xodr = os.path.join(PKG, "config", f"{town}.xodr")
        npz = os.path.join(PKG, "config", f"regions.{town.lower()}.approach.npz")
        if not os.path.exists(npz):
            print(f"missing {npz} — map_regions.py --approach-m 15 --suffix .approach")
            return 1
        table = load_region_table(npz)
        rm = segment(load(xodr), step=2.0, approach_m=15.0)
        pts, legs, uncovered = plan(town, xodr, table, rm,
                                    shortlist=a.shortlist, cover_m=a.cover_m,
                                    repeats=a.rare_repeats)
        if not pts:
            print(f"{town}: no drivable route")
            continue
        ln = _len(pts)
        mv = maneuvers(pts)
        # 10 Hz x 2 LiDAR x ~16k points x 16-20 B, the observed rate of the two-LiDAR
        # collection rig without cameras.
        secs = ln / a.speed_ms
        stats = {
            "town": town, "regions_total": len(rm.regions),
            "covered": len(rm.regions) - len(uncovered),
            "coverage_pct": 100.0 * (len(rm.regions) - len(uncovered))
            / len(rm.regions),
            "legs": len(legs), "length_m": round(ln, 1),
            "minutes": secs / 60.0, "bag_gb": secs * 7e6 / 1e9,
            "maneuvers": mv, "uncovered": sorted(uncovered),
            "uncovered_labels": sorted({rm.regions[r].label for r in uncovered}),
        }
        rows = predict_phases(pts, table, speed_ms=a.speed_ms)
        ppng, pcounts = plot_phases(
            rm, rows, os.path.join(a.out_dir, f"phases.{town.lower()}.png"), town)
        stats["predicted_frames"] = dict(pcounts)
        png = plot(rm, pts, os.path.join(a.out_dir, f"drivable.{town.lower()}.png"),
                   town, stats, uncovered)
        # THE DRIVEN POLYLINE ITSELF, in the CARLA frame, ~2 m spacing.
        # `drive_collection.py` must follow THIS, not re-trace centroid -> centroid.
        # Re-tracing looks equivalent and is not: a centroid snaps to whichever lane is
        # nearest, which can be the opposite carriageway, and the plan then loops the
        # block to reach it. The re-traced route can be several times longer, and the
        # vehicle spends the drive U-turning and recording mostly-stationary scans.
        path = [[round(x, 2), round(-y, 2), o] for x, y, o in pts]
        with open(os.path.join(a.out_dir, f"drivable.{town.lower()}.json"), "w") as f:
            json.dump({"stats": stats, "legs": legs, "path": path,
                       "targets": [{"rid": lg["target_rid"], "label": lg["label"],
                                    "planar": [round(rm.regions[lg["target_rid"]]
                                                     .centroid[0], 2),
                                               round(rm.regions[lg["target_rid"]]
                                                     .centroid[1], 2)],
                                    "carla": [round(rm.regions[lg["target_rid"]]
                                                    .centroid[0], 2),
                                              round(-rm.regions[lg["target_rid"]]
                                                    .centroid[1], 2)]}
                                   for lg in legs],
                       "note": "planar = OpenDRIVE/ROS (+y north). carla = CARLA "
                               "Python API (+y south). Bridge odometry is PLANAR."},
                      f, indent=2)
        print(f"\n{town}")
        print(f"  coverage    {stats['covered']}/{stats['regions_total']} "
              f"({stats['coverage_pct']:.0f}%)"
              + (f"  MISSING {stats['uncovered_labels']}" if uncovered else ""))
        print(f"  route       {len(legs)} legs, {ln / 1000:.2f} km, "
              f"~{stats['minutes']:.0f} min at {a.speed_ms:.0f} m/s")
        print(f"  maneuvers   {dict(sorted(mv.items()))}")
        print(f"  bag         ~{stats['bag_gb']:.0f} GB at 10 Hz x 2 LiDAR")
        tot = sum(pcounts.values())
        print(f"  phases      {tot} frames at 10 Hz: " + ", ".join(
            f"{k} {v} ({100 * v / tot:.0f}%)"
            for k, v in sorted(pcounts.items(), key=lambda kv: -kv[1])))
        print(f"  {png}")
        print(f"  {ppng}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
