#!/usr/bin/env python3
"""Plot the trajectories the vehicle ACTUALLY drove in CARLA, against the map.

Every other figure in this package comes from the offline simulation. This one reads the
per-tick JSONL the MPC writes during a real run — real odometry, real drivetrain, real
ground-truth clusters — so it is the only picture that is evidence the car drove.

Reads the newest log per mission from the MPC's debug_logs directory. Runs are identified
by `plan_name`, which the node records in every row.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import textwrap

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(PKG)
sys.path.insert(0, PKG)

LOGS = os.path.join(WS, "dgppo_ros_node_pkg", "dgppo_ros_node_pkg", "debug_logs")

# Categorical slots 1 and 2 of the reference palette (validated adjacent pair); identity
# is carried by dash-vs-solid too, so the routes separate without colour.
DESIRED, ACTUAL = "#2a78d6", "#eb6834"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#b9b8b2", "#fcfcfb"


def load_runs(pattern: str = "carla_mpc_*.jsonl") -> dict:
    """Newest run per plan_name, as {plan_name: [rows]}."""
    out: dict[str, list] = {}
    for path in sorted(glob.glob(os.path.join(LOGS, pattern))):
        rows = []
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("steer_to"):
                    rows.append(r)
        if not rows:
            continue
        name = rows[0].get("plan_name")
        if not name:
            # Logs from before plan_name was recorded. Skipped rather than guessed at:
            # a run identified by eye from its region sequence is how a figure ends up
            # mislabelled, and these are also root-owned (the container runs as root
            # against the mounted tree) so they cannot simply be cleared.
            continue
        out[name] = rows              # sorted() means later files win
    return out


#: Regions holding a cone, per plan name, to draw on the map. Empty by default; pass
#: --cones NAME=r1,r2 to draw them.
CONES: dict[str, tuple[int, ...]] = {}


def prompt_for(name: str) -> str:
    """The English instruction this run came from, read from its brain tree."""
    for path in glob.glob(os.path.join(PKG, "config", "mission.*.json")):
        try:
            tree = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        if tree.get("plan_name") == name:
            return tree.get("description", "")
    return ""


def speeds(rows: list) -> tuple[float, float]:
    """(commanded, actual) mean speed in m/s, using SIMULATED time.

    Wall time is wrong by the sim's speed-up factor (~10x here), which makes a
    well-tracked 5 m/s command look like 50 m/s. Falls back to wall time on older logs,
    and says so by returning a negative actual.
    """
    key = "t_sim" if "t_sim" in rows[0] else "t"
    tot_d = tot_t = 0.0
    for i in range(1, len(rows)):
        dt = rows[i][key] - rows[i - 1][key]
        if dt <= 0:
            continue
        tot_d += math.dist((rows[i - 1]["x"], rows[i - 1]["y"]),
                           (rows[i]["x"], rows[i]["y"]))
        tot_t += dt
    cmd = sum(r["v_cmd"] for r in rows) / len(rows)
    act = (tot_d / tot_t) if tot_t > 0 else 0.0
    return cmd, (act if key == "t_sim" else -act)


def summarise(rows: list) -> dict:
    regs, prev = [], None
    for r in rows:
        if r["gt_cluster"] != prev:
            regs.append(r["gt_cluster"])
            prev = r["gt_cluster"]
    dist = sum(math.dist((rows[i]["x"], rows[i]["y"]), (rows[i + 1]["x"], rows[i + 1]["y"]))
               for i in range(len(rows) - 1))
    yaws = [r["yaw_deg"] for r in rows]
    return {"regions": regs, "path_m": dist, "ticks": len(rows),
            "yaw_from": yaws[0], "yaw_to": yaws[-1],
            "complete": any(r.get("brain_state") == "COMPLETE" for r in rows)
                        or rows[-1].get("brain_state") == "COMPLETE"}


#: Metres from a road's reference line that count as drivable. Must match the MPC's
#: MpcConfig.road_half_width — this is the band the penalty holds the vehicle inside.
HALF_WIDTH = 7.0


def draw(ax, name: str, rows: list, table) -> None:
    import numpy as np

    # The DRIVABLE BAND: every reference-line sample fattened to the carriageway
    # half-width. This is the only thing the vehicle is constrained by — it is derived
    # from the MAP, not sensed. Nothing in Phase A perceives a wall, a building or a cone.
    wp_all = table.waypoints
    ax.scatter(wp_all[:, 0], wp_all[:, 1], s=HALF_WIDTH ** 2 * 3.2, c="#dfeaf7",
               alpha=.75, edgecolors="none", zorder=0)

    wp = table.waypoints
    for rid in table.region_ids:
        m = wp[:, 2].astype(int) == rid
        ax.scatter(wp[m, 0], wp[m, 1], s=6, c=MUTED, alpha=.55, edgecolors="none", zorder=1)

    s = summarise(rows)
    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    ax.plot(xs, ys, lw=2, color=ACTUAL, zorder=4, solid_capstyle="round")
    ax.scatter([xs[0]], [ys[0]], s=55, marker="o", facecolor=SURFACE,
               edgecolor=ACTUAL, lw=2, zorder=6)
    ax.scatter([xs[-1]], [ys[-1]], s=120, marker="*", color=ACTUAL,
               edgecolor=SURFACE, lw=.8, zorder=6)

    # where the steering target changed = where the plan advanced a step
    prev_t = None
    for r in rows:
        if r["steer_to"] != prev_t:
            if prev_t is not None:
                ax.scatter([r["x"]], [r["y"]], s=46, marker="D", facecolor=SURFACE,
                           edgecolor=ACTUAL, lw=1.6, zorder=7)
            prev_t = r["steer_to"]

    for rid in table.region_ids:
        cx, cy = table.centroid_of(rid)
        on = rid in s["regions"]
        junction = table.label_of(rid) == "junction"
        ax.annotate(str(rid), (cx, cy), fontsize=8.5 if on else 7,
                    fontweight="bold" if on else "normal",
                    color=INK if on else INK2, ha="center", va="center", zorder=8,
                    bbox=dict(boxstyle="circle,pad=0.28" if junction else "round,pad=0.22",
                              fc=SURFACE, ec=INK2 if on else MUTED,
                              lw=1.2 if on else .6, alpha=.96))

    # cones, where the mission put them — drawn so the reader can see what the cue was
    # actually responding to, rather than having to take the region ids on trust.
    for rid in CONES.get(name, ()):
        cx, cy = table.centroid_of(rid)
        ax.scatter([cx], [cy], s=190, marker="^", color="#eda100",
                   edgecolor=INK, lw=.8, zorder=5)
        ax.annotate("cone", (cx, cy), xytext=(0, -15), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK2, zorder=9)

    # mark every pose that left the band, so "it stayed on the road" is visible rather
    # than asserted
    import numpy as np
    off = [(r["x"], r["y"], table.nearest(r["x"], r["y"])[1]) for r in rows]
    strayed = [(x, y, d) for x, y, d in off if d > HALF_WIDTH]
    if strayed:
        ax.scatter([p[0] for p in strayed], [p[1] for p in strayed], s=34,
                   facecolor="none", edgecolor="#e34948", lw=1.6, zorder=8)
    max_off = max(d for _, _, d in off)
    has_term = "offroad_frac" in rows[0]

    cmd, act = speeds(rows)
    speed = (f"{cmd:.1f} m/s commanded, {act:.1f} actual" if act >= 0
             else f"{cmd:.1f} m/s commanded (no sim clock in this log)")
    # Wrapped, because these are English sentences and an unwrapped title runs straight
    # across the neighbouring panel.
    quoted = "\n".join(textwrap.wrap(f'"{prompt_for(name)}"', 52)) or name
    ax.set_title(quoted + "\n"
                 + f"regions {' → '.join(map(str, s['regions']))}  ·  "
                   f"{s['path_m']:.0f} m  ·  {s['ticks']} ticks\n"
                 + f"{speed}\n"
                 + ("ROAD TERM ON " if has_term else "road term OFF")
                 + f" · max {max_off:.1f} m off centreline"
                 + (f", {len(strayed)} poses outside the band" if strayed
                    else ", never left the band"),
                 fontsize=8.5, color=INK, loc="left", family="monospace")
    ax.set_aspect("equal")
    ax.grid(alpha=.12, lw=.5)
    ax.tick_params(labelsize=7, colors=INK2)
    for sp in ax.spines.values():
        sp.set_edgecolor(MUTED)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PKG, "config", "carla_runs.png"))
    ap.add_argument("--cones", action="append", default=[], metavar="NAME=r1,r2",
                    help="draw cones for plan NAME at these regions (repeatable)")
    args = ap.parse_args()
    for spec in args.cones:
        name, _, regs = spec.partition("=")
        CONES[name] = tuple(int(r) for r in regs.split(",") if r.strip())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from carla_gt_bridge.region_lookup import load_region_table

    runs = load_runs()
    if not runs:
        raise SystemExit(f"no runs with a plan_name in {LOGS}")
    table = load_region_table(os.path.join(PKG, "config", "regions.town05.npz"))

    n = len(runs)
    cols = 2 if n > 1 else 1
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(7.6 * cols, 6.8 * rows_n), dpi=130,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, rows) in zip(axes.flat, sorted(runs.items())):
        ax.set_facecolor(SURFACE)
        draw(ax, name, rows, table)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")

    fig.suptitle("Town05 — trajectories driven IN CARLA\n"
                 "real odometry, real drivetrain, ground-truth clusters",
                 fontsize=12, color=INK, y=.995)
    fig.text(.5, .042,
             "○ start    ◇ plan step advanced    ★ end    ▲ cone    "
             "◌ pose outside the drivable band    ·    circled ids are junctions",
             ha="center", fontsize=8.5, color=INK2)
    fig.text(.5, .004,
             "Blue band = drivable surface, from the MAP (reference lines ± 7 m), NOT "
             "from any sensor.\nNothing here perceives walls, buildings or cones. The MPC "
             "is penalised for leaving the band — a proxy for not hitting what lies "
             "beyond it.\nReal obstacle sensing arrives with the LiDAR in Phase B.",
             ha="center", fontsize=8, color=INK2, linespacing=1.5)
    fig.tight_layout(rect=(0, .058, 1, .955), h_pad=3.2, w_pad=2.0)
    fig.savefig(args.out, facecolor=SURFACE)
    print(args.out)
    for name, r in sorted(runs.items()):
        s = summarise(r)
        print(f"  {name:52s} {s['regions']}  {s['path_m']:6.0f} m  {s['ticks']:5d} ticks")


if __name__ == "__main__":
    main()
