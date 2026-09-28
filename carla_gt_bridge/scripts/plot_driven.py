#!/usr/bin/env python3
"""Draw what the vehicle ACTUALLY did, so a person can check it rather than trust a score.

WHY THIS EXISTS. Scores are summaries ("discriminates", "COMPLETED", "branch 'right'")
and can be wrong in ways no summary reveals -- e.g. a pair metric that asks whether two
runs differed, not whether either was right, passes runs that drove the mission
backwards. A picture of the path cannot hide that.

WHAT IS DRAWN
  grey dots     the town's road surface, from its own waypoints
  amber rings   junction region centroids, labelled with the region id
  the track     coloured by PLAN STEP, so you can see where the executor advanced
  green dot     start        red square   end
  dashed line   the straight line start->end, for reading how much turning happened
  cone marker   where a cone was placed, if the run had one

Reading it: the question is not "did it move" but "did it turn where the mission said".
For a matched pair, open both and compare -- if the two tracks are the same shape, the
world made no difference whatever the score says.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, PAD = 760, 560, 24


def load(path: str) -> list[dict]:
    out = []
    for line in open(path, errors="ignore"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def regions(town: str):
    f = os.path.join(PKG, "config", f"regions.{town}.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    return (d["waypoints"][:, :2], [int(x) for x in d["rids"]],
            [str(x) for x in d["labels"]], d["centroids"])


def svg(rows: list[dict], town: str, title: str, cone_xy=None) -> str:
    xy = np.array([[r["x"], r["y"]] for r in rows if "x" in r and "y" in r])
    if len(xy) < 2:
        return f'<div class="miss">{title}: no pose data</div>'
    reg = regions(town)
    allpts = np.vstack([xy, reg[0]]) if reg else xy
    lo, hi = allpts.min(0), allpts.max(0)
    span = np.maximum(hi - lo, 1e-6)
    k = min((W - 2 * PAD) / span[0], (H - 2 * PAD) / span[1])
    off = np.array([PAD, PAD]) + (np.array([W, H]) - 2 * PAD - span * k) / 2

    def T(p):
        q = (np.asarray(p, dtype=float) - lo) * k + off
        return float(q[0]), float(H - q[1])          # CARLA is y-up

    parts = []
    if reg:
        wp, rids, labs, cents = reg
        parts.append('<g class="road">' + "".join(
            f'<circle cx="{T(p)[0]:.1f}" cy="{T(p)[1]:.1f}" r="0.7"/>' for p in wp[::5]) + "</g>")
        js = [(r, c) for r, l, c in zip(rids, labs, cents) if l == "junction"]
        parts.append('<g class="junc">' + "".join(
            f'<circle cx="{T(c)[0]:.1f}" cy="{T(c)[1]:.1f}" r="5"/>' for _, c in js) + "</g>")
        parts.append('<g class="jid">' + "".join(
            f'<text x="{T(c)[0]:.1f}" y="{T(c)[1]-8:.1f}">{r}</text>' for r, c in js) + "</g>")

    # the track, segmented by plan step so advancement is visible
    steps = [r.get("step", 0) or 0 for r in rows if "x" in r]
    segs, cur, last = [], [], steps[0] if steps else 0
    for r, s in zip([r for r in rows if "x" in r], steps):
        if s != last and cur:
            segs.append((last, cur)); cur = [cur[-1]]; last = s
        cur.append((r["x"], r["y"]))
    if cur: segs.append((last, cur))
    for si, pts in segs:
        d = " ".join(("M" if i == 0 else "L") + f"{T(p)[0]:.1f},{T(p)[1]:.1f}"
                     for i, p in enumerate(pts))
        parts.append(f'<path class="tk s{si % 6}" d="{d}"/>')

    a, b = T(xy[0]), T(xy[-1])
    parts.append(f'<line class="chord" x1="{a[0]:.1f}" y1="{a[1]:.1f}" '
                 f'x2="{b[0]:.1f}" y2="{b[1]:.1f}"/>')
    parts.append(f'<circle class="start" cx="{a[0]:.1f}" cy="{a[1]:.1f}" r="5"/>')
    parts.append(f'<rect class="end" x="{b[0]-4:.1f}" y="{b[1]-4:.1f}" width="8" height="8"/>')
    if cone_xy is not None:
        c = T(cone_xy)
        parts.append(f'<path class="cone" d="M{c[0]:.1f},{c[1]-6:.1f} L{c[0]+5:.1f},'
                     f'{c[1]+4:.1f} L{c[0]-5:.1f},{c[1]+4:.1f} Z"/>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">'
            + "".join(parts) + "</svg>")


def summarise(rows: list[dict]) -> dict:
    seq, last = [], None
    for r in rows:
        c = r.get("gt_cluster")
        if c is not None and c != last:
            seq.append(int(c)); last = c
    yaws = [r["yaw_deg"] for r in rows if r.get("yaw_deg") is not None]
    xy = [(r["x"], r["y"]) for r in rows if "x" in r]
    dist = sum(float(np.hypot(b[0]-a[0], b[1]-a[1])) for a, b in zip(xy, xy[1:]))
    return {"regions": seq, "ticks": len(rows), "metres": round(dist, 1),
            "net_turn_deg": round((yaws[-1] - yaws[0] + 180) % 360 - 180, 1) if len(yaws) > 1 else None,
            "steps": max((r.get("step", 0) or 0) for r in rows) + 1 if rows else 0}


def match_log(run_dir: str, logs: list[str]) -> str | None:
    """The debug log this run produced.

    Matched on mtime against the run's own out.txt, taking the LAST log that finished
    while the run was alive. Nearest-in-time is not enough: several logs fall inside any
    200 s window when runs are closely spaced, and picking the wrong one silently
    attributes another mission's trajectory to this row, producing plausible-looking
    region sequences for void runs.
    """
    out = os.path.join(run_dir, "out.txt")
    if not os.path.exists(out):
        return None
    t_end = os.path.getmtime(out)
    # the log must have been last written DURING the run: after it started, at or before
    # it ended. `run-seconds` plus bring-up bounds the start.
    cands = [(os.path.getmtime(l), l) for l in logs
             if t_end - 600 <= os.path.getmtime(l) <= t_end + 30]
    if not cands:
        return None
    return max(cands)[1]


def render(cards, out: str, sweep: str) -> None:
    import html as H
    body = []
    for name, sv, st, mission in cards:
        if sv is None:
            body.append(f'<div class="card bad"><h3>{H.escape(name)}</h3>'
                        f'<p class="miss">{H.escape(st.get("note",""))}</p></div>')
            continue
        seq = " &rarr; ".join(str(x) for x in st["regions"][:12])
        badge = "ok" if st.get("completed") else "no"
        body.append(
            f'<div class="card"><h3>{H.escape(name)}'
            f'<span class="pill {badge}">{"completed" if st.get("completed") else "did not finish"}</span></h3>'
            f'<p class="mi">{H.escape(mission[:150])}</p>'
            f'<div class="fig">{sv}</div>'
            f'<div class="st"><span>regions <b>{seq or "—"}</b></span>'
            f'<span>steps <b>{st["steps"]}</b></span>'
            f'<span>distance <b>{st["metres"]} m</b></span>'
            f'<span>net turn <b>{st["net_turn_deg"]}&deg;</b></span></div></div>')
    page = f"""<title>Driven Paths</title>
<style>
:root{{--bg:#F4F3F0;--card:#FFF;--ink:#191816;--ink2:#4A4741;--ink3:#7B7871;--rule:#DDDAD3;
 --acc:#7C4A2D;--ok:#2F6B4F;--no:#9B3226;--road:#CFCCC5;--junc:#C89A3C;
 --mono:ui-monospace,Menlo,monospace;--sans:system-ui,sans-serif;--serif:ui-serif,Georgia,serif;}}
@media(prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#111110;--card:#1A1917;
 --ink:#ECEAE5;--ink2:#B4B0A8;--ink3:#85817A;--rule:#2C2A26;--acc:#D9A17C;--ok:#6FCF9A;
 --no:#F08B7C;--road:#31302C;--junc:#D9AE55;}}}}
:root[data-theme="dark"]{{--bg:#111110;--card:#1A1917;--ink:#ECEAE5;--ink2:#B4B0A8;--ink3:#85817A;
 --rule:#2C2A26;--acc:#D9A17C;--ok:#6FCF9A;--no:#F08B7C;--road:#31302C;--junc:#D9AE55;}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--ink);font-family:var(--sans);margin:0;line-height:1.5}}
.w{{max-width:86rem;margin:0 auto;padding:2.6rem 1.3rem 5rem;display:flex;flex-direction:column;gap:1.4rem}}
h1{{font-family:var(--serif);font-size:2.2rem;margin:0 0 .3rem;font-weight:600}}
.lede{{font-family:var(--serif);color:var(--ink2);margin:0 0 .3rem;max-width:64ch}}
.key{{display:flex;gap:1.3rem;flex-wrap:wrap;font-family:var(--mono);font-size:.7rem;
 color:var(--ink3);background:var(--card);border:1px solid var(--rule);border-radius:6px;padding:.7rem 1rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(30rem,1fr));gap:1rem}}
.card{{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:.9rem 1rem}}
.card.bad{{opacity:.55}}
h3{{font-family:var(--mono);font-size:.79rem;margin:0 0 .25rem;display:flex;gap:.6rem;align-items:center}}
.pill{{font-size:.58rem;padding:.06rem .4rem;border-radius:99px;border:1px solid;margin-left:auto}}
.pill.ok{{color:var(--ok);border-color:var(--ok)}} .pill.no{{color:var(--no);border-color:var(--no)}}
.mi{{font-size:.76rem;color:var(--ink3);margin:0 0 .5rem;font-style:italic}}
.fig{{background:var(--bg);border-radius:5px}} svg{{width:100%;height:auto;display:block}}
.st{{display:flex;gap:1rem;flex-wrap:wrap;font-family:var(--mono);font-size:.67rem;
 color:var(--ink3);margin-top:.5rem}}
.st b{{color:var(--ink2)}}
.miss{{font-family:var(--mono);font-size:.72rem;color:var(--no)}}
.road circle{{fill:var(--road)}}
.junc circle{{fill:none;stroke:var(--junc);stroke-width:1.3}}
.jid text{{font:600 8px var(--mono);fill:var(--ink3);text-anchor:middle}}
.tk{{fill:none;stroke-width:2.4;stroke-linecap:round;stroke-linejoin:round}}
.tk.s0{{stroke:#4C7FB8}} .tk.s1{{stroke:#5AA469}} .tk.s2{{stroke:#C87F3C}}
.tk.s3{{stroke:#9B6BB0}} .tk.s4{{stroke:#C05C6E}} .tk.s5{{stroke:#3F9A9A}}
.chord{{stroke:var(--ink3);stroke-width:1;stroke-dasharray:4 4;opacity:.5}}
.start{{fill:var(--ok)}} .end{{fill:var(--no)}} .cone{{fill:var(--junc)}}
</style>
<div class="w">
<h1>Driven paths &mdash; {H.escape(sweep)}</h1>
<p class="lede">What the vehicle actually did, drawn on the map, so a route can be
checked by eye rather than trusted from a score.</p>
<div class="key">
  <span>grey = road surface</span><span style="color:var(--junc)">amber ring = junction, labelled</span>
  <span style="color:var(--ok)">green dot = start</span><span style="color:var(--no)">red square = end</span>
  <span>track colour changes at each plan step</span><span>dashed = straight line start&rarr;end</span>
</div>
<div class="grid">{"".join(body)}</div>
</div>"""
    open(out, "w").write(page)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="grid", help="directory under reports/runs")
    ap.add_argument("--town", default="town05")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = os.path.join(PKG, "reports", "runs", a.sweep)
    logs = sorted(glob.glob(os.path.join(
        os.path.dirname(PKG), "dgppo_ros_node_pkg", "dgppo_ros_node_pkg",
        "debug_logs", "carla_mpc_*.jsonl")), key=os.path.getmtime)
    cards = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        own = os.path.join(d, "run.jsonl")
        src = own if os.path.exists(own) else match_log(d, logs)
        name = os.path.basename(d)
        if not src:
            cards.append((name, None, {"note": "no trajectory log found"}, ""))
            continue
        rows = load(src)
        if len(rows) < 5:
            cards.append((name, None, {"note": f"only {len(rows)} ticks"}, ""))
            continue
        mission = ""
        o = os.path.join(d, "out.txt")
        if os.path.exists(o):
            m = re.match(r'\s*mission\s*:\s*"(.+?)"', open(o, errors="ignore").readline())
            if m:
                mission = m.group(1)
        completed = os.path.exists(o) and "COMPLETED" in open(o, errors="ignore").read()
        s = summarise(rows)
        s["completed"] = completed
        s["log"] = os.path.basename(src)
        cards.append((name, svg(rows, a.town, name), s, mission))

    out = a.out or os.path.join(PKG, "reports", f"paths_{a.sweep}.html")
    render(cards, out, a.sweep)
    ok = sum(1 for _, sv, _, _ in cards if sv)
    print(f"  {ok}/{len(cards)} runs plotted -> {out}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
