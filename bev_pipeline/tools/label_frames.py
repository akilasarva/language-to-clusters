#!/usr/bin/env python3
"""Interactive per-frame navigation-mode labeler (primary ground truth).

Shows each frame's camera image + BEV side-by-side with the auto geometric
label's SUGGESTED nav mode pre-selected; you confirm or correct with a single
keystroke. Saves resumably to ``datasets/<env>/labels_human.npy`` (+ meta).

Vocabulary comes from config/nav_modes.yaml (edit that to change the target).
Needs a display (run locally / with X-forwarding). Everything autosaves on each
label, so partial labeling is safe and resumable.

Keys:
  1-9        assign that nav mode (see on-screen legend)
  <space>    accept the suggested mode + advance
  <right>/n  next frame     <left>/p  previous frame
  <backspace> clear this frame's label
  s          save now (also autosaves each label)
  q          save + quit

Usage:
  python3 tools/label_frames.py --env-dir datasets/full_campus_1hz \
      [--stride 1] [--modes config/nav_modes.yaml]
"""
import argparse
import json
import os
import sys
import textwrap

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)


def load_vocab(path):
    import yaml
    cfg = yaml.safe_load(open(path))
    modes = cfg["modes"]
    names = [m["name"] for m in modes]
    key_to_idx = {int(m["key"]): i for i, m in enumerate(modes)}
    return cfg, names, key_to_idx


def _bev_falsecolor(bev):
    def n(c):
        c = c.astype(float)
        return (c - c.min()) / (np.ptp(c) + 1e-9)
    return np.stack([n(bev[0]), n(bev[2]), n(bev[3])], -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("--modes", default=os.path.join(PKG, "config", "nav_modes.yaml"))
    ap.add_argument("--stride", type=int, default=1, help="label every Nth frame")
    ap.add_argument("--review-frames", default=None,
                    help="path to .npy list of frame indices to re-review ONLY "
                         "(overrides --stride); optional sibling _reasons.json shows why")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from PIL import Image

    cfg, names, key_to_idx = load_vocab(args.modes)
    dm = json.load(open(os.path.join(args.env_dir, "dataset_meta.json")))
    image_paths = dm["image_paths"]
    n_frames = len(image_paths)
    # geometry: the polar free-space profile + clearances (interpretable cues)
    gpath = os.path.join(args.env_dir, "modality_geom.npy")
    geom = np.load(gpath) if os.path.exists(gpath) else None
    MR, NS = 20.0, 36                       # max_range (m), n_sectors (must match GeomParams)

    # geometric auto-labels -> suggested nav mode
    geom_names = dm.get("geom_label_names", [])
    geom_labels = (np.load(os.path.join(args.env_dir, "labels_geom.npy"))
                   if os.path.exists(os.path.join(args.env_dir, "labels_geom.npy")) else None)
    sugg_map = cfg.get("geom_suggestion", {})

    # Optional trained-RandomForest predictions + confidence (leave-one-bag-out), used
    # as the suggestion when present. reports/rf_preds_<env>.json =
    # {idx: [class_name, confidence]}.
    rf_pred = {}
    env_base = os.path.basename(os.path.normpath(args.env_dir))
    _rp = os.path.join(PKG, "reports", f"rf_preds_{env_base}.json")
    if os.path.exists(_rp):
        for k, v in json.load(open(_rp)).items():
            if v[0] in names:
                rf_pred[int(k)] = (names.index(v[0]), float(v[1]))

    def geom_suggestion(i):
        if geom_labels is None:
            return 0
        gname = geom_names[geom_labels[i]]
        return names.index(sugg_map.get(gname, names[0])) if sugg_map.get(gname) in names else 0

    def suggestion(i):
        # the trained-RF prediction (with confidence) if available, else geom auto-label
        return rf_pred[i][0] if i in rf_pred else geom_suggestion(i)

    # resumable labels
    lbl_path = os.path.join(args.env_dir, "labels_human.npy")
    meta_path = os.path.join(args.env_dir, "labels_human_meta.json")
    labels = (np.load(lbl_path) if os.path.exists(lbl_path)
              else np.full(n_frames, -1, dtype=np.int64))
    if len(labels) != n_frames:
        labels = np.full(n_frames, -1, dtype=np.int64)

    def _has_img(i):
        rel = image_paths[i]
        return bool(rel) and os.path.exists(os.path.join(args.env_dir, rel))

    # only present frames that HAVE a camera image (can't judge the mode without it)
    reasons = {}
    if args.review_frames and os.path.exists(args.review_frames):
        # focused re-review: only the flagged candidate frames
        review = [int(i) for i in np.load(args.review_frames)]
        frames = [i for i in review if _has_img(i)] or review
        rpath = args.review_frames.replace(".npy", "_reasons.json")
        if os.path.exists(rpath):
            reasons = {int(k): v for k, v in json.load(open(rpath)).items()}
        start = 0
    else:
        frames = [i for i in range(0, n_frames, args.stride) if _has_img(i)]
        if not frames:
            frames = list(range(0, n_frames, args.stride))
        start = next((k for k, f in enumerate(frames) if labels[f] < 0), 0)
    state = {"pos": start}

    def save():
        np.save(lbl_path, labels)
        json.dump({"env": dm["env"], "mode_names": names,
                   "n_labeled": int((labels >= 0).sum()), "n_frames": n_frames,
                   "stride": args.stride}, open(meta_path, "w"), indent=2)

    # structure top-down: SAME polar_structure render the VLM receives (so the
    # human labels the identical representation). Pose-fallback mirrors the VLM
    # pipeline: polar_structure when odometry exists, single-frame RANSAC otherwise.
    from bev_pipeline.ground_removal import (plane_level_and_remove, keep_macro_structure,
                                             polar_structure)
    from bev_pipeline.ground_bev import gravity_ground, _accumulate
    from scipy.ndimage import gaussian_filter
    VIEW_R = 14.0                                   # match the VLM panel radius (R=14)
    _macro_cache = {}
    _poses = (np.load(os.path.join(args.env_dir, "poses.npy"))
              if os.path.exists(os.path.join(args.env_dir, "poses.npy")) else None)
    _gg_cache = {}

    def _gground(k):
        if k in _gg_cache:
            return _gg_cache[k]
        fp = os.path.join(args.env_dir, "points", f"frame_{k:05d}.npy")
        out = None
        if os.path.exists(fp) and _poses is not None:
            out = gravity_ground(np.load(fp), _poses[k, 3:7])
        _gg_cache[k] = out
        return out

    def _ground_panel(i, R=14.0, CELL=0.25, W=4):
        """Contrast-preserved ground-intensity top-down (bright=paved walkway,
        dark=grass): the walkway graph used to read path/junction/open. Uses
        multi-frame accumulation when odometry is present; falls back to a
        single-frame RANSAC ground (ringy/sparse) when poses are missing."""
        G = int(round(2 * R / CELL))
        pose_ok = (_poses is not None and np.all(np.isfinite(_poses[i])))
        if pose_ok:
            lst = [None] * len(_poses)
            for k in range(max(0, i - W), min(len(_poses), i + W + 1)):
                lst[k] = _gground(k)
            acc = _accumulate(lst, _poses, i, W, R)
        else:                                          # no odometry -> single frame
            fp = os.path.join(args.env_dir, "points", f"frame_{i:05d}.npy")
            if not os.path.exists(fp):
                return None
            p = np.load(fp); m = (np.abs(p[:, 0]) < R) & (np.abs(p[:, 1]) < R)
            _, acc = plane_level_and_remove(p[m], dist_thresh=0.25)
        if acc is None or len(acc) < 50:
            return None
        ix = np.clip(((acc[:, 1] + R) / CELL).astype(int), 0, G - 1)
        iy = np.clip(((acc[:, 0] + R) / CELL).astype(int), 0, G - 1)
        s = np.zeros((G, G)); c = np.zeros((G, G))
        np.add.at(s, (iy, ix), acc[:, 3]); np.add.at(c, (iy, ix), 1)
        m = c > 0; img = np.zeros((G, G))
        img[m] = s[m] / c[m]
        if m.sum() < 50:
            return None
        lo, hi = np.percentile(img[m], [5, 95])
        img = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1); img[~m] = 0
        return gaussian_filter(img, 0.6)

    def _macro_xy(i):
        """Top-down macro structure (walls/buildings, people/follower removed).

        Loads the frame's raw non-ground cloud, re-levels + RANSAC-removes the
        residual ground swirl, then drops small isolated blobs (pedestrians, a
        follower walking behind). Returns (M,3) xyz in the robocentric frame
        (+x forward, +y left) or None. Cached — computed once per frame.
        """
        if i in _macro_cache:
            return _macro_cache[i]
        fp = os.path.join(args.env_dir, "points", f"frame_{i:05d}.npy")
        out = None
        if os.path.exists(fp):
            p = np.load(fp)
            pose_ok = (_poses is not None and np.all(np.isfinite(_poses[i])))
            if pose_ok:                                # VLM path: tilt-robust polar structure
                out = polar_structure(p, _poses[i, 3:7], R=VIEW_R)
            else:                                      # no odometry -> single-frame fallback
                m = (np.abs(p[:, 0]) < VIEW_R) & (np.abs(p[:, 1]) < VIEW_R)
                p = p[m]
                if len(p) > 200:
                    ng, _ = plane_level_and_remove(p, dist_thresh=0.25)
                    out = keep_macro_structure(ng, eps=0.7, min_points=6,
                                               min_extent=1.5, min_count=60)
        _macro_cache[i] = out
        return out

    fig = plt.figure(figsize=(15, 7.8))
    axc = fig.add_axes([0.02, 0.26, 0.56, 0.66])    # camera (large)
    axg = fig.add_axes([0.62, 0.54, 0.36, 0.40])    # structure top-down
    axg2 = fig.add_axes([0.62, 0.08, 0.36, 0.40])   # ground walkway-graph
    # persistent, WRAPPED reason box under the camera (why this frame was flagged)
    reason_artist = fig.text(0.02, 0.235, "", ha="left", va="top", fontsize=10,
                             color="#7a2500", family="monospace",
                             bbox=dict(boxstyle="round,pad=0.5", fc="#fff3d6", ec="#d99a00"))
    cue_artist = fig.text(0.02, 0.105, "", ha="left", va="top", fontsize=9,
                          color="#111", family="monospace")
    legend = "   ".join(f"[{m['key']}] {m['name']}" for m in cfg["modes"])
    order = cfg.get("decision_order", [])
    rule = "decision order (first match): " + " > ".join(order) if order else ""
    descs = "  |  ".join(f"{m['name']}: {m.get('desc','')}" for m in cfg["modes"])
    # static, one-time (do NOT recreate per frame — that stacks/garbles text)
    fig.suptitle(legend, fontsize=11, y=0.99)
    fig.text(0.5, 0.05, rule, ha="center", fontsize=8, color="#333")
    fig.text(0.5, 0.015, descs, ha="center", fontsize=6.5, color="#666")

    def _geom_cues(i):
        """front/left/right clearance (m), open-direction count, overhead %."""
        if geom is None:
            return None
        g = geom[i]
        return dict(front=g[36] * MR, left=g[37] * MR, right=g[38] * MR,
                    n_open=int(round(g[39] * NS)), overhead=g[40] * 100.0,
                    profile=g[:36] * MR)

    def draw():
        i = frames[state["pos"]]
        axc.clear(); axc.axis("off"); axg.clear(); axg2.clear()
        rel = image_paths[i]
        if rel and os.path.exists(os.path.join(args.env_dir, rel)):
            axc.imshow(Image.open(os.path.join(args.env_dir, rel)))
        macro = _macro_xy(i)
        cues = _geom_cues(i)
        if macro is not None and len(macro):
            # top-down macro structure: robot at center, FORWARD = up. Plot
            # (-y, x) so +x(forward)=up, +y(left)=left. Colour by height.
            axg.scatter(-macro[:, 1], macro[:, 0], s=4, c=macro[:, 2],
                        cmap="plasma", vmin=0.0, vmax=4.0)
            for rr in (5, 10, 15):                       # range rings
                axg.add_patch(plt.Circle((0, 0), rr, fill=False, ec="#ccc", lw=0.6))
            axg.plot(0, 0, marker="^", ms=12, color="lime", mec="k", zorder=5)
            axg.set_xlim(-VIEW_R, VIEW_R); axg.set_ylim(-VIEW_R, VIEW_R)
            axg.set_aspect("equal"); axg.set_xticks([]); axg.set_yticks([])
            axg.text(0, VIEW_R * 0.93, "FORWARD", ha="center", fontsize=8, color="#333")
            axg.text(0, -VIEW_R * 0.97, "BEHIND", ha="center", fontsize=8, color="#333")
            axg.text(-VIEW_R * 0.9, 0, "LEFT", va="center", ha="left", fontsize=8, color="#333")
            axg.text(VIEW_R * 0.9, 0, "RIGHT", va="center", ha="right", fontsize=8, color="#333")
            axg.set_title("macro structure top-down (walls/buildings;\n"
                          "people & follower removed; rings 5/10/15 m)", fontsize=8)
        else:
            axg.set_title("(no macro structure)", fontsize=8)
            axg.set_xticks([]); axg.set_yticks([])
        # ground walkway-graph (bright=paved walk, dark=grass): read path/junction/open
        gimg = _ground_panel(i)
        if gimg is not None:
            axg2.imshow(gimg[:, ::-1], origin="lower", cmap="inferno",
                        extent=[-VIEW_R, VIEW_R, -VIEW_R, VIEW_R])
            axg2.plot(0, 0, marker="^", ms=11, color="#39f", mec="k", zorder=5)
            axg2.text(0, VIEW_R * 0.9, "FORWARD", ha="center", fontsize=8, color="#ddd")
            axg2.text(-VIEW_R * 0.9, 0, "LEFT", va="center", ha="left", fontsize=8, color="#bbb")
            axg2.text(VIEW_R * 0.9, 0, "RIGHT", va="center", ha="right", fontsize=8, color="#bbb")
            for rr in (5, 10):
                axg2.add_patch(plt.Circle((0, 0), rr, fill=False, ec="#888", lw=0.5))
            axg2.set_xlim(-VIEW_R, VIEW_R); axg2.set_ylim(-VIEW_R, VIEW_R)
            axg2.set_aspect("equal"); axg2.set_xticks([]); axg2.set_yticks([])
            axg2.set_title("ground walkway-graph (bright=paved walk, dark=grass;\n"
                           "1 way=path, 3+ crossing=junction, none=open)", fontsize=8)
        else:
            axg2.set_title("(no ground data)", fontsize=8)
            axg2.set_xticks([]); axg2.set_yticks([])
        if cues is not None:
            cue_txt = (f"front {cues['front']:.0f} m   left {cues['left']:.0f} m   "
                       f"right {cues['right']:.0f} m   open-dirs {cues['n_open']}   "
                       f"overhead {cues['overhead']:.0f}%")
        else:
            cue_txt = ""
        cur = labels[i]
        sug = suggestion(i)
        n_done = int((labels[frames] >= 0).sum())
        gtxt = (f"geom-RF: {names[rf_pred[i][0]]} (conf {rf_pred[i][1]:.2f})"
                if i in rf_pred else f"geom: {names[geom_suggestion(i)]}")
        axc.set_title(f"frame {i}  ({state['pos']+1}/{len(frames)})   labeled {n_done}/{len(frames)}"
                      f"    current: {names[cur] if cur>=0 else '—'}   "
                      f"[{gtxt}]  space=accept {names[sug]}", fontsize=10,
                      color=("green" if cur >= 0 else "black"))
        reason_artist.set_text(("⚑ FLAGGED — " + textwrap.fill(reasons[i], 74)) if i in reasons else "")
        cue_artist.set_text(cue_txt)
        fig.canvas.draw_idle()

    def on_key(ev):
        i = frames[state["pos"]]
        if ev.key in ("right", "n"):
            state["pos"] = min(state["pos"] + 1, len(frames) - 1)
        elif ev.key in ("left", "p"):
            state["pos"] = max(state["pos"] - 1, 0)
        elif ev.key == " ":
            labels[i] = suggestion(i); save()
            state["pos"] = min(state["pos"] + 1, len(frames) - 1)
        elif ev.key == "backspace":
            labels[i] = -1; save()
        elif ev.key and ev.key.isdigit() and int(ev.key) in key_to_idx:
            labels[i] = key_to_idx[int(ev.key)]; save()
            state["pos"] = min(state["pos"] + 1, len(frames) - 1)
        elif ev.key == "s":
            save(); print(f"saved ({(labels>=0).sum()} labeled)")
        elif ev.key == "q":
            save(); plt.close(fig); return
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    print(f"Labeling {len(frames)} frames of {dm['env']}. Legend: {legend}")
    print("space=accept suggestion, 1-9=set, arrows=nav, q=save+quit. Autosaves each label.")
    plt.show()
    save()
    print(f"Done. {int((labels>=0).sum())}/{n_frames} labeled -> {lbl_path}")


if __name__ == "__main__":
    main()
