#!/usr/bin/env python3
"""Interactive Open3D tool to draw/tag per-bag landmark bounding boxes.

Loads a cached env's accumulated global map + trajectory, shows it top-down, and lets you place oriented
boxes (center/size/heading via sliders), tag each with a type from
landmark_types.yaml, and save a ``<bag>.landmarks.yaml`` for the trajectory
labeler. Extends the band_tuner.py Open3D GUI pattern.

Requires a display (run locally, not headless).

Usage:
    python3 tools/landmark_bbox_editor.py --env-dir datasets/full_campus_1hz \
        --types config/landmark_types.yaml --out full_campus.landmarks.yaml
"""
import argparse
import math
import os
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)


def build_global_map(env_dir, z_min=1.0, stride=2, max_pts=400000):
    """Accumulate tall structure points into the odom frame + trajectory."""
    import json
    from bev_pipeline.frame_geometry import gravity_align, quat_to_matrix
    from bev_pipeline.ground_removal import plane_level_and_remove

    meta = json.load(open(os.path.join(env_dir, "meta.json")))
    poses = np.load(os.path.join(env_dir, "poses.npy"))
    pts_all, traj = [], []
    for i in range(0, len(meta["frames"]), stride):
        pose = poses[i]
        if not np.all(np.isfinite(pose)):
            continue
        p = np.load(os.path.join(env_dir, "points", f"frame_{i:05d}.npy"))
        lev = gravity_align(p, pose[3:7])
        ng, _ = plane_level_and_remove(lev)
        yaw = math.atan2(quat_to_matrix(pose[3:7])[1, 0], quat_to_matrix(pose[3:7])[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        xy = ng[:, :2] @ R.T + pose[:2]
        keep = ng[:, 2] > z_min
        pts_all.append(np.column_stack([xy[keep], ng[keep, 2]]))
        traj.append(pose[:3])
    pts = np.vstack(pts_all) if pts_all else np.zeros((0, 3))
    if len(pts) > max_pts:
        pts = pts[np.random.RandomState(0).choice(len(pts), max_pts, replace=False)]
    return pts, np.array(traj)


class BboxEditor:
    def __init__(self, pts, traj, types, out_path, bag_name, frame_id):
        import open3d as o3d
        import open3d.visualization.gui as gui
        import open3d.visualization.rendering as rendering
        self.o3d, self.gui, self.rendering = o3d, gui, rendering
        self.pts, self.traj = pts, traj
        self.types = list(types.keys())
        self.out_path, self.bag_name, self.frame_id = out_path, bag_name, frame_id
        self.landmarks = []          # list of dicts
        self.cur = {"cx": 0.0, "cy": 0.0, "length": 10.0, "width": 6.0,
                    "heading": 0.0, "type": self.types[0]}

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("Landmark BBox Editor", 1500, 950)
        w = self.window
        em = w.theme.font_size
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(w.renderer)
        self.scene.scene.set_background([0.05, 0.05, 0.05, 1.0])
        self._mat = rendering.MaterialRecord(); self._mat.shader = "defaultUnlit"; self._mat.point_size = 2.0
        self._line_mat = rendering.MaterialRecord(); self._line_mat.shader = "unlitLine"; self._line_mat.line_width = 3

        panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        self._type_combo = gui.Combobox()
        for t in self.types:
            self._type_combo.add_item(t)
        self._type_combo.set_on_selection_changed(lambda name, i: self._set("type", name))
        panel.add_child(gui.Label("Landmark type:")); panel.add_child(self._type_combo)

        self._sliders = {}
        for key, lo, hi in [("cx", -200, 200), ("cy", -300, 100),
                            ("length", 1, 60), ("width", 1, 40), ("heading", -180, 180)]:
            lbl = gui.Label(f"{key}: {self.cur[key]:.1f}")
            s = gui.Slider(gui.Slider.DOUBLE); s.set_limits(lo, hi); s.double_value = self.cur[key]
            s.set_on_value_changed(self._make_cb(key, lbl))
            panel.add_child(lbl); panel.add_child(s); self._sliders[key] = (s, lbl)

        add_btn = gui.Button("Add landmark"); add_btn.set_on_clicked(self._add)
        save_btn = gui.Button("Save YAML"); save_btn.set_on_clicked(self._save)
        undo_btn = gui.Button("Undo last"); undo_btn.set_on_clicked(self._undo)
        panel.add_child(add_btn); panel.add_child(undo_btn); panel.add_child(save_btn)
        self._status = gui.Label("0 landmarks"); panel.add_child(self._status)

        w.add_child(self.scene); w.add_child(panel); self._panel = panel
        w.set_on_layout(self._layout)
        self._draw()

    def _set(self, k, v):
        self.cur[k] = v

    def _make_cb(self, key, lbl):
        def cb(val):
            self.cur[key] = val
            lbl.text = f"{key}: {val:.1f}"
            self._draw()
        return cb

    def _box_lineset(self, cx, cy, length, width, heading_deg, color):
        th = math.radians(heading_deg)
        u = np.array([math.cos(th), math.sin(th)]); v = np.array([-math.sin(th), math.cos(th)])
        hl, hw = length / 2, width / 2
        corners2d = [np.array([cx, cy]) + a * hl * u + b * hw * v
                     for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        z = 1.0
        pts = [[c[0], c[1], z] for c in corners2d]
        lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
        ls = self.o3d.geometry.LineSet()
        ls.points = self.o3d.utility.Vector3dVector(pts)
        ls.lines = self.o3d.utility.Vector2iVector(lines)
        ls.colors = self.o3d.utility.Vector3dVector([color] * len(lines))
        return ls

    def _draw(self):
        self.scene.scene.clear_geometry()
        pcd = self.o3d.geometry.PointCloud()
        pcd.points = self.o3d.utility.Vector3dVector(self.pts)
        cols = np.tile([0.6, 0.6, 0.65], (len(self.pts), 1))
        pcd.colors = self.o3d.utility.Vector3dVector(cols)
        self.scene.scene.add_geometry("map", pcd, self._mat)
        if len(self.traj):
            tls = self.o3d.geometry.LineSet()
            tp = np.column_stack([self.traj[:, 0], self.traj[:, 1], np.full(len(self.traj), 1.0)])
            tls.points = self.o3d.utility.Vector3dVector(tp)
            tls.lines = self.o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(tp) - 1)])
            tls.colors = self.o3d.utility.Vector3dVector([[1, 0.2, 0.2]] * (len(tp) - 1))
            self.scene.scene.add_geometry("traj", tls, self._line_mat)
        for j, lm in enumerate(self.landmarks):
            self.scene.scene.add_geometry(
                f"lm{j}", self._box_lineset(lm["center"][0], lm["center"][1], lm["length"],
                                            lm["width"], lm["heading_deg"], [0.2, 1, 0.4]),
                self._line_mat)
        self.scene.scene.add_geometry(
            "cur", self._box_lineset(self.cur["cx"], self.cur["cy"], self.cur["length"],
                                     self.cur["width"], self.cur["heading"], [1, 1, 0.2]),
            self._line_mat)
        if not hasattr(self, "_camset"):
            bounds = self.scene.scene.bounding_box
            self.scene.setup_camera(60, bounds, bounds.get_center())
            self._camset = True

    def _add(self):
        self.landmarks.append({
            "id": len(self.landmarks), "type": self.cur["type"],
            "center": [self.cur["cx"], self.cur["cy"], 0.0],
            "heading_deg": self.cur["heading"], "length": self.cur["length"],
            "width": self.cur["width"]})
        self._status.text = f"{len(self.landmarks)} landmarks"
        self._draw()

    def _undo(self):
        if self.landmarks:
            self.landmarks.pop()
            self._status.text = f"{len(self.landmarks)} landmarks"
            self._draw()

    def _save(self):
        from bev_pipeline.landmark_schema import LandmarkSet, Landmark, save_landmarks
        ls = LandmarkSet(bag_name=self.bag_name, frame_id=self.frame_id,
                         landmarks=[Landmark(**lm) for lm in self.landmarks])
        save_landmarks(ls, self.out_path)
        self._status.text = f"saved {len(self.landmarks)} -> {self.out_path}"

    def _layout(self, ctx):
        r = self.window.content_rect
        pw = 20 * ctx.theme.font_size
        self.scene.frame = self.gui.Rect(r.x, r.y, r.width - pw, r.height)
        self._panel.frame = self.gui.Rect(r.x + r.width - pw, r.y, pw, r.height)

    def run(self):
        self.gui.Application.instance.run()


def main():
    import json
    from bev_pipeline.landmark_schema import load_landmark_types
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("--types", default=os.path.join(PKG, "config", "landmark_types.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    types = load_landmark_types(args.types)["types"]
    meta = json.load(open(os.path.join(args.env_dir, "meta.json")))
    bag_name = meta.get("env", "bag")
    frame_id = "odom"
    out = args.out or os.path.join(os.getcwd(), f"{bag_name}.landmarks.yaml")
    print(f"building global map from {args.env_dir} ...")
    pts, traj = build_global_map(args.env_dir)
    print(f"map: {len(pts)} pts, {len(traj)} trajectory poses. Launching GUI...")
    BboxEditor(pts, traj, types, out, bag_name, frame_id).run()


if __name__ == "__main__":
    main()
