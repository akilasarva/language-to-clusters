"""config/missions.yaml and scripts/run_missions.py: the nine missions are runnable."""
from __future__ import annotations

import json
import os
import sys

import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PKG, "scripts"))
DOC = yaml.safe_load(open(os.path.join(PKG, "config", "missions.yaml")))


def test_the_nine_missions_are_present_with_english():
    ids = [m["id"] for m in DOC["missions"]]
    assert ids == [f"M{i}" for i in range(1, 10)]
    assert all(isinstance(m["en"], str) and m["en"].strip() for m in DOC["missions"])


def test_every_world_is_well_formed_and_its_props_can_spawn():
    pose = {}
    for f in ("cones", "props"):
        p = os.path.join(PKG, "config", f"{f}.{DOC['town']}.json")
        d = json.load(open(p))
        for c in (d.get("cones") or []) + (d.get("props") or []):
            pose[int(c["region"])] = c
    names = set()
    for m in DOC["missions"]:
        for w in m["worlds"]:
            assert {"name", "start", "toward", "props"} <= set(w), (m["id"], w)
            assert (m["id"], w["name"]) not in names
            names.add((m["id"], w["name"]))
            missing = [r for r in w["props"] if r not in pose]
            assert not missing, f"{m['id']}/{w['name']}: no spawn pose for {missing}"


def test_neutral_plan_carries_geometry_and_no_constraints():
    from plan_geometry import neutral_plan, region_geometry
    plan = neutral_plan(DOC["town"], 0)
    geo = region_geometry(DOC["town"])
    for k in ("cluster_labels", "centroids", "bearing_map"):
        assert plan[k] == geo[k] and plan[k]
    assert len(plan["steps"]) == 1 and plan["steps"][0]["trigger"] == "traverse"
    for k in ("forbid_modes", "forbid_clusters", "require_modes", "hold_accept_clusters"):
        assert k not in plan and k not in plan["steps"][0]


def test_list_selects_worlds(capsys):
    import run_missions
    assert run_missions.main(["--arm", "ours", "--only", "M5", "--list"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 6
    assert run_missions.main(["--arm", "llm", "--only", "M1", "--list"]) == 1   # not sited


def test_container_path_maps_the_workspace_mount():
    import run_missions
    p = os.path.join(run_missions.WS, "carla_gt_bridge", "x.txt")
    assert run_missions.container_path(p) == "/ros_ws/src/carla_gt_bridge/x.txt"
