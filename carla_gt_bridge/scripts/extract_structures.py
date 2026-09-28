#!/usr/bin/env python3
"""Dump the STRUCTURES M1/M9 need, with positions: bridges, overpasses (stacked lanes) and car
parks. Runs inside the version-matched bridge container (host carla 0.10.0 vs server 0.9.14) via
run_extract_landmarks.sh:  EXTRACT=extract_structures.py bash run_extract_landmarks.sh
Output: /ros_ws/src/carla_gt_bridge/config/structures.<town>.raw.json, positions in CARLA frame
(left-handed; planar y = -carla y -- convert before touching regions.<town>.npz)."""
import argparse, json, sys
import carla

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--town", default="town05")
    a = ap.parse_args()
    c = carla.Client("127.0.0.1", 2000); c.set_timeout(120.0)
    w = c.get_world()
    if a.town.lower() not in w.get_map().name.lower():
        w = c.load_world(a.town.capitalize().replace("town", "Town"))
    out = {"town": a.town, "frame": "carla"}
    def bb(b):
        return {"x": b.location.x, "y": b.location.y, "z": b.location.z,
                "ex": b.extent.x, "ey": b.extent.y, "ez": b.extent.z, "yaw": b.rotation.yaw}
    out["bridge_bbs"] = [bb(b) for b in w.get_level_bbs(carla.CityObjectLabel.Bridge)]
    out["railtrack_bbs"] = [bb(b) for b in w.get_level_bbs(carla.CityObjectLabel.RailTrack)]
    print("railtrack bbs", len(out["railtrack_bbs"]), flush=True)
    print("bridge bbs", len(out["bridge_bbs"]), flush=True)
    names = {}
    for eo in w.get_environment_objects(carla.CityObjectLabel.Any):
        n = eo.name.lower()
        for key in ("park", "bridge", "overpass", "flyover", "garage", "lot"):
            if key in n:
                names.setdefault(key, []).append({"name": eo.name, "type": str(eo.type), **bb(eo.bounding_box)})
    out["named_objects"] = names
    print("named:", {k: len(v) for k, v in names.items()}, flush=True)
    wps = w.get_map().generate_waypoints(2.0)
    out["elevated_waypoints"] = [{"x": p.transform.location.x, "y": p.transform.location.y,
                                  "z": p.transform.location.z, "road": p.road_id, "lane": p.lane_id}
                                 for p in wps if p.transform.location.z > 3.0]
    print("elevated waypoints", len(out["elevated_waypoints"]), flush=True)
    park = []
    for p in wps:
        for side in (p.get_left_lane(), p.get_right_lane()):
            if side is not None and side.lane_type == carla.LaneType.Parking:
                park.append({"x": side.transform.location.x, "y": side.transform.location.y, "road": side.road_id})
    out["parking_lane_points"] = park
    print("parking lane points", len(park), flush=True)
    json.dump(out, open(f"/ros_ws/src/carla_gt_bridge/config/structures.{a.town}.raw.json", "w"), indent=1)
    print("WROTE", flush=True)

if __name__ == "__main__":
    sys.exit(main())
