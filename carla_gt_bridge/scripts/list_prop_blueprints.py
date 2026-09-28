#!/usr/bin/env python3
"""List static.prop blueprints on the running server, so a pose is never added for a
blueprint that does not exist. Runs INSIDE the bridge container (host carla is 0.10.0).
"""
import argparse, sys
import carla

ap = argparse.ArgumentParser()
ap.add_argument("--town", default="town05")
ap.add_argument("--match", default="")
a = ap.parse_args()
c = carla.Client("127.0.0.1", 2000); c.set_timeout(60.0)
w = c.get_world()
bp = [b.id for b in w.get_blueprint_library().filter("static.prop.*")]
print(f"{len(bp)} static.prop blueprints")
for pat in ("fountain", "bench", "cone", "kiosk", "shelter", "vending", "mailbox", "trash"):
    hits = sorted(x for x in bp if pat in x.lower())
    print(f"  {pat:10} -> {hits if hits else 'NONE'}")
sys.exit(0)
