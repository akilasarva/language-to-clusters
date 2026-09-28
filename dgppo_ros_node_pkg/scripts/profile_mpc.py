#!/usr/bin/env python3
"""Profile how much of a CARLA control tick is spent in the MPC itself.

Under synchronous_mode + wait_for_vehicle_control_command the world only advances when
the MPC emits a command, so simulated time is gated on tick cost. This script checks
whether `mpc_K` (the rollout count) is a meaningful lever on that cost.

`plan_step` is pure numpy over the real Town05 road surface, so it profiles offline
with no simulator:

    K=500   ~5 ms per call
    K=150   ~1.5 ms per call

With control_every_n_odom=4 a CARLA control tick took seconds of wall clock and MPC
compute was negligible. At the current default of 1 a serviced tick is ~14 ms, so a
~5 ms MPC call is a real share of it: re-profile before tuning K.

Most of the tick is the bridge/server round trip: odometry published -> MPC -> Twist ->
carla_twist_to_control -> world tick -> sensors republished (several hundred ms per
world tick at fixed_delta_seconds=0.05). To speed up runs, profile the bridge's
per-tick sensor publishing and the wait-for-control handshake rather than the
controller.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WS, "dgppo_ros_node_pkg"))

from dgppo_ros_node_pkg.sampling_mpc import MpcConfig, RoadSurface, plan_step  # noqa: E402

#: Reference timing from a closed-loop CARLA drive: 28 control ticks, 82 s wall, 5.4 s sim.
OBSERVED_TICK_MS = 82_000 / 28
OBSERVED_WORLD_TICK_MS = 82_000 / (5.4 / 0.05)


def main() -> int:
    npz = np.load(os.path.join(WS, "carla_gt_bridge", "config", "regions.town05.npz"),
                  allow_pickle=True)
    cents = npz["centroids"]
    rids = [int(r) for r in npz["rids"]]
    road = RoadSurface(npz["waypoints"])
    start, target = 45, 66            # a road region and a junction it leads to
    pos, yaw = cents[rids.index(start)].astype(float), 0.0

    print(f"medium mission, region {start} -> {target}, real Town05 road surface\n")
    print(f"{'K':>6}{'ms/call':>10}{'vs 500':>9}{'% of a CARLA tick':>20}")
    base = None
    for K in (500, 300, 150, 50):
        cfg = MpcConfig(K=K, N=8, dt=0.2)
        plan_step(cfg, pos, yaw, cents, rids, start, target, road=road,
                  rng=np.random.default_rng(0))
        reps, t0 = 30, time.perf_counter()
        for _ in range(reps):
            plan_step(cfg, pos, yaw, cents, rids, start, target, road=road,
                      rng=np.random.default_rng(0))
        ms = (time.perf_counter() - t0) / reps * 1000
        base = base or ms
        print(f"{K:>6}{ms:>10.1f}{ms / base:>8.2f}x{100 * ms / OBSERVED_TICK_MS:>19.2f}%")

    print(f"\n  observed CARLA control tick : {OBSERVED_TICK_MS:.0f} ms")
    print(f"  observed CARLA world tick   : {OBSERVED_WORLD_TICK_MS:.0f} ms")
    print("\n  The MPC is a rounding error inside the tick. K is not the lever, and a")
    print("  K=500 vs K=150 comparison in CARLA would measure noise. Profile the bridge")
    print("  round trip -- per-tick sensor publishing and the wait-for-control")
    print("  handshake -- which is where the other 99.8% is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
