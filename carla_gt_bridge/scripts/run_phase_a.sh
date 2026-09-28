#!/usr/bin/env bash
# Phase A: CARLA + bridge + the mission stack, in one command, with a low simulation load.
#
#   PLAN_FILE=<tree.json> bash run_phase_a.sh - dry     # dry run (no Twist published)
#   PLAN_FILE=<tree.json> bash run_phase_a.sh - drive   # actually driving
#   bash run_phase_a.sh --stop                          # tear everything down
#
# PLAN_FILE is a CONTAINER path (the workspace is mounted at /ros_ws). drive_english.py
# generates a tree from English and calls this script with it.
#
# WHY THE FLAGS ARE WHAT THEY ARE
#
#   -fps 10           Without a cap the world free-runs far faster than real time. The MPC
#                     ticks at 5 Hz, so there is no reason to simulate faster than that.
#   no_rendering_mode Phase A declares NO real sensors, so rendered frames are wasted work.
#                     Set through the CARLA API because the bridge launch has no argument
#                     for it. Must come OFF for Phase B (LiDAR) and C (camera).
#   RUN_SECONDS       A hard cap: bring it up, run, tear it down.
#
# THE BRIDGE LOADS THE TOWN; NOTHING ELSE MAY.
#
# bridge.py does:  if carla_world.get_map().name != parameters["town"]: load_world(...)
# A loaded map reports its name as "Carla/Maps/Town05" while the parameter is "Town05", so
# that comparison never matches and passing town:= always makes the bridge reload the world.
# If the town were also loaded here, carla_spawn_objects would race a world being torn down
# and rebuilt and hang. The reload also resets no_rendering_mode, so that setting is applied
# AFTER the bridge is up.
#
# carla_spawn_objects calls the /carla/spawn_object service through ros_compatibility's
# call_service(..., spin_until_response_received=True), i.e.
# `rclpy.spin_until_future_complete(self, future, self.executor, timeout)` with
# timeout=None -- if the service never answers it blocks forever and prints nothing
# (the log shows only "process started with pid").
#
# In SYNCHRONOUS mode carla's spawn_actor cannot complete until the world ticks, and the
# bridge tick contends with the callback doing the spawning, so the spawn can wedge.
# Asynchronous mode (SYNC=False, the default) removes the tick dependency: the server
# free-runs at -fps 10, spawn_actor returns immediately, and simulated time stays close to
# wall time. Set SYNC=True for determinism-critical work, but expect the spawn to hang
# intermittently; the retry loop below mitigates it.
#
# Everything runs INSIDE the bridge container (ROS Humble + carla 0.9.14, version-matched
# to the server) with ~/ros2_ws mounted at /ros_ws, so there is no cross-distro DDS.

set -euo pipefail

# ---------------------------------------------------------------------------
# ONE CARLA RUN AT A TIME, enforced by a lock. Gating on `pgrep` is not enough: it
# clears during the settle between runs, so a second run could start in the gap and its
# `docker rm -f` below would tear down a LIVE container.
#
# The lock lives here rather than in a sweep script because this is the single entry point
# to CARLA: anything that starts a run -- a sweep, drive_english, a hand-typed command --
# serialises on it without having to know about the others.
# LOCK_WAIT=0 fails fast instead of queueing.
# ---------------------------------------------------------------------------
LOCK_FILE="${LOCK_FILE:-/tmp/carla_gt_bridge.lock}"
exec 9>"$LOCK_FILE"
if ! flock -w "${LOCK_WAIT:-2400}" 9; then
    echo "another CARLA run holds $LOCK_FILE (waited ${LOCK_WAIT:-2400}s) — refusing to"
    echo "start, because tearing down its containers would corrupt that run."
    echo "Wait, or set LOCK_WAIT=0 to fail fast."
    exit 75
fi
echo "[lock] held $LOCK_FILE (pid $$)"

MISSION="${1:--}"
MODE="${2:-dry}"                 # dry | drive
RUN_SECONDS="${RUN_SECONDS:-180}"

# MPC rollouts per control tick, and the horizon. THE WALL-CLOCK LEVER: under
# SYNC=True + WAIT_FOR_CONTROL=True the world advances only when the MPC emits a command,
# so simulated time is gated on how long one tick takes (seconds per tick at K=500, well
# below real time). GPU load is not the constraint; tick cost is.
#
# mission.launch.py passes these through to the node. Lowering K trades path quality for
# wall clock and fails SOFTLY -- a worse path, not an error -- so compare a candidate value
# against a known-good run before trusting it.
MPC_K="${MPC_K:-500}"
MPC_N="${MPC_N:-8}"

# Which signal the MPC steers by: "region" (pure-pursuit aim into the target cluster,
# membership from the waypoint table, bearing demoted to a tie-breaker) or "centroid"
# (aim at the target centroid, Voronoi membership, bearing at full weight). "region"
# requires a road surface built with region ids.
# Default is "region". Under "centroid" the `region` aim branch is dead code (`use_region`
# gates on guidance == "region"), so a run that does not set this explicitly would measure
# a different system. Scripts that want the centroid behaviour set GUIDANCE=centroid.
GUIDANCE="${GUIDANCE:-region}"

# EDT rejection distance, and how many horizon steps the collision test looks at.
# Inside tight junctions a full-horizon test can reject most rollouts and stall the
# vehicle. 0 = check all N steps, the original behaviour.
SAFETY_RADIUS="${SAFETY_RADIUS:-1.5}"
COLLISION_HORIZON="${COLLISION_HORIZON:-0}"
# >=0 seeds the MPC sampler so the run replays exactly; <0 is OS entropy. Sweeps
# should set this to the repeat index.
MPC_SEED="${MPC_SEED:-0}"
# Map distortion for the robustness ablation: a similarity transform on the map's
# derived geometry (centroids, bearings) with cluster membership left true.
# AIM_SOURCE defaults to "carrot". Aiming at the target region's nearest point is unsound
# inside a junction (that point is ~31 m away against an ~8 m horizon, so the occupancy
# term is flat); the carrot puts the aim on drivable surface THROUGH the junction.
AIM_SOURCE="${AIM_SOURCE:-carrot}"; PURE_PURSUIT="${PURE_PURSUIT:-false}"
# Which controller drives. Default is "terrain", the curvature-fan controller.
# "region" selects the earlier sampling controller.
CONTROLLER="${CONTROLLER:-terrain}"; KAPPA_MAX="${KAPPA_MAX:-0.25}"
# 6.28 x kappa_max 0.25 = exactly 90 deg of turn per arc, which is terrain_mpc's
# `max_turn_rad` cap. Larger values (e.g. 10.0 -> 143 deg) HARD-FAIL at config
# construction.
ARC_LEN_M="${ARC_LEN_M:-6.28}"; FAN_SIZE="${FAN_SIZE:-65}"
FAN_SEGMENTS="${FAN_SEGMENTS:-1}"
# Applies to CONTROLLER=region only -- the terrain branch does not read this parameter and
# uses TerrainMpcConfig.w_obstacle (3.0) instead.
W_OBSTACLE="${W_OBSTACLE:-5.0}"
DIST_MODEL="${DIST_MODEL:-similarity}"; DIST_JITTER="${DIST_JITTER:-0.0}"
# Displace the drivable surface with the centroids, so membership and the goal point BOTH
# read a wrong map. With the waypoint table left pristine, membership would be unaffected
# by construction rather than by robustness.
DIST_SURFACE="${DIST_SURFACE:-false}"
DIST_JITTER_M="${DIST_JITTER_M:-0.0}"; DIST_ROT="${DIST_ROT:-0.0}"
DIST_PIVOT="${DIST_PIVOT:-map}"
DIST_ANGLE="${DIST_ANGLE:-0.0}"; DIST_SCALE="${DIST_SCALE:-1.0}"
DIST_TX="${DIST_TX:-0.0}"; DIST_TY="${DIST_TY:-0.0}"

# Controls emitted per world tick. Under WAIT_FOR_CONTROL the server holds each tick
# until it gets a vehicle control command, and the node publishes one only every
# CONTROL_EVERY_N-th odometry message, so any tick it does not service stalls on the
# server's ~1 s timeout. Values above 1 make runs far slower than real time, which is
# why the default is 1 and not the dt/fixed_delta value of 4.
CONTROL_EVERY_N="${CONTROL_EVERY_N:-1}"
TOWN="${TOWN:-Town05}"
# Where brain gets its cue answers. `topic` is the ground-truth oracle, which isolates
# the representation from perception. `vlm` asks a VLM against the live camera instead.
CUE_SOURCE="${CUE_SOURCE:-topic}"
# ONE VARIABLE DECIDES THE CAMERA. Rendering, the objects file and the server quality all
# derive from NEEDS_CAMERA, so they cannot disagree. Both the VLM cue source and a
# non-waypoint terrain source need a camera; `-quality-level=Low` with a camera produces
# 0 mpc ticks with no message naming the camera.
TERRAIN_SOURCE="${TERRAIN_SOURCE:-waypoint}"
TERRAIN_LABELLER="${TERRAIN_LABELLER:-carla_semantic}"
NEEDS_CAMERA=false
[ "$CUE_SOURCE" = vlm ] && NEEDS_CAMERA=true
[ "$TERRAIN_SOURCE" != waypoint ] && NEEDS_CAMERA=true
# Lowercase, because the config artifacts are named for the town in lower case
# (objects.town05.json, regions.town05.npz) while CARLA wants "Town05".
TOWN_LC="$(echo "$TOWN" | tr '[:upper:]' '[:lower:]')"
# PLAN_FILE: the brain tree to drive (required; see the usage at the top).
PLAN_FILE="${PLAN_FILE:-}"
ROLE="${ROLE:-ego_vehicle}"
CONTAINER=humble_dev_with_code
SERVER=carla_server
# Overridable so the same script works under a different account; a remote with another
# username only needs to export these.
WS="${WS:-$HOME/ros2_ws}"
CARLA_PYTHONAPI="${CARLA_PYTHONAPI:-$HOME/carla}"
CFG=/ros_ws/src/carla_gt_bridge/config

# Set LOG_DIR to keep the logs. They live INSIDE the container, so teardown destroys
# them. Copied out BEFORE the containers go.
save_logs() {
    [ -n "${LOG_DIR:-}" ] || return 0
    mkdir -p "$LOG_DIR"
    # bridge.log too: without it there is no way to tell whether synchronous_mode actually
    # took effect. Copy failures are reported rather than swallowed with `|| true`: a
    # container that died early otherwise leaves a run that looks healthy but has no logs.
    for f in mission spawn twist bridge; do
        if ! docker cp "$CONTAINER:/tmp/$f.log" "$LOG_DIR/$f.log" >/dev/null 2>&1; then
            echo "[logs] WARN could not copy /tmp/$f.log out of $CONTAINER"
        fi
    done
}

stop_all() {
    save_logs
    echo "[stop] removing containers"
    docker rm -f "$SERVER" "$CONTAINER" >/dev/null 2>&1 || true
    sleep 2
    nvidia-smi --query-gpu=memory.used,temperature.gpu,utilization.gpu \
               --format=csv,noheader 2>/dev/null || true
}

if [ "$MISSION" = "--stop" ]; then stop_all; exit 0; fi
if [ -z "$PLAN_FILE" ]; then
    echo "PLAN_FILE is required: a brain tree (container path under /ros_ws). Generate one"
    echo "from English with scripts/drive_english.py, or run scripts/run_missions.py."
    exit 2
fi
trap stop_all EXIT INT TERM       # never leave the GPU running on a failure

# THE LOCK (also held by run_collection.sh). Without it two servers can fight over port
# 2000 (one dies with SIGSEGV), and a `load_world` meant for one run can land on another
# run's server mid-run. Both scripts queue instead of colliding.
LOCK_FILE="${LOCK_FILE:-/tmp/carla_gt_bridge.lock}"
exec 9>"$LOCK_FILE"
if ! flock -w "${LOCK_WAIT:-3600}" 9; then
    echo "another CARLA run holds $LOCK_FILE -- refusing to start."; exit 1
fi
echo "[lock] held $LOCK_FILE (pid $$)"

echo "[1/6] clean slate"
docker rm -f "$SERVER" "$CONTAINER" >/dev/null 2>&1 || true

echo "[2/6] CARLA server (headless, low quality, 10 fps)"
docker run --rm -d --privileged --gpus all --net=host --name "$SERVER" \
    carlasim/carla:0.9.14 \
    /bin/bash -c "./CarlaUE4.sh -RenderOffScreen \
        -quality-level=${QUALITY:-$([ "$NEEDS_CAMERA" = true ] && echo Epic || echo Low)} \
        -nosound -fps 10" \
    >/dev/null
until (echo > /dev/tcp/127.0.0.1/2000) 2>/dev/null; do sleep 2; done
echo "      up"

echo "[3/6] bridge container"
docker run -d --name "$CONTAINER" --net=host \
    -v "$WS":/ros_ws -v "$CARLA_PYTHONAPI":/carla --user=root \
    carla-ros-bridge-dev:latest tail -f /dev/null >/dev/null

# Without -e the key does not cross into the container and brain logs "OPENAI_API_KEY
# not set -- VLM cue checks AND branch decisions will fail; decisions will fall back to
# the 'default' branch". A sweep can then complete looking healthy while every branch
# was taken by the fallback rather than by perception.
# PYTHONUNBUFFERED: brain records what the VLM actually answered with print()
# ([VLM] Cue query / Branch query), not the ROS logger. print() block-buffers when stdout
# is redirected to a file, so on a run killed at the cap those lines would be lost while
# get_logger() lines survive.
# LIDAR_Z / LIDAR_LOWER_FOV must be FORWARDED: lane_spawn.py, the only place the sensor is
# defined, runs inside the container. A host-side export does not reach it and the
# override silently does nothing.
DOCKER_ENV="-e OPENAI_API_KEY=${OPENAI_API_KEY:-} -e CUE_SOURCE=$CUE_SOURCE -e PYTHONUNBUFFERED=1 -e LIDAR_Z=${LIDAR_Z:-2.0} -e LIDAR_LOWER_FOV=${LIDAR_LOWER_FOV:--15.0} -e LIDAR_HEIGHTS=${LIDAR_HEIGHTS:-} -e SEMANTIC_LIDAR=${SEMANTIC_LIDAR:-} -e BEARING_SCOPE_M=${BEARING_SCOPE_M:-} -e BEARING_EXIT_GRACE_M=${BEARING_EXIT_GRACE_M:-} -e CUE_SCOPED_APPROACH=${CUE_SCOPED_APPROACH:-} -e CUE_EVIDENCE_M=${CUE_EVIDENCE_M:-} -e CUE_APPROACH_EVERY_M=${CUE_APPROACH_EVERY_M:-} -e DECIDE_V_MAX=${DECIDE_V_MAX:-} -e EXTRA_LANDMARKS=${EXTRA_LANDMARKS:-} -e BRAIN_MODE=${BRAIN_MODE:-} -e LLM_BRAIN_SCRIPT=${LLM_BRAIN_SCRIPT:-} -e LLM_MODEL=${LLM_MODEL:-} -e LLM_MISSION_FILE=${LLM_MISSION_FILE:-} -e CUE_SEMANTICS=${CUE_SEMANTICS:-}"
# VLM_FRAME_DUMP: save the exact frames sent to the model. A CONTAINER path, because
# brain runs inside; put it under /ros_ws (the mounted workspace) so the PNGs land on
# the host and survive teardown. Opt-in, since a sweep would otherwise write hundreds.
# Note: `[ -n x ] && DOCKER_ENV=...` would also be safe under `set -e` (bash exempts a
# failing command in a && list unless it is the final one); the `if` is for legibility.
if [ -n "${VLM_FRAME_DUMP:-}" ]; then
    DOCKER_ENV="$DOCKER_ENV -e VLM_FRAME_DUMP=$VLM_FRAME_DUMP"
fi
X="docker exec $DOCKER_ENV $CONTAINER bash -lc"

# Rebuild the packages every run. The container is recreated from the image each time and
# `/ros_ws` is the host tree, so install/ persists -- but entry points and package.xml do
# not update themselves, and a run against stale code is indistinguishable from a run
# against broken code. A few seconds is cheaper than that ambiguity.
echo "      rebuilding carla_gt_bridge / brain / dgppo_ros_node_pkg"
# A FAILED BUILD MUST STOP THE RUN; otherwise the run continues on whatever was installed
# before, and an MPC change can be logged as present but never rebuilt.
#
# CLEAN-BUILD carla_gt_bridge every run. Incremental symlink state in this package fails
# its build, which ABORTS dgppo_ros_node_pkg (the MPC) and leaves the run on stale code:
#   * dangling links   -- a config file globbed, then deleted:
#                         "can't copy '<path>': doesn't exist or not a regular file"
#   * generated config -- lane_spawn.py rewrites objects.<town>.json into the globbed
#                         config/ dir every run: "[Errno 17] File exists: build/... -> install/..."
#   * any EDITED file  -- same Errno 17, e.g. on launch/mission.launch.py
# All of these are build-directory state only, so clearing it is safe. The package is pure
# Python and builds in ~1 s. brain and dgppo_ros_node_pkg keep their incremental state.
$X "rm -rf /ros_ws/build/carla_gt_bridge /ros_ws/install/carla_gt_bridge" \
    >/dev/null 2>&1 || true
BUILD_OUT=$($X "source /opt/ros/humble/setup.bash && cd /ros_ws && colcon build --symlink-install \
    --packages-select carla_gt_bridge brain dgppo_ros_node_pkg 2>&1 | tail -12")
echo "$BUILD_OUT" | tail -2
case "$BUILD_OUT" in
    *"packages failed"*|*"package failed"*|*"packages aborted"*|*"package aborted"*)
        echo "      ERROR: the workspace did not build. The run would have used STALE code."
        echo "$BUILD_OUT" | sed 's/^/      | /'
        exit 1 ;;
esac
# brain imports openai at module scope; the image does not ship it. Harmless to re-run.
$X "python3 -c 'import openai' 2>/dev/null || pip install -q openai" >/dev/null 2>&1 || true
# The world is loaded by the BRIDGE, below -- see the comment at the top. Loading it here
# too causes the spawn hang.


echo "[4/6] bridge (it loads the town; nothing else may)"
docker exec -d $DOCKER_ENV "$CONTAINER" bash -lc \
    "source /carla_ws/install/setup.bash && ros2 launch carla_ros_bridge carla_ros_bridge.launch.py \
     town:=$TOWN timeout:=60 synchronous_mode:=${SYNC:-False} fixed_delta_seconds:=0.05 \
     synchronous_mode_wait_for_vehicle_control_command:=${WAIT_FOR_CONTROL:-False} \
     > /tmp/bridge.log 2>&1"
# Wait for the bridge to actually be up, rather than sleeping a guessed 15 s. It has to
# enumerate every actor in the town (Town05 has 250+ traffic lights alone), and how long
# that takes varies. Start carla_spawn_objects too early and the spawn silently does not
# happen — which then presents as "no odometry topic", pointing at the wrong thing.
for _ in $(seq 1 30); do
    if $X "source /carla_ws/install/setup.bash && ros2 topic list 2>/dev/null" \
         | grep -q "/carla/world_info"; then break; fi
    sleep 2
done
# ...and specifically for the SERVICE carla_spawn_objects blocks on. The world_info topic
# appears well before /carla/spawn_object is advertised, so waiting on the topic alone
# still raced: spawn_objects came up, blocked, and never spawned. It then presented as
# "no odometry topic", which points at the vehicle rather than at the service.
for _ in $(seq 1 30); do
    if $X "source /carla_ws/install/setup.bash && ros2 service list 2>/dev/null" \
         | grep -q "/carla/spawn_object"; then break; fi
    sleep 2
done
# Wait for the MAP, not just for a topic. The bridge loads the town itself and that takes
# tens of seconds; every previous readiness check passed while the world was still being
# rebuilt underneath, which is exactly what the spawn was racing.
for _ in $(seq 1 45); do
    got=$($X "source /carla_ws/install/setup.bash && python3 -c \"
import carla
print(carla.Client('localhost', 2000).get_world().get_map().name)\"" 2>/dev/null | tr -d '\r')
    case "$got" in *"$TOWN"*) break ;; esac
    sleep 2
done
echo "      bridge up, map is $got"
case "$got" in *"$TOWN"*) ;; *) echo "      ERROR: map is $got, wanted $TOWN"; exit 1 ;; esac
# no_rendering_mode goes on AFTER the load -- a world reload resets it.
#
# IT MUST BE OFF WHENEVER A CAMERA IS NEEDED. Camera sensors need the renderer; LiDAR does
# not. With it on and CUE_SOURCE=vlm, brain never receives a frame and `_run_vlm_decide`
# takes the `default` branch in EVERY world without failing, so a matched pair produces
# near-identical traces. Derived from NEEDS_CAMERA rather than given its own flag.
# NO_RENDERING/WITH_CAMERA can still be overridden so rendering and camera cost can be
# varied independently when diagnosing a stall.
if [ -z "${NO_RENDERING:-}" ]; then
    if [ "$NEEDS_CAMERA" = true ]; then NO_RENDERING=False; else NO_RENDERING=True; fi
fi
$X "source /carla_ws/install/setup.bash && python3 -c \"
import carla
w = carla.Client('localhost', 2000).get_world()
s = w.get_settings(); s.no_rendering_mode = $NO_RENDERING; w.apply_settings(s)
print('      no_rendering_mode', w.get_settings().no_rendering_mode)
assert w.get_settings().no_rendering_mode == $NO_RENDERING, 'no_rendering_mode did not take'
\""

# ONLY NOW can the spawn pose be computed: lane_spawn.py asks the live map which lane sits
# at the start region and which way it points. Run against the wrong town it silently
# answers about a different road.
echo "      computing the ego spawn from the live map ($TOWN_LC)"
# The seed comes from LANE_SPAWN_ARGS. When it is not given, DERIVE it from missions.py
# rather than falling back to lane_spawn.py's compiled-in defaults: a default that goes
# stale when the corridor's start region changes spawns the ego far from where the plan
# starts, and the run then reads as a control failure.
if [ -z "${LANE_SPAWN_ARGS:-}" ] && [ "$TOWN_LC" = "town05" ]; then
    LANE_SPAWN_ARGS=$(python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
sys.path.insert(0, os.getcwd())
import missions
_, _, _, cx, cy, cyaw = missions.spawn_seed()
print(f"--town town05 --region {missions.START_REGION} "
      f"--x {cx:.2f} --y {cy:.2f} --yaw {cyaw:.1f}")
PY
    ) || { echo "      ERROR: could not derive the spawn seed"; exit 1; }
    echo "      derived seed: $LANE_SPAWN_ARGS"
fi
$X "source /carla_ws/install/setup.bash && python3 /ros_ws/src/carla_gt_bridge/scripts/lane_spawn.py \
    ${LANE_SPAWN_ARGS:-}" | tail -4
# Applied HERE, not before the bridge: anything that reloads the world resets it.
# Spawn, then verify, then RETRY. The spawn can still race the bridge's startup (always
# presenting as "no odometry topic") even after waiting for /carla/world_info and the
# /carla/spawn_object service; the cause is not isolated, so the retry loop makes runs
# repeatable and the attempt counter makes the race visible.
# WHICH SENSOR SET. A camera lives in a SEPARATE generated objects file rather than being
# added to the Phase A one, so camera-free runs (rendering off) keep their original sensor
# set and tick cost.
OBJECTS="$CFG/objects.$TOWN_LC.json"
WITH_CAMERA="${WITH_CAMERA:-$NEEDS_CAMERA}"
if [ "$WITH_CAMERA" = true ]; then
    # DERIVE IT, DO NOT KEEP A COPY. lane_spawn.py has just REWRITTEN
    # objects.$TOWN_LC.json with the ego pose computed from the live map for THIS run's
    # --start region. A hand-maintained camera variant would be a snapshot of one start
    # region's pose and would spawn the ego far from the plan's start for any other.
    # So the camera is appended to whatever lane_spawn just wrote.
    python3 - "$WS/src/carla_gt_bridge/config/objects.$TOWN_LC.json" \
              "$WS/src/carla_gt_bridge/config/objects.$TOWN_LC.camera.gen.json" <<'PYCAM'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
blob = json.load(open(src))
ego = next(o for o in blob["objects"] if o.get("id") == "ego_vehicle")
ego["sensors"] = [s for s in ego["sensors"] if s.get("id") != "rgb_front"]
# The bridge builds the topic as /carla/<role_name>/<id>/image, so `rgb_front` is not
# cosmetic -- brain subscribes to /carla/ego_vehicle/rgb_front/image and any other id
# gives a live camera and a brain that never sees a frame.
ego["sensors"].append({
    "type": "sensor.camera.rgb", "id": "rgb_front",
    "spawn_point": {"x": 1.5, "y": 0.0, "z": 1.5,
                    "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    "image_size_x": 800, "image_size_y": 600, "fov": 90.0,
    # THROTTLED, and this is required: in synchronous mode the bridge services sensor data
    # every tick, and an unthrottled camera starves the control loop that the server is
    # waiting on (0 mpc ticks). Rendering alone does not cause the stall; the camera does.
    # brain's vlm_check_interval is 2.0 s, so anything above ~2 Hz is discarded anyway.
    "sensor_tick": 0.5})
blob["_note"] = ("GENERATED by run_phase_a.sh from objects.json for CUE_SOURCE=vlm. "
                 "Do not edit; it is overwritten every run.")
json.dump(blob, open(dst, "w"), indent=2)
print(f"      derived camera objects from {src.split('/')[-1]} -> rgb_front")
PYCAM
    OBJECTS="$CFG/objects.$TOWN_LC.camera.gen.json"
fi
echo "      sensors: $(basename "$OBJECTS")  (cue_source=$CUE_SOURCE)"
spawn_ego() {
    docker exec -d $DOCKER_ENV "$CONTAINER" bash -lc \
        "source /carla_ws/install/setup.bash && ros2 launch carla_spawn_objects \
         carla_spawn_objects.launch.py \
         objects_definition_file:=$OBJECTS > /tmp/spawn.log 2>&1"
}
ok=0
for attempt in 1 2 3; do
    spawn_ego
    for _ in $(seq 1 12); do
        if $X "source /carla_ws/install/setup.bash && ros2 topic list 2>/dev/null" \
             | grep -q "/carla/$ROLE/odometry"; then ok=1; break; fi
        sleep 2
    done
    [ "$ok" = "1" ] && { echo "      odometry up (spawn attempt $attempt)"; break; }
    echo "      spawn attempt $attempt produced no odometry; retrying"
    $X "pkill -f carla_spawn_objects" >/dev/null 2>&1 || true
    sleep 3
done
[ "$ok" = "1" ] || { echo "      ERROR: no odometry after 3 spawn attempts"; \
                     $X "tail -20 /tmp/spawn.log" || true; exit 1; }

# CONE=1 spawns a traffic cone at the decision junction. The SAME plan tree can be run
# twice with the world changed by a ROS service. Spawning a prop exercises the whole path
# the answer must travel: actor list -> gt_cue_node -> brain's branch pick -> a different
# route on the ground.
#
# SpawnObject takes a ROS pose, the same frame objects.json uses, so no conversion -- see
# carla_gt_bridge.frames.
# Cone poses come from config/cones.<town>.json (one pose per region, derived from the
# region table) rather than hand-typed CARLA coordinates: a cone one region off is
# answered CORRECTLY by the cue oracle about the WRONG place, which reads as a working run.
#
# CONES=2 puts a cone in both regions of cones.<town>.json, CONES=1 only in the last one.
# Prefer CONE_REGIONS (explicit regions) for anything a mission depends on.
CONES_JSON="$WS/src/carla_gt_bridge/config/cones.$TOWN_LC.json"
PROPS_JSON="$WS/src/carla_gt_bridge/config/props.$TOWN_LC.json"

# EXPLICIT REGIONS SPAWN independently of CONES: CONE_REGIONS both sets the launch
# argument gt_cue_node reads and spawns the props. Otherwise a "cone present" world could
# have no cone in it and drive identically to its own control.
#
# Each region brings its OWN blueprint from the json (`type`), because a depth-3 mission
# needs a cone at one junction and a bench at another.
if [ -n "${CONE_REGIONS:-}" ]; then
    echo "      spawning explicit regions: $CONE_REGIONS"
    CONE_REGIONS="$CONE_REGIONS" CONES_JSON="$CONES_JSON" PROPS_JSON="$PROPS_JSON" \
    DEFAULT_TYPE="${PROP_TYPE:-static.prop.constructioncone}" python3 -c "
import json, os
want = [int(x) for x in os.environ['CONE_REGIONS'].split(',') if x.strip()]
pose = {}
for env in ('CONES_JSON', 'PROPS_JSON'):
    fp = os.environ.get(env, '')
    if not os.path.exists(fp):
        continue
    d = json.load(open(fp))
    for c in (d.get('cones') or []) + (d.get('props') or []):
        pose[int(c['region'])] = c
for r in want:
    c = pose.get(r)
    if c is None:
        print('MISSING', r, 0, 0); continue
    print(c.get('type') or os.environ['DEFAULT_TYPE'], c['id'], c['x'], c['y'])" \
    | while read -r ptype id cx cy; do
        if [ "$ptype" = "MISSING" ]; then
            echo "      ERROR: region $id has no pose in cones/props json"; exit 1; fi
        echo "        $ptype at region-pose $id ($cx, $cy)"
        $X "source /carla_ws/install/setup.bash && ros2 service call /carla/spawn_object \
            carla_msgs/srv/SpawnObject \
            \"{type: '$ptype', id: '$id', \
               transform: {position: {x: $cx, y: $cy, z: 0.5}}, \
               attach_to: 0, random_pose: false}\" 2>&1 | grep -E 'response|error' | head -2"
    done
elif [ "${CONES:-0}" != "0" ]; then
    [ -f "$CONES_JSON" ] || { echo "      ERROR: $CONES_JSON missing — run"; \
        echo "        (cones.<town>.json holds one pose per cone region)"; exit 1; }
    N_CONES="${CONES:-0}"
    CONE_REGIONS="${CONE_REGIONS:-$(CONES_JSON="$CONES_JSON" N_CONES="$N_CONES" python3 -c "
import json, os
d = json.load(open(os.environ['CONES_JSON']))
c = d['cones'] if os.environ['N_CONES'] == '2' else d['cones'][-1:]
print(','.join(str(x['region']) for x in c))")}"
    echo "      spawning ${PROP_TYPE:-static.prop.constructioncone} in regions $CONE_REGIONS"
    CONES_JSON="$CONES_JSON" N_CONES="$N_CONES" python3 -c "
import json, os
d = json.load(open(os.environ['CONES_JSON']))
c = d['cones'] if os.environ['N_CONES'] == '2' else d['cones'][-1:]
for x in c: print(x['id'], x['x'], x['y'])" | while read -r id cx cy; do
        $X "source /carla_ws/install/setup.bash && ros2 service call /carla/spawn_object \
            carla_msgs/srv/SpawnObject \
            \"{type: '${PROP_TYPE:-static.prop.constructioncone}', id: '$id', \
               transform: {position: {x: $cx, y: $cy, z: 0.5}}, \
               attach_to: 0, random_pose: false}\" 2>&1 | grep -E 'response|error' | head -2"
    done
fi

echo "[5/6] twist_to_control + mission stack ($MISSION, $MODE)"
docker exec -d $DOCKER_ENV "$CONTAINER" bash -lc \
    "source /carla_ws/install/setup.bash && ros2 run carla_twist_to_control carla_twist_to_control \
     --ros-args -r __ns:=/carla/$ROLE > /tmp/twist.log 2>&1"
sleep 4
DRY=$([ "$MODE" = "drive" ] && echo false || echo true)
# Omitted entirely when empty: `cone_regions:=` with no value is a malformed launch
# argument and kills the whole launch, taking the mission log with it.
CONE_ARG=""
[ -n "${CONE_REGIONS:-}" ] && CONE_ARG="cone_regions:=${CONE_REGIONS}"
PLAN_ARG="plan_file:=$PLAN_FILE"
echo "      driving plan $PLAN_FILE"
docker exec -d $DOCKER_ENV "$CONTAINER" bash -lc \
    "source /carla_ws/install/setup.bash && source /ros_ws/install/setup.bash && \
     ros2 launch carla_gt_bridge mission.launch.py role_name:=$ROLE \
     town:=$TOWN_LC dry_run:=$DRY force_cone:=${FORCE_CONE:-auto} $CONE_ARG $PLAN_ARG \
     mpc_K:=$MPC_K mpc_N:=$MPC_N control_every_n_odom:=$CONTROL_EVERY_N \
     cue_place_first:=${CUE_PLACE_FIRST:-false} guidance:=$GUIDANCE \
     safety_radius:=$SAFETY_RADIUS collision_horizon:=$COLLISION_HORIZON mpc_seed:=$MPC_SEED \
     aim_source:=$AIM_SOURCE pure_pursuit:=$PURE_PURSUIT \
     controller:=$CONTROLLER kappa_max:=$KAPPA_MAX \
     terrain_source:=$TERRAIN_SOURCE terrain_labeller:=$TERRAIN_LABELLER \
     ${FORBID_TERRAIN:+forbid_terrain:=$FORBID_TERRAIN} terrain_decay_m:=${TERRAIN_DECAY_M:-0.0} \
     overlay_every_n:=${OVERLAY_EVERY_N:-0} \
     cam_x:=${CAM_X:-2.425} cam_z:=${CAM_Z:-1.573} \
     cam_fov:=${CAM_FOV:-130.0} cam_pitch_down_deg:=${CAM_PITCH_DOWN:-30.0} \
     arc_len_m:=$ARC_LEN_M fan_size:=$FAN_SIZE fan_segments:=$FAN_SEGMENTS \
     w_obstacle:=$W_OBSTACLE \
     distortion_model:=$DIST_MODEL distortion_jitter_frac:=$DIST_JITTER \
     distortion_surface:=$DIST_SURFACE \
     distortion_jitter_m:=$DIST_JITTER_M distortion_region_rot_deg:=$DIST_ROT \
     distortion_pivot:=$DIST_PIVOT distortion_angle_deg:=$DIST_ANGLE distortion_scale:=$DIST_SCALE distortion_tx:=$DIST_TX distortion_ty:=$DIST_TY \
     > /tmp/mission.log 2>&1"

# RUN_SECONDS IS A CAP, NOT A DURATION. Missions finish in tens of SIM seconds and the
# MPC then holds stop until the clock expires, so poll for completion instead.
# Three guards:
#   MIN_RUN_SECONDS  a completion string in the first seconds (stale log, echoed plan
#                    text) must not end the run -- otherwise this becomes a way to
#                    manufacture completions.
#   SETTLE_SECONDS   teardown must not race the last ticks out of the log, or the tick
#                    count and the outcome disagree.
#   no match => full cap, UNCHANGED. A stalled run still burns its budget and still
#                    classifies BUDGET. This must never turn a timeout into a success.
# EXIT_REASON is echoed so a scorer can tell "finished" from "ran out of clock" without
# inferring it from the tick count.
echo "[6/6] running up to ${RUN_SECONDS}s (exits early on plan completion)"
RUN_START=$(date +%s)
DEADLINE=$(( RUN_START + RUN_SECONDS ))
MIN_RUN="${MIN_RUN_SECONDS:-20}"
SETTLE="${SETTLE_SECONDS:-5}"
# STALL ABORT. The completion poll makes a finishing run cheap; without this, a STUCK
# run would still burn the whole cap.
#
# THIS CAN NEVER MANUFACTURE A SUCCESS: the reason is `stall`, never `complete`, so every
# scorer that keys off EXIT_REASON or off the completion string still calls it a failure.
# It only stops paying for a vehicle that has already stopped moving.
#
# DISPLACEMENT, not string equality: a frozen vehicle still creeps by fractions of a metre,
# so comparing the printed position to the previous one would reset the timer forever.
STALL_SECS="${STALL_SECONDS:-45}"      # 0 disables
STALL_MIN_M="${STALL_MIN_M:-2.0}"
STALL_ANCHOR=""
STALL_T=$RUN_START
EXIT_REASON="cap"
# COPY THE LOGS WHILE THE CONTAINER IS STILL UP. save_logs runs in stop_all, at teardown,
# and a container that died early takes its logs with it -- often the one log that would
# explain the failure. Mid-run copies cost nothing.
snapshot_logs() {
    [ -n "${LOG_DIR:-}" ] || return 0
    mkdir -p "$LOG_DIR"
    for f in mission bridge twist spawn; do
        docker cp "$CONTAINER:/tmp/$f.log" "$LOG_DIR/$f.live.log" >/dev/null 2>&1 || true
    done
}
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    snapshot_logs
    sleep 2
    [ $(( $(date +%s) - RUN_START )) -lt "$MIN_RUN" ] && continue
    if $X "grep -qE 'brain reports plan complete|Navigation plan complete' /tmp/mission.log" 2>/dev/null; then
        sleep "$SETTLE"
        EXIT_REASON="complete"
        break
    fi
    if [ "$STALL_SECS" -gt 0 ]; then
        POS=$($X "grep -oE '\\(-?[0-9.]+,-?[0-9.]+\\) yaw=' /tmp/mission.log | tail -1" 2>/dev/null \
              | grep -oE '\(-?[0-9.]+,-?[0-9.]+\)')
        if [ -n "$POS" ]; then
            if [ -z "$STALL_ANCHOR" ]; then
                STALL_ANCHOR="$POS"; STALL_T=$(date +%s)
            elif awk -v a="$STALL_ANCHOR" -v b="$POS" -v m="$STALL_MIN_M" 'BEGIN{
                    gsub(/[()]/,"",a); gsub(/[()]/,"",b);
                    split(a,A,","); split(b,B,",");
                    d=sqrt((A[1]-B[1])^2+(A[2]-B[2])^2); exit !(d>=m) }'; then
                STALL_ANCHOR="$POS"; STALL_T=$(date +%s)
            elif [ $(( $(date +%s) - STALL_T )) -ge "$STALL_SECS" ]; then
                echo "      STALL: moved <${STALL_MIN_M} m in ${STALL_SECS}s (last ${POS}); aborting"
                EXIT_REASON="stall"
                break
            fi
        fi
    fi
done
echo "      run ended: ${EXIT_REASON} after $(( $(date +%s) - RUN_START ))s"
# CONFIG IN FORCE, in its own section. These lines are printed ONCE at startup and would
# otherwise compete with the per-tick cue lines for the `tail -25` budget and never reach
# console.txt, the only log the harness keeps. Without them a run cannot distinguish
# "the distortion was applied and did not matter" from "the flag did nothing".
echo "----- config in force -----"
$X "grep -E 'MAP JITTERED|SURFACE DISPLACED|guidance=|aim=|safety_radius|mpc_seed' /tmp/mission.log | head -12" || true
echo "----- mission log (tail) -----"
$X "grep -E 'gt_cue_node|cue answers|sighting|advancing|BRANCH|Navigation plan|ERROR|WARN|Traceback|SURFACE DISPLACED|api_key|OPENAI' /tmp/mission.log | tail -25" || true
echo "----- last 12 raw lines -----"
$X "tail -12 /tmp/mission.log" || true
echo "----- GPU -----"
nvidia-smi --query-gpu=memory.used,temperature.gpu,utilization.gpu --format=csv,noheader
