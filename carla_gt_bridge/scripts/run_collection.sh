#!/usr/bin/env bash
# Collect one labelled-corpus bag: server + bridge + sensors + recording + the drive.
#
#   bash scripts/run_collection.sh Town01
#   TOWNS="Town01 Town07" bash scripts/run_collection.sh        # both, in sequence
#
# Produces  carla_data/collect_<town>_<stamp>/  which `scripts/label_frames.py` turns
# into ground truth.
#
# WHAT EACH CHOICE IS FOR:
#
#   the LOCK          Shared with every other CARLA entry point. Without it two servers
#                     can fight over port 2000 (one dies with SIGSEGV), and a load_world
#                     meant for one run can land on another run's server mid-run.
#   quality=Low       At Epic the server can segfault loading a town on an 8 GB GPU.
#                     Low still reports all Town01 buildings: get_environment_objects
#                     reads the object registry, not the LODs.
#   the BRIDGE loads  bridge.py does `if world.get_map().name != town: load_world(...)`,
#   the town          and "Carla/Maps/Town01" never equals "Town01", so if anything else
#                     loads it first the bridge reloads it underneath and
#                     carla_spawn_objects races a world being torn down. It hangs, and it
#                     hangs for the whole run.
#   objects.collect   objects.town01.json declares NO LiDAR and nothing else declares a
#                     SEMANTIC LiDAR, so without this file those topics would not exist
#                     and the bag could never be labelled.
#   no rendering      sensor.lidar.ray_cast is a physics raycast and works with
#                     no_rendering_mode ON. Only cameras need the renderer, and the GT
#                     plan uses no camera.
#   VERIFY THE TOPIC  The recording is checked for all six topics BEFORE the drive. A bag
#   LIST FIRST        missing odometry cannot be labelled at all.
set -euo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${WS:-$HOME/ros2_ws}"
CARLA_PYTHONAPI="${CARLA_PYTHONAPI:-$HOME/carla}"
DATA="${DATA:-$HOME/carla_data}"
SERVER="${SERVER:-carla_collect_server}"
CONTAINER="${CONTAINER:-carla_collect_bridge}"
LOCK_FILE="${LOCK_FILE:-/tmp/carla_gt_bridge.lock}"
TOWNS="${TOWNS:-${1:-Town01}}"
RUN_SECONDS="${RUN_SECONDS:-1500}"
SPEED_KMH="${SPEED_KMH:-28.8}"
# Smoke test: drive only the first N targets. Confirms the topic list, that BasicAgent
# actually moves THIS ego, and that the bag is labelable, in a fraction of the time.
MAX_TARGETS="${MAX_TARGETS:-0}"

# /carla/world_info, NOT /carla/map: this workspace's bridge publishes the OpenDRIVE as
# carla_msgs/CarlaWorldInfo on /carla/world_info and never publishes /carla/map at all.
# Other bridge builds may differ; the pre-drive topic check catches that.
# SYNCHRONOUS, WITH fixed_delta_seconds == 1 / rotation_frequency. NOT optional.
# A CARLA ray-cast LiDAR accumulates points between world ticks and publishes whatever
# it has swept. Under the async bridge the world free-runs faster than the sensor period,
# so each message carries only part of a revolution (roughly half azimuth coverage).
#
# It breaks both axes at once, and neither failure announces itself:
#   * the 72-bin scan the clusterer consumes is half max-range misses, and WHICH half
#     alternates -- so HDBSCAN would separate frames by sweep phase, not by scene;
#   * the semantic LiDAR is half-blind too, so `building_left` is missed whenever the
#     sweep covered the right, producing enclosure-label chatter that looks like
#     sensing dropout.
DELTA="${DELTA:-0.1}"          # must equal 1 / rotation_frequency in objects.collect.json

OBJECTS="${OBJECTS:-$PKG/config/objects.collect.json}"
TOPICS=(/carla/ego_vehicle/lidar /carla/ego_vehicle/semantic_lidar
        /carla/ego_vehicle/odometry /carla/world_info /clock /tf)
# EXTRA_TOPICS lets a variant rig add its own streams without editing the list. They are
# verified in [5/6] exactly like the rest, so a typo'd topic refuses the run rather than
# recording a bag with a hole in it.
for t in ${EXTRA_TOPICS:-}; do TOPICS+=("$t"); done

# DISK GUARD. Bags grow fast (tens of MB/s), and a full root filesystem loses more than
# the bag. MIN_FREE_GB is checked before the drive and again while it runs.
MIN_FREE_GB="${MIN_FREE_GB:-8}"
free_gb() { df -BG --output=avail $HOME | tail -1 | tr -dc '"'"'0-9'"'"'; }

exec 9>"$LOCK_FILE"
if ! flock -w "${LOCK_WAIT:-3600}" 9; then
    echo "another CARLA run holds $LOCK_FILE — refusing to start."
    exit 75
fi
echo "[lock] held $LOCK_FILE (pid $$)"

stop_all() {
    [ -n "${GUARD_PID:-}" ] && kill "$GUARD_PID" 2>/dev/null || true
    docker exec "$CONTAINER" bash -lc "pkill -INT -f 'ros2 bag record'" >/dev/null 2>&1 || true
    sleep 3
    docker rm -f "$SERVER" "$CONTAINER" >/dev/null 2>&1 || true
    echo "[cleanup] containers removed"
}
trap stop_all EXIT INT TERM

for TOWN in $TOWNS; do
    TOWN_LC="$(echo "$TOWN" | tr '[:upper:]' '[:lower:]')"
    ROUTE="$PKG/reports/collection/drivable.$TOWN_LC.json"
    STAMP="$(date +%Y%m%d_%H%M%S)"
    BAG="collect_${TOWN_LC}_${STAMP}"

    if [ ! -f "$ROUTE" ]; then
        echo "no route for $TOWN ($ROUTE). Run plan_drivable_route.py first."; exit 1
    fi

    echo; echo "############ $TOWN -> $DATA/$BAG ############"
    docker rm -f "$SERVER" "$CONTAINER" >/dev/null 2>&1 || true

    echo "[1/6] server"
    # NOT --rm: a container that dies on startup would take its logs with it, presenting
    # as "no such container" instead of an error message. stop_all removes it explicitly.
    docker run -d --privileged --gpus all --net=host --name "$SERVER" \
        carlasim/carla:0.9.14 \
        /bin/bash -c "./CarlaUE4.sh -RenderOffScreen -quality-level=${QUALITY:-Low} -nosound -fps 10" >/dev/null
    # BOUNDED WAIT. An unbounded `until (echo > /dev/tcp/...)` loop turns a server crash on
    # startup into a silent hang; this one gives up and prints the container's logs.
    up=0
    for i in $(seq 1 60); do
        if (echo > /dev/tcp/127.0.0.1/2000) 2>/dev/null; then up=1; break; fi
        if ! docker ps --format '{{.Names}}' | grep -qx "$SERVER"; then
            echo "      server container died during startup; last output:"
            docker logs "$SERVER" 2>&1 | tail -15 || echo "      (container already removed by --rm)"
            exit 1
        fi
        sleep 2
    done
    [ "$up" = 1 ] || { echo "      port 2000 never opened after 120s"; exit 1; }
    sleep 5
    echo "      up"

    echo "[2/6] bridge container"
    docker run -d --name "$CONTAINER" --net=host \
        -v "$WS":/ros_ws -v "$CARLA_PYTHONAPI":/carla -v "$DATA":/data --user=root \
        carla-ros-bridge-dev:latest tail -f /dev/null >/dev/null
    X="docker exec $CONTAINER bash -lc"

    # BasicAgent imports shapely and GlobalRoutePlanner imports networkx; the bridge
    # image ships neither, and the failure only appears AFTER the world is up and the
    # recorder is running. Install once, before anything expensive.
    $X "python3 -c 'import shapely, networkx' 2>/dev/null || \
        pip install -q shapely networkx" >/dev/null 2>&1 || true

    echo "[3/6] bridge (it loads the town; nothing else may)"
    docker exec -d "$CONTAINER" bash -lc \
        "source /carla_ws/install/setup.bash && ros2 launch carla_ros_bridge \
         carla_ros_bridge.launch.py town:=$TOWN timeout:=60 \
         synchronous_mode:=True \
         synchronous_mode_wait_for_vehicle_control_command:=False \
         fixed_delta_seconds:=$DELTA \
         > /tmp/bridge.log 2>&1"
    for _ in $(seq 1 60); do
        if $X "source /carla_ws/install/setup.bash && ros2 topic list 2>/dev/null" \
             | grep -q "/carla/world_info"; then break; fi
        sleep 2
    done
    echo "      bridge up"

    echo "[4/6] sensors ($(basename "$OBJECTS"))"
    docker exec -d "$CONTAINER" bash -lc \
        "source /carla_ws/install/setup.bash && ros2 launch carla_spawn_objects \
         carla_spawn_objects.launch.py \
         objects_definition_file:=/ros_ws/src/carla_gt_bridge/config/$(basename "$OBJECTS") \
         > /tmp/spawn.log 2>&1"
    ok=0
    for _ in $(seq 1 30); do
        if $X "source /carla_ws/install/setup.bash && ros2 topic list 2>/dev/null" \
             | grep -q "/carla/ego_vehicle/semantic_lidar"; then ok=1; break; fi
        sleep 2
    done
    if [ "$ok" != 1 ]; then
        echo "      semantic_lidar topic never appeared — spawn failed:"
        $X "tail -20 /tmp/spawn.log" || true
        exit 1
    fi
    echo "      ego + 2 lidars publishing"

    echo "      checking the LiDAR delivers FULL sweeps"
    $X "source /carla_ws/install/setup.bash && python3 -c \"
import numpy as np, rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
rclpy.init(); n=Node('sweepcheck'); got=[]
def cb(m):
    a=point_cloud2.read_points(m, skip_nans=True)
    az=np.degrees(np.arctan2(np.asarray(a['y']),np.asarray(a['x'])))%360.0
    got.append(100.0*len(np.unique((az//5).astype(int)))/72)
n.create_subscription(PointCloud2,'/carla/ego_vehicle/lidar',cb,10)
import time; t0=time.time()
while time.time()-t0<15 and len(got)<8: rclpy.spin_once(n,timeout_sec=0.5)
print('      azimuth coverage: %.0f%% over %d scans' % (np.median(got) if got else 0, len(got)))
assert got and np.median(got)>90, 'PARTIAL SWEEPS - the bridge is not synchronous with the sensor period'
\"" || { echo "      LiDAR is not delivering full sweeps; refusing to collect"; exit 1; }

    # THE FOV MUST BE THE ONE IN objects.collect.json, measured on the wire.
    # Changing lower_fov/upper_fov in the config and having it silently not apply is
    # A silent failure: the drive completes, the bag looks normal,
    # and every band scored against it describes the old rig.
    #
    # WHICH STATISTIC. p1 and p99 cannot share one test: p99 is scene-dependent at the TOP
    # of an upward FOV, because the highest beams point at open sky and return nothing -- the
    # percentile then sits below nominal by however much sky happens to be in view, which
    # would refuse a correct rig. A near-horizontal upper bound (e.g. +2.0) hides this, since
    # such a beam still strikes something at range in any town. So the two bounds get
    # different tests:
    #   lower  p1, tight. Downward beams always hit the road, so it is stable, and it is the
    #          DECISIVE one: a config that was ignored reads the default lower FOV here.
    #   upper  max, loose and one-sided. As soon as any top-beam ray returns at any
    #          azimuth the maximum reaches the top beam, so it cannot be starved the way
    #          p99 is, and a rig still on +2.0 misses by 10 degrees.
    echo "      checking the LiDAR FOV matches objects.collect.json"
    EXP_LO=$(python3 -c "import json;d=json.load(open('$OBJECTS'));print([s['lower_fov'] for o in d['objects'] for s in o.get('sensors',[]) if s['type']=='sensor.lidar.ray_cast' and s['id']=='lidar'][0])")
    EXP_HI=$(python3 -c "import json;d=json.load(open('$OBJECTS'));print([s['upper_fov'] for o in d['objects'] for s in o.get('sensors',[]) if s['type']=='sensor.lidar.ray_cast' and s['id']=='lidar'][0])")
    $X "source /carla_ws/install/setup.bash && python3 -c \"
import numpy as np, rclpy, time
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
lo, hi = float('$EXP_LO'), float('$EXP_HI')
rclpy.init(); n=Node('fovcheck'); got=[]
def cb(m):
    a=point_cloud2.read_points(m, skip_nans=True)
    x,y,z=np.asarray(a['x']),np.asarray(a['y']),np.asarray(a['z'])
    r=np.hypot(x,y); k=r>1.0
    if k.sum()<200: return
    el=np.degrees(np.arctan2(z[k],r[k]))
    got.append((np.percentile(el,1),el.max()))
n.create_subscription(PointCloud2,'/carla/ego_vehicle/lidar',cb,10)
t0=time.time()
while time.time()-t0<15 and len(got)<8: rclpy.spin_once(n,timeout_sec=0.5)
assert got, 'no scans arrived'
g=np.array(got); mlo,mhi=np.median(g[:,0]),np.median(g[:,1])
print('      elevation p1 %+.1f (config %+.1f), max %+.1f (config %+.1f)' % (mlo,lo,mhi,hi))
assert abs(mlo-lo)<1.0, (
    'LOWER FOV ON THE WIRE DOES NOT MATCH THE CONFIG - the spawner ignored it')
assert mhi<=hi+0.5, 'returns ABOVE the configured upper_fov - not the rig we asked for'
assert mhi>=hi-3.0, (
    'the top of the FOV never returns - upper_fov was ignored, or the ego is boxed in '
    'with no sky-facing structure at any azimuth')
\"" || { echo "      LiDAR FOV is not what the config asks for; refusing to collect"; exit 1; }

    AVAIL=$(free_gb)
    echo "      disk: ${AVAIL} GB free (guard aborts the drive below ${MIN_FREE_GB} GB)"
    [ "$AVAIL" -gt "$MIN_FREE_GB" ] || {
        echo "      only ${AVAIL} GB free; refusing to start a recording"; exit 1; }

    echo "[5/6] verifying every topic exists BEFORE driving"
    LIST="$($X "source /carla_ws/install/setup.bash && ros2 topic list 2>/dev/null")"
    miss=0
    for t in "${TOPICS[@]}"; do
        if echo "$LIST" | grep -qx "$t"; then echo "      ok   $t"
        else echo "      MISS $t"; miss=1; fi
    done
    [ "$miss" = 0 ] || { echo "a topic is missing; the bag would be unlabelable"; exit 1; }

    echo "      recording -> /data/$BAG"
    docker exec -d "$CONTAINER" bash -lc \
        "source /carla_ws/install/setup.bash && cd /data && \
         ros2 bag record -o $BAG ${TOPICS[*]} > /tmp/bag.log 2>&1"
    sleep 5

    # IN-FLIGHT half of the disk guard. A pre-drive size estimate is a guess (bag growth
    # rate varies by town and rig); this is a measurement. SIGINT, not SIGKILL, so ros2 bag
    # closes the db3 and writes metadata.yaml and a truncated bag stays readable.
    ( while sleep 20; do
        docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" || break
        A=$(free_gb)
        if [ "${A:-999}" -le "$MIN_FREE_GB" ]; then
            echo "      DISK GUARD: ${A} GB free <= ${MIN_FREE_GB} GB — stopping the recorder"
            docker exec "$CONTAINER" bash -lc "pkill -INT -f 'ros2 bag record'" || true
            break
        fi
      done ) &
    GUARD_PID=$!

    # THE TOWN, AS A SIDECAR FILE. `/carla/world_info` is a carla_msgs/CarlaWorldInfo,
    # and carla_msgs is built only in the Humble carla_ws — the HOST runs Jazzy and
    # cannot deserialize it, so `label_frames.py` could record the topic and still be
    # unable to read it. Writing the OpenDRIVE straight out of the CARLA API sidesteps
    # the message entirely and makes the bag self-describing with no ROS dependency.
    $X "python3 -c \"import sys; sys.path.insert(0,'/carla'); import carla
c = carla.Client('127.0.0.1', 2000); c.set_timeout(30.0)
open('/data/$BAG/map.xodr','w').write(c.get_world().get_map().to_opendrive())
print('      wrote map.xodr', c.get_world().get_map().name)\"" || \
        echo "      WARNING: could not write map.xodr"

    echo "[6/6] drive"
    $X "source /carla_ws/install/setup.bash && python3 \
        /ros_ws/src/carla_gt_bridge/scripts/drive_collection.py \
        --route /ros_ws/src/carla_gt_bridge/reports/collection/drivable.$TOWN_LC.json \
        --speed-kmh $SPEED_KMH --run-seconds $RUN_SECONDS \
        --max-targets $MAX_TARGETS --teleport-log /data/'$BAG'/teleports.json" || echo "      drive exited non-zero"

    echo "      stopping recorder"
    $X "pkill -INT -f 'ros2 bag record'" || true
    sleep 5
    docker rm -f "$SERVER" "$CONTAINER" >/dev/null 2>&1 || true

    # The bridge container runs as root, so everything it writes is root-owned in the
    # host tree (its logs too). Without this the bags cannot be deleted without a
    # container.
    docker run --rm -v "$DATA":/data --entrypoint chown carla-ros-bridge-dev:latest \
        -R "$(id -u):$(id -g)" "/data/$BAG" 2>/dev/null || true

    echo "      bag: $DATA/$BAG"
    python3 "$PKG/scripts/label_frames.py" --bag "$DATA/$BAG" --check || true
done
