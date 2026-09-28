#!/usr/bin/env bash
# Drive `third_right` with LiDARs at several heights and record a GT-labellable bag.
#
# WHY THIS IS STANDALONE, not a flag in run_phase_a.sh: the only thing this needs from
# run_phase_a.sh is that DOCKER_ENV forwards LIDAR_HEIGHTS and SEMANTIC_LIDAR into the
# container, which is checked below rather than assumed -- a host export does NOT reach
# lane_spawn.py.
#
# WHY third_right: a straight-line bag is almost entirely `path` / `open_space`, so a
# constant predictor scores near-perfectly on it and it cannot evaluate a classifier.
# third_right crosses THREE junctions (63, 60, 59) on a validated plan, which is where
# the class diversity is.
#
# WHY ONE BAG WITH FOUR LIDARS rather than four runs: same tick, same pose, same world,
# so mount height is the only variable. Separate runs would confound it with the
# trajectory, since same-seed runs can diverge.
set -u
cd $HOME/ros2_ws/src

OUT=${OUT:-carla_gt_bridge/reports/gt_collect}
HOSTOUT=$HOME/ros2_ws/src/$OUT
BAG_IN_C=/ros_ws/src/$OUT/bag
SECS=${SECS:-200}

# --- refuse to trample a run in flight -------------------------------------------------
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qE 'carla|humble'; then
    echo "REFUSING: CARLA containers are already up:"
    docker ps --format '   {{.Names}}\t{{.Status}}' | sed 's/^/   /'
    echo "   run_phase_a.sh does 'docker rm -f' at startup and tears down on EXIT, so"
    echo "   starting now would kill the other run and its teardown would kill this one."
    exit 3
fi

# --- the forwarding this depends on ----------------------------------------------------
if ! grep -q 'e LIDAR_HEIGHTS=' carla_gt_bridge/scripts/run_phase_a.sh; then
    echo "REFUSING: run_phase_a.sh does not forward LIDAR_HEIGHTS into the container."
    echo "   Without it lane_spawn.py sees nothing and spawns ONE lidar at the default"
    echo "   height, and the bag looks fine while containing the wrong thing."
    exit 4
fi

# The recorder runs as root inside the container, so its bag is root-owned and a host
# `rm -rf` fails with Permission denied. A leftover `bag/` directory would make the "did
# the recorder start" check below pass on the PREVIOUS run's directory, so remove it as
# root, through the same mount the recorder wrote it through.
docker run --rm -v $HOME/ros2_ws:/ros_ws --entrypoint bash \
    carla-ros-bridge-dev:latest -lc "rm -rf /ros_ws/src/$OUT" >/dev/null 2>&1 || true
rm -rf "$HOSTOUT" 2>/dev/null || true
mkdir -p "$HOSTOUT"
if [ -e "$HOSTOUT/bag" ]; then
    echo "REFUSING: $HOSTOUT/bag still exists after cleanup -- the start check below"
    echo "   would pass on it and a failed recording would read as a successful one."
    exit 8
fi
export LIDAR_HEIGHTS="${LIDAR_HEIGHTS:-2.0,1.0,0.6,0.3}"
export SEMANTIC_LIDAR=true
echo "heights: $LIDAR_HEIGHTS   semantic: on   route: third_right from region ${START:-0}   $(date -Is)"

setsid nohup env LIDAR_HEIGHTS="$LIDAR_HEIGHTS" SEMANTIC_LIDAR=true CONTROLLER=region \
  timeout 900 python3 carla_gt_bridge/scripts/drive_english.py \
    --english "go straight through two intersections then turn right at the third" \
    --start "${START:-0}" --town town05 --arm full \
    --plan carla_gt_bridge/reports/runs/ms_ordinal/third_right.plan.json \
    --cone-regions 60 --run-seconds "$SECS" --log-dir "$OUT/drive" \
  > "$HOSTOUT/drive.log" 2>&1 < /dev/null &
MISSION=$!

# Wait for the sensors to EXIST, not for a process to be alive (verify a
# background job by the output it created).
TOPICS=""
for i in $(seq 1 180); do
    T=$(docker exec humble_dev_with_code bash -lc \
          "source /carla_ws/install/setup.bash 2>/dev/null; ros2 topic list 2>/dev/null" 2>/dev/null)
    if echo "$T" | grep -q '/carla/ego_vehicle/semantic_lidar'; then
        TOPICS=$(echo "$T" | grep -E '/carla/ego_vehicle/(lidar|lidar_h[0-9]+|semantic_lidar|odometry)$|^/carla/map$' | tr '\n' ' ')
        echo "sensors up after ${i}s:"; for t in $TOPICS; do echo "   $t"; done
        break
    fi
    sleep 1
done
if [ -z "$TOPICS" ]; then
    echo "FAILED: semantic_lidar never appeared -- LIDAR_HEIGHTS/SEMANTIC_LIDAR did not reach lane_spawn.py"
    kill "$MISSION" 2>/dev/null || true; exit 5
fi
n_h=$(echo "$TOPICS" | tr ' ' '\n' | grep -c 'lidar_h')
echo "   $n_h height-tagged lidars present"

# ROSBAG2 IS BAKED INTO THE IMAGE, and it must NOT be installed here.
#
# Installing it between sensors-up and recording puts ~60 s on the critical path, while
# the mission drives `third_right` in ~12 s of WALL time: the bag would capture only a
# parked car at its final pose, and label_frames.py would label every frame `exit`.
# A symptom of this is a pose statistic whose median equals its p95 (the quantity never
# varied) while the run itself reports COMPLETED.
#
# ros-humble-rosbag2 was committed into carla-ros-bridge-dev:latest; the previous image
# is tagged carla-ros-bridge-dev:pre-rosbag2 to roll back to.
if ! docker exec humble_dev_with_code bash -lc \
      "source /opt/ros/humble/setup.bash && ros2 bag --help" >/dev/null 2>&1; then
    echo "FAILED: ros2 bag missing from the image. Do NOT install it here -- that puts"
    echo "   ~60 s on the critical path and the mission finishes in ~12 s wall."
    echo "   Rebuild the image instead (see the comment above)."
    kill "$MISSION" 2>/dev/null || true; exit 6
fi
echo "   ros2 bag available (from image)"

# The recorder's log goes to the MOUNTED tree, not /tmp: the container is torn down at the
# end of every run and takes /tmp with it, including any error message.
docker exec -d humble_dev_with_code bash -lc \
  "source /opt/ros/humble/setup.bash && source /carla_ws/install/setup.bash && \
   cd /ros_ws/src/$OUT && ros2 bag record -o $BAG_IN_C $TOPICS \
   > /ros_ws/src/$OUT/bagrec.log 2>&1"

# VERIFY IT STARTED, by the directory it must create -- never by pgrep.
for i in $(seq 1 30); do [ -d "$HOSTOUT/bag" ] && break; sleep 1; done
if [ ! -d "$HOSTOUT/bag" ]; then
    echo "FAILED: recorder created no bag directory after 30 s. Its log:"
    sed -n '1,20p' "$HOSTOUT/bagrec.log" 2>/dev/null || echo "   (no log written)"
    kill "$MISSION" 2>/dev/null || true; exit 7
fi
echo "recording -> $HOSTOUT/bag"

sleep "$SECS"
docker exec humble_dev_with_code bash -lc "pkill -INT -f 'ros2 bag record'" 2>/dev/null || true
sleep 8
kill "$MISSION" 2>/dev/null || true

echo "=== $(date -Is) done; bag:"
ls -la "$HOSTOUT/bag" 2>&1 | tail -5
du -sh "$HOSTOUT/bag" 2>/dev/null
# A bag that exists but holds nothing is the same failure wearing a directory.
SZ=$(du -sk "$HOSTOUT/bag" 2>/dev/null | cut -f1)
if [ -z "$SZ" ] || [ "$SZ" -lt 1000 ]; then
    echo "SUSPECT: bag is ${SZ:-0} KB -- expected tens of MB for 200 s of 5 lidars."
    sed -n '1,20p' "$HOSTOUT/bagrec.log" 2>/dev/null
fi
