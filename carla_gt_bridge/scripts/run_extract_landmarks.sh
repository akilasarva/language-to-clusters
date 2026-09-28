#!/usr/bin/env bash
# Bring up a CARLA server under the shared lock, extract the map-native landmark table,
# tear it down. The lock is not optional: two servers on port 2000 can crash one with
# SIGSEGV.
set -u
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TOWN="${TOWN:-town05}"
SERVER=carla_landmark_server
CONTAINER=carla_landmark_bridge
LOCK_FILE="${LOCK_FILE:-/tmp/carla_gt_bridge.lock}"
cleanup() { echo "[stop] removing containers"; docker rm -f "$SERVER" "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
exec 9>"$LOCK_FILE"
flock -w "${LOCK_WAIT:-1800}" 9 || { echo "another CARLA run holds $LOCK_FILE"; exit 1; }
echo "[lock] held $LOCK_FILE (pid $$)"
docker rm -f "$SERVER" "$CONTAINER" >/dev/null 2>&1 || true
echo "[1/3] server"
docker run -d --privileged --gpus all --net=host --name "$SERVER" carlasim/carla:0.9.14 \
    /bin/bash -c "./CarlaUE4.sh -RenderOffScreen -quality-level=Low -nosound -fps 10" >/dev/null
for i in $(seq 1 90); do
    (echo > /dev/tcp/127.0.0.1/2000) 2>/dev/null && break
    docker ps --format '{{.Names}}' | grep -qx "$SERVER" || { echo "server died"; docker logs "$SERVER" 2>&1 | tail -6; exit 2; }
    sleep 2
done
(echo > /dev/tcp/127.0.0.1/2000) 2>/dev/null || { echo "port never opened"; exit 2; }
echo "      up"
echo "[2/3] bridge container (matching PythonAPI: host carla is 0.10.0, server 0.9.14)"
docker run -d --name "$CONTAINER" --net=host -v "$WS":/ros_ws -v "$HOME/carla":/carla \
    --user=root carla-ros-bridge-dev:latest tail -f /dev/null >/dev/null
echo "[3/3] extract"
# PYTHONUNBUFFERED IS NOT OPTIONAL HERE. python block-buffers stdout when it is a PIPE, and
# this exec is always piped. A script that EXITS flushes, but one that hangs or is killed
# at teardown loses everything it printed, and the harness reports nothing wrong.
# run_phase_a.sh sets this for the same reason.
docker exec -e PYTHONUNBUFFERED=1 "$CONTAINER" bash -lc \
  "python3 -u /ros_ws/src/carla_gt_bridge/scripts/${EXTRACT:-extract_traffic_lights.py} --town $TOWN"
