#!/usr/bin/env bash
# Preflight the collection against a live server, under the SAME lock every other
# CARLA entry point takes, with ONE SERVER PER TOWN.
#
# WHY THE LOCK IS NOT OPTIONAL. Without it this preflight can start a second server while
# another run owns `carla_server`: the two contend for port 2000, one dies with SIGSEGV,
# and a `load_world('Town01')` meant for the preflight can land on the other run's server
# mid-run.
#
# WHY ONE SERVER PER TOWN. A second `load_world` in the same server lifetime can kill the
# server on an 8 GB GPU, and every later step then inherits a dead simulator. Restarting
# costs ~15 s per town and turns a silent partial result into a complete one.
#
# QUALITY=Low: at Epic the server can segfault loading a town. Low still reports all
# buildings, because `get_environment_objects` reads the level's object registry, not its
# LODs.
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="${SERVER:-carla_preflight}"
TOWNS="${TOWNS:-Town01 Town05 Town07 Town10HD}"
LOCK_FILE="${LOCK_FILE:-/tmp/carla_gt_bridge.lock}"
OUT_DIR="$PKG/reports/collection"

exec 9>"$LOCK_FILE"
if ! flock -w "${LOCK_WAIT:-0}" 9; then
    echo "another CARLA run holds $LOCK_FILE — refusing to start. Wait for it."
    exit 75
fi
echo "[lock] held $LOCK_FILE (pid $$)"
cleanup() { docker rm -f "$SERVER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

start_server() {
    docker rm -f "$SERVER" >/dev/null 2>&1 || true
    docker run --rm -d --privileged --gpus all --net=host --name "$SERVER" \
        carlasim/carla:0.9.14 \
        /bin/bash -c "./CarlaUE4.sh -RenderOffScreen -quality-level=${QUALITY:-Low} -nosound -fps 10" >/dev/null
    for i in $(seq 1 60); do
        if docker run --rm --net=host --entrypoint python3 carla-ros-bridge-dev:latest \
            -c "import carla; carla.Client('127.0.0.1',2000).get_world()" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    echo "server did not come up"; return 1
}

# The client must be 0.9.14: conda's is 0.10.0 and load_world() aborts the process with
# an uncaught std::exception on a version mismatch.
run_py() {
    docker run --rm --net=host -v "$PKG":/pkg --entrypoint python3 \
        carla-ros-bridge-dev:latest /pkg/scripts/preflight_collection.py "$@" 2>&1 \
        | grep -v "^WARNING" || true
}

PARTS=()
for T in $TOWNS; do
    echo; echo "=== $T ==="
    start_server
    run_py --towns "$T" --skip-probe --skip-tick-cost \
           --out "/pkg/reports/collection/preflight.$T.json"
    PARTS+=("$OUT_DIR/preflight.$T.json")
done

echo; echo "=== Town01 stock probe + tick cost (fresh server) ==="
start_server
run_py --towns --tick-town Town01 --out "/pkg/reports/collection/preflight.extras.json"
PARTS+=("$OUT_DIR/preflight.extras.json")

echo; echo "=== merge ==="
MERGE=()
for p in "${PARTS[@]}"; do MERGE+=("/pkg/reports/collection/$(basename "$p")"); done
run_py --merge "${MERGE[@]}" --out /pkg/reports/collection/preflight.json
