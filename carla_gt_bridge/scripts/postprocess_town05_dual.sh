#!/usr/bin/env bash
# Extract BOTH rigs from the Town05 dual bag and label once. Waits for the drive to end.
#
# One label CSV, not two: topology comes from the .xodr plus odometry and is
# sensor-independent, so both rigs join it by stamp_ns. Enclosure in that CSV describes
# the NEW rig only -- the semantic LiDAR sits on the new FOV -- so do not compare
# enclosure across the pair; a FOV change alone shifts enclosure label counts.
# NOT `set -u`: sourcing a ROS setup.bash under `set -u` dies on
# AMENT_TRACE_SETUP_FILES being unbound, and with stderr silenced the script exits
# before its first echo with no message at all.
set -o pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BAG="${BAG:-$HOME/carla_data/collect_town05_20260923_003643}"
WAIT_PID="${WAIT_PID:-}"
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

if [ -n "$WAIT_PID" ]; then
    echo "[post] waiting for the drive (pid $WAIT_PID)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
# the recorder closes the db3 and writes metadata.yaml on its way out
for _ in $(seq 1 60); do [ -f "$BAG/metadata.yaml" ] && break; sleep 5; done
echo "[post] $(date -Is) bag $(du -sh "$BAG" | cut -f1)"

for pair in "lidar:newfov" "lidar_oldfov:oldfov"; do
    T="${pair%%:*}"; N="${pair##*:}"
    OUT="$HOME/carla_data/town05_dual_${N}/town05_dual_${N}_pcds"
    echo "[post] extracting $N from /carla/ego_vehicle/$T"
    python3 "$PKG/scripts/bag_to_pcd.py" --bag "$BAG" \
        --topic "/carla/ego_vehicle/$T" --out "$OUT" 2>&1 | tail -1
done

echo "[post] labelling (topology is sensor-independent; one CSV for both rigs)"
python3 "$PKG/scripts/label_frames.py" --bag "$BAG" \
    --out "$PKG/reports/frame_labels/town05_dual.csv" 2>&1 | tail -6
echo "[post] $(date -Is) done"
df -h $HOME | tail -1
