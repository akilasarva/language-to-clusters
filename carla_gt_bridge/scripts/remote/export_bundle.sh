#!/usr/bin/env bash
# Build a transferable bundle of this CARLA setup. Run on THIS machine.
#
# WHAT HAS TO TRAVEL, AND WHY IT IS AWKWARD.
#
#   carlasim/carla:0.9.14        public on Docker Hub -- the remote pulls it, no transfer
#   carla-ros-bridge-dev:latest  15.7 GB, hand-built, NO DOCKERFILE ANYWHERE. It cannot be
#                                rebuilt from source, so it must be shipped as a saved
#                                image. This is the whole reason a bundle exists rather
#                                than a git clone plus a build.
#   the workspace                8.4 GB on disk, 257 MB compressed once three
#                                exclusions are applied, each a judgement call:
#                                  bev_pipeline/datasets   3.0 GB, only one module is
#                                                          imported (taxonomy_export)
#                                  debug_logs              400 MB of traces from PAST
#                                                          runs; a machine doing new ones
#                                                          does not need the history
#                                  old_dgppo               34 MB, superseded
#                                Pass --with-history to keep debug_logs, which the
#                                maneuver scorer's fallback reads for older runs that predate
#                                run.jsonl -- needed only if you score ON the remote.
#   $HOME/carla         119 MB, the CARLA PythonAPI the bridge mounts at /carla
#
#   NOT bundled, recreated on the far side instead:
#     the venv       absolute paths are baked into a venv; bootstrap.sh pip-installs it
#     Touchdown      re-fetched from lil-lab/touchdown, 9,325 instructions
#     build/install  colcon rebuilds them; shipping stale ones causes confusing failures
#
# Usage:  bash export_bundle.sh [--with-history] [/path/to/output/dir]
#
# Output is ~6 GB: the image dominates and everything else is noise beside it.
set -euo pipefail

WITH_HISTORY=0
if [ "${1:-}" = "--with-history" ]; then WITH_HISTORY=1; shift; fi
OUT="${1:-$HOME/carla_bundle}"
WS="$HOME/ros2_ws"
BRIDGE_IMAGE="carla-ros-bridge-dev:latest"

mkdir -p "$OUT"
echo "bundling into $OUT"

# --- preflight: fail before spending 20 minutes on a save that will not fit -----------
need_gb=25
have_gb=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc '0-9')
if [ "${have_gb:-0}" -lt "$need_gb" ]; then
  echo "  need ~${need_gb}G free in $OUT, have ${have_gb}G" >&2; exit 1
fi
docker image inspect "$BRIDGE_IMAGE" >/dev/null 2>&1 || {
  echo "  $BRIDGE_IMAGE not present -- nothing to export" >&2; exit 1; }

# --- the image: the big one -----------------------------------------------------------
echo "[1/4] saving $BRIDGE_IMAGE (15.7 GB uncompressed; expect 10-20 min)"
docker save "$BRIDGE_IMAGE" | gzip -1 > "$OUT/bridge-image.tar.gz"

# --- source: only what a run imports --------------------------------------------------
echo "[2/4] workspace source"
HIST_EXCLUDE=(--exclude='src/dgppo_ros_node_pkg/dgppo_ros_node_pkg/debug_logs')
[ "$WITH_HISTORY" = 1 ] && HIST_EXCLUDE=() && echo "      (keeping debug_logs: +400 MB)"
tar czf "$OUT/ros2_ws_src.tar.gz" -C "$WS" \
  --exclude='*/__pycache__' --exclude='*.pyc' \
  --exclude='src/bev_pipeline/datasets' --exclude='src/bev_pipeline/reports' \
  --exclude='src/dgppo_ros_node_pkg/dgppo_ros_node_pkg/old_dgppo' \
  --exclude='src/*/reports/runs' "${HIST_EXCLUDE[@]}" \
  src/carla_gt_bridge src/brain src/nl_planner src/dgppo_ros_node_pkg \
  src/baselines src/clustering src/proposals src/bev_pipeline

# --- the PythonAPI the bridge container mounts ----------------------------------------
echo "[3/4] CARLA PythonAPI"
tar czf "$OUT/carla_pythonapi.tar.gz" -C "$HOME" carla

# --- manifest: what this is and how to verify it arrived intact -----------------------
echo "[4/4] manifest"
{
  echo "bundle built $(date -Iseconds) on $(hostname)"
  echo "bridge image : $BRIDGE_IMAGE"
  docker image inspect "$BRIDGE_IMAGE" --format '  id {{.Id}}' 2>/dev/null
  echo "git HEAD     : $(git -C "$WS/src" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo
  echo "sha256:"
  (cd "$OUT" && sha256sum ./*.tar.gz)
} > "$OUT/MANIFEST.txt"

echo
du -sh "$OUT"/*.tar.gz
cat "$OUT/MANIFEST.txt"
echo
echo "copy the directory over, then run bootstrap.sh on the far side:"
echo "  rsync -avP --partial $OUT/ REMOTE:~/carla_bundle/"
