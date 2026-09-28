#!/usr/bin/env bash
set -o pipefail   # NOT -u: ROS setup.bash dies under it
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[train] waiting for postprocess (pid ${WAIT_PID:-none})"
while kill -0 "${WAIT_PID:-1}" 2>/dev/null; do sleep 30; done
CSV="$PKG/reports/frame_labels/town05_dual.csv"
for _ in $(seq 1 40); do [ -s "$CSV" ] && break; sleep 15; done
[ -s "$CSV" ] || { echo "[train] no $CSV -- postprocess did not produce labels"; exit 1; }
echo "[train] $(date -Is) labels present ($(wc -l < "$CSV") rows); training"
~/miniconda3/bin/python "$PKG/scripts/train_phase_town05.py" 2>&1 | grep -v Warning
echo "[train] $(date -Is) done"
