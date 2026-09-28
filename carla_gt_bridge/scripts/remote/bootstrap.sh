#!/usr/bin/env bash
# Stand up the CARLA driving setup on a fresh machine. Run on the REMOTE.
#
# Everything here is checked before anything is installed, because the failures that
# actually happen on a new box are environmental and silent:
#
#   no NVIDIA container runtime   `docker run --gpus all` fails with a runtime error that
#                                 reads like a Docker problem
#   no GPU visible in a container the CARLA server starts, listens on 2000, and renders
#                                 nothing -- runs complete and produce empty traces
#   OPENAI_API_KEY unset          brain logs "not set", every branch falls back to
#                                 `default`, and a sweep finishes looking healthy while
#                                 measuring nothing
#
# Usage:  bash bootstrap.sh [~/carla_bundle]
set -euo pipefail

BUNDLE="${1:-$HOME/carla_bundle}"
WS="$HOME/ros2_ws"
VENV="$HOME/.venvs/nlplanner"
TD="$HOME/data/touchdown"

say() { printf '\n== %s\n' "$1"; }
fail() { printf '   FAIL  %s\n' "$1" >&2; exit 1; }
ok()   { printf '   ok    %s\n' "$1"; }

say "preflight"
command -v docker >/dev/null || fail "docker not installed"
docker info >/dev/null 2>&1 || fail "docker daemon not reachable (add yourself to the docker group?)"
ok "docker"

command -v nvidia-smi >/dev/null || fail "nvidia-smi missing -- no driver, so CARLA cannot render"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/         /'

# The runtime check is the one worth doing properly: the driver being present on the host
# says nothing about whether a CONTAINER can see the GPU.
docker run --rm --gpus all ubuntu:22.04 true 2>/dev/null \
  || fail "containers cannot use the GPU -- install nvidia-container-toolkit and restart docker"
ok "GPU visible inside containers"

need=60
have=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
[ "${have:-0}" -ge "$need" ] || fail "need ~${need}G free in \$HOME, have ${have}G"
ok "disk (${have}G free)"

[ -f "$BUNDLE/bridge-image.tar.gz" ] || fail "no bundle at $BUNDLE -- copy it over first"
if [ -f "$BUNDLE/MANIFEST.txt" ]; then
  (cd "$BUNDLE" && sha256sum -c --status <(grep -A99 '^sha256:' MANIFEST.txt | tail -n +2)) \
    && ok "bundle checksums match" || fail "bundle is corrupt -- re-copy it"
fi

say "CARLA server image (public, ~10 GB)"
docker pull carlasim/carla:0.9.14

say "bridge image (from the bundle, ~16 GB)"
if docker image inspect carla-ros-bridge-dev:latest >/dev/null 2>&1; then
  ok "already present, skipping"
else
  gunzip -c "$BUNDLE/bridge-image.tar.gz" | docker load
fi

say "workspace"
mkdir -p "$WS"
tar xzf "$BUNDLE/ros2_ws_src.tar.gz" -C "$WS"
tar xzf "$BUNDLE/carla_pythonapi.tar.gz" -C "$HOME"
ok "$WS/src and ~/carla"

# A venv is not relocatable -- absolute paths are written into its scripts -- so it is
# rebuilt rather than copied. pydantic_ai stays OUT of the system python on purpose: it
# pulls a newer openai that breaks the raw client used elsewhere.
say "python environment (isolated on purpose)"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --disable-pip-version-check "pydantic-ai-slim[openai]" pyyaml numpy
"$VENV/bin/python" -c "import pydantic_ai, openai, yaml; print('   pydantic_ai', pydantic_ai.__version__, '| openai', openai.__version__)"

say "Touchdown corpus"
mkdir -p "$TD"
for split in train dev test; do
  [ -s "$TD/$split.json" ] || curl -sSL -o "$TD/$split.json" \
    "https://raw.githubusercontent.com/lil-lab/touchdown/master/data/$split.json"
done
n=$(cat "$TD"/*.json | wc -l)
[ "$n" -eq 9325 ] || fail "corpus is $n lines, expected 9325"
ok "9,325 instructions"

say "build"
# NOT built on the host. The nodes run INSIDE the bridge container against ROS humble,
# and run_phase_a.sh already does `colcon build --symlink-install` in there on every
# invocation. Building on the host would compile against whatever distro is in /opt/ros
# and produce install/ trees the container never reads.
ok "built inside the container by run_phase_a.sh; nothing to do on the host"

say "smoke test: does a mission actually drive?"
if [ -z "${OPENAI_API_KEY:-}" ]; then
  cat <<'MSG'
   OPENAI_API_KEY is unset. Plan generation needs it, and so does cue_source=vlm.
   Without it brain falls back to the `default` branch on every decision and a sweep
   will finish looking healthy while measuring nothing. Export it, then:

     python3 src/carla_gt_bridge/scripts/run_missions.py --arm ours --only M3
MSG
else
  ok "OPENAI_API_KEY present"
  echo "   run:  python3 src/carla_gt_bridge/scripts/run_missions.py --arm ours --only M3"
fi

say "done"
cat <<'MSG'
   Driving:      python3 src/carla_gt_bridge/scripts/run_missions.py --arm ours --only M3
   English in:   python3 src/carla_gt_bridge/scripts/drive_english.py --english "..." --start 0
   Stop it all:  bash src/carla_gt_bridge/scripts/run_phase_a.sh --stop

MSG
