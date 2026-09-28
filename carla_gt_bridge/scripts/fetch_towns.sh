#!/usr/bin/env bash
# Extract the CARLA OpenDRIVE maps. No server, no GPU — just the image's filesystem.
#
# The .xodr files are ~16 MB in total and are recoverable in one command, so they need not
# be committed. `--town` needs them (pick_corridor reads junction connectivity out of the
# .xodr).
set -euo pipefail
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config"
IMG="${IMG:-carlasim/carla:0.9.14}"
cid=$(docker create "$IMG" /bin/true)
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
tmp=$(mktemp -d)
docker cp "$cid:/home/carla/CarlaUE4/Content/Carla/Maps/OpenDrive" "$tmp/x" >/dev/null
for f in "$tmp"/x/Town*.xodr; do
    case "$f" in *_Opt.xodr) continue ;; esac      # _Opt is the same geometry
    cp "$f" "$DEST/$(basename "$f")"
done
rm -rf "$tmp"
ls -1 "$DEST"/Town*.xodr | sed 's|.*/|  |'
echo "-> $DEST"
