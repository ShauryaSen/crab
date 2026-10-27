#!/usr/bin/env bash
# Bring a fresh Linux box (e.g. a cloud session container) up to rendering shows.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if ! ldconfig -p | grep -q libOSMesa; then
  (apt-get install -y -q libosmesa6 >/dev/null 2>&1) || \
  (apt-get update -q >/dev/null 2>&1 && apt-get install -y -q libosmesa6 >/dev/null)
fi
command -v ffmpeg >/dev/null || (apt-get install -y -q ffmpeg >/dev/null 2>&1 || \
  (apt-get update -q >/dev/null 2>&1 && apt-get install -y -q ffmpeg >/dev/null))
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q mujoco onnxruntime numpy scipy librosa pillow matplotlib \
  fast-simplification trimesh
[ -f data/jumper/robot/jumper.xml ] || scripts/fetch_jumper.sh
echo "setup ok"
