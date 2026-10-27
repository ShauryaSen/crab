#!/usr/bin/env bash
# Fetch KingKong's Jumper robot model and exported policies (Apache-2.0) into data/jumper.
# crabshow only reads these as data (MJCF, ONNX, JSON); it never imports jumper's code.
set -euo pipefail
COMMIT=b2e58669848a3ba4f209597cd6bebdfe9d0533a4
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${JUMPER_SRC:-$ROOT/third_party/jumper}"
if [ ! -d "$SRC/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/KingKongRobotics/jumper "$SRC"
fi
git -C "$SRC" checkout -q "$COMMIT"
D="$ROOT/data/jumper"
mkdir -p "$D/policies"
rm -rf "$D/robot" && cp -r "$SRC/assets/jumper" "$D/robot"
for t in posture dance dance_brazilian dance_dream_wings dance_maze dance_waist \
         gesture_bow gesture_hello gesture_paw gesture_salute jump five_foot; do
  mkdir -p "$D/policies/$t"
  cp "$SRC/tasks/jumper/$t/out/example/"{actor.onnx,layout.json} "$D/policies/$t/"
  cp "$SRC/tasks/jumper/$t/out/example/"*.trajectory.json "$D/policies/$t/" 2>/dev/null || true
done
cp "$SRC/out/bundle_example/jumper/reference.json" "$D/policies/posture/"
cp "$SRC/LICENSE" "$SRC/NOTICE" "$D/"
echo "data/jumper ready"
