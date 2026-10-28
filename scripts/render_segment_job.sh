#!/usr/bin/env bash
# Render frames [A, B) of a plan and push the segment to its own branch.
# usage: scripts/render_segment_job.sh plans/rave_v3.json A B [WxH]
set -euo pipefail
PLAN=$1; A=$2; B=$3; SIZE=${4:-}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
scripts/setup.sh
# A pre-computed simulation, if one was published for this plan (big swarms).
SIMNAME=$(.venv/bin/python -I -c "import json,sys;p=json.load(open(sys.argv[1]));print(p.get('sim_name', p['name']))" "$PLAN")
if git ls-remote --exit-code origin "refs/heads/simpack/$SIMNAME" >/dev/null 2>&1; then
  git fetch -q origin "simpack/$SIMNAME"
  mkdir -p out/simpack
  for f in $(git ls-tree --name-only "origin/simpack/$SIMNAME" simpack/); do
    git show "origin/simpack/$SIMNAME:$f" > "out/simpack/$(basename "$f")"
  done
  N=$(.venv/bin/python -I -c "import json,sys;print(json.load(open(sys.argv[1]))['n'])" "$PLAN")
  .venv/bin/python -I scripts/simpack.py unpack out/simpack "out/${SIMNAME}_n${N}_sim.pkl"
fi
.venv/bin/python -I scripts/make_show.py "$PLAN" --frames "$A:$B" ${SIZE:+--size "$SIZE"}
NAME=$(.venv/bin/python -I -c "import json,sys;p=json.load(open(sys.argv[1]));print(p['name']+'_'+p['style'])" "$PLAN")${SIZE:+_$SIZE}
SEG=$(printf "out/segments/%s_%05d_%05d.mp4" "$NAME" "$A" "$B")
BR=$(printf "seg/%s-%05d" "$NAME" "$A")
git checkout -q -B "$BR"
mkdir -p segments && cp "$SEG" segments/
git add -f segments/
git -c user.name="Shaurya Sen" -c user.email="66851598+ShauryaSen@users.noreply.github.com" commit -q -m "render segment $NAME $A-$B"
for i in 1 2 3 4; do git push -q -f -u origin "$BR" && break || sleep $((2**i)); done
echo "pushed $BR"
