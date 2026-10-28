#!/usr/bin/env bash
# Simulate crabs [A, B) of a plan and push the pack to branch simpart/<sim>-<A>.
# usage: scripts/sim_part_job.sh plans/onecrab12k.json A B
set -euo pipefail
PLAN=$1; A=$2; B=$3
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
scripts/setup.sh
.venv/bin/python -I scripts/sim_part.py "$PLAN" "$A" "$B"
SIMNAME=$(.venv/bin/python -I -c "import json,sys;p=json.load(open(sys.argv[1]));print(p.get('sim_name', p['name']))" "$PLAN")
BR=$(printf "simpart/%s-%06d" "$SIMNAME" "$A")
git checkout -q -B "$BR"
mkdir -p simpack && cp out/simpack/*.part$(printf "%06d" "$A").npz simpack/
git add -f simpack/
git -c user.name="Shaurya Sen" -c user.email="66851598+ShauryaSen@users.noreply.github.com" commit -q -m "sim part $SIMNAME $A-$B"
for i in 1 2 3 4; do git push -q -f -u origin "$BR" && break || sleep $((2**i)); done
echo "pushed $BR"
