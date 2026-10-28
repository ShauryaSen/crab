"""Frames of the one-crab show on stand-in (standing) poses, to check the look before the sim lands."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import crabshow
from crabshow.compose import compose
from crabshow.video import preview_sheet
from crabshow.runtime import DATA
plan = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "plans/onecrab.json"))
s, m = compose(plan)
plan["_hero"] = int(m["hero"])
dflt = json.loads((DATA / "policies" / "posture" / "layout.json").read_text())["default_joint_pos"]
import mujoco
from crabshow.runtime import build_spec
mdl = build_spec().compile()
q0 = np.zeros(mdl.nq); q0[2] = 0.106; q0[3:7] = [np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)]
for j in range(mdl.njnt):
    if mdl.joint(j).name in dflt:
        q0[mdl.joint(j).qposadr[0]] = dflt[mdl.joint(j).name]
logs = []
for p in s.start:
    q = q0.copy(); q[:2] = p
    logs.append((np.array([0.0, 30.0]), np.array([q, q])))
times = [float(x) for x in sys.argv[1].split(",")]
preview_sheet(s, logs, plan, sys.argv[3] if len(sys.argv) > 3 else "out/onecrab_static.png", times, width=int(sys.argv[4]) if len(sys.argv) > 4 else 432, height=int(sys.argv[5]) if len(sys.argv) > 5 else 540)
print("ok")
