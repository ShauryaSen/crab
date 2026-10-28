"""Render the giant-crab dance as the swarm would show it: one standing crab per grid cell
of the extended canvas, lit with that cell's colour frame by frame, filmed top-down with
the hero's lens (defocus, bloom, grain). Lights only, no physics: a look test.

usage: python -I scripts/exp_giant_dance_render.py out/giant/dance_dream_wings_160.npy out/giant/test.mp4
"""
import json, subprocess, sys, copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import crabshow  # noqa: F401
from PIL import Image
from crabshow.crowd import CrowdRenderer
from crabshow.render import STYLES
from crabshow.runtime import DATA
from crabshow.video import bloom, defocus
from crabshow.looks import grain

_z = np.load(sys.argv[1])
grid = (_z["grid"] if hasattr(_z, "files") else _z).astype(float) / 255   # (F, rows, cols, 3)
dst = Path(sys.argv[2])
W, H = (int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "540x675").split("x"))
F, rows, cols, _ = grid.shape
lit = grid.max(-1) > 0.06
union = lit.any(0)
rr, cc = np.nonzero(union)
sp = 0.66
pts = np.stack([(cc - (cols - 1) / 2) * sp, ((rows - 1) / 2 - rr) * sp], 1)
n = len(pts)
print("crabs", n)

plan = json.load(open("plans/onecrab_v4.json"))
st = copy.deepcopy(STYLES[plan["style"]])
for k, v in plan.get("style_opts", {}).items():
    setattr(st, k, v)
R = CrowdRenderer(n, np.ones((n, 3)), st, W, H)
dflt = json.loads((DATA / "policies" / "posture" / "layout.json").read_text())["default_joint_pos"]
q0 = np.array(R.data.qpos[:R.nq1])
q0[:3] = [0, 0, 0.106]
q0[3:7] = [np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)]
for j in range(R.model.njnt):
    nm = R.model.joint(j).name.replace("c0_", "")
    if nm in dflt:
        q0[R.model.joint(j).qposadr[0] - R.qadr[0]] = dflt[nm]
Q = np.repeat(q0[None], n, 0)
Q[:, :2] = pts
R.set_poses(Q)
fr_dir = dst.with_suffix("")
fr_dir.mkdir(parents=True, exist_ok=True)
# Frame the resting crab the way the hero's opening shot does (fit its lit extent).
ext = np.ptp(pts[lit[0][rr, cc]], 0)
dist = 0.62 * max(ext[0] * H / W, ext[1]) / np.tan(np.radians(22.5))
cam = ([0, 0, 0.05], dist, 90, -89.5)
b = plan.get("bloom", {})
for k in range(F):
    g = grid[k][rr, cc]
    on = lit[k][rr, cc]
    cols_ = np.where(on[:, None], np.clip(g * 1.15, 0, 1), 0.05)
    R.set_lights(cols_, on.astype(float))
    img = Image.fromarray(R.frame(*cam))
    img = defocus(img, plan["defocus"], cam, H)
    img = bloom(img, strength=b.get("strength", 0.9), radius=b.get("radius", 18), threshold=b.get("threshold", 110))
    img = grain(img, k / 15, k)
    img.save(fr_dir / f"{k:04d}.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "15", "-i", str(fr_dir / "%04d.png"),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(dst)], check=True)
print("wrote", dst)
