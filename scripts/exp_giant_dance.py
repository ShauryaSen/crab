"""Experiment: the giant crab dances. One crab's real dance, rendered in the mosaic's studio
setup frame by frame, becomes the lights of the whole mosaic, so the picture made of crabs
moves like a crab. Saves the per-frame grid (for the show) and a top-down preview video.

usage: python -I scripts/exp_giant_dance.py dance_dream_wings 4.0 [cols]
"""
import json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import crabshow  # noqa: F401
from PIL import Image
from crabshow import mosaic
from crabshow.runtime import Crab

move, secs = sys.argv[1], float(sys.argv[2])
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 160
fps = 15
out = Path("out/giant")
out.mkdir(parents=True, exist_ok=True)

# 1. One crab dances in place, facing the camera.
c = Crab(0, 0, -np.pi / 2, 0, None)
c.run_clip(move, secs, start_frame=0)
t, q = np.array(c.log_t), np.array(c.log_qpos)
keep = np.searchsorted(t, np.arange(0, secs, 1 / fps))
q = q[np.clip(keep, 0, len(q) - 1)]
q[:, :2] -= q[0, :2]

# 2. Studio frames with the source portrait's crop.
w, h, ss = 800, 660, 2
R = mosaic._studio(w, h, ss)
rest = mosaic.studio_frame(R, mosaic.rest_pose(R), w, h)
m = rest.max(-1) > 12
ys, xs = np.nonzero(m)
pad = 8
box = (max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad)
# Extend the canvas at the same scale so moves beyond the resting outline still land on
# grid cells; the resting crab keeps exactly the mosaic's cells (offset by the margin).
src_h, src_w = box[1] - box[0], box[3] - box[2]
cell = src_w / cols
mr, mc = int(round(float(sys.argv[4]) if len(sys.argv) > 4 else 12)), int(round(float(sys.argv[4]) if len(sys.argv) > 4 else 12))
big_w, big_h = 800 + 2 * int(mc * cell) + 2, 660 + 2 * int(mr * cell) + 2
rows = int(round(cols * src_h / src_w))
R2 = mosaic._studio(big_w, big_h, ss)
fov0 = float(R2.model.vis.global_.fovy)
R2.model.vis.global_.fovy = float(np.degrees(2 * np.arctan(np.tan(np.radians(fov0 / 2)) * big_h / h)))
ox, oy = (big_w - w) // 2, (big_h - h) // 2       # where the 800x660 view sits inside
y0, x0 = box[0] + oy - mr * cell, box[2] + ox - mc * cell
H2, W2 = (rows + 2 * mr) * cell, (cols + 2 * mc) * cell
frames = []
for k, qk in enumerate(q):
    # A bigger canvas with the same focal scale: widen the view by rendering larger with
    # the same camera, then crop the extended box.
    img = mosaic.studio_frame(R2, qk, big_w, big_h)
    frames.append(img[int(y0):int(y0 + H2), int(x0):int(x0 + W2)])
    if k % 15 == 0:
        print("frame", k, flush=True)
grid = np.stack([np.asarray(Image.fromarray(f).resize((cols + 2 * mc, rows + 2 * mr), Image.LANCZOS))
                 for f in frames])
np.save(out / f"{move}_{cols}.npy", grid)
lit = grid.max(-1) > 0.06 * 255
union = lit.any(0)
base = lit[0]
frac = lit.mean(0)
print("grid", grid.shape, "base lit", base.sum(), "union lit", union.sum(), "extra crabs", (union & ~base).sum(),
      "| lit in >=10% of frames:", (frac >= 0.1).sum(), ">=25%:", (frac >= 0.25).sum())

# 3. Quick look: the grid frames side by side (upscaled), and a GIF.
strip = [Image.fromarray(g).resize((g.shape[1] * 3, g.shape[0] * 3), Image.NEAREST) for g in grid[::max(1, len(grid) // 6)][:6]]
W = sum(s.width for s in strip)
sheet = Image.new("RGB", (W, strip[0].height))
x = 0
for s_ in strip:
    sheet.paste(s_, (x, 0))
    x += s_.width
sheet.save(out / f"{move}_{cols}_sheet.png")
gif = [Image.fromarray(g).resize((g.shape[1] * 3, g.shape[0] * 3), Image.NEAREST) for g in grid]
gif[0].save(out / f"{move}_{cols}.gif", save_all=True, append_images=gif[1:], duration=int(1000 / fps), loop=0)
print("wrote", out / f"{move}_{cols}_sheet.png")
