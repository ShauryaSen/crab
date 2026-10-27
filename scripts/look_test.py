"""Render a couple of frames of a cached show with style overrides: quick look-dev."""
import sys, json, pickle
sys.path.insert(0, ".")
import numpy as np
import crabshow
from PIL import Image
from crabshow.compose import compose
from crabshow.render import Renderer, STYLES
from crabshow.show import sample, lights_at
from crabshow.video import camera_track, bloom
plan = json.load(open(sys.argv[1]))
over = json.loads(sys.argv[2])
times = [float(x) for x in sys.argv[3].split(",")]
style = STYLES[plan["style"]]
for k, v in over.items():
    setattr(style, k, tuple(v) if isinstance(v, list) else v)
score, _ = compose(plan)
logs = pickle.load(open(f"out/{plan['name']}_n{score.n}_sim.pkl", "rb"))
W, H = 432, 540
at = camera_track(plan, logs, score.duration, W, H)
R = Renderer(score.n, np.ones((score.n, 3)), style, W, H)
ims = []
for t in times:
    R.set_poses(sample(logs, t)); R.set_lights(*lights_at(score, t))
    ims.append(bloom(Image.fromarray(R.frame(*at(t))), radius=8))
sheet = Image.new("RGB", (W * len(ims), H))
for i, im in enumerate(ims):
    sheet.paste(im, (i * W, 0))
sheet.save(sys.argv[4])
