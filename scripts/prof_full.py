import sys, time, json, pickle
sys.path.insert(0, ".")
import numpy as np
import crabshow
from PIL import Image
from crabshow.compose import compose
from crabshow.render import Renderer, STYLES
from crabshow.show import sample, lights_at
from crabshow.video import bloom, camera_track
from crabshow.fx import FX
import mujoco
plan = json.load(open(sys.argv[1]))
W, H = int(sys.argv[2]), int(sys.argv[3])
st = STYLES[plan["style"]]
for kv in sys.argv[4:]:
    k, v = kv.split("=")
    setattr(st, k, json.loads(v))
score, _ = compose(plan)
logs = pickle.load(open(f"out/{plan.get('sim_name', plan['name'])}_n{score.n}_sim.pkl", "rb"))
at = camera_track(plan, logs, score.duration, W, H)
R = Renderer(score.n, np.ones((score.n, 3)), st, W, H)
fx = FX(plan, score, logs) if plan.get("fx") else None
for t in [16.0, 27.0]:
    p = sample(logs, t); R.set_poses(p); c, g = lights_at(score, t); R.set_lights(c, g)
    xy = np.array([q[:2] for q in p])
    t0 = time.time(); scn_extra = (lambda s: fx.draw(s, t, xy, c, g)) if fx else None
    R.r.update_scene(R.data, R.cam, R.opt)
    if fx: fx.draw(R.r.scene, t, xy, c, g)
    t1 = time.time(); img = R.frame(*at(t), extra=scn_extra); t2 = time.time()
    b = bloom(Image.fromarray(img)); t3 = time.time()
    print(f"t={t} fx {t1-t0:.2f}s frame(total incl fx) {t2-t1:.2f}s bloom {t3-t2:.2f}s ngeom {R.r.scene.ngeom}")
