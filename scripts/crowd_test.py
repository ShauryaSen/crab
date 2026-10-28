import sys, time, json, colorsys
sys.path.insert(0, ".")
import crabshow, numpy as np, copy
from PIL import Image
from crabshow.crowd import CrowdRenderer
from crabshow.render import STYLES
N = int(sys.argv[1])
dflt = json.load(open("data/jumper/policies/posture/layout.json"))["default_joint_pos"]
st = copy.deepcopy(STYLES["moonbeach"])
R = CrowdRenderer(st, 1080, 1350, faces=3000, max_geom=N * 45 + 5000)
q1 = np.array(R.data.qpos); q1[:3] = [0, 0, 0.106]; q1[3:7] = [0.7071, 0, 0, -0.7071]
for j in range(R.model.njnt):
    nm = R.model.joint(j).name.replace("c0_", "")
    if nm in dflt:
        q1[R.model.joint(j).qposadr[0]] = dflt[nm]
g = int(np.ceil(np.sqrt(N))); qs = []
for i in range(N):
    q = q1.copy(); q[0] = (i % g - g / 2) * 0.8; q[1] = (i // g - g / 2) * 0.8; qs.append(q)
cols = [colorsys.hsv_to_rgb((np.hypot(q[0], q[1]) / 12) % 1, 0.9, 1) for q in qs]
for k in range(2):
    t = time.time(); img = R.frame(qs, cols, np.ones(N), [0, 0, 0], 0.75 * g, 90, -60)
    print(N, "frame", round(time.time() - t, 2), "s")
Image.fromarray(img).save(f"out/crowd_{N}.png")
