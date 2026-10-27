import sys, time
sys.path.insert(0, ".")
import numpy as np, imageio
from crabshow.runtime import Crab
from crabshow.render import Renderer
from crabshow import formation as F

n = 12
pts = F.circle(n, 1.5)
crabs = []
for i, p in enumerate(pts):
    c = Crab(p[0], p[1], yaw=np.arctan2(-p[1], -p[0]))
    c.run_clip("dance", 2.0)
    crabs.append(c)
cols = [tuple(np.array(__import__("colorsys").hsv_to_rgb(i / n, 0.9, 1.0))) for i in range(n)]
for style in sys.argv[1:]:
    t = time.time()
    R = Renderer(n, cols, style, 540, 960)
    R.set_poses([c.log_qpos[-1] for c in crabs])
    img = R.frame([0, 0, 0.05], 5.0, 90, -35)
    imageio.imwrite(f"out/test_{style}.png", img)
    print(style, f"{time.time()-t:.1f}s")
