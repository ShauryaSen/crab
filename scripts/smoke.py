import sys, time
import numpy as np
sys.path.insert(0, ".")
from crabshow.runtime import Crab

for cmd in [(0, 0, 0), (0.3, 0, 0), (0, 0.3, 0), (0, 0, 1.0)]:
    c = Crab()
    t = time.time()
    c.run_walk(4.0, lambda _c, cmd=cmd: cmd)
    print("walk", cmd, "pos", c.pos.round(3), "yaw", round(c.yaw, 2), "z", round(c.data.qpos[2], 3),
          "fallen", c.fallen(), f"{time.time()-t:.1f}s")
for name in ["dance", "dance_brazilian", "gesture_hello"]:
    c = Crab()
    t = time.time()
    c.run_clip(name, 6.0)
    print(name, "pos", c.pos.round(3), "z", round(c.data.qpos[2], 3), "fallen", c.fallen(), f"{time.time()-t:.1f}s")
