import sys, time
sys.path.insert(0, ".")
import numpy as np
from crabshow.runtime import Crab, policy

for name in ["dance", "dance_brazilian", "dance_dream_wings", "dance_maze", "dance_waist", "gesture_hello", "gesture_bow", "gesture_salute", "gesture_paw"]:
    print(name, "frames", policy(name).clip_frames, end=" ")
    res = []
    for start in [0, 500, 3000]:
        if start >= policy(name).clip_frames: continue
        c = Crab()
        c.run_walk(2.0, lambda _c: (0.4, 0.0, 0.0))
        c.run_walk(0.5, lambda _c: (0, 0, 0))
        c.run_clip(name, 6.0, start_frame=start)
        f1 = c.fallen()
        c.run_walk(3.0, lambda _c: (0.0, 0.4, 0.0))
        res.append((start, f1, c.fallen(), c.pos.round(2).tolist()))
    print(res)
