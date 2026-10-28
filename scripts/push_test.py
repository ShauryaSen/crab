"""Robustness experiment: shove crabs mid-dance and mid-walk; count who stays up."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import crabshow
import mujoco
from crabshow.runtime import Crab

def trial(seed, mode, impulse):
    rng = np.random.default_rng(seed)
    c = Crab(seed=seed)
    c.run_walk(1.0, lambda _c: (0, 0, 0))
    # A sideways shove: a 0.1 s force on the base, random direction.
    a = rng.uniform(0, 2 * np.pi)
    F = impulse / 0.1 * np.array([np.cos(a), np.sin(a), 0.0])
    bid = c.model.body("base_link").id
    def shove(on):
        c.data.xfrc_applied[bid, :3] = F if on else 0.0
    if mode == "dance":
        shove(True); c.run_clip("dance_brazilian", 0.1, start_frame=646); shove(False)
        c.run_clip("dance_brazilian", 3.0, start_frame=651)
    else:
        shove(True); c.run_walk(0.1, lambda _c: (0.3, 0, 0)); shove(False)
        c.run_walk(3.0, lambda _c: (0.3, 0, 0))
    up = not c.fallen() and c.data.qpos[2] > 0.07
    return up

res = {}
for mode in ["walk", "dance"]:
    for imp in [5.0, 8.0, 12.0]:
        ok = sum(trial(s, mode, imp) for s in range(20))
        res[f"{mode} {imp}"] = ok
        print(mode, "impulse", imp, "N*s  ->", ok, "/ 20 stayed up", flush=True)
