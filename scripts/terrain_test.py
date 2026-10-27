import sys
sys.path.insert(0, ".")
import numpy as np
import crabshow
from crabshow.runtime import Crab
from crabshow.terrain import height
for amp in [0.04, 0.08, 0.12]:
    T = {"amp": amp, "seed": 0}
    ok = 0; res = []
    for k, (x, y, ang) in enumerate([(0, 0, 0), (2, 1, 90), (-3, 2, 45), (1, -3, 135)]):
        c = Crab(x, y, terrain=T, seed=k)
        a = np.radians(ang)
        c.run_walk(1.0, lambda _c: (0, 0, 0))
        c.run_walk(5.0, lambda _c: (0.5 * np.cos(a), 0.5 * np.sin(a), 0))
        f1 = c.fallen()
        c.run_clip("dance_brazilian", 5.0, start_frame=646)
        f2 = c.fallen()
        c.run_jump(); c.run_walk(1.0, lambda _c: (0, 0, 0))
        f3 = c.fallen()
        res.append((f1, f2, f3, round(float(np.linalg.norm(c.pos - [x, y])), 2)))
    slope = np.degrees(np.arctan(np.abs(np.gradient(height(np.linspace(-5,5,500), 0*np.linspace(-5,5,500), amp), 10/500)).max()))
    print(f"amp {amp}: max slope ~{slope:.0f} deg, (fell walking, fell dancing, fell jumping, dist):", res)
