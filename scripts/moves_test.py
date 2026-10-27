import sys
sys.path.insert(0, ".")
import numpy as np
import crabshow
from crabshow.runtime import Crab

# 1. Jump
for k in range(3):
    c = Crab(seed=k)
    c.run_walk(1.0, lambda _c: (0, 0, 0))
    z = []
    n0 = len(c.log_qpos)
    c.run_jump()
    zs = [q[2] for q in c.log_qpos[n0:]]
    c.run_walk(1.5, lambda _c: (0, 0, 0))
    print(f"jump {k}: peak z {max(zs):.3f} (stand 0.106), fallen after: {c.fallen()}, drift {np.linalg.norm(c.pos):.2f} m")
    # back-to-back jumps
c = Crab()
c.run_walk(0.5, lambda _c: (0, 0, 0))
for j in range(4):
    c.run_jump(); c.run_walk(0.3, lambda _c: (0, 0, 0))
print("4 jumps in a row, fallen:", c.fallen())

# 2. Height bob at different rates
for hz in [0.5, 1.067, 2.133, 4.267]:
    c = Crab()
    c.run_walk(1.0, lambda _c: (0, 0, 0))
    n0 = len(c.log_qpos)
    c.run_walk(4.0, lambda _c: (0, 0, 0), posture=lambda cr: (0, 0, 0, 0.035 * np.sin(2 * np.pi * hz * cr.t)))
    zs = np.array([q[2] for q in c.log_qpos[n0:]])
    print(f"bob {hz:.2f} Hz: z range {zs.min():.3f}..{zs.max():.3f}  fallen {c.fallen()}")
# twist/roll groove
c = Crab()
c.run_walk(4.0, lambda _c: (0, 0, 0), posture=lambda cr: (0.4 * np.sin(2*np.pi*1.067*cr.t), 0.2*np.cos(2*np.pi*1.067*cr.t), 0.15*np.sin(2*np.pi*2.13*cr.t), 0.02))
print("twist/pitch/roll groove fallen", c.fallen())
