"""Rolling sand dunes, the same heightfield for the physics and the render."""

from __future__ import annotations

import numpy as np

SIZE = 24.0     # half-extent, metres
RES = 0.06      # metres per cell


def height(x, y, amp=0.08, seed=0):
    """Dunes: a few long sine swells plus gentle ripples. amp is the swell amplitude."""
    rng = np.random.default_rng(seed)
    h = np.zeros_like(np.asarray(x, dtype=float))
    for k in range(4):
        wl = rng.uniform(2.5, 5.0)
        a = rng.uniform(0, np.pi)
        ph = rng.uniform(0, 2 * np.pi)
        h = h + amp / (k + 1) ** 0.6 * np.sin(2 * np.pi * (np.cos(a) * x + np.sin(a) * y) / wl + ph)
    h = h + amp * 0.08 * np.sin(2 * np.pi * (x * 0.8 + y) / 0.35)
    return h


def grid(amp=0.08, seed=0):
    n = int(2 * SIZE / RES) + 1
    xs = np.linspace(-SIZE, SIZE, n)
    X, Y = np.meshgrid(xs, xs)
    return height(X, Y, amp, seed)


def add_hfield(spec, name="dunes", amp=0.08, seed=0, contype=1, conaffinity=1, material=None):
    """Add the dunes as an hfield geom; returns the geom. Heights are offset so z=0 is
    the field's mean."""
    import mujoco

    h = grid(amp, seed)
    lo, hi = float(h.min()), float(h.max())
    span = max(hi - lo, 1e-6)
    n = h.shape[0]
    hf = spec.add_hfield(name=name, nrow=n, ncol=n, size=[SIZE, SIZE, span, 0.5])
    hf.userdata = ((h - lo) / span).astype(np.float32).ravel().tolist()
    g = spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=name,
                                pos=[0, 0, lo], contype=contype, conaffinity=conaffinity)
    if material:
        g.material = material
    return g
