"""The giant crab moves: one crab's real move, rendered in the mosaic's studio setup frame
by frame, becomes the lights of the whole swarm, so the picture made of crabs moves like a
crab. The asset is a grid per frame on the mosaic's grid plus a margin (so claws can swing
past the resting outline); the smiling eyes are painted onto each frame's face display.

build_asset("dance_dream_wings", 6.0) -> assets/giant_dance_dream_wings_160.npz
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ASSETS = Path(__file__).resolve().parent.parent / "assets"
EYE = np.array([0.75, 0.95, 1.0])


def build_asset(move, secs, start_frame=0, cols=160, margin=14, fps=15, lum_min=0.06):
    from . import mosaic
    from .runtime import Crab

    c = Crab(0, 0, -np.pi / 2, 0, None)
    c.run_clip(move, secs, start_frame=start_frame)
    t, q = np.array(c.log_t), np.array(c.log_qpos)
    q = q[np.clip(np.searchsorted(t, np.arange(0, secs, 1 / fps)), 0, len(q) - 1)]
    q[:, :2] -= q[0, :2]

    w, h, ss = 800, 660, 2
    R = mosaic._studio(w, h, ss)
    rest = mosaic.studio_frame(R, mosaic.rest_pose(R), w, h)
    ys, xs = np.nonzero(rest.max(-1) > 12)
    pad = 8
    box = (max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad)
    src_h, src_w = box[1] - box[0], box[3] - box[2]
    cell = src_w / cols
    rows = int(round(cols * src_h / src_w))
    # A wider view at the same scale, so the resting crab keeps the mosaic's exact cells.
    big_w, big_h = w + 2 * int(margin * cell) + 2, h + 2 * int(margin * cell) + 2
    R2 = mosaic._studio(big_w, big_h, ss)
    fov0 = float(R2.model.vis.global_.fovy)
    R2.model.vis.global_.fovy = float(np.degrees(2 * np.arctan(np.tan(np.radians(fov0 / 2)) * big_h / h)))
    ox, oy = (big_w - w) // 2, (big_h - h) // 2
    y0, x0 = box[0] + oy - margin * cell, box[2] + ox - margin * cell
    H2, W2 = (rows + 2 * margin) * cell, (cols + 2 * margin) * cell
    grid = []
    for qk in q:
        img = mosaic.studio_frame(R2, qk, big_w, big_h)[int(y0):int(y0 + H2), int(x0):int(x0 + W2)]
        g = np.asarray(Image.fromarray(img).resize((cols + 2 * margin, rows + 2 * margin),
                                                   Image.LANCZOS)).astype(float)
        lit = g.max(-1) > lum_min * 255
        _, display, _, happy, _ = mosaic.face_masks(lit)
        g[display] = 0
        g[happy] = EYE * 255
        grid.append(np.clip(g, 0, 255).astype(np.uint8))
    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / f"giant_{move}_{cols}.npz"
    np.savez_compressed(out, grid=np.stack(grid), margin=margin, fps=fps, move=move)
    return out


_cache = {}


def frames(path):
    if path not in _cache:
        z = np.load(path)
        _cache[path] = (z["grid"], float(z["fps"]))
    return _cache[path]


def lights(movie, t, n):
    """Colours and glow for every crab at time t of a giant move, or None outside it."""
    if not (movie["t0"] <= t < movie["t1"]):
        return None
    grid, fps = frames(str(ASSETS / movie["asset"]))
    k = int(np.clip((t - movie["t0"]) * fps * movie.get("speed", 1.0), 0, len(grid) - 1))
    rc = movie["rc"]
    g = grid[k][rc[:, 0], rc[:, 1]].astype(float) / 255
    on = g.max(-1) > 0.06
    cols = np.where(on[:, None], np.clip(g * 1.15, 0, 1), 0.05)
    return cols, on.astype(float)
