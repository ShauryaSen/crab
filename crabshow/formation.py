"""Formations: turn a word or a shape into N floor positions, and assign crabs to them.

All positions are metres in the floor plane, centred on the origin, x to the
audience's right and y away from the audience (the overhead camera looks down -z
with +y at the top of the frame).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _farthest_point_sample(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick n well-spread points out of a dense candidate cloud."""
    if len(points) <= n:
        return points
    chosen = [rng.integers(len(points))]
    dist = np.linalg.norm(points - points[chosen[0]], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(points - points[i], axis=1))
    return points[chosen]


def _relax(points: np.ndarray, mask_pts: np.ndarray, iters: int = 30) -> np.ndarray:
    """Lloyd relaxation over a dense mask so the crabs spread evenly through the glyphs."""
    p = points.copy()
    for _ in range(iters):
        d = ((mask_pts[:, None, :] - p[None, :, :]) ** 2).sum(-1)
        owner = d.argmin(1)
        for k in range(len(p)):
            sel = mask_pts[owner == k]
            if len(sel):
                p[k] = sel.mean(0)
    return p


def _mask_to_points(mask: np.ndarray, n: int, width_m: float, seed: int) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    scale = width_m / w
    pts = np.stack([(xs - w / 2) * scale, (h / 2 - ys) * scale], 1)
    rng = np.random.default_rng(seed)
    init = _farthest_point_sample(pts, n, rng)
    return _relax(init, pts)


def text(word: str, n: int, width_m: float, seed: int = 0) -> np.ndarray:
    """n points filling the letters of `word`, `width_m` wide."""
    font = ImageFont.truetype(FONT, 200)
    l, t, r, b = font.getbbox(word)
    img = Image.new("L", (r - l + 40, b - t + 40), 0)
    ImageDraw.Draw(img).text((20 - l, 20 - t), word, font=font, fill=255)
    return _mask_to_points(np.asarray(img) > 127, n, width_m, seed)


def heart(n: int, width_m: float, filled: bool = False, seed: int = 0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    outline = np.stack([x, y], 1)
    outline = outline / (x.max() - x.min()) * width_m
    if not filled:
        # Even spacing along the outline by arc length.
        seg = np.linalg.norm(np.diff(outline, axis=0, append=outline[:1]), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])[:-1]
        targets = np.linspace(0, s[-1] + seg[-1], n, endpoint=False)
        idx = np.searchsorted(s, targets).clip(0, len(outline) - 1)
        return outline[idx]
    from matplotlib.path import Path

    lo, hi = outline.min(0), outline.max(0)
    g = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], 300), np.linspace(lo[1], hi[1], 300)), -1)
    g = g.reshape(-1, 2)
    inside = g[Path(outline).contains_points(g)]
    rng = np.random.default_rng(seed)
    return _relax(_farthest_point_sample(inside, n, rng), inside)


def circle(n: int, radius_m: float, phase: float = 0.0) -> np.ndarray:
    a = phase + np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius_m * np.cos(a), radius_m * np.sin(a)], 1)


def grid(n: int, spacing_m: float) -> np.ndarray:
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    pts = [((c - (cols - 1) / 2) * spacing_m, ((rows - 1) / 2 - r) * spacing_m)
           for r in range(rows) for c in range(cols)]
    return np.array(pts[:n])


def spiral(n: int, spacing_m: float) -> np.ndarray:
    """Sunflower (Fermat) spiral: dense, round, evenly spaced."""
    k = np.arange(n) + 0.5
    r = spacing_m * np.sqrt(k) * 0.55
    a = k * np.pi * (3 - np.sqrt(5))
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def assign(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """perm such that crab i goes to goal[perm[i]], minimising total squared distance.

    Squared distance is used deliberately: for straight-line moves at a common speed the
    minimum-sum-of-squares assignment never makes two paths cross (Turpin et al., 2014),
    which is what keeps a swarm reformation collision-free without negotiation.
    """
    cost = ((start[:, None, :] - goal[None, :, :]) ** 2).sum(-1)
    _, perm = linear_sum_assignment(cost)
    return perm


def min_spacing(points: np.ndarray) -> float:
    d = np.linalg.norm(points[:, None] - points[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def fit_spacing(points: np.ndarray, spacing_m: float) -> np.ndarray:
    """Scale a formation about its centre so its closest pair is exactly spacing_m apart."""
    c = points.mean(0)
    return c + (points - c) * (spacing_m / min_spacing(points))
