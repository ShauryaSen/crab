"""A photomosaic of one Jumper made of many Jumpers.

Render one crab, downsample the picture to a grid, and put a crab on every pixel inside
the robot's silhouette, coloured as that pixel. Holes in the silhouette (the black face
display) are filled with unlit crabs, which later light up as the giant robot's eyes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

CACHE = Path(__file__).resolve().parent.parent / "out" / "mosaic"


def _render_source(w=480, h=400):
    from .render import Renderer, STYLES
    from .runtime import DATA

    dflt = json.loads((DATA / "policies" / "posture" / "layout.json").read_text())["default_joint_pos"]
    st = copy.deepcopy(STYLES["moonbeach"])
    st.glow, st.floor_reflect, st.shadows = 0.25, 0.0, False
    st.floor_rgb1 = st.floor_rgb2 = (0, 0, 0)
    st.ambient, st.key_light, st.ocean = 0.35, 1.2, None
    R = Renderer(1, np.array([[0.9, 0.12, 0.08]]), st, w, h)
    q = np.zeros(R.nq1)
    q[:3] = [0, 0, 0.106]
    q[3:7] = [np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)]
    for j in range(R.model.njnt):
        nm = R.model.joint(j).name.replace("c0_", "")
        if nm in dflt:
            q[R.model.joint(j).qposadr[0]] = dflt[nm]
    R.set_poses([q])
    R.set_lights(np.array([[0.95, 0.15, 0.08]]), np.array([1.0]))
    img = np.asarray(R.frame([0, 0, 0.045], 0.7, 90, -50))
    m = img.max(-1) > 15
    ys, xs = np.nonzero(m)
    pad = 6
    return Image.fromarray(img[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad])


def build(cols=84, spacing=0.66, lum_min=0.06):
    """Returns dict: pts (n,2), cols (n,3), rc (n,2) grid row/col, hole (n,) bool,
    eye_open (n,) bool, eye_happy (n,) bool, hero (index of the crab at the left eye)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    src_path = CACHE / "source.png"
    if not src_path.exists():
        _render_source().save(src_path)
    src = Image.open(src_path).convert("RGB")
    rows = int(round(cols * src.height / src.width))
    small = np.asarray(src.resize((cols, rows), Image.LANCZOS)).astype(float) / 255
    lit = small.max(-1) > lum_min
    from scipy.ndimage import binary_fill_holes, label

    filled = binary_fill_holes(lit)
    holes = filled & ~lit
    # The face display is the largest hole.
    lab, nlab = label(holes)
    sizes = [(lab == k).sum() for k in range(1, nlab + 1)]
    display = (lab == (int(np.argmax(sizes)) + 1)) if nlab else np.zeros_like(holes)
    r_idx, c_idx = np.nonzero(display)
    r0, r1, c0, c1 = r_idx.min(), r_idx.max(), c_idx.min(), c_idx.max()
    rm, w, h = (r0 + r1) / 2, c1 - c0 + 1, r1 - r0 + 1
    eye_c = [c0 + 0.27 * w, c0 + 0.73 * w]
    ax, ay = max(1.5, 0.12 * w), max(1.5, 0.36 * h)

    pts, colors, rc, hole, eo, eh = [], [], [], [], [], []
    for r in range(rows):
        for c in range(cols):
            if not filled[r, c]:
                continue
            pts.append(((c - (cols - 1) / 2) * spacing, ((rows - 1) / 2 - r) * spacing))
            colors.append(small[r, c])
            rc.append((r, c))
            hole.append(bool(display[r, c]))
            inside = any(((c - ec) / ax) ** 2 + ((r - rm) / ay) ** 2 <= 1.0 for ec in eye_c)
            eo.append(bool(display[r, c] and inside))
            # Happy eyes: an upturned arc ^ over each eye.
            arc = any(abs(((c - ec) / ax) ** 2 + ((r - (rm + 0.35 * ay)) / (ay * 0.9)) ** 2 - 1)
                      < 0.45 and r <= rm + 0.35 * ay for ec in eye_c)
            eh.append(bool(display[r, c] and arc))
    pts, rc = np.array(pts), np.array(rc)
    eye_open = np.array(eo)
    # Hero: the crab nearest the centre of the left eye.
    d = (rc[:, 1] - eye_c[0]) ** 2 + (rc[:, 0] - rm) ** 2
    hero = int(np.argmin(np.where(eye_open, d, np.inf)))
    return {"pts": pts, "cols": np.array(colors), "rc": rc, "hole": np.array(hole),
            "eye_open": eye_open, "eye_happy": np.array(eh), "hero": hero,
            "grid": (rows, cols), "spacing": spacing}
