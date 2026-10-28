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
# The committed source picture: every machine must build the identical mosaic.
SOURCE = Path(__file__).resolve().parent.parent / "assets" / "mosaic_source_studio.png"


def _studio(w, h, ss, faces=200000):
    """The studio renderer used for the source portrait (key, cool fill, warm rim)."""
    from .render import Renderer, STYLES
    from .runtime import DATA

    dflt = json.loads((DATA / "policies" / "posture" / "layout.json").read_text())["default_joint_pos"]
    st = copy.deepcopy(STYLES["moonbeach"])
    st.glow, st.floor_reflect, st.shadows = 0.06, 0.0, False
    st.floor_rgb1 = st.floor_rgb2 = (0, 0, 0)
    st.ambient, st.key_light, st.ocean, st.headlight = 0.12, 0.85, None, 0.18
    st.mesh_faces = 200000
    st.extra_lights = [((0.6, -0.4, 0.5), (-0.6, 0.4, -0.5), (0.25, 0.32, 0.45)),   # cool fill
                       ((-0.2, 0.8, 0.6), (0.2, -0.8, -0.5), (0.9, 0.75, 0.6))]     # warm rim
    st.mesh_faces = faces
    return Renderer(1, np.array([[0.9, 0.12, 0.08]]), st, w * ss, h * ss)


def rest_pose(R):
    """Standing pose, facing the camera, as in the source portrait."""
    from .runtime import DATA

    dflt = json.loads((DATA / "policies" / "posture" / "layout.json").read_text())["default_joint_pos"]
    q = np.zeros(R.nq1)
    q[:3] = [0, 0, 0.106]
    q[3:7] = [np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)]
    for j in range(R.model.njnt):
        nm = R.model.joint(j).name.replace("c0_", "")
        if nm in dflt:
            q[R.model.joint(j).qposadr[0]] = dflt[nm]
    return q


def studio_frame(R, q, w, h):
    """One studio render of the crab in pose q (base re-centred), downsampled to w x h."""
    q = np.array(q, dtype=float)
    R.set_poses([q])
    R.set_lights(np.array([[0.92, 0.13, 0.07]]), np.array([1.0]))
    big = np.asarray(R.frame([0, 0, 0.045], 0.7, 90, -50))
    return np.asarray(Image.fromarray(big).resize((w, h), Image.LANCZOS))


def _render_source(w=800, h=660, ss=2):
    """A studio portrait of one Jumper: key, cool fill and warm rim lights, full-detail
    meshes, rendered at ss x and downsampled for clean edges."""
    R = _studio(w, h, ss)
    q = rest_pose(R)
    R.set_poses([q])
    R.set_lights(np.array([[0.92, 0.13, 0.07]]), np.array([1.0]))
    big = np.asarray(R.frame([0, 0, 0.045], 0.7, 90, -50))
    img = np.asarray(Image.fromarray(big).resize((w, h), Image.LANCZOS))
    m = img.max(-1) > 12
    ys, xs = np.nonzero(m)
    pad = 8
    return Image.fromarray(img[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad])


def face_masks(lit):
    """From a lit/unlit grid: the filled silhouette, the face display (largest hole) and
    the eye shapes drawn on it (open, happy ^, closed line). All boolean grids."""
    from scipy.ndimage import binary_fill_holes, label

    filled = binary_fill_holes(lit)
    holes = filled & ~lit
    lab, nlab = label(holes)
    z = np.zeros_like(lit)
    if not nlab:
        return filled, z, z, z, z
    sizes = [(lab == k).sum() for k in range(1, nlab + 1)]
    display = lab == (int(np.argmax(sizes)) + 1)
    r_idx, c_idx = np.nonzero(display)
    r0, r1, c0, c1 = r_idx.min(), r_idx.max(), c_idx.min(), c_idx.max()
    rm, w, h = (r0 + r1) / 2, c1 - c0 + 1, r1 - r0 + 1
    eye_c = [c0 + 0.27 * w, c0 + 0.73 * w]
    ax, ay = max(1.5, 0.12 * w), max(1.5, 0.36 * h)
    R, C = np.mgrid[0:lit.shape[0], 0:lit.shape[1]]
    eo, eh, el = z.copy(), z.copy(), z.copy()
    for ec in eye_c:
        eo |= ((C - ec) / ax) ** 2 + ((R - rm) / ay) ** 2 <= 1.0
        # Happy eyes: an upturned arc ^ over each eye.
        eh |= (np.abs(((C - ec) / ax) ** 2 + ((R - (rm + 0.35 * ay)) / (ay * 0.9)) ** 2 - 1) < 0.45) \
            & (R <= rm + 0.35 * ay)
        # Closed eye: a flat line across the eye's middle (for a wink).
        el |= (np.abs(C - ec) <= ax * 0.95) & (np.abs(R - rm) <= max(1.1, 0.18 * ay))
    return filled, display, eo & display, eh & display, el & display


def build(cols=84, spacing=0.66, lum_min=0.06, extend=None):
    """Returns dict: pts (n,2), cols (n,3), rc (n,2) grid row/col, hole (n,) bool,
    eye_open / eye_happy / eye_line (n,) bool, dark (n,) bool, hero (index of the crab at the
    left eye). With extend (a giant-move asset), the grid grows by the asset's margin and
    every cell the move ever lights gets a crab too, dark until the move needs it."""
    CACHE.mkdir(parents=True, exist_ok=True)
    src_path = SOURCE if SOURCE.exists() else CACHE / "source_studio.png"
    if not src_path.exists():
        _render_source().save(src_path)
    src = Image.open(src_path).convert("RGB")
    rows = int(round(cols * src.height / src.width))
    small = np.asarray(src.resize((cols, rows), Image.LANCZOS)).astype(float) / 255
    lit = small.max(-1) > lum_min
    filled, display, eo, eh, el = face_masks(lit)
    m = 0
    extra = np.zeros_like(filled)
    if extend:
        # One asset or several: every cell any of them ever lights gets a crab.
        reach = None
        for path in ([extend] if isinstance(extend, str) else extend):
            z = np.load(path)
            m = int(z["margin"])
            g = z["grid"]
            assert g.shape[1:3] == (rows + 2 * m, cols + 2 * m), "asset made for another grid"
            r = (g.max(-1) > lum_min * 255).any(0)
            reach = r if reach is None else reach | r
        pad = lambda a: np.pad(a, m)
        small = np.pad(small, ((m, m), (m, m), (0, 0)))
        filled, display, eo, eh, el = (pad(a) for a in (filled, display, eo, eh, el))
        extra = reach & ~filled
    sel = filled | extra
    rr, cc = np.nonzero(sel)
    pts = np.stack([(cc - m - (cols - 1) / 2) * spacing, ((rows - 1) / 2 - (rr - m)) * spacing], 1)
    colors = np.where(extra[rr, cc][:, None], 0.0, small[rr, cc])
    rc = np.stack([rr, cc], 1)
    eye_open = eo[rr, cc]
    r_idx, c_idx = np.nonzero(display)
    rm = (r_idx.min() + r_idx.max()) / 2
    eye_c0 = c_idx.min() + 0.27 * (c_idx.max() - c_idx.min() + 1)
    # Hero: the crab nearest the centre of the left eye.
    d = (rc[:, 1] - eye_c0) ** 2 + (rc[:, 0] - rm) ** 2
    hero = int(np.argmin(np.where(eye_open, d, np.inf)))
    return {"pts": pts, "cols": colors, "rc": rc, "hole": display[rr, cc],
            "eye_open": eye_open, "eye_happy": eh[rr, cc], "eye_line": el[rr, cc],
            "dark": extra[rr, cc], "hero": hero, "grid": (rows + 2 * m, cols + 2 * m),
            "margin": m, "spacing": spacing}
