"""Post-processing looks: retro and stylised grades applied to finished frames.

Each look is f(img: PIL.Image, t: float, k: int) -> PIL.Image, deterministic per frame index
k so the grain and flicker animate but re-renders match.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def _rng(k):
    return np.random.default_rng(1000 + k)


def _grain(a, k, amount):
    return a + _rng(k).normal(0, amount, a.shape[:2])[..., None]


def vhs(img, t, k, stamp="OCT 28 1996"):
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]
    # Chroma bleed: shift red right, blue left; soften chroma.
    out = a.copy()
    sh = max(2, w // 270)
    out[:, sh:, 0] = a[:, :-sh, 0]
    out[:, :-sh, 2] = a[:, sh:, 2]
    out = np.asarray(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
                     .filter(ImageFilter.GaussianBlur(w / 900))).astype(np.float32)
    # Scanlines, tape noise, a rolling tracking band.
    out[::2] *= 0.86
    out = _grain(out, k, 9)
    band = int((t * 0.35 % 1.0) * h)
    bh = h // 40
    out[band:band + bh] = np.roll(out[band:band + bh], w // 60, axis=1) * 1.15 + 12
    # Lift blacks, warm, slightly desaturate.
    out = 18 + out * 0.9
    g = out.mean(-1, keepdims=True)
    out = g + (out - g) * 0.8
    out[..., 0] *= 1.04
    im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(MONO, int(w * 0.045))
    d.text((w * 0.06, h * 0.06), "PLAY ▶" if int(t * 2) % 2 == 0 else "PLAY", font=f,
           fill=(235, 235, 235))
    mm, ss = divmod(int(t), 60)
    d.text((w * 0.06, h * 0.89), f"{stamp}", font=f, fill=(235, 235, 235))
    d.text((w * 0.66, h * 0.89), f"0:{mm:02d}:{ss:02d}", font=f, fill=(235, 235, 235))
    return im


def film(img, t, k):
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]
    rng = _rng(k)
    # Gate weave: tiny jitter.
    dx, dy = rng.integers(-2, 3, 2)
    a = np.roll(np.roll(a, dx, 1), dy, 0)
    # Warm, faded grade with crushed highlights; strong grain; flicker.
    a = 10 + a * 0.92
    a[..., 0] *= 1.06
    a[..., 2] *= 0.88
    a = 255 * (a / 255) ** 0.95
    a = _grain(a, k, 14) * (1 + rng.normal(0, 0.02))
    # Dust and a hair now and then.
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for _ in range(rng.integers(2, 7)):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(1, 3) * w / 1000
        d.ellipse([x - r, y - r, x + r, y + r], fill=(235, 230, 220))
    if rng.uniform() < 0.15:
        x = rng.uniform(0, w)
        d.line([(x, 0), (x + rng.normal(0, 20), h)], fill=(25, 25, 25), width=1)
    return _vignette(im, 0.55)


def newsreel(img, t, k):
    a = np.asarray(img).astype(np.float32)
    g = a @ np.array([0.3, 0.59, 0.11])
    g = np.clip((g - 20) * 1.35, 0, 255)
    rng = _rng(k)
    g = g * (1 + rng.normal(0, 0.05)) + rng.normal(0, 16, g.shape)
    im = Image.fromarray(np.clip(np.stack([g, g * 0.97, g * 0.92], -1), 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    w, h = im.size
    for _ in range(rng.integers(0, 3)):         # vertical scratches
        x = rng.uniform(0, w)
        d.line([(x, 0), (x, h)], fill=(220, 220, 220), width=1)
    return _vignette(im, 0.8)


def eightbit(img, t, k, width=180, colors=24):
    w, h = img.size
    small = img.resize((width, int(h * width / w)), Image.BILINEAR)
    small = small.quantize(colors=colors, method=Image.Quantize.MEDIANCUT,
                           dither=Image.Dither.NONE).convert("RGB")
    return small.resize((w, h), Image.NEAREST)


def crt(img, t, k):
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]
    # Aperture-grille RGB mask and scanlines.
    mask = np.ones((h, w, 3), np.float32) * 0.7
    for c in range(3):
        mask[:, c::3, c] = 1.25
    a = a * mask
    a[1::3] *= 0.7
    # Barrel distortion.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx / w - 0.5) * 2, (yy / h - 0.5) * 2
    r2 = nx * nx + ny * ny
    sx = ((nx * (1 + 0.08 * r2)) / 2 + 0.5) * (w - 1)
    sy = ((ny * (1 + 0.08 * r2)) / 2 + 0.5) * (h - 1)
    valid = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    out = np.zeros_like(a)
    out[valid] = a[sy[valid].astype(int), sx[valid].astype(int)]
    glow = np.asarray(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
                      .filter(ImageFilter.GaussianBlur(w / 200))).astype(np.float32)
    out = out + 0.35 * glow
    return _vignette(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), 0.5)


def _vignette(im, strength):
    a = np.asarray(im).astype(np.float32)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    v = 1 - strength * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 2
    return Image.fromarray(np.clip(a * v[..., None], 0, 255).astype(np.uint8))


def grain(img, t, k, amount=7.0):
    """Fine film grain only: hides the dot pattern without changing the grade."""
    a = np.asarray(img).astype(np.float32)
    g = _rng(k).normal(0, amount, a.shape[:2])[..., None]
    lum = a.mean(-1, keepdims=True) / 255
    a = a + g * (0.35 + 0.65 * lum)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


LOOKS = {"grain": grain, "vhs": vhs, "film": film, "newsreel": newsreel, "8bit": eightbit, "crt": crt}
