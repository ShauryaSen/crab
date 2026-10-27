"""Frames to a finished video: camera path, glow/bloom, the typed prompt, the end card."""

from __future__ import annotations

import multiprocessing as mp
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .show import lights_at, sample

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def camera_at(keys, t):
    """keys: list of (t, lookat(3), distance, azimuth, elevation); smooth interpolation."""
    if t <= keys[0][0]:
        k = keys[0]
        return k[1], k[2], k[3], k[4]
    for a, b in zip(keys, keys[1:]):
        if a[0] <= t <= b[0]:
            s = (t - a[0]) / max(b[0] - a[0], 1e-9)
            s = s * s * (3 - 2 * s)
            lerp = lambda x, y: np.asarray(x) * (1 - s) + np.asarray(y) * s
            return lerp(a[1], b[1]), float(lerp(a[2], b[2])), float(lerp(a[3], b[3])), \
                float(lerp(a[4], b[4]))
    k = keys[-1]
    return k[1], k[2], k[3], k[4]


def bloom(img: Image.Image, strength=0.9, radius=18, threshold=110) -> Image.Image:
    a = np.asarray(img).astype(np.float32)
    lum = a.max(-1, keepdims=True)
    bright = np.clip((lum - threshold) / (255 - threshold), 0, 1) * a
    b = Image.fromarray(bright.astype(np.uint8))
    blur = np.zeros_like(a)
    for r, w in ((radius / 3, 0.5), (radius, 0.35), (radius * 3, 0.25)):
        blur += np.asarray(b.filter(ImageFilter.GaussianBlur(r))).astype(np.float32) * w
    out = a + strength * blur
    # Soft vignette.
    h, w_ = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w_]
    v = 1 - 0.35 * (((xx - w_ / 2) / (w_ / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 2
    out *= v[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def overlay_prompt(img: Image.Image, text: str, t: float, t0=0.3, cps=22.0, hold_until=7.0,
                   fade=0.6):
    """The prompt typed in a terminal-style box at the top of the frame."""
    if t < t0 or t > hold_until + fade:
        return img
    shown = text[:int((t - t0) * cps)]
    alpha = 1.0 if t <= hold_until else max(0.0, 1 - (t - hold_until) / fade)
    W, H = img.size
    size = int(W * 0.045)
    font = ImageFont.truetype(MONO, size)
    # Wrap.
    max_chars = int((W * 0.84) / (size * 0.6))
    words, lines, cur = ("> " + shown).split(" "), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip() if cur else w
    lines.append(cur)
    cursor = "█" if int(t * 2.5) % 2 == 0 else " "
    lines[-1] += cursor
    pad = int(size * 0.7)
    box_h = len(lines) * int(size * 1.35) + 2 * pad
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x0, y0 = int(W * 0.06), int(H * 0.08)
    d.rounded_rectangle([x0, y0, W - x0, y0 + box_h], radius=int(size * 0.6),
                        fill=(10, 12, 20, int(200 * alpha)), outline=(120, 255, 200, int(160 * alpha)),
                        width=3)
    for k, line in enumerate(lines):
        d.text((x0 + pad, y0 + pad + k * int(size * 1.35)), line, font=font,
               fill=(160, 255, 210, int(255 * alpha)))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def overlay_caption(img: Image.Image, lines, alpha=1.0, y_frac=0.80, size_frac=0.055):
    W, H = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    y = int(H * y_frac)
    for k, (text, scale) in enumerate(lines):
        font = ImageFont.truetype(SANS, int(W * size_frac * scale))
        l, t, r, b = d.textbbox((0, 0), text, font=font)
        d.text(((W - (r - l)) / 2, y), text, font=font, fill=(255, 255, 255, int(255 * alpha)),
               stroke_width=3, stroke_fill=(0, 0, 0, int(200 * alpha)))
        y += int((b - t) * 1.45)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def _render_chunk(args):
    (k0, k1, fps, score, logs, plan, outdir, width, height) = args
    from .render import Renderer

    R = Renderer(score.n, np.ones((score.n, 3)), plan.get("style", "night"), width, height)
    keys = [(c[0], c[1], c[2], c[3], c[4]) for c in plan["camera"]]
    for k in range(k0, k1):
        t = k / fps
        R.set_poses(sample(logs, t))
        cols, glow = lights_at(score, t)
        R.set_lights(cols, glow)
        img = Image.fromarray(R.frame(*camera_at(keys, t)))
        if R.style.glow > 0:
            img = bloom(img)
        img = overlay_prompt(img, plan["prompt"], t, hold_until=plan.get("prompt_hold", 6.0))
        for cap in plan.get("captions", []):
            if cap["t0"] <= t <= cap["t1"]:
                a = min(1.0, (t - cap["t0"]) / 0.4, (cap["t1"] - t) / 0.4)
                img = overlay_caption(img, [(l, s) for l, s in cap["lines"]], a,
                                      cap.get("y", 0.78))
        img.save(outdir / f"f{k:05d}.png", compress_level=1)
    return k1 - k0


def render_video(score, logs, plan, out_path, audio_path=None, fps=30, width=1080,
                 height=1920, procs=4):
    out_path = Path(out_path)
    frames_dir = out_path.with_suffix("")
    frames_dir = frames_dir.parent / (frames_dir.name + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    n_frames = int(score.duration * fps)
    chunks = np.linspace(0, n_frames, procs * 4 + 1).astype(int)
    jobs = [(a, b, fps, score, logs, plan, frames_dir, width, height)
            for a, b in zip(chunks, chunks[1:]) if b > a]
    # One chunk per worker process: software GL contexts do not survive being reused.
    with mp.get_context("spawn").Pool(procs, maxtasksperchild=1) as pool:
        for _ in pool.imap_unordered(_render_chunk, jobs):
            pass
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
           str(frames_dir / "f%05d.png")]
    if audio_path:
        cmd += ["-i", str(audio_path), "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
            "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True)
    return out_path
