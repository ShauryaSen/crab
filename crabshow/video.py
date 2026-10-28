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
    """keys: list of (t, offset(3), zoom, azimuth, elevation[, follow]); smooth interpolation.

    follow (default 1) is how much the look-at point tracks the swarm's centre; 0 holds the
    camera on the world origin so the whole swarm can be seen travelling.
    """
    keys = [list(k) + [1.0] * (6 - len(k)) for k in keys]
    if t <= keys[0][0]:
        return tuple(keys[0][1:])
    for a, b in zip(keys, keys[1:]):
        if a[0] <= t <= b[0]:
            s = (t - a[0]) / max(b[0] - a[0], 1e-9)
            s = s * s * (3 - 2 * s)
            return tuple(np.asarray(x) * (1 - s) + np.asarray(y) * s if isinstance(x, list)
                         else float(x * (1 - s) + y * s) for x, y in zip(a[1:], b[1:]))
    return tuple(keys[-1][1:])


def auto_track(logs, duration, width, height, fps=10, sigma_s=1.2, margin=1.25, score=None):
    """Smoothed swarm centroid and fit distance over time, sampled at `fps`. With a score,
    only crabs that are lit count, so crabs walking off into the dark don't widen the shot."""
    ts = np.arange(0, duration + 1 / fps, 1 / fps)
    cen, rx, ry = [], [], []
    for t in ts:
        xy = np.array([q[:2] for q in sample(logs, t)])
        if score is not None:
            lit = lights_at(score, t)[1] > 0.3
            if lit.sum() >= 3:
                xy = xy[lit]
        lo, hi = xy.min(0) - 0.35, xy.max(0) + 0.35
        cen.append((lo + hi) / 2)
        rx.append((hi[0] - lo[0]) / 2)
        ry.append((hi[1] - lo[1]) / 2)
    k = int(sigma_s * fps * 3)
    g = np.exp(-0.5 * (np.arange(-k, k + 1) / (sigma_s * fps)) ** 2)
    g /= g.sum()

    def smooth(x):
        x = np.asarray(x, dtype=float)
        pad = np.concatenate([np.repeat(x[:1], k, 0), x, np.repeat(x[-1:], k, 0)])
        if x.ndim == 1:
            return np.convolve(pad, g, mode="valid")
        return np.stack([np.convolve(pad[:, j], g, mode="valid") for j in range(x.shape[1])], 1)

    vfov = np.radians(45.0)
    hfov = 2 * np.arctan(np.tan(vfov / 2) * width / height)
    dist = np.maximum(smooth(rx) / np.tan(hfov / 2), smooth(ry) / np.tan(vfov / 2)) * margin
    return ts, smooth(cen), dist


def _pick_crab(logs, t, pick):
    xy = np.array([q[:2] for q in sample(logs, t)])
    if isinstance(pick, int):
        return pick
    c = xy.mean(0)
    if pick == "front":      # nearest the default camera (lowest y), near the middle
        score = xy[:, 1] + 0.3 * np.abs(xy[:, 0] - c[0])
    elif pick == "left":
        score = xy[:, 0] + 0.3 * np.abs(xy[:, 1] - c[1])
    elif pick == "right":
        score = -xy[:, 0] + 0.3 * np.abs(xy[:, 1] - c[1])
    else:                    # "center"
        score = np.linalg.norm(xy - c, axis=1)
    return int(np.argmin(score))


def shot_track(plan, logs, duration, width, height, score=None):
    """Edited camera: a list of shots with hard cuts between them.

    Each shot: {"t": start_s, "kind": "wide"|"close", "az": [a0, a1], "el": [e0, e1],
    "zoom": [z0, z1] (wide, x auto-fit) or "dist": [d0, d1] (close, metres),
    "crab": "front"|"center"|"left"|"right"|index (close), "follow": 0..1 (wide)}.
    Values move smoothly within a shot; cut to the next shot at its "t".
    """
    ts, cen, dist = auto_track(logs, duration, width, height, score=score)
    shots = sorted(plan["shots"], key=lambda s: s["t"])
    ends = [s["t"] for s in shots[1:]] + [duration + 1]
    picked = {}

    def lerp(v, u):
        if isinstance(v, (int, float)):
            return float(v)
        return float(v[0] + (v[1] - v[0]) * (u * u * (3 - 2 * u)))

    def at(t):
        k = max(i for i, s in enumerate(shots) if s["t"] <= t or i == 0)
        s, t1 = shots[k], ends[k]
        u = float(np.clip((t - s["t"]) / max(t1 - s["t"], 1e-6), 0, 1))
        az, el = lerp(s.get("az", 90), u), lerp(s.get("el", -70), u)
        if s.get("kind") == "zoom":
            # Powers-of-ten move: from the swarm's fit distance down to one crab's face
            # (or back out), log-linear in distance so the speed feels constant.
            if k not in picked:
                picked[k] = _pick_crab(logs, s["t"], s.get("crab", "center"))
            i = picked[k]
            e = u * u * (3 - 2 * u)
            if s.get("out"):
                e = 1 - e
            j = min(int(round(t * 10)), len(ts) - 1)
            far_look = np.array([cen[j][0], cen[j][1], 0.0])
            near_look = np.asarray(sample([logs[i]], t)[0][:3]) + np.array([0, 0, 0.03])
            look = far_look + (near_look - far_look) * e
            d_far = float(dist[j] * s.get("zoom_far", 1.1))
            d_near = s.get("dist_near", 0.9)
            dd = float(np.exp(np.log(d_far) + (np.log(d_near) - np.log(d_far)) * e))
            el = s.get("el_far", -89) + (s.get("el_near", -18) - s.get("el_far", -89)) * e
            return look, dd, az, el
        if s.get("kind") == "close":
            if k not in picked:
                picked[k] = _pick_crab(logs, s["t"], s.get("crab", "front"))
            i = picked[k]
            # Smooth the crab's position over ~0.3 s so the shot does not jitter.
            ps = np.array([sample([logs[i]], t + d)[0][:3] for d in np.linspace(-0.15, 0.15, 7)])
            look = ps.mean(0) + np.array([0, 0, 0.03])
            return look, lerp(s.get("dist", [1.2, 1.0]), u), az, el
        j = min(int(round(t * 10)), len(ts) - 1)
        f = s.get("follow", 1.0)
        look = f * np.array([cen[j][0], cen[j][1], 0.0]) + np.array(s.get("off", [0, 0, 0.05]))
        return look, float(dist[j] * lerp(s.get("zoom", 1.0), u)), az, el
    return at


def camera_track(plan, logs, duration, width, height, score=None):
    """Camera keys are [t, [dx, dy, dz], zoom, azimuth, elevation]; the look-at point and
    distance follow the swarm automatically, the keys add offsets and framing. A plan with
    "shots" uses the edited, hard-cut camera instead."""
    if plan.get("shots"):
        return shot_track(plan, logs, duration, width, height, score)
    ts, cen, dist = auto_track(logs, duration, width, height, score=score)
    keys = plan["camera"]

    def at(t):
        off, zoom, az, el, follow = camera_at(keys, t)
        i = min(int(round(t * 10)), len(ts) - 1)
        look = follow * np.array([cen[i][0], cen[i][1], 0.0]) + np.asarray(off)
        # Lower elevations see the formation foreshortened; pull back less.
        return look, float(dist[i] * zoom), az, el
    return at


_VIGNETTE: dict = {}


def bloom(img: Image.Image, strength=0.9, radius=18, threshold=110) -> Image.Image:
    """Glow around bright pixels. `radius` is in pixels at 540 px width and scales with
    the frame; the blur runs at quarter resolution, which a glow cannot tell apart."""
    a = np.asarray(img).astype(np.float32)
    h, w_ = a.shape[:2]
    lum = a.max(-1, keepdims=True)
    bright = np.clip((lum - threshold) / (255 - threshold), 0, 1) * a
    q = 4
    small = Image.fromarray(bright.astype(np.uint8)).resize((w_ // q, h // q), Image.BILINEAR)
    scale = w_ / 540.0 / q
    blur = np.zeros((h // q, w_ // q, 3), np.float32)
    for r, wt in ((radius / 3, 0.5), (radius, 0.35), (radius * 3, 0.25)):
        blur += np.asarray(small.filter(ImageFilter.GaussianBlur(r * scale))).astype(np.float32) * wt
    blur = np.asarray(Image.fromarray(np.clip(blur, 0, 255).astype(np.uint8))
                      .resize((w_, h), Image.BILINEAR)).astype(np.float32)
    out = a + strength * blur
    if (h, w_) not in _VIGNETTE:
        yy, xx = np.mgrid[0:h, 0:w_]
        _VIGNETTE[(h, w_)] = (1 - 0.35 * (((xx - w_ / 2) / (w_ / 2)) ** 2
                                          + ((yy - h / 2) / (h / 2)) ** 2) / 2)[..., None]
    out *= _VIGNETTE[(h, w_)]
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
    (k0, k1, fps, score, logs, plan, outdir, width, height, cam) = args
    from .render import Renderer

    R = Renderer(score.n, np.ones((score.n, 3)), plan.get("style", "night"), width, height,
                 terrain=plan.get("terrain"))
    fx = None
    if plan.get("fx"):
        from .fx import FX

        fx = FX(plan, score, logs)
    for k in range(k0, k1):
        t = k / fps
        poses = sample(logs, t)
        R.set_poses(poses)
        cols, glow = lights_at(score, t)
        R.set_lights(cols, glow)
        xy = np.array([q[:2] for q in poses])
        beat = 60.0 / plan["bpm"]

        def extra(scn, t=t, xy=xy, cols=cols, glow=glow):
            if fx is not None:
                fx.draw(scn, t, xy, cols, glow)
            if plan.get("eyes", True):
                from .fx import draw_eyes
                draw_eyes(scn, t, R.data, R.display, cols, glow, beat)
        img = Image.fromarray(R.frame(*cam[k], extra=extra))
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
                 height=1920, procs=None, frames=None):
    """Render to an MP4. `frames=(a, b)` renders only frames [a, b) as a silent segment,
    so one video can be split across machines and joined with `join_segments`."""
    import os

    procs = procs or os.cpu_count() or 4
    out_path = Path(out_path)
    frames_dir = out_path.with_suffix("")
    frames_dir = frames_dir.parent / (frames_dir.name + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    n_frames = int(score.duration * fps)
    at = camera_track(plan, logs, score.duration, width, height, score)
    cam = [at(k / fps) for k in range(n_frames)]
    f0, f1 = frames or (0, n_frames)
    f1 = min(f1, n_frames)
    chunks = np.linspace(f0, f1, procs * 4 + 1).astype(int)
    jobs = [(a, b, fps, score, logs, plan, frames_dir, width, height, cam)
            for a, b in zip(chunks, chunks[1:]) if b > a]
    # One chunk per worker process: software GL contexts do not survive being reused.
    with mp.get_context("spawn").Pool(procs, maxtasksperchild=1) as pool:
        for _ in pool.imap_unordered(_render_chunk, jobs):
            pass
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
           "-start_number", str(f0), "-i", str(frames_dir / "f%05d.png")]
    if audio_path and not frames:
        cmd += ["-i", str(audio_path), "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-frames:v", str(f1 - f0)]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
            "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True)
    return out_path


def preview_sheet(score, logs, plan, out_path, times, width=360, height=450):
    """A contact sheet of a few frames, for checking framing before a full render."""
    from .render import Renderer

    at = camera_track(plan, logs, score.duration, width, height, score)
    R = Renderer(score.n, np.ones((score.n, 3)), plan.get("style", "night"), width, height,
                 terrain=plan.get("terrain"))
    fx = None
    if plan.get("fx"):
        from .fx import FX

        fx = FX(plan, score, logs)
    tiles = []
    for t in times:
        poses = sample(logs, t)
        R.set_poses(poses)
        cols, glow = lights_at(score, t)
        R.set_lights(cols, glow)
        xy = np.array([q[:2] for q in poses])
        beat = 60.0 / plan["bpm"]

        def extra(scn, t=t, xy=xy, cols=cols, glow=glow):
            if fx is not None:
                fx.draw(scn, t, xy, cols, glow)
            if plan.get("eyes", True):
                from .fx import draw_eyes
                draw_eyes(scn, t, R.data, R.display, cols, glow, beat)
        img = Image.fromarray(R.frame(*at(t), extra=extra))
        if R.style.glow > 0:
            img = bloom(img, radius=8)
        ImageDraw.Draw(img).text((6, 6), f"t={t:.1f}s", fill=(255, 255, 0))
        tiles.append(img)
    cols = 6
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * height))
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % cols) * width, (k // cols) * height))
    sheet.save(out_path)
    return out_path


def join_segments(segments, audio_path, out_path):
    """Concatenate silent segments (same encode settings) and add the soundtrack."""
    lst = Path(out_path).with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in segments))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-i", str(audio_path), "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out_path)],
                   check=True)
    lst.unlink()
    return out_path
