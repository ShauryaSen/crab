"""Compose a Score from a choreography plan.

A plan is the small JSON an AI (or a person) writes from a prompt such as "make a
heart, spell HI, then go wild on the drop". Times are in bars of the track so every
change lands on the music.

    {
      "prompt": "make a heart, spell HI, dance on the drop",
      "n": 36, "style": "night", "bpm": 128,
      "sections": [["intro", 4], ["build", 4], ["drop", 8], ["outro", 2]],
      "acts": [
        {"bar": 0,  "do": "light_up", "palette": "rainbow"},
        {"bar": 4,  "do": "form", "shape": "heart", "bars": 3, "palette": "love"},
        {"bar": 8,  "do": "dance", "move": "dance_brazilian", "bars": 4, "ripple": "center"},
        {"bar": 12, "do": "form", "text": "HI", "bars": 2, "palette": "ice"},
        {"bar": 14, "do": "gesture", "move": "gesture_hello", "bars": 2},
        {"bar": 16, "do": "lights_out"}
      ]
    }
"""

from __future__ import annotations

import colorsys

import numpy as np

from . import formation as F
from .show import Cue, Light, Score

SPACING = 0.78          # metres between crab centres in a formation
FACE_CAMERA = -np.pi / 2

PALETTES = {
    "rainbow": lambda p, n: [colorsys.hsv_to_rgb((np.arctan2(y, x) / (2 * np.pi)) % 1, 0.85, 1)
                             for x, y in p],
    "love": lambda p, n: [colorsys.hsv_to_rgb(0.97 - 0.06 * (y - p[:, 1].min()) / (np.ptp(p[:, 1]) + 1e-9),
                                              0.85, 1) for x, y in p],
    "ice": lambda p, n: [(0.75 + 0.25 * t, 0.9 + 0.1 * t, 1.0)
                         for t in (p[:, 0] - p[:, 0].min()) / (np.ptp(p[:, 0]) + 1e-9)],
    "gold": lambda p, n: [(1.0, 0.7 + 0.2 * t, 0.2 + 0.2 * t)
                          for t in (p[:, 1] - p[:, 1].min()) / (np.ptp(p[:, 1]) + 1e-9)],
    "ocean": lambda p, n: [colorsys.hsv_to_rgb(0.5 + 0.12 * t, 0.9, 1)
                           for t in (p[:, 0] - p[:, 0].min()) / (np.ptp(p[:, 0]) + 1e-9)],
    "crab": lambda p, n: [(1.0, 0.25 + 0.15 * t, 0.1) for t in np.linspace(0, 1, n)],
    "white": lambda p, n: [(1.0, 1.0, 1.0)] * n,
}

# The most visually lively stretch of each dance clip, picked by reference joint speed.
LIVELY: dict[str, int] = {}


def lively_start(policy_name: str, seconds: float) -> int:
    if policy_name in LIVELY:
        return LIVELY[policy_name]
    from .runtime import policy

    p = policy(policy_name)
    qd = np.abs(p.ref["qd"]).sum(1)
    w = int(seconds * p.ref_meta["rec_hz"])
    if len(qd) <= w:
        LIVELY[policy_name] = 0
        return 0
    c = np.convolve(qd, np.ones(w), mode="valid")
    LIVELY[policy_name] = int(np.argmax(c))
    return LIVELY[policy_name]


def shape_points(act: dict, n: int) -> np.ndarray:
    if "text" in act:
        pts = F.text(act["text"], n, width_m=1.0 + 0.9 * len(act["text"]))
    else:
        shape = act.get("shape", "circle")
        if shape == "heart":
            pts = F.heart(n, 5.0, filled=act.get("filled", False))
        elif shape == "circle":
            pts = F.circle(n, 3.0)
        elif shape == "spiral":
            pts = F.spiral(n, SPACING)
        elif shape == "grid":
            pts = F.grid(n, SPACING)
        else:
            raise ValueError(shape)
    return F.fit_spacing(pts, SPACING)


def scatter(n: int, radius: float, seed: int = 1) -> np.ndarray:
    """Random start positions with no two closer than SPACING."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 2)
        if all(np.linalg.norm(p - q) >= SPACING * 1.1 for q in pts):
            pts.append(p)
    return np.array(pts)


def compose(plan: dict) -> tuple[Score, dict]:
    n = plan["n"]
    bar = 4 * 60.0 / plan["bpm"]
    beat = bar / 4
    total_bars = sum(b for _, b in plan["sections"])
    rng = np.random.default_rng(plan.get("seed", 3))
    start = scatter(n, plan.get("scatter_radius", 0.42 * np.sqrt(n)), plan.get("seed", 1))
    score = Score(n=n, start=start, start_yaw=FACE_CAMERA, duration=total_bars * bar)
    pos = start.copy()
    cols = np.zeros((n, 3))
    glow = np.zeros(n)
    meta = {"formations": []}

    for act in plan["acts"]:
        t = act["bar"] * bar
        do = act["do"]
        dur = act.get("bars", 1) * bar
        if do == "light_up":
            # Crabs switch on one by one, on eighth notes, in random order.
            order = rng.permutation(n)
            cols = np.array(PALETTES[act.get("palette", "rainbow")](pos, n))
            span = dur * 0.9
            for k, i in enumerate(order):
                g = glow.copy()
                g[i] = 1.0
                glow = g
                score.lights.append(Light(t + k * span / n, cols.copy(), glow.copy(), fade=0.08))
        elif do == "form":
            goal = shape_points(act, n)
            perm = F.assign(pos, goal)
            targets = goal[perm]
            score.cues.append(Cue(t, "move", dur, targets=targets, yaw=FACE_CAMERA))
            new_cols = np.array(PALETTES[act.get("palette", "white")](targets, n))
            score.lights.append(Light(t, new_cols, np.ones(n), fade=dur * 0.8))
            cols, glow, pos = new_cols, np.ones(n), targets
            meta["formations"].append({"bar": act["bar"], "label": act.get("text", act.get("shape"))})
        elif do in ("dance", "gesture"):
            move = act["move"]
            sf = act.get("start_frame")
            if sf is None:
                sf = lively_start(move, dur) if do == "dance" else 0
            offsets = None
            ripple = act.get("ripple")
            if ripple:
                c = pos.mean(0)
                if ripple == "center":
                    d = np.linalg.norm(pos - c, axis=1)
                elif ripple == "left":
                    d = pos[:, 0] - pos[:, 0].min()
                else:
                    d = rng.uniform(0, 1, n) * np.ptp(pos[:, 0])
                offsets = d / (np.ptp(d) + 1e-9) * act.get("ripple_s", beat * 2)
                # Light ripple on the same wave: a white flash travelling outwards.
                for k in range(8):
                    tk = t + offsets.max() * k / 7
                    flash = np.clip(1 - np.abs(offsets - (tk - t)) / (beat * 0.5), 0, 1)
                    score.lights.append(Light(tk, cols * (1 - flash[:, None] * 0.6)
                                              + flash[:, None] * 0.6, np.ones(n), fade=0.05))
                score.lights.append(Light(t + offsets.max() + beat, cols.copy(), np.ones(n), 0.2))
            score.cues.append(Cue(t, "clip", dur, policy=move, start_frame=sf, offsets=offsets,
                                  yaw=FACE_CAMERA))
            if act.get("pulse", do == "dance"):
                # Glow pulses on every beat while dancing.
                for k in range(int(round(dur / beat))):
                    tb = t + k * beat
                    score.lights.append(Light(tb, cols.copy(), np.full(n, 1.0), fade=0.02))
                    score.lights.append(Light(tb + beat * 0.5, cols.copy(), np.full(n, 0.55),
                                              fade=beat * 0.45))
                score.lights.append(Light(t + dur, cols.copy(), np.ones(n), fade=0.1))
        elif do == "lights_out":
            score.lights.append(Light(t, cols.copy(), np.zeros(n), fade=act.get("bars", 1) * bar))
        elif do == "color":
            cols = np.array(PALETTES[act["palette"]](pos, n))
            score.lights.append(Light(t, cols.copy(), glow.copy(), fade=act.get("fade", beat)))
        else:
            raise ValueError(do)
    score.lights.sort(key=lambda L: L.t)
    score.cues.sort(key=lambda c: c.t)
    return score, meta
