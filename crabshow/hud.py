"""Mission-control view: a top-down, schematic replay of a show, for "how it works" cuts.

Each crab is a dot in its show colour with a short trail. During a formation change a thin
line joins every crab to the spot the assignment gave it, and the HUD names the active
policy, the cue, the beat and the live minimum clearance between any two crabs.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .show import lights_at, sample

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

POLICY_NAMES = {"move": "posture  (walk, velocity + body-height command)",
                "hold": "posture  (stand, beat bounce)",
                "jump": "jump     (residual on recorded jump)",
                "clip": None}


class HUD:
    def __init__(self, plan, score, logs, width=1080, height=1350):
        self.plan, self.score, self.logs = plan, score, logs
        self.W, self.H = width, height
        self.bar = 4 * 60.0 / plan["bpm"]
        self.beat = self.bar / 4
        xy = np.concatenate([np.array([q[:, :2].min(0), q[:, :2].max(0)]) for _, q in logs])
        lo, hi = xy.min(0) - 1.0, xy.max(0) + 1.0
        self.c = (lo + hi) / 2
        span = hi - lo
        self.ppm = min(width * 0.9 / span[0], height * 0.62 / span[1])
        self.f = ImageFont.truetype(MONO, int(width * 0.022))
        self.fb = ImageFont.truetype(MONO_B, int(width * 0.03))
        self.fs = ImageFont.truetype(MONO, int(width * 0.018))
        self.labels = {}
        for a in plan["acts"]:
            self.labels[round(a["bar"] * self.bar - (0.3 if a["do"] == "jump" else 0), 2)] = a

    def _px(self, p):
        return (self.W / 2 + (p[0] - self.c[0]) * self.ppm,
                self.H * 0.47 - (p[1] - self.c[1]) * self.ppm)

    def _cue(self, t):
        cur = None
        for c in self.score.cues:
            if c.t <= t:
                cur = c
        if cur is not None and t > cur.t + cur.dur + 0.05:
            return None
        return cur

    def frame(self, t):
        W, H = self.W, self.H
        img = Image.new("RGB", (W, H), (6, 9, 16))
        d = ImageDraw.Draw(img, "RGBA")
        # Metre grid.
        for gx in range(-30, 31):
            x, _ = self._px((gx, 0))
            if 0 <= x <= W:
                d.line([(x, H * 0.12), (x, H * 0.82)], fill=(30, 40, 60, 255 if gx % 5 == 0 else 110))
        for gy in range(-30, 31):
            _, y = self._px((0, gy))
            if H * 0.12 <= y <= H * 0.82:
                d.line([(0, y), (W, y)], fill=(30, 40, 60, 255 if gy % 5 == 0 else 110))
        now = np.array([q[:2] for q in sample(self.logs, t)])
        cols, glow = lights_at(self.score, t)
        cols = np.clip(0.25 + 0.75 * cols * np.clip(glow[:, None] + 0.3, 0, 1), 0, 1)
        cue = self._cue(t)
        # Assignment lines and target rings during a move.
        if cue is not None and cue.kind == "move" and cue.targets is not None:
            for p, q, c in zip(now, cue.targets, cols):
                a, b = self._px(p), self._px(q)
                d.line([a, b], fill=(*(c * 255).astype(int), 90), width=1)
                r = 0.3 * self.ppm
                d.ellipse([b[0] - r, b[1] - r, b[0] + r, b[1] + r], outline=(*(c * 255).astype(int), 160))
        # Trails.
        for k in range(1, 16):
            past = np.array([q[:2] for q in sample(self.logs, t - k * 0.06)])
            a = int(120 * (1 - k / 16))
            for p, c in zip(past, cols):
                x, y = self._px(p)
                r = 2
                d.ellipse([x - r, y - r, x + r, y + r], fill=(*(c * 255).astype(int), a))
        # Crabs: body footprint (0.6 m wide), coloured.
        for p, c in zip(now, cols):
            x, y = self._px(p)
            rx, ry = 0.3 * self.ppm, 0.12 * self.ppm
            d.ellipse([x - rx, y - ry, x + rx, y + ry], fill=(*(c * 255).astype(int), 255))
        # Clearance.
        dm = np.linalg.norm(now[:, None] - now[None], axis=-1)
        np.fill_diagonal(dm, np.inf)
        clear = float(dm.min())
        # HUD text.
        b = t / self.beat
        frac = b - np.floor(b)
        pulse = np.exp(-frac * 5)
        d.text((W * 0.05, H * 0.03), "CRAB SWARM  ·  MISSION CONTROL", font=self.fb, fill=(150, 255, 210))
        d.text((W * 0.05, H * 0.075), f"t {t:5.2f}s   bar {t / self.bar + 1:5.2f}   "
               f"{self.plan['bpm']} BPM", font=self.f, fill=(170, 190, 220))
        r = 14
        bx, by = W * 0.9, H * 0.05
        d.ellipse([bx - r, by - r, bx + r, by + r], fill=(255, 80, 120, int(60 + 195 * pulse)))
        if cue is None:
            pol, what = "posture  (stand)", "hold"
        elif cue.kind == "clip":
            pol, what = f"{cue.policy}  (tracks a recorded clip)", f"{cue.policy}"
        else:
            pol, what = POLICY_NAMES[cue.kind], cue.kind
        act = None
        for tt, a in sorted(self.labels.items()):
            if tt <= t + 0.01:
                act = a
        label = ""
        if act:
            label = act["do"] + (" " + (act.get("text") or act.get("shape") or act.get("move") or "")).rstrip()
        lines = [
            (f"AGENTS     {self.score.n} × independent MuJoCo sims", (200, 210, 230)),
            (f"POLICY     {pol}", (255, 220, 120)),
            (f"CUE        {label}", (150, 255, 210)),
            ("ASSIGN     Hungarian, min Σd²  →  no path crossings" if what == "move" else
             "ASSIGN     —", (170, 190, 220)),
            (f"CLEARANCE  {clear:4.2f} m  (min over all pairs)",
             (120, 255, 140) if clear > 0.5 else (255, 120, 120)),
        ]
        y = H * 0.845
        for text, col in lines:
            d.text((W * 0.05, y), text, font=self.f, fill=col)
            y += H * 0.03
        # Timeline of acts.
        tl_y = H * 0.115
        total = self.score.duration
        d.line([(W * 0.05, tl_y), (W * 0.95, tl_y)], fill=(60, 80, 110), width=2)
        for tt, a in self.labels.items():
            x = W * 0.05 + W * 0.9 * max(tt, 0) / total
            d.line([(x, tl_y - 6), (x, tl_y + 6)], fill=(120, 150, 190), width=2)
        x = W * 0.05 + W * 0.9 * t / total
        d.ellipse([x - 6, tl_y - 6, x + 6, tl_y + 6], fill=(150, 255, 210))
        return img
