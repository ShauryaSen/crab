"""Show effects drawn into the MuJoCo scene each frame: an LED dance floor, lasers,
light pillars, light trails and confetti. They are render-only user geoms (no physics),
driven by the beat grid and by where the crabs are.

A plan enables them with an "fx" block, times in bars:

    "fx": {
      "floor":   {"from": 0, "tile": 0.5, "half": 9, "pattern": "ripple"},
      "lasers":  [{"from": 8, "to": 12, "n": 8}],
      "pillars": [{"at": 8, "beats": 2}],
      "trails":  {"len_s": 0.8},
      "confetti": [{"at": 8, "n": 500}]
    }
"""

from __future__ import annotations

import colorsys

import mujoco
import numpy as np

_EYE = np.eye(3).ravel()


def _add(scn, gtype, size, pos, rgba, emission=1.0, mat=_EYE):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, dtype=np.float64),
                        np.asarray(pos, dtype=np.float64), np.asarray(mat, dtype=np.float64),
                        np.asarray(rgba, dtype=np.float32))
    g.emission = emission
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    scn.ngeom += 1
    return g


def _beam(scn, a, b, width, rgba, emission=1.0):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), _EYE,
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    g.emission = emission
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    scn.ngeom += 1


class FX:
    def __init__(self, plan, score, logs):
        self.cfg = plan.get("fx", {})
        self.bar = 4 * 60.0 / plan["bpm"]
        self.beat = self.bar / 4
        self.score = score
        self.logs = logs
        rng = np.random.default_rng(11)
        self.confetti = []
        for c in self.cfg.get("confetti", []):
            n = c.get("n", 400)
            t0 = c["at"] * self.bar
            v = np.stack([rng.normal(0, 1.6, n), rng.normal(0, 1.6, n),
                          rng.uniform(4.0, 8.5, n)], 1)
            p0 = np.stack([rng.normal(0, 1.5, n), rng.normal(0, 1.5, n), np.zeros(n)], 1)
            hue = rng.uniform(0, 1, n)
            col = np.array([colorsys.hsv_to_rgb(h, 0.8, 1) for h in hue])
            spin = rng.uniform(0, 2 * np.pi, (n, 2))
            self.confetti.append((t0, p0, v, col, spin, c.get("life", 3.5)))

    # ------------------------------------------------------------------
    def _active(self, spans, t):
        for s in spans:
            if s["from"] * self.bar <= t < s["to"] * self.bar:
                return s
        return None

    def beat_phase(self, t):
        b = t / self.beat
        return b, b - np.floor(b)

    def draw(self, scn, t, xy, cols, glow):
        cfg = self.cfg
        b, frac = self.beat_phase(t)
        pulse = np.exp(-frac * 4.0)                   # 1 on the beat, decays
        if "floor" in cfg and t >= cfg["floor"].get("from", 0) * self.bar:
            self._floor(scn, t, b, pulse, xy, cols, glow, cfg["floor"])
        if "trails" in cfg:
            self._trails(scn, t, cols, glow, cfg["trails"])
        for s in cfg.get("lasers", []):
            if s["from"] * self.bar <= t < s["to"] * self.bar:
                self._lasers(scn, t, b, pulse, s)
        for s in cfg.get("pillars", []):
            t0 = s["at"] * self.bar
            if t0 <= t < t0 + s.get("beats", 2) * self.beat:
                k = (t - t0) / (s.get("beats", 2) * self.beat)
                self._pillars(scn, xy, cols, 1 - k)
        for c in self.confetti:
            self._confetti(scn, t, *c)

    # ------------------------------------------------------------------
    def _floor(self, scn, t, b, pulse, xy, cols, glow, f):
        tile = f.get("tile", 0.5)
        half = f.get("half", 9.0)
        n = int(2 * half / tile)
        cx = np.linspace(-half + tile / 2, half - tile / 2, n)
        X, Y = np.meshgrid(cx, cx)
        r = np.hypot(X, Y)
        pat = f.get("pattern", "ripple")
        energy = f.get("energy", 1.0)
        if isinstance(energy, list):     # [[bar, level], ...] envelope
            bars = [e[0] * self.bar for e in energy]
            energy = float(np.interp(t, bars, [e[1] for e in energy]))
        if pat == "ripple":
            # Rings travelling outwards, one per beat.
            w = np.clip(1 - np.abs(((r / 2.2) - (b % 4)) % 4 - 0) / 0.35, 0, 1)
            base = 0.10 + 0.28 * w + 0.12 * pulse
        elif pat == "checker":
            par = ((np.floor(X / tile) + np.floor(Y / tile) + np.floor(b)) % 2)
            base = 0.08 + 0.35 * par * pulse
        else:
            base = 0.1 + 0.2 * pulse * np.ones_like(X)
        hue = (0.62 + 0.08 * np.sin(r * 0.6 - b * 0.5) + 0.02 * b) % 1
        rgb = np.stack(np.vectorize(colorsys.hsv_to_rgb)(hue, 0.85, 1.0), -1)
        # Soft round edge so the floor has no hard border.
        bright = base * energy * np.clip(1.15 - (r / half) ** 3, 0, 1)
        # Optionally, tiles under crabs take the crab's colour.
        if f.get("under", False) and xy is not None and len(xy):
            ix = np.clip(((xy[:, 0] + half) / tile).astype(int), 0, n - 1)
            iy = np.clip(((xy[:, 1] + half) / tile).astype(int), 0, n - 1)
            for k in range(len(xy)):
                if glow[k] > 0.05:
                    rgb[iy[k], ix[k]] = cols[k]
                    bright[iy[k], ix[k]] = max(bright[iy[k], ix[k]], 0.55 * glow[k])
        s = [tile * 0.47, tile * 0.47, 0.004]
        for i in range(n):
            for j in range(n):
                v = bright[i, j]
                if v < 0.04:
                    continue
                _add(scn, mujoco.mjtGeom.mjGEOM_BOX, s, [X[i, j], Y[i, j], 0.002],
                     [*(rgb[i, j] * v), 1.0], emission=1.0)

    def _lasers(self, scn, t, b, pulse, s):
        n = s.get("n", 8)
        y0 = s.get("y", 10.0)
        for k in range(n):
            x0 = (k - (n - 1) / 2) * 1.6
            a = np.array([x0, y0, 0.25])
            sweep = np.sin(2 * np.pi * (b / 8) + k * 0.7)
            tgt = np.array([sweep * 10, -12, 3.5 + 2.5 * np.sin(b * 0.5 + k)])
            d = tgt - a
            end = a + d / np.linalg.norm(d) * 40
            hue = (k / n + b * 0.03) % 1
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
            alpha = 0.35 + 0.4 * pulse
            _beam(scn, a, end, 0.012, [*rgb, alpha], emission=1.0)
            _beam(scn, a, end, 0.045, [*rgb, alpha * 0.18], emission=1.0)   # soft halo

    def _pillars(self, scn, xy, cols, k):
        for p, c in zip(xy, cols):
            a = [p[0], p[1], 0.12]
            _beam(scn, a, [p[0], p[1], 9.0], 0.02, [*c, 0.7 * k], emission=1.0)
            _beam(scn, a, [p[0], p[1], 9.0], 0.08, [*c, 0.15 * k], emission=1.0)

    def _trails(self, scn, t, cols, glow, tr):
        L = tr.get("len_s", 0.8)
        step = tr.get("step_s", 0.05)
        ks = np.arange(1, int(L / step) + 1)
        from .show import sample

        hist = [np.array([q[:3] for q in sample(self.logs, t - k * step)]) for k in ks]
        now = np.array([q[:3] for q in sample(self.logs, t)])
        for j, (k, h) in enumerate(zip(ks, hist)):
            moving = np.linalg.norm(now[:, :2] - h[:, :2], axis=1) > 0.02 * k
            fade = 1 - k / (len(ks) + 1)
            for i in np.nonzero(moving & (glow > 0.2))[0]:
                _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.035 * fade + 0.01, 0, 0],
                     [h[i, 0], h[i, 1], 0.05], [*cols[i], 0.6 * fade], emission=1.0)

    def _confetti(self, scn, t, t0, p0, v, col, spin, life):
        dt = t - t0
        if dt < 0 or dt > life:
            return
        drag = 1.2
        # Ballistic with linear drag, closed form.
        e = (1 - np.exp(-drag * dt)) / drag
        g = 9.81
        pos = p0 + v * e[..., None] if np.ndim(e) else p0 + v * e
        pos[:, 2] = p0[:, 2] + (v[:, 2] + g / drag) * e - g / drag * dt + 0.2
        alive = pos[:, 2] > 0.01
        fade = min(1.0, (life - dt) / 0.8)
        for i in np.nonzero(alive)[0]:
            a, c = spin[i] + dt * 6
            mat = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a) * np.cos(c), np.cos(a) * np.cos(c),
                            -np.sin(c)], [np.sin(a) * np.sin(c), np.cos(a) * np.sin(c), np.cos(c)]])
            _add(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.05, 0.03, 0.003], pos[i],
                 [*col[i], fade], emission=0.8, mat=mat.ravel())
