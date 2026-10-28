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
from .show import Cue, Light, Score, peak_speed

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
    "spooky": lambda p, n: [colorsys.hsv_to_rgb(0.78 - 0.45 * t, 0.85, 1)
                            for t in (p[:, 0] - p[:, 0].min()) / (np.ptp(p[:, 0]) + 1e-9)],
    "pumpkin": lambda p, n: [(1.0, 0.42 + 0.1 * t, 0.02) for t in np.linspace(0, 1, n)],
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
        elif shape == "crab":
            pts = F.crab(n, 8.0)
        elif shape == "grid":
            pts = F.grid(n, SPACING)
        elif shape == "rows":
            k = act.get("rows", 3)
            cols = int(np.ceil(n / k))
            gap = act.get("gap", 1.4) * SPACING
            pts = np.array([((c - (cols - 1) / 2) * gap + (r % 2) * gap / 2,
                             ((k - 1) / 2 - r) * SPACING * 1.2)
                            for r in range(k) for c in range(cols)][:n])
        else:
            raise ValueError(shape)
    return F.fit_spacing(pts, SPACING)


def scatter(n: int, radius: float, seed: int = 1) -> np.ndarray:
    """Loose, irregular start positions: a jittered grid (looks random, packs safely)."""
    rng = np.random.default_rng(seed)
    g = F.grid(n, SPACING * 1.35)
    return g + rng.uniform(-1, 1, g.shape) * SPACING * 0.15


def scatter_random(n: int, radius: float, seed: int = 1) -> np.ndarray:
    """Random start positions with no two closer than SPACING."""
    rng = np.random.default_rng(seed)
    pts = []
    tries = 0
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 2)
        tries += 1
        if all(np.linalg.norm(p - q) >= SPACING * 1.1 for q in pts):
            pts.append(p)
        if tries > 200 * n:      # too tight for random packing: widen the field
            radius *= 1.1
            tries = 0
    return np.array(pts)


def _bob_spec(act, beat):
    if "every" not in act:
        return None
    e0, e1 = act["every"] if isinstance(act["every"], list) else (act["every"], act["every"])
    return (1.0 / (e0 * beat), 1.0 / (e1 * beat), act.get("amp", 0.035))


def _ripple(act, pos, rng, beat):
    ripple = act.get("ripple")
    if not ripple:
        return None
    c = pos.mean(0)
    if ripple == "center":
        d = np.linalg.norm(pos - c, axis=1)
    elif ripple == "left":
        d = pos[:, 0] - pos[:, 0].min()
    else:
        d = rng.uniform(0, 1, len(pos))
    return d / (np.ptp(d) + 1e-9) * act.get("ripple_s", beat * 2)


def compose(plan: dict) -> tuple[Score, dict]:
    n = plan["n"]
    bar = 4 * 60.0 / plan["bpm"]
    beat = bar / 4
    total_bars = sum(b for _, b in plan["sections"])
    rng = np.random.default_rng(plan.get("seed", 3))
    start = scatter(n, plan.get("scatter_radius", 0.42 * np.sqrt(n)), plan.get("seed", 1))
    score = Score(n=n, start=start, start_yaw=FACE_CAMERA, duration=total_bars * bar,
                  terrain=plan.get("terrain"))
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
            img_cols = None
            arts = {"pixel_crab": (F.CRAB_ART, F.CRAB_PALETTE),
                    "pumpkin": (F.PUMPKIN_ART, F.PUMPKIN_PALETTE)}
            if act.get("shape") in arts:
                art, pal = arts[act["shape"]]
                goal, img_cols = F.pixel_art(art, pal, act.get("spacing", 0.8))
                assert len(goal) == n, f"{act['shape']} needs n={len(goal)} crabs"
            else:
                goal = shape_points(act, act.get("count", n))
            k = len(goal)
            from scipy.optimize import linear_sum_assignment

            def pick(goal):
                """Which crab takes which goal (all crabs, or the k nearest when k < n)."""
                cost = ((pos[:, None, :] - goal[None, :, :]) ** 2).sum(-1)
                rows, cols_ = linear_sum_assignment(cost)
                return rows, cols_
            # Shift the new shape to where the swarm already is, then assign.
            for _ in range(3):
                rows, cols_ = pick(goal)
                goal = goal + (pos[rows] - goal[cols_]).mean(0)
            goal = goal - goal.mean(0) * act.get("recentre", 0.5)
            rows, cols_ = pick(goal)
            chosen = np.zeros(n, bool)
            chosen[rows] = True
            targets = pos.copy()
            targets[rows] = goal[cols_]
            perm = np.zeros(n, int)
            perm[rows] = cols_
            if k < n:
                # Everyone not in the shape keeps running, out into the dark.
                c = pos.mean(0)
                out = pos[~chosen] - c
                out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
                targets[~chosen] = pos[~chosen] + out * act.get("runaway_m", 5.0)
            if img_cols is not None:
                img_cols = img_cols[perm]
                meta["art_cols"] = img_cols.copy()
                if act.get("dark_face"):
                    # Carved pixels (the brightest) start unlit, as plain skin.
                    face = img_cols.max(1) >= 0.9
                    meta["face"] = face
                    skin = img_cols[~face].mean(0)
                    img_cols = img_cols.copy()
                    img_cols[face] = skin
            far = float(np.linalg.norm(targets - pos, axis=1)[chosen].max())
            vpk = peak_speed(far, dur)
            meta.setdefault("warnings", [])
            if vpk > 0.75:
                meta["warnings"].append(f"bar {act['bar']}: longest walk {far:.1f} m needs "
                                        f"{vpk:.2f} m/s peak; give it more bars")
            score.cues.append(Cue(t, "move", dur, targets=targets, yaw=FACE_CAMERA,
                                  bob=_bob_spec(act, beat)))
            new_cols = img_cols if img_cols is not None else \
                np.array(PALETTES[act.get("palette", "white")](targets, n))
            new_glow = chosen.astype(float)
            if k < n:
                new_cols = new_cols.copy()
                new_cols[~chosen] = cols[~chosen]
            score.lights.append(Light(t, new_cols, new_glow, fade=dur * 0.8))
            cols, glow, pos = new_cols, new_glow, targets
            meta["formations"].append({"bar": act["bar"], "label": act.get("text", act.get("shape"))})
        elif do == "bounce":
            # Stand in place and bounce; rate in beats per bounce, e.g. "every": [2, 0.5]
            # starts on every other beat and ends four bounces per beat.
            score.cues.append(Cue(t, "hold", dur, yaw=FACE_CAMERA, bob=_bob_spec(act, beat)))
        elif do == "jump":
            offsets = _ripple(act, pos, rng, beat)
            # Take-off comes ~0.3 s after the jump starts: start early so it lands on the beat.
            score.cues.append(Cue(t - 0.3, "jump", dur, offsets=offsets, yaw=FACE_CAMERA))
            od = offsets if offsets is not None else np.zeros(n)
            score.jumps += [(i, t + float(od[i])) for i in range(n)]
            flash_t = (offsets if offsets is not None else np.zeros(n))
            for k in range(6):
                tk = t + float(flash_t.max()) * k / 5
                f = np.clip(1 - np.abs(flash_t - (tk - t)) / (beat * 0.5), 0, 1) \
                    if offsets is not None else np.ones(n) * (k == 0)
                score.lights.append(Light(tk, cols * (1 - f[:, None]) + f[:, None], np.ones(n),
                                          fade=0.04))
            score.lights.append(Light(t + float(flash_t.max()) + beat, cols.copy(), np.ones(n),
                                      fade=0.3))
        elif do == "wave_claws":
            # Crabs forming the claws swing about the claw's shoulder (giant crab waves).
            # Claw crabs are those beyond |x| > claw_x and above y > claw_y.
            cx, cy = pos.mean(0)
            side = np.sign(pos[:, 0] - cx)
            in_claw = (np.abs(pos[:, 0] - cx) > act.get("claw_x", 3.5)) & \
                (pos[:, 1] - cy > act.get("claw_y", 0.5))
            pivot = np.stack([cx + side * act.get("pivot_x", 3.2),
                              np.full(n, cy + act.get("pivot_y", 0.0))], 1)
            weight = in_claw * (-side)      # mirror: both claws wave outward together
            score.cues.append(Cue(t, "track", dur, yaw=FACE_CAMERA, track={
                "pivot": pivot, "weight": weight.astype(float),
                "amp": act.get("amp_deg", 18) * np.pi / 180,
                "period": act.get("beats_per_wave", 2) * beat}))
        elif do == "jump_party":
            # Every crab jumps again and again at its own times: some land on the beat,
            # most do not, so the swarm boils. min_gap covers the jump plus landing.
            rate = act.get("per_bar", 0.9)
            gap = act.get("min_gap", 1.55)
            on_beat = act.get("on_beat", 0.35)
            jumps = []
            for i in range(n):
                ts_, tt = [], t + rng.uniform(0, beat * 2)
                while tt < t + dur - 1.0:
                    if rng.uniform() < on_beat:
                        tt = t + np.ceil((tt - t) / (beat / 2)) * (beat / 2)
                    ts_.append(float(tt))
                    tt += gap + rng.exponential(bar / max(rate, 1e-3) - gap if bar / rate > gap else 0.2)
                jumps.append(ts_)
                score.jumps += [(i, x) for x in ts_]
            score.cues.append(Cue(t, "party", dur, jumps=jumps, yaw=FACE_CAMERA,
                                  bob=_bob_spec({"every": 1, "amp": 0.03}, beat)))
        elif do == "shift":
            # The whole formation walks together, keeping its shape (no re-assignment).
            targets = pos + np.asarray(act["by"], dtype=float)
            score.cues.append(Cue(t, "move", dur, targets=targets, yaw=FACE_CAMERA))
            pos = targets
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
        elif do == "ignite":
            # Light the hidden face: upper pixels (eyes) first, then the rest (mouth).
            face = meta["face"]
            art = meta["art_cols"]
            idx = np.nonzero(face)[0]
            ys = pos[idx, 1]
            eyes = idx[ys >= np.median(ys)]
            mouth = idx[ys < np.median(ys)]
            for k, group in enumerate((eyes, mouth)):
                c2 = cols.copy()
                c2[group] = art[group]
                g2 = glow.copy()
                g2[group] = 1.0
                cols, glow = c2, g2
                score.lights.append(Light(t + k * act.get("gap_beats", 1) * beat, cols.copy(),
                                          glow.copy(), fade=0.12))
        elif do == "scale":
            # Every crab walks straight out from (or in toward) the centre: a rigid scaling,
            # so no two paths cross. "flee" uses it at top speed.
            c = pos.mean(0)
            targets = c + (pos - c) * act["by"]
            score.cues.append(Cue(t, "move", dur, targets=targets, yaw=FACE_CAMERA,
                                  bob=_bob_spec(act, beat)))
            far = float(np.linalg.norm(targets - pos, axis=1).max())
            meta.setdefault("warnings", [])
            if peak_speed(far, dur) > 1.0:
                meta["warnings"].append(f"bar {act['bar']}: scale needs {peak_speed(far, dur):.2f} m/s")
            pos = targets
        elif do == "flicker":
            # Candle flicker: the brightest crabs (a carved face) waver every 16th note.
            lum = cols.max(1)
            sel = lum >= act.get("min_lum", 0.9)
            step = beat / 4
            base = glow.copy()
            for k in range(int(act.get("bars", 1) * 16)):
                g = base.copy()
                g[sel] = np.clip(rng.uniform(0.45, 1.0, sel.sum()) ** 0.5, 0, 1)
                score.lights.append(Light(t + k * step, cols.copy(), g, fade=step * 0.8))
            score.lights.append(Light(t + dur, cols.copy(), base, fade=0.1))
        elif do == "color":
            cols = np.array(PALETTES[act["palette"]](pos, n))
            score.lights.append(Light(t, cols.copy(), glow.copy(), fade=act.get("fade", beat)))
        else:
            raise ValueError(do)
    score.lights.sort(key=lambda L: L.t)
    score.cues.sort(key=lambda c: c.t)
    return score, meta
