"""A show: a score of timed cues, simulated crab by crab, then rendered together.

A score is a list of cues on one timeline (seconds). Every crab receives every cue;
formation cues carry per-crab targets, dance cues may carry per-crab start offsets
(ripples). Simulation is per crab and independent, so it parallelises across processes.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np

from . import formation as F


@dataclass
class Cue:
    t: float                      # start time, seconds
    kind: str                     # "move" | "clip" | "hold"
    dur: float
    targets: np.ndarray | None = None   # (n, 2) for "move", indexed by crab
    policy: str = ""
    start_frame: int = 0
    offsets: np.ndarray | None = None   # (n,) seconds of delay per crab, for ripples
    yaw: float = -np.pi / 2       # heading to hold while moving/holding
    bob: tuple | None = None      # (hz_start, hz_end, amp_m, t_ref): body bounce while walking/holding
    jumps: list | None = None     # "party": per crab, the take-off times (s) of its jumps
    track: dict | None = None     # "track": per crab pivot (n,2), weight (n,), amp_rad, period_s
    only: np.ndarray | None = None  # (n,) bool: crabs that perform this cue; others hold


@dataclass
class Light:
    """Render-side lighting: colour and brightness per crab over time."""
    t: float
    colors: np.ndarray            # (n, 3)
    glow: np.ndarray              # (n,) 0..1
    fade: float = 0.3             # seconds to blend from the previous state


@dataclass
class Score:
    n: int
    start: np.ndarray             # (n, 2)
    start_yaw: float
    cues: list[Cue] = field(default_factory=list)
    lights: list[Light] = field(default_factory=list)
    duration: float = 0.0
    terrain: dict | None = None
    jumps: list = field(default_factory=list)   # (crab, take-off time) for render effects


RAMP = 0.25   # fraction of a move spent accelerating (and again decelerating)


def _trap(s, r=RAMP):
    """Trapezoidal velocity profile: position fraction (0..1) at time fraction s."""
    s = np.clip(s, 0.0, 1.0)
    vpk = 1.0 / (1.0 - r)
    if s < r:
        return 0.5 * vpk * s * s / r
    if s > 1 - r:
        e = 1 - s
        return 1 - 0.5 * vpk * e * e / r
    return 0.5 * vpk * r + vpk * (s - r)


def _dtrap(s, r=RAMP):
    s = np.clip(s, 0.0, 1.0)
    vpk = 1.0 / (1.0 - r)
    return vpk * min(1.0, s / r, (1 - s) / r)


def peak_speed(dist, dur):
    return dist / dur / (1.0 - RAMP)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def simulate_crab(args):
    """Run one crab through the score. Returns (times, qpos log)."""
    i, score = args
    from .runtime import Crab

    c = Crab(*score.start[i], yaw=score.start_yaw, seed=i, terrain=score.terrain)
    home = score.start[i].copy()     # where this crab is meant to be standing
    for cue in score.cues:
        # Fill any gap before this cue by standing at (and creeping back to) home.
        gap = cue.t - c.t
        if gap > 0.01:
            c.run_walk(gap, _hold_cmd(home, cue.yaw))
        pst = _bob(cue)
        if cue.only is not None and not cue.only[i]:
            if cue.t + cue.dur > c.t:
                c.run_walk(cue.t + cue.dur - c.t, _hold_cmd(home, cue.yaw))
            continue
        if cue.kind == "move":
            a, b = c.pos.copy(), cue.targets[i]
            c.run_walk(cue.dur, _move_cmd(a, b, cue.t, cue.dur, cue.yaw), posture=pst)
            home = b.copy()
        elif cue.kind == "hold":
            c.run_walk(cue.dur, _hold_cmd(home, cue.yaw), posture=pst)
        elif cue.kind == "track":
            c.run_walk(cue.dur, _track_cmd(home, cue, i), posture=pst)
        elif cue.kind == "party":
            # Jump whenever this crab's schedule says; bounce in between.
            for tj in cue.jumps[i]:
                pre = tj - 0.3 - c.t
                if pre > 0.01:
                    c.run_walk(pre, _hold_cmd(home, cue.yaw), posture=pst)
                if tj + 1.0 > cue.t + cue.dur:
                    break
                c.run_jump()
            rest = cue.t + cue.dur - c.t
            if rest > 0.01:
                c.run_walk(rest, _hold_cmd(home, cue.yaw), posture=pst)
        elif cue.kind == "jump":
            delay = float(cue.offsets[i]) if cue.offsets is not None else 0.0
            pre = max(0.0, delay + (cue.t - c.t))
            if pre > 0.01:
                c.run_walk(pre, _hold_cmd(home, cue.yaw))
            c.run_jump()
            rest = cue.t + cue.dur - c.t
            if rest > 0.01:
                c.run_walk(rest, _hold_cmd(home, cue.yaw))
        elif cue.kind == "clip":
            delay = float(cue.offsets[i]) if cue.offsets is not None else 0.0
            if delay > 0.01:
                c.run_walk(delay, _hold_cmd(home, cue.yaw))
            c.run_clip(cue.policy, cue.dur - delay, start_frame=cue.start_frame)
    end = score.duration - c.t
    if end > 0.01:
        c.run_walk(end, _hold_cmd(home, score.cues[-1].yaw if score.cues else 0.0))
    return np.array(c.log_t), np.array(c.log_qpos)


def _track_cmd(home, cue, i):
    """Follow a target that swings about a pivot: theta(t) = w * amp * sin(2 pi t / period).
    Crabs with weight 0 stand still; the rest sweep an arc (a waving claw)."""
    tr = cue.track
    if "path" in tr:
        # Waypoints, one hop per step: move for `move` of the step, then settle.
        P = np.asarray(tr["path"][i], dtype=float)
        K = len(P) - 1
        mv = tr.get("move", 0.85)

        def ref_p(t):
            u = np.clip((t - cue.t) / cue.dur, 0, 1) * K
            k = min(int(u), K - 1)
            f = np.clip((u - k) / mv, 0, 1)
            return P[k] + (P[k + 1] - P[k]) * (f * f * (3 - 2 * f))

        def f_path(crab):
            t = crab.t
            p, p2 = ref_p(t), ref_p(t + 0.02)
            v = (p2 - p) / 0.02 + 2.0 * (p - crab.pos)
            return _body_cmd(crab, v, cue.yaw)
        return f_path
    if "offset" in tr:
        off = np.asarray(tr["offset"][i], dtype=float)
        ramp = tr.get("ramp", 0.4)

        def prof(t):
            u = np.clip((t - cue.t) / cue.dur, 0, 1)
            up = np.clip(u / ramp, 0, 1)
            down = np.clip((1 - u) / ramp, 0, 1)
            x = min(up, down)
            return x * x * (3 - 2 * x)

        def f_off(crab):
            t = crab.t
            p, p2 = home + off * prof(t), home + off * prof(t + 0.02)
            v = (p2 - p) / 0.02 + 2.0 * (p - crab.pos)
            return _body_cmd(crab, v, cue.yaw)
        return f_off
    piv = np.asarray(tr["pivot"][i], dtype=float)
    w = float(tr["weight"][i])
    amp, per = tr["amp"], tr["period"]
    r = home - piv

    def ref(t):
        u = np.clip((t - cue.t) / 0.5, 0, 1) * np.clip((cue.t + cue.dur - t) / 0.5, 0, 1)
        th = w * amp * u * np.sin(2 * np.pi * (t - cue.t) / per)
        c, s = np.cos(th), np.sin(th)
        return piv + np.array([c * r[0] - s * r[1], s * r[0] + c * r[1]])

    def f(crab):
        t = crab.t
        p, p2 = ref(t), ref(t + 0.02)
        v = (p2 - p) / 0.02 + 2.0 * (p - crab.pos)
        return _body_cmd(crab, v, cue.yaw)
    return f


def _bob(cue):
    """Posture command for a beat bounce whose rate sweeps hz_start -> hz_end over the cue.

    Height only: [twist, pitch, roll, height offset]. Phase is continuous across the sweep
    so the bounce speeds up smoothly, and peaks land on beats when hz is a beat multiple.
    """
    if not cue.bob:
        return (0.0, 0.0, 0.0, 0.0)
    f0, f1, amp = cue.bob[:3]
    t0, T = cue.t, cue.dur

    def f(crab):
        u = np.clip((crab.t - t0) / T, 0, 1)
        # phase = integral of a linear frequency sweep
        ph = 2 * np.pi * T * (f0 * u + 0.5 * (f1 - f0) * u * u)
        a = amp * (1 - 0.4 * (f1 > 3) * u)         # fast bounces need a smaller stroke
        return (0.0, 0.0, 0.0, a * np.cos(ph))
    return f


def _body_cmd(crab, v_world, yaw_des, vmax=0.95):
    sp = np.linalg.norm(v_world)
    if sp > vmax:
        v_world = v_world * (vmax / sp)
    yaw = crab.yaw
    cy, sy = np.cos(yaw), np.sin(yaw)
    vx = cy * v_world[0] + sy * v_world[1]
    vy = -sy * v_world[0] + cy * v_world[1]
    wz = float(np.clip(2.0 * _wrap(yaw_des - yaw), -1.5, 1.5))
    return (vx, vy, wz)


def _move_cmd(a, b, t0, dur, yaw_des):
    def f(crab):
        s = (crab.t - t0) / dur
        ref = a + (b - a) * _trap(s)
        vff = (b - a) * _dtrap(s) / dur
        return _body_cmd(crab, vff + 2.0 * (ref - crab.pos), yaw_des)
    return f


def _hold_cmd(xy, yaw_des):
    def f(crab):
        err = xy - crab.pos
        v = 1.5 * err if np.linalg.norm(err) > 0.03 else np.zeros(2)
        cmd = _body_cmd(crab, v, yaw_des)
        if abs(cmd[2]) < 0.05 and np.linalg.norm(v) < 1e-6:
            return (0.0, 0.0, 0.0)
        return cmd
    return f


def simulate(score: Score, procs: int = 4):
    with mp.get_context("spawn").Pool(procs) as pool:
        out = pool.map(simulate_crab, [(i, score) for i in range(score.n)], chunksize=1)
    return out


def sample(logs, t):
    """qpos of every crab at time t (nearest logged frame at or before t)."""
    if hasattr(logs, "frame"):          # dense.DenseLogs
        return logs.frame(t)
    res = []
    for times, q in logs:
        k = int(np.searchsorted(times, t, side="right") - 1)
        res.append(q[max(0, min(k, len(q) - 1))])
    return res


def lights_at(score: Score, t: float):
    """Blend lighting cues (plus a white flash on every jump): (colors (n,3), glow (n,))."""
    cols, glow = _lights_at(score, t)
    for i, tj in score.jumps:
        u = (t - tj) / 0.35
        if 0 <= u <= 1:
            f = 1 - u
            cols[i] = cols[i] * (1 - f) + f
            glow[i] = max(glow[i], 1.0)
    return cols, glow


def _lights_at(score: Score, t: float):
    cols = np.zeros((score.n, 3))
    glow = np.zeros(score.n)
    for L in score.lights:
        if L.t > t:
            break
        w = 1.0 if L.fade <= 0 else float(np.clip((t - L.t) / L.fade, 0, 1))
        cols = cols * (1 - w) + L.colors * w
        glow = glow * (1 - w) + L.glow * w
    return cols, glow


def collisions(logs, t_step=0.1, min_dist=0.45):
    """Closest approach between any two crab bases over the show (a planning check)."""
    times = logs[0][0]
    worst = (np.inf, None, None)
    for t in np.arange(0, times[-1], t_step):
        xy = np.array([q[:2] for q in sample(logs, t)])
        d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        k = np.unravel_index(np.argmin(d), d.shape)
        if d[k] < worst[0]:
            worst = (float(d[k]), float(t), k)
    return worst
