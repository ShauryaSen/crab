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

    c = Crab(*score.start[i], yaw=score.start_yaw, seed=i)
    home = score.start[i].copy()     # where this crab is meant to be standing
    for cue in score.cues:
        # Fill any gap before this cue by standing at (and creeping back to) home.
        gap = cue.t - c.t
        if gap > 0.01:
            c.run_walk(gap, _hold_cmd(home, cue.yaw))
        if cue.kind == "move":
            a, b = c.pos.copy(), cue.targets[i]
            c.run_walk(cue.dur, _move_cmd(a, b, cue.t, cue.dur, cue.yaw))
            home = b.copy()
        elif cue.kind == "hold":
            c.run_walk(cue.dur, _hold_cmd(home, cue.yaw))
        elif cue.kind == "clip":
            delay = float(cue.offsets[i]) if cue.offsets is not None else 0.0
            if delay > 0.01:
                c.run_walk(delay, _hold_cmd(home, cue.yaw))
            c.run_clip(cue.policy, cue.dur - delay, start_frame=cue.start_frame)
    end = score.duration - c.t
    if end > 0.01:
        c.run_walk(end, _hold_cmd(home, score.cues[-1].yaw if score.cues else 0.0))
    return np.array(c.log_t), np.array(c.log_qpos)


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
    res = []
    for times, q in logs:
        k = int(np.searchsorted(times, t, side="right") - 1)
        res.append(q[max(0, min(k, len(q) - 1))])
    return res


def lights_at(score: Score, t: float):
    """Blend lighting cues: returns (colors (n,3), glow (n,))."""
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
