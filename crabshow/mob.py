"""The mob: a thousand crabs behaving like a thousand crabs, plus one patch dancing in sync.

A 2-D crowd simulation (10 Hz, soft separation) plans where every crab in the mob walks:
each leaves its slot after the ripple jump passes it, wanders between drifting hotspots
and its own neighbourhood, stops now and then to dance or jump, and at the end scuttles
back to its own slot. In the middle of the mob one patch scurries into a star, spins,
folds into a heart that beats, then joins the walk home. The plan is turned into a
per-crab script (see show.simulate_crab, cue kind "script"):

    ("jump", t_takeoff)
    ("clip", t0, t1, move, start_frame)
    ("path", t0, dt, xy (k, 2), yaw (k,))      reference sampled every dt from t0

Crabs outside the mob party in place (dance, gesture or jump) and are back facing the
camera before the zoom out.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

DT = 0.1
DANCES = ["dance_dream_wings", "dance_maze", "dance_waist", "dance"]
GESTURES = ["gesture_hello", "gesture_paw", "gesture_salute", "gesture_bow"]
FACE_CAMERA = -np.pi / 2


def star_points(k, r_out, r_in_frac=0.45, ring=0, ring_r=0.6):
    """k points: an evenly spaced 5-point star outline, plus `ring` points on a small
    ring at the centre. Returns (pts (k,2), arm (k,) 0..4 or -1 for the ring)."""
    m = k - ring
    ang = np.pi / 2 + np.arange(10) * np.pi / 5
    rad = np.where(np.arange(10) % 2 == 0, r_out, r_out * r_in_frac)
    V = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
    seg = np.linalg.norm(np.roll(V, -1, 0) - V, axis=1)
    s = (np.arange(m) + 0.5) / m * seg.sum()
    cum = np.concatenate([[0], np.cumsum(seg)])
    j = np.searchsorted(cum, s, side="right") - 1
    f = (s - cum[j]) / seg[j]
    P = V[j] + (np.roll(V, -1, 0)[j] - V[j]) * f[:, None]
    a = (np.degrees(np.arctan2(P[:, 1], P[:, 0])) - 90 + 36) % 360
    arm = (a // 72).astype(int)
    if ring:
        th = np.pi / 2 + np.arange(ring) * 2 * np.pi / ring
        P = np.concatenate([P, np.stack([ring_r * np.cos(th), ring_r * np.sin(th)], 1)])
        arm = np.concatenate([arm, -np.ones(ring, int)])
    return P, arm


def heart_points(k, size, inner=0, inner_scale=0.42):
    """k points: an evenly spaced heart outline plus `inner` points on a small heart."""
    def outline(m, sc):
        t = np.linspace(0, 2 * np.pi, 2000)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        C = np.stack([x, y + 2.5], 1) / 32 * 2 * sc
        d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
        s = (np.arange(m) + 0.5) / m * d[-1]
        return np.stack([np.interp(s, d, C[:, 0]), np.interp(s, d, C[:, 1])], 1)
    P = outline(k - inner, size)
    if inner:
        P = np.concatenate([P, outline(inner, size * inner_scale)])
    return P, np.concatenate([np.zeros(k - inner, int), np.ones(inner, int)])


def _assign(a, b):
    from scipy.optimize import linear_sum_assignment
    _, j = linear_sum_assignment(((a[:, None] - b[None]) ** 2).sum(-1))
    return b[j], j


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def build(pos, rng, beat, *, hero, t_drop, path_to, zone_m, patch_r, ripple_s, ripple_pow,
          ripple_d, star_t, spin_t, heart_t, beat_t, home_t, home_by, party_until,
          dark=None, wander_m=2.6, speed=(0.25, 0.55), hot_m2=30.0, sep=0.36,
          moves=None, popper_frac=0.04):
    """Plan the whole party. Times are in seconds. Returns a dict with per-crab scripts,
    jump and popper events, the patch (indices, arms) and the light keys of the patch."""
    n = len(pos)
    dark = np.zeros(n, bool) if dark is None else dark
    h = pos[hero]
    P = h + np.asarray(path_to, float)           # the patch centre, where the dolly ends
    # Distance from the dolly's line (hero -> P): the mob is the crabs near that line.
    ab = P - h
    u = np.clip(((pos - h) @ ab) / (ab @ ab), 0, 1)
    d_line = np.linalg.norm(pos - (h + u[:, None] * ab), axis=1)
    patch = np.linalg.norm(pos - P, axis=1) < patch_r
    mob = (d_line < zone_m) & ~patch
    # The ripple: one jump each, travelling out from the hero (slow near, fast far).
    d = np.linalg.norm(pos - h, axis=1)
    t_jump = t_drop + 0.55 + ripple_s * (d / ripple_d) ** ripple_pow
    t_jump[hero] = t_drop
    scripts = [[("jump", float(t_jump[i]))] for i in range(n)]
    jumps = [(i, float(t_jump[i])) for i in range(n)]
    poppers = []

    # ---- the patch: star -> spin -> heart -> beat, positions as functions of time ----
    pidx = np.nonzero(patch)[0]
    k = len(pidx)
    ring = max(6, k // 9)
    # Star size: outline spacing about 0.5 m.
    r_out = 0.5 * (k - ring) / 6.89
    S0, arm = star_points(k, r_out, ring=ring)
    S0, j = _assign(pos[pidx] - P, S0)
    arm = arm[j]
    spin_w = 0.5 / r_out                          # outer tips at 0.5 m/s
    ring_w = -0.9                                 # the centre ring turns the other way
    S_end = _rot_all(S0, arm, spin_w * (spin_t[1] - spin_t[0]), ring_w * (spin_t[1] - spin_t[0]))
    # The heart whose size and height make the longest fold shortest.
    best = None
    for sz in (0.75, 0.8, 0.9, 1.0):
        for dy in np.linspace(-0.25, 0.1, 8) * r_out:
            Hc, pc = heart_points(k, size=r_out * sz, inner=ring)
            Hc, jc = _assign(S_end, Hc + [0.0, dy])
            far = np.linalg.norm(Hc - S_end, axis=1).max()
            if best is None or far < best[0]:
                best = (far, Hc, pc[jc])
    _, H0, part = best
    pirouette = rng.uniform(0.9, 1.3, k) * np.where(rng.uniform(size=k) < 0.5, 1, -1)

    def patch_at(t):
        """Reference positions of the patch crabs at time t (before the walk home)."""
        home = pos[pidx] - P
        if t < star_t[0]:
            return home + P
        if t < star_t[1]:
            e = _smooth((t - star_t[0]) / (star_t[1] - star_t[0]))
            return home + (S0 - home) * e + P
        if t < spin_t[1]:
            tt = min(t, spin_t[1]) - spin_t[0]
            return _rot_all(S0, arm, spin_w * max(tt, 0), ring_w * max(tt, 0)) + P
        if t < heart_t[1]:
            e = _smooth((t - heart_t[0]) / (heart_t[1] - heart_t[0]))
            return S_end + (H0 - S_end) * e + P
        return H0 + P

    meta = {"patch": pidx, "arm": arm, "part": part, "P": P, "mob": np.nonzero(mob)[0],
            "r_out": r_out}

    # ---- the crowd simulation over mob + patch ----
    A = np.nonzero(mob | patch)[0]
    na = len(A)
    is_patch = patch[A]
    home = pos[A].copy()
    p = home.copy()
    release = t_jump[A] + 1.35 + rng.uniform(0, 0.8, na)
    ret = home_t[0] + rng.uniform(0, home_t[1] - home_t[0], na)
    ret[is_patch] = home_t[1]                      # the patch leaves last, all together
    spd = rng.uniform(*speed, na)
    side = np.where(rng.uniform(size=na) < 0.5, 1.0, -1.0)
    lo, hi = pos[A].min(0), pos[A].max(0)
    nh = max(3, int(np.prod(hi - lo) / hot_m2))
    hot = rng.uniform(lo, hi, (nh, 2))
    hot_v = rng.normal(0, 0.15, (nh, 2))
    mode = np.zeros(na, int)        # 0 waiting, 1 walking, 2 busy (dance/jump), 3 returning, 4 home, 5 patch
    target = home.copy()
    busy_until = np.zeros(na)
    yaw = np.full(na, FACE_CAMERA)
    t0 = float(release.min()) - 0.5
    t1 = home_by + 0.5
    steps = int(np.ceil((t1 - t0) / DT))
    rec_p = np.zeros((steps, na, 2))
    rec_y = np.zeros((steps, na))
    moving = np.zeros((steps, na), bool)
    events = [[] for _ in range(na)]     # (t, kind, payload) stops: dance/jump/popper
    pmap = {a: q for q, a in enumerate(pidx)}
    pq = np.array([pmap[a] for a in A[is_patch]])
    patch_pos_rows = np.nonzero(is_patch)[0]
    mv_names = list(moves) if moves else DANCES
    mv_w = np.array(list(moves.values()), float) if moves else np.ones(len(DANCES))
    mv_w /= mv_w.sum()
    P_avoid = r_out + 0.8
    late = np.zeros(na, bool)
    for s in range(steps):
        t = t0 + s * DT
        hot += hot_v * DT
        hot_v += rng.normal(0, 0.05, hot_v.shape)
        hot_v *= 0.98
        # Patch crabs follow the choreography until they head home.
        pp = patch_at(t)
        on_patch = is_patch & (t < ret)
        if on_patch.any():
            p[patch_pos_rows] = np.where(on_patch[patch_pos_rows, None], pp[pq], p[patch_pos_rows])
            mode[on_patch] = 5
        # Who starts / ends what.
        start = (mode == 0) & (t >= release) & ~is_patch
        mode[start] = 1
        target[start] = _new_target(rng, home[start], hot, wander_m)
        done = (mode == 2) & (t >= busy_until)
        mode[done] = 1
        target[done] = _new_target(rng, home[done], hot, wander_m)
        go_home = (t >= ret) & (mode != 3) & (mode != 4)
        mode[go_home] = 3
        target[go_home] = home[go_home]
        # Walkers that reach their target pick the next thing to do.
        reached = (mode == 1) & (np.linalg.norm(target - p, axis=1) < 0.15)
        for q in np.nonzero(reached)[0]:
            if t > ret[q] - 4.0:
                target[q] = _new_target(rng, home[q:q + 1], hot, wander_m * 0.5)[0]
                continue
            r = rng.uniform()
            if r < 0.30:
                mv = mv_names[rng.choice(len(mv_names), p=mv_w)]
                dur = rng.uniform(2.0, 4.0)
                if mv == "jumps":
                    events[q].append((t, "jump", None))
                    dur = 1.9
                elif mv == "popper":
                    events[q].append((t, "popper", dur))
                else:
                    events[q].append((t, "clip", (mv, dur)))
                mode[q], busy_until[q] = 2, t + dur
            else:
                target[q] = _new_target(rng, home[q:q + 1], hot, wander_m)[0]
        # Desired velocities.
        v = np.zeros((na, 2))
        walk = (mode == 1) | (mode == 3)
        dv = target - p
        dist = np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9
        sp = np.where(mode == 3, np.maximum(spd, 0.55), spd)
        if (mode == 3).any():
            # Hurry if late: reach home by home_by whatever happens.
            left = np.maximum(home_by - 0.6 - t, 0.3)
            need = dist[:, 0] / left
            sp = np.where(mode == 3, np.clip(np.maximum(sp, need * 1.15), 0, 0.8), sp)
        v[walk] = (dv / dist * np.minimum(sp[:, None], dist / 0.5))[walk]
        # Keep out of the dancing patch while it dances.
        if t < home_t[1]:
            dp = p - P
            r = np.linalg.norm(dp, axis=1, keepdims=True) + 1e-9
            inside = (r[:, 0] < P_avoid) & ~is_patch & walk & (mode != 3)
            v[inside] += (dp / r)[inside] * 0.6
        newp = p + v * DT
        # Soft separation between everyone in the mob; settled crabs do not get pushed.
        tree = cKDTree(newp)
        pairs = tree.query_pairs(sep, output_type="ndarray")
        if len(pairs):
            a_, b_ = pairs[:, 0], pairs[:, 1]
            dd = newp[a_] - newp[b_]
            L = np.linalg.norm(dd, axis=1, keepdims=True) + 1e-9
            push = dd / L * (sep - L) * 0.5
            free = ((mode == 1) | (mode == 3))
            fa, fb = free[a_].astype(float), free[b_].astype(float)
            wa = np.where(fa + fb > 0, fa / np.maximum(fa + fb, 1), 0) * 2 * 0.5
            wb = np.where(fa + fb > 0, fb / np.maximum(fa + fb, 1), 0) * 2 * 0.5
            corr = np.zeros_like(newp)
            np.add.at(corr, a_, push * wa[:, None])
            np.add.at(corr, b_, -push * wb[:, None])
            newp = newp + np.clip(corr, -0.03, 0.03)
        arrived = (mode == 3) & (np.linalg.norm(home - newp, axis=1) < 0.04)
        newp[arrived] = home[arrived]
        mode[arrived] = 4
        step_v = (newp - p) / DT
        mv_now = np.linalg.norm(step_v, axis=1) > 0.03
        # Crabs scuttle sideways: face across the direction they walk.
        hd = np.arctan2(step_v[:, 1], step_v[:, 0]) + side * np.pi / 2
        want = np.where(mv_now, hd, yaw)
        want = np.where(mode == 4, FACE_CAMERA, want)
        want = np.where((mode == 3) & (np.linalg.norm(home - newp, axis=1) < 0.4), FACE_CAMERA, want)
        dyaw = (want - yaw + np.pi) % (2 * np.pi) - np.pi
        yaw = yaw + np.clip(dyaw, -0.12, 0.12)
        if on_patch.any():
            # Patch crabs pirouette while the star turns.
            spinning = on_patch & (t >= spin_t[0]) & (t < spin_t[1])
            rows = patch_pos_rows[spinning[patch_pos_rows]]
            if len(rows):
                yaw[rows] += pirouette[pq[spinning[patch_pos_rows]]] * DT
            mv_now = mv_now | (on_patch & (t >= star_t[0]))
        p = newp
        rec_p[s], rec_y[s], moving[s] = p, yaw, mv_now
    late[:] = mode != 4
    meta["late"] = int(late.sum())
    meta["max_err"] = float(np.linalg.norm(p - home, axis=1).max())

    # The heart's inner crabs jump on every bar while it beats; the patch crabs hold still
    # for those jumps, so cut their path there.
    jump_at = {}
    for kk, q in enumerate(patch_pos_rows):
        if part[pq[kk]] == 1:
            jump_at[q] = [float(x) for x in np.arange(beat_t[0], beat_t[1] - 1.0, 4 * beat)]

    # ---- scripts from the recordings ----
    tgrid = t0 + np.arange(steps) * DT
    for q in range(na):
        i = A[q]
        sc = scripts[i]
        stops = sorted(events[q])
        for tj in jump_at.get(q, []):
            stops.append((tj - 0.35, "jump_inplace", tj))
        stops.sort(key=lambda e: e[0])
        # Moving stretches between stops become paths.
        mv_q = moving[:, q]
        busy = np.zeros(steps, bool)
        for (ts, kind, pay) in stops:
            if kind == "clip":
                dur = pay[1]
            elif kind == "popper":
                dur = pay
            elif kind == "jump_inplace":
                dur = 1.5
            else:
                dur = 1.9
            a0 = int((ts - t0) / DT)
            busy[max(a0, 0):max(a0 + int(dur / DT), 0)] = True
        active = mv_q & ~busy
        # Group consecutive active samples into paths (bridge 1-sample gaps).
        k0 = None
        for s in range(steps + 1):
            on = s < steps and active[s]
            if on and k0 is None:
                k0 = s
            elif not on and k0 is not None:
                a, b = max(k0 - 1, 0), min(s, steps - 1)
                if b - a >= 2:
                    sc.append(("path", float(tgrid[a]), DT, _smooth_path(rec_p[a:b + 1, q]),
                               np.unwrap(rec_y[a:b + 1, q]).copy()))
                k0 = None
        for (ts, kind, pay) in stops:
            if kind == "jump":
                sc.append(("jump", float(ts + 0.6)))
                jumps.append((i, float(ts + 0.6)))
            elif kind == "jump_inplace":
                sc.append(("jump", float(pay)))
                jumps.append((i, float(pay)))
            elif kind == "popper":
                sc.append(("clip", float(ts), float(ts + pay), "gesture_hello", 0))
                poppers.append((i, float(ts + rng.uniform(0.5, 1.2))))
            else:
                mv, dur = pay
                sf = int(rng.integers(0, 150)) if mv.startswith("dance") else 0
                sc.append(("clip", float(ts), float(ts + dur), mv, sf))
        sc.sort(key=lambda e: e[1])
    # ---- everyone else parties in place ----
    rest = np.nonzero(~(mob | patch))[0]
    for i in rest:
        if dark[i]:
            continue
        t_s = t_jump[i] + 1.7 + rng.uniform(0, 1.5)
        if t_s > party_until - 2:
            continue
        mv = mv_names[rng.choice(len(mv_names), p=mv_w)]
        if mv == "jumps":
            tj = t_s
            while tj < party_until - 1.2:
                scripts[i].append(("jump", float(tj)))
                jumps.append((i, float(tj)))
                tj += rng.uniform(1.6, 2.8)
        elif mv == "popper":
            scripts[i].append(("clip", float(t_s), float(party_until), "gesture_hello", 0))
        else:
            sf = int(rng.integers(0, 150)) if mv.startswith("dance") else 0
            scripts[i].append(("clip", float(t_s), float(party_until), mv, sf))
    meta["t_jump"] = t_jump
    meta["patch_at"] = patch_at
    meta["rec"] = (tgrid, A, rec_p)
    return {"scripts": scripts, "jumps": jumps, "poppers": poppers, "meta": meta}


def _smooth_path(xy, w=5):
    """Moving average that keeps both ends: shoving in the crowd sim jitters a little."""
    if len(xy) <= w:
        return xy.copy()
    pad = np.concatenate([np.repeat(xy[:1], w // 2, 0), xy, np.repeat(xy[-1:], w // 2, 0)])
    k = np.ones(w) / w
    out = np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(2)], 1)
    out[0], out[-1] = xy[0], xy[-1]
    return out


def _rot_all(S, arm, w_star, w_ring):
    th = np.where(arm >= 0, w_star, w_ring)
    c, s = np.cos(th), np.sin(th)
    return np.stack([c * S[:, 0] - s * S[:, 1], s * S[:, 0] + c * S[:, 1]], 1)


def _new_target(rng, home, hot, wander_m):
    """Somewhere to go: often a hotspot (crowds bunch up), otherwise near home; never
    more than wander_m from home so the walk back stays short."""
    m = len(home)
    to_hot = rng.uniform(size=m) < 0.55
    hk = hot[np.argmin(((home[:, None] - hot[None]) ** 2).sum(-1), axis=1)]
    tgt = np.where(to_hot[:, None], hk + rng.normal(0, 0.6, (m, 2)),
                   home + _disc(rng, m) * wander_m)
    off = tgt - home
    L = np.linalg.norm(off, axis=1, keepdims=True)
    return home + off * np.minimum(1, wander_m / (L + 1e-9))


def _disc(rng, m):
    r = np.sqrt(rng.uniform(size=m))
    a = rng.uniform(0, 2 * np.pi, m)
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)
