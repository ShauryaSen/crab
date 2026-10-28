"""Dense, memory-mapped crab trajectories for big swarms.

Per-crab logs (lists of (times, qpos)) are fine for a hundred crabs. For ten thousand they
are gigabytes, and every render process would get its own copy. Here they become one
float32 array Q[crab, frame, qpos] sampled at the video frame rate, saved as .npy and
memory-mapped, so all processes on a machine share one copy through the page cache.

`DenseLogs` behaves like the list-of-logs everywhere it is used: len(), iteration,
indexing (times, qpos) and `show.sample(logs, t)`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class DenseLogs:
    def __init__(self, path, fps=30.0):
        self.path = str(path)
        self.fps = fps
        self.Q = np.load(self.path, mmap_mode="r")
        self.times = np.arange(self.Q.shape[1]) / fps

    def __reduce__(self):
        return (DenseLogs, (self.path, self.fps))

    def __len__(self):
        return self.Q.shape[0]

    def __getitem__(self, i):
        return self.times, self.Q[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def frame(self, t):
        k = int(np.clip(np.floor(t * self.fps + 1e-6), 0, self.Q.shape[1] - 1))
        return self.Q[:, k]


def densify(logs, out_path, duration, fps=30.0):
    """Write list-of-logs (or an iterator of (times, qpos)) to a dense .npy file."""
    logs = list(logs) if not isinstance(logs, list) else logs
    F = int(np.floor(duration * fps)) + 1
    nq = logs[0][1].shape[1]
    Q = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                  shape=(len(logs), F, nq))
    ft = np.arange(F) / fps
    for i, (t, q) in enumerate(logs):
        k = np.clip(np.searchsorted(t, ft, side="right") - 1, 0, len(t) - 1)
        Q[i] = q[k]
    Q.flush()
    del Q
    return DenseLogs(out_path, fps)


def densify_packs(files, out_path, duration, fps=30.0):
    """Sim-pack parts (sorted by first crab) straight to a dense .npy, one crab at a time."""
    files = sorted(files, key=lambda p: int(np.load(p)["first"]))
    n = sum(int(np.load(f)["count"]) for f in files)
    F = int(np.floor(duration * fps)) + 1
    Q = None
    ft = np.arange(F) / fps
    i = 0
    for f in files:
        z = np.load(f)
        off = 0
        base, joints, tt = z["base"], z["joints"], z["t"]
        for L in z["lens"]:
            t = tt[off:off + L].astype(np.float64)
            if Q is None:
                Q = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                              shape=(n, F, base.shape[1] + joints.shape[1]))
            k = np.clip(np.searchsorted(t, ft, side="right") - 1, 0, L - 1)
            Q[i, :, :base.shape[1]] = base[off:off + L][k]
            Q[i, :, base.shape[1]:] = joints[off:off + L][k]
            off += L
            i += 1
    Q.flush()
    del Q
    return DenseLogs(out_path, fps)
