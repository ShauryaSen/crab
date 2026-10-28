"""Pack a simulation cache into git-friendly parts, or unpack it.

Base pose stays float32, joint angles go to float16 (about 0.003 rad, invisible on screen),
and crabs are split across parts so each file stays well under GitHub's 100 MB limit.

usage: python -I scripts/simpack.py pack   out/onecrab_n1988_sim.pkl out/simpack 4
       python -I scripts/simpack.py unpack out/simpack out/onecrab_n1988_sim.pkl
"""
import pickle, sys
from pathlib import Path

import numpy as np

mode = sys.argv[1]
if mode == "pack":
    src, dst, parts = Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    dst.mkdir(parents=True, exist_ok=True)
    logs = pickle.loads(src.read_bytes())
    n = len(logs)
    edges = np.linspace(0, n, parts + 1).astype(int)
    for k, (a, b) in enumerate(zip(edges, edges[1:])):
        ts = [logs[i][0].astype(np.float32) for i in range(a, b)]
        q = [logs[i][1] for i in range(a, b)]
        np.savez_compressed(dst / f"{src.stem}.part{k}.npz",
                            first=a, count=b - a,
                            lens=np.array([len(x) for x in ts]),
                            t=np.concatenate(ts),
                            base=np.concatenate([x[:, :7] for x in q]).astype(np.float32),
                            joints=np.concatenate([x[:, 7:] for x in q]).astype(np.float16))
        print("wrote", dst / f"{src.stem}.part{k}.npz")
elif mode == "unpack":
    src, dst = Path(sys.argv[2]), Path(sys.argv[3])
    files = sorted(src.glob(f"{dst.stem}.part*.npz"), key=lambda p: int(p.stem.split("part")[-1]))
    logs = []
    for f in files:
        z = np.load(f)
        off = 0
        for L in z["lens"]:
            t = z["t"][off:off + L].astype(np.float64)
            q = np.concatenate([z["base"][off:off + L], z["joints"][off:off + L]], 1).astype(np.float64)
            logs.append((t, q))
            off += L
    dst.write_bytes(pickle.dumps(logs))
    print("unpacked", len(logs), "crabs ->", dst)
