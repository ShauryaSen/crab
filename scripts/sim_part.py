"""Simulate crabs [a, b) of a plan and pack them for git.

usage: python -I scripts/sim_part.py plans/onecrab12k.json 0 500
"""
import json, sys, pickle, time
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import crabshow  # noqa: F401
from crabshow.compose import compose
from crabshow.show import simulate_crab


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    a, b = int(sys.argv[2]), int(sys.argv[3])
    score, _ = compose(plan)
    b = min(b, score.n)
    t = time.time()
    with mp.get_context("spawn").Pool() as pool:
        logs = pool.map(simulate_crab, [(i, score) for i in range(a, b)], chunksize=1)
    print(f"simulated crabs {a}-{b} in {time.time() - t:.0f}s")
    name = f"{plan.get('sim_name', plan['name'])}_n{score.n}_sim"
    out = Path("out/simpack")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{name}.part{a:06d}.npz"
    np.savez_compressed(f, first=a, count=b - a, lens=np.array([len(x[0]) for x in logs]),
                        t=np.concatenate([x[0] for x in logs]).astype(np.float32),
                        base=np.concatenate([x[1][:, :7] for x in logs]).astype(np.float32),
                        joints=np.concatenate([x[1][:, 7:] for x in logs]).astype(np.float16))
    print("wrote", f)


if __name__ == "__main__":
    main()
