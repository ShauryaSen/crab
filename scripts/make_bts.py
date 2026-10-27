"""'How it works' cut: the rendered show beside the mission-control HUD.

usage: python -I scripts/make_bts.py plans/rave_v3.json out/rave_v3_moonbeach.mp4
"""
import json, pickle, subprocess, sys
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import crabshow  # noqa: F401
from crabshow.compose import compose


def _chunk(args):
    plan, k0, k1, fps, outdir, w, h = args
    from crabshow.hud import HUD
    score, _ = compose(plan)
    sim = plan.get("sim_name", plan["name"])
    logs = pickle.loads(Path(f"out/{sim}_n{score.n}_sim.pkl").read_bytes())
    hud = HUD(plan, score, logs, w, h)
    for k in range(k0, k1):
        hud.frame(k / fps).save(outdir / f"h{k:05d}.png", compress_level=1)


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    main_mp4 = Path(sys.argv[2])
    fps, w, h = 30, 1080, 1350
    score, _ = compose(plan)
    n = int(score.duration * fps)
    d = Path("out") / f"{plan['name']}_hud_frames"
    d.mkdir(parents=True, exist_ok=True)
    import numpy as np
    cuts = np.linspace(0, n, 17).astype(int)
    with mp.get_context("spawn").Pool(4) as pool:
        pool.map(_chunk, [(plan, a, b, fps, d, w, h) for a, b in zip(cuts, cuts[1:])])
    out = Path("out") / f"{plan['name']}_howitworks.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(main_mp4), "-framerate", str(fps),
                    "-i", str(d / "h%05d.png"), "-filter_complex",
                    "[0:v][1:v]hstack=inputs=2,scale=1920:-2[v]", "-map", "[v]", "-map", "0:a",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-c:a", "copy",
                    "-shortest", "-movflags", "+faststart", str(out)], check=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
