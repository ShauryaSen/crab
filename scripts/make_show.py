"""plan JSON -> simulated swarm -> rendered, scored video.

usage: python -I scripts/make_show.py plans/heart_hi.json [--draft] [--n N]
"""
import argparse, json, pickle, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import crabshow  # noqa: F401  (sets the GL backend)
from crabshow import music
from crabshow.compose import compose
from crabshow.show import simulate, collisions
from crabshow.video import render_video

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--draft", action="store_true", help="half resolution")
    ap.add_argument("--style")
    ap.add_argument("--out", default="out")
    ap.add_argument("--resim", action="store_true")
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    if args.style:
        plan["style"] = args.style
    name = plan["name"] + "_" + plan["style"] + ("_draft" if args.draft else "")
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    t = time.time()
    score, meta = compose(plan)
    audio, starts = music.synth_track([tuple(s) for s in plan["sections"]], bpm=plan["bpm"])
    wav = out / f"{plan['name']}.wav"
    music.write_wav(str(wav), audio)

    cache = out / f"{plan['name']}_sim.pkl"
    if cache.exists() and not args.resim:
        logs = pickle.loads(cache.read_bytes())
    else:
        logs = simulate(score)
        cache.write_bytes(pickle.dumps(logs))
    print(f"simulated {score.n} crabs x {score.duration:.1f}s in {time.time()-t:.0f}s")
    d, when, pair = collisions(logs)
    print(f"closest approach {d:.2f} m at t={when:.1f}s between crabs {pair}")

    t = time.time()
    w, h = (540, 960) if args.draft else (1080, 1920)
    path = render_video(score, logs, plan, out / f"{name}.mp4", wav, width=w, height=h)
    print(f"rendered {path} in {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
