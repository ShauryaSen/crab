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
    ap.add_argument("--preview", action="store_true", help="contact sheet only")
    ap.add_argument("--size", help="WxH, e.g. 1080x1920 for a 9:16 Shorts cut")
    ap.add_argument("--frames", help="a:b -- render only frames [a, b) as a silent segment")
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    if args.style:
        plan["style"] = args.style
    name = plan["name"] + "_" + plan["style"] + ("_draft" if args.draft else "")
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    t = time.time()
    score, meta = compose(plan)
    if "hero" in meta:
        plan["_hero"] = int(meta["hero"])
    for w in meta.get("warnings", []):
        print("WARNING (plan):", w)
    synth = music.synth_rave if plan.get("music", "rave") == "rave" else music.synth_track
    audio, starts = synth([tuple(s) for s in plan["sections"]], bpm=plan["bpm"])
    wav = out / f"{plan['name']}_{plan.get('music', 'rave')}.wav"
    music.write_wav(str(wav), audio)

    cache = out / f"{plan.get('sim_name', plan['name'])}_n{score.n}_sim.pkl"
    if cache.exists() and not args.resim:
        logs = pickle.loads(cache.read_bytes())
    else:
        logs = simulate(score)
        cache.write_bytes(pickle.dumps(logs))
    print(f"simulated {score.n} crabs x {score.duration:.1f}s in {time.time()-t:.0f}s")
    d, when, pair = collisions(logs)
    print(f"closest approach {d:.2f} m at t={when:.1f}s between crabs {pair}")

    if args.preview:
        import numpy as np
        from crabshow.video import preview_sheet
        times = list(np.linspace(0.5, score.duration - 0.5, 12))
        print("preview", preview_sheet(score, logs, plan, out / f"{name}_preview.png", times))
        return

    t = time.time()
    w, h = (540, 676) if args.draft else (1080, 1350)
    if args.size:
        w, h = (int(x) for x in args.size.split("x"))
        name += f"_{w}x{h}"
    if args.frames:
        a, b = (int(x) for x in args.frames.split(":"))
        seg = out / "segments" / f"{name}_{a:05d}_{b:05d}.mp4"
        seg.parent.mkdir(exist_ok=True)
        path = render_video(score, logs, plan, seg, None, width=w, height=h, frames=(a, b))
    else:
        path = render_video(score, logs, plan, out / f"{name}.mp4", wav, width=w, height=h)
    print(f"rendered {path} in {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
