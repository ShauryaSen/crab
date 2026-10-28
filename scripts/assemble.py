"""Collect a plan's segment branches (seg/<name>-NNNNN) and join them with the soundtrack.

usage: python -I scripts/assemble.py plans/rave_v3.json [WxH] [--music halloween]
"""
import json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crabshow import music
from crabshow.video import join_segments

plan = json.loads(Path(sys.argv[1]).read_text())
size = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
name = plan["name"] + "_" + plan["style"] + (f"_{size}" if size else "")
out = Path("out")
segdir = out / "segments_in" / name
segdir.mkdir(parents=True, exist_ok=True)
refs = subprocess.run(["git", "ls-remote", "origin", f"refs/heads/seg/{name}-*"],
                      capture_output=True, text=True, check=True).stdout.split()
branches = sorted(r.removeprefix("refs/heads/") for r in refs if r.startswith("refs/"))
print("branches:", branches)
segs = []
for br in branches:
    subprocess.run(["git", "fetch", "-q", "origin", f"{br}:refs/remotes/origin/{br}"], check=True)
    files = subprocess.run(["git", "ls-tree", "--name-only", f"origin/{br}", "segments/"],
                           capture_output=True, text=True, check=True).stdout.split()
    for f in files:
        dst = segdir / Path(f).name
        dst.write_bytes(subprocess.run(["git", "show", f"origin/{br}:{f}"], capture_output=True,
                                       check=True).stdout)
        segs.append(dst)
segs.sort()
# Check coverage: ranges must tile [0, N) with no gaps.
ranges = [tuple(int(x) for x in s.stem.split("_")[-2:]) for s in segs]
print("ranges:", ranges)
for (a, b), (c, d) in zip(ranges, ranges[1:]):
    assert b == c, f"gap or overlap between {b} and {c}"
# --music overrides the plan's soundtrack, so one render can be tried with several.
if "--music" in sys.argv:
    plan["music"] = sys.argv[sys.argv.index("--music") + 1]
audio, label = music.soundtrack(plan)
wav = out / f"{plan['name']}_{label}.wav"
music.write_wav(str(wav), audio)
suffix = f"_{label}" if "--music" in sys.argv else ""
print("wrote", join_segments(segs, wav, out / f"{name}{suffix}.mp4"))
