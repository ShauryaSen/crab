"""Collect a plan's segment branches (seg/<name>-NNNNN) and join them with the soundtrack.

usage: python -I scripts/assemble.py plans/rave_v3.json
"""
import json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crabshow import music
from crabshow.video import join_segments

plan = json.loads(Path(sys.argv[1]).read_text())
name = plan["name"] + "_" + plan["style"]
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
synth = music.synth_rave if plan.get("music", "rave") == "rave" else music.synth_track
audio, _ = synth([tuple(s) for s in plan["sections"]], bpm=plan["bpm"])
wav = out / f"{plan['name']}_{plan.get('music', 'rave')}.wav"
music.write_wav(str(wav), audio)
print("wrote", join_segments(segs, wav, out / f"{name}.mp4"))
