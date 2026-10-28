"""Apply a post look to a video section: python -I scripts/apply_look.py in.mp4 out.mp4 look start dur"""
import subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image
from crabshow.looks import LOOKS

src, dst, look, start, dur = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
d = Path(tempfile.mkdtemp(dir="out"))
subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(start), "-t", str(dur), "-i", src,
                str(d / "f%05d.png")], check=True)
frames = sorted(d.glob("f*.png"))
for k, f in enumerate(frames):
    LOOKS[look](Image.open(f).convert("RGB"), start + k / 30, k).save(f)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(d / "f%05d.png"),
                "-ss", str(start), "-t", str(dur), "-i", src, "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac", "-shortest", dst],
               check=True)
for f in frames:
    f.unlink()
d.rmdir()
print("wrote", dst)
