import sys, time, json, pickle
sys.path.insert(0, ".")
import numpy as np
import crabshow
from crabshow.render import Renderer, STYLES
from crabshow.show import sample
from crabshow.video import bloom
from PIL import Image
logs = pickle.load(open("out/heart_hi_sim.pkl", "rb"))
n = len(logs)
style = STYLES["night"]
style.shadows = sys.argv[1] == "1"
style.floor_reflect = float(sys.argv[2])
t = time.time()
R = Renderer(n, np.ones((n, 3)), style, 540, 960)
print("build", round(time.time() - t, 1))
R.set_poses(sample(logs, 20.0)); R.set_lights(np.ones((n,3))*[1,.3,.5], np.ones(n))
for k in range(3):
    t = time.time(); img = R.frame([0, 0, 0], 10, 90, -60); t1 = time.time() - t
    t = time.time(); bloom(Image.fromarray(img)); t2 = time.time() - t
    print("frame", round(t1, 2), "bloom", round(t2, 2))
Image.fromarray(img).save("out/prof.png"); bloom(Image.fromarray(img)).save("out/prof_bloom.png")
