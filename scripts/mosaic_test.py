"""Photomosaic proof: crabs coloured as the pixels of a render of one Jumper."""
import sys, json, copy, time
sys.path.insert(0, ".")
import crabshow, numpy as np, mujoco
from PIL import Image
from crabshow.render import Renderer, STYLES
from crabshow.crowd import CrowdRenderer

GW = int(sys.argv[1]) if len(sys.argv) > 1 else 48          # mosaic columns
step = sys.argv[2] if len(sys.argv) > 2 else "both"
dflt = json.load(open("data/jumper/policies/posture/layout.json"))["default_joint_pos"]

def pose(model, nq1, x=0, y=0, yaw=-np.pi / 2):
    q = np.zeros(nq1); q[:3] = [x, y, 0.106]; q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    for j in range(model.njnt):
        nm = model.joint(j).name.replace("c0_", "")
        if nm in dflt:
            q[model.joint(j).qposadr[0]] = dflt[nm]
    return q

if step in ("src", "both"):
    # 1. A hero render of ONE crab, seen from above at a slight angle, on black.
    st = copy.deepcopy(STYLES["moonbeach"]); st.glow = 0.25; st.floor_reflect = 0; st.shadows = False
    st.floor_rgb1 = st.floor_rgb2 = (0, 0, 0); st.ambient = 0.35; st.key_light = 1.2
    R = Renderer(1, np.array([[0.9, 0.12, 0.08]]), st, 480, 400)
    R.set_poses([pose(R.model, R.nq1)])
    R.set_lights(np.array([[0.95, 0.15, 0.08]]), np.array([1.0]))
    img = R.frame([0, 0, 0.04], 0.75, 90, -62)
    Image.fromarray(img).save("out/mosaic_src.png")
    print("source rendered")

if step in ("mosaic", "both"):
    src = Image.open("out/mosaic_src.png").convert("RGB")
    GH = int(round(GW * src.height / src.width))
    small = np.asarray(src.resize((GW, GH), Image.LANCZOS)).astype(float) / 255
    lum = small.max(-1)
    sp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    pts, cols = [], []
    for r in range(GH):
        for c in range(GW):
            if lum[r, c] > 0.06:          # background pixels get no crab
                pts.append(((c - GW / 2) * sp, (GH / 2 - r) * sp)); cols.append(small[r, c])
    N = len(pts)
    print("crabs:", N)
    st = copy.deepcopy(STYLES["moonbeach"]); st.glow = 1.0; st.floor_rgb1 = st.floor_rgb2 = (0.03, 0.03, 0.04)
    st.floor_reflect = 0; st.shadows = False
    C = CrowdRenderer(st, 1080, 1350, faces=3000, max_geom=N * 45 + 5000)
    qs = [pose(C.model, C.nq1, x, y) for x, y in pts]
    cols = np.clip(np.array(cols) * 1.15, 0, 1)
    for name, look, dist, el in [("far", [0, 0, 0], GW * sp * 1.45, -89.5),
                                 ("mid", [pts[N // 2][0], pts[N // 2][1], 0], 9, -70),
                                 ("near", [pts[N // 2][0], pts[N // 2][1], 0.05], 1.4, -25)]:
        t = time.time()
        from crabshow.video import bloom
        im = Image.fromarray(C.frame(qs, cols, np.ones(N), look, dist, 90, el))
        if name == "far":
            im = bloom(im, strength=1.6, radius=10, threshold=40)
        im.save(f"out/mosaic_{name}.png")
        print(name, round(time.time() - t, 1), "s")
