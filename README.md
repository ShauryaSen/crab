# crab: type anything, a crab swarm performs it

An entry for the [KingKong Crab Robot Open-source Challenge](https://beunlimited.me/en/events/crab-robot-challenge-2026).

You write a prompt like *"light up, make a heart, go wild on the drop, then spell HI"*.
A swarm of [Jumper](https://github.com/KingKongRobotics/jumper) crab robots walks into
each formation and dances on the beat, lit like a drone show.

- **Nothing is keyframed.** Every crab is a separate MuJoCo physics simulation driven by
  KingKong's trained RL policies: the `posture` policy for walking and the dance or
  gesture policies for the moves.
- **The crabs never collide.** Formation changes use a minimum-sum-of-squares assignment,
  so the straight-line paths never cross. Each crab then follows a time-parameterised
  path, and all of them arrive together.
- **Moves from the stock policies only.** Walking uses the `posture` controller, whose
  body-height command also gives a bounce at any rate. The drop uses the `jump`
  controller, and the dances and gestures play recorded clips.
- **Changes land on the music.** The choreography is written in bars of the track.
  `crabshow/music.py` synthesises the original soundtrack and can also analyse any song
  for its beats and drops.

## Layout

| | |
|---|---|
| `crabshow/runtime.py` | Standalone Jumper runtime. It reads the robot MJCF and each policy's `layout.json` contract, and re-implements the observation builder and actuator model in plain MuJoCo + onnxruntime. |
| `crabshow/formation.py` | Words and shapes to positions; collision-free assignment. |
| `crabshow/compose.py` | A choreography plan (JSON) to a timed score of moves and lights. |
| `crabshow/show.py` | Per-crab simulation of a score, in parallel. |
| `crabshow/render.py`, `video.py` | Composite render (shared low-poly meshes), bloom, typed prompt, captions, mux, segment join. |
| `crabshow/fx.py` | Show effects: an LED dance floor, lasers, light pillars, trails and confetti, all on the beat. |
| `crabshow/hud.py` | A "mission control" top-down view for "how it works" cuts. |
| `crabshow/prompt.py` | Prompt to plan for common phrasings ("make a heart, jump on the drop, spell HI"). |
| `crabshow/terrain.py` | Dunes heightfield, shared by the physics and the render. |
| `plans/` | Example plans. |
| `SPARK.md` | Training brief for the GPU (tempo-robust dances). |

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install mujoco onnxruntime numpy scipy librosa pillow matplotlib
scripts/fetch_jumper.sh                      # robot model + exported policies -> data/jumper
.venv/bin/python -I scripts/make_show.py plans/rave_v3.json --preview   # storyboard
.venv/bin/python -I scripts/make_show.py plans/rave_v3.json             # 1080x1350 video
.venv/bin/python -I scripts/plan_from_prompt.py "light up, make a heart, then spell YO" yo
```

To render one video on many machines, give each one
`scripts/render_segment_job.sh <plan> <first frame> <end frame> [WxH]`. Each job pushes
its segment to a `seg/*` branch, and `scripts/assemble.py <plan> [WxH]` joins them.

Rendering uses software OpenGL (OSMesa) by default, so it runs on a headless CPU box.

## Credits

The robot model and policies are © KingKong Robotics, Apache-2.0 (fetched into `data/`,
not redistributed here). Music is synthesised by `crabshow/music.py`.
