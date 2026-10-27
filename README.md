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
| `crabshow/render.py`, `video.py` | Composite render, bloom, typed prompt, captions, mux. |
| `plans/` | Example plans. |
| `SPARK.md` | Training brief for the GPU (tempo-robust dances). |

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install mujoco onnxruntime numpy scipy librosa pillow matplotlib
scripts/fetch_jumper.sh                      # robot model + exported policies -> data/jumper
.venv/bin/python -I scripts/make_show.py plans/heart_hi.json --draft
```

Rendering uses software OpenGL (OSMesa) by default, so it runs on a headless CPU box.

## Credits

The robot model and policies are © KingKong Robotics, Apache-2.0 (fetched into `data/`,
not redistributed here). Music is synthesised by `crabshow/music.py`.
