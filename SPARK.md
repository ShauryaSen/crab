# Brief for the Claude session on the Spark

You are running on the CUPI team's shared NVIDIA DGX Spark (host `spark`, user
`shaurya`, ARM64 + GB10 GPU). This repo (`shauryasen/crab`) makes the
"type anything, a crab swarm performs it" videos for the KingKong Jumper
open-source challenge. The cloud session that writes this repo cannot reach the
Spark; you do the GPU work and hand results back through git.

## House rules (shared machine, these come first)

1. Before launching anything, run `nvidia-smi`. If someone else's job is using the
   GPU heavily, use a smaller `--num_envs`, or wait and tell the user.
2. Never kill, renice or otherwise touch a process you did not start.
3. Run long jobs in `tmux` with `nice -n 10`. Log to a file.
4. Work only inside `~/src/crab` and `~/src/jumper`. Do not modify the shared
   `isaac-lab-base` image or `~/src/isaaclab-learning`. Jumper does not use Isaac Lab:
   it trains with mjlab + MuJoCo Warp, so it gets its own venv.

## Step 0: smoke test (about 10 minutes; report the result before doing more)

```bash
cd ~/src && git clone https://github.com/KingKongRobotics/jumper.git && cd jumper
git checkout b2e58669848a3ba4f209597cd6bebdfe9d0533a4   # the commit crab was built against
python3 -m venv .venv && source .venv/bin/activate
# GB10 is Blackwell on aarch64: use the CUDA 12.8+ aarch64 wheels.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
python -m pytest tests/ -q
python scripts/train.py --task jumper.dance --num_envs 512 --max_iterations 20
```

Read `docs/AGENT_SETUP.md` in the jumper repo first: it has verification gates.
If MuJoCo Warp does not run on this GPU/arch, try `--backend native --device cpu`
(the Spark has 20 ARM cores) and report both outcomes to the user. **Don't spend more
than an hour fighting the install. Report back instead.**

## Job 1: dances that follow any tempo

The stock dance policies track a recorded clip at exactly 1x speed. To make crabs lock
onto any song's beat, we need dance policies that track the clip played back at a
variable rate.

- In the dance task's motion command (`tasks/jumper/common/dance/`), sample a playback
  speed `s ~ U(0.75, 1.35)` per episode (optionally a slow drift within the episode).
  Advance the clip index by `s` frames per control step, scale `ref_joint_vel` by `s`,
  and take `ref_future` lookaheads at `round(h * s)` frames.
- Keep the observation layout unchanged. That way `crabshow/runtime.py::Crab.run_clip(...,
  speed=s)` runs the result without code changes, because it already builds exactly that.
- Train `dance_brazilian` first (3304 frames), then `dance` and `dance_waist`. Start from
  the shipped checkpoint (`--resume`), not from scratch.
- Acceptance: at s=0.8, 1.0 and 1.25, the tracking error is within 1.3x of the stock
  policy's error at 1.0x, and there are no falls in 50 episodes.

## Hand-back

- Export with `python scripts/export.py --task <task> --checkpoint <path>`.
- Commit **only** `actor.onnx`, `layout.json`, the reference `*.trajectory.json` and a
  short `RESULTS.md` (numbers and the command you ran). Put them in this repo at
  `trained/<task>-tempo/` on the branch `spark-results`. Never commit `.pt` files or logs.
- A 10-second MP4 of `play.py --headless --video` at 0.8x and 1.25x is welcome. Keep it
  under 10 MB.

## Job 2 (after Job 1): dances and jumps on uneven sand

`crabshow/terrain.py` makes dunes. Walking on them works up to an amplitude of about
0.08 m (16° slopes). The `jump` and `dance_*` policies were trained on flat ground, so
a few crabs land badly on a slope and end up stuck in a crouch.

- Fine-tune `jump` and `dance_brazilian` with terrain randomisation: a heightfield of
  sine swells with amplitude `U(0, 0.06)` m and wavelength `U(2.5, 5)` m, matching
  `terrain.py`. Jumper's `scenes/rough.py` shows how scenes plug in. Resume from the
  shipped checkpoints.
- Acceptance: on dunes with amplitude 0.05, 50/50 jumps and 5-second dances end standing
  at a base height within 0.02 m of nominal.
- Hand back the same way as Job 1, under `trained/<task>-dunes/`.
