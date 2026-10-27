"""A standalone runtime for KingKong's Jumper policies: plain MuJoCo + onnxruntime.

It reads the robot MJCF and each policy's `layout.json` contract as data, and
re-implements the observation builder and action decode documented there (the
reference implementation is jumper's `deploy/fsm/src/{obs,action,trajectory}.rs`).
None of the jumper repository's Python is imported.

One `Crab` is one robot on its own floor. A swarm is many independent `Crab`s whose
paths are planned not to meet; they are composited into one scene only for rendering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

DATA = Path(__file__).resolve().parent.parent / "data" / "jumper"
ROBOT_XML = DATA / "robot" / "jumper.xml"

LEGS = ("LF", "RF", "LM", "RM", "LR", "RR")
FEET = ("LF_palm_pad_b_link", "RF_palm_pad_b_link", "LM_foot_tip_link", "RM_foot_tip_link",
        "LR_foot_tip_link", "RR_foot_tip_link")
CHASSIS = ("base_link", "upper_shell_link", "camera_link", "tof_sensor_link",
           "display_module_link")
LIMBS = ("LF_shoulder_link", "LF_upper_arm_link", "LF_forearm_link", "RF_shoulder_link",
         "RF_upper_arm_link", "RF_forearm_link", "LM_hip_link", "LM_thigh_link", "RM_hip_link",
         "RM_thigh_link", "LR_hip_link", "LR_thigh_link", "RR_hip_link", "RR_thigh_link")
TIPS = tuple(f"{s}_{n}" for s in ("LF", "RF") for n in (
    "palm_link", "finger_link", "finger_tip_link", "palm_pad_f_link", "finger_grip_insert_link",
    "palm_grip_insert_link"))

EFFORT = 1.7464
# Servo torque-speed curve (jumper assets/jumper/motor/motor_config.yaml).
W_KNEE, W_DECAY, W_MAX = 30.7352, 25.1851, 63.9838
STAND_Z = 0.10647


def _torque_limit(w):
    w = np.abs(w)
    lim = EFFORT * np.exp(-np.maximum(w - W_KNEE, 0.0) / W_DECAY)
    return np.where(w >= W_MAX, 0.0, lim)


def build_spec(floor: bool = True, terrain: dict | None = None) -> mujoco.MjSpec:
    """The robot as training saw it: motors, collision masks, contact params, solver."""
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    feet = {f"{n}_meshcol" for n in FEET}
    others = {f"{n}_meshcol" for n in CHASSIS + LIMBS + TIPS}
    all_legs = sum(1 << (i + 1) for i in range(6))
    for g in spec.geoms:
        if g.name in feet or g.name in others:
            leg = next((i for i, l in enumerate(LEGS) if g.name.startswith(l + "_")), None)
            if leg is None:
                g.contype, g.conaffinity = 1, 0
            else:
                bit = 1 << (leg + 1)
                g.contype, g.conaffinity = 1 | bit, all_legs & ~bit
            foot = g.name in feet
            g.condim = 3 if foot else 1
            g.priority = 1 if foot else 0
            if foot:
                g.friction = [1.2, 0.01, 0.01]
            g.solref = [0.02, 1.0]
            g.solimp = [0.8, 0.9, 0.002, 0.5, 2.0]
        elif g.contype or g.conaffinity:
            g.contype, g.conaffinity = 0, 0
    for j in spec.joints:
        if j.name.endswith("_joint"):
            a = spec.add_actuator(name=j.name.replace("_joint", "_motor"), target=j.name,
                                  trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.gear[0] = 1.0
            a.ctrllimited = True
            a.ctrlrange = [-EFFORT, EFFORT]
            a.forcelimited = True
            a.forcerange = [-EFFORT, EFFORT]
    if terrain:
        from .terrain import add_hfield

        add_hfield(spec, **terrain)
    elif floor:
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[0, 0, 0.01], contype=1, conaffinity=1)
        spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    o = spec.option
    o.iterations, o.ls_iterations, o.ccd_iterations = 10, 20, 50
    o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    o.solver = mujoco.mjtSolver.mjSOL_NEWTON
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return spec


def _projgrav(q):
    w, x, y, z = q
    return np.array([-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))])


class Policy:
    """One exported policy and its contract."""

    def __init__(self, name: str):
        self.name = name
        d = DATA / "policies" / name
        self.layout = json.loads((d / "layout.json").read_text())
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(d / "actor.onnx"), opts,
                                         providers=["CPUExecutionProvider"])
        L = self.layout
        self.obs_joints = L["obs_joint_order"]
        self.act_joints = L["action_joint_order"]
        self.default = L["default_joint_pos"]
        self.scale = L["action_scale"]
        c = L["control"]
        self.kp, self.kd = c["kp"], c["kd"]
        self.decimation, self.sim_dt = c["decimation"], c["sim_dt"]
        self.control_dt = c["control_dt"]
        self.terms = L["observation"]["terms"]
        self.ref = None
        ref = L.get("reference")
        if ref:
            r = json.loads((d / ref["file"]).read_text())
            order = r.get("joint_order", L["wire_joint_order"])
            self.ref = {k: np.asarray(v, dtype=np.float64) for k, v in r.items()
                        if k in ("q", "qd", "root_quat", "q_cmd")}
            self.ref_order = order
            self.ref_meta = ref

    @property
    def clip_frames(self) -> int:
        return len(self.ref["q"]) if self.ref else 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.sess.run(["actions"], {"obs": obs[None].astype(np.float32)})[0][0]


_POLICIES: dict[str, Policy] = {}


def policy(name: str) -> Policy:
    if name not in _POLICIES:
        _POLICIES[name] = Policy(name)
    return _POLICIES[name]


@dataclass
class Segment:
    """What a crab does for a while.

    kind "walk": posture policy, cmd = callable(crab) -> (vx, vy, wz) in the body frame.
    kind "clip": a dance/gesture policy played from `start_frame`.
    """
    kind: str
    duration: float
    policy: str = "posture"
    cmd: object = None
    start_frame: int = 0
    posture: tuple = (0.0, 0.0, 0.0, 0.0)


class Crab:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, seed=0, terrain: dict | None = None):
        self.spec = build_spec(terrain=terrain)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.jnames = [m.joint(i).name for i in range(m.njnt) if m.joint(i).name.endswith("_joint")]
        self.qadr = {n: m.joint(n).qposadr[0] for n in self.jnames}
        self.vadr = {n: m.joint(n).dofadr[0] for n in self.jnames}
        self.aid = {n: m.actuator(n.replace("_joint", "_motor")).id for n in self.jnames}
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.log_t: list[float] = []
        self.log_qpos: list[np.ndarray] = []
        p = policy("posture")
        d = self.data
        z0 = 0.0
        if terrain:
            from .terrain import height

            z0 = float(height(x, y, terrain.get("amp", 0.08), terrain.get("seed", 0)))
        d.qpos[:3] = [x, y, STAND_Z + z0 + 0.01]
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        for n in self.jnames:
            d.qpos[self.qadr[n]] = p.default[n]
        mujoco.mj_forward(m, d)
        self.target = np.array([p.default[n] for n in self.jnames])
        self.last_tau = np.zeros(len(self.jnames))

    # -- state -------------------------------------------------------------
    @property
    def pos(self):
        return self.data.qpos[:2].copy()

    @property
    def yaw(self):
        w, x, y, z = self.data.qpos[3:7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def q(self, names):
        return np.array([self.data.qpos[self.qadr[n]] for n in names])

    def qd(self, names):
        return np.array([self.data.qvel[self.vadr[n]] for n in names])

    def tau(self, names):
        idx = [self.jnames.index(n) for n in names]
        return self.last_tau[idx]

    def fallen(self):
        return _projgrav(self.data.qpos[3:7])[2] > -0.5

    # -- physics -----------------------------------------------------------
    def _substeps(self, pol: Policy, n: int, log_every: float):
        m, d = self.model, self.data
        m.opt.timestep = pol.sim_dt
        q_adr = np.array([self.qadr[j] for j in self.jnames])
        v_adr = np.array([self.vadr[j] for j in self.jnames])
        a_ids = np.array([self.aid[j] for j in self.jnames])
        for _ in range(n):
            qd = d.qvel[v_adr]
            tau = pol.kp * (self.target - d.qpos[q_adr]) - pol.kd * qd
            lim = _torque_limit(qd)
            tau = np.clip(tau, -lim, lim)
            d.ctrl[a_ids] = tau
            mujoco.mj_step(m, d)
            self.last_tau = tau
            self.t += pol.sim_dt
            if not self.log_t or self.t - self.log_t[-1] >= log_every - 1e-9:
                self.log_t.append(self.t)
                self.log_qpos.append(d.qpos.copy())

    def _set_target(self, pol: Policy, a: np.ndarray, base=None):
        tgt = dict(zip(self.jnames, self.target))
        for n in self.jnames:
            if n not in pol.act_joints:
                tgt[n] = pol.layout["unactuated_joints"].get(n, pol.default[n])
        for i, n in enumerate(pol.act_joints):
            b = pol.default[n] if base is None else base[n]
            tgt[n] = b + pol.scale * float(a[i])
        self.target = np.array([tgt[n] for n in self.jnames])

    def _imu(self):
        return self.data.qvel[3:6].copy(), _projgrav(self.data.qpos[3:7])

    # -- walking (posture policy) -------------------------------------------
    def run_walk(self, duration, cmd_fn, posture=(0, 0, 0, 0), log_every=1 / 30):
        pol = policy("posture")
        J = pol.obs_joints
        hist_terms = [t for t in pol.terms if t.get("history_length", 1) > 1]
        stride = hist_terms[0]["history_stride"]
        L = hist_terms[0]["history_length"]
        span = stride * (L - 1) + 1
        gp = next(t for t in pol.terms if t["name"] == "gait_phase")["params"]
        ring: list[dict] = []
        last_a = np.zeros(len(pol.act_joints))
        phase = 0.0
        steps = int(round(duration / pol.control_dt))
        for _ in range(steps):
            cmd = np.asarray(cmd_fn(self), dtype=np.float64)
            gyro, grav = self._imu()
            frame = {
                "joint_pos": self.q(J) - np.array([pol.default[n] for n in J]),
                "joint_vel": self.qd(J),
                "actions": last_a.copy(),
                "joint_torque": self.tau(J),
            }
            if not ring:
                ring = [frame] * span
            else:
                ring = ring[1:] + [frame]
            gate = float(np.linalg.norm(cmd) > gp["command_threshold"])
            parts = []
            for t in pol.terms:
                n = t["name"]
                if n == "base_ang_vel":
                    parts.append(gyro)
                elif n == "projected_gravity":
                    parts.append(grav)
                elif n in frame:
                    parts.extend(ring[span - 1 - stride * (L - 1 - j)][n] for j in range(L))
                elif n == "velocity_commands":
                    parts.append(cmd)
                elif n == "gait_phase":
                    parts.append(gate * np.array([np.sin(2 * np.pi * phase),
                                                  np.cos(2 * np.pi * phase)]))
                elif n == "posture_command":
                    pc = posture(self) if callable(posture) else posture
                    parts.append(np.asarray(pc, dtype=np.float64))
                else:
                    raise KeyError(n)
            obs = np.concatenate(parts)
            assert obs.shape[0] == pol.layout["observation"]["dim"], obs.shape
            a = pol.act(obs)
            last_a = a
            f = (np.hypot(cmd[0], cmd[1]) + abs(cmd[2]) * gp["turn_radius"]) / (2 * gp["stride"])
            f = np.clip(f, gp["freq_min"], gp["freq_max"])
            phase = (phase + f * pol.control_dt) % 1.0
            self._set_target(pol, a)
            self._substeps(pol, pol.decimation, log_every)

    # -- reference-clip policies (dances, gestures) ---------------------------
    def run_clip(self, name, duration, start_frame=0, log_every=1 / 30, speed=1.0):
        pol = policy(name)
        ref = pol.ref
        T = len(ref["q"])
        idx = [pol.ref_order.index(n) for n in pol.obs_joints]
        Q, QD = ref["q"][:, idx], ref["qd"][:, idx]
        dflt = np.array([pol.default[n] for n in pol.obs_joints])
        horizons = [int(round(s * pol.ref_meta["rec_hz"])) for s in pol.ref_meta["lookahead_s"]]
        last_a = np.zeros(len(pol.act_joints))
        steps = int(round(duration / pol.control_dt))
        k_f = float(start_frame)
        for _ in range(steps):
            k = min(int(k_f), T - 1)
            gyro, grav = self._imu()
            parts = []
            for t in pol.terms:
                n = t["name"]
                if n == "clip_phase":
                    parts.append([np.sin(2 * np.pi * k / T), np.cos(2 * np.pi * k / T)])
                elif n == "ref_joint_pos":
                    parts.append(Q[k] - dflt)
                elif n == "ref_joint_vel":
                    parts.append(QD[k] * speed)
                elif n == "ref_future":
                    for h in horizons:
                        parts.append(Q[min(k + int(round(h * speed)), T - 1)] - dflt)
                elif n == "ref_tilt_error":
                    parts.append(_projgrav(ref["root_quat"][k]) - grav)
                elif n == "base_ang_vel":
                    parts.append(gyro)
                elif n == "projected_gravity":
                    parts.append(grav)
                elif n == "joint_pos":
                    parts.append(self.q(pol.obs_joints) - dflt)
                elif n == "joint_vel":
                    parts.append(self.qd(pol.obs_joints))
                elif n == "actions":
                    parts.append(last_a)
                elif n == "joint_torque":
                    parts.append(self.tau(pol.obs_joints))
                else:
                    raise KeyError(n)
            obs = np.concatenate([np.asarray(p, dtype=np.float64) for p in parts])
            a = pol.act(obs)
            last_a = a
            self._set_target(pol, a)
            self._substeps(pol, pol.decimation, log_every)
            k_f += speed
        return int(k_f)

    # -- the jump (residual on the recorded command) ---------------------------
    def run_jump(self, log_every=1 / 30):
        """One recorded high jump, from stand to landing (about 1.2 s)."""
        pol = policy("jump")
        ref, meta = pol.ref, pol.ref_meta
        order = pol.ref_order
        oidx = [order.index(n) for n in pol.obs_joints]
        aidx = [order.index(n) for n in pol.act_joints]
        Q, QC = ref["q"], ref["q_cmd"]
        dflt = np.array([pol.default[n] for n in pol.obs_joints])

        def lerp(tab, idx):
            idx = float(np.clip(idx, 0, len(tab) - 1))
            lo = int(np.floor(idx))
            hi = min(lo + 1, len(tab) - 1)
            w = idx - lo
            return tab[lo] * (1 - w) + tab[hi] * w

        def q_at(tsg):
            idx = np.clip((meta["t_go"] + tsg) * meta["rec_hz"], meta["go_frame"], len(Q) - 1)
            return lerp(Q, idx)

        last_a = np.zeros(len(pol.act_joints))
        steps = int(round((meta["duration"] - meta["t_go"]) / pol.control_dt))
        for k in range(steps):
            tsg = k * pol.control_dt
            gyro, grav = self._imu()
            parts = []
            for t in pol.terms:
                n = t["name"]
                if n == "base_ang_vel":
                    parts.append(gyro)
                elif n == "projected_gravity":
                    parts.append(grav)
                elif n == "joint_pos":
                    parts.append(self.q(pol.obs_joints) - dflt)
                elif n == "joint_vel":
                    parts.append(self.qd(pol.obs_joints))
                elif n == "actions":
                    parts.append(last_a)
                elif n == "jump_phase":
                    parts.append([np.clip(tsg / meta["span"], 0, 1)])
                elif n == "ref_future":
                    for off in meta["lookahead_s"]:
                        parts.append(q_at(tsg + off)[oidx] - dflt)
                else:
                    raise KeyError(n)
            obs = np.concatenate([np.asarray(p, dtype=np.float64) for p in parts])
            a = pol.act(obs)
            last_a = a
            t_abs = tsg + meta["t_go"] + meta["baseline_lead_s"]
            base_row = lerp(QC, t_abs * meta["control_hz"])
            base = {n: base_row[order.index(n)] for n in pol.act_joints}
            self._set_target(pol, a, base=base)
            self._substeps(pol, pol.decimation, log_every)
