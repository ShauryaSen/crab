"""Composite many crabs' recorded motion into one scene and render it with MuJoCo.

Physics happens per crab in `runtime.Crab`; here each crab is a kinematic copy whose
qpos is replayed from its log, so a render costs nothing in contact solving.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


SHELL_RGBA = (0.85, 0.08, 0.05)   # the red parts of the stock model


@dataclass
class Style:
    name: str
    sky_top: tuple = (0.02, 0.03, 0.08)
    sky_bottom: tuple = (0.0, 0.0, 0.0)
    floor_rgb1: tuple = (0.76, 0.66, 0.48)
    floor_rgb2: tuple = (0.70, 0.60, 0.43)
    floor_reflect: float = 0.0
    floor_texture: str = "checker"     # "checker" | "flat"
    floor_scale: float = 1.0
    ambient: float = 0.3
    headlight: float = 0.4
    key_light: float = 0.6
    glow: float = 0.0                  # emission on the shells
    shadows: bool = True
    haze: tuple = (0.15, 0.25, 0.35, 1)
    extra_lights: list = field(default_factory=list)
    ocean: tuple | None = None         # rgb of a sea plane beyond y = +ocean_y
    mesh_faces: int = 60000            # visual triangles per crab after decimation
    ocean_y: float = 14.0


STYLES = {
    "moonbeach": Style(
        "moonbeach", sky_top=(0.01, 0.02, 0.07), sky_bottom=(0.06, 0.08, 0.17),
        floor_rgb1=(0.42, 0.40, 0.50), floor_rgb2=(0.38, 0.36, 0.46), floor_texture="flat",
        floor_reflect=0.12, ambient=0.05, headlight=0.03, key_light=0.35, glow=0.85,
        haze=(0.05, 0.07, 0.15, 1), ocean=(0.04, 0.13, 0.30)),
    "night": Style(
        "night", sky_top=(0.01, 0.02, 0.06), sky_bottom=(0.0, 0.0, 0.01),
        floor_rgb1=(0.20, 0.18, 0.17), floor_rgb2=(0.12, 0.11, 0.11), floor_texture="checker", floor_scale=0.5,
        floor_reflect=0.25, ambient=0.05, headlight=0.05, key_light=0.25, glow=0.85,
        haze=(0.02, 0.03, 0.08, 1)),
    "stadium": Style(
        "stadium", sky_top=(0.35, 0.55, 0.85), sky_bottom=(0.85, 0.9, 1.0),
        floor_rgb1=(0.20, 0.55, 0.22), floor_rgb2=(0.17, 0.50, 0.19), floor_texture="stripes",
        floor_scale=2.0, ambient=0.35, headlight=0.3, key_light=0.7, haze=(0.7, 0.8, 0.95, 1)),
    "beach": Style(
        "beach", sky_top=(0.40, 0.62, 0.92), sky_bottom=(0.95, 0.85, 0.70),
        floor_rgb1=(0.86, 0.76, 0.58), floor_rgb2=(0.83, 0.73, 0.55), floor_texture="flat",
        ambient=0.4, headlight=0.25, key_light=0.75, haze=(0.9, 0.85, 0.75, 1)),
}


def build_scene(n: int, colors, style: Style, width=1080, height=1920, terrain=None,
                per_crab_faces=None):
    from .lod import swarm_xml

    world = mujoco.MjSpec.from_string(swarm_xml(n, style.mesh_faces, per_crab_faces))
    world.compiler.degree = False
    world.option.timestep = 0.001
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    v = world.visual
    v.global_.offwidth, v.global_.offheight = width, height
    v.quality.shadowsize = 4096 if style.shadows else 0
    v.quality.offsamples = 4
    v.headlight.ambient = [style.ambient] * 3
    v.headlight.diffuse = [style.headlight] * 3
    v.headlight.specular = [0.1] * 3
    v.rgba.haze = list(style.haze)
    v.map.znear = 0.01    # x extent: 0.1 m with the extent pinned below
    world.stat.extent = 10.0     # else the sea and floor make it ~180 m and close-ups clip
    v.map.zfar = 200
    world.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                      builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                      rgb1=list(style.sky_top), rgb2=list(style.sky_bottom),
                      width=512, height=3072)
    if style.floor_texture == "stripes":
        builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    elif style.floor_texture == "checker":
        builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    else:
        builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
    world.add_texture(name="floortex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=builtin,
                      rgb1=list(style.floor_rgb1), rgb2=list(style.floor_rgb2),
                      width=512, height=512, mark=mujoco.mjtMark.mjMARK_NONE)
    mat = world.add_material(name="floormat", reflectance=style.floor_reflect,
                             texrepeat=[style.floor_scale, style.floor_scale], texuniform=True)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floortex"
    world.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[300, 300, 0.1],
                             material="floormat", contype=0, conaffinity=0,
                             pos=[0, 0, -0.15 if terrain else 0])
    if terrain:
        from .terrain import add_hfield

        add_hfield(world, contype=0, conaffinity=0, material="floormat", **terrain)
    key_dir = [-0.6, 0.5, -0.45] if style.ocean else [-0.3, 0.4, -1]
    key = world.worldbody.add_light(name="key", pos=[12, -10, 9], dir=key_dir,
                                    diffuse=[style.key_light] * 3, specular=[0.2] * 3,
                                    castshadow=style.shadows)
    key.cutoff = 60
    key.exponent = 1
    key.range = 500.0     # MuJoCo >= 3.3 lights stop at `range` (default 10 m)
    if style.ocean:     # the moon: a directional light, so the whole beach is lit evenly
        key.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    if style.ocean:
        sea = world.add_material(name="sea", rgba=[*style.ocean, 1], reflectance=0.6,
                                 specular=0.9, shininess=0.9)
        world.worldbody.add_geom(name="sea", type=mujoco.mjtGeom.mjGEOM_BOX,
                                 size=[80, 40, 0.05], pos=[0, style.ocean_y + 40, -0.02],
                                 material="sea", contype=0, conaffinity=0)
    for p, d, c in style.extra_lights:
        lt = world.worldbody.add_light(pos=list(p), dir=list(d), diffuse=list(c), castshadow=False)
        lt.range = 500.0
        lt.cutoff = 70

    for i in range(n):
        m = world.material(f"c{i}_shell")
        m.rgba = [*colors[i], 1.0]
        m.emission = style.glow
    model = world.compile()
    return model


class Renderer:
    def __init__(self, n, colors, style="night", width=1080, height=1920, terrain=None):
        self.style = STYLES[style] if isinstance(style, str) else style
        self.model = build_scene(n, colors, self.style, width, height, terrain)
        self.data = mujoco.MjData(self.model)
        self.r = mujoco.Renderer(self.model, height, width, max_geom=24000)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.n = n
        self.nq1 = (self.model.nq) // n
        self.shell = np.array([self.model.material(f"c{i}_shell").id for i in range(n)])
        self.display = np.array([self.model.geom(f"c{i}_display_module_link_visual").id
                                 for i in range(n)])

    def set_lights(self, colors, glow, max_emission=1.0):
        """colors (n,3) in 0..1, glow (n,) 0..1. Dark crabs fade to a dim body colour."""
        m = self.model
        base = 0.08
        if self.style.glow <= 0:
            m.mat_rgba[self.shell, :3] = colors
            return
        g = np.asarray(glow)[:, None]
        m.mat_rgba[self.shell, :3] = base + (np.asarray(colors) - base) * np.clip(g * 1.2, 0, 1)
        m.mat_emission[self.shell] = max_emission * self.style.glow * np.asarray(glow)

    def set_poses(self, qposes):
        for i, q in enumerate(qposes):
            self.data.qpos[i * self.nq1:(i + 1) * self.nq1] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)   # lights and cameras follow bodies

    def frame(self, lookat, distance, azimuth, elevation, extra=None):
        self.cam.lookat[:] = lookat
        self.cam.distance = distance
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        self.r.update_scene(self.data, self.cam, self.opt)
        if extra is not None:
            extra(self.r.scene)
        if self.style.shadows:
            self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self.style.floor_reflect > 0
        return self.r.render()
