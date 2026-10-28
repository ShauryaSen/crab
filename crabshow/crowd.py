"""Instanced crowd rendering: thousands of crabs from one set of meshes.

The scene model holds a single crab, which is never drawn itself. For every crab in the
crowd we run forward kinematics on a one-crab model with that crab's qpos and add its
visual meshes to the frame as user geoms, so mesh memory and compile time stay flat no
matter how many crabs there are. Per frame the cost is one mj_kinematics per crab plus
the triangles.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .render import STYLES, build_scene

SHELL_RGBA = (0.85, 0.08, 0.05)


class CrowdRenderer:
    def __init__(self, style="moonbeach", width=1080, height=1350, faces=3000, max_geom=120000):
        self.style = STYLES[style] if isinstance(style, str) else style
        self.style.mesh_faces = faces
        # One template crab, parked far below the floor (never visible).
        self.model = build_scene(1, np.ones((1, 3)), self.style, width, height)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[2] = -50
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        self.r = mujoco.Renderer(self.model, height, width, max_geom=max_geom)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        m = self.model
        self.vis = [g for g in range(m.ngeom)
                    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_bodyid[g] > 0]
        shell_mat = m.material("c0_shell").id
        self.is_shell = np.array([m.geom_matid[g] == shell_mat for g in self.vis])
        self.base_rgba = np.array([m.geom_rgba[g] for g in self.vis])
        self.display = m.geom("c0_display_module_link_visual").id
        self.nq1 = m.nq

    def frame(self, qposes, colors, glow, lookat, distance, azimuth, elevation, extra=None,
              cull=None):
        m, d = self.model, self.data
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.r.update_scene(d, self.cam, self.opt)
        scn = self.r.scene
        # Drop the template crab's own geoms from the scene.
        keep = 0
        for k in range(scn.ngeom):
            if scn.geoms[k].objtype != mujoco.mjtObj.mjOBJ_GEOM or \
                    m.geom_bodyid[scn.geoms[k].objid] == 0:
                if keep != k:
                    mujoco.mjv_initGeom(scn.geoms[keep], scn.geoms[k].type, scn.geoms[k].size,
                                        scn.geoms[k].pos, scn.geoms[k].mat, scn.geoms[k].rgba)
                    g, h = scn.geoms[keep], scn.geoms[k]
                    g.dataid, g.matid, g.objtype, g.objid = h.dataid, h.matid, h.objtype, h.objid
                    g.category, g.emission, g.texcoord = h.category, h.emission, h.texcoord
                    g.reflectance, g.specular, g.shininess = h.reflectance, h.specular, h.shininess
                keep += 1
        scn.ngeom = keep
        em = self.style.glow
        for i, q in enumerate(qposes):
            if cull is not None and not cull[i]:
                continue
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            gl = glow[i]
            shell = 0.08 + (np.asarray(colors[i]) - 0.08) * min(1.0, gl * 1.2) if em > 0 \
                else np.asarray(colors[i])
            for j, g in enumerate(self.vis):
                if scn.ngeom >= scn.maxgeom:
                    break
                vg = scn.geoms[scn.ngeom]
                rgba = np.array([*shell, 1.0]) if self.is_shell[j] else self.base_rgba[j]
                mujoco.mjv_initGeom(vg, mujoco.mjtGeom.mjGEOM_MESH, np.ones(3), d.geom_xpos[g],
                                    d.geom_xmat[g], rgba.astype(np.float32))
                vg.dataid = 2 * m.geom_dataid[g]
                vg.emission = em * gl if self.is_shell[j] else 0.0
                vg.specular, vg.shininess = 0.5, 0.6
                vg.category = mujoco.mjtCatBit.mjCAT_DYNAMIC
                scn.ngeom += 1
        d.qpos[2] = -50
        mujoco.mj_kinematics(m, d)
        if extra is not None:
            extra(scn)
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.style.shadows
        scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self.style.floor_reflect > 0
        return self.r.render()
