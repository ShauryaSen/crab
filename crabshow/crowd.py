"""Instanced crowd rendering: thousands of crabs from two shared mesh sets.

The scene model holds two template crabs (a cheap one and a detailed one), parked out of
sight. For every crab in the crowd we run forward kinematics on one template with that
crab's qpos and add its visual meshes to the frame as user geoms, so model size and
compile time stay flat however many crabs there are. The few crabs nearest the camera use
the detailed template; crabs outside the view cone are skipped; far crabs drop their small
parts. Eyes on each crab's face display are drawn here too.

`CrowdRenderer` mirrors `render.Renderer` (set_poses / set_lights / frame), so the video
pipeline can use either.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .render import STYLES, build_scene

FAR, NEAR = 0, 1
PARK = -60.0


class CrowdRenderer:
    def __init__(self, n, colors, style="moonbeach", width=1080, height=1350, terrain=None,
                 faces_far=2500, faces_near=60000, near_n=24, near_m=7.0, max_geom=None):
        self.style = STYLES[style] if isinstance(style, str) else style
        self.n, self.W, self.H = n, width, height
        self.model = build_scene(2, np.ones((2, 3)), self.style, width, height, terrain,
                                 per_crab_faces=[faces_far, faces_near])
        m = self.model
        self.data = mujoco.MjData(m)
        self.nq1 = (m.nq) // 2
        self.qadr = [0, self.nq1]
        for k in (0, 1):
            self.data.qpos[self.qadr[k] + 2] = PARK
            self.data.qpos[self.qadr[k] + 3] = 1.0
        mujoco.mj_kinematics(m, self.data)
        mujoco.mj_camlight(m, self.data)
        self.r = mujoco.Renderer(m, height, width, max_geom=max_geom or (n * 30 + 20000))
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.near_n, self.near_m = near_n, near_m
        root = [m.body(f"c{k}_base_link").id for k in (0, 1)]
        self.geoms, self.shell, self.rgba, self.big = [], [], [], []
        for k in (0, 1):
            gs = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                  and m.body_rootid[m.geom_bodyid[g]] == root[k]]
            mat = m.material(f"c{k}_shell").id
            self.geoms.append(np.array(gs))
            self.shell.append(np.array([m.geom_matid[g] == mat for g in gs]))
            self.rgba.append(np.array([m.geom_rgba[g] for g in gs], dtype=np.float32))
            # "Big" parts are all a far crab needs: the shell and the larger leg links.
            size = np.array([m.geom_rbound[g] for g in gs])
            self.big.append(size >= np.percentile(size, 45))
        self.display = [m.geom(f"c{k}_display_module_link_visual").id for k in (0, 1)]
        self.q = None
        self.cols = np.ones((n, 3))
        self.glow = np.ones(n)
        self.eyes = True

    # Renderer-compatible API -------------------------------------------------
    def set_poses(self, qposes):
        self.q = qposes

    def set_lights(self, colors, glow):
        self.cols = np.asarray(colors, dtype=float)
        self.glow = np.asarray(glow, dtype=float)

    def frame(self, lookat, distance, azimuth, elevation, extra=None, t=0.0, beat=0.47):
        m, d = self.model, self.data
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.r.update_scene(d, self.cam, self.opt)
        scn = self.r.scene
        self._drop_templates(scn)
        # Camera position and view cone, for culling and picking detail levels.
        az, el = np.radians(azimuth), np.radians(elevation)
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        campos = np.asarray(lookat) - distance * fwd
        vf = np.radians(45.0)
        hf = 2 * np.arctan(np.tan(vf / 2) * self.W / self.H)
        cone = np.cos(np.arctan(np.hypot(np.tan(vf / 2), np.tan(hf / 2))) + 0.15)
        xyz = np.array([q[:3] for q in self.q])
        rel = xyz - campos
        dist = np.linalg.norm(rel, axis=1) + 1e-9
        visible = (rel @ fwd) / dist > cone
        near = np.zeros(self.n, bool)
        order = np.argsort(dist)
        cand = [i for i in order[:self.near_n] if dist[i] < self.near_m]
        near[cand] = True
        em = self.style.glow
        for i in np.nonzero(visible)[0]:
            k = NEAR if near[i] else FAR
            a = self.qadr[k]
            d.qpos[a:a + self.nq1] = self.q[i]
            mujoco.mj_kinematics(m, d)
            gl = float(self.glow[i])
            col = self.cols[i]
            shell = (0.08 + (col - 0.08) * min(1.0, gl * 1.2)) if em > 0 else col
            far_part = (k == FAR) and dist[i] > 9.0
            for j, g in enumerate(self.geoms[k]):
                if far_part and not self.big[k][j]:
                    continue
                if scn.ngeom >= scn.maxgeom:
                    break
                vg = scn.geoms[scn.ngeom]
                rgba = np.array([*shell, 1.0], np.float32) if self.shell[k][j] else self.rgba[k][j]
                mujoco.mjv_initGeom(vg, mujoco.mjtGeom.mjGEOM_MESH, np.ones(3), d.geom_xpos[g],
                                    d.geom_xmat[g], rgba)
                vg.dataid = 2 * m.geom_dataid[g]
                vg.emission = em * gl if self.shell[k][j] else 0.0
                vg.specular, vg.shininess = 0.5, 0.6
                vg.category = mujoco.mjtCatBit.mjCAT_DYNAMIC
                scn.ngeom += 1
            if self.eyes and gl > 0.05 and dist[i] < 40:
                self._eyes(scn, d, self.display[k], col, gl, t, beat, i)
            d.qpos[a + 2] = PARK
        mujoco.mj_kinematics(m, d)
        if extra is not None:
            extra(scn)
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.style.shadows
        scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self.style.floor_reflect > 0
        return self.r.render()

    # -------------------------------------------------------------------------
    def _drop_templates(self, scn):
        """Remove the parked template crabs' own geoms from the scene."""
        m = self.model
        keep = 0
        for k in range(scn.ngeom):
            g = scn.geoms[k]
            is_crab = g.objtype == mujoco.mjtObj.mjOBJ_GEOM and m.geom_bodyid[g.objid] != 0
            if is_crab:
                continue
            if keep != k:
                h = scn.geoms[keep]
                mujoco.mjv_initGeom(h, g.type, g.size, g.pos, g.mat, g.rgba)
                h.dataid, h.matid, h.objtype, h.objid = g.dataid, g.matid, g.objtype, g.objid
                h.category, h.emission, h.texcoord = g.category, g.emission, g.texcoord
                h.reflectance, h.specular, h.shininess = g.reflectance, g.specular, g.shininess
            keep += 1
        scn.ngeom = keep

    def _eyes(self, scn, d, g, col, gl, t, beat, i, blink_every=4.0):
        p = d.geom_xpos[g]
        R = d.geom_xmat[g].reshape(3, 3)
        nrm, up, across = R[:, 0], R[:, 1], R[:, 2]
        ph = (t / beat + (i * 0.37) % blink_every) % blink_every
        open_ = 1.0 if ph > 0.18 else max(0.12, abs(ph - 0.09) / 0.09)
        c = (0.55 + 0.45 * np.asarray(col)) * gl
        mat = np.stack([nrm, up, across], 1).ravel()
        for s in (-1, 1):
            if scn.ngeom >= scn.maxgeom:
                return
            vg = scn.geoms[scn.ngeom]
            pos = p + nrm * 0.0035 + across * (s * 0.024) + up * 0.002
            mujoco.mjv_initGeom(vg, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                                np.array([0.0015, 0.0085 * open_, 0.0065]), pos, mat,
                                np.array([*c, 1.0], np.float32))
            vg.emission = 1.0
            vg.category = mujoco.mjtCatBit.mjCAT_DECOR
            scn.ngeom += 1
