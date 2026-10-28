"""A render-only swarm model: decimated visual meshes, loaded once and shared by every crab.

The stock robot carries ~430k visual triangles; 36 attached copies is 15M triangles and
several GB per software-GL context. Here each visual mesh is simplified once (cached under
data/jumper/robot_lod/) and the swarm MJCF lists every mesh a single time, with N renamed
copies of the body tree referencing them. Collision geoms are dropped.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .runtime import DATA, ROBOT_XML

LOD_DIR = DATA / "robot_lod"
SHELL_RGBA = (0.85, 0.08, 0.05)


def _decimate_meshes(root: ET.Element, target_faces_total: int) -> dict[str, str]:
    import fast_simplification
    import trimesh

    lod_dir = LOD_DIR / str(target_faces_total)
    lod_dir.mkdir(parents=True, exist_ok=True)
    meshdir = ROBOT_XML.parent
    visual_meshes = {g.get("mesh") for g in root.iter("geom")
                     if g.get("class") == "visual" and g.get("mesh")}
    sizes = {}
    files = {}
    for m in root.iter("mesh"):
        if m.get("name") in visual_meshes:
            mesh = trimesh.load(meshdir / m.get("file"), force="mesh")
            sizes[m.get("name")] = (mesh, len(mesh.faces))
            files[m.get("name")] = m.get("file")
    total = sum(n for _, n in sizes.values())
    out = {}
    for name, (mesh, nf) in sizes.items():
        dst = lod_dir / f"{name}.stl"
        if not dst.exists():
            keep = max(200, int(nf * target_faces_total / total))
            reduction = 1.0 - min(1.0, keep / nf)
            if reduction > 0.01:
                v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32),
                                                    mesh.faces.astype(np.int32),
                                                    target_reduction=reduction)
                mesh = trimesh.Trimesh(v, f, process=False)
            mesh.export(dst)
        out[name] = str(dst)
    return out


def swarm_xml(n: int, target_faces: int = 24000, per_crab_faces: list | None = None) -> str:
    """MJCF text for n kinematic crabs (free joints, no collisions) sharing one mesh set.

    per_crab_faces gives each crab its own detail level (crowd templates); meshes for each
    level are loaded once and named <mesh>__<faces>.
    """
    tree = ET.parse(ROBOT_XML)
    root = tree.getroot()
    levels = per_crab_faces or [target_faces] * n
    lods = {f: _decimate_meshes(root, f) for f in sorted(set(levels))}
    lod = lods[levels[0]]
    world = ET.Element("mujoco", {"model": "swarm"})
    ET.SubElement(world, "compiler", {"angle": "radian"})
    ET.SubElement(world, "option", {"timestep": "0.001"})
    world.append(copy.deepcopy(root.find("default")))
    asset = ET.SubElement(world, "asset")
    for m in root.find("asset"):
        if m.tag == "mesh" and m.get("name") in lod:
            for f, files in lods.items():
                ET.SubElement(asset, "mesh", {"name": f"{m.get('name')}__{f}",
                                              "file": files[m.get("name")]})
        elif m.tag != "mesh":
            asset.append(copy.deepcopy(m))
    for i in range(n):
        ET.SubElement(asset, "material", {"name": f"c{i}_shell", "rgba": "1 1 1 1",
                                          "specular": "0.6", "shininess": "0.8"})
    wb = ET.SubElement(world, "worldbody")
    base = root.find("worldbody").find("body")
    for i in range(n):
        b = copy.deepcopy(base)
        for el in b.iter():
            if el.tag == "geom" and (el.get("class") != "visual" or el.get("mesh") not in lod):
                el.set("_drop", "1")
            if el.get("name") and el.tag in ("body", "joint", "geom", "site", "camera", "light"):
                el.set("name", f"c{i}_{el.get('name')}")
        for parent in b.iter():
            for ch in list(parent):
                if ch.get("_drop"):
                    parent.remove(ch)
        for g in b.iter("geom"):
            if g.get("mesh"):
                g.set("mesh", f"{g.get('mesh')}__{levels[i]}")
            rgba = [float(x) for x in (g.get("rgba") or "0.5 0.5 0.5 1").split()]
            if np.allclose(rgba[:3], SHELL_RGBA, atol=0.01):
                g.set("rgba", "0.5 0.5 0.5 1")    # default rgba: the material's colour wins
                g.set("material", f"c{i}_shell")
        wb.append(b)
    return ET.tostring(world, encoding="unicode")
