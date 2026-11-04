"""Convert a Wavefront OBJ scene (MTL and textures in the same folder) to one GLB, using Blender.

Run it in background mode; no window opens:

    blender -b --factory-startup --python tools/assets/convert/obj-to-glb.py -- <in.obj> <out.glb>

Tested with Blender 4.5 LTS on Windows. Blender 4.2 and 5.x were not tested.

What it does and why:
- The OBJ importer reads the MTL file and resolves the texture paths next to the OBJ.
- Textures keep their original format (JPEG stays JPEG, PNG stays PNG). The OBJ
  sources are 8-bit, so re-encoding would only add loss.
- Blender's importer turns `illum 4` plus `Tf` into full glass transmission on
  every material. The Breakfast Room MTL has no `d` or `Tr` (opacity), so that
  reading would make the walls and floor transparent. We set transmission to 0
  on every material. The count is printed.
- The importer also guesses PBR values from old Phong fields, and for these
  scenes every guess is wrong (see `make_dielectric`):
  - Metallic = the average of the ambient colour `Ka` (when `illum` turns
    reflection on, as `illum 4` does). The Breakfast Room MTL has `Ka 1 1 1` on
    almost every material, so walls, floor and paint all came out fully metal
    and got no diffuse light at all.
  - Specular IOR Level = the average of `Ks`, and IOR = `Ni`. `Ni 1.0` means no
    Fresnel reflection, and `Ks` was tuned for Phong, not for a PBR lobe.
  We reset all three to a plain dielectric (Metallic 0, Principled's default
  specular, IOR 1.5), so glTF gets metallicFactor 0 and no specular or IOR
  extensions. Base colour, textures and roughness (from `Ns`) are kept.
- So materials named "Chrome" come out as dark, rough dielectrics. MTL has no
  trustworthy metal value, so this is a known limit, not fixed here.
- OBJ has no up axis in its data. We assume -Z forward and Y up, which is what
  Blender writes and what the glTF exporter expects. The script prints the
  bounding box so a wrong unit or axis shows up in the log.
"""

import os
import sys

import bpy
import numpy as np
from mathutils import Vector


def parse_args():
    if "--" not in sys.argv:
        sys.exit("usage: blender -b --python obj-to-glb.py -- <in.obj> <out.glb>")
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 2:
        sys.exit("usage: blender -b --python obj-to-glb.py -- <in.obj> <out.glb>")
    return os.path.abspath(args[0]), os.path.abspath(args[1])


def import_obj(obj_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(
        filepath=obj_path,
        forward_axis="NEGATIVE_Z",
        up_axis="Y",
    )


def scene_bounds_and_triangles():
    low = np.array([np.inf, np.inf, np.inf])
    high = np.array([-np.inf, -np.inf, -np.inf])
    triangles = 0
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    for obj in meshes:
        for corner in obj.bound_box:
            world = np.array(obj.matrix_world @ Vector(corner))
            low = np.minimum(low, world)
            high = np.maximum(high, world)
        obj.data.calc_loop_triangles()
        triangles += len(obj.data.loop_triangles)
    return len(meshes), triangles, low, high


def clear_transmission():
    cleared = 0
    for material in bpy.data.materials:
        if not material.use_nodes or material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            socket = node.inputs["Transmission Weight"]
            if socket.is_linked or socket.default_value != 0.0:
                socket.default_value = 0.0
                cleared += 1
    print(f"TRANSMISSION cleared on {cleared} materials")


# Principled BSDF inputs the OBJ importer derives from Phong-era MTL fields,
# and the plain-dielectric value each one gets instead.
#   Metallic           from Ka (ambient colour): 0, nothing in these scenes is metal we can trust.
#   Specular IOR Level from Ks: 0.5, Principled's default, which glTF writes as no
#                      KHR_materials_specular (specularFactor 1).
#   Specular Tint      from Ks colour: white, so the specular lobe is not tinted.
#   IOR                from Ni: 1.5, glTF's default, which gives the usual F0 of
#                      0.04 ((1.5 - 1) / (1.5 + 1))^2. Ni 1.0 would mean no reflection.
DIELECTRIC_INPUTS = {
    "Metallic": 0.0,
    "Specular IOR Level": 0.5,
    "Specular Tint": (1.0, 1.0, 1.0, 1.0),
    "IOR": 1.5,
}


def same_value(current, wanted):
    if isinstance(wanted, tuple):
        return all(abs(a - b) < 1e-6 for a, b in zip(current, wanted))
    return abs(current - wanted) < 1e-6


def make_dielectric():
    """Resets the importer's Phong-to-PBR guesses on every material to a plain dielectric.

    A linked socket ignores its own value, so links into these inputs are removed
    first. Prints the metallic value each material had, so the log shows what
    the importer did.
    """
    changed = {name: 0 for name in DIELECTRIC_INPUTS}
    for material in sorted(bpy.data.materials, key=lambda m: m.name):
        if not material.use_nodes or material.node_tree is None:
            continue
        tree = material.node_tree
        for node in tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            metallic = node.inputs["Metallic"]
            imported = "linked" if metallic.is_linked else f"{metallic.default_value:.3f}"
            print(f"MATERIAL {material.name}: imported metallic {imported}")
            for name, wanted in DIELECTRIC_INPUTS.items():
                socket = node.inputs[name]
                if not socket.is_linked and same_value(socket.default_value, wanted):
                    continue
                for link in list(socket.links):
                    tree.links.remove(link)
                socket.default_value = wanted
                changed[name] += 1
    summary = ", ".join(f"{name} {count}" for name, count in changed.items())
    print(f"DIELECTRIC reset on materials: {summary}")


def export_glb(out_path):
    bpy.ops.export_scene.gltf(
        filepath=out_path,
        export_format="GLB",
        # Keep the source JPEG/PNG bytes where glTF allows them.
        export_image_format="AUTO",
        export_yup=True,
        export_materials="EXPORT",
        export_cameras=False,
        export_lights=False,
        export_apply=False,
    )


def main():
    obj_path, out_path = parse_args()
    import_obj(obj_path)

    mesh_count, triangles, low, high = scene_bounds_and_triangles()
    print(f"SCENE meshes {mesh_count} triangles {triangles} materials {len(bpy.data.materials)}")
    print(f"BOUNDS min {low.round(2).tolist()} max {high.round(2).tolist()} size_m {(high - low).round(2).tolist()}")

    clear_transmission()
    make_dielectric()
    export_glb(out_path)
    print(f"EXPORTED {out_path} bytes {os.path.getsize(out_path)}")


main()
