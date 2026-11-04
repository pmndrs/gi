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
- Blender has no metallic value in MTL. Materials named "Chrome" therefore come
  out as dark, rough dielectrics. This is a known limit, not fixed here.
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
    export_glb(out_path)
    print(f"EXPORTED {out_path} bytes {os.path.getsize(out_path)}")


main()
