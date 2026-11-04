"""Convert an FBX scene (Amazon Lumberyard Bistro, ORCA) to one GLB, using Blender.

Run it in background mode; no window opens:

    blender -b --factory-startup --python tools/assets/convert/fbx-to-glb.py -- <in.fbx> <out.glb>

Tested with Blender 4.5 LTS on Windows. Blender 4.2 and 5.x were not tested.

Why this script does more than a plain FBX import and GLB export:

1. The Bistro FBX files reference DDS textures next to them. `use_image_search`
   finds them, so the importer does not need absolute paths.
2. The importer wires the packed "Specular" map into Specular IOR Level. In this
   pack the channels are R = occlusion, G = roughness, B = metalness, which is the
   glTF metallic-roughness layout. We move G and B into Roughness and Metallic.
   Occlusion (R) is dropped; the maps we checked store 0 there, and wiring it
   would darken everything.
3. The normal maps are DirectX style (green points down). glTF expects OpenGL style
   (green points up), so the green channel is inverted. The inverted copies are
   written to a scratch folder, not the source files.
4. The importer loads opacity from a second copy of each base-colour texture. The
   copy holds the same data, so the script reads the alpha from the base-colour
   texture and removes the duplicate node.
5. Alpha mode is chosen from the data, not from the material name:
   - no texel below 0.5 -> opaque;
   - almost all texels are 0 or 1 (a cutout) -> MASK with cutoff 0.5;
   - otherwise -> BLEND.
   The glTF exporter reads MASK from a `Greater Than 0.5` Math node on the alpha
   input, so the script adds that node.
6. Textures go out as WebP, not PNG. The Bistro DDS textures are 2048 px; PNG
   would make the GLB several GB. three r187 reads `EXT_texture_webp`.

Units: the FBX files are already in metres (unit scale 1.0). The script prints the
bounding box so a wrong unit shows up in the log.
"""

import os
import shutil
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Vector

# Alpha at or above this value is kept; below it the texel is cut out (MASK).
ALPHA_CUTOFF = 0.5

# An alpha map counts as a cutout when no more than this share of its texels lie
# between 0.05 and 0.95. Foliage in this pack measures 0.01 to 0.10; real
# translucency (glass) would sit far above it.
CUTOUT_MAX_SOFT_SHARE = 0.2

# A packed roughness/metal map this small holds one value. We copy that value
# into the material sockets instead of keeping a texture. The Bistro maps are
# 16x16 constants, so this covers almost every material.
CONSTANT_MAP_MAX_SIDE = 64

# WebP keeps alpha and is far smaller than PNG for 2048 px textures.
TEXTURE_FORMAT = "WEBP"
TEXTURE_QUALITY = 85


def parse_args():
    if "--" not in sys.argv:
        sys.exit("usage: blender -b --python fbx-to-glb.py -- <in.fbx> <out.glb>")
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 2:
        sys.exit("usage: blender -b --python fbx-to-glb.py -- <in.fbx> <out.glb>")
    return os.path.abspath(args[0]), os.path.abspath(args[1])


def import_fbx(fbx_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(
        filepath=fbx_path,
        use_image_search=True,
        axis_forward="-Z",
        axis_up="Y",
        global_scale=1.0,
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


def principled_node(material):
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def image_node_on(socket):
    """Return the image texture node that feeds this socket directly, or None."""
    if not socket.is_linked:
        return None
    node = socket.links[0].from_node
    return node if node.type == "TEX_IMAGE" and node.image is not None else None


def mean_rgba(image):
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(-1, 4).mean(axis=0), pixels


def fix_packed_roughness_metal(tree, bsdf):
    """Move the packed map's G (roughness) and B (metal) into the right sockets."""
    socket = bsdf.inputs["Specular IOR Level"]
    node = image_node_on(socket)
    if node is None:
        return
    tree.links.remove(socket.links[0])

    width, height = node.image.size
    if max(width, height) <= CONSTANT_MAP_MAX_SIDE:
        mean, _ = mean_rgba(node.image)
        bsdf.inputs["Roughness"].default_value = float(mean[1])
        bsdf.inputs["Metallic"].default_value = float(mean[2])
        tree.nodes.remove(node)
        return

    split = tree.nodes.new("ShaderNodeSeparateColor")
    tree.links.new(node.outputs["Color"], split.inputs["Color"])
    tree.links.new(split.outputs["Green"], bsdf.inputs["Roughness"])
    tree.links.new(split.outputs["Blue"], bsdf.inputs["Metallic"])


def green_flipped_image(source, scratch_dir, cache):
    """Return a copy of a DirectX normal map with its green channel inverted."""
    if source.name in cache:
        return cache[source.name]

    width, height = source.size
    _, pixels = mean_rgba(source)
    pixels[1::4] = 1.0 - pixels[1::4]

    staging = bpy.data.images.new(source.name + "_GL", width, height, alpha=True)
    staging.colorspace_settings.name = "Non-Color"
    staging.pixels.foreach_set(pixels)
    staging.update()
    path = os.path.join(scratch_dir, source.name + "_GL.png")
    staging.filepath_raw = path
    staging.file_format = "PNG"
    staging.save()
    bpy.data.images.remove(staging)

    flipped = bpy.data.images.load(path, check_existing=False)
    flipped.colorspace_settings.name = "Non-Color"
    cache[source.name] = flipped
    return flipped


def fix_normal_map(tree, scratch_dir, cache):
    for node in tree.nodes:
        if node.type != "NORMAL_MAP":
            continue
        source = image_node_on(node.inputs["Color"])
        if source is None:
            continue
        source.image = green_flipped_image(source.image, scratch_dir, cache)


def is_same_file(a, b):
    return a.image is not None and b.image is not None and a.image.filepath == b.image.filepath


def remove_if_unused(tree, node):
    if not any(output.is_linked for output in node.outputs):
        tree.nodes.remove(node)


def fix_alpha(tree, bsdf, stats):
    """Pick opaque, cutout or blend from the opacity data, and wire it correctly."""
    alpha_socket = bsdf.inputs["Alpha"]
    alpha_node = image_node_on(alpha_socket)
    if alpha_node is None:
        return "OPAQUE"

    _, pixels = mean_rgba(alpha_node.image)
    alpha = pixels[3::4]
    if (alpha < ALPHA_CUTOFF).sum() == 0:
        tree.links.remove(alpha_socket.links[0])
        remove_if_unused(tree, alpha_node)
        stats["OPAQUE"] = stats.get("OPAQUE", 0) + 1
        return "OPAQUE"

    soft_share = float(((alpha > 0.05) & (alpha < 0.95)).mean())
    mode = "MASK" if soft_share <= CUTOUT_MAX_SOFT_SHARE else "BLEND"
    stats[mode] = stats.get(mode, 0) + 1

    base_node = image_node_on(bsdf.inputs["Base Color"])
    if base_node is not None and is_same_file(base_node, alpha_node):
        # Same DDS file: take alpha from the base-colour texture, so the exporter
        # writes one texture instead of two.
        tree.links.remove(alpha_socket.links[0])
        tree.links.new(base_node.outputs["Alpha"], alpha_socket)
        remove_if_unused(tree, alpha_node)

    if mode == "MASK":
        threshold = tree.nodes.new("ShaderNodeMath")
        threshold.operation = "GREATER_THAN"
        threshold.inputs[1].default_value = ALPHA_CUTOFF
        source = alpha_socket.links[0].from_socket
        tree.links.remove(alpha_socket.links[0])
        tree.links.new(source, threshold.inputs[0])
        tree.links.new(threshold.outputs[0], alpha_socket)
    return mode


def fix_materials(scratch_dir):
    flipped_cache = {}
    alpha_stats = {}
    materials = [m for m in bpy.data.materials if m.use_nodes and m.node_tree is not None]
    for material in materials:
        tree = material.node_tree
        bsdf = principled_node(material)
        if bsdf is None:
            continue
        fix_packed_roughness_metal(tree, bsdf)
        fix_normal_map(tree, scratch_dir, flipped_cache)
        fix_alpha(tree, bsdf, alpha_stats)
    print(f"MATERIALS {len(materials)} alpha modes {alpha_stats} normal maps flipped {len(flipped_cache)}")


def export_glb(out_path):
    bpy.ops.export_scene.gltf(
        filepath=out_path,
        export_format="GLB",
        export_image_format=TEXTURE_FORMAT,
        export_image_quality=TEXTURE_QUALITY,
        export_yup=True,
        # Normal maps need a tangent frame; exporting it avoids a runtime guess.
        export_tangents=True,
        export_materials="EXPORT",
        export_cameras=False,
        export_lights=False,
        export_apply=False,
    )


def main():
    fbx_path, out_path = parse_args()
    import_fbx(fbx_path)

    mesh_count, triangles, low, high = scene_bounds_and_triangles()
    size = high - low
    print(f"SCENE meshes {mesh_count} triangles {triangles}")
    print(f"BOUNDS min {low.round(2).tolist()} max {high.round(2).tolist()} size_m {size.round(2).tolist()}")

    scratch_dir = tempfile.mkdtemp(prefix="fbx-to-glb-")
    try:
        fix_materials(scratch_dir)
        export_glb(out_path)
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
    print(f"EXPORTED {out_path} bytes {os.path.getsize(out_path)}")


main()
