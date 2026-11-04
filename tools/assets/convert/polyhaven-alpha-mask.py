"""Make the leaves of a Poly Haven tree alpha-cutout (MASK), and export one GLB, using Blender.

Run it in background mode; no window opens:

    blender -b --factory-startup --python tools/assets/convert/polyhaven-alpha-mask.py -- <in.gltf> <alpha.png> <out.glb>

Tested with Blender 4.5 LTS on Windows. Blender 4.2 and 5.x were not tested.

Why this script exists:
- The Poly Haven glTF marks the leaf material BLEND and uses an opaque JPEG as
  its diffuse. So the leaves have no cutouts as delivered. The cutout is in a
  separate grayscale PNG (leaves_alpha), which the glTF does not reference.
- Blender 4.5's glTF exporter decides alpha mode from the node setup, not from
  the material's blend setting. It writes MASK when the alpha input is
  `alpha > cutoff`, built as a Math node set to Greater Than with the cutoff
  as its second input. That is what this script builds, with a cutoff of 0.5.
- Only the leaf material changes. Other materials and the bin file are kept as
  they are.

The script finds the leaf material by its name ending in "_leaves". It stops with
an error if that name matches zero or several materials.
"""

import os
import sys

import bpy
import numpy as np

# glTF's default cutoff: an alpha at or above this value is kept, below it is cut out.
ALPHA_CUTOFF = 0.5

# Poly Haven names the leaf material "<asset>_leaves" in its glTF.
LEAF_MATERIAL_SUFFIX = "_leaves"


def parse_args():
    if "--" not in sys.argv:
        sys.exit("usage: blender -b --python polyhaven-alpha-mask.py -- <in.gltf> <alpha.png> <out.glb>")
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 3:
        sys.exit("usage: blender -b --python polyhaven-alpha-mask.py -- <in.gltf> <alpha.png> <out.glb>")
    return tuple(os.path.abspath(arg) for arg in args)


def import_gltf(gltf_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=gltf_path)


def find_leaf_material():
    matches = [m for m in bpy.data.materials if m.name.endswith(LEAF_MATERIAL_SUFFIX) and m.use_nodes]
    if len(matches) != 1:
        names = [m.name for m in bpy.data.materials]
        sys.exit(f"expected one material ending in {LEAF_MATERIAL_SUFFIX!r}, found {len(matches)}: {names}")
    return matches[0]


def principled_node(material):
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    sys.exit(f"no Principled BSDF in material {material.name}")


def cutout_share(image):
    """Share of alpha texels that stay after the cutoff. Printed so the report can state it."""
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    value = pixels[0::4]  # The PNG is grayscale; the red channel carries the value.
    return float((value >= ALPHA_CUTOFF).mean()), (width, height)


def add_alpha_cutout(material, alpha_path):
    tree = material.node_tree
    bsdf = principled_node(material)

    # The glTF importer can link an alpha input of its own. Replace it, so the
    # exporter sees only the cutout.
    for link in list(bsdf.inputs["Alpha"].links):
        tree.links.remove(link)

    alpha_image = bpy.data.images.load(alpha_path, check_existing=False)
    alpha_image.colorspace_settings.name = "Non-Color"
    share, size = cutout_share(alpha_image)
    print(f"ALPHA image {os.path.basename(alpha_path)} size {size} texels kept at cutoff {ALPHA_CUTOFF}: {share:.3f}")

    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = alpha_image
    texture.label = "leaf alpha"

    # Grayscale colour -> float uses the channel value, so the Color output is enough.
    threshold = tree.nodes.new("ShaderNodeMath")
    threshold.operation = "GREATER_THAN"
    threshold.inputs[1].default_value = ALPHA_CUTOFF
    tree.links.new(texture.outputs["Color"], threshold.inputs[0])
    tree.links.new(threshold.outputs[0], bsdf.inputs["Alpha"])
    print(f"MASK set on material {material.name}")


def export_glb(out_path):
    bpy.ops.export_scene.gltf(
        filepath=out_path,
        export_format="GLB",
        # Keep JPEG where the source is JPEG. Only the merged leaf texture needs alpha.
        export_image_format="AUTO",
        export_yup=True,
        # Keep the tangent frame that the source glTF did not have, so normal maps match across runtimes.
        export_tangents=True,
        export_materials="EXPORT",
        export_cameras=False,
        export_lights=False,
        export_apply=False,
    )


def main():
    gltf_path, alpha_path, out_path = parse_args()
    import_gltf(gltf_path)
    add_alpha_cutout(find_leaf_material(), alpha_path)
    export_glb(out_path)
    print(f"EXPORTED {out_path} bytes {os.path.getsize(out_path)}")


main()
