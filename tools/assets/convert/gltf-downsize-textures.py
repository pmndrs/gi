"""Shrink the textures of a glTF scene, and write a glTF that keeps the original structure. Uses Blender to read.

Run it in background mode; no window opens:

    blender -b --factory-startup --python tools/assets/convert/gltf-downsize-textures.py -- <in.gltf> <out.gltf> <max_side>

Tested with Blender 4.5 LTS on Windows. Blender 4.2 and 5.x were not tested.

Why this script edits the glTF JSON itself and does not re-export through Blender's glTF exporter:
- The exporter rebuilds the materials, and that changes how they look. Here only
  the image files change size.
- The JSON is read and written with Python's json module. Materials, meshes,
  accessors and nodes are copied unchanged, except for the two edits below.

Why the resizing is a plain block average written by this script:
- Each output pixel is the exact average of a block of input pixels, so the mean of
  every channel is kept (checked in float64 on all 72 textures; see the report).
- The PNG is written by this script with Python's zlib, so the bytes do not depend on
  Blender's image encoder. The sizes must divide evenly (4096 -> 1024 is a 4x4
  block average); the script stops if they don't.
- Blender is used only to read the pixels, as Non-Color, so the values are the
  stored bytes divided by 255 with no colour conversion.

What it does:
1. Each image is read, box-averaged to at most <max_side> pixels on its longer
   side (aspect ratio and alpha kept), and saved as 8-bit PNG into
   <out folder>/textures/. PNG is used for every image: normal and roughness maps
   should not go through JPEG. The file bytes depend on the zlib version that
   Blender bundles, so a different Python or zlib can change the file hashes.
2. The output glTF points each image at its new file. Its buffer URI is a relative
   path to the ORIGINAL .bin next to the input, so the large binary is not copied.
3. Every emissiveFactor component is clamped to [0, 1]. glTF requires that range,
   and the Intel scene has one value of 1.00000012 from float rounding.

The script stops with an error if an image is embedded (data URI or bufferView),
because this scene has none and the code does not handle them.
"""

import json
import os
import struct
import sys
import zlib

import bpy
import numpy as np

# Keep an emissive colour inside the glTF range [0, 1]; values above it come from float rounding.
EMISSIVE_MIN = 0.0
EMISSIVE_MAX = 1.0

# PNG colour types that carry an alpha channel: 4 (grey + alpha) and 6 (RGB + alpha).
PNG_COLOR_TYPES_WITH_ALPHA = (4, 6)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def parse_args():
    if "--" not in sys.argv:
        sys.exit("usage: blender -b --python gltf-downsize-textures.py -- <in.gltf> <out.gltf> <max_side>")
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 3:
        sys.exit("usage: blender -b --python gltf-downsize-textures.py -- <in.gltf> <out.gltf> <max_side>")
    in_path, out_path = os.path.abspath(args[0]), os.path.abspath(args[1])
    max_side = int(args[2])
    if max_side < 1:
        sys.exit("max_side must be at least 1")
    return in_path, out_path, max_side


def clamp_emissive(doc):
    clamped = 0
    for material in doc.get("materials", []):
        factor = material.get("emissiveFactor")
        if not factor:
            continue
        fixed = [min(max(value, EMISSIVE_MIN), EMISSIVE_MAX) for value in factor]
        if fixed != factor:
            material["emissiveFactor"] = fixed
            clamped += 1
    print(f"EMISSIVE clamped {clamped} material(s)")


def png_header(path):
    """Return (width, height, bit depth, colour type) from the IHDR chunk."""
    with open(path, "rb") as f:
        head = f.read(26)
    if head[:8] != PNG_SIGNATURE:
        sys.exit(f"not a PNG file: {path}")
    width = int.from_bytes(head[16:20], "big")
    height = int.from_bytes(head[20:24], "big")
    return width, height, head[24], head[25]


def scaled_size(width, height, max_side):
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    factor = longest // max_side
    if longest % max_side != 0 or width % factor != 0 or height % factor != 0:
        sys.exit(f"{width}x{height} cannot be reduced to {max_side} by an exact whole-number block average")
    return width // factor, height // factor


def block_average(pixels, width, height, factor):
    """Average factor x factor blocks of a flat RGBA float array (Blender row order)."""
    if factor == 1:
        return pixels
    grid = pixels.reshape(height, width, 4)
    blocks = grid.reshape(height // factor, factor, width // factor, factor, 4)
    return blocks.mean(axis=(1, 3)).reshape(-1)


def png_chunk(tag, data):
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def write_png(path, rows, has_alpha):
    """Write an 8-bit PNG from a uint8 array shaped (height, width, 4), top row first."""
    height, width = rows.shape[0], rows.shape[1]
    channels = rows if has_alpha else rows[:, :, :3]
    colour_type = 6 if has_alpha else 2
    # Filter type 0 (none) before each scanline.
    filtered = np.concatenate(
        [np.zeros((height, 1), dtype=np.uint8), channels.reshape(height, -1)],
        axis=1,
    )
    header = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    data = PNG_SIGNATURE + png_chunk(b"IHDR", header)
    data += png_chunk(b"IDAT", zlib.compress(filtered.tobytes(), 9))
    data += png_chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(data)


def downsize_images(doc, in_dir, out_dir, max_side):
    textures_dir = os.path.join(out_dir, "textures")
    os.makedirs(textures_dir, exist_ok=True)
    total_bytes = 0
    for image in doc.get("images", []):
        if "bufferView" in image or str(image.get("uri", "")).startswith("data:"):
            sys.exit("embedded images are not handled by this script")
        source = os.path.join(in_dir, image["uri"])
        name = os.path.splitext(os.path.basename(image["uri"]))[0] + ".png"
        target = os.path.join(textures_dir, name)

        width, height, bit_depth, colour_type = png_header(source)
        if bit_depth != 8:
            sys.exit(f"{name}: only 8-bit PNG is handled (found {bit_depth})")
        has_alpha = colour_type in PNG_COLOR_TYPES_WITH_ALPHA
        new_width, new_height = scaled_size(width, height, max_side)
        factor = width // new_width

        loaded = bpy.data.images.load(source, check_existing=False)
        # Non-Color: the pixel values are the stored bytes / 255, with no colour conversion.
        loaded.colorspace_settings.name = "Non-Color"
        pixels = np.empty(width * height * 4, dtype=np.float32)
        loaded.pixels.foreach_get(pixels)
        bpy.data.images.remove(loaded)

        small = block_average(pixels, width, height, factor)
        # Blender stores rows bottom-up; PNG stores them top-down.
        values = np.clip(np.round(small.reshape(new_height, new_width, 4) * 255.0), 0, 255).astype(np.uint8)
        write_png(target, np.flipud(values), has_alpha)

        image["uri"] = "textures/" + name
        image["mimeType"] = "image/png"
        total_bytes += os.path.getsize(target)
        print(f"TEXTURE {name} {width}x{height} -> {new_width}x{new_height} bytes {os.path.getsize(target)}")
    print(f"TEXTURES written {len(doc.get('images', []))} total bytes {total_bytes}")


def point_buffers_at_original(doc, in_dir, out_dir):
    for buffer in doc.get("buffers", []):
        uri = buffer.get("uri")
        if uri is None or uri.startswith("data:"):
            continue
        relative = os.path.relpath(os.path.join(in_dir, uri), out_dir)
        buffer["uri"] = relative.replace("\\", "/")
        print(f"BUFFER {uri} -> {buffer['uri']}")


def main():
    in_path, out_path, max_side = parse_args()
    in_dir = os.path.dirname(in_path)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    with open(in_path, encoding="utf-8") as f:
        doc = json.load(f)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    clamp_emissive(doc)
    downsize_images(doc, in_dir, out_dir, max_side)
    point_buffers_at_original(doc, in_dir, out_dir)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"EXPORTED {out_path} bytes {os.path.getsize(out_path)}")


main()
