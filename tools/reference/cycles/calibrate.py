"""
Known-answer checks for the unit conversions in units.py, rendered with the
same code path as the real references (blender_scene.py).

    blender --background --factory-startup --python tools/reference/cycles/calibrate.py

or `node tools/reference/cycles/run.ts --calibrate`.

Each case renders a small image and compares its centre (or mean) with a
number worked out by hand. A white (albedo 1) diffuse plane lit with
irradiance E has radiance E / pi. The cases:

  point      candela -> Blender watts:     E = I / d^2
  sun        lux -> sun strength:          E = lux * cos(incidence)   (also checks the direction)
  spot       cone angles and blend:        E = I cos^3(theta) / h^2 * smoothstep(cos outer, cos inner, cos theta)
  furnace    emissive radiance, albedo, bounce count and the indirect-only pass,
             through the glTF importer: a closed room with albedo 0.5 and
             emission 0.25 has radiance Le / (1 - rho) = 0.5 everywhere, of which
             the camera-visible emission is 0.25, direct light 0.125 and
             indirect 0.125 (rho^2 Le / (1 - rho)).
  linked metal  make_materials_diffuse_only on a Principled metal whose Metallic
             and Roughness are linked (as the glTF importer links them to a
             texture), seen by a camera looking straight down: E / pi with the
             sun's E, and the straight-down camera basis matches three's.

Prints one line per case and exits with code 1 if any is outside tolerance.
"""

import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import blender_scene as bs  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
SCRATCH = os.path.join(os.environ.get('TEMP', REPO_ROOT), 'ddgi-calibrate')
TOLERANCE = 0.01


def white_floor():
    """A 40 m square plane at y = 0 (manifest space) with a pure diffuse albedo-1 material."""
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    plane = bpy.context.object
    material = bpy.data.materials.new('white')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    diffuse = nodes.new('ShaderNodeBsdfDiffuse')
    diffuse.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    output = nodes.new('ShaderNodeOutputMaterial')
    material.node_tree.links.new(diffuse.outputs[0], output.inputs[0])
    plane.data.materials.append(material)


def glossy_metal_floor():
    """
    The same plane, but with the Principled BSDF the glTF importer makes for a
    textured metal: base colour white, Metallic, Roughness and Specular IOR Level
    LINKED to other nodes (Value nodes standing in for the metallicRoughness
    texture). Only make_materials_diffuse_only stands between this and a mirror.
    """
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    plane = bpy.context.object
    material = bpy.data.materials.new('linked-metal')
    material.use_nodes = True
    tree = material.node_tree
    principled = tree.nodes['Principled BSDF']
    principled.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    for name, value in (('Metallic', 1.0), ('Roughness', 0.2), ('Specular IOR Level', 0.5)):
        source = tree.nodes.new('ShaderNodeValue')
        source.outputs[0].default_value = value
        tree.links.new(source.outputs[0], principled.inputs[name])
    plane.data.materials.append(material)


def centre_value(path, half_size=2):
    pixels, width, height = bs.read_exr_pixels(path)
    total = 0.0
    count = 0
    for y in range(height // 2 - half_size, height // 2 + half_size):
        for x in range(width // 2 - half_size, width // 2 + half_size):
            total += pixels[(y * width + x) * 4]
            count += 1
    return total / count


def mean_value(path):
    pixels, width, height = bs.read_exr_pixels(path)
    return sum(pixels[i] for i in range(0, len(pixels), 4)) / (width * height)


def render_plane_case(light, target, samples=64):
    """Renders the plane lit by one manifest light; the camera looks at `target` from a slant, 1 degree wide."""
    scene = bs.reset_scene()
    bs.configure_cycles(scene, 32, 32, samples, 0, 'cpu')
    white_floor()
    bs.add_environment(scene, {'type': 'none'})
    bs.add_manifest_lights(scene, [light])
    bs.add_camera(scene, [target[0], 1.0, target[2] + 0.5], target, 1.0)
    _, files = bs.render_to_files(scene, SCRATCH, 'case-full', 'case-indirect')
    return centre_value(files[0])


def smoothstep(edge0, edge1, x):
    t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def case_point():
    intensity, height = 50.0, 2.0
    measured = render_plane_case(
        {'type': 'point', 'color': [1, 1, 1], 'intensity': intensity, 'position': [0, height, 0], 'range': 100}, (0.0, 0.0, 0.0)
    )
    return 'point 50 cd at 2 m, straight above', measured, intensity / height**2 / math.pi


def case_sun():
    lux = 3.0
    measured = render_plane_case(
        {'type': 'directional', 'color': [1, 1, 1], 'intensity': lux, 'direction': [1, -1, 0]}, (0.0, 0.0, 0.0)
    )
    return 'sun 3 lux, 45 degrees from the vertical', measured, lux * math.cos(math.radians(45)) / math.pi


def case_spot():
    intensity, height = 60.0, 2.0
    outer, inner, theta = 30.0, 10.0, 20.0
    x = height * math.tan(math.radians(theta))
    light = {
        'type': 'spot', 'color': [1, 1, 1], 'intensity': intensity, 'position': [0, height, 0],
        'direction': [0, -1, 0], 'range': 100, 'innerConeDeg': inner, 'outerConeDeg': outer,
    }
    measured = render_plane_case(light, (x, 0.0, 0.0))
    cos_theta = math.cos(math.radians(theta))
    mask = smoothstep(math.cos(math.radians(outer)), math.cos(math.radians(inner)), cos_theta)
    expected = intensity * cos_theta**3 / height**2 * mask / math.pi
    return 'spot 60 cd, cone 10 to 30 degrees, read at 20 degrees', measured, expected


def case_linked_metal_from_above():
    """
    Two fixes at once: a Principled metal whose Metallic and Roughness are linked
    (as the glTF importer links them to a texture) must come out pure Lambert, and
    a camera looking straight down must get a usable basis (forward x up is zero
    there). Sun 3 lux at 45 degrees on a white plane: radiance 3 cos(45) / pi. A
    metal would read about 0 here (its highlight points away from the camera).
    """
    scene = bs.reset_scene()
    bs.configure_cycles(scene, 32, 32, 64, 0, 'cpu')
    glossy_metal_floor()
    if not bs.find_non_diffuse_surfaces():
        raise SystemExit('calibration setup is wrong: the linked metal should be found before the reset')
    bs.make_materials_diffuse_only()
    bs.require_diffuse_only()
    bs.add_environment(scene, {'type': 'none'})
    lux = 3.0
    bs.add_manifest_lights(scene, [{'type': 'directional', 'color': [1, 1, 1], 'intensity': lux, 'direction': [1, -1, 0]}])
    camera = bs.add_camera(scene, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], 1.0)
    # three's fallback: image up is Blender +Y (manifest -Z) when looking straight down.
    image_up_y = camera.matrix_world.col[1][1]
    _, files = bs.render_to_files(scene, SCRATCH, 'case-full', 'case-indirect')
    return [
        ('linked metal reset to Lambert, seen straight down', centre_value(files[0]), lux * math.cos(math.radians(45)) / math.pi),
        ('straight-down camera: image up is Blender +Y (y component)', image_up_y, 1.0),
    ]


def case_furnace():
    scene = bs.reset_scene()
    bs.configure_cycles(scene, 96, 64, 128, 0, 'cpu')
    bs.import_gltf(os.path.join(REPO_ROOT, 'assets', 'generated', 'furnace-room.glb'))
    bs.make_materials_diffuse_only()
    bs.require_diffuse_only()
    bs.add_environment(scene, {'type': 'none'})
    manifest = bs.read_json(os.path.join(REPO_ROOT, 'scenes', 'manifests', 'furnace-room.json'))
    camera = manifest['cameras'][0]
    bs.add_camera(scene, camera['position'], camera['target'], camera['fovYDeg'])
    _, files = bs.render_to_files(scene, SCRATCH, 'furnace-full', 'furnace-indirect')
    return [
        ('furnace room, full image (Le / (1 - rho) = 0.5)', mean_value(files[0]), 0.5),
        ('furnace room, indirect-only image (rho^2 Le / (1 - rho) = 0.125)', mean_value(files[1]), 0.125),
    ]


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    results = [case_point(), case_sun(), case_spot()] + case_linked_metal_from_above() + case_furnace()
    failed = 0
    print('CALIBRATION (Blender %s)' % bpy.app.version_string)
    for name, measured, expected in results:
        error = abs(measured - expected) / expected
        status = 'ok  ' if error <= TOLERANCE else 'FAIL'
        failed += status == 'FAIL'
        print('CALIBRATION %s %-62s measured %.5f expected %.5f (%.2f%% off)' % (status, name, measured, expected, error * 100))
    sys.stdout.flush()
    os._exit(1 if failed else 0)


main()
