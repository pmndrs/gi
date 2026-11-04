"""
Builds a Cycles scene from a manifest: geometry, lights, environment, camera,
render settings and the compositor outputs. Used by render.py (the real
references) and calibrate.py (the known-answer checks), so both go through the
same code.

Run inside Blender 4.5 only (`import bpy`). Pinned to 4.5: the compositor and
glTF importer APIs differ in 5.x.

Settings that matter for a reference (and why):

  * Samples fixed, adaptive sampling OFF, denoising OFF: the EXR is the raw
    Monte Carlo estimate, so its noise can be measured and averaged.
  * Sample clamping OFF (direct and indirect): Blender's default indirect clamp
    of 10 darkens bright bounce light and is a bias.
  * Light sampling threshold 0: by default Cycles skips lights that would
    contribute under 1 percent, which silently removes dim sources from dark rooms.
  * Bounces: 32 total and 32 diffuse, glossy/transmission/volume 0. Albedo is at most
    0.8, so the energy left after 32 bounces is below 1e-3 of the first bounce; the
    pathtracer side uses the same count. No Russian-roulette-style albedo cap is
    applied (NVIDIA's 0.9 cap in the native harness is a harness detail).
  * Pixel filter BOX, width 1: matches a path tracer that jitters inside the
    pixel. Cycles' default Blackman-Harris at 1.5 px would blur edges.
  * Colour management: Standard view transform, look None, exposure 0, gamma 1.
    The EXRs are scene-linear and never see the view transform anyway; it is set
    so a stray PNG would not be tone-mapped either.
  * Light tree on (default). It changes which lights get sampled, not the
    expected value; with the threshold at 0 it has no bias.
"""

import json
import math
import os
import time

import bpy
import mathutils

from units import (
    directional_to_sun_strength,
    environment_to_background_strength,
    radiant_intensity_to_blender_watts,
    spot_cone_to_blender,
    to_blender_space,
)

MAX_BOUNCES = 32

# Axis change as a 3x3 matrix: manifest (x, y, z) -> Blender (x, -z, y).
AXIS_CHANGE = mathutils.Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))


# ---------------------------------------------------------------------------
# Scene and render settings
# ---------------------------------------------------------------------------

def reset_scene():
    """An empty scene with factory settings, so nothing from the user profile leaks in."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def configure_cycles(scene, width, height, samples, seed, device, max_bounces=MAX_BOUNCES):
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False

    cycles = scene.cycles
    cycles.samples = samples
    cycles.use_adaptive_sampling = False
    cycles.use_denoising = False
    cycles.seed = seed
    cycles.use_animated_seed = False
    cycles.sample_clamp_direct = 0.0
    cycles.sample_clamp_indirect = 0.0
    cycles.light_sampling_threshold = 0.0
    cycles.max_bounces = max_bounces
    cycles.diffuse_bounces = max_bounces
    cycles.glossy_bounces = 0
    cycles.transmission_bounces = 0
    cycles.volume_bounces = 0
    # Alpha-cutout leaves (Sponza ivy) are transparent bounces, not diffuse ones.
    cycles.transparent_max_bounces = max_bounces
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.blur_glossy = 0.0
    cycles.pixel_filter_type = 'BOX'
    cycles.filter_width = 1.0

    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    if device == 'cpu':
        cycles.device = 'CPU'
    else:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for compute_type in ('OPTIX', 'CUDA'):
            try:
                prefs.compute_device_type = compute_type
                prefs.get_devices()
                break
            except TypeError:
                continue
        for gpu in prefs.devices:
            gpu.use = gpu.type in ('OPTIX', 'CUDA') and 'NVIDIA' in gpu.name.upper()
        cycles.device = 'GPU'

    view_layer = scene.view_layers[0]
    # Passes used to build the indirect-only image (see add_compositor_outputs).
    view_layer.use_pass_diffuse_direct = True
    view_layer.use_pass_diffuse_indirect = True
    view_layer.use_pass_diffuse_color = True


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def import_gltf(path, transform=None):
    """
    Imports a glTF/GLB. `transform` is a manifest model transform
    {position, rotationDeg, scale} in manifest (Y-up) space, applied through a
    parent empty. Lights and cameras in the file are deleted: lights come from
    the manifest through units.py, never from the file's own conversion.
    """
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    imported = [obj for obj in bpy.data.objects if obj not in before]

    kept = [obj for obj in imported if obj.type not in ('LIGHT', 'CAMERA')]
    for obj in imported:
        if obj.type in ('LIGHT', 'CAMERA'):
            bpy.data.objects.remove(obj, do_unlink=True)
    imported = kept

    if transform is not None:
        parent = bpy.data.objects.new('model-transform', None)
        bpy.context.scene.collection.objects.link(parent)
        parent.matrix_world = manifest_transform_to_blender(transform)
        for obj in imported:
            if obj.parent is None or obj.parent not in imported:
                obj.parent = parent
    return imported


def manifest_transform_to_blender(transform):
    """T (manifest space) -> C * T * C^-1 (Blender space), where C is the axis change."""
    # three.js Euler 'XYZ' is R = Rx * Ry * Rz. Blender's 'XYZ' means the opposite
    # order (Rz * Ry * Rx), so the matrix is composed by hand.
    angle_x, angle_y, angle_z = [math.radians(a) for a in transform['rotationDeg']]
    rotation = (
        mathutils.Matrix.Rotation(angle_x, 3, 'X')
        @ mathutils.Matrix.Rotation(angle_y, 3, 'Y')
        @ mathutils.Matrix.Rotation(angle_z, 3, 'Z')
    )
    local = rotation @ (mathutils.Matrix.Identity(3) * transform['scale'])
    converted = AXIS_CHANGE @ local @ AXIS_CHANGE.inverted()
    matrix = converted.to_4x4()
    matrix.translation = mathutils.Vector(to_blender_space(transform['position']))
    return matrix


def make_materials_diffuse_only():
    """
    Turns every imported material into a Lambertian surface that keeps its
    base colour (and texture), emission and alpha.

    The platform models diffuse albedo only (see three-scene.ts: roughness 1,
    metalness 0), and GI references for DDGI should isolate diffuse transport.
    Blender's glTF importer builds a Principled BSDF with specular F0 = 4 percent
    even at roughness 1; that is zeroed here so the surface is truly Lambertian.
    """
    changed = 0
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type != 'BSDF_PRINCIPLED':
                continue
            set_input(node, 'Metallic', 0.0)
            set_input(node, 'Roughness', 1.0)
            set_input(node, 'Specular IOR Level', 0.0)
            set_input(node, 'Transmission Weight', 0.0)
            set_input(node, 'Coat Weight', 0.0)
            set_input(node, 'Sheen Weight', 0.0)
            changed += 1
    return changed


def set_input(node, name, value):
    socket = node.inputs.get(name)
    if socket is not None and not socket.is_linked:
        socket.default_value = value


# ---------------------------------------------------------------------------
# Lights and environment
# ---------------------------------------------------------------------------

def add_manifest_lights(scene, lights):
    for index, light in enumerate(lights):
        kind = light['type']
        name = light.get('label') or f'{kind}-{index}'
        color = light['color']
        casts_shadows = light.get('castsShadows', True)

        if kind == 'directional':
            data = bpy.data.lights.new(name, 'SUN')
            data.energy = directional_to_sun_strength(light['intensity'])
            # A point-like sun: the manifest light has no angular size, and a sun
            # disc (default 0.526 degrees in Blender's UI, 0 here) would soften shadows.
            data.angle = 0.0
        elif kind == 'point':
            data = bpy.data.lights.new(name, 'POINT')
            data.energy = radiant_intensity_to_blender_watts(light['intensity'])
            data.shadow_soft_size = 0.0
        else:
            data = bpy.data.lights.new(name, 'SPOT')
            data.energy = radiant_intensity_to_blender_watts(light['intensity'])
            data.shadow_soft_size = 0.0
            size, blend = spot_cone_to_blender(math.radians(light['outerConeDeg']), math.radians(light['innerConeDeg']))
            data.spot_size = size
            data.spot_blend = blend
        data.color = color
        data.use_shadow = casts_shadows
        # `range` (the point/spot window to zero) is not applied: Cycles has no
        # equivalent and the reference should be the unwindowed physical answer.

        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        if kind in ('directional', 'spot'):
            travel = mathutils.Vector(to_blender_space(light['direction'])).normalized()
            # Blender lights shine along their local -Z.
            obj.rotation_euler = travel.to_track_quat('-Z', 'Y').to_euler()
        if kind != 'directional':
            obj.location = to_blender_space(light['position'])


def add_environment(scene, environment):
    """
    'none' is a black world. 'color' is a uniform radiance. 'procedural' is
    Cycles' Nishita sky with the sun disc off (the sun comes from the
    directional light, so it is not counted twice); it is an approximation of
    the viewer's sky, not a match, and only Sponza uses it.
    """
    world = bpy.data.worlds.new('world')
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    background = nodes['Background']

    kind = environment['type']
    if kind == 'none':
        background.inputs['Strength'].default_value = 0.0
    elif kind == 'color':
        color = environment['color']
        background.inputs['Color'].default_value = (color[0], color[1], color[2], 1.0)
        background.inputs['Strength'].default_value = environment_to_background_strength(environment.get('intensity', 1.0))
    elif kind == 'procedural':
        sky = nodes.new('ShaderNodeTexSky')
        sky.sky_type = 'NISHITA'
        sky.sun_disc = False
        toward_sun = mathutils.Vector(to_blender_space(environment['sunDirection'])).normalized()
        sky.sun_elevation = math.asin(max(-1.0, min(1.0, toward_sun.z)))
        # Nishita's rotation is measured from +Y, clockwise seen from above.
        sky.sun_rotation = math.atan2(toward_sun.x, toward_sun.y) % (2.0 * math.pi)
        links.new(sky.outputs['Color'], background.inputs['Color'])
        background.inputs['Strength'].default_value = environment_to_background_strength(environment.get('intensity', 1.0))
    else:
        raise ValueError(f'unknown environment type {kind!r}')


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def add_camera(scene, position, target, fov_y_deg):
    """
    An explicit basis, not look-at constraints: forward = target - position,
    right = forward x up, up = right x forward; Blender cameras look down their
    local -Z with +Y up, so the matrix columns are (right, up, -forward).
    """
    eye = mathutils.Vector(to_blender_space(position))
    forward = (mathutils.Vector(to_blender_space(target)) - eye).normalized()
    world_up = mathutils.Vector((0.0, 0.0, 1.0))
    right = forward.cross(world_up).normalized()
    up = right.cross(forward)

    matrix = mathutils.Matrix.Identity(4)
    for row in range(3):
        matrix[row][0] = right[row]
        matrix[row][1] = up[row]
        matrix[row][2] = -forward[row]
        matrix[row][3] = eye[row]

    data = bpy.data.cameras.new('reference-camera')
    data.sensor_fit = 'VERTICAL'
    data.angle = math.radians(fov_y_deg)
    data.clip_start = 0.01
    data.clip_end = 2000.0
    obj = bpy.data.objects.new('reference-camera', data)
    obj.matrix_world = matrix
    scene.collection.objects.link(obj)
    scene.camera = obj
    return obj


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def add_compositor_outputs(scene, directory, full_name, indirect_name):
    """
    Wires two float EXR outputs.

    full     = the Combined pass.
    indirect = Diffuse Indirect x Diffuse Colour: light that reached the visible
               surface after at least one earlier bounce. Everything else is
               "direct": emission seen by the camera, and light that arrives at
               the visible surface without a bounce (light sampling and
               emissive hits by the surface's first BSDF ray). With purely
               diffuse materials full = emission + environment + direct + indirect.

    The pathtracer defines indirect the same way: total minus the first-hit
    direct term, with direct = emission + next-event light at the first hit.
    """
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    layers = tree.nodes.new('CompositorNodeRLayers')
    multiply = tree.nodes.new('CompositorNodeMixRGB')
    multiply.blend_type = 'MULTIPLY'
    multiply.inputs[0].default_value = 1.0
    tree.links.new(layers.outputs['DiffInd'], multiply.inputs[1])
    tree.links.new(layers.outputs['DiffCol'], multiply.inputs[2])

    composite = tree.nodes.new('CompositorNodeComposite')
    tree.links.new(layers.outputs['Image'], composite.inputs['Image'])

    output = tree.nodes.new('CompositorNodeOutputFile')
    output.base_path = directory
    output.format.file_format = 'OPEN_EXR'
    output.format.color_depth = '32'
    output.format.color_mode = 'RGB'
    output.format.exr_codec = 'ZIP'
    output.file_slots[0].path = full_name
    output.file_slots.new(indirect_name)
    tree.links.new(layers.outputs['Image'], output.inputs[0])
    tree.links.new(multiply.outputs['Image'], output.inputs[1])

    scene.render.use_compositing = True
    scene.frame_set(1)


def collect_outputs(directory, full_name, indirect_name):
    """
    The File Output node appends the frame number ("name0001.exr"). Strips it.
    Returns the final paths.
    """
    final = []
    for name in (full_name, indirect_name):
        produced = os.path.join(directory, f'{name}0001.exr')
        target = os.path.join(directory, f'{name}.exr')
        if os.path.exists(target):
            os.remove(target)
        os.replace(produced, target)
        final.append(target)
    return final


def render_to_files(scene, directory, full_name, indirect_name):
    """Renders once and returns (seconds, [full path, indirect path])."""
    os.makedirs(directory, exist_ok=True)
    add_compositor_outputs(scene, directory, full_name, indirect_name)
    started = time.time()
    bpy.ops.render.render(write_still=False)
    seconds = time.time() - started
    return seconds, collect_outputs(directory, full_name, indirect_name)


def read_exr_pixels(path):
    """Pixels as a flat RGBA float list, plus (width, height). For small calibration images."""
    image = bpy.data.images.load(path)
    width, height = image.size
    pixels = list(image.pixels)
    bpy.data.images.remove(image)
    return pixels, width, height


def read_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)
