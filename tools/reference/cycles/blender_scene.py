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
  * Pixel filter Blackman-Harris, 1.5 px wide (Cycles' default) for references
    (T1.13, 2026-11-05): anti-aliased edges at HD, so the viewer's wipe can be
    judged by eye. Cycles importance-samples the filter (no negative lobes), so
    an image's mean is unchanged; only edges are softened, by about half a
    pixel. References before T1.13 used a 1 px box; `--pixel-filter box`
    renders that way again. The calibration scenes keep the box (flat areas,
    where the filter makes no difference).
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

# Axis change as a 3x3 matrix: manifest (x, y, z) -> Blender (x, -z, y).
AXIS_CHANGE = mathutils.Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))


# ---------------------------------------------------------------------------
# Scene and render settings
# ---------------------------------------------------------------------------

def reset_scene():
    """An empty scene with factory settings, so nothing from the user profile leaks in."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


# Cycles' pixel filters by the names the command line uses.
PIXEL_FILTERS = {'blackman-harris': 'BLACKMAN_HARRIS', 'gaussian': 'GAUSSIAN', 'box': 'BOX'}


def configure_cycles(scene, width, height, samples, seed, device, max_bounces=MAX_BOUNCES, pixel_filter='box', filter_width=1.0):
    """
    Render settings (see the file comment). Returns a readable description of
    the device that renders, e.g. 'OPTIX: NVIDIA GeForce RTX 3060 Laptop GPU'.
    """
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
    cycles.pixel_filter_type = PIXEL_FILTERS[pixel_filter]
    cycles.filter_width = filter_width

    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    if device == 'cpu':
        cycles.device = 'CPU'
        device_description = f'CPU ({os.cpu_count()} logical cores)'
    else:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        compute_type_used = None
        for compute_type in ('OPTIX', 'CUDA'):
            try:
                prefs.compute_device_type = compute_type
                prefs.get_devices()
                compute_type_used = compute_type
                break
            except TypeError:
                continue
        for gpu in prefs.devices:
            gpu.use = gpu.type in ('OPTIX', 'CUDA') and 'NVIDIA' in gpu.name.upper()
        used = [gpu.name for gpu in prefs.devices if gpu.use and gpu.type == compute_type_used]
        if not used:
            raise SystemExit('--device gpu: Blender found no NVIDIA OptiX or CUDA device')
        cycles.device = 'GPU'
        device_description = f"{compute_type_used}: {', '.join(used)}"

    view_layer = scene.view_layers[0]
    # Passes used to build the indirect-only image (see add_compositor_outputs).
    view_layer.use_pass_diffuse_direct = True
    view_layer.use_pass_diffuse_indirect = True
    view_layer.use_pass_diffuse_color = True
    return device_description


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


# Principled BSDF inputs that must hold these values for the surface to be pure
# Lambert (Blender 4.5 names). Everything else that shapes the look is kept:
# Base Color (and its texture), Alpha (cutouts), Emission Color and Strength,
# and Normal (a normal map bends the shading normal; it is geometry, not a lobe).
#
#   Metallic 0              a metal has no diffuse at all
#   Roughness 1             no glossy lobe is left once specular is 0; 1 is the safe value
#   Diffuse Roughness 0     above 0, Blender 4.3+ shades with Oren-Nayar, not Lambert
#   Subsurface Weight 0     subsurface replaces part of the diffuse
#   Specular IOR Level 0    makes the dielectric F0 zero: no glossy lobe, and no energy
#                           taken from the diffuse for it (0.5 means "F0 from the IOR")
#   Specular Tint white     no effect at level 0; neutral so the check stays simple
#   Anisotropic, Transmission Weight, Coat Weight, Sheen Weight,
#   Thin Film Thickness 0   the remaining lobes, all off
#
# Blender's glTF importer LINKS some of these to textures: Metallic and Roughness
# go to the metallicRoughness texture (24 of the 28 New Sponza materials), and the
# KHR_materials_* extensions link Specular, Sheen, Coat, Transmission and so on. A
# linked socket ignores its default_value, so each one is unlinked first, then set.
NEUTRAL_PRINCIPLED_INPUTS = {
    'Metallic': 0.0,
    'Roughness': 1.0,
    'Diffuse Roughness': 0.0,
    'Subsurface Weight': 0.0,
    'Specular IOR Level': 0.0,
    'Specular Tint': (1.0, 1.0, 1.0, 1.0),
    'Anisotropic': 0.0,
    'Transmission Weight': 0.0,
    'Coat Weight': 0.0,
    'Sheen Weight': 0.0,
    'Thin Film Thickness': 0.0,
}

# Shader nodes that add a non-diffuse lobe by themselves. The glTF importer does
# not create any of them, so finding one means the scene is not what we expect.
NON_DIFFUSE_SHADER_NODES = {
    'BSDF_ANISOTROPIC',
    'BSDF_GLASS',
    'BSDF_GLOSSY',
    'BSDF_HAIR',
    'BSDF_HAIR_PRINCIPLED',
    'BSDF_METALLIC',
    'BSDF_REFRACTION',
    'BSDF_SHEEN',
    'BSDF_TOON',
    'BSDF_TRANSLUCENT',
    'BSDF_VELVET',
    'SUBSURFACE_SCATTERING',
}


def make_materials_diffuse_only():
    """
    Turns every imported material into a Lambertian surface that keeps its
    base colour (and texture), emission and alpha.

    The platform models diffuse albedo only (see three-scene.ts: roughness 1,
    metalness 0), and GI references for DDGI should isolate diffuse transport.
    Blender's glTF importer builds a Principled BSDF with specular F0 = 4 percent
    even at roughness 1, and links Metallic and Roughness to textures. Every
    input in NEUTRAL_PRINCIPLED_INPUTS is unlinked and set, so the surface is
    truly Lambertian. Returns (Principled BSDFs reset, texture links removed).
    """
    changed = 0
    unlinked = 0
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        tree = material.node_tree
        for node in tree.nodes:
            if node.type != 'BSDF_PRINCIPLED':
                continue
            for name, value in NEUTRAL_PRINCIPLED_INPUTS.items():
                unlinked += force_input(tree, node, name, value)
            changed += 1
    return changed, unlinked


def apply_textured_material_model():
    """
    Makes every imported material what the platform's GI sees in M4
    (docs/design/m4-materials.md, sections 2 and 3) while keeping the glTF's
    own textures. Run BEFORE make_materials_diffuse_only, which unlinks
    Metallic and Transmission and would lose what they said.

    On every Principled BSDF:
      Transmission    > 0 or linked: the surface lets light through (glass). GI
                      takes glass as `blocksLight: false`: no shadow, no bounce,
                      no tint. Alpha 0 makes Cycles pass straight through it too.
      Metallic        > 0 or linked: folded into the base colour as
                      base x (1 - metallic), which is three's and glTF's diffuse
                      colour of a metal and what the platform's hits read
                      ("albedo x (1 - metalness)"). make_materials_diffuse_only
                      then sets Metallic to 0, so no conductor lobe is left.
    Left as the importer made them:
      Base Color      the base colour texture times the factor (and vertex colours)
      Alpha           MASK: the importer's 1 - (alpha < cutoff) nodes, a hard
                      cutout. BLEND: the alpha itself, which Cycles treats as
                      the chance a ray passes (stochastic transparency)
      Emission        colour, texture and strength (KHR_materials_emissive_strength)
      Normal          normal maps stay, as in the other textured references

    Returns a summary for the reference's json: how many materials each rule
    changed or found, with their names (at most 40 listed per rule).
    """
    see_through = []
    metallic_folded = []
    cutout_or_blended = []
    emission_textured = []
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        tree = material.node_tree
        for node in [candidate for candidate in tree.nodes if candidate.type == 'BSDF_PRINCIPLED']:
            if input_is_on(node.inputs['Transmission Weight']):
                force_input(tree, node, 'Alpha', 0.0)
                see_through.append(material.name)
                continue
            if input_is_on(node.inputs['Metallic']):
                fold_metallic_into_base_color(tree, node)
                metallic_folded.append(material.name)
            alpha = node.inputs['Alpha']
            if alpha.is_linked or alpha.default_value < 1.0:
                cutout_or_blended.append(material.name)
            if node.inputs['Emission Color'].is_linked:
                emission_textured.append(material.name)

    def listing(names):
        return {'count': len(names), 'listed': sorted(names)[:40]}

    return {
        'seeThroughFromTransmission': listing(see_through),
        'metallicFoldedIntoBaseColor': listing(metallic_folded),
        'alphaCutoutOrBlended': listing(cutout_or_blended),
        'emissionTextured': listing(emission_textured),
    }


def input_is_on(socket):
    """A float input that is linked, or set above 0."""
    return socket.is_linked or socket.default_value > 0.0


def fold_metallic_into_base_color(tree, node):
    """
    Base Color <- Base Color x (1 - Metallic), built from whatever feeds each
    input now (a texture chain or a constant). Vector Math SCALE multiplies the
    colour's three channels by one factor; the colour's alpha is separate in
    Blender (the Alpha input), so cutouts are untouched.
    """
    base = node.inputs['Base Color']
    metallic = node.inputs['Metallic']

    one_minus_metallic = tree.nodes.new('ShaderNodeMath')
    one_minus_metallic.label = 'GI: 1 - metallic'
    one_minus_metallic.operation = 'SUBTRACT'
    one_minus_metallic.use_clamp = True
    one_minus_metallic.inputs[0].default_value = 1.0
    if metallic.is_linked:
        tree.links.new(metallic.links[0].from_socket, one_minus_metallic.inputs[1])
    else:
        one_minus_metallic.inputs[1].default_value = metallic.default_value

    diffuse_colour = tree.nodes.new('ShaderNodeVectorMath')
    diffuse_colour.label = 'GI: base x (1 - metallic)'
    diffuse_colour.operation = 'SCALE'
    if base.is_linked:
        tree.links.new(base.links[0].from_socket, diffuse_colour.inputs['Vector'])
    else:
        diffuse_colour.inputs['Vector'].default_value = tuple(base.default_value[:3])
    tree.links.new(one_minus_metallic.outputs['Value'], diffuse_colour.inputs['Scale'])
    tree.links.new(diffuse_colour.outputs['Vector'], base)


def force_input(tree, node, name, value):
    """Removes every link into input `name`, then sets its value. Returns the number of links removed."""
    socket = node.inputs.get(name)
    if socket is None:
        return 0
    links = list(socket.links)
    for link in links:
        tree.links.remove(link)
    socket.default_value = value
    return len(links)


def find_non_diffuse_surfaces():
    """
    Every reason a material in the scene is not pure Lambert, as readable lines
    ("'Material' / 'Principled BSDF': Metallic is linked"). Empty when the scene
    is clean. render.py runs it after make_materials_diffuse_only and refuses to
    render if it finds anything.
    """
    problems = []
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            where = f'{material.name!r} / {node.name!r}'
            if node.type in NON_DIFFUSE_SHADER_NODES:
                problems.append(f'{where}: a {node.type} shader node')
                continue
            if node.type != 'BSDF_PRINCIPLED':
                continue
            for name, neutral in NEUTRAL_PRINCIPLED_INPUTS.items():
                socket = node.inputs.get(name)
                if socket is None:
                    continue
                if socket.is_linked:
                    problems.append(f'{where}: {name} is linked')
                elif not values_match(socket.default_value, neutral):
                    problems.append(f'{where}: {name} is {format_value(socket.default_value)}, needs {format_value(neutral)}')
    return problems


def require_diffuse_only():
    """Stops the run, listing every culprit, if any material is not pure Lambert."""
    problems = find_non_diffuse_surfaces()
    if problems:
        listing = '\n  '.join(problems)
        raise SystemExit(f'{len(problems)} material input(s) are not diffuse-only; refusing to render a biased reference:\n  {listing}')


def values_match(actual, wanted, tolerance=1e-6):
    if isinstance(wanted, tuple):
        return all(abs(a - w) <= tolerance for a, w in zip(actual, wanted))
    return abs(actual - wanted) <= tolerance


def format_value(value):
    if isinstance(value, (int, float)):
        return f'{value:g}'
    return '(' + ', '.join(f'{component:g}' for component in value) + ')'


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


def sky_file_path(scene_id):
    """The shared sky file of a scene with a procedural environment (tools/sky/generate.ts)."""
    return os.path.join(REPO_ROOT, 'scenes', 'sky', f'{scene_id}.exr')


def add_environment(scene, environment, sky_file=None):
    """
    'none' is a black world. 'color' is a uniform radiance. 'procedural' is the
    shared sky file (`sky_file`, see sky_file_path): the same equirectangular
    half-float EXR the viewer and the platform read, with no sun disc (the sun
    is the directional light) and calibrated brightness, so `intensity` is
    already in it and the Background strength is 1.

    Orientation: Cycles' equirectangular lookup puts image u = 0.5 at Blender
    +X and u = 0.25 at Blender +Y, which is manifest -Z after the axis mapping
    (x, y, z) -> (x, -z, y); that is the layout the file is written in
    (scenes/src/sky/equirect.ts), so no rotation is applied.
    `calibrate.py` checks this by rendering known directions.
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
        if not sky_file or not os.path.exists(sky_file):
            raise SystemExit(f'sky file {sky_file} is missing; run `node tools/sky/generate.ts`')
        sky = nodes.new('ShaderNodeTexEnvironment')
        sky.image = bpy.data.images.load(sky_file, check_existing=True)
        # Float data as stored: no view transform or colour conversion on the way in.
        sky.image.colorspace_settings.name = 'Linear Rec.709'
        sky.projection = 'EQUIRECTANGULAR'
        sky.interpolation = 'Linear'
        links.new(sky.outputs['Color'], background.inputs['Color'])
        background.inputs['Strength'].default_value = 1.0
    else:
        raise ValueError(f'unknown environment type {kind!r}')


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

# The reference camera's clip distances (metres). Recorded with every render
# (render.py `shot`); nothing in the scenes is nearer or farther.
CAMERA_NEAR = 0.01
CAMERA_FAR = 2000.0


def add_camera(scene, position, target, fov_y_deg):
    """
    An explicit basis, not look-at constraints: forward = target - position,
    right = forward x up, up = right x forward; Blender cameras look down their
    local -Z with +Y up, so the matrix columns are (right, up, -forward). A
    camera looking straight up or down gets three.js's fallback (see below).
    """
    eye = mathutils.Vector(to_blender_space(position))
    forward = (mathutils.Vector(to_blender_space(target)) - eye).normalized()
    world_up = mathutils.Vector((0.0, 0.0, 1.0))
    if forward.cross(world_up).length < 1e-6:
        # Looking straight up or down: forward x up is zero and has no direction.
        # three's Matrix4.lookAt (camera.up = +Y) then nudges its backward axis by
        # 1e-4 along three's +Z, which is forward nudged along three's -Z, i.e.
        # Blender +Y. Doing the same frames the same picture as the viewer and the
        # path tracer: image right is +X; image up is Blender +Y looking down, -Y up.
        forward = (forward + mathutils.Vector((0.0, 1e-4, 0.0))).normalized()
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
    data.clip_start = CAMERA_NEAR
    data.clip_end = CAMERA_FAR
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
