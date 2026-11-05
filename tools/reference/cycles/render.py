"""
Renders a manifest scene at one camera bookmark with Cycles, to linear float EXRs.

Run through the Node wrapper (it finds Blender 4.5 and takes the GPU lock when
needed):

    node tools/reference/cycles/run.ts --scene cornell-box --camera 1 --samples 1024

or directly:

    blender --background --factory-startup --python tools/reference/cycles/render.py -- \
        --scene cornell-box --camera 1 --samples 1024 --width 512 --height 512

Writes to bench/references/<scene>/<camera name>/ :
    cycles-full.exr       everything the camera sees
    cycles-indirect.exr   only light that bounced at least once before the visible surface
    cycles.json           every setting, the sample count and the render time
See blender_scene.py for the settings and units.py for the unit conversions.

--materials-json <file> renders the scene "untextured": every material gets the
constant GI albedo and emission from a GI material table (written by
tools/gpu/pages/ddgi-sponza-accuracy/run.ts, as the three adapter sent them to
the platform), and the files are named cycles-untextured-*. See
apply_gi_material_table. It also writes the parts of DDGI's quantity (see
render_untextured):
    cycles-untextured-direct.exr          Diffuse Direct x Diffuse Colour
    cycles-untextured-lights-direct.exr   the same from a second render with only the
                                          manifest's lights (no sky, no emission, no bounces)
and run.ts adds them up into cycles-untextured-gi.exr.

The table also says which model parts were loaded (`parts`; only those are
imported) and which materials' meshes the three adapter left out of GI
because all their materials let light through (`leftOutMeshes`; their faces
are removed, see remove_left_out_faces).
"""

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time

import bmesh
import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import blender_scene as bs  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))


def parse_arguments():
    # Blender passes everything after a lone "--" to the script.
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', required=True, help='manifest id, e.g. cornell-box')
    parser.add_argument('--camera', default='1', help='camera bookmark: name, or 1-based number (default 1, the first)')
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--out-dir', default=None, help='default bench/references/<scene>/<camera name>')
    parser.add_argument('--name-suffix', default='', help='added to file names, e.g. -seed1')
    parser.add_argument('--threads', type=int, default=0, help='CPU threads; 0 means all')
    parser.add_argument(
        '--materials-json',
        default=None,
        help='GI material table: give every material its constant GI albedo and emission (no textures); '
        'files are named cycles-untextured-*',
    )
    return parser.parse_args(argv)


def pick_camera(manifest, wanted):
    cameras = manifest['cameras']
    if wanted.isdigit():
        index = int(wanted) - 1
        if not 0 <= index < len(cameras):
            raise SystemExit(f'camera {wanted} is out of range; the scene has {len(cameras)} bookmarks')
        return cameras[index]
    for camera in cameras:
        if camera['name'] == wanted:
            return camera
    raise SystemExit(f'no camera named {wanted!r}; known: {[c["name"] for c in cameras]}')


def load_geometry(manifest, model_parts=None):
    """
    Procedural parts come as one GLB per scene (pnpm scenes:export); model parts are downloaded files.
    `model_parts` (asset ids), when given, keeps only those model parts (the GI material table's `parts`).
    Returns {model part asset id: the mesh objects it imported}, for the GI material table.
    """
    procedural_done = False
    imported_by_part = {}
    for part in manifest['parts']:
        if model_parts is not None and part['kind'] == 'model' and part['asset'] not in model_parts:
            continue
        if part['kind'] == 'procedural':
            if procedural_done:
                continue
            procedural_done = True
            path = os.path.join(REPO_ROOT, 'assets', 'generated', f"{manifest['id']}.glb")
            if not os.path.exists(path):
                raise SystemExit(f'{path} is missing; run `pnpm scenes:export`')
            bs.import_gltf(path)
        else:
            path = os.path.join(REPO_ROOT, 'assets', 'downloads', part['asset'], part['path'])
            if not os.path.exists(path):
                raise SystemExit(f'{path} is missing; run `pnpm assets:fetch --only {part["asset"]}`')
            imported = bs.import_gltf(path, part['transform'])
            imported_by_part[part['asset']] = [obj for obj in imported if obj.type == 'MESH']
    return imported_by_part


# Blender adds ".001", ".002", ... when a material name is already taken (two
# parts can both have a "metal_door"); the table has the glTF name.
BLENDER_DUPLICATE_SUFFIX = re.compile(r'\.\d{3}$')


def apply_gi_material_table(table, imported_by_part):
    """
    Gives every imported material the constant GI albedo and emission from the
    table, so Cycles renders what the platform's GI sees in M1: one colour per
    material, no textures (DECISIONS 2026-11-05, T1.7: the adapter folds each
    texture's average into the colour; the core samples no textures until M4).

    On every Principled BSDF this unlinks and sets:
      Base Color      the table's albedo (texture, vertex colour and factor nodes cut off)
      Emission        colour = emission / its peak, strength = the peak (strength is radiance, units.py)
      Alpha           1: the M1 tracer treats every surface as opaque (alpha cutouts and
                      see-through materials arrive with M4), so Cycles does too; the meshes
                      the adapter leaves out of GI instead are removed (remove_left_out_faces)
      Normal          cut off: the platform shades with interpolated vertex normals
    Run after make_materials_diffuse_only; require_untextured and
    require_diffuse_only check the result. Materials are matched by model part
    and glTF name. A used material with no row, or a row with no material, stops
    the run. Returns the number of materials set.
    """
    rows = {(row['part'], row['name']): row for row in table['materials']}
    matched_rows = set()
    changed = 0
    for part, objects in imported_by_part.items():
        materials = {slot.material for obj in objects for slot in obj.material_slots if slot.material is not None}
        for material in materials:
            name = BLENDER_DUPLICATE_SUFFIX.sub('', material.name)
            row = rows.get((part, name))
            if row is None:
                raise SystemExit(f'material {material.name!r} of part {part} has no row in the GI material table')
            matched_rows.add((part, name))
            set_constant_material(material, row)
            changed += 1
    unmatched = sorted(set(rows) - matched_rows)
    if unmatched:
        raise SystemExit(f'GI material table rows with no material in the Blender scene: {unmatched}')
    return changed


def remove_left_out_faces(table, imported_by_part):
    """
    Removes the faces of materials whose meshes the three adapter left out of
    GI (the table's `leftOutMeshes`): a mesh whose materials all let light
    through is sent with contributesToGI false, since the M1 tracer treats
    every surface as opaque (packages/ddgi-three/src/scene-sync.ts). three
    makes one mesh per glTF primitive and a primitive has one material, so
    removing a material's faces removes exactly those meshes. A material with
    only some of its meshes left out can't be matched that way, and stops the
    run. Returns {part/name: faces removed}.
    """
    left_out = set()
    for row in table['materials']:
        count = row.get('leftOutMeshes', 0)
        if count == 0:
            continue
        if count != row['meshes']:
            raise SystemExit(
                f"material {row['name']!r} of part {row['part']}: {count} of its {row['meshes']} meshes were left out of GI; "
                'render.py can only remove all of a material\'s faces'
            )
        left_out.add((row['part'], row['name']))
    removed = {}
    done_meshes = set()
    for part, objects in imported_by_part.items():
        for obj in objects:
            mesh = obj.data
            # Objects can share mesh data; remove its faces once.
            if mesh.name in done_meshes:
                continue
            done_meshes.add(mesh.name)
            slots = [BLENDER_DUPLICATE_SUFFIX.sub('', slot.material.name) if slot.material else None for slot in obj.material_slots]
            removed_slots = {index for index, name in enumerate(slots) if (part, name) in left_out}
            if not removed_slots:
                continue
            editable = bmesh.new()
            editable.from_mesh(mesh)
            faces = [face for face in editable.faces if face.material_index in removed_slots]
            for index in removed_slots:
                key = f'{part}/{slots[index]}'
                removed[key] = removed.get(key, 0) + sum(1 for face in faces if face.material_index == index)
            bmesh.ops.delete(editable, geom=faces, context='FACES')
            editable.to_mesh(mesh)
            editable.free()
            mesh.update()
    missing = sorted(f'{part}/{name}' for part, name in left_out if f'{part}/{name}' not in removed)
    if missing:
        raise SystemExit(f'materials left out of GI with no faces in the Blender scene: {missing}')
    return removed


def set_constant_material(material, row):
    """One table row onto one material's Principled BSDF (see apply_gi_material_table)."""
    if not material.use_nodes:
        raise SystemExit(f'material {material.name!r} has no node tree')
    tree = material.node_tree
    principled = [node for node in tree.nodes if node.type == 'BSDF_PRINCIPLED']
    if len(principled) != 1:
        raise SystemExit(f'material {material.name!r} has {len(principled)} Principled BSDFs; expected 1')
    node = principled[0]
    albedo = row['albedo']
    emission = row['emission']
    peak = max(emission)
    bs.force_input(tree, node, 'Base Color', (albedo[0], albedo[1], albedo[2], 1.0))
    emission_color = (emission[0] / peak, emission[1] / peak, emission[2] / peak, 1.0) if peak > 0 else (0.0, 0.0, 0.0, 1.0)
    bs.force_input(tree, node, 'Emission Color', emission_color)
    bs.force_input(tree, node, 'Emission Strength', peak)
    bs.force_input(tree, node, 'Alpha', 1.0)
    for link in list(node.inputs['Normal'].links):
        tree.links.remove(link)


# Inputs that must not be linked in an untextured reference (set_constant_material).
UNTEXTURED_INPUTS = ('Base Color', 'Emission Color', 'Emission Strength', 'Alpha', 'Normal')


def require_untextured():
    """Stops the run if any Principled BSDF still has a linked colour, emission, alpha or normal input."""
    problems = []
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type != 'BSDF_PRINCIPLED':
                continue
            for name in UNTEXTURED_INPUTS:
                socket = node.inputs.get(name)
                if socket is not None and socket.is_linked:
                    problems.append(f'{material.name!r} / {node.name!r}: {name} is linked')
    if problems:
        listing = '\n  '.join(problems)
        raise SystemExit(f'{len(problems)} input(s) still textured; refusing to render an untextured reference:\n  {listing}')


# What DDGI's test view calls "indirect" (albedo / pi x the field), in Cycles' terms.
GI_DEFINITION = (
    'everything diffuse except the direct light of the manifest lights: (Diffuse Indirect + Diffuse Direct) x Diffuse Colour '
    'minus a lights-only direct render, so sky light and emitted light reaching the surface directly count, as in the DDGI field'
)


def multiply_passes(tree, layers, first, second):
    """A compositor node giving pass `first` x pass `second`."""
    multiply = tree.nodes.new('CompositorNodeMixRGB')
    multiply.blend_type = 'MULTIPLY'
    multiply.inputs[0].default_value = 1.0
    tree.links.new(layers.outputs[first], multiply.inputs[1])
    tree.links.new(layers.outputs[second], multiply.inputs[2])
    return multiply


def take_output(directory, name):
    """The File Output node appends the frame number ("name0001.exr"); strips it, as blender_scene.collect_outputs does."""
    produced = os.path.join(directory, f'{name}0001.exr')
    target = os.path.join(directory, f'{name}.exr')
    if os.path.exists(target):
        os.remove(target)
    os.replace(produced, target)
    return target


def render_untextured(scene, directory, prefix, suffix):
    """
    The untextured render, plus what DDGI's quantity needs (GI_DEFINITION).

    Cycles' Diffuse Indirect pass leaves out light that reaches the visible
    surface without a bounce, and that includes the sky and emissive surfaces
    seen straight from it. DDGI's field holds both: a probe ray that misses
    sees the sky, and one that hits an emitter sees its emission. Only the
    manifest's lights (sun, point, spot) are added analytically, outside the
    field. So, with Diffuse Colour as the albedo:
        DDGI's "indirect" = Indirect + Direct - (Direct from the lights alone)
    The lights' direct part comes from a second render with the sky and every
    emission at 0 and no bounces, with the same seed and samples, so its pixel
    jitter matches and edges cancel.

    Returns (seconds for both renders, the files).
    """
    full_name = f'{prefix}-full{suffix}'
    indirect_name = f'{prefix}-indirect{suffix}'
    direct_name = f'{prefix}-direct{suffix}'
    lights_direct_name = f'{prefix}-lights-direct{suffix}'
    os.makedirs(directory, exist_ok=True)

    # 1. The full scene: blender_scene's two outputs plus Diffuse Direct x Diffuse Colour.
    bs.add_compositor_outputs(scene, directory, full_name, indirect_name)
    tree = scene.node_tree
    layers = next(node for node in tree.nodes if node.type == 'R_LAYERS')
    output = next(node for node in tree.nodes if node.type == 'OUTPUT_FILE')
    output.file_slots.new(direct_name)
    tree.links.new(multiply_passes(tree, layers, 'DiffDir', 'DiffCol').outputs['Image'], output.inputs[2])
    started = time.time()
    bpy.ops.render.render(write_still=False)
    files = bs.collect_outputs(directory, full_name, indirect_name)
    files.append(take_output(directory, direct_name))

    # 2. The lights alone, direct only: no sky, no emission, no bounces.
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.0
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                node.inputs['Emission Strength'].default_value = 0.0
    scene.cycles.max_bounces = 0
    scene.cycles.diffuse_bounces = 0
    tree.nodes.clear()
    layers = tree.nodes.new('CompositorNodeRLayers')
    composite = tree.nodes.new('CompositorNodeComposite')
    tree.links.new(layers.outputs['Image'], composite.inputs['Image'])
    output = tree.nodes.new('CompositorNodeOutputFile')
    output.base_path = directory
    output.format.file_format = 'OPEN_EXR'
    output.format.color_depth = '32'
    output.format.color_mode = 'RGB'
    output.format.exr_codec = 'ZIP'
    output.file_slots[0].path = lights_direct_name
    tree.links.new(multiply_passes(tree, layers, 'DiffDir', 'DiffCol').outputs['Image'], output.inputs[0])
    bpy.ops.render.render(write_still=False)
    files.append(take_output(directory, lights_direct_name))
    return time.time() - started, files


def main():
    args = parse_arguments()
    manifest = bs.read_json(os.path.join(REPO_ROOT, 'scenes', 'manifests', f'{args.scene}.json'))
    camera = pick_camera(manifest, args.camera)

    out_dir = os.path.abspath(args.out_dir or os.path.join(REPO_ROOT, 'bench', 'references', manifest['id'], camera['name']))
    gi_table = None
    gi_table_info = None
    if args.materials_json:
        with open(args.materials_json, 'rb') as handle:
            table_bytes = handle.read()
        gi_table = json.loads(table_bytes.decode('utf-8'))
        # The hash lets the DDGI side check the reference was rendered from the table it compares with.
        gi_table_info = {
            'path': os.path.relpath(os.path.abspath(args.materials_json), REPO_ROOT).replace(os.sep, '/'),
            'sha256': hashlib.sha256(table_bytes).hexdigest(),
            'materials': len(gi_table['materials']),
        }
    prefix = 'cycles-untextured' if gi_table else 'cycles'
    full_name = f'{prefix}-full{args.name_suffix}'
    indirect_name = f'{prefix}-indirect{args.name_suffix}'

    scene = bs.reset_scene()
    bs.configure_cycles(scene, args.width, args.height, args.samples, args.seed, args.device)
    if args.threads:
        scene.render.threads_mode = 'FIXED'
        scene.render.threads = args.threads

    load_started = time.time()
    model_parts = gi_table.get('parts') if gi_table else None
    imported_by_part = load_geometry(manifest, model_parts)
    materials_changed, texture_links_removed = bs.make_materials_diffuse_only()
    gi_materials_set = 0
    left_out_faces = None
    if gi_table:
        gi_materials_set = apply_gi_material_table(gi_table, imported_by_part)
        require_untextured()
        left_out_faces = remove_left_out_faces(gi_table, imported_by_part)
    bs.require_diffuse_only()
    bs.add_manifest_lights(scene, manifest['lights'])
    bs.add_environment(scene, manifest['environment'], bs.sky_file_path(manifest['id']))
    bs.add_camera(scene, camera['position'], camera['target'], camera['fovYDeg'])
    load_seconds = time.time() - load_started

    triangles = sum(len(obj.data.polygons) for obj in bpy.data.objects if obj.type == 'MESH')
    # Record what the image was rendered from, so a reference made before a
    # scene changed (an added part, a new sky) can't be mistaken for a new one.
    sky_path = bs.sky_file_path(manifest['id'])
    sky_file = None
    if manifest['environment'].get('type') == 'procedural' and os.path.exists(sky_path):
        with open(sky_path, 'rb') as handle:
            sky_file = {'path': os.path.relpath(sky_path, bs.REPO_ROOT).replace(os.sep, '/'), 'sha256': hashlib.sha256(handle.read()).hexdigest()}
    if gi_table:
        render_seconds, files = render_untextured(scene, out_dir, prefix, args.name_suffix)
    else:
        render_seconds, files = bs.render_to_files(scene, out_dir, full_name, indirect_name)

    meta = {
        'tool': 'tools/reference/cycles/render.py',
        'blender': bpy.app.version_string,
        'scene': manifest['id'],
        'parts': [part for part in manifest['parts'] if model_parts is None or part['kind'] != 'model' or part['asset'] in model_parts],
        'leftOutOfGiFaces': left_out_faces,
        'environment': manifest['environment'],
        'skyFile': sky_file,
        'camera': camera,
        'width': args.width,
        'height': args.height,
        'samplesPerPixel': args.samples,
        'seed': args.seed,
        'device': args.device,
        'threads': scene.render.threads,
        'maxBounces': bs.MAX_BOUNCES,
        'denoising': False,
        'sampleClamp': 0,
        'lightSamplingThreshold': 0,
        'pixelFilter': 'BOX 1.0',
        'materials': f'diffuse only ({materials_changed} Principled BSDFs reset, {texture_links_removed} texture links removed, checked)'
        + (f'; untextured: {gi_materials_set} materials set to the GI table constants, alpha 1, no normal maps (checked)' if gi_table else ''),
        'giMaterials': gi_table_info,
        'indirectDefinition': 'Diffuse Indirect x Diffuse Colour (light that bounced at least once before the visible surface)',
        'loadSeconds': round(load_seconds, 2),
        'renderSeconds': round(render_seconds, 2),
        'triangles': triangles,
        'files': [os.path.basename(path) for path in files],
        'outDir': out_dir,
        'giComposite': (
            {
                'output': f'{prefix}-gi{args.name_suffix}.exr',
                'add': [f'{prefix}-indirect{args.name_suffix}.exr', f'{prefix}-direct{args.name_suffix}.exr'],
                'subtract': f'{prefix}-lights-direct{args.name_suffix}.exr',
                'definition': GI_DEFINITION,
            }
            if gi_table
            else None
        ),
        'host': platform.node(),
        'finishedAt': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    meta_path = os.path.join(out_dir, f'{prefix}{args.name_suffix}.json')
    with open(meta_path, 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2)
    print('REFERENCE_RESULT ' + json.dumps(meta))
    # Blender can hang for a long time at exit tearing down add-ons; everything is written.
    sys.stdout.flush()
    os._exit(0)


main()
