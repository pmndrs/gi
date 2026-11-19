"""
Renders a manifest scene at one camera bookmark with Cycles, to linear float EXRs.

Run through the Node wrapper (it finds Blender 4.5 and takes the GPU lock when
needed):

    node tools/reference/cycles/run.ts --scene cornell-box --camera 1 --samples 1024

or directly:

    blender --background --factory-startup --python tools/reference/cycles/render.py -- \
        --scene cornell-box --camera 1 --samples 1024 --width 512 --height 512

Writes to bench/references/<scene>/<camera name>/ :
    cycles-full.exr            everything the camera sees
    cycles-indirect.exr        only light that bounced at least once before the visible surface
    cycles-direct.exr          Diffuse Direct x Diffuse Colour
    cycles-lights-direct.exr   the same from a second render with only the manifest's
                               lights on (no sky, no emitter of any kind, no bounces)
    cycles.json                every setting, the sample count and the render time
and run.ts checks the lights-only image and adds the parts up into
cycles-gi.exr, the quantity DDGI's field holds (see render_with_gi_parts and
DECISIONS "What 'indirect light' means against Cycles in a scene with a sky").
See blender_scene.py for the settings and units.py for the unit conversions.

--materials-json <file> renders the scene "untextured": every material gets the
constant GI albedo and emission from a GI material table (written by
tools/gpu/pages/ddgi-sponza-accuracy/run.ts, as the three adapter sent them to
the platform), and the files are named cycles-untextured-* instead of cycles-*.
See apply_gi_material_table.

The table also says which model parts were loaded (`parts`; only those are
imported) and which materials' meshes the three adapter left out of GI
because all their materials let light through (`leftOutMeshes`; their faces
are removed, see remove_left_out_faces).

--textured renders the glTF's own materials as M4's GI sees them (T4.4):
base colour textures, alpha cutouts (MASK) and blending (BLEND), emission,
metals as base x (1 - metallic), glass (transmission) see-through. The files
are named cycles-textured-*. See blender_scene.apply_textured_material_model.
Without either option a render keeps the textures but treats metals as their
full base colour and glass as an opaque surface (the references before M4).

Every render records what it was made from (`sources`: each part's file and
sha256) and run.ts adds the sha256 of every image it wrote (`outputSha256`).
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time

import bmesh
import bpy
import mathutils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import blender_scene as bs  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))


def parse_arguments():
    # Blender passes everything after a lone "--" to the script.
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', required=True, help='manifest id, e.g. cornell-box')
    parser.add_argument('--camera', default='1', help='camera bookmark: name, or 1-based number (default 1, the first)')
    parser.add_argument(
        '--from-shot',
        default=None,
        help="a reference's json (cycles.json): render its stored camera and size instead of a bookmark's, to reproduce the view",
    )
    parser.add_argument('--samples', type=int, required=True)
    # 1920 x 1080 (T1.13): HD, and 16:9 like two-room's and Sponza's earlier 640 x 360,
    # so their means stay comparable; the viewer letterboxes to the stored aspect.
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', help='gpu (default) is OptiX on the NVIDIA GPU')
    parser.add_argument('--pixel-filter', choices=sorted(bs.PIXEL_FILTERS), default='blackman-harris')
    parser.add_argument('--filter-width', type=float, default=1.5, help='pixels (default 1.5)')
    parser.add_argument(
        '--denoise',
        action='store_true',
        help='also write <prefix>-full-denoised.exr (OpenImageDenoise with albedo and normal guides) for viewing; '
        'the raw EXRs are written as always and stay the ones accuracy is measured on',
    )
    parser.add_argument('--tone-mapping', default='aces', help="the viewer's tone mapper for this reference (stored, not applied to the EXRs)")
    parser.add_argument('--exposure-ev', type=float, default=None, help="the viewer's exposure for it; default the manifest's exposureEv")
    parser.add_argument(
        '--end-state',
        default=None,
        help='an entry of end-states.json: the scene with objects turned or a box added (TU.4); needs --textured. '
        'Output goes to bench/references/<end state>/<camera name>, and the end state may bring its own cameras',
    )
    parser.add_argument('--out-dir', default=None, help='default bench/references/<scene>/<camera name>')
    parser.add_argument('--name-suffix', default='', help='added to file names, e.g. -seed1')
    parser.add_argument('--threads', type=int, default=0, help='CPU threads; 0 means all')
    materials = parser.add_mutually_exclusive_group()
    materials.add_argument(
        '--materials-json',
        default=None,
        help='GI material table: give every material its constant GI albedo and emission (no textures); '
        'files are named cycles-untextured-*',
    )
    materials.add_argument(
        '--textured',
        action='store_true',
        help="the glTF's own materials as M4's GI sees them (textures, cutouts, blending, emission, "
        'metals as base x (1 - metallic), glass see-through); files are named cycles-textured-*',
    )
    return parser.parse_args(argv)


def pick_camera(manifest, wanted, extra_cameras=()):
    cameras = list(manifest['cameras']) + list(extra_cameras)
    if wanted.isdigit():
        index = int(wanted) - 1
        if not 0 <= index < len(cameras):
            raise SystemExit(f'camera {wanted} is out of range; the scene has {len(cameras)} bookmarks')
        return cameras[index]
    for camera in cameras:
        if camera['name'] == wanted:
            return camera
    raise SystemExit(f'no camera named {wanted!r}; known: {[c["name"] for c in cameras]}')


def camera_from_shot(path):
    """The stored camera and size of an earlier reference (its json's `shot`), to render the same view again."""
    with open(path, 'r', encoding='utf-8') as handle:
        shot = json.load(handle).get('shot')
    if not shot:
        raise SystemExit(f'{path} has no stored camera (`shot`); it was rendered before T1.13')
    camera = shot['camera']
    return (
        {'name': camera['bookmark'], 'position': camera['position'], 'target': camera['target'], 'fovYDeg': camera['fovYDeg']},
        camera['width'],
        camera['height'],
    )


def load_end_state(name):
    """An entry of end-states.json (the m4-textured-accuracy page reads the same file)."""
    states = bs.read_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'end-states.json'))
    if name not in states:
        raise SystemExit(f'no end state {name!r}; known: {[key for key in states if not key.startswith("$")]}')
    return states[name]


def srgb_to_linear(value):
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def add_end_state_boxes(state):
    """
    The end state's boxes, as the ddgi-move-cabinet page stands its cabinet (manifest space, Y up): centred at x,
    its bottom floorGap above the scene's lowest y and its back wallGap in front of the scene's lowest z
    (bounds of every mesh so far, as three's Box3 over the content). Diffuse, roughness 1, like three's material.
    """
    boxes = state.get('boxes') or []
    if not boxes:
        return []
    # The importer parents objects to the model transform after creating them: matrix_world is stale until this.
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ mathutils.Vector(corner) for obj in bpy.data.objects if obj.type == 'MESH' for corner in obj.bound_box]
    # Blender (x, y, z) is manifest (x, z, -y).
    min_y = min(corner.z for corner in corners)
    min_z = min(-corner.y for corner in corners)
    placed = []
    for box in boxes:
        sx, sy, sz = box['size']
        centre = (box['x'], min_y + sy / 2 + box['floorGap'], min_z + sz / 2 + box['wallGap'])
        bpy.ops.mesh.primitive_cube_add(size=1, location=bs.to_blender_space(centre))
        obj = bpy.context.active_object
        obj.name = box['name']
        obj.scale = (sx, sz, sy)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        hexcode = box['color']
        rgb = [srgb_to_linear(int(hexcode[index:index + 2], 16) / 255) for index in (0, 2, 4)]
        material = bpy.data.materials.new(box['name'])
        material.use_nodes = True
        principled = next(node for node in material.node_tree.nodes if node.type == 'BSDF_PRINCIPLED')
        principled.inputs['Base Color'].default_value = (*rgb, 1.0)
        principled.inputs['Roughness'].default_value = 1.0
        obj.data.materials.append(material)
        placed.append({'name': box['name'], 'centre': list(centre), 'size': box['size'], 'linearColor': rgb, 'sceneMinY': min_y, 'sceneMinZ': min_z})
    return placed


def apply_end_state_moves(state):
    """Turns the named objects about their pivots (viewer motion drivers: pivot x turn x pivot^-1 x rest)."""
    done = []
    for move in state.get('moves') or []:
        targets = [obj for obj in bpy.data.objects if obj.type == 'MESH' and (obj.name == move['object'] or obj.name.startswith(move['object'] + '.'))]
        if not targets:
            raise SystemExit(f"end state: no mesh object named {move['object']!r}; have {[o.name for o in bpy.data.objects if o.type == 'MESH']}")
        ax, ay, az = [math.radians(value) for value in move['rotationDeg']]
        turn = mathutils.Matrix.Rotation(ax, 3, 'X') @ mathutils.Matrix.Rotation(ay, 3, 'Y') @ mathutils.Matrix.Rotation(az, 3, 'Z')
        converted = bs.AXIS_CHANGE @ turn @ bs.AXIS_CHANGE.inverted()
        pivot = mathutils.Vector(bs.to_blender_space(move['pivot']))
        matrix = mathutils.Matrix.Translation(pivot) @ converted.to_4x4() @ mathutils.Matrix.Translation(-pivot)
        for obj in targets:
            obj.matrix_world = matrix @ obj.matrix_world
        done.append({'object': move['object'], 'meshObjects': [obj.name for obj in targets], 'pivot': move['pivot'], 'rotationDeg': move['rotationDeg']})
    return done


def file_sha256(path):
    # In 16 MB pieces: model files run to gigabytes (Bistro) inside an already busy Blender.
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for piece in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(piece)
    return digest.hexdigest()


# The repo paths whose uncommitted edits can change a reference: the render tools, the
# scenes (manifests, procedural builders, sky files) and the sky generator. Edits
# elsewhere (DDGI, the viewer, docs) can't, so they don't mark a render as made from
# uncommitted files. The GI material table is gitignored; its sha256 is in `shot.scene`.
RENDER_INPUT_PATHS = ('tools/reference', 'scenes', 'tools/sky')


def git_state():
    """
    The repo commit the render was made at, and whether files that affect it
    (RENDER_INPUT_PATHS) had uncommitted changes, with up to 20 of them listed.
    None when git isn't there.
    """
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--', *RENDER_INPUT_PATHS], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
        changed = [line[3:] for line in status.splitlines() if line.strip()]
        return {
            'commit': commit,
            'uncommittedChanges': bool(changed),
            'uncommittedScope': list(RENDER_INPUT_PATHS),
            'uncommittedFiles': changed[:20],
        }
    except (OSError, subprocess.CalledProcessError):
        return None


# The format of `shot` in a reference's json. The viewer (apps/viewer/src/scene/reference-files.ts)
# and the DDGI test pages read it; change the version when a field changes meaning.
SHOT_FORMAT = 'ddgi-reference-shot/1'


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


def gltf_dependencies(path):
    """
    The files a .gltf points at (buffers and images by `uri`), relative to its
    folder. A .glb holds everything itself, so it has none. Data URIs are
    inside the file and are skipped.
    """
    if not path.lower().endswith('.gltf'):
        return []
    with open(path, 'r', encoding='utf-8') as handle:
        gltf = json.load(handle)
    uris = [entry.get('uri') for entry in gltf.get('buffers', []) + gltf.get('images', [])]
    return sorted({uri for uri in uris if uri and not uri.startswith('data:')})


def part_sources(manifest, model_parts=None):
    """
    What the render was made from, file by file: for each loaded part its
    file's path and sha256, and for a .gltf the sha256 of every buffer and
    image it points at (so a changed texture changes the record). Procedural
    parts are the exported GLB (pnpm scenes:export). The same parts
    load_geometry loads.
    """
    from urllib.parse import unquote

    sources = []
    procedural_done = False
    for part in manifest['parts']:
        if model_parts is not None and part['kind'] == 'model' and part['asset'] not in model_parts:
            continue
        if part['kind'] == 'procedural':
            if procedural_done:
                continue
            procedural_done = True
            path = os.path.join(REPO_ROOT, 'assets', 'generated', f"{manifest['id']}.glb")
            entry = {'part': 'procedural', 'builders': [p['builder'] for p in manifest['parts'] if p['kind'] == 'procedural']}
        else:
            path = os.path.join(REPO_ROOT, 'assets', 'downloads', part['asset'], part['path'])
            entry = {'part': part['asset']}
        entry['path'] = os.path.relpath(path, REPO_ROOT).replace(os.sep, '/')
        entry['sha256'] = file_sha256(path)
        entry['bytes'] = os.path.getsize(path)
        dependencies = gltf_dependencies(path)
        if dependencies:
            folder = os.path.dirname(path)
            entry['dependencies'] = [{'uri': uri, 'sha256': file_sha256(os.path.join(folder, unquote(uri)))} for uri in dependencies]
        sources.append(entry)
    return sources


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


# ---------------------------------------------------------------------------
# Emitters other than the manifest lights
# ---------------------------------------------------------------------------

# Shader nodes that can emit light, and their (strength input, colour input)
# pairs. A pair emits when the strength is linked or above 0 and the colour is
# linked or not black (no colour input: the strength alone decides).
#   BSDF_PRINCIPLED     surface emission (the glTF importer's emissive)
#   EMISSION            the Emission shader, on a surface or as a volume
#   BACKGROUND          the world's shader; also allowed inside a material
#   PRINCIPLED_VOLUME   volume emission and blackbody glow
EMITTER_INPUTS = {
    'BSDF_PRINCIPLED': (('Emission Strength', 'Emission Color'),),
    'EMISSION': (('Strength', 'Color'),),
    'BACKGROUND': (('Strength', 'Color'),),
    'PRINCIPLED_VOLUME': (('Emission Strength', 'Emission Color'), ('Blackbody Intensity', None)),
}


def socket_is_on(socket, colour=False):
    """True if an input contributes: linked, or a value above 0 (a colour: any channel above 0)."""
    if socket.is_linked:
        return True
    if colour:
        return any(channel > 0.0 for channel in socket.default_value[:3])
    return socket.default_value > 0.0


def shader_trees():
    """Every shader node tree that can emit: the world's, each material's, and shader node groups (used inside those)."""
    trees = []
    for world in bpy.data.worlds:
        if world.use_nodes and world.node_tree is not None:
            trees.append((f'world {world.name!r}', world.node_tree))
    for material in bpy.data.materials:
        if material.use_nodes and material.node_tree is not None:
            trees.append((f'material {material.name!r}', material.node_tree))
    for group in bpy.data.node_groups:
        if group.bl_idname == 'ShaderNodeTree':
            trees.append((f'node group {group.name!r}', group))
    return trees


def find_emitters(scene):
    """
    Every source of light in the scene other than its light objects (which are
    the manifest's: import_gltf deletes the file's own lights), as readable
    lines. Empty when the lights are the only source.

    Conservative on purpose: an emitting node counts even when nothing is
    connected to it, and a shader node of a kind not in EMITTER_INPUTS that has
    an emission input is reported as unknown. A false alarm only makes the
    lights-only render skip its equal-to-direct check (or refuse to run); a
    missed emitter would put that light into the "analytic lights" share and
    take it out of the GI composite.
    """
    found = []
    world = scene.world
    if world is not None and not world.use_nodes and any(channel > 0.0 for channel in world.color[:3]):
        found.append(f'world {world.name!r}: colour {format_colour(world.color)} without nodes')
    for where, tree in shader_trees():
        # A world that is not the scene's does not render.
        if where.startswith('world ') and (world is None or tree != world.node_tree):
            continue
        for node in tree.nodes:
            pairs = EMITTER_INPUTS.get(node.type)
            if pairs is None:
                unknown = [socket.name for socket in node.inputs if 'emissi' in socket.name.lower()]
                if unknown:
                    found.append(f'{where} / {node.name!r}: a {node.type} node with emission inputs {unknown} (not a known kind)')
                continue
            for strength_name, colour_name in pairs:
                strength = node.inputs.get(strength_name)
                colour = node.inputs.get(colour_name) if colour_name else None
                if strength is None:
                    continue
                if socket_is_on(strength) and (colour is None or socket_is_on(colour, colour=True)):
                    found.append(f'{where} / {node.name!r}: {strength_name} on')
    return found


def format_colour(value):
    return '(' + ', '.join(f'{channel:g}' for channel in value[:3]) + ')'


def switch_off_emitters(scene):
    """
    Leaves the manifest's light objects as the only source of light, for the
    lights-only render:
    - the world becomes a new black one (no nodes, colour 0), whatever it held
      (sky texture, colour, any node setup);
    - every emitting input in every material and shader node group
      (EMITTER_INPUTS: Principled surface and volume emission, Emission and
      Background shaders, blackbody) is unlinked and set to 0.
    Then find_emitters must come back empty, or the run stops with the list.
    """
    black = bpy.data.worlds.new('lights-only-black')
    black.use_nodes = False
    black.color = (0.0, 0.0, 0.0)
    scene.world = black
    for _, tree in shader_trees():
        for node in tree.nodes:
            for strength_name, _ in EMITTER_INPUTS.get(node.type, ()):
                bs.force_input(tree, node, strength_name, 0.0)
    left = find_emitters(scene)
    if left:
        listing = '\n  '.join(left)
        raise SystemExit(f'{len(left)} source(s) of light other than the manifest lights are still on; refusing to render a lights-only image:\n  {listing}')


# What the lights-only render must look like, when that can be known without
# rendering it a second way (run.ts checks the images; see gi-composite.ts):
#   'equals-direct'  the manifest lights are the only source in the scene, so
#                    switching everything else off changes nothing: the
#                    lights-only direct image must equal the normal direct image
#   'black'          the scene has no manifest lights: the lights-only image must be 0
#   None             both kinds of source: no known answer to check against
def lights_only_expectation(emitters, light_count):
    if not emitters:
        return 'equals-direct'
    if light_count == 0:
        return 'black'
    return None


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


def add_denoised_output(tree, layers, output, name):
    """
    A second copy of the full image through OpenImageDenoise (the compositor's
    Denoise node, HDR, with Cycles' denoising albedo and normal as guides),
    for viewing only. The raw image is still written unchanged.
    """
    denoise = tree.nodes.new('CompositorNodeDenoise')
    denoise.use_hdr = True
    denoise.prefilter = 'ACCURATE'
    if hasattr(denoise, 'quality'):
        denoise.quality = 'HIGH'
    tree.links.new(layers.outputs['Image'], denoise.inputs['Image'])
    tree.links.new(layers.outputs['Denoising Normal'], denoise.inputs['Normal'])
    tree.links.new(layers.outputs['Denoising Albedo'], denoise.inputs['Albedo'])
    output.file_slots.new(name)
    tree.links.new(denoise.outputs['Image'], output.inputs[len(output.file_slots) - 1])


# How the viewing copy is denoised (render.py --denoise), as recorded in the json.
DENOISER = 'OpenImageDenoise (Blender compositor Denoise node: HDR, prefilter accurate, albedo and normal guides)'


def render_with_gi_parts(scene, directory, prefix, suffix, denoise=False):
    """
    The reference render, plus the parts of DDGI's quantity (GI_DEFINITION),
    for every scene: textured or untextured, with or without a sky.

    Cycles' Diffuse Indirect pass leaves out light that reaches the visible
    surface without a bounce, and that includes the sky and emissive surfaces
    seen straight from it. DDGI's field holds both: a probe ray that misses
    sees the sky, and one that hits an emitter sees its emission. Only the
    manifest's lights (sun, point, spot) are added analytically, outside the
    field. So, with Diffuse Colour as the albedo:
        DDGI's "indirect" = Indirect + Direct - (Direct from the lights alone)
    The lights' direct part comes from a second render with every other source
    switched off (switch_off_emitters) and no bounces, with the same seed and
    samples, so its pixel jitter matches and edges cancel. In a scene whose
    only sources are the manifest lights, the two direct images are the same
    and the composite is the indirect pass.

    Returns (seconds for both renders, seconds for the lights-only render, the files).
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
    denoised_name = f'{prefix}-full-denoised{suffix}'
    if denoise:
        add_denoised_output(tree, layers, output, denoised_name)
    started = time.time()
    bpy.ops.render.render(write_still=False)
    files = bs.collect_outputs(directory, full_name, indirect_name)
    files.append(take_output(directory, direct_name))
    if denoise:
        files.append(take_output(directory, denoised_name))

    # 2. The lights alone, direct only: every other source off, no bounces.
    lights_started = time.time()
    switch_off_emitters(scene)
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
    finished = time.time()
    return finished - started, finished - lights_started, files


def main():
    args = parse_arguments()
    end_state = None
    end_state_id = args.end_state
    if end_state_id:
        end_state = load_end_state(end_state_id)
        if args.scene != end_state['scene']:
            raise SystemExit(f"end state {end_state_id!r} is of scene {end_state['scene']!r}, not {args.scene!r}")
        if not args.textured:
            raise SystemExit('--end-state is supported with --textured only')
    manifest = bs.read_json(os.path.join(REPO_ROOT, 'scenes', 'manifests', f'{args.scene}.json'))
    width, height = args.width, args.height
    if args.from_shot:
        camera, width, height = camera_from_shot(args.from_shot)
    else:
        camera = pick_camera(manifest, args.camera, (end_state or {}).get('cameras', ()))
    bookmark_number = next((index + 1 for index, bookmark in enumerate(manifest['cameras']) if bookmark['name'] == camera['name']), None)

    out_dir = os.path.abspath(args.out_dir or os.path.join(REPO_ROOT, 'bench', 'references', end_state_id or manifest['id'], camera['name']))
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
    if gi_table:
        mode, prefix = 'untextured', 'cycles-untextured'
    elif args.textured:
        mode, prefix = 'textured', 'cycles-textured'
    else:
        mode, prefix = 'diffuse-textured', 'cycles'

    scene = bs.reset_scene()
    device_description = bs.configure_cycles(
        scene, width, height, args.samples, args.seed, args.device, pixel_filter=args.pixel_filter, filter_width=args.filter_width
    )
    if args.denoise:
        # Cycles' denoising albedo and normal passes: the guides for the viewing copy.
        scene.view_layers[0].cycles.denoising_store_passes = True
    if args.threads:
        scene.render.threads_mode = 'FIXED'
        scene.render.threads = args.threads

    load_started = time.time()
    model_parts = gi_table.get('parts') if gi_table else None
    imported_by_part = load_geometry(manifest, model_parts)
    end_state_applied = None
    if end_state:
        end_state_applied = {'id': end_state_id, 'moves': apply_end_state_moves(end_state), 'boxes': add_end_state_boxes(end_state)}
        print('END_STATE ' + json.dumps(end_state_applied))
    # Before make_materials_diffuse_only, which unlinks the Metallic and Transmission it reads.
    textured_materials = bs.apply_textured_material_model() if args.textured else None
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
    sources = part_sources(manifest, model_parts)
    # Record what the image was rendered from, so a reference made before a
    # scene changed (an added part, a new sky) can't be mistaken for a new one.
    sky_path = bs.sky_file_path(manifest['id'])
    sky_file = None
    if manifest['environment'].get('type') == 'procedural' and os.path.exists(sky_path):
        with open(sky_path, 'rb') as handle:
            sky_file = {'path': os.path.relpath(sky_path, bs.REPO_ROOT).replace(os.sep, '/'), 'sha256': hashlib.sha256(handle.read()).hexdigest()}
    # Taken before the lights-only render switches every other source off.
    emitters = find_emitters(scene)
    lights_only_expected = lights_only_expectation(emitters, len(manifest['lights']))
    render_seconds, lights_only_seconds, files = render_with_gi_parts(scene, out_dir, prefix, args.name_suffix, args.denoise)
    manifest_path = os.path.join(REPO_ROOT, 'scenes', 'manifests', f'{args.scene}.json')
    exposure_ev = args.exposure_ev if args.exposure_ev is not None else manifest.get('exposureEv', 0)
    # Everything needed to see this exact view again: in the viewer (camera, aspect,
    # exposure, tone mapping), in the DDGI test pages, or in Blender (--from-shot).
    shot = {
        'format': SHOT_FORMAT,
        'camera': {
            # Manifest space: three.js axes, Y up, metres. The camera's up is +Y, with
            # three's lookAt fallback when looking straight up or down (blender_scene.add_camera).
            'bookmark': camera['name'],
            'bookmarkNumber': bookmark_number,
            'position': camera['position'],
            'target': camera['target'],
            'up': [0, 1, 0],
            'fovYDeg': camera['fovYDeg'],
            'aspect': width / height,
            'width': width,
            'height': height,
            'near': bs.CAMERA_NEAR,
            'far': bs.CAMERA_FAR,
        },
        'display': {'exposureEv': exposure_ev, 'toneMapping': args.tone_mapping},
        'scene': {
            'manifest': manifest['id'],
            'manifestPath': os.path.relpath(manifest_path, REPO_ROOT).replace(os.sep, '/'),
            'manifestSha256': file_sha256(manifest_path),
            'skySha256': sky_file['sha256'] if sky_file else None,
            'giMaterialsSha256': gi_table_info['sha256'] if gi_table_info else None,
        },
        'render': {
            'git': git_state(),
            'blender': bpy.app.version_string,
            'device': device_description,
            'samplesPerPixel': args.samples,
            'seed': args.seed,
            'pixelFilter': {'type': bs.PIXEL_FILTERS[args.pixel_filter], 'widthPixels': args.filter_width},
            'renderSeconds': round(render_seconds, 2),
            # The raw EXRs are never denoised; with --denoise a separate viewing copy is.
            'denoised': {'file': f'{prefix}-full-denoised{args.name_suffix}.exr', 'denoiser': DENOISER} if args.denoise else None,
        },
    }

    meta = {
        'tool': 'tools/reference/cycles/render.py',
        'blender': bpy.app.version_string,
        'scene': manifest['id'],
        'parts': [part for part in manifest['parts'] if model_parts is None or part['kind'] != 'model' or part['asset'] in model_parts],
        'leftOutOfGiFaces': left_out_faces,
        'environment': manifest['environment'],
        'skyFile': sky_file,
        'camera': camera,
        'endState': end_state_applied,
        'shot': shot,
        'width': width,
        'height': height,
        'samplesPerPixel': args.samples,
        'seed': args.seed,
        'device': args.device,
        'deviceName': device_description,
        'threads': scene.render.threads,
        'maxBounces': bs.MAX_BOUNCES,
        # The raw images are never denoised; `denoisedFile` is a separate viewing copy (--denoise).
        'denoising': False,
        'denoisedFile': f'{prefix}-full-denoised{args.name_suffix}.exr' if args.denoise else None,
        'sampleClamp': 0,
        'lightSamplingThreshold': 0,
        'pixelFilter': f'{bs.PIXEL_FILTERS[args.pixel_filter]} {args.filter_width:g}',
        'materials': f'diffuse only ({materials_changed} Principled BSDFs reset, {texture_links_removed} texture links removed, checked)'
        + (f'; untextured: {gi_materials_set} materials set to the GI table constants, alpha 1, no normal maps (checked)' if gi_table else '')
        + ('; textured: the glTF materials as M4 GI sees them (metals folded into the base colour, glass see-through)' if args.textured else ''),
        # 'textured' (--textured), 'untextured' (--materials-json) or 'diffuse-textured' (neither).
        'mode': mode,
        'texturedMaterials': textured_materials,
        'sources': sources,
        'giMaterials': gi_table_info,
        'indirectDefinition': 'Diffuse Indirect x Diffuse Colour (light that bounced at least once before the visible surface)',
        'loadSeconds': round(load_seconds, 2),
        'renderSeconds': round(render_seconds, 2),
        'lightsOnlySeconds': round(lights_only_seconds, 2),
        'triangles': triangles,
        'files': [os.path.basename(path) for path in files],
        'outDir': out_dir,
        'jsonFile': f'{prefix}{args.name_suffix}.json',
        # Sources of light besides the manifest lights (sky, emissive materials), at most 20 listed.
        'otherEmitters': {'count': len(emitters), 'listed': emitters[:20]},
        'giComposite': {
            'output': f'{prefix}-gi{args.name_suffix}.exr',
            'add': [f'{prefix}-indirect{args.name_suffix}.exr', f'{prefix}-direct{args.name_suffix}.exr'],
            'subtract': f'{prefix}-lights-direct{args.name_suffix}.exr',
            'direct': f'{prefix}-direct{args.name_suffix}.exr',
            'definition': GI_DEFINITION,
            # run.ts checks the lights-only image against this before composing (gi-composite.ts).
            'lightsOnlyExpected': lights_only_expected,
        },
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


# Blender runs a --python script as __main__; calibrate.py imports this file for its functions.
if __name__ == '__main__':
    main()
