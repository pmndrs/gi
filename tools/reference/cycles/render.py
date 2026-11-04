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
"""

import argparse
import json
import os
import platform
import sys
import time

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


def load_geometry(manifest):
    """Procedural parts come as one GLB per scene (pnpm scenes:export); model parts are downloaded files."""
    procedural_done = False
    for part in manifest['parts']:
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
            bs.import_gltf(path, part['transform'])


def main():
    args = parse_arguments()
    manifest = bs.read_json(os.path.join(REPO_ROOT, 'scenes', 'manifests', f'{args.scene}.json'))
    camera = pick_camera(manifest, args.camera)

    out_dir = os.path.abspath(args.out_dir or os.path.join(REPO_ROOT, 'bench', 'references', manifest['id'], camera['name']))
    full_name = f'cycles-full{args.name_suffix}'
    indirect_name = f'cycles-indirect{args.name_suffix}'

    scene = bs.reset_scene()
    bs.configure_cycles(scene, args.width, args.height, args.samples, args.seed, args.device)
    if args.threads:
        scene.render.threads_mode = 'FIXED'
        scene.render.threads = args.threads

    load_started = time.time()
    load_geometry(manifest)
    materials_changed = bs.make_materials_diffuse_only()
    bs.add_manifest_lights(scene, manifest['lights'])
    bs.add_environment(scene, manifest['environment'])
    bs.add_camera(scene, camera['position'], camera['target'], camera['fovYDeg'])
    load_seconds = time.time() - load_started

    triangles = sum(len(obj.data.polygons) for obj in bpy.data.objects if obj.type == 'MESH')
    render_seconds, files = bs.render_to_files(scene, out_dir, full_name, indirect_name)

    meta = {
        'tool': 'tools/reference/cycles/render.py',
        'blender': bpy.app.version_string,
        'scene': manifest['id'],
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
        'materials': f'diffuse only ({materials_changed} Principled BSDFs reset)',
        'indirectDefinition': 'Diffuse Indirect x Diffuse Colour (light that bounced at least once before the visible surface)',
        'loadSeconds': round(load_seconds, 2),
        'renderSeconds': round(render_seconds, 2),
        'triangles': triangles,
        'files': [os.path.basename(path) for path in files],
        'host': platform.node(),
        'finishedAt': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    meta_path = os.path.join(out_dir, f'cycles{args.name_suffix}.json')
    with open(meta_path, 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2)
    print('REFERENCE_RESULT ' + json.dumps(meta))
    # Blender can hang for a long time at exit tearing down add-ons; everything is written.
    sys.stdout.flush()
    os._exit(0)


main()
