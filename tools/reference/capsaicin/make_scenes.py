"""Build Capsaicin-ready copies of our scenes (read-only inputs, outputs next to this file).

Why: scene_viewer.exe loads exactly one file from --load-external-scene and has no FOV flag, so we
  * merge Sponza main + curtains into one .gltf (buffers/images stay in place, referenced by relative URI),
  * inject our manifest cameras (as glTF perspective cameras, so yfov is exact) and our manifest lights
    (KHR_lights_punctual, with unit conversion for Capsaicin's importer) into every scene,
  * drop the cameras and lights the source files carry, so only the manifest rig is present.
Capsaicin camera index = 0 ("User") then the injected cameras in order: manifest camera 0 -> index 1, etc.
"""
import json, math, os, struct, sys

WG = r'C:\Users\denni\Documents\dev\webgpu-ddgi'
# Outputs go to the git-ignored artifacts folder; the script itself is tracked in tools/reference/capsaicin.
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
OUT = os.path.join(REPO, 'artifacts', 'native', 'capsaicin', 'scenes')
os.makedirs(OUT, exist_ok=True)
LM = 683.0  # Capsaicin divides glTF light intensity by 683 (lumens->watts)


def lookat_quat(eye, target, up=(0, 1, 0)):
    z = [eye[i] - target[i] for i in range(3)]
    n = math.sqrt(sum(c * c for c in z)); z = [c / n for c in z]
    x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]]
    n = math.sqrt(sum(c * c for c in x)); x = [c / n for c in x]
    y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
    m = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]  # columns x,y,z
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1) * 2
        q = [(m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, s / 4]
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2
        q = [s / 4, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s]
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2
        q = [(m[0][1] + m[1][0]) / s, s / 4, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s]
    else:
        s = math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2
        q = [(m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, s / 4, (m[1][0] - m[0][1]) / s]
    return q


def dir_quat(d):
    # rotation taking the node's -Z axis onto d (the way a glTF light/camera points)
    return lookat_quat((0, 0, 0), d)


def rig(j, man):
    """Strip existing cameras/lights from glTF dict j and add the manifest's."""
    for n in j.get('nodes', []):
        n.pop('camera', None)
        if 'extensions' in n and 'KHR_lights_punctual' in n['extensions']:
            del n['extensions']['KHR_lights_punctual']
            if not n['extensions']:
                del n['extensions']
    j.pop('cameras', None)
    ext = j.get('extensions', {})
    ext.pop('KHR_lights_punctual', None)
    j['extensions'] = ext
    nodes = j.setdefault('nodes', [])
    roots = j['scenes'][j.get('scene', 0)]['nodes']
    j['cameras'] = []
    for c in man['cameras']:
        j['cameras'].append({'type': 'perspective', 'name': c['name'],
                             'perspective': {'yfov': math.radians(c['fovYDeg']), 'aspectRatio': 16 / 9,
                                             'znear': 0.1, 'zfar': 10000.0}})
        nodes.append({'name': 'cam_' + c['name'], 'camera': len(j['cameras']) - 1,
                      'translation': c['position'], 'rotation': lookat_quat(c['position'], c['target'])})
        roots.append(len(nodes) - 1)
    lights = []
    for l in man['lights']:
        if l['intensity'] <= 0:
            continue
        if l['type'] == 'directional':
            gl = {'type': 'directional', 'color': l['color'], 'intensity': l['intensity'] * LM, 'name': l['label']}
            nd = {'name': l['label'], 'rotation': dir_quat(l['direction'])}
        else:
            # Capsaicin: W = I/683*4pi and shading uses W/d^2, so I_gltf = 683*candela/4pi gives candela/d^2.
            gl = {'type': 'point', 'color': l['color'], 'intensity': l['intensity'] * LM / (4 * math.pi),
                  'name': l['label']}
            if l.get('range'):
                gl['range'] = l['range']
            nd = {'name': l['label'], 'translation': l['position']}
        lights.append(gl)
        nd['extensions'] = {'KHR_lights_punctual': {'light': len(lights) - 1}}
        nodes.append(nd)
        roots.append(len(nodes) - 1)
    if lights:
        ext['KHR_lights_punctual'] = {'lights': lights}
        used = j.setdefault('extensionsUsed', [])
        if 'KHR_lights_punctual' not in used:
            used.append('KHR_lights_punctual')
    if not ext:
        j.pop('extensions')
    return j


def remap_index_keys(o, off):
    """Offset every textureInfo {"index": n} inside a material."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k == 'index' and isinstance(v, int):
                o[k] = v + off['textures']
            else:
                remap_index_keys(v, off)
    elif isinstance(o, list):
        for v in o:
            remap_index_keys(v, off)


ARRAYS = ('buffers', 'bufferViews', 'accessors', 'images', 'samplers', 'textures', 'materials', 'meshes', 'nodes')


def merge(paths):
    base = None
    for p in paths:
        src = os.path.join(WG, p)
        d = os.path.dirname(src)
        j = json.load(open(src, encoding='utf-8'))
        for b in j.get('buffers', []):
            b['uri'] = os.path.relpath(os.path.normpath(os.path.join(d, b['uri'])), OUT).replace('\\', '/')
        for im in j.get('images', []):
            if 'uri' in im:
                im['uri'] = os.path.relpath(os.path.normpath(os.path.join(d, im['uri'])), OUT).replace('\\', '/')
        if base is None:
            base = j
            continue
        off = {k: len(base.get(k, [])) for k in ARRAYS}
        for bv in j['bufferViews']:
            bv['buffer'] += off['buffers']
        for a in j['accessors']:
            if 'bufferView' in a:
                a['bufferView'] += off['bufferViews']
        for t in j['textures']:
            if 'sampler' in t:
                t['sampler'] += off['samplers']
            t['source'] += off['images']
        for m in j['materials']:
            remap_index_keys(m, off)
        for m in j['meshes']:
            for pr in m['primitives']:
                pr['attributes'] = {k: v + off['accessors'] for k, v in pr['attributes'].items()}
                if 'indices' in pr:
                    pr['indices'] += off['accessors']
                if 'material' in pr:
                    pr['material'] += off['materials']
        for n in j['nodes']:
            if 'mesh' in n:
                n['mesh'] += off['meshes']
            if 'children' in n:
                n['children'] = [c + off['nodes'] for c in n['children']]
        for k in ARRAYS:
            base[k] += j[k]
        base['scenes'][0]['nodes'] += [c + off['nodes'] for c in j['scenes'][j.get('scene', 0)]['nodes']]
    return base


def webp_to_jpeg_png(j, binchunk, max_dim=0):
    """Capsaicin's importer (cgltf + stb) cannot decode WebP and ignores EXT_texture_webp, so those materials render
    with white albedo. Re-encode every WebP image (JPEG q95, or PNG when it has alpha) and append it to the BIN chunk."""
    from concurrent.futures import ThreadPoolExecutor
    import subprocess
    data = bytearray(binchunk[8:])  # skip chunk header
    def conv(i):
        im = j['images'][i]; bv = j['bufferViews'][im['bufferView']]
        raw = bytes(data[bv.get('byteOffset', 0):bv.get('byteOffset', 0) + bv['byteLength']])
        has_alpha = subprocess.run(['magick', 'webp:-', '-format', '%A', 'info:'], input=raw, capture_output=True).stdout.strip() in (b'True', b'Blend', b'Set')
        fmt = 'png' if has_alpha else 'jpg'
        args = ['magick', 'webp:-'] + (['-resize', '%dx%d>' % (max_dim, max_dim)] if max_dim else []) + ([] if has_alpha else ['-quality', '95']) + [fmt + ':-']
        return i, fmt, subprocess.run(args, input=raw, capture_output=True, check=True).stdout
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(conv, range(len(j['images']))))
    for i, fmt, out in results:
        data += bytes(1) * ((4 - len(data) % 4) % 4)
        j['bufferViews'].append({'buffer': 0, 'byteOffset': len(data), 'byteLength': len(out)})
        data += out
        j['images'][i] = {'bufferView': len(j['bufferViews']) - 1, 'mimeType': 'image/png' if fmt == 'png' else 'image/jpeg',
                          'name': j['images'][i].get('name', 'img%d' % i)}
    for t in j['textures']:
        e = t.get('extensions', {}).pop('EXT_texture_webp', None)
        if e: t['source'] = e['source']
        if not t.get('extensions'): t.pop('extensions', None)
    j['buffers'][0]['byteLength'] = len(data)
    j['extensionsUsed'] = [e for e in j.get('extensionsUsed', []) if e != 'EXT_texture_webp']
    j['extensionsRequired'] = [e for e in j.get('extensionsRequired', []) if e != 'EXT_texture_webp']
    if not j['extensionsRequired']: j.pop('extensionsRequired')
    data += bytes(1) * ((4 - len(data) % 4) % 4)
    return struct.pack('<II', len(data), 0x004E4942) + bytes(data)


def glb_rig(src, dst, man, max_dim=0):
    d = open(src, 'rb').read()
    magic, ver, length = struct.unpack('<III', d[:12])
    jl, jt = struct.unpack('<II', d[12:20])
    j = json.loads(d[20:20 + jl].decode('utf-8'))
    rest = d[20 + jl:]
    if 'EXT_texture_webp' in j.get('extensionsUsed', []):
        rest = webp_to_jpeg_png(j, rest, max_dim)
    j = rig(j, man)
    js = json.dumps(j, separators=(',', ':')).encode('utf-8')
    js += b' ' * ((4 - len(js) % 4) % 4)
    open(dst, 'wb').write(struct.pack('<III', magic, ver, 12 + 8 + len(js) + len(rest))
                          + struct.pack('<II', len(js), jt) + js + rest)


def flat_plane(dst, albedo=0.5, half=100.0, cam_h=5.0, yfov_deg=60.0):
    """Sky-only furnace-style test (2026-11-19, D25): one upward-facing Lambert-ish plane (metallic 0, roughness 1,
    grey albedo), no lights, camera 1 looking straight down. Light it with environment 6 (White, radiance 1.0), so the
    exact answer is E = pi*L at every point and outgoing radiance = albedo*L for the diffuse lobe."""
    import base64
    pos = [(-half, 0, -half), (-half, 0, half), (half, 0, half), (half, 0, -half)]
    nrm = [(0, 1, 0)] * 4
    uv = [(0, 0), (0, 1), (1, 1), (1, 0)]
    idx = [0, 1, 2, 0, 2, 3]  # counter-clockwise seen from +Y, so the front face points up
    blob = b''.join(struct.pack('<3f', *v) for v in pos) + b''.join(struct.pack('<3f', *v) for v in nrm)         + b''.join(struct.pack('<2f', *v) for v in uv) + struct.pack('<6I', *idx)
    j = {'asset': {'version': '2.0', 'generator': 'ddgi make_scenes.py flat-plane'}, 'scene': 0,
         'scenes': [{'nodes': [0, 1]}],
         'nodes': [{'name': 'plane', 'mesh': 0},
                   {'name': 'cam_down', 'camera': 0, 'translation': [0, cam_h, 0],
                    'rotation': [-math.sqrt(0.5), 0, 0, math.sqrt(0.5)]}],  # -90 deg about X: -Z view axis -> -Y
         'cameras': [{'type': 'perspective', 'name': 'down',
                      'perspective': {'yfov': math.radians(yfov_deg), 'aspectRatio': 16 / 9, 'znear': 0.1, 'zfar': 10000.0}}],
         'meshes': [{'name': 'plane', 'primitives': [{'attributes': {'POSITION': 0, 'NORMAL': 1, 'TEXCOORD_0': 2},
                                                      'indices': 3, 'material': 0}]}],
         'materials': [{'name': 'grey', 'pbrMetallicRoughness': {'baseColorFactor': [albedo, albedo, albedo, 1.0],
                                                                 'metallicFactor': 0.0, 'roughnessFactor': 1.0}}],
         'buffers': [{'byteLength': len(blob), 'uri': 'data:application/octet-stream;base64,' + base64.b64encode(blob).decode()}],
         'bufferViews': [{'buffer': 0, 'byteOffset': 0, 'byteLength': 48, 'target': 34962},
                         {'buffer': 0, 'byteOffset': 48, 'byteLength': 48, 'target': 34962},
                         {'buffer': 0, 'byteOffset': 96, 'byteLength': 32, 'target': 34962},
                         {'buffer': 0, 'byteOffset': 128, 'byteLength': 24, 'target': 34963}],
         'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': 4, 'type': 'VEC3',
                        'min': [-half, 0, -half], 'max': [half, 0, half]},
                       {'bufferView': 1, 'componentType': 5126, 'count': 4, 'type': 'VEC3'},
                       {'bufferView': 2, 'componentType': 5126, 'count': 4, 'type': 'VEC2'},
                       {'bufferView': 3, 'componentType': 5125, 'count': 6, 'type': 'SCALAR'}]}
    json.dump(j, open(dst, 'w'), indent=1)


def man(name):
    return json.load(open(os.path.join(WG, 'scenes', 'manifests', name + '.json'), encoding='utf-8'))


for w in sys.argv[1:] or ['sponza', 'bistro-interior', 'bistro-exterior', 'two-room']:
    m = man(w) if not w.endswith('-1k') and w != 'flat-plane' else None
    if w == 'flat-plane':
        flat_plane(os.path.join(OUT, 'flat-plane.gltf'))
    elif w == 'sponza':
        j = rig(merge(['assets/downloads/sponza-intel/main_sponza_1k/NewSponza_Main_1k.gltf',
                       'assets/downloads/sponza-intel-curtains/pkg_a_curtains_1k/NewSponza_Curtains_1k.gltf']), m)
        json.dump(j, open(os.path.join(OUT, 'sponza-curtains.gltf'), 'w'))
    elif w == 'bistro-interior':
        glb_rig(os.path.join(WG, 'assets/downloads/bistro-interior/BistroInterior.glb'),
                os.path.join(OUT, 'bistro-interior.glb'), m)
    elif w == 'bistro-exterior':
        glb_rig(os.path.join(WG, 'assets/downloads/bistro-exterior/BistroExterior.glb'),
                os.path.join(OUT, 'bistro-exterior.glb'), m)
    elif w in ('bistro-interior-1k', 'bistro-exterior-1k'):
        # same scene with textures capped at 1024 px so it fits the 6 GB RTX 3060 Laptop for timing runs
        base = w[:-3]
        glb_rig(os.path.join(WG, 'assets/downloads/%s/%s.glb' % (base, 'BistroInterior' if 'interior' in base else 'BistroExterior')),
                os.path.join(OUT, w + '.glb'), man(base), 1024)
    elif w == 'two-room':
        glb_rig(os.path.join(WG, 'assets/generated/two-room.glb'), os.path.join(OUT, 'two-room.glb'), m)
    print('wrote', w)
