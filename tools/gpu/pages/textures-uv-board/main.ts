/**
 * M4, T4.1 done-when 2: a uv board. The albedo (and emission) at ray hits,
 * from `giSurfaceAtCone`, against a CPU reference sampler.
 *
 * Eight 1 m boards, each a quad with its own material mapping of one test
 * texture (red = x ramp, green = y ramp, blue = a 32-texel checker):
 * repeat with tiling, mirror, clamp, an offset, a rotation with mixed wrap,
 * uv set 1, a shear with clamp and mirror, and one with a metalness map
 * (blue channel, its own mapping) and an emissive map on a board stretched
 * 2× in x. Rays hit a 12 × 12 grid on each board head-on and at 50°, with
 * three cone spreads (0: mip 0; 0.004 and 0.03: mips between).
 *
 * The CPU takes each hit the GPU reports (instance, triangle, barycentrics,
 * distance) and works out the rest itself: the uv of the mapping's set, the
 * 3×2 transform, the ray-cone mip, the wrap, and a trilinear sample of the
 * store's own layer read back (LayerSampler: sRGB-decoded, bilinear with
 * repeat, blended between mips), then albedo = colour × texture × (1 −
 * metalness × map) and emission = emissive × texture. This checks the
 * mapping and the mip, not the layer (textures-sources does that).
 *
 * Passes when 99.5% of hits are within 0.01 of the CPU in every channel of
 * albedo and emission, and the mean difference is under 0.002 (filtering
 * hardware rounds its weights to about 1/256).
 */

import { assert, runGpuTest } from '../../client.ts';
import type { InstanceHandle, TextureHandle } from '../../../../packages/core/src/platform/contract/handles.ts';
import type { ColorRGB } from '../../../../packages/core/src/platform/contract/math.ts';
import type { MaterialDescriptor, UvMapping, UvWrap } from '../../../../packages/core/src/platform/contract/scene.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { STORE_LAYER_SIZE, STORE_MIP_LEVELS } from '../../../../packages/core/src/platform/textures/texture-store-wgsl.ts';
import { ready, ShadeKernel } from '../hit-shading-shared/harness.ts';
import { LayerSampler, readLayer } from '../textures-shared/textures.ts';

const SIZE = 256;
const GRID = 12;
const SPREADS = [0, 0.004, 0.03];
/** Head-on and turned 50° about y. */
const TILTS_DEGREES = [0, 50];
const RAY_DISTANCE = 2;
/** Board centres along x: room for the stretched board. */
const BOARD_SPACING = 2.5;
const HIT_TOLERANCE = 0.01;
const WITHIN_SHARE = 0.995;
const MEAN_TOLERANCE = 0.002;
/** As giSurfaceAtCone: the footprint stops growing below this |cos|. */
const MIN_FOOTPRINT_COSINE = 0.1;

type Vec3 = [number, number, number];
type Matrix = [number, number, number, number, number, number];

/** The quad: corners at ±0.5 in x and y, facing +z. uv set 0 has v down the board (as images), set 1 is another layout. */
const QUAD_POSITIONS = new Float32Array([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0]);
const QUAD_NORMALS = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
const QUAD_UVS0 = new Float32Array([0, 1, 1, 1, 1, 0, 0, 0]);
const QUAD_UVS1 = new Float32Array([0.25, 0.25, 0.25, 0.75, 0.9, 0.8, 0.8, 0.1]);
const QUAD_INDICES = new Uint32Array([0, 1, 2, 0, 2, 3]);

interface Board {
    readonly name: string;
    readonly material: MaterialDescriptor;
    /** Scale of the board in x (world), for a board whose world area differs from its uv area. */
    readonly scaleX: number;
}

function mapping(matrix: Matrix, wrapU: UvWrap = 'repeat', wrapV: UvWrap = 'repeat', channel: 0 | 1 = 0): UvMapping {
    return { matrix, wrapU, wrapV, channel };
}

function rotation(degrees: number, scale: number, offset: [number, number]): Matrix {
    const angle = (degrees * Math.PI) / 180;
    const cos = Math.cos(angle) * scale;
    const sin = Math.sin(angle) * scale;
    // Column-major 3×2: (m0, m1) is the first column.
    return [cos, sin, -sin, cos, offset[0], offset[1]];
}

/** The boards, all mapping the one test texture. */
function boards(handle: TextureHandle): Board[] {
    const albedo = (uv: UvMapping): MaterialDescriptor => ({ albedoTexture: handle, albedoUv: uv });
    return [
        { name: 'repeat x3', material: albedo(mapping([3, 0, 0, 3, 0, 0])), scaleX: 1 },
        { name: 'mirror x3', material: albedo(mapping([3, 0, 0, 3, -1, -1], 'mirror', 'mirror')), scaleX: 1 },
        { name: 'clamp x2', material: albedo(mapping([2, 0, 0, 2, -0.5, -0.5], 'clamp', 'clamp')), scaleX: 1 },
        { name: 'offset', material: albedo(mapping([1, 0, 0, 1, 0.3, 0.6])), scaleX: 1 },
        { name: 'rotated 30 x1.5, repeat/mirror', material: albedo(mapping(rotation(30, 1.5, [0.2, 0.1]), 'repeat', 'mirror')), scaleX: 1 },
        { name: 'uv set 1 x2', material: albedo(mapping([2, 0, 0, 2, 0, 0], 'repeat', 'repeat', 1)), scaleX: 1 },
        { name: 'shear, clamp/mirror', material: albedo(mapping([2.5, 0, 0.7, -1.5, 0.1, 0.4], 'clamp', 'mirror')), scaleX: 1 },
        {
            name: 'metalness and emission, stretched',
            material: {
                albedo: [0.9, 0.8, 1],
                albedoTexture: handle,
                albedoUv: mapping([1, 0, 0, 1, 0, 0]),
                metalness: 0.8,
                metalnessTexture: { texture: handle, channel: 'b', uv: mapping([2, 0, 0, 2, 0.1, 0.1]) },
                emissive: [2, 1, 0.5],
                emissiveTexture: handle,
                emissiveUv: mapping(rotation(-20, 1.2, [0.05, 0.3]), 'mirror', 'clamp'),
            },
            scaleX: 2,
        },
    ];
}

function boardTransform(index: number, scaleX: number): number[] {
    return [scaleX, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, index * BOARD_SPACING, 0, 0, 1];
}

/** The test texture: red ramps with x, green with y, blue is a checker of 32-texel squares. sRGB-encoded bytes. */
function testTexture(): Uint8Array<ArrayBuffer> {
    const pixels = new Uint8Array(SIZE * SIZE * 4);
    for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
            const at = (y * SIZE + x) * 4;
            pixels[at] = x;
            pixels[at + 1] = y;
            pixels[at + 2] = ((x >> 5) + (y >> 5)) % 2 === 0 ? 0 : 255;
            pixels[at + 3] = 255;
        }
    }
    return pixels;
}

const KERNEL = /* wgsl */ `
@group(1) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> outputs: array<vec4f>;

// Per ray, in: (origin, spread), (direction, 0). Out: (albedo, t), (emission, 0), (instance, triangle, barycentrics) as bits.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= arrayLength(&inputs) / 2u) {
        return;
    }
    let originAndSpread = inputs[2u * index];
    let ray = GiRay(originAndSpread.xyz, 0.0, normalize(inputs[2u * index + 1u].xyz), 100.0);
    let hit = giTraceClosest(ray);
    var albedo = vec3f(-1.0);
    var emission = vec3f(-1.0);
    if (hit.t >= 0.0) {
        let surface = giSurfaceAtCone(ray, hit, originAndSpread.w);
        albedo = surface.albedo;
        emission = surface.emission;
    }
    outputs[3u * index] = vec4f(albedo, hit.t);
    outputs[3u * index + 1u] = vec4f(emission, 0.0);
    outputs[3u * index + 2u] = vec4f(bitcast<f32>(hit.instance), bitcast<f32>(hit.triangle), hit.barycentrics);
}
`;

// ---------------------------------------------------------------------------
// The CPU reference
// ---------------------------------------------------------------------------

function sub(a: Vec3, b: Vec3): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function cross(a: Vec3, b: Vec3): Vec3 {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function length(a: Vec3): number {
    return Math.hypot(a[0], a[1], a[2]);
}

function wrap(coordinate: number, mode: UvWrap, halfTexel: number): number {
    if (mode === 'clamp') return Math.min(Math.max(coordinate, halfTexel), 1 - halfTexel);
    if (mode === 'mirror') {
        const fract = coordinate * 0.5 - Math.floor(coordinate * 0.5);
        const folded = 1 - Math.abs(2 * fract - 1);
        return Math.min(Math.max(folded, halfTexel), 1 - halfTexel);
    }
    return coordinate;
}

interface Hit {
    readonly board: Board;
    readonly triangle: number;
    readonly barycentrics: [number, number];
    readonly t: number;
    readonly direction: Vec3;
    readonly spread: number;
}

function sampleMapping(sampler: LayerSampler, hit: Hit, uv: UvMapping | undefined): [number, number, number, number] {
    const matrix = uv?.matrix ?? [1, 0, 0, 1, 0, 0];
    const uvs = uv?.channel === 1 ? QUAD_UVS1 : QUAD_UVS0;
    const corners = [0, 1, 2].map((corner) => QUAD_INDICES[hit.triangle * 3 + corner]);
    const weights = [1 - hit.barycentrics[0] - hit.barycentrics[1], hit.barycentrics[0], hit.barycentrics[1]].map((weight) => Math.min(Math.max(weight, 0), 1));
    const cornerUv = corners.map((vertex) => [uvs[vertex * 2], uvs[vertex * 2 + 1]]);
    const u = cornerUv.reduce((sum, value, corner) => sum + value[0] * weights[corner], 0);
    const v = cornerUv.reduce((sum, value, corner) => sum + value[1] * weights[corner], 0);
    const mappedU = matrix[0] * u + matrix[2] * v + matrix[4];
    const mappedV = matrix[1] * u + matrix[3] * v + matrix[5];

    // The ray-cone mip, as giShadeTextureLod.
    const world = corners.map((vertex) => [QUAD_POSITIONS[vertex * 3] * hit.board.scaleX, QUAD_POSITIONS[vertex * 3 + 1], QUAD_POSITIONS[vertex * 3 + 2]] as Vec3);
    const worldCross = cross(sub(world[1], world[0]), sub(world[2], world[0]));
    const doubleWorldArea = length(worldCross);
    const normal: Vec3 = [worldCross[0] / doubleWorldArea, worldCross[1] / doubleWorldArea, worldCross[2] / doubleWorldArea];
    const cosine = Math.max(Math.abs(hit.direction[0] * normal[0] + hit.direction[1] * normal[1] + hit.direction[2] * normal[2]), MIN_FOOTPRINT_COSINE);
    const width = (hit.spread * hit.t) / cosine;
    const edge1 = [cornerUv[1][0] - cornerUv[0][0], cornerUv[1][1] - cornerUv[0][1]];
    const edge2 = [cornerUv[2][0] - cornerUv[0][0], cornerUv[2][1] - cornerUv[0][1]];
    const doubleUvArea = Math.abs(edge1[0] * edge2[1] - edge1[1] * edge2[0]);
    const transformArea = Math.abs(matrix[0] * matrix[3] - matrix[2] * matrix[1]);
    const texelsPerMetre = Math.sqrt((doubleUvArea * transformArea * STORE_LAYER_SIZE * STORE_LAYER_SIZE) / doubleWorldArea);
    const lod = Math.min(Math.max(Math.log2(Math.max(width * texelsPerMetre, 1e-30)), 0), STORE_MIP_LEVELS - 1);
    const halfTexel = (0.5 * Math.pow(2, Math.ceil(lod))) / STORE_LAYER_SIZE;
    return sampler.sample(wrap(mappedU, uv?.wrapU ?? 'repeat', halfTexel), wrap(mappedV, uv?.wrapV ?? 'repeat', halfTexel), lod);
}

function cpuSurface(sampler: LayerSampler, hit: Hit): { albedo: Vec3; emission: Vec3 } {
    const material = hit.board.material;
    const colour: ColorRGB = material.albedo ?? [1, 1, 1];
    const texel = material.albedoTexture !== undefined ? sampleMapping(sampler, hit, material.albedoUv) : [1, 1, 1, 1];
    let metalness = material.metalness ?? 0;
    if (material.metalnessTexture) metalness *= sampleMapping(sampler, hit, material.metalnessTexture.uv)[2];
    const albedo = [0, 1, 2].map((channel) => Math.min(colour[channel] * texel[channel] * (1 - metalness), 1)) as Vec3;
    const emissive = material.emissive ?? [0, 0, 0];
    const emissiveTexel = material.emissiveTexture !== undefined ? sampleMapping(sampler, hit, material.emissiveUv) : [1, 1, 1, 1];
    const emission = [0, 1, 2].map((channel) => emissive[channel] * (material.emissiveIntensity ?? 1) * emissiveTexel[channel]) as Vec3;
    return { albedo, emission };
}

await runGpuTest(async ({ device, log }) => {
    const platform = new GIPlatformImpl(device, { label: 'textures-uv-board' });
    const scene = platform.scene;
    const texture = scene.createTexture({ source: { data: testTexture(), width: SIZE, height: SIZE }, srgb: true, label: 'uv board' });
    const mesh = scene.createMesh({ positions: QUAD_POSITIONS, normals: QUAD_NORMALS, uvs: QUAD_UVS0, uvs1: QUAD_UVS1, indices: QUAD_INDICES, label: 'board' });
    const list = boards(texture);
    const instances: InstanceHandle[] = list.map((board, index) =>
        scene.createInstance({ mesh, materials: [scene.createMaterial({ ...board.material, label: board.name })], transform: boardTransform(index, board.scaleX), label: board.name }),
    );
    await ready(device, platform);
    const layer = platform.sceneCopy.textureLayer(texture);
    assert(layer !== null && platform.textures.texture, 'the board texture has no layer');
    const sampler = new LayerSampler(await readLayer(device, platform.textures.texture, layer));
    const boardBySlot = new Map(instances.map((handle, index) => [platform.sceneCopy.instanceSlot(handle), list[index]]));

    // Rays: per board, a grid of points, each hit head-on and tilted, at every spread.
    const rays: { origin: Vec3; direction: Vec3; spread: number }[] = [];
    list.forEach((board, index) => {
        const centreX = index * BOARD_SPACING;
        for (let i = 0; i < GRID; i++) {
            for (let j = 0; j < GRID; j++) {
                const target: Vec3 = [centreX + (((i + 0.5) / GRID - 0.5) * 0.96) * board.scaleX, ((j + 0.5) / GRID - 0.5) * 0.96, 0];
                for (const tilt of TILTS_DEGREES) {
                    const angle = (tilt * Math.PI) / 180;
                    const direction: Vec3 = [-Math.sin(angle), 0, -Math.cos(angle)];
                    const origin: Vec3 = [target[0] - direction[0] * RAY_DISTANCE, target[1], target[2] - direction[2] * RAY_DISTANCE];
                    for (const spread of SPREADS) rays.push({ origin, direction, spread });
                }
            }
        }
    });
    const inputs = new Float32Array(rays.length * 8);
    rays.forEach((ray, index) => {
        inputs.set([...ray.origin, ray.spread, ...ray.direction, 0], index * 8);
    });
    const kernel = await ShadeKernel.create(device, platform, 'textures-uv-board', KERNEL);
    const output = await kernel.run(inputs, rays.length * 48, rays.length);
    const floats = new Float32Array(output);
    const words = new Uint32Array(output);

    let within = 0;
    let differenceSum = 0;
    let compared = 0;
    let worst = { difference: 0, board: '', gpu: [0, 0, 0] as number[], cpu: [0, 0, 0] as number[], spread: 0 };
    const perBoard: Record<string, { hits: number; worst: number }> = {};
    rays.forEach((ray, index) => {
        const base = index * 12;
        const t = floats[base + 3];
        assert(t >= 0, `ray ${index} missed its board`);
        const board = boardBySlot.get(words[base + 8]);
        assert(board, `ray ${index} hit instance ${words[base + 8]}, not a board`);
        const hit: Hit = { board, triangle: words[base + 9], barycentrics: [floats[base + 10], floats[base + 11]], t, direction: ray.direction, spread: ray.spread };
        const cpu = cpuSurface(sampler, hit);
        const gpu = [floats[base], floats[base + 1], floats[base + 2], floats[base + 4], floats[base + 5], floats[base + 6]];
        const expected = [...cpu.albedo, ...cpu.emission];
        let rayWorst = 0;
        for (let channel = 0; channel < 6; channel++) {
            const difference = Math.abs(gpu[channel] - expected[channel]);
            differenceSum += difference;
            compared++;
            rayWorst = Math.max(rayWorst, difference);
        }
        if (rayWorst <= HIT_TOLERANCE) within++;
        const record = (perBoard[board.name] ??= { hits: 0, worst: 0 });
        record.hits++;
        record.worst = Math.max(record.worst, rayWorst);
        if (rayWorst > worst.difference) worst = { difference: rayWorst, board: board.name, gpu, cpu: expected, spread: ray.spread };
    });
    const withinShare = within / rays.length;
    const meanDifference = differenceSum / compared;
    log(`${rays.length} hits on ${list.length} boards: ${(withinShare * 100).toFixed(2)}% within ${HIT_TOLERANCE}, mean difference ${meanDifference.toExponential(2)}, worst ${worst.difference.toFixed(4)} on '${worst.board}' (spread ${worst.spread})`);
    platform.dispose();

    assert(withinShare >= WITHIN_SHARE, `only ${(withinShare * 100).toFixed(2)}% of hits are within ${HIT_TOLERANCE} of the CPU sampler (worst ${worst.difference.toFixed(4)} on '${worst.board}': GPU ${worst.gpu.map((value) => value.toFixed(4))}, CPU ${worst.cpu.map((value) => value.toFixed(4))})`);
    assert(meanDifference <= MEAN_TOLERANCE, `mean difference ${meanDifference} is over ${MEAN_TOLERANCE}`);
    return { hits: rays.length, withinShare, meanDifference, worst, perBoard };
});
