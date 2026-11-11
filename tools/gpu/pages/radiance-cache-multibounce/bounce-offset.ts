/**
 * Register D39 on `radiance-cache-multibounce`: Capsaicin's `offsetPosition`
 * for the second-bounce ray (`rcOffsetPosition`, geometry.hlsl:35-47), and
 * how many second-bounce rays it sends somewhere else than the platform's
 * push (`giShadeOffsetRayOrigin`), which the port used before RC6.2.
 *
 * - `checkOffsetPositionTwin`: the WGSL function on 4,096 points and
 *   normals (both signs, near and far from the origin, coordinates under
 *   1/32 included) against its CPU twin (`offsetPosition` in
 *   capsaicin-math.ts), bit for bit.
 * - `measureBounceOffset`: the technique on a scene (Capsaicin's defaults,
 *   multibounce on) for a few frames from an empty cache, then a page kernel
 *   that repeats `PopulateMultibounceCells`' steps for every first-bounce
 *   entry of the latest frame with the production functions (the discard
 *   draw, `giSurfaceAt`, the cosine sample) and traces the second-bounce ray
 *   twice, from Capsaicin's origin and from the platform's. Nothing here is
 *   in the production shaders.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { offsetPosition, type Direction } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { CAPSAICIN_WGSL } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/capsaicin.ts';
import { HASH_GRID_COUNTER } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/capacity.ts';
import {
    INV_PI_WGSL,
    SAMPLE_DIFFUSE_WGSL,
    VISIBILITY_BACK_FACE_BIT,
    VISIBILITY_WORD,
    VISIBILITY_WORDS,
} from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-population-wgsl.ts';
import { readBuffer } from '../ddgi-shared/harness.ts';
import { readRange } from '../radiance-cache-world-cache/corner.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
/** Frames after the trees are built, from an empty cache: the cache holds tiles, the screen probes have history. */
const WARM_FRAMES = 8;
/** A hit closer than this to its ray's origin counts as caught by the surface it left, or the one beside it (m). */
const NEAR_HIT = 1e-3;

// ---- offsetPosition against its CPU twin -------------------------------------------------------

const TWIN_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
@group(0) @binding(0) var<storage, read> twinInputs: array<vec4f>;    // position, then normal, per point
@group(0) @binding(1) var<storage, read_write> twinResults: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= arrayLength(&twinResults)) {
        return;
    }
    twinResults[id.x] = vec4f(rcOffsetPosition(twinInputs[2u * id.x].xyz, twinInputs[2u * id.x + 1u].xyz), 0.0);
}
`;

/** Points spread over 1e-3 to 1e3 m on both sides of the origin (a fifth of the coordinates under 1/32), with unit normals. */
function twinCases(count: number): { positions: Direction[]; normals: Direction[] } {
    let state = 2026;
    const random = (): number => {
        state = (Math.imul(state, 1103515245) + 12345) >>> 0;
        return state / 2 ** 32;
    };
    const positions: Direction[] = [];
    const normals: Direction[] = [];
    for (let index = 0; index < count; index++) {
        const coordinate = (): number => {
            const sign = random() < 0.5 ? -1 : 1;
            const magnitude = random() < 0.2 ? random() / 32 : 10 ** (random() * 6 - 3);
            return Math.fround(sign * magnitude);
        };
        positions.push([coordinate(), coordinate(), coordinate()]);
        const normal = [random() * 2 - 1, random() * 2 - 1, random() * 2 - 1];
        const size = Math.hypot(normal[0], normal[1], normal[2]) || 1;
        normals.push([Math.fround(normal[0] / size), Math.fround(normal[1] / size), Math.fround(normal[2] / size)]);
    }
    // Axis-aligned normals, as on Cornell's walls: one coordinate moves, by 256 steps.
    normals[0] = [0, 0, 1];
    normals[1] = [0, -1, 0];
    normals[2] = [-1, 0, 0];
    return { positions, normals };
}

export async function checkOffsetPositionTwin(device: GPUDevice): Promise<{ points: number; mismatches: number; examples: string[] }> {
    const count = 4096;
    const { positions, normals } = twinCases(count);
    const inputs = new Float32Array(count * 8);
    positions.forEach((position, index) => {
        inputs.set(position, index * 8);
        inputs.set(normals[index], index * 8 + 4);
    });
    const input = device.createBuffer({ size: inputs.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(input, 0, inputs);
    const output = device.createBuffer({ size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code: TWIN_WGSL }), entryPoint: 'main' } });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output } }] }));
    pass.dispatchWorkgroups(count / 64);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const results = new Float32Array(await readBuffer(device, output, count * 16));
    input.destroy();
    output.destroy();
    let mismatches = 0;
    const examples: string[] = [];
    positions.forEach((position, index) => {
        const expected = offsetPosition(position, normals[index]);
        const gpu = [results[index * 4], results[index * 4 + 1], results[index * 4 + 2]];
        if (gpu.some((value, axis) => !Object.is(value, expected[axis]))) {
            mismatches++;
            if (examples.length < 4) examples.push(`${position.join(', ')} along ${normals[index].join(', ')}: GPU ${gpu.join(', ')}, twin ${expected.join(', ')}`);
        }
    });
    return { points: count, mismatches, examples };
}

// ---- The second-bounce rays from both origins --------------------------------------------------

/** Words per first-bounce entry in the page kernel's results. */
const RESULT_WORDS = 10;
const RESULT = {
    /** Bits: kept (not discarded), the platform-origin ray hit, the Capsaicin-origin ray hit, either one's hit on a front face. */
    flags: 0,
    platformT: 1,
    capsaicinT: 2,
    platformInstance: 3,
    platformTriangle: 4,
    capsaicinInstance: 5,
    capsaicinTriangle: 6,
    /** How far each origin sits off the surface along its normal, and how far Capsaicin's moves along the surface (f32). */
    platformLift: 7,
    capsaicinLift: 8,
    capsaicinSlide: 9,
} as const;
const FLAG = { kept: 1, platformHit: 2, capsaicinHit: 4, platformFront: 8, capsaicinFront: 16 } as const;

const MEASURE_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${INV_PI_WGSL}
${SAMPLE_DIFFUSE_WGSL}
struct OffsetParams {
    entry_count: u32,        // first-bounce entries (VisibilityCountBuffer0)
    frame_index: u32,        // g_HashGridCacheConstants.frame_index of the frame that made them
    visibility_buffer: u32,  // g_HashGridCache_RayLists.visibility_buffer: the entries' first word
    discard_probability: f32, // discard_multibounce_ray_probability
}
@group(1) @binding(0) var<uniform> offsetParams: OffsetParams;
@group(1) @binding(1) var<storage, read> listWords: array<u32>;
@group(1) @binding(2) var<storage, read_write> offsetResults: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let did = id.x;
    if (did >= offsetParams.entry_count) {
        return;
    }
    let result = did * ${RESULT_WORDS}u;
    // PopulateMultibounceCells' steps (hash-grid-population-wgsl.ts): the discard draw first.
    var random = rcMakeRandom(did, offsetParams.frame_index);
    if (rcRand(&random) <= offsetParams.discard_probability) {
        offsetResults[result + ${RESULT.flags}u] = 0u;
        return;
    }
    // The visibility entry (HashGridCache_LoadVisibility) and its surface (HashGridCache_VisibilitySurface).
    let entry = offsetParams.visibility_buffer + ${VISIBILITY_WORDS}u * did;
    let origin = vec3f(bitcast<f32>(listWords[entry + ${VISIBILITY_WORD.origin}u]),
                       bitcast<f32>(listWords[entry + ${VISIBILITY_WORD.origin}u + 1u]),
                       bitcast<f32>(listWords[entry + ${VISIBILITY_WORD.origin}u + 2u]));
    let direction = vec3f(unpack2x16float(listWords[entry + ${VISIBILITY_WORD.direction}u]),
                          unpack2x16float(listWords[entry + ${VISIBILITY_WORD.direction}u + 1u]).x);
    let hit_distance = bitcast<f32>(listWords[entry + ${VISIBILITY_WORD.hitDistance}u]);
    let instance_word = listWords[entry + ${VISIBILITY_WORD.instance}u];
    let barycentrics = vec2f(bitcast<f32>(listWords[entry + ${VISIBILITY_WORD.barycentrics}u]),
                             bitcast<f32>(listWords[entry + ${VISIBILITY_WORD.barycentrics}u + 1u]));
    let first_ray = GiRay(origin, 0.0, direction, max(hit_distance, 0.0));
    let first_hit = GiHit(hit_distance, instance_word & ~${VISIBILITY_BACK_FACE_BIT}u, listWords[entry + ${VISIBILITY_WORD.triangle}u], barycentrics,
                          (instance_word & ${VISIBILITY_BACK_FACE_BIT}u) == 0u);
    let surface = giSurfaceAt(first_ray, first_hit);
    let sample = HashGridCache_SampleBRDFAndEvaluateDiffuse(surface.albedo, surface.normal, rcRand2(&random));

    // The same ray from both origins.
    let platform_origin = giShadeOffsetRayOrigin(surface.position, surface.normal);
    let capsaicin_origin = rcOffsetPosition(surface.position, surface.normal);
    let platform_hit = giTraceClosest(GiRay(platform_origin, 0.0, sample.direction, RC_MAX_HIT_DISTANCE));
    let capsaicin_hit = giTraceClosest(GiRay(capsaicin_origin, 0.0, sample.direction, RC_MAX_HIT_DISTANCE));

    var flags = ${FLAG.kept}u;
    if (platform_hit.t >= 0.0) {
        flags |= ${FLAG.platformHit}u | select(0u, ${FLAG.platformFront}u, platform_hit.frontFace);
    }
    if (capsaicin_hit.t >= 0.0) {
        flags |= ${FLAG.capsaicinHit}u | select(0u, ${FLAG.capsaicinFront}u, capsaicin_hit.frontFace);
    }
    let capsaicin_shift = capsaicin_origin - surface.position;
    let capsaicin_lift = dot(capsaicin_shift, surface.normal);
    offsetResults[result + ${RESULT.flags}u] = flags;
    offsetResults[result + ${RESULT.platformT}u] = bitcast<u32>(platform_hit.t);
    offsetResults[result + ${RESULT.capsaicinT}u] = bitcast<u32>(capsaicin_hit.t);
    offsetResults[result + ${RESULT.platformInstance}u] = platform_hit.instance;
    offsetResults[result + ${RESULT.platformTriangle}u] = platform_hit.triangle;
    offsetResults[result + ${RESULT.capsaicinInstance}u] = capsaicin_hit.instance;
    offsetResults[result + ${RESULT.capsaicinTriangle}u] = capsaicin_hit.triangle;
    offsetResults[result + ${RESULT.platformLift}u] = bitcast<u32>(dot(platform_origin - surface.position, surface.normal));
    offsetResults[result + ${RESULT.capsaicinLift}u] = bitcast<u32>(capsaicin_lift);
    offsetResults[result + ${RESULT.capsaicinSlide}u] = bitcast<u32>(length(capsaicin_shift - capsaicin_lift * surface.normal));
}
`;

/** What the two origins did with the same second-bounce rays (the file comment). */
export interface BounceOffsetMeasure {
    readonly frame: number;
    readonly firstBounceEntries: number;
    /** Entries that kept their second-bounce ray (the 30% of the 0.7 discard). */
    readonly secondBounceRays: number;
    /** The production kernel's second-bounce entries this frame, and the inserts the cache refused (a refused hit makes no entry). */
    readonly productionEntries: number;
    readonly bucketOverflows: number;
    readonly hitsFromCapsaicinOrigin: number;
    readonly hitsFromPlatformOrigin: number;
    /** Rays whose hit differs: a hit against a miss either way, or another triangle. */
    readonly changedHit: number;
    readonly hitOnlyFromCapsaicin: number;
    readonly hitOnlyFromPlatform: number;
    readonly otherTriangle: number;
    /** Of the other-triangle rays: one of the two hit within {@link NEAR_HIT} of its origin. */
    readonly otherTriangleNearHit: { readonly capsaicin: number; readonly platform: number };
    /** Back-face hits from each origin: a ray that starts behind a surface sees back faces (a leak would show here). */
    readonly backFaceHits: { readonly capsaicin: number; readonly platform: number };
    /** Of the changed rays: Capsaicin's hits a back face where the platform's hits a front face, and the other way. */
    readonly changedToBackFace: number;
    readonly changedToFrontFace: number;
    /** Hits within {@link NEAR_HIT} of the origin, from each. */
    readonly nearHits: { readonly capsaicin: number; readonly platform: number };
    /** The push off the surface along the normal (m): median and largest, for each origin; Capsaicin's slide along the surface. */
    readonly lift: {
        readonly capsaicinMedian: number;
        readonly capsaicinLargest: number;
        /** Capsaicin's origins that end up on or below the surface along its normal (none expected). */
        readonly capsaicinNotAbove: number;
        readonly platformMedian: number;
        readonly platformLargest: number;
        readonly capsaicinSlideLargest: number;
    };
    readonly examples: readonly string[];
}

function median(values: Float32Array): number {
    if (values.length === 0) return 0;
    const sorted = Float32Array.from(values).sort();
    return sorted[Math.floor(sorted.length / 2)];
}

/**
 * The technique on a scene, `WARM_FRAMES` frames from an empty cache with the
 * trees built, then every kept second-bounce ray of the latest frame from
 * both origins.
 */
export async function measureBounceOffset(
    device: GPUDevice,
    name: string,
    camera: PageCamera,
    describe: (platform: GIPlatformImpl) => Promise<(() => void) | void>,
): Promise<BounceOffsetMeasure> {
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-multibounce ${name}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique());
    platform.setOutput(technique);
    await platform.prepare();
    const beforeFrame = (await describe(platform)) ?? (() => {});
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    try {
        const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
        const frame = (): void => {
            beforeFrame();
            const encoder = device.createCommandEncoder({ label: `radiance-cache-multibounce ${name}` });
            platform.encode(encoder, input);
            screen.encode(encoder, camera);
            platform.encodeScreen(encoder, screen.screen);
            device.queue.submit([encoder.finish()]);
        };
        frame();
        await platform.rays.whenBuilt();
        technique.resetHistory();
        for (let index = 0; index < WARM_FRAMES; index++) frame();
        await device.queue.onSubmittedWorkDone();
        if (platform.rays.building) throw new Error(`${name}: the platform still reports trees building`);

        const cache = technique.worldCache;
        const lists = cache?.rayLists;
        if (!cache || !lists) throw new Error(`${name}: no world cache lists after the warm frames`);
        const counters = new Uint32Array(await readRange(device, cache.buffers.words, 0, 64 * 4));
        const entryCount = counters[HASH_GRID_COUNTER.visibilityCount0];
        // The latest frame's index, which the cache's constants hold as g_FrameIndex (as the corner's checks read it).
        const frameIndex = (technique.probes?.frameIndex ?? 0) >>> 0;

        const kernel = await platform.kernels.computeKernel({
            label: 'radiance-cache-multibounce bounce offset',
            modules: [platform.shade.module],
            code: MEASURE_WGSL,
            entryPoint: 'main',
            ownGroups: [
                {
                    entries: [
                        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                    ],
                },
            ],
        });
        const params = new ArrayBuffer(16);
        new Uint32Array(params, 0, 3).set([entryCount, frameIndex, lists.layout.visibilityBuffer]);
        new Float32Array(params, 12, 1)[0] = Math.fround(technique.settings.hashGridCacheDiscardMultibounceRayProbability);
        const paramsBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(paramsBuffer, 0, params);
        const resultBytes = Math.max(entryCount, 1) * RESULT_WORDS * 4;
        const results = device.createBuffer({ size: resultBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const encoder = device.createCommandEncoder({ label: 'radiance-cache-multibounce bounce offset' });
        const pass = encoder.beginComputePass();
        kernel.bind(pass);
        pass.setBindGroup(
            1,
            device.createBindGroup({
                layout: kernel.ownLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: paramsBuffer } },
                    { binding: 1, resource: { buffer: lists.buffer } },
                    { binding: 2, resource: { buffer: results } },
                ],
            }),
        );
        pass.dispatchWorkgroups(Math.ceil(entryCount / 64));
        pass.end();
        device.queue.submit([encoder.finish()]);
        const words = new Uint32Array(await readBuffer(device, results, resultBytes));
        const floats = new Float32Array(words.buffer);
        paramsBuffer.destroy();
        results.destroy();

        let kept = 0;
        let capsaicinHits = 0;
        let platformHits = 0;
        let hitOnlyFromCapsaicin = 0;
        let hitOnlyFromPlatform = 0;
        let otherTriangle = 0;
        let otherNearCapsaicin = 0;
        let otherNearPlatform = 0;
        let backCapsaicin = 0;
        let backPlatform = 0;
        let changedToBackFace = 0;
        let changedToFrontFace = 0;
        let nearCapsaicin = 0;
        let nearPlatform = 0;
        const capsaicinLifts: number[] = [];
        const platformLifts: number[] = [];
        let slideLargest = 0;
        const examples: string[] = [];
        for (let entry = 0; entry < entryCount; entry++) {
            const base = entry * RESULT_WORDS;
            const flags = words[base + RESULT.flags];
            if ((flags & FLAG.kept) === 0) continue;
            kept++;
            capsaicinLifts.push(floats[base + RESULT.capsaicinLift]);
            platformLifts.push(floats[base + RESULT.platformLift]);
            slideLargest = Math.max(slideLargest, floats[base + RESULT.capsaicinSlide]);
            const capsaicinHit = (flags & FLAG.capsaicinHit) !== 0;
            const platformHit = (flags & FLAG.platformHit) !== 0;
            const capsaicinFront = (flags & FLAG.capsaicinFront) !== 0;
            const platformFront = (flags & FLAG.platformFront) !== 0;
            const capsaicinT = floats[base + RESULT.capsaicinT];
            const platformT = floats[base + RESULT.platformT];
            if (capsaicinHit) {
                capsaicinHits++;
                if (!capsaicinFront) backCapsaicin++;
                if (capsaicinT < NEAR_HIT) nearCapsaicin++;
            }
            if (platformHit) {
                platformHits++;
                if (!platformFront) backPlatform++;
                if (platformT < NEAR_HIT) nearPlatform++;
            }
            let changed = false;
            if (capsaicinHit && !platformHit) {
                hitOnlyFromCapsaicin++;
                changed = true;
            } else if (platformHit && !capsaicinHit) {
                hitOnlyFromPlatform++;
                changed = true;
            } else if (
                capsaicinHit &&
                (words[base + RESULT.capsaicinInstance] !== words[base + RESULT.platformInstance] || words[base + RESULT.capsaicinTriangle] !== words[base + RESULT.platformTriangle])
            ) {
                otherTriangle++;
                changed = true;
                if (capsaicinT < NEAR_HIT) otherNearCapsaicin++;
                if (platformT < NEAR_HIT) otherNearPlatform++;
            }
            if (changed && capsaicinHit && platformHit) {
                if (!capsaicinFront && platformFront) changedToBackFace++;
                if (capsaicinFront && !platformFront) changedToFrontFace++;
            }
            if (changed && examples.length < 6) {
                examples.push(
                    `entry ${entry}: Capsaicin's origin ${capsaicinHit ? `t ${capsaicinT.toPrecision(4)} ${capsaicinFront ? 'front' : 'back'}` : 'miss'}, ` +
                        `the platform's ${platformHit ? `t ${platformT.toPrecision(4)} ${platformFront ? 'front' : 'back'}` : 'miss'}`,
                );
            }
        }
        const capsaicinLiftValues = Float32Array.from(capsaicinLifts);
        const platformLiftValues = Float32Array.from(platformLifts);
        return {
            frame: frameIndex,
            firstBounceEntries: entryCount,
            secondBounceRays: kept,
            productionEntries: counters[HASH_GRID_COUNTER.visibilityCount1],
            bucketOverflows: counters[HASH_GRID_COUNTER.bucketOverflowCount],
            hitsFromCapsaicinOrigin: capsaicinHits,
            hitsFromPlatformOrigin: platformHits,
            changedHit: hitOnlyFromCapsaicin + hitOnlyFromPlatform + otherTriangle,
            hitOnlyFromCapsaicin,
            hitOnlyFromPlatform,
            otherTriangle,
            otherTriangleNearHit: { capsaicin: otherNearCapsaicin, platform: otherNearPlatform },
            backFaceHits: { capsaicin: backCapsaicin, platform: backPlatform },
            changedToBackFace,
            changedToFrontFace,
            nearHits: { capsaicin: nearCapsaicin, platform: nearPlatform },
            lift: {
                capsaicinMedian: median(capsaicinLiftValues),
                capsaicinLargest: capsaicinLifts.reduce((largest, value) => Math.max(largest, value), 0),
                capsaicinNotAbove: capsaicinLifts.filter((value) => !(value > 0)).length,
                platformMedian: median(platformLiftValues),
                platformLargest: platformLifts.reduce((largest, value) => Math.max(largest, value), 0),
                capsaicinSlideLargest: slideLargest,
            },
            examples,
        };
    } finally {
        screen.dispose();
        platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}
