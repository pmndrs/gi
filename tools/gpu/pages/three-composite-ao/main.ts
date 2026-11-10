/**
 * Curtained Sponza follow-up (after the curtain investigation): a
 * short-range occlusion term applied to DDGI's GI only, the three-native way
 * (three's GTAO, `GTAONode`) and as rays through the platform's tracer.
 *
 * Why: on Sponza's curtains the light changes inside the folds faster than
 * probes 1 m apart can follow, so DDGI reads +23% against Cycles. A
 * ray-traced occlusion term out to half a probe spacing closed it in the
 * engine-free test view (ddgi-sponza-accuracy, `shortRangeOcclusion`). This
 * page measures the same idea in three's pipeline, and what both cost.
 *
 * Two modes (the job comes from run.ts, `window.threeCompositeAoJob`):
 *
 * - **match**: the scene loaded the viewer's way, every material replaced
 *   by a Lambert material with the constant the GI platform received (the
 *   untextured Cycles reference's materials), see-through meshes hidden (the
 *   reference removes their faces), DDGI converged like the accuracy page
 *   (warm-up passes with a running average, then passes at the normal
 *   hysteresis), then frozen. Returned, at the stored camera:
 *   - the composite's GI term: the composite (T1.14, `ddgi`) minus
 *     the scene pass's own colour, which is `giAlbedo × E / π` per pixel;
 *   - the `giAlbedo` weight (for run.ts to tell materials apart);
 *   - three's GTAO (`ao()`) from the same pass's depth and normal, once per
 *     radius in the job. run.ts multiplies the GI term by it, so the AO
 *     touches GI only, never direct light.
 * - **timing**: Sponza with its own materials at 2560 × 1440, GPU-bound,
 *   short sections, the variants interleaved frame by frame (composite-check.ts `timeInterleaved`):
 *   - three's pipeline: the composite alone, and with GTAO at full and half
 *     resolution feeding the output (the multiply is on the whole colour
 *     here; on GI only it would be the same one multiply);
 *   - the ray-traced term as a compute pass on the platform's tracer: per
 *     pixel (or per 2 × 2 block at half resolution) N short rays from a
 *     surface buffer that primary rays fill beforehand (not timed: a
 *     renderer has depth and normals already).
 *
 * Not a shading path: everything here is measurement.
 */

import { FloatType, MeshLambertMaterial, RenderPipeline, RenderTarget, type Material, type Mesh, type Node, type Object3D } from 'three/webgpu';
import { AccumulatedOcclusion, ssaoOn, upscaledComposite, type UpscalePreset } from '../three-composite-shared/variants.ts';
import { nodeObject, sample, unpackRGBToNormal, uv, vec4 } from 'three/tsl';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import { assert } from '../../client.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { ThreeGI } from '../../../../packages/ddgi-three/src/three-gi.ts';
import { GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT } from '../../../../packages/ddgi-three/src/composite.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { floatsToBase64 } from '../ddgi-shared/harness.ts';
import { runSteps, runThreePage, type ThreePageContext } from '../three-shared/harness.ts';
import {
    CompositeRoutes,
    HdrReader,
    PassClock,
    findManifest,
    shadowCameras,
    shotCamera,
    advanceThreeFrame,
    timeInterleaved,
    type CompositeScene,
    type FrameVariant,
} from '../three-composite-shared/composite-check.ts';

export type CompositeAoScene = 'sponza' | 'sponza-base' | 'cornell-box-point';

export interface CompositeAoJob {
    readonly mode: 'match' | 'timing';
    readonly scene: CompositeAoScene;
    readonly width: number;
    readonly height: number;
    /** GTAO radii (metres, view space) for match mode. */
    readonly radii: readonly number[];
    /** GTAO samples per pixel (three's default 16). */
    readonly samples: number;
    /** GTAO's resolution against the pass's (three's `resolutionScale`; 0.5 is half). Match mode. */
    readonly resolutionScale: number;
    /**
     * Match mode, the occlusion measured (T1.14 cost study, three-composite-shared/variants.ts):
     * `gtao` (three's GTAO, one frame), `gtao-temporal` (GTAO's temporal pattern averaged over
     * `temporalFrames` frames, static camera), or `ssao` (three's SSAO at its default half
     * resolution, with its blur; `resolutionScale` doesn't apply).
     */
    readonly aoKind: 'gtao' | 'gtao-temporal' | 'ssao';
    readonly temporalFrames: number;
    /**
     * Match mode: also the GI term through @pmndrs/upscaler at this preset, at the output size,
     * with GTAO inside the composite at the preset's render scale (one image per radius, at
     * `resolutionScale` of the render scale), after `upscaleFrames` frames on a still camera.
     */
    readonly upscale: UpscalePreset | null;
    readonly upscaleFrames: number;
    /** Full DDGI passes with a running average, then at the normal hysteresis (as ddgi-sponza-accuracy). */
    readonly warmupPasses: number;
    readonly passes: number;
    /** Timing: GTAO's radius (it doesn't change the cost; the sample count does). */
    readonly timingRadius: number;
    /** Timing: which section this run times (each its own run, so each starts from a cool GPU). */
    readonly timingPart: 'gtao' | 'rays';
    /** Timing: ray counts of the ray-traced term, each at full and half resolution. */
    readonly timingRays: readonly number[];
    /** Timing: the ray term's reach, metres (half the 1 m probe spacing in the accuracy runs). */
    readonly rayRadius: number;
    readonly rounds: number;
    /**
     * Light point and spot lights with no range window (three's `distance = 0`),
     * as Cycles does and as the test view's `ignoreLightRange` does. The viewer's
     * builder gives three lights the manifest's range (Cornell point: 20 m),
     * whose window dims the light the probes see by a few per cent.
     */
    readonly ignoreLightRange: boolean;
}

const DEFAULT_JOB: CompositeAoJob = {
    mode: 'match',
    scene: 'sponza',
    width: 640,
    height: 360,
    radii: [0.1, 0.25, 0.5],
    samples: 16,
    resolutionScale: 1,
    aoKind: 'gtao',
    temporalFrames: 16,
    upscale: null,
    upscaleFrames: 32,
    warmupPasses: 16,
    passes: 64,
    timingRadius: 0.25,
    timingPart: 'gtao',
    timingRays: [4, 8],
    rayRadius: 0.5,
    rounds: 5,
    ignoreLightRange: true,
};

declare global {
    interface Window {
        threeCompositeAoJob?: Partial<CompositeAoJob>;
    }
}

const job: CompositeAoJob = { ...DEFAULT_JOB, ...window.threeCompositeAoJob };

/** The DDGI technique's default `maxProbesPerFrame`. */
const MAX_PROBES_PER_FRAME = 4096;
const TIMING_SIZE = { width: 2560, height: 1440 } as const;
/** Idle seconds after loading (which heats the GPU) and before a timed section. */
const COOL_SECONDS = 30;

interface LoadedAoScene extends CompositeScene {
    readonly ddgi: DDGITechnique;
}

/**
 * The viewer's loader, DDGI on the manifest's volume through ThreeGI,
 * converged the accuracy page's way, then frozen. `sponza-base` is Sponza
 * with only its base package (no curtains), as the `-base` Cycles reference.
 */
async function loadAoScene(context: ThreePageContext, width: number, height: number): Promise<LoadedAoScene> {
    const { device, log } = context;
    const full = findManifest(job.scene === 'sponza-base' ? 'sponza' : job.scene);
    const manifest = job.scene === 'sponza-base' ? { ...full, parts: full.parts.filter((part) => part.kind !== 'model' || part.asset === 'sponza-intel') } : full;
    const loaded = await loadScene(manifest, () => {});
    const scene = loaded.scene;
    if (job.ignoreLightRange) {
        scene.traverse((object) => {
            const light = object as Object3D & { isPointLight?: boolean; isSpotLight?: boolean; distance?: number };
            if (light.isPointLight || light.isSpotLight) light.distance = 0;
        });
    }
    // The sky dome is for viewing; GI takes the sky from the platform environment.
    const content = new Set<Object3D>();
    loaded.content.traverse((object) => content.add(object));
    scene.traverse((object) => {
        if ((object as Mesh).isMesh && !content.has(object)) object.userData.giContributes = false;
    });
    const camera = shotCamera(manifest, width, height);
    const platform = new GIPlatformImpl(device, { label: `three composite ao ${job.scene}` });
    const volume = manifest.volume;
    const ddgi = platform.addTechnique(new DDGITechnique({ volume: { center: volume.center, size: volume.size, spacing: volume.probeSpacing }, warmUpPasses: 0 }));
    platform.setOutput(ddgi);
    await platform.prepare();
    const environment = manifest.environment.type === 'procedural' ? platformSkyEnvironment(manifest) : null;
    const threeGI = new ThreeGI(platform, scene, { budget: { units: 0 }, environment });
    threeGI.update(camera);
    await platform.rays.whenBuilt();
    threeGI.update(camera);

    const field = ddgi.field;
    assert(field, 'DDGI has no field');
    const probeCount = field.grid.probeCount;
    // Whole passes split into equal frames, as ddgi-sponza-accuracy.
    const framesPerPass = Math.ceil(probeCount / MAX_PROBES_PER_FRAME);
    threeGI.budget = { units: Math.ceil(probeCount / framesPerPass) };
    const normalHysteresis = ddgi.settings.hysteresis;
    const runPasses = (passes: number) => runSteps(device, passes * framesPerPass, () => threeGI.update(camera, 1 / 60));
    for (let pass = 0; pass < job.warmupPasses; pass++) {
        // Pass k keeps k/(k+1): the field is the plain mean of the passes so far.
        ddgi.settings.hysteresis = Math.min(normalHysteresis, pass / (pass + 1));
        await runPasses(1);
    }
    ddgi.settings.hysteresis = normalHysteresis;
    await runPasses(job.passes);
    threeGI.budget = { paused: true };
    log(`${job.scene}: ${probeCount} probes, ${job.warmupPasses} + ${job.passes} passes, then frozen`);
    return { manifest, loaded, scene, camera, platform, threeGI, ddgi };
}

/**
 * Every material becomes a Lambert material with the constant the GI
 * platform received, so three's indirect-diffuse weight is exactly the GI
 * albedo (the reference's materials, no specular share). See-through meshes,
 * which the adapter left out of GI and the reference removed, are hidden.
 */
function useLambertGiTable(scene: LoadedAoScene): { replaced: number; hidden: number } {
    const replacements = new Map<Material, MeshLambertMaterial>();
    let hidden = 0;
    const replace = (source: Material): Material => {
        const existing = replacements.get(source);
        if (existing) return existing;
        const descriptor = scene.threeGI.sync.sentMaterial(source);
        if (!descriptor) return source;
        const albedo = descriptor.albedo ?? [1, 1, 1];
        const emissive = descriptor.emissive ?? [0, 0, 0];
        const intensity = descriptor.emissiveIntensity ?? 1;
        const material = new MeshLambertMaterial({ name: `${source.name} (GI table, Lambert)`, side: source.side });
        // Linear values: three's working colour space.
        material.color.setRGB(albedo[0], albedo[1], albedo[2]);
        material.emissive.setRGB(emissive[0] * intensity, emissive[1] * intensity, emissive[2] * intensity);
        replacements.set(source, material);
        return material;
    };
    scene.loaded.content.traverse((object) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh) return;
        if (scene.threeGI.sync.sentSeeThrough(mesh) === true) {
            mesh.visible = false;
            hidden++;
            return;
        }
        mesh.material = Array.isArray(mesh.material) ? mesh.material.map(replace) : replace(mesh.material);
    });
    return { replaced: replacements.size, hidden };
}

/** three's GTAO on a composite pass's own depth and (packed) normal. */
function gtaoFor(routes: CompositeRoutes, scene: CompositeScene) {
    const normalTexture = routes.compositePass.getTextureNode(GI_NORMAL_OUTPUT);
    const normal = sample((coordinate) => unpackRGBToNormal(normalTexture.sample(coordinate).rgb));
    const node = ao(routes.compositePass.getTextureNode('depth'), normal, scene.camera);
    node.samples.value = job.samples;
    node.resolutionScale = job.resolutionScale;
    return node;
}

/** rgb of an rgba readback, as floats (for the report). */
function rgbOf(rgba: Float32Array): Float32Array {
    const rgb = new Float32Array((rgba.length / 4) * 3);
    for (let pixel = 0; pixel < rgba.length / 4; pixel++) for (let channel = 0; channel < 3; channel++) rgb[pixel * 3 + channel] = rgba[pixel * 4 + channel];
    return rgb;
}

async function match(context: ThreePageContext): Promise<void> {
    const { renderer, data, log } = context;
    const { width, height } = job;
    const scene = await loadAoScene(context, width, height);
    data.materials = useLambertGiTable(scene);
    renderer.setSize(width, height, false);
    const routes = new CompositeRoutes(scene);
    const gtao = gtaoFor(routes, scene);
    // The occlusion read back per radius: GTAO itself, GTAO averaged over frames, or SSAO.
    const temporal = job.aoKind === 'gtao-temporal' ? new AccumulatedOcclusion(gtao, job.temporalFrames) : null;
    const ssaoNode = job.aoKind === 'ssao' ? (ssaoOn(routes.compositePass) as unknown as { radius: { value: number }; getTextureNode(): ReturnType<typeof gtao.getTextureNode> }) : null;
    const reader = new HdrReader(renderer, width, height);
    await routes.compositePass.compileAsync(renderer);
    await reader.warm(routes.composite as unknown as Node, 4);
    const composite = await reader.read(routes.composite as unknown as Node);
    // The same frame's textures: the pass's colour before the composite, and the GI weight.
    const passColour = await reader.read(routes.compositePass.getTextureNode('output') as unknown as Node, false);
    const weight = await reader.read(routes.compositePass.getTextureNode(GI_ALBEDO_OUTPUT) as unknown as Node, false);
    const giTerm = new Float32Array(composite.length);
    for (let index = 0; index < composite.length; index++) giTerm[index] = composite[index] - passColour[index];
    data.giTermBase64 = floatsToBase64(rgbOf(giTerm));
    data.weightBase64 = floatsToBase64(rgbOf(weight));
    const aoImages: Record<string, string> = {};
    const aoTexture = temporal ? temporal.getTextureNode() : ssaoNode ? ssaoNode.getTextureNode() : gtao.getTextureNode();
    const aoView = vec4(aoTexture.sample(uv()).r, aoTexture.sample(uv()).r, aoTexture.sample(uv()).r, 1) as unknown as Node;
    // An average needs its frames; one frame of GTAO or SSAO needs two to compile and draw.
    const aoFrames = temporal ? job.temporalFrames + 2 : 2;
    for (const radius of job.radii) {
        gtao.radius.value = radius;
        if (ssaoNode) ssaoNode.radius.value = radius;
        await reader.warm(aoView, aoFrames);
        const image = await reader.read(aoView);
        aoImages[String(radius)] = floatsToBase64(rgbOf(image));
        let sum = 0;
        let count = 0;
        for (let pixel = 0; pixel < width * height; pixel++) {
            if (weight[pixel * 4] + weight[pixel * 4 + 1] + weight[pixel * 4 + 2] <= 0) continue;
            sum += image[pixel * 4];
            count++;
        }
        log(`GTAO radius ${radius} m: mean AO over GI pixels ${(sum / count).toFixed(4)}`);
    }
    data.aoBase64 = aoImages;
    data.aoKind = job.aoKind;
    reader.dispose();
    if (job.upscale) data.upscaledBase64 = await upscaledGiTerms(context, scene, job.upscale, width, height);
}

/**
 * The GI term at the output size through @pmndrs/upscaler (variants.ts, `upscaledComposite`
 * with `giTermOnly`), with GTAO inside at each radius: the upscaler's own pipeline, its jitter
 * and history, `upscaleFrames` frames on a still camera, read back in linear HDR.
 */
async function upscaledGiTerms(context: ThreePageContext, scene: CompositeScene, preset: UpscalePreset, width: number, height: number): Promise<Record<string, string>> {
    const { renderer, device, log } = context;
    const images: Record<string, string> = {};
    const target = new RenderTarget(width, height, { type: FloatType, depthBuffer: false });
    const variants: Array<[string, false | { resolutionScale: number; radius: number; samples: number }]> = [
        ['no AO', false],
        ...job.radii.map((radius) => [`GTAO ${radius} m`, { resolutionScale: job.resolutionScale, radius, samples: job.samples }] as [string, { resolutionScale: number; radius: number; samples: number }]),
    ];
    for (const [name, occlusion] of variants) {
        const built = upscaledComposite(scene.scene, scene.camera, scene.threeGI, preset, { occlusion, giTermOnly: true });
        const pipeline = new RenderPipeline(renderer, built.node as unknown as Node);
        // Linear HDR out: the readback compares radiance, not a tone-mapped picture.
        pipeline.outputColorTransform = false;
        for (let frame = 0; frame < job.upscaleFrames; frame++) {
            advanceThreeFrame(renderer);
            renderer.setRenderTarget(target);
            pipeline.render();
            renderer.setRenderTarget(null);
            if (frame % 4 === 3) await device.queue.onSubmittedWorkDone();
        }
        const pixels = new Float32Array((await renderer.readRenderTargetPixelsAsync(target, 0, 0, width, height)) as Float32Array);
        images[`upscaled ${preset}, ${name}`] = floatsToBase64(rgbOf(pixels));
        log(`upscaled GI term (${preset}, ${name}): ${job.upscaleFrames} frames`);
        pipeline.dispose();
    }
    target.dispose();
    return images;
}

// -- timing ------------------------------------------------------------------------------------

const RAY_TERM_WGSL = /* wgsl */ `
struct AoTimingSettings {
    position: vec4f,
    forward: vec4f,   // xyz: unit forward; w: tan(fovY / 2)
    right: vec4f,     // xyz: unit right; w: width / height
    up: vec4f,
    size: vec4u,      // full width, height; stride (1 full, 2 half); rays per pixel
    reach: vec4f,     // x: ray length, metres
}
@group(1) @binding(0) var<uniform> aoTiming: AoTimingSettings;
// Per full-resolution pixel: position (w 1 for a surface) and normal.
@group(1) @binding(1) var<storage, read_write> aoSurface: array<vec4f>;
@group(1) @binding(2) var<storage, read_write> aoFactor: array<f32>;

const AO_GOLDEN_ANGLE: f32 = 2.39996322972865;

// Not timed: what a renderer's depth and normal targets already hold.
@compute @workgroup_size(8, 8)
fn aoPrimary(@builtin(global_invocation_id) id: vec3u) {
    let size = aoTiming.size.xy;
    if (any(id.xy >= size)) {
        return;
    }
    let ndc = vec2f((f32(id.x) + 0.5) / f32(size.x) * 2.0 - 1.0, 1.0 - (f32(id.y) + 0.5) / f32(size.y) * 2.0);
    let direction = normalize(aoTiming.forward.xyz + aoTiming.right.xyz * ndc.x * aoTiming.forward.w * aoTiming.right.w + aoTiming.up.xyz * ndc.y * aoTiming.forward.w);
    let ray = GiRay(aoTiming.position.xyz, 0.0, direction, 1.0e27);
    let hit = giTraceClosest(ray);
    let pixel = id.y * size.x + id.x;
    if (hit.t < 0.0) {
        aoSurface[2u * pixel] = vec4f(0.0);
        return;
    }
    let surface = giSurfaceAt(ray, hit);
    aoSurface[2u * pixel] = vec4f(surface.position, 1.0);
    aoSurface[2u * pixel + 1u] = vec4f(surface.normal, 0.0);
}

// Timed: the short-range term, as the accuracy runs measured it (cosine spiral,
// each hit takes away 1 − its albedo), for one pixel of each stride × stride block.
@compute @workgroup_size(8, 8)
fn aoRays(@builtin(global_invocation_id) id: vec3u) {
    let stride = aoTiming.size.z;
    let outputSize = (aoTiming.size.xy + stride - 1u) / stride;
    if (any(id.xy >= outputSize)) {
        return;
    }
    let source = id.xy * stride;
    let pixel = source.y * aoTiming.size.x + source.x;
    let positionAndValid = aoSurface[2u * pixel];
    var factor = 1.0;
    if (positionAndValid.w > 0.0) {
        let position = positionAndValid.xyz;
        let normal = aoSurface[2u * pixel + 1u].xyz;
        let helper = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(normal.x) > 0.9);
        let tangent = normalize(cross(helper, normal));
        let bitangent = cross(normal, tangent);
        let origin = giShadeOffsetRayOrigin(position, normal);
        let rotation = f32(bitcast<u32>(position.x * 7.0 + position.y * 13.0 + position.z * 17.0) & 1023u) / 1024.0 * 6.28318530717959;
        let rays = aoTiming.size.w;
        var occlusion = 0.0;
        for (var index = 0u; index < rays; index++) {
            let u = (f32(index) + 0.5) / f32(rays);
            let angle = f32(index) * AO_GOLDEN_ANGLE + rotation;
            let discRadius = sqrt(u);
            let direction = normalize(tangent * (discRadius * cos(angle)) + bitangent * (discRadius * sin(angle)) + normal * sqrt(max(0.0, 1.0 - u)));
            let ray = GiRay(origin, 0.0, direction, aoTiming.reach.x);
            let hit = giTraceClosest(ray);
            if (hit.t >= 0.0) {
                let albedo = giSurfaceAt(ray, hit).albedo;
                occlusion += 1.0 - (albedo.r + albedo.g + albedo.b) / 3.0;
            }
        }
        factor = 1.0 - occlusion / f32(rays);
    }
    aoFactor[id.y * outputSize.x + id.x] = factor;
}
`;

const RAY_TERM_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ray term timing',
    entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
};

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/**
 * GPU time of the ray-traced term per frame: for each variant (rays,
 * stride), bursts of DISPATCHES_PER_SAMPLE dispatches in one timed compute
 * pass, variants alternating each round. Short: a few seconds in all.
 */
async function timeRayTerm(context: ThreePageContext, scene: LoadedAoScene): Promise<Record<string, number>> {
    const { device, log } = context;
    assert(device.features.has('timestamp-query'), 'timing needs timestamp-query');
    const { width, height } = TIMING_SIZE;
    const make = (entryPoint: string) =>
        scene.platform.kernels.computeKernel({ label: `ray term ${entryPoint}`, modules: [scene.platform.shade.module], code: RAY_TERM_WGSL, entryPoint, ownGroups: [RAY_TERM_GROUP] });
    const [primaryKernel, raysKernel] = await Promise.all([make('aoPrimary'), make('aoRays')]);
    const surface = device.createBuffer({ label: 'ray term surface', size: width * height * 32, usage: GPUBufferUsage.STORAGE });
    const factor = device.createBuffer({ label: 'ray term factor', size: width * height * 4, usage: GPUBufferUsage.STORAGE });
    const settings = device.createBuffer({ label: 'ray term settings', size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bookmark = scene.manifest.cameras[0];
    const forward = bookmark.target.map((value, axis) => value - bookmark.position[axis]);
    const forwardLength = Math.hypot(...forward);
    const f = forward.map((value) => value / forwardLength);
    const rightRaw = [-f[2], 0, f[0]]; // cross(forward, up) with up = +y
    const rightLength = Math.hypot(...rightRaw);
    const r = rightRaw.map((value) => value / rightLength);
    const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
    const writeSettings = (stride: number, rays: number) => {
        const bytes = new ArrayBuffer(96);
        new Float32Array(bytes, 0, 16).set([...bookmark.position, 0, ...f, Math.tan((bookmark.fovYDeg * Math.PI) / 360), ...r, width / height, ...u, 0]);
        new Uint32Array(bytes, 64, 4).set([width, height, stride, rays]);
        new Float32Array(bytes, 80, 4).set([job.rayRadius, 0, 0, 0]);
        device.queue.writeBuffer(settings, 0, bytes);
    };
    const group = (kernel: typeof raysKernel) =>
        device.createBindGroup({
            layout: kernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: settings } },
                { binding: 1, resource: { buffer: surface } },
                { binding: 2, resource: { buffer: factor } },
            ],
        });
    writeSettings(1, 1);
    {
        const encoder = device.createCommandEncoder({ label: 'ray term surface' });
        const pass = encoder.beginComputePass();
        primaryKernel.bind(pass);
        pass.setBindGroup(1, group(primaryKernel));
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
    }
    const raysGroup = group(raysKernel);
    const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
    const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    // One dispatch per sample: a full-resolution dispatch takes tens of milliseconds on Sponza, and the
    // whole section must stay short (the laptop RTX throttles after about 20 s of load).
    const DISPATCHES_PER_SAMPLE = 1;
    const sampleMs = async (stride: number, rays: number): Promise<number> => {
        writeSettings(stride, rays);
        const encoder = device.createCommandEncoder({ label: 'ray term timed' });
        const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
        raysKernel.bind(pass);
        pass.setBindGroup(1, raysGroup);
        const groupsX = Math.ceil(Math.ceil(width / stride) / 8);
        const groupsY = Math.ceil(Math.ceil(height / stride) / 8);
        for (let index = 0; index < DISPATCHES_PER_SAMPLE; index++) pass.dispatchWorkgroups(groupsX, groupsY);
        pass.end();
        encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
        encoder.copyBufferToBuffer(resolve, 0, readback, 0, 16);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const [start, end] = new BigUint64Array(readback.getMappedRange().slice(0));
        readback.unmap();
        return Number(end - start) / 1e6 / DISPATCHES_PER_SAMPLE;
    };
    const variants = job.timingRays.flatMap((rays) => [1, 2].map((stride) => ({ id: `rays${rays}-${stride === 1 ? 'full' : 'half'}`, rays, stride })));
    // Compile and ramp: every variant a few times, then timed rounds.
    for (const variant of variants) for (let index = 0; index < 1; index++) await sampleMs(variant.stride, variant.rays);
    const perVariant = new Map<string, number[]>(variants.map((variant) => [variant.id, []]));
    const start = performance.now();
    for (let round = 0; round < job.rounds; round++) {
        const order = round % 2 === 0 ? variants : [...variants].reverse();
        for (const variant of order) for (let index = 0; index < 2; index++) perVariant.get(variant.id)!.push(await sampleMs(variant.stride, variant.rays));
    }
    log(`ray term timed section took ${((performance.now() - start) / 1000).toFixed(1)} s`);
    const result: Record<string, number> = {};
    for (const variant of variants) {
        result[variant.id] = median(perVariant.get(variant.id)!);
        log(`  ray term ${variant.id}: ${result[variant.id].toFixed(3)} ms per frame`);
    }
    for (const buffer of [surface, factor, settings, resolve, readback]) buffer.destroy();
    querySet.destroy();
    return result;
}

async function timing(context: ThreePageContext): Promise<void> {
    const { renderer, device, data, log } = context;
    renderer.shadowMap.enabled = true;
    const scene = await loadAoScene(context, TIMING_SIZE.width, TIMING_SIZE.height);
    if (job.timingPart === 'rays') {
        // Loading heated the GPU; idle before the short timed section.
        log(`cooling for ${COOL_SECONDS} s before the ray term`);
        await new Promise((resolve) => setTimeout(resolve, COOL_SECONDS * 1000));
        data.rayTerm = await timeRayTerm(context, scene);
        return;
    }
    renderer.setSize(TIMING_SIZE.width, TIMING_SIZE.height, false);
    const routes = new CompositeRoutes(scene);
    const composite = new RenderPipeline(renderer, routes.composite as unknown as Node);
    const withGtao = (resolutionScale: number) => {
        const gtao = gtaoFor(routes, scene);
        gtao.radius.value = job.timingRadius;
        gtao.resolutionScale = resolutionScale;
        const occlusion = gtao.getTextureNode().sample(uv()).r;
        // Consumed by the output so GTAO runs each frame (see the file comment).
        const colour = nodeObject(routes.composite as unknown as Node) as unknown as ReturnType<typeof vec4>;
        const output = vec4(colour.rgb.mul(occlusion), colour.a) as unknown as Node;
        return new RenderPipeline(renderer, output);
    };
    const gtaoFull = withGtao(1);
    const gtaoHalf = withGtao(0.5);
    const variants: FrameVariant[] = [
        { id: 'composite', apply: () => {}, draw: () => composite.render() },
        { id: 'composite+gtao', apply: () => {}, draw: () => gtaoFull.render() },
        { id: 'composite+gtao-half', apply: () => {}, draw: () => gtaoHalf.render() },
    ];
    // Compile every variant, then idle: loading the scene heated the GPU.
    for (const variant of variants) {
        for (let frame = 0; frame < 6; frame++) {
            advanceThreeFrame(renderer);
            variant.draw();
            await device.queue.onSubmittedWorkDone();
        }
    }
    log(`cooling for ${COOL_SECONDS} s before timing`);
    await new Promise((resolve) => setTimeout(resolve, COOL_SECONDS * 1000));
    const clock = new PassClock(renderer, shadowCameras(scene));
    const results = await timeInterleaved(
        renderer,
        device,
        clock,
        variants,
        [
            { name: 'gtao full', minuend: 'composite+gtao', subtrahend: 'composite', part: 'total' },
            { name: 'gtao half', minuend: 'composite+gtao-half', subtrahend: 'composite', part: 'total' },
        ],
        { label: 'GTAO on the composite', seconds: 10, cyclesPerBatch: 4, rampSeconds: 2, log },
    );
    const gtao = {
        fullMs: results.deltas['gtao full'].median,
        halfMs: results.deltas['gtao half'].median,
        samples: job.samples,
    };
    log(`GTAO adds ${gtao.fullMs.toFixed(3)} ms at full resolution, ${gtao.halfMs.toFixed(3)} ms at half (${job.samples} samples)`);
    data.gtao = { ...gtao, variants: results };
    for (const pipeline of [composite, gtaoFull, gtaoHalf]) pipeline.dispose();
}

await runThreePage(
    async (context) => {
        context.data.job = job;
        if (job.mode === 'match') await match(context);
        else await timing(context);
        context.log('done');
    },
    job.mode === 'match' ? { width: job.width, height: job.height } : TIMING_SIZE,
    { trackTimestamp: true },
);
