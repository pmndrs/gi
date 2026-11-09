/**
 * Shared by the `three-composite` (Cornell) and `three-composite-sponza`
 * pages (T1.14): a scene loaded the viewer's way with DDGI through the three
 * adapter, the two ways of showing GI side by side (per fragment, and the
 * composite on three's `pass()` and `mrt()`), HDR images of both, the match
 * statistics, and a GPU timer for whole frames.
 *
 * Not a page itself (no index.html).
 *
 * **The stored cameras.** The Cycles references store their camera (T1.13,
 * `ddgi-reference-shot/1`): each scene's first bookmark at 1920 × 1080, near
 * 0.01, far 2000. The pages use the same camera. Sponza's HD reference
 * isn't rendered yet, so its stored shot is the bookmark by definition.
 */

import { FloatType, MeshStandardMaterial, NodeMaterial, PerspectiveCamera, QuadMesh, RenderTarget, UnsignedByteType, Vector2 } from 'three/webgpu';
import type { Camera, Material, Mesh, Node, Object3D, Scene, WebGPURenderer } from 'three/webgpu';
import { type context, isBackgroundDepth, mrt, normalView, output, packNormalToRGB, pass, uv, vec4 } from 'three/tsl';
import type { MaterialDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { ThreeGI } from '../../../../packages/ddgi-three/src/three-gi.ts';
import { DDGICompositeNode, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT, type DDGICompositeOptions } from '../../../../packages/ddgi-three/src/composite.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { loadScene, type LoadedScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { runSteps } from '../three-shared/harness.ts';

/** The stored shot's camera planes (render.py, T1.13). */
export const SHOT_NEAR = 0.01;
export const SHOT_FAR = 2000;
/** The stored shot's size (render.py's default, T1.13). */
export const SHOT_SIZE = { width: 1920, height: 1080 } as const;

// -- the scene ---------------------------------------------------------------------------------

export interface CompositeSceneOptions {
    readonly width: number;
    readonly height: number;
    /** GI frames before the field is frozen. */
    readonly giFrames: number;
    /** Probe updates per GI frame (a fixed `units` budget, so the run doesn't depend on the budget controller). */
    readonly probesPerFrame: number;
    readonly log: (message: string) => void;
}

export interface CompositeScene {
    readonly manifest: SceneManifest;
    readonly loaded: LoadedScene;
    readonly scene: Scene;
    readonly camera: PerspectiveCamera;
    readonly platform: GIPlatformImpl;
    readonly threeGI: ThreeGI;
}

export function findManifest(sceneId: string): SceneManifest {
    const manifest = loadAllManifests().find((candidate) => candidate.id === sceneId);
    if (!manifest) throw new Error(`unknown scene '${sceneId}'`);
    return manifest;
}

/** The first bookmark, as the stored shot has it. */
export function shotCamera(manifest: SceneManifest, width: number, height: number): PerspectiveCamera {
    const bookmark = manifest.cameras[0];
    const camera = new PerspectiveCamera(bookmark.fovYDeg, width / height, SHOT_NEAR, SHOT_FAR);
    camera.position.set(...bookmark.position);
    camera.lookAt(...bookmark.target);
    camera.updateMatrixWorld(true);
    return camera;
}

/**
 * Loads the scene with the viewer's loader, runs DDGI on the manifest's
 * volume through `ThreeGI` for `giFrames` frames, then freezes the field
 * (`paused`), so every image after this shades the same field.
 */
export async function loadCompositeScene(device: GPUDevice, sceneId: string, options: CompositeSceneOptions): Promise<CompositeScene> {
    const manifest = findManifest(sceneId);
    const loaded = await loadScene(manifest, () => {});
    const scene = loaded.scene;
    // The sky dome is for viewing; GI takes the sky from the platform environment.
    const content = new Set<Object3D>();
    loaded.content.traverse((object) => content.add(object));
    scene.traverse((object) => {
        if ((object as Mesh).isMesh && !content.has(object)) object.userData.giContributes = false;
    });
    const camera = shotCamera(manifest, options.width, options.height);
    const platform = new GIPlatformImpl(device, { label: `three composite ${sceneId}` });
    const volume = manifest.volume;
    platform.setOutput(platform.addTechnique(new DDGITechnique({ volume: { center: volume.center, size: volume.size, spacing: volume.probeSpacing }, warmUpPasses: 0 })));
    await platform.prepare();
    const environment = manifest.environment.type === 'procedural' ? platformSkyEnvironment(manifest) : null;
    const threeGI = new ThreeGI(platform, scene, { budget: { units: 0 }, environment });
    threeGI.update(camera);
    await platform.rays.whenBuilt();
    threeGI.update(camera);
    threeGI.budget = { units: options.probesPerFrame };
    await runSteps(device, options.giFrames, () => threeGI.update(camera, 1 / 60));
    threeGI.budget = { paused: true };
    options.log(`${sceneId}: ${loaded.stats.triangles} triangles; ${options.giFrames} GI frames of ${options.probesPerFrame} probes; coverage per pixel ${threeGI.light.computeCoverage}`);
    return { manifest, loaded, scene, camera, platform, threeGI };
}

/**
 * Replaces every material of the model with the constant one the GI
 * platform received (SceneSync.sentMaterial: albedo = colour ×
 * (1 − metalness), emission; since T4.3 without textures, which GI now
 * samples itself), as the Sponza accuracy page's GI
 * material table and the untextured Cycles reference use. Roughness keeps
 * the material's own factor; side, transparency and opacity are kept.
 * Returns how many materials were replaced.
 */
export function useGiMaterialTable(scene: CompositeScene): number {
    const replacements = new Map<Material, MeshStandardMaterial>();
    const replace = (source: Material): Material => {
        const existing = replacements.get(source);
        if (existing) return existing;
        const descriptor: MaterialDescriptor | undefined = scene.threeGI.sync.sentMaterial(source);
        if (!descriptor) return source;
        // Since T4.3 the adapter sends metalness apart (the platform applies it): fold it here, as the platform does without a map.
        const diffuse = 1 - (descriptor.metalness ?? 0);
        const albedo = (descriptor.albedo ?? [1, 1, 1]).map((channel) => channel * diffuse);
        const emissive = descriptor.emissive ?? [0, 0, 0];
        const intensity = descriptor.emissiveIntensity ?? 1;
        const lit = source as Material & { roughness?: number };
        const material = new MeshStandardMaterial({
            name: `${source.name} (GI table)`,
            roughness: typeof lit.roughness === 'number' ? lit.roughness : 1,
            metalness: 0,
            side: source.side,
            transparent: source.transparent,
            opacity: source.opacity,
        });
        // Linear values: three's working colour space.
        material.color.setRGB(albedo[0], albedo[1], albedo[2]);
        material.emissive.setRGB(emissive[0] * intensity, emissive[1] * intensity, emissive[2] * intensity);
        replacements.set(source, material);
        return material;
    };
    scene.loaded.content.traverse((object) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh) return;
        mesh.material = Array.isArray(mesh.material) ? mesh.material.map(replace) : replace(mesh.material);
    });
    return replacements.size;
}

// -- three's frame -----------------------------------------------------------------------------

/** The parts of three's renderer its animation loop touches to start a frame (`Animation.js`, r187). */
interface FrameAdvance {
    info: { autoReset: boolean; reset(): void; frame: number; render: { drawCalls: number } };
    _nodes: { nodeFrame: { update(): void; frameId: number } };
}

/**
 * Starts a new three frame the way its animation loop does. Passes update
 * once per frame (`PassNode.updateBeforeType = FRAME`), and three's frame
 * counter only advances inside `setAnimationLoop` (NOTES), so pages that
 * draw frames themselves call this first.
 */
export function advanceThreeFrame(renderer: WebGPURenderer): void {
    const frame = renderer as unknown as FrameAdvance;
    if (frame.info.autoReset) frame.info.reset();
    frame._nodes.nodeFrame.update();
    frame.info.frame = frame._nodes.nodeFrame.frameId;
}

// -- the two ways of showing GI ---------------------------------------------------------------

/**
 * The scene passes of both routes, built the way a user builds them:
 * - `perFragmentPass`: `pass(scene, camera)`, GI through the light per fragment (today's path);
 * - `opaquePass`: the same without transparent objects, to find the pixels transparent objects cover;
 * - `compositePass`: `pass()` with `mrt({ output, giAlbedo, normal })`, and `composite = ddgi(...)`.
 */
export class CompositeRoutes {
    readonly perFragmentPass;
    readonly opaquePass;
    readonly compositePass;
    readonly composite: DDGICompositeNode;
    /** Per pixel: 1 where a surface is (not the cleared background), the giAlbedo weight's luminance, its alpha, and the linear depth (0 near, 1 far). */
    readonly coverage: Node;

    /**
     * `byteTargets`: store `giAlbedo` and the normal as 8-bit unorm instead of
     * the pass's half float, the bandwidth saving three's SSGI example makes.
     * `occlusion`: the composite's short-range occlusion on GI. Off unless
     * asked for, because the match checks compare the composite with GI per
     * fragment, which has none.
     * `camera`: the camera the passes render with (default the scene's shot camera).
     * `context`: a pass context for all three passes (three's `builtinAOContext`, say), set before `ddgi()`.
     */
    constructor(
        scene: CompositeScene,
        options: {
            readonly byteTargets?: boolean;
            readonly occlusion?: DDGICompositeOptions['occlusion'];
            readonly camera?: Camera;
            readonly context?: ReturnType<typeof context>;
        } = {},
    ) {
        const camera = options.camera ?? scene.camera;
        this.perFragmentPass = pass(scene.scene, camera);
        this.opaquePass = pass(scene.scene, camera);
        this.opaquePass.transparent = false;
        this.compositePass = pass(scene.scene, camera);
        if (options.context) {
            for (const scenePass of [this.perFragmentPass, this.opaquePass, this.compositePass]) scenePass.contextNode = options.context;
        }
        this.compositePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView) }));
        if (options.byteTargets) {
            this.compositePass.getTexture(GI_ALBEDO_OUTPUT).type = UnsignedByteType;
            this.compositePass.getTexture(GI_NORMAL_OUTPUT).type = UnsignedByteType;
        }
        this.composite = new DDGICompositeNode(this.compositePass, scene.threeGI, { occlusion: options.occlusion ?? false });
        const depth = this.compositePass.getTextureNode('depth').sample(uv()).r;
        const weight = this.compositePass.getTextureNode(GI_ALBEDO_OUTPUT).sample(uv());
        const surface = isBackgroundDepth(depth).select(0, 1);
        const linearDepth = this.compositePass.getLinearDepthNode();
        this.coverage = vec4(surface, weight.r.mul(0.2126).add(weight.g.mul(0.7152)).add(weight.b.mul(0.0722)), weight.a, linearDepth) as unknown as Node;
    }
}

/** Draws nodes full screen into float targets and reads them back: linear HDR, before any tone mapping. */
export class HdrReader {
    private readonly quads = new Map<Node, QuadMesh>();
    readonly target: RenderTarget;

    constructor(
        private readonly renderer: WebGPURenderer,
        readonly width: number,
        readonly height: number,
    ) {
        this.target = new RenderTarget(width, height, { type: FloatType, depthBuffer: false });
    }

    private quadFor(node: Node): QuadMesh {
        let quad = this.quads.get(node);
        if (!quad) {
            const material = new NodeMaterial();
            material.fragmentNode = node;
            material.name = 'hdr readback';
            quad = new QuadMesh(material);
            this.quads.set(node, quad);
        }
        return quad;
    }

    /**
     * One frame of `node` (passes it reads render first), read back as rgba floats.
     * `newFrame` false reads the passes' textures as the last frame left them.
     */
    async read(node: Node, newFrame = true): Promise<Float32Array> {
        if (newFrame) advanceThreeFrame(this.renderer);
        const quad = this.quadFor(node);
        const previous = this.renderer.getRenderTarget();
        this.renderer.setRenderTarget(this.target);
        quad.render(this.renderer);
        this.renderer.setRenderTarget(previous);
        const pixels = await this.renderer.readRenderTargetPixelsAsync(this.target, 0, 0, this.width, this.height);
        return new Float32Array(pixels as Float32Array);
    }

    /** Compiles and draws a few frames, so the pipelines are ready before the images that count. */
    async warm(node: Node, frames = 3): Promise<void> {
        for (let frame = 0; frame < frames; frame++) await this.read(node);
    }

    dispose(): void {
        this.target.dispose();
        for (const quad of this.quads.values()) (quad.material as Material).dispose();
    }
}

// -- statistics --------------------------------------------------------------------------------

export function luminanceAt(pixels: Float32Array, pixel: number): number {
    return 0.2126 * pixels[pixel * 4] + 0.7152 * pixels[pixel * 4 + 1] + 0.0722 * pixels[pixel * 4 + 2];
}

function percentile(sorted: Float64Array, fraction: number): number {
    if (sorted.length === 0) return NaN;
    return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))];
}

/**
 * A pixel's relative difference is measured against at least this share of
 * the mean luminance, so near-black pixels (where a tiny absolute
 * difference is a huge ratio and invisible) don't dominate the statistics.
 */
export const RELATIVE_FLOOR_SHARE = 0.01;

export interface MatchStatistics {
    /** Pixels compared: a surface, and no transparent object in front of it. */
    readonly pixels: number;
    readonly meanRelative: number;
    readonly p95Relative: number;
    readonly p99Relative: number;
    readonly maxRelative: number;
    /** Share of compared pixels off by more than 2%. */
    readonly over2Percent: number;
    /** Mean luminance of the composite over the per-fragment image's, minus 1. */
    readonly meanBias: number;
    /** GI's share of the per-fragment image's light on those pixels: (on − off) / on, from the means. */
    readonly giShare: number;
    readonly floor: number;
}

/**
 * Per-pixel relative luminance difference between the composite and the
 * per-fragment image over `mask` pixels: |L_c − L_f| / max(L_f, floor),
 * floor = {@link RELATIVE_FLOOR_SHARE} × the mean of L_f over the mask.
 */
export function matchStatistics(perFragment: Float32Array, composite: Float32Array, giOff: Float32Array, mask: Uint8Array): MatchStatistics {
    let count = 0;
    let sumReference = 0;
    let sumComposite = 0;
    let sumOff = 0;
    for (let pixel = 0; pixel < mask.length; pixel++) {
        if (!mask[pixel]) continue;
        count++;
        sumReference += luminanceAt(perFragment, pixel);
        sumComposite += luminanceAt(composite, pixel);
        sumOff += luminanceAt(giOff, pixel);
    }
    const floor = RELATIVE_FLOOR_SHARE * (count > 0 ? sumReference / count : 0);
    const relative = new Float64Array(count);
    let index = 0;
    let sumRelative = 0;
    let over2 = 0;
    for (let pixel = 0; pixel < mask.length; pixel++) {
        if (!mask[pixel]) continue;
        const reference = luminanceAt(perFragment, pixel);
        const value = Math.abs(luminanceAt(composite, pixel) - reference) / Math.max(reference, floor);
        relative[index++] = value;
        sumRelative += value;
        if (value > 0.02) over2++;
    }
    relative.sort();
    return {
        pixels: count,
        meanRelative: count > 0 ? sumRelative / count : NaN,
        p95Relative: percentile(relative, 0.95),
        p99Relative: percentile(relative, 0.99),
        maxRelative: count > 0 ? relative[count - 1] : NaN,
        over2Percent: count > 0 ? over2 / count : NaN,
        meanBias: sumComposite / sumReference - 1,
        giShare: (sumReference - sumOff) / sumReference,
        floor,
    };
}

/**
 * Pixels to compare: a surface (from the composite pass's depth), and the
 * same per-fragment colour with and without transparent objects, so no
 * transparent object covers it.
 */
export function opaqueMask(coverage: Float32Array, perFragment: Float32Array, opaqueOnly: Float32Array): Uint8Array {
    const pixels = coverage.length / 4;
    const mask = new Uint8Array(pixels);
    for (let pixel = 0; pixel < pixels; pixel++) {
        if (coverage[pixel * 4] < 0.5) continue;
        const full = luminanceAt(perFragment, pixel);
        const opaque = luminanceAt(opaqueOnly, pixel);
        if (Math.abs(full - opaque) > 1e-6 * Math.max(full, 1e-3)) continue;
        mask[pixel] = 1;
    }
    return mask;
}

export function formatMatch(name: string, stats: MatchStatistics): string {
    const percent = (value: number) => `${(value * 100).toFixed(3)}%`;
    return `${name}: mean ${percent(stats.meanRelative)}, p95 ${percent(stats.p95Relative)}, p99 ${percent(stats.p99Relative)}, max ${percent(stats.maxRelative)}, >2% on ${percent(stats.over2Percent)} of ${stats.pixels} pixels; mean bias ${percent(stats.meanBias)}; GI is ${percent(stats.giShare)} of the light`;
}

// -- GPU time per frame ------------------------------------------------------------------------

interface RenderContextShape {
    camera: Camera | null;
    fullscreenPass?: boolean;
}

interface QueryPool {
    currentQueryIndex: number;
    timestamps: Map<string, number>;
}

interface BackendShape {
    timestampQueryPool?: Partial<Record<'render' | 'compute', QueryPool | null>>;
    beginRender(context: RenderContextShape): void;
    get(object: object): { timestampUID?: string };
}

/** A pass longer than this is a timestamp glitch (Intel sometimes returns an end before its start; NOTES). */
const MAX_PLAUSIBLE_PASS_MS = 60_000;

/** Which frame a pass belongs to: the variant drawn and the cycle (one frame of every variant). */
export interface FrameTag {
    readonly variant: string;
    readonly cycle: number;
}

/** One frame's GPU passes, ms by kind (`shadow`, `scene`, `quad`), and their sum. */
export interface FramePasses extends FrameTag {
    readonly byLabel: Record<string, number>;
    readonly total: number;
}

/**
 * three's GPU pass times, one frame at a time. Each pass is labelled by kind
 * when it begins: `shadow` (a shadow camera's pass), `quad` (a full-screen
 * pass: the pipeline's output and tone mapping, where the composite runs) or
 * `scene` (the scene pass), and tagged with the frame the page says it is
 * drawing (`tag`). three keys each pass's time by a uid that holds the frame
 * number (`Backend.getTimestampUID`), so the tags last until the pool is
 * resolved. Needs `trackTimestamp: true` on the renderer.
 */
export class PassClock {
    /** The frame being drawn; set before each frame. */
    tag: FrameTag | null = null;
    private readonly passes = new Map<string, FrameTag & { label: string }>();

    constructor(
        private readonly renderer: WebGPURenderer,
        shadowCameras: ReadonlySet<Camera>,
    ) {
        const backend = renderer.backend as unknown as BackendShape;
        const beginRender = backend.beginRender.bind(backend);
        backend.beginRender = (context: RenderContextShape) => {
            const uid = backend.get(context).timestampUID;
            if (uid && this.tag) {
                const label = context.camera !== null && shadowCameras.has(context.camera) ? 'shadow' : context.fullscreenPass ? 'quad' : 'scene';
                this.passes.set(uid, { ...this.tag, label });
            }
            beginRender(context);
        };
    }

    /**
     * Resolves three's query pool: every tagged frame's passes drawn so far,
     * and how many pass times were implausible. three submits the resolve at
     * once and maps the result when the GPU gets there, so frames drawn while
     * this is pending land in the next read (they keep their tags).
     */
    async read(): Promise<{ frames: FramePasses[]; invalid: number }> {
        const backend = this.renderer.backend as unknown as BackendShape;
        const frames = new Map<string, { variant: string; cycle: number; byLabel: Record<string, number> }>();
        let invalid = 0;
        const pool = backend.timestampQueryPool?.render;
        if (pool && pool.currentQueryIndex > 0) {
            await this.renderer.resolveTimestampsAsync('render');
            for (const [uid, ms] of pool.timestamps) {
                const pass = this.passes.get(uid);
                if (!pass) continue;
                // Only this resolve's passes: frames drawn since (still in flight) keep their tags.
                this.passes.delete(uid);
                if (!Number.isFinite(ms) || ms < 0 || ms > MAX_PLAUSIBLE_PASS_MS) {
                    invalid++;
                    continue;
                }
                const key = `${pass.variant}\n${pass.cycle}`;
                let frame = frames.get(key);
                if (!frame) {
                    frame = { variant: pass.variant, cycle: pass.cycle, byLabel: {} };
                    frames.set(key, frame);
                }
                frame.byLabel[pass.label] = (frame.byLabel[pass.label] ?? 0) + ms;
            }
        }
        return {
            frames: [...frames.values()].map((frame) => ({ ...frame, total: Object.values(frame.byLabel).reduce((sum, ms) => sum + ms, 0) })),
            invalid,
        };
    }
}

/** One way of drawing a frame. */
export interface FrameVariant {
    readonly id: string;
    /** Before each of the variant's frames (which light is visible, say). Must be cheap. */
    apply(): void;
    /** One frame's GPU work. Must not wait. */
    draw(): void;
    /**
     * GPU time this variant spends outside three's passes, in ms, as it last
     * measured it (the upscaler times its own compute passes), or null.
     */
    extraGpuMs?(): number | null;
}

export interface InterleavedOptions {
    /** What the section is for, kept in its result (the report prints it). */
    readonly label: string;
    /** Length of the timed section, seconds. */
    readonly seconds: number;
    /** Cycles drawn back to back between two reads of the query pool (each read waits for the GPU once). */
    readonly cyclesPerBatch: number;
    /** Seconds of GPU-bound cycles before the timed section, so the clock has ramped up from idle. */
    readonly rampSeconds: number;
    readonly log: (message: string) => void;
}

/** A per-cycle difference's distribution, ms. */
export interface Spread {
    readonly median: number;
    readonly p10: number;
    readonly p90: number;
    /** Median absolute deviation from the median. */
    readonly mad: number;
    /** Medians of the first, middle and last third of the cycles: a drift check. */
    readonly thirds: [number, number, number];
    readonly cycles: number;
}

export interface InterleavedTiming {
    /** What the section is for (`InterleavedOptions.label`). */
    readonly label: string;
    /** Wall-clock window of the timed section (`Date.now()`), for matching GPU clock samples. */
    readonly windowStartMs: number;
    readonly windowEndMs: number;
    readonly width: number;
    readonly height: number;
    readonly cycles: number;
    /**
     * Per variant: median GPU ms per frame, in total and by pass kind; the
     * draw-call counts seen; and the median CPU time of recording one frame
     * (a page whose frames cost more CPU than GPU can't be GPU-bound).
     */
    readonly variants: Record<string, { readonly totalMs: number; readonly byLabel: Record<string, number>; readonly drawCalls: number[]; readonly cpuMs: number; readonly extraGpuMs: number | null }>;
    /** Per named difference: the distribution of its per-cycle values. */
    readonly deltas: Record<string, Spread>;
    readonly invalidPasses: number;
}

function quantile(sorted: readonly number[], fraction: number): number {
    if (sorted.length === 0) return NaN;
    const position = fraction * (sorted.length - 1);
    const below = Math.floor(position);
    const above = Math.min(sorted.length - 1, below + 1);
    return sorted[below] + (sorted[above] - sorted[below]) * (position - below);
}

export function median(values: readonly number[]): number {
    return quantile([...values].sort((a, b) => a - b), 0.5);
}

export function spreadOf(values: readonly number[]): Spread {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = quantile(sorted, 0.5);
    const third = Math.max(1, Math.floor(values.length / 3));
    return {
        median: middle,
        p10: quantile(sorted, 0.1),
        p90: quantile(sorted, 0.9),
        mad: median(values.map((value) => Math.abs(value - middle))),
        thirds: [median(values.slice(0, third)), median(values.slice(third, values.length - third)), median(values.slice(values.length - third))],
        cycles: values.length,
    };
}

/**
 * A difference to report: per cycle, the minuend frame's part minus the
 * subtrahend frame's. The two frames ran next to each other in the same
 * cycle, so at the same clock.
 */
export interface DeltaDefinition {
    readonly name: string;
    readonly minuend: string;
    readonly subtrahend: string;
    /** Which part of the frame: all of it, or one kind of pass. */
    readonly part: 'total' | 'scene' | 'quad' | 'shadow';
}

/**
 * GPU time of each variant with the variants interleaved frame by frame
 * (A B C D A B C D ...). Every pass of every frame is timed on the GPU, and
 * each difference is taken per cycle, between frames that ran next to each
 * other, then summarised over all cycles. Clock drift during the run moves
 * every variant alike, so it cancels in the differences instead of landing
 * on whichever variant ran while the GPU was hot (the laptop RTX 3060
 * throttles after about 20 s of GPU-bound load, 7a8128c).
 *
 * **GPU-bound, really.** The GPU must never wait for the page, or it drops
 * to a low-power clock and every millisecond reads long (Budget v2 hit this:
 * waiting for each frame left the RTX about 40% busy). So:
 * - the CPU stays at least one batch ahead: batch k is drawn while batch
 *   k − 1 is still on the GPU, and the page only waits for batch k − 1's
 *   timestamps (three resolves them as soon as they're asked for, and maps
 *   them when the GPU gets there);
 * - there are no other waits or read-backs in the timed section;
 * - drawing a frame must change no renderer state: each variant has its
 *   own pass and camera (the caller's job), because toggling a light or a
 *   material between frames makes three refresh every object, and the page
 *   becomes CPU-bound (it measured 150 ms of wall time per frame on Sponza).
 *
 * The variants must already be compiled (draw each a few times first).
 */
export async function timeInterleaved(
    renderer: WebGPURenderer,
    device: GPUDevice,
    clock: PassClock,
    variants: readonly FrameVariant[],
    deltas: readonly DeltaDefinition[],
    options: InterleavedOptions,
): Promise<InterleavedTiming> {
    const drawCalls = () => (renderer as unknown as FrameAdvance).info.render.drawCalls;
    const callsSeen = new Map<string, Set<number>>(variants.map((variant) => [variant.id, new Set<number>()]));
    const cpuTimes = new Map<string, number[]>(variants.map((variant) => [variant.id, []]));
    const extraTimes = new Map<string, number[]>(variants.map((variant) => [variant.id, []]));
    const drawCycle = (cycle: number | null) => {
        for (const variant of variants) {
            const cpuStart = performance.now();
            advanceThreeFrame(renderer);
            variant.apply();
            clock.tag = cycle === null ? null : { variant: variant.id, cycle };
            variant.draw();
            if (cycle !== null) {
                cpuTimes.get(variant.id)!.push(performance.now() - cpuStart);
                callsSeen.get(variant.id)!.add(drawCalls());
                const extra = variant.extraGpuMs?.() ?? null;
                if (extra !== null) extraTimes.get(variant.id)!.push(extra);
            }
        }
        clock.tag = null;
    };

    // The ramp: untimed batches, two in flight, until the clock has come up from idle.
    const rampStart = performance.now();
    let previous: Promise<void> | null = null;
    while (performance.now() - rampStart < options.rampSeconds * 1000) {
        for (let index = 0; index < options.cyclesPerBatch; index++) drawCycle(null);
        const done = device.queue.onSubmittedWorkDone();
        if (previous) await previous;
        previous = done;
    }

    const frames: FramePasses[] = [];
    let invalid = 0;
    let cycle = 0;
    const collect = (read: { frames: FramePasses[]; invalid: number }) => {
        frames.push(...read.frames);
        invalid += read.invalid;
    };
    const windowStartMs = Date.now();
    const start = performance.now();
    // The ramp's last batch is still on the GPU: its frames are untagged, so a read just empties the pool.
    let pendingRead: Promise<void> = clock.read().then(() => {});
    while (performance.now() - start < options.seconds * 1000) {
        for (let index = 0; index < options.cyclesPerBatch; index++) drawCycle(cycle++);
        // Wait for the batch before this one (this one stays queued on the GPU), then ask for this one's times.
        await pendingRead;
        pendingRead = clock.read().then(collect);
    }
    await pendingRead;
    const windowEndMs = Date.now();
    // Anything resolved after the last read started (three resolves on overflow by itself).
    collect(await clock.read());

    // Frames by cycle; a cycle missing any variant's frame (a dropped timestamp) is left out.
    const byCycle = new Map<number, Map<string, FramePasses>>();
    for (const frame of frames) {
        let cycleFrames = byCycle.get(frame.cycle);
        if (!cycleFrames) {
            cycleFrames = new Map();
            byCycle.set(frame.cycle, cycleFrames);
        }
        cycleFrames.set(frame.variant, frame);
    }
    const complete = [...byCycle.entries()]
        .filter(([, cycleFrames]) => variants.every((variant) => cycleFrames.has(variant.id)))
        .sort(([a], [b]) => a - b)
        .map(([, cycleFrames]) => cycleFrames);

    const partOf = (frame: FramePasses, part: DeltaDefinition['part']) => (part === 'total' ? frame.total : (frame.byLabel[part] ?? 0));
    const results: Record<string, { totalMs: number; byLabel: Record<string, number>; drawCalls: number[]; cpuMs: number; extraGpuMs: number | null }> = {};
    for (const variant of variants) {
        const own = complete.map((cycleFrames) => cycleFrames.get(variant.id)!);
        const labels = new Set(own.flatMap((frame) => Object.keys(frame.byLabel)));
        const byLabel: Record<string, number> = {};
        for (const label of labels) byLabel[label] = median(own.map((frame) => frame.byLabel[label] ?? 0));
        const extra = extraTimes.get(variant.id)!;
        results[variant.id] = {
            totalMs: median(own.map((frame) => frame.total)),
            byLabel,
            drawCalls: [...callsSeen.get(variant.id)!],
            cpuMs: median(cpuTimes.get(variant.id)!),
            extraGpuMs: extra.length > 0 ? median(extra) : null,
        };
    }
    const differences: Record<string, Spread> = {};
    for (const delta of deltas) {
        const values = complete.map((cycleFrames) => partOf(cycleFrames.get(delta.minuend)!, delta.part) - partOf(cycleFrames.get(delta.subtrahend)!, delta.part));
        differences[delta.name] = spreadOf(values);
    }
    const size = renderer.getDrawingBufferSize(new Vector2());
    options.log(`${options.label}, ${size.width} x ${size.height}: ${complete.length} complete cycles in ${((windowEndMs - windowStartMs) / 1000).toFixed(1)} s`);
    for (const [id, result] of Object.entries(results)) {
        const parts = Object.entries(result.byLabel)
            .map(([label, ms]) => `${label} ${ms.toFixed(3)}`)
            .join(', ');
        const extra = result.extraGpuMs === null ? '' : `, plus ${result.extraGpuMs.toFixed(3)} ms of upscaler compute`;
        options.log(`  ${id}: ${result.totalMs.toFixed(3)} ms per frame on the GPU (${parts})${extra}, ${result.cpuMs.toFixed(2)} ms to record; draw calls ${result.drawCalls.join('/')}`);
    }
    for (const [name, spread] of Object.entries(differences)) {
        options.log(`  ${name}: median ${spread.median.toFixed(3)} ms, p10 to p90 ${spread.p10.toFixed(3)} to ${spread.p90.toFixed(3)}, MAD ${spread.mad.toFixed(3)}, thirds ${spread.thirds.map((ms) => ms.toFixed(3)).join('/')}`);
    }
    return {
        label: options.label,
        windowStartMs,
        windowEndMs,
        width: size.width,
        height: size.height,
        cycles: complete.length,
        variants: results,
        deltas: differences,
        invalidPasses: invalid,
    };
}

/** The scene's shadow cameras, so their passes get their own label. */
export function shadowCameras(scene: CompositeScene): Set<Camera> {
    const cameras = new Set<Camera>();
    for (const light of scene.loaded.directLights) {
        const shadow = (light as { shadow?: { camera?: Camera } }).shadow;
        if (shadow?.camera) cameras.add(shadow.camera);
    }
    return cameras;
}
