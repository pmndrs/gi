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

import { FloatType, UnsignedByteType, MeshStandardMaterial, NodeMaterial, PerspectiveCamera, QuadMesh, RenderTarget } from 'three/webgpu';
import type { Camera, Material, Mesh, Node, Object3D, Scene, WebGPURenderer } from 'three/webgpu';
import { isBackgroundDepth, mrt, normalView, output, packNormalToRGB, pass, uv, vec4 } from 'three/tsl';
import type { MaterialDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { ThreeGI } from '../../../../packages/ddgi-three/src/three-gi.ts';
import { ddgiComposite, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT, type DDGICompositeNode } from '../../../../packages/ddgi-three/src/composite.ts';
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
    platform.setOutput(platform.addTechnique(new DDGITechnique({ volume: { center: volume.center, size: volume.size, spacing: volume.probeSpacing } })));
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
 * platform received (SceneSync.sentMaterial: albedo = colour × texture
 * average × (1 − metalness), emission), as the Sponza accuracy page's GI
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
        const albedo = descriptor.albedo ?? [1, 1, 1];
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
 * - `compositePass`: `pass()` with `mrt({ output, giAlbedo, normal })`, and `composite = ddgiComposite(...)`.
 */
export class CompositeRoutes {
    readonly perFragmentPass;
    readonly opaquePass;
    readonly compositePass;
    readonly composite: DDGICompositeNode;
    /** Per pixel: 1 where a surface is (not the cleared background), the giAlbedo weight's luminance, and its alpha. */
    readonly coverage: Node;

    /**
     * `byteTargets`: store `giAlbedo` and the normal as 8-bit unorm instead of
     * the pass's half float, the bandwidth saving three's SSGI example makes.
     */
    constructor(scene: CompositeScene, options: { readonly byteTargets?: boolean } = {}) {
        this.perFragmentPass = pass(scene.scene, scene.camera);
        this.opaquePass = pass(scene.scene, scene.camera);
        this.opaquePass.transparent = false;
        this.compositePass = pass(scene.scene, scene.camera);
        this.compositePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView) }));
        if (options.byteTargets) {
            this.compositePass.getTexture(GI_ALBEDO_OUTPUT).type = UnsignedByteType;
            this.compositePass.getTexture(GI_NORMAL_OUTPUT).type = UnsignedByteType;
        }
        this.composite = ddgiComposite(this.compositePass, scene.threeGI);
        const depth = this.compositePass.getTextureNode('depth').sample(uv()).r;
        const weight = this.compositePass.getTextureNode(GI_ALBEDO_OUTPUT).sample(uv());
        const surface = isBackgroundDepth(depth).select(0, 1);
        this.coverage = vec4(surface, weight.r.mul(0.2126).add(weight.g.mul(0.7152)).add(weight.b.mul(0.0722)), weight.a, 0) as unknown as Node;
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

/**
 * three's GPU pass times, labelled by kind: `shadow` (a shadow camera's
 * pass), `quad` (a full-screen pass: the pipeline's output and tone
 * mapping, where the composite runs) and `scene` (everything else: the
 * scene pass). Needs `trackTimestamp: true` on the renderer.
 */
export class PassClock {
    private readonly labels = new Map<string, string>();

    constructor(
        private readonly renderer: WebGPURenderer,
        shadowCameras: ReadonlySet<Camera>,
    ) {
        const backend = renderer.backend as unknown as BackendShape;
        const beginRender = backend.beginRender.bind(backend);
        backend.beginRender = (context: RenderContextShape) => {
            const uid = backend.get(context).timestampUID;
            if (uid) {
                const label = context.camera !== null && shadowCameras.has(context.camera) ? 'shadow' : context.fullscreenPass ? 'quad' : 'scene';
                this.labels.set(uid, label);
            }
            beginRender(context);
        };
    }

    /** Reads and empties three's query pools: milliseconds by label, and whether any pass time was implausible. */
    async read(): Promise<{ byLabel: Record<string, number>; invalid: boolean }> {
        const backend = this.renderer.backend as unknown as BackendShape;
        const byLabel: Record<string, number> = {};
        let invalid = false;
        for (const kind of ['render', 'compute'] as const) {
            const pool = backend.timestampQueryPool?.[kind];
            if (!pool || pool.currentQueryIndex === 0) continue;
            await this.renderer.resolveTimestampsAsync(kind);
            for (const [uid, ms] of pool.timestamps) {
                if (!Number.isFinite(ms) || ms < 0 || ms > MAX_PLAUSIBLE_PASS_MS) {
                    invalid = true;
                    continue;
                }
                const label = kind === 'compute' ? 'compute' : (this.labels.get(uid) ?? 'scene');
                byLabel[label] = (byLabel[label] ?? 0) + ms;
            }
        }
        this.labels.clear();
        return { byLabel, invalid };
    }
}

/** One way of drawing a frame. */
export interface FrameVariant {
    readonly id: string;
    /** Before the variant's frames in each round (which light is visible, say). */
    apply(): void;
    /** One frame's GPU work. Must not wait. */
    draw(): void;
}

export interface VariantTiming {
    /** Median over rounds of each round's median, ms per frame. */
    readonly totalMs: number;
    /** three's draw calls in the variant's last frame: the same count in every sample, or objects were skipped (still compiling). */
    readonly drawCalls: number[];
    readonly byLabel: Record<string, number>;
    readonly perRound: number[];
    readonly samples: number;
}

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    if (sorted.length === 0) return NaN;
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export interface TimingOptions {
    readonly rounds: number;
    /** Samples per variant per round. */
    readonly samples: number;
    /** Frames per sample, drawn back to back with no wait: GPU-bound. */
    readonly burst: number;
    /** Samples a variant is drawn for before its first timed samples (its frame time and draw calls must also settle). */
    readonly settleSamples: number;
    /** Seconds of GPU-bound frames before anything is timed, so the clock has ramped up from idle. */
    readonly rampSeconds: number;
    /**
     * Seconds of idle GPU after the pipelines compile and before the ramp, so
     * a GPU heated by loading the scene cools first (82 → 63 °C took 10 s idle).
     */
    readonly coolSeconds: number;
    readonly log: (message: string) => void;
}

/**
 * GPU time per frame of each variant, GPU-bound (Dennis, 2026-11-05: judge
 * with the GPU loaded): each sample draws `burst` frames back to back and
 * waits once. Variants alternate order each round; each is drawn until its
 * frame time settles first (pipelines compile in the background).
 *
 * Keep the whole run short (about 10–15 s): a sustained GPU-bound load heats
 * the laptop RTX 3060 to 87 °C in about 20 s, after which it holds
 * 450–600 MHz (the clock investigation, 7a8128c). Start it cool.
 */
export async function timeVariants(renderer: WebGPURenderer, device: GPUDevice, clock: PassClock, variants: readonly FrameVariant[], options: TimingOptions): Promise<Record<string, VariantTiming>> {
    const drawCalls = () => (renderer as unknown as FrameAdvance).info.render.drawCalls;
    const sample = async (variant: FrameVariant) => {
        let calls = 0;
        for (let frame = 0; frame < options.burst; frame++) {
            advanceThreeFrame(renderer);
            variant.draw();
            calls = drawCalls();
        }
        await device.queue.onSubmittedWorkDone();
        const passes = await clock.read();
        const byLabel: Record<string, number> = {};
        for (const [label, ms] of Object.entries(passes.byLabel)) byLabel[label] = ms / options.burst;
        return { byLabel, total: Object.values(byLabel).reduce((sum, ms) => sum + ms, 0), invalid: passes.invalid, calls };
    };
    const settle = async (variant: FrameVariant, minimum: number) => {
        const totals: number[] = [];
        const calls: number[] = [];
        for (let index = 0; index < 200; index++) {
            const result = await sample(variant);
            totals.push(result.total);
            calls.push(result.calls);
            if (totals.length < minimum) continue;
            const recent = totals.slice(-4);
            const recentCalls = calls.slice(-4);
            const typical = median(recent);
            if (recentCalls.every((count) => count === recentCalls[0]) && recent.every((ms) => Math.abs(ms - typical) <= 0.1 * typical + 0.02)) return;
        }
    };
    // Every variant once (pipelines compile), then GPU-bound frames until the clock has ramped up from idle.
    for (const variant of variants) {
        variant.apply();
        await settle(variant, options.settleSamples);
    }
    options.log(`cooling for ${options.coolSeconds} s, then ${options.rampSeconds} s of GPU-bound frames before timing`);
    await new Promise((resolve) => setTimeout(resolve, options.coolSeconds * 1000));
    const timedStart = performance.now();
    const rampStart = performance.now();
    while (performance.now() - rampStart < options.rampSeconds * 1000) await sample(variants[0]);
    const taken = new Map<string, Array<Array<{ byLabel: Record<string, number>; total: number; calls: number }>>>(variants.map((variant) => [variant.id, []]));
    for (let round = 0; round < options.rounds; round++) {
        const order = round % 2 === 0 ? variants : [...variants].reverse();
        for (const variant of order) {
            variant.apply();
            await settle(variant, 2);
            const samples = [];
            for (let index = 0; index < options.samples; index++) {
                const result = await sample(variant);
                if (!result.invalid) samples.push(result);
            }
            taken.get(variant.id)!.push(samples);
        }
        options.log(`timing round ${round + 1} of ${options.rounds} done`);
    }
    options.log(`timed section (ramp and rounds) took ${((performance.now() - timedStart) / 1000).toFixed(1)} s`);
    const results: Record<string, VariantTiming> = {};
    for (const variant of variants) {
        const rounds = taken.get(variant.id)!.filter((samples) => samples.length > 0);
        const labels = new Set<string>();
        for (const samples of rounds) for (const result of samples) Object.keys(result.byLabel).forEach((label) => labels.add(label));
        const byLabel: Record<string, number> = {};
        for (const label of labels) byLabel[label] = median(rounds.map((samples) => median(samples.map((result) => result.byLabel[label] ?? 0))));
        const perRound = rounds.map((samples) => median(samples.map((result) => result.total)));
        const drawCallCounts = [...new Set(rounds.flatMap((samples) => samples.map((result) => result.calls)))];
        results[variant.id] = { totalMs: median(perRound), drawCalls: drawCallCounts, byLabel, perRound, samples: rounds.reduce((sum, samples) => sum + samples.length, 0) };
        const parts = Object.entries(byLabel)
            .map(([label, ms]) => `${label} ${ms.toFixed(3)}`)
            .join(', ');
        options.log(`  ${variant.id}: ${results[variant.id].totalMs.toFixed(3)} ms per frame (${parts}); rounds ${perRound.map((ms) => ms.toFixed(3)).join('/')}; draw calls ${drawCallCounts.join('/')}`);
    }
    return results;
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
