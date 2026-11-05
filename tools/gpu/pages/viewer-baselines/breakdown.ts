/**
 * Where DDGI's GI-on frame time goes, pass by pass (T1.9: Sponza's +43 ms
 * on the Intel UHD). Runs instead of the baselines when
 * `VITE_DDGI_BREAKDOWN=1`:
 *
 *   VITE_DDGI_BREAKDOWN=1 VITE_DDGI_SCENES=cornell-box,two-room,sponza \
 *     node tools/reference/bench-lock.ts run --what "frame breakdown" -- \
 *     pnpm test:gpu --filter viewer-baselines --gpu intel --json breakdown.json
 *
 * Same view as the baselines (the scene's first bookmark, 1280x720 by
 * default, the viewer's own DDGI technique and render path, frames drawn
 * through three's animation loop and waited for). For every frame, three's
 * timestamped passes are labelled by what drew them (a shadow map, the
 * scene, the output pass) by watching which render context each pass
 * belongs to; the platform's own passes (`ddgi/trace`, `ddgi/blend`, ...,
 * `platform/top-level refit`) come from `gi.stats.passes`.
 *
 * Variants, measured in alternating rounds like the baselines:
 *  - `off`: GI off (DDGI's light hidden from the camera, no update);
 *  - `on`: the viewer's normal frame (2 ms update budget);
 *  - `on-paused`: GI on but no probe update: the shading cost alone;
 *  - `off-prepass` and `on-prepass`: the same with a depth prepass first, so
 *    each pixel is shaded once (the lever for overdraw; not in the viewer).
 *    The prepass draws every opaque mesh with a depth-only material that keeps
 *    the source's side and alpha cut-out; the main pass then passes the
 *    depth test only for the front surface (three's default `LessEqual`).
 *  - `off-plain` and `on-plain`: every opaque mesh drawn with one shared,
 *    untextured standard material (front or double-sided as its own), so
 *    every scene runs the same fragment shader: what's left of the GI cost
 *    per fragment then depends on the geometry and the field, not on the
 *    scene's materials;
 *  - `off-no-msaa` and `on-no-msaa`: the scene target without the viewer's
 *    4x multisampling;
 *  - `off-alu`: GI off, every opaque mesh drawn with one unlit material
 *    whose fragment shader is a fixed run of arithmetic and reads no
 *    texture. Its cost over `off-plain` is a fixed amount of work per
 *    fragment-shader invocation, so comparing it between scenes shows how
 *    many invocations a scene's pixels really cost (small triangles shade
 *    whole 2x2 quads), with no texture cache or GI field involved.
 * `VITE_DDGI_VARIANTS=off,on,...` runs only some of them.
 *
 * GPU-bound timing (`VITE_DDGI_BURST=8`): each sample draws that many frames
 * back to back (the first from three's animation loop, the rest advancing
 * three's frame the way its loop does), waits once, and divides three's pass
 * times by the count, so the GPU never idles long enough to clock down. The
 * platform's passes are its newest timed frame. With 1 (the default) every
 * frame is waited for, as in the baselines.
 *
 * Plus two untimed checks:
 *  - fragments per pixel: every opaque mesh drawn with an additive "count 1"
 *    material in three's own draw order and depth test (MSAA 4 like the
 *    scene target, resolved), so the mean is how many fragments the scene
 *    pass shades per pixel (quad overshading not included);
 *  - the prepass picture against the normal one (update paused, so the field
 *    is the same): mean colour and the share of pixels that differ, to show
 *    the prepass dropped nothing.
 */

import { AddEquation, CustomBlending, DataUtils, DoubleSide, FrontSide, HalfFloatType, OneFactor, type Camera, type Material, type Mesh, type Object3D, type Scene, type Texture } from 'three';
import { Fn, Loop, positionWorld, texture, vec3 } from 'three/tsl';
import { MeshBasicNodeMaterial, MeshStandardNodeMaterial, RenderTarget, type Node, type WebGPURenderer } from 'three/webgpu';
import type { SceneManifest } from '../../../../scenes/src/manifest';
import { ViewerCore } from '../../../../apps/viewer/src/core/viewer-core';
import { median } from '../../../../apps/viewer/src/bench/measurement';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests';
import type { DdgiTechnique } from '../../../../apps/viewer/src/techniques/ddgi-technique';
import type { FrameBudget } from '../../../../apps/viewer/src/ddgi/library';

export interface BreakdownOptions {
    scenes: string[];
    variants: readonly VariantId[];
    /** Frames drawn back to back per sample (see the file comment); 1 waits for every frame. */
    burst: number;
    rounds: number;
    sampleFrames: number;
    width: number;
    height: number;
    log(message: string): void;
}

export type VariantId = 'off' | 'on' | 'on-paused' | 'off-prepass' | 'on-prepass' | 'off-plain' | 'on-plain' | 'off-no-msaa' | 'on-no-msaa' | 'off-alu';
export const VARIANTS: readonly VariantId[] = ['off', 'on', 'on-paused', 'off-prepass', 'on-prepass', 'off-plain', 'on-plain', 'off-no-msaa', 'on-no-msaa', 'off-alu'];

/** What a render pass drew, from its render context. */
type PassLabel = 'shadow' | 'depth prepass' | 'scene' | 'output' | 'three compute';

/** How the scene pass is drawn this frame. */
type DrawMode = 'normal' | 'prepass' | 'plain' | 'alu';

interface FrameRecord {
    /** three's passes by label, ms. */
    three: Record<string, number>;
    /** The platform's passes by name, ms (empty when it did no work this frame). */
    platform: Record<string, number>;
    probesUpdated: number;
    invalid: boolean;
}

export interface VariantResult {
    /** Median over rounds of each round's median, ms. */
    threeMs: Record<string, number>;
    platformMs: Record<string, number>;
    threeTotalMs: number;
    platformTotalMs: number;
    frameMs: number;
    /** The frame total per round (median of its frames), for the spread. */
    frameMsPerRound: number[];
    probesUpdated: number;
    framesUsed: number;
}

export interface SceneBreakdown {
    scene: string;
    triangles: number;
    meshes: number;
    materials: number;
    probes: number;
    computeCoverage: boolean | null;
    variants: Partial<Record<VariantId, VariantResult>>;
    fragmentsPerPixel: number | null;
    fragmentsPerPixelP95: number | null;
    prepassImage: { meanNormal: number[]; meanPrepass: number[]; pixelsDiffering: number } | null;
    error: string | null;
}

export interface BreakdownResult {
    scenes: SceneBreakdown[];
    adapter: { vendor: string; architecture: string; description: string; device: string; features: string[] };
    options: Omit<BreakdownOptions, 'log'>;
    gpuErrors: string[];
}

// -- three internals the measurement reads (r187) ------------------------------------------------

interface RenderContextShape {
    camera: Camera | null;
    sampleCount: number;
    width: number;
    height: number;
}

interface QueryPool {
    currentQueryIndex: number;
    timestamps: Map<string, number>;
}

interface BackendShape {
    trackTimestamp: boolean;
    timestampQueryPool?: Partial<Record<'render' | 'compute', QueryPool | null>>;
    beginRender(context: RenderContextShape): void;
    beginCompute(group: unknown): void;
    get(object: object): { timestampUID?: string };
}

/** The parts of the viewer's DDGI technique the breakdown steers or reads. Private there; this is a measurement tool. */
interface DdgiInternals {
    threeGI: { budget: FrameBudget; light: { computeCoverage: boolean } } | null;
    platform: { stats: { frame: number; passes: ReadonlyArray<{ name: string; gpuMs: number | undefined; frame: number }> } };
    ddgi: { counters: Record<string, number> };
    target: RenderTarget;
    node: Node;
    buildNode(): Node;
    camera: Camera;
}

/** The parts of three's renderer its animation loop touches to start a frame (`Animation.js`, r187). */
interface FrameAdvance {
    info: { autoReset: boolean; reset(): void; frame: number };
    _nodes: { nodeFrame: { update(): void; frameId: number } };
}

/** The viewer's scene target is multisampled 4x (`ScenePassTechnique`). */
const VIEWER_MSAA_SAMPLES = 4;

/**
 * Labels three's timestamped passes. three keys each pass's duration by a uid it builds
 * just before `beginRender`; the wrapper reads that uid and remembers which render context
 * (and so which camera, target and draw mode) it belongs to.
 */
class PassLabeler {
    private readonly labels = new Map<string, PassLabel>();
    mode: DrawMode = 'normal';

    constructor(
        private readonly backend: BackendShape,
        private readonly viewCamera: () => Camera | null,
        private readonly shadowCameras: () => ReadonlySet<Camera>,
    ) {
        const beginRender = backend.beginRender.bind(backend);
        backend.beginRender = (context: RenderContextShape) => {
            const uid = backend.get(context).timestampUID;
            if (uid) this.labels.set(uid, this.classify(context));
            beginRender(context);
        };
        const beginCompute = backend.beginCompute.bind(backend);
        backend.beginCompute = (group: unknown) => {
            const uid = typeof group === 'object' && group !== null ? backend.get(group).timestampUID : undefined;
            if (uid) this.labels.set(uid, 'three compute');
            beginCompute(group);
        };
    }

    private classify(context: RenderContextShape): PassLabel {
        if (context.camera && this.shadowCameras().has(context.camera)) return 'shadow';
        if (context.camera && context.camera === this.viewCamera()) return this.mode === 'prepass' ? 'depth prepass' : 'scene';
        return 'output';
    }

    labelOf(uid: string): string {
        return this.labels.get(uid) ?? 'unlabelled';
    }

    forget(): void {
        this.labels.clear();
    }
}

/** A pass longer than this is a timestamp glitch (the Intel UHD sometimes returns end before start). */
const MAX_PLAUSIBLE_PASS_MS = 60_000;

/** Reads and empties three's query pools, summing durations by label. */
async function readThreePasses(renderer: WebGPURenderer, labeler: PassLabeler): Promise<{ byLabel: Record<string, number>; invalid: boolean }> {
    const backend = renderer.backend as unknown as BackendShape;
    const byLabel: Record<string, number> = {};
    let invalid = false;
    for (const kind of ['render', 'compute'] as const) {
        const pool = backend.timestampQueryPool?.[kind];
        if (!pool || pool.currentQueryIndex === 0) continue;
        await renderer.resolveTimestampsAsync(kind);
        for (const [uid, ms] of pool.timestamps) {
            if (!Number.isFinite(ms) || ms < 0 || ms > MAX_PLAUSIBLE_PASS_MS) {
                invalid = true;
                continue;
            }
            const label = labeler.labelOf(uid);
            byLabel[label] = (byLabel[label] ?? 0) + ms;
        }
    }
    labeler.forget();
    return { byLabel, invalid };
}

// -- material swaps for the prepass and the fragment count ----------------------------------------

/** Keeps a source material's face culling and alpha cut-out on a stand-in, so it covers the same pixels. */
function copyCoverage(source: Material, standIn: MeshBasicNodeMaterial): void {
    standIn.side = source.side;
    if (source.alphaTest > 0) {
        standIn.alphaTest = source.alphaTest;
        const map = (source as Material & { map?: Texture | null }).map;
        if (map) standIn.opacityNode = texture(map).a.mul(source.opacity);
        else standIn.opacity = source.opacity;
    }
}

function depthOnlyMaterial(source: Material): MeshBasicNodeMaterial {
    const material = new MeshBasicNodeMaterial();
    material.name = `depth only (${source.name})`;
    material.colorWrite = false;
    copyCoverage(source, material);
    return material;
}

/** Adds 1 to the pixel for each fragment that passes the depth test (One + One blending, depth test and write as usual). */
function countMaterial(source: Material): MeshBasicNodeMaterial {
    const material = new MeshBasicNodeMaterial({ color: 0xffffff });
    material.name = `fragment count (${source.name})`;
    copyCoverage(source, material);
    // The colour stays 1 whatever the cut-out's alpha is: One, not SrcAlpha.
    material.blending = CustomBlending;
    material.blendEquation = AddEquation;
    material.blendSrc = OneFactor;
    material.blendDst = OneFactor;
    material.blendSrcAlpha = OneFactor;
    material.blendDstAlpha = OneFactor;
    return material;
}

/** One untextured standard material per face side, shared by every mesh (the `plain` variants). */
function plainMaterials(): Map<number, MeshStandardNodeMaterial> {
    const bySide = new Map<number, MeshStandardNodeMaterial>();
    for (const side of [FrontSide, DoubleSide]) {
        const material = new MeshStandardNodeMaterial({ color: 0xb0b0b0, roughness: 1, metalness: 0 });
        material.name = `plain (${side === FrontSide ? 'front' : 'double'})`;
        material.side = side;
        bySide.set(side, material);
    }
    return bySide;
}

/** Iterations of the `alu` material's loop: enough to dwarf the rest of its shader. */
const ALU_ITERATIONS = 1024;

/** One unlit, texture-free, fixed-cost material per face side (the `alu` variant). */
function aluMaterials(): Map<number, MeshBasicNodeMaterial> {
    const bySide = new Map<number, MeshBasicNodeMaterial>();
    // Each step depends on the last and on the fragment's position, so the compiler can neither
    // fold the loop nor share it between fragments.
    const busyColour = Fn(() => {
        const value = positionWorld.x.add(positionWorld.y).toVar();
        Loop(ALU_ITERATIONS, () => {
            value.assign(value.mul(1.37).sin().add(0.5));
        });
        return vec3(value.abs().mul(0.01));
    });
    for (const side of [FrontSide, DoubleSide]) {
        const material = new MeshBasicNodeMaterial();
        material.colorNode = busyColour();
        material.name = `alu (${side === FrontSide ? 'front' : 'double'})`;
        material.side = side;
        bySide.set(side, material);
    }
    return bySide;
}

/**
 * Swaps every opaque mesh's material for a stand-in (one per source material), and hides
 * meshes it can't stand in for (transparent ones, and materials that refuse overrides such
 * as the sky dome). Returns the function that puts everything back.
 */
function swapMaterials<StandIn extends Material>(scene: Scene, cache: Map<Material, StandIn>, make: (source: Material) => StandIn): () => void {
    const restores: Array<() => void> = [];
    const standIn = (source: Material): StandIn => {
        let material = cache.get(source);
        if (!material) {
            material = make(source);
            cache.set(source, material);
        }
        return material;
    };
    scene.traverse((object: Object3D) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh || !mesh.visible) return;
        const sources = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        const drawable = sources.every((source) => !source.transparent && source.allowOverride !== false);
        if (!drawable) {
            mesh.visible = false;
            restores.push(() => (mesh.visible = true));
            return;
        }
        const original = mesh.material;
        mesh.material = Array.isArray(original) ? original.map(standIn) : standIn(original);
        restores.push(() => (mesh.material = original));
    });
    return () => {
        for (const restore of restores) restore();
    };
}

// -- the breakdown --------------------------------------------------------------------------------

function useFirstBookmark(core: ViewerCore, manifest: SceneManifest): void {
    const bookmark = manifest.cameras[0];
    core.camera.position.set(...bookmark.position);
    core.camera.fov = bookmark.fovYDeg;
    core.camera.lookAt(...bookmark.target);
    core.camera.updateProjectionMatrix();
    core.camera.updateMatrixWorld(true);
}

function countSceneParts(scene: Scene): { meshes: number; materials: number } {
    const materials = new Set<Material>();
    let meshes = 0;
    scene.traverse((object) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh) return;
        meshes++;
        for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) materials.add(material);
    });
    return { meshes, materials: materials.size };
}

function halfToFloats(data: ArrayLike<number>, isHalf: boolean): Float32Array {
    const out = new Float32Array(data.length);
    for (let index = 0; index < data.length; index++) out[index] = isHalf ? DataUtils.fromHalfFloat(data[index]) : data[index];
    return out;
}

/** Reads an RGBA render target as floats (half-float targets come back as raw 16-bit words). */
async function readTarget(renderer: WebGPURenderer, target: RenderTarget): Promise<Float32Array> {
    const raw = (await renderer.readRenderTargetPixelsAsync(target, 0, 0, target.width, target.height)) as ArrayLike<number> & { BYTES_PER_ELEMENT: number };
    return halfToFloats(raw, raw.BYTES_PER_ELEMENT === 2 && target.texture.type === HalfFloatType);
}

class SceneRun {
    private readonly labeler: PassLabeler;
    private readonly depthMaterials = new Map<Material, MeshBasicNodeMaterial>();
    private readonly countMaterials = new Map<Material, MeshBasicNodeMaterial>();
    private readonly plainBySource = new Map<Material, MeshStandardNodeMaterial>();
    private readonly plainBySide = plainMaterials();
    private readonly aluBySource = new Map<Material, MeshBasicNodeMaterial>();
    private readonly aluBySide = aluMaterials();
    /** Scene targets (and the output node reading each) by sample count; the viewer's own is kept. */
    private readonly targets = new Map<number, { target: RenderTarget; node: Node }>();
    private readonly internals: DdgiInternals;
    private readonly originalRender: WebGPURenderer['render'];
    private drawMode: DrawMode = 'normal';
    private platformFrameAtStart = 0;
    private normalBudget: FrameBudget | null = null;

    constructor(
        private readonly core: ViewerCore,
        private readonly technique: DdgiTechnique,
        private readonly scene: Scene,
        shadowCameras: ReadonlySet<Camera>,
        private readonly burst: number,
    ) {
        this.internals = technique as unknown as DdgiInternals;
        this.labeler = new PassLabeler(core.renderer.backend as unknown as BackendShape, () => technique.camera, () => shadowCameras);
        // The prepass wraps the technique's own scene render: same target, same camera.
        this.originalRender = core.renderer.render.bind(core.renderer);
        core.renderer.render = (scene: Object3D, camera: Camera) => {
            const isScenePass = scene === this.scene && camera === technique.camera;
            if (isScenePass && this.drawMode === 'prepass') return this.renderWithPrepass(this.scene, camera);
            if (isScenePass && this.drawMode === 'plain') return this.renderPlain(this.scene, camera);
            if (isScenePass && this.drawMode === 'alu') return this.renderAlu(this.scene, camera);
            return this.originalRender(scene, camera);
        };
    }

    restore(): void {
        this.core.renderer.render = this.originalRender;
        this.useSamples(VIEWER_MSAA_SAMPLES);
        for (const [samples, entry] of this.targets) if (samples !== VIEWER_MSAA_SAMPLES) entry.target.dispose();
        for (const material of [...this.depthMaterials.values(), ...this.countMaterials.values(), ...this.plainBySide.values(), ...this.aluBySide.values()]) material.dispose();
    }

    private renderPlain(scene: Scene, camera: Camera): void {
        const putBack = swapMaterials(scene, this.plainBySource, (source) => this.plainBySide.get(source.side === FrontSide ? FrontSide : DoubleSide)!);
        try {
            this.originalRender(scene, camera);
        } finally {
            putBack();
        }
    }

    private renderAlu(scene: Scene, camera: Camera): void {
        const putBack = swapMaterials(scene, this.aluBySource, (source) => this.aluBySide.get(source.side === FrontSide ? FrontSide : DoubleSide)!);
        try {
            this.originalRender(scene, camera);
        } finally {
            putBack();
        }
    }

    /** Draws the scene into a target with this many samples; the viewer picks the new output node up on its next frame. */
    private useSamples(samples: number): void {
        const internals = this.internals;
        if (!this.targets.has(VIEWER_MSAA_SAMPLES)) this.targets.set(VIEWER_MSAA_SAMPLES, { target: internals.target, node: internals.node });
        let entry = this.targets.get(samples);
        if (!entry) {
            internals.target = new RenderTarget(1, 1, { type: HalfFloatType, samples });
            entry = { target: internals.target, node: internals.buildNode() };
            this.targets.set(samples, entry);
        }
        internals.target = entry.target;
        internals.node = entry.node;
    }

    /** One more frame right away, started the way three's animation loop starts one. */
    private renderExtraFrame(): void {
        const renderer = this.core.renderer as unknown as FrameAdvance;
        if (renderer.info.autoReset) renderer.info.reset();
        renderer._nodes.nodeFrame.update();
        renderer.info.frame = renderer._nodes.nodeFrame.frameId;
        this.core.renderFrame();
    }

    private renderWithPrepass(scene: Scene, camera: Camera): void {
        const renderer = this.core.renderer;
        this.labeler.mode = 'prepass';
        const putBack = swapMaterials(scene, this.depthMaterials, depthOnlyMaterial);
        try {
            this.originalRender(scene, camera);
        } finally {
            putBack();
        }
        this.labeler.mode = 'normal';
        // Keep the prepass's depth: no clear before the shaded pass.
        const autoClear = renderer.autoClear;
        renderer.autoClear = false;
        try {
            this.originalRender(scene, camera);
        } finally {
            renderer.autoClear = autoClear;
        }
    }

    private get threeGI(): NonNullable<DdgiInternals['threeGI']> {
        const threeGI = this.internals.threeGI;
        if (!threeGI) throw new Error('DDGI has no ThreeGI yet');
        return threeGI;
    }

    apply(variant: VariantId): void {
        this.normalBudget ??= this.threeGI.budget;
        this.core.setLayout({ giMode: variant.startsWith('off') ? 'off' : 'on' });
        this.threeGI.budget = variant === 'on-paused' ? { paused: true } : this.normalBudget;
        this.drawMode = variant.endsWith('prepass') ? 'prepass' : variant.endsWith('plain') ? 'plain' : variant.endsWith('alu') ? 'alu' : 'normal';
        this.useSamples(variant.endsWith('no-msaa') ? 1 : VIEWER_MSAA_SAMPLES);
        this.platformFrameAtStart = this.internals.platform.stats.frame;
    }

    resetBudget(): void {
        if (this.normalBudget) this.threeGI.budget = this.normalBudget;
        this.drawMode = 'normal';
        this.useSamples(VIEWER_MSAA_SAMPLES);
    }

    async frame(): Promise<FrameRecord & { drawCalls: number }> {
        await this.core.renderThroughLoop();
        const drawCalls = this.core.renderer.info.render.drawCalls;
        for (let extra = 1; extra < this.burst; extra++) this.renderExtraFrame();
        await this.core.device.queue.onSubmittedWorkDone();
        const three = await readThreePasses(this.core.renderer, this.labeler);
        for (const label of Object.keys(three.byLabel)) three.byLabel[label] /= this.burst;
        const platform: Record<string, number> = {};
        let invalid = three.invalid;
        if (this.core.layout.giMode !== 'off') {
            // Only passes timed since this variant began: earlier ones belong to another variant.
            for (const pass of this.internals.platform.stats.passes) {
                if (pass.frame <= this.platformFrameAtStart) continue;
                if (pass.gpuMs === undefined || !(pass.gpuMs >= 0)) {
                    invalid = true;
                    continue;
                }
                platform[pass.name] = (platform[pass.name] ?? 0) + pass.gpuMs;
            }
        }
        // With GI off or the update paused nothing is updated; the counter would still show the last update.
        const updating = this.core.layout.giMode !== 'off' && !this.threeGI.budget.paused;
        const probesUpdated = updating ? (this.internals.ddgi.counters['probes updated'] ?? 0) : 0;
        return { three: three.byLabel, platform, probesUpdated, invalid, drawCalls };
    }

    /** Draws until the draw-call count and the frame time stop changing (pipelines compile in the background). */
    async settle(minimumFrames: number): Promise<void> {
        const calls: number[] = [];
        const times: number[] = [];
        for (let index = 0; index < 600; index++) {
            const record = await this.frame();
            calls.push(record.drawCalls);
            times.push(Object.values(record.three).reduce((sum, ms) => sum + ms, 0));
            if (calls.length < minimumFrames) continue;
            const recentCalls = calls.slice(-8);
            const recentTimes = times.slice(-8);
            const typical = median(recentTimes);
            if (recentCalls.every((count) => count === recentCalls[0]) && recentTimes.every((ms) => Math.abs(ms - typical) <= 0.25 * typical + 0.02)) return;
        }
    }

    /** Mean and 95th percentile of fragments shaded per pixel by the scene pass (see the file comment). */
    async fragmentsPerPixel(width: number, height: number): Promise<{ mean: number; p95: number }> {
        const target = new RenderTarget(width, height, { type: HalfFloatType, samples: 4 });
        const renderer = this.core.renderer;
        const camera = this.technique.camera;
        const putBack = swapMaterials(this.scene, this.countMaterials, countMaterial);
        const previousTarget = renderer.getRenderTarget();
        const background = this.scene.background;
        try {
            // Start from 0 everywhere: a black clear, whatever the scene's background is.
            this.scene.background = null;
            renderer.setRenderTarget(target);
            renderer.setClearColor(0x000000, 0);
            this.originalRender(this.scene, camera);
            // Compiles in the background: draw until it's all there.
            for (let attempt = 0; attempt < 50; attempt++) {
                await this.core.device.queue.onSubmittedWorkDone();
                await new Promise((resolve) => setTimeout(resolve, 20));
                this.originalRender(this.scene, camera);
            }
            await this.core.device.queue.onSubmittedWorkDone();
            const pixels = await readTarget(renderer, target);
            const counts: number[] = [];
            for (let index = 0; index < pixels.length; index += 4) counts.push(pixels[index]);
            counts.sort((a, b) => a - b);
            const mean = counts.reduce((sum, count) => sum + count, 0) / counts.length;
            return { mean, p95: counts[Math.min(counts.length - 1, Math.floor(0.95 * counts.length))] };
        } finally {
            this.scene.background = background;
            renderer.setRenderTarget(previousTarget);
            putBack();
            target.dispose();
            // The count's timestamps would land in the next timed window.
            await readThreePasses(renderer, this.labeler);
        }
    }

    /** The scene target's picture after a few frames of `variant`, with the update paused. */
    async picture(prepass: boolean): Promise<Float32Array> {
        this.apply('on-paused');
        this.drawMode = prepass ? 'prepass' : 'normal';
        for (let index = 0; index < 10; index++) await this.frame();
        return readTarget(this.core.renderer, this.internals.target);
    }
}

function meanColour(pixels: Float32Array): number[] {
    const sums = [0, 0, 0];
    const count = pixels.length / 4;
    for (let index = 0; index < pixels.length; index += 4) for (let channel = 0; channel < 3; channel++) sums[channel] += pixels[index + channel];
    return sums.map((sum) => sum / count);
}

/** Share of pixels whose luminance differs by more than 2% (plus a small absolute floor for near-black). */
function shareDiffering(a: Float32Array, b: Float32Array): number {
    let differing = 0;
    const luminance = (pixels: Float32Array, index: number) => 0.2126 * pixels[index] + 0.7152 * pixels[index + 1] + 0.0722 * pixels[index + 2];
    for (let index = 0; index < a.length; index += 4) {
        const la = luminance(a, index);
        const lb = luminance(b, index);
        if (Math.abs(la - lb) > 0.02 * Math.max(la, lb) + 1e-4) differing++;
    }
    return differing / (a.length / 4);
}

function summarizeVariant(rounds: FrameRecord[][]): VariantResult {
    const perRound = rounds.map((frames) => frames.filter((frame) => !frame.invalid));
    const labels = new Set<string>();
    const names = new Set<string>();
    for (const frames of perRound) {
        for (const frame of frames) {
            Object.keys(frame.three).forEach((label) => labels.add(label));
            Object.keys(frame.platform).forEach((name) => names.add(name));
        }
    }
    const roundMedian = (pick: (frame: FrameRecord) => number) => median(perRound.filter((frames) => frames.length > 0).map((frames) => median(frames.map(pick))));
    const threeMs: Record<string, number> = {};
    for (const label of labels) threeMs[label] = roundMedian((frame) => frame.three[label] ?? 0);
    const platformMs: Record<string, number> = {};
    for (const name of names) platformMs[name] = roundMedian((frame) => frame.platform[name] ?? 0);
    const threeTotal = (frame: FrameRecord) => Object.values(frame.three).reduce((sum, ms) => sum + ms, 0);
    const platformTotal = (frame: FrameRecord) => Object.values(frame.platform).reduce((sum, ms) => sum + ms, 0);
    return {
        threeMs,
        platformMs,
        threeTotalMs: roundMedian(threeTotal),
        platformTotalMs: roundMedian(platformTotal),
        frameMs: roundMedian((frame) => threeTotal(frame) + platformTotal(frame)),
        frameMsPerRound: perRound.filter((frames) => frames.length > 0).map((frames) => median(frames.map((frame) => threeTotal(frame) + platformTotal(frame)))),
        probesUpdated: roundMedian((frame) => frame.probesUpdated),
        framesUsed: perRound.reduce((sum, frames) => sum + frames.length, 0),
    };
}

function formatVariant(id: VariantId, result: VariantResult): string {
    const three = Object.entries(result.threeMs)
        .map(([label, ms]) => `${label} ${ms.toFixed(2)}`)
        .join(', ');
    const platform = Object.entries(result.platformMs)
        .map(([name, ms]) => `${name} ${ms.toFixed(3)}`)
        .join(', ');
    return `  ${id}: frame ${result.frameMs.toFixed(2)} ms (three ${result.threeTotalMs.toFixed(2)}: ${three}; platform ${result.platformTotalMs.toFixed(2)}${platform ? `: ${platform}` : ''}); probes ${result.probesUpdated}; rounds ${result.frameMsPerRound.map((ms) => ms.toFixed(1)).join('/')}`;
}

async function waitForDdgi(core: ViewerCore, technique: DdgiTechnique): Promise<void> {
    await technique.warmUp();
    if (technique.failed) throw new Error(`DDGI could not start: ${technique.failed}`);
    while (!technique.ready) {
        for (let step = 0; step < technique.prepareStepsPerBatch && !technique.ready; step++) technique.encodePrepareStep();
        await core.device.queue.onSubmittedWorkDone();
    }
}

export async function runBreakdown(canvas: HTMLCanvasElement, options: BreakdownOptions): Promise<BreakdownResult> {
    canvas.width = options.width;
    canvas.height = options.height;
    canvas.style.width = `${options.width}px`;
    canvas.style.height = `${options.height}px`;
    const core = await ViewerCore.create(canvas);
    core.resize(options.width, options.height);
    const manifests = loadAllManifests();
    const results: SceneBreakdown[] = [];

    for (const id of options.scenes) {
        const manifest = manifests.find((candidate) => candidate.id === id);
        const result: SceneBreakdown = {
            scene: id,
            triangles: 0,
            meshes: 0,
            materials: 0,
            probes: 0,
            computeCoverage: null,
            variants: {},
            fragmentsPerPixel: null,
            fragmentsPerPixelP95: null,
            prepassImage: null,
            error: null,
        };
        results.push(result);
        if (!manifest) {
            result.error = `unknown scene '${id}'`;
            continue;
        }
        let run: SceneRun | null = null;
        try {
            options.log(`scene ${id}: loading`);
            core.layout = { left: 'none', right: null, giMode: 'on', split: 0.5 };
            const loaded = await core.loadScene(manifest, () => {});
            useFirstBookmark(core, manifest);
            result.triangles = loaded.stats.triangles;
            Object.assign(result, countSceneParts(loaded.scene));
            core.setLayout({ left: 'ddgi', right: null, giMode: 'on' });
            const technique = core.getTechnique('ddgi') as DdgiTechnique;
            await waitForDdgi(core, technique);
            const shadowCameras = new Set<Camera>();
            for (const light of loaded.directLights) {
                const shadow = (light as { shadow?: { camera?: Camera } }).shadow;
                if (shadow?.camera) shadowCameras.add(shadow.camera);
            }
            run = new SceneRun(core, technique, loaded.scene, shadowCameras, options.burst);
            result.probes = Number(technique.describeSettings().probes);
            options.log(`scene ${id}: ${result.triangles} triangles, ${result.meshes} meshes, ${result.materials} materials, ${result.probes} probes`);

            const frames = new Map<VariantId, FrameRecord[][]>(options.variants.map((variant) => [variant, []]));
            const shown = new Set<VariantId>();
            for (let round = 0; round < options.rounds; round++) {
                const order = round % 2 === 0 ? options.variants : [...options.variants].reverse();
                for (const variant of order) {
                    run.apply(variant);
                    // The first showing waits about 30 frames for compiles (fewer samples when each is a burst of frames).
                    await run.settle(shown.has(variant) ? 8 : Math.max(8, Math.ceil(30 / options.burst)));
                    shown.add(variant);
                    const sampled: FrameRecord[] = [];
                    for (let index = 0; index < options.sampleFrames; index++) sampled.push(await run.frame());
                    frames.get(variant)!.push(sampled);
                }
            }
            for (const variant of options.variants) {
                const summary = summarizeVariant(frames.get(variant)!);
                result.variants[variant] = summary;
                options.log(formatVariant(variant, summary));
            }
            result.computeCoverage = (technique as unknown as DdgiInternals).threeGI?.light.computeCoverage ?? null;

            const normal = await run.picture(false);
            const prepass = await run.picture(true);
            result.prepassImage = { meanNormal: meanColour(normal), meanPrepass: meanColour(prepass), pixelsDiffering: shareDiffering(normal, prepass) };
            run.resetBudget();
            core.setLayout({ giMode: 'on' });
            const fragments = await run.fragmentsPerPixel(options.width, options.height);
            result.fragmentsPerPixel = fragments.mean;
            result.fragmentsPerPixelP95 = fragments.p95;
            options.log(
                `  fragments per pixel ${fragments.mean.toFixed(2)} (p95 ${fragments.p95.toFixed(1)}); prepass picture: ${(result.prepassImage.pixelsDiffering * 100).toFixed(2)}% of pixels differ, ` +
                    `mean ${result.prepassImage.meanNormal.map((value) => value.toFixed(4)).join('/')} against ${result.prepassImage.meanPrepass.map((value) => value.toFixed(4)).join('/')}; coverage ${result.computeCoverage}`,
            );
        } catch (error) {
            result.error = error instanceof Error ? error.message : String(error);
            options.log(`scene ${id}: ${result.error}`);
        } finally {
            run?.restore();
        }
    }

    const info = core.adapterInfo;
    const gpuErrors = [...core.gpuErrors];
    const features = [...core.device.features].sort();
    core.dispose();
    return {
        scenes: results,
        adapter: { vendor: info.vendor, architecture: info.architecture, description: info.description, device: info.device, features },
        options: { scenes: options.scenes, variants: options.variants, burst: options.burst, rounds: options.rounds, sampleFrames: options.sampleFrames, width: options.width, height: options.height },
        gpuErrors,
    };
}
