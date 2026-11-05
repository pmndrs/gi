/**
 * Shared by the `three-vb-gi` and `three-screen-gi` pages: Sponza (with
 * curtains) loaded by the viewer, DDGI warmed up through the viewer's own
 * technique, and a small timing harness that draws frames itself instead of
 * through the viewer's render pipeline, so each page can choose the drawing
 * pipeline, the render targets and the passes.
 *
 * Not a page itself (no index.html).
 *
 * How a frame is timed (GPU-bound, as Dennis asked: back to back, clocks up):
 * each sample draws `burst` frames back to back, waits for the queue once,
 * and divides three's per-pass GPU times by `burst`. three's passes are
 * labelled by what the page was drawing when the pass began (the label the
 * page sets before each renderer call), except shadow-map passes, which are
 * labelled `shadow` by their camera. The DDGI field is frozen after the
 * warm-up (no probe updates) unless a variant updates it itself, so every
 * variant shades the same field.
 */

import { DataUtils, HalfFloatType, type Camera, type Material, type Mesh, type Object3D, type PerspectiveCamera, type Scene } from 'three';
import type { RenderTarget, WebGPURenderer } from 'three/webgpu';
import type { SceneManifest } from '../../../../scenes/src/manifest';
import { ViewerCore } from '../../../../apps/viewer/src/core/viewer-core';
import { median } from '../../../../apps/viewer/src/bench/measurement';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests';
import type { DdgiTechnique } from '../../../../apps/viewer/src/techniques/ddgi-technique';
import type { FrameBudget, GILight, ThreeGI } from '../../../../apps/viewer/src/ddgi/library';
import { LAYER_DDGI, LAYER_DIRECT_LIGHTS, LAYER_VIEW_GEOMETRY } from '../../../../apps/viewer/src/scene/layers';

// -- three internals the harness reads (r187) ----------------------------------------------------

interface RenderContextShape {
    camera: Camera | null;
}

interface QueryPool {
    currentQueryIndex: number;
    timestamps: Map<string, number>;
}

interface BackendShape {
    timestampQueryPool?: Partial<Record<'render' | 'compute', QueryPool | null>>;
    beginRender(context: RenderContextShape): void;
    beginCompute(group: unknown): void;
    get(object: object): { timestampUID?: string };
}

/** The parts of three's renderer its animation loop touches to start a frame (`Animation.js`, r187). */
interface FrameAdvance {
    info: { autoReset: boolean; reset(): void; frame: number; render: { drawCalls: number } };
    _nodes: { nodeFrame: { update(): void; frameId: number } };
}

/** The parts of the viewer's DDGI technique the harness reads. Private there; this is a measurement tool. */
interface DdgiInternals {
    threeGI: ThreeGI | null;
    platform: { stats: { frame: number; passes: ReadonlyArray<{ name: string; gpuMs: number | undefined; frame: number }> } };
    ddgi: { counters: Record<string, number> };
}

// -- set-up ----------------------------------------------------------------------------------------

export interface SponzaSetup {
    readonly core: ViewerCore;
    readonly renderer: WebGPURenderer;
    readonly device: GPUDevice;
    readonly scene: Scene;
    /** DDGI's own viewing camera, placed at the scene's first bookmark. Its layers are the page's to set. */
    readonly camera: PerspectiveCamera;
    readonly threeGI: ThreeGI;
    readonly giLight: GILight;
    /** The budget the viewer runs with (budget v2, calibrated in the warm-up). */
    readonly normalBudget: FrameBudget;
    readonly shadowCameras: ReadonlySet<Camera>;
    readonly internals: DdgiInternals;
    readonly triangles: number;
    readonly startupMs: number;
}

export function useFirstBookmark(core: ViewerCore, manifest: SceneManifest): void {
    const bookmark = manifest.cameras[0];
    core.camera.position.set(...bookmark.position);
    core.camera.fov = bookmark.fovYDeg;
    core.camera.lookAt(...bookmark.target);
    core.camera.updateProjectionMatrix();
    core.camera.updateMatrixWorld(true);
}

async function waitForDdgi(core: ViewerCore, technique: DdgiTechnique): Promise<void> {
    await technique.warmUp();
    if (technique.failed) throw new Error(`DDGI could not start: ${technique.failed}`);
    while (!technique.ready) {
        for (let step = 0; step < technique.prepareStepsPerBatch && !technique.ready; step++) technique.encodePrepareStep();
        await core.device.queue.onSubmittedWorkDone();
    }
    if (technique.failed) throw new Error(`DDGI failed in its warm-up: ${technique.failed}`);
}

/**
 * Loads the scene through the viewer, warms DDGI up the viewer's way (16 passes or 8 s, then
 * the budget calibration), and freezes the field. From here on the page draws every frame itself.
 */
export async function loadSceneWithDdgi(canvas: HTMLCanvasElement, sceneId: string, width: number, height: number, log: (message: string) => void): Promise<SponzaSetup> {
    const startedAt = performance.now();
    canvas.width = width;
    canvas.height = height;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const core = await ViewerCore.create(canvas);
    core.renderer.setPixelRatio(1);
    core.resize(width, height);
    const manifest = loadAllManifests().find((candidate) => candidate.id === sceneId);
    if (!manifest) throw new Error(`unknown scene '${sceneId}'`);
    log(`${sceneId}: loading`);
    core.layout = { left: 'none', right: null, giMode: 'on', split: 0.5 };
    const loaded = await core.loadScene(manifest, () => {});
    useFirstBookmark(core, manifest);
    core.setLayout({ left: 'ddgi', right: null, giMode: 'on' });
    const technique = core.getTechnique('ddgi') as DdgiTechnique;
    await waitForDdgi(core, technique);
    technique.syncCamera(core.camera);
    const internals = technique as unknown as DdgiInternals;
    const threeGI = internals.threeGI;
    if (!threeGI) throw new Error('DDGI has no ThreeGI after its warm-up');
    const normalBudget = threeGI.budget;
    // Freeze the field: only a variant that asks for it updates probes.
    threeGI.budget = { paused: true };
    const shadowCameras = new Set<Camera>();
    for (const light of loaded.directLights) {
        const shadow = (light as { shadow?: { camera?: Camera } }).shadow;
        if (shadow?.camera) shadowCameras.add(shadow.camera);
    }
    const startupMs = performance.now() - startedAt;
    log(`${sceneId}: ${loaded.stats.triangles} triangles; DDGI ready after ${(startupMs / 1000).toFixed(1)} s; coverage per pixel ${threeGI.light.computeCoverage}`);
    return {
        core,
        renderer: core.renderer,
        device: core.device,
        scene: loaded.scene,
        camera: technique.camera,
        threeGI,
        giLight: threeGI.light,
        normalBudget,
        shadowCameras,
        internals,
        triangles: loaded.stats.triangles,
        startupMs,
    };
}

// -- camera layers ---------------------------------------------------------------------------------

/** The layer the pages put their own lights on (the viewer uses 0-5). */
export const LAYER_PAGE_LIGHT = 11;

/**
 * What the viewing camera sees: the model and the direct lights always; DDGI's per-fragment
 * light (`fragment`), the page's own light (`page`), or neither (`off`). Lights follow the
 * camera's layers in three, so this switches GI without touching intensities.
 */
export function setGiLayers(camera: Camera, gi: 'off' | 'fragment' | 'page'): void {
    camera.layers.disableAll();
    camera.layers.enable(LAYER_VIEW_GEOMETRY);
    camera.layers.enable(LAYER_DIRECT_LIGHTS);
    if (gi === 'fragment') camera.layers.enable(LAYER_DDGI);
    if (gi === 'page') camera.layers.enable(LAYER_PAGE_LIGHT);
}

// -- timing ----------------------------------------------------------------------------------------

/** A pass longer than this is a timestamp glitch. */
const MAX_PLAUSIBLE_PASS_MS = 60_000;

/**
 * Labels three's timestamped passes. three keys each pass's duration by a uid it builds just
 * before `beginRender` / `beginCompute`; the wrapper records the label current at that moment.
 */
export class PassTimer {
    /** What the page is drawing now; set before each renderer call. */
    label = 'unlabelled';
    private readonly labels = new Map<string, string>();

    constructor(
        private readonly renderer: WebGPURenderer,
        private readonly shadowCameras: ReadonlySet<Camera>,
    ) {
        const backend = renderer.backend as unknown as BackendShape;
        const beginRender = backend.beginRender.bind(backend);
        backend.beginRender = (context: RenderContextShape) => {
            const uid = backend.get(context).timestampUID;
            const isShadow = context.camera !== null && this.shadowCameras.has(context.camera);
            if (uid) this.labels.set(uid, isShadow ? 'shadow' : this.label);
            beginRender(context);
        };
        const beginCompute = backend.beginCompute.bind(backend);
        backend.beginCompute = (group: unknown) => {
            const uid = typeof group === 'object' && group !== null ? backend.get(group).timestampUID : undefined;
            if (uid) this.labels.set(uid, `${this.label} compute`);
            beginCompute(group);
        };
    }

    /** Reads and empties three's query pools, summing durations by label. */
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
                const label = this.labels.get(uid) ?? 'unlabelled';
                byLabel[label] = (byLabel[label] ?? 0) + ms;
            }
        }
        this.labels.clear();
        return { byLabel, invalid };
    }
}

/** Starts a new three frame the way its animation loop does, so frame-type nodes (shadow maps) update once per frame. */
export function advanceThreeFrame(renderer: WebGPURenderer): void {
    const frame = renderer as unknown as FrameAdvance;
    if (frame.info.autoReset) frame.info.reset();
    frame._nodes.nodeFrame.update();
    frame.info.frame = frame._nodes.nodeFrame.frameId;
}

function drawCalls(renderer: WebGPURenderer): number {
    return (renderer as unknown as FrameAdvance).info.render.drawCalls;
}

/** One way of drawing the frame. `draw` records one frame's GPU work; it must not wait. */
export interface Variant {
    readonly id: string;
    /** Called once before the variant's frames in each round (pipeline, layers, targets). */
    apply(): void;
    draw(): void;
    /** Called after the variant's frames in each round. */
    leave?(): void;
}

export interface VariantResult {
    /** Median over rounds of each round's median, ms per frame, by pass label. */
    passMs: Record<string, number>;
    /** Every three pass in the frame. */
    totalMs: number;
    /** Every pass except shadow maps (the same in every variant). */
    totalWithoutShadowMs: number;
    /** The platform's own GPU passes (DDGI's update), when the variant updates the field. */
    platformMs: number;
    probesUpdated: number;
    /** Each round's median frame total, for the spread. */
    totalPerRound: number[];
    drawCalls: number;
    framesUsed: number;
}

interface Sample {
    passMs: Record<string, number>;
    platformMs: number;
    probesUpdated: number;
    invalid: boolean;
    drawCalls: number;
}

export interface MeasureOptions {
    rounds: number;
    samples: number;
    burst: number;
    log(message: string): void;
}

export class FrameHarness {
    readonly timer: PassTimer;
    /** The platform's frame number when the current variant began. */
    private platformFloor = 0;

    constructor(private readonly setup: SponzaSetup) {
        this.timer = new PassTimer(setup.renderer, setup.shadowCameras);
    }

    /** One sample: `burst` frames back to back, one wait, three's pass times divided by `burst`. */
    async sample(variant: Variant, burst: number): Promise<Sample> {
        const { renderer, device, internals } = this.setup;
        let calls = 0;
        for (let frame = 0; frame < burst; frame++) {
            advanceThreeFrame(renderer);
            variant.draw();
            calls = drawCalls(renderer);
        }
        await device.queue.onSubmittedWorkDone();
        const passes = await this.timer.read();
        for (const label of Object.keys(passes.byLabel)) passes.byLabel[label] /= burst;
        // The platform keeps its newest timed frame, which arrives a frame or two late: count it only
        // if it was encoded after this variant began (otherwise it belongs to another variant).
        let platformMs = 0;
        for (const pass of internals.platform.stats.passes) {
            if (pass.frame <= this.platformFloor || pass.gpuMs === undefined || !(pass.gpuMs >= 0)) continue;
            platformMs += pass.gpuMs;
        }
        const probesUpdated = platformMs > 0 ? (internals.ddgi.counters['probes updated'] ?? 0) : 0;
        return { passMs: passes.byLabel, platformMs, probesUpdated, invalid: passes.invalid, drawCalls: calls };
    }

    /** Draws until the draw-call count and the frame time stop changing (pipelines compile in the background). */
    async settle(variant: Variant, minimumSamples: number, burst: number): Promise<void> {
        const calls: number[] = [];
        const times: number[] = [];
        for (let index = 0; index < 400; index++) {
            const sample = await this.sample(variant, burst);
            calls.push(sample.drawCalls);
            times.push(Object.values(sample.passMs).reduce((sum, ms) => sum + ms, 0));
            if (calls.length < minimumSamples) continue;
            const recentCalls = calls.slice(-6);
            const recentTimes = times.slice(-6);
            const typical = median(recentTimes);
            if (recentCalls.every((count) => count === recentCalls[0]) && recentTimes.every((ms) => Math.abs(ms - typical) <= 0.25 * typical + 0.02)) return;
        }
    }

    /** Rounds of every variant, in alternating order, each settled first. */
    async measure(variants: readonly Variant[], options: MeasureOptions): Promise<Record<string, VariantResult>> {
        const samples = new Map<string, Sample[][]>(variants.map((variant) => [variant.id, []]));
        const shown = new Set<string>();
        for (let round = 0; round < options.rounds; round++) {
            const order = round % 2 === 0 ? variants : [...variants].reverse();
            for (const variant of order) {
                variant.apply();
                this.platformFloor = this.setup.internals.platform.stats.frame;
                try {
                    // The first showing waits longer: shaders compile in the background.
                    await this.settle(variant, shown.has(variant.id) ? 4 : Math.max(8, Math.ceil(40 / options.burst)), options.burst);
                    shown.add(variant.id);
                    const taken: Sample[] = [];
                    for (let index = 0; index < options.samples; index++) taken.push(await this.sample(variant, options.burst));
                    samples.get(variant.id)!.push(taken);
                } finally {
                    variant.leave?.();
                }
            }
            options.log(`round ${round + 1} of ${options.rounds} done`);
        }
        const results: Record<string, VariantResult> = {};
        for (const variant of variants) {
            results[variant.id] = summarize(samples.get(variant.id)!);
            options.log(formatResult(variant.id, results[variant.id]));
        }
        return results;
    }
}

function summarize(rounds: Sample[][]): VariantResult {
    const perRound = rounds.map((taken) => taken.filter((sample) => !sample.invalid)).filter((taken) => taken.length > 0);
    const labels = new Set<string>();
    for (const taken of perRound) for (const sample of taken) Object.keys(sample.passMs).forEach((label) => labels.add(label));
    const roundMedian = (pick: (sample: Sample) => number) => median(perRound.map((taken) => median(taken.map(pick))));
    const total = (sample: Sample) => Object.values(sample.passMs).reduce((sum, ms) => sum + ms, 0);
    const passMs: Record<string, number> = {};
    for (const label of labels) passMs[label] = roundMedian((sample) => sample.passMs[label] ?? 0);
    return {
        passMs,
        totalMs: roundMedian(total),
        totalWithoutShadowMs: roundMedian((sample) => total(sample) - (sample.passMs.shadow ?? 0)),
        platformMs: roundMedian((sample) => sample.platformMs),
        probesUpdated: roundMedian((sample) => sample.probesUpdated),
        totalPerRound: perRound.map((taken) => median(taken.map(total))),
        drawCalls: roundMedian((sample) => sample.drawCalls),
        framesUsed: perRound.reduce((sum, taken) => sum + taken.length, 0),
    };
}

function formatResult(id: string, result: VariantResult): string {
    const passes = Object.entries(result.passMs)
        .map(([label, ms]) => `${label} ${ms.toFixed(2)}`)
        .join(', ');
    const platform = result.platformMs > 0 ? `; DDGI update ${result.platformMs.toFixed(2)} ms, ${result.probesUpdated} probes` : '';
    return `  ${id}: ${result.totalMs.toFixed(2)} ms (${passes})${platform}; rounds ${result.totalPerRound.map((ms) => ms.toFixed(2)).join('/')}; ${result.drawCalls} draw calls`;
}

// -- pictures --------------------------------------------------------------------------------------

/** Reads an RGBA render target as floats (half-float targets come back as raw 16-bit words). */
export async function readTarget(renderer: WebGPURenderer, target: RenderTarget, textureIndex = 0): Promise<Float32Array> {
    const raw = (await renderer.readRenderTargetPixelsAsync(target, 0, 0, target.width, target.height, textureIndex)) as ArrayLike<number> & { BYTES_PER_ELEMENT: number };
    const isHalf = raw.BYTES_PER_ELEMENT === 2 && target.textures[textureIndex].type === HalfFloatType;
    const out = new Float32Array(raw.length);
    for (let index = 0; index < raw.length; index++) out[index] = isHalf ? DataUtils.fromHalfFloat(raw[index]) : raw[index];
    return out;
}

export function meanColour(pixels: Float32Array): number[] {
    const sums = [0, 0, 0];
    const count = pixels.length / 4;
    for (let index = 0; index < pixels.length; index += 4) for (let channel = 0; channel < 3; channel++) sums[channel] += pixels[index + channel];
    return sums.map((sum) => sum / count);
}

/** Share of pixels whose luminance differs by more than `tolerance` (relative, plus a small floor for near-black). */
export function shareDiffering(a: Float32Array, b: Float32Array, tolerance = 0.02): number {
    let differing = 0;
    const luminance = (pixels: Float32Array, index: number) => 0.2126 * pixels[index] + 0.7152 * pixels[index + 1] + 0.0722 * pixels[index + 2];
    for (let index = 0; index < a.length; index += 4) {
        const la = luminance(a, index);
        const lb = luminance(b, index);
        if (Math.abs(la - lb) > tolerance * Math.max(la, lb) + 1e-4) differing++;
    }
    return differing / (a.length / 4);
}

export interface PictureComparison {
    meanA: number[];
    meanB: number[];
    /** Relative difference of the mean luminance, B against A. */
    meanLuminanceRatio: number;
    differingOver2Percent: number;
    differingOver10Percent: number;
}

export function comparePictures(a: Float32Array, b: Float32Array): PictureComparison {
    const meanA = meanColour(a);
    const meanB = meanColour(b);
    const luminance = (rgb: number[]) => 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    return {
        meanA,
        meanB,
        meanLuminanceRatio: luminance(meanB) / luminance(meanA),
        differingOver2Percent: shareDiffering(a, b, 0.02),
        differingOver10Percent: shareDiffering(a, b, 0.1),
    };
}

// -- scene walking ---------------------------------------------------------------------------------

/** Every visible mesh with its materials (one per geometry group for multi-material meshes). */
export function visibleMeshes(scene: Scene): Mesh[] {
    const meshes: Mesh[] = [];
    scene.traverse((object: Object3D) => {
        const mesh = object as Mesh;
        if (mesh.isMesh && mesh.visible) meshes.push(mesh);
    });
    return meshes;
}

export function materialsOf(mesh: Mesh): Material[] {
    return Array.isArray(mesh.material) ? mesh.material : [mesh.material];
}

export function triangleCount(mesh: Mesh): number {
    const geometry = mesh.geometry;
    const count = geometry.index ? geometry.index.count : (geometry.attributes.position?.count ?? 0);
    return Math.floor(count / 3);
}

// -- page plumbing ---------------------------------------------------------------------------------

export function listFromEnv(value: unknown): string[] | null {
    return typeof value === 'string' && value.trim() !== '' ? value.split(',').map((item) => item.trim()) : null;
}

export function countFromEnv(value: unknown, fallback: number): number {
    const parsed = typeof value === 'string' ? Number(value) : NaN;
    return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

/** The page's log, shown on the page and kept for the report. */
export function createLog(element: HTMLElement): { lines: string[]; log(message: string): void } {
    const lines: string[] = [];
    return {
        lines,
        log(message: string): void {
            lines.push(message);
            element.textContent = lines.join('\n');
        },
    };
}

/** Everything the page itself sees as an error, so it lands in the report next to the numbers. */
export function collectPageErrors(): string[] {
    const errors: string[] = [];
    window.addEventListener('error', (event) => errors.push(`page error: ${event.message}`));
    window.addEventListener('unhandledrejection', (event) => errors.push(`unhandled rejection: ${String(event.reason)}`));
    const originalConsoleError = console.error.bind(console);
    console.error = (...args: unknown[]) => {
        errors.push(`console.error: ${args.map(String).join(' ')}`);
        originalConsoleError(...args);
    };
    return errors;
}
