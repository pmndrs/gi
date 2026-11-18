/**
 * T1.14 on Sponza with its curtains (`sponza`): the GI composite against GI
 * per fragment, and what showing GI costs each way
 * (`docs/design/three-composite.md`, DECISIONS 2026-11-05, T1.14 and
 * "curtained Sponza, option A").
 *
 * 1. **Cost**, GPU-bound, the field frozen (shading only; DECISIONS choice
 *    (b) judges the update separately). Every variant is three's own
 *    `RenderPipeline` with the same output (tone mapping to the canvas), and
 *    the sun's shadow map is in every frame. Variants, drawn interleaved
 *    frame by frame (off, mrt, composite, ..., forward, off, mrt, ...), so
 *    all of them see the same clock (composite-check.ts, `timeInterleaved`):
 *    - `off`: `pass(scene, camera)`, the GI light hidden;
 *    - `mrt`: the pass with `mrt({ output, giAlbedo, normal })` and the
 *      composite's mark (no per-fragment GI), showing its colour;
 *    - `composite`: that pass and `ddgi` without occlusion;
 *    - `composite+gtao`: the composite with three's GTAO on the GI term
 *      (1 m radius, full resolution): the default path when this page was
 *      written (M1; the composite has no occlusion by default since
 *      2026-11-18, DECISIONS), and still the name of the bar's delta;
 *    - `composite+gtao-half`: the same at half resolution;
 *    - `forward`: GI per fragment (today's path), on a pass with the same
 *      MRT outputs as `mrt` but no mark, so `forward − mrt` is what GI per
 *      fragment adds to the same pass.
 *    Each difference is the median of per-cycle differences between
 *    neighbouring frames, with its spread. Each variant has its own pass
 *    and camera, and the GI light is on a layer only the GI variants'
 *    cameras see: nothing changes between frames, so three's caches hold
 *    and the page stays GPU-bound (timeInterleaved says why that matters).
 *    Two passes with the same targets but different lights make three
 *    rebuild their objects every frame (measured: 130–280 ms of CPU per
 *    frame), so `off` is the only variant with a single target, and
 *    `forward` has the MRT targets of the GI variants. The sun's shadow map
 *    is drawn once, while compiling, not every frame: the scene is static,
 *    the shadow pass is the same in every variant (so it cancels in every
 *    difference), and drawing it each frame under each variant's pass
 *    context cost 130 ms of CPU per frame.
 *    Two timed sections:
 *    - **section 1, 2560 × 1440:** the bar, the cost of showing GI on the
 *      default path (`composite+gtao` − `off`) ≤ 1.0 ms on the RTX 3060;
 *    - **section 2, 1920 × 1080:** the same variants at the reference
 *      size, for the composite pass's cost there.
 *    Run it through `run.ts`, which adds each section's GPU clock and use,
 *    the clock-normalised numbers, and the bench lock's verdict.
 * 2. **Match** at the stored camera (bookmark 1, 1920 × 1080), the composite
 *    without occlusion against per fragment (the composite page's check):
 *    first with Sponza's own materials, then with the untextured GI material
 *    table. Bar: mean ≤ 0.5%, p95 ≤ 2% on the GI-table materials.
 *
 * Settings (vite exposes `VITE_*` variables to the page):
 *   VITE_DDGI_SKIP        'timing' or 'match' to skip a part
 *   VITE_DDGI_GI_FRAMES   GI frames of 4096 probes before the field freezes (default 120)
 *   VITE_DDGI_SECONDS     the 2560 × 1440 timed section, seconds (default 10)
 *   VITE_DDGI_SECONDS_1080 the 1920 × 1080 timed section, seconds (default 4; 0 skips it)
 *   VITE_DDGI_COOL        idle seconds after loading, before timing (default 25)
 *   VITE_DDGI_RAMP        GPU-bound seconds before each timed section, for the clock to ramp up (default 2)
 *   VITE_DDGI_STATIC_SHADOWS '0': redraw the sun's shadow map every frame (default '1': once; see above)
 *   VITE_DDGI_SET         which variants (measurement for Dennis's decision; shared/variants.ts):
 *                           'default': off, mrt, composite, composite + GTAO full and half, forward;
 *                           'native': off, composite, and composite with GTAO full, half, quarter,
 *                             half averaged over 16 frames (static camera), SSAO;
 *                           'upscaled': through @pmndrs/upscaler at its Quality and Performance presets,
 *                             each off (upscaleScene, as the viewer does), composite, composite + GTAO, + GTAO half.
 *   VITE_DDGI_TARGETS     'half' (default) or 'byte': the format of giAlbedo and the normal.
 *                         One format per run: in r187 two MRT passes over one scene whose targets
 *                         differ only in format reuse each other's pipelines (NOTES).
 *
 * Manual (it needs the downloaded Sponza). Timing goes through run.ts under the bench lock:
 *   node tools/reference/bench-lock.ts run --what "three composite sponza" -- node tools/gpu/pages/three-composite-sponza/run.ts --gpu nvidia
 */

import { RenderPipeline, UnsignedByteType, type Node, type PerspectiveCamera } from 'three/webgpu';
import { mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import { assert } from '../../client.ts';
import { markCompositePass, type MarkablePass } from '../../../../packages/ddgi-three/src/composite-mark.ts';
import { AccumulatedOcclusion, compositePassAt, gtaoOn, ssaoOn, upscaledComposite, upscaledScene, upscalerMs, type UpscalePreset } from '../three-composite-shared/variants.ts';
import { DDGICompositeNode } from '../../../../packages/ddgi-three/src/composite.ts';
import { giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT } from '../../../../packages/ddgi-three/src/composite.ts';
import { runThreePage, type ThreePageContext } from '../three-shared/harness.ts';
import {
    CompositeRoutes,
    HdrReader,
    PassClock,
    SHOT_SIZE,
    advanceThreeFrame,
    formatMatch,
    loadCompositeScene,
    matchStatistics,
    opaqueMask,
    shadowCameras,
    timeInterleaved,
    useGiMaterialTable,
    type CompositeScene,
    type DeltaDefinition,
    type FrameVariant,
    type InterleavedTiming,
    type MatchStatistics,
} from '../three-composite-shared/composite-check.ts';

const TIMING_SIZE = { width: 2560, height: 1440 } as const;
const GI_FRAMES = 120;
const PROBES_PER_FRAME = 4096;
const MEAN_BAR = 0.005;
const P95_BAR = 0.02;
/** Cycles drawn back to back between two reads of the timestamp pool. */
const CYCLES_PER_BATCH = 4;
/** Frames each variant is drawn, with a wait, before timing: pipelines compile and draw calls settle. */
const COMPILE_FRAMES = 6;

const env = import.meta.env as Record<string, unknown>;
/** A layer only the GI variants' cameras see; the GI light lives there while timing. */
const GI_LAYER = 30;
const byteTargets = env.VITE_DDGI_TARGETS === 'byte';

function numberFromEnv(value: unknown, fallback: number): number {
    const parsed = typeof value === 'string' && value.trim() !== '' ? Number(value) : NaN;
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

/** The differences reported, each per cycle between neighbouring frames. */
const DELTAS: readonly DeltaDefinition[] = [
    { name: 'showing GI, default path (composite + GTAO full) - off', minuend: 'composite+gtao', subtrahend: 'off', part: 'total' },
    { name: 'showing GI, composite + GTAO half - off', minuend: 'composite+gtao-half', subtrahend: 'off', part: 'total' },
    { name: 'showing GI, composite without occlusion - off', minuend: 'composite', subtrahend: 'off', part: 'total' },
    { name: 'GI per fragment on the same MRT pass (forward - mrt)', minuend: 'forward', subtrahend: 'mrt', part: 'total' },
    { name: 'showing GI, forward per fragment with the MRT targets - off', minuend: 'forward', subtrahend: 'off', part: 'total' },
    { name: 'MRT extra (mrt - off)', minuend: 'mrt', subtrahend: 'off', part: 'total' },
    { name: 'composite pass (quad: composite - mrt)', minuend: 'composite', subtrahend: 'mrt', part: 'quad' },
    { name: 'GTAO full (composite + GTAO - composite)', minuend: 'composite+gtao', subtrahend: 'composite', part: 'total' },
    { name: 'GTAO half (composite + GTAO half - composite)', minuend: 'composite+gtao-half', subtrahend: 'composite', part: 'total' },
];

interface CostReport {
    readonly sections: InterleavedTiming[];
    readonly targets: string;
}

function setViewSize(context: ThreePageContext, scene: CompositeScene, size: { width: number; height: number }): void {
    context.renderer.setSize(size.width, size.height, false);
    scene.camera.aspect = size.width / size.height;
    scene.camera.updateProjectionMatrix();
}

async function timing(context: ThreePageContext, scene: CompositeScene): Promise<CostReport> {
    const { renderer, device, log } = context;
    setViewSize(context, scene, TIMING_SIZE);
    const light = scene.threeGI.light;
    // GI on its own layer: the "off" camera leaves it out, so no variant toggles the light (timeInterleaved).
    light.layers.set(GI_LAYER);
    const cameras: PerspectiveCamera[] = [];
    const cameraFor = (withGI: boolean): PerspectiveCamera => {
        const camera = scene.camera.clone();
        if (withGI) camera.layers.enable(GI_LAYER);
        cameras.push(camera);
        return camera;
    };
    const resizeCameras = () => {
        for (const camera of cameras) {
            camera.aspect = scene.camera.aspect;
            camera.updateProjectionMatrix();
        }
    };

    const pipelines: RenderPipeline[] = [];
    const pipelineOf = (node: unknown) => {
        const pipeline = new RenderPipeline(renderer, node as Node);
        pipelines.push(pipeline);
        return pipeline;
    };
    const set = typeof env.VITE_DDGI_SET === 'string' && env.VITE_DDGI_SET !== '' ? env.VITE_DDGI_SET : 'default';
    const built = set === 'native' ? nativeSet(scene, cameraFor, pipelineOf) : set === 'upscaled' ? upscaledSet(scene, cameraFor, pipelineOf) : defaultSet();
    const variants = built.variants;
    const deltas = built.deltas;
    log(`variant set '${set}': ${variants.map((variant) => variant.id).join(', ')}`);

    function defaultSet(): { variants: FrameVariant[]; deltas: readonly DeltaDefinition[] } {
        const off = pipelineOf(pass(scene.scene, cameraFor(false)));
        // GI per fragment on the MRT targets, unmarked (see the file comment): giAlbedo writes 0 there.
        const forwardPass = pass(scene.scene, cameraFor(true));
        forwardPass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView) }));
        for (const name of [GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT]) {
            const target = forwardPass.getTexture(name);
            if (byteTargets) target.type = UnsignedByteType;
        }
        const forward = pipelineOf(forwardPass.getTextureNode('output'));

        // The MRT pass alone: the composite's outputs and mark, its colour shown as is.
        const mrtPass = pass(scene.scene, cameraFor(true));
        mrtPass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView) }));
        markCompositePass(mrtPass as unknown as MarkablePass);
        // An MRT output is only drawn once its texture exists (MRTNode.setup).
        for (const name of [GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT]) {
            const target = mrtPass.getTexture(name);
            if (byteTargets) target.type = UnsignedByteType;
        }
        const mrtOnly = pipelineOf(mrtPass.getTextureNode('output'));

        const pipelineFor = (occlusion: ConstructorParameters<typeof CompositeRoutes>[1]) => {
            const routes = new CompositeRoutes(scene, { byteTargets, ...occlusion, camera: cameraFor(true) });
            return pipelineOf(routes.composite);
        };
        const composite = pipelineFor({ occlusion: false });
        // The composite's own GTAO at 1 m, full resolution (M1's default; opt-in since 2026-11-18).
        const compositeGtao = pipelineFor({ occlusion: {} });
        const compositeGtaoHalf = pipelineFor({ occlusion: { resolutionScale: 0.5 } });

        const nothing = () => {};
        const variants: FrameVariant[] = [
            { id: 'off', apply: nothing, draw: () => off.render() },
            { id: 'mrt', apply: nothing, draw: () => mrtOnly.render() },
            { id: 'composite', apply: nothing, draw: () => composite.render() },
            { id: 'composite+gtao', apply: nothing, draw: () => compositeGtao.render() },
            { id: 'composite+gtao-half', apply: nothing, draw: () => compositeGtaoHalf.render() },
            { id: 'forward', apply: nothing, draw: () => forward.render() },
        ];
        return { variants, deltas: DELTAS };
    }

    // Compile every variant, then let the GPU cool: loading the scene heated it.
    for (const variant of variants) {
        for (let frame = 0; frame < COMPILE_FRAMES; frame++) {
            advanceThreeFrame(renderer);
            variant.apply();
            variant.draw();
            await device.queue.onSubmittedWorkDone();
        }
    }
    const coolSeconds = numberFromEnv(env.VITE_DDGI_COOL, 25);
    log(`compiled ${variants.length} variants; idle for ${coolSeconds} s`);
    await new Promise((resolve) => setTimeout(resolve, coolSeconds * 1000));

    if (env.VITE_DDGI_STATIC_SHADOWS !== '0') {
        // The scene doesn't move, so the shadow map drawn while compiling stays right; no variant redraws it.
        for (const light of scene.loaded.directLights) {
            const shadow = (light as { shadow?: { autoUpdate: boolean } }).shadow;
            if (shadow) shadow.autoUpdate = false;
        }
        log('shadow maps drawn once (static)');
    }
    const clock = new PassClock(renderer, shadowCameras(scene));
    const rampSeconds = numberFromEnv(env.VITE_DDGI_RAMP, 2);
    const sections: InterleavedTiming[] = [];
    sections.push(
        await timeInterleaved(renderer, device, clock, variants, deltas, {
            label: 'section 1 of 2: 2560 x 1440, the bar',
            seconds: numberFromEnv(env.VITE_DDGI_SECONDS, 10),
            cyclesPerBatch: CYCLES_PER_BATCH,
            rampSeconds,
            log,
        }),
    );
    const seconds1080 = numberFromEnv(env.VITE_DDGI_SECONDS_1080, 4);
    if (seconds1080 > 0) {
        setViewSize(context, scene, SHOT_SIZE);
        resizeCameras();
        // A resize reallocates every pass target: a few frames of each first.
        for (let frame = 0; frame < 2; frame++) {
            for (const variant of variants) {
                advanceThreeFrame(renderer);
                variant.apply();
                variant.draw();
            }
            await device.queue.onSubmittedWorkDone();
        }
        await clock.read();
        sections.push(
            await timeInterleaved(renderer, device, clock, variants, deltas, {
                label: 'section 2 of 2: 1920 x 1080, the composite pass at the reference size',
                seconds: seconds1080,
                cyclesPerBatch: CYCLES_PER_BATCH,
                rampSeconds: 0.5,
                log,
            }),
        );
    }
    light.layers.enableAll();
    for (const pipeline of pipelines) pipeline.dispose();
    return { sections, targets: byteTargets ? 'byte' : 'half' };
}

type CameraFor = (withGI: boolean) => PerspectiveCamera;
type PipelineOf = (node: unknown) => RenderPipeline;
const nothing = () => {};

/**
 * The 'native' set (file comment): cheaper ways to show GI at native resolution. Each
 * composite has its own pass at full resolution.
 */
function nativeSet(scene: CompositeScene, cameraFor: CameraFor, pipelineOf: PipelineOf): { variants: FrameVariant[]; deltas: readonly DeltaDefinition[] } {
    const off = pipelineOf(pass(scene.scene, cameraFor(false)));
    const compositeWith = (make: (scenePass: ReturnType<typeof compositePassAt>) => ConstructorParameters<typeof DDGICompositeNode>[2]) => {
        const scenePass = compositePassAt(scene.scene, cameraFor(true), 1, false);
        return pipelineOf(new DDGICompositeNode(scenePass, scene.threeGI, make(scenePass)));
    };
    const pipelines: Record<string, RenderPipeline> = {
        off,
        composite: compositeWith(() => ({ occlusion: false })),
        'composite+gtao': compositeWith(() => ({ occlusion: {} })),
        'composite+gtao-half': compositeWith(() => ({ occlusion: { resolutionScale: 0.5 } })),
        'composite+gtao-quarter': compositeWith(() => ({ occlusion: { resolutionScale: 0.25 } })),
        'composite+gtao-half-temporal': compositeWith((scenePass) => ({ occlusion: { source: new AccumulatedOcclusion(gtaoOn(scenePass, { resolutionScale: 0.5 }), 16) } })),
        'composite+ssao': compositeWith((scenePass) => ({ occlusion: { source: ssaoOn(scenePass) } })),
    };
    const variants = Object.entries(pipelines).map(([id, pipeline]) => ({ id, apply: nothing, draw: () => pipeline.render() }));
    const deltas: DeltaDefinition[] = Object.keys(pipelines)
        .filter((id) => id !== 'off')
        .map((id) => ({ name: `showing GI, ${id} - off`, minuend: id, subtrahend: 'off', part: 'total' as const }));
    return { variants, deltas };
}

/**
 * The 'upscaled' set (file comment): 2560 x 1440 output through @pmndrs/upscaler at its
 * Quality and Performance presets; the scene pass, the composite and GTAO at the
 * preset's render scale. "off" is the viewer's own path (`upscaleScene`). Each variant's
 * upscaler reports its own compute time (it isn't in three's passes).
 */
function upscaledSet(scene: CompositeScene, cameraFor: CameraFor, pipelineOf: PipelineOf): { variants: FrameVariant[]; deltas: readonly DeltaDefinition[] } {
    const variants: FrameVariant[] = [];
    const deltas: DeltaDefinition[] = [];
    for (const preset of ['quality', 'performance'] as const satisfies readonly UpscalePreset[]) {
        const offNode = upscaledScene(scene.scene, cameraFor(false), preset);
        const off = pipelineOf(offNode);
        variants.push({ id: `${preset} off`, apply: nothing, draw: () => off.render(), extraGpuMs: () => upscalerMs(offNode) });
        const options = {
            composite: false,
            'composite+gtao': { resolutionScale: 1 },
            'composite+gtao-half': { resolutionScale: 0.5 },
        } as const;
        for (const [name, occlusion] of Object.entries(options)) {
            const built = upscaledComposite(scene.scene, cameraFor(true), scene.threeGI, preset, { occlusion });
            const pipeline = pipelineOf(built.node);
            const id = `${preset} ${name}`;
            variants.push({ id, apply: nothing, draw: () => pipeline.render(), extraGpuMs: () => upscalerMs(built.node) });
            deltas.push({ name: `showing GI, ${id} - off`, minuend: id, subtrahend: `${preset} off`, part: 'total' });
        }
    }
    return { variants, deltas };
}

async function match(context: ThreePageContext, scene: CompositeScene, name: string): Promise<MatchStatistics> {
    const { renderer } = context;
    setViewSize(context, scene, SHOT_SIZE);
    const routes = new CompositeRoutes(scene, { byteTargets });
    const reader = new HdrReader(renderer, SHOT_SIZE.width, SHOT_SIZE.height);
    for (const scenePass of [routes.perFragmentPass, routes.opaquePass, routes.compositePass]) await scenePass.compileAsync(renderer);
    for (const node of [routes.perFragmentPass, routes.opaquePass, routes.composite] as unknown as Node[]) await reader.warm(node, 4);
    const perFragment = await reader.read(routes.perFragmentPass as unknown as Node);
    const opaqueOnly = await reader.read(routes.opaquePass as unknown as Node);
    const composite = await reader.read(routes.composite as unknown as Node);
    const coverage = await reader.read(routes.coverage, false);
    const light = scene.threeGI.light;
    light.intensity = 0;
    const giOff = await reader.read(routes.perFragmentPass as unknown as Node);
    light.intensity = 1;
    reader.dispose();
    const mask = opaqueMask(coverage, perFragment, opaqueOnly);
    const stats = matchStatistics(perFragment, composite, giOff, mask);
    context.log(formatMatch(name, stats));
    return stats;
}

await runThreePage(
    async (context) => {
        const { renderer, device, data, log } = context;
        renderer.shadowMap.enabled = true;
        const skip = typeof env.VITE_DDGI_SKIP === 'string' ? env.VITE_DDGI_SKIP : '';
        const giFrames = numberFromEnv(env.VITE_DDGI_GI_FRAMES, GI_FRAMES);
        const scene = await loadCompositeScene(device, 'sponza', { ...SHOT_SIZE, giFrames, probesPerFrame: PROBES_PER_FRAME, log });
        // The cost is judged by run.ts (it knows the GPU clock); the page only measures.
        if (skip !== 'timing') data.timing = await timing(context, scene);
        if (skip !== 'match') {
            const targets = byteTargets ? '8-bit targets' : 'half-float targets';
            const textured = await match(context, scene, `sponza match, its own materials, ${targets}`);
            const replaced = useGiMaterialTable(scene);
            log(`${replaced} materials replaced by the GI material table`);
            const untextured = await match(context, scene, `sponza match, GI material table, ${targets}`);
            data.match = { targets, textured, untextured };
            assert(untextured.meanRelative <= MEAN_BAR, `sponza (GI table): mean ${(untextured.meanRelative * 100).toFixed(3)}% > ${MEAN_BAR * 100}%`);
            assert(untextured.p95Relative <= P95_BAR, `sponza (GI table): p95 ${(untextured.p95Relative * 100).toFixed(3)}% > ${P95_BAR * 100}%`);
        }
        log('done');
    },
    SHOT_SIZE,
    { trackTimestamp: true },
);
