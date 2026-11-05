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
 *    - `composite+gtao`: the composite with its default occlusion (three's
 *      GTAO on the GI term, 1 m radius, full resolution): the default path;
 *    - `composite+gtao-half`: the same at half resolution;
 *    - `forward`: the plain pass, GI per fragment (today's path).
 *    Each difference is the median of per-cycle differences between
 *    neighbouring frames, with its spread. The bar: the cost of showing GI
 *    on the default path (`composite+gtao` − `off`) ≤ 1.0 ms at 2560 × 1440
 *    on the RTX 3060. A shorter section at 1920 × 1080 follows. Run it
 *    through `run.ts`, which adds each section's GPU clock and the
 *    clock-normalised numbers.
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
 *   VITE_DDGI_TARGETS     'half' (default) or 'byte': the format of giAlbedo and the normal.
 *                         One format per run: in r187 two MRT passes over one scene whose targets
 *                         differ only in format reuse each other's pipelines (NOTES).
 *
 * Manual (it needs the downloaded Sponza). Timing goes through run.ts under the bench lock:
 *   node tools/reference/bench-lock.ts run --what "three composite sponza" -- node tools/gpu/pages/three-composite-sponza/run.ts --gpu nvidia
 */

import { RenderPipeline, UnsignedByteType, type Node } from 'three/webgpu';
import { mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import { assert } from '../../client.ts';
import { markCompositePass, type MarkablePass } from '../../../../packages/ddgi-three/src/composite-mark.ts';
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
    { name: 'showing GI, forward per fragment - off', minuend: 'forward', subtrahend: 'off', part: 'total' },
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

    const plain = new RenderPipeline(renderer, pass(scene.scene, scene.camera) as unknown as Node);

    // The MRT pass alone: the composite's outputs and mark, its colour shown as is.
    const mrtPass = pass(scene.scene, scene.camera);
    mrtPass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView) }));
    markCompositePass(mrtPass as unknown as MarkablePass);
    // An MRT output is only drawn once its texture exists (MRTNode.setup).
    for (const name of [GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT]) {
        const target = mrtPass.getTexture(name);
        if (byteTargets) target.type = UnsignedByteType;
    }
    const mrtOnly = new RenderPipeline(renderer, mrtPass.getTextureNode('output') as unknown as Node);

    const pipelineFor = (occlusion: ConstructorParameters<typeof CompositeRoutes>[1]) => {
        const routes = new CompositeRoutes(scene, { byteTargets, ...occlusion });
        return new RenderPipeline(renderer, routes.composite as unknown as Node);
    };
    const composite = pipelineFor({ occlusion: false });
    // The composite's default occlusion: its own GTAO at 1 m.
    const compositeGtao = pipelineFor({ occlusion: {} });
    const compositeGtaoHalf = pipelineFor({ occlusion: { resolutionScale: 0.5 } });

    const showGI = (visible: boolean) => () => {
        light.visible = visible;
    };
    const variants: FrameVariant[] = [
        { id: 'off', apply: showGI(false), draw: () => plain.render() },
        { id: 'mrt', apply: showGI(true), draw: () => mrtOnly.render() },
        { id: 'composite', apply: showGI(true), draw: () => composite.render() },
        { id: 'composite+gtao', apply: showGI(true), draw: () => compositeGtao.render() },
        { id: 'composite+gtao-half', apply: showGI(true), draw: () => compositeGtaoHalf.render() },
        { id: 'forward', apply: showGI(true), draw: () => plain.render() },
    ];

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

    const clock = new PassClock(renderer, shadowCameras(scene));
    const rampSeconds = numberFromEnv(env.VITE_DDGI_RAMP, 2);
    const sections: InterleavedTiming[] = [];
    sections.push(await timeInterleaved(renderer, device, clock, variants, DELTAS, { seconds: numberFromEnv(env.VITE_DDGI_SECONDS, 10), cyclesPerBatch: CYCLES_PER_BATCH, rampSeconds, log }));
    const seconds1080 = numberFromEnv(env.VITE_DDGI_SECONDS_1080, 4);
    if (seconds1080 > 0) {
        setViewSize(context, scene, SHOT_SIZE);
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
        sections.push(await timeInterleaved(renderer, device, clock, variants, DELTAS, { seconds: seconds1080, cyclesPerBatch: CYCLES_PER_BATCH, rampSeconds: 0.5, log }));
    }
    light.visible = true;
    for (const pipeline of [plain, mrtOnly, composite, compositeGtao, compositeGtaoHalf]) pipeline.dispose();
    return { sections, targets: byteTargets ? 'byte' : 'half' };
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
    const stats = matchStatistics(perFragment, composite, giOff, opaqueMask(coverage, perFragment, opaqueOnly));
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
