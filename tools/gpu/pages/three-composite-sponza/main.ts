/**
 * T1.14, part 1, on Sponza with its curtains (`sponza`): the GI composite
 * against GI per fragment, and what showing GI costs each way
 * (`docs/design/three-composite.md`, DECISIONS 2026-11-05, T1.14).
 *
 * 1. **Cost** at 2560 × 1440, GPU-bound (frames back to back, no wait; the
 *    field frozen, so this is shading only: DECISIONS choice (b) judges the
 *    update separately). Short (about 10–15 s) and first, so it runs on a
 *    cool GPU: the laptop RTX throttles to 450–600 MHz after about 20 s of
 *    GPU-bound load (7a8128c). Start the page on a GPU below about 70 °C,
 *    and time with a short warm-up (`VITE_DDGI_GI_FRAMES`); the shading cost
 *    doesn't depend on how converged the field is. Every variant is three's own `RenderPipeline` with
 *    the same output (tone mapping to the canvas); the sun's shadow map is in
 *    every frame. Variants:
 *    - `off`: `pass(scene, camera)`, the GI light hidden;
 *    - `forward`: the same pass, GI per fragment (today's path);
 *    - `mrt`: the pass with `mrt({ output, giAlbedo, normal })` and the
 *      composite's mark (no per-fragment GI), showing its colour: what the
 *      extra outputs cost;
 *    - `composite`: that pass and `ddgiComposite`.
 *    With `VITE_DDGI_TARGETS=byte`, `giAlbedo` and the normal are stored as
 *    8-bit unorm instead of half float (three's SSGI example does this for
 *    bandwidth), for the timing and the match alike. One page run uses one
 *    format: in r187, two MRT passes over the same scene whose targets differ
 *    only in format reuse each other's pipelines, which WebGPU rejects
 *    ("Attachment state ... is not compatible").
 *    Cost of showing GI = composite − off (bar ≤ 1.0 ms on the RTX 3060);
 *    forward − off for comparison.
 * 2. **Match** at the stored camera (bookmark 1, 1920 × 1080), as the
 *    three-composite page does for Cornell: first with Sponza's own
 *    materials, then with the untextured GI material table (each material
 *    the constant the GI platform received, as the Sponza accuracy page and
 *    the untextured Cycles reference use). Bar: mean ≤ 0.5%, p95 ≤ 2% on
 *    the GI-table materials; the textured result is reported as well.
 *
 * Settings (vite exposes `VITE_*` variables to the page):
 *   VITE_DDGI_ROUNDS      timing rounds (default 6; variants alternate order, so clock drift hits all alike)
 *   VITE_DDGI_FRAMES      samples per variant per round (default 3)
 *   VITE_DDGI_RAMP        seconds of GPU-bound frames before timing, for the clock to ramp up (default 3)
 *   VITE_DDGI_COOL        seconds of idle GPU before the ramp, after loading heated it (default 25)
 *   VITE_DDGI_BURST       frames per sample, back to back (default 8)
 *   VITE_DDGI_SETTLE      samples before a variant's first timed ones (default 6)
 *   VITE_DDGI_GI_FRAMES   GI frames of 4096 probes before the field freezes (default 120)
 *   VITE_DDGI_SKIP        'timing' or 'match' to skip a part
 *   VITE_DDGI_TARGETS     'half' (default) or 'byte': the format of giAlbedo and the normal
 *
 * Manual (it needs the downloaded Sponza), and the timing holds the bench lock:
 *   node tools/reference/bench-lock.ts run --what "three composite sponza" -- pnpm test:gpu --filter three-composite-sponza --json out.json
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
    formatMatch,
    loadCompositeScene,
    matchStatistics,
    opaqueMask,
    shadowCameras,
    timeVariants,
    useGiMaterialTable,
    type CompositeScene,
    type FrameVariant,
    type MatchStatistics,
    type VariantTiming,
} from '../three-composite-shared/composite-check.ts';

const TIMING_SIZE = { width: 2560, height: 1440 } as const;
const GI_FRAMES = 120;
const PROBES_PER_FRAME = 4096;
const MEAN_BAR = 0.005;
const P95_BAR = 0.02;
/** The cost bar (DECISIONS 2026-11-05, T1.14): showing GI ≤ 1.0 ms on Sponza at 1440p, on the RTX 3060. */
const COST_BAR_MS = 1.0;

const env = import.meta.env as Record<string, unknown>;
const byteTargets = env.VITE_DDGI_TARGETS === 'byte';

function countFromEnv(value: unknown, fallback: number): number {
    const parsed = typeof value === 'string' ? Number(value) : NaN;
    return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

async function timing(context: ThreePageContext, scene: CompositeScene): Promise<{ variants: Record<string, VariantTiming>; derived: Record<string, number> }> {
    const { renderer, device, log } = context;
    renderer.setSize(TIMING_SIZE.width, TIMING_SIZE.height, false);
    scene.camera.aspect = TIMING_SIZE.width / TIMING_SIZE.height;
    scene.camera.updateProjectionMatrix();
    const light = scene.threeGI.light;

    const plainPass = pass(scene.scene, scene.camera);
    const plain = new RenderPipeline(renderer, plainPass as unknown as Node);

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

    const routes = new CompositeRoutes(scene, { byteTargets });
    const composite = new RenderPipeline(renderer, routes.composite as unknown as Node);

    const showGI = (visible: boolean) => () => {
        light.visible = visible;
    };
    const variants: FrameVariant[] = [
        { id: 'off', apply: showGI(false), draw: () => plain.render() },
        { id: 'forward', apply: showGI(true), draw: () => plain.render() },
        { id: 'mrt', apply: showGI(true), draw: () => mrtOnly.render() },
        { id: 'composite', apply: showGI(true), draw: () => composite.render() },
    ];
    const clock = new PassClock(renderer, shadowCameras(scene));
    const options = {
        rounds: countFromEnv(env.VITE_DDGI_ROUNDS, 5),
        samples: countFromEnv(env.VITE_DDGI_FRAMES, 3),
        rampSeconds: countFromEnv(env.VITE_DDGI_RAMP, 3),
        coolSeconds: countFromEnv(env.VITE_DDGI_COOL, 25),
        burst: countFromEnv(env.VITE_DDGI_BURST, 8),
        settleSamples: countFromEnv(env.VITE_DDGI_SETTLE, 6),
        log,
    };
    log(`timing at ${TIMING_SIZE.width} x ${TIMING_SIZE.height}: ${variants.map((variant) => variant.id).join(', ')}; ${options.rounds} rounds x ${options.samples} samples x ${options.burst} frames`);
    const results = await timeVariants(renderer, device, clock, variants, options);
    light.visible = true;
    const total = (id: string) => results[id].totalMs;
    const quad = (id: string) => results[id].byLabel.quad ?? 0;
    const derived = {
        showingGiCompositeMs: total('composite') - total('off'),
        showingGiForwardMs: total('forward') - total('off'),
        mrtExtraMs: total('mrt') - total('off'),
        compositePassMs: quad('composite') - quad('off'),
    };
    log(Object.entries(derived).map(([name, ms]) => `${name} ${ms.toFixed(3)}`).join('; '));
    plain.dispose();
    mrtOnly.dispose();
    composite.dispose();
    return { variants: results, derived };
}

async function match(context: ThreePageContext, scene: CompositeScene, name: string): Promise<MatchStatistics> {
    const { renderer } = context;
    renderer.setSize(SHOT_SIZE.width, SHOT_SIZE.height, false);
    scene.camera.aspect = SHOT_SIZE.width / SHOT_SIZE.height;
    scene.camera.updateProjectionMatrix();
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
        const giFrames = countFromEnv(env.VITE_DDGI_GI_FRAMES, GI_FRAMES);
        const scene = await loadCompositeScene(device, 'sponza', { ...SHOT_SIZE, giFrames, probesPerFrame: PROBES_PER_FRAME, log });
        if (skip !== 'timing') data.timing = await timing(context, scene);
        if (skip !== 'match') {
            const targets = byteTargets ? '8-bit targets' : 'half-float targets';
            const textured = await match(context, scene, `sponza match, its own materials, ${targets}`);
            const replaced = useGiMaterialTable(scene);
            log(`${replaced} materials replaced by the GI material table`);
            const untextured = await match(context, scene, `sponza match, GI material table, ${targets}`);
            data.match = { targets, textured, untextured };
        }
        log('done');

        const timingResult = data.timing as { derived: Record<string, number> } | undefined;
        if (timingResult) {
            const cost = timingResult.derived.showingGiCompositeMs;
            const vendor = context.adapterSummary.vendor;
            if (vendor === 'nvidia') assert(cost <= COST_BAR_MS, `showing GI with the composite costs ${cost.toFixed(3)} ms > ${COST_BAR_MS} ms`);
        }
        const matchResult = data.match as { untextured: MatchStatistics } | undefined;
        if (matchResult) {
            const stats = matchResult.untextured;
            assert(stats.meanRelative <= MEAN_BAR, `sponza (GI table): mean ${(stats.meanRelative * 100).toFixed(3)}% > ${MEAN_BAR * 100}%`);
            assert(stats.p95Relative <= P95_BAR, `sponza (GI table): p95 ${(stats.p95Relative * 100).toFixed(3)}% > ${P95_BAR * 100}%`);
        }
    },
    SHOT_SIZE,
    { trackTimestamp: true },
);
