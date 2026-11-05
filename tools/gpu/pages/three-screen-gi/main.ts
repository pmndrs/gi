/**
 * What "GI once per pixel" would cost through three's forward path, on
 * Sponza with curtains (the frame breakdown's proposal in DECISIONS). A page
 * only: the adapter is unchanged; the screen-space pieces are in
 * `screen-gi.ts`.
 *
 * Variants (every one draws the viewer's 4x multisampled half-float scene
 * target from the scene's first bookmark; the DDGI field is frozen after the
 * warm-up, so every variant shades the same field):
 *  - `forward-off`: no GI light (the baseline every "GI adds" is against);
 *  - `forward-gi`: the adapter's light, GI evaluated per fragment (today);
 *  - `forward-gi-update`: the same plus DDGI's update at the viewer's budget,
 *    for the update's own cost in this run;
 *  - `mrt-off`: the main pass with `mrt({ output, normal })`, no GI: what
 *    writing depth and normals from the main pass adds;
 *  - `mrt-screen`: that main pass with materials reading a per-pixel GI
 *    texture by `screenUV` (`ScreenGILight`), then the screen pass that fills
 *    the texture from this pass's depth and normals. The materials read the
 *    texture the previous frame made: GI one frame late (or a deferred
 *    composite), see the report;
 *  - `prepass-screen`: a depth-and-normal prepass (1 sample, every opaque mesh
 *    re-rasterised with a normal-mapped stand-in), the screen pass, then the
 *    main pass reading the texture: GI from the same frame;
 *  - `prepass4-screen`: the same with a 4x multisampled prepass.
 * Pass labels: `main`, `prepass`, `screen`, `shadow` (the sun's shadow map).
 *
 * Then the pictures: the per-fragment frame against each screen route (mean
 * and share of pixels that differ), to show the screen path draws the same GI.
 *
 * Settings (vite exposes `VITE_*` variables to the page):
 *   VITE_DDGI_WIDTH / VITE_DDGI_HEIGHT   view size (default 2560 x 1440)
 *   VITE_DDGI_ROUNDS                     rounds over the variants (default 3)
 *   VITE_DDGI_FRAMES                     samples per variant per round (default 10)
 *   VITE_DDGI_BURST                      frames per sample, back to back (default 8)
 *   VITE_DDGI_VARIANTS                   only these variants
 *   VITE_DDGI_SCENES                     the scene (default sponza)
 *
 * Timing runs hold the bench lock:
 *   node tools/reference/bench-lock.ts run --what "vb and screen gi" -- pnpm test:gpu --filter three-screen-gi --json out.json
 */

import { HalfFloatType, NearestFilter, type Material, type Mesh } from 'three';
import { mrt, output } from 'three/tsl';
import { RenderTarget, type MeshBasicNodeMaterial } from 'three/webgpu';
import { reportGpuTest, summarizeAdapterInfo } from '../../client.ts';
import {
    FrameHarness,
    LAYER_PAGE_LIGHT,
    collectPageErrors,
    comparePictures,
    countFromEnv,
    createLog,
    listFromEnv,
    loadSceneWithDdgi,
    materialsOf,
    readTarget,
    setGiLayers,
    visibleMeshes,
    type PictureComparison,
    type SponzaSetup,
    type Variant,
    type VariantResult,
} from '../three-screen-gi-shared/sponza-bench.ts';
import { NORMAL_DEPTH_NODE, ScreenGILight, ScreenGIPass, normalDepthMaterial } from './screen-gi.ts';

/** The viewer's scene target is multisampled 4x (`ScenePassTechnique`). */
const VIEWER_MSAA_SAMPLES = 4;

const startTime = performance.now();
const { lines: logLines, log } = createLog(document.getElementById('log') as HTMLElement);
const pageErrors = collectPageErrors();
const env = import.meta.env as Record<string, unknown>;

function halfFloatTarget(width: number, height: number, samples: number, count = 1): RenderTarget {
    const target = new RenderTarget(width, height, { type: HalfFloatType, samples, count });
    // The screen pass and the materials read exactly one texel per pixel.
    for (const texture of target.textures) {
        texture.minFilter = NearestFilter;
        texture.magFilter = NearestFilter;
    }
    return target;
}

/**
 * Swaps every opaque mesh's materials for prepass stand-ins and hides what a prepass leaves
 * out (transparent meshes, and the sky dome, which refuses overrides). Returns the undo.
 */
function swapToPrepass(setup: SponzaSetup, cache: Map<Material, MeshBasicNodeMaterial>): () => void {
    const restores: Array<() => void> = [];
    const standIn = (source: Material): MeshBasicNodeMaterial => {
        let material = cache.get(source);
        if (!material) {
            material = normalDepthMaterial(source);
            cache.set(source, material);
        }
        return material;
    };
    for (const mesh of visibleMeshes(setup.scene)) {
        const sources = materialsOf(mesh);
        if (!sources.every((source) => !source.transparent && source.allowOverride !== false)) {
            mesh.visible = false;
            restores.push(() => (mesh.visible = true));
            continue;
        }
        const original = mesh.material;
        (mesh as Mesh).material = Array.isArray(original) ? original.map(standIn) : standIn(original);
        restores.push(() => ((mesh as Mesh).material = original));
    }
    return () => {
        for (const restore of restores) restore();
    };
}

interface ScreenGiResult {
    scene: string;
    width: number;
    height: number;
    triangles: number;
    startupMs: number;
    computeCoverage: boolean;
    variants: Record<string, VariantResult>;
    derived: Record<string, number>;
    pictures: Record<string, PictureComparison>;
    options: { rounds: number; samples: number; burst: number };
}

function derive(results: Record<string, VariantResult>): Record<string, number> {
    const pass = (id: string, label: string) => results[id]?.passMs[label] ?? NaN;
    const total = (id: string) => results[id]?.totalWithoutShadowMs ?? NaN;
    return {
        forwardOffMainMs: pass('forward-off', 'main'),
        forwardGiMainMs: pass('forward-gi', 'main'),
        forwardGiAddsMs: total('forward-gi') - total('forward-off'),
        updateMs: results['forward-gi-update']?.platformMs ?? NaN,
        mrtAddsMs: pass('mrt-off', 'main') - pass('forward-off', 'main'),
        mrtRouteMainMs: pass('mrt-screen', 'main'),
        mrtRouteScreenPassMs: pass('mrt-screen', 'screen'),
        mrtRouteReadAddsMs: pass('mrt-screen', 'main') - pass('mrt-off', 'main'),
        mrtRouteGiAddsMs: total('mrt-screen') - total('forward-off'),
        prepassMs: pass('prepass-screen', 'prepass'),
        prepassRouteScreenPassMs: pass('prepass-screen', 'screen'),
        screenReadAddsMs: pass('prepass-screen', 'main') - pass('forward-off', 'main'),
        prepassRouteGiAddsMs: total('prepass-screen') - total('forward-off'),
        prepass4Ms: pass('prepass4-screen', 'prepass'),
        prepass4RouteGiAddsMs: total('prepass4-screen') - total('forward-off'),
    };
}

/** The loaded scene, kept for the report and the clean-up (a holder, so TypeScript sees it change). */
const page: { setup: SponzaSetup | null } = { setup: null };

async function run(): Promise<ScreenGiResult> {
    const width = countFromEnv(env.VITE_DDGI_WIDTH, 2560);
    const height = countFromEnv(env.VITE_DDGI_HEIGHT, 1440);
    const sceneId = listFromEnv(env.VITE_DDGI_SCENES)?.[0] ?? 'sponza';
    const options = { rounds: countFromEnv(env.VITE_DDGI_ROUNDS, 3), samples: countFromEnv(env.VITE_DDGI_FRAMES, 10), burst: countFromEnv(env.VITE_DDGI_BURST, 8) };
    const setup = await loadSceneWithDdgi(document.getElementById('view') as HTMLCanvasElement, sceneId, width, height, log);
    page.setup = setup;
    const { renderer, scene, camera, threeGI } = setup;
    const harness = new FrameHarness(setup);
    const timer = harness.timer;

    const mainTarget = halfFloatTarget(width, height, VIEWER_MSAA_SAMPLES);
    const mrtTarget = halfFloatTarget(width, height, VIEWER_MSAA_SAMPLES, 2);
    mrtTarget.textures[0].name = 'output';
    mrtTarget.textures[1].name = 'normal';
    const prepassTarget = halfFloatTarget(width, height, 1);
    const prepass4Target = halfFloatTarget(width, height, VIEWER_MSAA_SAMPLES);
    const giTarget = new RenderTarget(width, height, { type: HalfFloatType, samples: 1, depthBuffer: false });
    giTarget.texture.minFilter = NearestFilter;
    giTarget.texture.magFilter = NearestFilter;

    const shading = setup.giLight.shading;
    const screenFromMrt = new ScreenGIPass(shading, mrtTarget.textures[1]);
    const screenFromPrepass = new ScreenGIPass(shading, prepassTarget.texture);
    const screenFromPrepass4 = new ScreenGIPass(shading, prepass4Target.texture);
    const screenLight = new ScreenGILight(giTarget.texture);
    screenLight.name = 'screen GI';
    screenLight.layers.set(LAYER_PAGE_LIGHT);
    scene.add(screenLight);
    const mainMrt = mrt({ output, normal: NORMAL_DEPTH_NODE });
    const prepassMaterials = new Map<Material, MeshBasicNodeMaterial>();
    const computeCoverage = () => threeGI.light.computeCoverage;

    const drawScene = (target: RenderTarget, label: string) => {
        timer.label = label;
        renderer.setRenderTarget(target);
        renderer.render(scene, camera);
    };
    const drawScreenPass = (pass: ScreenGIPass) => {
        timer.label = 'screen';
        renderer.setRenderTarget(giTarget);
        pass.render(renderer, camera, computeCoverage());
    };
    const drawPrepass = (target: RenderTarget) => {
        const putBack = swapToPrepass(setup, prepassMaterials);
        try {
            drawScene(target, 'prepass');
        } finally {
            putBack();
        }
    };
    const withMrt = (draw: () => void) => {
        renderer.setMRT(mainMrt);
        try {
            draw();
        } finally {
            renderer.setMRT(null);
        }
    };

    const all: Variant[] = [
        { id: 'forward-off', apply: () => setGiLayers(camera, 'off'), draw: () => drawScene(mainTarget, 'main') },
        { id: 'forward-gi', apply: () => setGiLayers(camera, 'fragment'), draw: () => drawScene(mainTarget, 'main') },
        {
            id: 'forward-gi-update',
            apply: () => {
                setGiLayers(camera, 'fragment');
                threeGI.budget = setup.normalBudget;
            },
            draw: () => {
                threeGI.update(camera);
                drawScene(mainTarget, 'main');
            },
            leave: () => (threeGI.budget = { paused: true }),
        },
        { id: 'mrt-off', apply: () => setGiLayers(camera, 'off'), draw: () => withMrt(() => drawScene(mrtTarget, 'main')) },
        {
            id: 'mrt-screen',
            apply: () => setGiLayers(camera, 'page'),
            draw: () => {
                withMrt(() => drawScene(mrtTarget, 'main'));
                drawScreenPass(screenFromMrt);
            },
        },
        {
            id: 'prepass-screen',
            apply: () => setGiLayers(camera, 'page'),
            draw: () => {
                drawPrepass(prepassTarget);
                drawScreenPass(screenFromPrepass);
                drawScene(mainTarget, 'main');
            },
        },
        {
            id: 'prepass4-screen',
            apply: () => setGiLayers(camera, 'page'),
            draw: () => {
                drawPrepass(prepass4Target);
                drawScreenPass(screenFromPrepass4);
                drawScene(mainTarget, 'main');
            },
        },
    ];
    const wanted = listFromEnv(env.VITE_DDGI_VARIANTS);
    const variants = wanted ? all.filter((variant) => wanted.includes(variant.id)) : all;
    log(`variants: ${variants.map((variant) => variant.id).join(', ')}; ${options.rounds} rounds x ${options.samples} samples x ${options.burst} frames`);
    const results = await harness.measure(variants, { ...options, log });
    const derived = derive(results);
    log(Object.entries(derived).map(([name, ms]) => `${name} ${ms.toFixed(2)}`).join('; '));

    // Pictures: a few frames of each, then the resolved colour.
    const picture = async (id: string, target: RenderTarget): Promise<Float32Array | null> => {
        const variant = all.find((candidate) => candidate.id === id)!;
        variant.apply();
        for (let frame = 0; frame < 6; frame++) await harness.sample(variant, 1);
        variant.leave?.();
        return readTarget(renderer, target, 0);
    };
    const pictures: Record<string, PictureComparison> = {};
    const off = await picture('forward-off', mainTarget);
    const perFragment = await picture('forward-gi', mainTarget);
    const viaMrt = await picture('mrt-screen', mrtTarget);
    const viaPrepass = await picture('prepass-screen', mainTarget);
    if (off && perFragment && viaMrt && viaPrepass) {
        pictures['GI on against off (per fragment)'] = comparePictures(off, perFragment);
        pictures['MRT route against per fragment'] = comparePictures(perFragment, viaMrt);
        pictures['prepass route against per fragment'] = comparePictures(perFragment, viaPrepass);
        for (const [name, comparison] of Object.entries(pictures)) {
            log(`${name}: mean luminance x${comparison.meanLuminanceRatio.toFixed(4)}; pixels differing >2% ${(comparison.differingOver2Percent * 100).toFixed(2)}%, >10% ${(comparison.differingOver10Percent * 100).toFixed(2)}%`);
        }
    }

    // Leave the per-pixel picture on the canvas for the screenshot.
    setGiLayers(camera, 'page');
    timer.label = 'screenshot';
    renderer.setRenderTarget(null);
    renderer.render(scene, camera);
    await setup.device.queue.onSubmittedWorkDone();
    await timer.read();

    const result: ScreenGiResult = {
        scene: sceneId,
        width,
        height,
        triangles: setup.triangles,
        startupMs: setup.startupMs,
        computeCoverage: threeGI.light.computeCoverage,
        variants: results,
        derived,
        pictures,
        options,
    };
    return result;
}

try {
    const result = await run();
    const setup = page.setup!;
    reportGpuTest({
        status: pageErrors.length === 0 ? 'pass' : 'fail',
        message: pageErrors.length === 0 ? undefined : pageErrors.join(' | '),
        data: result as unknown as Record<string, unknown>,
        deviceErrors: [...setup.core.gpuErrors],
        adapter: summarizeAdapterInfo(setup.core.adapterInfo, setup.device.features),
        startTime,
        logLines,
    });
} catch (error) {
    const message = error instanceof Error ? `${error.message}
${error.stack ?? ''}` : String(error);
    reportGpuTest({ status: 'fail', message: [message, ...pageErrors].join(' | '), deviceErrors: page.setup ? [...page.setup.core.gpuErrors] : [], startTime, logLines });
} finally {
    page.setup?.core.dispose();
}
