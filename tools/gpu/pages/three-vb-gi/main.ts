/**
 * GI on against off through three's GPU-driven drawing pipelines, on Sponza
 * with curtains, against the standard forward path in the same run.
 *
 * three r187's Visibility-Buffer pipeline rasterises small triangles into a
 * visibility buffer and shades each pixel once in a resolve pass, so our GI
 * light (which works in every pipeline, T0.8) would run once per pixel there
 * instead of once per fragment of every sub-pixel triangle. Both GPU-driven
 * pipelines also pick simplified levels of detail (1 px error by default).
 * Only FrontSide, opaque, channel-0 materials take the GPU-driven path;
 * everything else (DoubleSide above all) falls back to the standard path in
 * the same render (NOTES, three.js).
 *
 * Variants (the viewer's 4x multisampled half-float scene target, the scene's
 * first bookmark, the DDGI field frozen after the warm-up):
 *  - `standard-off`, `standard-gi`: the standard pipeline (the viewer today);
 *  - `vb-off`, `vb-gi`: the Visibility-Buffer pipeline;
 *  - `meshlet-off`, `meshlet-gi`: the Meshlet pipeline (VITE_THREE_MESHLET=0 skips it);
 *  - `fallback-off`, `fallback-gi`: the standard pipeline drawing only the
 *    meshes the Visibility-Buffer pipeline leaves on the standard path: what
 *    the fallback costs inside a VB frame (an upper bound: with the VB meshes
 *    hidden, nothing in front of the fallback meshes hides any of them);
 *  - `vb-frontside-off`, `vb-frontside-gi`: the VB pipeline with the
 *    fallback materials that are only double-sided or vertex-coloured forced
 *    FrontSide without vertex colours, so it takes them too (a cost bound for
 *    VB-friendly content; the picture changes).
 * Pass labels: `main` (render passes), `main compute` (the GPU-driven
 * pipelines' culling and software rasteriser, including their shadow-map
 * culling), `shadow` (shadow-map render passes).
 *
 * Also reported: which meshes and materials fall back and why, the
 * pipelines' preparation time (meshlets and levels of detail, on the CPU),
 * and the pictures (VB against standard, GI on).
 *
 * Settings: VITE_DDGI_WIDTH / VITE_DDGI_HEIGHT (default 2560 x 1440),
 * VITE_DDGI_ROUNDS (3), VITE_DDGI_FRAMES (10), VITE_DDGI_BURST (8),
 * VITE_DDGI_VARIANTS, VITE_DDGI_SCENES (sponza), VITE_THREE_VB_SAMPLES (4).
 *
 * Timing runs hold the bench lock:
 *   node tools/reference/bench-lock.ts run --what "vb and screen gi" -- pnpm test:gpu --filter three-vb-gi --json out.json
 */

import { DoubleSide, FrontSide, HalfFloatType, type Material, type Mesh, type Object3D } from 'three';
import { RenderTarget } from 'three/webgpu';
import { MeshletDrawingPipeline } from 'three/addons/pipelines/MeshletDrawingPipeline.js';
import { VisibilityBufferDrawingPipeline } from 'three/addons/pipelines/VisibilityBufferDrawingPipeline.js';
import { reportGpuTest, summarizeAdapterInfo } from '../../client.ts';
import {
    FrameHarness,
    collectPageErrors,
    comparePictures,
    countFromEnv,
    createLog,
    listFromEnv,
    loadSceneWithDdgi,
    readTarget,
    setGiLayers,
    triangleCount,
    visibleMeshes,
    type PictureComparison,
    type SponzaSetup,
    type Variant,
    type VariantResult,
} from '../three-screen-gi-shared/sponza-bench.ts';

const startTime = performance.now();
const { lines: logLines, log } = createLog(document.getElementById('log') as HTMLElement);
const pageErrors = collectPageErrors();
const env = import.meta.env as Record<string, unknown>;

/** What the drawing pipelines look like from here (three r187; `drawingPipeline` is experimental and untyped). */
interface DrawingPipelineShape {
    _shadowPass: boolean;
    _isMeshlet(object: Object3D, geometry: unknown, material: Material, group: { start: number; count: number } | null): boolean;
    prepareAsync?(scene: Object3D, camera: unknown): Promise<void> | undefined;
}

interface RendererWithPipeline {
    drawingPipeline: DrawingPipelineShape;
}

/** One decision of the VB pipeline about one mesh (or one material group of it). */
interface PathDecision {
    mesh: Mesh;
    material: Material;
    taken: boolean;
    triangles: number;
}

/** Why a material can't take the GPU-driven path, by the checks in `MeshletDrawingPipeline.isSupportedMaterial` (r187). */
function fallbackReason(material: Material): string {
    const extended = material as Material & { transmission?: number; alphaHash?: boolean; vertexColors?: boolean; flatShading?: boolean; wireframe?: boolean; displacementMap?: unknown; maskNode?: unknown; mrtNode?: unknown };
    if (material.transparent) return 'transparent';
    if ((extended.transmission ?? 0) > 0) return 'transmission';
    if (material.side !== FrontSide) return material.side === DoubleSide ? 'DoubleSide' : 'BackSide';
    if (extended.vertexColors) return 'vertex colours';
    if (extended.flatShading) return 'flat shading';
    if (extended.displacementMap) return 'displacement map';
    if (material.depthWrite === false || material.depthTest === false || material.colorWrite === false) return 'depth or colour write off';
    if (extended.maskNode || extended.mrtNode) return 'mask or MRT node';
    for (const value of Object.values(material)) {
        if (value && typeof value === 'object' && (value as { isTexture?: boolean }).isTexture && (value as { channel: number }).channel !== 0) return 'texture on uv channel > 0';
    }
    return 'other (object, geometry or node checks)';
}

/** Records the VB pipeline's per-object decisions while the main pass is drawn. */
function watchDecisions(pipeline: DrawingPipelineShape, isMainPass: () => boolean): Map<string, PathDecision> {
    const decisions = new Map<string, PathDecision>();
    const original = pipeline._isMeshlet.bind(pipeline);
    pipeline._isMeshlet = (object, geometry, material, group) => {
        const taken = original(object, geometry, material, group);
        if (isMainPass() && !pipeline._shadowPass && (object as Mesh).isMesh) {
            const mesh = object as Mesh;
            const triangles = group ? Math.floor(group.count / 3) : triangleCount(mesh);
            decisions.set(`${mesh.uuid}:${group ? group.start : '-'}`, { mesh, material, taken, triangles: Number.isFinite(triangles) ? triangles : triangleCount(mesh) });
        }
        return taken;
    };
    return decisions;
}

interface FallbackSummary {
    meshesTaken: number;
    meshesFallback: number;
    trianglesTaken: number;
    trianglesFallback: number;
    materialsTaken: number;
    materialsFallback: number;
    /** Taken by the GPU-driven path but alpha-tested: VB draws these with the Meshlet pipeline's hardware rasteriser. */
    materialsAlphaTestedTaken: number;
    fallbackByReason: Record<string, { materials: number; meshes: number; triangles: number }>;
    fallbackMaterials: Array<{ name: string; reason: string; meshes: number; triangles: number }>;
}

function summarizeDecisions(decisions: Map<string, PathDecision>): FallbackSummary {
    const taken = [...decisions.values()].filter((decision) => decision.taken);
    const fallback = [...decisions.values()].filter((decision) => !decision.taken);
    const byMaterial = new Map<Material, { meshes: number; triangles: number }>();
    for (const decision of fallback) {
        const entry = byMaterial.get(decision.material) ?? { meshes: 0, triangles: 0 };
        entry.meshes++;
        entry.triangles += decision.triangles;
        byMaterial.set(decision.material, entry);
    }
    const fallbackByReason: FallbackSummary['fallbackByReason'] = {};
    const fallbackMaterials: FallbackSummary['fallbackMaterials'] = [];
    for (const [material, entry] of byMaterial) {
        const reason = fallbackReason(material);
        const row = (fallbackByReason[reason] ??= { materials: 0, meshes: 0, triangles: 0 });
        row.materials++;
        row.meshes += entry.meshes;
        row.triangles += entry.triangles;
        fallbackMaterials.push({ name: material.name, reason, ...entry });
    }
    fallbackMaterials.sort((a, b) => b.triangles - a.triangles);
    const takenMaterials = new Set(taken.map((decision) => decision.material));
    return {
        meshesTaken: new Set(taken.map((decision) => decision.mesh)).size,
        meshesFallback: new Set(fallback.map((decision) => decision.mesh)).size,
        trianglesTaken: taken.reduce((sum, decision) => sum + decision.triangles, 0),
        trianglesFallback: fallback.reduce((sum, decision) => sum + decision.triangles, 0),
        materialsTaken: takenMaterials.size,
        materialsFallback: byMaterial.size,
        materialsAlphaTestedTaken: [...takenMaterials].filter((material) => material.alphaTest > 0).length,
        fallbackByReason,
        fallbackMaterials,
    };
}

interface VbGiResult {
    scene: string;
    width: number;
    height: number;
    samples: number;
    triangles: number;
    startupMs: number;
    computeCoverage: boolean;
    prepareMs: Record<string, number>;
    fallback: FallbackSummary | null;
    variants: Record<string, VariantResult>;
    derived: Record<string, number>;
    pictures: Record<string, PictureComparison>;
    options: { rounds: number; samples: number; burst: number };
}

function derive(results: Record<string, VariantResult>): Record<string, number> {
    const total = (id: string) => results[id]?.totalWithoutShadowMs ?? NaN;
    return {
        standardOffMs: total('standard-off'),
        standardGiMs: total('standard-gi'),
        standardGiAddsMs: total('standard-gi') - total('standard-off'),
        vbOffMs: total('vb-off'),
        vbGiMs: total('vb-gi'),
        vbGiAddsMs: total('vb-gi') - total('vb-off'),
        meshletOffMs: total('meshlet-off'),
        meshletGiMs: total('meshlet-gi'),
        meshletGiAddsMs: total('meshlet-gi') - total('meshlet-off'),
        fallbackOffMs: total('fallback-off'),
        fallbackGiMs: total('fallback-gi'),
        fallbackGiAddsMs: total('fallback-gi') - total('fallback-off'),
        vbFrontsideOffMs: total('vb-frontside-off'),
        vbFrontsideGiMs: total('vb-frontside-gi'),
        vbFrontsideGiAddsMs: total('vb-frontside-gi') - total('vb-frontside-off'),
        fallbackShareOfVbGiFrame: total('fallback-gi') / total('vb-gi'),
        fallbackShareOfVbGiAdds: (total('fallback-gi') - total('fallback-off')) / (total('vb-gi') - total('vb-off')),
    };
}

const page: { setup: SponzaSetup | null } = { setup: null };

async function run(): Promise<VbGiResult> {
    const width = countFromEnv(env.VITE_DDGI_WIDTH, 2560);
    const height = countFromEnv(env.VITE_DDGI_HEIGHT, 1440);
    const samples = countFromEnv(env.VITE_THREE_VB_SAMPLES, 4);
    const sceneId = listFromEnv(env.VITE_DDGI_SCENES)?.[0] ?? 'sponza';
    const withMeshlet = env.VITE_THREE_MESHLET !== '0';
    const options = { rounds: countFromEnv(env.VITE_DDGI_ROUNDS, 3), samples: countFromEnv(env.VITE_DDGI_FRAMES, 10), burst: countFromEnv(env.VITE_DDGI_BURST, 8) };
    const setup = await loadSceneWithDdgi(document.getElementById('view') as HTMLCanvasElement, sceneId, width, height, log);
    page.setup = setup;
    const { renderer, scene, camera, threeGI } = setup;
    const harness = new FrameHarness(setup);
    const timer = harness.timer;
    const withPipeline = renderer as unknown as RendererWithPipeline;
    const target = new RenderTarget(width, height, { type: HalfFloatType, samples });

    const standard = withPipeline.drawingPipeline;
    const visibility = new VisibilityBufferDrawingPipeline() as unknown as DrawingPipelineShape;
    const meshlet = withMeshlet ? (new MeshletDrawingPipeline() as unknown as DrawingPipelineShape) : null;
    const decisions = watchDecisions(visibility, () => timer.label === 'main');

    // Meshlets and levels of detail are built on the CPU before the first frame (meshoptimizer).
    const prepareMs: Record<string, number> = {};
    setGiLayers(camera, 'fragment');
    for (const [name, pipeline] of [['vb', visibility], ['meshlet', meshlet]] as const) {
        if (!pipeline) continue;
        withPipeline.drawingPipeline = pipeline;
        const started = performance.now();
        await renderer.compileAsync(scene, camera);
        prepareMs[name] = performance.now() - started;
        log(`${name}: meshlets, levels of detail and pipelines prepared in ${(prepareMs[name] / 1000).toFixed(1)} s`);
    }
    withPipeline.drawingPipeline = standard;

    const draw = () => {
        timer.label = 'main';
        renderer.setRenderTarget(target);
        renderer.render(scene, camera);
    };

    // Which meshes the VB pipeline leaves on the standard path: a few VB frames record its decisions.
    withPipeline.drawingPipeline = visibility;
    for (let frame = 0; frame < 10; frame++) await harness.sample({ id: 'decisions', apply: () => {}, draw }, 1);
    withPipeline.drawingPipeline = standard;
    const fallback = summarizeDecisions(decisions);
    log(
        `VB path: ${fallback.meshesTaken} meshes / ${fallback.trianglesTaken} triangles / ${fallback.materialsTaken} materials (${fallback.materialsAlphaTestedTaken} alpha-tested); ` +
            `standard fallback: ${fallback.meshesFallback} meshes / ${fallback.trianglesFallback} triangles / ${fallback.materialsFallback} materials`,
    );
    for (const [reason, row] of Object.entries(fallback.fallbackByReason)) log(`  fallback (${reason}): ${row.materials} materials, ${row.meshes} meshes, ${row.triangles} triangles`);
    const takenMeshes = new Set<Mesh>();
    const fallbackMeshes = new Set<Mesh>();
    for (const decision of decisions.values()) (decision.taken ? takenMeshes : fallbackMeshes).add(decision.mesh);

    /** Hides every mesh the VB pipeline draws itself, so only its standard-path fallback is left. */
    let restoreHidden: Array<() => void> = [];
    const showFallbackOnly = () => {
        for (const mesh of visibleMeshes(scene)) {
            if (!takenMeshes.has(mesh) || fallbackMeshes.has(mesh)) continue;
            mesh.visible = false;
            restoreHidden.push(() => (mesh.visible = true));
        }
    };
    const showAll = () => {
        for (const restore of restoreHidden) restore();
        restoreHidden = [];
    };

    /**
     * "What if the content suited the GPU-driven path": the fallback materials that are only
     * double-sided or vertex-coloured, drawn FrontSide without vertex colours, so the VB
     * pipeline takes them too. The picture changes (curtains seen from behind vanish; vertex
     * colours are dropped); it is a cost bound, not a fix.
     */
    const adjustable = new Set<Material>();
    for (const decision of decisions.values()) {
        const reason = fallbackReason(decision.material);
        if (!decision.taken && (reason === 'DoubleSide' || reason === 'vertex colours')) adjustable.add(decision.material);
    }
    let restoreMaterials: Array<() => void> = [];
    const forceGpuDriven = () => {
        for (const material of adjustable) {
            const extended = material as Material & { vertexColors: boolean };
            const side = material.side;
            const vertexColors = extended.vertexColors;
            material.side = FrontSide;
            extended.vertexColors = false;
            material.needsUpdate = true;
            restoreMaterials.push(() => {
                material.side = side;
                extended.vertexColors = vertexColors;
                material.needsUpdate = true;
            });
        }
    };
    const restoreForced = () => {
        for (const restore of restoreMaterials) restore();
        restoreMaterials = [];
    };

    const variant = (id: string, pipeline: DrawingPipelineShape, gi: 'off' | 'fragment', change: 'none' | 'fallback only' | 'forced' = 'none'): Variant => ({
        id,
        apply: () => {
            withPipeline.drawingPipeline = pipeline;
            setGiLayers(camera, gi);
            if (change === 'fallback only') showFallbackOnly();
            if (change === 'forced') forceGpuDriven();
        },
        draw,
        leave: () => {
            if (change === 'fallback only') showAll();
            if (change === 'forced') restoreForced();
            withPipeline.drawingPipeline = standard;
        },
    });
    const all: Variant[] = [
        variant('standard-off', standard, 'off'),
        variant('standard-gi', standard, 'fragment'),
        variant('vb-off', visibility, 'off'),
        variant('vb-gi', visibility, 'fragment'),
        ...(meshlet ? [variant('meshlet-off', meshlet, 'off'), variant('meshlet-gi', meshlet, 'fragment')] : []),
        variant('fallback-off', standard, 'off', 'fallback only'),
        variant('fallback-gi', standard, 'fragment', 'fallback only'),
        variant('vb-frontside-off', visibility, 'off', 'forced'),
        variant('vb-frontside-gi', visibility, 'fragment', 'forced'),
    ];
    const wanted = listFromEnv(env.VITE_DDGI_VARIANTS);
    const variants = wanted ? all.filter((candidate) => wanted.includes(candidate.id)) : all;
    log(`variants: ${variants.map((candidate) => candidate.id).join(', ')}; ${options.rounds} rounds x ${options.samples} samples x ${options.burst} frames; ${samples} samples per pixel`);
    const results = await harness.measure(variants, { ...options, log });
    const derived = derive(results);
    log(Object.entries(derived).map(([name, value]) => `${name} ${value.toFixed(3)}`).join('; '));

    const picture = async (id: string): Promise<Float32Array> => {
        const chosen = all.find((candidate) => candidate.id === id)!;
        chosen.apply();
        for (let frame = 0; frame < 6; frame++) await harness.sample(chosen, 1);
        const pixels = await readTarget(renderer, target, 0);
        chosen.leave?.();
        return pixels;
    };
    const pictures: Record<string, PictureComparison> = {};
    const standardGi = await picture('standard-gi');
    pictures['VB against standard, GI on'] = comparePictures(standardGi, await picture('vb-gi'));
    pictures['VB, GI on against off'] = comparePictures(await picture('vb-off'), await picture('vb-gi'));
    pictures['VB with fallbacks forced FrontSide against standard, GI on'] = comparePictures(standardGi, await picture('vb-frontside-gi'));
    if (meshlet) pictures['Meshlet against standard, GI on'] = comparePictures(standardGi, await picture('meshlet-gi'));
    for (const [name, comparison] of Object.entries(pictures)) {
        log(`${name}: mean luminance x${comparison.meanLuminanceRatio.toFixed(4)}; pixels differing >2% ${(comparison.differingOver2Percent * 100).toFixed(2)}%, >10% ${(comparison.differingOver10Percent * 100).toFixed(2)}%`);
    }

    // The VB picture on the canvas for the screenshot.
    withPipeline.drawingPipeline = visibility;
    setGiLayers(camera, 'fragment');
    timer.label = 'screenshot';
    renderer.setRenderTarget(null);
    renderer.render(scene, camera);
    await setup.device.queue.onSubmittedWorkDone();
    await timer.read();
    withPipeline.drawingPipeline = standard;

    return {
        scene: sceneId,
        width,
        height,
        samples,
        triangles: setup.triangles,
        startupMs: setup.startupMs,
        computeCoverage: threeGI.light.computeCoverage,
        prepareMs,
        fallback,
        variants: results,
        derived,
        pictures,
        options,
    };
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
    const message = error instanceof Error ? `${error.message}\n${error.stack ?? ''}` : String(error);
    reportGpuTest({ status: 'fail', message: [message, ...pageErrors].join(' | '), deviceErrors: page.setup ? [...page.setup.core.gpuErrors] : [], startTime, logLines });
} finally {
    page.setup?.core.dispose();
}
