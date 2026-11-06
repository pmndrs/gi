/**
 * T1.14, part 1: the GI composite on three's pipeline against GI per
 * fragment, on the Cornell scenes (`docs/design/three-composite.md`).
 *
 * 1. **Match.** On `cornell-box-point` and the emissive `cornell-box`, at the
 *    stored camera (bookmark 1, 1920 × 1080; composite-check.ts), the
 *    composite's HDR image before tone mapping against the per-fragment
 *    path's, over pixels with a surface and no transparent object in front:
 *    per-pixel relative luminance difference mean ≤ 0.5%, p95 ≤ 2%
 *    (DECISIONS 2026-11-05, T1.14). The field is frozen first, so both
 *    images shade the same field.
 * 2. **Transparent objects keep GI.** Two spheres in the point-light Cornell,
 *    both `transparent: true`: one at opacity 1, which covers what's behind
 *    it, so its pixels must match the per-fragment image as above and get
 *    GI (the image with GI against the image with the GI light at intensity
 *    0); one at opacity 0.5, whose pixels are reported (GI on the surface
 *    behind it is evaluated at the sphere, composite.ts says why).
 * 3. **The render-time update** (design, "No per-frame call"): can the GI
 *    frame run inside three's render, from a light node's update, and still
 *    land before the render that samples it? A test light whose node
 *    (`updateType = RENDER`, as `GILightNode`) calls `ThreeGI.update`. From
 *    an empty field, K frames each way on fresh platforms:
 *    - explicit: `threeGI.update()`, then the render;
 *    - render-time: the render alone, the update inside it.
 *    The render-time image after K frames must match the explicit one
 *    after K (not after K − 1), and the GI frame's queue submit must come
 *    after the scene pass's `beginRender` and before that pass's own
 *    submit. A second explicit run gives the run-to-run noise.
 * 4. **More cases, the same match bar** (the Opus review of T1.14), each on
 *    `cornell-box-point`:
 *    - `grid`: a baked `LightProbeGrid` at priority 2 (above GI's 1) over the
 *      left half of the box. Per fragment the grid wins the probe slot there;
 *      the composite must add no GI there, or GI is counted twice;
 *    - `ao`: AO in `giAlbedo`: three's `builtinAOContext` (a constant 0.6) on
 *      every pass, and a sphere with an `aoMap` (0.4). Both darken indirect
 *      diffuse per fragment; the composite must carry them in the weight;
 *    - `environment`: a `scene.environment` (a flat grey equirect), so the
 *      environment-diffuse workaround and per-pixel coverage run in both
 *      paths.
 */

import {
    AnalyticLightNode,
    Color,
    DataTexture,
    EquirectangularReflectionMapping,
    FloatType,
    Light,
    Mesh,
    MeshStandardMaterial,
    NodeUpdateType,
    RGBAFormat,
    Scene,
    SphereGeometry,
    type Node,
    type NodeFrame,
} from 'three/webgpu';
import { builtinAOContext, float } from 'three/tsl';
import { LightProbeGrid } from 'three/addons/lighting/LightProbeGrid.js';
import { assert } from '../../client.ts';
import { runThreePage, type ThreePageContext } from '../three-shared/harness.ts';
import {
    CompositeRoutes,
    HdrReader,
    SHOT_SIZE,
    formatMatch,
    loadCompositeScene,
    matchStatistics,
    opaqueMask,
    type CompositeScene,
    type MatchStatistics,
} from '../three-composite-shared/composite-check.ts';

const SCENES = ['cornell-box-point', 'cornell-box'] as const;
/** GI frames before freezing, every probe each frame (Cornell has about 2,000). */
const GI_FRAMES = 120;
const PROBES_PER_FRAME = 4096;
/** The bars (DECISIONS 2026-11-05, T1.14). */
const MEAN_BAR = 0.005;
const P95_BAR = 0.02;
/** The opacity-1 sphere must take at least this share of its light from GI, or the check says nothing. */
const MIN_SPHERE_GI_SHARE = 0.02;
/** Render-time check: frames from an empty field, every probe each frame. */
const RENDER_TIME_FRAMES = 4;

type Images = { perFragment: Float32Array; composite: Float32Array; opaqueOnly: Float32Array; coverage: Float32Array; giOff: Float32Array };

async function compileRoutes(context: ThreePageContext, routes: CompositeRoutes): Promise<void> {
    for (const scenePass of [routes.perFragmentPass, routes.opaquePass, routes.compositePass]) await scenePass.compileAsync(context.renderer);
}

/** Every image the comparison needs, each from a fresh three frame. */
async function takeImages(scene: CompositeScene, routes: CompositeRoutes, reader: HdrReader): Promise<Images> {
    const perFragmentNode = routes.perFragmentPass as unknown as Node;
    for (const node of [perFragmentNode, routes.opaquePass as unknown as Node, routes.composite as unknown as Node]) await reader.warm(node);
    const perFragment = await reader.read(perFragmentNode);
    const opaqueOnly = await reader.read(routes.opaquePass as unknown as Node);
    const composite = await reader.read(routes.composite as unknown as Node);
    // Same three frame: the composite pass's textures as the composite just read them.
    const coverage = await reader.read(routes.coverage, false);
    const light = scene.threeGI.light;
    const intensity = light.intensity;
    light.intensity = 0;
    const giOff = await reader.read(perFragmentNode);
    light.intensity = intensity;
    return { perFragment, composite, opaqueOnly, coverage, giOff };
}

/**
 * Loads the scene, lets `prepare` change it after the field froze (it returns the routes'
 * options), and compares the composite with GI per fragment.
 */
async function matchScene(
    context: ThreePageContext,
    sceneId: string,
    name = `${sceneId} match`,
    prepare: (scene: CompositeScene) => Promise<ConstructorParameters<typeof CompositeRoutes>[1]> = async () => ({}),
): Promise<{ stats: MatchStatistics; scene: CompositeScene; routes: CompositeRoutes; reader: HdrReader }> {
    const scene = await loadCompositeScene(context.device, sceneId, { ...SHOT_SIZE, giFrames: GI_FRAMES, probesPerFrame: PROBES_PER_FRAME, log: context.log });
    const routes = new CompositeRoutes(scene, await prepare(scene));
    const reader = new HdrReader(context.renderer, SHOT_SIZE.width, SHOT_SIZE.height);
    await compileRoutes(context, routes);
    const images = await takeImages(scene, routes, reader);
    const mask = opaqueMask(images.coverage, images.perFragment, images.opaqueOnly);
    const stats = matchStatistics(images.perFragment, images.composite, images.giOff, mask);
    context.log(formatMatch(name, stats));
    return { stats, scene, routes, reader };
}

/** A paused GI frame: ThreeGI re-reads the scene (another probe light, an environment) and sets coverage, without touching the field. */
function resync(scene: CompositeScene): void {
    scene.threeGI.update(scene.camera, 0);
}

/** The review's cases (file comment, 4): each returns the routes' options after changing the scene. */
const EXTRA_CASES: Record<string, (context: ThreePageContext, scene: CompositeScene) => Promise<ConstructorParameters<typeof CompositeRoutes>[1]>> = {
    grid: async (context, scene) => {
        // A grid that saw only a flat grey background: a known, flat irradiance, unlike GI's.
        const bakeScene = new Scene();
        bakeScene.background = new Color(0.5, 0.5, 0.5);
        const grid = new LightProbeGrid(2.8, 5.6, 5.8, 2, 2, 2);
        grid.position.set(1.4, 2.74, 2.8);
        grid.priority = 2;
        grid.bake(context.renderer, bakeScene, { cubemapSize: 8 });
        scene.scene.add(grid);
        resync(scene);
        context.log(`grid: LightProbeGrid at priority ${grid.priority} over the left half; GI computes coverage per pixel: ${scene.threeGI.light.computeCoverage}`);
        return {};
    },
    ao: async (context, scene) => {
        const aoTexture = new DataTexture(new Uint8Array([102, 102, 102, 255, 102, 102, 102, 255, 102, 102, 102, 255, 102, 102, 102, 255]), 2, 2, RGBAFormat);
        aoTexture.needsUpdate = true;
        const sphere = new Mesh(new SphereGeometry(0.6, 48, 24), new MeshStandardMaterial({ color: 0xd0d0d0, roughness: 0.8, aoMap: aoTexture }));
        sphere.position.set(3.6, 0.9, 4.0);
        sphere.userData.giContributes = false;
        scene.scene.add(sphere);
        context.log('ao: builtinAOContext(0.6) on every pass, and a sphere with an aoMap of 0.4');
        return { context: builtinAOContext(float(0.6)) };
    },
    environment: async (context, scene) => {
        const grey = new Float32Array(16).fill(0.3);
        const environment = new DataTexture(grey, 2, 2, RGBAFormat, FloatType);
        environment.mapping = EquirectangularReflectionMapping;
        environment.needsUpdate = true;
        scene.scene.environment = environment;
        resync(scene);
        context.log(`environment: a flat grey scene.environment; GI computes coverage per pixel: ${scene.threeGI.light.computeCoverage}`);
        return {};
    },
};

/** Pixels where hiding `object` changes the per-fragment image: what it covers. */
async function coveredBy(object: Mesh, routes: CompositeRoutes, reader: HdrReader, withObject: Float32Array): Promise<Uint8Array> {
    object.visible = false;
    const without = await reader.read(routes.perFragmentPass as unknown as Node);
    object.visible = true;
    const pixels = withObject.length / 4;
    const mask = new Uint8Array(pixels);
    for (let pixel = 0; pixel < pixels; pixel++) {
        for (let channel = 0; channel < 3; channel++) {
            if (Math.abs(withObject[pixel * 4 + channel] - without[pixel * 4 + channel]) > 1e-6) mask[pixel] = 1;
        }
    }
    return mask;
}

function sphere(opacity: number, x: number): Mesh {
    const mesh = new Mesh(new SphereGeometry(0.55, 48, 24), new MeshStandardMaterial({ color: 0xbfbfbf, roughness: 0.7, transparent: true, opacity }));
    mesh.name = `transparent sphere, opacity ${opacity}`;
    mesh.position.set(x, 0.9, 4.2);
    // Added after the field froze; keep it out of GI explicitly so a later sync can't add it.
    mesh.userData.giContributes = false;
    return mesh;
}

interface TransparentReport {
    readonly opaqueSphere: MatchStatistics;
    readonly halfSphere: MatchStatistics;
    /** GI's share of each sphere's light in the composite image: (with GI − GI light at 0) / with GI. */
    readonly opaqueSphereGiShareComposite: number;
    readonly halfSphereGiShareComposite: number;
}

function meanLuminance(pixels: Float32Array, mask: Uint8Array): number {
    let sum = 0;
    let count = 0;
    for (let pixel = 0; pixel < mask.length; pixel++) {
        if (!mask[pixel]) continue;
        sum += 0.2126 * pixels[pixel * 4] + 0.7152 * pixels[pixel * 4 + 1] + 0.0722 * pixels[pixel * 4 + 2];
        count++;
    }
    return count > 0 ? sum / count : NaN;
}

async function transparentCheck(context: ThreePageContext, scene: CompositeScene, routes: CompositeRoutes, reader: HdrReader): Promise<TransparentReport> {
    const opaqueSphere = sphere(1, 1.5);
    const halfSphere = sphere(0.5, 4.0);
    scene.scene.add(opaqueSphere, halfSphere);
    await compileRoutes(context, routes);
    const images = await takeImages(scene, routes, reader);
    const light = scene.threeGI.light;
    light.intensity = 0;
    const compositeOff = await reader.read(routes.composite as unknown as Node);
    light.intensity = 1;
    const opaqueMaskPixels = await coveredBy(opaqueSphere, routes, reader, images.perFragment);
    const halfMaskPixels = await coveredBy(halfSphere, routes, reader, images.perFragment);
    const share = (mask: Uint8Array) => 1 - meanLuminance(compositeOff, mask) / meanLuminance(images.composite, mask);
    const report: TransparentReport = {
        opaqueSphere: matchStatistics(images.perFragment, images.composite, images.giOff, opaqueMaskPixels),
        halfSphere: matchStatistics(images.perFragment, images.composite, images.giOff, halfMaskPixels),
        opaqueSphereGiShareComposite: share(opaqueMaskPixels),
        halfSphereGiShareComposite: share(halfMaskPixels),
    };
    context.log(formatMatch('transparent sphere, opacity 1', report.opaqueSphere));
    context.log(formatMatch('transparent sphere, opacity 0.5', report.halfSphere));
    context.log(`GI share in the composite image: opacity-1 sphere ${(report.opaqueSphereGiShareComposite * 100).toFixed(2)}%, opacity-0.5 sphere ${(report.halfSphereGiShareComposite * 100).toFixed(2)}%`);
    scene.scene.remove(opaqueSphere, halfSphere);
    return report;
}

// -- the render-time update ----------------------------------------------------------------------

/** A light that lights nothing; its node runs `onRender` in its render-time update, as a GI light's node would. */
class RenderTimeUpdateLight extends Light {
    readonly isRenderTimeUpdateLight = true;

    constructor(readonly onRender: (frame: NodeFrame) => void) {
        super(0xffffff, 0);
    }
}

class RenderTimeUpdateNode extends AnalyticLightNode<RenderTimeUpdateLight> {
    static get type(): string {
        return 'RenderTimeUpdateNode';
    }

    constructor(light: RenderTimeUpdateLight | null = null) {
        super(light);
        // The GI light node's own update type (gi-light.ts): once per render() that builds it.
        this.updateType = NodeUpdateType.RENDER;
    }

    update(frame: NodeFrame): boolean | undefined {
        this.light!.onRender(frame);
        return undefined;
    }

    setup(): Node | undefined {
        return undefined;
    }
}

RenderTimeUpdateLight.registerNode(RenderTimeUpdateNode);

/** What reaches the queue, in order: render-pass starts and submitted command buffers, by label. */
class QueueLog {
    readonly events: string[] = [];
    private readonly labels = new WeakMap<object, string>();

    constructor(device: GPUDevice, renderer: ThreePageContext['renderer']) {
        const createEncoder = device.createCommandEncoder.bind(device);
        device.createCommandEncoder = (descriptor?: GPUCommandEncoderDescriptor) => {
            const encoder = createEncoder(descriptor);
            const label = descriptor?.label ?? '';
            const finish = encoder.finish.bind(encoder);
            encoder.finish = (finishDescriptor?: GPUCommandBufferDescriptor) => {
                const buffer = finish(finishDescriptor);
                this.labels.set(buffer, label);
                return buffer;
            };
            return encoder;
        };
        const queue = device.queue;
        const submit = queue.submit.bind(queue);
        queue.submit = (buffers: Iterable<GPUCommandBuffer>) => {
            for (const buffer of buffers) this.events.push(`submit ${this.labels.get(buffer) ?? '?'}`);
            submit(buffers);
        };
        const backend = renderer.backend as unknown as { beginRender(context: { fullscreenPass?: boolean; id?: number }): void };
        const beginRender = backend.beginRender.bind(backend);
        backend.beginRender = (renderContext) => {
            this.events.push(`beginRender ${renderContext.fullscreenPass ? 'quad' : 'scene'} renderContext_${renderContext.id ?? '?'}`);
            beginRender(renderContext);
        };
    }
}

interface RenderTimeReport {
    readonly renderTimeAgainstExplicit: MatchStatistics;
    readonly renderTimeAgainstExplicitOneFrameBehind: MatchStatistics;
    readonly explicitRunToRun: MatchStatistics;
    readonly updatesPerFrame: number[];
    /** The queue events of the last render-time frame. */
    readonly lastFrameEvents: string[];
    readonly giSubmitInsideScenePass: boolean;
}

async function renderTimeCheck(context: ThreePageContext): Promise<RenderTimeReport> {
    const sceneId = 'cornell-box-point';
    const fresh = () => loadCompositeScene(context.device, sceneId, { ...SHOT_SIZE, giFrames: 0, probesPerFrame: PROBES_PER_FRAME, log: context.log });
    const reader = new HdrReader(context.renderer, SHOT_SIZE.width, SHOT_SIZE.height);

    /** K frames of "update, then render", keeping every image. */
    const explicitRun = async () => {
        const scene = await fresh();
        const routes = new CompositeRoutes(scene);
        await compileRoutes(context, routes);
        scene.threeGI.budget = { units: PROBES_PER_FRAME };
        const images: Float32Array[] = [];
        for (let frame = 0; frame < RENDER_TIME_FRAMES; frame++) {
            scene.threeGI.update(scene.camera, 1 / 60);
            images.push(await reader.read(routes.perFragmentPass as unknown as Node));
        }
        const coverage = await reader.read(routes.coverage, true);
        scene.threeGI.dispose();
        scene.platform.dispose();
        return { images, coverage };
    };

    const first = await explicitRun();
    const second = await explicitRun();

    const scene = await fresh();
    const routes = new CompositeRoutes(scene);
    await compileRoutes(context, routes);
    scene.threeGI.budget = { units: PROBES_PER_FRAME };
    const updatesPerFrame: number[] = [];
    let updatesThisFrame = 0;
    let lastFrameId = -1;
    const driver = new RenderTimeUpdateLight((frame) => {
        // Once per three frame: a frame can hold several lit renders (cube captures, reflections).
        if (frame.frameId === lastFrameId) return;
        lastFrameId = frame.frameId;
        updatesThisFrame++;
        scene.threeGI.update(scene.camera, 1 / 60);
    });
    driver.layers.enableAll();
    scene.scene.add(driver);
    const log = new QueueLog(context.device, context.renderer);
    let renderTime: Float32Array | null = null;
    let lastFrameEvents: string[] = [];
    for (let frame = 0; frame < RENDER_TIME_FRAMES; frame++) {
        updatesThisFrame = 0;
        const start = log.events.length;
        renderTime = await reader.read(routes.perFragmentPass as unknown as Node);
        updatesPerFrame.push(updatesThisFrame);
        lastFrameEvents = log.events.slice(start);
    }
    const sceneBegin = lastFrameEvents.findIndex((event) => event.startsWith('beginRender scene'));
    const sceneContext = sceneBegin >= 0 ? lastFrameEvents[sceneBegin].split(' ')[2] : '';
    const giSubmit = lastFrameEvents.indexOf('submit ddgi-three frame');
    const sceneSubmit = lastFrameEvents.findIndex((event, index) => index > sceneBegin && event === `submit ${sceneContext}`);
    const giSubmitInsideScenePass = sceneBegin >= 0 && giSubmit > sceneBegin && sceneSubmit > giSubmit;

    const mask = new Uint8Array(first.coverage.length / 4);
    for (let pixel = 0; pixel < mask.length; pixel++) mask[pixel] = first.coverage[pixel * 4] > 0.5 ? 1 : 0;
    const last = RENDER_TIME_FRAMES - 1;
    const black = new Float32Array(first.images[last].length);
    const report: RenderTimeReport = {
        renderTimeAgainstExplicit: matchStatistics(first.images[last], renderTime!, black, mask),
        renderTimeAgainstExplicitOneFrameBehind: matchStatistics(first.images[last - 1], renderTime!, black, mask),
        explicitRunToRun: matchStatistics(first.images[last], second.images[last], black, mask),
        updatesPerFrame,
        lastFrameEvents,
        giSubmitInsideScenePass,
    };
    context.log(formatMatch(`render-time update against explicit, both after ${RENDER_TIME_FRAMES} GI frames`, report.renderTimeAgainstExplicit));
    context.log(formatMatch(`render-time update against explicit one GI frame behind`, report.renderTimeAgainstExplicitOneFrameBehind));
    context.log(formatMatch('explicit against explicit (run-to-run noise)', report.explicitRunToRun));
    context.log(`GI updates per frame in render-time mode: ${updatesPerFrame.join(', ')}; last frame's queue: ${lastFrameEvents.join(' | ')}`);
    scene.threeGI.dispose();
    scene.platform.dispose();
    reader.dispose();
    return report;
}

await runThreePage(
    async (context) => {
        const { renderer, data, log } = context;
        renderer.shadowMap.enabled = true;
        const matches: Record<string, MatchStatistics> = {};
        for (const sceneId of SCENES) {
            const { stats, scene, routes, reader } = await matchScene(context, sceneId);
            matches[sceneId] = stats;
            data.match = matches;
            if (sceneId === 'cornell-box-point') data.transparent = await transparentCheck(context, scene, routes, reader);
            reader.dispose();
            scene.threeGI.dispose();
            scene.platform.dispose();
        }
        for (const [name, prepare] of Object.entries(EXTRA_CASES)) {
            const { stats, scene, reader } = await matchScene(context, 'cornell-box-point', `${name} match`, (loaded) => prepare(context, loaded));
            matches[name] = stats;
            data.match = matches;
            reader.dispose();
            scene.threeGI.dispose();
            scene.platform.dispose();
        }
        data.renderTimeUpdate = await renderTimeCheck(context);
        log('done');

        for (const [sceneId, stats] of Object.entries(matches)) {
            assert(stats.pixels > 0, `${sceneId}: no pixels compared`);
            // With scene.environment, GI replaces the environment's diffuse light, so turning GI off can brighten the image: the size counts, not the sign.
            assert(Math.abs(stats.giShare) > 0.05, `${sceneId}: GI changes the light by only ${(stats.giShare * 100).toFixed(2)}%, too little for the check to mean anything`);
            assert(stats.meanRelative <= MEAN_BAR, `${sceneId}: composite against per fragment, mean ${(stats.meanRelative * 100).toFixed(3)}% > ${MEAN_BAR * 100}%`);
            assert(stats.p95Relative <= P95_BAR, `${sceneId}: composite against per fragment, p95 ${(stats.p95Relative * 100).toFixed(3)}% > ${P95_BAR * 100}%`);
        }
        const transparent = data.transparent as TransparentReport;
        assert(transparent.opaqueSphere.pixels > 1000, 'the opacity-1 sphere covers too few pixels');
        assert(transparent.opaqueSphere.meanRelative <= MEAN_BAR && transparent.opaqueSphere.p95Relative <= P95_BAR, 'the opacity-1 transparent sphere differs from its per-fragment image');
        assert(transparent.opaqueSphereGiShareComposite >= MIN_SPHERE_GI_SHARE, `the transparent sphere gets ${(transparent.opaqueSphereGiShareComposite * 100).toFixed(2)}% of its light from GI in the composite`);
    },
    SHOT_SIZE,
);
