/**
 * M4, T4.5: GI's indirect light on a textured scene, at the cameras of its
 * textured Cycles references (`cycles-textured-gi.exr`), on the path M1's HD
 * accuracy was judged on (DECISIONS 2026-11-06, T1.14's default): the
 * composite (`ddgi()`) through `@pmndrs/upscaler` at its Performance preset,
 * with three's GTAO on the GI term at half the reduced pass's resolution
 * (1 m, 16 samples), read back at the reference's 1920 × 1080. That GTAO
 * was the composite's default until 2026-11-18; the default since has no
 * occlusion (DECISIONS 2026-11-18), so the page measures that by default and
 * `run.ts --occlusion` turns the GTAO on.
 *
 * What differs from M1's run (`three-composite-ao`, match mode): the scene
 * keeps its own materials. M1 replaced them with Lambert materials holding
 * the constant colour the GI platform received, for the untextured
 * reference; since T4.1 and T4.3 GI samples the textures itself, and the
 * textured references (T4.4) render the glTF's own materials the way GI
 * reads them. So the GI term here is three's `giAlbedo` (the material's
 * diffuse colour, textures included) × the field's E/π × GTAO.
 *
 * Steps (the job comes from run.ts, `window.m4TexturedAccuracyJob`):
 * 1. Load the scene with the viewer's loader (Bistro with T2.17's texture
 *    cap, `maxTextureSize`), point and spot lights without a range window
 *    (as Cycles lights them, M1's `ignoreLightRange`), the sky dome left
 *    out of GI (GI takes the sky from the platform environment).
 * 2. DDGI on the manifest's volume through `ThreeGI`, converged the way the
 *    M1 accuracy pages converge: `warmupPasses` whole passes with a running
 *    average, then `passes` at the normal hysteresis; then frozen.
 * 3. For each view: the GI term through the upscaler, `upscaleFrames`
 *    frames on a still camera (its jitter and history), read back in linear
 *    HDR, before tone mapping.
 *
 * Each view's GI term (rgb floats, rows as read back) is posted to the
 * runner's server (`UPLOAD_PATH`), not put in the report: three 1920 × 1080
 * float images in base64 made the report too big to read back within
 * DevTools' 30 s.
 *
 * Also returned, for the record: the texture store's memory
 * (`stats.memory.textures`) and the probe count.
 *
 * Not a shading path: everything here is measurement.
 */

import { Box3, BoxGeometry, Euler, FloatType, Matrix4, Mesh as ThreeMesh, MeshStandardMaterial, PerspectiveCamera, RenderPipeline, RenderTarget, Vector3, type Mesh, type Node, type Object3D } from 'three/webgpu';
import { assert } from '../../client.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { ThreeGI } from '../../../../packages/ddgi-three/src/three-gi.ts';
import { DEFAULT_GI_OCCLUSION_RADIUS } from '../../../../packages/ddgi-three/src/composite.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { runSteps, runThreePage, type ThreePageContext } from '../three-shared/harness.ts';
import { advanceThreeFrame, findManifest, SHOT_FAR, SHOT_NEAR } from '../three-composite-shared/composite-check.ts';
import { upscaledComposite, type UpscalePreset } from '../three-composite-shared/variants.ts';
import endStates from '../../../reference/cycles/end-states.json';
import { UPLOAD_PATH } from './upload-path.ts';

/** A reference's stored camera (its json's `shot.camera`). */
export interface AccuracyView {
    readonly name: string;
    readonly position: [number, number, number];
    readonly target: [number, number, number];
    readonly fovYDeg: number;
}

export interface TexturedAccuracyJob {
    readonly scene: string;
    /** An entry of tools/reference/cycles/end-states.json (TU.4): objects turned about a pivot and boxes added, as the Cycles end-state render has them. */
    readonly endState: string | null;
    /**
     * A diagnostic for foliage (TU.4, sponza-full): 'default' leaves the loaded materials alone; 'cutout' makes every
     * blended (`transparent`, no alphaTest) material an alpha-tested one (alphaTest 0.5, opaque pass), 'opaque' drops
     * its alpha altogether. Neither is what the reference has (it keeps the glTF's BLEND); they tell where a difference
     * comes from.
     */
    readonly materialVariant: 'default' | 'cutout' | 'opaque';
    readonly views: readonly AccuracyView[];
    /** The references' size; every view is read back at it. */
    readonly width: number;
    readonly height: number;
    /** T2.17's cap on three's own textures (Bistro on the 6 GB RTX), or null for the files' sizes. GI's store is 256² a layer either way. */
    readonly maxTextureSize: number | null;
    /** Whole DDGI passes with a running average, then passes at the normal hysteresis (M1's accuracy pages). */
    readonly warmupPasses: number;
    readonly passes: number;
    /** The upscaler preset; M1's default path is Performance. */
    readonly preset: UpscalePreset;
    /**
     * GTAO on the GI term, as M1's default path had it (true, `run.ts --occlusion`), or none (false: the
     * shipping composite's default since 2026-11-18; run both to tell the occlusion's share of a
     * difference from the field's). Its resolution against the reduced pass, and its samples (M1's default path: 0.5, 16).
     */
    readonly occlusion: boolean;
    readonly occlusionScale: number;
    readonly occlusionSamples: number;
    /** Frames on a still camera through the upscaler before the read-back. */
    readonly upscaleFrames: number;
}

const DEFAULT_JOB: TexturedAccuracyJob = {
    scene: 'textured-furnace',
    endState: null,
    materialVariant: 'default',
    views: [],
    width: 1920,
    height: 1080,
    maxTextureSize: null,
    warmupPasses: 16,
    passes: 64,
    preset: 'performance',
    occlusion: false,
    occlusionScale: 0.5,
    occlusionSamples: 16,
    upscaleFrames: 32,
};

declare global {
    interface Window {
        m4TexturedAccuracyJob?: Partial<TexturedAccuracyJob>;
    }
}

const job: TexturedAccuracyJob = { ...DEFAULT_JOB, ...window.m4TexturedAccuracyJob };

/** The DDGI technique's default `maxProbesPerFrame`: whole passes are split into frames of at most this many probes. */
const MAX_PROBES_PER_FRAME = 4096;

interface EndState {
    readonly scene: string;
    readonly moves?: readonly { readonly object: string; readonly pivot: [number, number, number]; readonly rotationDeg: [number, number, number] }[];
    readonly boxes?: readonly { readonly name: string; readonly size: [number, number, number]; readonly color: string; readonly x: number; readonly floorGap: number; readonly wallGap: number }[];
}

/**
 * Sets the scene to an end state before GI first sees it: each move as the viewer's motion drivers place an
 * object (pivot x turn x pivot^-1 x rest), each box as ddgi-move-cabinet stands its cabinet. The same
 * end-states.json drives Cycles (render.py --end-state). Returns what it did, with the content's bounds the boxes stood on.
 */
function applyEndState(state: EndState, content: Object3D): Record<string, unknown> {
    const applied: Record<string, unknown> = {};
    content.updateMatrixWorld(true);
    for (const move of state.moves ?? []) {
        const target = content.getObjectByName(move.object);
        assert(target, `end state: no object ${move.object}`);
        const pivot = new Vector3(...move.pivot);
        const turn = new Matrix4().makeRotationFromEuler(new Euler(...(move.rotationDeg.map((degrees) => (degrees * Math.PI) / 180) as [number, number, number]), 'XYZ'));
        const world = new Matrix4().makeTranslation(pivot.x, pivot.y, pivot.z).multiply(turn).multiply(new Matrix4().makeTranslation(-pivot.x, -pivot.y, -pivot.z)).multiply(target.matrixWorld);
        // The parent is the identity here (procedural parts add their meshes to the content group), checked below.
        assert(target.parent !== null && target.parent.matrixWorld.equals(new Matrix4()), `end state: ${move.object}'s parent is not the identity`);
        target.matrix.copy(world);
        target.matrix.decompose(target.position, target.quaternion, target.scale);
        target.updateMatrixWorld(true);
        applied[move.object] = { position: target.position.toArray(), rotationDeg: move.rotationDeg };
    }
    const bounds = new Box3().setFromObject(content);
    applied.bounds = { min: bounds.min.toArray(), max: bounds.max.toArray() };
    for (const box of state.boxes ?? []) {
        const mesh = new ThreeMesh(new BoxGeometry(...box.size), new MeshStandardMaterial({ color: Number.parseInt(box.color, 16), roughness: 1, metalness: 0 }));
        mesh.name = box.name;
        mesh.position.set(box.x, bounds.min.y + box.size[1] / 2 + box.floorGap, bounds.min.z + box.size[2] / 2 + box.wallGap);
        content.add(mesh);
        applied[box.name] = { position: mesh.position.toArray() };
    }
    content.updateMatrixWorld(true);
    return applied;
}

/** rgb of an rgba read-back. */
function rgbOf(rgba: Float32Array): Float32Array<ArrayBuffer> {
    const rgb = new Float32Array((rgba.length / 4) * 3);
    for (let pixel = 0; pixel < rgba.length / 4; pixel++) for (let channel = 0; channel < 3; channel++) rgb[pixel * 3 + channel] = rgba[pixel * 4 + channel];
    return rgb;
}

async function measure(context: ThreePageContext): Promise<void> {
    const { device, renderer, data, log } = context;
    assert(job.views.length > 0, 'the job names no views');
    const manifest = findManifest(job.scene);
    const loaded = await loadScene(manifest, () => {}, job.maxTextureSize === null ? {} : { maxTextureSize: job.maxTextureSize });
    const scene = loaded.scene;
    if (job.endState) {
        const state = (endStates as unknown as Record<string, EndState>)[job.endState];
        assert(state && state.scene === job.scene, `no end state ${job.endState} of ${job.scene}`);
        const applied = applyEndState(state, loaded.content);
        log(`end state ${job.endState}: ${JSON.stringify(applied)}`);
        data.endState = applied;
    }
    if (job.materialVariant !== 'default') {
        const changed = new Set<string>();
        scene.traverse((object) => {
            const mesh = object as Mesh;
            if (!mesh.isMesh) return;
            for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
                if (!material.transparent || material.alphaTest > 0) continue;
                material.transparent = false;
                material.depthWrite = true;
                material.alphaHash = false;
                material.alphaTest = job.materialVariant === 'cutout' ? 0.5 : 0;
                material.needsUpdate = true;
                changed.add(material.name);
            }
        });
        log(`material variant ${job.materialVariant}: ${changed.size} blended materials changed: ${[...changed].join(', ')}`);
        data.variantMaterials = [...changed];
    }
    // Cycles lights point and spot lights with no range window (M1's `ignoreLightRange`).
    scene.traverse((object) => {
        const light = object as Object3D & { isPointLight?: boolean; isSpotLight?: boolean; distance?: number };
        if (light.isPointLight || light.isSpotLight) light.distance = 0;
    });
    // The sky dome is for viewing; GI takes the sky from the platform environment.
    const content = new Set<Object3D>();
    loaded.content.traverse((object) => content.add(object));
    scene.traverse((object) => {
        if ((object as Mesh).isMesh && !content.has(object)) object.userData.giContributes = false;
    });

    const camera = new PerspectiveCamera(job.views[0].fovYDeg, job.width / job.height, SHOT_NEAR, SHOT_FAR);
    const placeCamera = (view: AccuracyView) => {
        camera.fov = view.fovYDeg;
        camera.position.set(...view.position);
        camera.lookAt(...view.target);
        camera.updateProjectionMatrix();
        camera.updateMatrixWorld(true);
    };
    placeCamera(job.views[0]);

    const platform = new GIPlatformImpl(device, { label: `m4 textured accuracy ${job.scene}` });
    const volume = manifest.volume;
    const ddgi = platform.addTechnique(new DDGITechnique({ volume: { center: volume.center, size: volume.size, spacing: volume.probeSpacing }, warmUpPasses: 0 }));
    platform.setOutput(ddgi);
    await platform.prepare();
    const environment = manifest.environment.type === 'procedural' ? platformSkyEnvironment(manifest) : null;
    const threeGI = new ThreeGI(platform, scene, { budget: { units: 0 }, environment });
    threeGI.update(camera);
    await platform.rays.whenBuilt();
    threeGI.update(camera);

    const field = ddgi.field;
    assert(field, 'DDGI has no field');
    const probeCount = field.grid.probeCount;
    // Whole passes split into equal frames, as the M1 accuracy pages do.
    const framesPerPass = Math.ceil(probeCount / MAX_PROBES_PER_FRAME);
    threeGI.budget = { units: Math.ceil(probeCount / framesPerPass) };
    const normalHysteresis = ddgi.settings.hysteresis;
    const runPasses = (passes: number) => runSteps(device, passes * framesPerPass, () => threeGI.update(camera, 1 / 60));
    for (let pass = 0; pass < job.warmupPasses; pass++) {
        // Pass k keeps k/(k+1): the field is the plain mean of the passes so far.
        ddgi.settings.hysteresis = Math.min(normalHysteresis, pass / (pass + 1));
        await runPasses(1);
    }
    ddgi.settings.hysteresis = normalHysteresis;
    await runPasses(job.passes);
    threeGI.budget = { paused: true };
    const textureBytes = platform.stats.memory.textures;
    log(`${job.scene}: ${probeCount} probes, ${job.warmupPasses} + ${job.passes} passes, then frozen; texture store ${(textureBytes / 2 ** 20).toFixed(1)} MB`);
    data.probeCount = probeCount;
    data.textureStoreBytes = textureBytes;
    data.recompiles = platform.stats.recompiles;

    renderer.setSize(job.width, job.height, false);
    const target = new RenderTarget(job.width, job.height, { type: FloatType, depthBuffer: false });
    const occlusion = job.occlusion ? { resolutionScale: job.occlusionScale, radius: DEFAULT_GI_OCCLUSION_RADIUS, samples: job.occlusionSamples } : false;
    const uploaded: string[] = [];
    for (const view of job.views) {
        placeCamera(view);
        // A fresh pipeline per view: the upscaler's history starts at this camera.
        const built = upscaledComposite(scene, camera, threeGI, job.preset, { occlusion, giTermOnly: true });
        const pipeline = new RenderPipeline(renderer, built.node as unknown as Node);
        // Linear HDR out: the read-back compares radiance, not a tone-mapped picture.
        pipeline.outputColorTransform = false;
        for (let frame = 0; frame < job.upscaleFrames; frame++) {
            advanceThreeFrame(renderer);
            renderer.setRenderTarget(target);
            pipeline.render();
            renderer.setRenderTarget(null);
            if (frame % 4 === 3) await device.queue.onSubmittedWorkDone();
        }
        const pixels = new Float32Array((await renderer.readRenderTargetPixelsAsync(target, 0, 0, job.width, job.height)) as Float32Array);
        let nonFinite = 0;
        for (const value of pixels) if (!Number.isFinite(value)) nonFinite++;
        assert(nonFinite === 0, `${view.name}: ${nonFinite} non-finite values in the GI term`);
        const response = await fetch(`${UPLOAD_PATH}?view=${encodeURIComponent(view.name)}`, { method: 'POST', body: rgbOf(pixels) });
        assert(response.ok, `${view.name}: the upload failed (${response.status})`);
        uploaded.push(view.name);
        log(`${view.name}: GI term through the upscaler (${job.preset}, ${job.occlusion ? `GTAO at ${job.occlusionScale}` : 'no occlusion'}), ${job.upscaleFrames} frames`);
        pipeline.dispose();
    }
    target.dispose();
    data.uploadedViews = uploaded;
    data.recompilesAfter = platform.stats.recompiles;
}

await runThreePage(
    async (context) => {
        context.data.job = { ...job, views: job.views.map((view) => view.name) };
        await measure(context);
        context.log('done');
    },
    { width: job.width, height: job.height },
);
