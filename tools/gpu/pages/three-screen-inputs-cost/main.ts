/**
 * RC1.1's cost: what filling the platform's `FrameScreen` from three's scene
 * pass costs on the GPU, on Sponza with its curtains at its first bookmark,
 * against the T1.14 composite's own MRT pass (`mrt({ output, giAlbedo,
 * normal })`, the default path's pass). GPU-bound, the variants drawn
 * interleaved frame by frame (composite-check.ts, `timeInterleaved`), each
 * with its own pass and camera so three's caches hold:
 * - `composite-mrt`: the composite's pass, its colour shown;
 * - `fill-normal`: that pass plus the world normal (`giNormal`), the screen
 *   stage attached (`screenInputs(...).encodeBefore`): depth and normals,
 *   what a technique asking for `screen: true` gets;
 * - `fill-no-instance`: plus `velocity` and `giDiffuseAlbedo`;
 * - `fill-all`: plus `giInstance`, every channel.
 * No screen technique is attached, so the stage hands the textures over and
 * records only the instance ids' resolve (`fill-all`): the differences are
 * the extra MRT outputs. The resolve is a small compute pass in the stage's
 * own encoder, outside three's timestamps, so it isn't counted.
 * (Timing deferred by Dennis, 2026-11-09: at 1280 × 720 this page is not
 * GPU-bound yet, see PROGRESS.)
 *
 * Two sections: 1280 × 720 and 2560 × 1440. Timing goes through run.ts under
 * the bench lock, which adds the GPU clock and the lock's verdict and can
 * write the rows to bench/results/screen-inputs.csv:
 *   node tools/reference/bench-lock.ts run --what "rc1.1 screen inputs cost" -- node tools/gpu/pages/three-screen-inputs-cost/run.ts --gpu nvidia --csv
 *
 * Settings (vite exposes `VITE_*` variables to the page):
 *   VITE_DDGI_GI_FRAMES     GI frames of 4096 probes before the field freezes (default 8: the cost doesn't depend on the field)
 *   VITE_DDGI_SECONDS_720   the 1280 × 720 section, seconds (default 10; 0 skips it)
 *   VITE_DDGI_SECONDS_1440  the 2560 × 1440 section, seconds (default 10; 0 skips it)
 *   VITE_DDGI_COOL          idle seconds after loading, before timing (default 25)
 *   VITE_DDGI_RAMP          GPU-bound seconds before each section (default 2)
 */

import { RenderPipeline, type Node, type PerspectiveCamera } from 'three/webgpu';
import { mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import { markCompositePass, type MarkablePass } from '../../../../packages/three/src/composite-mark.ts';
import { giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT } from '../../../../packages/three/src/composite.ts';
import { screenInputs } from '../../../../packages/three/src/screen-inputs.ts';
import type { ScreenChannelRequest } from '../../../../packages/core/src/platform/contract/technique.ts';
import { runThreePage, type ThreePageContext } from '../three-shared/harness.ts';
import { fullBrowserVersion } from '../trace-bench/run.ts';
import { PassClock, advanceThreeFrame, loadCompositeScene, shadowCameras, timeInterleaved, type CompositeScene, type DeltaDefinition, type FrameVariant, type InterleavedTiming } from '../three-composite-shared/composite-check.ts';

const SECTIONS = [
    { id: '720', width: 1280, height: 720, secondsKey: 'VITE_DDGI_SECONDS_720' },
    { id: '1440', width: 2560, height: 1440, secondsKey: 'VITE_DDGI_SECONDS_1440' },
] as const;
const PROBES_PER_FRAME = 4096;
const CYCLES_PER_BATCH = 4;
/** Frames each variant is drawn, with a wait, before timing: pipelines compile and draw calls settle. */
const COMPILE_FRAMES = 6;

/** Variant ids, and the channels each fill asks for beyond depth and normals. */
const FILLS: ReadonlyArray<{ readonly id: string; readonly channels: ScreenChannelRequest }> = [
    { id: 'fill-normal', channels: {} },
    { id: 'fill-no-instance', channels: { velocity: true, albedo: true } },
    { id: 'fill-all', channels: { velocity: true, albedo: true, instance: true } },
];

const DELTAS: readonly DeltaDefinition[] = [
    { name: 'filling every channel (fill-all - composite-mrt)', minuend: 'fill-all', subtrahend: 'composite-mrt', part: 'total' },
    { name: 'depth and normals only (fill-normal - composite-mrt)', minuend: 'fill-normal', subtrahend: 'composite-mrt', part: 'total' },
    { name: 'every channel but instance ids (fill-no-instance - composite-mrt)', minuend: 'fill-no-instance', subtrahend: 'composite-mrt', part: 'total' },
    { name: 'instance ids (fill-all - fill-no-instance)', minuend: 'fill-all', subtrahend: 'fill-no-instance', part: 'total' },
    { name: 'velocity and albedo (fill-no-instance - fill-normal)', minuend: 'fill-no-instance', subtrahend: 'fill-normal', part: 'total' },
];

const env = import.meta.env as Record<string, unknown>;

function numberFromEnv(value: unknown, fallback: number): number {
    const parsed = typeof value === 'string' && value.trim() !== '' ? Number(value) : NaN;
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

function setViewSize(context: ThreePageContext, cameras: readonly PerspectiveCamera[], size: { width: number; height: number }): void {
    context.renderer.setSize(size.width, size.height, false);
    for (const camera of cameras) {
        camera.aspect = size.width / size.height;
        camera.updateProjectionMatrix();
    }
}

/** The composite's MRT pass, marked (no GI per fragment, as T1.14's `mrt` variant), on its own camera. */
function compositePass(scene: CompositeScene, camera: PerspectiveCamera) {
    const scenePass = pass(scene.scene, camera);
    markCompositePass(scenePass as unknown as MarkablePass);
    return scenePass;
}

async function timing(context: ThreePageContext, scene: CompositeScene): Promise<InterleavedTiming[]> {
    const { renderer, device, log } = context;
    const cameras: PerspectiveCamera[] = [];
    const pipelines: RenderPipeline[] = [];
    const cameraOf = () => {
        const camera = scene.camera.clone();
        cameras.push(camera);
        return camera;
    };
    const variants: FrameVariant[] = [];
    const nothing = () => {};

    const base = compositePass(scene, cameraOf());
    base.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView) }));
    // An MRT output is drawn only once its texture exists (NOTES).
    for (const name of [GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT]) base.getTexture(name);
    const basePipeline = new RenderPipeline(renderer, base.getTextureNode('output') as unknown as Node);
    pipelines.push(basePipeline);
    variants.push({ id: 'composite-mrt', apply: nothing, draw: () => basePipeline.render() });

    for (const fill of FILLS) {
        const scenePass = compositePass(scene, cameraOf());
        const screen = screenInputs(scenePass, scene.threeGI, { channels: fill.channels });
        scenePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), ...screen.outputs }));
        for (const name of [GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT]) scenePass.getTexture(name);
        const pipeline = new RenderPipeline(renderer, screen.encodeBefore(scenePass.getTextureNode('output') as unknown as Node));
        pipelines.push(pipeline);
        variants.push({ id: fill.id, apply: nothing, draw: () => pipeline.render() });
        log(`${fill.id}: outputs ${Object.keys(screen.outputs).join(', ')}`);
    }

    const sections: InterleavedTiming[] = [];
    let clock: PassClock | null = null;
    for (const section of SECTIONS) {
        const seconds = numberFromEnv(env[section.secondsKey], 10);
        if (seconds <= 0) continue;
        setViewSize(context, cameras, section);
        // Compile (and, after a resize, reallocate every target): a few frames of each, with waits.
        for (let frame = 0; frame < COMPILE_FRAMES; frame++) {
            for (const variant of variants) {
                advanceThreeFrame(renderer);
                variant.draw();
            }
            await device.queue.onSubmittedWorkDone();
        }
        if (!clock) {
            const coolSeconds = numberFromEnv(env.VITE_DDGI_COOL, 25);
            log(`compiled ${variants.length} variants; idle for ${coolSeconds} s`);
            await new Promise((resolve) => setTimeout(resolve, coolSeconds * 1000));
            // The scene doesn't move: the shadow map drawn while compiling stays right (three-composite-sponza says why).
            for (const light of scene.loaded.directLights) {
                const shadow = (light as { shadow?: { autoUpdate: boolean } }).shadow;
                if (shadow) shadow.autoUpdate = false;
            }
            clock = new PassClock(renderer, shadowCameras(scene));
        } else {
            await clock.read();
        }
        sections.push(
            await timeInterleaved(renderer, device, clock, variants, DELTAS, {
                label: `${section.width} x ${section.height}`,
                seconds,
                cyclesPerBatch: CYCLES_PER_BATCH,
                rampSeconds: numberFromEnv(env.VITE_DDGI_RAMP, 2),
                log,
            }),
        );
    }
    for (const pipeline of pipelines) pipeline.dispose();
    return sections;
}

await runThreePage(
    async (context) => {
        const { renderer, device, data, log } = context;
        renderer.shadowMap.enabled = true;
        data.userAgent = navigator.userAgent;
        data.browserVersion = await fullBrowserVersion();
        const giFrames = numberFromEnv(env.VITE_DDGI_GI_FRAMES, 8);
        const scene = await loadCompositeScene(device, 'sponza', { width: SECTIONS[0].width, height: SECTIONS[0].height, giFrames, probesPerFrame: PROBES_PER_FRAME, log });
        data.timing = { sections: await timing(context, scene) };
        log('done');
    },
    { width: SECTIONS[0].width, height: SECTIONS[0].height },
    { trackTimestamp: true },
);
