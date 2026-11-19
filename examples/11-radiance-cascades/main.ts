import * as THREE from 'three/webgpu';
import { RealtimeRadianceCascades } from './RealtimeRadianceCascades.js';
import { makePassage, REPLAY_FRAMES, type CameraPose, type LampCue } from './scene.ts';

import { makeDoorChamber, type DoorMode } from './door-scene.ts';
import { createReconstruction, type ReconstructionMode } from './reconstruction.ts';
import { createNativeAdapter, type RawBackend } from './native-adapter.ts';
import { frameTimestamps, receiverMeter } from './measurement.ts';

type View = 'lit' | 'indirect' | 'direct';
interface VerifyOptions {
    frames: number;
    frame?: number;
    advance?: boolean;
    camera?: CameraPose;
    cue?: LampCue;
    view?: View;
    measure?: boolean;
    occupancy?: boolean;
    captureSize?: 'display';
    linearReceiver?: boolean;
    door?: DoorMode;
}

const params = new URLSearchParams(location.search);
const verify = params.get('verify') === '1';
const status = document.querySelector<HTMLOutputElement>('#status')!;
const metrics = document.querySelector<HTMLOutputElement>('#metrics')!;
if (matchMedia('(max-width: 720px), (max-height: 420px)').matches) {
    document.querySelector<HTMLDetailsElement>('#panel')!.open = false;
}
const play = document.querySelector<HTMLButtonElement>('#play')!;
const step = document.querySelector<HTMLButtonElement>('#step')!;
const replay = document.querySelector<HTMLButtonElement>('#replay')!;
function select(id: string) {
    const element = document.querySelector<HTMLSelectElement>(`#${id}`)!;
    const requested = params.get(id);
    if (Array.from(element.options).some(option => option.value === requested)) element.value = requested!;
    return element;
}
const view = select('view');
const pose = select('camera');
const cue = select('cue');
const backend = select('backend'), scene = select('scene'), door = select('door');
const resolve = select('resolve'), ratio = select('ratio'), outputSize = select('outputSize');
const sharpness = document.querySelector<HTMLInputElement>('#sharpness')!;
if (params.has('sharpness')) sharpness.value = params.get('sharpness')!;
if (verify && !params.has('backend')) backend.value = 'three';
door.closest('label')!.hidden = scene.value !== 'door';
cue.closest('label')!.hidden = scene.value === 'door';
if (scene.value === 'door') {
    pose.options[1].text = 'Inspect receiver';
    document.querySelector('#description')!.textContent = 'The shutter closes over a constant amber lamp. Inspect the opposite wall to watch its indirect light fade and return.';
}
const nextFrame = () => new Promise<number>(resolve => requestAnimationFrame(resolve));

async function start() {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('This example requires a WebGPU adapter.');
    const bindings = backend.value === 'three' ? 9 : 8;
    if (adapter.limits.maxStorageBuffersPerShaderStage < bindings) {
        throw new Error(`The cascade tracer needs ${bindings} storage buffers per stage. This adapter supports ${adapter.limits.maxStorageBuffersPerShaderStage}.`);
    }
    const timing = adapter.features.has('timestamp-query');
    const renderer = new THREE.WebGPURenderer({
        antialias: false,
        powerPreference: 'high-performance',
        trackTimestamp: timing,
        requiredLimits: {
            maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxBufferSize: adapter.limits.maxBufferSize,
        },
    });
    document.body.append(renderer.domElement);
    await renderer.init();
    if (!(renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend) throw new Error('A WebGPU renderer is required.');
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    const camera = new THREE.PerspectiveCamera(62, 1, 0.05, 40);
    const chamber = scene.value === 'door' ? makeDoorChamber() : null;
    const world = chamber ?? makePassage();
    const stride = params.get('stride') === '1' ? 1 : 2;
    const options = {
        spacing: 0.10, levelDistance: 1.5, levels: 10, maxProbes: 4096,
        blendingRadius: 2, orientations: false, rayStride: stride, secondary: true,
        qualityDecay: 0.94, quality: 1, adaptiveLOD: false, adaptiveQuality: false, velocity: false,
    };
    const moving = chamber ? [chamber.door.mesh] : [];
    const reference = backend.value === 'three' ? new RealtimeRadianceCascades(options) : null;
    status.textContent = 'Preparing scene and cascade passes…';
    await nextFrame();
    reference?.setScene(renderer,world.scene,moving);
    const rc = reference ?? await createNativeAdapter(renderer,world.scene,stride,moving);
    const reconstruction = createReconstruction(renderer,world.scene,camera,rc);
    const device = (renderer.backend as unknown as RawBackend).device;
    const meter = verify ? receiverMeter(renderer) : null;
    let width = 0, height = 0;
    function resize() {
        const scale = outputSize.value === 'full' ? 1 : Math.min(1,960 / innerWidth,540 / innerHeight);
        width = Math.max(1,Math.floor(innerWidth * scale));
        height = Math.max(1,Math.floor(innerHeight * scale));
        reconstruction.configure({ mode:resolve.value as ReconstructionMode, ratio:Number(ratio.value),
            width,height,sharpness:Number(sharpness.value) });
        sharpness.disabled = ratio.disabled = resolve.value !== 'upscale';
    }
    resize();
    let frame = 0;
    let renderedFrames = 0;
    let playing = !verify;
    let lamp = '';
    let pendingTiming: Promise<void> | null = null;
    let computeMs: number | null = null;
    let queueMs: number | null = null;
    let samplingQueue = false;
    let failed = false;
    let intervals: number[] = [];
    let lastTime = 0;
    let lastHud = 0;

    function timings() {
        if (!timing || pendingTiming) return;
        // Neither query pool may fill. Read the tagged compute group independently of presentation.
        pendingTiming = Promise.all([reference?.readGPUTime(renderer) ?? renderer.resolveTimestampsAsync('compute').then(() => null), renderer.resolveTimestampsAsync('render')])
            .then(([ms]) => { if (ms !== null && Number.isFinite(ms)) computeMs = ms; })
            .catch(() => { computeMs = null; })
            .finally(() => { pendingTiming = null; });
    }
    function draw(readTiming = true) {
        world.pose(camera, frame, pose.value as CameraPose);
        lamp = world.light(frame, cue.value as LampCue);
        chamber?.door.update(frame,door.value as DoorMode);
        rc.mode.value = view.value === 'indirect' ? 1 : 0;
        rc.intensity.value = view.value === 'direct' ? 0 : 1;
        const sample = !verify && !reference && !samplingQueue && device.features.has('timestamp-query') && renderedFrames % 30 === 0
            ? frameTimestamps(device,1) : null;
        sample?.mark(0);
        reconstruction.render();
        if (sample) {
            sample.mark(1);
            samplingQueue = true;
            void sample.finish().then(([value]) => { queueMs = value; }).catch(() => { queueMs = null; })
                .finally(() => { samplingQueue = false; });
        }
        renderedFrames++;
        if (readTiming) timings();
    }
    function hud() {
        play.textContent = playing ? 'Pause' : 'Play';
        const mean = intervals.length ? intervals.reduce((a, b) => a + b, 0) / intervals.length : 0;
        const value = reference ? computeMs : queueMs;
        const ms = value === null ? '—' : `${value.toFixed(2)} ms`;
        status.textContent = `Frame ${frame} / ${REPLAY_FRAMES - 1} · ${lamp} · ${playing ? 'playing' : 'paused'}`;
        metrics.textContent = `${reconstruction.renderWidth}×${reconstruction.renderHeight} → ${width}×${height} · 1 ray / ${stride * stride} px\nProbe storage ${(rc.memory / 2 ** 20).toFixed(1)} MiB\n${reference ? 'RC compute' : 'GPU frame span'} ${ms}\nFrame interval ${mean ? mean.toFixed(2) + ' ms' : '—'}`;
    }
    function advance() { frame = (frame + 1) % REPLAY_FRAMES; }
    function stopClock() { playing = false; lastTime = 0; intervals = []; }
    play.onclick = () => {
        if (playing) stopClock();
        else playing = true;
        hud();
    };
    step.onclick = () => { stopClock(); advance(); draw(); hud(); };
    replay.onclick = () => {
        const url = new URL(location.href);
        for (const control of [view, pose, cue,backend,scene,door,resolve,ratio,sharpness,outputSize]) url.searchParams.set(control.id, control.value);
        history.replaceState(null, '', url);
        // A new addon starts with empty probes, ray sequence and history. Keep inspection settings.
        location.reload();
    };
    scene.onchange = backend.onchange = () => replay.click();
    for (const control of [pose,cue,view,door]) control.onchange = () => { reconstruction.resetHistory(); draw(); hud(); };
    function reconfigure() { resize(); draw(); hud(); }
    for (const control of [resolve,ratio,outputSize]) control.onchange = reconfigure;
    sharpness.oninput = reconfigure;
    addEventListener('resize',reconfigure);
    play.disabled = step.disabled = replay.disabled = false;
    status.textContent = 'Preparing cascade passes…';
    await nextFrame();
    draw();
    hud();

    if (verify) {
        // Only verification requests read pixels. Capture in the same task as the final present.
        const capture = document.createElement('canvas');
        capture.width = 64;
        capture.height = 36;
        const ctx = capture.getContext('2d', { willReadFrequently: true })!;
        async function drainTimers() {
            if (timing) await Promise.all([renderer.resolveTimestampsAsync('compute'), renderer.resolveTimestampsAsync('render')]);
        }
        Object.assign(window, { __rcRailVerify: {
            async render(request: VerifyOptions) {
                stopClock();
                if (!Number.isInteger(request.frames) || request.frames < 1) throw new RangeError('frames must be a positive integer');
                if (request.frame !== undefined) frame = ((Math.trunc(request.frame) % REPLAY_FRAMES) + REPLAY_FRAMES) % REPLAY_FRAMES;
                if ((request.camera && request.camera !== pose.value) || (request.view && request.view !== view.value)
                    || (request.cue && request.cue !== cue.value)) reconstruction.resetHistory();
                if (request.door) door.value = request.door;
                if (request.view) view.value = request.view;
                if (request.camera) pose.value = request.camera;
                if (request.cue) cue.value = request.cue;
                const measured = request.measure === true;
                if (measured && request.frames > 120) throw new RangeError('timestamp windows are limited to 120 frames');
                const frameIntervalsMs: number[] = [];
                const queryIds: string[] = [];
                const peakProbes = reference?.capacities.map(() => 0) ?? [];
                const linearReceiver = [];
                const gpu = measured && device.features.has('timestamp-query') ? frameTimestamps(device,request.frames) : null;
                let previousTick: number | null = null;
                let firstTick: number | null = null;
                if (measured) {
                    // Start with empty pools, including queries queued during asynchronous warm-up reads.
                    await pendingTiming;
                    await drainTimers();
                }
                for (let i = 0; i < request.frames; i++) {
                    const tick = await nextFrame();
                    firstTick ??= tick;
                    if (measured && previousTick !== null) frameIntervalsMs.push(tick - previousTick);
                    previousTick = tick;
                    if (request.advance !== false && (request.frame === undefined || i > 0)) advance();
                    gpu?.mark(2*i);
                    draw(!measured);
                    gpu?.mark(2*i+1);
                    if (measured && timing && reference) {
                        // Same identifier used by the pinned addon's readGPUTime(). Resolve all frames in one batch.
                        const uid = (rc as unknown as { _timestampUID: string | null })._timestampUID;
                        if (uid !== null) queryIds.push(uid);
                    }
                    if (i === request.frames - 1) {
                        const w = request.captureSize === 'display' ? width : 64;
                        const h = request.captureSize === 'display' ? height : 36;
                        if (capture.width !== w) capture.width = w;
                        if (capture.height !== h) capture.height = h;
                        ctx.drawImage(renderer.domElement, 0, 0, w, h);
                    }
                    if (request.linearReceiver && chamber && meter) {
                        linearReceiver.push({ luma: await meter.sample(rc.getCompositeNode()), elapsedMs: tick - firstTick, doorOpenness: chamber.door.openness });
                    }
                    if (request.occupancy && reference) {
                        const counts = await reference.readStats(renderer);
                        for (let n = 0; n < counts.length; n++) peakProbes[n] = Math.max(peakProbes[n], counts[n]);
                    }
                }
                const gpuFrameMs = gpu ? await gpu.finish() : [];
                const gpuComputeMs: number[] = [];
                if (measured) {
                    await drainTimers();
                    const pool = (renderer.backend as unknown as {
                        timestampQueryPool: { compute?: { timestamps: Map<string, number> } };
                    }).timestampQueryPool.compute;
                    for (const uid of new Set(queryIds)) {
                        const value = pool?.timestamps.get(uid);
                        if (value !== undefined && Number.isFinite(value)) gpuComputeMs.push(value);
                    }
                }
                hud();
                return {
                    backend: reference ? 'three' : 'interval', linearReceiver,
                    reconstruction: { mode:resolve.value,ready:reconstruction.ready,displayWidth:width,displayHeight:height,
                        renderWidth:reconstruction.renderWidth,renderHeight:reconstruction.renderHeight,updates:reconstruction.updates,
                        historyResets:reconstruction.historyResets,jitterX:reconstruction.jitterX,jitterY:reconstruction.jitterY },
                    pixels: ctx.getImageData(0, 0, capture.width, capture.height).data, width:capture.width, height:capture.height, frame, renderedFrames, computeMs,
                    occupancy: request.occupancy && reference ? { peakProbes, capacities: reference.capacities, probeBytes: rc.memory } : undefined,
                    measurement: measured ? {
                        frameIntervalsMs, gpuFrameMs, gpuComputeMs, timestampSupported: timing, renderWidth: reconstruction.renderWidth, renderHeight: reconstruction.renderHeight,
                        stride, probeBytes: rc.memory, hitScratchBytes: rc.hitScratchBytes,
                    } : undefined,
                };
            },
        } });
    }
    renderer.setAnimationLoop((now: number) => {
        if (!playing || failed) return;
        try {
            advance();
            draw();
            if (lastTime > 0) {
                intervals.push(now - lastTime);
                if (intervals.length > 60) intervals.shift();
            }
            lastTime = now;
            if (now - lastHud > 250) { lastHud = now; hud(); }
        } catch (error) {
            failed = true;
            fail(error);
        }
    });
    addEventListener('pagehide', (event: PageTransitionEvent) => {
        if (event.persisted) return;
        renderer.setAnimationLoop(null);
        rc.dispose();
        reconstruction.dispose();
        meter?.dispose();
        renderer.dispose();
    });
}

function fail(error: unknown) {
    status.textContent = error instanceof Error ? error.message : String(error);
    play.disabled = step.disabled = replay.disabled = true;
    console.error(error);
}
void start().catch(fail);
