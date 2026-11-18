import * as THREE from 'three/webgpu';
import { RealtimeRadianceCascades } from './RealtimeRadianceCascades.js';
import { makePassage, REPLAY_FRAMES, type CameraPose, type LampCue } from './scene.ts';

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
const nextFrame = () => new Promise<number>(resolve => requestAnimationFrame(resolve));

async function start() {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('This example requires a WebGPU adapter.');
    if (adapter.limits.maxStorageBuffersPerShaderStage < 9) {
        throw new Error(`The cascade tracer needs 9 storage buffers per stage. This adapter supports ${adapter.limits.maxStorageBuffersPerShaderStage}.`);
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
    const world = makePassage();
    const stride = params.get('stride') === '1' ? 1 : 2;
    const options = {
        spacing: 0.10, levelDistance: 1.5, levels: 10, maxProbes: 4096,
        blendingRadius: 2, orientations: false, rayStride: stride, secondary: true,
        qualityDecay: 0.94, quality: 1, adaptiveLOD: false, adaptiveQuality: false, velocity: false,
    };
    const rc = new RealtimeRadianceCascades(options);
    let width = 0;
    let height = 0;
    function resize() {
        const scale = Math.min(1, 960 / innerWidth, 540 / innerHeight);
        width = Math.max(1, Math.floor(innerWidth * scale));
        height = Math.max(1, Math.floor(innerHeight * scale));
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
    }
    resize();
    addEventListener('resize', resize);
    status.textContent = 'Building the gallery BVH…';
    await nextFrame();
    rc.setScene(renderer, world.scene);
    const pipeline = new THREE.RenderPipeline(renderer);
    let lastComposite: THREE.Node | null = null;
    let frame = 0;
    let renderedFrames = 0;
    let playing = !verify;
    let lamp = '';
    let pendingTiming: Promise<void> | null = null;
    let computeMs: number | null = null;
    let failed = false;
    let intervals: number[] = [];
    let lastTime = 0;
    let lastHud = 0;

    function timings() {
        if (!timing || pendingTiming) return;
        // Neither query pool may fill. Read the tagged compute group independently of presentation.
        pendingTiming = Promise.all([rc.readGPUTime(renderer), renderer.resolveTimestampsAsync('render')])
            .then(([ms]) => { if (ms !== null && Number.isFinite(ms)) computeMs = ms; })
            .catch(() => { computeMs = null; })
            .finally(() => { pendingTiming = null; });
    }
    function display() {
        rc.mode.value = view.value === 'indirect' ? 1 : 0;
        rc.intensity.value = view.value === 'direct' ? 0 : 1;
        const composite = rc.getCompositeNode();
        if (composite !== lastComposite) {
            lastComposite = composite;
            pipeline.outputNode = composite;
            pipeline.needsUpdate = true;
        }
        pipeline.render();
    }
    function draw(readTiming = true) {
        world.pose(camera, frame, pose.value as CameraPose);
        lamp = world.light(frame, cue.value as LampCue);
        rc.update(renderer, world.scene, camera);
        display();
        renderedFrames++;
        if (readTiming) timings();
    }
    function hud() {
        play.textContent = playing ? 'Pause' : 'Play';
        const mean = intervals.length ? intervals.reduce((a, b) => a + b, 0) / intervals.length : 0;
        const ms = computeMs === null ? 'unavailable' : `${computeMs.toFixed(2)} ms`;
        status.textContent = `Frame ${frame} / ${REPLAY_FRAMES - 1} · ${lamp} · ${playing ? 'playing' : 'paused'}`;
        metrics.textContent = `${width}×${height} · 1 ray / ${stride * stride} px\nProbe storage ${(rc.memory / 2 ** 20).toFixed(1)} MiB\nRC compute ${ms}\nFrame interval ${mean ? mean.toFixed(2) + ' ms' : '—'}`;
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
        for (const control of [view, pose, cue]) url.searchParams.set(control.id, control.value);
        history.replaceState(null, '', url);
        // A new addon starts with empty probes, ray sequence and history. Keep inspection settings.
        location.reload();
    };
    for (const control of [pose, cue]) control.onchange = () => { draw(); hud(); };
    view.onchange = () => { display(); hud(); };
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
                if (request.view) view.value = request.view;
                if (request.camera) pose.value = request.camera;
                if (request.cue) cue.value = request.cue;
                const measured = request.measure === true;
                if (measured && request.frames > 120) throw new RangeError('timestamp windows are limited to 120 frames');
                const frameIntervalsMs: number[] = [];
                const queryIds: string[] = [];
                const peakProbes = rc.capacities.map(() => 0);
                let previousTick: number | null = null;
                if (measured) {
                    // Start with empty pools, including queries queued during asynchronous warm-up reads.
                    await pendingTiming;
                    await drainTimers();
                }
                for (let i = 0; i < request.frames; i++) {
                    const tick = await nextFrame();
                    if (measured && previousTick !== null) frameIntervalsMs.push(tick - previousTick);
                    previousTick = tick;
                    if (request.advance !== false && (request.frame === undefined || i > 0)) advance();
                    draw(!measured);
                    if (measured && timing) {
                        // Same identifier used by the pinned addon's readGPUTime(). Resolve all frames in one batch.
                        const uid = (rc as unknown as { _timestampUID: string | null })._timestampUID;
                        if (uid !== null) queryIds.push(uid);
                    }
                    if (i === request.frames - 1) ctx.drawImage(renderer.domElement, 0, 0, 64, 36);
                    if (request.occupancy) {
                        const counts = await rc.readStats(renderer);
                        for (let n = 0; n < counts.length; n++) peakProbes[n] = Math.max(peakProbes[n], counts[n]);
                    }
                }
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
                    pixels: ctx.getImageData(0, 0, 64, 36).data, width: 64, height: 36, frame, renderedFrames, computeMs,
                    occupancy: request.occupancy ? { peakProbes, capacities: rc.capacities, probeBytes: rc.memory } : undefined,
                    measurement: measured ? {
                        frameIntervalsMs, gpuComputeMs, timestampSupported: timing, renderWidth: width, renderHeight: height,
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
        pipeline.dispose();
        renderer.dispose();
    });
}

function fail(error: unknown) {
    status.textContent = error instanceof Error ? error.message : String(error);
    play.disabled = step.disabled = replay.disabled = true;
    console.error(error);
}
void start().catch(fail);
