/**
 * RC3.6: more than two bounces in the radiance cache's world cache, the two new
 * settings checked against analytic answers (correctness gates, expected
 * values written before the run). The comparison of every way of getting the
 * later bounces (accuracy against Cycles, lag, ghosting, flicker, memory) is
 * `radiance-cache-accuracy` and `radiance-cache-bounces-change`.
 *
 * **The furnace** (as radiance-cache-ddgi-bounce's): a floor (y = 0, facing
 * up) and a ceiling (y = 3, facing down), 400 m wide, albedo ρ = 0.5, a black
 * sky, a sun straight down (E = 4) that casts no shadow, so it lights the
 * floor and never the ceiling's underside. A ceiling cell has no direct light,
 * so a probe ray that hits the ceiling reads the bounces after the first:
 * GI-1.2's second bounce ρ L_f1 = ρ² E / π = 0.3183, or every bounce
 * ρ L_f1 / (1 − ρ²) = 0.4244 (L_f1 = ρ E / π, the floor's direct light). The
 * rays judged hit the ceiling within 8 m of the camera (tiles under 2 m).
 *
 * 1. **GI-1.2's rays** (`cacheFeedback` off, the baseline), camera looking
 *    down at the floor: the near ceiling rays read the second bounce, 0.3183,
 *    within 2% (radiance-cache-ddgi-bounce's case 1).
 * 2. **Cache feedback** (`cacheFeedback: true`, register E12): the same rays
 *    read every bounce, 0.4244, within 2%. The fixed point of a second hit
 *    bringing its cell's direct and indirect value: I_c = ρ (L_f1 + I_f),
 *    I_f = ρ I_c, so I_c = ρ L_f1 / (1 − ρ²). Reading the indirect value twice
 *    converges to ρ L_f1 / (1 − 2ρ²) = 0.6366 and not at all to 0.3183.
 * 3. **Strips** (half the floor gone, as radiance-cache-ddgi-bounce's 4-6),
 *    cache feedback: every bounce ρ f L_f1 / (1 − ρ² f) = 0.1819. Recorded,
 *    not gated, as GI-1.2's strips are (the lead, 2026-11-23: a far strip-floor
 *    cell sharing a hash column with the ceiling counts the ceiling's unlit
 *    hits, register D63, so rays landing there bring less: 0.974 × with
 *    GI-1.2's rays).
 * 4. **Temporal feedback, hits off last frame's screen** (`useTemporalFeedback:
 *    true`, register D68), the camera looking down: the ceiling is never on
 *    the screen (the view's top edge is 15° below level), so no probe ray
 *    takes last frame's image and the near ceiling rays read GI-1.2's second
 *    bounce, 0.3183, within 2%. A lookup that read a texel for a hit outside
 *    the screen would read the lit floor there.
 * 5. **Temporal feedback, hits on last frame's screen**, the camera at
 *    y = 1.5 looking level along −z (60° vertical field: the ceiling fills
 *    the top of the picture, the floor the bottom): the near ceiling rays
 *    whose hit last frame's camera saw well inside its screen (5% margin)
 *    read last frame's image there, the ceiling's lit colour (no direct light,
 *    albedo / π × E). With the floor and ceiling both on screen that image
 *    carries more bounces than the cache's two, and never more than every
 *    bounce: their mean lies between 1.05 × 0.3183 and 1.02 × 0.4244, and no
 *    near ceiling ray's mean (on screen or not) is over 1.02 × 0.4244.
 *    Capsaicin's way, adding the lit value into the cell as direct light while
 *    the cell's indirect value is added again, reads more than every bounce.
 *    Before measuring, the run checks the frame's probe rays: some near
 *    ceiling hits are on screen and some aren't.
 * 6-7. **Temporal feedback, hits last frame's camera didn't see** (the plane
 *    test): the level view with a black slab (albedo 0) 0.5 m under the
 *    ceiling over z from −8 to −4, facing down like the ceiling. From the
 *    camera the slab hides the ceiling from z = −8.5 to −12; probe rays from
 *    the floor there still reach it. Those hidden ceiling hits (|x| < 3) land
 *    on screen where the slab's underside shows, with the same normal and
 *    last frame's lit image black; only the plane test (the shown point 0.5 m
 *    off the hit's plane) keeps them on their cells. Section 6 runs GI-1.2's
 *    rays (no feedback), section 7 temporal feedback: the hidden rays' mean
 *    within 5% of section 6's (both read the cells, whose samples the
 *    feedback doesn't change). Reading the slab's black there reads about 0.
 *
 * Every section: no NaN, infinite or negative ray, and no device errors. The
 * frames: 600 from an emptied cache (each bounce takes a frame and the cells'
 * 16-sample means; the screen loop takes the probes' 8-sample means a turn).
 *
 * Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-bounces --gpu nvidia`.
 */

import type { LightDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import type { Vec3 } from '../../../../packages/ddgi-core/src/platform/contract/math.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { halfBitsToFloat, unpackHalf3 } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import type { RadianceCacheSettings } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { WORLD_CACHE_PROBE_RAY_BYTES, WORLD_CACHE_PROBE_RAYS_HEADER_BYTES } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/world-cache.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { PAGE_COMPARE_VALUES } from '../radiance-cache-probes/compare-env.ts';
import { setCompareValues } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/compare-toggles.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
const FRAMES = 600;

const FURNACE = { halfWidth: 200, ceilingY: 3, albedo: 0.5, sunIntensity: 4 } as const;
const STRIPS = { width: 0.05, period: 0.1, halfLength: 100, fill: 0.5 } as const;
/** Looking down at the floor: the ceiling is never on the screen. */
const DOWN_CAMERA: PageCamera = { position: [0, 2, 0], target: [0, 0, -2], fovYDeg: 60, near: 0.05, far: 1000 };
/** Looking level from halfway up: the ceiling fills the top of the picture, the floor the bottom. */
const LEVEL_CAMERA: PageCamera = { position: [0, 1.5, 0], target: [0, 1.5, -10], fovYDeg: 60, near: 0.05, far: 1000 };
/** Rays judged: their hit on the ceiling within this distance of the camera (tiles under 2 m). */
const NEAR_HIT_METRES = 8;
/** An on-screen hit: inside last frame's screen by this share of it on every side. */
const SCREEN_MARGIN = 0.05;
/** Sections 6-7: the black slab under the ceiling (y, z range), and the ceiling hits it hides from the level camera. */
const SLAB = { y: 2.5, zNear: -4, zFar: -8, halfWidth: 50 } as const;
const HIDDEN = { zNear: -8.5, zFar: -12, halfWidth: 3 } as const;
/** Section 7: the hidden rays' mean within this of section 6's. */
const HIDDEN_TOLERANCE = 0.05;

const FLOOR_DIRECT = (FURNACE.albedo * FURNACE.sunIntensity) / Math.PI;
const EXPECTED = {
    /** ρ L_f1 = ρ² E / π: GI-1.2's second bounce. */
    secondBounce: FURNACE.albedo * FLOOR_DIRECT,
    /** ρ L_f1 / (1 − ρ²): every bounce after the first. */
    everyBounce: (FURNACE.albedo * FLOOR_DIRECT) / (1 - FURNACE.albedo ** 2),
    /** ρ f L_f1 / (1 − ρ² f): the strips' every bounce. */
    stripsEveryBounce: (FURNACE.albedo * STRIPS.fill * FLOOR_DIRECT) / (1 - FURNACE.albedo ** 2 * STRIPS.fill),
} as const;
const TOLERANCE = 0.02;
/** Section 5: the on-screen rays' mean at least this much over the second bounce (feedback took hold). */
const FEEDBACK_GAIN = 1.05;

const FURNACE_SUN: LightDescriptor = { type: 'directional', direction: [0, -1, 0], color: [1, 1, 1], intensity: FURNACE.sunIntensity, castsShadows: false, label: 'furnace sun' };
const HALF_TABLE = Float32Array.from({ length: 65536 }, (_, bits) => halfBitsToFloat(bits));
const round = (value: number, digits = 4): number => Number(value.toFixed(digits));

interface Scenario {
    readonly technique: RadianceCacheTechnique;
    readonly camera: PageCamera;
    frame(): void;
    dispose(): void;
}

function quad(platform: GIPlatformImpl, label: string, corners: readonly Vec3[], normal: Vec3, albedo: number = FURNACE.albedo): void {
    const scene = platform.scene;
    const mesh = scene.createMesh({ positions: new Float32Array(corners.flat()), normals: new Float32Array([...normal, ...normal, ...normal, ...normal]), indices: new Uint32Array([0, 1, 2, 0, 2, 3]), label });
    const material = scene.createMaterial({ albedo: [albedo, albedo, albedo], label });
    scene.createInstance({ mesh, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], label });
}

function stripFloor(platform: GIPlatformImpl): void {
    const { width, period, halfLength } = STRIPS;
    const count = Math.round((2 * halfLength) / period);
    const positions: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];
    for (let strip = 0; strip < count; strip++) {
        const z0 = -halfLength + strip * period;
        const z1 = z0 + width;
        const base = strip * 4;
        positions.push(-halfLength, 0, z1, halfLength, 0, z1, halfLength, 0, z0, -halfLength, 0, z0);
        normals.push(0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0);
        indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
    const scene = platform.scene;
    const mesh = scene.createMesh({ positions: new Float32Array(positions), normals: new Float32Array(normals), indices: new Uint32Array(indices), label: 'strip floor' });
    const material = scene.createMaterial({ albedo: [FURNACE.albedo, FURNACE.albedo, FURNACE.albedo], label: 'strip floor' });
    scene.createInstance({ mesh, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], label: 'strip floor' });
}

async function startScenario(device: GPUDevice, name: string, settings: Partial<RadianceCacheSettings>, camera: PageCamera, strips: boolean, slab: boolean): Promise<Scenario> {
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-bounces ${name}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique(settings));
    // VITE_RC_COMPARE (RC4.5) under the section's own settings.
    setCompareValues(technique.settings, PAGE_COMPARE_VALUES);
    Object.assign(technique.settings, settings);
    platform.setOutput(technique);
    await platform.prepare();
    const half = FURNACE.halfWidth;
    if (strips) stripFloor(platform);
    else quad(platform, 'floor', [[-half, 0, half], [half, 0, half], [half, 0, -half], [-half, 0, -half]], [0, 1, 0]);
    quad(platform, 'ceiling', [[-half, FURNACE.ceilingY, -half], [half, FURNACE.ceilingY, -half], [half, FURNACE.ceilingY, half], [-half, FURNACE.ceilingY, half]], [0, -1, 0]);
    if (slab) {
        // Black on both sides: its underside faces down like the ceiling, its top faces up.
        const w = SLAB.halfWidth;
        quad(platform, 'slab underside', [[-w, SLAB.y, SLAB.zFar], [w, SLAB.y, SLAB.zFar], [w, SLAB.y, SLAB.zNear], [-w, SLAB.y, SLAB.zNear]], [0, -1, 0], 0);
        quad(platform, 'slab top', [[-w, SLAB.y, SLAB.zNear], [w, SLAB.y, SLAB.zNear], [w, SLAB.y, SLAB.zFar], [-w, SLAB.y, SLAB.zFar]], [0, 1, 0], 0);
    }
    platform.scene.createLight(FURNACE_SUN);
    platform.scene.setEnvironment({ type: 'color', color: [0, 0, 0], intensity: 1 });
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const frame = (): void => {
        const encoder = device.createCommandEncoder({ label: `radiance-cache-bounces ${name}` });
        platform.encode(encoder, { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) });
        screen.encode(encoder, camera);
        platform.encodeScreen(encoder, screen.screen);
        device.queue.submit([encoder.finish()]);
    };
    frame();
    await platform.rays.whenBuilt();
    frame();
    await device.queue.onSubmittedWorkDone();
    technique.resetHistory();
    return {
        technique,
        camera,
        frame,
        dispose() {
            screen.dispose();
            platform.dispose();
        },
    };
}

async function readRange(device: GPUDevice, source: GPUBuffer, offset: number, byteLength: number): Promise<ArrayBuffer> {
    const size = Math.max(4, Math.ceil(byteLength / 4) * 4);
    const staging = device.createBuffer({ label: 'radiance-cache-bounces readback', size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder({ label: 'radiance-cache-bounces readback' });
    encoder.copyBufferToBuffer(source, offset, staging, 0, size);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return copy;
}

/** Where `point` lands on `camera`'s screen (0..1, y down), or null behind it. */
function screenUv(camera: PageCamera, point: Vec3): [number, number] | null {
    const { view, projection } = frameCamera(camera, WIDTH, HEIGHT);
    const apply = (m: ArrayLike<number>, p: readonly number[]) => [0, 1, 2, 3].map((row) => m[row] * p[0] + m[4 + row] * p[1] + m[8 + row] * p[2] + m[12 + row] * p[3]);
    const clip = apply(projection, apply(view, [...point, 1]));
    if (clip[3] <= 0) return null;
    return [0.5 * (clip[0] / clip[3]) + 0.5, 0.5 - 0.5 * (clip[1] / clip[3])];
}

interface RayReading {
    /** Near ceiling rays: mean (channel mean) and count, all, on last frame's screen, and off it. */
    readonly near: { readonly mean: number; readonly rays: number };
    readonly onScreen: { readonly mean: number; readonly rays: number };
    readonly offScreen: { readonly mean: number; readonly rays: number };
    /** Sections 6-7: ceiling hits the slab hides from the camera (|x| < 3, z from −8.5 to −12), any distance. */
    readonly hidden: { readonly mean: number; readonly rays: number };
    readonly bad: number;
}

async function readRays(device: GPUDevice, scenario: Scenario): Promise<RayReading> {
    const probes = scenario.technique.probes;
    if (!probes) throw new Error('no probes after encodeScreen');
    const rayCount = new Uint32Array(await readRange(device, probes.probeRays, 0, 4))[0];
    const rayWords = new Uint32Array(await readRange(device, probes.probeRays, 0, WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + rayCount * WORLD_CACHE_PROBE_RAY_BYTES));
    const rayFloats = new Float32Array(rayWords.buffer);
    const results = probes.rayScratch.rayResults;
    const radianceWords = new Uint32Array(await readRange(device, results.buffer, results.offset ?? 0, rayCount * 8));
    const eye = scenario.camera.position;
    const sums = { near: [0, 0], onScreen: [0, 0], offScreen: [0, 0], hidden: [0, 0] };
    let bad = 0;
    for (let ray = 0; ray < rayCount; ray++) {
        const base = (WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + ray * WORLD_CACHE_PROBE_RAY_BYTES) / 4;
        const hitT = rayFloats[base + 6];
        if (!(hitT >= 0)) continue;
        const direction = unpackHalf3([rayWords[base + 4], rayWords[base + 5]]);
        const hit: Vec3 = [rayFloats[base] + hitT * direction[0], rayFloats[base + 1] + hitT * direction[1], rayFloats[base + 2] + hitT * direction[2]];
        const value = (HALF_TABLE[radianceWords[ray * 2] & 0xffff] + HALF_TABLE[radianceWords[ray * 2] >>> 16] + HALF_TABLE[radianceWords[ray * 2 + 1] & 0xffff]) / 3;
        if (!(Number.isFinite(value) && value >= 0)) {
            bad++;
            continue;
        }
        const onCeiling = Math.abs(hit[1] - FURNACE.ceilingY) <= 1e-3;
        if (onCeiling && Math.abs(hit[0]) < HIDDEN.halfWidth && hit[2] < HIDDEN.zNear && hit[2] > HIDDEN.zFar) {
            sums.hidden[0] += value;
            sums.hidden[1]++;
        }
        if (!onCeiling || Math.hypot(hit[0] - eye[0], hit[2] - eye[2]) > NEAR_HIT_METRES) continue;
        sums.near[0] += value;
        sums.near[1]++;
        const uv = screenUv(scenario.camera, hit);
        const inside = uv !== null && uv.every((coordinate) => coordinate > SCREEN_MARGIN && coordinate < 1 - SCREEN_MARGIN);
        const outside = uv === null || uv.some((coordinate) => coordinate < 0 || coordinate > 1);
        if (inside) {
            sums.onScreen[0] += value;
            sums.onScreen[1]++;
        } else if (outside) {
            sums.offScreen[0] += value;
            sums.offScreen[1]++;
        }
    }
    const tally = ([sum, rays]: number[]) => ({ mean: sum / Math.max(rays, 1), rays });
    return { near: tally(sums.near), onScreen: tally(sums.onScreen), offScreen: tally(sums.offScreen), hidden: tally(sums.hidden), bad };
}

async function runFrames(device: GPUDevice, scenario: Scenario, frames: number): Promise<void> {
    for (let index = 0; index < frames; index++) {
        scenario.frame();
        if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
    }
    await device.queue.onSubmittedWorkDone();
}

interface Section {
    readonly name: string;
    readonly settings: Partial<RadianceCacheSettings>;
    readonly camera: PageCamera;
    readonly strips: boolean;
    /** Sections 6-7: the black slab under the ceiling. */
    readonly slab?: boolean;
    /** Checks the reading (with the earlier sections' readings by number); returns the section's numbers. */
    judge(reading: RayReading, failures: string[], earlier: ReadonlyMap<string, RayReading>): Record<string, unknown>;
}

const within = (value: number, expected: number) => Math.abs(value / expected - 1) <= TOLERANCE;

const SECTIONS: readonly Section[] = [
    {
        name: "1 furnace, GI-1.2's rays (second bounce)",
        settings: { secondBounce: 'gi1-rays', cacheFeedback: false },
        camera: DOWN_CAMERA,
        strips: false,
        judge(reading, failures) {
            if (!within(reading.near.mean, EXPECTED.secondBounce)) failures.push(`${this.name}: near ceiling rays read ${round(reading.near.mean)}, expected ${round(EXPECTED.secondBounce)} within 2%`);
            return { expected: round(EXPECTED.secondBounce), ratio: round(reading.near.mean / EXPECTED.secondBounce) };
        },
    },
    {
        name: '2 furnace, cache feedback (every bounce)',
        settings: { secondBounce: 'gi1-rays', cacheFeedback: true },
        camera: DOWN_CAMERA,
        strips: false,
        judge(reading, failures) {
            if (!within(reading.near.mean, EXPECTED.everyBounce)) failures.push(`${this.name}: near ceiling rays read ${round(reading.near.mean)}, expected every bounce ${round(EXPECTED.everyBounce)} within 2%`);
            return { expected: round(EXPECTED.everyBounce), ratio: round(reading.near.mean / EXPECTED.everyBounce) };
        },
    },
    {
        name: '3 strips, cache feedback (every bounce; recorded)',
        settings: { secondBounce: 'gi1-rays', cacheFeedback: true },
        camera: DOWN_CAMERA,
        strips: true,
        judge(reading) {
            return { expected: round(EXPECTED.stripsEveryBounce), ratio: round(reading.near.mean / EXPECTED.stripsEveryBounce), gated: false };
        },
    },
    {
        name: "4 furnace, temporal feedback, hits off last frame's screen (second bounce)",
        settings: { secondBounce: 'gi1-rays', useTemporalFeedback: true },
        camera: DOWN_CAMERA,
        strips: false,
        judge(reading, failures) {
            if (reading.onScreen.rays > 0) failures.push(`${this.name}: ${reading.onScreen.rays} near ceiling hits are on the screen; the section needs none`);
            if (!within(reading.near.mean, EXPECTED.secondBounce)) failures.push(`${this.name}: near ceiling rays read ${round(reading.near.mean)}, expected ${round(EXPECTED.secondBounce)} within 2% (no hit was on screen)`);
            return { expected: round(EXPECTED.secondBounce), ratio: round(reading.near.mean / EXPECTED.secondBounce) };
        },
    },
    {
        name: "5 furnace, temporal feedback, hits on last frame's screen (between two and every bounce)",
        settings: { secondBounce: 'gi1-rays', useTemporalFeedback: true },
        camera: LEVEL_CAMERA,
        strips: false,
        judge(reading, failures) {
            if (reading.onScreen.rays < 1000 || reading.offScreen.rays < 100) failures.push(`${this.name}: ${reading.onScreen.rays} near ceiling hits on screen and ${reading.offScreen.rays} off it; the section needs both`);
            const low = FEEDBACK_GAIN * EXPECTED.secondBounce;
            const high = (1 + TOLERANCE) * EXPECTED.everyBounce;
            if (!(reading.onScreen.mean >= low && reading.onScreen.mean <= high)) {
                failures.push(`${this.name}: the on-screen near ceiling rays read ${round(reading.onScreen.mean)}, expected between ${round(low)} (more than two bounces) and ${round(high)} (every bounce)`);
            }
            if (reading.near.mean > high) failures.push(`${this.name}: the near ceiling rays read ${round(reading.near.mean)}, over every bounce ${round(EXPECTED.everyBounce)} (+2%)`);
            return {
                onScreenOverSecondBounce: round(reading.onScreen.mean / EXPECTED.secondBounce),
                onScreenOverEveryBounce: round(reading.onScreen.mean / EXPECTED.everyBounce),
                offScreenOverSecondBounce: round(reading.offScreen.mean / EXPECTED.secondBounce),
                onScreenRays: reading.onScreen.rays,
                offScreenRays: reading.offScreen.rays,
            };
        },
    },
    {
        name: "6 furnace with a slab, GI-1.2's rays (the control for 7)",
        settings: { secondBounce: 'gi1-rays', useTemporalFeedback: false },
        camera: LEVEL_CAMERA,
        strips: false,
        slab: true,
        judge(reading, failures) {
            if (reading.hidden.rays < 1000) failures.push(`${this.name}: only ${reading.hidden.rays} ceiling hits hidden behind the slab`);
            return { hiddenMean: round(reading.hidden.mean), hiddenRays: reading.hidden.rays };
        },
    },
    {
        name: "7 furnace with a slab, temporal feedback, hits last frame's camera didn't see",
        settings: { secondBounce: 'gi1-rays', useTemporalFeedback: true },
        camera: LEVEL_CAMERA,
        strips: false,
        slab: true,
        judge(reading, failures, earlier) {
            const control = earlier.get('6');
            if (!control) {
                failures.push(`${this.name}: needs section 6's reading`);
                return {};
            }
            if (reading.hidden.rays < 1000) failures.push(`${this.name}: only ${reading.hidden.rays} ceiling hits hidden behind the slab`);
            const ratio = reading.hidden.mean / Math.max(control.hidden.mean, 1e-12);
            if (Math.abs(ratio - 1) > HIDDEN_TOLERANCE) {
                failures.push(`${this.name}: the hidden ceiling rays read ${round(reading.hidden.mean)}, ${round(ratio)} x section 6's ${round(control.hidden.mean)} (within ${HIDDEN_TOLERANCE * 100}%: the slab's black taken for the ceiling's light?)`);
            }
            return { hiddenMean: round(reading.hidden.mean), hiddenRays: reading.hidden.rays, overControl: round(ratio) };
        },
    },
];

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    const device = await adapter.requestDevice({ requiredLimits: requiredLimits(adapter), requiredFeatures: optionalFeatures(adapter) });
    releaseDeviceOnPageHide(device);
    deviceErrors = watchDevice(device);
    const failures: string[] = [];
    const sections: Record<string, unknown> = {};
    const only = String((import.meta.env as Record<string, unknown>).VITE_RC_BOUNCES_SECTIONS ?? '');
    const readings = new Map<string, RayReading>();
    for (const section of SECTIONS) {
        const number = section.name.split(' ')[0];
        if (only && !only.split(',').includes(number)) continue;
        const scenario = await startScenario(device, section.name, section.settings, section.camera, section.strips, section.slab ?? false);
        try {
            await runFrames(device, scenario, FRAMES);
            const reading = await readRays(device, scenario);
            if (reading.bad > 0) failures.push(`${section.name}: ${reading.bad} rays NaN, infinite or negative`);
            if (reading.near.rays < 1000) failures.push(`${section.name}: only ${reading.near.rays} near ceiling rays`);
            sections[section.name] = { nearMean: round(reading.near.mean), nearRays: reading.near.rays, ...section.judge(reading, failures, readings) };
            readings.set(number, reading);
            console.log(section.name, JSON.stringify(sections[section.name]));
        } finally {
            scenario.dispose();
            await device.queue.onSubmittedWorkDone();
        }
    }
    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 && deviceErrors.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data: { width: WIDTH, height: HEIGHT, frames: FRAMES, compare: PAGE_COMPARE_VALUES, expected: Object.fromEntries(Object.entries(EXPECTED).map(([key, value]) => [key, round(value)])), sections },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
