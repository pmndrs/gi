/**
 * Shared by the T2.4c pages (key probes and propagation): Sponza started as
 * the viewer starts it, with its sun, bookmark 1's view and a lamp to switch
 * on; frames at a fixed batch; converging a field; the sun aimed and turned;
 * the probes' mean irradiance read back; and the field's errors against a
 * converged reference. The same procedures as T2.4b's `ddgi-schedule-sponza`
 * (which keeps its own copy), so the numbers compare.
 *
 * Not a page itself (no index.html).
 */

import { PointLight, Vector3, type DirectionalLight } from 'three/webgpu';
import { assert } from '../../client.ts';
import { cameraWeight, frustumPlanes } from '../../../../packages/ddgi/src/camera.ts';
import { isKeyProbe, PROBE_HISTORY_FLAG } from '../../../../packages/ddgi/src/layout.ts';
import { NEAR_CAMERA_WEIGHT } from '../../../../packages/ddgi/src/settings.ts';
import type { DDGIField } from '../../../../packages/ddgi/src/ddgi-technique.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { imageMean, readProbeData, TestView, type TestViewImages } from './harness.ts';
import { applyHysteresisMode, FIXED_097, parseHysteresisMode } from './hysteresis-modes.ts';
import { probeCoordinates, readHistory, type ProbeHistoryRecord } from './response.ts';
import { startViewerScene, yieldToCallbacks, type ViewerScene } from './viewer-scene.ts';

export type Vec3 = [number, number, number];

/** The test view's size (as T2.4b's Sponza page). */
export const VIEW_WIDTH = 320;
export const VIEW_HEIGHT = 180;

/** Converging (as T2.4b's Sponza page): fast frames at a fast fixed hysteresis, then blocks at 0.97 until the view settles. */
const CONVERGE_UNITS = 4096;
const FAST_FRAMES = 160;
const FAST_HYSTERESIS = 0.5;
const BLOCK_FRAMES = 40;
const MAX_BLOCKS = 16;
const CONVERGED_TOLERANCE = 0.002;
/** Frames of the default rule at the batch after converging, so every probe's history is at its steady count. */
export const SETTLE_FRAMES = 1200;

export interface SponzaUnderTest {
    readonly scene: ViewerScene;
    readonly view: TestView;
    readonly sun: DirectionalLight;
    readonly lamp: PointLight;
    /** The view's GI-only mean at bookmark 1. */
    readonly mean: () => Promise<number>;
    /** The view's images at bookmark 1. */
    readonly images: () => Promise<TestViewImages>;
}

export async function startSponza(device: GPUDevice, label: string, lampPosition: Vec3 = [3, 1.5, 0.5], sceneId = 'sponza'): Promise<SponzaUnderTest> {
    const manifest = loadAllManifests().find((entry) => entry.id === sceneId);
    assert(manifest, `no ${sceneId} manifest`);
    let lamp: PointLight | null = null;
    const scene = await startViewerScene(device, manifest, {
        gpuTiming: true,
        label,
        prepareScene: (loaded) => {
            // Off until a page switches it on; no range (three's distance 0).
            lamp = new PointLight(0xffffff, 0, 0, 2);
            lamp.position.set(...lampPosition);
            loaded.scene.add(lamp);
        },
    });
    assert(lamp, 'no lamp');
    const sun = scene.loaded.directLights.find((light) => (light as DirectionalLight).isDirectionalLight) as DirectionalLight | undefined;
    assert(sun, `${sceneId} has no sun`);
    const view = await TestView.create(device, scene.platform);
    const field = scene.ddgi.field;
    assert(field, 'no field');
    const bookmark = manifest.cameras[0];
    const images = async () => view.render(field, bookmark, VIEW_WIDTH, VIEW_HEIGHT);
    const mean = async () => imageMean((await images()).indirect);
    return { scene, view, sun, lamp, mean, images };
}

/** Frames through the adapter at a fixed batch, waiting for the GPU every few. */
export async function sponzaFrames(test: SponzaUnderTest, count: number, units: number): Promise<void> {
    for (let index = 0; index < count; index++) {
        test.scene.frame({ units });
        if (index % 4 === 3) {
            await test.scene.device.queue.onSubmittedWorkDone();
            await yieldToCallbacks();
        }
    }
    await test.scene.device.queue.onSubmittedWorkDone();
}

/** Converges the field (the file comment); then, with `settleUnits`, SETTLE_FRAMES at that batch. Returns the view's mean. */
export async function convergeSponza(test: SponzaUnderTest, label: string, log: (line: string) => void, settleUnits: number | null): Promise<number> {
    const settings = test.scene.ddgi.settings;
    applyHysteresisMode(settings, { name: 'fast', settings: { ...FIXED_097.settings, hysteresis: FAST_HYSTERESIS } });
    await sponzaFrames(test, FAST_FRAMES, CONVERGE_UNITS);
    applyHysteresisMode(settings, FIXED_097);
    let mean = await test.mean();
    let settled = false;
    for (let block = 1; block <= MAX_BLOCKS && !settled; block++) {
        await sponzaFrames(test, BLOCK_FRAMES, CONVERGE_UNITS);
        const next = await test.mean();
        settled = block >= 2 && Math.abs(next / mean - 1) < CONVERGED_TOLERANCE;
        mean = next;
        if (settled) log(`${label}: converged after ${block} blocks, view mean ${next.toPrecision(5)}`);
    }
    assert(settled, `${label}: the view did not settle`);
    applyHysteresisMode(settings, parseHysteresisMode('default'));
    if (settleUnits !== null) await sponzaFrames(test, SETTLE_FRAMES, settleUnits);
    return test.mean();
}

/** The sun's direction (from the light toward its target). */
export function sunDirection(sun: DirectionalLight): Vector3 {
    return new Vector3().subVectors(sun.target.getWorldPosition(new Vector3()), sun.getWorldPosition(new Vector3())).normalize();
}

/** Points the sun along `direction`, keeping its target; the adapter sees it at the next update. */
export function aimSun(test: SponzaUnderTest, direction: Vector3): void {
    const target = test.sun.target.getWorldPosition(new Vector3());
    const distance = test.sun.getWorldPosition(new Vector3()).distanceTo(target);
    test.sun.position.copy(target).addScaledVector(direction, -distance);
    test.sun.updateMatrixWorld();
    test.scene.threeGI.markChanged(test.sun);
}

/**
 * The sun turned `degrees` lower: its elevation lowered, its azimuth kept.
 * `direction` is the way the light travels. Until T2.6a this turned about
 * cross(direction, up) by −degrees, which raises the sun instead: Sponza's
 * from 70.6° over the zenith to 79.4° (NOTES).
 */
export function turnedSun(direction: Vector3, degrees: number): Vector3 {
    const elevation = Math.asin(-direction.y) - (degrees * Math.PI) / 180;
    const horizontal = new Vector3(direction.x, 0, direction.z).normalize();
    return new Vector3(horizontal.x * Math.cos(elevation), -Math.sin(elevation), horizontal.z * Math.cos(elevation));
}

/** The sun's elevation and azimuth (degrees) from the way its light travels; the azimuth from +X toward +Z. */
export function sunAngles(direction: Vector3): { elevation: number; azimuth: number } {
    return { elevation: (Math.asin(-direction.y) * 180) / Math.PI, azimuth: (Math.atan2(-direction.z, -direction.x) * 180) / Math.PI };
}

/** The way the light of a sun at `elevation` and `azimuth` (degrees, as {@link sunAngles}) travels. */
export function sunFromAngles(elevation: number, azimuth: number): Vector3 {
    const e = (elevation * Math.PI) / 180;
    const a = (azimuth * Math.PI) / 180;
    return new Vector3(-Math.cos(e) * Math.cos(a), -Math.sin(e), -Math.cos(e) * Math.sin(a));
}

/** The sky's brightness times `factor` (the adapter sends the environment again: a global change). */
export function dimSky(test: SponzaUnderTest, factor: number): void {
    const environment = test.scene.threeGI.environment;
    assert(environment && environment.type === 'procedural', 'the scene has no procedural sky to dim');
    test.scene.threeGI.environment = { ...environment, intensity: (environment.intensity ?? 1) * factor };
}

/** Each probe's mean irradiance, whether it is active, and its record. */
export interface ProbeMeans {
    readonly mean: Vec3[];
    readonly active: boolean[];
    readonly history: ProbeHistoryRecord[];
}

export async function probeMeans(test: SponzaUnderTest): Promise<ProbeMeans> {
    const field = test.scene.ddgi.field!;
    const [history, data] = await Promise.all([readHistory(test.scene.device, field), readProbeData(test.scene.device, field)]);
    return { mean: history.map((record) => record.mean), active: data.map((probe) => probe.state === 0), history };
}

/** Which probes are near and in view (camera weight ≥ 0.5 from the scene's camera) and which are keys. */
export interface ProbeClasses {
    readonly near: (probe: number) => boolean;
    readonly key: (probe: number) => boolean;
}

export function probeClasses(test: SponzaUnderTest, field: DDGIField): ProbeClasses {
    const camera = test.scene.camera;
    camera.updateMatrixWorld();
    const planes = frustumPlanes({ view: [...camera.matrixWorldInverse.elements], projection: [...camera.projectionMatrix.elements] });
    const grid = field.grid;
    const near = (probeIndex: number) => {
        const point = probeCoordinates(probeIndex, grid.counts).map((value, axis) => grid.origin[axis] + value * grid.spacing[axis]) as Vec3;
        return cameraWeight(point, { position: [camera.position.x, camera.position.y, camera.position.z] }, planes, grid) >= NEAR_CAMERA_WEIGHT;
    };
    const key = (probeIndex: number) => isKeyProbe(probeCoordinates(probeIndex, grid.counts), grid.counts);
    return { near, key };
}

/** Probes still waiting for their keys (the awaiting flag). */
export function awaitingCount(means: ProbeMeans): number {
    return means.history.filter((record) => (record.flags & PROBE_HISTORY_FLAG.awaitingChange) !== 0).length;
}

/** Each active probe's error against a reference (relative, its largest channel; dark probes against a floor of 1% of the volume's mean). */
export function probeErrors(now: ProbeMeans, reference: ProbeMeans, include: (probe: number) => boolean = () => true): { errors: number[]; probes: number[] } {
    const luminance = (rgb: Vec3) => (rgb[0] + rgb[1] + rgb[2]) / 3;
    const activeMeans = reference.mean.filter((_, probe) => reference.active[probe]).map(luminance);
    const floor = 0.01 * (activeMeans.reduce((sum, value) => sum + value, 0) / Math.max(1, activeMeans.length));
    const errors: number[] = [];
    const probes: number[] = [];
    for (let probe = 0; probe < reference.mean.length; probe++) {
        if (!reference.active[probe] || !now.active[probe] || !include(probe)) continue;
        let largest = 0;
        for (let channel = 0; channel < 3; channel++) largest = Math.max(largest, Math.abs(now.mean[probe][channel] - reference.mean[probe][channel]) / Math.max(reference.mean[probe][channel], floor));
        errors.push(largest);
        probes.push(probe);
    }
    return { errors, probes };
}

/**
 * The field's error weighted by its light (T2.4b check 5's measure,
 * DECISIONS 2026-11-06): the sum of |now − reference| over the sum of the
 * reference, every active probe and channel (those `include` takes).
 */
export function lightWeightedError(now: ProbeMeans, reference: ProbeMeans, include: (probe: number) => boolean = () => true): number {
    let difference = 0;
    let sum = 0;
    for (let probe = 0; probe < reference.mean.length; probe++) {
        if (!reference.active[probe] || !now.active[probe] || !include(probe)) continue;
        for (let channel = 0; channel < 3; channel++) {
            difference += Math.abs(now.mean[probe][channel] - reference.mean[probe][channel]);
            sum += reference.mean[probe][channel];
        }
    }
    return sum > 0 ? difference / sum : 0;
}

/**
 * How much of a change a field has taken in, at the probes `include` takes:
 * `removed` is 1 − Σ|now − reference| / Σ|before − reference| (the share of
 * the light-weighted error the change left that is gone; noise counts
 * against it), `along` is Σ (now − before)·(reference − before) / Σ
 * (reference − before)² (how far the field moved along the change), and
 * `change` the change's size, Σ|reference − before| / Σ reference.
 */
export function changeTakenIn(before: ProbeMeans, now: ProbeMeans, reference: ProbeMeans, include: (probe: number) => boolean): { removed: number; along: number; change: number; probes: number } {
    let errorBefore = 0;
    let errorNow = 0;
    let projected = 0;
    let squared = 0;
    let referenceSum = 0;
    let probes = 0;
    for (let probe = 0; probe < reference.mean.length; probe++) {
        if (!reference.active[probe] || !now.active[probe] || !before.active[probe] || !include(probe)) continue;
        probes++;
        for (let channel = 0; channel < 3; channel++) {
            const step = reference.mean[probe][channel] - before.mean[probe][channel];
            errorBefore += Math.abs(step);
            errorNow += Math.abs(reference.mean[probe][channel] - now.mean[probe][channel]);
            projected += (now.mean[probe][channel] - before.mean[probe][channel]) * step;
            squared += step * step;
            referenceSum += reference.mean[probe][channel];
        }
    }
    return {
        removed: errorBefore > 0 ? 1 - errorNow / errorBefore : 0,
        along: squared > 0 ? projected / squared : 0,
        change: referenceSum > 0 ? errorBefore / referenceSum : 0,
        probes,
    };
}

export const meanOf = (values: readonly number[]) => values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);

export function quantile(values: readonly number[], share: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(share * sorted.length))] ?? Number.NaN;
}

function toSrgb8(linear: number): number {
    const clamped = Math.min(1, Math.max(0, linear));
    const encoded = clamped <= 0.0031308 ? clamped * 12.92 : 1.055 * clamped ** (1 / 2.4) - 0.055;
    return Math.round(encoded * 255);
}

/** An rgb float image as a PNG (base64), scaled by `exposure` and sRGB-encoded, for the runner's JSON. */
export async function pngBase64(rgb: Float32Array, width: number, height: number, exposure: number): Promise<string> {
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let pixel = 0; pixel < width * height; pixel++) {
        for (let channel = 0; channel < 3; channel++) rgba[pixel * 4 + channel] = toSrgb8(rgb[pixel * 3 + channel] * exposure);
        rgba[pixel * 4 + 3] = 255;
    }
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext('2d');
    assert(context, 'no 2d canvas for PNG encoding');
    context.putImageData(new ImageData(rgba, width, height), 0, 0);
    const bytes = new Uint8Array(await (await canvas.convertToBlob({ type: 'image/png' })).arrayBuffer());
    let binary = '';
    for (let offset = 0; offset < bytes.length; offset += 0x8000) binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
    return btoa(binary);
}
