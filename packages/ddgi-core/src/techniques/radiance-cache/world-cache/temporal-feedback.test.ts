import { describe, expect, it } from 'vitest';
import { compareToggle } from '../compare-toggles.ts';
import { RadianceCacheTechnique } from '../radiance-cache-technique.ts';
import { DEFAULT_RADIANCE_CACHE_SETTINGS, PREPARE_SETTINGS } from '../settings.ts';
import { DEFAULT_HASH_GRID_CACHE_OPTIONS } from './capacity.ts';
import { HASH_GRID_CONSTANTS_WORD } from './hash-grid-wgsl.ts';
import { generateReservoirsWgsl } from './hash-grid-population-wgsl.ts';
import { hashGridCacheBufferLayout, hashGridCacheCapacity } from './capacity.ts';
import {
    TEMPORAL_FEEDBACK_NORMAL_THRESHOLD,
    TEMPORAL_FEEDBACK_PLANE_TOLERANCE,
    TEMPORAL_FEEDBACK_UNIFORM_BYTES,
    TEMPORAL_FEEDBACK_WGSL,
    TEMPORAL_FEEDBACK_WORD,
    temporalFeedbackUniformWords,
} from './temporal-feedback.ts';

/** A view matrix looking from `eye` at `target` (y up) and a WebGPU perspective (depth 0 to 1), column-major. */
function camera(eye: readonly [number, number, number], target: readonly number[], fovYDeg: number, aspect: number, near: number, far: number): { view: number[]; projection: number[] } {
    const sub = (a: readonly number[], b: readonly number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const normalize = (a: number[]) => a.map((value) => value / Math.hypot(a[0], a[1], a[2]));
    const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const forward = normalize(sub(target, eye));
    const right = normalize(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);
    const view = [right[0], up[0], -forward[0], 0, right[1], up[1], -forward[1], 0, right[2], up[2], -forward[2], 0, -dot(right, eye), -dot(up, eye), dot(forward, eye), 1];
    const f = 1 / Math.tan((fovYDeg * Math.PI) / 360);
    const projection = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0];
    return { view, projection };
}

/** Column-major 4 × 4 times a point. */
function transform(m: ArrayLike<number>, p: readonly number[]): number[] {
    return [0, 1, 2, 3].map((row) => m[row] * p[0] + m[4 + row] * p[1] + m[8 + row] * p[2] + m[12 + row] * p[3]);
}

describe('temporal feedback (RC3.6, register D68)', () => {
    it("is off by default, as Capsaicin's gi1_use_temporal_feedback; on, the technique asks for the lit image and albedo, and a change restarts it", () => {
        expect(DEFAULT_RADIANCE_CACHE_SETTINGS.useTemporalFeedback).toBe(false);
        expect(DEFAULT_HASH_GRID_CACHE_OPTIONS.useTemporalFeedback).toBe(false);
        expect(new RadianceCacheTechnique({ useTemporalFeedback: true }).requirements).toEqual({ queries: ['rays'], screen: { velocity: true, albedo: true, lighting: true } });
        expect(PREPARE_SETTINGS).toContain('useTemporalFeedback');
        expect(compareToggle('useTemporalFeedback')).toMatchObject({ urlName: 'rctemporal', register: 'D68', defaultValue: false, capsaicin: false, restartsTechnique: true });
    });

    it('lays the uniform out as the WGSL struct does', () => {
        // mat4x4f, mat4x4f, vec3f + f32, vec2u, f32, u32: 160 bytes.
        expect(TEMPORAL_FEEDBACK_UNIFORM_BYTES).toBe(160);
        expect(TEMPORAL_FEEDBACK_WORD).toEqual({ previousViewProjection: 0, previousInverseViewProjection: 16, previousEye: 32, planeTolerance: 35, size: 36, normalThreshold: 38 });
        const fields = /struct HashGridCacheTemporalFeedback \{([^}]*)\}/.exec(TEMPORAL_FEEDBACK_WGSL)?.[1].split('\n').map((line) => line.replace(/\/\/.*$/, '').trim()).filter(Boolean);
        expect(fields).toEqual(['previous_view_projection: mat4x4f,', 'previous_inverse_view_projection: mat4x4f,', 'previous_eye: vec3f,', 'plane_tolerance: f32,', 'size: vec2u,', 'normal_threshold: f32,', 'padding: u32,']);
        expect(HASH_GRID_CONSTANTS_WORD.temporalFeedback).toBe(34);
        expect(HASH_GRID_CONSTANTS_WORD.cacheFeedback).toBe(33);
    });

    it("projects a world point into last frame's screen and back through the uniform's matrices (camera far from the origin)", () => {
        const eye: [number, number, number] = [1000.5, 12, -2000.25];
        const bytes = temporalFeedbackUniformWords({ camera: { position: eye, ...camera(eye, [1004, 11, -2006], 60, 16 / 9, 0.05, 1000) }, width: 1280, height: 720 });
        const floats = new Float32Array(bytes);
        const words = new Uint32Array(bytes);
        const viewProjection = floats.subarray(0, 16);
        const inverse = floats.subarray(16, 32);
        expect(Array.from(floats.subarray(32, 35))).toEqual(eye.map(Math.fround));
        expect(floats[TEMPORAL_FEEDBACK_WORD.planeTolerance]).toBeCloseTo(TEMPORAL_FEEDBACK_PLANE_TOLERANCE, 6);
        expect(floats[TEMPORAL_FEEDBACK_WORD.normalThreshold]).toBeCloseTo(TEMPORAL_FEEDBACK_NORMAL_THRESHOLD, 6);
        expect([words[36], words[37]]).toEqual([1280, 720]);

        // A point in front of the camera, relative to the eye as the kernel makes it.
        const world = [1003, 11.5, -2004];
        const relative = [world[0] - eye[0], world[1] - eye[1], world[2] - eye[2], 1];
        const clip = transform(viewProjection, relative);
        expect(clip[3]).toBeGreaterThan(0);
        const ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];
        for (const value of ndc.slice(0, 2)) expect(Math.abs(value)).toBeLessThan(1);
        // Back from NDC and the stored depth: the same point (what the plane test compares).
        const back = transform(inverse, [ndc[0], ndc[1], ndc[2], 1]);
        const shown = [back[0] / back[3], back[1] / back[3], back[2] / back[3]];
        for (let axis = 0; axis < 3; axis++) expect(shown[axis]).toBeCloseTo(relative[axis], 3);
    });

    it('is read in GenerateReservoirs from group 3, before the light sample, and keeps the probe ray off the resolve list', () => {
        const code = generateReservoirsWgsl(hashGridCacheBufferLayout(hashGridCacheCapacity(DEFAULT_HASH_GRID_CACHE_OPTIONS)));
        expect(code).toContain('@group(3) @binding(0) var<uniform> g_HashGridCache_TemporalFeedback');
        const feedback = code.indexOf('HashGridCache_TemporalFeedback(surface)');
        expect(feedback).toBeGreaterThan(0);
        expect(feedback).toBeLessThan(code.indexOf('giDirectChooseLight'));
        expect(code).toMatch(/if \(reads_cell\) \{\s*let resolve_index/);
        expect(code).toContain('let bypass_cache = reads_cell && ray_length < cell_size;');
    });
});

describe('cache feedback (RC3.6, register E12)', () => {
    it("is off by default, read every frame, with a compare toggle whose Capsaicin value is off", () => {
        expect(DEFAULT_RADIANCE_CACHE_SETTINGS.cacheFeedback).toBe(false);
        expect(DEFAULT_HASH_GRID_CACHE_OPTIONS.cacheFeedback).toBe(false);
        expect(PREPARE_SETTINGS).not.toContain('cacheFeedback');
        expect(compareToggle('cacheFeedback')).toMatchObject({ urlName: 'rcfeedback', register: 'E12', defaultValue: false, capsaicin: false, restartsTechnique: false, resetsHistory: false });
    });
});
