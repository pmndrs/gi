/**
 * The manifests' `motions` (T2.1, docs/design/m2-lights.md §7): the format's
 * validator and `motionState`, and the built-in pair every scene with a lamp
 * has, checked against the scene itself: the sweep glides at 1 m/s and comes
 * back, the step jumps every 3 s, and neither moves the lamp through a wall.
 *
 * The geometry check: the path from the lamp's own place to each keyframe in
 * turn, thickened by CLEARANCE_M (the segment and six copies moved that far
 * along each axis), must hit no triangle. Since the lamp's place is open
 * space, every keyframe is then open space too, not inside a wall. Procedural
 * scenes are built here; scenes from downloaded models are checked when the
 * files are there (`pnpm assets:fetch`).
 */

import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { extractGltfGeometry, parseGlb } from '../../tools/gpu/pages/trace-bench/gltf-geometry';
import { loadManifests } from './export';
import { BUILT_IN_MOTIONS, motionState, STEP_SECONDS, SWEEP_SPEED, validateManifest, type ManifestMotion, type SceneManifest, type Vec3 } from './manifest';
import { modelTransformMatrix } from './model-bounds';
import { buildProcedural } from './procedural';

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const downloadsDirectory = process.env.DDGI_DOWNLOADS_DIR ?? join(repoRoot, 'assets', 'downloads');
/** How far a lamp's path keeps from every surface. */
const CLEARANCE_M = 0.1;

function sweep(from: Vec3, to: Vec3, seconds: number): ManifestMotion {
    return { id: 'light-sweep', label: 'sweep', target: 'light:lamp', loop: true, keyframes: [{ t: 0, position: from }, { t: seconds, position: to }, { t: 2 * seconds, position: from }] };
}

function withLamp(motions: unknown): Record<string, unknown> {
    return {
        id: 'test-scene',
        title: 'Test',
        description: 'A test scene.',
        tags: ['test'],
        units: 'meters',
        upAxis: 'y',
        parts: [{ kind: 'procedural', builder: 'two-room' }],
        cameras: [
            { name: 'a', position: [0, 1, 0], target: [1, 1, 1], fovYDeg: 60 },
            { name: 'b', position: [0, 2, 0], target: [1, 1, 1], fovYDeg: 60 },
        ],
        lights: [
            { type: 'point', label: 'lamp', color: [1, 1, 1], intensity: 1, position: [1, 1, 1], range: 0 },
            { type: 'directional', label: 'sun', color: [1, 1, 1], intensity: 1, direction: [0, -1, 0] },
        ],
        motions,
        environment: { type: 'none' },
        volume: { center: [0, 0, 0], size: [4, 4, 4], probeSpacing: 0.5 },
        nativeCapable: true,
        credits: [{ asset: null, license: 'MIT', attribution: 'MIT, this repo.' }],
    };
}

describe('the motions format', () => {
    it('accepts a sweep and a step', () => {
        const step: ManifestMotion = { id: 'light-step', label: 'step', target: 'light:lamp', interpolation: 'step', loop: true, keyframes: [{ t: 0, position: [1, 1, 1] }, { t: 3, position: [2, 1, 1] }, { t: 6, position: [1, 1, 1] }] };
        expect(validateManifest(withLamp([sweep([1, 1, 1], [2, 1, 1], 1), step])).errors).toEqual([]);
    });

    it('names what is wrong', () => {
        const errors = (motions: unknown) => validateManifest(withLamp(motions)).errors.join('\n');
        expect(errors([{ ...sweep([1, 1, 1], [2, 1, 1], 1), target: 'light:nobody' }])).toContain('no light has the label "nobody"');
        expect(errors([{ ...sweep([1, 1, 1], [2, 1, 1], 1), target: 'lamp' }])).toContain('must be "light:<label>"');
        expect(errors([sweep([1, 1, 1], [2, 1, 1], 1), sweep([1, 1, 1], [2, 1, 1], 1)])).toContain('is used twice');
        expect(errors([{ ...sweep([1, 1, 1], [2, 1, 1], 1), keyframes: [{ t: 0, position: [1, 1, 1] }] }])).toContain('at least two keyframes');
        expect(errors([{ ...sweep([1, 1, 1], [2, 1, 1], 1), keyframes: [{ t: 1, position: [1, 1, 1] }, { t: 2, position: [1, 1, 1] }] }])).toContain('the first keyframe is at 0 s');
        expect(errors([{ ...sweep([1, 1, 1], [2, 1, 1], 1), keyframes: [{ t: 0, position: [1, 1, 1] }, { t: 0, position: [1, 1, 1] }] }])).toContain('later than the keyframe before');
        expect(errors([{ ...sweep([1, 1, 1], [2, 1, 1], 1), keyframes: [{ t: 0, position: [1, 1, 1] }, { t: 1, intensity: 2 }] }])).toContain('the same fields as the first keyframe');
        expect(errors([{ ...sweep([1, 1, 1], [2, 1, 1], 1), target: 'light:sun' }])).toContain('a directional light has no position');
        expect(errors([{ ...sweep([1, 1, 1], [2, 1, 1], 1), interpolation: 'cubic' }])).toContain('"linear" or "step"');
        expect(errors([{ ...sweep([1, 1, 1], [2, 1, 1], 1), keyframes: [{ t: 0, intensity: -1 }, { t: 1, intensity: 1 }] }])).toContain('0 or more');
    });

    it('plays a motion: glides linearly, holds for a step, loops', () => {
        const glide = sweep([0, 1, 0], [4, 1, 0], 4);
        expect(motionState(glide, 1).position).toEqual([1, 1, 0]);
        expect(motionState(glide, 6).position).toEqual([2, 1, 0]);
        expect(motionState(glide, 8 + 1).position).toEqual([1, 1, 0]);
        const step: ManifestMotion = { ...glide, interpolation: 'step', keyframes: [{ t: 0, position: [0, 1, 0], intensity: 1 }, { t: 3, position: [4, 1, 0], intensity: 2 }, { t: 6, position: [0, 1, 0], intensity: 1 }] };
        expect(motionState(step, 2.99)).toEqual({ position: [0, 1, 0], intensity: 1 });
        expect(motionState(step, 3)).toEqual({ position: [4, 1, 0], intensity: 2 });
        expect(motionState(step, 6.5).position).toEqual([0, 1, 0]);
        // A motion that doesn't loop stays at its last keyframe.
        expect(motionState({ ...glide, loop: false }, 100).position).toEqual([0, 1, 0]);
    });
});

/** Every triangle of a scene in world space, nine numbers each, or null when a downloaded model is missing. */
function sceneTriangles(manifest: SceneManifest): Float32Array | null {
    const chunks: Float32Array[] = [];
    const add = (positions: Float32Array, indices: Uint32Array, matrix: ArrayLike<number>) => {
        const triangles = new Float32Array(indices.length * 3);
        for (let corner = 0; corner < indices.length; corner++) {
            const v = indices[corner] * 3;
            const [x, y, z] = [positions[v], positions[v + 1], positions[v + 2]];
            for (let axis = 0; axis < 3; axis++) triangles[corner * 3 + axis] = matrix[axis] * x + matrix[4 + axis] * y + matrix[8 + axis] * z + matrix[12 + axis];
        }
        chunks.push(triangles);
    };
    for (const part of manifest.parts) {
        if (part.kind === 'procedural') {
            const built = buildProcedural(part.builder, part.options);
            for (const instance of built.instances) add(built.meshes[instance.mesh].positions, built.meshes[instance.mesh].indices, instance.transform);
        } else if (part.kind === 'model') {
            const file = join(downloadsDirectory, part.asset, part.path);
            if (!existsSync(file)) return null;
            let geometry: ReturnType<typeof extractGltfGeometry>;
            if (file.endsWith('.glb')) {
                const bytes = readFileSync(file);
                const { gltf, binary } = parseGlb(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
                geometry = extractGltfGeometry(gltf, [binary!]);
            } else {
                const gltf = JSON.parse(readFileSync(file, 'utf8'));
                const buffers = (gltf.buffers as { uri: string }[]).map((buffer) => new Uint8Array(readFileSync(join(dirname(file), decodeURIComponent(buffer.uri)))));
                geometry = extractGltfGeometry(gltf, buffers);
            }
            const placement = modelTransformMatrix(part.transform);
            for (const instance of geometry.instances) {
                const mesh = geometry.meshes[instance.meshIndex];
                const local = instance.objectToWorld as unknown as number[];
                // The manifest's placement after the file's own node transform.
                const world = new Array<number>(16).fill(0);
                for (let row = 0; row < 4; row++) for (let column = 0; column < 4; column++) for (let k = 0; k < 4; k++) world[column * 4 + row] += placement[k * 4 + row] * local[column * 4 + k];
                add(mesh.positions, mesh.indices, world);
            }
        }
    }
    const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const all = new Float32Array(total);
    let at = 0;
    for (const chunk of chunks) {
        all.set(chunk, at);
        at += chunk.length;
    }
    return all;
}

/** Whether the segment from `a` to `b` crosses any triangle (Möller-Trumbore, both faces). */
function segmentHits(triangles: Float32Array, a: Vec3, b: Vec3): boolean {
    const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    for (let i = 0; i < triangles.length; i += 9) {
        const e1 = [triangles[i + 3] - triangles[i], triangles[i + 4] - triangles[i + 1], triangles[i + 5] - triangles[i + 2]];
        const e2 = [triangles[i + 6] - triangles[i], triangles[i + 7] - triangles[i + 1], triangles[i + 8] - triangles[i + 2]];
        const p = [d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]];
        const det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
        if (Math.abs(det) < 1e-14) continue;
        const inverse = 1 / det;
        const s = [a[0] - triangles[i], a[1] - triangles[i + 1], a[2] - triangles[i + 2]];
        const u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * inverse;
        if (u < 0 || u > 1) continue;
        const q = [s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]];
        const v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inverse;
        if (v < 0 || u + v > 1) continue;
        const t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inverse;
        if (t >= 0 && t <= 1) return true;
    }
    return false;
}

function clearPath(triangles: Float32Array, a: Vec3, b: Vec3): boolean {
    // Only triangles whose box meets the thickened segment's box can be hit (Sponza has 5.7 M).
    const low = [0, 1, 2].map((axis) => Math.min(a[axis], b[axis]) - CLEARANCE_M);
    const high = [0, 1, 2].map((axis) => Math.max(a[axis], b[axis]) + CLEARANCE_M);
    const near: number[] = [];
    for (let i = 0; i < triangles.length; i += 9) {
        let outside = false;
        for (let axis = 0; axis < 3 && !outside; axis++) {
            const values = [triangles[i + axis], triangles[i + 3 + axis], triangles[i + 6 + axis]];
            outside = Math.max(...values) < low[axis] || Math.min(...values) > high[axis];
        }
        if (!outside) for (let value = 0; value < 9; value++) near.push(triangles[i + value]);
    }
    const nearby = Float32Array.from(near);
    const offsets: Vec3[] = [[0, 0, 0], [CLEARANCE_M, 0, 0], [-CLEARANCE_M, 0, 0], [0, CLEARANCE_M, 0], [0, -CLEARANCE_M, 0], [0, 0, CLEARANCE_M], [0, 0, -CLEARANCE_M]];
    return offsets.every((offset) => !segmentHits(nearby, [a[0] + offset[0], a[1] + offset[1], a[2] + offset[2]], [b[0] + offset[0], b[1] + offset[1], b[2] + offset[2]]));
}

/** Whether a scene's downloaded models are all there. */
function modelsPresent(manifest: SceneManifest): boolean {
    return manifest.parts.every((part) => part.kind !== 'model' || existsSync(join(downloadsDirectory, part.asset, part.path)));
}

const pathLength = (points: readonly Vec3[]) => points.slice(1).reduce((sum, point, index) => sum + Math.hypot(point[0] - points[index][0], point[1] - points[index][1], point[2] - points[index][2]), 0);

describe('the built-in motions of each scene with a lamp (m2-lights.md §7)', async () => {
    const manifests = await loadManifests();
    // Scenes whose own lights include a lamp, and Sponza, which has one for its motions (off until one plays).
    const lit = manifests.filter((manifest) => manifest.lights.some((light) => light.type !== 'directional'));

    it('are in every scene with a lamp, and only there', () => {
        expect(lit.map((manifest) => manifest.id).sort()).toEqual(['bistro-interior', 'cabinet-row', 'closed-door', 'cornell-box-point', 'furnished-room', 'sponza', 'thin-wall', 'two-room']);
        for (const manifest of manifests) {
            const ids = (manifest.motions ?? []).map((motion) => motion.id).sort();
            expect(ids, manifest.id).toEqual(lit.includes(manifest) ? [BUILT_IN_MOTIONS.step, BUILT_IN_MOTIONS.sweep] : []);
        }
    });

    for (const manifest of lit) {
        describe(manifest.id, () => {
            const sweepMotion = manifest.motions?.find((motion) => motion.id === BUILT_IN_MOTIONS.sweep);
            const stepMotion = manifest.motions?.find((motion) => motion.id === BUILT_IN_MOTIONS.step);

            it('sweeps one lamp at 1 m/s and back, and steps the same lamp every 3 s', () => {
                expect(sweepMotion && stepMotion).toBeTruthy();
                expect(stepMotion!.target).toBe(sweepMotion!.target);
                expect(sweepMotion!.loop && stepMotion!.loop).toBe(true);
                for (const motion of [sweepMotion!, stepMotion!]) {
                    const frames = motion.keyframes;
                    expect(frames[frames.length - 1].position, `${motion.id} ends where it starts`).toEqual(frames[0].position);
                }
                const sweepFrames = sweepMotion!.keyframes;
                for (let index = 1; index < sweepFrames.length; index++) {
                    const distance = pathLength([sweepFrames[index - 1].position!, sweepFrames[index].position!]);
                    expect(distance / (sweepFrames[index].t - sweepFrames[index - 1].t), `sweep leg ${index}`).toBeCloseTo(SWEEP_SPEED, 2);
                }
                expect(stepMotion!.interpolation).toBe('step');
                stepMotion!.keyframes.forEach((frame, index) => expect(frame.t).toBeCloseTo(index * STEP_SECONDS, 6));
            });

            // Built inside the test, not while collecting: Sponza's take seconds to read.
            it.skipIf(!modelsPresent(manifest))(`keeps the lamp ${CLEARANCE_M} m clear of every surface, from where it hangs to every keyframe`, { timeout: 120_000 }, () => {
                const triangles = sceneTriangles(manifest);
                const label = sweepMotion!.target.slice('light:'.length);
                const lamp = manifest.lights.find((light) => light.label === label);
                expect(lamp && lamp.type !== 'directional').toBe(true);
                const home = (lamp as { position: Vec3 }).position;
                for (const motion of [sweepMotion!, stepMotion!]) {
                    const points: Vec3[] = [home, ...motion.keyframes.map((frame) => frame.position!)];
                    for (let index = 1; index < points.length; index++) {
                        expect(clearPath(triangles!, points[index - 1], points[index]), `${motion.id}: ${JSON.stringify(points[index - 1])} -> ${JSON.stringify(points[index])}`).toBe(true);
                    }
                }
            });
        });
    }
});
