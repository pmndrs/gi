/**
 * T2.10 (TASKS, T2.M1's review): `setTransforms` with 10,000 transforms
 * inside a real three adapter frame, with three rendering every frame.
 *
 * T2.M1 item 1 timed the call alone (`scene-set-transforms`: 1.4 ms median,
 * a platform with no three and no technique). T1.7 saw the adapter's work run
 * 2-4x slower in real frames, after three's render has used the caches. So
 * here the call runs where a host makes it: an `InstancedMesh` of 10,000
 * small spheres in the loaded scene, every instance sliding every frame
 * (the host's work: new instance matrices, not timed), then the adapter's
 * loop (`threeGI.update`, which sends every moved instance with one
 * `setTransforms` call, then three's `render`), DDGI updating at its 2 ms
 * budget.
 *
 * Timed: each `setTransforms` call (the platform's scene API wrapped on the
 * page, nothing changed in the library), and beside it the adapter's whole
 * sync (`threeGI.timings.syncMs`, which includes the call and the adapter's
 * own check of 10,000 instance matrices) and the platform's encode.
 * Reported against the 2 ms bar of T2.M1 item 1; `performance.now()` steps
 * in 0.1 ms with jitter (NOTES), so the median of many frames is the number.
 */

import { InstancedMesh, Matrix4, MeshStandardMaterial, SphereGeometry, type Scene } from 'three/webgpu';
import type { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';

export const SET_TRANSFORMS_INSTANCES = 10_000;
/** T2.M1 item 1's bar (TASKS): 10,000 transforms through one call in at most 2 ms of CPU. */
export const SET_TRANSFORMS_TARGET_MS = 2;
const SPHERE_RADIUS = 0.05;
/** How far each instance slides from its home, per axis (metres): every instance moves every frame. */
const SLIDE_METRES = 0.1;

export interface MovingInstances {
    readonly mesh: InstancedMesh;
    /** Moves every instance (a slide, so the linear part stays) and tells three the matrices changed. */
    move(frame: number): void;
}

/** 10,000 small spheres at random places in the box (a fixed seed, so every run has the same layout). */
export function addMovingInstances(scene: Scene, center: readonly number[], size: readonly number[]): MovingInstances {
    const mesh = new InstancedMesh(new SphereGeometry(SPHERE_RADIUS, 8, 6), new MeshStandardMaterial({ color: 0xb0b0b0 }), SET_TRANSFORMS_INSTANCES);
    mesh.name = 't2.10 moving instances';
    // The scene's bounding sphere would be computed once; the instances stay inside the volume.
    mesh.frustumCulled = false;
    let seed = 12345;
    const random = (): number => {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        return seed / 4294967296;
    };
    const homes = new Float32Array(SET_TRANSFORMS_INSTANCES * 3);
    const phases = new Float32Array(SET_TRANSFORMS_INSTANCES);
    for (let index = 0; index < SET_TRANSFORMS_INSTANCES; index++) {
        for (let axis = 0; axis < 3; axis++) homes[index * 3 + axis] = center[axis] + (random() - 0.5) * size[axis] * 0.9;
        phases[index] = random() * Math.PI * 2;
    }
    const matrix = new Matrix4();
    const move = (frame: number): void => {
        for (let index = 0; index < SET_TRANSFORMS_INSTANCES; index++) {
            const offset = SLIDE_METRES * Math.sin(frame * 0.1 + phases[index]);
            matrix.makeTranslation(homes[index * 3] + offset, homes[index * 3 + 1] + offset * 0.5, homes[index * 3 + 2] - offset);
            mesh.setMatrixAt(index, matrix);
        }
        mesh.instanceMatrix.needsUpdate = true;
    };
    move(0);
    scene.add(mesh);
    return { mesh, move };
}

/** Wraps the platform's `scene.setTransforms` so each call's CPU time and size are recorded. */
export function timeSetTransforms(platform: GIPlatformImpl): { readonly calls: Array<{ ms: number; transforms: number }> } {
    const calls: Array<{ ms: number; transforms: number }> = [];
    const api = platform.scene;
    const original = api.setTransforms.bind(api);
    api.setTransforms = (instances: Uint32Array, matrices: Float32Array): void => {
        const start = performance.now();
        original(instances, matrices);
        calls.push({ ms: performance.now() - start, transforms: instances.length });
    };
    return { calls };
}
