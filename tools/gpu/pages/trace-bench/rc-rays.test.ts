/**
 * Tests of the `rc` ray-set generator (RC1.5): counts, the cosine
 * distribution, the hemisphere side, the multi-bounce ratio, the light
 * choice and determinism. No GPU: hits come from the CPU walk of a small
 * scene, packed as the GPU's `GiHitRecord`s.
 */

import { describe, expect, it } from 'vitest';
import {
    assembleTraceScene,
    createCpuTracer,
    HIT_RECORD_BYTES,
    HIT_RECORD_WORD,
    RAY_FLOATS,
} from '../../../../packages/core/src/platform/trace/index.ts';
import { createRandom } from '../../../../packages/core/src/platform/trace/oracle-fixtures.ts';
import { RC_HALTON_FIRST_INDEX, RC_MULTI_BOUNCE_RATIO, RC_RAYS_PER_PROBE, RC_TILE_SIZE, RC_VIEW } from './config.ts';
import { rcPredictions, traceBenchRows } from './csv-rows.ts';
import type { BenchInstance, BenchMeshGeometry } from './gltf-geometry.ts';
import {
    halton,
    hitSurfaces,
    makeBounceRays,
    makeLightShadowRays,
    makePlacementRays,
    makeProbeRays,
    multiBounceRayCount,
    offsetRayOrigin,
    placeProbes,
    rayProblems,
    rcLightsFromManifest,
    SUN_SHADOW_RAY_LENGTH,
    tileGrid,
} from './rc-rays.ts';
import type { SceneResult, TraceBenchData } from './results.ts';
import { buildMeshBvhWithTmbvh } from './tmbvh-convert.ts';

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const VIEW = { width: RC_VIEW.width, height: RC_VIEW.height, tileSize: RC_TILE_SIZE };

/** A closed box room, 20 m wide, 6 m tall, 20 m deep, its walls two triangles each (normals inward or outward, either way). */
function boxRoom(): BenchMeshGeometry {
    const [x0, x1, y0, y1, z0, z1] = [-10, 10, 0, 6, -10, 10];
    const corners = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ];
    const faces = [
        [0, 1, 2, 3], [5, 4, 7, 6], [4, 0, 3, 7], [1, 5, 6, 2], [4, 5, 1, 0], [3, 2, 6, 7],
    ];
    return {
        positions: Float32Array.from(corners.flat()),
        indices: Uint32Array.from(faces.flatMap(([a, b, c, d]) => [a, b, c, a, c, d])),
    };
}

/** A floor quad only (y = 0, 20 x 20 m): rays that miss it see "sky". */
function floorOnly(): BenchMeshGeometry {
    return {
        positions: Float32Array.from([-10, 0, -10, 10, 0, -10, 10, 0, 10, -10, 0, 10]),
        indices: Uint32Array.from([0, 2, 1, 0, 3, 2]),
    };
}

/** Traces packed rays on the CPU walk and packs the answers as the GPU's hit records. */
function traceOnCpu(meshes: BenchMeshGeometry[], instances: BenchInstance[], rays: Float32Array): ArrayBuffer {
    const data = assembleTraceScene(
        meshes.map((mesh) => buildMeshBvhWithTmbvh(mesh, 4).bvh),
        instances.map((instance, index) => ({ meshIndex: instance.meshIndex, instanceIndex: index, objectToWorld: instance.objectToWorld })),
    );
    const tracer = createCpuTracer(data);
    const count = rays.length / RAY_FLOATS;
    const records = new ArrayBuffer(count * HIT_RECORD_BYTES);
    const floats = new Float32Array(records);
    const words = new Uint32Array(records);
    const recordWords = HIT_RECORD_BYTES / 4;
    for (let ray = 0; ray < count; ray++) {
        const base = ray * RAY_FLOATS;
        const hit = tracer.closest({
            origin: [rays[base], rays[base + 1], rays[base + 2]],
            direction: [rays[base + 4], rays[base + 5], rays[base + 6]],
            tMin: rays[base + 3],
            tMax: rays[base + 7],
        });
        const record = ray * recordWords;
        floats[record + HIT_RECORD_WORD.t] = hit ? hit.t : -1;
        if (!hit) continue;
        words[record + HIT_RECORD_WORD.instance] = hit.instanceIndex;
        words[record + HIT_RECORD_WORD.triangle] = hit.triangleIndex;
        words[record + HIT_RECORD_WORD.frontFace] = hit.frontFace ? 1 : 0;
        floats[record + HIT_RECORD_WORD.barycentrics] = hit.u;
        floats[record + HIT_RECORD_WORD.barycentrics + 1] = hit.v;
    }
    return records;
}

function direction(rays: Float32Array, ray: number): [number, number, number] {
    const base = ray * RAY_FLOATS;
    return [rays[base + 4], rays[base + 5], rays[base + 6]];
}

function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

describe('rc ray sets: probe placement', () => {
    const camera = { position: [0, 1.7, 0] as [number, number, number], target: [10, 1.7, 0] as [number, number, number], fovYDeg: 60 };

    it('has 80 x 45 = 3,600 tiles at 1280x720 with 16-pixel tiles', () => {
        expect(tileGrid(VIEW)).toEqual({ columns: 80, rows: 45, count: 3600 });
    });

    it('gives Halton (2, 3) points in [0, 1)', () => {
        expect(halton(1, 2)).toBe(0.5);
        expect(halton(2, 3)).toBeCloseTo(2 / 3, 12);
        expect(halton(3, 2)).toBe(0.75);
    });

    it('aims one unit ray per tile into the view, inside the field of view, at a jittered point', () => {
        const rays = makePlacementRays(camera, VIEW, RC_HALTON_FIRST_INDEX);
        expect(rays.length).toBe(3600 * RAY_FLOATS);
        expect(rayProblems('placement', rays)).toEqual([]);
        const tanHalfY = Math.tan((60 * Math.PI) / 360);
        const tanHalfX = tanHalfY * (1280 / 720);
        let outsideView = 0;
        for (let tile = 0; tile < 3600; tile++) {
            const [x, y, z] = direction(rays, tile);
            // Forward is +x; in the camera's frame right is +z (forward x up) and up is +y.
            const inside = x > 0 && Math.abs(z / x) <= tanHalfX + 1e-6 && Math.abs(y / x) <= tanHalfY + 1e-6;
            if (!inside) outsideView++;
        }
        expect(outsideView).toBe(0);
        // Tile 0 is top-left: up, and to the left (-z when looking down +x with y up).
        const [x0, y0, z0] = direction(rays, 0);
        expect(y0 / x0).toBeGreaterThan(tanHalfY * (1 - 2 * 16 / 720));
        expect(z0 / x0).toBeLessThan(-tanHalfX * (1 - 2 * 16 / 1280));
        // The last tile is bottom-right.
        const [x1, y1, z1] = direction(rays, 3599);
        expect(y1 / x1).toBeLessThan(-tanHalfY * (1 - 2 * 16 / 720));
        expect(z1 / x1).toBeGreaterThan(tanHalfX * (1 - 2 * 16 / 1280));
        // Jittered: tile 0's point isn't the tile's corner or its centre.
        expect(y0 / x0).not.toBeCloseTo(tanHalfY, 6);
    });

    it('places one probe per tile that sees geometry, none for sky, on the camera side of the surface', () => {
        const meshes = [floorOnly()];
        const instances = [{ meshIndex: 0, objectToWorld: IDENTITY }];
        // Looking down at the floor from above, tilted so the top of the view sees past its edge.
        const view = { position: [0, 3, -9] as [number, number, number], target: [0, 0, 2] as [number, number, number], fovYDeg: 70 };
        const rays = makePlacementRays(view, VIEW, RC_HALTON_FIRST_INDEX);
        const probes = placeProbes(rays, traceOnCpu(meshes, instances, rays), meshes, instances);
        expect(probes.count).toBeGreaterThan(0);
        expect(probes.count).toBeLessThan(3600);
        expect(probes.count + probes.missCount + probes.degenerateCount).toBe(3600);
        for (let probe = 0; probe < probes.count; probe++) {
            // The floor's normal faced toward the camera (above it) is +y, and the probe sits just above the floor.
            expect(probes.normals[probe * 3 + 1]).toBeCloseTo(1, 12);
            expect(probes.origins[probe * 3 + 1]).toBeGreaterThan(0);
            expect(probes.origins[probe * 3 + 1]).toBeLessThan(1e-3);
        }
    });

    it('pushes origins off the surface by 256 rounding steps of the largest coordinate, as the platform does', () => {
        const offset = offsetRayOrigin([100, 0, 0], [0, 1, 0]);
        expect(offset[1]).toBeCloseTo(256 * 100 * 2 ** -23, 12);
        // Near the origin the step stays that of 1/16 m.
        expect(offsetRayOrigin([0, 0, 0], [0, 0, 1])[2]).toBeCloseTo(256 * 0.0625 * 2 ** -23, 15);
    });
});

describe('rc ray sets: probe rays', () => {
    const normals = [
        [0, 1, 0],
        [0, 0, -1],
        [0.6, 0, 0.8],
        [-0.48, 0.6, -0.64],
    ];
    const probesPerNormal = 250;
    function probesAlong(normal: number[]) {
        return {
            origins: new Float64Array(probesPerNormal * 3),
            normals: Float64Array.from(Array.from({ length: probesPerNormal }, () => normal).flat()),
            count: probesPerNormal,
        };
    }

    it('fires 64 unit rays per probe, all in the hemisphere about the normal, from the probe', () => {
        const probes = { origins: Float64Array.from([1, 2, 3, 4, 5, 6]), normals: Float64Array.from([0, 1, 0, 1, 0, 0]), count: 2 };
        const rays = makeProbeRays(probes, RC_RAYS_PER_PROBE, createRandom(1));
        expect(rays.length).toBe(2 * 64 * RAY_FLOATS);
        expect(rayProblems('probe', rays)).toEqual([]);
        for (let ray = 0; ray < 128; ray++) {
            const probe = Math.floor(ray / 64);
            expect(Array.from(rays.slice(ray * RAY_FLOATS, ray * RAY_FLOATS + 3))).toEqual([1 + 3 * probe, 2 + 3 * probe, 3 + 3 * probe]);
            expect(dot(direction(rays, ray), probes.normals.subarray(probe * 3, probe * 3 + 3))).toBeGreaterThan(0);
            expect(rays[ray * RAY_FLOATS + 7]).toBeGreaterThan(1e29); // no length limit
        }
    });

    it('is cosine-weighted: E[cos] = 2/3, E[cos^2] = 1/2, and no lean sideways, for any normal', () => {
        for (const normal of normals) {
            const rays = makeProbeRays(probesAlong(normal), RC_RAYS_PER_PROBE, createRandom(7));
            const count = rays.length / RAY_FLOATS;
            let cosine = 0;
            let cosineSquared = 0;
            let belowSurface = 0;
            const mean = [0, 0, 0];
            for (let ray = 0; ray < count; ray++) {
                const d = direction(rays, ray);
                const c = dot(d, normal);
                if (!(c > 0)) belowSurface++;
                cosine += c;
                cosineSquared += c * c;
                for (let axis = 0; axis < 3; axis++) mean[axis] += d[axis] - c * normal[axis];
            }
            expect(belowSurface).toBe(0);
            expect(cosine / count).toBeCloseTo(2 / 3, 2);
            expect(cosineSquared / count).toBeCloseTo(1 / 2, 2);
            // The mean of the part along the surface is ~0: the per-probe rotation spreads the angles evenly.
            expect(Math.hypot(mean[0], mean[1], mean[2]) / count).toBeLessThan(0.01);
        }
    });

    it('differs between probes (per-probe rotation and jitter) and repeats exactly with the same seed', () => {
        const probes = probesAlong([0, 1, 0]);
        const a = makeProbeRays(probes, RC_RAYS_PER_PROBE, createRandom(42));
        const b = makeProbeRays(probes, RC_RAYS_PER_PROBE, createRandom(42));
        const c = makeProbeRays(probes, RC_RAYS_PER_PROBE, createRandom(43));
        expect(Array.from(a)).toEqual(Array.from(b));
        expect(Array.from(a)).not.toEqual(Array.from(c));
        expect(direction(a, 0)).not.toEqual(direction(a, 64));
    });

    it('refuses a ray count that is not a square grid', () => {
        expect(() => makeProbeRays(probesAlong([0, 1, 0]), 60, createRandom(1))).toThrow(/square/);
    });
});

describe('rc ray sets: multi-bounce and shadow rays', () => {
    const meshes = [boxRoom()];
    const instances = [{ meshIndex: 0, objectToWorld: IDENTITY }];
    const probes = { origins: Float64Array.from([0, 1, 0, 3, 0.001, 2]), normals: Float64Array.from([0, 1, 0, 0, 1, 0]), count: 2 };
    const probeRays = makeProbeRays(probes, RC_RAYS_PER_PROBE, createRandom(3));
    const surfaces = hitSurfaces(probeRays, traceOnCpu(meshes, instances, probeRays), meshes, instances);

    it('finds a surface for every probe ray inside a closed room, with the normal toward the ray', () => {
        expect(surfaces.count).toBe(128);
        expect(surfaces.missCount).toBe(0);
        for (let surface = 0; surface < surfaces.count; surface++) {
            const ray = surfaces.rayIndex[surface];
            expect(dot(surfaces.normals.subarray(surface * 3, surface * 3 + 3), direction(probeRays, ray))).toBeLessThan(0);
        }
    });

    it('makes 0.3x the probe rays, each from a probe-ray hit, into the hemisphere of that hit', () => {
        const count = multiBounceRayCount(probeRays.length / RAY_FLOATS, RC_MULTI_BOUNCE_RATIO);
        expect(count).toBe(Math.round(128 * 0.3));
        const rays = makeBounceRays(surfaces, count, createRandom(5));
        expect(rays.length).toBe(count * RAY_FLOATS);
        expect(rayProblems('bounce', rays)).toEqual([]);
        const used = new Set<number>();
        for (let ray = 0; ray < count; ray++) {
            const origin = rays.subarray(ray * RAY_FLOATS, ray * RAY_FLOATS + 3);
            // The source: the surface whose offset origin this is.
            let source = -1;
            for (let surface = 0; surface < surfaces.count; surface++) {
                const expected = offsetRayOrigin(surfaces.positions.subarray(surface * 3, surface * 3 + 3), surfaces.normals.subarray(surface * 3, surface * 3 + 3));
                if (expected.every((value, axis) => Math.fround(value) === origin[axis])) source = surface;
            }
            expect(source).toBeGreaterThanOrEqual(0);
            used.add(source);
            expect(dot(direction(rays, ray), surfaces.normals.subarray(source * 3, source * 3 + 3))).toBeGreaterThan(0);
        }
        // Fewer rays than hits: each hit is used at most once.
        expect(used.size).toBe(count);
    });

    it('still makes the full count when there are fewer hits than rays', () => {
        const rays = makeBounceRays({ positions: surfaces.positions.slice(0, 6), normals: surfaces.normals.slice(0, 6), count: 2 }, 10, createRandom(5));
        expect(rays.length).toBe(10 * RAY_FLOATS);
        expect(makeBounceRays(surfaces, 0, createRandom(5)).length).toBe(0);
    });

    const white: [number, number, number] = [1, 1, 1];
    const sun = { type: 'directional' as const, color: white, intensity: 3, direction: [0, -1, 0] as [number, number, number] };
    const lamp = (position: [number, number, number], intensity = 40, range = 0) => ({ type: 'point' as const, color: white, intensity, position, range });

    it('with only the sun, aims the hits it reaches at it, unbounded, and gives the rest none', () => {
        const shadow = makeLightShadowRays(surfaces, rcLightsFromManifest([sun]), createRandom(9));
        // Only the floor (normal +y inside the room) faces the sun straight up; walls and the ceiling don't.
        let facingUp = 0;
        for (let surface = 0; surface < surfaces.count; surface++) if (surfaces.normals[surface * 3 + 1] > 0) facingUp++;
        expect(shadow.count).toBe(facingUp);
        expect(shadow.noLightCount).toBe(surfaces.count - facingUp);
        for (let ray = 0; ray < shadow.count; ray++) {
            expect(direction(shadow.rays, ray)).toEqual([0, 1, 0]);
            expect(shadow.rays[ray * RAY_FLOATS + 7]).toBe(Math.fround(SUN_SHADOW_RAY_LENGTH));
        }
    });

    it('gives every hit a ray when some light reaches it: the walls and ceiling the sun misses go to the lamp', () => {
        // A lamp in the middle of the closed room reaches every inside surface.
        const shadow = makeLightShadowRays(surfaces, rcLightsFromManifest([sun, lamp([0, 3, 0])]), createRandom(9));
        expect(shadow.count).toBe(surfaces.count);
        expect(shadow.noLightCount).toBe(0);
        expect(shadow.picksPerLight[0] + shadow.picksPerLight[1]).toBe(surfaces.count);
        expect(shadow.picksPerLight[1]).toBeGreaterThan(0);
        expect(rayProblems('shadow', shadow.rays)).toEqual([]);
        // A lamp ray ends at the lamp.
        for (let ray = 0; ray < shadow.count; ray++) {
            const base = ray * RAY_FLOATS;
            if (shadow.rays[base + 7] > 1e6) continue;
            const end = [0, 1, 2].map((axis) => shadow.rays[base + axis] + shadow.rays[base + 4 + axis] * shadow.rays[base + 7]);
            expect(Math.hypot(end[0], end[1] - 3, end[2])).toBeLessThan(1e-4);
        }
    });

    it('chooses lights in proportion to their unshadowed irradiance, and never one that cannot reach the hit', () => {
        // 4,000 copies of one floor point at the origin, facing up.
        const count = 4000;
        const floor = { positions: new Float64Array(count * 3), normals: Float64Array.from({ length: count * 3 }, (_, index) => (index % 3 === 1 ? 1 : 0)), count };
        const lights = rcLightsFromManifest([
            lamp([0, 2, 0], 30), // straight above, 2 m: 30 / 4 = 7.5
            lamp([0, 2, 0], 10), // the same place, a third as bright: 2.5
            lamp([0, 2, 0], 0), // zero intensity: never
            lamp([0, -2, 0], 1000), // below the floor (cos < 0): never
            lamp([0, 20, 0], 1000, 12), // out of its 12 m range: never
        ]);
        const shadow = makeLightShadowRays(floor, lights, createRandom(21));
        expect(shadow.count).toBe(count);
        expect(shadow.picksPerLight.slice(2)).toEqual([0, 0, 0]);
        expect(shadow.picksPerLight[0] / count).toBeCloseTo(0.75, 1);
        expect(shadow.picksPerLight[1] / count).toBeCloseTo(0.25, 1);
        // No light reaches: none of them can.
        const dark = makeLightShadowRays(floor, rcLightsFromManifest([lamp([0, 2, 0], 0), lamp([0, -2, 0])]), createRandom(21));
        expect(dark.count).toBe(0);
        expect(dark.noLightCount).toBe(count);
        expect(() => rcLightsFromManifest([])).toThrow(/no light/);
    });
});

describe('rc CSV rows and predictions', () => {
    const mode = (tracer: 'platform' | 'ours-tinybvh', raySet: string, kind: 'closest' | 'any', rays: number, ms: number, loadedMs?: number) => ({
        tracer,
        mode: kind,
        rays,
        raySet,
        timing: { medianMs: ms, minMs: ms, maxMs: ms, samplesMs: [ms] },
        timingSource: 'timestamp-query' as const,
        mraysPerSecond: rays / (ms * 1000),
        ...(loadedMs === undefined ? {} : { loaded: { medianMs: loadedMs, minMs: loadedMs, maxMs: loadedMs, dispatches: 100, seconds: 1, mraysPerSecond: rays / (loadedMs * 1000) } }),
    });
    const set = (name: string, kind: 'closest' | 'any', rays: number) => ({
        set: name, mode: kind, rays, hits: rays / 2, hitFraction: 0.5, oracleRays: 16, oracleMismatches: 0, oracleAmbiguous: 0, agreementRate: 1,
    });
    const scene = {
        scene: 'sponza',
        meshCount: 1, instanceCount: 1, uniqueTriangles: 10, instancedTriangles: 10,
        ours: null, tmbvh: null, agreement: [], skipped: [],
        rc: {
            camera: { name: 'a', position: [0, 0, 0], target: [1, 0, 0], fovYDeg: 75 },
            width: 1280, height: 720, tileSize: 16, tiles: 3600, probesPlaced: 3000, tilesMissed: 600,
            raysPerProbe: 64, multiBounceRatio: 0.3, seed: 1, lightCount: 2, lightHasSun: true,
            sets: [
                set('placement', 'closest', 3600),
                set('probe', 'closest', 192_000),
                set('bounce', 'closest', 57_600),
                { ...set('shadow-probe', 'any', 100_000), noLight: 20_000, raysPerHit: 0.9, picksPerLight: [90_000, 10_000] },
                set('shadow-bounce', 'any', 30_000),
            ],
        },
        modes: [
            mode('platform', 'placement', 'closest', 3600, 0.05, 0.04),
            mode('platform', 'probe', 'closest', 192_000, 2, 1.6),
            mode('platform', 'bounce', 'closest', 57_600, 0.6, 0.5),
            mode('platform', 'shadow-probe', 'any', 100_000, 1, 0.8),
            mode('platform', 'shadow-bounce', 'any', 30_000, 0.3, 0.25),
            mode('ours-tinybvh', 'probe', 'closest', 192_000, 2),
        ],
    } as unknown as SceneResult;

    it('sums probe, bounce and both shadow sets (not placement) at the loaded rates', () => {
        const predictions = rcPredictions(scene);
        // ours-tinybvh lacks the other per-frame sets, so it gets no prediction.
        expect(predictions.map((entry) => entry.tracer)).toEqual(['platform']);
        expect(predictions[0].source).toBe('loaded');
        expect(predictions[0].sets.map((entry) => entry.set)).toEqual(['probe', 'bounce', 'shadow-probe', 'shadow-bounce']);
        expect(predictions[0].totalMs).toBeCloseTo(1.6 + 0.5 + 0.8 + 0.25, 9);
    });

    it('writes rc rows with the set in the name and no probe-grid rows', () => {
        const data = { benchmark: 'trace-bench', device: { maxStorageBufferBindingSize: 1, maxBufferSize: 2 }, scenes: [scene], synthetic: [] } as unknown as TraceBenchData;
        const rows = traceBenchRows(data);
        const value = (metric: string) => rows.find((row) => row.metric === metric)?.value;
        expect(value('note/ray_set_rc')).toBe(1);
        expect(value('rc/probes_placed')).toBe(3000);
        expect(value('rc/probe/rays')).toBe(192_000);
        expect(value('rc/shadow-probe/hits_no_light')).toBe(20_000);
        expect(value('rc/shadow-probe/rays_per_hit')).toBe(0.9);
        expect(value('rc/shadow-probe/rays_to_light1')).toBe(10_000);
        expect(value('rc/probe/platform/closest/mrays_per_s')).toBeCloseTo(96, 9);
        expect(value('rc/probe/platform/closest/loaded_mrays_per_s')).toBeCloseTo(120, 9);
        expect(value('rc/probe/platform/closest/predicted_ms')).toBeCloseTo(1.6, 9);
        expect(value('rc/total/platform/predicted_ms')).toBeCloseTo(3.15, 9);
        expect(value('rc/total/rays')).toBe(192_000 + 57_600 + 100_000 + 30_000);
        expect(rows.some((row) => row.metric.startsWith('rays/') || row.metric.startsWith('probes/'))).toBe(false);
        expect(rows.some((row) => row.metric.startsWith('platform/closest'))).toBe(false);
    });
});
