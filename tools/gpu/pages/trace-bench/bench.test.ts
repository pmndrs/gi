/**
 * Tests of the benchmark's pure parts: statistics, the ray generators, the
 * glTF reader, the triangle conversion and the CSV rows. No GPU needed.
 */

import { describe, expect, it } from 'vitest';
import {
    buildMeshBvh,
    HIT_RECORD_BYTES,
    RAY_FLOATS,
    type MeshGeometry,
} from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import { createRandom, randomTriangleSoup } from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';
import { BenchConfigError, parseBenchConfig, type BenchProfileName } from './config.ts';
import { browserFromUserAgent, deviceNameFromAdapter, formatValue, repeatedRunRows, traceBenchRows } from './csv-rows.ts';
import { extractGltfGeometry, modelTransformMatrix, parseGlb, type GltfJson } from './gltf-geometry.ts';
import { makeProbeGrid, makeProbeRays, makeShadowRays, makeSphereDirections, probePosition, worldBounds } from './rays.ts';
import type { TraceBenchData } from './results.ts';
import { compareBlocked, compareClosestDistances, median, megaRaysPerSecond } from './statistics.ts';
import { makeSphereMesh } from './synthetic.ts';
import { convertWatertightToMollerTrumbore } from './triangle-conversion.ts';

describe('statistics', () => {
    it('takes the median of odd and even counts', () => {
        expect(median([3, 1, 2])).toBe(2);
        expect(median([4, 1, 3, 2])).toBe(2.5);
    });

    it('counts hit/miss and distance disagreements separately', () => {
        const a = Float32Array.from([-1, 5, 5, 10, -1]);
        const b = Float32Array.from([-1, 5.001, 6, -1, 3]);
        const agreement = compareClosestDistances(a, b);
        expect(agreement.agree).toBe(2); // both miss, and 5 vs 5.001 is inside 1e-3 relative
        expect(agreement.distanceMismatch).toBe(1);
        expect(agreement.hitMissMismatch).toBe(2);
        expect(agreement.rate).toBeCloseTo(2 / 5);
    });

    it('compares blocked flags', () => {
        expect(compareBlocked(Uint32Array.from([1, 0, 1]), Uint32Array.from([1, 1, 1])).rate).toBeCloseTo(2 / 3);
    });

    it('converts to million rays per second', () => {
        expect(megaRaysPerSecond(2_000_000, 20)).toBeCloseTo(100);
    });
});

describe('config', () => {
    const defaults: { scenes: string[]; profile: BenchProfileName } = { scenes: ['furnished-room'], profile: 'quick' };

    it('uses the profile when the URL is empty', () => {
        const config = parseBenchConfig('', defaults);
        expect(config.raysPerProbe).toBe(64);
        expect(config.scenes).toEqual(['furnished-room']);
        expect(config.tracers).toEqual(['ours-mt', 'ours-wt', 'tmbvh']);
    });

    it('lets the URL override the profile', () => {
        const config = parseBenchConfig('?scene=sponza,bistro-interior&tracer=tmbvh&rays=16&profile=full&iterations=3', defaults);
        expect(config.scenes).toEqual(['sponza', 'bistro-interior']);
        expect(config.tracers).toEqual(['tmbvh']);
        expect(config.raysPerProbe).toBe(16);
        expect(config.targetRays).toBe(2_097_152);
        expect(config.timedDispatches).toBe(3);
    });

    it('keeps a dispatch under 400 ms unless told otherwise', () => {
        expect(parseBenchConfig('', defaults).maxDispatchMs).toBe(400);
        expect(parseBenchConfig('?maxDispatchMs=0', defaults).maxDispatchMs).toBe(0);
    });

    it('rejects nonsense with a message', () => {
        expect(() => parseBenchConfig('?rays=0', defaults)).toThrow(BenchConfigError);
        expect(() => parseBenchConfig('?tracer=nope', defaults)).toThrow(/ours-mt/);
    });
});

describe('probe rays', () => {
    it('fills the box with a grid close to the requested probe count', () => {
        const grid = makeProbeGrid({ min: [0, 0, 0], max: [40, 10, 20] }, 1000);
        expect(grid.probeCount).toBeGreaterThan(800);
        expect(grid.probeCount).toBeLessThan(1250);
        // Cells are roughly cubic.
        expect(grid.spacing[0] / grid.spacing[1]).toBeGreaterThan(0.7);
        expect(grid.spacing[0] / grid.spacing[1]).toBeLessThan(1.4);
        const first = probePosition(grid, 0);
        const last = probePosition(grid, grid.probeCount - 1);
        for (let axis = 0; axis < 3; axis++) {
            expect(first[axis]).toBeGreaterThan(0);
            expect(last[axis]).toBeLessThan([40, 10, 20][axis]);
        }
    });

    it('makes the same unit directions every time', () => {
        const a = makeSphereDirections(64, 7);
        const b = makeSphereDirections(64, 7);
        expect(Array.from(a)).toEqual(Array.from(b));
        for (let i = 0; i < 64; i++) expect(Math.hypot(a[i * 3], a[i * 3 + 1], a[i * 3 + 2])).toBeCloseTo(1, 6);
    });

    it('packs every probe times every direction in the GiRay layout', () => {
        const grid = makeProbeGrid({ min: [0, 0, 0], max: [2, 2, 2] }, 8);
        const directions = makeSphereDirections(4, 1);
        const rays = makeProbeRays(grid, directions);
        expect(rays.length).toBe(grid.probeCount * 4 * RAY_FLOATS);
        // Ray 5 belongs to probe 1 and uses direction 1.
        const base = 5 * RAY_FLOATS;
        expect(Array.from(rays.slice(base, base + 3))).toEqual(probePosition(grid, 1).map(Math.fround));
        expect(Array.from(rays.slice(base + 4, base + 7))).toEqual(Array.from(directions.slice(3, 6)));
    });
});

describe('shadow rays', () => {
    // One floor triangle in the y = 0 plane, facing up, seen by a ray going straight down.
    const floor = {
        positions: Float32Array.from([-10, 0, -10, 10, 0, -10, 0, 0, 10]),
        indices: Uint32Array.from([0, 2, 1]),
    };
    const instance = { meshIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };
    const downRay = Float32Array.from([0, 5, 0, 0, 0, -1, 0, 1e30]);

    function hitRecord(t: number): ArrayBuffer {
        const buffer = new ArrayBuffer(HIT_RECORD_BYTES);
        new Float32Array(buffer)[0] = t;
        new Uint32Array(buffer).set([0, 0], 1); // instance 0, triangle 0
        return buffer;
    }

    it('starts the shadow ray just above the surface and aims it at the sun', () => {
        const shadow = makeShadowRays({
            closestRays: downRay,
            hitRecords: hitRecord(5),
            meshes: [floor],
            instances: [instance],
            towardSun: [0, 1, 0],
            normalOffset: 0.01,
        });
        expect(shadow.count).toBe(1);
        expect(shadow.hitCount).toBe(1);
        expect(shadow.rays[0]).toBeCloseTo(0, 6);
        expect(shadow.rays[1]).toBeCloseTo(0.01, 6);
        expect(Array.from(shadow.rays.slice(4, 7))).toEqual([0, 1, 0]);
    });

    it('skips misses and surfaces that face away from the sun', () => {
        const miss = makeShadowRays({ closestRays: downRay, hitRecords: hitRecord(-1), meshes: [floor], instances: [instance], towardSun: [0, 1, 0], normalOffset: 0.01 });
        expect(miss.count).toBe(0);
        expect(miss.hitCount).toBe(0);
        const away = makeShadowRays({ closestRays: downRay, hitRecords: hitRecord(5), meshes: [floor], instances: [instance], towardSun: [0, -1, 0], normalOffset: 0.01 });
        expect(away.count).toBe(0);
        expect(away.facingAwayCount).toBe(1);
    });
});

describe('world bounds', () => {
    it('applies each instance transform', () => {
        const mesh = { positions: Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0]), indices: Uint32Array.from([0, 1, 2]) };
        const moved = { meshIndex: 0, objectToWorld: [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 10, 0, 0, 1] };
        const box = worldBounds([mesh], [moved]);
        expect(box.min).toEqual([10, 0, 0]);
        expect(box.max).toEqual([12, 2, 0]);
    });
});

describe('glTF reader', () => {
    /** Two nodes using one mesh with two primitives; one node is translated and mirrored. */
    function tinyModel(): { gltf: GltfJson; binary: Uint8Array } {
        const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1]);
        const indexA = new Uint16Array([0, 1, 2]);
        const indexB = new Uint16Array([0, 1, 2]);
        const binary = new Uint8Array(positions.byteLength + 16);
        binary.set(new Uint8Array(positions.buffer), 0);
        binary.set(new Uint8Array(indexA.buffer), 72);
        binary.set(new Uint8Array(indexB.buffer), 80);
        const gltf: GltfJson = {
            scene: 0,
            scenes: [{ nodes: [0, 1] }],
            nodes: [{ mesh: 0 }, { mesh: 0, translation: [5, 0, 0], scale: [-1, 1, 1] }],
            meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 2 }, { attributes: { POSITION: 1 }, indices: 3 }] }],
            accessors: [
                { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
                { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
                { bufferView: 2, componentType: 5123, count: 3, type: 'SCALAR' },
                { bufferView: 3, componentType: 5123, count: 3, type: 'SCALAR' },
            ],
            bufferViews: [
                { buffer: 0, byteOffset: 0, byteLength: 36 },
                { buffer: 0, byteOffset: 36, byteLength: 36 },
                { buffer: 0, byteOffset: 72, byteLength: 6 },
                { buffer: 0, byteOffset: 80, byteLength: 6 },
            ],
        };
        return { gltf, binary };
    }

    it('merges a mesh\'s primitives and keeps one instance per node', () => {
        const { gltf, binary } = tinyModel();
        const geometry = extractGltfGeometry(gltf, [binary]);
        expect(geometry.meshes).toHaveLength(1);
        expect(geometry.meshes[0].indices.length).toBe(6);
        // The second primitive's indices were shifted past the first primitive's three vertices.
        expect(Array.from(geometry.meshes[0].indices)).toEqual([0, 1, 2, 3, 4, 5]);
        expect(geometry.instances).toHaveLength(2);
        expect(geometry.instances[1].objectToWorld.slice(12, 15)).toEqual([5, 0, 0]);
        expect(geometry.instances[1].objectToWorld[0]).toBe(-1);
    });

    it('applies the root transform on top of the node transforms', () => {
        const { gltf, binary } = tinyModel();
        const root = modelTransformMatrix({ position: [0, 1, 0], rotationDeg: [0, 0, 0], scale: 2 });
        const geometry = extractGltfGeometry(gltf, [binary], root);
        expect(geometry.instances[0].objectToWorld.slice(12, 15)).toEqual([0, 1, 0]);
        expect(geometry.instances[1].objectToWorld.slice(12, 15)).toEqual([10, 1, 0]);
        expect(geometry.instances[0].objectToWorld[0]).toBe(2);
    });

    it('reads a .glb container', () => {
        const { gltf, binary } = tinyModel();
        const json = new TextEncoder().encode(JSON.stringify({ ...gltf, buffers: [{ byteLength: binary.length }] }));
        const jsonPadded = new Uint8Array(Math.ceil(json.length / 4) * 4).fill(0x20);
        jsonPadded.set(json);
        const total = 12 + 8 + jsonPadded.length + 8 + binary.length;
        const file = new ArrayBuffer(total);
        const view = new DataView(file);
        view.setUint32(0, 0x46546c67, true);
        view.setUint32(4, 2, true);
        view.setUint32(8, total, true);
        view.setUint32(12, jsonPadded.length, true);
        view.setUint32(16, 0x4e4f534a, true);
        new Uint8Array(file, 20, jsonPadded.length).set(jsonPadded);
        const binHeader = 20 + jsonPadded.length;
        view.setUint32(binHeader, binary.length, true);
        view.setUint32(binHeader + 4, 0x004e4942, true);
        new Uint8Array(file, binHeader + 8, binary.length).set(binary);

        const parsed = parseGlb(file);
        expect(parsed.binary?.length).toBe(binary.length);
        expect(extractGltfGeometry(parsed.gltf, [parsed.binary!]).instances).toHaveLength(2);
    });

    it('refuses compressed meshes with a clear message', () => {
        const { gltf, binary } = tinyModel();
        expect(() => extractGltfGeometry({ ...gltf, extensionsRequired: ['KHR_draco_mesh_compression'] }, [binary])).toThrow(/KHR_draco/);
    });
});

describe('Moller-Trumbore conversion', () => {
    it('gives bit-identical triangles to a real Moller-Trumbore build', () => {
        const geometry: MeshGeometry = randomTriangleSoup(createRandom(11), 500, 20, 3);
        const watertight = buildMeshBvh(geometry, { triangleTest: 'watertight' });
        const reference = buildMeshBvh(geometry, { triangleTest: 'moller-trumbore' });
        const converted = watertight.triangles.slice(0);
        convertWatertightToMollerTrumbore(converted);
        expect(Array.from(new Uint32Array(converted))).toEqual(Array.from(new Uint32Array(reference.triangles)));
    });
});

describe('synthetic mesh', () => {
    it('is a closed sphere of a few hundred triangles', () => {
        const sphere = makeSphereMesh();
        expect(sphere.indices!.length / 3).toBe(2 * 16 * 11);
    });
});

describe('CSV rows', () => {
    const data = {
        benchmark: 'trace-bench',
        device: { maxStorageBufferBindingSize: 1, maxBufferSize: 2 },
        scenes: [
            {
                scene: 'sponza',
                meshCount: 1, instanceCount: 1, uniqueTriangles: 10, instancedTriangles: 10,
                probeCount: 1, raysPerProbe: 1, closestRays: 1, closestRaysRequested: 2, closestHits: 1, shadowRays: 1,
                probesRequested: 2, thinned: true,
                ours: null,
                tmbvh: null,
                modes: [
                    {
                        tracer: 'ours-wt', mode: 'closest', rays: 1000,
                        timing: { medianMs: 2, minMs: 1.5, maxMs: 3, samplesMs: [2] },
                        timingSource: 'timestamp-query', mraysPerSecond: 0.5,
                    },
                ],
                agreement: [{ pair: 'ours-wt vs tmbvh', closestRate: 0.999, closestRays: 1000, hitMissMismatch: 1, distanceMismatch: 0, anyRate: 0.99 }],
            },
        ],
        synthetic: [],
    } as unknown as TraceBenchData;

    it('names metrics tracer/mode/metric_unit', () => {
        const metrics = traceBenchRows(data).map((row) => `${row.scene} ${row.metric}`);
        expect(metrics).toContain('sponza ours-wt/closest/mrays_per_s');
        expect(metrics).toContain('sponza ours-wt/closest/gpu_ms_median');
        expect(metrics).toContain('sponza agreement/ours-wt_vs_tmbvh/closest_rate');
        expect(metrics).toContain('device limit/max_storage_buffer_binding_size_bytes');
    });

    it('says how many probes were traced and whether the grid was thinned', () => {
        const rows = traceBenchRows(data);
        const value = (metric: string) => rows.find((row) => row.metric === metric)?.value;
        expect(value('probes/count')).toBe(1);
        expect(value('probes/requested')).toBe(2);
        expect(value('probes/thinned')).toBe(1);
    });

    it('turns repeated runs into per-run rows and medians', () => {
        const run = (speed: number) => [
            { scene: 'sponza', metric: 'ours-wt/closest/mrays_per_s', value: speed },
            { scene: 'sponza', metric: 'rays/closest', value: 100 },
        ];
        const rows = repeatedRunRows([run(50), run(70), [...run(60), { scene: 'sponza', metric: 'only/in_run3', value: 1 }]]);
        const value = (metric: string) => rows.find((row) => row.metric === metric)?.value;
        expect(value('run1/ours-wt/closest/mrays_per_s')).toBe(50);
        expect(value('run3/only/in_run3')).toBe(1);
        expect(value('median/ours-wt/closest/mrays_per_s')).toBe(60);
        expect(value('median/rays/closest')).toBe(100);
        expect(value('median/runs')).toBe(3);
        // A metric missing from some runs gets no median: it would not be a median of all runs.
        expect(value('median/only/in_run3')).toBeUndefined();
    });

    it('formats numbers, browsers and devices', () => {
        expect(formatValue(3)).toBe('3');
        expect(formatValue(0.123456789)).toBe('0.123457');
        expect(browserFromUserAgent('Mozilla/5.0 HeadlessChrome/156.0.0.0 Safari/537.36', '156.0.8078.74')).toBe('Chrome 156.0.8078.74 headless');
        expect(browserFromUserAgent('Mozilla/5.0 HeadlessChrome/156.0.0.0 Safari/537.36')).toBe('Chrome 156.0.0.0 headless');
        expect(deviceNameFromAdapter({ vendor: 'nvidia', architecture: 'ampere', description: 'NVIDIA GeForce RTX 3060 Laptop GPU' })).toBe('RTX 3060 Laptop');
        expect(deviceNameFromAdapter({ vendor: 'intel', architecture: 'gen-12lp', description: 'Intel(R) UHD Graphics' })).toBe('Intel UHD');
        expect(deviceNameFromAdapter({ vendor: 'nvidia', architecture: 'ampere', description: '' })).toBe('RTX 3060 Laptop');
    });
});
