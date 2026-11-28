/**
 * The scene-copy checker must catch faults. `checkReadback` is what the GPU
 * page (tools/gpu/pages/scene-copy) uses to judge the WGSL accessors, so a
 * checker that let a wrong answer through would let the page pass a broken
 * shader. Each case here starts from a correct readback, plants one fault
 * a broken accessor could produce, and requires the checker to report it.
 */

import { describe, expect, it } from 'vitest';
import { PlatformScene } from '../platform-scene.ts';
import { createFakeGpu, type FakeBuffer } from './fake-gpu.ts';
import { buildBasicScene, checkReadback, makeQueries, materialIndices, SceneModel } from './fixture.ts';
import type { InstanceQueryResult, SceneReadback } from './readback.ts';
import { SceneReader } from './scene-reader.ts';

function correctReadback() {
    const gpu = createFakeGpu();
    const scene = new PlatformScene(gpu.device);
    const model = new SceneModel(scene);
    buildBasicScene(model);
    const encoder = gpu.createEncoder();
    scene.encode(encoder);
    gpu.submit(encoder);
    const [vertices, tables] = scene.module.resources().map((resource) => (resource as GPUBufferBinding).buffer as unknown as FakeBuffer);
    const reader = new SceneReader(vertices.bytes, tables.bytes);
    const queries = makeQueries(model);
    const materials = materialIndices(model);
    return { model, reader, queries, materials, readback: reader.readAll(queries, materials) };
}

/** Index of the first query on a live cube triangle: it has normals and both uv sets. */
function cubeQuery(setup: ReturnType<typeof correctReadback>): number {
    const index = setup.queries.findIndex((query) => query.instance === setup.model.scene.instanceSlot([...setup.model.instances.keys()][0]));
    expect(index).toBeGreaterThanOrEqual(0);
    return index;
}

function withQuery(readback: SceneReadback, index: number, change: (query: InstanceQueryResult) => Partial<InstanceQueryResult>): SceneReadback {
    const queries = readback.queries.map((query, at) => (at === index ? { ...query, ...change(query) } : query));
    return { ...readback, queries };
}

describe('the scene-copy checker', () => {
    it('passes a correct readback', () => {
        const setup = correctReadback();
        expect(checkReadback(setup.model, setup.queries, setup.materials, setup.readback)).toEqual([]);
    });

    const faults: { name: string; expect: RegExp; plant: (setup: ReturnType<typeof correctReadback>) => SceneReadback }[] = [
        {
            name: 'a flipped shading normal',
            expect: /normalAt/,
            plant: (setup) => withQuery(setup.readback, cubeQuery(setup), (query) => ({ normalAt: query.normalAt.map((value) => -value) })),
        },
        {
            name: 'a slightly wrong uv',
            expect: /uv[01]/,
            plant: (setup) => withQuery(setup.readback, cubeQuery(setup), (query) => ({ uv: [query.uv[0] + 0.01, query.uv[1]] })),
        },
        {
            name: 'the wrong material',
            expect: /materialAt/,
            plant: (setup) => {
                const index = cubeQuery(setup);
                const other = setup.materials.find((material) => material !== setup.readback.queries[index].materialAt)!;
                return withQuery(setup.readback, index, () => ({ materialAt: other }));
            },
        },
        {
            name: 'an off-by-one vertex index',
            expect: /vertices/,
            plant: (setup) => withQuery(setup.readback, cubeQuery(setup), (query) => ({ vertices: [query.vertices[0] + 1, query.vertices[1], query.vertices[2]] })),
        },
        {
            name: 'the neighbouring triangle (off-by-one triangle index)',
            expect: /vertices|corners/,
            plant: (setup) => {
                const index = cubeQuery(setup);
                const query = setup.queries[index];
                return withQuery(setup.readback, index, () => setup.reader.query({ ...query, triangle: query.triangle + 1 }));
            },
        },
        {
            name: 'a transposed normal matrix',
            expect: /normal matrix/,
            plant: (setup) => {
                const index = setup.queries.findIndex((query) => query.instance === 1); // the turned, non-uniformly scaled cube
                return withQuery(setup.readback, index, (query) => ({
                    normalMatrix: [0, 1, 2].flatMap((column) => [0, 1, 2].map((row) => query.normalMatrix[row * 3 + column])),
                }));
            },
        },
        {
            name: 'a free slot that reports triangles',
            expect: /free slot/,
            plant: (setup) => {
                const index = setup.queries.findIndex((query) => query.instance >= setup.model.instances.size);
                return withQuery(setup.readback, index, () => ({ triangleCount: 12 }));
            },
        },
        {
            name: 'a wrong albedo',
            expect: /albedo/,
            plant: (setup) => ({
                ...setup.readback,
                materials: setup.readback.materials.map((material, at) => (at === 1 ? { ...material, albedo: [material.albedo[0] + 0.01, material.albedo[1], material.albedo[2]] } : material)),
            }),
        },
        {
            name: 'lights out of order (a point light among the directional ones)',
            expect: /directional lights must come first|kind/,
            plant: (setup) => ({ ...setup.readback, lights: [setup.readback.lights[2], ...setup.readback.lights.slice(1, 2), setup.readback.lights[0], ...setup.readback.lights.slice(3)] }),
        },
        {
            name: 'a wrong environment',
            expect: /environment kind/,
            plant: (setup) => ({ ...setup.readback, environment: { ...setup.readback.environment, kind: 1 } }),
        },
    ];

    for (const fault of faults) {
        it(`fails ${fault.name}`, () => {
            const setup = correctReadback();
            const failures = checkReadback(setup.model, setup.queries, setup.materials, fault.plant(setup));
            expect(failures.length).toBeGreaterThan(0);
            expect(failures.join('\n')).toMatch(fault.expect);
        });
    }
});
