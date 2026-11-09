/**
 * The page's test kernels: thin entry points around the hash grid's shared
 * WGSL (group 1, `HashGridCache.wgsl`), with the page's inputs and results in
 * group 2. They call the production functions as the RC3.2 passes will;
 * nothing here is in the production shaders.
 */

import { HASH_GRID_CACHE_GROUP_ENTRIES } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-wgsl.ts';
import type { HashGridCache } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-cache.ts';
import type { ComputeKernel, KernelBuilder } from '../../../../packages/ddgi-core/src/platform/contract/wgsl.ts';
import { WORLD_CACHE_PROBE_RAYS_WGSL } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/world-cache.ts';

export const TEST_WORKGROUP_SIZE = 64;

/** Words per test hit in the input: position xyz, hit distance, direction xyz, unused. */
export const HIT_WORDS = 8;

/** Results per thread, by kernel. */
export const RESULT_WORDS = { hashes: 2, descriptors: 4, insert: 3, find: 2 } as const;

/** Counter words: how many inserts found a new tile. */
export const COUNTER = { newTiles: 0 } as const;

export const TEST_ENTRY_POINT = {
    hashes: 'testHashes',
    descriptors: 'testDescriptors',
    insert: 'testInsert',
    find: 'testFind',
    touch: 'testTouch',
} as const;

export type TestKernelName = keyof typeof TEST_ENTRY_POINT;

const SHADER_STAGE_COMPUTE = 0x4;

const IO_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache-hash-grid io',
    entries: [
        { binding: 0, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
    ],
};

const TEST_WGSL = /* wgsl */ `
struct TestParams {
    item_count: u32,     // hits (or hash inputs)
    thread_count: u32,   // threads of testInsert; thread i inserts hit i % item_count
    unused0: u32,
    unused1: u32,
}

@group(2) @binding(0) var<uniform> test_params: TestParams;
@group(2) @binding(1) var<storage, read> test_input: array<u32>;
@group(2) @binding(2) var<storage, read_write> test_results: array<u32>;
@group(2) @binding(3) var<storage, read_write> test_counters: array<atomic<u32>>;

fn testHitData(hit_index: u32) -> HashGridCache_Data {
    let base = hit_index * ${HIT_WORDS}u;
    let position = bitcast<vec3f>(vec3u(test_input[base], test_input[base + 1u], test_input[base + 2u]));
    let hit_distance = bitcast<f32>(test_input[base + 3u]);
    let direction = bitcast<vec3f>(vec3u(test_input[base + 4u], test_input[base + 5u], test_input[base + 6u]));
    return HashGridCache_Data(g_HashGridCacheConstants.eye_position, position, direction, hit_distance);
}

@compute @workgroup_size(${TEST_WORKGROUP_SIZE})
fn ${TEST_ENTRY_POINT.hashes}(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= test_params.item_count) {
        return;
    }
    test_results[id.x * 2u] = pcgHash(test_input[id.x]);
    test_results[id.x * 2u + 1u] = xxHash(test_input[id.x]);
}

@compute @workgroup_size(${TEST_WORKGROUP_SIZE})
fn ${TEST_ENTRY_POINT.descriptors}(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= test_params.item_count) {
        return;
    }
    let desc = HashGridCache_GetDesc(testHitData(id.x));
    test_results[id.x * 4u] = desc.bucket_index;
    test_results[id.x * 4u + 1u] = desc.tile_hash;
    test_results[id.x * 4u + 2u] = desc.cell_offset.x;
    test_results[id.x * 4u + 3u] = desc.cell_offset.y;
}

// The cache half of PopulateScreenProbesHandleHit (gi1.comp:986-1036) without the visibility
// list: insert, touch the tile, and a new tile joins the packed list with its cells cleared.
@compute @workgroup_size(${TEST_WORKGROUP_SIZE})
fn ${TEST_ENTRY_POINT.insert}(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= test_params.thread_count) {
        return;
    }
    let inserted = HashGridCache_InsertCell(testHitData(id.x % test_params.item_count));
    if (inserted.cell_index != kGI1_InvalidId) {
        HashGridCache_TouchTile(inserted.tile_index);
        if (inserted.is_new_tile) {
            HashGridCache_AddNewTile(inserted.tile_index);
            atomicAdd(&test_counters[${COUNTER.newTiles}u], 1u);
        }
    }
    test_results[id.x * 3u] = inserted.cell_index;
    test_results[id.x * 3u + 1u] = inserted.tile_index;
    test_results[id.x * 3u + 2u] = select(0u, 1u, inserted.is_new_tile);
}

@compute @workgroup_size(${TEST_WORKGROUP_SIZE})
fn ${TEST_ENTRY_POINT.find}(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= test_params.item_count) {
        return;
    }
    let found = HashGridCache_FindCell(testHitData(id.x));
    test_results[id.x * 2u] = found.cell_index;
    test_results[id.x * 2u + 1u] = found.tile_index;
}

// A reader touching what it finds (Capsaicin's reflections, gi1.comp:2882-2888).
@compute @workgroup_size(${TEST_WORKGROUP_SIZE})
fn ${TEST_ENTRY_POINT.touch}(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= test_params.item_count) {
        return;
    }
    let found = HashGridCache_FindCell(testHitData(id.x));
    if (found.cell_index != kGI1_InvalidId) {
        HashGridCache_TouchTile(found.tile_index);
    }
}
`;

/** Words the probe-ray reader writes for one ray. */
export const PROBE_RAY_READ_WORDS = 14;

/**
 * Reads ray `test_ray.index` and the count through `WorldCacheProbeRays` (the
 * world cache's struct), as words: count, origin xyz, flags, direction xyz
 * (decoded), hitT, instance, triangle, front face, barycentrics xy.
 */
const PROBE_RAY_READER_WGSL = /* wgsl */ `
${WORLD_CACHE_PROBE_RAYS_WGSL}
struct TestRay {
    index: u32,
    unused0: u32,
    unused1: u32,
    unused2: u32,
}
@group(1) @binding(0) var<uniform> test_ray: TestRay;
@group(1) @binding(1) var<storage, read> test_probe_rays: WorldCacheProbeRays;
@group(1) @binding(2) var<storage, read_write> test_ray_words: array<u32>;

@compute @workgroup_size(1)
fn testReadProbeRay() {
    let ray = test_probe_rays.rays[test_ray.index];
    let direction = worldCacheProbeRayDirection(ray);
    test_ray_words[0] = test_probe_rays.count;
    test_ray_words[1] = bitcast<u32>(ray.origin.x);
    test_ray_words[2] = bitcast<u32>(ray.origin.y);
    test_ray_words[3] = bitcast<u32>(ray.origin.z);
    test_ray_words[4] = ray.flags;
    test_ray_words[5] = bitcast<u32>(direction.x);
    test_ray_words[6] = bitcast<u32>(direction.y);
    test_ray_words[7] = bitcast<u32>(direction.z);
    test_ray_words[8] = bitcast<u32>(ray.hitT);
    test_ray_words[9] = ray.hitInstance;
    test_ray_words[10] = ray.hitTriangle;
    test_ray_words[11] = ray.hitFrontFace;
    test_ray_words[12] = bitcast<u32>(ray.hitBarycentrics.x);
    test_ray_words[13] = bitcast<u32>(ray.hitBarycentrics.y);
}
`;

/** The reader kernel: bind a uniform with the ray index, the probe-ray buffer and a result buffer. */
export async function createProbeRayReader(kernels: KernelBuilder): Promise<ComputeKernel> {
    return kernels.computeKernel({
        label: 'radiance-cache-hash-grid probe-ray reader',
        modules: [],
        code: PROBE_RAY_READER_WGSL,
        entryPoint: 'testReadProbeRay',
        ownGroups: [
            {
                label: 'probe-ray reader',
                entries: [
                    { binding: 0, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
                    { binding: 1, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 2, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
                ],
            },
        ],
    });
}

/** The page's IO buffers, shared by every test kernel. */
export interface TestIo {
    readonly params: GPUBuffer;
    readonly input: GPUBuffer;
    readonly results: GPUBuffer;
    readonly counters: GPUBuffer;
}

export function createTestIo(device: GPUDevice, maxInputWords: number, maxResultWords: number): TestIo {
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    return {
        params: device.createBuffer({ label: 'test params', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
        input: device.createBuffer({ label: 'test input', size: maxInputWords * 4, usage: storage }),
        results: device.createBuffer({ label: 'test results', size: maxResultWords * 4, usage: storage }),
        counters: device.createBuffer({ label: 'test counters', size: 256, usage: storage }),
    };
}

/** A compiled test kernel with its two groups bound. */
export interface TestKernel {
    readonly kernel: ComputeKernel;
    readonly cacheGroup: GPUBindGroup;
    readonly ioGroup: GPUBindGroup;
}

export async function createTestKernels(device: GPUDevice, kernels: KernelBuilder, cache: HashGridCache, io: TestIo): Promise<Record<TestKernelName, TestKernel>> {
    const code = `${cache.wgsl(1)}\n${TEST_WGSL}`;
    const names = Object.keys(TEST_ENTRY_POINT) as TestKernelName[];
    const built = await Promise.all(
        names.map((name) =>
            kernels.computeKernel({
                label: `radiance-cache-hash-grid ${name}`,
                modules: [],
                code,
                entryPoint: TEST_ENTRY_POINT[name],
                ownGroups: [{ label: 'hash grid', entries: HASH_GRID_CACHE_GROUP_ENTRIES }, IO_GROUP],
            }),
        ),
    );
    const result = {} as Record<TestKernelName, TestKernel>;
    names.forEach((name, index) => {
        const kernel = built[index];
        result[name] = {
            kernel,
            cacheGroup: cache.createBindGroup(kernel.ownLayouts[0]),
            ioGroup: device.createBindGroup({
                label: `radiance-cache-hash-grid ${name} io`,
                layout: kernel.ownLayouts[1],
                entries: [
                    { binding: 0, resource: { buffer: io.params } },
                    { binding: 1, resource: { buffer: io.input } },
                    { binding: 2, resource: { buffer: io.results } },
                    { binding: 3, resource: { buffer: io.counters } },
                ],
            }),
        };
    });
    return result;
}
