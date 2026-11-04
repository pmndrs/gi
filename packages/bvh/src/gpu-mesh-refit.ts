/**
 * Refit of one mesh's tree (a BLAS) from moved vertices, on the GPU (stage
 * B1): for deforming meshes, whose triangles move but keep their
 * connectivity. The tree keeps its shape; its triangles and boxes follow the
 * vertex buffer the adapter supplies (a skinned or morphed mesh's output, say).
 *
 * Two steps in the caller's compute pass:
 *
 * 1. `meshRefitTriangles`, one thread per triangle slot (tree order): reads
 *    the slot's original triangle number, fetches its three corners from the
 *    vertex buffer, writes them into the tree's triangle record (so the
 *    tracer's watertight test sees the moved triangle) and writes the slot's
 *    box for the refit.
 * 2. The B1a refit (gpu-refit.ts) over the mesh's nodes: a bottom-up climb
 *    with a per-node counter, its subgroup pairing, and its repair pass.
 *
 * Any builder's tree works (the GPU LBVH's or tinybvh's): only the layout
 * matters. The refit takes the tree's shape from a CPU copy of the layout
 * (`setTree`), which a GPU-built tree gets once with `GpuLbvhBuilder.readLayout`;
 * the shape never changes while the mesh deforms, so that is a one-off at
 * build time, not a per-frame readback.
 *
 * A refit tree slowly gets worse as triangles move away from where it was
 * built. The refit's quality signal (`latestStats.relativeCost`, with
 * `diagnostics: true`) is the same as the top level's, so a caller can rebuild
 * (with the GPU LBVH) when it drifts, as B1a does at 1.25x.
 */

import { GPU_LAYOUT_TRIANGLE_BYTES, GPU_LAYOUT_TRIANGLE_WORD, type GpuLayoutMeshBvh } from './gpu-layout.ts';
import { GpuBvhRefit, type BvhRefitStats, type GpuBvhRefitOptions } from './gpu-refit.ts';
import { leafSlotsOf, parentSlotsOf, PRIMITIVE_BOX_FLOATS } from './refit-cpu.ts';
import { dispatchWorkgroups2d, WORKGROUP_INDEX_WGSL } from './gpu-workgroup-scan.ts';

const WORKGROUP_SIZE = 64;
const PARAM_WORDS = 8;
const USAGE = { COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80 } as const;

/** WGSL of the triangle step. Plain WGSL: the refit after it needs subgroups, this does not. */
export function meshRefitTrianglesWgsl(): string {
    const triangle = GPU_LAYOUT_TRIANGLE_WORD;
    return /* wgsl */ `
struct MeshRefitParams {
    triangleCount: u32,
    triangleWordOffset: u32,   // where the mesh's first triangle record starts, in words
    vertexCount: u32,
    positionStride: u32,       // floats from one vertex to the next
    positionOffset: u32,       // float index of vertex 0's x
    indexFormat: u32,          // 0: a plain triangle list, 16 or 32: index bits
    indexOffset: u32,          // in indices, not bytes
    unused: u32,
}

@group(0) @binding(0) var<storage, read_write> refitTriangles: array<u32>;
@group(0) @binding(1) var<storage, read> refitPositions: array<f32>;
@group(0) @binding(2) var<storage, read> refitIndices: array<u32>;
// Two per triangle slot, in tree order: (min.xyz, 0), (max.xyz, 0). The refit reads them.
@group(0) @binding(3) var<storage, read_write> refitBoxes: array<vec4f>;
@group(0) @binding(4) var<uniform> refitParams: MeshRefitParams;

const REFIT_WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;
const REFIT_TRIANGLE_WORDS: u32 = ${GPU_LAYOUT_TRIANGLE_BYTES / 4}u;
const REFIT_CORNER_STRIDE: u32 = ${triangle.vertex1 - triangle.vertex0}u;
const REFIT_ORIGINAL_INDEX: u32 = ${triangle.originalIndex}u;
${WORKGROUP_INDEX_WGSL}

fn refitVertexOf(original: u32, corner: u32) -> u32 {
    let entry = refitParams.indexOffset + original * 3u + corner;
    var vertex = original * 3u + corner;
    if (refitParams.indexFormat == 32u) {
        vertex = refitIndices[entry];
    } else if (refitParams.indexFormat == 16u) {
        vertex = (refitIndices[entry >> 1u] >> ((entry & 1u) * 16u)) & 0xffffu;
    }
    // The indices were checked when the tree was built; this only keeps a bad buffer's reads in range.
    return min(vertex, max(refitParams.vertexCount, 1u) - 1u);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn meshRefitTriangles(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let slot = flatWorkgroupIndex(workgroup, groups) * REFIT_WORKGROUP_SIZE + localIndex;
    if (slot >= refitParams.triangleCount) {
        return;
    }
    let base = refitParams.triangleWordOffset + slot * REFIT_TRIANGLE_WORDS;
    let original = refitTriangles[base + REFIT_ORIGINAL_INDEX];
    var lower = vec3f(0x1.fffffep+127f);
    var upper = vec3f(-0x1.fffffep+127f);
    for (var corner = 0u; corner < 3u; corner += 1u) {
        let at = refitParams.positionOffset + refitVertexOf(original, corner) * refitParams.positionStride;
        let position = vec3f(refitPositions[at], refitPositions[at + 1u], refitPositions[at + 2u]);
        let word = base + corner * REFIT_CORNER_STRIDE;
        refitTriangles[word] = bitcast<u32>(position.x);
        refitTriangles[word + 1u] = bitcast<u32>(position.y);
        refitTriangles[word + 2u] = bitcast<u32>(position.z);
        lower = min(lower, position);
        upper = max(upper, position);
    }
    refitBoxes[slot * 2u] = vec4f(lower, 0.0);
    refitBoxes[slot * 2u + 1u] = vec4f(upper, 0.0);
}
`;
}

/** The moved vertices, on the GPU. Buffers need `STORAGE` usage. */
export interface MeshRefitVertices {
    readonly positionBuffer: GPUBuffer;
    readonly vertexCount: number;
    /** Floats from one vertex to the next. Default 3. */
    readonly positionStrideFloats?: number;
    /** Float index of vertex 0's x. Default 0. */
    readonly positionOffsetFloats?: number;
    /** The mesh's indices, as when the tree was built; omit for a plain triangle list. */
    readonly indexBuffer?: GPUBuffer | null;
    readonly indexFormat?: 'uint16' | 'uint32';
    readonly indexOffset?: number;
}

/** Where the mesh's tree lives on the GPU: its own buffers, or its place in a scene's shared ones. */
export interface MeshRefitTarget {
    /** Nodes, the mesh's root at `nodeWordOffset`. */
    readonly nodes: GPUBuffer;
    /** Triangle records, the mesh's first at `triangleWordOffset`. */
    readonly triangles: GPUBuffer;
}

/** Refits one mesh's tree from moved vertices, every frame if need be. Needs `subgroups` (for the refit). */
export class GpuMeshRefit {
    static isSupported(device: GPUDevice): boolean {
        return GpuBvhRefit.isSupported(device);
    }

    private readonly device: GPUDevice;
    private readonly label: string;
    private readonly refit: GpuBvhRefit;
    private pipeline: GPUComputePipeline | null = null;
    private readonly params: GPUBuffer;
    private boxes: GPUBuffer;
    private triangleCount = 0;
    private triangleWordOffset = 0;
    private group: { group: GPUBindGroup; key: readonly unknown[] } | null = null;

    constructor(device: GPUDevice, options: GpuBvhRefitOptions = {}) {
        this.device = device;
        this.label = options.label ?? 'mesh refit';
        this.refit = new GpuBvhRefit(device, { ...options, label: this.label });
        this.params = device.createBuffer({ label: `${this.label} params`, size: PARAM_WORDS * 4, usage: USAGE.UNIFORM | USAGE.COPY_DST });
        this.boxes = device.createBuffer({ label: `${this.label} boxes`, size: 64, usage: USAGE.STORAGE });
    }

    async prepare(): Promise<void> {
        if (this.pipeline) return;
        const module = this.device.createShaderModule({ label: this.label, code: meshRefitTrianglesWgsl() });
        const [pipeline] = await Promise.all([
            this.device.createComputePipelineAsync({ label: this.label, layout: 'auto', compute: { module, entryPoint: 'meshRefitTriangles' } }),
            this.refit.prepare(),
        ]);
        this.pipeline = pipeline;
    }

    get ready(): boolean {
        return this.pipeline !== null && this.refit.ready;
    }

    /** The refit's stats with `diagnostics: true` (see gpu-refit.ts). */
    get latestStats(): BvhRefitStats | null {
        return this.refit.latestStats;
    }

    /**
     * Takes the tree's shape from a CPU copy of its layout, and where the mesh
     * sits in the GPU buffers (in words; 0 for a tree in its own buffers).
     */
    setTree(layout: GpuLayoutMeshBvh, placement: { nodeWordOffset?: number; triangleWordOffset?: number; generation?: number } = {}): void {
        const words = new Uint32Array(layout.nodes, 0, layout.nodeCount * 16);
        const nodeWordOffset = placement.nodeWordOffset ?? 0;
        this.refit.setShape({
            nodeWordOffset,
            nodeCount: layout.nodeCount,
            parentSlots: parentSlotsOf(words, 0, layout.nodeCount, layout.parents),
            leafSlots: leafSlotsOf(words, 0, layout.nodeCount),
            generation: placement.generation ?? 0,
        });
        this.triangleCount = layout.triangleCount;
        const boxBytes = Math.max(64, layout.triangleCount * PRIMITIVE_BOX_FLOATS * 4);
        if (this.boxes.size < boxBytes) {
            this.boxes.destroy();
            this.boxes = this.device.createBuffer({ label: `${this.label} boxes`, size: boxBytes, usage: USAGE.STORAGE });
        }
        this.triangleWordOffset = placement.triangleWordOffset ?? 0;
    }

    /** Records the triangle step and the refit into `pass`. The vertices must be written by an earlier command. */
    encode(pass: GPUComputePassEncoder, target: MeshRefitTarget, vertices: MeshRefitVertices): void {
        if (!this.pipeline) throw new Error('@pmndrs/bvh: call prepare() before encode()');
        if (this.triangleCount === 0) return;
        const indexFormat = vertices.indexBuffer ? (vertices.indexFormat === 'uint16' ? 16 : 32) : 0;
        this.device.queue.writeBuffer(
            this.params,
            0,
            new Uint32Array([
                this.triangleCount,
                this.triangleWordOffset,
                vertices.vertexCount,
                vertices.positionStrideFloats ?? 3,
                vertices.positionOffsetFloats ?? 0,
                indexFormat,
                vertices.indexOffset ?? 0,
                0,
            ]),
        );
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup(target.triangles, vertices.positionBuffer, vertices.indexBuffer ?? vertices.positionBuffer));
        dispatchWorkgroups2d(pass, Math.ceil(this.triangleCount / WORKGROUP_SIZE), this.device.limits.maxComputeWorkgroupsPerDimension);
        this.refit.encode(pass, target.nodes, this.boxes);
    }

    /** Stats readback (diagnostics only), as for the top-level refit: poll at a frame's start, copy after the submit's encode. */
    pollStats(): void {
        this.refit.pollStats();
    }

    copyStats(encoder: GPUCommandEncoder): void {
        this.refit.copyStats(encoder);
    }

    dispose(): void {
        this.refit.dispose();
        this.params.destroy();
        this.boxes.destroy();
    }

    private bindGroup(triangles: GPUBuffer, positions: GPUBuffer, indices: GPUBuffer): GPUBindGroup {
        const key = [triangles, positions, indices, this.boxes];
        if (this.group && this.group.key.every((entry, index) => entry === key[index])) return this.group.group;
        if (!this.pipeline) throw new Error('@pmndrs/bvh: call prepare() first');
        const group = this.device.createBindGroup({
            label: this.label,
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: triangles } },
                { binding: 1, resource: { buffer: positions } },
                { binding: 2, resource: { buffer: indices } },
                { binding: 3, resource: { buffer: this.boxes } },
                { binding: 4, resource: { buffer: this.params } },
            ],
        });
        this.group = { group, key };
        return group;
    }
}
