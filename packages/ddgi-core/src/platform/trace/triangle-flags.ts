/**
 * `gi/triangle-flags` (T4.2; docs/design/m4-materials.md §3): keeps the
 * tracer's triangle flag bit 0, "consult the material", right without
 * rebuilding any tree.
 *
 * A triangle is flagged when its material slot, in *any* instance of its
 * mesh, could let a ray pass (`giMaterialNeedsLookup`, material-test-wgsl.ts:
 * see-through, single-sided, a cutout or stochastic alpha that can open a
 * hole). It is a per-mesh union, so it can only cost an extra lookup, never
 * give a wrong answer: the walk's `giMaterialPasses` reads the instance's own
 * material.
 *
 * Two small passes, both reading the truth from the scene copy:
 *
 * 1. `giTriangleFlagsMasks`, one thread per instance slot: for each slot of
 *    the instance's material list that needs a lookup, OR the slot's bit
 *    into its mesh's 32-bit mask. Slots 0-30 have a bit each; bit 31 stands
 *    for every slot from 31 up (few meshes have that many, and the union
 *    rule makes sharing a bit safe). The list's last entry also stands in
 *    for slots past the end of the list (`giMaterialAt` clamps to it), so it
 *    sets its own bit and every bit above.
 * 2. `giTriangleFlagsWrite`, one thread per tree-ordered triangle of the
 *    meshes asked for: the triangle's original index (word 3 of its record)
 *    gives its material slot through the scene copy
 *    (`giSceneTriangleMaterialSlot`), and bit 0 of its flags word (word 7)
 *    becomes the mask's bit for that slot. Other flag bits are kept.
 *
 * **The install hook.** Whatever writes a mesh's triangle records into the
 * tracer's shared triangle buffer (a tree placed, a tree replaced, a refit
 * that rewrites whole records) leaves word 7 as the builder wrote it, which
 * is 0. So every install path calls
 *
 *     PlatformRayQueries.requestTriangleFlags(meshSlot: number): void
 *
 * (ray-queries.ts) for the mesh whose records it wrote. The rewrite is
 * recorded once per frame, at the end of `PlatformRayQueries.update`, after
 * every placement, in the same encoder; the GPU then runs it after the
 * frame's tree writes and copies and before any technique traces. A path
 * that writes triangle records *after* `update` in the frame (in a later
 * pass) must either keep word 7 or call {@link TriangleFlagsKernels.encode}
 * itself after its writes. tinybvh's and the GPU's H-PLOC/LBVH trees from
 * T2.12 go through `placeTrees`, which calls the hook; M3's T3.2/T3.3 install
 * paths call the same function (agreed with the M3 lead).
 *
 * The platform also asks for a rewrite when the scene's materials change in
 * a way that can flip a slot: a material edited, an instance given other
 * materials, an instance created or shown (scene-change regions that are not
 * moves; see `requestTriangleFlagsForChanges` in ray-queries.ts). Nothing
 * here compiles after `prepare`.
 */

import type { ComputeKernel, KernelBuilder, WgslModule } from '../contract/wgsl.ts';
import { SHADER_STAGE_COMPUTE } from '../kernel-builder.ts';
import { TRIANGLE_WORD, TRIANGLE_WORDS } from './layout.ts';
import { materialNeedsLookup, TRIANGLE_FLAG_CONSULT_MATERIAL, type WalkMaterial } from './material-test-wgsl.ts';

/** One mesh's tree-ordered triangles in the shared triangle buffer. */
export interface MeshTriangleRange {
    /** The mesh's slot in the scene copy (`giInstanceMesh`). */
    readonly meshSlot: number;
    /** First triangle record, in triangles. */
    readonly triangleOffset: number;
    readonly triangleCount: number;
}

/** Bits in a mesh's slot mask; the last one stands for every slot from it up. */
export const SLOT_MASK_BITS = 32;
const LAST_SLOT_BIT = SLOT_MASK_BITS - 1;

/** The mask bit of a material slot (CPU twin of `giTriangleFlagsSlotBit`). */
export function slotMaskBit(slot: number): number {
    return (1 << Math.min(slot, LAST_SLOT_BIT)) >>> 0;
}

/**
 * The bits one instance's material list sets in its mesh's mask (CPU twin of
 * pass 1): each entry that needs a lookup sets its slot's bit, and the last
 * entry also every bit above, since slots past the list's end use it.
 */
export function instanceSlotMask(materials: readonly WalkMaterial[]): number {
    let mask = 0;
    materials.forEach((material, position) => {
        if (!materialNeedsLookup(material)) return;
        const bit = slotMaskBit(position);
        // bit and every bit above it: ~(bit - 1).
        mask |= position === materials.length - 1 ? ~(bit - 1) : bit;
    });
    return mask >>> 0;
}

/**
 * Writes the consult flag over one mesh's tree-ordered triangle records on
 * the CPU (CPU twin of pass 2, for the reference tracers and tests).
 * `slotOf` gives a triangle's material slot from its original index.
 */
export function writeTriangleFlagsOnCpu(triangleWords: Uint32Array, range: Omit<MeshTriangleRange, 'meshSlot'>, mask: number, slotOf: (originalIndex: number) => number): void {
    for (let triangle = range.triangleOffset; triangle < range.triangleOffset + range.triangleCount; triangle++) {
        const base = triangle * TRIANGLE_WORDS;
        const consult = (mask & slotMaskBit(slotOf(triangleWords[base + TRIANGLE_WORD.originalIndex]))) !== 0;
        const flags = triangleWords[base + TRIANGLE_WORD.flags] & ~TRIANGLE_FLAG_CONSULT_MATERIAL;
        triangleWords[base + TRIANGLE_WORD.flags] = (flags | (consult ? TRIANGLE_FLAG_CONSULT_MATERIAL : 0)) >>> 0;
    }
}

const WORKGROUP_SIZE = 64;
/** Pass 1 loops over the instance slots, so a few hundred workgroups cover any scene (NOTES: no indirect dispatch for this). */
const MAX_MASK_WORKGROUPS = 256;
/** Bytes per range record: mesh slot, first triangle, count, first thread. */
const RANGE_BYTES = 16;
const RANGE_WORDS = RANGE_BYTES / 4;

/** Shared by both passes: the slot bits, as the CPU twins above. */
const SLOT_BITS_WGSL = /* wgsl */ `
// Slots 0-30 have a bit each; bit 31 stands for every slot from 31 up.
fn giTriangleFlagsSlotBit(slot: u32) -> u32 {
    return 1u << min(slot, ${LAST_SLOT_BIT}u);
}
`;

const MASKS_WGSL = /* wgsl */ `
${SLOT_BITS_WGSL}
// One 32-bit slot mask per mesh slot, cleared before this pass.
@group(1) @binding(0) var<storage, read_write> giFlagsMaskWords: array<atomic<u32>>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn giTriangleFlagsMasks(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) workgroupCount: vec3u) {
    let stride = workgroupCount.x * ${WORKGROUP_SIZE}u;
    for (var instance = id.x; instance < giInstanceCount(); instance += stride) {
        // Free slots report no triangles. Hidden instances count: the mask is a union,
        // and showing one later is a change that asks for a rewrite anyway.
        if (giInstanceTriangleCount(instance) == 0u) {
            continue;
        }
        let mesh = giInstanceMesh(instance);
        if (mesh >= arrayLength(&giFlagsMaskWords)) {
            continue;   // no tree of this mesh is being rewritten
        }
        // The same list read as giMaterialAt's (scene/wgsl.ts).
        let list = giSceneInstanceMaterialList(instance);
        var mask = 0u;
        for (var position = 0u; position < list.y; position += 1u) {
            if (giMaterialNeedsLookup(giMaterial(giSceneMaterialListEntry(list.x, position)))) {
                let bit = giTriangleFlagsSlotBit(position);
                // The last entry also covers the slots past the list's end, which use it.
                mask |= select(bit, ~(bit - 1u), position + 1u == list.y);
            }
        }
        if (mask != 0u) {
            atomicOr(&giFlagsMaskWords[mesh], mask);
        }
    }
}
`;

const WRITE_WGSL = /* wgsl */ `
${SLOT_BITS_WGSL}
struct GiFlagsRange {
    meshSlot: u32,
    triangleOffset: u32,   // first triangle record, in triangles
    triangleCount: u32,
    firstThread: u32,      // the ranges are numbered one after another
}

// The tracer's triangle records (12 words each), rewritten in place.
@group(1) @binding(0) var<storage, read_write> giFlagsTriangles: array<u32>;
@group(1) @binding(1) var<storage, read> giFlagsMasks: array<u32>;
// Exactly the ranges of this rewrite, none empty, in thread order.
@group(1) @binding(2) var<storage, read> giFlagsRanges: array<GiFlagsRange>;

const GI_FLAGS_TRIANGLE_WORDS: u32 = ${TRIANGLE_WORDS}u;
const GI_FLAGS_ORIGINAL_INDEX_WORD: u32 = ${TRIANGLE_WORD.originalIndex}u;
const GI_FLAGS_FLAGS_WORD: u32 = ${TRIANGLE_WORD.flags}u;
const GI_FLAGS_CONSULT: u32 = ${TRIANGLE_FLAG_CONSULT_MATERIAL}u;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn giTriangleFlagsWrite(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroupCount: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    // 2D dispatch, so a scene's worth of triangles stays under 65535 workgroups in x.
    let thread = (workgroup.y * workgroupCount.x + workgroup.x) * ${WORKGROUP_SIZE}u + localIndex;
    let rangeCount = arrayLength(&giFlagsRanges);
    let lastRange = giFlagsRanges[rangeCount - 1u];
    if (thread >= lastRange.firstThread + lastRange.triangleCount) {
        return;
    }
    // The last range starting at or before this thread.
    var low = 0u;
    var high = rangeCount - 1u;
    while (low < high) {
        let middle = (low + high + 1u) / 2u;
        if (giFlagsRanges[middle].firstThread <= thread) {
            low = middle;
        } else {
            high = middle - 1u;
        }
    }
    let range = giFlagsRanges[low];
    let base = (range.triangleOffset + thread - range.firstThread) * GI_FLAGS_TRIANGLE_WORDS;
    let originalIndex = giFlagsTriangles[base + GI_FLAGS_ORIGINAL_INDEX_WORD];
    let mask = giFlagsMasks[range.meshSlot];

    var consult = false;
    if (originalIndex < giMeshTriangleCount(range.meshSlot)) {
        consult = (mask & giTriangleFlagsSlotBit(giSceneTriangleMaterialSlot(range.meshSlot, originalIndex))) != 0u;
    } else {
        // The tree is older than the mesh's current shape (its new tree is still
        // building), so the index may not name a triangle now: consult whenever
        // any slot might need it, as the union rule allows.
        consult = mask != 0u;
    }
    let flagsWord = base + GI_FLAGS_FLAGS_WORD;
    giFlagsTriangles[flagsWord] = (giFlagsTriangles[flagsWord] & ~GI_FLAGS_CONSULT) | select(0u, GI_FLAGS_CONSULT, consult);
}
`;

const MASKS_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'gi/triangle-flags masks',
    entries: [{ binding: 0, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } }],
};

const WRITE_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'gi/triangle-flags write',
    entries: [
        { binding: 0, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
    ],
};

/** The mask buffer's smallest size, in mesh slots; it doubles when a higher slot is rewritten. */
const INITIAL_MASK_WORDS = 64;

/**
 * Encodes a buffer used by one encode is kept for before it is destroyed:
 * the platform asks hosts to submit each frame's encoder before the next
 * frame's `encode`, and one encode more covers a host that records two
 * encodes before one submit (destroying a buffer an unsubmitted encoder
 * uses would make that submit fail).
 */
const ENCODES_BEFORE_DESTROY = 2;

/**
 * The two passes of `gi/triangle-flags` and their small buffers.
 *
 * **Each `encode` brings its own inputs, in its own encoder.** The masks are
 * cleared with `encoder.clearBuffer` and the ranges go in a buffer made for
 * that encode (mapped at creation), not with `queue.writeBuffer`: a queue
 * write lands before *any* command buffer submitted after it, so two encodes
 * recorded before one submit would both have read the second one's ranges
 * (T4.2's review). A ranges buffer (16 bytes a mesh, made only in frames
 * that rewrite flags) and a mask buffer that was outgrown are destroyed
 * {@link ENCODES_BEFORE_DESTROY} encodes later, not at the next one, for the
 * same host.
 */
export class TriangleFlagsKernels {
    private readonly device: GPUDevice;
    private readonly label: string;
    private readonly materialTestModule: WgslModule;
    private masksKernel: ComputeKernel | null = null;
    private writeKernel: ComputeKernel | null = null;

    /** One 32-bit slot mask per mesh slot, cleared in each encode's own encoder before pass 1. */
    private masks: GPUBuffer;
    private maskCapacity = INITIAL_MASK_WORDS;
    /** Encodes so far, and the buffers to destroy once `ENCODES_BEFORE_DESTROY` more have passed. */
    private encodes = 0;
    private retired: { readonly buffer: GPUBuffer; readonly encode: number }[] = [];

    constructor(device: GPUDevice, materialTestModule: WgslModule, label = 'gi') {
        this.device = device;
        this.label = label;
        this.materialTestModule = materialTestModule;
        this.masks = this.createMasks(INITIAL_MASK_WORDS);
    }

    /** Whether both passes are compiled. */
    get ready(): boolean {
        return this.masksKernel !== null && this.writeKernel !== null;
    }

    /** Compiles both passes (once; the platform's `prepare` calls this). */
    async prepare(kernels: KernelBuilder): Promise<void> {
        if (this.ready) return;
        const modules = [this.materialTestModule];
        [this.masksKernel, this.writeKernel] = await Promise.all([
            kernels.computeKernel({ label: 'gi/triangle-flags masks', modules, code: MASKS_WGSL, entryPoint: 'giTriangleFlagsMasks', ownGroups: [MASKS_GROUP] }),
            kernels.computeKernel({ label: 'gi/triangle-flags write', modules, code: WRITE_WGSL, entryPoint: 'giTriangleFlagsWrite', ownGroups: [WRITE_GROUP] }),
        ]);
    }

    /**
     * Records both passes into `encoder`: the consult flag rewritten over each
     * of `meshes`' triangle ranges in `triangles` (the tracer's shared
     * triangle buffer, bound read-write). `instanceCount` (live instances)
     * sizes pass 1; it loops over every instance slot whatever it is.
     */
    encode(encoder: GPUCommandEncoder, triangles: GPUBuffer, meshes: readonly MeshTriangleRange[], instanceCount: number): void {
        if (!this.masksKernel || !this.writeKernel) throw new Error('@pmndrs/gi: gi/triangle-flags was used before prepare');
        this.encodes++;
        this.retired = this.retired.filter((entry) => {
            const done = this.encodes - entry.encode >= ENCODES_BEFORE_DESTROY;
            if (done) entry.buffer.destroy();
            return !done;
        });
        const nonEmpty = meshes.filter((mesh) => mesh.triangleCount > 0);
        if (nonEmpty.length === 0) return;

        // Masks for every mesh slot up to the highest one rewritten; pass 1 skips other meshes.
        this.ensureMaskWords(Math.max(...nonEmpty.map((mesh) => mesh.meshSlot)) + 1);
        encoder.clearBuffer(this.masks);

        // This encode's ranges, in a buffer of its own (see the class comment).
        const ranges = this.device.createBuffer({
            label: `${this.label} gi/triangle-flags ranges`,
            size: nonEmpty.length * RANGE_BYTES,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
        });
        const rangeWords = new Uint32Array(ranges.getMappedRange());
        let threads = 0;
        nonEmpty.forEach((mesh, index) => {
            rangeWords.set([mesh.meshSlot, mesh.triangleOffset, mesh.triangleCount, threads], index * RANGE_WORDS);
            threads += mesh.triangleCount;
        });
        ranges.unmap();
        this.retired.push({ buffer: ranges, encode: this.encodes });

        const masksGroup = this.device.createBindGroup({
            label: `${this.label} gi/triangle-flags masks`,
            layout: this.masksKernel.ownLayouts[0],
            entries: [{ binding: 0, resource: { buffer: this.masks } }],
        });
        const writeGroup = this.device.createBindGroup({
            label: `${this.label} gi/triangle-flags write`,
            layout: this.writeKernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: triangles } },
                { binding: 1, resource: { buffer: this.masks } },
                // Exactly this rewrite's ranges: the kernel takes their number from the binding's size.
                { binding: 2, resource: { buffer: ranges } },
            ],
        });

        const pass = encoder.beginComputePass({ label: `${this.label} gi/triangle-flags` });
        this.masksKernel.bind(pass);
        pass.setBindGroup(1, masksGroup);
        pass.dispatchWorkgroups(Math.min(MAX_MASK_WORKGROUPS, Math.max(1, Math.ceil(instanceCount / WORKGROUP_SIZE))));
        // Dispatches in one pass run in order with their storage writes visible to the next (WebGPU's
        // usage scopes are per dispatch), so pass 2 reads finished masks.
        this.writeKernel.bind(pass);
        pass.setBindGroup(1, writeGroup);
        const workgroups = Math.ceil(threads / WORKGROUP_SIZE);
        const x = Math.min(workgroups, this.device.limits.maxComputeWorkgroupsPerDimension);
        pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
        pass.end();
    }

    dispose(): void {
        for (const entry of this.retired) entry.buffer.destroy();
        this.retired = [];
        this.masks.destroy();
    }

    /** A bigger mask buffer when a higher mesh slot is rewritten; the old one is retired (see the class comment). */
    private ensureMaskWords(words: number): void {
        if (words <= this.maskCapacity) return;
        let capacity = this.maskCapacity;
        while (capacity < words) capacity *= 2;
        this.retired.push({ buffer: this.masks, encode: this.encodes });
        this.masks = this.createMasks(capacity);
        this.maskCapacity = capacity;
    }

    private createMasks(words: number): GPUBuffer {
        return this.device.createBuffer({
            label: `${this.label} gi/triangle-flags masks`,
            size: words * 4,
            // COPY_DST for clearBuffer.
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
    }
}
