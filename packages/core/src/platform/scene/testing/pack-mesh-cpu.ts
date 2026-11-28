/**
 * `gi/pack-mesh` (pack-mesh-wgsl.ts) run on the CPU, step for step, for unit
 * tests in Node: the fake device runs it when a test submits a dispatch of
 * that kernel. Floats go through `Math.fround` where the WGSL computes in
 * f32, so the octahedral words come out as the GPU's would (up to the GPU's
 * own rounding, which the GPU page `scene-gpu-packing` checks on hardware).
 * Test-only: not exported from the package.
 */

import { PACK_HEADER_WORD, PACK_HEADER_WORDS, PACK_JOB_KIND, PACK_JOB_WORD, PACK_JOB_WORDS, PACK_MESH_WORKGROUP_SIZE } from '../pack-mesh-wgsl.ts';

const f32 = Math.fround;

function isFiniteBits(bits: number): boolean {
    return (bits & 0x7f800000) !== 0x7f800000;
}

const bitsView = new DataView(new ArrayBuffer(4));

function floatOfBits(bits: number): number {
    bitsView.setUint32(0, bits, true);
    return bitsView.getFloat32(0, true);
}

/** WGSL's `pack2x16snorm`: floor(0.5 + 32767 × clamp(value, -1, 1)) per component, first in the low half. */
function pack2x16snorm(u: number, v: number): number {
    const component = (value: number) => Math.floor(0.5 + f32(32767 * Math.min(1, Math.max(-1, value)))) & 0xffff;
    return (component(u) | (component(v) << 16)) >>> 0;
}

/** The kernel's `packOctahedral`. */
export function packOctahedralF32(xBits: number, yBits: number, zBits: number): number {
    if (!(isFiniteBits(xBits) && isFiniteBits(yBits) && isFiniteBits(zBits))) return 0;
    const x = floatOfBits(xBits);
    const y = floatOfBits(yBits);
    const z = floatOfBits(zBits);
    const largest = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
    if (!(largest > 0)) return 0;
    const sx = f32(x / largest);
    const sy = f32(y / largest);
    const sz = f32(z / largest);
    const sum = f32(f32(Math.abs(sx) + Math.abs(sy)) + Math.abs(sz));
    let u = f32(sx / sum);
    let v = f32(sy / sum);
    if (z < 0) {
        const signU = u >= 0 ? 1 : -1;
        const signV = v >= 0 ? 1 : -1;
        const foldedU = f32((1 - Math.abs(v)) * signU);
        const foldedV = f32((1 - Math.abs(u)) * signV);
        u = foldedU;
        v = foldedV;
    }
    return pack2x16snorm(u, v);
}

/** The kernel's `findJob`: the last job whose first item is at or before `item`. */
function findJob(jobs: Uint32Array, item: number): number {
    let low = 0;
    let high = jobs[PACK_HEADER_WORD.jobCount] - 1;
    while (low < high) {
        const middle = Math.floor((low + high + 1) / 2);
        if (jobs[PACK_HEADER_WORDS + middle * PACK_JOB_WORDS + PACK_JOB_WORD.firstItem] <= item) low = middle;
        else high = middle - 1;
    }
    return low;
}

/**
 * Runs one dispatch of `workgroupsX` × `workgroupsY` workgroups, thread by
 * thread as the kernel does: `jobs` is the job table the packer bound (header
 * and records), `staging` and `vertices` the two buffers' words.
 */
export function runPackMeshDispatch(staging: Uint32Array, vertices: Uint32Array, jobs: Uint32Array, workgroupsX: number, workgroupsY: number): void {
    const threadsPerRow = jobs[PACK_HEADER_WORD.threadsPerRow];
    const totalItems = jobs[PACK_HEADER_WORD.totalItems];
    for (let row = 0; row < workgroupsY; row++) {
        for (let column = 0; column < workgroupsX * PACK_MESH_WORKGROUP_SIZE; column++) {
            const item = row * threadsPerRow + column;
            if (item >= totalItems) continue;
            const base = PACK_HEADER_WORDS + findJob(jobs, item) * PACK_JOB_WORDS;
            const kind = jobs[base + PACK_JOB_WORD.kind];
            const source = jobs[base + PACK_JOB_WORD.source];
            const local = item - jobs[base + PACK_JOB_WORD.firstItem];
            const destination = jobs[base + PACK_JOB_WORD.destination] + local;
            if (kind === PACK_JOB_KIND.normals) {
                const at = source + local * 3;
                vertices[destination] = packOctahedralF32(staging[at], staging[at + 1], staging[at + 2]);
            } else if (kind === PACK_JOB_KIND.narrow) {
                const first = 2 * local;
                const low = staging[source + first] & 0xffff;
                const high = first + 1 < jobs[base + PACK_JOB_WORD.indexCount] ? staging[source + first + 1] & 0xffff : 0;
                vertices[destination] = (low | (high << 16)) >>> 0;
            } else if (kind === PACK_JOB_KIND.widen) {
                const word = staging[source + (local >> 1)];
                vertices[destination] = (word >>> ((local & 1) * 16)) & 0xffff;
            } else {
                vertices[destination] = staging[source + local];
            }
        }
    }
}
