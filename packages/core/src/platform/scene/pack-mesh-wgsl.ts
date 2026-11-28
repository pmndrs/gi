/**
 * `gi/pack-mesh`: turns the host's raw mesh arrays, staged on the GPU, into
 * the vertex buffer's packed words (T2.13; layout in mesh-packing.ts).
 *
 * One dispatch handles every mesh in one staging buffer. The CPU writes a job
 * table: each job is one run of output words of one kind (below), and every
 * output word is one thread. A thread finds its job by binary search over the
 * jobs' first item, so thousands of meshes (Bistro exterior has 1,592) still
 * take one dispatch per staging buffer, not one per mesh (NOTES: many small
 * dispatches are what made per-mesh GPU work slow).
 *
 * The kinds:
 * - **copy:** words that are already in their packed form (positions, uvs,
 *   material slots, and indices whose width the host already gave).
 * - **normals:** three float words per vertex in, one octahedral word out.
 *   Same steps as `encodeOctahedral` (octahedral.ts), in f32 instead of f64,
 *   so a word may differ from the CPU's by one step (1/32767) per component.
 * - **narrow:** 32-bit indices to two 16-bit indices per word, for a mesh of
 *   at most 65,536 vertices whose host gave a `Uint32Array`.
 * - **widen:** 16-bit indices to one 32-bit index per word, for a mesh of more
 *   than 65,536 vertices whose host gave a `Uint16Array` (rare: only possible
 *   when some vertices are never used).
 */

/** Threads per workgroup. 256 is the default device's limit. */
export const PACK_MESH_WORKGROUP_SIZE = 256;

/** `PackJob.kind`. */
export const PACK_JOB_KIND = {
    copy: 0,
    normals: 1,
    narrow: 2,
    widen: 3,
} as const;

/** Words in one job record; see `PackJob` in the WGSL. */
export const PACK_JOB_WORDS = 8;

/** Words of the job table's header, before the first job; see `PackHeader`. */
export const PACK_HEADER_WORDS = 4;

/** Word positions inside a job record. */
export const PACK_JOB_WORD = {
    kind: 0,
    /** First word to read, in the staging buffer. */
    source: 1,
    /** First word to write, in the vertex buffer. */
    destination: 2,
    /** Output words this job writes: one thread each. */
    items: 3,
    /** The job's first thread among all of this dispatch's threads (a running sum of `items`). */
    firstItem: 4,
    /** For `narrow`: how many 32-bit indices there are to read (the last word may hold only one). */
    indexCount: 5,
} as const;

/** Word positions inside the header. */
export const PACK_HEADER_WORD = {
    jobCount: 0,
    /** Every job's `items`, summed: threads at or past this do nothing. */
    totalItems: 1,
    /** Threads in one row of the dispatch (workgroups along x × the workgroup size); see `itemOf`. */
    threadsPerRow: 2,
} as const;

export const PACK_MESH_ENTRY_POINT = 'giPackMesh';

export const PACK_MESH_WGSL = /* wgsl */ `
struct PackHeader {
    jobCount: u32,
    totalItems: u32,
    threadsPerRow: u32,
    unused: u32,
}

struct PackJob {
    kind: u32,
    source: u32,
    destination: u32,
    items: u32,
    firstItem: u32,
    indexCount: u32,
    unused0: u32,
    unused1: u32,
}

struct PackJobs {
    header: PackHeader,
    jobs: array<PackJob>,
}

const PACK_KIND_COPY = ${PACK_JOB_KIND.copy}u;
const PACK_KIND_NORMALS = ${PACK_JOB_KIND.normals}u;
const PACK_KIND_NARROW = ${PACK_JOB_KIND.narrow}u;
const PACK_KIND_WIDEN = ${PACK_JOB_KIND.widen}u;

// The host's arrays as they were given, one mesh after another (gpu-mesh-packing.ts).
@group(0) @binding(0) var<storage, read> staging: array<u32>;
@group(0) @binding(1) var<storage, read_write> vertices: array<u32>;
@group(0) @binding(2) var<storage, read> packJobs: PackJobs;

// An IEEE float's bits with every exponent bit set are an infinity or a NaN.
// Tested on the bits, not on the float: WGSL lets an implementation assume
// floats are never NaN, so a float comparison could be optimised away.
fn packIsFinite(bits: u32) -> bool {
    return (bits & 0x7f800000u) != 0x7f800000u;
}

// encodeOctahedral (octahedral.ts) in f32. A zero or non-finite vector encodes
// as (0, 0), which decodes to +Z, as on the CPU.
fn packOctahedral(xBits: u32, yBits: u32, zBits: u32) -> u32 {
    if (!(packIsFinite(xBits) && packIsFinite(yBits) && packIsFinite(zBits))) {
        return 0u;
    }
    let normal = vec3f(bitcast<f32>(xBits), bitcast<f32>(yBits), bitcast<f32>(zBits));
    // Divided by its largest component first, so |x| + |y| + |z| can't overflow
    // f32 for huge (still finite) inputs; the CPU sums in f64, where it can't either.
    let largest = max(abs(normal.x), max(abs(normal.y), abs(normal.z)));
    if (!(largest > 0.0)) {
        return 0u;
    }
    let scaled = normal / largest;
    var uv = scaled.xy / (abs(scaled.x) + abs(scaled.y) + abs(scaled.z));
    if (normal.z < 0.0) {
        // Fold the lower hemisphere over the diagonals. The sign is +1 for zero
        // (signNotZero on the CPU), and the CPU decides on the same f32 inputs,
        // so both sides always take the same branch and the same signs.
        let signs = select(vec2f(-1.0), vec2f(1.0), uv >= vec2f(0.0));
        uv = (vec2f(1.0) - abs(uv.yx)) * signs;
    }
    // pack2x16snorm rounds as floor(0.5 + 32767 × clamp(value)), as the CPU's
    // Math.round does, and puts the first component in the low 16 bits.
    return pack2x16snorm(uv);
}

// The thread's output word among all of the dispatch's. Two dimensions,
// because one dimension allows at most 65,535 workgroups (16.8 M threads),
// and Sponza writes about 28 M words.
fn itemOf(id: vec3u) -> u32 {
    return id.y * packJobs.header.threadsPerRow + id.x;
}

// The last job whose first item is at or before this item.
fn findJob(item: u32) -> u32 {
    var low = 0u;
    var high = packJobs.header.jobCount - 1u;
    while (low < high) {
        let middle = (low + high + 1u) / 2u;
        if (packJobs.jobs[middle].firstItem <= item) {
            low = middle;
        } else {
            high = middle - 1u;
        }
    }
    return low;
}

@compute @workgroup_size(${PACK_MESH_WORKGROUP_SIZE})
fn ${PACK_MESH_ENTRY_POINT}(@builtin(global_invocation_id) id: vec3u) {
    let item = itemOf(id);
    if (item >= packJobs.header.totalItems) {
        return;
    }
    let job = packJobs.jobs[findJob(item)];
    let local = item - job.firstItem;
    let destination = job.destination + local;
    switch job.kind {
        case PACK_KIND_NORMALS: {
            let source = job.source + local * 3u;
            vertices[destination] = packOctahedral(staging[source], staging[source + 1u], staging[source + 2u]);
        }
        case PACK_KIND_NARROW: {
            // Output word 'local' holds indices 2 × local (low half) and 2 × local + 1 (high half).
            let first = 2u * local;
            let low = staging[job.source + first] & 0xffffu;
            var high = 0u;
            if (first + 1u < job.indexCount) {
                high = staging[job.source + first + 1u] & 0xffffu;
            }
            vertices[destination] = low | (high << 16u);
        }
        case PACK_KIND_WIDEN: {
            // Index 'local' sits in staged word local / 2: even indices low, odd ones high.
            let word = staging[job.source + local / 2u];
            vertices[destination] = (word >> ((local & 1u) * 16u)) & 0xffffu;
        }
        default: {
            vertices[destination] = staging[job.source + local];
        }
    }
}
`;
