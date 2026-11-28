/**
 * A workgroup-wide exclusive prefix sum built on subgroup operations, used by
 * the platform's scan (scan.ts). It came from the frozen BVH copy
 * (`packages/bvh/src/gpu-workgroup-scan.ts`) with the scan, when gi moved onto
 * the published `@pmndrs/bvh` (pmndrs/gi#63): the BVH package's own scan is
 * internal and subgroups-only, and its declared no-subgroups fallback is
 * tinybvh, so it never needs the workgroup-memory path below. This is gi's
 * code now (docs/plans/sibling-migration.md §3, item 3).
 *
 * **Order without assumptions.** A prefix sum needs an order of the threads,
 * and the obvious one, `local_invocation_index`, says nothing about which
 * subgroup a thread lands in or at which lane: WebGPU fixes neither the
 * subgroup size (4 to 128, and it may differ between pipelines on one device:
 * Intel's UHD here runs 8 or 16 wide) nor the lane layout. So the threads make
 * their own order. At the start of a kernel, one lane of each subgroup claims
 * a block of thread numbers from a workgroup counter, as many as its subgroup
 * has active lanes, and each lane takes the number of its rank among them.
 * Thread numbers then run in subgroup lane order, which is exactly the order
 * `subgroupExclusiveAdd` sums in, and subgroups are numbered (their "slot") in
 * the order they claimed. Nothing depends on the subgroup size or on which
 * lanes exist; a kernel only has to use `ScanOrder.thread` to decide which
 * items a thread owns.
 *
 * **The scan.** Within a subgroup, `subgroupExclusiveAdd` gives each lane its
 * prefix and `subgroupAdd` the subgroup's total. The elected lane stores the
 * total in its slot; after a barrier, one thread per word turns the slot totals
 * into exclusive prefixes (a short serial walk: at most 64 slots); after a
 * second barrier every thread adds its slot's prefix. Several words can be
 * scanned at once.
 *
 * **The declared fallback without subgroups** (DECISIONS 2026-11-04: subgroups
 * first, one fallback): the same three names, `ScanOrder`, `scanTakeOrder` and
 * `scanExclusive`, built on workgroup memory alone. The order is then simply
 * `local_invocation_index` (no lanes are involved, so nothing about them is
 * assumed), and each word is scanned by a Hillis-Steele scan over the
 * workgroup's threads. A kernel written against these names runs on either
 * path unchanged.
 */

/** How a workgroup combines its threads' values: subgroup operations first, or the declared fallback in workgroup memory. */
export type ScanReduction = 'subgroups' | 'workgroup';

/** Threads per workgroup of every kernel that uses this scan. The default device's largest workgroup. */
export const SCAN_WORKGROUP_SIZE = 256;

/** The smallest subgroup WebGPU allows, which bounds how many subgroups one workgroup can have. */
const MIN_SUBGROUP_SIZE = 4;

/** Most subgroups in one workgroup. */
export const MAX_SUBGROUPS_PER_WORKGROUP = SCAN_WORKGROUP_SIZE / MIN_SUBGROUP_SIZE;

/**
 * WGSL declaring `ScanOrder`, `scanTakeOrder(localIndex)` and
 * `scanExclusive(values, order, localIndex)` for `words` words per thread,
 * on the given path. With `subgroups` (the default) the including module must
 * `enable subgroups;`. Both functions contain subgroup operations or
 * barriers, so call them from uniform control flow, and call `scanTakeOrder`
 * once per workgroup, before anything else uses the order.
 */
export function workgroupScanWgsl(words: number, reduction: ScanReduction = 'subgroups'): string {
    if (!Number.isInteger(words) || words < 1) throw new RangeError(`a workgroup scan needs at least one word, got ${words}`);
    return reduction === 'subgroups' ? subgroupScanWgsl(words) : workgroupMemoryScanWgsl(words);
}

function subgroupScanWgsl(words: number): string {
    return /* wgsl */ `
// ---- Workgroup scan in subgroup order (workgroup-scan.ts) ----------------

const SCAN_WORDS: u32 = ${words}u;
const SCAN_MAX_SUBGROUPS: u32 = ${MAX_SUBGROUPS_PER_WORKGROUP}u;

struct ScanOrder {
    thread: u32,     // this thread's place in the workgroup's order
    slot: u32,       // its subgroup's place among the workgroup's subgroups
    elected: bool,   // the one lane per subgroup that stores the subgroup's totals
}

// (slots claimed << 16) | threads claimed. Workgroup memory starts at zero.
var<workgroup> scanClaims: atomic<u32>;
var<workgroup> scanSlotTotals: array<u32, ${MAX_SUBGROUPS_PER_WORKGROUP * words}>;
// After scanExclusive: each word's sum over the whole workgroup.
var<workgroup> scanTotals: array<u32, ${words}>;

// 'localIndex' is unused here: the order comes from the subgroups (the fallback uses it).
fn scanTakeOrder(localIndex: u32) -> ScanOrder {
    let lanes = subgroupAdd(1u);
    let rank = subgroupExclusiveAdd(1u);
    let elected = subgroupElect();
    var claim = 0u;
    if (elected) {
        claim = atomicAdd(&scanClaims, (1u << 16u) | lanes);
    }
    // The elected lane is the lowest active one, which is where broadcastFirst reads.
    claim = subgroupBroadcastFirst(claim);
    return ScanOrder((claim & 0xffffu) + rank, claim >> 16u, elected);
}

// Exclusive prefix sums of 'values' over the workgroup, word by word, in
// ScanOrder.thread order; each word's workgroup total is left in scanTotals.
// A caller that scans again in a loop must put a barrier between the rounds.
fn scanExclusive(values: array<u32, SCAN_WORDS>, order: ScanOrder, localIndex: u32) -> array<u32, SCAN_WORDS> {
    var prefix: array<u32, SCAN_WORDS>;
    for (var word = 0u; word < SCAN_WORDS; word += 1u) {
        prefix[word] = subgroupExclusiveAdd(values[word]);
        let total = subgroupAdd(values[word]);
        if (order.elected) {
            scanSlotTotals[order.slot * SCAN_WORDS + word] = total;
        }
    }
    workgroupBarrier();
    if (localIndex < SCAN_WORDS) {
        let slots = atomicLoad(&scanClaims) >> 16u;
        var running = 0u;
        for (var slot = 0u; slot < slots; slot += 1u) {
            let at = slot * SCAN_WORDS + localIndex;
            let total = scanSlotTotals[at];
            scanSlotTotals[at] = running;
            running += total;
        }
        scanTotals[localIndex] = running;
    }
    workgroupBarrier();
    for (var word = 0u; word < SCAN_WORDS; word += 1u) {
        prefix[word] += scanSlotTotals[order.slot * SCAN_WORDS + word];
    }
    return prefix;
}
`;
}

function workgroupMemoryScanWgsl(words: number): string {
    return /* wgsl */ `
// ---- Workgroup scan in workgroup memory, the fallback without subgroups (workgroup-scan.ts) ----

const SCAN_WORDS: u32 = ${words}u;
const SCAN_THREADS: u32 = ${SCAN_WORKGROUP_SIZE}u;

struct ScanOrder {
    thread: u32,     // this thread's place in the workgroup's order: its local index
    slot: u32,       // always 0: there are no subgroups to number
    elected: bool,   // thread 0 only
}

// One word per thread while a word is scanned; the words take turns.
var<workgroup> scanPartials: array<u32, ${SCAN_WORKGROUP_SIZE}>;
// After scanExclusive: each word's sum over the whole workgroup.
var<workgroup> scanTotals: array<u32, ${words}>;

fn scanTakeOrder(localIndex: u32) -> ScanOrder {
    return ScanOrder(localIndex, 0u, localIndex == 0u);
}

// Exclusive prefix sums of 'values' over the workgroup, word by word, in
// ScanOrder.thread order; each word's workgroup total is left in scanTotals.
// Hillis-Steele: after the round at 'distance', each entry holds the sum of
// the 2 * distance entries ending at it, so log2(threads) rounds give the
// inclusive prefix. Each round reads before a barrier and writes after it,
// so no thread reads an entry another is changing. A caller that scans again
// in a loop must put a barrier between the rounds (as with subgroups).
fn scanExclusive(values: array<u32, SCAN_WORDS>, order: ScanOrder, localIndex: u32) -> array<u32, SCAN_WORDS> {
    var prefix: array<u32, SCAN_WORDS>;
    for (var word = 0u; word < SCAN_WORDS; word += 1u) {
        scanPartials[order.thread] = values[word];
        workgroupBarrier();
        for (var distance = 1u; distance < SCAN_THREADS; distance *= 2u) {
            var before = 0u;
            if (order.thread >= distance) {
                before = scanPartials[order.thread - distance];
            }
            workgroupBarrier();
            scanPartials[order.thread] += before;
            workgroupBarrier();
        }
        let inclusive = scanPartials[order.thread];
        prefix[word] = inclusive - values[word];
        if (order.thread == SCAN_THREADS - 1u) {
            scanTotals[word] = inclusive;
        }
        // The next word overwrites scanPartials; the total must be visible on return.
        workgroupBarrier();
    }
    return prefix;
}
`;
}

/** WGSL for the flat workgroup index of a 2D dispatch (big inputs exceed the per-dimension limit). */
export const WORKGROUP_INDEX_WGSL = /* wgsl */ `
fn flatWorkgroupIndex(workgroup: vec3u, groups: vec3u) -> u32 {
    return workgroup.y * groups.x + workgroup.x;
}
`;

/** Records `workgroups` workgroups as a 2D dispatch that stays under the per-dimension limit. */
export function dispatchWorkgroups2d(pass: GPUComputePassEncoder, workgroups: number, maxPerDimension: number): void {
    if (workgroups <= 0) return;
    const x = Math.min(workgroups, maxPerDimension);
    pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
}
