/**
 * Which probes update each frame (docs/design/ddgi.md, "Which probes update").
 *
 * All probes are walked in one fixed scrambled order, `order[i] = (i × stride)
 * mod N`, with `stride` coprime with N and near 0.618 N (the golden ratio's
 * fraction). Consecutive entries then land far apart, so each frame's set is
 * spread through the volume instead of being one slab of it, and every probe
 * comes up exactly once per pass because a coprime stride visits every
 * residue. A CPU cursor walks the order; nothing is read back from the GPU.
 */

/** 1/φ: consecutive multiples of it mod 1 are as evenly spread as any sequence can be. */
const GOLDEN_FRACTION = (Math.sqrt(5) - 1) / 2;

function greatestCommonDivisor(a: number, b: number): number {
    while (b !== 0) [a, b] = [b, a % b];
    return a;
}

/** The stride coprime with `probeCount` nearest to 0.618 × `probeCount` (1 for tiny counts). */
export function scrambleStride(probeCount: number): number {
    if (probeCount <= 2) return 1;
    const target = Math.round(probeCount * GOLDEN_FRACTION);
    // Search outwards from the target; a coprime number is always close (1 is one, at worst).
    for (let distance = 0; distance < probeCount; distance++) {
        for (const candidate of [target - distance, target + distance]) {
            if (candidate >= 1 && candidate < probeCount && greatestCommonDivisor(candidate, probeCount) === 1) return candidate;
        }
    }
    return 1;
}

/** Walks the scrambled order and hands out each frame's update list. */
export class UpdateCursor {
    readonly probeCount: number;
    readonly stride: number;
    /** Index into the scrambled order of the next probe to update. */
    private position = 0;
    /** Probe updates handed out since the last reset (capped at the probe count for "not updated yet"). */
    private handedOut = 0;

    constructor(probeCount: number) {
        this.probeCount = probeCount;
        this.stride = scrambleStride(probeCount);
    }

    /** The `i`-th probe of the scrambled order. */
    probeAt(index: number): number {
        // Both factors are below 2^26 for any atlas that fits a device, so the product stays exact in a double.
        return (index * this.stride) % this.probeCount;
    }

    /**
     * Fills `list` with the next `count` probes (fewer if `count` exceeds the
     * probe count, so no probe updates twice in one frame) and advances the
     * cursor. Returns how many were written.
     */
    next(count: number, list: Uint32Array): number {
        // NaN asks for nothing; Infinity asks for every probe.
        const requested = Number.isNaN(count) ? 0 : Math.floor(count);
        const take = Math.max(0, Math.min(requested, this.probeCount, list.length));
        for (let entry = 0; entry < take; entry++) {
            list[entry] = this.probeAt(this.position);
            this.position = (this.position + 1) % this.probeCount;
        }
        this.handedOut = Math.min(this.probeCount, this.handedOut + take);
        return take;
    }

    /** Probes that have not had an update since the last reset. */
    get notUpdatedYet(): number {
        return this.probeCount - this.handedOut;
    }

    /** Every probe counts as not updated again (`resetHistory`). The walk itself carries on. */
    resetHistory(): void {
        this.handedOut = 0;
    }
}
