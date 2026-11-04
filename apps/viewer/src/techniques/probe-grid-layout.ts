/**
 * Picks how many probes a `LightProbeGrid` gets for a manifest's suggested
 * volume. No engine imports, so it is unit-tested.
 *
 * The manifest gives a probe spacing sized for DDGI (Sponza: 1 m, which would
 * be about 17,000 probes). `LightProbeGrid` bakes every probe from six
 * rendered faces of the whole scene, so its examples use a few hundred. The
 * spacing is widened until the probe count fits the budget, and the result is
 * recorded with every baseline number.
 */

export interface ProbeGridLayout {
    /** Probes along x, y and z. */
    counts: [number, number, number];
    total: number;
    /** Metres between probes along the longest axis of the volume. */
    spacing: number;
}

export const DEFAULT_MAX_PROBES = 600;

export function chooseProbeGridLayout(size: readonly [number, number, number], spacing: number, maxProbes: number): ProbeGridLayout {
    if (!(spacing > 0)) throw new Error(`probe spacing must be above 0, got ${spacing}`);
    if (!(maxProbes >= 8)) throw new Error(`the probe budget must be at least 8, got ${maxProbes}`);
    let currentSpacing = spacing;
    for (;;) {
        const counts = size.map((extent) => Math.max(2, Math.ceil(extent / currentSpacing))) as [number, number, number];
        const total = counts[0] * counts[1] * counts[2];
        // With 2 probes per axis the budget can't shrink further (2x2x2 = 8).
        if (total <= maxProbes || total <= 8) return { counts, total, spacing: currentSpacing };
        currentSpacing *= 1.05;
    }
}
