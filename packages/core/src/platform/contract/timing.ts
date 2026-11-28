/**
 * The slice of the GPU timer (T0.3) that techniques use. The full `GpuTimer`
 * lives in `platform/` once T0.3 lands; this is the part the contract needs.
 */

export interface GpuTimerScope {
    /**
     * Timestamp writes for a compute pass, named `<technique>/<pass>` in stats.
     * Returns `undefined` when timing is off or unsupported, which is exactly
     * what `beginComputePass({ timestampWrites })` accepts, so callers need no
     * branch.
     */
    computePass(name: string): GPUComputePassTimestampWrites | undefined;
}
