/**
 * `pnpm test:gpu --max-storage-buffers <n>`: run the pages as if the GPU
 * allowed only `n` storage buffers per shader stage.
 *
 * Why: Chrome on Apple GPUs (the M5 Pro) allows 10, the RTX 3060 16, and a
 * pipeline over 10 fails only on the Mac (pmndrs/gi#49). WebGPU enforces the
 * limits a device was created with, not the adapter's, so a device asked for
 * with 10 on the RTX rejects an 11-buffer pipeline exactly as the Mac does.
 *
 * The runner adds {@link deviceLimitCapsScript} to every document before its
 * own scripts (DevTools `Page.addScriptToEvaluateOnNewDocument`), so every
 * page, three.js's own device included, sees the cap:
 *
 * - `GPUSupportedLimits` reports at most the cap (adapter and device), so
 *   code that asks for the adapter's maximum (`requiredLimits(adapter)`)
 *   asks for the cap;
 * - `GPUAdapter.requestDevice` lowers any requested value above the cap to
 *   it, so a page that names a larger number still gets the capped device.
 *
 * A page that requests nothing keeps WebGPU's default (8), as on the Mac.
 */

/** The limits the cap applies to: the per-stage storage-buffer limits (the vertex and fragment ones exist in newer browsers). */
export const STORAGE_BUFFER_LIMITS = ['maxStorageBuffersPerShaderStage', 'maxStorageBuffersInVertexStage', 'maxStorageBuffersInFragmentStage'] as const;

/** Each storage-buffer limit at most `count`. */
export function storageBufferCaps(count: number): Record<string, number> {
    return Object.fromEntries(STORAGE_BUFFER_LIMITS.map((name) => [name, count]));
}

/**
 * Browser script that caps the named limits: the reported values and what a
 * `requestDevice` call can get. Self-contained (it runs in the page, not in
 * Node); a limit the browser doesn't have is left alone.
 */
export function deviceLimitCapsScript(caps: Readonly<Record<string, number>>): string {
    return `(() => {
    const caps = ${JSON.stringify(caps)};
    if (typeof GPUSupportedLimits === 'undefined' || typeof GPUAdapter === 'undefined') return;
    for (const [name, cap] of Object.entries(caps)) {
        const descriptor = Object.getOwnPropertyDescriptor(GPUSupportedLimits.prototype, name);
        if (!descriptor || !descriptor.get) continue;
        const read = descriptor.get;
        Object.defineProperty(GPUSupportedLimits.prototype, name, {
            ...descriptor,
            get() {
                return Math.min(read.call(this), cap);
            },
        });
    }
    const requestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = function (descriptor) {
        const requested = (descriptor && descriptor.requiredLimits) || {};
        const requiredLimits = { ...requested };
        for (const [name, cap] of Object.entries(caps)) {
            if (requiredLimits[name] !== undefined && requiredLimits[name] > cap) requiredLimits[name] = cap;
        }
        return requestDevice.call(this, { ...(descriptor || {}), requiredLimits });
    };
    globalThis.__gpuLimitCaps = caps;
})();`;
}
