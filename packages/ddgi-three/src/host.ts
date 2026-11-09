/**
 * Making the device for three and the GI platform together.
 *
 * The host owns the device (SPEC, "GPU device"): three takes it through
 * `new WebGPURenderer({ device })`, and the platform through `createGI`.
 * When three makes its own device it asks for every feature the adapter has
 * and reads `core-features-and-limits` from the device to decide whether it
 * runs in compatibility mode (NOTES), so this does the same; that also covers
 * the platform's optional features (subgroups, timestamp-query, ...). Limits
 * are the platform's `requiredLimits`: the adapter's largest storage bindings
 * and buffers, which large scenes need. Plus the adapter's
 * `maxColorAttachmentBytesPerSample` (RC1.1): a scene pass with the
 * composite's outputs and the screen inputs' writes about 40-50 bytes per
 * pixel, past WebGPU's default of 32 (the RTX 3060 offers 128).
 */

import { WebGPURenderer } from 'three/webgpu';
import { requiredLimits, type RequiredLimits } from '@pmndrs/ddgi-core';

type RendererParameters = NonNullable<ConstructorParameters<typeof WebGPURenderer>[0]>;

export interface GIDeviceOptions {
    /** The adapter to use. Default: `navigator.gpu.requestAdapter({ powerPreference })`. */
    readonly adapter?: GPUAdapter;
    /** Default `'high-performance'`. */
    readonly powerPreference?: GPUPowerPreference;
    /** More limits on top of the platform's; each is capped at what the adapter has. */
    readonly requiredLimits?: RequiredLimits;
    readonly label?: string;
}

export interface GIDevice {
    readonly adapter: GPUAdapter;
    readonly device: GPUDevice;
}

/** The limits to request: the platform's, plus the host's extras, never more than the adapter has (so `requestDevice` can't fail on them). */
export function deviceLimits(adapter: GPUAdapter, extra: RequiredLimits = {}): Record<string, number> {
    const limits: Record<string, number> = {};
    const available = adapter.limits as unknown as Record<string, number>;
    const renderTargets: RequiredLimits = { maxColorAttachmentBytesPerSample: adapter.limits.maxColorAttachmentBytesPerSample };
    for (const [name, value] of Object.entries({ ...requiredLimits(adapter), ...renderTargets, ...extra })) {
        if (typeof value !== 'number' || typeof available[name] !== 'number') continue;
        limits[name] = Math.min(value, available[name]);
    }
    return limits;
}

/** An adapter and a device with every adapter feature and the platform's limits. */
export async function requestGIDevice(options: GIDeviceOptions = {}): Promise<GIDevice> {
    if (typeof navigator === 'undefined' || !navigator.gpu) throw new Error('@pmndrs/ddgi: WebGPU is not available in this browser.');
    const adapter = options.adapter ?? (await navigator.gpu.requestAdapter({ powerPreference: options.powerPreference ?? 'high-performance' }));
    if (!adapter) throw new Error('@pmndrs/ddgi: no WebGPU adapter (requestAdapter() returned null).');
    const device = await adapter.requestDevice({
        label: options.label ?? 'ddgi',
        requiredFeatures: [...adapter.features] as GPUFeatureName[],
        requiredLimits: deviceLimits(adapter, options.requiredLimits),
    });
    return { adapter, device };
}

export interface GIRendererOptions extends GIDeviceOptions {
    /** A device you already made (it should have the features and limits above). */
    readonly device?: GPUDevice;
    /** Anything else for `WebGPURenderer`, except `device`. */
    readonly renderer?: Omit<RendererParameters, 'device'>;
}

export interface GIRenderer {
    readonly renderer: WebGPURenderer;
    readonly device: GPUDevice;
    /** Null when you passed your own device. */
    readonly adapter: GPUAdapter | null;
}

/** A `WebGPURenderer` on a device made by {@link requestGIDevice} (or yours), initialized. */
export async function createGIRenderer(options: GIRendererOptions = {}): Promise<GIRenderer> {
    const { adapter, device } = options.device ? { adapter: options.adapter ?? null, device: options.device } : await requestGIDevice(options);
    const renderer = new WebGPURenderer({ ...options.renderer, device } as RendererParameters);
    await renderer.init();
    return { adapter, device, renderer };
}
