/**
 * The device the radiance-cache pages run on: the adapter's limits as the
 * platform asks for them (`requiredLimits`), except storage buffers per
 * shader stage, held at the platform's budget of 8
 * (`MIN_STORAGE_BUFFERS_PER_STAGE`). Why: the adapters we test on allow more
 * (16 on the RTX 3060 and the Intel UHD, 10 on the M5), so a kernel that
 * binds 9 to 10 would pass there and fail on a default device. Held at 8, the
 * kernel builder's `GILimitError` catches it on every machine (register D6).
 */

import { MIN_STORAGE_BUFFERS_PER_STAGE, optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';

export interface BudgetDevice {
    readonly device: GPUDevice;
    /** What the adapter itself allows, for the page's report. */
    readonly adapterStorageBuffersPerStage: number;
}

export async function requestBudgetDevice(adapter: GPUAdapter): Promise<BudgetDevice> {
    const limits = { ...requiredLimits(adapter), maxStorageBuffersPerShaderStage: MIN_STORAGE_BUFFERS_PER_STAGE };
    const device = await adapter.requestDevice({ requiredLimits: limits, requiredFeatures: optionalFeatures(adapter) });
    return { device, adapterStorageBuffersPerStage: adapter.limits.maxStorageBuffersPerShaderStage };
}
