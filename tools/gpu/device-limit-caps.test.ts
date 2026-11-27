import { afterEach, describe, expect, it } from 'vitest';
import { deviceLimitCapsScript, storageBufferCaps } from './device-limit-caps.ts';

/** Stand-ins for the browser's classes: a limit is a prototype getter, as WebIDL attributes are. */
function installFakeWebGpu(adapterStorageBuffers: number): { requested: GPUDeviceDescriptor[]; adapter: { limits: { maxStorageBuffersPerShaderStage: number }; requestDevice(descriptor?: GPUDeviceDescriptor): Promise<unknown> } } {
    class FakeLimits {
        constructor(private readonly storage: number) {}
        get maxStorageBuffersPerShaderStage(): number {
            return this.storage;
        }
        get maxBindGroups(): number {
            return 4;
        }
    }
    const requested: GPUDeviceDescriptor[] = [];
    class FakeAdapter {
        readonly limits = new FakeLimits(adapterStorageBuffers);
        requestDevice(descriptor: GPUDeviceDescriptor = {}): Promise<unknown> {
            requested.push(descriptor);
            const storage = (descriptor.requiredLimits?.maxStorageBuffersPerShaderStage as number | undefined) ?? 8;
            return Promise.resolve({ limits: new FakeLimits(storage) });
        }
    }
    const scope = globalThis as Record<string, unknown>;
    scope.GPUSupportedLimits = FakeLimits;
    scope.GPUAdapter = FakeAdapter;
    return { requested, adapter: new FakeAdapter() };
}

afterEach(() => {
    const scope = globalThis as Record<string, unknown>;
    delete scope.GPUSupportedLimits;
    delete scope.GPUAdapter;
    delete scope.__gpuLimitCaps;
});

describe('the device-limit caps (--max-storage-buffers)', () => {
    it('reports at most the cap, and lowers a larger request to it', async () => {
        const { requested, adapter } = installFakeWebGpu(16);
        new Function(deviceLimitCapsScript(storageBufferCaps(10)))();
        expect(adapter.limits.maxStorageBuffersPerShaderStage).toBe(10);
        // A page asking for the adapter's maximum (requiredLimits) asks for the cap.
        const device = (await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 16, maxBindGroups: 4 } })) as { limits: { maxStorageBuffersPerShaderStage: number } };
        expect(requested[0].requiredLimits).toEqual({ maxStorageBuffersPerShaderStage: 10, maxBindGroups: 4 });
        expect(device.limits.maxStorageBuffersPerShaderStage).toBe(10);
    });

    it('leaves a device that asks for nothing at the default, and lower values alone', async () => {
        const { requested, adapter } = installFakeWebGpu(8);
        new Function(deviceLimitCapsScript(storageBufferCaps(10)))();
        expect(adapter.limits.maxStorageBuffersPerShaderStage).toBe(8);
        const device = (await adapter.requestDevice()) as { limits: { maxStorageBuffersPerShaderStage: number } };
        expect(requested[0].requiredLimits).toEqual({});
        expect(device.limits.maxStorageBuffersPerShaderStage).toBe(8);
        // Limits the cap doesn't name are untouched.
        expect((adapter.limits as unknown as { maxBindGroups: number }).maxBindGroups).toBe(4);
    });
});
