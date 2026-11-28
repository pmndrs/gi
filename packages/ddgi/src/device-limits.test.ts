// What the platform asks a device for, against DDGI's kernels. The platform's own tests
// import no technique (PKG.1), so this check lives with the kernel it measures.
import { describe, expect, it } from 'vitest';
import { KERNEL_COMPUTE_LIMITS } from '@pmndrs/gi-core/platform/device-requirements';
import { BLEND_WORKGROUP_SIZE } from './wgsl/blend.ts';

describe('the compute limits the platform asks for', () => {
    it('asks for workgroups as large as the largest kernel uses (DDGI\'s blend)', () => {
        expect(KERNEL_COMPUTE_LIMITS.maxComputeWorkgroupSizeX).toBeGreaterThanOrEqual(BLEND_WORKGROUP_SIZE);
        expect(KERNEL_COMPUTE_LIMITS.maxComputeInvocationsPerWorkgroup).toBeGreaterThanOrEqual(BLEND_WORKGROUP_SIZE);
    });
});
