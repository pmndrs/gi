/**
 * `FilterScreenProbes` (gi1.comp:1356-1443; the kernel is in
 * wgsl/screen-irradiance.ts) as its own small class, recorded the way
 * Capsaicin runs it (gi1.cpp:2588-2614): two passes over every possible ray,
 * across the screen from the atlas into the second atlas, then down it from
 * there back into the atlas. The caller copies the atlas into the second
 * atlas first (Capsaicin's copy after the blend, gi1.cpp:2540-2541), so the
 * second pass finds the probes the first doesn't write (the reprojected ones)
 * there too.
 *
 * `ScreenIrradiance` owns one for the frame. It takes its inputs as plain
 * buffers and textures, so a test can run the same kernel on atlases of its
 * own without reaching into the technique.
 */

import type { ComputeKernel, KernelBuilder } from '../../platform/contract/wgsl.ts';
import type { ScreenProbeLayout } from './settings.ts';
import { BLUR_DIRECTION_BYTES, BLUR_DIRECTIONS, FILTER_PROBES_GROUP, FILTER_PROBES_WGSL, SCREEN_IRRADIANCE_ENTRY_POINT, SCREEN_IRRADIANCE_WORKGROUP } from './wgsl/screen-irradiance.ts';

const BUFFER_USAGE = { uniform: 0x40, copyDst: 0x08 } as const;

const TECHNIQUE_ID = 'radiance-cache';

/** What the two passes read and write. */
export interface ScreenProbeFilterInputs {
    readonly layout: ScreenProbeLayout;
    /** `RcFrame` (wgsl/screen-probes.ts): the inverse view-projection, the eye, the screen's size and its encodings are read. */
    readonly frameUniforms: GPUBuffer;
    /** `RcScreenProbesConstants` (wgsl/screen-irradiance.ts): the view axis, the cell size, the mask's mip count. */
    readonly screenConstants: GPUBuffer;
    /** The compacted probes (`RcProbeSpawn`): the count, then the seeds. */
    readonly probeSpawn: GPUBuffer;
    /** The probe mask with its whole mip chain (`FilterProbeMask`'s). */
    readonly probeMask: GPUTexture;
    /** The host's depth and normals (`FrameScreen`). */
    readonly depth: GPUTexture;
    readonly normal: GPUTexture;
    /** `rgba16float`, atlas-sized: the first pass reads `atlas` and writes `secondAtlas`, the second the other way round. */
    readonly atlas: GPUTexture;
    readonly secondAtlas: GPUTexture;
}

export class ScreenProbeFilter {
    private groups: { readonly inputs: ScreenProbeFilterInputs; readonly passes: readonly [GPUBindGroup, GPUBindGroup] } | null = null;

    private constructor(
        private readonly device: GPUDevice,
        private readonly kernel: ComputeKernel,
        /** `g_BlurDirection` of each pass, written once. */
        private readonly blurDirections: readonly [GPUBuffer, GPUBuffer],
    ) {}

    /** Compiles the kernel through the platform's kernel builder. */
    static async create(context: { readonly device: GPUDevice; readonly kernels: KernelBuilder }): Promise<ScreenProbeFilter> {
        const entryPoint = SCREEN_IRRADIANCE_ENTRY_POINT.filterProbes;
        const kernel = await context.kernels.computeKernel({ label: `${TECHNIQUE_ID}/${entryPoint}`, modules: [], code: FILTER_PROBES_WGSL, entryPoint, ownGroups: [FILTER_PROBES_GROUP] });
        const device = context.device;
        const blurDirection = (direction: readonly [number, number], label: string): GPUBuffer => {
            const buffer = device.createBuffer({ label: `${TECHNIQUE_ID} FilterScreenProbes ${label}`, size: BLUR_DIRECTION_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
            device.queue.writeBuffer(buffer, 0, new Int32Array([direction[0], direction[1], 0, 0]));
            return buffer;
        };
        return new ScreenProbeFilter(device, kernel, [blurDirection(BLUR_DIRECTIONS[0], 'across'), blurDirection(BLUR_DIRECTIONS[1], 'down')]);
    }

    get memoryBytes(): number {
        return 2 * BLUR_DIRECTION_BYTES;
    }

    /**
     * Records both passes into `pass` (each dispatch is its own usage scope,
     * so the second reads what the first wrote).
     */
    record(pass: GPUComputePassEncoder, inputs: ScreenProbeFilterInputs): void {
        const passes = this.groupsFor(inputs);
        const workgroups = inputs.layout.maxRayCount / SCREEN_IRRADIANCE_WORKGROUP.probe;
        const x = Math.max(1, Math.min(workgroups, this.device.limits.maxComputeWorkgroupsPerDimension));
        this.kernel.bind(pass);
        for (const group of passes) {
            pass.setBindGroup(1, group);
            // Split over x and y; the kernel counts workgroups in rows (rcWorkgroupIndex).
            pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
        }
    }

    dispose(): void {
        for (const buffer of this.blurDirections) buffer.destroy();
        this.groups = null;
    }

    private groupsFor(inputs: ScreenProbeFilterInputs): readonly [GPUBindGroup, GPUBindGroup] {
        const current = this.groups;
        if (current && (Object.keys(inputs) as (keyof ScreenProbeFilterInputs)[]).every((key) => current.inputs[key] === inputs[key])) return current.passes;
        const view = (texture: GPUTexture): GPUTextureView => texture.createView({ aspect: texture.format.startsWith('depth') ? 'depth-only' : 'all', baseMipLevel: 0, mipLevelCount: 1 });
        // g_ScreenProbes_PreviousProbeBuffer and g_ScreenProbes_ProbeBuffer of each pass (gi1.cpp:2605-2610).
        const group = (index: 0 | 1, read: GPUTexture, write: GPUTexture): GPUBindGroup => {
            const resources: GPUBindingResource[] = [
                { buffer: inputs.frameUniforms },
                { buffer: inputs.screenConstants },
                view(inputs.depth),
                view(inputs.normal),
                // The whole mip chain: FindClosestProbe walks up it.
                inputs.probeMask.createView({ baseMipLevel: 0, mipLevelCount: inputs.layout.probeMaskMipCount }),
                { buffer: inputs.probeSpawn },
                read.createView(),
                write.createView(),
                { buffer: this.blurDirections[index] },
            ];
            return this.device.createBindGroup({
                label: `${TECHNIQUE_ID}/${SCREEN_IRRADIANCE_ENTRY_POINT.filterProbes} pass ${index + 1}`,
                layout: this.kernel.ownLayouts[0],
                entries: resources.map((resource, binding) => ({ binding, resource })),
            });
        };
        const passes = [group(0, inputs.atlas, inputs.secondAtlas), group(1, inputs.secondAtlas, inputs.atlas)] as const;
        this.groups = { inputs, passes };
        return passes;
    }
}
