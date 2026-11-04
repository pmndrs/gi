/**
 * One entry point for mesh trees in the tracer's layout, whichever builder
 * makes them (the Builder contract, docs/design/bvh-framework.md): the GPU
 * LBVH (B1) when the device has subgroups, tinybvh in WebAssembly (B0)
 * otherwise. That is the BVH's one declared fallback (DECISIONS 2026-11-04:
 * subgroups first, one fallback per feature).
 *
 * The fallback also covers single meshes: a mesh too big for the device's
 * buffers, or a GPU build that was refused (gpu-lbvh.ts, "Checks"), is built
 * by tinybvh instead, and the result says so.
 *
 * tinybvh runs on the calling thread here. ddgi-core runs it in a worker
 * (`mesh-bvh-async.ts`); this package stays free of workers and engines.
 */

import type { GpuLayoutMeshBvh } from './gpu-layout.ts';
import { GpuLbvhBuilder, planGpuLbvhBatches, type GpuLbvhOptions } from './gpu-lbvh.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder, type TinyBvhInput } from './tinybvh.ts';

export type MeshLayoutBuilderKind = 'gpu-lbvh' | 'tinybvh';

export interface MeshLayoutResult {
    readonly layout: GpuLayoutMeshBvh;
    readonly builtBy: MeshLayoutBuilderKind;
    /** Why tinybvh built this mesh although the GPU builder is in use, or null. */
    readonly fallbackReason: string | null;
}

export interface MeshLayoutBuilder {
    /** The builder meshes go to first. */
    readonly kind: MeshLayoutBuilderKind;
    /** Why the GPU builder is not in use, or null when it is. */
    readonly fallbackReason: string | null;
    /** Builds every mesh's tree. With the GPU builder, meshes go in as few batches as the device's buffers allow. */
    buildLayouts(geometries: readonly TinyBvhInput[], options?: GpuLbvhOptions): Promise<MeshLayoutResult[]>;
}

/**
 * The GPU LBVH builder if `device` has subgroups, tinybvh otherwise (or with no
 * device). `prefer: 'tinybvh'` forces the CPU builder, for comparisons.
 */
export async function createMeshLayoutBuilder(
    device: GPUDevice | null,
    options: { prefer?: MeshLayoutBuilderKind; label?: string } = {},
): Promise<MeshLayoutBuilder> {
    let fallbackReason: string | null = null;
    if (options.prefer === 'tinybvh') fallbackReason = 'the CPU builder was asked for';
    else if (!device) fallbackReason = 'no GPU device';
    else if (!GpuLbvhBuilder.isSupported(device)) fallbackReason = 'the device has no "subgroups" feature';

    let tinyBvh: Promise<TinyBvhBuilder> | null = null;
    const buildOnCpu = async (geometry: TinyBvhInput, gpuOptions: GpuLbvhOptions): Promise<GpuLayoutMeshBvh> => {
        tinyBvh ??= createTinyBvhBuilder();
        return (await tinyBvh).buildLayout(geometry, { maxLeafPrimitives: gpuOptions.maxLeafPrimitives }).layout;
    };

    if (fallbackReason !== null || !device) {
        return {
            kind: 'tinybvh',
            fallbackReason,
            async buildLayouts(geometries, gpuOptions = {}) {
                const results: MeshLayoutResult[] = [];
                for (const geometry of geometries) results.push({ layout: await buildOnCpu(geometry, gpuOptions), builtBy: 'tinybvh', fallbackReason: null });
                return results;
            },
        };
    }

    const gpu = new GpuLbvhBuilder(device, { label: options.label });
    await gpu.prepare();
    return {
        kind: 'gpu-lbvh',
        fallbackReason: null,
        async buildLayouts(geometries, gpuOptions = {}) {
            const results: (MeshLayoutResult | null)[] = geometries.map(() => null);
            const plan = planGpuLbvhBatches(device, geometries);
            for (const index of plan.tooBig) {
                const reason = `the mesh is too big for this device's GPU buffers (${Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize)} bytes)`;
                results[index] = { layout: await buildOnCpu(geometries[index], gpuOptions), builtBy: 'tinybvh', fallbackReason: reason };
            }
            for (const batch of plan.batches) {
                const trees = await gpu.buildMany(batch.map((index) => geometries[index]), gpuOptions);
                try {
                    const layouts = await gpu.readLayouts(trees);
                    for (const [at, layout] of layouts.entries()) {
                        const index = batch[at];
                        const problem = trees.meshes[at].problem;
                        if (layout && !problem) results[index] = { layout, builtBy: 'gpu-lbvh', fallbackReason: null };
                        else results[index] = { layout: await buildOnCpu(geometries[index], gpuOptions), builtBy: 'tinybvh', fallbackReason: problem };
                    }
                } finally {
                    // The layouts are CPU copies; the GPU trees are not needed after the readback.
                    trees.destroy();
                }
            }
            return results as MeshLayoutResult[];
        },
    };
}
