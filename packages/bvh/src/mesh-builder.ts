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
import { checkGpuLbvhFits, GpuLbvhBuilder, type GpuLbvhOptions } from './gpu-lbvh.ts';
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
    /** Builds every mesh's tree. With the GPU builder, all fitting meshes go in one batch. */
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
            const onGpu: number[] = [];
            for (const [index, geometry] of geometries.entries()) {
                const triangles = geometry.indices ? geometry.indices.length / 3 : geometry.positions.length / 9;
                try {
                    checkGpuLbvhFits(device, triangles);
                    onGpu.push(index);
                } catch (error) {
                    const reason = error instanceof Error ? error.message : String(error);
                    results[index] = { layout: await buildOnCpu(geometry, gpuOptions), builtBy: 'tinybvh', fallbackReason: reason };
                }
            }
            const builds = await gpu.buildMany(onGpu.map((index) => geometries[index]), gpuOptions);
            try {
                for (const [at, build] of builds.entries()) {
                    const index = onGpu[at];
                    if (build.problem) {
                        results[index] = { layout: await buildOnCpu(geometries[index], gpuOptions), builtBy: 'tinybvh', fallbackReason: build.problem };
                    } else {
                        results[index] = { layout: await gpu.readLayout(build.tree), builtBy: 'gpu-lbvh', fallbackReason: null };
                    }
                }
            } finally {
                // The layouts are CPU copies; the GPU trees are not needed after the readback.
                for (const build of builds) build.tree.destroy();
            }
            return results as MeshLayoutResult[];
        },
    };
}
