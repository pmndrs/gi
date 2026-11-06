/**
 * The GPU builders' declared fallback (B1, B2), on a real device: the page
 * asks for no optional features, so the device has no `subgroups` (the
 * adapter may have them; the device doesn't). Then:
 *
 * - `createMeshLayoutBuilder` picks tinybvh (B0) and says why, whichever GPU
 *   builder is asked for;
 * - the GPU LBVH, the GPU H-PLOC and the BLAS refit refuse to start rather
 *   than fail later;
 * - the trees it builds pass the layout checker.
 */

import { GpuHplocBuilder } from '../../../../packages/bvh/src/gpu-hploc.ts';
import { GpuLbvhBuilder } from '../../../../packages/bvh/src/gpu-lbvh.ts';
import { GpuMeshRefit } from '../../../../packages/bvh/src/gpu-mesh-refit.ts';
import { checkGpuLayout } from '../../../../packages/bvh/src/layout-check.ts';
import { createMeshLayoutBuilder } from '../../../../packages/bvh/src/mesh-builder.ts';
import { assert, runGpuTest } from '../../client.ts';

function randomSoup(count: number, seed: number): Float32Array {
    let state = seed >>> 0;
    const random = () => ((state = (state * 1664525 + 1013904223) >>> 0) / 2 ** 32);
    return Float32Array.from({ length: count * 9 }, () => random() * 10);
}

await runGpuTest(async ({ adapter, device, log }) => {
    assert(!device.features.has('subgroups'), 'the device was made with subgroups; this page needs a device without');
    const builder = await createMeshLayoutBuilder(device);
    assert(builder.kind === 'tinybvh', `the mesh builder chose ${builder.kind} on a device without subgroups`);
    log(`fallback chosen: ${builder.fallbackReason}`);
    const askedForLbvh = await createMeshLayoutBuilder(device, { prefer: 'gpu-lbvh' });
    assert(askedForLbvh.kind === 'tinybvh', `asked for the LBVH, the mesh builder chose ${askedForLbvh.kind} on a device without subgroups`);

    for (const [name, start] of [
        ['LBVH', () => new GpuLbvhBuilder(device)],
        ['H-PLOC', () => new GpuHplocBuilder(device)],
    ] as const) {
        let refused = false;
        try {
            start();
        } catch {
            refused = true;
        }
        assert(refused, `the GPU ${name} builder started on a device without subgroups`);
    }
    assert(!GpuMeshRefit.isSupported(device), 'the BLAS refit says it can run without subgroups');

    const geometries = [{ positions: randomSoup(1, 1) }, { positions: randomSoup(3, 2) }, { positions: randomSoup(20_000, 3) }];
    const results = await builder.buildLayouts(geometries);
    const failures: string[] = [];
    results.forEach((result, index) => {
        if (result.builtBy !== 'tinybvh') failures.push(`mesh ${index} was built by ${result.builtBy}`);
        for (const problem of checkGpuLayout(result.layout, geometries[index])) failures.push(`mesh ${index}: ${problem}`);
    });
    assert(failures.length === 0, failures.join('\n'));
    return { adapterHasSubgroups: adapter.features.has('subgroups'), fallbackReason: builder.fallbackReason, nodes: results.map((result) => result.layout.nodeCount) };
});
