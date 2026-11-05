/**
 * B2: the GPU H-PLOC builder's trees, on this GPU.
 *
 * Every mesh of the B1 cases (../bvh-shared/tree-cases.ts) is built in one
 * batch, twice:
 *
 * 1. **Default** (B1's 30-bit cubic codes, SAH leaf collapse): each tree goes
 *    through the layout checker (packages/bvh/src/layout-check.ts), and its SAH
 *    cost and node count are compared with the CPU twin's (hploc-cpu.ts), which
 *    builds the same tree but for float rounding.
 * 2. **Against tinybvh** (tinybvh's 63-bit codes and tie rule, one-triangle
 *    leaves: the tree tinybvh's `BuildHPLOC` makes): checked again, and the
 *    SAH cost must be within 1% of tinybvh's `BuildHPLOC` tree of the same
 *    mesh (the B2 done-when). One case is refused instead: on 3000 identical
 *    triangles tinybvh's rule chains the tree to depth 128 (tinybvh's own tree
 *    is that deep), more than H-PLOC's info word, and the tracer's stack, hold.
 * 3. **tinybvh's codes, the default ties**: against the CPU twin, as 1.
 *
 * A mesh with an index past its last vertex must be refused, while the other
 * meshes of its batch are not: a refused mesh is how the builder hands a mesh
 * to its fallback (mesh-builder.ts). (A NaN corner is not refused: WGSL's
 * `min` and `max` may drop the NaN, so the boxes stay finite and the tree is built.)
 *
 * The ray oracle for H-PLOC trees is the trace-oracle page (variant `watertight-hploc`).
 */

import { GpuHplocBuilder, type HplocOptions } from '../../../../packages/bvh/src/gpu-hploc.ts';
import { buildHplocOnCpu } from '../../../../packages/bvh/src/hploc-cpu.ts';
import { checkGpuLayout, treeSahCost } from '../../../../packages/bvh/src/layout-check.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder, type TinyBvhInput } from '../../../../packages/bvh/src/tinybvh.ts';
import { assert, runGpuTest } from '../../client.ts';
import { randomSoup, treeCases, type TreeCase } from '../bvh-shared/tree-cases.ts';

/** The CPU twin computes the same tree but for float rounding (hploc-cpu.ts); 1% covers that. */
const TWIN_TOLERANCE = 0.01;
/** The B2 done-when: the same SAH cost as tinybvh's BuildHPLOC, within 1%. */
const TINYBVH_TOLERANCE = 0.01;

interface Variant {
    name: string;
    options: HplocOptions;
    /** The tree the GPU's must match, and what it is called. */
    reference: { name: string; tolerance: number; build(geometry: TinyBvhInput): ReturnType<typeof buildHplocOnCpu> };
    /** Cases the GPU must refuse with these options. */
    refused?: readonly string[];
}

function variants(tinyBvh: TinyBvhBuilder): Variant[] {
    return [
        {
            name: 'default',
            options: {},
            reference: { name: 'the CPU twin', tolerance: TWIN_TOLERANCE, build: (geometry) => buildHplocOnCpu(geometry) },
        },
        {
            name: 'tinybvh codes and ties, leaves of 1',
            options: { mortonCodes: 'tinybvh-63', tieBreak: 'tinybvh', maxLeafPrimitives: 1 },
            reference: {
                name: "tinybvh's BuildHPLOC",
                tolerance: TINYBVH_TOLERANCE,
                build: (geometry) => tinyBvh.buildLayout(geometry, { builder: 'hploc', maxLeafPrimitives: 1 }).layout,
            },
            refused: ['all triangles equal'],
        },
        {
            name: 'tinybvh codes',
            options: { mortonCodes: 'tinybvh-63' },
            reference: { name: 'the CPU twin', tolerance: TWIN_TOLERANCE, build: (geometry) => buildHplocOnCpu(geometry, { mortonCodes: 'tinybvh-63' }) },
        },
    ];
}

async function checkVariant(
    builder: GpuHplocBuilder,
    variant: Variant,
    cases: readonly TreeCase[],
    log: (message: string) => void,
): Promise<{ failures: string[]; trees: Record<string, unknown> }> {
    const failures: string[] = [];
    const trees: Record<string, unknown> = {};
    const maxLeafPrimitives = variant.options.maxLeafPrimitives;
    const built = await builder.buildMany(
        cases.map((entry) => entry.input ?? entry.geometry),
        variant.options,
    );
    const layouts = await builder.readLayouts(built);
    built.destroy();
    for (const [index, layout] of layouts.entries()) {
        const { name, geometry } = cases[index];
        const where = `${variant.name}, ${name}`;
        if (variant.refused?.includes(name)) {
            if (layout) failures.push(`${where}: built, but must be refused`);
            log(`${where}: refused as expected (${built.meshes[index].problem})`);
            continue;
        }
        if (!layout) {
            failures.push(`${where}: ${built.meshes[index].problem}`);
            continue;
        }
        const problems = checkGpuLayout(layout, geometry, { maxLeafPrimitives });
        for (const problem of problems) failures.push(`${where}: ${problem}`);
        const cost = treeSahCost(layout);
        const reference = variant.reference.build(geometry);
        const referenceCost = treeSahCost(reference);
        if (Math.abs(cost - referenceCost) > variant.reference.tolerance * referenceCost) {
            failures.push(`${where}: SAH cost ${cost.toFixed(4)}, ${variant.reference.name}'s ${referenceCost.toFixed(4)}`);
        }
        trees[name] = {
            triangles: layout.triangleCount,
            nodes: layout.nodeCount,
            referenceNodes: reference.nodeCount,
            depth: layout.maxDepth,
            sahCost: Number(cost.toFixed(4)),
            sahCostVsReference: Number((cost / referenceCost).toFixed(5)),
        };
        log(
            `${where}: ${layout.triangleCount} triangles, ${layout.nodeCount} nodes (${reference.nodeCount}), depth ${layout.maxDepth} (${reference.maxDepth}), ` +
                `SAH ${cost.toFixed(3)} (${variant.reference.name} ${referenceCost.toFixed(3)}), ${problems.length === 0 ? 'valid' : 'INVALID'}`,
        );
    }
    return { failures, trees };
}

/** An index past the last vertex must get the mesh refused, and only that mesh. */
async function checkRefusal(builder: GpuHplocBuilder, log: (message: string) => void): Promise<string[]> {
    const positions = randomSoup(200, 11);
    const indices = Uint32Array.from({ length: 600 }, (_, index) => index);
    indices[77] = 600;
    const built = await builder.buildMany([{ positions: randomSoup(200, 10) }, { positions, indices }]);
    const [good, bad] = built.meshes;
    built.destroy();
    log(`mesh with a bad index: ${bad.problem ?? 'NOT refused'}; its neighbour: ${good.problem ?? 'fine'}`);
    const failures: string[] = [];
    if (!bad.problem) failures.push('a mesh with an index past its last vertex was not refused');
    if (good.problem) failures.push(`a good mesh next to a bad one was refused: ${good.problem}`);
    return failures;
}

await runGpuTest(
    async ({ device, log }) => {
        assert(GpuHplocBuilder.isSupported(device), 'this device has no subgroups; the fallback is tested by bvh-lbvh-fallback');
        const builder = new GpuHplocBuilder(device);
        await builder.prepare();
        log(`subgroups ${device.adapterInfo.subgroupMinSize}-${device.adapterInfo.subgroupMaxSize} wide: ${builder.slots} slot(s) per lane`);
        const tinyBvh = await createTinyBvhBuilder();
        const { cases, owned } = treeCases(device);
        const failures: string[] = [];
        const data: Record<string, unknown> = { slots: builder.slots, subgroupMinSize: device.adapterInfo.subgroupMinSize, subgroupMaxSize: device.adapterInfo.subgroupMaxSize };
        for (const variant of variants(tinyBvh)) {
            const result = await checkVariant(builder, variant, cases, log);
            failures.push(...result.failures);
            data[variant.name] = result.trees;
        }
        failures.push(...(await checkRefusal(builder, log)));
        for (const buffer of owned) buffer.destroy();
        assert(failures.length === 0, failures.slice(0, 15).join('\n'));
        return data;
    },
    { optionalFeatures: ['subgroups'] },
);
