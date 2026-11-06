import { describe, expect, it } from 'vitest';
import { checkFieldLimits, fieldBytes, fieldLayout, keyDeltaStrideWords, propagationScratchStrideWords, scratchLayout } from './layout.ts';
import { ALL_LIGHT_GROUPS_MASK, defaultLightGroups, layerPlan, lightGroupCount, MAX_IRRADIANCE_LAYERS, writeLayerWeights } from './light-groups.ts';
import { probeGrid } from './settings.ts';
import { blendKernelWgsl } from './wgsl/blend.ts';
import { composeKernelWgsl } from './wgsl/compose.ts';
import { propagateKernelWgsl } from './wgsl/propagate.ts';
import { writeKernelWgsl } from './wgsl/write.ts';

describe('light groups: the layer plan (T2.5, m2-lights.md §5)', () => {
    it('keeps one layer holding everything with groups off, and composes nothing', () => {
        for (const off of [null, undefined, { ...defaultLightGroups(0) }, { ...defaultLightGroups(-1) }]) {
            const plan = layerPlan(off);
            expect(plan.layers).toEqual([{ kind: 'everything', group: 0, groupMask: ALL_LIGHT_GROUPS_MASK, holdsEnvironment: true }]);
            expect(plan.composed).toBe(false);
            expect(plan.groupLayers).toEqual([0, 0, 0, 0]);
        }
    });

    it('gives each group a layer, groups past the count share the last, and the environment comes after them', () => {
        const plan = layerPlan(defaultLightGroups(2));
        expect(plan.layers.map((layer) => layer.kind)).toEqual(['lightGroup', 'lightGroup', 'environment']);
        expect(plan.groupLayers).toEqual([0, 1, 1, 1]);
        expect(plan.layers.map((layer) => layer.groupMask)).toEqual([0b0001, 0b1110, 0]);
        expect(plan.environmentLayer).toBe(2);
        expect(plan.layers.filter((layer) => layer.holdsEnvironment)).toHaveLength(1);
        // Every group's light goes somewhere, once.
        expect(plan.layers.reduce((mask, layer) => mask | layer.groupMask, 0)).toBe(ALL_LIGHT_GROUPS_MASK);
        expect(layerPlan(defaultLightGroups(4)).groupLayers).toEqual([0, 1, 2, 3]);
        expect(layerPlan(defaultLightGroups(4)).layers).toHaveLength(5);
        expect(lightGroupCount({ ...defaultLightGroups(9) })).toBe(4);
        expect(lightGroupCount({ ...defaultLightGroups(2.4) })).toBe(2);
        expect(layerPlan(defaultLightGroups(4)).layers.length).toBeLessThanOrEqual(MAX_IRRADIANCE_LAYERS);
    });

    it('writes each layer its weight, and says when one changed (bit for bit as the f32 uniform holds it)', () => {
        const groups = defaultLightGroups(2);
        const plan = layerPlan(groups);
        const weights = new Float32Array(MAX_IRRADIANCE_LAYERS * 4).fill(Number.NaN);
        expect(writeLayerWeights(plan, groups, weights)).toBe(true);
        expect(Array.from(weights.subarray(0, 12))).toEqual([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]);
        expect(writeLayerWeights(plan, groups, weights)).toBe(false);
        groups.weights[1] = [0.1, 0.5, 2];
        groups.environmentWeight = [0, 0, 0];
        expect(writeLayerWeights(plan, groups, weights)).toBe(true);
        expect(Array.from(weights.subarray(4, 12))).toEqual([Math.fround(0.1), 0.5, 2, 0, 0, 0, 0, 0]);
        // The same weights again (0.1 as it was) change nothing.
        groups.weights[1] = [0.1, 0.5, 2];
        expect(writeLayerWeights(plan, groups, weights)).toBe(false);
        // Not a number or negative counts as 1; a missing group's weight is 1.
        groups.weights = [[Number.NaN, -1, 3]];
        writeLayerWeights(plan, groups, weights);
        expect(Array.from(weights.subarray(0, 8))).toEqual([1, 1, 3, 0, 1, 1, 1, 0]);
    });
});

describe('light groups: the layout', () => {
    const grid = probeGrid({ center: [4.1, 1.4, 2.0], size: [8.4, 3.0, 4.2], spacing: 0.5 });

    it('puts the layers side by side in the layered atlas, and keeps the composed one a single block', () => {
        const one = fieldLayout(grid, 6, 14, 'rg16float');
        const three = fieldLayout(grid, 6, 14, 'rg16float', 3);
        expect(one.layeredIrradiance).toEqual(one.irradiance);
        expect(three.irradiance).toEqual(one.irradiance);
        expect(three.layeredIrradiance.width).toBe(3 * one.irradiance.width);
        // Layers and the composed atlas: 3 + 1 blocks of 512 bytes a probe, against 1 with groups off.
        expect(fieldBytes(three) - fieldBytes(one)).toBe(3 * grid.probeCount * 512);
    });

    it('checks the layered atlas against the texture width limit', () => {
        // 210 probes of 8 texels in 5 blocks: 8,400 texels, past 8,192; one block is 1,680.
        const wideGrid = { ...grid, counts: [210, 7, 9] as [number, number, number], probeCount: 210 * 63 };
        const limits = { maxTextureDimension2D: 8192, maxTextureArrayLayers: 256 };
        expect(() => checkFieldLimits(fieldLayout(wideGrid, 6, 14, 'rg16float', 5), limits)).toThrow(/maxTextureDimension2D/);
        expect(() => checkFieldLimits(fieldLayout(wideGrid, 6, 14, 'rg16float'), limits)).not.toThrow();
    });

    it("sizes the scratch, the key deltas and propagation's scratch per layer", () => {
        const one = scratchLayout(6, 14);
        const three = scratchLayout(6, 14, 3);
        expect(three.irradianceWords).toBe(one.irradianceWords);
        expect(three.distanceOffset).toBe(3 * one.irradianceWords);
        expect(three.strideWords - one.strideWords).toBe(2 * one.irradianceWords);
        expect(keyDeltaStrideWords(6, 3) - keyDeltaStrideWords(6)).toBe(2 * 72);
        expect(propagationScratchStrideWords(6, 3)).toBe(3 * propagationScratchStrideWords(6));
    });

    it('builds every kernel with one layer as before, and loops the layers with several', () => {
        const scratch = scratchLayout(6, 14, 3);
        const blend = blendKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch, reduction: 'workgroup' });
        expect(blend).toContain('const DDGI_IRRADIANCE_LAYERS: u32 = 3u;');
        expect(blend).toContain('for (var turn = 0u; turn < DDGI_IRRADIANCE_LAYERS; turn++)');
        const write = writeKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch, distanceFormat: 'rg16float' });
        expect(write).toContain('const DDGI_IRRADIANCE_LAYERS: u32 = 3u;');
        const propagate = propagateKernelWgsl({ irradianceTexels: 6, irradianceLayers: 3, reduction: 'workgroup', listCapacity: 10 });
        expect(propagate).toContain('const DDGI_IRRADIANCE_LAYERS: u32 = 3u;');
        expect(blendKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch: scratchLayout(6, 14), reduction: 'workgroup' })).toContain('const DDGI_IRRADIANCE_LAYERS: u32 = 1u;');
        expect(() => composeKernelWgsl({ irradianceTexels: 6, irradianceLayers: 1 })).toThrow(RangeError);
        // Compose binds 2 storage buffers: the list and its state.
        expect(composeKernelWgsl({ irradianceTexels: 6, irradianceLayers: 3 }).match(/var<storage/g)).toHaveLength(2);
    });
});
