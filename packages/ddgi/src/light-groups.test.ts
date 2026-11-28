import { describe, expect, it } from 'vitest';
import { DIRECT_SKY_LIGHT_BIT } from '@pmndrs/gi-core/platform/contract/shade';
import { checkFieldLimits, fieldBytes, fieldLayout, keyDeltaStrideWords, propagationScratchStrideWords, scratchLayout } from './layout.ts';
import { ALL_LIGHT_GROUPS_MASK, defaultLightGroups, layerPlan, lightGroupCount, MAX_IRRADIANCE_LAYERS, sunLayerWeight, writeLayerWeights } from './light-groups.ts';
import { probeGrid } from './settings.ts';
import { blendKernelWgsl } from './wgsl/blend.ts';
import { composeKernelWgsl } from './wgsl/compose.ts';
import { propagateKernelWgsl } from './wgsl/propagate.ts';
import { DDGI_LIGHT_GROUP_MASK } from './wgsl/trace.ts';
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

describe('the sun layer (T2.6b, m2-lights.md §6.1)', () => {
    const sun = { type: 'directional', color: [1, 0.5, 0.25], intensity: 4, direction: [0, -1, 0] } as const;

    it('adds the sun as the last layer: after everything else with groups off, after the environment with groups', () => {
        const alone = layerPlan(null, true);
        expect(alone.layers.map((layer) => layer.kind)).toEqual(['everything', 'sun']);
        expect(alone.sunLayer).toBe(1);
        expect(alone.environmentLayer).toBe(0);
        expect(alone.composed).toBe(true);
        expect(alone.groupLayers).toEqual([0, 0, 0, 0]);
        expect(layerPlan(null).sunLayer).toBe(-1);
        const withGroups = layerPlan(defaultLightGroups(2), true);
        expect(withGroups.layers.map((layer) => layer.kind)).toEqual(['lightGroup', 'lightGroup', 'environment', 'sun']);
        expect(withGroups.sunLayer).toBe(3);
        expect(withGroups.environmentLayer).toBe(2);
        // The sun's layer holds no group's light and no environment.
        expect(withGroups.layers[3]).toMatchObject({ groupMask: 0, holdsEnvironment: false });
    });

    it('holds at most 6 layers: 4 groups, the environment and the sun, which the blend accepts', () => {
        const full = layerPlan(defaultLightGroups(4), true);
        expect(full.layers).toHaveLength(MAX_IRRADIANCE_LAYERS);
        expect(full.sunLayer).toBe(5);
        expect(layerPlan(defaultLightGroups(9), true).layers).toHaveLength(MAX_IRRADIANCE_LAYERS);
        const scratch = scratchLayout(6, 14, full.layers.length);
        expect(() => blendKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch, reduction: 'subgroups', sunLayer: full.sunLayer })).not.toThrow();
        expect(() => blendKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch, reduction: 'subgroups', sunLayer: 6 })).toThrow(RangeError);
    });

    it("weighs the sun layer by the sun's colour x intensity, times its group's weight with groups on", () => {
        expect(sunLayerWeight(layerPlan(null, true), null, sun)).toEqual([4, 2, 1]);
        expect(sunLayerWeight(layerPlan(null, true), null, null)).toEqual([0, 0, 0]);
        const groups = defaultLightGroups(2);
        groups.weights[0] = [0.5, 0.5, 0.5];
        groups.weights[1] = [1, 0, 2];
        const plan = layerPlan(groups, true);
        expect(sunLayerWeight(plan, groups, sun)).toEqual([2, 1, 0.5]);
        // A sun in group 3 shares group 1's layer (2 groups), so group 1's weight.
        expect(sunLayerWeight(plan, groups, { ...sun, group: 3 })).toEqual([4, 0, 2]);
        const weights = new Float32Array(MAX_IRRADIANCE_LAYERS * 4).fill(Number.NaN);
        expect(writeLayerWeights(plan, groups, weights, sunLayerWeight(plan, groups, sun))).toBe(true);
        expect(Array.from(weights.subarray(12, 16))).toEqual([2, 1, 0.5, 0]);
        // A new sun colour is a weight change; the same again is none. A sun at 0 weighs 0, not 1.
        expect(writeLayerWeights(plan, groups, weights, [2, 1, 0.5])).toBe(false);
        expect(writeLayerWeights(plan, groups, weights, [0, 0, 0])).toBe(true);
        expect(Array.from(weights.subarray(12, 16))).toEqual([0, 0, 0, 0]);
    });

    it("weighs the sun layer by the sun's light in the probe's mean, and leaves the blend as it was without one", () => {
        const withSun = blendKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch: scratchLayout(6, 14, 2), reduction: 'workgroup', sunLayer: 1 });
        expect(withSun).toContain('const DDGI_SUN_LAYER: u32 = 1u;');
        expect(withSun).toContain('ddgi.sunLight.rgb');
        const without = blendKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch: scratchLayout(6, 14, 2), reduction: 'workgroup' });
        expect(without).not.toContain('ddgi.sunLight');
        // Propagation writes a fine probe's mean the same way.
        const propagateWithSun = propagateKernelWgsl({ irradianceTexels: 6, irradianceLayers: 2, reduction: 'workgroup', listCapacity: 10, sunLayer: 1 });
        expect(propagateWithSun).toContain('const DDGI_SUN_LAYER: u32 = 1u;');
        expect(propagateWithSun).toContain('select(vec3f(1.0), ddgi.sunLight.rgb, layer == DDGI_SUN_LAYER) * value');
        expect(propagateKernelWgsl({ irradianceTexels: 6, irradianceLayers: 2, reduction: 'workgroup', listCapacity: 10 })).not.toContain('ddgi.sunLight');
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
        expect(blend).toContain('let packed = ddgiRays[ddgiRayResult(slot, ray, layer)];');
        expect(blend).not.toContain('var<workgroup> ddgiRayRadiance');
        const write = writeKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch, distanceFormat: 'rg16float' });
        expect(write).toContain('const DDGI_IRRADIANCE_LAYERS: u32 = 3u;');
        const propagate = propagateKernelWgsl({ irradianceTexels: 6, irradianceLayers: 3, reduction: 'workgroup', listCapacity: 10 });
        expect(propagate).toContain('const DDGI_IRRADIANCE_LAYERS: u32 = 3u;');
        expect(blendKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch: scratchLayout(6, 14), reduction: 'workgroup' })).toContain('const DDGI_IRRADIANCE_LAYERS: u32 = 1u;');
        expect(() => composeKernelWgsl({ irradianceTexels: 6, irradianceLayers: 1 })).toThrow(RangeError);
        // The blend's workgroup memory holds at most MAX_IRRADIANCE_LAYERS layers; past it, a clear error when it is set up.
        expect(() => blendKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch: scratchLayout(6, 14, MAX_IRRADIANCE_LAYERS + 1), reduction: 'workgroup' })).toThrow(/at most 6 irradiance layers/);
        expect(() => blendKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch: scratchLayout(6, 14, MAX_IRRADIANCE_LAYERS), reduction: 'workgroup' })).not.toThrow();
        // Compose binds 2 storage buffers: the list and its state.
        expect(composeKernelWgsl({ irradianceTexels: 6, irradianceLayers: 3 }).match(/var<storage/g)).toHaveLength(2);
    });
});

describe("the platform's direct sky light (RC3.4): DDGI never opts in", () => {
    it('the sky light bit is above every light group, and DDGI\'s trace mask never holds it', () => {
        expect(DIRECT_SKY_LIGHT_BIT & ALL_LIGHT_GROUPS_MASK).toBe(0);
        // Its trace's group mask holds group bits only (gi/direct's layer masks are within
        // ALL_LIGHT_GROUPS_MASK), and the functions gi/direct calls never read the bit.
        expect(DDGI_LIGHT_GROUP_MASK & DIRECT_SKY_LIGHT_BIT).toBe(0);
    });
});
