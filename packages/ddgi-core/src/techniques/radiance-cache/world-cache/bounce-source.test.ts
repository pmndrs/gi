import { describe, expect, it } from 'vitest';
import type { ShadingModule, ShadingParameter } from '../../../platform/contract/shading-module.ts';
import { DEFAULT_RADIANCE_CACHE_SETTINGS } from '../settings.ts';
import { RadianceCacheShadingModule } from '../shading.ts';
import { BOUNCE_SOURCE_GROUP, SAMPLE_BOUNCE_SOURCE_ENTRY_POINT, sampleBounceSourceCellsWgsl, shadingModuleLayoutEntries, shadingModuleUniformBytes } from './bounce-source.ts';
import { DEFAULT_HASH_GRID_CACHE_OPTIONS, hashGridCacheBufferLayout, hashGridCacheCapacity } from './capacity.ts';
import { hashGridCacheClearOptions, hashGridCacheSamplesSky, sameClearOptions } from './hash-grid-cache.ts';
import { hashGridCacheRayListLayout } from './hash-grid-population-wgsl.ts';

/** A stand-in for another technique's module (the radiance cache never imports one): DDGI's shape, plus a uniform. */
function fakeModule(withUniform: boolean): ShadingModule {
    const texture = (): GPUTexture => ({}) as GPUTexture;
    const parameters: ShadingParameter[] = [
        { kind: 'texture', name: 'fieldIrradiance', dimension: '2d-array', sampleType: 'float', texture },
        { kind: 'sampler', name: 'fieldIrradianceSampler', texture: 'fieldIrradiance', filtering: 'linear', addressMode: 'clamp-to-edge' },
        { kind: 'texture', name: 'fieldTable', dimension: '2d', sampleType: 'unfilterable-float', texture },
    ];
    if (withUniform) parameters.push({ kind: 'vec4f', name: 'fieldScale', value: () => new Float32Array([1, 2, 3, 4]) });
    return {
        techniqueId: 'fake-field',
        code: 'fn giIrradiance(p: vec3f, n: vec3f, v: vec3f, uv: vec2f, a: texture_2d_array<f32>, s: sampler, t: texture_2d<f32>) -> vec3f { return vec3f(1.0); }',
        parameters,
        provides: { radiance: false, worldSpaceIrradiance: true },
        globalsCode: (group: number) => `// globals at group ${group}\nfn giIrradianceAt(p: vec3f, n: vec3f, v: vec3f, uv: vec2f) -> vec3f { return vec3f(1.0); }`,
        writeUniforms: () => {},
        version: 1,
    };
}

describe("experiment E10: the world cache's second bounce from a stacked technique", () => {
    it("is 'auto' by default (the fallback when one is stacked, rays otherwise), its indirect cap 4", () => {
        expect(DEFAULT_RADIANCE_CACHE_SETTINGS.secondBounce).toBe('auto');
        expect(DEFAULT_HASH_GRID_CACHE_OPTIONS.secondBounce).toBe('auto');
        expect(DEFAULT_RADIANCE_CACHE_SETTINGS.fallbackIndirectSampleCount).toBe(4);
        expect(DEFAULT_HASH_GRID_CACHE_OPTIONS.fallbackIndirectSampleCount).toBe(4);
    });

    it("reads only a world-space module: the radiance cache's own (screen space) says it is none", () => {
        const own = new RadianceCacheShadingModule({ shadingTextures: () => ({}) as never });
        expect(own.provides.worldSpaceIrradiance).toBe(false);
        expect(fakeModule(false).provides.worldSpaceIrradiance).toBe(true);
    });

    it("binds a shading module's textures and samplers in order, then its vec4f uniforms in one buffer", () => {
        const entries = shadingModuleLayoutEntries(fakeModule(true));
        expect(entries.map((entry) => entry.binding)).toEqual([0, 1, 2, 3]);
        expect(entries[0].texture).toEqual({ sampleType: 'float', viewDimension: '2d-array' });
        expect(entries[1].sampler).toEqual({ type: 'filtering' });
        expect(entries[2].texture).toEqual({ sampleType: 'unfilterable-float', viewDimension: '2d' });
        expect(entries[3].buffer).toEqual({ type: 'uniform' });
        expect(shadingModuleUniformBytes(fakeModule(true))).toBe(16);
        // DDGI's module has no vec4f parameters (its version 2): no buffer, as globalsCode declares none.
        expect(shadingModuleLayoutEntries(fakeModule(false))).toHaveLength(3);
        expect(shadingModuleUniformBytes(fakeModule(false))).toBe(0);
    });

    it("includes the module's globals in group 3 and weighs its irradiance by its coverage", () => {
        const layout = hashGridCacheBufferLayout(hashGridCacheCapacity(DEFAULT_HASH_GRID_CACHE_OPTIONS));
        const code = sampleBounceSourceCellsWgsl(layout, fakeModule(false));
        expect(BOUNCE_SOURCE_GROUP).toBe(3);
        expect(code).toContain(`// globals at group ${BOUNCE_SOURCE_GROUP}`);
        expect(code).toContain(`fn ${SAMPLE_BOUNCE_SOURCE_ENTRY_POINT}(`);
        expect(code).toContain('giIrradianceAt(surface.position, surface.normal, worldCacheProbeRayDirection(probe_ray), vec2f(0.0))');
        expect(code).toContain('saturate(giCoverageAt(surface.position, surface.normal))');
        // One sample per first-bounce entry into its cell's indirect scratch, and the bypass routing (D38).
        expect(code).toContain('HASHGRIDCACHE_UPDATECELLVALUEINDIRECTBUFFER + 4u * cell_index');
        expect(code).toContain('first_bounce_bypass_buffer');
    });

    it('empties the cache when the estimator that runs changes, and only with multibounce on', () => {
        const options = { ...DEFAULT_HASH_GRID_CACHE_OPTIONS };
        const rays = hashGridCacheClearOptions(options, 'gi1-rays');
        const fallback = hashGridCacheClearOptions(options, 'fallback');
        expect(sameClearOptions(rays, fallback)).toBe(false);
        expect(sameClearOptions(rays, hashGridCacheClearOptions(options))).toBe(true);
        const single = { ...options, useMultibounce: false };
        expect(sameClearOptions(hashGridCacheClearOptions(single, 'gi1-rays'), hashGridCacheClearOptions(single, 'fallback'))).toBe(true);
    });

    it("leaves the sky out of the cache's light choice with the fallback, whose field holds it (RC3.4's option)", () => {
        expect(hashGridCacheSamplesSky({ environmentLightEnable: true }, 'gi1-rays')).toBe(true);
        expect(hashGridCacheSamplesSky({ environmentLightEnable: true }, 'fallback')).toBe(false);
        expect(hashGridCacheSamplesSky({ environmentLightEnable: false }, 'gi1-rays')).toBe(false);
        // RC3.4's default: the sky is a light (register D59), except with the fallback.
        expect(hashGridCacheSamplesSky(DEFAULT_HASH_GRID_CACHE_OPTIONS, 'gi1-rays')).toBe(true);
        expect(hashGridCacheSamplesSky(DEFAULT_HASH_GRID_CACHE_OPTIONS, 'fallback')).toBe(false);
    });

    it('drops the second-bounce lists without second-bounce rays: 9 words a probe ray instead of 32', () => {
        const numCells = hashGridCacheCapacity(DEFAULT_HASH_GRID_CACHE_OPTIONS).numCells;
        for (const rays of [80 * 45 * 64, 120 * 68 * 64]) {
            const full = hashGridCacheRayListLayout(rays, numCells);
            const firstOnly = hashGridCacheRayListLayout(rays, numCells, false);
            expect(full.bytes).toBe(32 * rays * 4);
            expect(firstOnly.bytes).toBe(9 * rays * 4);
            expect(firstOnly.maxCombinedRayCount).toBe(rays);
            // The lists every mode uses keep their room.
            expect(firstOnly.resolveBuffer - firstOnly.visibilityRayBuffer).toBe(rays);
            expect(firstOnly.multibounceDirectBuffer - firstOnly.firstBounceBypassBuffer).toBe(rays);
        }
    });
});
