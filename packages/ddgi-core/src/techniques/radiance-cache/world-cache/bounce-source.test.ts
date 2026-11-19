import { describe, expect, it } from 'vitest';
import type { ShadingModule, ShadingParameter } from '../../../platform/contract/shading-module.ts';
import { DEFAULT_RADIANCE_CACHE_SETTINGS } from '../settings.ts';
import { BOUNCE_SOURCE_GROUP, SAMPLE_BOUNCE_SOURCE_ENTRY_POINT, sampleBounceSourceCellsWgsl, shadingModuleLayoutEntries, shadingModuleUniformBytes } from './bounce-source.ts';
import { DEFAULT_HASH_GRID_CACHE_OPTIONS, hashGridCacheBufferLayout, hashGridCacheCapacity } from './capacity.ts';
import { hashGridCacheClearOptions, sameClearOptions } from './hash-grid-cache.ts';

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
        provides: { radiance: false },
        globalsCode: (group: number) => `// globals at group ${group}\nfn giIrradianceAt(p: vec3f, n: vec3f, v: vec3f, uv: vec2f) -> vec3f { return vec3f(1.0); }`,
        writeUniforms: () => {},
        version: 1,
    };
}

describe("experiment E10: the world cache's second bounce from a stacked technique", () => {
    it('is off by default: GI-1.2 traces its second-bounce rays', () => {
        expect(DEFAULT_RADIANCE_CACHE_SETTINGS.secondBounce).toBe('gi1-rays');
        expect(DEFAULT_HASH_GRID_CACHE_OPTIONS.secondBounce).toBe('gi1-rays');
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

    it("includes the module's globals in group 3 and calls its parameterless giIrradianceAt", () => {
        const layout = hashGridCacheBufferLayout(hashGridCacheCapacity(DEFAULT_HASH_GRID_CACHE_OPTIONS));
        const code = sampleBounceSourceCellsWgsl(layout, fakeModule(false));
        expect(BOUNCE_SOURCE_GROUP).toBe(3);
        expect(code).toContain(`// globals at group ${BOUNCE_SOURCE_GROUP}`);
        expect(code).toContain(`fn ${SAMPLE_BOUNCE_SOURCE_ENTRY_POINT}(`);
        expect(code).toContain('giIrradianceAt(surface.position, surface.normal, worldCacheProbeRayDirection(probe_ray), vec2f(0.0))');
        // One sample per first-bounce entry into its cell's indirect scratch, and the bypass routing (D38).
        expect(code).toContain('HASHGRIDCACHE_UPDATECELLVALUEINDIRECTBUFFER + 4u * cell_index');
        expect(code).toContain('first_bounce_bypass_buffer');
    });

    it('empties the cache when the estimator that runs changes, and only with multibounce on', () => {
        const options = { ...DEFAULT_HASH_GRID_CACHE_OPTIONS };
        const rays = hashGridCacheClearOptions(options, 'gi1-rays');
        const volume = hashGridCacheClearOptions(options, 'ddgi-volume');
        expect(sameClearOptions(rays, volume)).toBe(false);
        expect(sameClearOptions(rays, hashGridCacheClearOptions(options))).toBe(true);
        const single = { ...options, useMultibounce: false };
        expect(sameClearOptions(hashGridCacheClearOptions(single, 'gi1-rays'), hashGridCacheClearOptions(single, 'ddgi-volume'))).toBe(true);
    });
});
