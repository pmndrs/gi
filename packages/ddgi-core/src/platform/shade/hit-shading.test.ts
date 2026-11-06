import { describe, expect, it } from 'vitest';
import type { KernelBuilder, WgslModule } from '../contract/wgsl.ts';
import { countGroupStorageBuffers, orderModules, placeModules } from '../kernel-builder.ts';
import { SCENE_MODULE_ENTRIES, sceneModuleWgsl } from '../scene/wgsl.ts';
import { liftSunAboveMinimum, unusableEnvironmentTexture } from '../sky/platform-sky.ts';
import { projectionWgsl, RADIANCE_SH_COUNT, SKY_HARMONICS_LAYOUT } from '../sky/projection-wgsl.ts';
import { SH_COEFFICIENT_COUNT } from '../sky/spherical-harmonics.ts';
import { SKY_CHANGE_MODULE_ENTRIES, SKY_RADIANCE_SH_BYTES, skyChangeModuleWgsl } from './sky-change.ts';
import { DEFAULT_HIT_SHADING_SETTINGS, PlatformHitShading } from './hit-shading.ts';
import { SHADE_MODULE_ENTRIES, SHADE_UNIFORM_LAYOUT, shadeModuleWgsl } from './wgsl.ts';

/** Just enough of a GPUDevice to construct hit shading in Node. */
function fakeDevice(features: string[] = []): GPUDevice {
    const texture = (descriptor: GPUTextureDescriptor) => ({ ...descriptor, createView: () => ({}), destroy() {} });
    return {
        features: new Set(features),
        createTexture: texture,
        createBuffer: (descriptor: GPUBufferDescriptor) => ({ ...descriptor, destroy() {} }),
        createSampler: () => ({}),
        queue: { writeBuffer() {}, writeTexture() {} },
    } as unknown as GPUDevice;
}

/** A kernel builder these tests never call: hit shading only compiles gi/direct when a technique prepares it. */
const NO_KERNELS: KernelBuilder = {
    computeKernel: () => Promise.reject(new Error('no kernels in these tests')),
};

function stubModule(name: string, entries: WgslModule['entries'], storageBufferCount: number): WgslModule {
    return { name, dependencies: [], code: () => '', entries, resources: () => [], version: 0, storageBufferCount };
}

/** Storage buffers a module's entries really declare, counted the way the kernel builder counts a technique's groups. */
function declaredStorageBuffers(module: WgslModule): number {
    return countGroupStorageBuffers([{ entries: module.entries.map((entry, binding) => ({ ...entry, binding })) }]);
}

describe('hit shading', () => {
    const scene = stubModule('gi-scene', SCENE_MODULE_ENTRIES, SCENE_MODULE_ENTRIES.length);
    const rays = stubModule('gi-rays', [], 3);

    it('keeps the scene copy and hit shading at 2 storage buffers together (the binding budget)', () => {
        const shade = new PlatformHitShading(fakeDevice(), NO_KERNELS, scene, rays);
        expect(shade.module.storageBufferCount).toBe(0);
        expect(declaredStorageBuffers(shade.module)).toBe(0);
        expect(declaredStorageBuffers(scene)).toBe(2);
        expect(scene.storageBufferCount + shade.module.storageBufferCount).toBe(2);
        // With the tracer's 3, a DDGI update kernel has 3 of its 8 left for its own buffers (it uses 2).
        const modules = orderModules([shade.module]);
        expect(modules.map((module) => module.name)).toEqual(['gi-scene', 'gi-rays', 'gi-light-falloff', 'gi-shade']);
        // The falloff is the same object as the platform's, so a kernel listing both includes it once.
        expect(orderModules([shade.lightFalloff, shade.module]).filter((module) => module === shade.lightFalloff)).toHaveLength(1);
        expect(shade.lightFalloff.storageBufferCount + shade.lightFalloff.entries.length).toBe(0);
        expect(modules.reduce((sum, module) => sum + module.storageBufferCount, 0)).toBe(5);
    });

    it('declares the contract functions and its bindings at the slot it is given', () => {
        const code = shadeModuleWgsl({ group: 0, firstBinding: 7 });
        for (const name of ['struct GiSurface', 'fn giSurfaceAt(ray: GiRay, hit: GiHit) -> GiSurface', 'fn giDirectIrradiance(surface: GiSurface, groupMask: u32, random: vec2f) -> vec3f', 'fn giSkyRadiance(direction: vec3f) -> vec3f', 'fn giSkyIrradiance(surface: GiSurface) -> vec3f']) {
            expect(code).toContain(name);
        }
        expect(code).toContain('@group(0) @binding(7) var<uniform> giShade');
        expect(code).toContain('@group(0) @binding(8) var giSkyRadianceMap');
        expect(code).toContain('@group(0) @binding(9) var giSkySampler');
        // Every global name follows the platform's gi prefix rule.
        for (const match of code.matchAll(/^(?:fn|struct|const|var<[^>]*>|var)\s+([A-Za-z_]\w*)/gm)) {
            expect(match[1].toLowerCase().startsWith('gi'), match[1]).toBe(true);
        }
        // The scene module it calls into defines what it uses.
        const sceneCode = sceneModuleWgsl({ group: 0, firstBinding: 0 });
        for (const used of ['giInstanceTriangle', 'giInstanceGeometricNormal', 'giInstanceNormalAt', 'giInstanceUvAt', 'giMaterialAt', 'giLightCount', 'giDirectionalLightCount', 'giLight']) {
            expect(sceneCode).toContain(`fn ${used}(`);
        }
    });

    it('places its four entries after the modules it depends on, the light grid picks last', () => {
        const shade = new PlatformHitShading(fakeDevice(), NO_KERNELS, scene, rays);
        const placed = placeModules(orderModules([shade.module]));
        expect(placed[2].slot.firstBinding).toBe(SCENE_MODULE_ENTRIES.length);
        expect(shade.module.entries).toBe(SHADE_MODULE_ENTRIES);
        expect(shade.module.resources()).toHaveLength(4);
        expect(SHADE_MODULE_ENTRIES[3].texture?.sampleType).toBe('uint');
        expect(shadeModuleWgsl({ group: 0, firstBinding: 7 })).toContain('@group(0) @binding(10) var giLightGridPicks: texture_2d<u32>');
        // No grid until a scene has many lights: the module's resources are those it was made with.
        expect(shade.module.version).toBe(0);
    });

    it("lays the light grid's lookup out where the WGSL struct has it", () => {
        // WGSL's uniform layout for the struct's fields from maxShadowedLightsPerHit on: scalars align to 4, vec3 to 16 (size 12).
        const code = shadeModuleWgsl({ group: 0, firstBinding: 0 });
        const body = /struct GiShadeUniforms \{([^}]*)\}/.exec(code)?.[1] ?? '';
        const fields = body
            .split('\n')
            .map((line) => line.trim().replace(/,$/, ''))
            .filter((line) => line.includes(':') && !line.startsWith('//'))
            .map((line) => line.split(':').map((part) => part.trim()) as [string, string]);
        const first = fields.findIndex(([name]) => name === 'maxShadowedLightsPerHit');
        let offset = SHADE_UNIFORM_LAYOUT.maxShadowedLightsPerHit;
        const offsets = new Map<string, number>();
        for (const [name, type] of fields.slice(first)) {
            const isVector = type === 'vec3f' || type === 'vec3u';
            if (isVector) offset = Math.ceil(offset / 16) * 16;
            offsets.set(name, offset);
            offset += isVector ? 12 : 4;
        }
        expect(offsets.get('bounceAlbedoCap')).toBe(SHADE_UNIFORM_LAYOUT.bounceAlbedoCap);
        expect(offsets.get('lightGridActive')).toBe(SHADE_UNIFORM_LAYOUT.lightGrid);
        expect(offsets.get('lightGridOrigin')).toBe(SHADE_UNIFORM_LAYOUT.lightGrid + 8);
        expect(offsets.get('lightGridInverseCellSize')).toBe(SHADE_UNIFORM_LAYOUT.lightGrid + 20);
        expect(offsets.get('lightGridCells')).toBe(SHADE_UNIFORM_LAYOUT.lightGrid + 24);
        expect(offsets.get('lightGridOutsideCell')).toBe(SHADE_UNIFORM_LAYOUT.lightGrid + 36);
        expect(Math.ceil(offset / 16) * 16).toBe(SHADE_UNIFORM_LAYOUT.byteLength);
    });

    it("starts from the contract defaults and lays its uniforms out after the sky's harmonics", () => {
        const shade = new PlatformHitShading(fakeDevice(), NO_KERNELS, scene, rays);
        expect(shade.settings).toEqual(DEFAULT_HIT_SHADING_SETTINGS);
        // Radiance SH bands 0-2 at 0 (so a uniform binding of it alone needs no offset), then the 18 irradiance coefficients.
        expect(SHADE_UNIFORM_LAYOUT.skyRadiance).toBe(0);
        expect(SHADE_UNIFORM_LAYOUT.skyIrradiance).toBe(RADIANCE_SH_COUNT * 16);
        expect(SHADE_UNIFORM_LAYOUT.skyRadianceMapScale).toBe((RADIANCE_SH_COUNT + SH_COEFFICIENT_COUNT) * 16);
        expect(SHADE_UNIFORM_LAYOUT.maxShadowedLightsPerHit).toBe(SKY_HARMONICS_LAYOUT.byteLength);
        expect(SKY_HARMONICS_LAYOUT.byteLength).toBe(448);
        expect(SHADE_UNIFORM_LAYOUT.byteLength % 16).toBe(0);
        const code = shadeModuleWgsl({ group: 0, firstBinding: 0 });
        expect(code.indexOf('skyRadiance: array<vec4f, 9>')).toBeLessThan(code.indexOf('skyIrradiance: array<vec4f, 18>'));
    });

    it("hands techniques the sky's change as a module of two uniforms and no storage buffers", () => {
        const shade = new PlatformHitShading(fakeDevice(), NO_KERNELS, scene, rays);
        expect(shade.skyChange.entries).toBe(SKY_CHANGE_MODULE_ENTRIES);
        expect(shade.skyChange.storageBufferCount).toBe(0);
        expect(declaredStorageBuffers(shade.skyChange)).toBe(0);
        expect(shade.skyChange.entries.every((entry) => entry.buffer?.type === 'uniform')).toBe(true);
        // "Now" is the radiance SH range of hit shading's own uniforms; "before" a copy of the same size.
        const [now, before] = shade.skyChange.resources() as GPUBufferBinding[];
        expect(now.offset).toBe(SHADE_UNIFORM_LAYOUT.skyRadiance);
        expect(now.size).toBe(SKY_RADIANCE_SH_BYTES);
        expect((before.buffer as unknown as GPUBufferDescriptor).size).toBe(SKY_RADIANCE_SH_BYTES);
        const code = skyChangeModuleWgsl({ group: 0, firstBinding: 4 });
        expect(code).toContain('fn giSkyRadianceShNow(index: u32) -> vec3f');
        expect(code).toContain('fn giSkyRadianceShBefore(index: u32) -> vec3f');
        expect(code).toContain('@group(0) @binding(5) var<uniform> giSkyRadianceBefore');
        for (const match of code.matchAll(/^(?:fn|struct|const|var<[^>]*>|var)\s+([A-Za-z_]\w*)/gm)) {
            expect(match[1].toLowerCase().startsWith('gi'), match[1]).toBe(true);
        }
    });

    it('records the sky as seen by copying the radiance SH into "before"', () => {
        const shade = new PlatformHitShading(fakeDevice(), NO_KERNELS, scene, rays);
        const copies: unknown[][] = [];
        const encoder = { copyBufferToBuffer: (...args: unknown[]) => copies.push(args) } as unknown as GPUCommandEncoder;
        shade.recordSkySeen(encoder);
        const [now, before] = shade.skyChange.resources() as GPUBufferBinding[];
        expect(copies).toEqual([[now.buffer, SHADE_UNIFORM_LAYOUT.skyRadiance, before.buffer, 0, SKY_RADIANCE_SH_BYTES]]);
    });

    it('builds the subgroup reduction only on devices that have subgroups', () => {
        expect(new PlatformHitShading(fakeDevice(['subgroups']), NO_KERNELS, scene, rays).sky.reduction).toBe('subgroups');
        expect(new PlatformHitShading(fakeDevice(), NO_KERNELS, scene, rays).sky.reduction).toBe('workgroup');
        expect(() => new PlatformHitShading(fakeDevice(), NO_KERNELS, scene, rays, { skyReduction: 'subgroups' })).toThrow(/subgroups/);
        expect(projectionWgsl('subgroups')).toMatch(/^enable subgroups;/);
        expect(projectionWgsl('subgroups')).toContain('subgroupAdd(');
        expect(projectionWgsl('workgroup')).not.toContain('subgroup_invocation_id');
        // T2.6a: the reductions loop over the coefficients instead of writing out 18 inlined copies
        // (the cause of M1's slow compile), so the kernel holds one call site of the reduction per entry point.
        for (const reduction of ['subgroups', 'workgroup'] as const) {
            expect(projectionWgsl(reduction).match(/= giSkyReduce\(/g)).toHaveLength(2);
        }
    });
});

describe('the sky source checks', () => {
    const texture = (overrides: Partial<GPUTexture>): GPUTexture =>
        ({ format: 'rgba16float', usage: 0x4, dimension: '2d', width: 512, height: 256, depthOrArrayLayers: 1, ...overrides }) as GPUTexture;

    it('accepts float textures of the right shape and names what is wrong with the others', () => {
        expect(unusableEnvironmentTexture(texture({}), 'equirect')).toBeNull();
        expect(unusableEnvironmentTexture(texture({ format: 'rgba32float' }), 'equirect')).toBeNull();
        expect(unusableEnvironmentTexture(texture({ width: 64, height: 64, depthOrArrayLayers: 6 }), 'cube')).toBeNull();
        expect(unusableEnvironmentTexture(texture({ format: 'rgba8uint' }), 'equirect')).toMatch(/format/);
        expect(unusableEnvironmentTexture(texture({ usage: 0x2 }), 'equirect')).toMatch(/TEXTURE_BINDING/);
        expect(unusableEnvironmentTexture(texture({}), 'cube')).toMatch(/6 layers/);
        expect(unusableEnvironmentTexture(texture({ depthOrArrayLayers: 6 }), 'cube')).toMatch(/square/);
    });

    it('lifts a sun below 5 degrees to 5 degrees, keeping its azimuth', () => {
        expect(liftSunAboveMinimum([0, 1, 0])).toEqual([0, 1, 0]);
        const lifted = liftSunAboveMinimum([3, -1, 4]);
        expect(Math.asin(lifted[1]) * (180 / Math.PI)).toBeGreaterThanOrEqual(5);
        expect(Math.atan2(lifted[2], lifted[0])).toBeCloseTo(Math.atan2(4, 3), 12);
    });
});
