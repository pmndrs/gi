import { describe, expect, it } from 'vitest';
import type { KernelBuilder, WgslModule } from '../contract/wgsl.ts';
import { countGroupStorageBuffers, orderModules, placeModules } from '../kernel-builder.ts';
import { SCENE_MODULE_ENTRIES, sceneModuleWgsl } from '../scene/wgsl.ts';
import { liftSunAboveMinimum, unusableEnvironmentTexture } from '../sky/platform-sky.ts';
import { projectionWgsl, RADIANCE_SH_COUNT, SKY_HARMONICS_LAYOUT } from '../sky/projection-wgsl.ts';
import { SH_COEFFICIENT_COUNT } from '../sky/spherical-harmonics.ts';
import { createTextureSamplingModule, TEXTURE_SAMPLING_ENTRIES, textureSamplingWgsl } from '../textures/texture-sampling-wgsl.ts';
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

/** The texture store as its sampling module binds it: a view, a sampler and a version (textures/texture-sampling-wgsl.ts). */
const STORE_BINDING = { view: { label: 'texture store' } as unknown as GPUTextureView, sampler: { label: 'store sampler' } as unknown as GPUSampler, version: 0 };

/** A store stand-in that hands out one sampling module, as `PlatformTextureStore.samplingModule` does. */
function storeFor(binding: { view: GPUTextureView; sampler: GPUSampler; version: number }) {
    let module: WgslModule | null = null;
    return { samplingModule: (sceneModule: WgslModule) => (module ??= createTextureSamplingModule(sceneModule, binding)) };
}

const TEXTURES = storeFor(STORE_BINDING);

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
        const shade = new PlatformHitShading(fakeDevice(), NO_KERNELS, scene, rays, TEXTURES);
        expect(shade.module.storageBufferCount).toBe(0);
        expect(declaredStorageBuffers(shade.module)).toBe(0);
        expect(declaredStorageBuffers(scene)).toBe(2);
        expect(scene.storageBufferCount + shade.module.storageBufferCount).toBe(2);
        // With the tracer's 3, a DDGI update kernel has 3 of its 8 left for its own buffers (it uses 2).
        const modules = orderModules([shade.module]);
        expect(modules.map((module) => module.name)).toEqual(['gi-scene', 'gi-rays', 'gi-texture-store', 'gi-light-falloff', 'gi-shade']);
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

    it('places its four entries after the modules it depends on, the texture store with its sampler among them', () => {
        const shade = new PlatformHitShading(fakeDevice(), NO_KERNELS, scene, rays, TEXTURES);
        const placed = placeModules(orderModules([shade.module]));
        // Scene, rays (no entries in this stub), the texture store's two, the falloff (none), then hit shading's.
        expect(placed[2].module.name).toBe('gi-texture-store');
        expect(placed[2].slot.firstBinding).toBe(SCENE_MODULE_ENTRIES.length);
        expect(placed[4].slot.firstBinding).toBe(SCENE_MODULE_ENTRIES.length + TEXTURE_SAMPLING_ENTRIES.length);
        expect(shade.module.entries).toBe(SHADE_MODULE_ENTRIES);
        expect(shade.module.resources()).toHaveLength(4);
        expect(SHADE_MODULE_ENTRIES[3].texture?.sampleType).toBe('uint');
        expect(TEXTURE_SAMPLING_ENTRIES[0].texture?.viewDimension).toBe('2d-array');
        expect(TEXTURE_SAMPLING_ENTRIES[1].sampler?.type).toBe('filtering');
        expect(placed[2].module.resources()).toEqual([STORE_BINDING.view, STORE_BINDING.sampler]);
        const code = shadeModuleWgsl({ group: 0, firstBinding: 7 });
        expect(code).toContain('@group(0) @binding(10) var giLightGridPicks: texture_2d<u32>');
        const samplingCode = textureSamplingWgsl({ group: 0, firstBinding: 11 });
        expect(samplingCode).toContain('@group(0) @binding(11) var giTextureStore: texture_2d_array<f32>');
        expect(samplingCode).toContain('@group(0) @binding(12) var giTextureSampler: sampler');
        // No grid until a scene has many lights: the module's resources are those it was made with.
        expect(shade.module.version).toBe(0);
    });

    it("binds the texture store through a module that follows the store's version, so kernels rebind when its array grows", () => {
        const binding = { ...STORE_BINDING, version: 0 };
        const shade = new PlatformHitShading(fakeDevice(), NO_KERNELS, scene, rays, storeFor(binding));
        const sampling = shade.module.dependencies.find((module) => module.name === 'gi-texture-store');
        expect(sampling).toBeDefined();
        const before = sampling!.version;
        binding.version = 1;
        expect(sampling!.version).not.toBe(before);
        expect(sampling!.storageBufferCount).toBe(0);
    });

    it('writes the sampleTextures setting when it changes, on by default', () => {
        const writes: { offset: number; data: unknown }[] = [];
        const device = fakeDevice();
        (device.queue as unknown as { writeBuffer: unknown }).writeBuffer = (_buffer: unknown, offset: number, data: unknown) => writes.push({ offset, data });
        const shade = new PlatformHitShading(device, NO_KERNELS, scene, rays, TEXTURES);
        expect(shade.settings.sampleTextures).toBe(true);
        const sampleTexturesWord = () => writes.filter((write) => write.offset === SHADE_UNIFORM_LAYOUT.sampleTextures).map((write) => (write.data as Uint32Array)[0]);
        // Only the settings upload is looked at (a private step of encode, which also bakes the sky).
        (shade as unknown as { uploadSettings(): void }).uploadSettings();
        shade.settings.sampleTextures = false;
        (shade as unknown as { uploadSettings(): void }).uploadSettings();
        (shade as unknown as { uploadSettings(): void }).uploadSettings();
        expect(sampleTexturesWord()).toEqual([1, 0]);
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
        expect(offsets.get('sampleTextures')).toBe(SHADE_UNIFORM_LAYOUT.sampleTextures);
        expect(Math.ceil(offset / 16) * 16).toBe(SHADE_UNIFORM_LAYOUT.byteLength);
    });

    it("starts from the contract defaults and lays its uniforms out after the sky's harmonics", () => {
        const shade = new PlatformHitShading(fakeDevice(), NO_KERNELS, scene, rays, TEXTURES);
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
        const shade = new PlatformHitShading(fakeDevice(), NO_KERNELS, scene, rays, TEXTURES);
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
        const shade = new PlatformHitShading(fakeDevice(), NO_KERNELS, scene, rays, TEXTURES);
        const copies: unknown[][] = [];
        const encoder = { copyBufferToBuffer: (...args: unknown[]) => copies.push(args) } as unknown as GPUCommandEncoder;
        shade.recordSkySeen(encoder);
        const [now, before] = shade.skyChange.resources() as GPUBufferBinding[];
        expect(copies).toEqual([[now.buffer, SHADE_UNIFORM_LAYOUT.skyRadiance, before.buffer, 0, SKY_RADIANCE_SH_BYTES]]);
    });

    it('builds the subgroup reduction only on devices that have subgroups', () => {
        expect(new PlatformHitShading(fakeDevice(['subgroups']), NO_KERNELS, scene, rays, TEXTURES).sky.reduction).toBe('subgroups');
        expect(new PlatformHitShading(fakeDevice(), NO_KERNELS, scene, rays, TEXTURES).sky.reduction).toBe('workgroup');
        expect(() => new PlatformHitShading(fakeDevice(), NO_KERNELS, scene, rays, TEXTURES, { skyReduction: 'subgroups' })).toThrow(/subgroups/);
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
