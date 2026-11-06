import { describe, expect, it } from 'vitest';
import type { WgslModule } from '../contract/wgsl.ts';
import { countGroupStorageBuffers, orderModules, placeModules } from '../kernel-builder.ts';
import { SCENE_MODULE_ENTRIES, sceneModuleWgsl } from '../scene/wgsl.ts';
import { liftSunAboveMinimum, unusableEnvironmentTexture } from '../sky/platform-sky.ts';
import { projectionWgsl } from '../sky/projection-wgsl.ts';
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
        const shade = new PlatformHitShading(fakeDevice(), scene, rays);
        expect(shade.module.storageBufferCount).toBe(0);
        expect(declaredStorageBuffers(shade.module)).toBe(0);
        expect(declaredStorageBuffers(scene)).toBe(2);
        expect(scene.storageBufferCount + shade.module.storageBufferCount).toBe(2);
        // With the tracer's 3, a DDGI update kernel has 3 of its 8 left for its own buffers (it uses 2).
        const modules = orderModules([shade.module]);
        expect(modules.map((module) => module.name)).toEqual(['gi-scene', 'gi-rays', 'gi-shade']);
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

    it('places its three entries after the modules it depends on', () => {
        const shade = new PlatformHitShading(fakeDevice(), scene, rays);
        const placed = placeModules(orderModules([shade.module]));
        expect(placed[2].slot.firstBinding).toBe(SCENE_MODULE_ENTRIES.length);
        expect(shade.module.entries).toBe(SHADE_MODULE_ENTRIES);
        expect(shade.module.resources()).toHaveLength(3);
    });

    it('starts from the contract defaults and lays its uniforms out after the 18 SH coefficients', () => {
        const shade = new PlatformHitShading(fakeDevice(), scene, rays);
        expect(shade.settings).toEqual(DEFAULT_HIT_SHADING_SETTINGS);
        expect(SHADE_UNIFORM_LAYOUT.maxShadowedLightsPerHit).toBe(288);
        expect(SHADE_UNIFORM_LAYOUT.byteLength % 16).toBe(0);
    });

    it('builds the subgroup reduction only on devices that have subgroups', () => {
        expect(new PlatformHitShading(fakeDevice(['subgroups']), scene, rays).sky.reduction).toBe('subgroups');
        expect(new PlatformHitShading(fakeDevice(), scene, rays).sky.reduction).toBe('workgroup');
        expect(() => new PlatformHitShading(fakeDevice(), scene, rays, { skyReduction: 'subgroups' })).toThrow(/subgroups/);
        expect(projectionWgsl('subgroups')).toMatch(/^enable subgroups;/);
        expect(projectionWgsl('subgroups')).toContain('subgroupAdd(');
        expect(projectionWgsl('workgroup')).not.toContain('subgroup_invocation_id');
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
