import { describe, expect, it } from 'vitest';
import type { ShadingParameter } from '../../platform/contract/shading-module.ts';
import { DDGITechnique } from './ddgi-technique.ts';
import { DDGI_SHADING_PARAMETER_NAMES, DDGIShadingModule, type DDGIFieldTextures, type DDGIShadingSource } from './shading.ts';
import { DDGI_DEBUG_VIEW } from './wgsl/shading.ts';

/** A stand-in GPUTexture: the module only hands it on. */
function fakeTexture(label: string): GPUTexture {
    return { label } as GPUTexture;
}

function fakeSource(): DDGIShadingSource & { textures: DDGIFieldTextures } {
    const textures = { irradiance: fakeTexture('irradiance'), distance: fakeTexture('distance'), probeData: fakeTexture('probe data'), volumeTable: fakeTexture('volume table') };
    return { textures, fieldTextures: () => textures };
}

/** The contract's rule for now (docs/api/platform.md, walk-through gap 1). */
const MAX_TEXTURES = 6;

describe('the DDGI shading module', () => {
    const source = fakeSource();
    const shading = new DDGIShadingModule(source);
    const textures = shading.parameters.filter((parameter): parameter is Extract<ShadingParameter, { kind: 'texture' }> => parameter.kind === 'texture');

    it('takes textures and samplers only, at most 6 textures: the three atlases and the volume table', () => {
        expect(textures.length).toBeLessThanOrEqual(MAX_TEXTURES);
        expect(shading.parameters.map((parameter) => parameter.kind)).toEqual(['texture', 'sampler', 'texture', 'sampler', 'texture', 'texture']);
        expect(textures.map((texture) => texture.dimension)).toEqual(['2d-array', '2d-array', '2d-array', '2d']);
        // rgba32float, read with textureLoad: it must not ask for a filtering binding.
        expect(textures[3].name).toBe(DDGI_SHADING_PARAMETER_NAMES.volumeTable);
        expect(textures[3].sampleType).toBe('unfilterable-float');
    });

    it('names each sampler after its filterable texture', () => {
        for (const parameter of shading.parameters) {
            if (parameter.kind !== 'sampler') continue;
            expect(parameter.name).toBe(`${parameter.texture}Sampler`);
            expect(textures.find((texture) => texture.name === parameter.texture)?.sampleType).toBe('float');
        }
    });

    it('declares no bindings and no storage in its code, and says it has no giRadiance', () => {
        expect(shading.code).not.toMatch(/@group|@binding|var<storage/);
        expect(shading.code).toMatch(/fn giIrradiance\(/);
        expect(shading.code).toMatch(/fn giCoverage\(/);
        expect(shading.code).toMatch(/fn giDebugColor\(/);
        // The points view's irradianceFunction (DV-03): one probe's own light.
        expect(shading.code).toMatch(/fn giProbeIrradiance\(\s*point: u32,\s*direction: vec3f,/);
        expect(shading.code).not.toMatch(/fn giRadiance\(/);
        expect(shading.provides.radiance).toBe(false);
        // A probe field: another technique can read it at its own hits (the radiance cache's second bounce, P6.3).
        expect(shading.provides.worldSpaceIrradiance).toBe(true);
    });

    it('uses the parameter names as the WGSL parameter names, in order', () => {
        const signature = /fn giIrradiance\(([^)]*)\)/s.exec(shading.code)?.[1] ?? '';
        const names = signature
            .split(',')
            .map((part) => part.trim().split(':')[0])
            .filter((name) => name.length > 0);
        expect(names).toEqual(['position', 'normal', 'viewDirection', 'screenUV', ...shading.parameters.map((parameter) => parameter.name)]);
    });

    it('hands out the current textures, and has no uniforms to write', () => {
        expect(textures.map((texture) => texture.texture().label)).toEqual(['irradiance', 'distance', 'probe data', 'volume table']);
        expect(shading.parameters.filter((parameter) => parameter.kind === 'vec4f')).toEqual([]);
        // Nothing to write, so even an empty target is fine.
        const target = new Float32Array(4).fill(7);
        shading.writeUniforms(target);
        expect([...target]).toEqual([7, 7, 7, 7]);
        shading.writeUniforms(new Float32Array(0));
    });

    it('writes globals with bindings from 0 in parameter order, no uniform buffer, and parameterless wrappers', () => {
        const globals = shading.globalsCode(2);
        expect(globals.startsWith(shading.code)).toBe(true);
        const n = DDGI_SHADING_PARAMETER_NAMES;
        expect(globals).toContain(`@group(2) @binding(0) var ${n.irradiance}: texture_2d_array<f32>;`);
        expect(globals).toContain(`@group(2) @binding(1) var ${n.irradianceSampler}: sampler;`);
        expect(globals).toContain(`@group(2) @binding(2) var ${n.distance}: texture_2d_array<f32>;`);
        expect(globals).toContain(`@group(2) @binding(3) var ${n.distanceSampler}: sampler;`);
        expect(globals).toContain(`@group(2) @binding(4) var ${n.probeData}: texture_2d_array<f32>;`);
        expect(globals).toContain(`@group(2) @binding(5) var ${n.volumeTable}: texture_2d<f32>;`);
        // WGSL has no empty structs: with no vec4f parameters there is no uniform buffer.
        expect(globals).not.toMatch(/var<uniform>/);
        expect(globals).toMatch(/fn giIrradianceAt\(position: vec3f, normal: vec3f, viewDirection: vec3f, screenUV: vec2f\) -> vec3f/);
        expect(globals).toMatch(/fn giCoverageAt\(position: vec3f, normal: vec3f\) -> f32/);
        expect(globals).toMatch(/fn giDebugColorAt\(position: vec3f, normal: vec3f, view: u32, screenUV: vec2f\) -> vec3f/);
        expect(globals).toMatch(/fn giProbeIrradianceAt\(point: u32, direction: vec3f\) -> vec3f/);
        expect(globals).not.toContain('var<storage');
    });

    it('is version 2: the volume table replaced the volume uniforms (M3, several volumes)', () => {
        expect(shading.version).toBe(2);
    });
});

describe('the DDGI technique and its shading module', () => {
    const technique = new DDGITechnique({ volume: { center: [0, 0, 0], size: [2, 2, 2], spacing: 1 } });

    it('publishes the module, and screen debug views with ids distinct from the points view', () => {
        expect(technique.shading).toBeInstanceOf(DDGIShadingModule);
        expect(technique.shading.techniqueId).toBe('ddgi');
        const ids = technique.debugViews.map((view) => view.id);
        expect(new Set(ids).size).toBe(ids.length);
        const screens = technique.debugViews.filter((view) => view.kind === 'screen');
        expect(screens.map((view) => view.id).sort()).toEqual([DDGI_DEBUG_VIEW.giOnly, DDGI_DEBUG_VIEW.coverage, DDGI_DEBUG_VIEW.probeStates].sort());
        for (const view of screens) expect(view.description.length).toBeGreaterThan(20);
    });

    it('says plainly that textures exist only after prepare', () => {
        const texture = technique.shading.parameters.find((parameter) => parameter.kind === 'texture');
        expect(texture && texture.kind === 'texture' ? () => texture.texture() : null).toThrow(/after prepare/);
    });
});
