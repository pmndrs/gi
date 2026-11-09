/**
 * Sampling the texture store, as one platform WGSL module (`gi-texture-store`)
 * that every kernel reading material textures shares:
 *
 * - hit shading (`giSurfaceAtCone`, shade/wgsl.ts), at the mip a ray cone's
 *   footprint covers;
 * - the walk's alpha rules (`giMaterialPasses`, trace/material-test-wgsl.ts),
 *   at a fixed fine mip.
 *
 * So the uv mapping (uv set, 3×2 transform, wrap per axis) has one
 * definition, and the store's array and sampler are bound once in a kernel
 * that lists both (the kernel builder includes a module once).
 *
 * ```wgsl
 * fn giTextureSampleMapped(layer: u32, mapping: GiUvMapping, uvs: GiTriangleUvs, weights: vec3f, lod: f32) -> vec4f
 * fn giTextureMappedUv(mapping: GiUvMapping, uvs: GiTriangleUvs, weights: vec3f) -> vec2f
 * fn giTextureWrap(coordinate: f32, wrap: u32, halfTexel: f32) -> f32
 * ```
 *
 * It depends on the scene module (`GiUvMapping`, `GiTriangleUvs`, the wrap
 * constants) and binds the store's `texture_2d_array` and its sampler: no
 * storage buffer.
 */

import type { BindingSlot, ModuleBindingEntry, WgslModule } from '../contract/wgsl.ts';
import { STORE_LAYER_SIZE, STORE_MIP_LEVELS } from './texture-store-wgsl.ts';

/** `GPUShaderStage.COMPUTE`. The global does not exist in Node, where the tests import this file. */
const SHADER_STAGE_COMPUTE = 0x4;

/** The module's two bindings: the store's array (a `2d-array` view), then its sampler. */
export const TEXTURE_SAMPLING_ENTRIES: readonly ModuleBindingEntry[] = [
    { visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
    { visibility: SHADER_STAGE_COMPUTE, sampler: { type: 'filtering' } },
];

/** What the module binds: the store's array, its sampler, and when the array was last replaced. */
export interface TextureStoreBinding {
    readonly view: GPUTextureView;
    /** Increments when the array is replaced (it grew): bind groups holding `view` are stale. */
    readonly version: number;
    /** Trilinear and repeating (clamp and mirror are worked out on the uv, {@link textureSamplingWgsl}). */
    readonly sampler: GPUSampler;
}

/** The sampler every kernel samples the store with: trilinear, repeating. */
export const TEXTURE_STORE_SAMPLER: GPUSamplerDescriptor = {
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
};

/** The module's source, with its bindings at `slot`: the array, then the sampler. */
export function textureSamplingWgsl(slot: BindingSlot): string {
    return /* wgsl */ `
// ---- Texture store sampling (platform/textures/texture-sampling-wgsl.ts) --------------

// One ${STORE_LAYER_SIZE}² layer with mips per texture (a material's texture words name
// its layer). rgba8unorm-srgb, so samples are linear light; alpha is linear.
@group(${slot.group}) @binding(${slot.firstBinding}) var giTextureStore: texture_2d_array<f32>;
// Linear in and between mips, repeating: clamp and mirror are worked out on the uv (giTextureWrap).
@group(${slot.group}) @binding(${slot.firstBinding + 1}) var giTextureSampler: sampler;

// The store's layers: texels on a side, and the last mip.
const GI_TEXTURE_SIZE: f32 = ${STORE_LAYER_SIZE}.0;
const GI_TEXTURE_LAST_LOD: f32 = ${STORE_MIP_LEVELS - 1}.0;

// One axis of a mapped uv, wrapped as the mapping says. 'repeat' is left to
// the sampler (it repeats). 'clamp' keeps the coordinate half a texel inside
// the edges, and 'mirror' folds it (1 - |(u mod 2) - 1|: forwards on even
// tiles, backwards on odd ones) and keeps it inside the same way, so the
// repeating sampler's filter never reaches across to the opposite edge.
// 'halfTexel' is that of the coarser of the two mips the sample blends.
fn giTextureWrap(coordinate: f32, wrap: u32, halfTexel: f32) -> f32 {
    if (wrap == GI_UV_WRAP_CLAMP) {
        return clamp(coordinate, halfTexel, 1.0 - halfTexel);
    }
    if (wrap == GI_UV_WRAP_MIRROR) {
        // fract is floored, so negative coordinates fold the same way.
        let folded = 1.0 - abs(2.0 * fract(coordinate * 0.5) - 1.0);
        return clamp(folded, halfTexel, 1.0 - halfTexel);
    }
    return coordinate;
}

// The texture coordinate of a point of a triangle, before wrapping: the uv
// set's corners 'uvs' (giInstanceTriangleUvs with mapping.channel) blended by
// barycentric 'weights' (of corners 0, 1, 2), then the mapping's 3x2 transform.
fn giTextureMappedUv(mapping: GiUvMapping, uvs: GiTriangleUvs, weights: vec3f) -> vec2f {
    let uv = uvs.uv0 * weights.x + uvs.uv1 * weights.y + uvs.uv2 * weights.z;
    return mapping.transform * vec3f(uv, 1.0);
}

// A material texture (store layer 'layer') at a point of a triangle, through
// its mapping, at mip 'lod' (trilinear between the two mips around it).
fn giTextureSampleMapped(layer: u32, mapping: GiUvMapping, uvs: GiTriangleUvs, weights: vec3f, lod: f32) -> vec4f {
    let mapped = giTextureMappedUv(mapping, uvs, weights);
    let halfTexel = 0.5 * exp2(ceil(lod)) / GI_TEXTURE_SIZE;
    let wrapped = vec2f(giTextureWrap(mapped.x, mapping.wrapU, halfTexel), giTextureWrap(mapped.y, mapping.wrapV, halfTexel));
    return textureSampleLevel(giTextureStore, giTextureSampler, wrapped, layer, lod);
}
`;
}

/**
 * The module over one store. The store makes it once
 * (`PlatformTextureStore.samplingModule`), so hit shading and the walk list
 * the same object and a kernel with both binds the store once.
 */
export function createTextureSamplingModule(sceneModule: WgslModule, binding: TextureStoreBinding): WgslModule {
    return {
        name: 'gi-texture-store',
        dependencies: [sceneModule],
        code: (slot: BindingSlot) => textureSamplingWgsl(slot),
        entries: TEXTURE_SAMPLING_ENTRIES,
        resources: () => [binding.view, binding.sampler],
        get version() {
            return binding.version;
        },
        storageBufferCount: 0,
    };
}
