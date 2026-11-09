/**
 * Materials in the walk (T4.2; docs/design/m4-materials.md §3): the rules
 * that let a ray pass a triangle it hit, kept in one file so the walk, the
 * flags kernel and the CPU reference tracers can't drift apart.
 *
 * At a candidate hit, the ray **passes** (the hit is ignored, and the walk
 * goes on looking behind it) when:
 *
 * 1. the material doesn't block light (`blocksLight: false`: glass, hidden);
 * 2. the material is single-sided (`doubleSided: false`) and the hit is on
 *    its back face (the side opposite the one the mesh was authored as front,
 *    also through a mirrored transform: `GiHit.frontFace`'s rule);
 * 3. it is a cutout (`alphaMode: 'cutout'`) and its alpha at the hit is below
 *    the cutoff (`alphaCutoff`, 0.5 when none is set);
 * 4. it is stochastic (`alphaMode: 'stochastic'`, blended surfaces; DECISIONS
 *    2026-11-09, "M4 answers") and a hash of the hit is at least its alpha,
 *    so a share 1 − alpha of the rays gets through.
 *
 * Otherwise the hit stands, for closest hit and any hit alike.
 *
 * **Alpha** is the material's `opacity` times the albedo texture's alpha at
 * the hit: the albedo texture's uv mapping (uv set, 3×2 transform, wrap),
 * sampled from the texture store at a fixed fine mip, {@link ALPHA_TEST_LOD}
 * (the shared sampling module, textures/texture-sampling-wgsl.ts). A fixed
 * mip, not the ray cone's, because a cutout is geometry: a far ray must see
 * the same holes as a near one. Mip 1, not 0, is half the texture's detail,
 * which is plenty for a hole's shape and reads the same texels as hit
 * shading's nearer hits. Without an albedo texture (or before its layer is
 * made) the alpha is the opacity alone. The rules read the texture whatever
 * hit shading's `sampleTextures` setting says: holes are geometry, not shading.
 *
 * **The stochastic hash** mixes the instance, the triangle and the bits of
 * the hit's barycentrics (PCG, Jarzynski and Olano 2020). The barycentrics
 * carry the ray: two rays meeting the same triangle at the same point to the
 * last bit are rare, and DDGI turns its probe rays every frame, so the
 * pattern differs from frame to frame without a frame number (which would
 * need a uniform in the walk). A fixed ray traced every frame sees the same
 * answer each time; many rays see 1 − alpha of them pass.
 *
 * **Only flagged triangles pay.** The tracer's triangle flag bit 0
 * ({@link TRIANGLE_FLAG_CONSULT_MATERIAL}) says "consult the material". The
 * flags kernel (`gi/triangle-flags`, triangle-flags.ts) sets it on every
 * triangle whose material slot, in any instance of its mesh, could let a ray
 * pass ({@link materialNeedsLookup}). The walk (wgsl.ts) calls
 * `giMaterialPasses` only for those, so an opaque scene costs one bit test of
 * a word the walk has already loaded. Because the flag is a union over the
 * mesh's instances, an opaque instance of a mesh that is see-through
 * elsewhere pays a lookup and gets "blocks": an extra lookup at worst, never
 * a wrong answer.
 *
 * What this file gives:
 * - {@link createMaterialTestModule}: `giMaterialNeedsLookup` and
 *   `giMaterialPasses` over the platform's scene copy and texture store, as a
 *   binding-free module that depends on the scene and sampling modules;
 * - {@link NO_MATERIALS_WGSL}: `giMaterialPasses` for a tracer that has no
 *   scene copy (the low-level batch kernel of `trace-gpu.ts`, which the
 *   oracle and the trace bench use). Its trees come straight from the
 *   builders, which never set the flag, so the call is never made;
 * - the same rules on the CPU ({@link materialNeedsLookup},
 *   {@link materialLetsRayPass}, {@link materialAlphaAt},
 *   {@link stochasticHashValue}) for the reference tracers.
 */

import type { AlphaMode, MaterialDescriptor, UvMapping } from '../contract/scene.ts';
import type { BindingSlot, WgslModule } from '../contract/wgsl.ts';
import { DEFAULT_ALPHA_CUTOFF } from '../scene/layout.ts';
import { sampleMappedOnCpu, type CpuStoreLayer } from '../textures/texture-store-cpu.ts';
import { TRIANGLE_FLAG_ALPHA_TEST } from './layout.ts';

/**
 * Triangle `flags` bit 0: "consult the material" (T4.2). layout.ts reserved
 * this bit in T0.10 as "needs an alpha test"; M4 widened it to every rule
 * above, so the name here says what it means now.
 */
export const TRIANGLE_FLAG_CONSULT_MATERIAL = TRIANGLE_FLAG_ALPHA_TEST;

/** The store mip the alpha rules read (the file comment says why 1). */
export const ALPHA_TEST_LOD = 1;

/** Mixed into the stochastic hash first, so it doesn't start from the same state as other hashes of the same numbers. */
const STOCHASTIC_SALT = 0x5f3759df;

/** An albedo texture as the CPU rules read it: its store layer made on the CPU, and its uv mapping. */
export interface WalkAlphaTexture {
    readonly layer: CpuStoreLayer;
    readonly uv?: UvMapping;
}

/**
 * What the rules read from a material: `GiMaterial`'s fields of the same
 * names, resolved as the scene copy writes its record (see
 * {@link walkMaterialOf}).
 */
export interface WalkMaterial {
    readonly blocksLight: boolean;
    readonly doubleSided: boolean;
    /** Defaults to `'opaque'`. */
    readonly alphaMode?: AlphaMode;
    /** Used when `alphaMode` is `'cutout'`; defaults to 0.5 there (`DEFAULT_ALPHA_CUTOFF`). */
    readonly alphaCutoff?: number;
    /** Defaults to 1. */
    readonly opacity?: number;
    /** The albedo texture, if any: its alpha times `opacity` is the surface's alpha. */
    readonly albedoTexture?: WalkAlphaTexture;
}

/** The material a host's descriptor makes, as the rules read it (the scene copy's resolution of `alphaMode` and `alphaCutoff`). */
export function walkMaterialOf(descriptor: MaterialDescriptor, albedoTexture?: WalkAlphaTexture): WalkMaterial {
    const alphaMode: AlphaMode = descriptor.alphaMode ?? ((descriptor.alphaCutoff ?? 0) > 0 ? 'cutout' : 'opaque');
    return {
        blocksLight: descriptor.blocksLight !== false,
        doubleSided: descriptor.doubleSided !== false,
        alphaMode,
        alphaCutoff: alphaMode === 'cutout' ? (descriptor.alphaCutoff ?? DEFAULT_ALPHA_CUTOFF) : 0,
        opacity: descriptor.opacity ?? 1,
        albedoTexture,
    };
}

/** The cutoff the record holds: the material's (0.5 by default) for a cutout, 0 otherwise. */
function cutoffOf(material: WalkMaterial): number {
    return material.alphaMode === 'cutout' ? (material.alphaCutoff ?? DEFAULT_ALPHA_CUTOFF) : 0;
}

/**
 * The material a triangle of an instance uses, for the CPU tracers:
 * `instanceIndex` as `GiHit.instance` reports it, `triangleIndex` the
 * triangle's index in the mesh as the host gave it.
 */
export type WalkMaterialLookup = (instanceIndex: number, triangleIndex: number) => WalkMaterial;

/**
 * A triangle's corner uvs in one uv set, `[u0, v0, u1, v1, u2, v2]`, for the
 * CPU tracers' alpha rules; null when the mesh has no such set (the uv is
 * then (0, 0), as `giInstanceTriangleUvs` gives).
 */
export type WalkUvLookup = (instanceIndex: number, triangleIndex: number, channel: 0 | 1) => ArrayLike<number> | null;

/** Whether the alpha rules can open a hole in this material (CPU twin of `giMaterialMayOpenHoles`). */
function materialMayOpenHoles(material: WalkMaterial): boolean {
    const opacity = material.opacity ?? 1;
    const textured = material.albedoTexture !== undefined;
    if (material.alphaMode === 'cutout') {
        const cutoff = cutoffOf(material);
        return cutoff > 0 && (textured || opacity < cutoff);
    }
    if (material.alphaMode === 'stochastic') return textured || opacity < 1;
    return false;
}

/** Whether any rule can let a ray through this material, so its triangles must be flagged. CPU twin of `giMaterialNeedsLookup`. */
export function materialNeedsLookup(material: WalkMaterial): boolean {
    return !material.blocksLight || !material.doubleSided || materialMayOpenHoles(material);
}

const NO_UVS = [0, 0, 0, 0, 0, 0];

/**
 * The material's alpha at a point of a triangle: `opacity` times the albedo
 * texture's alpha at {@link ALPHA_TEST_LOD} (CPU twin of `giMaterialAlpha`).
 * `u`, `v` are the weights of corners 1 and 2; `uvsOf` gives the corner uvs
 * of a uv set.
 */
export function materialAlphaAt(material: WalkMaterial, u: number, v: number, uvsOf: (channel: 0 | 1) => ArrayLike<number> | null): number {
    const opacity = material.opacity ?? 1;
    const texture = material.albedoTexture;
    if (!texture) return opacity;
    const clamp = (weight: number): number => Math.min(Math.max(weight, 0), 1);
    const weights: [number, number, number] = [clamp(1 - u - v), clamp(u), clamp(v)];
    const uvs = uvsOf(texture.uv?.channel ?? 0) ?? NO_UVS;
    return opacity * sampleMappedOnCpu(texture.layer, texture.uv, uvs, weights, ALPHA_TEST_LOD)[3];
}

/** PCG's output hash (Jarzynski and Olano, "Hash Functions for GPU Rendering", JCGT 2020), on 32-bit words. */
function pcgHash(value: number): number {
    const state = (Math.imul(value >>> 0, 747796405) + 2891336453) >>> 0;
    const word = Math.imul(((state >>> (((state >>> 28) + 4) >>> 0)) ^ state) >>> 0, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
}

const floatBits = new Float32Array(1);
const floatWords = new Uint32Array(floatBits.buffer);
function bitsOfFloat32(value: number): number {
    floatBits[0] = value;
    return floatWords[0];
}

/**
 * The stochastic rule's hash of a hit, in [0, 1) (CPU twin of
 * `giMaterialStochasticValue`): the instance, the triangle, and the f32 bits
 * of the barycentrics `u`, `v` (weights of corners 1 and 2).
 */
export function stochasticHashValue(instanceIndex: number, triangleIndex: number, u: number, v: number): number {
    let hash = pcgHash((instanceIndex ^ STOCHASTIC_SALT) >>> 0);
    hash = pcgHash((hash ^ triangleIndex) >>> 0);
    hash = pcgHash((hash ^ bitsOfFloat32(u)) >>> 0);
    hash = pcgHash((hash ^ bitsOfFloat32(v)) >>> 0);
    return (hash >>> 8) / 16777216;
}

/**
 * Whether a ray that hit this material goes on through it. CPU twin of
 * `giMaterialPasses`. `alphaAt` gives the material's alpha at the hit
 * ({@link materialAlphaAt}) and `stochasticValue` the hash
 * ({@link stochasticHashValue}); each is only called when a rule needs it.
 */
export function materialLetsRayPass(material: WalkMaterial, frontFace: boolean, alphaAt: () => number = () => 1, stochasticValue: () => number = () => 0): boolean {
    if (!material.blocksLight) return true;
    if (!material.doubleSided && !frontFace) return true;
    if (material.alphaMode === 'cutout') return alphaAt() < cutoffOf(material);
    if (material.alphaMode === 'stochastic') return stochasticValue() >= alphaAt();
    return false;
}

/** The rules over the scene copy: needs the scene module's `giMaterialAt`, `giMaterial` and `giInstanceTriangleUvs`, and the sampling module's `giTextureSampleMapped`. */
const MATERIAL_TEST_WGSL = /* wgsl */ `
// ---- Materials in the walk (trace/material-test-wgsl.ts) ------------------------------

// The store mip the alpha rules read: fixed, as a hole is geometry (ALPHA_TEST_LOD).
const GI_MATERIAL_ALPHA_TEST_LOD: f32 = ${ALPHA_TEST_LOD}.0;
const GI_MATERIAL_STOCHASTIC_SALT: u32 = ${STOCHASTIC_SALT}u;

// Whether the alpha rules can open a hole in this material: a cutout whose
// alpha can fall below its cutoff, or a stochastic material whose alpha can
// be below 1. A texture word is GI_NO_TEXTURE until its layer is made; the
// platform logs a material change when it lands, which rewrites the flags.
fn giMaterialMayOpenHoles(material: GiMaterial) -> bool {
    let textured = material.albedoTexture != GI_NO_TEXTURE;
    if (material.alphaMode == GI_ALPHA_CUTOUT) {
        return material.alphaCutoff > 0.0 && (textured || material.opacity < material.alphaCutoff);
    }
    if (material.alphaMode == GI_ALPHA_STOCHASTIC) {
        return textured || material.opacity < 1.0;
    }
    return false;
}

// Whether any rule in giMaterialPasses can let a ray through this material.
// gi/triangle-flags sets the consult flag on triangles whose slot this is true for.
fn giMaterialNeedsLookup(material: GiMaterial) -> bool {
    return !material.blocksLight || !material.doubleSided || giMaterialMayOpenHoles(material);
}

// The material's alpha at a point of a triangle: opacity x the albedo
// texture's alpha there, through the albedo texture's uv mapping, at the
// fixed alpha-test mip. The opacity alone without a texture.
fn giMaterialAlpha(material: GiMaterial, instance: u32, triangle: u32, barycentrics: vec2f) -> f32 {
    if (material.albedoTexture == GI_NO_TEXTURE) {
        return material.opacity;
    }
    // Clamped as hit shading clamps them: the walk's barycentrics can be a hair outside [0, 1].
    let weights = clamp(vec3f(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y), vec3f(0.0), vec3f(1.0));
    let uvs = giInstanceTriangleUvs(instance, triangle, material.albedoUv.channel);
    let texel = giTextureSampleMapped(material.albedoTexture, material.albedoUv, uvs, weights, GI_MATERIAL_ALPHA_TEST_LOD);
    return material.opacity * texel.a;
}

// PCG's output hash (Jarzynski and Olano, "Hash Functions for GPU Rendering", JCGT 2020).
fn giMaterialPcgHash(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// The stochastic rule's random number for a hit, in [0, 1): a hash of the
// instance, the triangle and the barycentrics' bits, which carry the ray
// (the file comment says why no frame number is needed).
fn giMaterialStochasticValue(instance: u32, triangle: u32, barycentrics: vec2f) -> f32 {
    var hash = giMaterialPcgHash(instance ^ GI_MATERIAL_STOCHASTIC_SALT);
    hash = giMaterialPcgHash(hash ^ triangle);
    hash = giMaterialPcgHash(hash ^ bitcast<u32>(barycentrics.x));
    hash = giMaterialPcgHash(hash ^ bitcast<u32>(barycentrics.y));
    // The top 24 bits, exactly representable, over 2^24.
    return f32(hash >> 8u) * (1.0 / 16777216.0);
}

// Whether a ray that hit triangle 'triangle' (its index in the mesh as the host
// gave it) of instance 'instance' (the scene-copy slot) goes on through it.
// 'barycentrics' are the hit's weights of corners 1 and 2, 'frontFace' is
// GiHit.frontFace: the side the mesh was authored as front, also when
// mirrored. Called by the walk only for triangles with the consult flag.
fn giMaterialPasses(instance: u32, triangle: u32, barycentrics: vec2f, frontFace: bool) -> bool {
    // The lookup: instance -> its material list -> the triangle's slot -> the record.
    let material = giMaterial(giMaterialAt(instance, triangle));

    // 1. See-through (glass, hidden): every ray passes.
    if (!material.blocksLight) {
        return true;
    }
    // 2. Single-sided: a ray that meets the back face passes.
    if (!material.doubleSided && !frontFace) {
        return true;
    }
    // 3. Cutout: a hole where the alpha is below the cutoff. The alpha is at most
    //    the opacity, so an opacity below the cutoff needs no texture read.
    if (material.alphaMode == GI_ALPHA_CUTOUT) {
        if (material.opacity < material.alphaCutoff) {
            return true;
        }
        return giMaterialAlpha(material, instance, triangle, barycentrics) < material.alphaCutoff;
    }
    // 4. Stochastic: a share 1 - alpha of the rays passes. A random number at or
    //    above the opacity passes whatever the texture says, without reading it.
    if (material.alphaMode == GI_ALPHA_STOCHASTIC) {
        let random = giMaterialStochasticValue(instance, triangle, barycentrics);
        if (random >= material.opacity) {
            return true;
        }
        return random >= giMaterialAlpha(material, instance, triangle, barycentrics);
    }
    return false;
}
`;

/**
 * The material test as a platform module: no bindings of its own, the scene
 * module and the texture store's sampling module before it. The `rays`
 * module and `gi/triangle-flags` list it.
 */
export function createMaterialTestModule(sceneModule: WgslModule, textureModule: WgslModule): WgslModule {
    return {
        name: 'gi-material-test',
        dependencies: [sceneModule, textureModule],
        code: (_slot: BindingSlot) => MATERIAL_TEST_WGSL,
        entries: [],
        resources: () => [],
        version: 0,
        storageBufferCount: 0,
    };
}

/**
 * `giMaterialPasses` for a tracer with no scene copy, so no materials (the
 * low-level batch kernel in trace-gpu.ts, used by the oracle and the trace
 * bench). Its trees come straight from the builders, which leave the consult
 * flag clear, so the walk never calls this; every triangle blocks.
 */
export const NO_MATERIALS_WGSL = /* wgsl */ `
// ---- No materials: this tracer has no scene copy (trace/material-test-wgsl.ts) ----
fn giMaterialPasses(instance: u32, triangle: u32, barycentrics: vec2f, frontFace: bool) -> bool {
    return false;
}
`;
