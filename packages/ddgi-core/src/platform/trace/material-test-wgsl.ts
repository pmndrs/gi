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
 * 3. *(T4.2's second half, with T4.1's texture store)* it is a cutout and the
 *    albedo texture's alpha at the hit is below the cutoff;
 * 4. *(same)* it is stochastic and a hash of the ray and triangle is at least
 *    the alpha.
 *
 * Otherwise the hit stands, for closest hit and any hit alike.
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
 *   `giMaterialPasses` over the platform's scene copy, as a binding-free
 *   module that depends on the scene module;
 * - {@link NO_MATERIALS_WGSL}: `giMaterialPasses` for a tracer that has no
 *   scene copy (the low-level batch kernel of `trace-gpu.ts`, which the
 *   oracle and the trace bench use). Its trees come straight from the
 *   builders, which never set the flag, so the call is never made;
 * - the same rules on the CPU ({@link materialNeedsLookup},
 *   {@link materialLetsRayPass}) for the reference tracers.
 */

import type { BindingSlot, WgslModule } from '../contract/wgsl.ts';
import { TRIANGLE_FLAG_ALPHA_TEST } from './layout.ts';

/**
 * Triangle `flags` bit 0: "consult the material" (T4.2). layout.ts reserved
 * this bit in T0.10 as "needs an alpha test"; M4 widened it to every rule
 * above, so the name here says what it means now.
 */
export const TRIANGLE_FLAG_CONSULT_MATERIAL = TRIANGLE_FLAG_ALPHA_TEST;

/** What the rules read from a material: `GiMaterial`'s fields of the same names. */
export interface WalkMaterial {
    readonly blocksLight: boolean;
    readonly doubleSided: boolean;
}

/**
 * The material a triangle of an instance uses, for the CPU tracers:
 * `instanceIndex` as `GiHit.instance` reports it, `triangleIndex` the
 * triangle's index in the mesh as the host gave it.
 */
export type WalkMaterialLookup = (instanceIndex: number, triangleIndex: number) => WalkMaterial;

/** Whether any rule can let a ray through this material, so its triangles must be flagged. CPU twin of `giMaterialNeedsLookup`. */
export function materialNeedsLookup(material: WalkMaterial): boolean {
    // Alpha rules (T4.2's second half): also true for a cutout or a stochastic material.
    return !material.blocksLight || !material.doubleSided;
}

/** Whether a ray that hit this material on the given side goes on through it. CPU twin of `giMaterialPasses`. */
export function materialLetsRayPass(material: WalkMaterial, frontFace: boolean): boolean {
    if (!material.blocksLight) return true;
    if (!material.doubleSided && !frontFace) return true;
    // Alpha rules (T4.2's second half): cutout and stochastic go here.
    return false;
}

/** The rules over the scene copy: needs the scene module's `giMaterialAt` and `giMaterial`. */
const MATERIAL_TEST_WGSL = /* wgsl */ `
// ---- Materials in the walk (trace/material-test-wgsl.ts) ------------------------------

// Whether any rule in giMaterialPasses can let a ray through this material.
// gi/triangle-flags sets the consult flag on triangles whose slot this is true for.
fn giMaterialNeedsLookup(material: GiMaterial) -> bool {
    // Alpha rules (T4.2's second half): also true for a cutout or a stochastic material.
    return !material.blocksLight || !material.doubleSided;
}

// Whether a ray that hit triangle 'triangle' (its index in the mesh as the host
// gave it) of instance 'instance' (the scene-copy slot) goes on through it.
// 'frontFace' is GiHit.frontFace: the side the mesh was authored as front, also
// when mirrored. Called by the walk only for triangles with the consult flag.
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

    // ---- Alpha rules: T4.2's second half, once T4.1's texture store exists ----
    // 3. Cutout: the uv at 'barycentrics' (giInstanceUvAt with the albedo
    //    texture's mapping), the albedo layer's alpha at a fixed fine mip
    //    (alphaTestLod), and pass when alpha < material.alphaCutoff.
    // 4. Stochastic: pass when a hash of the triangle and 'barycentrics' (which
    //    differ for every ray) is >= alpha, so on average 1 - alpha gets through.
    // giMaterialNeedsLookup (and its CPU twin) must name both modes too.

    return false;
}
`;

/**
 * The material test as a platform module: no bindings of its own, the scene
 * module before it. The `rays` module and `gi/triangle-flags` list it.
 */
export function createMaterialTestModule(sceneModule: WgslModule): WgslModule {
    return {
        name: 'gi-material-test',
        dependencies: [sceneModule],
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
