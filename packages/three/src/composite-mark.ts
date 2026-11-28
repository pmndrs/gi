/**
 * The mark that tells GI's light node a scene pass is composited (T1.14,
 * `docs/design/three-composite.md`), and the one rule both sides use to
 * decide which surfaces the composite lights.
 *
 * The composite marks its scene pass through the pass's `contextNode`, the
 * way three's GTAO marks a pass with `builtinAOContext`. PassNode merges that
 * context into `renderer.contextNode` while the pass renders, so every
 * material built for that pass sees the mark in `builder.context`, and only
 * those: shadow maps, cube captures and other passes don't. three caches
 * shaders by the renderer's context node (`RenderObject.getCacheKey`), so a
 * material gets one shader for the marked pass and another elsewhere.
 *
 * In a marked pass a surface is either composited or not, and the GI light
 * and `giAlbedo` must agree on which, or GI is counted twice or lost:
 *
 * - **composited** (opaque, and a lighting model whose indirect-diffuse
 *   weight `giAlbedo` reproduces): the light adds no indirect diffuse;
 *   `giAlbedo` writes the weight, and the composite adds `giAlbedo × E / π`;
 * - **not composited** (transparent, unlit, or a model `giAlbedo` can't
 *   reproduce): the light shades GI per fragment as in an unmarked pass, and
 *   `giAlbedo` writes 0.
 *
 * {@link compositeTakesGI} is that rule. Both sides call it with the same
 * builder, so they can't disagree.
 */

import type { Material, Node, NodeBuilder } from 'three/webgpu';
import { context } from 'three/tsl';

type ContextNode = ReturnType<typeof context>;

/**
 * The context key of the mark. A string key, because PassNode merges
 * contexts by spreading their keys (`mergeContext`), and a symbol would be
 * dropped by nothing but is harder to read in a debugger.
 */
export const GI_COMPOSITE_MARK = 'threeGiComposite';

/** The part of a scene pass the mark touches (three's `PassNode`). */
export interface MarkablePass {
    contextNode: ContextNode | null;
    /** Node's setter: bumps the pass's version so its cached merged context is rebuilt. */
    needsUpdate: boolean;
}

/** Whether a context (a pass's `contextNode`, or a builder's context) carries the mark. */
export function hasCompositeMark(contextData: object | null | undefined): boolean {
    return (contextData as Record<string, unknown> | null | undefined)?.[GI_COMPOSITE_MARK] === true;
}

/**
 * Marks a scene pass as composited, keeping whatever context it already has
 * (GTAO's `builtinAOContext`, say). The existing context node becomes the
 * child of the new one; `getFlowContextData` merges every context node in
 * the chain, so both sets of keys reach the materials. Marking twice is a
 * no-op.
 */
export function markCompositePass(scenePass: MarkablePass): void {
    const existing = scenePass.contextNode;
    if (existing && hasCompositeMark(flowContextData(existing))) return;
    const mark = { [GI_COMPOSITE_MARK]: true };
    scenePass.contextNode = existing ? context(existing as unknown as Node, mark) : context(mark);
    // PassNode caches the merged context by its own version (`_getContextNode`).
    scenePass.needsUpdate = true;
}

/** Every context value in a context-node chain, merged (three's `ContextNode.getFlowContextData`, missing from the typings). */
export function flowContextData(contextNode: ContextNode): Record<string, unknown> {
    return (contextNode as unknown as { getFlowContextData(): Record<string, unknown> }).getFlowContextData();
}

/** Physical-material features whose effect on indirect diffuse `giAlbedo` doesn't reproduce. */
const UNSUPPORTED_PHYSICAL_FEATURES = ['clearcoat', 'sheen', 'iridescence', 'transmission', 'dispersion', 'diffuseRoughness'] as const;

type MaterialFlags = Material &
    Partial<{
        isMeshStandardMaterial: boolean;
        isMeshPhysicalMaterial: boolean;
        isMeshLambertMaterial: boolean;
        isMeshPhongMaterial: boolean;
        isNodeMaterial: boolean;
        lights: boolean;
        lightsNode: unknown;
        fragmentNode: unknown;
    }> &
    Record<string, unknown>;

/** The lighting model a material's indirect diffuse goes through, if `giAlbedo` reproduces it. */
export type CompositedModel = 'physical' | 'lambert';

/**
 * Which of three's lighting models lights this material, if `giAlbedo` can
 * reproduce its indirect-diffuse weight; null otherwise. Works on the
 * material as the builder holds it: a classic material (`MeshStandardMaterial`)
 * or a node material (node materials copy the classic flags,
 * `NodeMaterial.setDefaultValues`).
 *
 * - Standard, and Physical without the features below: `PhysicalLightingModel`,
 *   weight `diffuseContribution × (1 − F_ss − F_ms)` (r187
 *   `PhysicalLightingModel.indirectDiffuse`);
 * - Lambert and Phong: `PhongLightingModel`, weight `diffuseColor`.
 *
 * Physical materials with clearcoat, sheen, iridescence, transmission,
 * dispersion or rough diffuse (EON) change that weight in ways `giAlbedo`
 * doesn't reproduce, so they stay per fragment.
 */
export function compositedModel(material: Material | null | undefined): CompositedModel | null {
    const flags = material as MaterialFlags | null | undefined;
    if (!flags) return null;
    // A node material that isn't lit (MeshBasicNodeMaterial, or `lights = false`) gets no GI at all.
    if (flags.isNodeMaterial && flags.lights !== true) return null;
    // A custom lights list or a custom fragment replaces three's lighting.
    if (flags.lightsNode || flags.fragmentNode) return null;
    if (flags.isMeshPhysicalMaterial) {
        for (const feature of UNSUPPORTED_PHYSICAL_FEATURES) {
            const amount = flags[feature];
            if ((typeof amount === 'number' && amount > 0) || (flags[`${feature}Node`] ?? null) !== null) return null;
        }
        return 'physical';
    }
    if (flags.isMeshStandardMaterial) return 'physical';
    if (flags.isMeshLambertMaterial || flags.isMeshPhongMaterial) return 'lambert';
    return null;
}

/**
 * The rule (file comment): this build's surface is composited when the pass
 * carries the mark and the material is opaque with a supported model.
 * Transparent surfaces blend over what's behind them, so a composite after
 * the pass can't light them; they keep GI per fragment.
 */
export function compositeTakesGI(builder: Pick<NodeBuilder, 'context' | 'material'>): boolean {
    if (!hasCompositeMark(builder.context as object)) return false;
    const material = builder.material;
    if (!material || material.transparent) return false;
    return compositedModel(material) !== null;
}
