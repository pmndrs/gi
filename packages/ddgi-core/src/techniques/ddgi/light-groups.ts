/**
 * Light groups (T2.5, docs/design/m2-lights.md §5): one irradiance layer per
 * light group, added up with the host's weights into the atlas renderers
 * sample.
 *
 * By linearity, the field for "group g at weight w" is w × the field for
 * group g at weight 1, bounces included, as long as each group's bounce
 * reads only its own layer (SPEC's HomeFig rule 1). So DDGI keeps a layer per
 * group, each traced at the lights' real colour and intensity (rule 3), and
 * an environment layer for the sky and emissive surfaces (rule 2: the sky's
 * shape changes with time of day, so it is re-traced, never just scaled).
 * `ddgi/compose` writes Σ weight × layer per texel into the composed atlas.
 * A weight change is a compose of every tile in the frame it happens: no
 * probe is marked, no history is cut, no ray is traced for it.
 *
 * **Groups off** (`lightGroups` null or a count of 0, the default) keeps one
 * layer holding everything, and no composed atlas: the kernels run M1's
 * arithmetic with a layer loop of one.
 *
 * **Where the layers live:** as blocks side by side inside each array layer
 * of the irradiance atlas (layer L's tile of probe (x, z) at
 * ((L × nx + x) × tile, z × tile)), not as more array layers: the atlas
 * already uses one array layer per probe row, and the default limit is 256.
 *
 * This file is the plan only (which layer holds what, and the weights):
 * plain numbers, unit-tested without a GPU, read by the technique and its
 * WGSL generators. T2.6b adds a sun layer to the same plan (§6.3): a layer
 * kind, its place in the list and its weight; the kernels loop over
 * whatever the plan lists.
 */

import type { ColorRGB } from '../../platform/contract/math.ts';

/** Light groups of the contract: a light's `group` is 0..3 (contract/scene.ts). */
export const MAX_LIGHT_GROUPS = 4;

/** Every group's bit: the lights that reach the probes at all. */
export const ALL_LIGHT_GROUPS_MASK = (1 << MAX_LIGHT_GROUPS) - 1;

/**
 * The most irradiance layers a volume keeps: four groups and the environment,
 * with room for T2.6b's sun layer. `ddgi/blend` keeps every layer's texel
 * estimates and old values in workgroup memory beside the rays' directions:
 * 8 KB + 1,152 bytes a layer at 6×6 texels, 14.9 KB at 6 layers, within the
 * 16 KB default. The compose weights uniform is sized for it.
 */
export const MAX_IRRADIANCE_LAYERS = 6;

/**
 * `DDGISettings.lightGroups`: light groups on, how many, and their weights.
 * `count` sizes the GPU resources (a new `prepare` when it changes); the
 * weights are live, read every frame.
 */
export interface DDGILightGroups {
    /**
     * Groups with a layer of their own, 1..4: lights of group g < count go into
     * layer g, lights of a higher group into the last one (count − 1). 0 is the
     * same as `lightGroups: null` (off). New `prepare` when changed.
     */
    count: number;
    /**
     * Per group, a linear RGB multiplier on top of the lights' own colour and
     * intensity (1, 1, 1 is the lights as they are). Missing entries are 1.
     * Live: a change shows in the next frame, with no update lag and no noise.
     */
    weights: ColorRGB[];
    /** The environment layer's multiplier (the sky and emissive surfaces). Live. */
    environmentWeight: ColorRGB;
}

/** What a layer holds. */
export type IrradianceLayerKind =
    /** Groups off: the one layer, every light, the sky and emission. */
    | 'everything'
    /** One light group's lights (its direct light and its own bounces). */
    | 'lightGroup'
    /** The sky (rays that miss) and emissive surfaces, and their own bounces. */
    | 'environment';

export interface IrradianceLayer {
    readonly kind: IrradianceLayerKind;
    /** For `lightGroup`: which group's weight scales it. */
    readonly group: number;
    /** The contract groups (bits 0..3) whose direct light goes into this layer. */
    readonly groupMask: number;
    /** Whether rays that miss (the sky) and emissive surfaces add their light here. */
    readonly holdsEnvironment: boolean;
}

/** Which layer holds what, for one volume (fixed until the next `prepare`). */
export interface LayerPlan {
    readonly layers: readonly IrradianceLayer[];
    /** Light groups with a layer of their own (0 with groups off). */
    readonly groupCount: number;
    /** The layer the sky and emission go into. */
    readonly environmentLayer: number;
    /** For contract groups 0..3, the layer their direct light goes into (`DirectLightFrame.groupLayers`). */
    readonly groupLayers: readonly number[];
    /** False with one layer: renderers sample the irradiance atlas itself and nothing is composed. */
    readonly composed: boolean;
}

/** The light groups in use: 0 when off, else `count` limited to 1..4 (whole groups). */
export function lightGroupCount(lightGroups: DDGILightGroups | null | undefined): number {
    if (!lightGroups) return 0;
    const count = lightGroups.count;
    if (!Number.isFinite(count) || count <= 0) return 0;
    return Math.min(MAX_LIGHT_GROUPS, Math.max(1, Math.round(count)));
}

/**
 * The layers for a light-group setting: one layer holding everything with
 * groups off; otherwise one per group (groups at or past the count share the
 * last group's), then the environment.
 */
export function layerPlan(lightGroups: DDGILightGroups | null | undefined): LayerPlan {
    const groupCount = lightGroupCount(lightGroups);
    if (groupCount === 0) {
        return {
            layers: [{ kind: 'everything', group: 0, groupMask: ALL_LIGHT_GROUPS_MASK, holdsEnvironment: true }],
            groupCount: 0,
            environmentLayer: 0,
            groupLayers: [0, 0, 0, 0],
            composed: false,
        };
    }
    const groupLayers = Array.from({ length: MAX_LIGHT_GROUPS }, (_, group) => Math.min(group, groupCount - 1));
    const layers: IrradianceLayer[] = [];
    for (let group = 0; group < groupCount; group++) {
        const groupMask = groupLayers.reduce((mask, layer, contractGroup) => (layer === group ? mask | (1 << contractGroup) : mask), 0);
        layers.push({ kind: 'lightGroup', group, groupMask, holdsEnvironment: false });
    }
    layers.push({ kind: 'environment', group: 0, groupMask: 0, holdsEnvironment: true });
    return { layers, groupCount, environmentLayer: groupCount, groupLayers, composed: true };
}

/** A weight as the kernel takes it: three finite, non-negative numbers (anything else counts as 1). */
function sanitizedWeight(weight: ColorRGB | undefined): [number, number, number] {
    if (!weight) return [1, 1, 1];
    return [0, 1, 2].map((channel) => {
        const value = weight[channel];
        return Number.isFinite(value) && value >= 0 ? value : 1;
    }) as [number, number, number];
}

/**
 * Each layer's weight, 4 floats a layer (rgb and 0), in the plan's order:
 * a light group's from `weights`, the environment's from
 * `environmentWeight`. Written into `target` (MAX_IRRADIANCE_LAYERS × 4
 * floats); returns whether anything changed, so the technique knows when a
 * weight change needs every tile composed again.
 */
export function writeLayerWeights(plan: LayerPlan, lightGroups: DDGILightGroups | null | undefined, target: Float32Array): boolean {
    let changed = false;
    plan.layers.forEach((layer, index) => {
        const weight =
            layer.kind === 'lightGroup' ? sanitizedWeight(lightGroups?.weights[layer.group]) : layer.kind === 'environment' ? sanitizedWeight(lightGroups?.environmentWeight) : ([1, 1, 1] as const);
        for (let channel = 0; channel < 3; channel++) {
            // Compared as the f32 the uniform holds, so a weight set again to the same value changes nothing.
            if (target[index * 4 + channel] !== Math.fround(weight[channel])) changed = true;
            target[index * 4 + channel] = weight[channel];
        }
        target[index * 4 + 3] = 0;
    });
    return changed;
}

/** Default weights for `count` groups: everything as the lights are. */
export function defaultLightGroups(count: number): DDGILightGroups {
    return { count, weights: Array.from({ length: MAX_LIGHT_GROUPS }, () => [1, 1, 1] as ColorRGB), environmentWeight: [1, 1, 1] };
}
