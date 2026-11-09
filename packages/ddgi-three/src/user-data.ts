/**
 * Per-object GI settings a host puts in three's `object.userData`.
 *
 * - `userData.giContributes` (boolean): whether the object blocks and bounces
 *   light for GI. Defaults to true, except for skinned meshes (characters),
 *   which default to false (DECISIONS 2026-11-04: characters don't feed GI;
 *   things that should, like trees, opt in). An object that doesn't
 *   contribute still *receives* GI through the light.
 * - `userData.giLightGroup` (0..3): the light group of a directional, point or
 *   spot light. Defaults to 0. DDGI can re-weight a whole group instantly.
 * - `userData.giCastsShadows` (boolean, on a light): whether GI traces shadow
 *   rays for it. Defaults to three's `castShadow`. Set it `true` on lights
 *   three draws without shadows only because there are too many for shadow
 *   maps (hundreds of candles under three's `ClusteredLighting`, which skips
 *   shadowed lights): GI's shadows cost no shadow map, and without them the
 *   light leaks through walls into GI.
 * - `material.userData.giSingleSided` (boolean, on a *material*): rays pass
 *   through the back of its surfaces. Defaults to false: every material is
 *   double-sided for GI whatever three's `side`, because back-face hits are
 *   how probes find out they sit inside geometry (scene contract,
 *   `doubleSided`), and most glTF materials are `FrontSide`.
 * - `material.userData.giBlocksLight` (boolean, on a *material*): `true` makes
 *   a material three draws see-through (alpha-blended, `transmission`, low
 *   `opacity`) block and bounce light for GI like an opaque one. Without it,
 *   such materials let light through (convert.ts, `materialDescriptor`), and
 *   a mesh made only of them is left out of GI's rays (scene-sync.ts). Only
 *   `true` is read.
 * - `userData.giDeforming` (boolean, on a mesh): whether GI follows the
 *   mesh's changing shape (deforming.ts). A `SkinnedMesh` or a mesh with
 *   morph targets deforms by default once it contributes to GI; `false` keeps
 *   it in its bind pose. `true` on a plain mesh says its position attribute
 *   is animated on the CPU, so each edit is sent as a pose instead of a new
 *   shape.
 * - `userData.giDeformBounds` (a three `Box3`, or `{ min: [x, y, z], max:
 *   [x, y, z] }`, on a deforming mesh): a box in the mesh's own space that
 *   every pose stays inside. Without it the adapter estimates one
 *   (deforming.ts).
 *
 * The flag is read from the object itself, not inherited from its parents.
 */

import type { Aabb } from '@pmndrs/ddgi-core';
import type { Object3D } from 'three';

export const GI_USER_DATA = {
    contributes: 'giContributes',
    lightGroup: 'giLightGroup',
    castsShadows: 'giCastsShadows',
    singleSided: 'giSingleSided',
    blocksLight: 'giBlocksLight',
    deforming: 'giDeforming',
    deformBounds: 'giDeformBounds',
} as const;

/** Light groups the platform has (scene contract: 0..3). */
export const LIGHT_GROUP_COUNT = 4;

export function contributesToGI(object: Object3D): boolean {
    return contributesFlag(object, (object as { isSkinnedMesh?: boolean }).isSkinnedMesh === true);
}

/** {@link contributesToGI} with the skinned-mesh test already made (the scene sync reads it once per object). */
export function contributesFlag(object: Object3D, skinned: boolean): boolean {
    const value = object.userData[GI_USER_DATA.contributes];
    return typeof value === 'boolean' ? value : !skinned;
}

/** Whether a material opted out of double-sided GI (`userData.giSingleSided === true`). */
export function singleSidedForGI(userData: Record<string, unknown>): boolean {
    return userData[GI_USER_DATA.singleSided] === true;
}

/** Whether a see-through material opted in to blocking light for GI (`userData.giBlocksLight === true`). */
export function blocksLightOptIn(userData: Record<string, unknown>): boolean {
    return userData[GI_USER_DATA.blocksLight] === true;
}

/** Whether GI shadows a light: `userData.giCastsShadows` when it is a boolean, otherwise three's `castShadow`. */
export function castsShadowsForGI(light: Object3D): boolean {
    const value = light.userData[GI_USER_DATA.castsShadows];
    return typeof value === 'boolean' ? value : light.castShadow;
}

/** The light group from userData, or 0 when absent or not a valid group. */
export function lightGroup(object: Object3D): number {
    const value = object.userData[GI_USER_DATA.lightGroup];
    return Number.isInteger(value) && value >= 0 && value < LIGHT_GROUP_COUNT ? (value as number) : 0;
}

/** `userData.giDeforming` when it is a boolean, otherwise undefined (the mesh's kind decides). */
export function deformingFlag(object: Object3D): boolean | undefined {
    const value = object.userData[GI_USER_DATA.deforming];
    return typeof value === 'boolean' ? value : undefined;
}

/**
 * `userData.giDeformBounds` as the platform's box, or undefined when absent
 * or not a box of finite numbers. Accepts a three `Box3` (`min` and `max` as
 * `Vector3`) or plain `[x, y, z]` triples.
 */
export function deformBoundsFlag(object: Object3D): Aabb | undefined {
    const value = object.userData[GI_USER_DATA.deformBounds] as { min?: unknown; max?: unknown } | undefined;
    if (!value || typeof value !== 'object') return undefined;
    const min = triple(value.min);
    const max = triple(value.max);
    return min && max ? { min, max } : undefined;
}

function triple(value: unknown): [number, number, number] | undefined {
    if (Array.isArray(value)) {
        const numbers = value as unknown[];
        if (numbers.length === 3 && numbers.every((number) => typeof number === 'number' && Number.isFinite(number))) return [numbers[0] as number, numbers[1] as number, numbers[2] as number];
        return undefined;
    }
    const vector = value as { x?: unknown; y?: unknown; z?: unknown } | null;
    if (vector && typeof vector === 'object' && [vector.x, vector.y, vector.z].every((number) => typeof number === 'number' && Number.isFinite(number))) {
        return [vector.x as number, vector.y as number, vector.z as number];
    }
    return undefined;
}
