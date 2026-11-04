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
 * - `material.userData.giSingleSided` (boolean, on a *material*): rays pass
 *   through the back of its surfaces. Defaults to false: every material is
 *   double-sided for GI whatever three's `side`, because back-face hits are
 *   how probes find out they sit inside geometry (scene contract,
 *   `doubleSided`), and most glTF materials are `FrontSide`.
 *
 * The flag is read from the object itself, not inherited from its parents.
 */

import type { Object3D } from 'three';

export const GI_USER_DATA = {
    contributes: 'giContributes',
    lightGroup: 'giLightGroup',
    singleSided: 'giSingleSided',
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

/** The light group from userData, or 0 when absent or not a valid group. */
export function lightGroup(object: Object3D): number {
    const value = object.userData[GI_USER_DATA.lightGroup];
    return Number.isInteger(value) && value >= 0 && value < LIGHT_GROUP_COUNT ? (value as number) : 0;
}
