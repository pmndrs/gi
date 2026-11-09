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
 *   a material three draws see-through block and bounce light for GI like an
 *   opaque one: glass (`transmission`) blocks, and an alpha-blended material
 *   is opaque instead of stochastic (convert.ts, `materialDescriptor`). Alpha
 *   cutouts keep their holes. The material colour overrides (`giAlbedo`,
 *   `giAlbedoMap`, `giEmissive`) are described at the end of this file. Only
 *   `true` is read.
 *
 * The flag is read from the object itself, not inherited from its parents.
 */

import type { Color, Object3D, Texture } from 'three';

export const GI_USER_DATA = {
    contributes: 'giContributes',
    lightGroup: 'giLightGroup',
    castsShadows: 'giCastsShadows',
    singleSided: 'giSingleSided',
    blocksLight: 'giBlocksLight',
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

// ---------------------------------------------------------------------------
// Material colour overrides (M4, T4.3)
// ---------------------------------------------------------------------------

/**
 * What GI sees of a material, said by the host when the adapter can't work
 * it out (a `colorNode` graph or a `ShaderMaterial`), or should see
 * something else. Each replaces one input of convert.ts's
 * `materialDescriptor`, on the *material*:
 *
 * - `material.userData.giAlbedo`: the albedo colour, instead of `color`. The
 *   albedo map and metalness still apply.
 * - `material.userData.giAlbedoMap` (a three `Texture`): the albedo texture,
 *   instead of `map`, sampled with its own uv mapping.
 * - `material.userData.giEmissive`: the emissive colour, instead of
 *   `emissive`. `emissiveIntensity` and `emissiveMap` still apply.
 *
 * A colour is a three `Color` (linear, as three keeps it), an `[r, g, b]`
 * array (linear), or anything `new Color(value)` takes (a hex number or a
 * CSS string, both sRGB, as three reads them). Anything else is ignored.
 */
export const GI_MATERIAL_USER_DATA = {
    albedo: 'giAlbedo',
    albedoMap: 'giAlbedoMap',
    emissive: 'giEmissive',
} as const;

/** A userData colour into `target` (see {@link GI_MATERIAL_USER_DATA}), or null when the value isn't a colour. */
function readColorValue(value: unknown, target: Color): Color | null {
    if (value === undefined || value === null) return null;
    if (typeof value === 'number') return Number.isFinite(value) ? target.setHex(value) : null;
    if (typeof value === 'string') return value.trim() === '' ? null : target.setStyle(value);
    if (Array.isArray(value)) {
        if (value.length !== 3 || !value.every((channel) => typeof channel === 'number' && Number.isFinite(channel))) return null;
        return target.setRGB(value[0], value[1], value[2]);
    }
    const color = value as { r?: unknown; g?: unknown; b?: unknown };
    if (typeof color === 'object' && [color.r, color.g, color.b].every((channel) => typeof channel === 'number' && Number.isFinite(channel))) {
        return target.setRGB(color.r as number, color.g as number, color.b as number);
    }
    return null;
}

/** `userData.giAlbedo` written into `target` (linear), or null when absent. */
export function giAlbedoOverride(userData: Record<string, unknown>, target: Color): Color | null {
    return readColorValue(userData[GI_MATERIAL_USER_DATA.albedo], target);
}

/** `userData.giEmissive` written into `target` (linear), or null when absent. */
export function giEmissiveOverride(userData: Record<string, unknown>, target: Color): Color | null {
    return readColorValue(userData[GI_MATERIAL_USER_DATA.emissive], target);
}

/** `userData.giAlbedoMap` when it is a three texture, otherwise null. */
export function giAlbedoMapOverride(userData: Record<string, unknown>): Texture | null {
    const value = userData[GI_MATERIAL_USER_DATA.albedoMap] as { isTexture?: boolean } | undefined;
    return value && typeof value === 'object' && value.isTexture === true ? (value as Texture) : null;
}
