/**
 * How three draws a scene's direct lights: its default lighting, or, for
 * scenes with many shadowless point lights (`sponza-candles`' 1,024 candles),
 * three's `ClusteredLighting`.
 *
 * *Why:* three's default lighting writes shader code for every light into
 * every material. With 1,024 point lights the page's main thread was blocked
 * for more than 30 s building shaders (T2.1, RTX 3060, Chrome 156), so the
 * scene could not be shown. Clustered lighting loops over a light list per
 * screen cluster instead, so the shaders don't grow with the light count.
 *
 * The candles have no range (three's `distance` 0, as Intel's file has them),
 * so every cluster lists every light, up to MAX_CLUSTERED_LIGHTS: the picture
 * is three's exact unshadowed direct light then, at a cost per pixel that
 * grows with the light count. Past it (the `candles` setting above 1,024)
 * three's picture leaves the rest out; GI has them all. Shadowed lights are
 * never clustered (three keeps them on its default path).
 *
 * Only this file decides it, so the viewer and the GPU pages that render a
 * loaded scene draw the same way.
 */

import { ClusteredLighting } from 'three/addons/lighting/ClusteredLighting.js';
import type { Light, Lighting, PointLight } from 'three/webgpu';

/** From this many shadowless point and spot lights on, three draws them clustered. */
export const CLUSTERED_LIGHTS_FROM = 64;
/** The most lights three's clustered lighting holds, and lists per cluster. */
export const MAX_CLUSTERED_LIGHTS = 1024;
/** Clusters of 64 x 64 pixels and 8 depth slices: lights with no range are in every cluster anyway, so fine clusters would only cost memory. */
const TILE_PIXELS = 64;
const DEPTH_SLICES = 8;

/** The lighting three should draw `lights` with: `defaultLighting` unless there are many shadowless point or spot lights. */
export function sceneLighting(lights: readonly Light[], defaultLighting: Lighting): Lighting {
    const shadowless = lights.filter((light) => ((light as PointLight).isPointLight || (light as { isSpotLight?: boolean }).isSpotLight) && !light.castShadow).length;
    if (shadowless < CLUSTERED_LIGHTS_FROM) return defaultLighting;
    const capacity = Math.min(shadowless, MAX_CLUSTERED_LIGHTS);
    return new ClusteredLighting(capacity, TILE_PIXELS, DEPTH_SLICES, capacity);
}
