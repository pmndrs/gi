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
 * That per-pixel cost is what made the viewer unusable on `sponza-candles`
 * (T2.8): every pixel looped over all 1,024 candles, about 0.4 s a frame on
 * the RTX 3060 at 1280 x 720, so DDGI's start-up frames crawled behind it and
 * never finished. The viewer (only the viewer: the GPU pages that draw a
 * loaded scene keep the exact lists) gives such lights a reach with
 * {@link limitClusteredLightReach}, three's own way to cull lights per cluster
 * (`light.distance`): see there for what that changes.
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

/**
 * With a reach (limitClusteredLightReach), lights are culled per cluster, so
 * finer clusters pay: 32 x 32 pixels and 24 depth slices are three's own
 * defaults. A cluster lists at most this many lights, enough for the densest
 * clusters of 1,024 candles at a reach of several metres.
 */
const REACHED_TILE_PIXELS = 32;
const REACHED_DEPTH_SLICES = 24;
const REACHED_MAX_LIGHTS_PER_CLUSTER = 512;

/**
 * The reach the viewer gives many shadowless lights with no range of their
 * own, in metres (limitClusteredLightReach). On `sponza-candles` (1,024
 * candles, RTX 3060, 1280 x 720, headless correctness runs, not timed under
 * the clock rule) the viewer drew 2 frames a second unranged, 9 at a reach of
 * 6 m, 18 at 4 m and 33 at 3 m. 3 m keeps each candle's own pool of light
 * on the wall or floor next to it, and the frame rate usable; a link can ask
 * for more (`reach=6`) or for the exact unranged picture (`reach=0`).
 */
export const DEFAULT_CLUSTERED_LIGHT_REACH_M = 3;

function isShadowlessLocalLight(light: Light): boolean {
    return ((light as PointLight).isPointLight || (light as { isSpotLight?: boolean }).isSpotLight === true) && !light.castShadow;
}

function shadowlessLocalLights(lights: readonly Light[]): Light[] {
    return lights.filter(isShadowlessLocalLight);
}

/** The lighting three should draw `lights` with: `defaultLighting` unless there are many shadowless point or spot lights. */
export function sceneLighting(lights: readonly Light[], defaultLighting: Lighting): Lighting {
    const shadowless = shadowlessLocalLights(lights);
    if (shadowless.length < CLUSTERED_LIGHTS_FROM) return defaultLighting;
    const capacity = Math.min(shadowless.length, MAX_CLUSTERED_LIGHTS);
    // Lights with a range are culled per cluster, so finer clusters cut the lists; unranged ones are in every cluster.
    const ranged = shadowless.every((light) => (light as PointLight).distance > 0);
    if (ranged) return new ClusteredLighting(capacity, REACHED_TILE_PIXELS, REACHED_DEPTH_SLICES, Math.min(capacity, REACHED_MAX_LIGHTS_PER_CLUSTER));
    return new ClusteredLighting(capacity, TILE_PIXELS, DEPTH_SLICES, capacity);
}

/**
 * Gives the shadowless point and spot lights of a clustered scene (from
 * CLUSTERED_LIGHTS_FROM of them) with no range of their own a reach of
 * `reachMetres` (three's `distance`), so three's clustered lighting culls
 * them per cluster instead of every pixel looping over every light. Returns
 * how many lights it changed; a reach of 0 or less changes nothing.
 *
 * What it changes, in plain terms: three's light falls off with distance
 * squared as before, then fades to nothing at the reach (three's window,
 * (1 − (d/reach)⁴)², 12% down at half the reach). Light from candles further
 * away is left out. three draws these lights without shadows, so far
 * candles lit surfaces straight through walls anyway; the reach mostly takes
 * away light that shouldn't be there. GI reads `distance` as the light's
 * range too (the three adapter), so GI and three keep the same lights and
 * agree with each other; GI's far candle light is shadowed and real, and it
 * is left out as well.
 */
export function limitClusteredLightReach(lights: readonly Light[], reachMetres: number): number {
    if (!(reachMetres > 0)) return 0;
    const shadowless = shadowlessLocalLights(lights);
    if (shadowless.length < CLUSTERED_LIGHTS_FROM) return 0;
    let changed = 0;
    for (const light of shadowless) {
        const local = light as PointLight;
        if (local.distance > 0) continue;
        local.distance = reachMetres;
        changed++;
    }
    return changed;
}
