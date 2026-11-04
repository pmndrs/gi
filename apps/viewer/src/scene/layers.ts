/**
 * Three.js layers, used so one scene can be drawn by several techniques at
 * once (the split-screen wipe) and so "GI only" can hide the direct lights.
 *
 * How it works: every object stays on layer 0, which is what the
 * `LightProbeGrid` bake cameras and the VXGI voxelizer look at, so baking
 * always sees geometry and lights together. Viewing cameras look at the
 * other layers instead, and each technique chooses which of them to include:
 *
 *   VIEW_GEOMETRY    the model, the sky and the reference cube
 *   DIRECT_LIGHTS    the manifest's lights and the emissive-quad area lights
 *   PROBE_GRID       the `LightProbeGrid` (only its own cameras see it)
 *   DDGI             DDGI's GI light (only DDGI's camera sees it)
 *   DDGI_PROBES      DDGI's probe spheres (DDGI's camera, when they are switched on)
 *
 * Lights follow the camera's layers too (`Renderer.js` tests `light.layers`),
 * which is what makes "GI only" possible without touching light intensities.
 * Changing an intensity would make VXGI re-inject its lighting.
 */

import type { Layers } from 'three';

export const LAYER_VIEW_GEOMETRY = 1;
export const LAYER_DIRECT_LIGHTS = 2;
export const LAYER_PROBE_GRID = 3;
export const LAYER_DDGI = 4;
export const LAYER_DDGI_PROBES = 5;

/**
 * - `on`: direct light plus the technique's GI.
 * - `only`: the GI alone; the direct lights are hidden from the viewing camera.
 * - `off`: direct light only, drawn through this technique's own pipeline. The
 *   benchmark uses it as the reference that the GI cost is measured against.
 */
export type GiMode = 'on' | 'only' | 'off';

/**
 * Which layers a viewing camera sees.
 * @param giMode see {@link GiMode}
 * @param giLayers the technique's own GI layers (the probe grid), left out when GI is off
 */
export function applyViewLayers(layers: Layers, giMode: GiMode, giLayers: readonly number[]): void {
    layers.disableAll();
    layers.enable(LAYER_VIEW_GEOMETRY);
    if (giMode !== 'only') layers.enable(LAYER_DIRECT_LIGHTS);
    if (giMode !== 'off') for (const layer of giLayers) layers.enable(layer);
}
