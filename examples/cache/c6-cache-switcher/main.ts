/**
 * The cache switcher: the radiance cache with one world cache at a time,
 * picked from `RADIANCE_CACHE_VARIANT_PRESETS` (P6.5's variants), on two
 * rooms or an open courtyard. The panel shows the variant's card, its GPU
 * time per pass and memory, and the cache's own debug views.
 *
 * - A preset is a set of compare values: `applyCompareValues` puts them into
 *   the settings, which are given at construction, because some are read
 *   only then (the world cache, the neural tail, the screen channels
 *   temporal feedback asks for). A switch reloads the page (panel.ts).
 * - A preset that needs a DDGI volume gets the scene's suggested probe box,
 *   stacked under the cache with `setFallback` before `prepare`, as the
 *   viewer does; DDGI warms up before the first picture.
 * - The rest is C1's wiring: `screenInputs` on three's MRT pass, the screen
 *   stage run before `ddgi()`, the composite.
 */

import * as THREE from 'three/webgpu';
import { mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import { createGI } from '@pmndrs/gi/core';
import { DDGITechnique } from '@pmndrs/gi/ddgi';
import {
    applyCompareValues,
    DEFAULT_RADIANCE_CACHE_SETTINGS,
    RADIANCE_CACHE_DEBUG_VIEW,
    RadianceCacheTechnique,
    variantPreset,
    type RadianceCacheDebugViewId,
    type RadianceCacheSettings,
} from '@pmndrs/gi/radiance-cache';
import { ddgi, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT, screenInputs, ThreeGI } from '@pmndrs/gi/three';
import { createRendererAndDevice } from '../../shared/renderer.ts';
import { loadScene } from '../../shared/scene.ts';
import { showCacheStats } from '../../shared/cache-stats.ts';
import { showPassTiming } from '../../shared/pass-timing.ts';
import { CacheDebugOverlay } from './debug-overlay.ts';
import { keepCamera, showSwitcher } from './panel.ts';

const SCENES = [
    { id: 'two-room', label: 'Two rooms: closed', manifest: 'two-room' },
    { id: 'courtyard', label: 'Courtyard: open sky', manifest: 'courtyard-tree' },
] as const;
const OVERLAYS = [
    { id: 'none', label: 'None: the lit picture', view: null },
    { id: 'probe-atlas', label: 'Probe atlas', view: RADIANCE_CACHE_DEBUG_VIEW.probeAtlas },
    { id: 'probe-mask', label: 'Probe mask', view: RADIANCE_CACHE_DEBUG_VIEW.probeMask },
    { id: 'probe-sh', label: "Probes' SH", view: RADIANCE_CACHE_DEBUG_VIEW.sphericalHarmonics },
] as const;

const params = new URLSearchParams(location.search);
const preset = variantPreset(params.get('variant') ?? '') ?? variantPreset('gi12-ddgi')!;
const place = SCENES.find((entry) => entry.id === params.get('scene')) ?? SCENES[0];

// The panel first, so its card is there while GI starts; the overlay applies once the pipeline exists.
let overlayView: RadianceCacheDebugViewId | null = null;
let applyOverlay = (): void => {};
const panel = showSwitcher({
    variant: preset,
    scene: place.id,
    scenes: SCENES,
    overlays: OVERLAYS,
    onOverlay: (view) => {
        overlayView = view;
        applyOverlay();
    },
});

// The screen inputs need a single-sampled pass.
const { renderer, device } = await createRendererAndDevice({ antialias: false });
const { scene, camera, controls, probes } = loadScene(place.manifest, renderer);
keepCamera(camera, controls, place.id);

// The variant: every compare toggle at its value, the rest at the defaults.
const settings: RadianceCacheSettings = { ...DEFAULT_RADIANCE_CACHE_SETTINGS };
applyCompareValues(settings, preset.values);

const gi = createGI(device, { gpuTiming: device.features.has('timestamp-query') });
const cache = gi.addTechnique(new RadianceCacheTechnique(settings));
gi.setOutput(cache);
let under: DDGITechnique | null = null;
if (preset.needs.ddgiVolume) {
    under = gi.addTechnique(new DDGITechnique({ volume: probes }));
    gi.setFallback(cache, under); // before prepare: the cache builds its DDGI lookup then
}
const preparing = gi.prepare();
// The cache ignores the budget; it hands DDGI its probes a frame.
const threeGI = new ThreeGI(gi, scene, { preparing, budget: { units: 256 } });
await preparing;
if (settings.neuralCache === 'tail' && !cache.neuralTail) {
    panel.note("The neural tail is off here, so this is the grid alone: the site doesn't bundle @pathtracer/mlp (the console says why).");
}

// DDGI warms up first, so the cache's later bounces don't start from a black field.
while (under?.warmingUp) {
    threeGI.update(camera, 0);
    await device.queue.onSubmittedWorkDone();
}

const pipeline = new THREE.RenderPipeline(renderer);
const scenePass = pass(scene, camera);
const screen = screenInputs(scenePass, threeGI);
scenePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), ...screen.outputs }));
const lit = screen.encodeBefore(ddgi(scenePass, threeGI));
const overlay = new CacheDebugOverlay(device, cache);
const debug = screen.encodeBefore(overlay.node);
applyOverlay = () => {
    overlay.view = overlayView;
    pipeline.outputNode = overlayView === null ? lit : debug;
    pipeline.outputColorTransform = overlayView === null; // the debug image is already display colour
    pipeline.needsUpdate = true;
};
applyOverlay();

const mib = (bytes = 0): string => `${Math.round(bytes / 2 ** 20).toLocaleString('en')} MiB`;
panel.readouts(() => {
    const { memory } = gi.stats;
    const counters = cache.counters;
    const lines = [`GPU memory: radiance cache ${mib(memory.techniques['radiance-cache'])}${under ? `, DDGI ${mib(memory.techniques.ddgi)}` : ''}`];
    const capacity = counters['world cache cells'] ?? counters['world cache slots'];
    if (capacity) lines.push(`World cache: room for ${capacity.toLocaleString('en')} ${settings.worldCache === 'sharc' ? 'slots' : 'cells'}; how many are in use stays on the GPU`);
    const steps = counters['neural tail training steps'];
    if (steps !== undefined) lines.push(`Neural tail: ${steps.toLocaleString('en')} training steps`);
    return lines;
});
showPassTiming(gi, {
    container: panel.timing,
    // A pass the shared list doesn't describe yet: CountScreenProbes, a scan and ScatterScreenProbes (docs/design/radiance-cache.md).
    descriptions: { 'radiance-cache/LookupScreenProbes': 'Counts the probes reprojection kept and gathers them into one list, for the passes after it.' },
});
showCacheStats(device, gi, cache, () => (under ? `${preset.id}, DDGI under it` : preset.id));

renderer.setAnimationLoop(() => {
    threeGI.update(camera); // the frame's GI work; the screen stage runs inside the pipeline
    pipeline.render();
    overlay.draw(); // the debug view of this frame's probes, shown next frame
});
