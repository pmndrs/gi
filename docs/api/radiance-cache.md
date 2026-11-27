# The radiance cache (`@pmndrs/gi/radiance-cache`)

The radiance cache is the GI platform's second technique, next to DDGI. It lights what the camera sees:

- **Screen probes.** One probe per 16 × 16 pixel tile is placed on a visible surface each frame. Each traces 64 rays, keeps its light from frame to frame and moves with the camera.
- **A world cache.** The probes' rays end in a hash grid of light at their hits, with cells sized by their distance from the camera. A second layer of cells carries the next bounce.
- **A denoiser.** The irradiance per pixel goes through AMD's GI-1.0 denoiser before it is shown.

It gives indirect diffuse light only. three.js draws direct light, emission and reflections as it always does.

There is no volume to place. In exchange it needs the host's depth, normals and motion every frame, about 1.1 GiB of GPU memory, and a device with more than WebGPU's default limits.

It is built from AMD's GI-1.0, as GI-1.2 in [Capsaicin](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin) v1.3 ships it (MIT). It isn't a one-to-one port. Capsaicin's errors are fixed, among them:

- the SH projection's weight and the empty probe cells' backup: together, Capsaicin reads a constant sky 1.30 times too bright at the first frame and 0.87 times at steady state;
- the second bounce's 1 / (1 − p) weight and its misses, which counted no sample;
- the bypass paths that added light to the wrong probe ray.

Every difference from Capsaicin, with its reason, is in the design doc's divergence register ([docs/design/radiance-cache.md](../design/radiance-cache.md)). Its settings each name the register row they come from.

The code is in [`packages/ddgi-core/src/techniques/radiance-cache/`](../../packages/ddgi-core/src/techniques/radiance-cache/). [Every export](#every-export-of-pmndrsgiradiance-cache) is listed at the end. Which of the two techniques to use, and how they stack, is in [the guide](../guides/ddgi-or-radiance-cache.md). The working example is [`examples/12-radiance-cache`](../../examples/12-radiance-cache/).

## Getting started with three.js

The cache runs on a platform you make yourself (`createGI` from `/core`). three.js reaches it through `/three`, the same pieces as for DDGI:

- `ThreeGI` mirrors the scene;
- `screenInputs` hands the scene pass's depth, normals and motion to the cache;
- `ddgi()` adds the result once per pixel.

```ts
import * as THREE from 'three/webgpu';
import { mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import { createGI } from '@pmndrs/gi/core';
import { RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';
import { createGIRenderer, ddgi, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT, screenInputs, ThreeGI } from '@pmndrs/gi/three';

// A device with the platform's limits; no multisampling (the screen inputs need a single-sampled pass).
const { renderer, device } = await createGIRenderer({ renderer: { antialias: false } });

const gi = createGI(device);
const cache = gi.addTechnique(new RadianceCacheTechnique());
gi.setOutput(cache);
const preparing = gi.prepare();
const threeGI = new ThreeGI(gi, scene, { preparing }); // copies the scene in while the kernels compile
await preparing;

const pipeline = new THREE.RenderPipeline(renderer);
const scenePass = pass(scene, camera);
const screen = screenInputs(scenePass, threeGI);
scenePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), ...screen.outputs }));
pipeline.outputNode = screen.encodeBefore(ddgi(scenePass, threeGI));

renderer.setAnimationLoop(() => {
    threeGI.update(camera); // the frame's GI work
    pipeline.render(); // the scene pass, then the cache's screen stage, then the composite
});
```

What each step does:

- **`createGIRenderer`** makes three's renderer on a device with the adapter's full limits. three's own device (`new WebGPURenderer()`) allows 8 storage buffers per shader stage and a 128 MiB storage binding. The cache needs 10 and about 1 GiB, so it can't run there (see "What it needs from the device").
- **`ThreeGI`** mirrors the scene into the platform, as for DDGI ([three.md, "`ThreeGI`"](three.md#threegi)). The cache ignores the frame budget: all of its work runs every frame.
- **`screenInputs`** adds the outputs the cache reads to your MRT pass: a world-space normal and three's `velocity`. Depth is the pass's own. `screen.encodeBefore(node)` runs the cache's screen stage once a frame, after the pass draws and before `node` reads the result ([three.md, "Screen inputs"](three.md#screen-inputs-screeninputsscenepass-threegi-options)).
- **`ddgi()`** is the composite. Despite its name it isn't DDGI-only: it adds `giAlbedo × E / π` from whichever technique is the platform's output. With the cache, E is read by screen position from the denoised irradiance.
- **`threeGI.update(camera)`** runs before the pipeline each frame. Alternatively, set `updateOnRender` on the `ThreeGI`.

Things to know:

- **Full resolution only, for now.** The cache hasn't been run under `@pmndrs/upscaler` yet. The screen stage and the composite would need the upscaler's jitter (`ThreeGI`'s `jitter` option) and its reduced pass.
- **Surfaces the composite doesn't take** (transparent ones, and Physical materials with clearcoat, sheen and the like; [three.md, "`ddgi()`"](three.md#ddgiscenepass-volume-options-and-gialbedo)) get GI from the GI light that `ThreeGI` adds. That light reads the cache's image at the fragment's screen position, so a transparent surface gets the GI of the opaque surface behind it.
- **Let the screen stage run before the pass first draws.** It sets the pass's MRT blend modes on its first run, and a pass drawn earlier keeps pipelines without them (three.md, "Screen inputs").

### Stacking DDGI under the cache

The world cache traces one bounce beyond the probes' hits by default (GI-1.2's second bounce). Stack a DDGI volume under the cache, and each of those hits reads DDGI's irradiance instead of tracing a ray. That gives every later bounce and the light of emissive surfaces, and it costs the world cache about half its work. The setting is `secondBounce: 'auto'`, the default: it uses DDGI when one is stacked and GI-1.2's own rays otherwise. Against Cycles, the closed test rooms go from 22–56% too dark to within −1.3% and +5.2% ([the guide](../guides/ddgi-or-radiance-cache.md) has the numbers).

```ts
import { createGI } from '@pmndrs/gi/core';
import { DDGITechnique } from '@pmndrs/gi/ddgi';
import { RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';
import { ThreeGI } from '@pmndrs/gi/three';

const gi = createGI(device);
const cache = gi.addTechnique(new RadianceCacheTechnique());
gi.setOutput(cache);
const under = gi.addTechnique(new DDGITechnique({ volume: { center: [0, 2.5, 0], size: [6, 5, 6], spacing: 1 } }));
gi.setFallback(cache, under); // before prepare: the cache builds its DDGI lookup then
const preparing = gi.prepare();
const threeGI = new ThreeGI(gi, scene, { preparing, budget: { units: 256 } }); // DDGI's probes a frame
await preparing;

// DDGI's own warm-up first, so the cache's later bounces don't start from a black field.
while (under.warmingUp) {
    threeGI.update(camera, 0);
    await device.queue.onSubmittedWorkDone();
}
```

Then build the pipeline as above.

- **Call `setFallback` before `prepare`.** The cache compiles its DDGI lookup at `prepare`. A fallback set later is picked up too, but the cache traces rays until the lookup is built.
- **The budget is DDGI's.** A `units` budget gives DDGI that many probe updates a frame; the cache ignores it. A millisecond budget (`gpuMs`) is split evenly between the two techniques today, so half of it goes nowhere.
- **Outside the volume** DDGI's coverage is 0, so a hit there gets no later bounce rather than a wrong one. Make the volume cover what the camera sees. A volume that follows the camera hasn't been tried under the cache yet.
- **The sky is counted once.** With DDGI under it, the cache leaves the sky out of its own light choice, because DDGI's field holds it.
- **`neuralCache: 'tail'` turns off** with a warning when DDGI supplies the second bounce: DDGI's field already carries the later bounces.
- **Memory:** DDGI adds its own, 55 MB on Sponza with 19,425 probes.

## Without three.js: `/core`

The cache is engine-free. On your own `GPUDevice`, it is a technique like DDGI ([platform.md](platform.md), [the adapter guide](../guides/adapter.md)):

```ts
import { createGI, requiredLimits } from '@pmndrs/gi/core';
import { RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';

const device = await adapter.requestDevice({ requiredLimits: requiredLimits(adapter) });
const gi = createGI(device);
const cache = gi.addTechnique(new RadianceCacheTechnique());
gi.setOutput(cache);
await gi.prepare(); // throws GILimitError naming the limit on a device that is too small
// ...describe the scene through gi.scene, as in the adapter guide...

// Every frame: the world-space work, your depth and normal pass, then the screen stage.
gi.encode(encoder, { deltaTime, budget: { units: 0 }, camera: frameCamera });
// ...draw depth, world-space normals and motion into your own textures...
gi.encodeScreen(encoder, { width, height, depth, normal, normalEncoding: 'snorm-float', velocity });
// ...then shade, reading gi.shading (the cache answers by screen position)...
```

- **`gi.screenNeeds`** says which screen channels to draw. The cache always reads `depth`, `normal` and `velocity` (motion in NDC, current minus previous). With `ssgi: true` it also reads `albedo` (base colour × (1 − metalness)) and `lighting` (your lit colour before GI). The textures must be the screen's size and single-sampled.
- **`camera`** is the unjittered camera. If your depth is drawn with a jittered projection, pass the jitter as `camera.jitter` in NDC; the cache rebuilds positions from depth with it.
- **Reading the result.** `gi.shading` is the cache's shading module, the same shape every technique publishes (adapter guide, step 9). Its `giIrradiance` reads the denoised irradiance (the raw one with `denoiser: false`) at `screenUV` and ignores position and normal, so it answers only for the pixel being shaded on this screen. Its coverage is 1 everywhere. You can also read `cache.probes.denoisedIrradiance` (an `rgba16float` texture, E in rgb) in a full-screen pass of your own: that is what three's composite does.

## What it needs from the device

| Need | Why | Without it |
| --- | --- | --- |
| `maxStorageBuffersPerShaderStage` ≥ 10 | The largest passes bind 10 storage buffers, the platform's 5 included (pmndrs/gi#49). Chrome on Apple GPUs allows exactly 10; the RTX 3060 and Intel UHD allow 16; three's own device gets WebGPU's default, 8. Declared in the technique's `requirements.limits`, so `gi.limitNeeds` lists it. | `prepare` throws `GILimitError` naming the limit. |
| A storage binding and buffer far above 128 MiB | The world cache's hash grid is about 1 GiB at Capsaicin's capacity, in two buffers. | `prepare` throws `GILimitError` naming `maxStorageBufferBindingSize`. |
| `maxColorAttachmentBytesPerSample` above 32 | three's MRT pass with the composite's outputs and the screen inputs needs up to 56 bytes a pixel. | The screen stage throws `GILimitError` at its first run. |
| `shader-f16` and the adapter's `maxComputeWorkgroupStorageSize` | Only for `neuralCache: 'tail'`. f32 and smaller tiles are the fallback. | The tail runs in f32, or turns off with a warning if its network doesn't fit. |

`requiredLimits(adapter)` and `optionalFeatures(adapter)` from `/core` ask for all of these, and `createGIRenderer` / `requestGIDevice` from `/three` pass them on.

From the host it asks for rays through the platform's tracer (the scene copy every technique shares) and, every frame, the screen channels above.

## Memory

At the default settings the cache holds about **1.1 GiB** of GPU memory at 1280 × 720 (1,124 MiB, of which the world cache is 1,053 MiB). That is Capsaicin's own hash-grid capacity (2^14 buckets × 16 tiles × 85 cells), kept on purpose. The screen probes' share grows with the screen; the world cache's doesn't.

- `cache.memoryBytes` gives the technique's bytes, and `gi.stats.memory` the platform's total.
- The world cache's capacity comes from `hashGridCacheNumBuckets`, `hashGridCacheNumTilesPerBucket` and `hashGridCacheTileCellRatio`, read at `prepare`. Only Capsaicin's values are checked.
- `worldCache: 'sharc'` uses 56 MiB instead (below).
- Each cell's last-reset frame, 32 MiB, is made when `cellChangeReset` (on by default) or `cellShading: 'budgeted'` is first used. The budgeted lists add 5.5 MB at 1280 × 720.
- `neuralCache: 'tail'` adds 26 MiB at 1280 × 720.

## Settings

`new RadianceCacheTechnique(settings)` takes any of the settings below; the rest keep their defaults (`DEFAULT_RADIANCE_CACHE_SETTINGS`). `cache.settings` is the live object:

- Almost every setting is **read every frame**: change it on `cache.settings` and the next frame uses it.
- Some are **read at `prepare`**: `worldCache`, `ssgi`, `neuralCache` and `neural`, the hash grid's capacity (`hashGridCacheNumBuckets`, `hashGridCacheNumTilesPerBucket`, `hashGridCacheTileCellRatio`) and three of SHaRC's keys. To change one, make the technique again (the viewer does).
- `ssgi` is also fixed at construction, because it decides which screen channels the technique asks for.

```ts
import { RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';

const cache = new RadianceCacheTechnique({ denoiserHistoryClip: 3 });
cache.settings.cellShading = 'budgeted'; // live, from the next frame
```

Costs below are measured on the RTX 3060 laptop at 1280 × 720, from the design doc and its register rows. Its timings come from runs the bench runner marked thermally limited, so read them as relative.

### The defaults that matter

| Setting | Default | What it does, and what it costs |
| --- | --- | --- |
| `secondBounce` | `'auto'` | Where the world cache's second bounce comes from. `'auto'`: from a technique stacked under the cache (`gi.setFallback(cache, ddgi)`) whose shading module answers anywhere in the world, else GI-1.2's own rays. `'gi1-rays'`: always the rays (Capsaicin's way). `'fallback'`: always the stacked technique, with a warning when there is none. With DDGI: every later bounce, about half the world cache's work, faster reaction (register E10). Changing what runs empties the world cache. |
| `useMultibounce` | `true` | GI-1.2's second bounce. Off: one bounce beyond the probes' hits. The second-bounce buffers (535 MB) stay allocated either way. |
| `denoiser` | `true` | The image-space GI denoiser (`Gi1Denoiser` from `denoiser`, below). Off shows the raw interpolation, which flickers. |
| `worldCache` | `'gi1-hash-grid'` | The world cache's backend. `'sharc'` is a variant (below). Read at `prepare`. |
| `cellShading` | `'per-hit'` | How cells are lit. `'per-hit'`: every hit draws a light and traces a shadow ray (GI-1.2). `'budgeted'` is a variant (below). |
| `neuralCache` | `'off'` | `'tail'` is a variant (below). Read at `prepare`. |
| `ssgi` | `false` | Capsaicin's near-field screen-space GI and bent-normal occlusion. Off because its noise outweighed what it added (Sponza changed 21–27% a frame with one slice). On, the cache also asks for `albedo` and `lighting`, six more screen-sized textures, and `ssgiHistoryFrames` of lag after an occluder moves. Read at `prepare` and fixed at construction. |
| `cellChangeReset` | `true` | Where the scene changed, a hit's cell drops its 16-sample history once per change, so a doubled sun is followed in 2 frames instead of 37. Costs a short noise rise after a change (Sponza 3.40% → 4.08% frame to frame) and 32 MiB for each cell's last-reset frame. |
| `environmentLightEnable` | `true` | The sky is one of the lights a cache hit samples, as in Capsaicin. Off: cache hits get no sky light; the sky reaches the probes only through rays that miss. With DDGI under the cache the sky is left out of the cache's lights either way, because DDGI's field holds it. |

### Screen probes

| Setting | Default | What it does, and what it costs |
| --- | --- | --- |
| `interpolationJitter` | `true` | Each pixel's probe lookup is jittered by up to a spawn tile, for the denoiser to average away. Off only to measure it. |
| `probeFilter` | `true` | Each probe's cells averaged with up to three neighbours each way before the SH projection (Capsaicin always runs it). |
| `probeFilterHistory` | `false` | `true` keeps the filtered probes as next frame's history (Capsaicin's), which compounds the blur under the running mean (Cornell's mean E +3.6%). |
| `probeBlend` | `'running-mean'` | How a probe cell carries its past: a running mean over up to `historySampleCap` frames. `'capsaicin-hysteresis'` is Capsaicin's, which reads dark on noisy cells. Changing it starts the probes' history over. |
| `historySampleCap` | `8` | Frames a cell's running mean counts. Higher is steadier and slower to follow a change; 1 makes each frame stand alone. |
| `emptyCellFill` | `'keep-previous'` | A cell no ray reached keeps its value. `'capsaicin'` refills it with Capsaicin's divided backup, the cause of GI-1.2's drift dark over time. Changing it starts the history over. |
| `backupFillsAsHistory` | `false` | `true` keeps those fills as history (Capsaicin's). Changing it starts the history over. |
| `rayGuidingUniformShare` | `0.125` | The share of each probe's rays that sample their own cell instead of being drawn by the probe's light (ray guiding). Without it a cell that went dark is never sampled again (0 is Capsaicin's rule). |
| `rayGuidingCorrection` | `true` | Weights each cell's step by the rays it got over the rays it expected, so ray guiding doesn't bias the running mean dark. |
| `probeChangeReset` | `true` | Where the scene changed, a cell whose ray crosses the change forgets its count, so its next ray replaces it. |
| `probeHistory` | `true` | `false` drops the probes' history before every frame: for comparisons, not a product setting. |

The screen probes' size, rays and spawn rate are fixed: `PROBE_SIZE` (8 × 8 cells), `PROBE_CELLS` (64 rays), `PROBE_SPAWN_TILE_SIZE` (one probe per 16 × 16 pixels a frame), `SAMPLING_MODE` (`'QuarterSpp'`, a quarter of a ray per pixel). `screenProbeLayout(width, height)` gives the counts for a screen.

### World cache (GI-1.2's hash grid)

| Setting | Default | What it does, and what it costs |
| --- | --- | --- |
| `hashGridCacheCellSize` | `32` | A cell's size in projected pixels. Smaller cells follow detail better and need more of them. |
| `hashGridCacheMinCellSize` | `0.1` | The smallest cell, in scene units (metres). |
| `hashGridCacheTileCellRatio` | `8` | Cells across a tile (8 × 8). Read at `prepare`. |
| `hashGridCacheNumBuckets` | `14` | log2 of the bucket count (16,384). With the next one, sets the capacity and so the memory. Read at `prepare`; only Capsaicin's values are checked. |
| `hashGridCacheNumTilesPerBucket` | `4` | log2 of the tiles per bucket (16). Read at `prepare`. |
| `hashGridCacheMaxSampleCount` | `16` | A cell's temporal sample cap. Higher is steadier and slower to follow light changes. |
| `hashGridCacheDiscardMultibounceRayProbability` | `0.7` | The share of first-bounce hits that trace no second-bounce ray. |
| `hashGridCacheMaxMultibounceSampleCount` | `16` | The second bounce's sample cap with GI-1.2's rays. |
| `fallbackIndirectSampleCount` | `4` | The second bounce's sample cap when it comes from a stacked technique. Its value carries no ray noise, so a lower cap follows changes sooner. |
| `cellChangeResetHistory` | `0` | With `cellChangeReset`: samples of a changed cell's history kept beside the new ones (0 to 15). 3 trades a little lag for less noise. |
| `useDirectLighting` | `true` | Probe rays that miss read the sky, and rays that hit an emissive surface take its emission. |
| `useTemporalFeedback` | `false` | Capsaicin's option to feed last frame's lit image back at hits. Not built: nothing reads it. |
| `useResampling` | `false` | Capsaicin's world-space ReSTIR of light samples. Not built: nothing reads it. |
| `reservoirCacheCellSize` | `16` | ReSTIR's cell size, with `useResampling`. Nothing reads it. |
| `disableAlphaTesting` | `false` | Capsaicin's option. Nothing reads it: the platform's tracer always honours cut-outs. |

### Denoiser

| Setting | Default | What it does, and what it costs |
| --- | --- | --- |
| `denoiserHistoryClamp` | `'neighbourhood'` | Clips the long history to a short one (about 4 frames) ± `denoiserHistoryClip` standard errors, so a lighting change shows within a few frames. `'none'` is Capsaicin's unclipped history: 45–160 frames to follow a step change. |
| `denoiserHistoryClip` | `2` | The clamp's half width. Smaller follows changes sooner and flickers more. |
| `denoiserChangeReset` | `true` | A short history for 8 frames where the scene changed. |

### SSGI (with `ssgi: true`)

| Setting | Default | What it does |
| --- | --- | --- |
| `ssgiSliceCount` | `1` | Slices a pixel traces a frame. |
| `ssgiStepCount` | `2` | Steps along each slice. |
| `ssgiViewRadius` | `0.2` | The search radius, in scene units. |
| `ssgiFalloffRange` | `0.05` | The falloff, as a share of the radius. |
| `ssgiHistoryFrames` | `16` | Frames the slice sums are averaged over. More is steadier; an occluder that moves takes up to this many frames to show. |

### Variants

Three backends and levers sit beside GI-1.2's, off by default. Each is measured against it, and none is the default until they have been compared (see "Variants being compared" below).

| Setting | Default | What it does, and what it costs |
| --- | --- | --- |
| `worldCache: 'sharc'` | | A SHaRC-style hash cache, ported from pmndrs/pathtracer, in the world-cache slot. It follows each probe hit with short update paths, so its cells carry every bounce without DDGI. 56 MiB instead of about 1 GiB. A quarter of the probes update a frame; its update paths took 7.8 ms on Sponza, where GI-1.2's whole world cache took 12.9 ms. |
| `sharc` | `{}` | SHaRC's settings over its defaults: `log2Entries` 20 (1M slots), `minCell` 0.02, `cellPixels` 16, `normalBins` true, `historyFrames` 32, `staleFrames` 64, `fallbackLevels` 3, `pyramid` true, `minSamples` 16, `bounceRays` 2, `cacheTail` true, `updateFraction` 0.25, `changeReset` true, `changeKeepSamples` 4. `log2Entries`, `fallbackLevels` and `pyramid` are read at `prepare`. |
| `cellShading: 'budgeted'` | | id Tech 8's "shade cells, not hits": hits only find their cells, and at most `cellShadingBudget` cells a frame are lit, by priority (new or changed first). Fewer shadow rays; each cell is noisier and older. World-cache time on Sponza 12.3 ms per hit, 11.8 ms at 20,000 cells. |
| `cellShadingBudget` | `20000` | Cells lit a frame with `'budgeted'`. More reacts faster: a doubled sun takes 24 / 8 / 6 frames at 5k / 20k / 80k. |
| `neuralCache: 'tail'` | | A small neural network (pmndrs/pathtracer's `@pathtracer/mlp`) answers the end of GI-1.2's second bounce, so the cells carry every bounce. Runs only with the GI-1.2 grid and its own second-bounce rays. About 2.8 ms on Sponza and 26 MiB. Needs the optional package (below). |
| `neural` | `{}` | The tail's settings over its defaults: `network` `'compact'` (64 → 3 × 32 → 3), `precision` `'auto'`, `positionBands` 12, `batch` 16384, `stepsPerFrame` 2, `warmupStepsPerFrame` 16, `warmupSteps` 256, `cellRecords` 16384, `screenStride` 8, `screenRecords` 16384, `changeBoostFrames` 16, `learningRate` 0.002, `ema` `'off'`, `seed` 1, `bounds` null, `boundsHalfSize` 64. Read at `prepare`. |

### The denoiser and its package

The denoiser is `Gi1Denoiser` from the sibling library [`denoiser`](https://www.npmjs.com/package/denoiser) (pmndrs/denoiser, imported from `denoiser/ffx`), a port of AMD's MIT code. `@pmndrs/gi` depends on `denoiser@2.1.0`, pinned, so it installs with the package. You never make or pass one: the cache makes its own. It reads the cache's irradiance and the screen's depth, normals and motion, and the composite reads its output.

### The neural tail and `@pathtracer/mlp`

`neuralCache: 'tail'` loads `@pathtracer/mlp` at run time with a dynamic `import()`. That package isn't on npm yet (pmndrs/pathtracer#65) and isn't a dependency of `@pmndrs/gi`. Where it can't be loaded, the tail turns off with one warning and the cache runs as GI-1.2: nothing fails, and your bundler doesn't try to resolve it. In this repository, `node tools/linked-siblings.ts --fetch` clones it next to the checkout at its pinned commit.

It also turns off, with a warning, with `worldCache: 'sharc'`, with the second bounce from a stacked technique, or when its network doesn't fit the device's workgroup memory.

## Variants being compared

The SHaRC cache, budgeted cells, the neural tail and DDGI underneath are first-class alternatives to GI-1.2's own world cache. The default will be picked after a side-by-side comparison of accuracy against Cycles, lag and ghosting after changes, flicker, memory, GPU time and what each needs from the host (TASKS P6.5; DECISIONS 2026-11-23). Until then the defaults stay as above: the GI-1.2 hash grid, with DDGI's bounces when a volume is stacked.

## Reading results and debug views

| Member | What it is |
| --- | --- |
| `cache.debugViews` | The technique's three screen debug views: the probe atlas, the probe mask and the probes' SH. A host draws them through the shading module's `giDebugColor` by id (`RADIANCE_CACHE_DEBUG_VIEW`). |
| `cache.encodeDebugView(encoder, view, target, exposure?)` | Draws a debug view into your own `rgba8unorm` storage texture, engine-free. |
| `cache.probes` | The latest screen's GPU resources (`RadianceCacheProbes`): the layout, the probe atlas and mask, the SH, the irradiance and `denoisedIrradiance`, and more. Null before the first screen stage. For tests, debug views and your own composites. |
| `cache.counters` | What the CPU knows without a readback: the spawn tiles, the most probe rays, the world cache's capacity and memory. |
| `cache.memoryBytes` | GPU bytes the technique holds. |
| `cache.resetHistory()` | Forgets the world cache and the probes' history at the next frame (a scene swap). |
| `cache.resetScreenHistory()` | Forgets the probes' history only (a camera cut). The world cache keeps its cells. |
| `cache.worldCache`, `cache.worldCacheBackend`, `cache.neuralTail` | The world cache (GI-1.2's grid, or whichever backend fills the slot) and the neural tail once prepared. For tests and pages; their types aren't exported. |

**GPU time:** with `gpuTiming`, `gi.stats.passes` doesn't hold the cache's passes today. The platform resolves its timestamps at the end of `encode`, and the cache's work all runs later, in `encodeScreen`. Time it with the browser's GPU profiler for now.

## Known limits

- **Indirect diffuse only.** No glossy reflections (GI-1.1's need roughness and metalness in the platform's materials), and hits are Lambert.
- **Textures at hits** are each texture's mean colour, as for DDGI's rays. The pixel itself uses its real texture through `giAlbedo`.
- **Emissive surfaces aren't lights** for the cache. Their light reaches a probe only when its ray hits them; it doesn't reach the cells near them. DDGI underneath fills this in.
- **Not under the upscaler yet**, and full resolution only.
- **About 1.1 GiB of GPU memory** at the default capacity, whatever the scene's size. Integrated and mobile GPUs may not have it; the Intel UHD runs it for correctness only.
- **GPU time is not tuned yet.** The world cache alone took about 12 ms on Sponza at 1280 × 720 (RTX 3060 laptop, thermally limited runs); native Capsaicin's whole diffuse GI takes 7.95 ms there with hardware ray tracing.
- **A degenerate camera throws.** A canvas of 0 × 0 pixels (a hidden iframe) gives three a camera whose projection can't be inverted, and the screen stage throws inside three's render.
- **Moving objects:** cells and probes near a change forget their history (`cellChangeReset`, `probeChangeReset`, `denoiserChangeReset`), but no cache entry moves with its object yet.
- **three.js r187 only**, through `screenInputs`, which reads one field of three's WebGPU backend to reach its render targets.
- **Not checked on the Apple M5 Pro in this form yet**, beyond the storage-buffer fit (#49).

## Every export of `@pmndrs/gi/radiance-cache`

Every name `@pmndrs/gi/radiance-cache` exports. A unit test (`tools/docs/docs-cover-exports.test.ts`) fails when the package exports a name this section doesn't mention. Its three.js pieces (`screenInputs`, `GILight`, `ddgi()`, `giAlbedo`, `GI_ALBEDO_OUTPUT`, `GI_NORMAL_OUTPUT`) are `@pmndrs/gi/three`'s, and the platform (`createGI`, `requiredLimits`) is `@pmndrs/gi/core`'s.

### The technique

| Name | What it is |
| --- | --- |
| `RadianceCacheTechnique` | The technique: `new RadianceCacheTechnique(settings?)`, then `gi.addTechnique` and `gi.setOutput`. Members above. |
| `RadianceCacheProbes` | What `cache.probes` returns. |
| `RadianceCacheShadingModule` | Its shading module (`cache.shading`): reads the irradiance by `screenUV`; coverage 1. |
| `RADIANCE_CACHE_DEBUG_VIEW`, `RadianceCacheDebugViewId` | The debug views' ids: `probeAtlas` 0, `probeMask` 1, `sphericalHarmonics` 2. |

### Settings

| Name | What it is |
| --- | --- |
| `RadianceCacheSettings` | Every setting ("Settings" above). |
| `DEFAULT_RADIANCE_CACHE_SETTINGS` | Their defaults, frozen. |
| `WorldCacheKind` | `'gi1-hash-grid' \| 'sharc'`. |
| `SCREEN_HISTORY_SETTINGS` | The settings whose change starts the probes' history over: `probeBlend`, `emptyCellFill`, `backupFillsAsHistory`. |
| `PROBE_SIZE`, `PROBE_CELLS`, `PROBE_SPAWN_TILE_SIZE`, `SAMPLING_MODE` | The screen probes' fixed sizes: 8 × 8 cells, 64 rays, one probe per 16 × 16 pixels a frame, `'QuarterSpp'`. |
| `screenProbeLayout(width, height)`, `ScreenProbeLayout` | The probe counts, atlas size and most rays for a screen. |

### Compare toggles

Every change from Capsaicin's behaviour has a setting whose other value brings Capsaicin's way back. These list them, for a panel that switches them live (the viewer's "Radiance cache · compare" section; design doc, "Compare toggles").

| Name | What it is |
| --- | --- |
| `RADIANCE_CACHE_COMPARE_TOGGLES`, `CompareToggle` | The list: each toggle's setting, group, label, link name, Capsaicin's value and register row. |
| `COMPARE_GROUPS`, `COMPARE_GROUP_LABELS`, `CompareGroup` | The groups: screen probes, world cache, denoiser, experiments. |
| `compareToggle(setting)`, `compareTogglesIn(group)` | One toggle by setting, or a group's toggles. |
| `CompareValues`, `capsaicinCompareValues()` | A set of toggle values, and Capsaicin's for every toggle. |
| `applyCompareValues(settings, values)` | Puts values into a technique's live settings. |
| `readCompareValues(params)`, `writeCompareValues(values, params)`, `COMPARE_URL_NAMES`, `compareValuesFromText(text)` | The values to and from a page's URL parameters, and from text. |
| `HOST_COMPARE_DEFAULTS`, `hostCompareValue(values, setting)` | The host's own choices in a compare set (whether to stack DDGI), which never go into the technique's settings. |
