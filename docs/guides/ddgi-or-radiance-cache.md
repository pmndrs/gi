# DDGI or the radiance cache

`@pmndrs/gi` has two GI techniques on one platform. They can run alone, or stacked, with DDGI under the cache. This page says how they differ and when to use which. The numbers are the ones the docs already hold, with their sources.

## How they differ

| | DDGI (`@pmndrs/gi/ddgi`) | Radiance cache (`@pmndrs/gi/radiance-cache`) |
| --- | --- | --- |
| **Where light is kept** | A grid of probes in a box you place (`DDGIVolume`). | Probes on the surfaces the camera sees, one per 16 × 16 pixels, over a world-space hash grid of cells sized by distance from the camera. No volume. |
| **Detail** | The probe spacing (1 m in the examples). Each probe knows how far it sees, so light leaks little through walls. | Follows the screen: finer near the camera. |
| **What it needs each frame** | The camera. | Depth, normals and motion from your scene pass, at full resolution. |
| **three.js set-up** | `new DDGIVolume(…)` and `start`, like `LightProbeGrid`. GI per fragment, or once per pixel with `ddgi()`. | A platform you make (`createGI`), `ThreeGI`, `screenInputs` and `ddgi()` on three's `RenderPipeline` ([radiance-cache.md](../api/radiance-cache.md)). |
| **GPU work** | Inside a budget you set, 2 ms by default. Showing it through the composite: 0.41 ms on Sponza at 1440p under the upscaler ([three.md](../api/three.md)). | All of it every frame; no budget. The world cache alone took about 12 ms on Sponza at 1280 × 720, RTX 3060 laptop (thermally limited runs, so relative only; [design doc, "Lever E8"](../design/radiance-cache.md#lever-e8-shade-cells-not-hits-p62)). |
| **Memory** | Grows with the probes: 55 MB on Sponza with 19,425 probes. | About 1.1 GiB at the default capacity, whatever the scene. |
| **Device** | Runs on three's own device for small scenes; large ones need `createGIRenderer`. Presets down to `low` for integrated GPUs. | Needs `createGIRenderer` (or `requiredLimits`): 10 storage buffers per stage and a binding near 1 GiB. |
| **With `@pmndrs/upscaler`** | Yes: the intended pipeline for dense scenes. | Not yet. |
| **Saved bakes, `LightProbeGrid` export** | Yes. | No. |
| **Against Cycles** | Cornell +0.2%; Sponza +23% without GTAO, about +6% with it at half resolution ([three.md](../api/three.md)). | Alone, with GI-1.2's own second bounce: the closed test rooms (two-room, both Cornells) 22–56% too dark. With DDGI under it: within −1.3% and +5.2%; Sponza +15% to +21% ([design doc, "Experiment E10"](../design/radiance-cache.md#experiment-e10-ddgi-supplies-the-second-bounce-p63)). |

Both give indirect diffuse light only; three.js draws direct light, emission and reflections.

## Which to use

- **Start with DDGI.** It has the shortest set-up and runs on the widest range of GPUs. It works per fragment on any material, follows lights, objects and the sun, and has saved bakes and a static `LightProbeGrid` fallback.
- **Try the radiance cache** when a fixed probe grid is too coarse for what the camera sees, or no volume fits the scene well, and the device has the memory. Expect it to be heavier and less tuned than DDGI today; its variants are still being compared ([radiance-cache.md, "Variants being compared"](../api/radiance-cache.md#variants-being-compared)).
- **Use them together** when you use the cache: DDGI under it is how the cache gets its later bounces.

## Stacking DDGI under the cache

The cache stays the output. DDGI goes under it as its fallback:

1. `gi.addTechnique(cache)`, `gi.setOutput(cache)`;
2. `gi.addTechnique(ddgi)`, then `gi.setFallback(cache, ddgi)`, before `gi.prepare()`.

Each frame the platform runs DDGI first. Where a probe ray of the cache hits a surface, the world cache reads DDGI's irradiance there instead of tracing GI-1.2's second-bounce ray (`secondBounce: 'auto'`, the default). The code is in [radiance-cache.md, "Stacking DDGI under the cache"](../api/radiance-cache.md#stacking-ddgi-under-the-cache) and [`examples/12-radiance-cache`](../../examples/12-radiance-cache/).

What it changes:

- **Every bounce, not two.** DDGI's field carries its own bounces, so the cache's cells get them all. It also holds the sky and the emissive surfaces DDGI's rays see, which the cache's own lights lack.
- **Less work for the world cache.** No second-bounce ray is traced, so about half the world cache's work goes. On Sponza the live tiles drop from 49,356 to 18,566.
- **DDGI's own cost is added:** its probe updates (give it a `units` budget; the cache ignores budgets) and its memory.
- **Make the volume cover what the camera sees.** Outside it DDGI's coverage is 0, and a hit there gets no later bounce.
- **The neural tail turns off** (`neuralCache: 'tail'`), with a warning: DDGI already carries the later bounces.
