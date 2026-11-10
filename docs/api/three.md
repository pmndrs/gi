# The three.js adapter (`@pmndrs/ddgi/three`)

The adapter puts the GI platform's global illumination into a three.js r187 `WebGPURenderer` scene. It does three jobs:

- it mirrors the three.js scene into the platform;
- it runs one GI frame per rendered frame;
- it lights every lit material through one GI light, or, on three's post-processing pipeline, adds GI once per pixel (`ddgi()`).

The code is in [`packages/ddgi-three/src/`](../../packages/ddgi-three/src/). The platform underneath is described in [platform.md](platform.md). Working examples: [`examples/`](../../examples/) (`01-hello` and `02-composite` are the code below). [Every export](#every-export-of-pmndrsddgithree) of `@pmndrs/ddgi/three` is listed at the end. (`@pmndrs/ddgi` itself exports the same names.)

## Getting started

The same shape as three's `LightProbeGrid`, in as many lines:

```ts
import { GIVolume } from '@pmndrs/ddgi/three';

const volume = new GIVolume(6, 5, 6, 6, 6, 6);   // like new LightProbeGrid(6, 5, 6, 6, 6, 6)
volume.position.set(0, 2.5, 0);
scene.add(volume);
await volume.start(renderer, scene);              // like probes.bake(renderer, scene)

renderer.setAnimationLoop(() => renderer.render(scene, camera));   // GI updates itself
```

`start` uses three's own GPU device, warms the probes up, and from then on GI updates itself inside `renderer.render`. There is no per-frame call.

### Dense scenes: GI once per pixel, through the upscaler

Lit materials normally take GI per fragment. In scenes with many tiny triangles (Sponza's curtains), that runs many times per pixel. On three's post-processing pipeline, as three's SSGI and GTAO examples use it, `ddgi()` adds GI once per pixel instead.

**The intended pipeline puts [`@pmndrs/upscaler`](https://github.com/pmndrs/upscaler) after it** (DECISIONS 2026-11-06). The scene pass and the composite run at the upscaler's render resolution, and the upscaler brings the result to the display. This is its composable path (`upscale()`), because `upscaleScene()` makes its own scene pass and can't take the composite:

```ts
import * as THREE from 'three/webgpu';
import { mrt, normalView, output, packNormalToRGB, pass, rtt, velocity } from 'three/tsl';
import { QualityMode, getQualityModeRatio, upscale } from '@pmndrs/upscaler';
import { ddgi, giAlbedo } from '@pmndrs/ddgi/three';

const renderer = new THREE.WebGPURenderer({ antialias: false });   // the upscaler anti-aliases
// ... the scene, and await volume.start(renderer, scene) as above ...

const renderScale = 1 / getQualityModeRatio(QualityMode.Performance);
const pipeline = new THREE.RenderPipeline(renderer);
const scenePass = pass(scene, camera);
scenePass.setMRT(mrt({ output, giAlbedo, normal: packNormalToRGB(normalView), velocity }));
scenePass.setResolutionScale(renderScale);
const gi = rtt(ddgi(scenePass, volume), null, null, { type: THREE.HalfFloatType, resolutionScale: renderScale });
pipeline.outputNode = upscale(gi, scenePass.getTextureNode('depth'), scenePass.getTextureNode('velocity'), camera, { quality: QualityMode.Performance });

renderer.setAnimationLoop(() => pipeline.render());
```

- `velocity` in the MRT is three's own; the upscaler keeps it free of its jitter.
- `rtt(…, { type: HalfFloatType, resolutionScale })` makes the composite a texture at the render resolution. Without it, `upscale()` would materialise it at display size. Keep `type`: an options object without it gives an 8-bit target, which clips HDR.
- On the reduced pass, `ddgi()` runs three's GTAO on the GI term at half the render resolution by default (below).
- Without the upscaler, `pipeline.outputNode = ddgi(scenePass, volume)` on a full-resolution pass works too, with GTAO opt-in (below).

**What it costs** (Sponza with its curtains, 2560 × 1440 output, RTX 3060 laptop, GPU-bound, the field frozen; T1.14):

| Path | Showing GI | Sponza GI against Cycles (curtain pixels) |
| --- | --- | --- |
| Upscaler Performance, GTAO at half the render resolution (the default) | 0.73 ms | +6.6% (+15.7%) |
| Upscaler Quality, the same | 1.25 ms | +6.1% (+14.6%) |
| No upscaler, no occlusion | 1.3–1.4 ms | +23% (+44%) |
| No upscaler, GTAO at half resolution (`occlusion: { resolutionScale: 0.5 }`) | 3.2 ms | +5.8% (+13.6%) |
| No upscaler, GTAO at full resolution (`occlusion: {}`) | 8.0 ms | +4.4% (+10.7%; 640 × 360 reference) |
| GI per fragment (no pipeline) | 3.4 ms | as the composite without occlusion |

- **The upscaler's own cost is separate:** 1.55 ms per frame (Performance) or 1.72 ms (Quality), with or without GI. On Sponza at 1440p the upscaled frame without GI costs more than native (2.58 + 1.72 ms against 3.0 ms). You'd use it for the frame as a whole, not for GI.
- The Sponza accuracy figures use the HD Cycles reference at 1920 × 1080 (except where marked). The base package (no curtains) reads −1.2% against its 640 × 360 reference, the only one it has so far; Cornell reads +0.2% at 1920 × 1080.

- `giAlbedo` is the weight three's lighting gives indirect diffuse light, per pixel: the diffuse colour × (1 − F_ss − F_ms) × ambient occlusion for Standard and Physical materials, as r187's `PhysicalLightingModel` applies it.
- `ddgi()` adds `giAlbedo × E / π` to the pass's linear HDR colour before the pipeline's tone mapping. It matches the per-fragment image to 0.04–0.08% on average (p95 under 0.25%) on Cornell and Sponza, on both test GPUs.
- In that pass the GI light adds no indirect diffuse on the surfaces the composite lights, so nothing is counted twice. Other passes, shadow maps and cube captures keep GI per fragment.
- Call `volume.start` before the pipeline's first render.

## The pieces

### `GIVolume`

`GIVolume` has the same shape as three's `LightProbeGrid`, so moving a scene from one to the other is a small change:

- `new GIVolume(width, height, depth, nx, ny, nz)`;
- the box is centred on the object's position and aligned with the world axes;
- there is one probe at the centre of each cell;
- `boundingBox`, `priority` and `intensity` work as they do on `LightProbeGrid`. `intensity = 0` turns GI off.

**`await volume.start(renderer, scene, options?)`** does the set-up:

1. It takes three's device (`renderer.backend.device`) and initialises the renderer if needed.
2. It makes the platform and a DDGI technique on the volume's probe grid, and prepares them.
3. It adds the GI light to the scene (through a `ThreeGI`).
4. It warms the probes up. The first frames copy the scene and build the ray-tracing trees. Then up to 16 passes over every probe run with a running average, stopping early after 8 s (at least 2 passes). Then the budget is calibrated on the GPU.
5. It switches on the render-time update.

| Option | What it does |
| --- | --- |
| `preset` | A quality preset: `'low'`, `'medium'` or `'high'`, or a preset object from `ddgiPreset` with fields changed (below). Default none, which is the same as `'high'`. |
| `budget` | GPU work per frame for GI updates after the warm-up. Default the preset's, or `{ gpuMs: 2 }`. |
| `ddgi` | DDGI settings over the preset's and the defaults (rays per probe, hysteresis…). The probe grid comes from the volume. |
| `environment`, `sky` | GI's sky, as for `ThreeGI` below. |
| `light` | Settings for the GI light. |
| `warmUp` | `false` to skip it (GI then fills in live), or `{ passes, maxSeconds, minPasses, calibrate }`. |
| `camera` | The camera the warm-up runs with, read each frame; default one at the volume's centre. Pass the camera you render with when `ddgi.volumes` has a camera-following volume, so its window warms up where the camera is (otherwise it jumps there afterwards and warms up again). `restart` takes it too. |
| `bake` | A saved bake (below) to start from instead of warming up. |
| `bakeIfSceneChanged` | `'load stale'` (default) or `'refuse'`, for a bake saved before the scene changed. |
| `warn` | Where warnings go. |

**Quality presets** (T5.1; the table and what each costs is in [platform.md, "Quality presets"](platform.md#quality-presets)). One word sets the rays per probe, the probe tiles, the budget and the size of GI's copy of the textures. A volume made without probe counts also takes the preset's spacing (2 m for `low`); counts you give are kept. `budget` and `ddgi` still win over the preset:

```ts
import { ddgiPreset, recommendPreset } from '@pmndrs/ddgi/core';

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error('No WebGPU adapter');
await volume.start(renderer, scene, { preset: recommendPreset(adapter.info, adapter.limits) });

// Or start from a preset and change one field:
const preset = ddgiPreset('low');
preset.ddgi.raysPerProbe = 160;
await volume.start(renderer, scene, { preset });
```

`recommendPreset` is advice only: it gives `low` for integrated and mobile GPUs and `high` otherwise, and nothing applies it unless you pass it. Changing presets later means `dispose()` and `start` again: rays and tile sizes size DDGI's buffers when it starts.

If the scene needs more than the renderer's device allows, `start` throws a `GILimitError` (`ThreeGILimitError`) naming the limit. three makes its device with default limits (a 128 MiB storage binding), which large scenes such as Sponza with its curtains outgrow. The fix is to make the renderer with the platform's limits: `const { renderer } = await createGIRenderer()`, or `new WebGPURenderer({ requiredLimits: requiredLimits(adapter) })` with `requiredLimits` from `@pmndrs/ddgi/core`.

Other members:

- `volume.recalibrate()` measures again how much probe work fits the budget, over the next frames. Use it after a big scene change.
- `await volume.restart({ budget, warmUp, camera })` starts the probes over: it forgets the field and the calibration and runs the warm-up again, for after the scene changed a lot. `volume.isStarted` says whether `start` has finished.
- `volume.threeGI` (budget, `markChanged`, timings) and `volume.platform` (stats, debug views) are there after `start`.
- `volume.light` is the GI light (`ddgi()` reads it).
- `await volume.exportLightProbeGrid()` gives a snapshot of the GI as three's `LightProbeGrid` ("Exporting a `LightProbeGrid`" below).
- `volume.dispose()` removes the GI light and frees the platform.

**Saved bakes** (T5.2). Save what the probes have learned and start from it later, so a page opens with finished GI and keeps updating from there:

```ts
const bytes = await volume.saveBake();          // a .ddgibake file's bytes (one GPU readback, here only)
// ...later, on a fresh page with the same scene and settings:
await volume.start(renderer, scene, { bake: bytes });   // no warm-up passes
// or into GI that is already running:
const loaded = volume.loadBake(bytes);          // { stale, changedParts, savedAt, library }
```

- The file is engine-neutral (ddgi-core's `DDGITechnique.saveField` / `loadField`, platform.md "Saved bakes"), so it doesn't depend on three.
- It loads only into a volume with the same layout: the same probe grid and place, tile sizes, light groups and sun layer, and camera-following volumes. Anything else throws `SavedFieldError` (from `@pmndrs/ddgi/core`) with a message naming what differs.
- The file records a fingerprint of the scene it was baked for. If the scene changed since (an object, a material, a texture, a light, the sky), it loads anyway as a starting point: `loaded.stale` is true, `loaded.changedParts` says which parts changed, and every probe updates from the loaded light. `{ ifSceneChanged: 'refuse' }` (`bakeIfSceneChanged` for `start`) throws instead.
- With `start(..., { bake })`, `volume.loadedBake` says what was loaded. `start` still waits for the ray-tracing trees and calibrates the budget; it only skips the warm-up passes.
- Images the CPU can't read cheaply (`ImageBitmap`, which `GLTFLoader` gives) count in the fingerprint by their size and colour space, not their pixels, so a repainted image of the same size isn't noticed.

`GIVolume.fromDDGIVolume(settings)` makes a volume whose probes are exactly a DDGI box and spacing (as the viewer's scene files describe volumes). `volume.updateBoundingBox()` sets `boundingBox` from its position and size, and `volume.useDefaultSpacing(metres)` is what a preset's spacing does to a volume made without probe counts.

`volume.ddgiVolume()` gives the matching DDGI settings, for building a platform by hand:

- the spacing is `width / nx` (and the same on the other axes);
- the box is one cell smaller than the volume, so the outermost probes sit half a cell inside it.

Moving or resizing the volume later needs new settings: `dispose()` and `start` again.

### The render-time update

After `start`, the GI light's node runs each GI frame from its own render-time update, as three's `LightProbeGridNode` updates itself. In r187 that update runs inside the render pass, after its command encoder exists and before it is submitted. GI's work is submitted from there, so the GPU runs it before the render that samples it (checked on the GPU: the image equals an explicit update-then-render exactly). What it means in practice:

- **Once per frame.** It runs at most once per JavaScript task (a `requestAnimationFrame` or `setAnimationLoop` callback), however many lit renders the frame draws.
- **The first lit render's camera.** If cube captures or a reflection pass render before the main view, GI's update order follows their camera that frame.
- **No lit render, no GI frame.** A frame that draws nothing lit by GI (all of it culled, or the light's layer hidden) does no GI work, so the probes don't converge meanwhile.
- **Errors** in the GI frame are reported once (`warn`) and stop the updates; three keeps rendering.

`threeGI.updateOnRender = false` turns it off. Call `threeGI.update(camera)` yourself before each render instead.

### `ddgi(scenePass, volume, options?)` and `giAlbedo`

The composite node for three's pipeline (see "Dense scenes" above). Built like three's `SSGINode` and `GTAONode`.

- The pass's MRT must write `giAlbedo` and `normal: packNormalToRGB(normalView)` (three's SSGI and GTAO convention). The node says so if it doesn't.
- It marks its pass through `scenePass.contextNode`, keeping any context you already set (GTAO's `builtinAOContext`, say). Set your own context before calling `ddgi()`.
- **Which surfaces it lights:** opaque surfaces with Standard, Physical, Lambert or Phong materials. Physical materials with clearcoat, sheen, iridescence, transmission, dispersion or rough diffuse keep GI per fragment, and so do transparent surfaces and unlit ones.
- **Transparent surfaces in front of composited ones.** The composited GI behind a transparent surface is evaluated at the transparent surface, because the pass's normal and depth are its own. Behind a 50%-opaque sphere in Cornell that reads about 19% too dark. Where that matters, draw transparent objects after the composite.
- **Short-range occlusion on GI.** `ddgi()` can run three's own GTAO (`ao()`) on the pass's depth and normal, with a 1 m radius, and multiply the GI term by it, and only the GI term. Probes 1 m apart can't follow light that changes inside small folds: Sponza's curtains read +23% against Cycles without it.
  - **The default follows the pass.** On a reduced pass (`scenePass.setResolutionScale` below 1, as under the upscaler), GTAO runs at half the pass's resolution. On a full-resolution pass there is no occlusion: GTAO is opt-in there, because it costs several times the composite (table above). The choice is made when the pipeline builds, so the pass's scale may be set before or after `ddgi()`.
  - `occlusion: { radius, resolutionScale, samples }` sets it. It needs 16 samples (three's default): with 8, Sponza reads +16%.
  - Its size follows the scene pass, not the canvas: a pass drawn smaller (`scenePass.setResolutionScale`, as under `@pmndrs/upscaler`) gets GTAO at its own size, and `resolutionScale: 0.5` is always half the pass's pixels.
  - `occlusion: false` turns it off.
  - `occlusion: { source: gtaoNode }` uses a GTAO node your pipeline already makes from this pass, instead of a second one.
  - If the pass already applies AO to its materials (`builtinAOContext`), `giAlbedo` already carries it, and the default is no second occlusion.
  - `node.occlusionNode` is the composite's own GTAO node, for live changes (`radius.value`). With the default it is made at the build.
- The fog of a fogged scene isn't applied to the composited GI.
- The targets should stay half float: 8-bit `giAlbedo` and normal miss the match by about 10× on Sponza.
- If the platform's shading module changes shape (another technique, say), set `pipeline.needsUpdate = true`.

### `createGIRenderer()` and `requestGIDevice()`

These make the GPU device and hand it to three (`new WebGPURenderer({ device })`). The host owns the device; the platform never makes one.

- The device gets every feature the adapter offers, as three asks for when it makes its own device. three needs `core-features-and-limits` to run in full (not compatibility) mode.
- It gets the platform's `requiredLimits`. Large scenes such as Sponza with the curtains need them.
- Pass `{ device }` to use a device you made yourself.

### `ThreeGI`

For hosts that want control (several techniques, a shared platform), `ThreeGI` is the scene's link to a platform you built yourself:

```ts
import { createGI, DDGITechnique } from '@pmndrs/ddgi/core';
import { createGIRenderer, GIVolume, ThreeGI } from '@pmndrs/ddgi/three';

const volume = new GIVolume(6, 5, 6, 6, 6, 6);
volume.position.set(0, 2.5, 0);
volume.updateMatrixWorld();
const { renderer, device } = await createGIRenderer();
const gi = createGI(device);
gi.setOutput(gi.addTechnique(new DDGITechnique({ volume: volume.ddgiVolume() })));
await gi.prepare();
const threeGI = new ThreeGI(gi, scene, { volume });

renderer.setAnimationLoop(() => {
    threeGI.update(camera);           // first: GI's work for this frame
    renderer.render(scene, camera);   // then three draws, reading GI
});
```

`new ThreeGI(gi, scene, options)` needs the platform's output technique and a prepared platform, or one still preparing with `prepare`'s promise passed as `preparing`; otherwise it throws. It adds the GI light to the scene once the platform is prepared.

With `preparing` it may be made while the platform still prepares (T2.14), so the scene is copied in while the kernels compile. Until `prepare` resolves, `update` only copies the scene (the scene sync's CPU work) and the light stays out of the scene, so a render meanwhile draws without GI. When it resolves the light joins the scene, so a host that only renders (`updateOnRender`) gets GI without calling `update`; if it rejects, the next `update` throws the error. Chrome sends WebGPU calls to its GPU process when the JavaScript task ends, so let the task end between `prepare()` and a long copy, or the compiles wait for the copy. `GIVolume.start` does this:

```ts
import { ThreeGI } from '@pmndrs/ddgi/three';

const preparing = gi.prepare();
await new Promise((resolve) => setTimeout(resolve, 0)); // the compile requests go out
const threeGI = new ThreeGI(gi, scene, { volume, preparing });
threeGI.update(camera, 0);                               // the scene copy, while compiling
await preparing;                                         // the light is in the scene; the next update runs GI
```

`update(camera, deltaTime?)` is called once per frame, before `renderer.render` (or by the GI light itself, with `updateOnRender`). Each call:

- sends the scene's changes;
- records the GI work for the frame;
- submits it straight away.

The extra submit is needed because three r187 records into its own command encoders and gives no way to add work to them. The queue runs submissions in order, so GI's work always finishes before three draws.

| Option | What it does |
| --- | --- |
| `budget` | GPU time per frame for GI updates, default `{ gpuMs: 2 }`. You can change it at any time. |
| `volume` | A `GIVolume`. The GI light follows its `intensity` and `priority`. |
| `light` | Settings for the GI light: `intensity`, `priority`, `replaceEnvironmentDiffuse`. |
| `environment` | A platform environment to use instead of the scene's (the viewer's sky files, say). |
| `sky` | A pmndrs/sky `Sky`, used as GI's sky. Its `exposure` becomes `sunIlluminance`. |
| `objectsPerFrame`, `editChecksPerFrame`, `lightsPerFrame` | How much of the scene is checked each frame (see "How changes are found"). |
| `updateMatrixWorld` | Update world matrices in every `update` (see below). Default: no. |
| `updateOnRender` | Run `update` from the GI light's render-time update (see above). Default: no. Can be switched later. |
| `warn` | Where warnings go. Each message is shown once. |

Other members:

- `threeGI.light` is the GI light. Set `light.intensity = 0` to turn GI off. Never remove the light: that recompiles every material.
- `threeGI.timings` gives the CPU time of the last update.
- `threeGI.report` lists what the last update sent.
- `threeGI.markChanged(object)` is described below.
- `threeGI.setGroupWeight(group, color, intensity = 1)` and `threeGI.setEnvironmentWeight(color, intensity = 1)` are described in "Light groups" below.
- `dispose()` removes what the adapter added.

### Light groups

Dimming or recolouring a whole group of lights in GI shows in the same frame, with no update lag and no noise (T2.5; docs/design/m2-lights.md §5). Put each light in a group with `userData.giLightGroup` (0 to 3), turn groups on in DDGI's settings, and set a weight:

```ts
await volume.start(renderer, scene, { ddgi: { lightGroups: { count: 2, weights: [[1, 1, 1], [1, 1, 1]], environmentWeight: [1, 1, 1] } } });
const threeGI = volume.threeGI!;           // there once start has finished
threeGI.setGroupWeight(1, 'white', 0.25);   // group 1 at a quarter
threeGI.setGroupWeight(0, '#ffcc88');       // group 0 warmer
threeGI.setEnvironmentWeight('white', 0.5); // the sky and emissive surfaces at half
```

- The colour is any three colour (`Color`, `'#ffcc88'`, `0xffcc88`), read in three's linear working colour space, times `intensity`, as a light's colour and intensity are.
- A weight multiplies the lights' own colour and intensity, in GI only: dim three's own lights too for the direct light. Changing a light's `intensity` still works as before: it re-traces the probes it reaches.
- Groups are off by default. Each group costs a layer of the field: on Sponza about 25 MB and 5% of each probe update (4 groups: +10%; two-room +23%), and `count` needs a new `prepare`.
- Lights in a group at or past `count` share the last group's layer.

### The sun layer

With DDGI's `sunLayer` setting, a change of the sun's colour or brightness shows in GI exactly, in the frame it happens, with nothing traced again (T2.6b; docs/design/m2-lights.md §6.1). Change the sun as usual:

```ts
await volume.start(renderer, scene, { ddgi: { sunLayer: true } });
sun.color.set('#ffb070');   // an evening sun: GI follows in the same frame
sun.intensity = 1.5;
```

- The sun is the scene's first `DirectionalLight` (the first the adapter met). DDGI traces it in a layer of its own at a white reference and weighs that layer by the light's colour × intensity.
- Turning the sun still re-traces, nearest the camera first, as before.
- Dim or hide it (`intensity` 0 or `visible` false is fine: GI sees a sun at 0), but don't remove it: a removed sun makes another directional light the sun, and every probe is traced again.
- Off by default. It costs a layer of the field (Sponza: 58 MB of GI becomes 93 MB) and its share of every probe update (PROGRESS, T2.6b). It works with light groups: the sun's layer is then also weighed by its group's weight. 4 groups, the environment and the sun are the most a volume holds.

### Time of day

Move the sun and the sky together; the adapter follows both every frame. The sky can be three's `SkyMesh` (the sun's brightness then comes from the brightest directional light) or a pmndrs/sky `Sky` (the `sky` option). With either, GI makes its own copy of the sky on the GPU again whenever the sun has turned more than 0.05°. A sky texture of your own is read again only when it changes.

```ts
import { SkyMesh } from 'three/addons/objects/SkyMesh.js';

const sky = new SkyMesh();
sky.scale.setScalar(1000);
scene.add(sky);
await volume.start(renderer, scene, { ddgi: { sunLayer: true } });

renderer.setAnimationLoop(() => {
    const angle = ((hours - 6) / 12) * Math.PI; // 6 am to 6 pm
    sun.position.set(Math.cos(angle), Math.sin(angle), 0.3).multiplyScalar(50); // a DirectionalLight shines from its position to its target
    sky.sunPosition.value.copy(sun.position).normalize();
    renderer.render(scene, camera);
});
```

- A turning sun moves the sunlit patches, so the probes are traced again, nearest the camera first. With the sun layer (above), a change of the sun's colour or brightness alone shows in the same frame.

### Several volumes, and a volume that follows the camera

`start`'s `ddgi.volumes` replaces the volume's single box with a list (up to 8). Where boxes overlap the higher `priority` answers. A box with `follow: 'camera'` is a window of probes that moves with the camera, for large or open scenes (platform.md, ["Several volumes"](platform.md#several-volumes-and-the-camera-following-volume)):

```ts
volume.updateMatrixWorld(); // ddgiVolume() reads the volume's world position
await volume.start(renderer, scene, {
    camera, // the camera you render with, so the window warms up where it is
    ddgi: {
        volumes: [
            { center: [0, 8, 0], size: [46.5, 16.5, 46.5], spacing: 1.5, follow: 'camera', priority: 1 },
            volume.ddgiVolume(), // the GIVolume's own box, behind the window
        ],
    },
});
```

- Pass `camera`, or the window warms up around the volume's centre and then jumps to the camera on the first frame, warming up again.
- The GI light still follows the `GIVolume`'s `intensity` and `priority`.
- `exportLightProbeGrid` exports one volume, the highest-priority one; with a camera window first, that is the window as it was at the export. Exporting a second volume or the camera window hasn't been checked yet.

### Moving, adding and removing objects

Nothing to call: change the scene as you always would, and GI follows (see "How changes are found" below for how, and how late). Two `userData` flags say what GI should do with an object:

```ts
door.rotation.y = Math.PI / 2; // found by the next render, sent by the next GI update
scene.add(lamp); // added objects reach GI in the next update
character.userData.giContributes = false; // lit by GI, but GI's rays pass through it (set before adding it)
scene.add(character);
flag.userData.giDeforming = true; // its position attribute is animated on the CPU: send each edit as a pose
volume.threeGI?.markChanged(door); // optional: an edit, a hide or a move outside three's render, seen at once
```

- An object that doesn't contribute to GI still wakes the probes in empty air around it as it moves, so a character crossing a room is lit (T3.1, T3.4).
- A moving object that glows (an emissive material) updates every probe its light reaches, also around corners (T3.4).
- Skinned and morphed meshes: "Deforming meshes" below.

### Exporting a `LightProbeGrid`

A snapshot of the GI as three's own `LightProbeGrid` (r187), for a static fallback where DDGI can't run (T5.3; docs/design/m5-coverage.md §3). It needs no `bake`: the grid is filled straight from our probes.

```ts
import { lightProbeGridFromJSON, lightProbeGridToJSON } from '@pmndrs/ddgi/three';

const grid = await volume.exportLightProbeGrid();   // one GPU projection, one readback
const text = JSON.stringify(lightProbeGridToJSON(grid));

// Later, or on another page, without DDGI:
const probes = lightProbeGridFromJSON(text);
scene.add(probes);
```

- The probes sit exactly where ours do. `LightProbeGrid` puts its probes at cell centres, so the grid is `counts × spacing` wide and centred on our grid's centre.
- Each probe holds 9 L2 spherical-harmonic coefficients per colour in three's own convention (the radiance SH `LightProbe` stores), so three's `getShIrradianceAt` gives back our irradiance. The core does the projection: `exportIrradianceSH(technique, { volume })` in `@pmndrs/ddgi/core` returns engine-free data (counts, origin, spacing, coefficients, probe states), and `toLightProbeGrid(data, { intensity, priority })` builds the grid from it. `volume.exportLightProbeGrid()` does both.
- **Light groups:** the export is the field you see, so with light groups or the sun layer on it is every layer times its weight as of the last rendered frame. A weight set since then shows after the next frame.
- Probes that live GI doesn't read (inside walls, in empty air, not updated yet) get the mean of their nearest active neighbours, because a grid with no visibility blends every probe. `exportIrradianceSH(..., { fillInactiveProbes: false })` keeps their raw values.
- To light with the grid instead of live GI, add it and set the GI's `intensity` to 0 (GI's priority, 1, beats the grid's 0 otherwise). Don't call `bake` on an exported grid: it would bake into a different texture than the one the grid shows.
- **What the fallback loses**, compared with live GI: detail above L2 (DDGI's 8 × 8 tiles hold more), visibility (no distances, so light leaks through walls between probes), and relocation (probes are placed on the lattice). That difference is the fallback's known cost and is recorded, not tuned: in the GI-only view, the point-light Cornell box's mean is within 1% of live GI's at every bookmark (per pixel 2-4%), and Sponza's is 8% darker at the colonnade and 4% at the upper gallery (per pixel 11% and 7%), on the M5 Pro (`three-lightprobegrid-export`, `three-lightprobegrid-export-sponza`).
- **WebGL:** `LightProbeGridWebGL` reads the same packing, but only the WebGPU `LightProbeGrid` is built and checked so far.

**The JSON format is ours, version 1.** three's own `LightProbeGrid` JSON (PR #33658, `toJSON` / `fromJSON`) was still open when this was written, so we don't match it yet. Ours is shaped like it, but stores the coefficients as plain numbers instead of the PR's fixed point, so a round trip gives every float back exactly:

```json
{
  "formatVersion": 1, "type": "LightProbeGrid", "generator": "@pmndrs/ddgi",
  "width": 6, "height": 6, "depth": 6, "resolution": [12, 12, 12], "position": [2.78, 2.74, 2.8],
  "intensity": 1, "priority": 0,
  "coefficients": [ 27 numbers per probe ]
}
```

- `coefficients`: 27 numbers per probe. Probes go x fastest, then y, then z. Within a probe come the 9 coefficients in three's order (constant, y, z, x, xy, yz, 3z² − 1, xz, x² − y²), each as r, g, b.
- `lightProbeGridFromJSON` refuses another `type` or `formatVersion`, and says which. When three settles its format, we'll read and write that too.

### `GILight`

This is the light that brings GI into three's materials. `ThreeGI` makes one, so you rarely make one yourself.

- It works with every lit node material and every drawing pipeline: standard, Meshlet and Visibility-Buffer. It never changes your materials.
- GI arrives as irradiance through three's light-probe slot.
- Where GI and a `LightProbeGrid` overlap, the higher `priority` wins. GI's default priority, 1, beats the grid's default, 0.
- `ThreeGI`'s light is on every layer, so it lights whichever camera renders.

### For custom integrations

- `wrapGpuTexture(gpuTexture, { dimension, sampleType, filtering, addressMode })` wraps a same-device `GPUTexture` as an `ExternalTexture`, described the way three needs it.
- `wgslParameterType(parameter)` and `entryWgsl(shadingModule)` give the WGSL the GI light builds from a shading module.
- The core exports DDGI's debug-view colours as `DDGI_DEBUG_COLORS`.

### `SceneSync`

This is the mirroring on its own, for hosts that call `gi.encode` themselves: `const sync = new SceneSync(gi.scene, { device })`, then `sync.update(scene)` once per frame before `gi.encode`. `ThreeGI` uses one (`threeGI.sync`). Its options and report are in the table at the end.

## What is mirrored

**Meshes.**
- One platform mesh per `BufferGeometry`: positions, indices, vertex normals, uv, uv1 and groups.
- Geometry without an index works.
- uv1 is sent only when a material samples it.

**Instances.**
- One per `Mesh`, and one per instance of an `InstancedMesh`.
- GI follows `visible`, including the parents', as three draws it.

**Materials.** GI's rays see each material's real colours and textures: the platform samples the textures where rays hit (M4). Nothing is averaged or folded in by the adapter.

| three material | GI albedo | GI emission |
| --- | --- | --- |
| Standard, Physical | `color` × `map` × (1 − `metalness` × `metalnessMap`'s blue channel), as three and glTF apply metalness. Metals bounce no diffuse light. | `emissive` × `emissiveIntensity` × `emissiveMap` |
| Lambert, Phong, Toon | `color` × `map` | as above |
| Basic | `color` × `map` | none |
| Node versions of these (`MeshStandardNodeMaterial`...) with plain properties | as their plain versions | as their plain versions |
| A node material with a `colorNode`, a `ShaderMaterial` or `RawShaderMaterial` | `color` if it's set (a node material's `color` counts when it isn't white), otherwise grey 0.5. `map` is left out, since three ignores it there. A one-time warning names the material. | `emissive` if it has one |
| A node material with only an `emissiveNode` | as its plain version | `emissive` (the graph isn't read) |

- Each texture keeps its own uv mapping: offset, repeat, rotation, wrap, and uv channel 0 or 1. A texture on channel 2 or 3 is sampled with channel 0 (a warning).
- `flipY` textures (three's default for images it loads itself) are sampled the right way up: the adapter turns v upside down for them. glTF textures don't flip.
- **Your own values, on the material's `userData`:**
  - `giAlbedo`: the albedo colour GI uses instead of `color`;
  - `giAlbedoMap`: a three texture GI uses instead of `map`;
  - `giEmissive`: the emissive colour GI uses instead of `emissive`.

  A colour is a three `Color` or `[r, g, b]` (linear), or a hex number or CSS string (sRGB, as `new Color()` reads them). Set them on custom-shaded materials to say what GI should see; they also silence the warning.
- **Alpha** (alpha = `opacity` × the albedo map's alpha):
  - `alphaTest` above 0 (glTF `MASK`) is a cutout: rays pass where alpha is below `alphaTest`. Foliage and fences.
  - Alpha-blended materials (`transparent` with no `alphaTest`; glTF `BLEND`, so decals too) and `alphaHash` ones are stochastic: a ray passes with probability 1 − alpha, so many rays see the right average (DECISIONS 2026-11-09). A blended material **blocks light by its alpha** (the core's stochastic pass, T4.2). Before M4 blended surfaces let all light through.
- **Light passes straight through** (`blocksLight: false`):
  - glass (`transmission` above 0);
  - `ShadowMaterial` (a ground that only catches shadows);
  - hidden materials (`visible = false`).
- To make a see-through material block and bounce light like an opaque one, set `material.userData.giBlocksLight = true`: glass then blocks, and a blended material is opaque for GI instead of stochastic. Cutouts keep their holes. It doesn't bring back a hidden material.
- Every material is double-sided for GI, whatever three's `side`. Back-face hits are how GI finds probes inside walls, and most glTF materials are `FrontSide`. To opt one out, set `material.userData.giSingleSided = true`.
- **Textures GI reads:** images (`ImageBitmap` from GLTFLoader, image elements, canvases, `ImageData`), 8-bit RGBA `DataTexture`s, and `CompressedTexture`s (KTX2Loader, gputex) in BC1–BC5, BC7, ETC2, EAC or ASTC, sRGB or linear as the texture's `colorSpace` says. The platform keeps its own 256 × 256 copy of each, with mips, so the textures can be any size.
- A texture's pixels are sent again when its `version` changes (`texture.needsUpdate = true`); the material edit checks find it (see "How changes are found").

**Lights.**
- Directional, point and spot lights, in three's physical units.
- `distance = 0` means no range limit.
- A spot light's `angle` and `penumbra` become the outer and inner cones.
- `userData.giLightGroup` (0 to 3) sets the light group.
- GI traces shadows for a light when three's `castShadow` is on, or when `userData.giCastsShadows` is `true`. Use the second for lights three draws without shadows only because there are too many for shadow maps (hundreds of candles under three's `ClusteredLighting`, which skips shadowed lights). Without GI shadows their light leaks through walls into GI. `false` turns GI's shadows off for a light three shadows.

**The sky**, the first of these that applies:
1. your `environment` option;
2. a pmndrs/sky;
3. three's `SkyMesh`, with the sun's brightness taken from the brightest directional light;
4. `scene.environment`, then `scene.background`, as an equirect or cube texture. Cube maps are mirrored in x to match three, and the y rotation and intensity are passed on.
5. a background `Color`;
6. none.

**Per object.** Set `userData.giContributes = false` to keep an object out of GI's rays. It still receives GI. Skinned meshes are left out unless they set it to `true`.

**Deforming meshes** (T3.7). GI follows the changing shape of a mesh that takes part in GI's rays and is:
- a `SkinnedMesh` (it must also set `userData.giContributes = true`, see above);
- a mesh with position morph targets;
- a plain mesh with `userData.giDeforming = true`, whose position attribute you animate on the CPU (set `needsUpdate` as usual). Each edit is then sent as a new pose, not as a new shape.

How it works and what to know:
- Each such mesh gets its own platform mesh (a shared geometry is copied per mesh), made `deforming`.
- When the pose's inputs change (the bones' world matrices, the morph influences, the position attribute's version), the adapter works the positions out on the CPU, the way three's own `getVertexPosition` does, and sends them with `setDeformedPositions`. A still pose costs a few compares an update. A skinned rig that only moves as a whole sends no pose.
- **GPU skinning** (T3.3): pass your renderer to `ThreeGI` (`new ThreeGI(gi, scene, { renderer })`), and a skinned mesh that contributes to GI (without position morph targets) is posed by three's own `computeSkinning`, written into a texture GI reads on the GPU. No position reaches the CPU. Without the renderer, skinned meshes are posed on the CPU as above.
- `userData.giDeforming = false` keeps a skinned or morphed mesh in its bind pose. Changing `giDeforming` later (or adding morph targets) is picked up like a `giContributes` change: through the edit checks, or at once with `markChanged`. The object then starts over in GI.
- `userData.giDeformBounds` (a `Box3`, or `{ min: [x, y, z], max: [x, y, z] }`) is a box in the mesh's own space that every pose stays in. Without it the adapter estimates one: exact for morph targets with influences from 0 to 1; for skinned meshes, the bind pose's box grown by half its largest side on every side; for CPU-animated ones, by a quarter. A pose that leaves the box is still traced correctly; the platform grows the box and warns once.
- A pose reaches GI's rays in the next frame: the platform refits the mesh's tree on the GPU from it (T3.3). CPU poses still cost CPU work per vertex; GPU skinning (above) doesn't.
- On a device without subgroups, deforming meshes stay in their rest pose for GI, with one warning.

**See-through surfaces** are handled triangle by triangle, since M4. Every mesh goes into GI's rays whole, and each triangle follows its own material: a glass pane inside a wall mesh lets light through the pane only, and a decal such as Sponza's `dirt_decal` blocks light by its alpha (the core's stochastic pass, T4.2). The adapter no longer leaves any mesh out for its materials, and no longer warns about meshes that mix see-through and opaque materials.

## What isn't mirrored

You get a one-time warning for:
- `BatchedMesh`;
- ambient, hemisphere and `LightProbe` lights. They also add diffuse light on top of GI in three.
- `RectAreaLight`. Give the surface an emissive material instead.
- point or spot lights whose `decay` isn't 2;
- materials shaded by a `colorNode` graph, or a `ShaderMaterial` (GI guesses; see "Materials" above);
- `alphaMap`, which GI doesn't read yet: the alpha it uses is `opacity` × the albedo map's alpha;
- textures it can't read: render targets, video, compressed array and cube textures, compressed formats outside the list above (BC6H, the signed ones), and data textures that aren't 8-bit RGBA. Materials using one shade with their colour alone.

Not mirrored, with no warning:
- **Camera layers.** GI follows `visible` only. An object on a layer the camera doesn't show still blocks and bounces light: hide it, or set `giContributes = false`.
- Morph targets and skinning of meshes that don't take part in GI's rays (characters, by default). GI uses their bind pose, which only matters for where their nearby probes wake.
- The x and z parts of an environment rotation.
- Vertex colours (`vertexColors`, glTF `COLOR_0`). Later (DECISIONS 2026-11-09).
- The colour of tinted glass: glass lets all light through, untinted.

## How changes are found, and how late

three doesn't record when a world matrix changes, and looking at every object every frame would cost more the bigger the scene. Since T3.7 three tells the adapter about the commonest changes itself, and a fixed amount of work per frame finds the rest.

What three tells the adapter (nothing to call):

- **Added and removed objects.** three fires `childadded` and `childremoved` on the parent whenever `add`, `remove`, `attach`, `clear` or `removeFromParent` changes its children. The adapter listens on every object of the scene, so an added or removed object, with everything under it, reaches GI in the next `update`.
- **Moves.** The adapter replaces `updateMatrixWorld` on each mesh it mirrors with one that calls three's and then compares the new world matrix with the one last sent. three's render runs it on every object, so a moved mesh, or the meshes under a moved group, are found by the render after the move and sent by the next `update`. It costs about 50 ns a mesh inside three's render (T3.0). The method is put back by assignment when the mesh leaves GI or on `dispose`. The adapter never `delete`s it: on Bistro interior that left three's own matrix update 3× slower.
- **Instance matrices.** Each `InstancedMesh`'s `instanceMatrix.version` is compared every `update` (an integer each), so edited instances reach GI in the next `update`. When its update ranges (`instanceMatrix.addUpdateRange(...)`) name the instances that changed, only those are looked at; three clears the ranges at `render`, after `ThreeGI.update` in the documented order, so they are complete. Otherwise every instance is compared and only changed ones are sent.
- **Deforming meshes:** their poses are checked every `update` (see "Deforming meshes" above).

What the adapter still looks for, a fixed amount per frame:

- **The walk.** Each frame checks `objectsPerFrame` objects for new, moved, hidden or removed ones, then carries on where it stopped. A whole pass takes `ceil(objects / objectsPerFrame)` frames. It no longer compares the transforms of meshes whose `updateMatrixWorld` reports their moves, while three updates the scene's matrices itself (`scene.matrixWorldAutoUpdate`, three's default). It is the safety net for the rest: a hidden parent, a world matrix written by hand on a scene that doesn't update its own matrices.
  - **This assumes three renders the scene,** or something calls its `updateMatrixWorld`, every frame. Two kinds of move are then not caught: in a host that leaves `matrixWorldAutoUpdate` on but doesn't render this scene with three, and under an ancestor whose own `updateMatrixWorld` doesn't call its children's. Such hosts should call `markChanged(object)` after a move, or set `scene.matrixWorldAutoUpdate = false`, which makes the walk compare transforms again.
- **The hot list.** Once an object is seen moving, hidden or shown, it is checked every frame until it has been still for 60 frames. Continuous motion is therefore sent every frame.
- **`markChanged(object)`.** Tells the adapter something changed (added, moved, edited or removed). The object and everything under it are picked up in the next `update`. It is still the way to have an edit (a material or geometry) or a `visible` change seen at once.
- **Moves go out in one batch.** Every transform an `update` finds changed is sent with one `setTransforms` call at its end.
- **Edits.** Each frame, `editChecksPerFrame` geometries, materials and objects are checked for edits: geometry, material values (including each texture's `version`, uv mapping and `flipY`, and the `userData` overrides), and the `giContributes` flag.
- **Lights** (T2.1). The first 64 lights and every directional light are checked every frame, so a scene with a few lamps sees each light change in the frame it happens. Past 64, lights follow the objects' rules: `lightsPerFrame` (default 64) of them a frame, round robin, and a light found changed stays checked every frame until it has been still for 60 frames. So 1,024 candles cost 128 light checks on a still frame, not 1,024; an unmarked candle change, hide or removal reaches GI within `ceil((lights − 64) / 64)` frames (15 for 1,024 candles), and a flickering candle every frame once seen. `markChanged(light)` sends it the next update.

The first pass over a new scene is done in full, so GI has the whole scene from the first frame.

The defaults are `objectsPerFrame` 64 and `editChecksPerFrame` 8. How late changes arrive on Bistro interior (2,621 objects, 2,062 meshes), with three rendering every frame in the documented order (`update`, then `render`) and no `markChanged` (RTX 3060 laptop, T3.7, `three-change-tracking`):

| Change | Sent by the adapter | Before T3.7 (the walk alone) |
| --- | --- | --- |
| An object added with `scene.add` | the next `update`; rays hit it 2 frames later, once its mesh tree is built | up to 41 frames |
| An object removed with `removeFromParent` | the next `update`; rays pass it the frame after | up to 82 frames (two passes) |
| A still object's first move | 2 frames: the render after the move finds it, the next `update` sends it | up to 41 frames (23 measured, T1.7) |

- With `markChanged`, a change reaches GI in the next update.
- An unmarked edit on Bistro interior takes up to about 260 frames for a geometry, or 9 for a material.
- The adapter's CPU time on a still frame of Bistro interior read 0.2 ms as a median of 300 frames, and 0.2 ms with one object moving every frame (the browser's timer steps in 0.1 ms). The hooks add about 0.17 ms to three's own `scene.updateMatrixWorld()` there (0.42 ms against 0.25 ms warm, about 80 ns a mesh), paid inside three's render. Not judged: M3 has no timing gates (DECISIONS 2026-11-09).
- Bigger budgets find what the walk looks for sooner and cost more per frame.

By default `update` doesn't recompute world matrices: it reads the ones three's last render made. A move therefore reaches GI one frame late, which DDGI's blending over many frames hides. Recomputing them would make three do it twice per frame (1 ms on Bistro interior). `updateMatrixWorld: true` turns it on. A host that moves an object or light outside three's render (a gizmo, an animation step) and wants GI to see it in the same frame can call that object's `updateMatrixWorld()` right after the move, as the viewer's motion player does.

## Environment light counted twice

r187 adds the environment's diffuse light (from `scene.environment`, or a material's own `envMap`) on top of GI, so it would be counted twice. The GI light fixes this on Standard and Physical materials: it fades the environment's diffuse light where GI covers the point, and keeps the specular. This is `replaceEnvironmentDiffuse`, on by default.

- Coverage costs as much as the GI lookup itself, so it is worked out only when something needs it: a `scene.environment`, a material `envMap`, or a `LightProbeGrid` that three will light with. That means a grid that is visible, has an intensity above 0, and is on one of the camera's layers.
- Lambert and Phong materials get no diffuse light from the environment at all, so they need no fix.
- r188 may add a switch for this (PR #35017). The adapter will use it once it lands.

## Every export of `@pmndrs/ddgi/three`

Every name `@pmndrs/ddgi/three` exports, in one place. A unit test (`tools/docs/docs-cover-exports.test.ts`) fails when the package exports a name this page doesn't mention.

### The simple path

| Name | What it is |
| --- | --- |
| `GIVolume` | GI shaped like `LightProbeGrid`: `new GIVolume(width, height, depth, nx, ny, nz)`, then `start` ("`GIVolume`" above). |
| `GIVolumeStartOptions` | `start`'s options: `preset`, `budget`, `ddgi`, `environment`, `sky`, `light`, `warmUp`, `camera`, `warn`, `label`, `onCreated`. |
| `StartedGI` | What `start` made: the `platform`, the DDGI `technique` and the `threeGI` (also passed to `onCreated`). |
| `WarmUpSettings`, `DEFAULT_WARM_UP` | The warm-up's settings (`passes`, `maxSeconds`, `minPasses`, `calibrate`) and their defaults (16, 8 s, 2, true). |
| `DEFAULT_GI_BUDGET` | The budget after the warm-up when no preset sets one: `{ gpuMs: 2 }`. |
| `ThreeGILimitError` | A `GILimitError` whose message says how to make three's device with the limits GI needs. |

### The device

| Name | What it is |
| --- | --- |
| `createGIRenderer(options?)` | A `WebGPURenderer` on a device with every adapter feature and the platform's limits, initialised. |
| `GIRendererOptions`, `GIRenderer` | Its options (the device options, your own `device`, the renderer's own options) and its result (`renderer`, `device`, `adapter`). |
| `requestGIDevice(options?)` | Just the adapter and the device, made the same way. |
| `GIDeviceOptions`, `GIDevice` | Its options (`adapter`, `powerPreference`, extra `requiredLimits`, `label`) and its result. |

### Control by hand

| Name | What it is |
| --- | --- |
| `ThreeGI` | The scene's link to a platform you built yourself: `update(camera)`, `budget`, `light`, `markChanged`, `setGroupWeight`, `setEnvironmentWeight`, `timings`, `report`, `dispose` ("`ThreeGI`" above). |
| `ThreeGIOptions` | Its options (table above). |
| `ThreeGITimings` | The CPU time of the last `update`, in three parts. |
| `SceneSync` | The mirroring on its own, for hosts that call `gi.encode` themselves: `new SceneSync(gi.scene, { device })`, then `update(scene)` each frame, `markChanged`, `counts`, `report`, `dispose`. |
| `SceneSyncOptions` | Its options: `device`, `renderer` (for GPU skinning), `warn`, `objectsPerFrame`, `editChecksPerFrame`, `lightsPerFrame`. |
| `SyncFrameReport` | What the last update sent: objects created and destroyed, transforms, visibility, materials, meshes, lights, poses, and how many objects it looked at. |
| `DEFAULT_OBJECTS_PER_FRAME`, `DEFAULT_EDIT_CHECKS_PER_FRAME`, `HOT_FRAMES` | 64 objects walked a frame, 8 edit checks a frame, and 60 frames that a changed object stays checked every frame ("How changes are found"). |

### The GI light

| Name | What it is |
| --- | --- |
| `GILight` | The light that brings GI into three's materials ("`GILight`" above). Toggle GI with its `intensity`; never remove it. |
| `GILightOptions` | `intensity`, `priority`, `replaceEnvironmentDiffuse`. |
| `GI_LIGHT_PRIORITY` | GI's default priority, 1, which beats `LightProbeGrid`'s 0 where both cover a point. |

### The composite (GI once per pixel)

| Name | What it is |
| --- | --- |
| `ddgi(scenePass, source, options?)` | The composite node for three's post-processing pipeline ("`ddgi(scenePass, volume, options?)`" above). |
| `DDGICompositeNode` | The node `ddgi()` returns; `occlusionNode` is its own GTAO node. |
| `DDGICompositeOptions` | Its one option, `occlusion`: GTAO settings, `false`, or `{ source }`. |
| `CompositeGISource` | What `ddgi()` reads GI from: anything with the GI light as `light`, so a started `GIVolume` or a `ThreeGI`. |
| `GIOcclusionSettings` | GTAO on the GI term: `radius`, `resolutionScale`, `samples`. |
| `GIOcclusionSource` | An AO node your pipeline already makes from the same pass (three's GTAO node fits), passed as `occlusion: { source }`. |
| `DEFAULT_GI_OCCLUSION_RADIUS` | 1 m. |
| `giAlbedo`, `GIAlbedoNode` | The MRT output with the weight three gives indirect diffuse light per pixel; put it in the scene pass's `mrt()`. |
| `GI_ALBEDO_OUTPUT`, `GI_NORMAL_OUTPUT` | The MRT names `ddgi()` reads: `'giAlbedo'` and `'normal'`. |

### The `LightProbeGrid` export

| Name | What it is |
| --- | --- |
| `toLightProbeGrid(data, options?)` | A `LightProbeGrid` from the core's `exportIrradianceSH` data ("Exporting a `LightProbeGrid`" above). |
| `ToLightProbeGridOptions` | Its `intensity` and `priority`. |
| `ExportedLightProbeGrid` | The grid it makes: a three `LightProbeGrid` filled from our probes, which keeps its coefficients for the JSON. |
| `lightProbeGridToJSON(grid)`, `lightProbeGridFromJSON(json)` | Our JSON format, version 1, both ways. |
| `LightProbeGridJSON`, `LIGHT_PROBE_GRID_JSON_FORMAT_VERSION` | The JSON's shape, and its version (1). |
| `packCoefficientAtlas(coefficients, resolution)` | The coefficients in `LightProbeGrid`'s own texture packing (seven RGBA sub-volumes along z), as a `Data3DTexture`. |

### Settings on objects, and custom integrations

| Name | What it is |
| --- | --- |
| `GI_USER_DATA` | The `userData` keys the adapter reads on objects and lights: `giContributes`, `giLightGroup`, `giCastsShadows`, `giSingleSided`, `giBlocksLight`, `giDeforming`, `giDeformBounds` ("What is mirrored"). |
| `GI_MATERIAL_USER_DATA` | The material `userData` keys for what GI sees of a material: `giAlbedo`, `giAlbedoMap`, `giEmissive`. |
| `PmndrsSkyLike` | What the adapter reads from a pmndrs/sky `Sky` (the `sky` option). |
| `wrapGpuTexture(gpuTexture, options)`, `GpuTextureWrapOptions` | A same-device `GPUTexture` as a three `ExternalTexture`, described the way three needs it. |
| `entryWgsl(shadingModule)`, `wgslParameterType(parameter)` | The WGSL the GI light builds from a shading module, for custom nodes. |
