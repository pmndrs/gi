# The three.js adapter (`@pmndrs/ddgi/three`)

The adapter puts the GI platform's global illumination into a three.js r187 `WebGPURenderer` scene. It does three jobs:

- it mirrors the three.js scene into the platform;
- it runs one GI frame per rendered frame;
- it lights every lit material through one GI light, or, on three's post-processing pipeline, adds GI once per pixel (`ddgi()`).

The code is in [`packages/ddgi-three/src/`](../../packages/ddgi-three/src/). The platform underneath is described in [platform.md](platform.md). Working examples: [`examples/three/`](../../examples/three/).

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
| `budget` | GPU work per frame for GI updates after the warm-up, default `{ gpuMs: 2 }`. |
| `ddgi` | DDGI settings over the defaults (rays per probe, hysteresis…). The probe grid comes from the volume. |
| `environment`, `sky` | GI's sky, as for `ThreeGI` below. |
| `light` | Settings for the GI light. |
| `warmUp` | `false` to skip it (GI then fills in live), or `{ passes, maxSeconds, minPasses, calibrate }`. |
| `warn` | Where warnings go. |

If the scene needs more than the renderer's device allows, `start` throws a `GILimitError` (`ThreeGILimitError`) naming the limit. three makes its device with default limits (a 128 MiB storage binding), which large scenes such as Sponza with its curtains outgrow. The fix is to make the renderer with the platform's limits: `const { renderer } = await createGIRenderer()`, or `new WebGPURenderer({ requiredLimits: requiredLimits(adapter) })` with `requiredLimits` from `@pmndrs/ddgi/core`.

Other members:

- `volume.recalibrate()` measures again how much probe work fits the budget, over the next frames. Use it after a big scene change.
- `volume.threeGI` (budget, `markChanged`, timings) and `volume.platform` (stats, debug views) are there after `start`.
- `volume.light` is the GI light (`ddgi()` reads it).
- `volume.dispose()` removes the GI light and frees the platform.

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

`new ThreeGI(gi, scene, options)` needs a prepared platform. It adds the GI light to the scene.

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
| `objectsPerFrame`, `editChecksPerFrame` | How much of the scene is checked each frame (see "How changes are found"). |
| `updateMatrixWorld` | Update world matrices in every `update` (see below). Default: no. |
| `updateOnRender` | Run `update` from the GI light's render-time update (see above). Default: no. Can be switched later. |
| `warn` | Where warnings go. Each message is shown once. |

Other members:

- `threeGI.light` is the GI light. Set `light.intensity = 0` to turn GI off. Never remove the light: that recompiles every material.
- `threeGI.timings` gives the CPU time of the last update.
- `threeGI.report` lists what the last update sent.
- `threeGI.markChanged(object)` is described below.
- `dispose()` removes what the adapter added.

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

This is the mirroring on its own, for hosts that call `gi.encode` themselves.

## What is mirrored

**Meshes.**
- One platform mesh per `BufferGeometry`: positions, indices, vertex normals, uv, uv1 and groups.
- Geometry without an index works.
- uv1 is sent only when a material samples it.

**Instances.**
- One per `Mesh`, and one per instance of an `InstancedMesh`.
- GI follows `visible`, including the parents', as three draws it.

**Materials.**
- Albedo is colour × (1 − metalness): metals bounce no diffuse light.
- Emission is emissive × emissiveIntensity.
- `alphaTest` becomes the alpha cutoff.
- These materials let light through (`blocksLight: false`):
  - glass (`transmission` above 0);
  - alpha-blended materials: `transparent` with no `alphaTest`, at any `opacity`. glTF's `alphaMode: BLEND` loads this way, so decals do too;
  - alpha-tested `transparent` materials with `opacity` below 0.5;
  - hidden materials (`visible = false`).
- To keep a see-through material blocking and bouncing light like an opaque one, set `material.userData.giBlocksLight = true`. A blended one with a texture then gets an alpha cutoff of 0.5. This doesn't bring back a hidden material.
- glTF `alphaMode: MASK` (`alphaTest`) is a cutout, and it blocks light.
- Every material is double-sided for GI, whatever three's `side`. Back-face hits are how GI finds probes inside walls, and most glTF materials are `FrontSide`. To opt one out, set `material.userData.giSingleSided = true`.
- Textures go to the platform as handles. The core doesn't sample textures yet (that comes in M4), so until then the adapter multiplies by texture averages, worked out once per texture:
  - The GI albedo is the colour × the average of map texel × (1 − metalness × metalnessMap texel). three reads metalness from the map's blue channel, as glTF's metallicRoughness stores it.
  - When the two maps share a uv mapping, the product is taken texel by texel. Otherwise the two averages are multiplied.
  - The emission is multiplied by the `emissiveMap`'s average.
  - Averages are taken in linear light and weighted by alpha, so cut-out texels don't count as black.
  - Images over 256 px a side are first drawn at 256 px by the browser.

**Lights.**
- Directional, point and spot lights, in three's physical units.
- `distance = 0` means no range limit.
- A spot light's `angle` and `penumbra` become the outer and inner cones.
- `userData.giLightGroup` (0 to 3) sets the light group.

**The sky**, the first of these that applies:
1. your `environment` option;
2. a pmndrs/sky;
3. three's `SkyMesh`, with the sun's brightness taken from the brightest directional light;
4. `scene.environment`, then `scene.background`, as an equirect or cube texture. Cube maps are mirrored in x to match three, and the y rotation and intensity are passed on.
5. a background `Color`;
6. none.

**Per object.** Set `userData.giContributes = false` to keep an object out of GI's rays. It still receives GI. Skinned meshes are left out unless they set it to `true`.

**See-through meshes.** Until M4, GI's rays treat every surface as opaque: they don't yet skip see-through materials triangle by triangle. So the adapter leaves out of GI's rays any mesh whose materials **all** let light through. Such a mesh still receives GI. Without this, a decal with no diffuse colour, such as Sponza's `dirt_decal`, would be a black sheet that darkens the room.
- A mesh that mixes blocking and see-through materials stays in GI, and all of it is treated as opaque. You get a one-time warning naming it.
- A material edit or swap that changes this is picked up like any other edit (see below).

## What isn't mirrored

You get a one-time warning for:
- `BatchedMesh`;
- ambient, hemisphere and `LightProbe` lights. They also add diffuse light on top of GI in three.
- `RectAreaLight`. Give the surface an emissive material instead.
- point or spot lights whose `decay` isn't 2.

Not mirrored, with no warning:
- **Camera layers.** GI follows `visible` only. An object on a layer the camera doesn't show still blocks and bounces light: hide it, or set `giContributes = false`.
- Morph targets and skinning. GI uses the bind pose.
- The x and z parts of an environment rotation.
- Render-target and compressed textures. Their pixels can't be read.
- See-through parts of a mesh that also has opaque materials. They are treated as opaque (a warning names the mesh; see "See-through meshes").

## How changes are found, and how late

three doesn't record when a world matrix changes. So finding a moved object means looking at it, and looking at every object every frame would cost more the bigger the scene.

The adapter does a fixed amount of work per frame instead:

- **The walk.** Each frame checks `objectsPerFrame` objects for new, moved, hidden or removed ones, then carries on where it stopped. A whole pass takes `ceil(objects / objectsPerFrame)` frames. Removals can take up to two passes.
- **The hot list.** Once an object is seen moving, it is checked every frame until it has been still for 60 frames. Continuous motion is therefore sent every frame.
- **`markChanged(object)`.** Tells the adapter something changed (added, moved, edited or removed). The object and everything under it are picked up in the next `update`. Use it when you move things and want no delay.
- **Edits.** Each frame, `editChecksPerFrame` geometries, materials and objects are checked for edits: geometry, material values, the `giContributes` flag, and whether a mesh's materials still all let light through.
- **Lights** are checked every frame.

The first pass over a new scene is done in full, so GI has the whole scene from the first frame.

The defaults are `objectsPerFrame` 64 and `editChecksPerFrame` 8. The table shows how late changes arrive with them (RTX 3060 laptop, T1.7):

| | Objects | A still object's first move reaches GI within | Measured |
| --- | --- | --- | --- |
| Sponza with curtains | 596 | 10 frames | 3 frames |
| Bistro interior | 2,622 | 41 frames (about 0.7 s at 60 fps) | 23 frames |

- With `markChanged`, a move reaches GI in the next update.
- An unmarked edit on Bistro interior takes up to about 260 frames for a geometry, or 9 for a material.
- The adapter's CPU time on a still frame is about 0.06–0.10 ms on Sponza and 0.11–0.15 ms on Bistro interior.
- Bigger budgets find changes sooner and cost more per frame.

By default `update` doesn't recompute world matrices: it reads the ones three's last render made. A move therefore reaches GI one frame late, which DDGI's blending over many frames hides. Recomputing them would make three do it twice per frame (1 ms on Bistro interior). `updateMatrixWorld: true` turns it on.

## Environment light counted twice

r187 adds the environment's diffuse light (from `scene.environment`, or a material's own `envMap`) on top of GI, so it would be counted twice. The GI light fixes this on Standard and Physical materials: it fades the environment's diffuse light where GI covers the point, and keeps the specular. This is `replaceEnvironmentDiffuse`, on by default.

- Coverage costs as much as the GI lookup itself, so it is worked out only when something needs it: a `scene.environment`, a material `envMap`, or a `LightProbeGrid` that three will light with. That means a grid that is visible, has an intensity above 0, and is on one of the camera's layers.
- Lambert and Phong materials get no diffuse light from the environment at all, so they need no fix.
- r188 may add a switch for this (PR #35017). The adapter will use it once it lands.
