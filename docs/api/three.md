# The three.js adapter (`@pmndrs/ddgi/three`)

The adapter puts the GI platform's global illumination into a three.js r187 `WebGPURenderer` scene. It does three jobs:

- it mirrors the three.js scene into the platform;
- it runs one GI frame per `update`;
- it lights every lit material through one GI light.

The code is in [`packages/ddgi-three/src/`](../../packages/ddgi-three/src/). The platform underneath is described in [platform.md](platform.md).

## Getting started

```ts
import { createGI, DDGITechnique } from '@pmndrs/ddgi/core';
import { createGIRenderer, GIVolume, ThreeGI } from '@pmndrs/ddgi/three';

const { renderer, device } = await createGIRenderer();     // one device for three and GI
const gi = createGI(device);

const volume = new GIVolume(20, 8, 20, 40, 16, 40);         // like new LightProbeGrid(...)
volume.position.set(0, 4, 0);
scene.add(volume);
scene.updateMatrixWorld();

gi.setOutput(gi.addTechnique(new DDGITechnique({ volume: volume.ddgiVolume() })));
await gi.prepare();
const threeGI = new ThreeGI(gi, scene, { volume });

renderer.setAnimationLoop(() => {
    threeGI.update(camera);           // first: GI's work for this frame
    renderer.render(scene, camera);   // then three draws, reading GI
});
```

## The pieces

### `createGIRenderer()` and `requestGIDevice()`

These make the GPU device and hand it to three (`new WebGPURenderer({ device })`). The host owns the device; the platform never makes one.

- The device gets every feature the adapter offers, as three asks for when it makes its own device. three needs `core-features-and-limits` to run in full (not compatibility) mode.
- It gets the platform's `requiredLimits`. Large scenes such as Sponza with the curtains need them.
- Pass `{ device }` to use a device you made yourself.

### `ThreeGI`

`new ThreeGI(gi, scene, options)` needs a prepared platform. It adds the GI light to the scene.

`update(camera, deltaTime?)` is called once per frame, before `renderer.render`. Each call:

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
| `warn` | Where warnings go. Each message is shown once. |

Other members:

- `threeGI.light` is the GI light. Set `light.intensity = 0` to turn GI off. Never remove the light: that recompiles every material.
- `threeGI.timings` gives the CPU time of the last update.
- `threeGI.report` lists what the last update sent.
- `threeGI.markChanged(object)` is described below.
- `dispose()` removes what the adapter added.

### `GIVolume`

`GIVolume` has the same shape as three's `LightProbeGrid`, so moving a scene from one to the other is a small change:

- `new GIVolume(width, height, depth, nx, ny, nz)`;
- the box is centred on the object's position and aligned with the world axes;
- there is one probe at the centre of each cell;
- `boundingBox`, `priority` and `intensity` work as they do on `LightProbeGrid`.

`volume.ddgiVolume()` gives the matching DDGI settings:

- the spacing is `width / nx` (and the same on the other axes);
- the box is one cell smaller than the volume, so the outermost probes sit half a cell inside it.

Moving or resizing the volume later needs new settings and another `prepare`.

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
- Glass (`transmission`) and mostly see-through `transparent` materials let light through.
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

## How changes are found, and how late

three doesn't record when a world matrix changes. So finding a moved object means looking at it, and looking at every object every frame would cost more the bigger the scene.

The adapter does a fixed amount of work per frame instead:

- **The walk.** Each frame checks `objectsPerFrame` objects for new, moved, hidden or removed ones, then carries on where it stopped. A whole pass takes `ceil(objects / objectsPerFrame)` frames. Removals can take up to two passes.
- **The hot list.** Once an object is seen moving, it is checked every frame until it has been still for 60 frames. Continuous motion is therefore sent every frame.
- **`markChanged(object)`.** Tells the adapter something changed (added, moved, edited or removed). The object and everything under it are picked up in the next `update`. Use it when you move things and want no delay.
- **Edits.** Each frame, `editChecksPerFrame` geometries, materials and objects are checked for edits: geometry, material values and the `giContributes` flag.
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
