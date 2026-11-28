# Platform API (`@pmndrs/gi/core` and `@pmndrs/gi/ddgi`)

Status: the contract was **signed off by Dennis on 2026-11-04**, revised after the spikes (T0.8 three.js, T0.9 Babylon, T0.10 BVH) and in each milestone since. The changes are listed under "Revisions" at the end. Brought up to date for M2 to M5 in T5.4 (2026-11-10).

This page covers the engine-free part of the library: `@pmndrs/gi/core`, the GI platform, and `@pmndrs/gi/ddgi`, DDGI on top of it, with the helpers around them. Neither needs three or any other engine, and `/core` holds no DDGI code: a technique of your own needs only `/core`. The contract's TypeScript is in [`packages/ddgi-core/src/platform/contract/`](../../packages/ddgi-core/src/platform/contract/) (types only, plus two error classes), and DDGI's is in [`packages/ddgi-core/src/techniques/ddgi/`](../../packages/ddgi-core/src/techniques/ddgi/). This page explains them in plain language, shows each feature with a short example, and lists every export of [`/core`](#every-export-of-pmndrsgicore) and of [`/ddgi`](#every-export-of-pmndrsgiddgi) at the end. It also walks two future GI techniques (voxels and surfels) through the contract, to check it isn't shaped only for DDGI.

**Writing an adapter for an engine?** Start with [the adapter guide](../guides/adapter.md), which walks through the plain-WebGPU hello example step by step. For three.js, see [three.md](three.md).

## The picture

```
 host app / engine                      platform (engine-free)                   techniques
 ─────────────────                      ──────────────────────                   ──────────
 GPUDevice ───────────────────────────▶ createGI(device)
 adapter mirrors scene ── handles ────▶ scene: meshes, materials, instances,
                                        lights, environment  ──▶ GPU scene copy ─┐
                                        queries: rays (BVH)  ◀────────────────────┤ DDGI
 each frame: encode(encoder, frame) ──▶ uploads, BVH top level, refits ──────────▶│ (later: voxels,
                                        budget → work units                       │  surfels, ...)
 renderer shades with ◀── shading module (WGSL + textures + vec4 uniforms) ◀──────┘
```

The host owns the device, the scene and the frame. The platform keeps its own simplified copy of the scene and records its GPU work into the host's command encoder. Techniques see only the platform. They never see the host.

## The files

| File | What it defines |
| --- | --- |
| `platform.ts` | `GIPlatform`, the object a host talks to, and its options. `GIPlatformModule` declares `createGI`, `requiredLimits`, `optionalFeatures` and `isSupported`, which `gi-platform.ts` and `device-requirements.ts` implement. |
| `scene.ts` | `SceneApi`: create, update and destroy meshes, textures, materials, instances and lights, and set the environment. Light units. |
| `handles.ts`, `math.ts` | Opaque handles; plain vectors, colors, matrices and boxes. |
| `frame.ts` | The per-frame input: time, the budget, the camera and, optionally, the screen buffers. |
| `scene-copy.ts` | What techniques can read about the scene, and `SceneChanges` (what moved since the last frame). |
| `queries.ts` | Scene queries. v1 has one backend, `rays`, which offers closest-hit and any-hit in WGSL. |
| `shade.ts` | Hit shading: the surface at a hit, direct light with shadows, and sky. Also the "bounce source" idea. |
| `technique.ts` | `GITechnique`, the contract a GI method implements, and the `TechniqueContext` it receives. |
| `shading-module.ts` | `ShadingModule`, what a renderer reads while shading pixels. |
| `wgsl.ts` | How WGSL pieces and their bindings are combined into compute kernels. |
| `stats.ts`, `timing.ts` | Stats, memory, GPU pass timings and debug views. |
| `errors.ts` | `GINotReadyError` and `GILimitError`. |

## How a host uses it

1. **Device.** The host asks the adapter for the device using `requiredLimits(adapter)` and `optionalFeatures(adapter)`, then calls `createGI(device)`. The platform never creates, configures or destroys a device. It watches `device.lost`, and after a loss `isReady` turns false and calls throw `GINotReadyError('device-lost')`. The host makes a new platform on the new device, the same rule as the upscaler.
2. **Scene.** The adapter calls `gi.scene.createMesh(...)`, `createMaterial(...)`, `createInstance(...)`, `createLight(...)` and `setEnvironment(...)`, then sends only what changed (`setTransform`, `updateLight`, ...). The calls are cheap. Changes are queued and uploaded inside the next `encode`. Many moves at once go through `setTransforms(instances, matrices)`, one call with 16 numbers per instance (T2.M1, Chrome on the RTX laptop: 10,000 sliding in 1.4 ms of CPU, 3.5 ms turning).
3. **Techniques.** `gi.addTechnique(new DDGITechnique(settings))`, `gi.setOutput(ddgi)`, then `await gi.prepare()`, which compiles everything asynchronously.
4. **Every frame.** `gi.encode(encoder, { deltaTime, budget, camera })` comes before the host's own passes, in the same encoder. There are no extra submits and no readbacks frame after frame. The exception is a frame that uploads meshes (the scene at load, and meshes added or replaced later): the GPU's build of those meshes' first trees is submitted by the platform and reads back a few words of stats per mesh (T2.12 and T3.2, DECISIONS 2026-11-08 and 2026-11-09). Deforming meshes add two (T3.3, DECISIONS 2026-11-10; both confirmed by Dennis on 2026-11-18): while a mesh is being posed, its refit's stats are read back at most once every 15 frames; and a drift rebuild submits its own H-PLOC build and reads back that tree's stats and layout. Hosts with screen-space techniques also call `gi.encodeScreen(encoder, screen)` after their depth/normal pass. It does nothing when no technique needs it.
5. **Shading.** The adapter injects `gi.shading` into its engine once, and rebuilds only when `shading.version` changes.

The lifecycle names match the pmndrs siblings: `prepare()`, `isReady`, `encode()`, `resetHistory()`, `dispose()`, `gpuTiming`, and settings as live objects. Note one rename from SPEC: the per-frame call is `encode(encoder, frame)`, not `update(encoder, budget)`. The budget is a field of `frame`.

## Quality presets

A host can pick one word instead of thirty settings (T5.1; docs/design/m5-coverage.md §1). `ddgiPreset(name)` returns plain data, a new object each call, so a host can change single fields:

```ts
import { createGI } from '@pmndrs/gi/core';
import { DDGITechnique, ddgiPreset, recommendPreset } from '@pmndrs/gi/ddgi';

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error('No WebGPU adapter');
const preset = ddgiPreset(recommendPreset(adapter.info, adapter.limits));   // 'low', 'medium' or 'high'
const gi = createGI(device, { textureLayerSize: preset.textureLayerSize });
const ddgi = gi.addTechnique(new DDGITechnique({ ...preset.ddgi, volume }));
// ... prepare, then every frame:
gi.encode(encoder, { deltaTime, budget: preset.budget, camera });
```

A preset has four parts:

- `ddgi`: the DDGI settings it changes (rays per probe, the two tile sizes and the warm-up batch). Every other setting keeps its default. Spread it before your own settings, so yours win.
- `budget`: the per-frame budget to pass as `frame.budget`.
- `textureLayerSize`: `createGI`'s option for the texture store, the platform's own copy of every material texture that rays read.
- `probeSpacing`: metres between probes, for hosts that don't set a spacing themselves (three: a `DDGIVolume` made without probe counts). A host that sets one keeps it.

| | `low` | `medium` | `high` (the defaults) | What going down costs in look |
| --- | --- | --- | --- | --- |
| Rays per probe (`raysPerProbe`) | 128 | 192 | 256 | Each update is noisier, so GI flickers a little more and takes a little longer to settle. The first 32 rays never blend, so 128 still leaves 96. |
| Distance tile (`distanceTexels`) | 8² | 10² | 14² | What each probe can see is coarser: thin walls and small gaps blur a little and leak sooner. This tile is most of a probe's memory and blend work. |
| Irradiance tile (`irradianceTexels`) | 6² | 6² | 6² | Not lowered: it is already small, and 4² would make the light's direction visibly blocky. |
| Budget (`budget.gpuMs`) | 1 ms | 1.5 ms | 2 ms | Fewer probes updated a frame, so light follows changes more slowly. The platform's floor still sweeps every probe within a few seconds. |
| Texture store (`textureLayerSize`) | 128² | 256² | 256² | Only rays hitting a texture up close see fewer texels: GI rays mostly read coarse mips. A quarter of the memory. |
| Spacing when the host sets none (`probeSpacing`) | 2 m | 1.5 m | 1 m | Light that changes over less than the spacing (a doorway, a corner) blurs. 2 m has an eighth of 1 m's probes. |
| Warm-up batch (`maxProbesPerFrame`) | 1,024 | 4,096 | 4,096 | The warm-up takes a little longer, in shorter frames. On the Intel UHD (two-room refined to 25,194 probes, `ddgi-low-warmup`): 4,096 gave 47–70 ms frames for 3.6 s, 1,024 about 20 ms frames for 4.6 s (TU.1, DECISIONS 2026-11-19). It is also the most probes any frame may update. |

`high` is exactly the defaults: what M1 to M4 were checked with on the RTX 3060, and what the M5 Pro runs. A unit test holds it to that (`presets.test.ts`). `low` is for the correctness devices (the Intel UHD, the iPad, older iOS). Values are from the settings' own ranges, not tuned (DECISIONS 2026-11-10, "Feature coverage before tuning").

**What they cost, measured** (`ddgi-presets`, M5 Pro, Chrome headless, the manifest's spacing; every probe updated every frame for 600 frames, then the update timed and the test view's indirect mean read at Cornell's `front` and two-room's `lit-room` views):

| Scene | Preset | Probes | Rays a frame | DDGI memory | Update, GPU | Indirect mean against `high` |
| --- | --- | --- | --- | --- | --- | --- |
| Cornell | low | 1,728 | 221,184 | 8.9 MB | 0.65 ms | −0.50% |
| Cornell | medium | 1,728 | 331,776 | 12.1 MB | 1.06 ms | +0.13% |
| Cornell | high | 1,728 | 442,368 | 16.5 MB | 1.97 ms | 0 |
| Two-room | low | 1,071 | 137,088 | 7.5 MB | 0.88 ms | −0.56% |
| Two-room | medium | 1,071 | 205,632 | 9.9 MB | 1.22 ms | −0.14% |
| Two-room | high | 1,071 | 274,176 | 13.0 MB | 1.65 ms | 0 |

With the preset's own spacing instead, Cornell would have 27 / 64 / 216 probes and two-room 30 / 54 / 180.

Every preset passes the correctness checks on the M5 Pro (`VITE_DDGI_PRESET=<name> pnpm test:gpu --filter ddgi-furnace`, and the same for the others; rows in `bench/results/checks.csv` under `t5.1/`):

| Check | `low` | `medium` | `high` | Limit |
| --- | --- | --- | --- | --- |
| `ddgi-furnace`, mean error | 0.31% | 0.26% | 0.22% | 2% |
| `ddgi-leak`, two-room against Cycles | 0.86% | 0.87% | 0.87% | 1% |
| `ddgi-leak`, thin wall's dark side (reported) | 3.20% | 2.08% | 0.59% | none |
| `ddgi-leak`, closed door's dark side (reported) | 0.10% | 0.08% | 0.06% | none |
| `ddgi-textured-furnace`, textured against flat | +0.15% (128² store) | +0.10% | +0.06% | 1% |
| `platform-recompiles` | 0 | 0 | 0 | 0 |

The thin wall shows the coarser distance tiles' cost: its dark side gets 3.2% of the lit side's light at `low`, 0.6% at `high`.

**Changing presets needs a new `prepare`.** Rays and tile sizes size DDGI's buffers and atlases, as they always have, so switching a running technique to another preset means new settings and `prepare` again (or a new technique). That's how the settings work, not a recompile bug. The texture store's size is fixed for the platform's life.

**`recommendPreset(adapterInfo, limits, hints?)`** says which preset to start from: `low` for integrated and mobile GPUs, `medium` for a Mac that may have a base M chip, `high` otherwise. It is advice only: nothing applies it behind the host's back. Browsers say little about the GPU (Chrome gives a vendor and an architecture, and usually no chip name), so its rules are guesses from what they do say, in order:

1. a software adapter, a phone or tablet (`hints.mobile`, or read from `navigator`), or a mobile GPU maker (Qualcomm, ARM, Imagination, Samsung, MediaTek): `low`;
2. an adapter whose buffers are capped under 1 GiB: `low`;
3. Intel: `low` unless the architecture is an Arc card's;
4. Apple on a Mac: a description that names the chip decides (a base M chip gives `low`, a Pro, Max or Ultra `high`). Browsers don't name it, so the CPU core count stands in (`hints.cpuCores`, or `navigator.hardwareConcurrency`): more than 10 cores gives `high`, anything else `medium`. Base M chips have 8 to 10 cores and the Pro chips start at 12. Safari reports at most 8 cores, so every Mac gets `medium` there;
5. AMD: `low` when the description names integrated Radeon graphics, otherwise `high`; NVIDIA and anything else: `high`.

## Lights and light groups

Lights are directional, point or spot, in three.js's physical units (see "The scene" below). `range` is required for point and spot lights; `range: 0` means no limit, so such a light reaches, and costs, every probe.

```ts
const lamp = gi.scene.createLight({ type: 'point', position: [0, 2.5, 0], color: [1, 0.85, 0.7], intensity: 25, range: 10, group: 1 });
const spot = gi.scene.createLight({
    type: 'spot',
    position: [3, 4, 0],
    direction: [0, -1, 0],
    color: [1, 1, 1],
    intensity: 60,
    range: 0, // no limit
    innerConeAngle: 0.3, // radians, where the falloff starts
    outerConeAngle: 0.5, // where the light reaches zero
    castsShadows: false, // cheaper, but its light leaks through walls into GI
});
gi.scene.updateLight(lamp, { intensity: 5 }); // any field but its type
gi.scene.destroyLight(spot);
```

A light change re-traces only the probes the light reaches (its influence radius: where its light falls below 1/1,000 of the scene's brightest), nearest the camera first. Hundreds of lights are fine: past 8 point and spot lights, rays pick from a grid of nearby lights instead of testing every one (`HitShadingSettings`, below).

**Emissive surfaces aren't light sources to hit shading.** Their light reaches the GI only when a probe's ray happens to hit them, so a small, bright emitter is noisy in the probes. Sampling emitters directly (next-event estimation, promised for M2 in DECISIONS 2026-11-05) wasn't built; area lights, which would cover it, are [issue #29](https://github.com/pmndrs/gi/issues/29).

**Light groups** let a host dim or recolour a whole group of lights in GI in the same frame, with no rays traced again and no noise (T2.5). A light's group (0 to 3) is scene data, on the light. The weights belong to DDGI, and are turned on in its settings:

```ts
import { DDGITechnique } from '@pmndrs/gi/ddgi';

const ddgi = gi.addTechnique(
    new DDGITechnique({
        volume,
        lightGroups: { count: 2, weights: [[1, 1, 1], [1, 1, 1]], environmentWeight: [1, 1, 1] },
    }),
);
gi.setOutput(ddgi);
await gi.prepare();

ddgi.setGroupWeight(1, [0.25, 0.2, 0.15]); // group 1 at a quarter, a little warmer
ddgi.setEnvironmentWeight([0.5, 0.5, 0.5]); // the sky and emissive surfaces at half
```

- A weight is a linear RGB multiplier on top of the lights' own colour and intensity, in GI only. Adapters reach it through `gi.outputTechnique` (three: `ThreeGI.setGroupWeight`).
- Lights in a group at or past `count` share the last group's layer. `count` needs a new `prepare`; the weights are read every frame.
- Each group costs a layer of the field: on Sponza about 25 MB and 5% of each probe update. Off by default.

## Time of day: the sky and the sun layer

A `procedural` sky (Preetham's clear sky, calibrated against open ground) is made on the GPU, so it can follow the sun every frame. It is made again when the sun has turned more than 0.05° since the last time, or when `turbidity`, `intensity` or `sunIlluminance` change. The sun itself is a directional light, and keeping the two in step is the host's job:

```ts
// Each frame: the sun from 6 am to 6 pm, and the sky with it.
const angle = ((timeOfDay - 6) / 12) * Math.PI;
const towardSun: [number, number, number] = [Math.cos(angle), Math.sin(angle), 0.3];
gi.scene.updateLight(sun, { direction: [-towardSun[0], -towardSun[1], -towardSun[2]] }); // the way its light travels
gi.scene.setEnvironment({ type: 'procedural', sunDirection: towardSun, turbidity: 2, sunIlluminance: 3 });
```

- A turning sun moves the sunlit patches, so the probes are traced again, nearest the camera first.
- **The sun layer** makes a change of the sun's *colour or brightness* exact in the frame it happens, with nothing traced again (T2.6b). DDGI traces the sun in a layer of its own at a white reference and weighs that layer by the sun's colour × intensity:

```ts
import { DDGITechnique } from '@pmndrs/gi/ddgi';

const ddgi = gi.addTechnique(new DDGITechnique({ volume, sunLayer: true }));
gi.setOutput(ddgi);
await gi.prepare();
gi.scene.updateLight(sun, { color: [1, 0.55, 0.3], intensity: 1.5 }); // an evening sun: GI follows in the same frame
```

- The sun is the scene's first directional light. Dim it to 0 rather than destroying it: a destroyed sun makes another directional light the sun, and every probe is traced again.
- Off by default. It costs a layer (Sponza: 58 MB of GI becomes 93 MB) and its share of every probe update. It works with light groups; 4 groups, the environment and the sun are the most one technique holds.

## Moving, adding and removing objects

Every change is a cheap call that is queued and uploaded in the next `encode`; calls between two frames are merged. Tables grow by reallocating buffers, never by changing shader code, so adding objects compiles nothing (`gi.stats.recompiles` stays 0).

```ts
const crate = gi.scene.createInstance({ mesh: meshes[0], materials: [materials[0]], transform: matrix });
gi.scene.setTransform(crate, nextMatrix); // it moved
gi.scene.setTransforms(movedInstances, movedMatrices); // many at once: handles in a Uint32Array, 16 numbers each
gi.scene.setVisible(crate, false);
gi.scene.destroyInstance(crate);

// A character: it receives GI through the renderer, but GI's rays ignore it.
gi.scene.createInstance({ mesh: meshes[1], materials: [materials[1]], transform: matrix, contributesToGI: false });
```

- `setTransforms` is for many moves a frame: 10,000 objects sliding cost 1.4 ms of CPU (Chrome, RTX laptop).
- A mesh added or replaced (`createMesh`, `updateMesh`) gets its ray-tracing tree soon after it is uploaded: built on the GPU in that frame on a device with subgroups, and a better tree from a worker replaces it later. Rays miss a new mesh until its first tree arrives.
- **What techniques see** of all this is the change feed, `SceneChanges`, handed to every technique each frame (`TechniqueFrame.changes`). Hosts never read it; it is what lets DDGI update "probes near a change first":

```ts
import type { SceneChanges } from '@pmndrs/gi/core';

function describeChanges(changes: SceneChanges): void {
    for (const region of changes.regions) {
        // A world box whose lighting is out of date, and why: 'geometry', 'material' or 'light'.
        // A move gives two regions, where the object was and where it is.
        if (region.reason === 'geometry' && region.moved !== undefined) {
            const glows = region.emits ? ', and it gives off light' : '';
            console.log(`instance ${region.instance} moved ${region.moved.toFixed(2)} m${glows}`);
        }
    }
    for (const box of changes.receivers) console.log('something that only receives GI is now in', box.min, box.max);
    for (const light of changes.lights) console.log(`light ${light.index} reached ${light.influenceRadius.before} m, now ${light.influenceRadius.after} m`);
    if (changes.global.length > 0) console.log('changed everywhere:', changes.global.join(', '));
}
```

- `DirtyRegion.moved`: how far the object went, in metres (the farthest any corner of its box travelled), so a technique can size the change. `instance` is its slot, to follow it across frames.
- `DirtyRegion.emits`: the object gives off light (an emissive material). An emitter that moves changes the light also where it can't be seen, through the surfaces it lights, so DDGI treats such a change more like a light's (T3.4).
- `receivers`: boxes around objects that don't contribute to GI and were created, shown, moved or deformed (up to 256 a frame, then one merged box). DDGI wakes the probes in empty air around them, so a character walking through a room is lit.
- `lights`: each changed light, before and after, with its influence radius on both sides. `global` lists changes with no useful box: the sun, the environment, or too many lights in one frame.

## Deforming meshes

A mesh made with `deforming: true` keeps its triangles and takes new vertex positions every frame (cloth, foliage in the wind, skinned meshes that block light):

```ts
const flag = gi.scene.createMesh({
    positions: restPositions,
    indices: flagIndices,
    deforming: true,
    deformBounds: { min: [-1, 0, -0.5], max: [1, 2, 0.5] }, // in the mesh's own space: every pose stays inside
});

gi.scene.setDeformedPositions(flag, posedPositions); // a Float32Array: copied at the call
gi.scene.setDeformedPositions(flag, { buffer: skinnedVertices, offset: 0, stride: 32 }); // a GPU buffer: copied in the next encode
gi.scene.setDeformedPositions(flag, { texture: posedPositionTexture }); // an rgba32float texture: vertex i at (i mod width, i / width)
```

- Rays see a pose in the next frame: the platform copies it into its scene copy and refits the mesh's tree on the GPU, in your encoder, with no wait. A tree that has drifted too far from a good shape is rebuilt now and then.
- **`deformBounds`** is logged as the mesh's changed region each time a pose arrives, and kept in the top-level tree, so a pose never changes the tree's shape. It is required for positions on the GPU, which the platform can't read. A CPU pose outside it still traces correctly: the box grows, with one warning.
- A GPU buffer needs `STORAGE` usage (xyz as floats at the start of each vertex; `stride` at least 12). A texture needs `COPY_SRC` usage and a width that is a multiple of 16: the form a three.js TSL compute writes.
- A deforming mesh's normals are ignored: hits use each posed triangle's own normal.
- Without the `subgroups` feature the refit can't run: the mesh stays in its rest pose, with one warning.

## Several volumes and the camera-following volume

One DDGI technique can hold up to 8 boxes of probes (`volumes`). They share one set of textures, and where they overlap the one with the higher `priority` answers, fading into the next across its edge. A volume with `follow: 'camera'` is a window of probes that moves with the camera (T3.6):

```ts
import { DDGITechnique } from '@pmndrs/gi/ddgi';

const ddgi = gi.addTechnique(
    new DDGITechnique({
        volumes: [
            // 32 × 12 × 32 probes 1.5 m apart around the camera: answers first wherever it covers.
            { center: [0, 8, 0], size: [46.5, 16.5, 46.5], spacing: 1.5, follow: 'camera', priority: 1 },
            // The whole street, coarser, for everything the window doesn't cover.
            { center: [0, 10, 0], size: [240, 20, 120], spacing: 3 },
        ],
    }),
);
gi.setOutput(ddgi);
await gi.prepare();
```

- **How the window moves.** Its probes sit on a lattice fixed in the world. When the camera is more than 0.75 of a spacing from the window's middle, the window moves by a whole plane of probes, and the plane it leaves behind is reused ahead. New planes are updated first. The covered area grows into them by a fifteenth of a spacing a frame, and only over planes that have finished warming up, and its edge fades out over 10 spacings, so nothing pops.
- It needs at least 4 probes on every axis, and the camera every frame (`FrameInput.camera`). A jump of more than half the window (a teleport) starts the window's probes over.
- Memory: the 32 × 12 × 32 window above is 12,288 probes, about 19 MB.
- A point outside every volume reads the nearest one. Each volume takes one texture layer per probe row in y; past the device's `maxTextureArrayLayers`, `prepare` throws a `GILimitError`.
- `ddgi.field.volumes` lists the volumes as built, highest priority first.

## Materials and textures

The material model, and how an engine's materials map onto it, are walked through in [the adapter guide, step 4](../guides/adapter.md#step-4-materials-and-textures). In short, a ray's hit sees `albedo` × the albedo texture × (1 − `metalness` × the metalness texture), the emission, and the alpha (`opacity` × the albedo texture's alpha) acting as the `alphaMode` says. M4 added textures at hits, `metalness`, `opacity`, `alphaMode` and compressed sources:

```ts
const baseColor = gi.scene.createTexture({ source: await createImageBitmap(imageBlob), srgb: true, label: 'leaves' });
const metalRough = gi.scene.createTexture({ source: { format: 'bc7-rgba-unorm', levels: compressedLevels }, srgb: false });

const leaves = gi.scene.createMaterial({
    albedo: [1, 1, 1],
    albedoTexture: baseColor,
    albedoUv: { channel: 0, matrix: [4, 0, 0, 4, 0, 0], wrapU: 'repeat', wrapV: 'mirror' }, // four times across
    alphaMode: 'cutout', // holes where alpha is below the cutoff
    alphaCutoff: 0.5,
});
const paintedMetal = gi.scene.createMaterial({
    albedo: [0.7, 0.7, 0.7],
    metalness: 1,
    metalnessTexture: { texture: metalRough, channel: 'b' }, // glTF keeps metalness in the blue channel
});
const tintedDecal = gi.scene.createMaterial({ albedo: [0.4, 0.3, 0.2], opacity: 0.5, alphaMode: 'stochastic' }); // blocks half the light
```

- **Sources:** `ImageBitmap`, image, canvas and `OffscreenCanvas` elements, `ImageData`, `PixelData` (8-bit RGBA bytes), a same-device `GPUTexture`, or `CompressedPixelData` (`{ format, levels }`, what three's `CompressedTexture.mipmaps` holds). The platform keeps its own copy of each: one layer of its texture store, 256 × 256 by default, with mips, made in the first `encode` after `createTexture`. The source may be freed after that.
- **Compressed textures** need the format's device feature (`texture-compression-bc`, `-etc2` or `-astc`; `optionalFeatures` asks for them). Only one mip level is uploaded, the smallest that is still at least the store's layer size. Without the feature the platform warns once and the materials keep their flat colour.
- **Where a texture is sampled** (`UvMapping`): the uv set, a 3 × 2 matrix, and `'repeat'`, `'clamp'` or `'mirror'` on each axis. Texture coordinate (0, 0) is the source's first texel, the top-left of an image.
- Textures are read at the mip a ray's footprint covers, so GI's rays mostly read coarse mips. The texture store's memory is in `gi.stats.memory.textures` (about 0.35 MB a texture at 256²).
- **Textures off, for comparisons:** `gi.hitShadingSettings.sampleTextures = false` makes ray hits read each texture's mean colour (its last mip), the same over the whole surface, instead of the texel they hit: albedo, emissive and metalness alike. It is live (no recompile) and defaults to `true`. Alpha cut-outs still read their texture, since a hole is geometry. Changing it logs no scene change, so the probes blend towards the new look over their usual updates; `gi.resetHistory()` shows it at once. Through three, `volume.platform` is the same object once the volume has started.

## Exporting the field as spherical harmonics

`exportIrradianceSH` projects each probe of one volume onto L2 spherical harmonics (9 coefficients per colour) on the GPU, with one readback (T5.3). The result is engine-free data: the grid's counts, origin and spacing, and the coefficients in three.js's convention. three's adapter turns it into a `LightProbeGrid` for a static fallback ([three.md, "Exporting a `LightProbeGrid`"](three.md#exporting-a-lightprobegrid)).

```ts
import { exportIrradianceSH, L2_SH_FLOATS_PER_PROBE } from '@pmndrs/gi/ddgi';

const grid = await exportIrradianceSH(ddgi, { volume: 0 }); // the highest-priority volume
console.log(grid.counts, grid.origin, grid.spacing);
const firstProbe = grid.coefficients.subarray(0, L2_SH_FLOATS_PER_PROBE); // 9 coefficients, each as r, g, b
```

- Probes go x fastest, then y, then z. Within a probe come the 9 coefficients in three's order, each as r, g, b.
- With light groups or the sun layer, the export is the field you see: every layer times its weight as of the last frame.
- Probes live GI doesn't read (inside walls, in empty air, not updated yet) get the mean of their nearest active neighbours (`filledProbes` counts them), because a grid without visibility blends every probe. `fillInactiveProbes: false` keeps their raw values.
- An explicit call, so its readback is allowed: the no-readback rule is about the frame loop.
- The pieces it is made of are exported for checks and other formats: `shProjectionDirections` (directions spread evenly over the sphere), `l2ShBasis` (the 9 basis functions at a direction), `l2CosineLobe` (the factor between radiance and irradiance coefficients per band), `projectIrradianceToSH` (the projection on the CPU), and `fillInactiveProbes`.

## Every export of `@pmndrs/gi/core`

Every name `@pmndrs/gi/core` exports, in one place: the platform, with no technique in it. DDGI's names are in [the next section](#every-export-of-pmndrsgiddgi). A unit test (`tools/docs/docs-cover-exports.test.ts`) fails when the package exports a name this page doesn't mention, so these lists stay complete.

### Making the platform

| Name | What it is |
| --- | --- |
| `createGI(device, options?)` | Makes the platform on your device. |
| `GIPlatform` | The platform: `scene`, `addTechnique`, `setOutput`, `prepare`, `encode`, `shading`, `stats`, `hitShadingSettings`, `dispose`... |
| `HostHitShadingSettings` | The hit-shading settings a host may change live, `gi.hitShadingSettings`: `sampleTextures` (above, "Materials and textures"). |
| `GIPlatformOptions` | `createGI`'s options: `gpuTiming`, `label`, `buildIn`, `textureLayerSize`. |
| `GIPlatformModule` | The contract's declaration of `createGI` and the three helpers below. |
| `ScreenNeeds` | `gi.screenNeeds`: which screen channels the attached techniques read (`screen`, `velocity`, `albedo`, `instance`, `lighting`), so a host renders only those (RC1.1). |
| `gi.limitNeeds` | The device limits the attached techniques declare above WebGPU's defaults (`TechniqueRequirements.limits`), the largest of each; `prepare` names the first the device is below (#49). |
| `requiredLimits(adapter)` | The device limits to ask for: the adapter's largest bindings and buffers, and what the kernels need. |
| `RequiredLimits` | The type `requiredLimits` returns: limit names to numbers. |
| `optionalFeatures(adapter)` | Features worth asking for, where the adapter has them. |
| `isSupported(device)` | Whether a device can run the platform at all. |
| `GINotReadyError` | Thrown by calls made before `prepare` finished (`reason: 'preparing'`) or after the device was lost (`'device-lost'`). |
| `GILimitError` | Thrown when the scene or settings need more than the device allows; names the limit, what was needed and what the device has. |

### Describing the scene

| Name | What it is |
| --- | --- |
| `SceneApi` | `gi.scene`: create, update and destroy meshes, textures, materials, instances and lights; set the sky; `bounds`. |
| `Handle` | A number tagged with what it refers to, so a mesh handle can't be passed as a material. |
| `MeshHandle`, `MaterialHandle`, `TextureHandle`, `InstanceHandle`, `LightHandle` | The five kinds of handle. |
| `Vec3`, `ColorRGB`, `Mat4`, `Aabb` | Plain data: `[x, y, z]`, linear `[r, g, b]`, a column-major 4 × 4 matrix, a box with `min` and `max`. |
| `MeshDescriptor` | A mesh: positions, indices, optional normals, `uvs`, `uvs1`, `groups`, `deforming`, `deformBounds`. |
| `MeshGroup` | A run of triangles using one material slot: `start`, `count`, `materialSlot`. |
| `DeformedPositions` | Positions on the GPU for `setDeformedPositions`: a buffer or a texture. |
| `DeformedPositionsBuffer` | Positions in a `STORAGE` buffer: `buffer`, `offset`, `stride`. |
| `DeformedPositionsTexture` | Positions in an `rgba32float` texture: `texture`. |
| `TextureDescriptor` | A texture: `source`, `srgb`, `label`. |
| `TextureSource` | Anything a texture can be read from (images, canvases, pixels, a `GPUTexture`, compressed pixels). |
| `PixelData` | 8-bit RGBA bytes with a width and height. |
| `CompressedPixelData` | A compressed image: its `format` and its mip `levels`, largest first. |
| `CompressedPixelLevel` | One compressed mip level: `data`, `width`, `height`. |
| `UvMapping` | Where a texture is sampled: `channel`, `matrix`, `wrapU`, `wrapV`. |
| `UvWrap` | `'repeat'`, `'clamp'` or `'mirror'`. |
| `TextureChannel` | `'r'`, `'g'`, `'b'` or `'a'`. |
| `MetalnessTexture` | A metalness map: a texture, the channel holding metalness, and its `uv`. |
| `AlphaMode` | `'opaque'`, `'cutout'` or `'stochastic'`. |
| `MaterialDescriptor` | The material model: albedo, metalness, opacity, alpha, emission, `blocksLight`, `doubleSided`, and their textures. |
| `InstanceDescriptor` | A placement: `mesh`, `materials` (one per slot), `transform`, `contributesToGI`, `visible`. |
| `LightDescriptor` | Any light: one of the three below. |
| `DirectionalLightDescriptor`, `PointLightDescriptor`, `SpotLightDescriptor` | The three kinds of light, with their units. |
| `LightChanges` | What `updateLight` takes: any fields of the light's own kind except `type`. |
| `LIGHT_GROUP_COUNT` | How many light groups there are (4): `LightDescriptor.group` is 0 to 3. |
| `EnvironmentDescriptor` | The sky: `'none'`, `'color'`, `'texture'` (cube or equirect) or `'procedural'`. |

### The frame

| Name | What it is |
| --- | --- |
| `FrameInput` | What `encode` takes each frame: `deltaTime`, `budget`, `camera`, and `screen` for screen-space techniques. |
| `FrameBudget` | How much GI work a frame may do: `gpuMs`, `units`, `targetFps`, `paused`. |
| `FrameCamera` | The camera as plain data: position, view and projection matrices, near and far. |
| `FrameScreen` | Screen buffers (depth, normals, motion) for screen-space techniques. DDGI ignores them. |
| `OptionalScreenChannel` | The `FrameScreen` channels a technique asks for by name: `'velocity'`, `'albedo'`, `'instance'` (RC1.1), `'lighting'` (the host's own lit colour before GI, linear HDR; RC4.2). |
| `ScreenChannel`, `SCREEN_CHANNELS` | Every screen channel by name (`'depth'`, `'normal'`, `'velocity'`, `'albedo'`, `'instance'`), and the list of them. |
| `ScreenInputsView` | An engine-free debug view: one `FrameScreen` channel drawn into an `rgba8unorm` texture of display values, for a host to show (RC1.1). |
| `ScreenInputsViewSettings`, `DEFAULT_SCREEN_INPUTS_VIEW_SETTINGS`, `SCREEN_INPUTS_VIEW_FORMAT` | Its settings (velocity scale, depth range), their defaults, and its texture format. |
| `instanceDebugColor`, `instanceDebugHash` | The colour `ScreenInputsView` gives an instance id, and the hash behind it. |

### Reading GI while shading

| Name | What it is |
| --- | --- |
| `ShadingModule` | `gi.shading`: the WGSL, its `parameters`, `globalsCode(group)`, `writeUniforms`, `version` ([the adapter guide, step 9](../guides/adapter.md#step-9-sampling-gi-in-your-shaders)). |
| `ShadingParameter` | One resource the module's functions take: a texture, a sampler or a `vec4f`. |
| `DebugView` | A debug view a technique offers: a screen view by id, or points (probes) in a texture, optionally with the shading-module function that gives each point's own light (`irradianceFunction`). |

### Stats and GPU timing

| Name | What it is |
| --- | --- |
| `GIStats` | `gi.stats`: frame number, pass timings, memory, counters, CPU time of `encode`, and `recompiles`. |
| `MemoryReport` | GPU memory by part: scene copy, acceleration structures, textures, each technique, total. |
| `PassTiming` | One timed GPU pass in `stats.passes`: `name`, `gpuMs`, `frame`. |
| `GpuTimer` | The timestamp-query timer the platform uses; also usable on its own. Results arrive a few frames late. |
| `GpuTimerOptions`, `GpuTimerSample`, `GpuTimerStatus` | Its options, one frame's results, and its state (`'unsupported'`, `'pending'`, `'ready'`, `'failed'`). |
| `GpuTimerScope` | The part of the timer a technique sees: `computePass(name)`. |
| `TimestampRing`, `TimestampSlot`, `TimestampSlotState`, `PassQueryIndices` | The timer's ring of query slots, so frames don't overwrite results not yet read. |
| `timestampsToPasses`, `totalByName`, `TimedPass` | Turning raw timestamps into pass times, and adding up passes with the same name. |

### For writing a technique

A technique gets the platform's services when attached, and never sees the host. These are described under "What techniques see" below and in the walk-throughs.

| Name | What it is |
| --- | --- |
| `GITechnique` | What a technique implements: `attach`, `prepare`, `encode`, its `shading` module, `settings`, `debugViews`, `dispose`... |
| `TechniqueRequirements` | What it needs: query kinds, ray batches, screen buffers, a fallback, device limits above the defaults. |
| `ScreenChannelRequest` | `TechniqueRequirements.screen` in its long form: which optional screen channels it reads (RC1.1). |
| `TechniqueContext` | What it gets when attached: the device, the scene copy, queries, hit shading, the kernel builder, the timer. |
| `TechniqueFrame`, `TechniqueScreenFrame` | What its `encode` (and screen stage) gets each frame: the camera, the changes, its share of the budget. |
| `WorkFloorRate` | What its fewest units per frame may depend on: the frame rate and the calibrated batch. |
| `SceneCopy` | The scene as techniques read it: WGSL accessors, counts, memory, the first directional light. |
| `SceneFingerprint` | `SceneCopy.fingerprint()`: content hashes of the scene by part (geometry, materials, textures, lights, sky), which saved bakes match against. |
| `fingerprintDifferences(saved, current)` | Which parts (geometry, materials, textures, lights, sky) of two `SceneFingerprint`s differ; every part when their versions differ. For saved bakes. |
| `halfToFloat(bits)` | A binary16 value's 16 bits to a number; reads saved bakes' half-float fields. |
| `SceneChanges`, `DirtyRegion`, `ChangeReason`, `LightChange` | The change feed ("Moving, adding and removing objects" above). |
| `ChangeEventTracker`, `ChangeEvent`, `ChangeEventOptions` | For techniques: the change feed as events over time (`change-events.ts`): one event per source with a box grown to cover it, epochs at its start, every `rearmFrames` while it goes on and once when it stops, so a technique forgets history once per epoch rather than on every frame of a change that goes on (the radiance cache's E8, E9 and D46). |
| `SceneQueries`, `QueryKind`, `RayQueries` | Scene queries; v1 has one backend, `rays` (`giTraceClosest`, `giTraceAny`, batches). |
| `HitShading`, `HitShadingSettings` | Shared hit shading: the surface at a hit, direct light with shadows, the sky; and its settings. |
| `DIRECT_SKY_LIGHT_BIT`, `DIRECT_SKY_LIGHT` | The sky as a light in hit shading's one-light choice (RC3.4): the group-mask bit that makes `giDirectChooseLight` weigh the sky (`1 << LIGHT_GROUP_COUNT`, so group bits alone never set it), and the light index it returns for the sky. |
| `DirectLightFrame`, `DeferredDirectLight`, `DirectWorkList`, `RayScratch` | Direct light deferred to one platform pass (`gi/direct`), and its scratch buffer. |
| `BounceSource` | Another technique's shading module used for bounce light. |
| `WgslModule`, `BindingSlot`, `ModuleBindingEntry` | A piece of WGSL and the resources it binds. |
| `KernelBuilder`, `ComputeKernel`, `ComputeKernelOptions` | How WGSL modules become compute pipelines, within the binding limits. |
| `RAY_BYTES`, `HIT_RECORD_BYTES`, `HIT_RECORD_WORD` | Sizes and layout of the buffers `encodeBatch` reads and writes. |
| `GpuComputeUtilities` | `TechniqueContext.compute`: the platform's GPU utilities for work whose size only the GPU knows (RC1.3). |
| `GpuPrefixScan`, `PrefixScanOptions` | An exclusive prefix sum of `u32`s, recorded into the caller's pass, and its options. |
| `GpuCompaction`, `CompactionOptions` | Stream compaction: the kept items' indices, in order, and their count on the GPU; and its options. |
| `GpuIndirectArgs`, `IndirectArgsOptions` | Indirect dispatch arguments made from a count the GPU computed; and its options. |
| `GpuWord` | One `u32` a pass wrote on the GPU (a total, a count) for later passes to read. |
| `GpuScreenHistory`, `ScreenHistoryOptions`, `ScreenHistoryTextures` | `compute.createScreenHistory`: the previous frame's depth (`r32float`) and normals (decoded, `rgba16float`), copied by one compute pass at the end of a frame, for techniques that reproject (RC4.1); and its options and textures. |
| `GpuScreenSpaceGI`, `ScreenSpaceGIOptions`, `ScreenSpaceGIInputs`, `ScreenSpaceGISettings`, `ScreenSpaceGITextures`, `DEFAULT_SCREEN_SPACE_GI_SETTINGS` | `compute.createScreenSpaceGI` (`gi/ssgi`): Capsaicin's SSGI from the depth buffer, one compute pass a frame: the ambient visibility and bent normal (`rgba16float`, world space, signed) and the near-field irradiance from a lighting image the caller gives (RC4.2; the radiance cache reads it, DDGI could). |
| `ComputeReduction` | How a workgroup combines its threads' values: `'subgroups'` or the `'workgroup'` memory fallback. |

## Every export of `@pmndrs/gi/ddgi`

Every name `@pmndrs/gi/ddgi` exports: DDGI, the first technique on the platform. It runs on a platform from `@pmndrs/gi/core` (`gi.addTechnique(new DDGITechnique(settings))`).

### DDGI

| Name | What it is |
| --- | --- |
| `DDGITechnique` | DDGI. `new DDGITechnique(settings)`; then `settings` (live), `field`, `counters`, `debugViews`, `shading`, `setGroupWeight`, `setEnvironmentWeight`, `resetHistory`, `resetProbes` ("Resetting offsets and states" below). |
| `ProbeResetRequest` | What `DDGITechnique.resetProbes` takes: `{ offsets?: boolean; states?: boolean }`, each false unless given. |
| `DDGISettings` | Every DDGI setting. Most are read every frame; the volumes, tile sizes, `maxProbesPerFrame`, `lightGroups.count` and `sunLayer` need a new `prepare`. |
| `DDGISettingsInput` | What the constructor takes: a `volume` or `volumes`, and any other settings. |
| `DDGILightGroups` | The type of `DDGISettings.lightGroups`: how many groups get a layer of their own (a new `prepare` when it changes), and their live weights. |
| `DDGIVolumeSettings` | One box of probes: `center`, `size`, `spacing`, `priority`, `follow`. |
| `DEFAULT_DDGI_SETTINGS` | The defaults (the `high` preset). |
| `DEFAULT_WARM_UP_PASSES`, `DEFAULT_WARM_UP_MAX_SECONDS`, `DEFAULT_WARM_UP_MIN_PASSES` | The warm-up's defaults: 16 passes over every probe, stopping after 8 s, but not before 2. |
| `DDGIField` | What `ddgi.field` gives test tools: the textures, the volume table, and each volume's grid. |
| `probeGrid(volume)`, `ProbeGrid` | The probe grid a volume setting describes: counts, origin, spacing. |
| `PROBE_STATE`, `PROBE_STATE_LABELS` | Probe states (active, off inside geometry, off with no surface nearby, not updated yet) and their names. |
| `PROBE_HISTORY_WORD`, `PROBE_HISTORY_WORDS` | The layout of each probe's history record, for tools that read it. |
| `DistanceFormat` | The distance texture's format: `'rg16float'` with `texture-formats-tier1`, otherwise `'rgba16float'`. |

### Reading DDGI while shading

| Name | What it is |
| --- | --- |
| `DDGIShadingModule` | DDGI's shading module (`ddgi.shading`). |
| `DDGI_SHADING_PARAMETER_NAMES` | The names of DDGI's shading parameters. |
| `DDGI_SAMPLE_WGSL` | The one WGSL function that samples DDGI's field, for engine-free test views. |
| `DDGI_DEBUG_VIEW` | DDGI's debug view ids: `points`, `giOnly`, `coverage`, `probeStates`. |
| `DDGI_PROBE_IRRADIANCE_FUNCTION` | `'giProbeIrradiance'`, the shading-module function the `points` view names for each probe's own light. |
| `DDGI_DEBUG_COLORS` | The colours its debug views use, for checks. |

### Quality presets

| Name | What it is |
| --- | --- |
| `ddgiPreset(name)` | A preset as plain data: `ddgi` settings, `budget`, `textureLayerSize`, `probeSpacing` ("Quality presets" above). |
| `recommendPreset(adapterInfo, limits, hints?)` | Which preset to start from on this GPU. Advice only. |
| `DDGI_PRESET_NAMES`, `isDDGIPresetName(value)` | The names, and a check that a string is one. |
| `DDGIPreset`, `DDGIPresetName`, `DDGIPresetSettings` | A preset, its name, and the DDGI settings it sets. |
| `PresetAdapterInfo`, `PresetAdapterLimits`, `RecommendPresetHints` | What `recommendPreset` reads: the adapter's info and limits, and optional hints (`mobile`, `cpuCores`). |

### The spherical-harmonics export

| Name | What it is |
| --- | --- |
| `exportIrradianceSH(technique, options?)` | One volume of the field as L2 spherical harmonics ("Exporting the field" above). |
| `ExportIrradianceSHOptions`, `IrradianceSHGrid` | Its options (`volume`, `fillInactiveProbes`) and its result. |
| `projectIrradianceToSH`, `fillInactiveProbes`, `l2ShBasis`, `l2CosineLobe`, `shProjectionDirections` | Its pieces, on the CPU. |
| `EXPORT_SH_DIRECTIONS`, `L2_SH_COEFFICIENTS`, `L2_SH_FLOATS_PER_PROBE` | 1,024 directions per probe; 9 coefficients; 27 floats per probe. |

### Saved bakes

| Name | What it is |
| --- | --- |
| `LoadFieldOptions`, `LoadedField` | `DDGITechnique.loadField`'s options (`ifSceneChanged`: `load stale` or `refuse`) and what it returns (`stale`, `changedParts`, `savedAt`, `library`) ("Saved bakes" below). |
| `SavedFieldError`, `SavedFieldProblem` | The error a refused file throws, and its `problem`: `not a saved field`, `format version`, `layout` or `scene changed`. |
| `SAVED_FIELD_FORMAT_VERSION` | The `.ddgibake` format version this code writes and reads (1). |
| `decodeSavedField`, `SavedFieldHeader`, `SavedFieldLayout` | Reads a file's header and sections without a GPU (to show what a bake holds), and the header's types: the layout that must match, the fingerprint, the camera windows. |

## Design choices, and why

### The scene

- **Four material ideas.** These are albedo (color and/or texture), emission, alpha cutout, and whether the surface blocks light. M4 adds metalness (it scales the albedo), opacity and an alpha mode. `doubleSided` is there too, because back-face hits are how probes learn they sit inside a wall. Everything else about an engine material is ignored. That keeps adapters small.
- **Textures** come in as any image source, a same-device `GPUTexture`, or compressed mips (`CompressedPixelData`, M4). The platform always makes its own downsized copy, a layer with mips in its texture store (256² by default, or 128² or 64² with `GIPlatformOptions.textureLayerSize`), so the host can free the source.
- **`contributesToGI`** per instance. Characters and small props still *receive* GI through the shading module, but rays ignore them, which is what shipping engines do.
- **Deforming meshes** keep their triangles. The host hands over a GPU buffer of positions each frame, and the BVH is refit on the GPU from it.
- **Lights** use three.js's physical units: directional intensity is irradiance, point and spot intensity is radiant intensity, and emission and sky are radiance. So the three adapter passes values straight through. `range` is required for point and spot lights. `range: 0` means "no limit", as in three.js (`distance = 0`): such a light is never culled, so it costs every probe. Light **group** membership (0–3) is scene data, so it lives on the light. Group *weights* belong to the technique.
- **Nothing recompiles.** Every table (meshes, instances, materials, lights) is fixed capacity and grows by reallocating, never by changing shader code. `gi.stats.recompiles` counts the compute pipelines made while the platform is ready, so a host can see that it stays 0 (T2.M1).
- **Moves say how far they went.** A moved instance's two dirty regions (where it was, where it is) carry `moved`, the farthest any corner of its mesh's box travelled, and its slot (`instance`), so a technique can size the change and follow the object (DDGI's geometry marks and mover mask, docs/design/ddgi.md).

### What techniques see

- **The scene copy** is read through WGSL accessor functions (`giInstanceTransform`, `giInstanceTriangle`, ...), so the table layouts can change without breaking techniques.
- **`SceneChanges`** lists the dirty world-space boxes since the last frame, with a reason: geometry, material or light. A moved object gives two boxes, where it was and where it is now. Changes with no useful box (the sun, the environment) are flagged as global. This is what lets a technique update "probes near a change first".
  - *Since T2.4a* it also lists every changed light, `lights: LightChange[]`: the light before and after (`null` for created or destroyed), its index in the light table, and its **influence radius** on each side: where its irradiance, without three's range window, falls below 1/1,000 of the scene's reference (the brightest directional light's, or the brightest local light's at 1 m), capped by `range`. A light's box is that radius, so a light with `range: 0` is local now, not global. The light table starts with room for 64 lights and doubles as it fills (4,096 is 256 KB); growing rebuilds bind groups, never pipelines (T2.1, `platform-recompiles`). Up to 1,024 a frame; past that the list is empty, `lightsOverflowed` is true and the frame counts as global (`'lights'`). Several edits of a light in one frame arrive as one change, and a light that ends where it began is left out. Technique-facing; hosts never see it.
- **Scene queries are a backend, not a BVH.** v1 has `rays`: `giTraceClosest(ray) -> GiHit` and `giTraceAny(ray) -> bool`. Techniques can call them inside their own kernels, or run a batch from buffers. A host could swap in another tracer that keeps the same WGSL signatures.
- **Hit shading** is shared: `giSurfaceAt(ray, hit)`, `giDirectIrradiance(surface, groupMask, random)`, `giSkyRadiance(direction)`. Direct light caps the number of shadowed lights per hit and picks the rest stochastically, so hit cost stays flat as the light count rises toward the 32-light minimum. `bounceAlbedoCap` exists so we can match native RTXGI (0.9) when comparing; like native's, it caps the albedo of the bounce term only (`giBounceAlbedo`), not of direct light.
- **The sky** (T1.4; re-baked on the GPU since T2.6a, `platform/sky/`): `setEnvironment` is baked into a 256 × 128 radiance map for rays that miss, and the sky's harmonics integrated over that map: irradiance SH (bands 0, 1, 2, 4) for `giSkyIrradiance`, and radiance SH (bands 0-2) for techniques that follow the sky's shape. A `procedural` sky is evaluated on the GPU (Preetham, calibrated against open ground as the sky files are), so time of day can re-bake it every frame: it re-bakes when the sun has turned more than 0.05° since the last bake, or turbidity, `intensity` or `sunIlluminance` changed (its cost: PROGRESS, T2.6a). Other sources re-bake on every `setEnvironment`. `HitShading.skyChange` gives kernels the radiance SH now and when the techniques were last handed an environment change, so a technique can size a sky change on the GPU (DDGI's mark does).
- **Bounce light comes from a bounce source,** which is any technique's shading module. For DDGI it's its own previous field (unlimited bounces). For a stacked technique it's the fallback technique below it.

### Building kernels, and the binding budget

WGSL has no imports, so every service hands over a `WgslModule`: some source text plus a list of the resources it binds.

**Why the platform's modules share one bind group:** a default WebGPU device allows only **4 bind groups** per pipeline. A DDGI trace kernel uses the scene copy, the ray queries, hit shading, a bounce source and its own data. That's five sources. So the kernel builder packs every platform module into group 0, numbering the bindings consecutively, and the technique gets groups 1–3. Modules also report how many storage buffers they use, because a default device allows only **8 per shader stage**. The builder throws a clear `GILimitError` instead of letting the pipeline fail to compile.

Every platform WGSL name starts with `gi`/`Gi`, so modules don't collide.

### The shading module (the engine seam)

This is the part the three.js and Babylon spikes test. Two engine facts shape it:

1. **Only textures, samplers and a few `vec4f` uniforms.** three r187 can bind our `GPUTexture` (`ExternalTexture`) but not our buffers, and compatibility-mode devices may allow no storage buffers in fragment shaders. So anything a pixel reads is a texture. Lookup tables, such as the list of probe volumes, are stored as small `rgba32float` textures. That keeps the uniform count fixed however many volumes there are.
2. **Resources are function parameters, not globals.** three's `wgslFn` and Babylon's material plugins each declare resources their own way. So the core's WGSL functions take every resource as an argument, in the order the module lists them. Each adapter declares the resources its own way and calls through. A plain WebGPU renderer that wants globals gets them from `globalsCode(group)`, which writes the `@group/@binding` lines and wrapper functions.

The functions are:

- `giIrradiance(position, normal, viewDirection, screenUV, ...)` returns irradiance E. The renderer applies albedo/π.
- `giCoverage(position, normal, ...)` returns 0..1. Renderers fade to their environment light where this is low.
- `provides.worldSpaceIrradiance` (P6.3) says `giIrradiance` answers at any world point from the position and normal alone, `screenUV` unused (DDGI: true; the radiance cache: false). A technique that reads another as its bounce source in compute (the radiance cache's `secondBounce`) requires it.
- `giRadiance(position, direction, roughness, ...)` is optional. DDGI doesn't provide it yet (`provides.radiance` is false): a rough-only fallback is only right at roughness 1, and anything smoother needs a blend with the host's environment specular. It isn't built yet: it is on SPEC's after-v1 list, as [issue #31](https://github.com/pmndrs/gi/issues/31).
- `giDebugColor(position, normal, view, screenUV, ...)` drives the debug views.
- A technique may add its own functions. DDGI's `giProbeIrradiance(point, direction, ...)` returns the irradiance one probe holds for a surface facing `direction`, from that probe's tile alone (`point` is its index in the `points` debug view). The `points` view names it in `irradianceFunction`, so an adapter can shade each probe's sphere by its own light, RTXGI's standard probe view (TU.5, DV-03).

`screenUV` is there for screen-space techniques (see the surfel walk-through). DDGI ignores it. `giDebugColor` got it in RC2.2 (revision below), so the radiance cache can draw its probe atlas, probe mask and SH views by the pixel's place.

## Walk-through 1: a voxel GI technique

Something like three's VXGI: voxelize the scene, inject direct light into the voxels, and cone-trace for indirect light.

| Step | What it uses | Fits? |
| --- | --- | --- |
| Voxelize the scene | Loops over `giInstanceTriangle` from the scene copy in a compute kernel. No rasterizer and no engine involved. | Yes |
| Re-voxelize after changes | `SceneChanges.regions` with reason `geometry`: re-voxelize only the bricks those boxes touch. | Yes. This is cheaper than three's VXGI, which re-voxelizes everything. |
| Inject direct light | Builds a `GiSurface` per surface voxel and calls `giDirectIrradiance`. Shadow rays come from `giTraceAny`. No engine shadow maps needed. | Yes |
| Multiple bounces | Uses its own previous voxel radiance as its bounce source. | Yes |
| Light groups | Passes `groupMask` to direct light, one voxel radiance layer per group. | Yes, if memory allows |
| Show it | The shading module samples 3D textures (`dimension: '3d'`) at chosen mip levels for cone tracing. | Yes, but see gap 1 |
| Cheaper gather at half resolution | Cone-traces in `encodeScreen` into a screen texture, then the shading module reads it by `screenUV`. | **Was a gap, now fixed:** added `encodeScreen`, `FrameScreen` and `screenUV`. |

Gaps:

1. **Texture count per material.** Anisotropic voxels want 6 3D textures. three materials already use many texture slots (maps, shadow maps), and the default limit is 16 sampled textures per stage. **Rule for now:** a shading module uses at most 6 textures. DDGI needs about 4: irradiance, distance, probe data and the volume table.
2. **Voxels as a shared query backend.** A first voxel technique can voxelize on its own. If a second technique wants voxels, voxelization moves into a `voxels` query backend. The `QueryKind` type already reserves the name.

## Walk-through 2: a surfel GI technique

Something like EA's GIBS: spawn surfels on visible surfaces, trace rays from them, and gather per pixel from nearby surfels.

| Step | What it uses | Fits? |
| --- | --- | --- |
| Spawn surfels where the screen lacks coverage | Needs depth and normals every frame. | **Was a gap, now fixed:** `FrameScreen` plus the `encodeScreen` stage, with `requirements.screen = true`. |
| Attach surfels to objects so they move along | Needs to know which instance covers each pixel, then stores instance plus local position and uses `giInstanceTransform`. | **Was a gap, now fixed:** optional `FrameScreen.instance` (`r32uint`). This is extra adapter work, so only techniques that need it ask. |
| Trace rays from surfels | `giTraceClosest`, `giSurfaceAt`, `giDirectIrradiance`. | Yes |
| Bounces, and rays that leave surfel coverage | Uses DDGI as the fallback (`setFallback(surfels, ddgi)`), and DDGI is encoded first. | Yes. This is the "techniques stack" rule from SPEC. |
| Per-pixel gather | Surfels live in buffers with a hash grid, and fragments can't read buffers in three. So the gather happens in `encodeScreen` into a screen texture, which the shading module reads by `screenUV`. | Yes, with the fixes above |
| Upscaler | The screen pass runs at render resolution and can reuse the upscaler's motion vectors (`FrameScreen.velocity`). | Yes |

Gaps:

3. **Getting screen buffers out of three.** The three adapter would need a depth/normal (and maybe instance-ID) pre-pass, and must run `encodeScreen` between that pass and lighting. That's adapter work, not a contract problem. It only matters when we build a screen-space technique, after v1.
   - *Closed by RC1.1 (2026-11-09):* `screenInputs(scenePass, threeGI)` adds the outputs to the host's own MRT scene pass and runs `encodeScreen` after that pass and before the node that reads the result (docs/api/three.md, "Screen inputs"). No pre-pass: the host's pass writes them. three's targets are handed over without a copy through one version-pinned backend read (DECISIONS 2026-11-09, register row D13).

## Other open gaps

4. **Deformed vertices from three.js.** three's skinned and morphed positions aren't exposed as a GPU buffer. Decided in M3 (m3-moving.md §7): the CPU form of `setDeformedPositions` first (T3.1), a storage `ExternalTexture` or an upstream `ExternalBufferAttribute` for the GPU form later.
   - *Answered in M3 (T3.3):* `setDeformedPositions` also takes positions on the GPU, as a buffer or an `rgba32float` texture (`DeformedPositionsTexture`). The three adapter poses skinned meshes with three's own `computeSkinning` into a storage `ExternalTexture` and hands that over. See "Deforming meshes" above.
5. **Compressed textures** (KTX2). `TextureSource` accepts a `GPUTexture`, but getting three's `GPUTexture` means reaching into the renderer's backend, which we've ruled out. Alternatives: decode the KTX2 ourselves, or have the adapter keep the original image. To decide in M4.
   - *Answered in M4 (T4.1):* `CompressedPixelData` (`{ format, levels }`, what three's `CompressedTexture.mipmaps` holds). The platform uploads the one level nearest above 256² and the GPU decodes it. See "During M4" below.
6. **Splitting the budget between stacked techniques.** For now the platform splits the millisecond budget evenly between techniques and turns each share into units with that technique's measured `msPerUnit` (T1.5b). This needs a real policy once there are two techniques.
7. **Storage-buffer budget for tracing kernels.** The scene copy (about 3), the BVH (about 2) and lights (1) leave about 2 of the default 8 for the technique. T0.10 must count the real numbers. If it's too tight, `requiredLimits` asks for more where the adapter has it (most desktop adapters allow 10 or more), and we pack tables into fewer buffers.
8. **Texture tiling (UV transforms).** Engines tile, offset and rotate textures, and may read a second uv set. Without that information, the GI copy of a tiled floor samples the wrong texels.
   - The contract now has optional `albedoUv` and `emissiveUv` (`UvMapping`: uv channel, a 3×2 matrix, and wrap per axis) and a second uv set, `MeshDescriptor.uvs1`. From three.js the adapter passes `texture.matrix` and `texture.channel` straight through.
   - Still to decide in M4: how the texture atlas honours `repeat` and `mirror`. A tile in an atlas can't use the sampler's own wrapping, so it needs wrap in the shader plus padding, or a texture array instead of an atlas.
   - *Answered in M4 (T4.1):* a texture array, one layer per texture. `repeat` is the sampler's, `clamp` and `mirror` are worked out on the uv in the shader. See "During M4" below.
   - Found by the parent session while answering Dennis, 2026-11-04.
9. **Smooth normals at ray hits.** Hit shading used the triangle normal only, so low-poly curved surfaces lit in facets. `MeshDescriptor.normals` is now optional, and `GiSurface.normal` interpolates it when present. Front/back tests keep the triangle normal.
   - Still to decide in M1: whether to store normals compressed (octahedral, 4 bytes per vertex instead of 12) once the scene copy's memory is measured.

## Questions for the spikes

- **T0.8 (three.js):**
  - Can a `wgslFn` take `texture_2d_array<f32>`, `texture_3d<f32>`, `sampler` and several `vec4f` parameters fed from `ExternalTexture` and uniform nodes?
  - Does the `Light`-node path call such a function under all three drawing pipelines?
  - If parameters don't work, the fallback is for the adapter to generate TSL texture nodes from `ShadingModule.parameters` and to inline the WGSL as a code node. Either way, the contract's parameter list is what the adapter needs.
- **T0.9 (Babylon):** Can a material plugin declare our textures and uniforms (`getSamplers`, `getUniforms`), wrap a `GPUTexture` from our device, and paste our parameter-style WGSL?
- **T0.10 (BVH):**
  - Are the `GiRay`/`GiHit` structs and the inline `giTraceClosest`/`giTraceAny` signatures enough?
  - What is the real storage-buffer count?
  - Should a batch mode exist at all, or is inline tracing always better?
  - Answered under "After T0.10" below.

## What happens after the spikes

The answers go in PROGRESS. Any change to the contract is made in both the TypeScript and this page, and listed in a "Revisions" section here.

## Saved bakes (DDGI, T5.2)

A DDGI field can be saved and loaded later, so a page starts with finished GI and keeps updating from there (docs/design/m5-coverage.md §2). It is DDGI's, not the platform's: `DDGITechnique.saveField()` and `loadField(bytes, options?)`.

```ts
const bytes = await ddgi.saveField();               // Uint8Array, a .ddgibake file
// a fresh platform, the same scene and DDGI settings:
await gi.prepare();
const loaded = ddgi.loadField(bytes);               // before the first frame: no warm-up passes
if (loaded.stale) console.log('the scene changed since:', loaded.changedParts);
```

- **Saving** copies the atlases and histories out and reads them back once, in a submit of its own. It is an explicit host call; nothing reads back in the frame loop.
- **Loading** writes the sections with `queue.writeTexture` and `writeBuffer`, which land before the host's next submit. It compiles nothing and makes no new GPU objects (`stats.recompiles` doesn't move). The technique waits for the ray-tracing trees, as its warm-up does, and then carries on updating with no warm-up passes. The counters `saved fields loaded` and `saved field stale` say what happened.
- **Refused** with `SavedFieldError` (its `problem`: `not a saved field`, `format version`, `layout`, `scene changed`) and a plain message: a file that isn't one, a format version this code doesn't read, or a layout that doesn't match (the message names the field: tile texels, the irradiance layers of light groups and the sun layer, a volume's counts, spacing, place, priority or `follow`). A field saved on a device with `rg16float` distances loads on one without (and back): the texels are converted.
- **A changed scene** (its fingerprint differs) loads as stale unless `ifSceneChanged: 'refuse'`: the first frame that updates probes marks every probe as a whole change, so the field updates from the loaded values at priority instead of from black.
- **The scene changes the next frame brings aren't marked:** the fingerprint was checked against the scene with them, and on a fresh platform they are its creation (every object and light added), which would cut every loaded history. So load after the scene is described and before the first frame; a change made between `loadField` and the next frame goes unmarked.
- **A camera-following volume's window is saved** (where it was on the lattice, and its valid extent) and put back at load. It moves on with the camera from there; a camera far from it is a teleport, which starts that volume over as any teleport does.

**The file (`.ddgibake`), little-endian:** `DDGB`, a `u32` format version (1), a `u32` header length, the header as UTF-8 JSON, then the sections, each starting on a 256-byte boundary (`techniques/ddgi/saved-field.ts`).

- The header: `formatVersion`, `library` (`@pmndrs/gi 0.0.0`), `createdAt`, `layout` (what has to match: tile texels, irradiance layers, the distance format, each volume's counts, spacing, first probe, priority and `follow`), `sections` (name, format, byte offset and length, and a texture's width, height and layers), `fingerprint`, `clockSeconds` (the technique's clock, which the histories' last-update times are on), and `cameraWindows`.
- The sections: `irradiance` (every irradiance layer side by side, `rgba16float`), `distance`, `probeData` (relocation offsets and states), and `probeHistory/<volume>` for each volume (64 bytes a probe). Texture rows are packed with no padding. The composed atlas isn't saved: the first frame after a load composes it.
- The probe history is settled before it is written: it keeps each probe's sample count, update count, last-update time, mean and noise, and drops what only meant something in the frames it was saved in (owed updates, marks, wakes, frame stamps, the change clock).

**The scene fingerprint** is `SceneCopy.fingerprint()` (technique-facing, platform/scene/scene-fingerprint.ts): a 64-bit hash per part.

| Part | What is hashed |
| --- | --- |
| `geometry` | Each GI instance's mesh content (positions, indices, normals, uv sets, groups, triangle count, deform box), transform and visibility. A deforming mesh counts with its rest pose. |
| `materials` | Each GI instance's material list: every number of each material, its textures by content hash. Labels are left out. |
| `textures` | The content of every texture those materials use: the bytes of `PixelData`, `ImageData` and compressed levels (over 1 MB, an even 1 MB sample plus the length). `ImageBitmap`, canvases and `GPUTexture`s count by size, format and colour space only; image elements by URL and size. |
| `lights` | Every light's numbers and shadow parameters (`castsShadows`), and its group. There are no shadow matrices to leave out. |
| `sky` | The environment's settings; a texture environment by its texture's size and format, not its texels. |

Each part is order-free, so two loads that create objects in another order agree. Bump `FINGERPRINT_VERSION` when what a part covers changes.

## Resetting offsets and states (DDGI, DV-12)

`resetHistory()` forgets everything: every probe starts again from its grid point, unlit, and the warm-up runs again. DDGI also has a lighter reset for the two things relocation and classification decide, for when the geometry around the probes changed in a way they won't undo by themselves. It is DDGI's, not the platform's: `DDGITechnique.resetProbes(what)`. The probes keep their irradiance, distances and history, and there is no warm-up.

```ts
ddgi.resetProbes({ offsets: true });                // every probe back to its grid point
ddgi.resetProbes({ states: true });                 // every probe that was switched off comes back on
ddgi.resetProbes({ offsets: true, states: true });  // both
ddgi.resetProbes({});                               // nothing: each flag is false unless given
```

- **Offsets:** each probe's relocation offset goes back to 0 and its state stays. Relocation moves it again over its usual updates. A probe that relocation had moved out of geometry is inside it again until its next update, is switched off for that update (as at the start), and comes back on from where relocation moves it. A probe that is off for being inside geometry stays put, since relocation has had its go at it; reset the states too to give it another.
- **States:** each probe that classification switched off ("inside geometry" or "no surface nearby") becomes "not updated yet", which is how an off probe always comes back on. Its next update traces all its rays and classification decides again; if it stays on, it starts its light from nothing and shading uses it after two blended updates. Active probes and probes not updated yet keep their state, since classification looks at them at every update anyway. Offsets stay. RTXGI's reset makes every probe active instead; ours goes through "not updated yet" so shading never reads an off probe's stale tiles or a tile that was never blended. So in the frame of a states-only reset, nothing renderers see changes.
- **When:** in the next `encode`, before any update, as one small GPU pass (`ddgi/reset-probes`). The offset and the state share each probe's probe-data texel and the CPU knows neither, so the pass reads a copy of that texture and writes it back with the `points` debug view. The copy (8 bytes a probe) is made at the first reset and kept until the next `prepare`. Calls before a frame add up (`{ offsets }` then `{ states }` is both); a `resetHistory` or `loadField` after the call, before the frame, replaces it. Before `prepare` it does nothing.
- **A camera-following volume:** its planes count as warm only when few of their probes are "not updated yet", so after a states reset the valid extent waits for those probes' next update before it grows over them.

Through three, `volume.resetProbes(what)` (three.md).

## Revisions

### GPU mesh trees are final, 2026-11-28 (pmndrs/gi#26)

- **Technique-facing meaning of `RayQueries.building`:** true until every mesh has its tree, the GPU's (H-PLOC) or, where the GPU gives none, tinybvh's (the declared fallback). No tree replaces a mesh's tree of the same version any more, so there is no second, "refining" phase; a deforming mesh's drift rebuild still doesn't count. No WGSL or binding changed.
- **Host-visible:** `stats.counters['platform mesh trees']` loses "GPU trees replaced by tinybvh" (always 0 now). "meshes on tinybvh trees" counts fallback trees only. `stats.memory.acceleration` is smaller where the GPU's trees filled the node buffer: it no longer grows with an eighth of headroom for the swap.
- **Implementation-only, for the repository's tools:** `MeshTreeOptions.holdRefinements`, `rays.releaseRefinements()`, `rays.whenRefined()`, `rays.refining` and `MeshTreeCounts.swaps` are gone. `gpuFirst: false` still forces the fallback (M1's path), so tools test it on a device with subgroups. The page `ddgi-tree-swap` became `ddgi-gpu-trees`.

### Apple's storage-buffer limit, 2026-11-26 (pmndrs/gi#49)

- **Technique-facing: `TechniqueRequirements.limits`** (optional, `RequiredLimits`): device limits a technique's kernels need above WebGPU's defaults. The radiance cache declares `maxStorageBuffersPerShaderStage: 10` (its most in any pass, packed to fit Apple GPUs' 10 in Chrome; a device three.js makes itself has 8). `prepare` throws a `GILimitError` naming the first declared limit the device is below, before compiling anything.
- **Host-facing: `GIPlatform.limitNeeds`** (read-only): the largest of each limit the attached techniques declare, so a host can check a device or pass them to `requestDevice`. `requiredLimits(adapter)` already asks for the adapter's maximum storage buffers per stage, so a host using it gets them where the adapter has them.

### Radiance-cache phase 2, 2026-11-19 (RC2.2)

- **Host- and technique-facing: `giDebugColor(position, normal, view, screenUV, ...)`.** The debug colour takes the pixel's screen position (0..1, top left 0), as `giIrradiance` has it, so a screen-space technique can draw its views by the pixel's place (the radiance cache's probe atlas, probe mask and SH). `globalsCode` writes `giDebugColorAt(position, normal, view, screenUV)`. DDGI's module takes the argument and ignores it; the viewer's debug material passes three's `screenUV`.
- **Host-facing: `FrameCamera.jitter`** (optional): this frame's sub-pixel jitter as an NDC offset (x right, y up), when the host draws its screen buffers with a jittered projection; the jittered projection is `translate(jitter, 0) × projection`. Screen techniques rebuild positions from `FrameScreen.depth` with it, so they land where the depth was drawn (RC2.1 found half a pixel's error without it). `projection` stays the unjittered one. World-space techniques ignore it. The three adapter fills it, and `projection`, from `ThreeGI.jitter`, a source the host sets when it jitters, giving the NDC jitter and the unjittered projection (for `@pmndrs/upscaler`, `upscalerJitter` after `beginFrame`: `Upscaler.jitter` of `(x, y)` pixels is `(−2x / width, +2y / height)`, as three's `setViewOffset` moves points).

### Issue #20, 2026-11-19

- **Host-facing through DDGI (DV-12; Dennis, 2026-11-19: "yes build it"): `DDGITechnique.resetProbes(what)` and `ProbeResetRequest`** ("Resetting offsets and states" above), and three's `DDGIVolume.resetProbes`. The engine-neutral `resetHistory()` is unchanged. Implementation-only: the `ddgi/reset-probes` kernel, compiled at `prepare` with the others, and its timed pass of that name in frames with a reset.

### After the M3-M4 questions, 2026-11-18

- **Host-facing (DECISIONS 2026-11-18, "Both public"): `GIPlatform.hitShadingSettings`**, typed `HostHitShadingSettings`, holds `sampleTextures` (live, default true; "Materials and textures" above). It is the same live object hit shading reads, so a change lands at the next `encode`. The other `HitShadingSettings` fields stay technique- and tool-facing. `RayQueries.capabilities.perTriangleMaterials` stays public, unchanged.

### During M5, 2026-11-10

- **Host-facing (the gi rename, DECISIONS 2026-11-10): the package is `@pmndrs/gi`, and `/core` is the platform alone.** DDGI's names (everything in "Every export of `@pmndrs/gi/ddgi`") moved from `/core` to `@pmndrs/gi/ddgi`; nothing in the platform or DDGI changed name or shape. The three adapter's `GIVolume` is now `DDGIVolume` (three.md).
- **Host-facing (T5.1): quality presets.** `ddgiPreset(name)`, `recommendPreset(adapterInfo, limits, hints?)`, `DDGI_PRESET_NAMES`, `isDDGIPresetName` and their types, from `@pmndrs/gi/ddgi` (section "Quality presets" above). Plain data over the existing settings; nothing in the frame changed.
- **Host-facing (T5.1): `GIPlatformOptions.textureLayerSize`** (`64 | 128 | 256`, default 256): the texture store's layer size, fixed for the platform's life. At 256 the store, its kernel and the sampling module's WGSL are what M4 checked, to the character. Smaller layers have one mip fewer per halving; the ray cone's mip and the wrap's half texel are worked out in the store's own texels. Implementation-only: `storeLayout(layerSize)` and `textureStoreWgsl(layerSize)` (textures/texture-store-wgsl.ts), and `PlatformTextureStore`'s `layerSize` option. The CPU mirrors and the repository's reference tracer stay at 256.
- **Technique-facing (T5.2): `SceneCopy.fingerprint(): SceneFingerprint`**, content hashes of the scene by part, for saved bakes ("Saved bakes" above). Worked out on the CPU when called. Meshes and textures are hashed once when created or updated, since the scene keeps no CPU copy of them after upload: 1.9 ns a word on the M5 Pro (48 MB in 22 ms), on the host's thread.
- **Host-facing through DDGI (T5.2): `DDGITechnique.saveField` / `loadField`**, the `.ddgibake` format, `SavedFieldError` ("Saved bakes" above). Implementation-only: the irradiance and distance atlases gain `COPY_DST` usage (the load writes them).

### Radiance-cache phase 2, 2026-11-09 (RC2.1)

- **Technique-facing: `TechniqueScreenFrame.camera` (and `frame`, `deltaTime`) are now those of the frame's `encode`.** Before, `encodeScreen` handed screen stages an identity camera, so a technique couldn't turn the host's depth back into world positions. A screen technique reconstructs positions with `camera.projection × camera.view`, so `projection` must be the one `FrameScreen.depth` was drawn with (WebGPU's 0..1 depth); hosts call `encodeScreen` after the frame's `encode`, as before. Implementation: `GIPlatformImpl` keeps the latest `encode`'s camera. Hosts that jitter their projection for an upscaler hand GI the unjittered one (`FrameCamera`), which is half a pixel off the depth; RC2.2 decides whether `FrameScreen` needs its own projection.

### Radiance-cache phase 1, 2026-11-09 (RC1.1)

- **Host- and technique-facing: `FrameScreen.albedo`** (optional): the diffuse albedo, linear rgb, base colour × (1 − metalness), the fold the adapters already apply to GI albedo for rays, so a pixel and a ray hit on the same surface agree. Any float or unorm format. It is not three's composite weight `giAlbedo`, which also carries the specular lobe's share and AO. `FrameScreen.depth` now says what formats it takes: a depth format (read as `texture_depth_2d`) or a one-channel float colour format.
- **Technique-facing: `TechniqueRequirements.screen` is `boolean | ScreenChannelRequest`**: `true` is depth and normals; `{ velocity, albedo, instance }` asks for those channels too. `encodeScreen` throws, naming the technique and channel, if a screen stage would run without a channel its technique asked for.
- **Host-facing: `requiredLimits(adapter)` also asks for the adapter's `maxColorAttachmentBytesPerSample`**: a host scene pass that writes the screen inputs beside its own outputs needs 48-56 bytes per pixel, past WebGPU's default 32 (RC1.1 review).
- **Stated in the contract: every `FrameScreen` channel describes the opaque surface under the depth sample**; a blended surface that writes no depth leaves the channels as the surface behind it (RC1.1 review).
- **Host-facing: `GIPlatform.screenNeeds`** (read-only): the union of the attached techniques' screen requirements (`{ screen, velocity, albedo, instance }`), so a host renders only what is read. DDGI asks for none, so nothing changes for it.
- **Host-facing: `SceneApi.instanceIndex(instance)`**: an instance's index on the GPU, what rays report as `GiHit.instance` and `FrameScreen.instance` holds plus 1. Fixed for the instance's life.
- **Engine-free debug view: `ScreenInputsView`** (`platform/screen/`): one channel of a `FrameScreen` drawn into an `rgba8unorm` texture of display values, for a host to show.

### During M3, 2026-11-09

- **Technique-facing meaning of `RayQueries.building`, and when the platform submits (T3.2): meshes added or replaced while the scene runs get GPU (H-PLOC) trees too**, built in the frame they are uploaded and placed after the stats readback, with tinybvh's tree swapped in later, as at load. So `building` is true while any mesh's current version has no tree yet, whether it came at load or later, and a frame that uploads meshes submits one GPU build of its own and reads back its stats (DECISIONS 2026-11-09). One tinybvh build per mesh runs at a time; a mesh replaced again meanwhile keeps only its newest version waiting. Implementation-only: `rays.memory` gains `trianglesInUse`.
- **Host-facing (T3.1): `MeshDescriptor.deformBounds` and `setDeformedPositions` with a `Float32Array`.** A `deforming` mesh may give `deformBounds`, a box in its own space that every pose stays inside; its instances keep that box in the top level, and each new pose is logged as that box (in world space, per instance). `setDeformedPositions(mesh, positions)` takes the CPU form (Dennis, 2026-11-05: yes): xyz per vertex, copied at the call, the newest of a frame kept. The GPU forms came with T3.3 (next entry). *As built for the prototype (replaced by T3.3's refit):* the pose is written into the mesh's positions in the scene copy (plain xyz floats, so no layout change) and the mesh is reported as uploaded again, so the tracer rebuilds its tree as for `updateMesh`; new poses wait while trees are being built, so a mesh posed every frame still gets trees (about one in three frames on the RTX for 1,152 triangles). A pose outside `deformBounds` grows the box, with one warning.
- **Host-facing (T3.3): `setDeformedPositions` takes positions on the GPU, and a pose reaches rays in the next frame.** `DeformedPositions` is now a buffer (`STORAGE` usage, any stride of 12 bytes or more, an offset) or an `rgba32float` texture (`COPY_SRC`, width a multiple of 16, vertex `i` at texel `(i mod width, i / width)`: the form a three.js TSL compute writes into a storage `ExternalTexture`). The mesh must have `deformBounds` for a GPU pose: the platform can't read it to grow the box, so its `DirtyRegion.moved` is the box's diagonal. Every pose (CPU or GPU) is copied into the scene copy and the mesh's tree is refit on the GPU (`GpuMeshRefit`) in the same frame's encoder: no submit, no tree build, no wait (T3.1's rebuild per pose and its hold are gone). A tree whose refit has drifted past 1.1× its built SAH cost for about 60 frames is rebuilt with H-PLOC from the scene copy's positions (DECISIONS 2026-11-09), through T3.2's runtime path, with one readback of its stats and its layout. The refit's stats (the drift signal) are read back at most once every 15 frames per posed mesh. A deforming mesh's normals are dropped: hit shading uses the posed triangle's own normal. Without subgroups (no refit) poses are ignored, with one warning, and the mesh stays in its rest pose. Implementation-only: `SceneUpload.meshesPosed`; `stats.counters['platform deforming meshes']`; the timed pass `platform/mesh refit`.
- **Technique-facing (T3.1): `SceneChanges.receivers: readonly Aabb[]`** replaces the `movedReceivers` count: the world box, where it is now, of each visible instance that doesn't contribute to GI and was created, shown, moved or deformed, up to 256 a frame (then one merged box). For waking probes in empty space (m3-moving.md §5.3). **`DirtyRegion.moved`** is also set for a deforming mesh's region: the largest vertex travel since the pose before, times the instance's largest axis scale.
- **Technique-facing (T3.4): `DirtyRegion.emits`** is true on a geometry region whose instance gives off light (a material with a non-black `emissive` and a positive `emissiveIntensity`, as its materials are when the region is logged; a merged region emits if any of its regions did). An emitter that moves or deforms changes the light also where it can't be seen, through the surfaces it lights, so DDGI marks such a change more like a light's (docs/design/ddgi.md).
### During M4, 2026-11-09

All optional and backward compatible (docs/design/m4-materials.md, "Contract additions"; T4.1).

- **Host-facing: textures are sampled at ray hits.** The platform keeps a texture store: one `texture_2d_array` of 256 × 256 layers with full mips, `rgba8unorm-srgb`, a layer per texture. A texture's layer is made in the next `encode` after `createTexture` (`gi/texture-store`, linear-light downsampling and mips). Until then its materials shade with their flat colour; when it lands, the platform logs a `material` change over the instances using them. The array grows by copying layers and rebuilding bind groups, never by recompiling. Texture coordinate (0, 0) is the source's first texel (the top-left of an image), as in WebGPU.
- **Host-facing: `CompressedPixelData`** (`{ format, levels: { data, width, height }[] }`) is a `TextureSource`. The level nearest above 256² is uploaded and the GPU decodes it. It needs the format's device feature; without it the platform warns once and those materials keep their flat colour. This answers gap 5.
- **Host-facing: new `MaterialDescriptor` fields.** `metalness` (0–1) and `metalnessTexture: { texture, channel, uv? }`: the albedo at a hit is `albedo × albedoTexture × (1 − metalness × metalnessTexture[channel])`, as three and glTF do. `opacity` (0–1) and `alphaMode: 'opaque' | 'cutout' | 'stochastic'` (with `alphaCutoff` above 0 and no mode it is `'cutout'`, as before; `'cutout'` with no cutoff uses 0.5). Rays use `opacity` and `alphaMode` from T4.2 on.
- **Implementation-facing for now: `HitShadingSettings.sampleTextures`** (live, default true; host-facing since 2026-11-18 as `GIPlatform.hitShadingSettings.sampleTextures`, above). Off, every texture is read at its last mip, its mean colour, the same over the whole surface: the flat colours M1's adapter folded in (albedo, emissive and metalness alike, so a glTF metal-mapped material isn't treated as fully metal). While a metalness map's layer is still being made, the material counts as metalness 0, so it doesn't go black while it loads. Hosts reach it as `GIPlatform.hitShadingSettings` (Dennis, 2026-11-18).
- **Host-facing: `stats.memory.textures`**, the texture store's bytes (0.35 MB a layer, the array grown in doubles), counted in `total`.
- **Host-facing: `requiredLimits` asks for `maxTextureArrayLayers`** (the default 256 would cap a scene at 256 textures; adapters give 2,048), and `optionalFeatures` for `texture-compression-bc`, `-etc2` and `-astc` where present. Since 2026-11-20 (DECISIONS) `requiredLimits` also asks for the adapter's `maxComputeWorkgroupStorageSize`, and `optionalFeatures` for `shader-f16` where present (f32 is the fallback).
- **Technique-facing: `giSurfaceAtCone(ray, hit, spread)`.** Textures are read at the mip a ray cone's footprint covers: `log2(spread × t / max(|cos θ|, 0.1) × texels per metre)`, texels per metre from the hit triangle's uv area (scaled by the uv transform's determinant) against its world area. A technique passes its rays' spread, `sqrt(4π / rays per sphere)`; DDGI passes it from `raysPerProbe`. `giSurfaceAt` uses 0.22 (256 rays). The uv mapping (uv set, 3×2 transform, wrap per axis) is applied for each texture: `repeat` by the sampler, `clamp` and `mirror` on the uv. `GiMaterial` gains `metalness`, `metalnessTexture`, `metalnessChannel`, `metalnessUv`, `opacity` and `alphaMode`, and its texture words are store layers (`GI_NO_TEXTURE` until a layer is made). The scene module gains `giInstanceTriangleUvs`. The hit-shading module binds two more resources, the store's array and its sampler, and still no storage buffer.
- Since T4.3 the three adapter passes real textures (images, data and compressed) and no longer folds texture averages.
- **Technique-facing (T4.2, first half): materials in the walk.** `giTraceClosest` and `giTraceAny` now let rays through see-through materials (`blocksLight: false`) and the back faces of single-sided ones (`doubleSided: false`), triangle by triangle, in every instance (docs/design/m4-materials.md §3). `RayQueries.capabilities.perTriangleMaterials` is true; `alphaCutouts` stays false until the second half. No WGSL signature changed. **Binding cost:** the `rays` module now depends on the scene module (the walk reads its material tables), so a kernel that lists `rays` alone binds the scene copy's 2 storage buffers too: inline tracing 3 + 2, a batch kernel 7 of 8. Kernels that list hit shading had both already, so DDGI's count is unchanged. Implementation-only: `rays.requestTriangleFlags(meshSlot)` (the install hook every path that writes tree triangles calls; trace/triangle-flags.ts) and `rays.triangleFlagRewrites`. Hosts see nothing new. Since T4.3 the three adapter no longer leaves see-through meshes out of GI; the walk handles them per triangle.
- **Technique-facing (T4.2, second half): alpha in the walk.** `RayQueries.capabilities.alphaCutouts` is true. Rays also pass a cutout where its alpha (`opacity` × the albedo texture's alpha, through the albedo texture's uv mapping) is below `alphaCutoff` (0.5 by default), and a stochastic material with probability 1 − alpha (a hash of the frame number, the instance, the triangle and the hit's barycentrics, so rays that repeat exactly from frame to frame, such as shadow rays from a still camera, average out instead of freezing into one pattern), for closest hit and any hit alike. The alpha is read from the texture store at a fixed mip, 1 (`ALPHA_TEST_LOD`), whatever `sampleTextures` says: a hole is geometry. No WGSL signature changed. **Bindings:** the material test binds one uniform (the frame number, written by the `rays` backend each update). The store's array and its sampler now come from one module of their own, `gi-texture-store` (`giTextureSampleMapped`, `giTextureMappedUv`, `giTextureWrap`; textures/texture-sampling-wgsl.ts), which hit shading and the walk both list, so a kernel with both binds them once. The `rays` module now binds them too: still no storage buffer, so a batch kernel stays at 7 of 8 and inline tracing at 3 + 2. Hit shading's own entries drop from six to four (the uniforms, the sky map, its sampler, the light grid's picks).
- **Implementation-only (T4.2, second half): the alpha-test mip is a constant, not a setting** (`ALPHA_TEST_LOD` = 1, trace/material-test-wgsl.ts): holes are geometry, and no host or technique has needed another mip.
- **Host-facing (T4.2, second half):** `MaterialDescriptor.alphaMode` and `opacity` now act on rays (above). The store's mips weight colour by alpha, so the coarse mips of cutout foliage keep the leaves' colour. A texture source that throws when read (a closed `ImageBitmap`, transferred `PixelData`) fails alone, with a warning; a `GPUTexture` source is checked for this device first (one or two frames), so another device's texture is refused with a warning instead of invalidating the host's frame.

### During M2, 2026-11-08

- **Technique-facing meaning of `RayQueries.building` (T2.12): true until every mesh has its first tree**, the GPU's (H-PLOC, built at load) or tinybvh's, whichever arrives first. tinybvh's trees replacing GPU trees afterwards don't count: rays see the same triangles either way, and the swap changes no binding layout and compiles nothing. **Host-visible: `stats.counters['platform mesh trees']`** (meshes on GPU and tinybvh trees, meshes waiting, tinybvh builds running, GPU trees built and refused, swaps, the last GPU build's time to its stats), and `stats.memory.acceleration` includes the GPU build's output buffers while they live (a few frames at load). Implementation-only, for the repository's tools: `GIPlatformImplOptions.meshTrees` (`gpuFirst`, `holdRefinements`, `gpuMaxDepth`, and `firstTreeBuilder`, a stand-in first-tree builder for unit tests in Node) and `rays.whenRefined()`, `rays.releaseRefinements()`, `rays.meshTreeCounts`. DECISIONS 2026-11-08.

### During M2, 2026-11-07

- **Technique-facing (T2.6b): the sun kept apart, for a sun layer.** `DirectLightFrame.sunLayer` names a result layer for the sun; `gi/direct` then puts the sun's direct light there at a white reference (colour 1, intensity 1) instead of into its group's layer, and the technique weighs that layer by the sun's colour × intensity, so a change of the sun's colour or brightness needs no new rays. The sun is the scene's first directional light, light 0 of the light table. Hit shading gains `giDirectSplitIrradiance(surface, groupMask, sunApart, random)` (the `GiDirectGroups` struct has a `sun` field, 0 unless asked for) and `giDirectReachesSplit(surface, groupMask, sunApart)`, which weighs the sun in white so a sun at intensity 0 still lists the hits it faces (its layer keeps updating); the old functions are these with `sunApart` false. The ray scratch header's word 11 holds the sun layer. **`SceneCopy.firstDirectionalLight`** (`{ light, descriptor }` or null) gives techniques the sun as it is now: its colour and intensity for the weight, its handle to see it become another light (it changes only when it is destroyed, or a first directional light is created). Hosts don't see this; DDGI's `settings.sunLayer` uses it.

### During M2, 2026-11-06

- **Technique-facing (T2.5): light groups in `gi/direct` and the ray scratch.** `direct.createScratch` takes `layers` (1 to 8, default 1): a result per ray per layer, ray `i`'s layer `l` at result `i × layers + l` (`RayScratch.layers`). `DirectLightFrame.groupLayers` names the layer each light group's direct light goes into (default all 0). Tracing kernels store a layer with `giStoreRayLayerResult(ray, layer, radiance, half)`; `giStoreRayResult` is layer 0. Hit shading gains `giDirectGroupIrradiance(surface, groupMask, random) -> GiDirectGroups`, the same light choice and shadow rays split by group; `giDirectIrradiance` is its sum. With one layer everything is as before.
- **Host-facing (T2.5): `GIPlatform.outputTechnique`** (read-only): the technique `setOutput` chose. **Technique-facing: optional `GITechnique.setGroupWeight(group, weight)` and `setEnvironmentWeight(weight)`**, linear RGB multipliers on a whole light group (DDGI implements them with `settings.lightGroups`). Adapters reach them through `outputTechnique` (three: `ThreeGI.setGroupWeight`).

- **Technique-facing (T2.2): `HitShading.direct`, deferred direct light (`gi/direct`; docs/design/m2-lights.md §2).** A tracing kernel no longer has to trace shadow rays itself. It keeps its rays' results in a `RayScratch` from `direct.createScratch` (one storage buffer: a header, the results, a work list), includes `direct.traceWgsl(variable, workList)`, stores results with `giStoreRayResult`, and puts each hit a light can reach (`giNeedsDirect`) on the list with `giSurfaceRecord` and `giDeferDirect`. Then `direct.encode` runs `gi/direct` over the list in one pass timed as `<technique>/gi-direct`, so it counts towards that technique's budget, and empties the list. A ray's result is two `u32`s: radiance as halves, plus one half that is the technique's own (DDGI's hit distance). `gi/direct` adds `albedo / π × direct` to the radiance. The work list is filled with subgroups first; the declared fallback is one atomic per hit (`DirectWorkList`). Hit shading also gains `giDirectReaches` (does any light of the groups light this side) and `giDirectRandom` (`gi/direct`'s random numbers for a ray). Not built yet from the design's sketch: `giDirectLightSamples` (T2.3's light grid) and the `gridBounds` argument (T2.3); `groupLayers` came with T2.5 (below). Hosts don't see this.
- **`PlatformHitShading` takes the kernel builder** (constructor), because `gi/direct` is a platform kernel compiled on a technique's first `direct.prepare()`. Platform-internal.
- **Host-facing settings (T2.3): `HitShadingSettings.exactLightLimit` (default 8), `lightGridMaxCells` (default 2,048) and `lightGridShadowedLightsPerHit` (default 1).** With more point and spot lights than `exactLightLimit`, hit shading draws them from the light grid (`platform/shade/light-grid.ts`; docs/design/m2-lights.md §3): the platform rebuilds it each unpaused frame over the scene's bounds (`gi/light-grid`, a platform pass), and `giDirectIrradiance` and `giDirectReaches` read their hit's cell. No WGSL signature changed; techniques get the grid through the hit-shading module, which now also binds the grid's picks as a `uint` texture (still no storage buffer). `lightGridShadowedLightsPerHit` then counts the shadow rays a hit traces among its cell's 16 picks (`maxShadowedLightsPerHit` still applies to the exact path). As built, the grid covers the scene's bounds, so the design's `gridBounds` argument and `giDirectLightSamples` were not added (light selection stays inside `giDirectIrradiance`, which keeps one shadow-ray call site).

### During M1, 2026-11-04

- **Host-facing (budget v2, 2026-11-05, after its review): `GIPlatform.recalibrate()`.** Measures again how much work fits in `gpuMs`, at the next millisecond frame. The platform calibrates by itself after `prepare`; a geometry change only marks the calibration stale (reported in `stats.counters['<technique> budget']`), since a burst during play makes frames heavier for a moment; a new `gpuMs` is solved from what was measured. `FrameBudget.gpuMs` now says it needs `gpuTiming` (without timings the hand-out is `units`, or 512).
- **Technique-facing (budget v2, 2026-11-05): optional `GITechnique.minUnitsPerFrame(rate)`,** the fewest units the technique wants per frame under a millisecond budget, given the host's frame rate and the calibrated batch (`WorkFloorRate`). It wins over the millisecond ceiling; `budget.units` caps it. DDGI implements it from `minProbesPerFrame`.
- **Host-facing, accepted by Dennis with budget v2 (2026-11-05): `FrameBudget.targetFps`, optional.** The frame rate the host aims for. Budget v2 (docs/design/ddgi.md) hands out a calibrated number of work units each frame and steps it down only when frames are late, so it needs the target interval; without `targetFps` it uses the rate it observes from `deltaTime` when frames are on time. Without it, an observed rate that isn't a display refresh rate (up to 165 Hz) means an uncapped host, which gets no frame-rate ceiling. It also sets the frame rate for floors given in seconds (DDGI's `minProbesPerFrame: 'auto'`, a full sweep within 4 s). `gpuMs` now means "per frame, measured with the GPU's clocks up", and `units` caps the floor too. (Budget v2 implementation, 2026-11-05.)

- **`requiredLimits` also covers compute workgroups:** it asks for 256 invocations per workgroup (and size X 256), capped at the adapter, because compatibility-mode devices default to 128. Host-facing only in that hosts get one more limit in the list they already pass through. (T1.6, 2026-11-05)

- **Technique-facing: `GITechnique.encode` may return the number of units it actually used** (T1.5b). The platform's budget meter divides the measured GPU time by that, not by the units it handed out, so a budget bigger than the work doesn't inflate the next frame's hand-out. Returning nothing means "all of them". Hosts don't see this.

- **Host-facing, approved by Dennis (2026-11-05): a procedural environment gets an optional `sunIlluminance`,** the illuminance of the sun the sky is calibrated to (default 1); `intensity` stays a plain multiplier, as for 'color' and 'texture'. It's the same quantity as pmndrs/sky's `exposure`. See DECISIONS.
- **Stated in the contract (T1.4):** the `equirect` layout (nadir at memory row 0, u = 0 at −X, as three's `equirectUV`), and the cube convention (WebGPU's faces, direction unchanged). three r187 looks cubes up at (−x, y, z), so the three adapter mirrors them.

- **Technique-facing: `ComputeKernelOptions.ownLayouts` became `ownGroups`,** bind-group layout *descriptors*, so the kernel builder can count the technique's own storage buffers too and throw `GILimitError` instead of a pipeline validation error past the device's limit. `ComputeKernel.ownLayouts` returns the layouts it made. Hosts don't see this. (T1.3 review, 2026-11-05)
- **`requiredLimits` says what's true now:** small scenes run at default limits; large ones (Sponza with curtains, Bistro exterior) need the raised limits it asks for, which are the adapter's maximum binding and buffer sizes. Nothing is split across bindings. Dennis approved raising limits (DECISIONS 2026-11-05).

- **Fix: `updateLight` takes `LightChanges`,** a union of each light type's partial fields. The old type, `Partial<Omit<LightDescriptor, 'type'>>`, kept only the keys every light type shares (that's how `Omit` treats a union), so a host couldn't update a point light's `position` or `range` without a cast. The documented behaviour (any field except `type`) is unchanged. Found by T1.2; fixed by the M1 lead, and Dennis was told.

### After T0.10 (BVH), 2026-11-04

- **Confirmed: `GiRay`, `GiHit` and the inline `giTraceClosest` / `giTraceAny` are enough.** The oracle and the benchmarks ran through exactly these. Batches write `GiHitRecord`, because WGSL `bool` can't be stored in a buffer (already in the contract).
- **The tracer's real binding cost:** inline tracing binds **3 storage buffers** (top level, nodes, triangles). A batch kernel binds **5** (those plus the ray and hit buffers). Measured on both GPUs, against a default limit of 8 per shader stage.
- **Budget for a DDGI trace kernel, on a default device (8 storage buffers):**
  - ray queries: 3;
  - scene copy and hit shading together: at most 2. Vertex attributes (normals, uvs) go in one buffer with offsets, and instances, materials and lights in one table buffer;
  - the technique's own data: the remaining 3, though DDGI mostly needs textures.
  - The scene-copy task in M1 (T1.2) holds this. If it can't, the platform asks the host for a raised limit (the host owns the device and already raises limits for three) before it packs further.
- **Both tracing modes stay. Inline is the default.**
  - DDGI traces, shades and accumulates in one kernel, so rays and hits never go to memory. The DDGI update loop (T1.5) uses inline tracing.
  - `encodeBatch` stays for techniques that want separate passes, and the oracle and benchmarks use it. It costs 2 more storage buffers.
  - *Since T2.14:* its two kernels are compiled in `prepare` only when an attached technique sets `requirements.rayBatches` (the repository's tools pass the implementation's `rayBatches` option). Otherwise `encodeBatch` throws; it never compiles on first use, so `stats.recompiles` stays 0.
  - T0.10 didn't time inline against batch. T1.5 does, if the update cost comes near its budget.
- **Change: `range: 0` on a point or spot light means "no limit".** Before, `range` had to be positive. three's `distance = 0` already means unlimited, so the three adapter passes it straight through. Such a light is never culled. This is host-facing, but it only allows a value three already uses.

### After T0.8 (three.js) and T0.9 (Babylon), 2026-11-04

> The Babylon notes below describe the T0.9 snapshot (Babylon 9.33.2). From M1 on, three.js is the only adapter until v1 is validated, and contract changes are no longer checked against Babylon (DECISIONS 2026-11-04).

Both engines took the shading module as designed. A light node in three and a material plugin in Babylon each called a function shaped like `giIrradiance(position, normal, viewDirection, screenUV, ...)`. Its resources were a 2D-array texture, a 3D texture, an `rgba32float` table texture, samplers and `vec4f` uniforms. Pixel readbacks matched the expected E/π within 0.2% on both GPUs.

- **Change: samplers belong to a texture.** A `sampler` parameter now names the texture it belongs to and is called `<texture>Sampler`. three's `wgslFn` feeds a sampler from a texture node, using that texture's filter and wrap settings. Babylon automatically binds a sampler named `<texture>Sampler` to that texture.
- **Confirmed: `giCoverage` is needed.** Both engines add their environment's diffuse on top of ours unless told not to. The three adapter scales the environment's diffuse by `1 − coverage` (r187 has no switch; PR #35017 would add one). Babylon zeroes its environment irradiance inside a plugin. Environment specular stays on in both.
- **Confirmed: the host owns the device, even when the engine insists on making it.**
  - three takes our device (`new WebGPURenderer({ device })`) with raised limits.
  - Babylon always makes its own device, so its adapter passes our limits through `deviceDescriptor` and hands `engine._device` to the core.
  - Babylon also destroys the device on `dispose()` and makes a new one after device loss. Both fit the existing rule: on loss, recreate the platform on the new device.
- **Confirmed: toggle GI with intensity.** In three, setting intensity to 0 cost 0 new pipelines. Removing the light cost 1–2 new render pipelines per material, and re-adding it cost the same again.
- **Adapter notes, not contract changes:**
  - three's 3D `ExternalTexture` needs `is3DTexture = true`.
  - Babylon is left-handed, so its adapter flips z in the call.
  - `screenUV` needs a viewport uniform from the adapter.
  - Babylon's foreign-texture wrapper destroys our texture on dispose unless reset first.
  - **Metals bounce no diffuse light.** The adapter folds metalness into albedo:
    - GI albedo = base colour × (1 − metalness);
    - with a metalness texture, the core samples it at the hit (`metalnessTexture`, M4) and applies the same product; nothing is baked.
    - Without this, a metal bounces light like paint. Every adapter needs this rule, and the M1 adapter task checks it.
    - Roughness stays out: DDGI bounces diffuse light only, the standard DDGI limitation. Revisit it for the RT-reflections technique.
  - Details are in NOTES.
