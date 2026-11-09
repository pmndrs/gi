# webgpu-ddgi v1 spec

30 October 2026. Written after the [audit](docs/AUDIT-2026-10-30.md). This is the spec for a **fresh start in a new private repository**: new core, new code, nothing ported automatically. The old project stays as it is, at its `review-baseline-2026-10-30` tag. This repo links the native RTXGI fork as a submodule at `native/rtxgi-ddgi`. The research notes and a few proven pieces of math and tests are reused deliberately, as listed below.

## What we're building

A **universal, open-source WebGPU dynamic diffuse GI library**, to be published as **`pmndrs/ddgi`**. It's a sibling of [`pmndrs/upscaler`](https://github.com/pmndrs/upscaler) and [`pmndrs/denoiser`](https://github.com/pmndrs/denoiser), and follows their conventions.

- It has no dependency on any rendering library.
- three.js is the first adapter and the one we develop against.
- Babylon, Pixi, react-three-fiber and anyone's own WebGPU renderer must be able to plug in the same way, without changes to the core.
  - Until v1 is validated, three.js is the only adapter we build or test. The other adapters come in M5 or later; see DECISIONS 2026-11-04.
- It is built for nobody in particular: games, configurators, archviz, editors, demos. No single product's needs shape the API. (HomeFig was where the first experiments happened; its lessons are recorded below, and it may become one user.)

**Scenes are whatever the host builds.** The library never assumes a file format or a way of making scenes. A scene can be:

- written in code;
- generated procedurally;
- edited live in a tool;
- assembled from loaded models (glTF, FBX, USD, anything the host can load);
- any mix of these.

The host owns the scene. The adapter describes it to the core as meshes, materials, instances and lights.

For a three.js user it should feel like this:

- Add a probe volume to whatever scene they've built: interiors, Sponza, a courtyard with trees, a procedural level.
- The bounce lighting looks right.
- It keeps up while the camera moves, objects are dragged, lights and the sun change, and animated things like a tree swaying in the wind cast moving shadows.
- The frame cost leaves room for the rest of the app.

### A foundation, not just one technique

DDGI is the first GI method built here, not the last. Nearly every modern GI path builds on a DDGI-style probe field or uses one as its fallback:

- voxel GI;
- SDF tracing;
- surfels;
- radiance cascades;
- ray-traced reflections.

So the code is split in two:

1. **A shared GI platform.** It holds everything a GI method needs from the host and the scene.
2. **Techniques that plug into it.** DDGI is the first.

Later experiments add techniques without changing adapters or host code. See "Built to host other GI methods" under Architecture.

### What "moving" means

The common cases, in priority order:

1. **The camera.** It moves constantly. GI must stay stable, with no flicker or popping, and the area around the camera must have probes when the scene is larger than one volume.
2. **Dragged objects:** cabinets, furniture, props. These are rigid objects moved by hand.
3. **Lights and sun:** moved, toggled, recolored, time of day.
4. **Animated geometry:** a tree swaying, a curtain or a fan. The shape deforms every frame but keeps the same triangles. It should cast moving shadows and occlusion into the GI.

### How games handle this, and what we copy

- **Settling time.** Probe GI in games always lags a little. RTXGI-style DDGI blends each probe update into the old value (hysteresis around 0.97), so a light change starts showing on the next frame. It is mostly there within a few tenths of a second and fully settled in about a second. Games hide the remainder because direct light and shadows change instantly; only the bounce trails. Unreal's Lumen and Godot's SDFGI have similar lag. Our targets (visible next frame, within 5% in 0.5 s on a discrete GPU) are in line with that.
- **Characters and small moving things usually don't affect probe GI.** Probes are spaced a metre or more apart, so a person-sized object barely changes what a probe sees. Shipping engines (RTXGI titles, Enlighten, Godot SDFGI) let characters *receive* GI but leave them out of GI rays. Their local darkening comes from cheaper effects: screen-space AO, contact shadows, capsule shadows.
- **Large movers do matter, so we include them:** doors, furniture, vehicles, big foliage.
- **Our default:**
  - Every object has a "contributes to GI" flag. It's on for static and large objects, and the adapter can turn it off for characters and small props.
  - Deforming meshes such as trees opt in.
  - Skeletal characters as GI occluders are a stretch goal; local contact darkening is the host renderer's job (AO, contact shadows).

### Licensing

- **MIT.** The core is implemented from the published papers (Majercik et al. 2019 and 2021).
- **Nothing from NVIDIA's RTXGI SDK.** No source or shader code is copied, because it's under NVIDIA's own license, not MIT. The native RTXGI harness is used only as an external comparison tool.
- **Vendored dependencies must be MIT-compatible,** and each one is listed in `THIRD_PARTY.md`.

## Architecture

### three.js version

- **Pinned to r187** (released 30 October 2026).
- **Tracking r188** (due 25 November). See "What three.js already has" below for how r187 and the open PRs shape the adapter.
- **Sibling libraries:** `@pmndrs/upscaler` (a WebGPU port of FSR3-style temporal upscaling) and `@pmndrs/denoiser` (Intel OIDN networks and AMD FidelityFX temporal denoisers in WGSL) are first-class neighbours. See "Working with the pmndrs siblings".

### How the three.js adapter plugs in (r187)

Checked against the r187 source. Runtime confirmation is an M0 spike.

- **GPU device.** The host owns the device and our core never creates one. This is the same rule as the upscaler ("your device, never ours") and the denoiser.
  - `DDGI.requiredLimits(adapter)` tells the host what to request, for example via `new WebGPURenderer({ requiredLimits })`, or by creating the device and passing `{ device }`.
  - The core works at default limits for typical scenes by splitting large buffers across bindings, and reports clearly when a scene needs more. Three's default device has a 128 MB storage-binding cap.
    - *Superseded 2026-11-05 (DECISIONS):* nothing is split across bindings. Small scenes run at default limits; large ones (Sponza with curtains, Bistro exterior) need the raised limits `requiredLimits` asks for, which Dennis approved.
  - Our core listens with `addEventListener('uncapturederror')` and never assigns `onuncapturederror`, which three overwrites.
- **Our textures in three's shaders.** r187's `ExternalTexture` wraps a `GPUTexture` from the same device; three binds it without allocating or destroying it. This is how the adapter reads our probe atlases, with no reaching into `renderer.backend`.
  - Set `image` dimensions and the right `type` for the sample format.
  - On reallocation, wrap the new texture in a new `ExternalTexture`. But the core keeps textures fixed-capacity, so this should be rare.
- **Our buffers in three's shaders: not supported.** That would need an upstream PR of about 20 lines (an `ExternalBufferAttribute`). So the platform rule is: **anything a fragment shader reads lives in textures, with small uniforms mirrored from the CPU.** This also suits compatibility-mode devices, which may allow zero fragment-stage storage buffers. Compute-only data (BVH, rays, hits) stays in buffers.
- **Hooking into lighting.** r187 replaced `addLight` with `Light.registerNode()`. The adapter ships a GI volume as a three `Light` subclass with a registered `AnalyticLightNode`, the same pattern as three's `LightProbeGrid`:
  - Adding it to the scene lights every lit node material, with no material changes.
  - It works under the new Meshlet and Visibility-Buffer drawing pipelines. Those overwrite `material.contextNode`, so **never inject through `contextNode`**.
  - It writes three's light-probe irradiance slot with a priority, so a fragment never adds our GI and a `LightProbeGrid` together. That slot is an internal convention; it gets a regression test and the three version is pinned.
  - It returns irradiance E (three applies 1/π) and keeps its uniforms in `renderGroup`.
  - It is toggled with `intensity`, because adding or removing a light recompiles materials.
- **Matching the LightProbeGrid shape.** The three adapter's volume mirrors `LightProbeGrid`'s API: extends `Light`; `(width, height, depth, nx, ny, nz)` centred on `position`; probes at cell centres; `boundingBox`, `priority`, `intensity`. It becomes a drop-in upgrade.
- **Bake export.** An option exports a snapshot to `LightProbeGrid` JSON (open PR #33658) as a static or WebGL fallback.
- **Environment double-counting.** In r187, `scene.environment` adds its diffuse light separately from probe GI, so diffuse is counted twice, and r187 has no switch.
  - Open PR #35017 adds `material.envMapDiffuseEnabled`. The adapter feature-detects it and turns env diffuse off for receivers our GI covers, keeping env specular.
  - Until it lands, the adapter documents the issue and offers a material-level workaround.
  - It warns when `AmbientLight`, `HemisphereLight` or `LightProbe` would also add diffuse.
- **Coverage.** The shading module returns irradiance *and* a coverage value (how much of the answer the GI volume is confident about). That fits the "SSGI → probes → environment" coverage chain being discussed upstream.
- **Sky.** The platform owns sky lighting for ray misses in WGSL (low-order SH or a small octahedral map, projected by compute from a host-provided sky texture or a procedural sky), so the core stays engine-neutral.
  - three's PMREM (much faster once your compute PR #35130 lands) keeps handling specular from the same sky.
  - Note that three's env "irradiance" is a rough-GGX approximation, not a true cosine convolution. Expect a small, known mismatch where GI fades into the environment.

### Working with the pmndrs siblings

- **Shared device, shared frame.** All three libraries run on the host's `GPUDevice` and record into the host's frame. One device carries:
  - GI updates;
  - the upscaler's temporal passes;
  - the denoiser's passes.
- **The upscaler** renders the scene at reduced resolution, so GI sampling cost falls with it.
  - GI must be stable from frame to frame and under sub-pixel jitter, so it doesn't fight the upscaler's history. Probe GI is world-space, so jitter doesn't change it.
  - A GI "reactive" hint (how much GI changed this frame in screen space) can feed the upscaler's reactive mask after big light changes.
  - Its published temporal guides (motion, disocclusion) are reused by future screen-space GI techniques instead of being recomputed.
- **The denoiser** pairs with later techniques that trace noisy rays per pixel: RT reflections via its FidelityFX reflection denoiser, and path-traced references via its OIDN networks.
  - The reference renderers in `tools/` denoise progressive path tracing with it, so "converged" comes sooner.
  - The DDGI probe field itself doesn't need it: blending over time already denoises.
- **One test bench.** The viewer composes all three on the same scenes:
  - GI on/off;
  - upscaled vs native resolution;
  - denoised RT reflections once they exist.

  Frame-cost targets are reported both ways.

### The rule that shapes everything

**The core speaks only WebGPU and plain data.**

- It never imports three.js or any other engine.
- It never walks an engine's scene graph.
- It never touches an engine's internals.

"The core" here means the platform packages plus every technique. Everything engine-specific lives in an adapter. Lint rules in `eslint.config.mjs` enforce this inside `ddgi-core` (`pnpm lint`; there is no separate CI check yet): no engine or adapter import anywhere in the core; a technique imports only `platform/contract` and its own folder (not the platform's internals, `@pmndrs/bvh` or another technique); the platform never imports a technique. Tests may use platform internals as fixtures.

### Packages

**Published shape: one package, `@pmndrs/ddgi`, with one import path per layer,** matching `@pmndrs/upscaler`:

- `@pmndrs/ddgi/core`: the platform plus the DDGI technique, engine-free;
- `@pmndrs/ddgi` or `@pmndrs/ddgi/three`: the three.js adapter;
- `@pmndrs/ddgi/babylon` and `/pixi`: later;
- `@react-three/ddgi`: later react-three-fiber bindings.

Engines are optional peer dependencies.

**Inside the repo,** the platform and the technique stay separate modules (the `@gi/*` names below), with a lint-enforced boundary (techniques reach the platform only through `platform/contract`). When the second technique arrives, the platform can move into its own package (for example `@pmndrs/gi-core`) without changing its API.

**Platform: shared by every GI method**

| Package | Depends on | Job |
| --- | --- | --- |
| `@gi/core` | WebGPU only | The host contract (device, scene handles, frame driving, shading injection); the GPU scene copy (geometry, instances, materials, texture atlas, lights, environment); change tracking and dirty regions; the frame budget; timestamps and stats; debug-data plumbing |
| `@gi/trace` | WebGPU only | Scene queries: BVH builder (in a worker), refit, WGSL closest-hit and any-hit traversal. Behind an interface, so SDF or voxel tracing can be swapped in later |
| `@gi/shade` | core, trace | Shading at ray hits: material lookup, direct light with shadow rays, emission, sky on miss, plus a hook to add the active GI technique's answer for further bounces. Every ray-based technique and RT reflections reuse it |

**Techniques: plug into the platform**

| Package | Depends on | Job |
| --- | --- | --- |
| `@gi/ddgi` | platform | Probe volumes, the update scheduler, ray generation, blending, borders, relocation, classification, light groups, saved bakes, and the DDGI sampling module |
| later: `@gi/voxel`, `@gi/sdf`, `@gi/surfels`, `@gi/cascades` | platform (+ ddgi as fallback) | Future experiments on the same contract |

**Hosts and tools**

| Package | Depends on | Job |
| --- | --- | --- |
| `@gi/three` | platform, three | Mirrors a three.js scene into the platform, injects the active technique's shading module through TSL, converts light units, coordinates with reflection probes and env maps |
| `@gi/webgpu-example` | platform | A minimal plain-WebGPU renderer that proves everything works without an engine; the template for custom adapters |
| `@gi/babylon`, `@gi/pixi`, ... | platform, engine | Later adapters on the same contract |
| `apps/viewer` | three adapter | The human-facing judge-and-compare tool; switches techniques |
| `tools/` | none | Native launcher, reference renderers, image comparison, test-scene manifests |

The repo keeps its `webgpu-ddgi` name for now, since DDGI is the first deliverable. A rename to something like `webgpu-gi` can come with the public release.

### How a renderer plugs in (the adapter contract)

An adapter does four things. Nothing else is required.

1. **Share the GPU device.**
   - The host gives the core its `GPUDevice`; the core never creates, configures or destroys one. `isSupported(device)` and `requiredLimits(adapter)` help the host pick limits.
   - On device loss, `isReady` turns false and calls throw a typed not-ready error. The host recreates the core on the new device, the same rule as the upscaler.
2. **Describe the scene by handles, incrementally.** Calls look like `createMesh(positions, indices, uvs)`, `createMaterial({albedo, albedoTexture, emissive, alphaCutoff})`, `createInstance(mesh, material, transform)`, `setTransform(instance, matrix)`, `createLight({type, color, intensity, position, direction, range, cone})`, `setEnvironment(...)`.
   - The adapter watches its engine for changes and sends only what changed.
   - The core keeps its own GPU copy. That copy is simplified on purpose: diffuse albedo, emissive and cutout only.
3. **Drive a frame.** The host calls `gi.update(commandEncoder, budget)` once per frame.
   - The core records its work into the host's encoder: no extra queue submits and no CPU readback, frame after frame. The one exception is at load: the GPU's build of the scene's first mesh trees is submitted once by the core and reads back a few words of stats per mesh (T2.12; DECISIONS 2026-11-08, for Dennis to confirm).
   - The host decides when to call it, how much time to spend, and whether to pause.
4. **Read the GI while shading.** The active technique publishes a **shading module**, the same shape for every technique. It has three parts:
   - WGSL functions:
     - `giIrradiance(position, normal, viewDirection) -> vec3f` for diffuse;
     - optionally `giRadiance(position, direction, roughness) -> vec3f` for glossy, which techniques that can answer it provide (DDGI gives a rough-only fallback);
   - the bind-group layout it needs;
   - the live bindings.

   Each adapter injects this into its own shading system once:
   - three.js through a TSL node;
   - Babylon through a material plugin;
   - a custom renderer by pasting the WGSL in.

   Switching from DDGI to a future technique changes the module, not the adapter.

The core runs in metres, right-handed and Y-up, with linear color and its own documented light units. Adapters convert into these.

**Lifecycle conventions shared with the pmndrs siblings:**

- `configure(...)`, then `await prepare()`, which compiles pipelines asynchronously.
- Calls throw a typed not-ready error until `isReady` is true.
- `encode(encoder, ...)` once per frame.
- `resetHistory()` for cuts.
- `dispose()` frees only what the core allocated.
- `gpuTiming: true` fills per-pass timings when `timestamp-query` is available, and silently does nothing otherwise.
- Settings are a live object read every frame.
- Every stage has a debug view.
- An executable plain-WebGPU "hello" example is the reference integration.

### Plug-in points inside the core

- **Ray tracer.** The core asks for "closest hit" and "any hit" over batches of rays stored on the GPU.
  - The default is `@gi/trace`.
  - A host could supply its own: another BVH library, or hardware ray tracing if WebGPU ever exposes it.
- **Probe volumes.** v1 ships box grids:
  - several can coexist and blend at their edges;
  - one can follow the camera (scrolling), so scenes bigger than one volume always have probes near the viewer.

  The interface leaves room for other layouts later.
- **References.** Reference renderers plug into `tools/` through one interface: "render this manifest scene from this camera to a linear EXR". v1 vendors:
  - three-gpu-pathtracer, through the three adapter;
  - Blender Cycles, scripted, for converged ground truth;
  - the native RTXGI harness and its path tracer.

### Built to host other GI methods

The rules that keep later techniques cheap to add:

- **Techniques never talk to the host.** They see only the platform: the scene copy, scene queries, hit shading, the budget and change events. That is how one adapter can serve every technique.
- **Scene queries are an interface, not a BVH.**
  - DDGI uses closest-hit and any-hit rays.
  - A voxel technique adds a voxelized view of the same scene copy.
  - An SDF technique adds distance fields.
  - Each is another query backend over the same handles.
- **Hit shading is shared.** Lighting a ray hit (material, direct light, shadows, sky, emission) is written once in `@gi/shade` and reused by DDGI, surfels, cascades, RT reflections and the reference checks.
- **Techniques can stack.** A technique can use another as its fallback for distant or secondary bounces, e.g. surfels or cascades falling back to DDGI. The platform supports more than one active technique; only one feeds the final shading module.
- **One output contract.** Every technique publishes the same shading module, debug data and stats, so the viewer, tests and comparisons work for all of them unchanged.
- **Check it on paper in M0.** Sketch how a voxel technique and a surfel technique would sit on this API before writing DDGI. Revise the API if they don't fit.

### Reflections

DDGI is diffuse-only by design. Its probes store very blurred light (about 8x8 texels per probe), which can't represent a sharp reflection, so reflections are not a DDGI output. They layer on top, and DDGI makes all of them better:

1. **Reflection probes** (three.js's box-projected probes, your own, or a host's). When the host renders a reflection probe's capture, the scene in that capture is lit with GI too, so reflections show bounce light and darken correctly in unlit rooms. The adapter can trigger recapture of probes near a GI change.
2. **Screen-space reflections** fall back to reflection probes, or to `giRadiance`, when a ray leaves the screen.
3. **Ray-traced glossy reflections** reuse what the platform already has: our tracer finds the hit, `@gi/shade` lights it, and DDGI supplies the bounce light at the hit. This is the standard RTXGI pairing. It makes a natural post-v1 technique, with no new scene work.

**v1 scope for reflections:**

- `giIrradiance` works inside reflection-probe capture passes.
- The three.js adapter plays correctly with three's reflection probes and env maps:
  - env-map *diffuse* is switched off where GI covers it, so it isn't counted twice;
  - env-map and probe *specular* stay on.
- DDGI's rough-only `giRadiance` fallback is available.

A ray-traced reflections technique is the first candidate after v1.

### Tracing decision

Software BVH tracing in WGSL, owned by `@gi/trace`:

- **Two levels.** Each mesh gets its own BVH, built once in a worker. A small top-level BVH over all object instances is rebuilt every frame, so moving objects is cheap and adding objects never recompiles shaders.
- **Deforming meshes** (a tree in the wind) keep their BVH shape. A compute pass refits the boxes every frame from deformed vertex positions that the adapter supplies as a GPU buffer, so there's no rebuild and no CPU work.
- **Learn from three-mesh-bvh, don't depend on it.** It's a three.js library, and its WebGPU path is hosted by three. The old project's ray oracle and thin-wall tests carry over to check our tracer, and three-mesh-bvh's results remain a cross-check.
- **Measure first.** M0 measures our traversal against three-mesh-bvh on Sponza before we commit.

## What v1 must do

### Scenes and content

- Any scene the host builds works, with no special preparation of its meshes or materials. The test set mixes hand-built scenes, procedural ones and scenes assembled from downloaded models, and covers interiors and exteriors:
  - **Primitives and known answers:** Cornell box, the classic two-room leak test, a furnace room, thin walls, a closed door, a hollow-panel cabinet row.
  - **Classics:** Sponza (Crytek and Intel versions), Bistro (interior and exterior), San Miguel if it fits in memory.
  - **Small environments:** furnished rooms, a courtyard, a small outdoor scene with an animated, alpha-cutout tree. These come from Sketchfab, Poly Haven and similar sources.

  Every asset's source and license is recorded in its scene manifest. Redistributable assets ship with the examples. Others are downloaded by a script and never committed.
- **Materials:** the core only needs four things from a material:
  - diffuse albedo, as a color and/or a texture;
  - emissive color and texture;
  - an alpha cutout;
  - whether it blocks light (glass doesn't).

  Each adapter maps its engine's materials onto that. The core keeps downsized copies of the albedo and emissive textures in its own atlas.
- **Lights:** directional (sun), point and spot.
  - A sky or environment lights the scene through windows.
  - Emissive surfaces bounce light.
  - Capacity is at least 32 lights.
  - Adding or removing a light never recompiles shaders.

### Dynamic behaviour

- **Lights:** moving, dimming, recoloring or toggling a light updates the GI continuously, with no bake button.
- **Camera:** GI stays stable as the camera moves. A camera-following volume keeps probes around the viewer, and its scrolling never shows as a seam or a pop.
- **Rigid objects:** moving, rotating, adding and removing objects updates both what blocks light and what bounces it.
- **Deforming objects:** vertex-animated meshes (foliage, cloth, fans) cast moving occlusion into the GI.
- **Sun:** the direction and time of day can be animated.
- **No blackouts:** GI never goes black or pops while updating. It blends from the old answer to the new one.
- **Budgeted updates:** work per frame has a cap, set in milliseconds or probes per frame. Probes near a change are updated first.

### GI method

- Standard RTXGI-style DDGI:
  - randomly rotated rays on every update;
  - blending over time (hysteresis);
  - unlimited bounces by sampling the current field;
  - stored distances plus a Chebyshev test for visibility;
  - probe relocation and classification on the GPU.
- **Optional light groups.** The field can be split into a few layers, for example sky, sun and artificial lights. Bounce light scales linearly with emission, so dimming or recoloring a whole group is an instant weight change, with no update lag. Moving a light still needs updates. An earlier HomeFig prototype measured it linear to 0.006%.
- **Double-sided distance hits.** Rays that hit back faces mark probes that sit inside geometry, which feeds classification and relocation.
- **No per-pixel ray tracing in the default shading path.**
- Debug outputs: probe positions and states, GI only, and how fresh each probe is. The core exposes these as data. Each adapter draws them.

### Hardware tiers

Revised 2026-11-05 (DECISIONS). Dennis: "we arent building for the intel. We are building for RTX3060 and up." DDGI is for mid- to high-end GPUs.

- **Target hardware (the performance targets apply here):** full continuous updates. Test devices:
  - an RTX 3060 Laptop (Windows, Chrome: Dawn on D3D12 and Vulkan), the floor of the range;
  - a Mac with an M5 Pro and 64 GB (Safari and Chrome: Metal).
- **Correctness devices (no performance targets):** integrated and mobile GPUs must run correctly and pass every correctness check, with settings scaled down by presets (M5). Test devices:
  - the Intel UHD in the Windows laptop;
  - an iPad;
  - an older iOS device. If its Safari has no WebGPU, it isn't a target; it documents the cutoff.
- **Android:** a stretch goal, tested through a device lab if needed.
- Every milestone reports its numbers per test device and browser. Performance numbers are judged on the target hardware only.
- **Saved bakes:** the field saves and loads in a versioned, engine-neutral format.

### Out of scope for v1

- Building a reflection system. v1 feeds existing reflection probes and SSR and provides the platform pieces for ray-traced reflections; the RT reflections technique itself comes after v1.
- Skeletal characters as light blockers is a stretch goal (they always *receive* GI).
- Large-scale cascades for open worlds. One camera-following volume plus fixed volumes is in scope.
- WebGL.
- Shipping Babylon and Pixi adapters. Proving the contract fits them is in scope (M0 spike).

## How we'll judge it

The viewer measures every number itself, with GPU timestamp queries, on named scenes and cameras. Targets are proposals until M0 gives us real web and native baselines.

**Costs are measured with the GPU loaded** (revised 2026-11-05, DECISIONS): GPU-bound, uncapped or back to back, with the clocks up. Dennis: "We should load the GPU and test at a 'normal' baseline not a super quiet, low power mode." A GPU idling at 60 Hz drops into a low-power state that inflates per-pass milliseconds without costing any frame time, so light-load numbers aren't criteria.

| What | How measured | Proposed v1 target (target hardware, 1440p) |
| --- | --- | --- |
| Cost of showing GI | GPU ms of the GI sampling in the frame, including any extra render targets and the composite pass it needs | ≤ 1.0 ms (stretch 0.5 ms) |
| Cost of updating GI | GPU ms of `gi.update` per frame at default budget | ≤ 2.0 ms, user-adjustable |
| Whole frame | p50 / p95 frame time, GI on vs off | reported; the bar is the sum of the two rows above (≤ 3 ms p50 at the default budget), and no p95 spikes (revised 2026-11-05, DECISIONS) |
| Light change response | frames until visible; time until within 5% of settled | next frame; ≤ 0.5 s |
| Object move response | same, for a cabinet move and a door closing | ≤ 1 s, no leak through the closed door |
| Stability | frame-to-frame flicker on a still scene | mean ≤ 0.1%, p95 ≤ 0.5% per-pixel change per frame (set by eye in M1 and locked, DECISIONS 2026-11-06) |
| Leaks | dark-room light as a share of lit-room light, thin-wall fixtures and furnished rooms | ≤ 1%, and none visible in review |
| Accuracy | indirect light vs a converged path-traced reference (mean error, FLIP) | mean within ±10%; FLIP ≤ 0.15 (8×8 blocks, against the 1920×1080 references; DECISIONS 2026-11-06) |
| Memory | measured GPU bytes owned by the core | reported for every scene; defaults sized for mid- to high-end GPUs (a 6 GB RTX 3060 must hold Sponza with room for the host). The furnished room's 64 MB stays as a guide, not a gate (measured 40.7 MB) |
| Startup | from the host handing the scene to GI, to the first frame drawn with GI, with a cold shader compile; time from file request reported too | ≤ 2 s on the target hardware (Dennis: "2s cold is fine", from "Handoff") |
| Engine neutrality | same scene through the plain-WebGPU example and the three adapter | GI images match within a small tolerance |
| Adapter cost | CPU ms per frame the three adapter spends syncing a still scene / one moving object | ≤ 0.2 ms / ≤ 0.5 ms |

**Human sign-off is part of every milestone.** You look at it and decide whether it's right. Numbers support that judgment; they don't replace it.

## The viewer

- **Scene picker:** every manifest scene: primitives, two-rooms, Sponza, Bistro, small environments. Each opens on a useful, well-lit view at normal exposure.
- **Navigation:** orbit and fly, with camera bookmarks shared with the native harness through the scene manifest.
- **Editing:**
  - Drag lights and objects with gizmos.
  - Toggle lights and change their color and intensity.
  - A sun/time-of-day slider.
  - Scripted motions: a light sweep, a door opening, a cabinet slide.
- **Comparing:**
  - GI off / on / GI only.
  - A split-screen wipe between two modes.
  - A wipe against a reference image (path-traced or native).
- **Stats:** frame ms, GPU ms per GI step, probes per frame, memory, adapter sync ms.
- **Debug:** probe spheres colored by state.
- **Help:** every control has a hover explanation of what it does and what you'll see.

## Milestones

Each milestone ends with:

1. a viewer build you play with;
2. a one-page plain-language note: what works, what doesn't, numbers, screenshots or a clip;
3. automated checks that run from one command on either GPU.

You sign off before the next milestone starts.

### M0: Clean start, tools, and the risky questions

- **New private repository** under DennisSmolek. It starts with:
  - this spec;
  - the audit;
  - the useful research notes;
  - the native RTXGI fork as a submodule.

  The old repo is left untouched.
- **Workspace setup:** the package layout above, TypeScript, lint (including the core's no-engine-imports rule), formatting, and a headless-Chrome GPU test runner driven over the DevTools protocol (the `@pmndrs/upscaler` approach) with `--gpu nvidia|intel`.
- **Test-scene manifests:** how to build or load each test scene (code, or model files plus placement), units, cameras, lights and volume bounds, read by the viewer, the native harness and the reference renderers. Scenes: Cornell, two-rooms, Sponza, Bistro, and a furnished interior.
- **Native harness:**
  - GPU picked by setting instead of a hard-coded "RTX 3060".
  - A `-Scene` launcher.
  - A clean display mode.
  - Cameras loaded from the manifest.
  - Sponza downloaded.
  - Per-pass GPU timings recorded (the built-in F4 benchmark) for each scene.
  - A D3D12 build, so PIX can time native and Chrome frames in one tool.
- **References:** Blender Cycles scripted from the manifest. three-gpu-pathtracer wrapped as a reference plug-in, denoised progressively with `@pmndrs/denoiser`.
- **Tooling:**
  - **Image comparison:** NVIDIA FLIP is installed.
  - **Frame capture and timing:**
    - RenderDoc for Vulkan/D3D12 frame captures (installed);
    - PIX for Chrome's D3D12 backend (installed);
    - Nsight Graphics for NVIDIA GPU traces;
    - Xcode's Metal debugger and Safari Web Inspector on the Mac and iOS devices.
- **Spikes.** Small throwaway tests to answer the questions that could force an architecture change:
  1. **three.js sharing (answered on paper; confirm at runtime):**
     - three r187 runs on our `GPUDevice` with our limits.
     - A registered GI `Light` node samples our atlases through `ExternalTexture`.
     - This holds under the Standard, Meshlet and Visibility-Buffer drawing pipelines.
     - Env diffuse can be switched off without breaking env specular.
  2. **Babylon:** the same question through its WebGPU engine and a material plugin. This keeps the contract from becoming three-shaped.
  3. **Tracing speed:** our two-level BVH traversal in plain WGSL vs the latest three-mesh-bvh WebGPU path (with its recent upstream speedups) on Sponza, rays per second for closest-hit and any-hit.
  4. **Timestamp queries:** availability and overhead on both GPUs in Chrome.
  5. **Second-technique check (paper):** walk a voxel GI and a surfel GI design through the platform API and the shading-module contract, and fix what doesn't fit.
  6. **Reflection-probe coexistence:** GI inside a three.js reflection-probe capture, and env diffuse off with env specular on.

**Done when:** WebGPU timings work on the Windows NVIDIA and Intel GPUs (the Mac M5 Pro, iPad and older iOS device are deferred to a later milestone: DECISIONS 2026-11-04); native timings exist for every manifest scene; each spike has a written yes/no answer with numbers; and the contract above is confirmed or revised from what the spikes found.

### M1: Core DDGI loop, static scene, two hosts

- The core runs the full DDGI update in a budget. Content: static geometry, constant-color materials, sun plus sky.
- **Both hosts:** the plain-WebGPU example and the three adapter render it.
- **Viewer v1:** scene picker, navigation, GI modes, split wipe, stats, probe debug, and three's `LightProbeGrid` and VXGI as baseline techniques on the same scenes.

**Done when:** the GI cost, accuracy, leak and stability targets are met on Sponza (with its curtains; untextured materials until the M4 atlas), Cornell and two-rooms on the target hardware, with memory reported; the two hosts match; the Intel UHD runs every correctness check; and you've signed off on the look.

### M2: Dynamic lights

- A fixed-capacity light buffer.
- Change detection with prioritized, faster-blending updates near the change.
- Light groups, so group-wide dimmer and color changes are instant.
- Sun and sky with time of day.
- Viewer: gizmos, sliders, the scripted light sweep, and side-by-side with native doing the same motion.

**Done when:** the light-response targets are met on the target hardware, the correctness devices run it correctly, and the sweep looks right to you next to native.

### M3: Moving objects

- A top-level BVH rebuilt every frame.
- Worker-built per-mesh BVHs.
- GPU refit for deforming meshes from adapter-supplied vertex buffers.
- The camera-following volume.
- Add and remove objects without recompiles.
- Probes near moved geometry re-checked and prioritized.
- Three adapter: efficient change tracking for transforms, adds and removes.
- Viewer: drag furniture, animated door, add and duplicate cabinets, a wind-animated tree in the outdoor scene, and a camera fly-through of a scene larger than one volume.

**Done when:** the object-response and adapter-cost targets are met, the recompile counter stays at zero, and a closing door darkens the other room without leaks.

### M4: Real materials and scenes

This can overlap M2 and M3.

- Ray hits return UVs.
- An albedo and emissive texture atlas in the core.
- Alpha cutouts during traversal.
- The three adapter maps three.js materials (whatever their source) onto the core's material model.
- Fully textured versions of every test scene, including Bistro and the alpha-cutout tree.

**Done when:** all target scenes load unmodified and pass the accuracy, leak and cost targets, and you've reviewed each one.

### M5: Tiers, packaging, release prep

- Quality presets, including scaled-down settings for the correctness devices.
- Saved bakes.
- Public API docs, plus an adapter-writing guide built on the plain-WebGPU example.
- Examples site.
- Release prep for a pmndrs-style public home.

### After v1

- Ray-traced glossy reflections as the first new technique.
- Voxel, SDF, surfel and radiance-cascade experiments on the same platform.
- react-three-fiber bindings, Babylon adapter, then Pixi: after v1 is validated on three.js (M5 or later). The contract is checked against Babylon only as of the T0.9 snapshot until then.
- Room-shaped volumes and multi-level cascades.
- Optional high-quality modes.

## What we deliberately reuse

These are carried over by rewriting, not by copying files. Source paths refer to the old repository, [DennisSmolek/ddgi at review-baseline-2026-10-30](https://github.com/DennisSmolek/ddgi/tree/review-baseline-2026-10-30/).

| What | Source | Why |
| --- | --- | --- |
| Octahedral layout, convolution and Chebyshev sampling math | `experiments/gpu-ddgi/src/shaders.js` (old) | Verified against a CPU reference; rewrite readably |
| Plain-language TSL display node patterns | `browser/homefig-bench/app/src/gi/ddgi/` | Cleanest old code; informs the three adapter |
| Ray-hit oracle, thin-wall and transform test cases | `experiments/gpu-ray-provider/tests` | Check our tracer |
| Energy/convolution GPU tests | `experiments/gpu-ddgi/tests` | Become core unit tests |
| WebGPU call and timestamp counter idea | `scripts/browser/profile-probe.js` | Basis for stats |
| Native fork and capture tools | `native/references/rtxgi-ddgi` | Comparison and timing |
| Findings | [audit](docs/AUDIT-2026-10-30.md) | View bias, ray rotation, recursion-cap matching, camera-basis rays |

## What three.js already has (r187)

three.js is building its own GI quickly. We should know exactly where it stops, compare against it, and contribute where it helps.

| | What it is | Dynamic? | Leaks | Notes |
| --- | --- | --- | --- | --- |
| `LightProbeGrid` (r186+, faster in r187) | L2 spherical harmonics per probe in a 3D texture, baked by rasterizing cubemaps (8 probes × 6 faces per draw via `ArrayCamera`). Multi-bounce by re-baking with the previous grid as a light | Partly: probes can be re-baked N per frame, but values are overwritten (no blending), a re-bake briefly drops bounce light, and nothing detects changes | No visibility, so it leaks through walls | Sponza (490 probes, 32² faces): 3.1 s bake on a Pascal GPU. One grid per fragment, chosen by priority |
| PR #35174, DDGI visibility for `LightProbeGrid` (open draft, stalled) | Adds distance moments (rasterized), Chebyshev weighting, an optional 4-probe tetrahedral mode, and hiding (not moving) probes buried in geometry | Visibility is captured once (static geometry) | Fixes many | Measured shading cost on Sponza: 3.08 ms → 4.00 ms (8 probes) or 3.38 ms (4 probes). mrdoob was unhappy with the cost. Unlikely in r188 |
| VXGI (r186+, leak fix in r187) | Voxel cone tracing: CPU-collected triangles, compute voxelization into dense 3D textures (default 128³), lights injected from shadow maps; gathered as a full-screen pass that needs TRAA | Lights yes. Geometry no: a change means full re-voxelization ("too expensive for per-frame animation") | Thin walls under a voxel leak; r187 fills solid interiors | Plugs in via a render-level GI context, not a light |
| `LightMapper` (r187) + `RayTracedRadiosity` (r188) | Lightmap baking in UV space. r188's radiosity traces rays with three-mesh-bvh's WebGPU BVH | Bake-time only | n/a | Upstream BVH speedups (SAH, nearest-first, packed data) roughly doubled ray throughput |
| Reflection probes (PR #35046, open draft, r188) | Box-projected cube-array probes installed as the scene environment | n/a | n/a | They also add diffuse via the environment, hence the need for #35017 |

**What that means for us:**

- **Our reason to exist is the gap three leaves:**
  - updates that blend in instead of popping;
  - moving and deforming geometry;
  - leak control that's cheap enough to ship;
  - ray-traced updates that don't re-render the scene per probe;
  - an engine-neutral core.
- **Compare, don't guess.** The viewer runs three's `LightProbeGrid` and VXGI on the same scenes as baseline techniques. Our targets have to beat the #35174 shading costs above on Sponza.
- **Borrow good ideas:**
  - the 4-probe tetrahedral sampling mode, as a quality/cost option;
  - VXGI's "fill closed interiors as solid", for probe classification;
  - expressing biases as fractions of probe spacing.
- **Contribute where it helps everyone:**
  - findings and cost data on #35174;
  - the small `ExternalBufferAttribute` PR;
  - the GI-coverage seam;
  - BVH improvements upstream to three-mesh-bvh rather than a fork.

  A full DDGI is unlikely to be accepted into three itself. Maintainers want small PRs, WebGL parity and BVH work routed through three-mesh-bvh. So this stays an external package that fits three cleanly.

## Lessons from HomeFig's DDGI branch

HomeFig's `cursor/ddgi-probe-gi` branch (commit `e8727ff`) is a separate, much more mature DDGI effort inside the HomeFig editor:

- It bakes probes by rendering cubemaps, not by tracing rays.
- It caches a per-probe G-buffer, so it can relight without re-rendering the scene.
- It is tightly tied to three.js r185 internals and to HomeFig itself.

We don't take its code. We do take its ideas and its measured lessons. None of these are tied to an engine.

**Adopt:**

- **Light groups:** see GI method above. Three rules matter:
  1. While a group updates, its bounce must read only its own layer, or the layers contaminate each other.
  2. The sky stays a fixed layer, because time of day changes its shape, not just its brightness.
  3. Bake each group in its real colors and scale by RGB ratios, not in white.
- **Leak fixes, by measured effect:**
  - Double-sided distance was the biggest single fix (84×).
  - Probe density came second (22×).
  - Probe texture resolution made no difference.
  - *Reducing* bias made leaks worse, by up to 34×.
- **Fixed-capacity probe textures:** changing the grid only changes how the textures are interpreted. Keep a compact list of active probes so dispatches skip gaps.
- **Content fingerprints for saved bakes:** hash source texture bytes and shadow *parameters* (not matrices), and bump a cache version on any meaning change.
- **Test oracles:**
  - a furnace test;
  - a "directional furnace" (emissive ceiling, closed-form answer);
  - a sealed two-room leak meter, measured as a share of the lit side;
  - a closed-door doorway test;
  - a cabinet room with hollow 18 mm panels.

  Never use a small, hard-edged emitter as an oracle; it swings ±11%.

**Avoid:**

- View bias pulling sample points into furniture: it caused black patches under cabinets.
- Setting the weight-crush threshold equal to the backface floor (both 0.2), which lets fully back-facing exterior probes leak.
- Grid coarsening that drops vertical probe layers: it blacked out backsplashes.
- Probes just outside the building seeing through openings.
- Area lights without shadows.
- Probe readings run 13–27% high near surfaces, which matters for anything like a light meter.
- Alpha cutouts and transmission were never handled there. We budget them explicitly (M4).

**Measurement traps they hit:**

- Texture readbacks need 256-byte row padding.
- Hidden browser tabs do no GPU work.
- `textureLoad` on multi-target render attachments returned junk.
- A scissor doesn't limit a WebGPU clear.

**Useful numbers** (RTX 4090, native Dawn):

- Relighting 512 cached probes: about 1 ms of GPU time for the sun, 2.3 ms for sun plus sky.
- Compacting a sparse grid cut one pass from 80.8 to 2.2 ms.
- A real 432-probe kitchen took 27.8 s to cold-bake by cubemap rendering. Removing that cost is the main reason to trace rays.

**Test assets:**

- None. Its kitchens are procedural and its product models aren't redistributable, so we use public scenes instead.

## How we work

- **One goal per milestone,** described as what you'll be able to see and do.
- **Readable TypeScript and WGSL:** comments explain *why*, settings are real configuration objects, and there are no minified one-liners or magic indices.
- **One test command from a clean clone on any test device.** Tests check behavior and cost, never other reports.
- **Native and reference comparisons explain differences.** They don't block progress unless a stated target fails.
- **Plain-language status:** "verified" and "not verified" are stated plainly.

## Decisions so far

- **Hardware:** target hardware is the RTX 3060 Laptop and up, and a Mac M5 Pro 64 GB. Correctness devices: the Intel UHD (Windows), an iPad and an older iOS device. Android is a stretch goal (revised 2026-11-05).
- **Name:** `pmndrs/ddgi`, published as `@pmndrs/ddgi` with `/core`, `/three`, `/babylon`, `/pixi` import paths, following `@pmndrs/upscaler`.
- **Neighbours:** `@pmndrs/upscaler` and `@pmndrs/denoiser`, sharing device, conventions and the test bench.
- **Tooling:** approved. See M0.

## Open questions

None blocking. M0 can start.
