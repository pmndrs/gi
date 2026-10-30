# webgpu-ddgi v1 spec

30 October 2026. Written after the [audit](docs/AUDIT-2026-10-30.md). This is the spec for a **fresh start in a new private repository**: new core, new code, nothing ported automatically. The old project stays as it is, at its `review-baseline-2026-10-30` tag. This repo links the native RTXGI fork as a submodule at `native/rtxgi-ddgi`. The research notes and a few proven pieces of math and tests are reused deliberately, as listed below.

## What we're building

A **universal, open-source WebGPU dynamic diffuse GI library**, aimed at a pmndrs-style home (working name `pmndrs/ddgi`).

- It has no dependency on any rendering library.
- three.js is the first adapter and the one we develop against.
- Babylon, Pixi, react-three-fiber and anyone's own WebGPU renderer must be able to plug in the same way, without changes to the core.
- It is built for nobody in particular: games, configurators, archviz, editors, demos. No single product's needs shape the API. (HomeFig was where the first experiments happened; its lessons are recorded below, and it may become one user.)

For a three.js user it should feel like this:

- Add a probe volume to any glTF scene: interiors, Sponza, a courtyard with trees.
- The bounce lighting looks right.
- It keeps up while the camera moves, objects are dragged, lights and the sun change, and animated things like a tree swaying in the wind cast moving shadows.
- The frame cost leaves room for the rest of the app.

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

- **Pinned to r186,** the current release.
- **Tracking r187/r188,** whose already-merged work is relevant:
  - **Compute-based PMREM:** fast environment updates feed sky lighting on ray misses, and time of day.
  - **Probe-related PRs, such as reflection probes:** DDGI supplies diffuse light and reflection probes supply specular, so the adapter must slot alongside them without double-counting the environment's diffuse light.
- **Upscalers and denoisers:** your upscaler and denoiser libraries are first-class neighbours. GI must be temporally stable enough not to fight a temporal upscaler, and it must work at render resolution.

### The rule that shapes everything

**The core speaks only WebGPU and plain data.**

- It never imports three.js or any other engine.
- It never walks an engine's scene graph.
- It never touches an engine's internals.

Everything engine-specific lives in an adapter. A lint rule and a CI check enforce this: the core package may only import from itself.

### Packages

Package names below use the placeholder `@ddgi/*`. For pmndrs this would become something like `@pmndrs/ddgi`, `@pmndrs/ddgi-three`, and later `@react-three/ddgi`.

| Package | Depends on | Job |
| --- | --- | --- |
| `@ddgi/core` | WebGPU only | Probe volumes, update scheduling, all GI compute passes, the GI's own scene copy (geometry, instances, materials, lights, sky), saving and loading bakes, stats |
| `@ddgi/trace` | WebGPU only | Ray tracing for the core: BVH builder (runs in a worker) and WGSL traversal. Behind an interface, so another tracer can replace it |
| `@ddgi/three` | core, three | Mirrors a three.js scene into the core. Provides the TSL node that adds GI to three's materials. Converts three's light units |
| `@ddgi/webgpu-example` | core | A minimal plain-WebGPU renderer that uses the core directly. It proves the core works without any engine, and it's the reference for anyone writing their own adapter |
| `@ddgi/babylon`, `@ddgi/pixi`, … | core, engine | Later adapters built to the same contract |
| `apps/viewer` | three adapter | The human-facing judge-and-compare tool |
| `tools/` | — | Native launcher, reference renderers, image comparison, scene manifests |

### How a renderer plugs in (the adapter contract)

An adapter does four things. Nothing else is required.

1. **Share the GPU device.**
   - The host gives the core its `GPUDevice`. The core never creates a second device next to a renderer.
   - The core handles device loss by telling the host and rebuilding when asked.
2. **Describe the scene by handles, incrementally.** Calls look like `createMesh(positions, indices, uvs)`, `createMaterial({albedo, albedoTexture, emissive, alphaCutoff})`, `createInstance(mesh, material, transform)`, `setTransform(instance, matrix)`, `createLight({type, color, intensity, position, direction, range, cone})`, `setEnvironment(...)`.
   - The adapter watches its engine for changes and sends only what changed.
   - The core keeps its own GPU copy. That copy is simplified on purpose: diffuse albedo, emissive and cutout only.
3. **Drive a frame.** The host calls `ddgi.update(commandEncoder, budget)` once per frame.
   - The core records its work into the host's encoder: no extra queue submits and no CPU readback.
   - The host decides when to call it, how much time to spend, and whether to pause.
4. **Read the GI while shading.** The core publishes a **sampling module** made of three parts:
   - WGSL source for `ddgiIrradiance(position, normal, viewDirection) -> vec3f`;
   - the bind-group layout it needs;
   - the live bindings (probe textures, sampler, a small uniform buffer).

   Each adapter injects this into its own shading system. three.js does it through a TSL function node, Babylon through a material plugin, and a custom renderer just pastes the WGSL in.

The core runs in metres, right-handed and Y-up, with linear color and its own documented light units. Adapters convert into these.

### Plug-in points inside the core

- **Ray tracer.** The core asks for "closest hit" and "any hit" over batches of rays stored on the GPU.
  - The default is `@ddgi/trace`.
  - A host could supply its own: another BVH library, or hardware ray tracing if WebGPU ever exposes it.
- **Probe volumes.** v1 ships box grids:
  - several can coexist and blend at their edges;
  - one can follow the camera (scrolling), so scenes bigger than one volume always have probes near the viewer.

  The interface leaves room for other layouts later.
- **References.** Reference renderers plug into `tools/` through one interface: "render this manifest scene from this camera to a linear EXR". v1 vendors:
  - three-gpu-pathtracer, through the three adapter;
  - Blender Cycles, scripted, for converged ground truth;
  - the native RTXGI harness and its path tracer.

### Tracing decision

Software BVH tracing in WGSL, owned by `@ddgi/trace`:

- **Two levels.** Each mesh gets its own BVH, built once in a worker. A small top-level BVH over all object instances is rebuilt every frame, so moving objects is cheap and adding objects never recompiles shaders.
- **Deforming meshes** (a tree in the wind) keep their BVH shape. A compute pass refits the boxes every frame from deformed vertex positions that the adapter supplies as a GPU buffer, so there's no rebuild and no CPU work.
- **Learn from three-mesh-bvh, don't depend on it.** It's a three.js library, and its WebGPU path is hosted by three. The old project's ray oracle and thin-wall tests carry over to check our tracer, and three-mesh-bvh's results remain a cross-check.
- **Measure first.** M0 measures our traversal against three-mesh-bvh on Sponza before we commit.

## What v1 must do

### Scenes and content

- Ordinary glTF scenes load unmodified (through the three adapter). The test set covers interiors and exteriors:
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

- **Tier A (discrete GPU, measured on this RTX 3060 Laptop):** full continuous updates.
- **Tier B (integrated and mobile GPUs: this machine's Intel UHD, Apple silicon, Android flagships):** continuous at a smaller budget, or load a saved bake and update slowly. Measurement decides which. Only the Intel UHD is available here, so Apple and Android need devices or a device lab.
- **Saved bakes:** the field saves and loads in a versioned, engine-neutral format.

### Out of scope for v1

- Glossy and specular GI, reflections.
- Skeletal characters as light blockers is a stretch goal (they always *receive* GI).
- Large-scale cascades for open worlds. One camera-following volume plus fixed volumes is in scope.
- WebGL.
- Shipping Babylon and Pixi adapters. Proving the contract fits them is in scope (M0 spike).

## How we'll judge it

The viewer measures every number itself, with GPU timestamp queries, on named scenes and cameras. Targets are proposals until M0 gives us real web and native baselines.

| What | How measured | Proposed v1 target (Tier A, 1440p) |
| --- | --- | --- |
| Cost of showing GI | GPU ms of the GI sampling in the frame | ≤ 1.0 ms (stretch 0.5 ms) |
| Cost of updating GI | GPU ms of `ddgi.update` per frame at default budget | ≤ 2.0 ms, user-adjustable |
| Whole frame | p50 / p95 frame time, GI on vs off | GI adds ≤ 3 ms p50 and no p95 spikes |
| Light change response | frames until visible; time until within 5% of settled | next frame; ≤ 0.5 s (Tier B ≤ 1.5 s) |
| Object move response | same, for a cabinet move and a door closing | ≤ 1 s, no leak through the closed door |
| Stability | frame-to-frame flicker on a still scene | threshold set by eye in M1, then locked |
| Leaks | dark-room light as a share of lit-room light, thin-wall fixtures and furnished rooms | ≤ 1%, and none visible in review |
| Accuracy | indirect light vs a converged path-traced reference (mean error, FLIP) | mean within ±10%; FLIP threshold set in M1 |
| Memory | measured GPU bytes owned by the core | ≤ 64 MB for a single furnished-room volume; reported for every scene |
| Startup | load to first visible GI; shader compile stalls; BVH build time | ≤ 2 s on Tier A; others reported |
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
- **Workspace setup:** the package layout above, TypeScript, lint (including the core's no-engine-imports rule), formatting, and a Playwright-based GPU test runner as a normal dev dependency with `--gpu nvidia|intel`.
- **Scene manifests:** glTF, units, cameras, lights and volume bounds, read by the viewer, the native harness and the reference renderers. Scenes: Cornell, two-rooms, Sponza, Bistro, and a furnished interior.
- **Native harness:**
  - GPU picked by setting instead of a hard-coded "RTX 3060".
  - A `-Scene` launcher.
  - A clean display mode.
  - Cameras loaded from the manifest.
  - Sponza downloaded.
  - Per-pass GPU timings recorded (the built-in F4 benchmark) for each scene.
  - A D3D12 build, so PIX can time native and Chrome frames in one tool.
- **References:** Blender Cycles scripted from the manifest. three-gpu-pathtracer wrapped as a reference plug-in.
- **Spikes.** Small throwaway tests to answer the questions that could force an architecture change:
  1. **three.js sharing:** can three's `WebGPURenderer` run on our `GPUDevice`, and can a TSL node bind GPU textures and buffers that the core owns, without reaching into three's internals? If not, what's the smallest supported workaround, or an upstream change?
  2. **Babylon:** the same question through its WebGPU engine and a material plugin. This keeps the contract from becoming three-shaped.
  3. **Tracing speed:** our two-level BVH traversal in plain WGSL vs three-mesh-bvh on Sponza, rays per second for closest-hit and any-hit.
  4. **Timestamp queries:** availability and overhead on both GPUs in Chrome.

**Done when:** native timings exist for every manifest scene; each spike has a written yes/no answer with numbers; and the contract above is confirmed or revised from what the spikes found.

### M1: Core DDGI loop, static scene, two hosts

- The core runs the full DDGI update in a budget. Content: static geometry, constant-color materials, sun plus sky.
- **Both hosts:** the plain-WebGPU example and the three adapter render it.
- **Viewer v1:** scene picker, navigation, GI modes, split wipe, stats, probe debug.

**Done when:** the GI cost, accuracy, leak, stability and memory targets are met on Sponza (untextured), Cornell and two-rooms; the two hosts match; Tier B is measured and its mode decided; and you've signed off on the look.

### M2: Dynamic lights

- A fixed-capacity light buffer.
- Change detection with prioritized, faster-blending updates near the change.
- Light groups, so group-wide dimmer and color changes are instant.
- Sun and sky with time of day.
- Viewer: gizmos, sliders, the scripted light sweep, and side-by-side with native doing the same motion.

**Done when:** the light-response targets are met on Tier A and B, and the sweep looks right to you next to native.

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
- The three adapter maps glTF and three materials onto the core's material model.
- Fully textured versions of every test scene, including Bistro and the alpha-cutout tree.

**Done when:** all target scenes load unmodified and pass the accuracy, leak and cost targets, and you've reviewed each one.

### M5: Tiers, packaging, release prep

- Quality presets per tier.
- Saved bakes.
- Public API docs, plus an adapter-writing guide built on the plain-WebGPU example.
- Examples site.
- Release prep for a pmndrs-style public home.

### After v1

- react-three-fiber bindings, Babylon adapter, then Pixi.
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
- **One test command from a clean clone on either GPU.** Tests check behavior and cost, never other reports.
- **Native and reference comparisons explain differences.** They don't block progress unless a stated target fails.
- **Plain-language status:** "verified" and "not verified" are stated plainly.

## Open questions for you

1. **Hardware:** is this laptop (RTX 3060 + Intel UHD) the whole test matrix for now, or should we line up a Mac and an Android device?
2. **Package scope/name:** `@ddgi/*` is a placeholder until a pmndrs home is discussed.
3. **Your libraries:** names and repos of your upscaler and denoiser libraries, and links to the r187/188 PRs you mentioned (compute PMREM, reflection probes), so the adapter is designed alongside them.
4. **Optional tools:** Nsight Graphics and RenderDoc for GPU frame captures.
