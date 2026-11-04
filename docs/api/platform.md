# Platform API (draft for T0.11)

Status: **first draft, before the spikes.** T0.8 (three.js), T0.9 (Babylon) and T0.10 (BVH) will test the riskiest parts. This page gets revised after them.

The TypeScript is in [`packages/ddgi-core/src/platform/contract/`](../../packages/ddgi-core/src/platform/contract/). It holds types only, plus two error classes. This page explains it in plain language and walks two future GI techniques (voxels and surfels) through it, to check it isn't shaped only for DDGI.

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
| `platform.ts` | `GIPlatform`, the object a host talks to. Also `createGI`, `requiredLimits`, `optionalFeatures` and `isSupported`. |
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
2. **Scene.** The adapter calls `gi.scene.createMesh(...)`, `createMaterial(...)`, `createInstance(...)`, `createLight(...)` and `setEnvironment(...)`, then sends only what changed (`setTransform`, `updateLight`, ...). The calls are cheap. Changes are queued and uploaded inside the next `encode`.
3. **Techniques.** `gi.addTechnique(new DDGITechnique(settings))`, `gi.setOutput(ddgi)`, then `await gi.prepare()`, which compiles everything asynchronously.
4. **Every frame.** `gi.encode(encoder, { deltaTime, budget, camera })` comes before the host's own passes, in the same encoder. There are no extra submits and no readbacks. Hosts with screen-space techniques also call `gi.encodeScreen(encoder, screen)` after their depth/normal pass. It does nothing when no technique needs it.
5. **Shading.** The adapter injects `gi.shading` into its engine once, and rebuilds only when `shading.version` changes.

The lifecycle names match the pmndrs siblings: `prepare()`, `isReady`, `encode()`, `resetHistory()`, `dispose()`, `gpuTiming`, and settings as live objects. Note one rename from SPEC: the per-frame call is `encode(encoder, frame)`, not `update(encoder, budget)`. The budget is a field of `frame`.

## Design choices, and why

### The scene

- **Four material ideas.** These are albedo (color and/or texture), emission, alpha cutout, and whether the surface blocks light. `doubleSided` is there too, because back-face hits are how probes learn they sit inside a wall. Everything else about an engine material is ignored. That keeps adapters small.
- **Textures** come in as any image source, or as a same-device `GPUTexture`. The platform always makes its own downsized atlas copy, so the host can free the source.
- **`contributesToGI`** per instance. Characters and small props still *receive* GI through the shading module, but rays ignore them, which is what shipping engines do.
- **Deforming meshes** keep their triangles. The host hands over a GPU buffer of positions each frame, and the BVH is refit on the GPU from it.
- **Lights** use three.js's physical units: directional intensity is irradiance, point and spot intensity is radiant intensity, and emission and sky are radiance. So the three adapter passes values straight through. `range` is required for point and spot lights, because unbounded lights can't be culled. Light **group** membership (0–3) is scene data, so it lives on the light. Group *weights* belong to the technique.
- **Nothing recompiles.** Every table (meshes, instances, materials, lights) is fixed capacity and grows by reallocating, never by changing shader code.

### What techniques see

- **The scene copy** is read through WGSL accessor functions (`giInstanceTransform`, `giInstanceTriangle`, ...), so the table layouts can change without breaking techniques.
- **`SceneChanges`** lists the dirty world-space boxes since the last frame, with a reason: geometry, material or light. A moved object gives two boxes, where it was and where it is now. Changes with no useful box (the sun, the environment) are flagged as global. This is what lets a technique update "probes near a change first".
- **Scene queries are a backend, not a BVH.** v1 has `rays`: `giTraceClosest(ray) -> GiHit` and `giTraceAny(ray) -> bool`. Techniques can call them inside their own kernels, or run a batch from buffers. A host could swap in another tracer that keeps the same WGSL signatures.
- **Hit shading** is shared: `giSurfaceAt(ray, hit)`, `giDirectIrradiance(surface, groupMask, random)`, `giSkyRadiance(direction)`. Direct light caps the number of shadowed lights per hit and picks the rest stochastically, so hit cost stays flat as the light count rises toward the 32-light minimum. `bounceAlbedoCap` exists so we can match native RTXGI (0.9) when comparing.
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
- `giRadiance(position, direction, roughness, ...)` is optional. DDGI gives a rough-only fallback.
- `giDebugColor(position, normal, view, ...)` drives the debug views.

`screenUV` is there for screen-space techniques (see the surfel walk-through). DDGI ignores it.

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

## Other open gaps

4. **Deformed vertices from three.js.** `setDeformedPositions` wants a GPU buffer. three's skinned and morphed positions aren't exposed as one. The adapter may have to run its own skinning compute. To decide in M3.
5. **Compressed textures** (KTX2). `TextureSource` accepts a `GPUTexture`, but getting three's `GPUTexture` means reaching into the renderer's backend, which we've ruled out. Alternatives: decode the KTX2 ourselves, or have the adapter keep the original image. To decide in M4.
6. **Splitting the budget between stacked techniques.** For now the platform shares units in proportion to each technique's `msPerUnit`. This needs a real policy once there are two techniques.
7. **Storage-buffer budget for tracing kernels.** The scene copy (about 3), the BVH (about 2) and lights (1) leave about 2 of the default 8 for the technique. T0.10 must count the real numbers. If it's too tight, `requiredLimits` asks for more where the adapter has it (most desktop adapters allow 10 or more), and we pack tables into fewer buffers.

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

## What happens after the spikes

The answers go in PROGRESS. Any change to the contract is made in both the TypeScript and this page, and listed in a "Revisions" section here.

## Revisions

### After T0.8 (three.js) and T0.9 (Babylon), 2026-11-04

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
  - Details are in NOTES.
