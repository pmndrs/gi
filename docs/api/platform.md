# Platform API (draft for T0.11)

Status: **signed off by Dennis, 2026-11-04, and revised after the spikes** (T0.8 three.js, T0.9 Babylon, T0.10 BVH). The changes are listed under "Revisions" at the end.

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
2. **Scene.** The adapter calls `gi.scene.createMesh(...)`, `createMaterial(...)`, `createInstance(...)`, `createLight(...)` and `setEnvironment(...)`, then sends only what changed (`setTransform`, `updateLight`, ...). The calls are cheap. Changes are queued and uploaded inside the next `encode`. Many moves at once go through `setTransforms(instances, matrices)`, one call with 16 numbers per instance (T2.M1, Chrome on the RTX laptop: 10,000 sliding in 1.4 ms of CPU, 3.5 ms turning).
3. **Techniques.** `gi.addTechnique(new DDGITechnique(settings))`, `gi.setOutput(ddgi)`, then `await gi.prepare()`, which compiles everything asynchronously.
4. **Every frame.** `gi.encode(encoder, { deltaTime, budget, camera })` comes before the host's own passes, in the same encoder. There are no extra submits and no readbacks frame after frame; at load, the GPU's build of the scene's first mesh trees is submitted once by the platform and reads back a few words of stats per mesh (T2.12, DECISIONS 2026-11-08). Hosts with screen-space techniques also call `gi.encodeScreen(encoder, screen)` after their depth/normal pass. It does nothing when no technique needs it.
5. **Shading.** The adapter injects `gi.shading` into its engine once, and rebuilds only when `shading.version` changes.

The lifecycle names match the pmndrs siblings: `prepare()`, `isReady`, `encode()`, `resetHistory()`, `dispose()`, `gpuTiming`, and settings as live objects. Note one rename from SPEC: the per-frame call is `encode(encoder, frame)`, not `update(encoder, budget)`. The budget is a field of `frame`.

## Design choices, and why

### The scene

- **Four material ideas.** These are albedo (color and/or texture), emission, alpha cutout, and whether the surface blocks light. M4 adds metalness (it scales the albedo), opacity and an alpha mode. `doubleSided` is there too, because back-face hits are how probes learn they sit inside a wall. Everything else about an engine material is ignored. That keeps adapters small.
- **Textures** come in as any image source, a same-device `GPUTexture`, or compressed mips (`CompressedPixelData`, M4). The platform always makes its own downsized copy, a 256² layer with mips in its texture store, so the host can free the source.
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
- **Hit shading** is shared: `giSurfaceAt(ray, hit)`, `giDirectIrradiance(surface, groupMask, random)`, `giSkyRadiance(direction)`. Direct light caps the number of shadowed lights per hit and picks the rest stochastically, so hit cost stays flat as the light count rises toward the 32-light minimum. `bounceAlbedoCap` exists so we can match native RTXGI (0.9) when comparing.
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

## Revisions

### During M4, 2026-11-09

All optional and backward compatible (docs/design/m4-materials.md, "Contract additions"; T4.1).

- **Host-facing: textures are sampled at ray hits.** The platform keeps a texture store: one `texture_2d_array` of 256 × 256 layers with full mips, `rgba8unorm-srgb`, a layer per texture. A texture's layer is made in the next `encode` after `createTexture` (`gi/texture-store`, linear-light downsampling and mips). Until then its materials shade with their flat colour; when it lands, the platform logs a `material` change over the instances using them. The array grows by copying layers and rebuilding bind groups, never by recompiling. Texture coordinate (0, 0) is the source's first texel (the top-left of an image), as in WebGPU.
- **Host-facing: `CompressedPixelData`** (`{ format, levels: { data, width, height }[] }`) is a `TextureSource`. The level nearest above 256² is uploaded and the GPU decodes it. It needs the format's device feature; without it the platform warns once and those materials keep their flat colour. This answers gap 5.
- **Host-facing: new `MaterialDescriptor` fields.** `metalness` (0–1) and `metalnessTexture: { texture, channel, uv? }`: the albedo at a hit is `albedo × albedoTexture × (1 − metalness × metalnessTexture[channel])`, as three and glTF do. `opacity` (0–1) and `alphaMode: 'opaque' | 'cutout' | 'stochastic'` (with `alphaCutoff` above 0 and no mode it is `'cutout'`, as before; `'cutout'` with no cutoff uses 0.5). Rays use `opacity` and `alphaMode` from T4.2 on.
- **Implementation-facing for now: `HitShadingSettings.sampleTextures`** (live, default true). Off, every material shades with its flat colours, as before M4. Hosts can't reach it through the `GIPlatform` contract yet. The repository's tools and the viewer's textures toggle reach it through `GIPlatformImpl.shade.settings`. Exposing it to hosts is a decision for later, if a host needs it.
- **Host-facing: `stats.memory.textures`**, the texture store's bytes (0.35 MB a layer, the array grown in doubles), counted in `total`.
- **Host-facing: `requiredLimits` asks for `maxTextureArrayLayers`** (the default 256 would cap a scene at 256 textures; adapters give 2,048), and `optionalFeatures` for `texture-compression-bc`, `-etc2` and `-astc` where present.
- **Technique-facing: `giSurfaceAtCone(ray, hit, spread)`.** Textures are read at the mip a ray cone's footprint covers: `log2(spread × t / max(|cos θ|, 0.1) × texels per metre)`, texels per metre from the hit triangle's uv area (scaled by the uv transform's determinant) against its world area. A technique passes its rays' spread, `sqrt(4π / rays per sphere)`; DDGI passes it from `raysPerProbe`. `giSurfaceAt` uses 0.22 (256 rays). The uv mapping (uv set, 3×2 transform, wrap per axis) is applied for each texture: `repeat` by the sampler, `clamp` and `mirror` on the uv. `GiMaterial` gains `metalness`, `metalnessTexture`, `metalnessChannel`, `metalnessUv`, `opacity` and `alphaMode`, and its texture words are store layers (`GI_NO_TEXTURE` until a layer is made). The scene module gains `giInstanceTriangleUvs`. The hit-shading module binds two more resources, the store's array and its sampler, and still no storage buffer.
- Since T4.3 the three adapter passes real textures (images, data and compressed) and no longer folds texture averages.
- **Technique-facing (T4.2, first half): materials in the walk.** `giTraceClosest` and `giTraceAny` now let rays through see-through materials (`blocksLight: false`) and the back faces of single-sided ones (`doubleSided: false`), triangle by triangle, in every instance (docs/design/m4-materials.md §3). `RayQueries.capabilities.perTriangleMaterials` is true; `alphaCutouts` stays false until the second half. No WGSL signature changed. **Binding cost:** the `rays` module now depends on the scene module (the walk reads its material tables), so a kernel that lists `rays` alone binds the scene copy's 2 storage buffers too: inline tracing 3 + 2, a batch kernel 7 of 8. Kernels that list hit shading had both already, so DDGI's count is unchanged. Implementation-only: `rays.requestTriangleFlags(meshSlot)` (the install hook every path that writes tree triangles calls; trace/triangle-flags.ts) and `rays.triangleFlagRewrites`. Hosts see nothing new. Since T4.3 the three adapter no longer leaves see-through meshes out of GI; the walk handles them per triangle.
- **Technique-facing (T4.2, second half): alpha in the walk.** `RayQueries.capabilities.alphaCutouts` is true. Rays also pass a cutout where its alpha (`opacity` × the albedo texture's alpha, through the albedo texture's uv mapping) is below `alphaCutoff` (0.5 by default), and a stochastic material with probability 1 − alpha (a hash of the instance, the triangle and the hit's barycentrics; no frame number, since a technique's rays differ from frame to frame), for closest hit and any hit alike. The alpha is read from the texture store at a fixed mip, 1 (`ALPHA_TEST_LOD`), whatever `sampleTextures` says: a hole is geometry. No WGSL signature changed. **Bindings:** the store's array and its sampler now come from one module of their own, `gi-texture-store` (`giTextureSampleMapped`, `giTextureMappedUv`, `giTextureWrap`; textures/texture-sampling-wgsl.ts), which hit shading and the walk both list, so a kernel with both binds them once. The `rays` module now binds them too: still no storage buffer, so a batch kernel stays at 7 of 8 and inline tracing at 3 + 2. Hit shading's own entries drop from six to four (the uniforms, the sky map, its sampler, the light grid's picks).
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
