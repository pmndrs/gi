# M4 design: real materials and scenes (prototype)

Status: **draft for Dennis, 2026-11-09, revised the same day for prototype level.** Dennis: "Lets not get hung up on timings and get things built. when we have clear gi we can make decisions on paths but we've been behind on getting the milestiones at least prototype level for a week. Id like to clear m2, m3, and m4 at least very soon". So this plan:

- aims at textured GI, alpha cutouts and material mapping that **look right**;
- has done-whens about correctness and the look: accuracy against references, leaks, Dennis's visual check. Costs are noted but not gated, and there are no VALID runs or clock rules;
- picks the simplest approach that works and says why;
- runs **in parallel with M3** (branch `m3`), with the files each task touches listed so the two don't collide.

It builds on SPEC §M4, [platform.md](../api/platform.md) (gaps 5 and 8), [three.md](../api/three.md), [trace.md](trace.md) and [m3-moving.md](m3-moving.md). M3's T3.12 (compressed textures for three through gputex) stays in M3. Section 5 says how it fits.

## In one paragraph

Today GI sees a textured surface as one flat colour (the three adapter multiplies by each texture's average, M1), and every surface blocks light unless its whole mesh is see-through. M4 makes rays see real materials:

- **The core keeps a small texture store:** a texture array of 256² layers with mips, sampled where rays hit, with the mip chosen from the ray's footprint.
- **The tracer looks at a triangle's material only when a flag says so**, to skip see-through, single-sided-back-facing and cut-out hits.
- **The three adapter passes real textures** (including compressed ones) and drops the averages.
- **Every test scene is shown textured** against textured Cycles references.

## What's already there

Much of "ray hits return UVs" exists from M1:

- `GiHit` carries the instance, the original triangle index and the barycentrics.
- The scene copy stores uv sets 0 and 1 (`MESH_WORD.uvs0/uvs1`), and `giInstanceUvAt` interpolates either.
- Material records hold each texture's slot and its uv mapping (channel, 3×2 transform, wrap per axis).
- `GiSurface.uv` exists.
- The tracer's 48-byte triangle record has a flags word whose bit 0 is reserved "needs alpha test (M4)". It is always 0 today.

So **M4 needs no change to the mesh or vertex layout.** It adds words to the material record only, which matters for running beside M3 (see "Files and M3").

## 1. Hits: uvs and the mip

- `giSurfaceAt` applies each texture's mapping: the uv set, then the transform, then wrap.
- **Wrap:** `repeat` comes from the sampler (one `repeat` sampler). `clamp` is `clamp(uv, h, 1 − h)` (h: half a texel) and `mirror` is `1 − |(uv mod 2) − 1|`, both in the shader. This is the answer to platform.md gap 8.
- **Mip, by ray cones** (Akenine-Möller et al., Ray Tracing Gems ch. 20):
  - `lod = log2(spread × t / max(|cos θ|, 0.1) × texelsPerMetre)`;
  - `spread = sqrt(4π / raysPerProbe)`, 0.22 rad at DDGI's 256 rays;
  - `texelsPerMetre` comes from the hit triangle's uv area and world area.

  A probe ray stands for 1/256 of a sphere, so sampling the average over its footprint is both right and less noisy. GI rays mostly land on coarse mips (a hit 3 m away on a Sponza wall reads lod ~6), which is why 256² layers are plenty.
- New `giSurfaceAtCone(ray, hit, spread)` lets a technique pass its own spread. `giSurfaceAt` uses the platform default. `gi/direct` is unchanged: it reads the albedo the trace kernel put in the surface record.

## 2. The texture store

**Choice: one `texture_2d_array`, every layer 256 × 256 with a full mip chain, `rgba8unorm` with an sRGB view.** Not an atlas.

*Why:* in an array, each layer wraps and mips on its own, so there are no gutters and no repacking. A texture gets a free layer from the existing `SlotAllocator`, and growing the array means copying layers and rebuilding a bind group, never recompiling. Non-square or small textures are stretched to 256². GI reads coarse mips, so nobody will see it. Memory is 0.33 MB per layer: about 27 MB for Sponza and 35 MB for Bistro exterior (estimated ~100 albedo and emissive maps). `requiredLimits` asks for the adapter's `maxTextureArrayLayers`, since the default is 256 and adapters give 2,048.

- **What a hit reads:**
  - albedo = material colour × albedo layer × (1 − metalness × the metalness layer's blue channel), as three and glTF do;
  - emission = emissive × intensity × emissive layer.

  The metalness map is a separate sample with its own mapping, so there is no composition step. Without the map it is `1 − metalness`, as today.
- **Sources:** each source gets onto the GPU one way, then one kernel, `gi/texture-store`, downsamples it into the layer in linear light and builds the mips.
  - images: `copyExternalImageToTexture` into a staging texture at full size, released after the submit;
  - `PixelData`: `writeTexture` into staging;
  - same-device `GPUTexture`: read directly;
  - **new, compressed mips** (`CompressedPixelData`: a GPU format plus `levels: {data, width, height}[]`, which is what three's `CompressedTexture.mipmaps` holds from KTX2Loader or gputex): only the level nearest above 256² is uploaded (64 KB in BC7), and the hardware decodes it when sampled. This closes platform.md gap 5 without touching three's backend.
- **When:** textures are built in the next `encode` after `createTexture`, all of them, as the scene copy's packing is (T2.13). Until its layer exists, a material shades with its flat colour. When the layer lands, the platform logs a material change over the instances using it, so DDGI's mark refreshes those probes.

## 3. Materials in the walk

At a candidate hit the ray **passes** when:

1. the material doesn't block light (`blocksLight: false`: glass, hidden);
2. the material is single-sided and the hit is on its back face;
3. it is a cutout and alpha < cutoff;
4. it is stochastic (open question 1) and a hash of the ray and triangle is ≥ alpha.

Otherwise the hit stands, for closest hit and any hit alike.

- **Only flagged triangles pay.** Triangle flag bit 0 means "consult the material". It is set on triangles whose material slot needs rules 1–4 in any instance of the mesh. A new instance flag, "consults", says whether this instance needs it. The walk does the lookup only when both bits are set. The flag word is already loaded, so opaque scenes cost nothing extra.
- **The lookup** (in its own WGSL function, `giMaterialPasses`, called from one line of the leaf test): instance → material list → the triangle's slot → the material record. Rules 3 and 4 also fetch the uvs and sample the albedo layer's alpha at a fixed fine mip (`alphaTestLod`, default 1). That needs no coverage-preserving mips.
- **Keeping flags right without rebuilding trees:** a small kernel, `gi/triangle-flags`, rewrites the flags over one mesh's triangles from a per-mesh mask of slots that need a lookup. It runs when a tree is installed and when a material change flips a slot.
- **Bindings:** tracing kernels already bind the scene copy. The store adds one sampled texture and one sampler, and **no storage buffer**.
- Per-triangle see-through replaces the adapter's mesh-level exclusion (three.md "See-through meshes"), so a glass pane inside a wall mesh and Sponza's `dirt_decal` just work.
- *Cost (noted, not gated):* opaque scenes nothing; foliage rays maybe 0.3–0.6× speed. Budget v2 spends fewer probes there by itself.

## 4. The three adapter

| three material | Albedo | Emission | Cutout / see-through |
| --- | --- | --- | --- |
| Standard, Physical (and their node versions with plain properties) | `color` × `map`; `metalness` × `metalnessMap` (b) | `emissive` × `emissiveIntensity` × `emissiveMap` | `alphaTest` → cutout; `transmission` > 0 → `blocksLight: false`; blended → open question 1 |
| Lambert, Phong, Toon, Basic | `color` × `map` | as above (Basic: none) | as above |
| `colorNode` graphs, `ShaderMaterial` | `color` / `map` if set, else grey 0.5, one-time warning | `emissive` if set | as above |
| Any | `userData.giAlbedo`, `giAlbedoMap`, `giEmissive` override | | `giBlocksLight`, `giSingleSided` as today |

- **Textures:** `texture.image` (`ImageBitmap` from GLTFLoader, image elements, canvases); 8-bit `DataTexture`s as `PixelData`; **`CompressedTexture` as compressed mips**. Render targets are still skipped with a warning. uv channels 0 and 1 map to the core's two sets. Edits are found through the existing material edit checks, now also comparing each map's `version`.
- **Removed:** `texture-average.ts` and its fold (about 1.1 s of Bistro's main thread at load, T2.13), and the mesh-level see-through exclusion.
- **Later:** `alphaMap` (rare in glTF; a one-time warning for now), vertex colours (needs a mesh layout word; open question 2).

## 5. Compressed textures so Bistro fits in 6 GB (T3.12, in M3)

Bistro exterior's 224 textures at 2048² take 5.0 GB uncompressed with mips, which overflows the 6 GB RTX with DDGI's 0.6 GB (T2.17). In BC7 (and BC5 for normal maps) they take **1.25 GB**.

**Upstream gputex alone is enough.** I checked the npm package (0.14.0; 0.14.1 is current, published 2026-10-31):

- `gputex/three`'s `compressTexture(source, { device, mipmaps: true, hint, colorSpace, flipY })` and `GputexLoader` return a three `CompressedTexture` with every mip, on the device you pass (three's).
- They encode on the GPU and read the bytes back once at load; three re-uploads them. That costs load time, not frames or VRAM.
- HomeFig's extra method (`encodeMipChainIntoTexture`, no readback) would need three to draw from a `GPUTexture` it doesn't own. It isn't needed.

Notes for T3.12:

- Request `texture-compression-bc` (and ASTC/ETC2 where present) on the device: `optionalFeatures()` doesn't list it today.
- glTF textures need `flipY: false`. Copy wrap, repeat, offset, rotation, channel and colour space across.
- Normal maps use hint `'normal'`. On ASTC-only devices gputex puts normals in `.ra`, which three doesn't read, so prefer BC.
- **Keep `texture.mipmaps`' bytes:** GI reads its 256² level from them (section 2).

With both: three's textures 1.25 GB, DDGI ~0.6 GB, GI's store ~35 MB. That leaves several GB on the RTX.

## Files and M3

M3 (branch `m3`) owns tree building and refit (`trace/mesh-bvh*`, `gpu-mesh-trees.ts`, `top-level*`, `sah-builder.ts`, `packages/bvh`), the change feed (`scene/change-log.ts`, `scene/instance-transforms.ts`, the contract's change-feed types) and volumes. M4 stays out of those.

Paths below are under `packages/ddgi-core/src/platform/` unless they say otherwise.

| Task | Touches |
| --- | --- |
| T4.1 | **new** `textures/` (store, `gi/texture-store`, sources); `shade/wgsl.ts`, `shade/hit-shading.ts` (`giSurfaceAt`, `giSurfaceAtCone`); `scene/layout.ts` **material record words only** (`MATERIAL_WORD`: texture layer, metalness, metalness map, opacity, alpha mode); `scene/wgsl.ts` (`GiMaterial`); `scene/platform-scene.ts` (texture entries); `contract/scene.ts` (texture source, material fields); `device-requirements.ts`; `gi-platform.ts` (wiring) |
| T4.2 | **new** `trace/material-test-wgsl.ts` (`giMaterialPasses`), **new** `trace/triangle-flags.ts` (`gi/triangle-flags`); `trace/wgsl.ts`, **one call in the leaf test** plus an instance flag bit; `trace/ray-queries.ts` (capabilities, store binding, running the flags kernel after a tree install); `trace/reference-tracer.ts`, `trace/cpu-traversal.ts`, `trace/oracle-fixtures.ts` (CPU reference with materials) |
| T4.3 | `packages/ddgi-three/src/`: `convert.ts`, `scene-sync.ts`, `user-data.ts`, `texture-average.ts` (deleted); `docs/api/three.md` |
| T4.4 | `apps/viewer/src/scene/` and the viewer's debug views; scene manifests and the procedural scene builders; `tools/reference/cycles` (textured renders); new GPU test pages |
| T4.5 | `docs/milestones/M4.md`, PROGRESS, `bench/results` |

**The two shared points:**

- **`scene/layout.ts`:** M4 changes only the material record, and M3's T3.3 changes the mesh and vertex parts. One layout change at a time: whoever goes first lands a layout-only commit and tells the other, who merges it before touching the layout (the M3 lead's rule).
- **The tree-install call site:** the call to `gi/triangle-flags` goes in `ray-queries.ts`, where trees are installed. If M3's T3.2 or T3.3 adds an install path, it calls the same function. Whichever lands second wires it.

M3's walk edits are in `trace/wgsl.ts`'s traversal (refit nodes). M4's is one call in the leaf's triangle test. The alpha logic lives in its own file.

## Tasks

| ID | Task | Agent | Depends on |
| --- | --- | --- | --- |
| T4.1 | Texture store and textured hits: store, four sources incl. compressed mips, `gi/texture-store`, uv mapping and ray-cone mip in `giSurfaceAt`, albedo/emissive/metalness at hits, the contract additions | implementer | none |
| T4.2 | Materials in the walk: flags, `giMaterialPasses`, `gi/triangle-flags`; see-through and single-sided first (no textures needed), then cutout and stochastic; CPU reference | implementer | T4.1 for cutouts only |
| T4.3 | three adapter materials v2 (section 4) | implementer | T4.1, T4.2 |
| T4.4 | Textured scenes: four small procedural ones (below), every real scene textured in the viewer, textured Cycles HD references, debug views (GI albedo at hits; textures on/off) | wrangler + implementer | none (pages need T4.1) |
| T4.5 | M4 check and note, with Dennis's look | main session + gpu-tester | all, T3.12 for Bistro |

T4.1, T4.2's see-through half and T4.4 can start at once.

**Procedural scenes:**

- a *textured furnace*: a checkerboard with mean albedo 0.5;
- a *uv board*: repeat, mirror, clamp, offset, rotation, uv set 1;
- an *alpha checker wall* between a lit and a dark room;
- a *glass pane in a wall mesh*.

**Real scenes:** Sponza (Intel) with curtains, ivy and trees; Sponza (Khronos); Bistro exterior and interior; the furnished room; the courtyard tree.

### Done-whens (correctness and the look)

- **T4.1.** Done when:
  - the textured furnace matches a flat 0.5 albedo furnace within 1%;
  - the uv board's albedo at hits matches a CPU reference sampler on both GPUs;
  - each source kind (image, `PixelData`, `GPUTexture`, BC7 mips) gives the same layer within 2/255;
  - adding 500 textures leaves the recompile counter at 0;
  - with textures off, fields match M3's.
- **T4.2.** Done when:
  - hits and misses agree with the CPU reference tracer on ≥ 99.9% of rays (rays within a texel of an alpha edge excluded) on the alpha checker and the glass pane, on both GPUs;
  - the checker lets through 50% ± 3% of the open wall's light;
  - the leak checks and two-room's door still pass;
  - an opaque scene's field is unchanged.
- **T4.3.** Done when:
  - a test matrix (each material type × map, metalnessMap, alphaTest, transparent, transmission, `CompressedTexture`) gives the expected descriptors;
  - Sponza's decal and curtains are traced;
  - no texture average is computed;
  - a mixed opaque/glass mesh lets light through its glass only.
- **T4.4.** Done when every scene above opens textured from its manifest with no per-scene material code, and each real scene has a textured Cycles reference (source and checksum recorded).
- **T4.5.** Done when:
  - each textured scene's indirect light is within ±10% mean of its reference (FLIP reported);
  - leaks are ≤ 1% and none is visible;
  - Bistro exterior runs on the 6 GB RTX without T2.17's cap (after T3.12);
  - Dennis has looked at each scene.

  Costs (probes per frame, store memory, load time) are written down, not gated.

## Contract additions

Host-facing. The platform API was signed off on 2026-11-04, so Dennis sees them. All are optional and backward compatible:

- `CompressedPixelData` as a `TextureSource`;
- `MaterialDescriptor.metalness` and `metalnessTexture: { texture, channel, uv? }`;
- `MaterialDescriptor.opacity`;
- `MaterialDescriptor.alphaMode: 'opaque' | 'cutout' | 'stochastic'` (with `alphaCutoff` and no mode it is `'cutout'`, as now);
- `requiredLimits` asks for `maxTextureArrayLayers`.

Technique-facing: `giSurfaceAtCone` and `RayQueries.capabilities.alphaCutouts = true`.

## Later

- `alphaMap`.
- Coverage-preserving alpha mips (Castaño 2010), if cut-out leaves look thin.
- A BC7 store via HomeFig's gputex fork.
- Tinted glass.
- Video textures.

## Open questions for Dennis

1. **Blended materials** (glTF BLEND: decals, Sponza's cypresses). Today they let all light through. *Recommend stochastic transparency:* a ray passes with probability 1 − alpha. It is the right average, it costs the same as a cutout, and DDGI's blending smooths it. Glass (`transmission`) keeps letting all light through.
2. **Vertex colours** (glTF `COLOR_0`, three's `vertexColors`). They need a mesh-layout word, which M3's T3.3 also changes. *Recommend later,* unless a test scene depends on them. None of ours is known to.
