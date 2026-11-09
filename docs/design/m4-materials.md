# M4 design draft: real materials and scenes

Status: **draft for Dennis, 2026-11-09.** Written on branch `m4-design` from `m2` at 8ab2faf. It is the input for M4's breakdown. SPEC says M4 "can overlap M2 and M3", and Dennis wants a mostly working base DDGI soon, to compare newer GI methods against. So this page proposes starting M4's core lanes alongside M3 (open question 8).

It builds on SPEC §M4, [platform.md](../api/platform.md) (open gaps 5 and 8), [three.md](../api/three.md) ("What is mirrored", "See-through meshes"), [trace.md](trace.md) (the triangle record's alpha-test flag and original index) and [m3-moving.md](m3-moving.md). M3's T3.12 (compressed textures for three through GPUTex, DECISIONS 2026-11-09) and T3.16 (texture averages) overlap with this page. Section 6 says how they fit together.

## In one paragraph

Today GI sees every textured surface as one flat colour: the three adapter multiplies each material by its texture's average (M1, DECISIONS 2026-11-04), and every surface blocks light unless the whole mesh is see-through. M4 makes rays see real materials:

- **Hits know their texture coordinates and footprint.** Hit shading already gets the triangle and its barycentrics, and the scene copy already stores uvs. M4 interpolates the uvs at the hit, applies the material's uv transform, and works out how large the ray's footprint is on the texture, which picks the mip.
- **A texture store in the core.** Each albedo or emissive texture gets a downsized copy (256 × 256 by default, with mips) as one layer of a texture array. That means no atlas, so `repeat` and `mirror` wrapping and mips work per layer without padding.
- **The tracer honours materials per triangle.** Alpha cutouts (leaves, fences, chains), see-through materials such as glass, and optionally blended materials as stochastic transparency, all in the same lookup, paid only by the triangles that need it.
- **The three adapter maps every material** onto this, from any source (glTF, FBX, code, KTX2 or GPUTex-compressed textures), and drops M1's texture-average fold.
- **Fully textured test scenes** with textured Cycles references: Sponza with its add-ons, Bistro exterior and interior, the furnished room, and the courtyard with its alpha-cutout tree. Plus small procedural scenes whose answers are known.
- **Bistro fits on the 6 GB RTX** because three's own textures are block-compressed on the GPU with upstream gputex (T3.12), and GI's copy reads the compressed mips directly.

## Words used here

- **Texture store:** the core's own copy of the scene's albedo and emissive textures, small and mipmapped, which GI rays sample. (SPEC calls it "the atlas"; section 2 explains why it is a texture array rather than an atlas.)
- **Layer:** one texture's slot in the store's texture array.
- **Mip, lod:** a texture's successively halved copies. "lod 3" is the third halving. Sampling a coarser mip averages more texels.
- **Ray cone, footprint:** a ray is treated as a thin cone. Where it hits, the cone's width is the patch of surface the ray stands for. DDGI's probe rays each stand for about 1/256 of the sphere, so their footprints are large.
- **Cutout (alpha test):** texels whose alpha is below a cutoff are holes. glTF's `alphaMode: MASK`.
- **Stochastic transparency:** a ray passes a surface with probability (1 − alpha). Averaged over many rays and frames, the surface lets through the right share of light.
- **BC7, BC5, ASTC, ETC2:** block-compressed GPU texture formats. BC7 stores colour at 1 byte per texel, against 4 for plain RGBA8.
- **Consult flag:** a bit on a triangle in the tracer's own record that says "look at this triangle's material before accepting a hit". The bit exists already (trace.md: "flags bit 0 marks needs alpha test for M4").

## What M4 must deliver (SPEC)

| Target | RTX 3060 and up, Apple M5 Pro (1440p) |
| --- | --- |
| Scenes | every test scene loads **unmodified**, textured, through the viewer's normal loader |
| Accuracy (indirect light against textured Cycles HD references) | mean within ±10%; FLIP ≤ 0.15 (8×8 blocks) |
| Leaks | ≤ 1%, and none visible in review |
| Update cost | ≤ 2.0 ms a frame at the default budget. Textures cost probes, not milliseconds: Budget v2 recalibrates, so the cost shows as fewer probes a frame (bar below) |
| Sampling cost | ≤ 1.0 ms at 1440p: unchanged, because shading pixels never reads the texture store |
| Memory | reported for every scene. Bistro exterior must fit the 6 GB RTX with room for three (section 6) |
| Startup | ≤ 2 s from hand-off, cold compile, with textures counted (open question 6) |
| Dennis's review | each scene, in the viewer |

M1–M3's checks still pass, and untextured scenes trace exactly as fast as before (section 3: no flagged triangles, no extra cost).

## What others do, and what we copy

| Who | Textures and alpha in GI rays | What we take |
| --- | --- | --- |
| **RTXGI-DDGI** (SDK test harness) | Hit shading samples the material's albedo with an explicit mip; alpha-tested geometry uses any-hit shaders in DXR/Vulkan RT | Albedo at hits from real textures; alpha tests only on geometry flagged for them |
| **"Texture Level of Detail Strategies for Real-Time Ray Tracing"** (Akenine-Möller et al., Ray Tracing Gems ch. 20, 2019) | **Ray cones:** a mip from the cone's width at the hit, the triangle's texel density and the hit angle | The mip formula (section 1.2) |
| **Lumen** (Wright et al., SIGGRAPH 2022 Advances) | Its surface cache stores low-resolution material "cards" per mesh, so GI never samples full textures | GI needs only low-resolution material data: a small fixed store size is the norm |
| **Castaño, "Computing Alpha Mipmaps"** (2010) | Alpha mips scaled so the share of texels above the cutoff stays the same at every mip | Coverage-preserving alpha mips, so leaves don't thin out at coarse mips |
| **Enderton et al., "Stochastic Transparency"** (I3D 2010) | Each sample keeps or drops a semi-transparent surface at random, by alpha | Stochastic transparency for blended materials, which fits DDGI's many rays and temporal blending (open question 3) |
| **three.js r187** | Renders textures itself. `KTX2Loader` gives `CompressedTexture`s with mip data | The adapter reads the mip data three already holds (section 2.4) |
| **gputex** (verekia, MIT, npm 0.14.x) | Encodes images to BC7/BC5/ASTC/ETC2 on the GPU at load. Its `gputex/three` entry returns a `CompressedTexture` with mips | three's own textures in BC7 (section 6) |

## 1. Ray hits return texture coordinates

### 1.1 What exists

Most of this is plumbed already:

- `GiHit` carries the instance, the original triangle index and the barycentrics. Batches write them in `GiHitRecord`.
- The scene copy stores uv set 0 and uv set 1 per vertex (`MESH_WORD.uvs0`, `uvs1`), and `giInstanceUvAt(instance, triangle, barycentrics, channel)` interpolates either set.
- Material records already hold each texture's slot and its `UvMapping` (channel, the 3×2 transform, wrap per axis: `MATERIAL_WORD.albedoUv*`, `emissiveUv*`).
- `GiSurface.uv` is uv set 0, interpolated, without the transform.

### 1.2 What M4 adds

- **`giSurfaceAt` samples the material** at the hit (section 2.3), using each texture's own mapping: channel, transform, wrap. `GiSurface` keeps `uv`, and gains `uvFootprint` (section below) for techniques that sample anything else.
- **The footprint, by ray cones.** A probe ray stands for a cone of half-angle about `spread = sqrt(4π / raysPerProbe)`: 0.22 rad at DDGI's 256 rays. Probe rays start at a point, so the cone's width at a hit `t` metres away is `w = spread × t / max(|cos θ|, 0.1)`, where θ is the angle between the ray and the surface normal.
  - The triangle's texel density comes from the hit triangle itself: `texelsPerMetre = storeSize × sqrt(uvArea × |det M| / worldArea)`, with `M` the uv transform's 2×2 part and the world area taken after the instance transform. Everything is at hand in `giSurfaceAt`: the corners and the uvs.
  - `lod = log2(w × texelsPerMetre) + textureLodBias` (bias default 0), clamped to the layer's mips.
  - *Why it matters for DDGI:* a probe ray stands for 1/256 of the probe's sphere, so the texel it should see is the average over its footprint. Sampling the full-resolution texel instead adds noise that hysteresis then has to average out. Prefiltering is both more correct and quieter.
  - *What it means in practice* (Sponza-like walls, a 256² store over about 2 m, 128 texels a metre): a hit 3 m away covers about 0.66 m, about 84 texels, so lod 6.4. A hit 0.5 m away is lod 3.8. GI rays mostly read coarse mips, which is why a 256² store is enough (section 2.5).
- **Contract (technique-facing):** `giSurfaceAt(ray, hit)` keeps its signature and uses the platform's default spread. `giSurfaceAtCone(ray, hit, spread)` lets a technique pass its own (DDGI derives it from `raysPerProbe`). A later reflection technique would pass the cone it carries instead.
- **The deferred pass is unchanged.** `gi/direct` reads the albedo the trace kernel put in the surface record (`giSurfaceRecord`), so textures are sampled once per hit, in the trace.

## 2. The texture store

### 2.1 Atlas or texture arrays

SPEC says "atlas", and platform.md gap 8 left this open: "A tile in an atlas can't use the sampler's own wrapping, so it needs wrap in the shader plus padding, or a texture array instead of an atlas."

| | One 2D atlas | Texture arrays (recommended) |
| --- | --- | --- |
| `repeat` | shader wrap into the tile, plus gutters so bilinear filtering doesn't read the neighbour | the sampler's own `repeat`, within the layer |
| `mirror`, `clamp` | shader math, plus gutters | shader math on the uv before sampling (below), no gutters |
| Mips | each tile's mips bleed into neighbours unless gutters grow as 2^mip, or the lod is clamped per tile | each layer has its own clean mip chain |
| Mixed sizes | packs any size | one size per array: every texture is resized to it |
| Adding and removing textures | packing, fragmentation, occasional repacking | a free layer from the existing `SlotAllocator`. Growth copies layers into a bigger array and rebuilds bind groups, never pipelines |
| Limits | one 2D texture of at most 8,192² (default limit) | 256 layers by default; adapters give 2,048 (D3D12, Vulkan, Metal), and `requiredLimits` asks for the adapter's maximum |
| Bindings | 1 texture, 1 sampler | 1 texture, 1 sampler |

**Recommendation: one `texture_2d_array`, every layer at the store size, 256 × 256 by default, with a full mip chain.** Non-square textures are stretched to square. uvs are normalised, so they still map correctly, and GI's coarse mips make the lost aspect invisible. Textures smaller than the store size are stretched up. That wastes a little memory, which is acceptable at these sizes (section 2.5); a second array for small textures can come later if memory says so.

### 2.2 Wrapping and the sampler

- **One sampler,** `repeat` on both axes, linear filtering, linear between mips (trilinear). WGSL can't index samplers at run time, and a sampler per wrap combination would cost 9 bindings, so `clamp` and `mirror` are applied to the uv in the shader:
  - `repeat`: the uv as it is (the sampler wraps);
  - `clamp`: `clamp(uv, h, 1 − h)`, with `h` half a texel at the chosen mip, so filtering never reaches the far edge;
  - `mirror`: `m = uv mod 2` (always positive), then `1 − |m − 1|`. At the mirror line the bilinear footprint reaches half a texel across the opposite edge. At GI mips that is invisible; the exact fix would need gutters, which arrays avoid.
- Wrap is per axis, from `UvMapping.wrapU` / `wrapV`, which the scene copy already packs (`UV_PACK_SHIFT`). The 3×2 transform is applied first, as three does.

### 2.3 What is stored, and how a hit reads it

- **Format:** `rgba8unorm` with an `rgba8unorm-srgb` view (`viewFormats`). The build kernels write the plain view as storage (sRGB storage textures don't exist) and encode sRGB themselves. Rays sample through the sRGB view, so the hardware decodes to linear before filtering.
- **Albedo layers** hold the GI albedo, composed when the store copy is made (2.4):
  - rgb = map × (1 − metalness × metalnessMap's channel), the fold platform.md's T0.8 notes promised for M4;
  - alpha = map alpha × alpha map's channel, when there is a separate alpha map (three's `alphaMap`).

  Composition happens only when the parts share a uv mapping, as glTF's do. When they don't, the albedo layer stays plain, and the other map is applied at the hit from its own layer's 1 × 1 mip (its average): one more sample, no CPU readback. That replaces M1's CPU averaging in every case.
- **Emissive layers** hold the emissive map. The material's emissive colour × intensity multiplies it, as in three. 8-bit storage is fine because the range comes from the intensity, as it does for three's emissive maps.
- **At the hit:** albedo = `material.albedo × sample(albedo layer)` (× vertex colour, 2.6), capped by `bounceAlbedoCap` as now; emission = `material.emissive × intensity × sample(emissive layer)`. That is one or two extra samples per hit, plus the uv fetches, for textured materials only.
- **A texture that isn't ready yet** (still uploading, 2.4) samples as absent: the flat material colour is used, as for an untextured material. When its layer lands, the platform logs a material change over every instance using it, so DDGI's mark updates those probes like any other material edit.

### 2.4 Getting pixels in (sources and cost)

`TextureSource` already takes images, `ImageData`, `PixelData` and a same-device `GPUTexture`. Each takes one route onto the GPU, then the same build kernel:

- **Images** (`ImageBitmap`, `HTMLImageElement`, canvases): `copyExternalImageToTexture` into a staging texture at full size, then `gi/texture-store` downsamples on the GPU into the layer. Staging textures come from a small pool and are released after the frame's submit. A 2048² staging texture is 16 MB, and the pool holds 4.
- **`PixelData` / `ImageData`:** `writeTexture` into staging, then the same.
- **`GPUTexture`** (same device): sampled directly by the downsample.
- **New: compressed mips** (`CompressedPixelData`: a GPU format such as `bc7-rgba-unorm-srgb`, plus `levels: { data, width, height }[]`). This is what three's `CompressedTexture` holds in `mipmaps`, whether it came from `KTX2Loader` or from gputex. The core uploads only **the level nearest above the store size** (64 KB for a 256² BC7 level) into a compressed staging texture, and the downsample reads it through the hardware's decoder. It needs the matching device feature (`texture-compression-bc`, `-astc` or `-etc2`). The host already enabled it, or three couldn't use the texture either. Without it the texture is skipped with a one-time warning, and the material falls back to its flat colour. This closes platform.md gap 5 without reaching into three's backend.
- **Build kernel `gi/texture-store`** (compute; one dispatch per mip level over all layers built that frame):
  1. downsample the source into the layer's mip 0 in linear light (a box filter over the source texels each store texel covers; the source's own nearest mip first when it has mips);
  2. compose metalness and alpha (2.3);
  3. build the mip chain;
  4. **coverage-preserving alpha** for layers used with a cutout: per mip, scale alpha so the share of texels at or above the cutoff matches mip 0 (Castaño 2010). One workgroup per layer runs the bisection in workgroup memory.
- **When it runs:** at load, every texture is built in the load burst, as the scene copy is (T2.13), and counts towards "scene ready" (open question 6). Textures created later are built at most `textureUploadsPerFrame` (default 4) a frame inside `encode`.
- **Main-thread cost:** `copyExternalImageToTexture` is the unknown. On Bistro, M1's texture averages took about 1.1 s of main thread (T2.13). M4 removes them. T4.0 measures the copy for Bistro's images and the compressed-level route. The compressed route copies 64 KB per texture and should be close to free.
- **Updates:** new host-facing call `updateTexture(texture, descriptor)` for changed contents (a three texture's `version`). It rebuilds that layer under the same budget. Video textures are out of scope for v1.

### 2.5 Memory

Per layer at 256²: 256 × 256 × 4 B × 4/3 for mips = **0.33 MB**. At 512²: 1.4 MB. At 128²: 0.09 MB.

| Scene | Albedo + emissive textures (estimate; T4.0 counts) | Store at 256² | at 512² |
| --- | --- | --- | --- |
| Sponza (Intel, with add-ons) | ~80 | ~27 MB | ~112 MB |
| Bistro exterior | ~100 (of its 224 textures; the rest are normal and specular maps) | ~35 MB | ~140 MB |
| Worst case: all 224 Bistro textures | 224 | 78 MB | 313 MB |

Plus the staging pool while textures build (at most 4 × 16 MB, released after load), and nothing per vertex: uvs are already in the scene copy.

**Settings:** `textureStore.size` (128, 256 or 512; default 256) and `textureStore.maxLayers` (capped by the device limit). Past the device's limit, `prepare` or `createTexture` throws `GILimitError` naming it. Growth doubles the array (a copy, a new bind group, no recompile), and `stats.memory.textures` reports it.

### 2.6 Vertex colours

glTF `COLOR_0` and three's `vertexColors` multiply albedo, and many kit-bashed and procedural scenes use them for all their colour. The material model has nowhere to put them. Proposed (host-facing, open question 2): `MeshDescriptor.colors` (rgba per vertex, stored as 4 bytes unorm in the scene copy, 1 word per vertex), and `MaterialDescriptor.vertexColors: boolean`. Vertex alpha multiplies texture alpha for cutouts, as in three.

## 3. Alpha cutouts and see-through materials in traversal

### 3.1 What decides whether a ray stops

At a candidate hit, in this order:

1. **The material doesn't block light** (`blocksLight: false`: glass, hidden materials): the ray passes, for closest hit and any hit alike.
2. **Single-sided material, back face** (`doubleSided: false`): the ray passes.
3. **Cutout** (`alphaCutoff`): the ray passes where alpha < cutoff. Alpha = texture alpha × vertex alpha (if on) × `opacity`.
4. **Stochastic** (`alphaMode: 'stochastic'`, open question 3): the ray passes with probability 1 − alpha, from a hash of the ray, the frame and the triangle, so a semi-transparent surface lets through the right share of light, on average.
5. Otherwise the hit stands.

Today, rule 1 applies only to whole meshes (three.md "See-through meshes": the adapter leaves out a mesh whose materials all let light through, and treats mixed meshes as opaque). M4 makes it per triangle, so a window in a wall mesh, a decal sharing a mesh with the wall, or Sponza's `dirt_decal` all work without excluding anything. The adapter's mesh-level exclusion and its warning are removed.

### 3.2 Paying only where needed: the consult flag

The walk reads each candidate triangle's 48-byte record anyway (trace.md), and its flags word already has a bit reserved for this. So:

- **Triangle flag bit 0, "consult the material":** set on every triangle whose material slot, in **any** instance of its mesh, uses rules 1–4. It is a union, because trees are per mesh and materials are per instance. A union can only cause an extra lookup, never a wrong answer.
- **Instance flag "consults":** set when any of the instance's own materials uses rules 1–4. The walk consults only when both bits are set, so an opaque instance of a mesh that elsewhere has a glass pane pays nothing.
- **Unflagged triangles cost nothing new:** the bit is in a word the walk already loads. Untextured, all-opaque scenes trace exactly as fast (done-when).
- **The lookup** (flagged hits only): instance → its material list → the triangle's material slot (scene copy) → the material record. Rules 1 and 2 stop there. Rules 3 and 4 also fetch the three uv indices and uvs, work out the mip (below) and sample the albedo layer's alpha. That is about ten dependent loads and one sample per flagged candidate.
- **Keeping the flags right without rebuilding trees:** a small kernel, `gi/triangle-flags`, rewrites the flags word over one mesh's tree-ordered triangles from a per-mesh mask of "consult" slots. It runs when a tree for a flagged mesh is installed (tinybvh's, H-PLOC's, or a refit's rest-pose tree in M3) and when a material change flips a slot's need. That is one path for every builder, and nothing recompiles.
- **The mip for alpha:** closest-hit rays use the ray cone (1.2). Any-hit rays (shadow rays in `gi/direct`, which start at a surface) use the same cone grown from their travelled distance, with a floor of `alphaTestMinLod` (default 2). Thanks to coverage-preserving mips, a leaf card keeps its silhouette at those mips.
- **Contract:** `RayQueries.capabilities.alphaCutouts` becomes `true`. The tracer module now also reads the scene copy and the texture store. In every kernel that traces, the scene copy is already bound (hit shading needs it). The store adds **one sampled texture and one sampler, and no storage buffer**, so `ddgi/trace` stays within its 8 storage buffers (m3-moving.md: 7, plus the mover mask). Batch kernels (the oracle, the bench) go from 5 to 7 storage buffers.

### 3.3 What it costs

Unflagged triangles: nothing. Flagged ones: the lookup above, per candidate. On foliage most candidates are flagged, and many are rejected (holes), so the walk continues past them. Engines see the same thing with any-hit shaders. **Planning number: rays through the courtyard tree's crown trace at 0.3–0.6× the speed of the same tree with solid cards.** T4.0 measures it before T4.4 commits. If it is much worse, two levers: a per-leaf "fully opaque" bit set by the store build (a triangle whose uv footprint is all above the cutoff skips the sample), and fewer rays a frame through foliage-heavy probes by Budget v2's calibration, which happens anyway.

### 3.4 Effects on DDGI

- **Classification and relocation** see fewer back-face hits through foliage and glass. A probe behind a glass pane is no longer "inside geometry". That is correct, and T4.4 re-runs the leak checks.
- **The distance atlas** sees through holes, so Chebyshev visibility behind a hedge is partial, as it should be.
- **The mover mask and marks** (M2, M3) are unchanged: they act on instances, not texels.

## 4. The three adapter: materials from any source

### 4.1 The mapping

The adapter reads three's material **properties**, so it doesn't matter whether the material came from glTF, FBX, OBJ, code or an editor. Per material type:

| three material | Albedo | Emission | Cutout / see-through |
| --- | --- | --- | --- |
| `MeshStandardMaterial`, `MeshPhysicalMaterial` | `color` × `map`, metalness folded with `metalnessMap` (channel b, as glTF's metallicRoughness) | `emissive` × `emissiveIntensity` × `emissiveMap` | `alphaTest` → cutout; `alphaMap` (channel g) composed into alpha; `opacity`; `transmission` > 0 → `blocksLight: false` |
| `MeshLambertMaterial`, `MeshPhongMaterial`, `MeshToonMaterial` | `color` × `map` (no metalness) | as above | as above (no transmission) |
| `MeshBasicMaterial` | `color` × `map` (it is drawn unlit, but in a GI ray it is a surface with that colour) | none | as above |
| Node materials (`MeshStandardNodeMaterial` and others) with plain properties | as their non-node counterparts | | |
| Node materials with a `colorNode` or `emissiveNode`, `ShaderMaterial`, `RawShaderMaterial` | the plain `color` / `map` if present, otherwise mid-grey 0.5, **with a one-time warning naming the material** (a TSL graph can't be evaluated outside three) | `emissive` if present | `alphaTest`, `transparent`, `opacity` as usual |
| Any | `userData.giAlbedo` (colour), `giAlbedoMap` (texture), `giEmissive`, `giEmissiveMap` override what the adapter would read, for custom shaders | | `giBlocksLight`, `giSingleSided` as today |
| Ignored, as today | `normalMap`, `roughness`, `aoMap`, `lightMap`, `envMap`, clearcoat, sheen, iridescence | | |

**Blended materials** (`transparent` without `alphaTest`; glTF `BLEND`): today they let all light through (decals, Sponza's cypress trees). Open question 3 proposes stochastic transparency instead, with alpha = opacity × texture alpha. Glass (`transmission`) stays `blocksLight: false`. Tinted glass colouring the light behind it is out of scope for v1.

### 4.2 Textures

- **One platform texture per three `Texture`** (keyed by the texture, re-sent when `texture.version` changes, through the existing material edit checks). Composed store layers (2.3) are the core's business. The adapter only says which maps a material uses.
- **Sources:** `texture.image` (`ImageBitmap` from `GLTFLoader`, `HTMLImageElement`, canvas) as the image source; `DataTexture` 8-bit RGBA as `PixelData`, other data types converted to 8-bit once; **`CompressedTexture` (KTX2Loader, gputex) as `CompressedPixelData` from `texture.mipmaps`**. Render-target textures are still skipped with a warning: their pixels live in three's backend.
- **uv channels** 0 and 1 map to the core's two sets. Channels 2 and 3 fall back to 0 with a warning, as now.
- **The M1 fold is removed:** `texture-average.ts` and the `TextureAverages` plumbing go, and with them about 1.1 s of Bistro's main-thread load (T2.13).
- **The mesh-level see-through exclusion is removed** (3.1). A mesh whose materials all let light through simply has every triangle flagged, so its rays pass.

### 4.3 Change tracking

Material and texture edits stay on the budgeted edit checks (three.md: `editChecksPerFrame`), now also comparing each map's `version`. `markChanged(material)` still gives the next frame. The adapter's still-frame cost must stay ≤ 0.2 ms (done-when).

## 5. Fully textured test scenes

### 5.1 Procedural scenes with known answers (new, small)

| Scene | What it checks | Known answer |
| --- | --- | --- |
| **Textured furnace** | the furnace room with a checkerboard albedo whose mean is 0.5 | the same irradiance as a flat 0.5 albedo, within noise: textures are energy-correct and filtered in linear light |
| **uv wrap board** | quads using `repeat`, `mirror` and `clamp`, offset, rotation and uv set 1, each a distinct colour pattern | hit albedo against a CPU reference sampler (oracle-style) |
| **Alpha checker** | a wall with a checkerboard cutout between a lit and a dark room | the light through equals the open share (50%) within noise; hit/miss against the CPU reference tracer |
| **Glass in a wall mesh** | one mesh, opaque walls plus a glass pane (`transmission`) | rays pass the pane only; leak checks unchanged elsewhere |

### 5.2 Real scenes, textured

| Scene | Asset (licence, as in `tools/assets/sources.json`) | What it exercises |
| --- | --- | --- |
| Sponza (Intel) + curtains, ivy, trees | CC BY 4.0 + Intel preface | albedo maps, the curtains' colour, ivy and cypress alpha (the cypress is BLEND, the ivy declared OPAQUE: NOTES) |
| Sponza (Khronos / Crytek) | CRYENGINE licence, download-only | the classic alpha-masked plants and chains |
| Bistro exterior and interior | CC BY 4.0 | 224 textures at 2048², metallic maps, emissive signs, foliage masks; the 6 GB fit |
| Furnished room | CC BY 3.0 | everyday materials, emissive lamp shades |
| Courtyard with the wind tree | CC0 (T3.11's tree when it lands) | the alpha-cutout tree, swaying (M3) and cut out (M4) together |

- **Unmodified** means the viewer loads each file as published, through three's normal loaders. No baked textures, no manifest edits to materials. The texture cap from T2.17 is gone once T3.12 lands.
- **References:** textured Cycles HD references (1920 × 1080, stored cameras) for Sponza with curtains, Bistro exterior and interior, the furnished room and the courtyard. Today's M1 references are "untextured" (the GI material table applied in Cycles, DECISIONS 2026-11-04). They stay for regression, and the textured ones are new. Bistro may not fit Cycles on the 6 GB GPU. If not, it renders on the CPU overnight. A reference render isn't a build, so the "nothing takes minutes" rule doesn't apply, but each one is rendered once and checksummed.
- **Native:** the RTXGI harness loads glTF materials. Whether it samples albedo textures and alpha in its probe rays is checked in T4.9. If it does, textured scenes join the native side-by-side; if not, they compare against Cycles only.

## 6. Compressed textures so Bistro fits in 6 GB

**The problem (T2.17):** Bistro exterior's 224 textures at 2048² are 5.0–5.1 GB uncompressed with mips (224 × 22.4 MB), in three's own memory. With DDGI's 0.6 GB that overflows the 6 GB RTX 3060 Laptop, and a GI frame took 517 ms instead of 47 ms. The test pages use a texture cap (`VITE_DDGI_MAX_TEXTURE_SIZE`) until this is fixed. Dennis chose GPUTex, with KTX2 as the fallback (DECISIONS 2026-11-09), as M3's T3.12.

**With BC7** (1 byte a texel; normal maps BC5, also 1 byte): 224 × 5.6 MB = **1.25 GB**.

| On the 6 GB RTX 3060 Laptop, Bistro exterior | Today | With T3.12 + M4 |
| --- | --- | --- |
| three's textures | 5.0 GB | 1.25 GB |
| DDGI (field, scene copy, trees) | ~0.6 GB | ~0.6 GB (~0.37 GB after T3.13 trims) |
| GI texture store | 0 | ~35 MB (78 MB worst case) |
| Left for three's geometry, render targets, the upscaler, the browser | negative | about 3.5–4 GB |

**Is upstream gputex enough? Yes, for three's rendering.** Checked against the npm package (0.14.0 tarball; 0.14.1 is current, published 2026-10-31):

- `gputex/three` exports `compressTexture(source, { device, mipmaps: true, hint, colorSpace, flipY })` and `GputexLoader`. On a compressed path they return a three **`CompressedTexture` with every mip**, on the device you pass (three's own, so no second WebGPU device).
- Internally it encodes on the GPU, reads the blocks back (`mapAsync`), and builds the `CompressedTexture` from bytes (`buildCompressedTexture`). three then uploads the bytes again. That round trip costs load time and a transient CPU copy (5.6 MB per Bistro texture), not frame time or VRAM. Load time is the host's and outside the startup target. Dennis ranks dynamic behaviour over startup (DECISIONS 2026-11-06).
- HomeFig's fork adds one method, `Encoder.encodeMipChainIntoTexture(src, target, { layer })`, which writes straight into a `GPUTexture` layer with no readback. That saves the round trip, but three would then have to draw from a `GPUTexture` it doesn't own (`ExternalTexture` as a material `map`, unverified in r187). Not needed for T3.12.
- **Where the fork would fit:** compressing the core's own store into BC7 layers (4× less memory). At 35 MB that isn't worth a core dependency (open question 4).

**What T3.12 must take care of** (notes for whoever builds it):

- Ask for `texture-compression-bc` (and `-astc`, `-etc2` where present) when requesting the device. `optionalFeatures(adapter)` doesn't list them today. The viewer's device request, or `requestGIDevice()`, adds them. The M5 Pro has BC in Chrome on Apple silicon. Check Safari.
- glTF textures have `flipY = false`. Pass `flipY: false` to gputex. Copy `wrapS/T`, `repeat`, `offset`, `rotation`, `channel` and `colorSpace` to the new texture.
- Use hint `'normal'` for normal maps (BC5). On ASTC-only devices gputex stores normals in `.ra` (`astcNormalRemap`), which three's normal map node doesn't read. Prefer BC wherever available, and fall back to uncompressed normals elsewhere.
- Keep `texture.mipmaps` (the bytes) on the `CompressedTexture` at least until GI has read its level. The GI copy reads it there (2.4), and so does KTX2's output.
- Prewarm the encoders (`prewarmCompressTexture`) during the scene load: about 100 ms of compile once.

**How it fits M4:** T3.12 makes three's textures small, and M4's compressed-mips source (2.4) lets GI read those same textures for 64 KB each. They are independent. M4 works with uncompressed textures too (images route), and T3.12 works without M4. If M4 starts before T3.12, Bistro's checks keep using the cap.

## How it fits the existing kernels (no parallel GPU paths)

| Piece | After M3 | M4 |
| --- | --- | --- |
| `giSurfaceAt` | flat material colour; uv set 0 interpolated | + texture samples per mapping with a ray-cone mip; `giSurfaceAtCone` |
| Scene copy | uvs, material records with uv mappings and texture slots | + texture layer per slot, `opacity`, `alphaMode`, `vertexColors`, metalness and alpha refs; + vertex colours (1 word per vertex, when present) |
| Texture store | none (texture handles only) | new: one `texture_2d_array` + one sampler; `gi/texture-store` at load and on updates |
| Tracer walk | every triangle opaque; mesh-level exclusion in the adapter | consult flag × instance flag → material lookup, cutout, stochastic, see-through, single-sided |
| Triangle flags | written once from `triangleFlags` (always 0) | `gi/triangle-flags` after each tree install and on material changes |
| `gi/direct` | albedo from the surface record | unchanged (the record now holds textured albedo); any-hit shadow rays get alpha tests |
| three adapter | texture-average fold; see-through mesh exclusion | real textures as handles (images, data, compressed mips); fold and exclusion removed |
| Storage buffers | `ddgi/trace` 8 | unchanged (the store is a sampled texture) |

## Cost against the budgets (estimates, RTX 3060, to be measured)

| Item | When | Cost |
| --- | --- | --- |
| Textured `giSurfaceAt` | per probe-ray hit | 1–2 samples + ~6 loads; **probes a frame at 2 ms on Sponza ≥ 0.85× M3's** (planning bar) |
| Alpha tests in the walk | flagged candidates only | 0 on opaque scenes; foliage rays 0.3–0.6× (planning number, T4.0) |
| `gi/texture-store` | load; then ≤ 4 textures a frame | ~0.05–0.2 ms GPU per texture (estimate) |
| `copyExternalImageToTexture` | load, per image | unknown: T4.0 (the risk) |
| Compressed-mips upload | load, per texture | 64 KB `writeTexture` each |
| `gi/triangle-flags` | tree installs, material flips | < 0.05 ms per 100k triangles |
| GI sampling in pixels | every frame | unchanged |
| Memory | | store ~27–35 MB (Sponza, Bistro at 256²); staging pool ≤ 64 MB at load only |

## Proposed task breakdown

M4 keeps the house rules: features over timing, one VALID run per gate, an Opus review before done, and one fix attempt for a missed bar, then a known miss in DECISIONS.

**Order and lanes** (open question 8 proposes starting alongside M3):

- *Lane A, textures:* T4.1 → T4.2 → T4.3, then T4.6 (adapter).
- *Lane B, traversal:* T4.4's see-through and single-sided parts need no textures and can start at once; its cutout and stochastic parts follow T4.2.
- *Lane C, scenes:* T4.7 (assets and references) from the start.
- T4.0 first, small, to settle the two unknowns.
- *File overlap with M3:* T4.4 edits the walk's leaf test (`trace/wgsl.ts`); M3's T3.2 and T3.3 edit tree building and refit, not the walk. They share `gi/triangle-flags`' call sites at tree install, so whichever lands second wires it.

| ID | Task | Agent | Size | Depends on |
| --- | --- | --- | --- | --- |
| T4.0 | Spike: main-thread and GPU cost of getting Bistro's textures in (image copy vs compressed mips; does three r187 keep `CompressedTexture.mipmaps` after upload, from KTX2Loader and gputex); alpha-test trace speed on the courtyard tree; count Bistro's albedo and emissive textures | implementer | S | none |
| T4.1 | Hit uvs and footprint: `giSurfaceAt` applies each mapping (channel, transform, wrap), the ray-cone lod, `giSurfaceAtCone`; CPU reference | implementer | S | none |
| T4.2 | Texture store: the array, slots and growth, `gi/texture-store` (downsample in linear light, composition, mips, coverage-preserving alpha), all four sources incl. compressed mips, upload budget, pending textures and their material marks, `updateTexture`, memory stats, `requiredLimits` for layers | implementer | M | T4.1, contract (T4.5) |
| T4.3 | Textured hit shading: albedo, emissive, metalness and vertex colours sampled at hits; `sampleTextures` debug setting; textured furnace and uv board pages | implementer | M | T4.2 |
| T4.4 | Materials in the walk: consult and instance flags, `gi/triangle-flags`, see-through and single-sided per triangle (first), then cutout and stochastic; CPU reference tracer and oracle cases; `capabilities.alphaCutouts` | implementer | M | T4.0; T4.2 for the cutout half |
| T4.5 | Contract additions (section 7) in the TypeScript and platform.md, after Dennis's answers; the plain-WebGPU example textured, for the neutrality check | main session, then implementer | S | Dennis's answers |
| T4.6 | three adapter materials v2: the mapping table (4.1), texture sources incl. `CompressedTexture`, vertex colours, overrides, fold and mesh exclusion removed, texture-version checks | implementer | M | T4.3, T4.4 |
| T4.7 | Textured scenes and references: the four procedural scenes (5.1); textured Cycles HD references (5.2); the viewer loads every scene textured by default | wrangler + implementer | M | none (pages need T4.3) |
| T4.8 | Viewer: debug views of GI's albedo at hits and of the chosen mip; a textures on/off toggle; hover help | implementer | S | T4.3 |
| T4.9 | M4 measurements: the SPEC table on every textured scene (RTX, M5 Pro if set up), correctness on the Intel UHD, memory per scene, Dennis's review | gpu-tester | M | all above, T3.12 for Bistro |
| T4.10 | M4 note and M5 breakdown | main session | S | T4.9 |

T3.12 (compressed textures for three) stays in M3 as decided. If M4 runs first, it moves to M4 unchanged.

### Draft done-whens

- **T4.0 Spike.** Done when each question has a written answer with numbers:
  1. main-thread ms per texture and in total for Bistro exterior's albedo and emissive images through `copyExternalImageToTexture`, and through the compressed-mips route on gputex output;
  2. whether `CompressedTexture.mipmaps` still holds the bytes after three's upload (KTX2Loader and gputex);
  3. closest-hit and any-hit Mrays/s through the courtyard tree's crown with a prototype alpha test, against solid cards;
  4. Bistro's count of albedo and emissive textures, and the store's memory at 128, 256 and 512.
- **T4.1 Hit uvs.** Done when:
  1. the uv at hits matches a CPU reference within 1e-5 for both uv sets, with transforms, on both GPUs (oracle page);
  2. the lod formula's unit tests pass on a known quad at known distances and angles (lod within 0.05).
- **T4.2 Texture store.** Done when:
  1. every layer's mip 0 matches a CPU downsample in linear light within 2/255 per channel, for each source kind (image, `PixelData`, `GPUTexture`, BC7 mips);
  2. growing past 256 layers and destroying and recreating 500 textures leaves the recompile counter at 0;
  3. a cutout layer's coverage at every mip is within 2 percentage points of mip 0's;
  4. Bistro exterior's store memory is reported, ≤ 80 MB at 256², and its main-thread texture time at load is at most M2's texture-average time (about 1.1 s; T4.0 sets the real bar).
- **T4.3 Textured hit shading.** Done when:
  1. the textured furnace's irradiance is within 1% of the flat 0.5 furnace (field mean, after convergence);
  2. the uv board's hit albedo matches the CPU reference sampler (mean per-quad error ≤ 1/255);
  3. Sponza's probes a frame at 2 ms are ≥ 0.85× M3's (one VALID run); untextured scenes' are unchanged within 3%;
  4. with `sampleTextures` off, fields match M3's within 0.1%.
- **T4.4 Materials in the walk.** Done when:
  1. hit/miss and closest-hit triangle agree with the CPU reference tracer on ≥ 99.9% of rays (rays within a texel of an alpha edge excluded) for the alpha checker and the glass-in-wall mesh, on both GPUs;
  2. the alpha checker's light through is 50% ± 3% of the open-wall case; stochastic at alpha 0.5 is 50% ± 3% after convergence;
  3. scenes with no flagged triangles trace within 3% of M3's speed (bench page);
  4. M1's leak checks and two-room's door case still pass; the courtyard tree's rays/s is recorded.
- **T4.5 Contract.** Done when the additions agreed with Dennis are in `contract/*.ts` and platform.md's Revisions, and the plain-WebGPU example's textured scene matches the three adapter's GI within T1.7's neutrality tolerance.
- **T4.6 three adapter v2.** Done when:
  1. a test matrix (each material type in 4.1 × map, metalnessMap, alphaMap, alphaTest, transparent, transmission, vertex colours, `CompressedTexture`) gives the expected descriptors;
  2. Sponza's `dirt_decal` and curtains are traced (no mesh exclusion), and a mixed opaque/glass mesh lets light through its glass only;
  3. the still-frame adapter cost stays ≤ 0.2 ms on Sponza and Bistro interior;
  4. no texture average is computed at load (profile).
- **T4.7 Scenes and references.** Done when every scene in 5.1 and 5.2 opens textured in the viewer from its manifest with no per-scene material code, each 5.2 scene has a textured HD Cycles reference with its source, settings and checksum recorded, and licences are in `THIRD_PARTY.md`.
- **T4.8 Viewer.** Done when the albedo-at-hits and mip views show on every scene, the textures toggle switches live, and Dennis can use them without instructions.
- **T4.9 M4 measurements.** The SPEC table above on each textured scene, plus memory per scene and Bistro exterior on the 6 GB RTX without the cap. Done when each target is met or its miss is explained and agreed with Dennis.

## 7. Contract changes (for Dennis)

Host-facing (the platform API was signed off 2026-11-04, so each is a decision):

1. `TextureSource` gains **`CompressedPixelData`** `{ format: GPUTextureFormat, levels: { data, width, height }[] }`. This closes platform.md gap 5.
2. **`updateTexture(texture, descriptor)`**, for changed contents.
3. `MaterialDescriptor` gains:
   - `metalness?: number` and `metalnessTexture?: { texture, channel, uv? }` (the fold platform.md planned for the atlas);
   - `alphaTexture?: { texture, channel, uv? }` (three's `alphaMap`);
   - `opacity?: number` (multiplies alpha);
   - `alphaMode?: 'opaque' | 'cutout' | 'stochastic'` (a material with `alphaCutoff` and no `alphaMode` is `'cutout'`, so existing hosts don't change);
   - `vertexColors?: boolean`.
4. `MeshDescriptor.colors?: Float32Array | Uint8Array` (rgba per vertex).
5. Settings: `textureStore.size`, `textureStore.maxLayers`, `textureUploadsPerFrame`, `textureLodBias`, `alphaTestMinLod`, `sampleTextures` (debug).
6. `requiredLimits` also asks for the adapter's `maxTextureArrayLayers`.

Technique-facing: `giSurfaceAtCone`, `GiSurface.uvFootprint`, `RayQueries.capabilities.alphaCutouts = true`, `stats.memory.textures`.

## Risks, and what to spike first

1. **Main-thread cost of copying images in** (T4.0). If `copyExternalImageToTexture` on Bistro's 2048² bitmaps costs as much as the averages did, the gain is only correctness. The compressed-mips route avoids it wherever T3.12's gputex is used, which the viewer will be.
2. **Foliage traversal cost** (T4.0). Alpha-tested crowns can make rays several times slower. Levers: the "fully opaque leaf" bit, and Budget v2 spends fewer probes there by itself.
3. **Startup.** T2.12 measured 1.90 s against 2.0 s on Sponza. Building ~80 store layers in the load burst may break the bar (open question 6).
4. **Stochastic transparency's noise.** Semi-transparent surfaces add ray noise that hysteresis must absorb. T1.12's stability threshold applies. If it fails on Sponza's cypresses, the fallback is a cutout at 0.5 for blended materials with an alpha texture.
5. **Accuracy may move.** Textured albedo changes where light bounces (a texture's average is right in total but not per patch), so the M1 accuracy numbers (+6.6% on Sponza's composite path) are re-measured against new references, not compared with the old ones.
6. **ASTC-only devices** (iPad, older iOS): normal maps compressed by gputex read wrongly in three (`astcNormalRemap`). A T3.12 concern for three's rendering, not for GI.

## Later

- A second, smaller array for small textures, if store memory matters.
- BC7 store layers through the gputex fork's `encodeMipChainIntoTexture` (4× less store memory), if 512² stores become the default.
- Tinted transmission (coloured glass tinting the light behind it).
- Textured roughness for the RT-reflections technique after v1.
- Video textures.

## Open questions for Dennis, with recommendations

1. **Texture arrays instead of an atlas, 256² per layer by default** (section 2.1). *Recommend yes:* hardware wrapping and clean mips per layer, no gutters or repacking, and ~35 MB on Bistro. The setting allows 128 or 512.
2. **The contract additions in section 7** (compressed sources, `updateTexture`, metalness and alpha maps, opacity and `alphaMode`, vertex colours). *Recommend yes to all.* Each is small, optional and backward compatible. Without metalness maps in the core, metals bounce light like paint again once the averages go. Without vertex colours, kit-bashed and procedural scenes lose their colour.
3. **Blended materials as stochastic transparency in GI rays** (glTF BLEND: decals, Sponza's cypresses), instead of letting all light through. *Recommend yes,* with glass (`transmission`) still passing light. It is the physically right average, it fits DDGI's many rays and blending, and it makes decals bounce their own colour. The fallback is risk 4.
4. **Compressing GI's own texture store** with the gputex fork. *Recommend no for v1:* 35 MB doesn't justify a core dependency or the fork. Revisit if 512² becomes the default.
5. **gputex in the library?** *Recommend no:* compression is the host's loader choice. The adapter reads any `CompressedTexture` (gputex or KTX2) without depending on either, and only the viewer and test pages use gputex (T3.12, upstream 0.14.x; 0.14.1 is the current release, more than two weeks old).
6. **Textures count towards the 2 s startup.** *Recommend yes:* GI that starts flat and then shifts as textures land looks like a pop. If the bar breaks after one fix attempt, it is recorded as a known miss.
7. **Custom shaders** (`colorNode`, `ShaderMaterial`): read the plain `color`/`map` if set, otherwise mid-grey 0.5, with a one-time warning and `userData.giAlbedo`/`giAlbedoMap` overrides. *Recommend yes.*
8. **Start M4's core lanes now, alongside M3**, rather than after M3's sign-off. *Recommend yes:* textured GI on Sponza and Bistro is the biggest visible gap for a base DDGI to compare against, the lanes touch different files from M3's (one shared call site), and T4.0, T4.1 and T4.7 have no dependencies.
9. **T3.16's texture-average items** (async averages from three's smallest mip). *Recommend dropping them* if M4 runs alongside M3: T4.6 removes the averages.
