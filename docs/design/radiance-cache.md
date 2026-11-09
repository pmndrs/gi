# radiance-cache: AMD's two-level radiance cache on the platform

Status: phase 1 (platform groundwork) started 2026-11-09 on branch `radiance-cache` (cut from `m2` at 5de856b). Adapted from the research plan (draft 2, 2026-11-09) in the `ddgi` research repo, `research/gi-1.0/PLAN.md`, with its sources: `source-map.md` (Capsaicin's passes with path:line), `native-capsaicin.md` (Capsaicin's per-pass times on our scenes), `paper.md`, `idtech8-performance.md`, `pmndrs-siblings.md`, `related-work.md`. Dennis's decisions are in DECISIONS (2026-11-09, "radiance-cache").

Based on AMD GI-1.0/1.2. The reference is AMD Capsaicin v1.3 (`914b915`, MIT), which ships GI-1.2: GI-1.0, plus GI-1.1's glossy reflections, plus a second-bounce layer in the world cache.

## What it is

The second technique on the platform, next to DDGI, at `packages/ddgi-core/src/techniques/radiance-cache/`, published as `@pmndrs/ddgi/radiance-cache`.

- **Screen cache:** one 8x8 hemi-octahedral probe per 16x16 screen tile, placed on visible surfaces, filled over several frames and reused by reprojection. Detail where the camera looks, quick reaction to change.
- **World cache:** a spatial hash grid of outgoing radiance at ray hits, cells sized by camera distance, freed when unused. Probe rays end in it. A second layer of cells carries the next bounce.
- **On the platform:** the scene copy, the tracer, hit shading, the light grid, the work budget and the change feed. WGSL plus TypeScript orchestration, engine-free. three.js r187 is the only adapter until the technique is validated (the same rule as DDGI).

Why: it is what DDGI is not. DDGI is a fixed probe grid, stable but blurry and slow to react. This one is screen-driven: detail where the camera looks, fast reaction to moving objects, no volume to place. The two stack well (id Tech 8 ships that stack; see "Extensions").

## How we work: the faithful port first

1. **Build it as AMD built it.** Capsaicin's default configuration, pass by pass: its algorithms, ray counts, rates, cache sizes and per-hit cache population. That is the **baseline**. If it is slow on our tracer, that is information, not a reason to change it up front.
2. **Every difference from Capsaicin goes in the divergence register** below: what we do instead, its kind, why, and the way back, including workarounds others use (browser vendors, other WebGPU projects, engines).
3. **Performance ideas are levers, not rewrites.** Each lever is a switch beside the faithful path, measured against the baseline on the same scene and camera, and adopted only with a written reason. A slow faithful path is kept and marked for attention.
4. **Priority versions are extra options.** A cheaper or prioritised variant of a pass can be built behind a switch. It doesn't replace the baseline.

## What we port, and from where

| Source | License | Use |
| --- | --- | --- |
| Capsaicin v1.3 (`914b915`), `src/core/src/render_techniques/gi1/` | MIT (AMD) | The baseline. Port pass by pass; ported files keep AMD's notice. About 5,000 useful HLSL lines. |
| Capsaicin's SSGI technique (near-field GI, bent-normal occlusion) | MIT (AMD) | Part of the baseline: GI-1 reads its output. About 350 lines. |
| Capsaicin's blue-noise tables (`blue_noise_sampler_samples.h`) | No license header | Not copied. Regenerated (D7). |
| GI-1.0 paper (arXiv 2310.19855), GI-1.1 paper, GI-1.2 page | Papers | Algorithms and reasons. |
| FidelityFX Brixelizer GI | MIT (headers to verify) | Ported in another workstream; plugs in as a world cache and tracing backend ("World cache as a slot"). |
| NVIDIA SHaRC, RTXGI, NRC | NVIDIA RTX SDK license | Never ported. Ideas from public docs only. |
| Lumen, id Tech 8, EA surfels | Proprietary | Slides only. |

**The baseline is Capsaicin's default configuration:**

- In: screen probes (one probe per 16x16 tile per frame, `QuarterSpp`, at full render resolution), reprojection, adaptive patching of empty tiles, luminance-CDF ray guiding, the off-screen LRU probe cache, biased-hysteresis blending, the probe filter, SH projection and interpolation, the hash-grid cache with its 4-level tile MIPs, decay and **per-hit population** (one sampled light and one shadow ray per hit), multi-bounce cells (GI-1.2), SSGI near-field and bent-normal occlusion, the image-space GI denoiser and the final resolve.
- In, once the platform has the material data: glossy reflections (GI-1.1). They are on by default in Capsaicin; their absence is D8.
- Out (off by default in Capsaicin, so not part of its default behaviour): the DXR 1.0 path, world-space ReSTIR, temporal radiance feedback, the debug-stat kernels.

## The native yardstick

Capsaicin GI-1.2 on our own scenes, from our manifest cameras (field of view matched), 1280x720, RTX 3060 Laptop with hardware ray tracing, warmed averages over frames 101 to 300 (`native-capsaicin.md`, "Our scenes in Capsaicin"). Bistro uses 1k textures: at 2048 px it doesn't fit the 6 GB card in Capsaicin either. Run-to-run noise is about 15% (laptop GPU, clocks not locked, also driving the display).

| Group (ms) | Sponza + curtains | Bistro interior | Bistro exterior | Two-room |
| --- | --- | --- | --- | --- |
| Probe tracing and cell population | 4.75 | 7.81 | 7.56 | 1.31 |
| Cache update and resolve | 0.74 | 0.67 | 0.80 | 0.65 |
| Screen probes | 1.38 | 1.18 | 1.31 | 1.51 |
| Denoise and resolve | 1.08 | 0.71 | 0.95 | 0.56 |
| **Diffuse subtotal** | **7.95** | **10.38** | **10.62** | **4.03** |
| Reflections (GI-1.1) | 0.69 | 0.31 | 0.44 | 0.17 |
| Reference path tracer, 1 spp | 23.8 | 34.8 | 53.0 | 7.4 |

- **Compare per pass, on the same machine, scene and camera.** Our baseline's time for each Capsaicin pass sits next to Capsaicin's. Passes far above native go on the attention list.
- **No frame-time target before the baseline exists.** Dennis sets it from the per-pass table (phase 5).
- **Accuracy is judged against path tracers, not Capsaicin's GI image.** Under punctual lights Capsaicin's GI-1.2 image is far darker than its own path tracer (mean linear luminance on Sponza 0.00085 against 0.016; Bistro interior 0.029 against 0.197; two-room 0.83 against 1.08). Its screen pass lights from emissives and the environment only; punctual light arrives through the cache's light sampling, and its renderer expects a separate direct-light technique. Our port is the same: indirect only, with three drawing direct light. So Capsaicin's GI image is the reference for behaviour (stability, reaction, artifacts) and per-pass cost; Capsaicin's path tracer and our Cycles HD references are the reference for accuracy. Phase 5 confirms the dark result from Capsaicin's source before any difference is called ours.

**What to expect from tracing.** Our compute tracer runs about 105–120 M closest-hit rays/s on Sponza on the RTX 3060, and 48–60 M shadow rays/s in real dispatches (NOTES, "Tracing"). Capsaicin's defaults trace one probe per 16x16 tile with 64 rays each (at 1280x720: 80 × 45 = 3,600 probes, 230,400 probe rays), about 0.3× that again for multi-bounce, and about one shadow ray per hit. Phase 1's ray cost table (RC1.5) turns that into a predicted tracing time per scene before any of the technique exists.

## Divergence register

Every way our port differs from Capsaicin's default behaviour. **Kind:** P platform (WebGPU or our platform lacks something), L license, Q affects quality, S speed lever (only after the baseline). A row's "way back" says how to remove the difference, including what others do.

| ID | Capsaicin | Ours | Kind | Why | Way back / others' workarounds |
| --- | --- | --- | --- | --- | --- |
| D1 | DXR 1.1 inline ray queries; hardware traversal; alpha test in the loop | The platform's compute BVH (`giTraceClosest`, `giTraceAny`); cutouts from the material | P | WebGPU has no ray tracing | Our BVH work (wide nodes, H-PLOC, any-hit fixes); Brixelizer's SDF backend (in progress elsewhere); any WebGPU ray-query proposal; compare against three-mesh-bvh's WebGPU walk and Webgiya |
| D2 | Bindless material textures sampled at every hit | Texture-average albedo through `giSurfaceAt` (as DDGI) | P, Q | WebGPU has no bindless textures; the platform's hit record is per material | Texture arrays or an atlas with real UVs at hits; compressed textures (GPUTex, T3.12) to afford it; the WebGPU bindless proposal |
| D3 | Mesh-shader visibility buffer (instance, primitive, barycentrics) | Host-provided screen buffers through `FrameScreen` (depth, normal, velocity, albedo, instance id); three fills them from its MRT scene pass | P | WebGPU has no mesh shaders; three already renders these | three's own VisibilityBuffer pipeline; a platform raster pass over the scene copy for hosts that can't provide them |
| D4 | `LightSamplerGridStream` (64 streamed reservoirs per cell, wave intrinsics in the build) | The platform light grid (16 picks per cell weighed at the hit, N shadow rays per hit) | P, Q | The platform rule "hit shading is shared"; no second light system | Port the stream sampler as a platform option (subgroups in the build) if the light grid's image differs measurably from Capsaicin's |
| D5 | Read-write `RGBA16F` textures | `texture-formats-tier2` read-write storage textures | P | Core WebGPU allows read-write storage only on 32-bit one-channel formats | Declared fallback: ping-pong (a sampled read plus a write-only storage texture). RC1.4: the feature works on both GPUs, on our device and three's, and the fallback was run and works too |
| D6 | Unbounded buffer arrays (about 30 hash-cache buffers) | A few large packed storage buffers at fixed offsets | P | Per-stage storage-binding count limit | None needed; same data |
| D7 | Shipped blue-noise and Sobol tables | Regenerated tables (or the platform's blue noise) | L | The tables have no license header | Ask AMD for terms if the regenerated noise behaves differently |
| D8 | GGX BRDF at hits from a packed material; glossy reflections on by default | Lambert at hits; no reflections until the platform has roughness and metalness | P, Q | Platform materials are diffuse-only (`contract/scene.ts`) | Roughness and metalness in the material descriptor (also needed by planned RT reflections) |
| D9 | FSR 2/3 in the frame chain | `@pmndrs/upscaler` | P | Our sibling library | Equivalent; watch upscaler issue #7 (jitter against GI history) |
| D10 | D3D12 per-pass timestamps | WebGPU `timestamp-query` per pass | P | Measurement only | The platform's `GpuTimer` already does this |
| D11 | Half-precision packing via `f32tof16` | `pack2x16float` / `unpack2x16float`; `shader-f16` where it helps | P | WGSL equivalents | None needed |
| D12 | One queue (id Tech 8 uses async compute) | One WebGPU queue | P | WebGPU has no async compute | WebGPU multi-queue proposals; order passes to keep the GPU busy |
| D13 | The visibility buffer is Capsaicin's own texture, read in place | `ddgi-three` hands three's MRT targets to `FrameScreen` without a copy (except the instance id: blending needs an 8-bit target, which a small pass turns into the `r32uint` texture each frame) by reading one field of three r187's WebGPU backend, `renderer.backend.get(texture).texture`, in one function (`three-gpu-texture.ts`) that refuses any other three revision | P | three r187 has no public way to get a render target's `GPUTexture` (only the other way: `ExternalTexture` wraps ours). A copy into our own textures would cost bandwidth every frame; the backend field is what three's own XR path writes | A public three API returning a texture's `GPUTexture` (none asked for upstream as of 2026-11-09: no issue or PR found; #31653 added the reverse); or the host renders into `GPUTexture`s we own, wrapped as `ExternalTexture`s in the pass's render target, which needs resizes handled by us; or a copy (`copyTextureToTexture`) per frame |
| D14 | A D3D12 device with the hardware's limits: about 1 GiB of hash cache in about 30 buffers | When three r187 makes its own device (`new WebGPURenderer()` with no `device`), it has WebGPU's default limits: a 128 MiB storage binding, a 256 MiB buffer and 8 storage buffers per stage, on the RTX 3060 and the Intel UHD (Chrome 157, RC1.4). Our device (`requestGIDevice`) gets 2 GiB bindings and 16 storage buffers per stage on both. The same holds for `maxColorAttachmentBytesPerSample`: three's own device keeps 32 bytes, while the screen inputs with the composite need up to 56 (RC1.1) | P | three requests every feature it knows that the adapter has (all five RC1.4 features work on its device), but no limits: `requiredLimits` defaults to `{}` (`three/src/renderers/webgpu/WebGPUBackend.js:97, 245-248`) | Declared fallback: none at the default limits. The technique throws `GILimitError` naming `maxStorageBufferBindingSize` ("World-cache memory"). The host passes our device (`createGIRenderer`, or `requestGIDevice` then `new WebGPURenderer({ device })`) or passes the adapter's limits as `new WebGPURenderer({ requiredLimits })` (three's own parameter, `WebGPUBackend.js:71`). Upstream: a three change that asks for the adapter's storage limits when it makes its own device. A smaller hash-cache capacity would fit 128 MiB, but isn't Capsaicin's default |
| D15 | f32 math, half precision only in storage (`f32tof16`) | f16 math must stay below 65,504; don't rely on overflow giving infinity (the Intel UHD keeps a wider intermediate). Measured: `300h * 300h / 1000h` is infinity on the RTX 3060 and 90 on the Intel UHD (Chrome 157, RC1.4) | P | WGSL makes a run-time f16 overflow an indeterminate value (§15.7.3), so both answers are allowed; and radiance easily passes 65,504 | Declared fallback: f32 math with `pack2x16float` storage (D11), which is also Capsaicin's way. An f16 kernel clamps or pre-scales its inputs, and is checked on both GPUs. Not a gap for the faithful baseline, which doesn't do f16 math |
| D16 | `SampleScreenProbes` picks a ray's lobe by the surface's BRDF: with probability `calculateBRDFProbability(F0, ...)` it samples GGX, otherwise the diffuse cell (`gi1.comp:952-978`; `gi1_disable_specular_materials` is false by default) | Every probe ray takes the diffuse branch, as Capsaicin built with `DISABLE_SPECULAR_MATERIALS`. The random draws keep Capsaicin's order (the GGX pair is drawn and dropped) | P, Q | The platform's materials are diffuse-only (D8), and `FrameScreen` has no roughness or F0. Note (RC2.1 review): with Lambert-only materials Capsaicin still sends a few percent of probe rays down the GGX branch, since F0 = lerp(0.04, albedo, metallicity) (`materials.hlsl:186`) is never 0; this row drops that share. Capsaicin's default build also draws one more `rand()` (`gi1.comp:960`) than the `DISABLE_SPECULAR` build our order follows, which only matters under D18 | With D8's way back: roughness and metalness in the material descriptor, and a roughness channel in `FrameScreen` (RC1.1's channel requests), then the branch as Capsaicin has it |
| D17 | The probes read Capsaicin's own visibility buffer: the ray origin is the exact point on the triangle moved off it by `offsetPosition` (`gi1.comp:1135-1150`), the probe frame is the geometry normal, and a sky pixel is one with a zero geometry normal (`gi1.comp:570-571`) | The probes read the host's `FrameScreen`: the position comes from `depth` through the frame's inverse view-projection (camera-relative, f64 on the CPU), the origin is pushed along the normal by the platform's rounding offset plus what four steps of the stored depth move the point; the frame is `FrameScreen.normal` (three's MRT gives the shading normal, not the geometric one); a sky pixel is one at the far plane's depth | P, Q | No visibility buffer on WebGPU (D3); `FrameScreen` carries depth and normals, and its normal is whatever the host draws | A visibility channel in `FrameScreen` (instance, triangle, barycentrics; three r187's VisibilityBuffer pipeline could fill it) or a geometric-normal channel; until then RC2.2 checks the image for self-shadowing at grazing angles |
| D18 | `MakeRandom(seed, frame)` starts its PCG from `g_RandomSeedBuffer[seed % size]`, a table of `mt19937` numbers the CPU makes at start-up (`random_number_generator.hlsl:108-116`; from `std::random_device` unless deterministic) | The same PCG and the same increments, started from `pcgHash(seed)` (`math/hash.hlsl:33-39`) instead of a table entry | P | No table to make and bind; the table is random data, so its numbers can't be matched anyway (by default they differ run to run in Capsaicin too) | Upload a seed table if a statistic ever differs; none expected |
| D19 | A missed ray's hit distance is `FLT_MAX` packed with `f32tof16` (`gi1.comp:1094, 1156`); a cell no ray reached gets `MAX_HIT_DISTANCE` (1e9) in the probe atlas (`gi1.comp:1232`) | Both store 65,504, the largest finite half | P | WGSL's `pack2x16float` and stores to `rgba16float` give an indeterminate value outside the half range (§16.9, D15's rule) | None needed: 65,504 m reads as "far" wherever a hit distance is read (phase 4's reprojection and filter) |
| D20 | A missed probe ray samples the environment cube map (`g_EnvironmentBuffer`, `gi1.comp:1082`) | `giSkyRadiance`: the platform's sky, baked to a 256 × 128 radiance map (shared hit shading, as DDGI) | P, Q | The platform rule: hit shading, the sky included, is shared between techniques | A larger sky bake if fine sky detail shows in the probes; the bake's size is a platform setting |
| D21 | Emissive triangles are lights: `LightBuilder` gathers them every frame (`gather_area_lights.comp`) into the light list the sampler draws from, so a hit beside an emissive surface is lit by it through one light sample and a shadow ray | The platform's lights are the host's punctual lights (directional, point, spot) and the sky; an emissive surface reaches a probe only when a probe ray hits it (its emission, `gi1.comp:2439-2452`). Seen in RC2.1 on `cornell-box`, lit by its emissive ceiling alone: the walls' probe cells hold no direct light, only the cells whose ray hit the ceiling are bright | P, Q | Hit shading is shared and has no area lights; DDGI gets emissive light only through its field's bounces too | Emissive triangles as platform lights: gathered into the light table (or the light grid) with their area and radiance, sampled in `giDirectIrradiance` and `gi/direct`. A platform task, needed before RC2.2's Cornell image can match Capsaicin's |

Speed levers (kind S) get a row only when one is adopted. Phase 1 adds rows for any gap RC1.1 or RC1.4 finds (a three backend step, a missing feature), numbered from D13. D13 is held for RC1.1's three backend step, if one is needed; RC1.4's rows are D14 and D15.

## What the platform has, and what phase 1 adds

**Already there** (branch `m2`): inline tracing from our own kernels (`giTraceClosest`, `giTraceAny`) plus batched rays (`platform/contract/queries.ts`); shared hit shading (`giSurfaceAt`, `giDirectIrradiance` with shadow rays, the sky, emission; `contract/shade.ts`); the light grid (close to Capsaicin's sampler, D4); the `BounceSource` slot so techniques stack; the change feed (`DirtyRegion.moved`); the GPU-refit top level; Budget v2 and timing; velocity in three r187's MRT; `@pmndrs/upscaler` in the viewer; `requiredLimits(adapter)` asking for the adapter's maximum buffer and binding sizes (about 2 GiB on both laptops).

**Gaps phase 1 closes:**
1. **Screen inputs in compute.** `FrameScreen` has depth, normal, optional velocity and instance id, but no albedo, and nothing in `ddgi-three` fills it (`docs/api/platform.md`, gap 3).
2. **No platform/technique lint rule.** SPEC says the boundary is lint-enforced; only the engine boundary is.
3. **GPU utilities:** prefix scan, stream compaction and indirect-dispatch arguments as platform pieces. `packages/bvh` has a subgroup scan (`gpu-scan.ts`, `gpu-workgroup-scan.ts`) and DDGI's scheduler has its own scan with a fallback; share rather than add a third.

Diffuse-only materials (D8) stay a gap until the reflections work.

### Screen inputs: the flexible route

The technique needs the per-pixel data three already renders. The platform takes it through `FrameScreen`, as textures the **host** provides. The three adapter fills that contract from the host's MRT scene pass. That keeps the core engine-free, lets any host hand over its own buffers, and lets a host without them use a platform raster pass over the scene copy later (D3's way back). If three r187 won't hand over its render targets as `GPUTexture`s without reaching into its backend, the adapter takes the smallest version-pinned step and records it in the register and DECISIONS.

## World cache as a slot

The baseline world cache is Capsaicin's hash grid. Behind one interface ("radiance leaving this hit point toward this ray", plus "these cells need updating"), others can plug in: Brixelizer GI's brick cache (from the separate port, which also brings SDF tracing as a second query backend), a Lumen-style surface cache, or the DDGI volume as the last level (id Tech 8 style). The interface is designed in phase 3 with the Brixelizer port's needs in view, so plugging it in later doesn't change the screen cache.

### The world-cache interface (lead, 2026-11-09, phase 3)

The seam sits where Capsaicin's screen cache hands its probe rays to the hash grid: after the probe rays are traced, before they are blended. Everything between is the world cache's business, so a Brixelizer brick cache, a surface cache or the DDGI field can replace the hash grid without the screen cache changing.

- **Input:** the frame's traced probe rays in one GPU buffer: per ray its origin, direction, hit record (instance, primitive, barycentrics, distance, front face; a miss is marked) and a "bypass" flag, plus the ray count in a GPU counter (RC1.3's indirect arguments size the dispatches). Rays that took Capsaicin's bypass path (emissive hits, rays shorter than a cell) are already lit and carry the flag.
- **Output:** for every ray without the bypass flag, the world cache **adds** the radiance leaving the hit toward the ray's origin into that ray's radiance slot (the buffer `BlendScreenProbes` reads). Misses are the screen cache's (the sky).
- **TypeScript:** a `WorldCache` object the technique owns: `prepare(context)` (allocate; may throw `GILimitError` naming the limit), `encodeFrame(encoder, frame, rays)` (all of its per-frame work: purge, insert, populate, update, resolve), `memoryBytes`, `stats`, `dispose()`. The hash grid is the default and only implementation in phase 3.
- **WGSL for other readers:** an optional `worldCacheLookup(position, normal, direction) -> vec4f` (radiance and a validity weight) for later consumers that look up without inserting (Capsaicin's reflections read `HashGridCache_FindCell`). A cache that can't answer returns weight 0.
- **Why here:** Capsaicin's per-hit population, multi-bounce and resolve stay intact inside the hash grid, so the faithful path is unchanged. A brick or surface cache has its own population (from its own tracing or SDF), and needs only the hits and a place to write radiance.

## World-cache memory

Capsaicin's defaults allocate about 1 GiB for the hash cache (2^14 buckets × 16 tiles × 85 cells, two value arrays and two per-cell scratch arrays; `source-map.md` §4.2). We allocate the same, under the adapter-maximum limits the platform already asks for; never a cap from WebGPU's default 128 MiB binding (DECISIONS 2026-11-09). A device that can't fit it throws `GILimitError` naming the limit. Capacity is a setting chosen by measured bucket overflow and quality. Sizing the update scratch per updated tile instead of per cell is a lever (it frees several hundred MB). The technique reports its GPU memory in stats.

## Where the code goes

- **Branch:** `radiance-cache`, cut from `m2`. Platform changes are made as separate commits (one per phase-1 row, each on its own branch `rc1-<row>` merged into `radiance-cache`) so each can merge to the main line on its own and M3/M4 agents can use them.
- **Technique:** `packages/ddgi-core/src/techniques/radiance-cache/`, importing only `platform/contract` (RC1.2 enforces it). Import path `@pmndrs/ddgi/radiance-cache` until the package-split decision (DECISIONS: revisit when a second technique ships).
- **Adapter:** screen-input wiring and a composite node in `packages/ddgi-three`, following the additive composite (`albedo/π × E` before tone mapping).
- **Denoisers and upscaling:** contributed to `pmndrs/denoiser` and `pmndrs/upscaler` and used through their public APIs.

## Phases

One phase active at a time, each with checkable done-whens. Implementers and reviewers run on Opus; Sonnet does test runs and measurement. GPU timing under the bench lock, never while a native Capsaicin run is going. The RTX 3060 is the target; the Intel UHD is for correctness only.

### Phase 1: platform groundwork (RC1.1–RC1.5)

The rows and their done-whens are on the board (TASKS.md, "radiance-cache track"). In short: screen inputs from three's MRT pass with albedo; the lint rule; scan, compaction and indirect arguments; feature checks on both GPUs; the ray cost table.

**Feature fallbacks, declared up front.** RC1.4 checked each by use and readback (`tools/gpu/pages/platform-features`) on the RTX 3060 and the Intel UHD in Chrome 157.0.8103.38, on our device and on the device three r187 makes for itself: every feature works on both GPUs and both devices, so no fallback is needed there. The gaps are the limits row (D14) and f16 overflow, which differs between the GPUs (D15).

| Feature | Used for | Fallback | RC1.4 result |
| --- | --- | --- | --- |
| Indirect compute dispatch | Work sized on the GPU (spawned probes, cells to update) | Core WebGPU; no fallback planned. If it were missing: a fixed-size dispatch with an early return on the GPU count | Works, y > 1 included. A dimension over `maxComputeWorkgroupsPerDimension` (65,535 on both) runs nothing, silently, so GPU-written arguments must split (RC1.3) |
| `texture-formats-tier2` | Read-write `rgba16float` storage textures (D5) | Ping-pong: a sampled read plus a write-only storage texture | Works. Without it, a read-write `rgba16float` binding is refused with a validation error at `createBindGroupLayout` (captured, no crash), and the ping-pong fallback was run and works |
| `subgroups` | Scans, reductions, compaction (subgroups-first rule, DECISIONS 2026-11-04) | Workgroup-memory versions, tested | Works. Sizes: RTX 32; Intel reports 8–16 and ran 16 |
| `shader-f16` | Half-precision math in denoise and filter kernels (id Tech 8 lever 4) | f32 math with `pack2x16float` storage (D11) | Works: `array<f16>` storage and f16 rounding on both GPUs. Overflow differs and is allowed to (D15): `300h * 300h / 1000h` is infinity on the RTX, 90 on the Intel; the page reports it, not checks it. The fallback stands for any value that can pass 65,504 |
| `timestamp-query` | Per-pass times (D10) | No per-pass times; the budget uses `units` (as DDGI) | Works, ordered and non-zero |
| Adapter-maximum storage limits | The hash cache (about 1 GiB) and the packed buffers (D6) | None at WebGPU's default limits: `GILimitError` naming the limit | Our device has them (2 GiB bindings, 16 storage buffers per stage, both GPUs); three's own device doesn't (D14) |

### Phase 2: screen probes, first image

Spawn (16x16 tiles, Halton jitter), trace 64 rays per probe, light hits directly (Capsaicin's cache-bypass path), blend, project to SH, interpolate, composite through three. Done when: a single-bounce image on Cornell, two-room and Sponza; probe atlas, mask and SH in the viewer's debug views; a constant-sky Lambert check (E = πL within 1%) passes on a GPU test.

**What RC2.1 built, and the staging until phases 3 and 4** (not divergences: each is Capsaicin's own path for a frame with no history and no world cache, and the phase that brings the rest is named).

- *Passes,* in `encodeScreen`, each a WGSL kernel named after Capsaicin's entry point: `ClearCounters`; `ClearProbeMask` every frame (Capsaicin runs it at allocation; here it leaves every tile as `ReprojectScreenProbes` leaves a tile with no previous probe, `gi1.comp:457`, until phase 4's reprojection); `SpawnScreenProbes`; the platform's stream compaction (Capsaicin's `gfxCommandScanSum`) and `CompactScreenProbes`, which runs a thread per kept tile through RC1.3's indirect arguments and writes the same list in the same order; `SampleScreenProbes`; `PopulateScreenProbesMain`; the platform's `gi/direct`; `BlendScreenProbes`. Workgroup sizes are Capsaicin's. The per-ray passes are dispatched over every possible ray with an early return past the probe count, as Capsaicin dispatches them (`gi1.cpp:2323-2542`): they are not sized on the GPU there.
- *Sampling with no history:* with no previous probe, Capsaicin's 3 × 3 taps and cached tiles find nothing, every cell's previous radiance is (0, 0, 0, −1), the luminance CDF is zero and ray guiding samples each thread's own cell (`gi1.comp:972`): 64 rays, one uniform point in each hemi-octahedral cell. Phase 4 adds the taps, the CDF and the history write. The specular branch is D16.
- *Lighting every hit on the cache-bypass path:* until the world cache, every probe-ray hit is lit as Capsaicin lights a hit that bypasses it. A miss reads the sky (`PopulateScreenProbesHandleMiss`; D20). An emissive front face gives its emission and nothing more (`GenerateReservoirs`, `gi1.comp:2439-2452`). Any other hit gives the light it reflects toward the probe, `albedo / π × E_direct` from one light sample and its shadow ray (`GenerateReservoirs` then `PopulateCells`, `gi1.comp:2500-2598, 2040-2106`, which adds it to the probe ray when the bypass flag is set); here the platform's `gi/direct` traces that pass over the hits the populate kernel lists, and adds into the ray's result (light selection as D4). Phase 3 keeps this path for emissive hits and rays shorter than a cell and sends the rest to the cache.
- *Blending without history:* `BlendScreenProbes` as Capsaicin, without the biased hysteresis against the previous probe and the off-screen probe cache (phase 4). A cell no ray reached takes Capsaicin's backup, the mean of the reached cells divided by the number of empty cells (`gi1.comp:1226-1233`). With no history the only empty cells are those whose ray half precision moved across an edge (the direction is stored as three halves, `packHalf3`, as in Capsaicin), so the next cell has two rays.
- *The probe rays for the world cache:* one record per ray (origin, flags, direction as three halves, the hit record in `GiHitRecord`'s order; 48 bytes) and the ray count on the GPU, written by `SampleScreenProbes` (direction) and `PopulateScreenProbes` (origin, hit record and flags), so phase 3's world cache gets what "The world-cache interface" asks for. The radiance slot is the platform ray scratch's result for the same ray. Every phase 2 ray carries the bypass flag.
- *Debug views:* the probe atlas and the probe mask, drawn by an engine-free kernel into an image the host shows (`RadianceCacheTechnique.encodeDebugView`). The contract's debug views go through the shading module's `giDebugColor`, which has no screen position; so the viewer's entry, with the three composite, is RC2.2's.

### Phase 3: world cache, faithful

The hash grid exactly as Capsaicin: descriptor and hashes in one shared WGSL function, CAS insert into 16-slot buckets, 4-level tile MIPs, decay and purge, filtered lookup, per-hit population with one sampled light and shadow ray, then GI-1.2's multi-bounce cells. The world-cache interface is designed here. Done when: Capsaicin's default capacity runs under adapter-maximum limits with memory in stats; bucket overflow is measured; multi-bounce visibly fills two-room's closed room compared to phase 2; GPU tests cover insert, find, collision, overflow and decay.

### Phase 4: temporal quality, faithful

Reprojection with the plane and normal tests, adaptive patching, luminance-CDF ray guiding, the off-screen LRU probe cache, the probe filter, SSGI near-field and bent-normal occlusion, and the image-space GI denoiser (built as a `pmndrs/denoiser` contribution). Done when: Dennis judges the image steady by eye in the viewer on Cornell and Sponza, and a camera fly-through shows no probe popping.

### Phase 5: baseline comparison and attention list

Same scene and camera: Capsaicin GI-1.2, Capsaicin's path tracer, our baseline, our DDGI, the Cycles HD references. A per-pass table, ours against Capsaicin's, GPU loaded; each pass over native gets a cause (tracer, missing feature, our code) and an attention-list row. Accuracy against the path tracer per scene. Done when the table, the attention list and the image comparison are in, and Dennis sets the frame-time target from them.

### Phase 6: levers and dynamic scenes

Levers one at a time against the baseline, in the order the attention list says matters. In parallel, dynamic behaviour: on `DirtyRegion.moved`, invalidate or fast-decay hash tiles overlapping the old and new bounds; lights and sun re-populate affected cells first. Done when the M2/M3 scenarios (dragged cabinet, sun sweep, swaying tree) settle within SPEC's targets.

## Levers to try after the baseline

Each is a switch beside the faithful path, measured against it, and adopted with a reason in the register. id Tech 8 (Sousa, SIGGRAPH 2025) is the main source; details in the research folder's `idtech8-performance.md`.

1. **Shade cells, not hits:** probe rays store a hit record and look up a cell; lighting runs on a fixed budget of cells per frame by priority. Targets the largest native group (4.75–7.81 ms).
2. **DDGI supplies the second bounce** instead of GI-1.2's multi-bounce rays.
3. **Half- and quarter-resolution tiers** for probes and the gather, or `@pmndrs/upscaler`.
4. **Denoise in workgroup memory with f16** at gather resolution.
5. **Compact formats:** shared-exponent history, 16-byte hit records, SH in three `rgba16float`.
6. **Interleaved refresh of distant cache regions;** benchmark over a full update cycle.
7. **Short probe rays that end in the world cache** (Lumen). Our tracer gains little from short rays (NOTES), so measure first.
8. **Scratch per updated tile** instead of per cell (memory).
9. **Brixelizer's SDF tracing** for long rays.
10. **Priority versions of passes,** for example key-probe-style ordering of screen-probe updates.

## Extensions after the baseline (ranked)

1. DDGI as the third cache level (screen cache, world cache, then the DDGI volume), through the `BounceSource` slot.
2. Glossy reflections (GI-1.1), once materials carry roughness and metalness (D8).
3. Brixelizer and surface-cache world caches through the slot.
4. Object-attached cache entries for moving objects (EA surfels; Webgiya, MIT, is a WebGPU precedent).
5. Lumen-style improvements: product importance sampling, a GPU priority queue for cache updates.
6. Split Radiance Cascades (arXiv 2607.20384) as a world-cache experiment.

## Siblings: what goes back

- **`@pmndrs/upscaler`:** used as-is. A GI-history path for its issue #7 if phase 5 runs into it.
- **`@pmndrs/denoiser`:** GI-1.0's image-space denoiser (phase 4), hit distance and cache confidence as guides, the "coverage" input mode (its issue #23), a bilateral upsample node for half-resolution GI. Probe-space filtering and hysteresis stay in the technique.
- **Textures:** GPUTex where real hit textures (D2) or big scenes need the memory.

## Risks

| Risk | Mitigation |
| --- | --- |
| The faithful baseline is far slower than native in tracing | Expected and measured: RC1.5 predicts it, phase 5 lists it, BVH work and levers follow |
| Getting three's buffers into compute | RC1.1 first; the smallest pinned step if three needs one, in the register |
| Flicker the numbers miss | Dennis's look by eye is a phase 4 done-when |
| Temporal accumulation fighting the upscaler's jitter | Placement decided in phase 5, with upscaler issue #7 |
| World cells stale after objects move | Dirty-region invalidation in phase 6; object-attached entries later |
| Clashing with M2/M3 work | A separate branch; platform changes as separate, small merges |
| The denoiser branch is unmerged and Apple-only so far | Our contribution includes an RTX 3060 run |
