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
2. **Every difference from Capsaicin goes in [radiance-cache-divergences.md](radiance-cache-divergences.md):** what we do instead, its kind, why, and the way back, including workarounds others use (browser vendors, other WebGPU projects, engines), with its **cause** (platform can't, our rule, quick setup, ...), its **evidence** (measured, cited, assumed), its impact, the effort to match and a recommendation, so that later work can challenge the differences that weren't forced.
3. **Performance ideas are levers, not rewrites.** Each lever is a switch beside the faithful path, measured against the baseline on the same scene and camera, and adopted only with a written reason. A slow faithful path is kept and marked for attention.
4. **Priority versions are extra options.** A cheaper or prioritised variant of a pass can be built behind a switch. It doesn't replace the baseline.

## What we port, and from where

| Source | License | Use |
| --- | --- | --- |
| Capsaicin v1.3 (`914b915`), `src/core/src/render_techniques/gi1/` | MIT (AMD) | The baseline. Port pass by pass; ported files keep AMD's notice. About 5,000 useful HLSL lines. |
| Capsaicin's SSGI technique (near-field GI, bent-normal occlusion) | MIT (AMD) | Part of the baseline: GI-1 reads its output. About 350 lines. |
| Capsaicin's blue-noise tables (`blue_noise_sampler_samples.h`) | No license header | Not copied. Regenerated (D7). |
| GI-1.0 paper (arXiv 2310.19855), GI-1.1 paper, GI-1.2 page | Papers | Algorithms and reasons. |
| FidelityFX Brixelizer GI | MIT (headers to verify) | Ported in another workstream as a distance-field library (no GI, DEC-010); plugs in as a tracing backend ("World cache as a slot"). |
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

## Divergences from Capsaicin

Every way our port differs from Capsaicin's default behaviour is in [radiance-cache-divergences.md](radiance-cache-divergences.md): what we do instead, why (forced by the platform, our own rule, a quick setup, a fix), the evidence, the impact on the image and speed, the effort to match Capsaicin and a recommendation. It merges what used to be this doc's divergence register and challenge list.

Every new divergence gets a row there (numbered from the next free ID) in the same commit as the code.

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

The baseline world cache is Capsaicin's hash grid. Behind one interface ("radiance leaving this hit point toward this ray", plus "these cells need updating"), others can plug in: a Lumen-style surface cache (cards), or the DDGI volume as the last level (id Tech 8 style). *Corrected 2026-11-11:* the separate Brixelizer port is a distance-field library, not a GI cache (its DEC-010 excludes GI), so it comes in as a tracing backend, not a world cache (see "Brixelizer: what it is and where it fits"). The interface is designed in phase 3 with the Brixelizer port's needs in view, so plugging it in later doesn't change the screen cache.

### The world-cache interface (lead, 2026-11-09, phase 3)

The seam sits where Capsaicin's screen cache hands its probe rays to the hash grid: after the probe rays are traced, before they are blended. Everything between is the world cache's business, so a surface cache or the DDGI field can replace the hash grid without the screen cache changing.

- **Input:** the frame's traced probe rays in one GPU buffer: per ray its origin, direction, hit record (instance, primitive, barycentrics, distance, front face; a miss is marked) and a "bypass" flag, plus the ray count in a GPU counter (RC1.3's indirect arguments size the dispatches). The flag marks a ray the screen cache lit itself, which the world cache leaves alone. *Amended (RC3.2):* Capsaicin decides its bypass cases (emissive hits, rays shorter than their cell) inside the cache's own passes (`GenerateReservoirs`, `PopulateCells`), and the short-ray test needs the cache's cell size, so the hash grid decides and lights them itself; the screen cache sets the flag on no ray.
- **Output:** for every ray without the bypass flag, the world cache **adds** the radiance leaving the hit toward the ray's origin into that ray's radiance slot (the buffer `BlendScreenProbes` reads). Misses are the screen cache's (the sky).
- **TypeScript:** a `WorldCache` object the technique owns: `prepare(context)` (allocate; may throw `GILimitError` naming the limit), `encodeFrame(encoder, frame, rays)` (all of its per-frame work: purge, insert, populate, update, resolve), `memoryBytes`, `stats`, `dispose()`. The hash grid is the default and only implementation in phase 3.
- **WGSL for other readers:** an optional `worldCacheLookup(position, normal, direction) -> vec4f` (radiance and a validity weight) for later consumers that look up without inserting (Capsaicin's reflections read `HashGridCache_FindCell`). A cache that can't answer returns weight 0.
- **Built (RC3.1):** `techniques/radiance-cache/world-cache/world-cache.ts` holds the types and the probe-ray layout (`WorldCacheProbeRays`, 48 bytes a ray after a 16-byte header with the count, RC2.1's `RcProbeRays` byte for byte; radiance slots a separate `array<vec2u>` of `packHalf4`); `hash-grid-cache.ts` is the hash grid behind it. `encodeFrame` takes `{ frame, camera, width, height }` (cells are sized by pixels covered). In RC3.1 it runs the purge and `ClearCounters` only.
- **Built (RC3.2):** the technique owns a `HashGridCache` (made in `prepare` at its settings, `GILimitError` on a device that can't hold it; its memory and figures in the technique's `memoryBytes` and counters; `resetHistory` empties it at the next frame) and calls `encodeFrame` between `PopulateScreenProbes` and `BlendScreenProbes`. `encodeFrame` runs Capsaicin's sections in its order: `PurgeRadianceCache`, then the cache half of `PopulateScreenProbes`, `GenerateReservoirs`, `PopulateCells`, `GenerateUpdateTilesDispatch` with `UpdateTiles`, and `ResolveCells` (`world-cache/hash-grid-population-wgsl.ts`), each sized on the GPU through RC1.3's indirect arguments. It is GI-1.2's multibounce path (Capsaicin's default): the resolve list holds every non-emissive hit, and the indirect values (0 until RC3.3) are carried. The world cache's context adds `shade` (it lights hits through platform hit shading) and `timer`. Every pass binds at most the platform's 8 storage buffers per stage (D6). Register D30-D33 and D35-D37 (D34 is RC2.1's M5 fix).
- **Built (RC3.3):** GI-1.2's multibounce, in Capsaicin's order: `PopulateMultibounceCells` after the insert, `GenerateMultibounceReservoirs` after `GenerateReservoirs`, `UpdateMultibounceCells` after `UpdateTiles` (its samples reach the indirect values in the next frame's `UpdateTiles`, Capsaicin's one-frame lag). `useMultibounce` (Capsaicin's `gi1_use_multibounce`, default on) sizes the cache at `prepare`, as Capsaicin's define does (RC6.2, D42): off, the indirect arrays (535 MB) and the second-bounce kernels aren't made, and turning it on takes effect at the next `prepare`; turning it off also works between prepares. Off is Capsaicin's single-bounce path, and changing it empties the cache. The second-bounce entries share the visibility lists with the first bounce's (ten words each, D30); their brdf and pdf are a list of their own. `PopulateMultibounceCells` binds exactly the platform's 8 storage buffers (hit shading's 5, the cache words, the cell values, the lists), `GenerateMultibounceReservoirs` 7, `UpdateMultibounceCells` 4. Register D39-D42.
- **Built (RC6.1):** the sky is one of the lights the cache's hits draw, as Capsaicin's environment light is (register D33): `GenerateReservoirs` and `GenerateMultibounceReservoirs` call the platform's `giDirectChooseLightWithSky` (the sky one candidate ahead of the lights, a direction from `giSkyLightSample` over the baked sky), and `PopulateCells` traces the sky's unbounded shadow ray through `giDirectLightRay`. The light builder's `environment_light_enable` is the technique's `environmentLightEnable` (default true), a word of the hash grid's constants. DDGI's `gi/direct` is unchanged.
- **Why here:** Capsaicin's per-hit population, multi-bounce and resolve stay intact inside the hash grid, so the faithful path is unchanged. A brick or surface cache has its own population (from its own tracing or SDF), and needs only the hits and a place to write radiance.

### Budgeted cell shading, id Tech 8's way (lever 1; lead, 2026-11-11)

Dennis chose this on 2026-11-11 (DECISIONS). It's id Tech 8's design (Sousa, SIGGRAPH 2025, S15-S20) on top of the hash grid we already have. Probe rays only find their cell, and a separate pass lights a fixed number of cells a frame, most urgent first. A cell's light is reused until it's lit again. It lives inside the hash-grid world cache (`encodeFrame`), so the world-cache interface and the screen cache don't change.

**Settings** (all on the hash grid's constants; `perHit` is Capsaicin's and the default):
- `cellShading: 'perHit' | 'budget'`, read at prepare, because `budget` adds buffers and kernels.
- `cellShadingBudget`: cells lit per frame (id: about 20,000 for a whole level; ours to tune by measurement).
- `cellShadingSamples`: light draws (shadow rays) per lit cell per frame, default 1, so a frame's shadow rays are at most budget × samples.
- `cellShadingMaxAge`: frames after which a cell still in use counts as stale and goes ahead of fresher cells.

**The frame in `budget` mode:**
1. **Insert, unchanged.** Probe-ray hits find or insert their cell. The bypass cases (emissive front faces, rays shorter than their cell) are still lit per hit, as Capsaicin does; they're few.
2. **Pick one hit per cell.** Each hit in a cell this frame offers its visibility entry (it already carries the hit record, D30). The cell keeps the hit with the smallest hash of (ray, frame), an `atomicMin` on a per-cell word, so its point is a uniformly random hit each frame. Over frames the cell's light is the average over the area its hits cover, as in `perHit`.
3. **Choose the cells to light.** Cells used this frame are ranked: new cells (no samples yet) first, then by frames since last lit, stale ones (`cellShadingMaxAge` or more) ahead of the rest. The first `cellShadingBudget` are taken, through a histogram of age classes and the platform's scan, with no sort. Unused cells aren't lit; nothing reads them.
4. **Light the chosen cells.** For each one, rebuild its point from the kept hit record through `giSurfaceAt`, which also supplies the material (the average today; real textures with lever 27). Then run `GenerateReservoirs`' light choice (the sky included, RC6.1) and `PopulateCells`' shadow ray, `cellShadingSamples` times. The result goes into the cell through `UpdateTiles`' temporal accumulation and sample cap, as now.
5. **Second bounce.** With `useMultibounce`, the second-bounce ray also leaves from the chosen cells' points (one per lit cell) instead of from every hit, so it's on the same budget. Lever 2, the bounce from DDGI, can replace it later.
6. **Resolve, unchanged.** Probe rays read their cell's filtered light (`ResolveCells`), so they read cells not lit this frame at their last value. That's the lag.

**Memory:** one 32-bit word per cell for the kept hit, plus one for its last-lit frame, sized from the cache's capacity at prepare. It's stated in `memoryBytes`, and zero in `perHit` mode.

**What it costs the image:** lighting can lag by up to `cellShadingMaxAge` frames, and with a small budget more. A cell sees fewer samples per frame than in `perHit`, so it converges more slowly. Both are measured, not assumed.

**Why it's switchable:** the per-hit kernels stay as they are. `budget` adds four kernels beside them (pick, histogram, choose, light), and Capsaicin's population is skipped in that mode. Every RC page runs in `perHit`, and the new checks run in both modes, so neither path rots.

**Its partners:** lever 22 (a frame budget) turns `cellShadingBudget` as a dial. Lever 27 (bindless) gets cheaper here, because textures are read once per lit cell instead of once per hit.

### Card cache experiment (extension 3; lead, 2026-11-11): design study

Dennis wants Lumen's surface cache tried as an experiment. It's a second `WorldCache` behind the slot, so it's a switch by design (`worldCache: 'hashGrid' | 'cards'`). Facts below are from Wright, Narkowicz and Kelly, SIGGRAPH 2022 Advances ([deck](https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-Lumen-Wright%20et%20al.pdf); page numbers as p.N), and the current UE docs ([Lumen Technical Details](https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-technical-details-in-unreal-engine)). The deck is 2022, so current UE may differ.

**What Lumen does.**
- **Cards** are axis-aligned projections of each mesh, chosen when the mesh is imported: surfels are clustered, and the clusters with the most coverage become cards. UE allows up to 12 per mesh by default, and the fallback is six box faces (p.60). Generation takes about 0.2 ms for a 1.5M-triangle mesh.
- **The atlas** is 4096² in 128² pages, filled by a virtual cache: low-resolution pages around the camera always resident, high-resolution pages on demand from GPU feedback (p.65-67).
- **Channels,** BC-compressed at runtime: albedo, opacity, depth, normal and emissive, about 88 MB in all (p.69).
- **Capture** rasterises each mesh with an orthographic camera into its cards, 512² texels a frame by priority (p.68-69). Cards are fixed at import, so deforming meshes and characters aren't covered.
- **Lighting** is done in the atlas on two budgets (p.73-77):
  - Direct light: 1024² texels a frame, the first 8 lights of each 8 × 8 tile, with shadow maps plus traced shadow rays.
  - Indirect light: 512² texels a frame, through one 4 × 4 hemisphere probe per 4 × 4 texels.
  - Priority: last used minus last updated, chosen through a histogram, which is our lever 1's selection.
- **Lookup** (p.70): from the mesh's card grid, take the 3 cards facing the hit best; weight 4 depth taps per card by the difference from the hit's depth, by the normals and bilinearly; blend. A hit no card covers loses its energy rather than leaking.
- **Cost:** 0.47-0.93 ms for the surface cache on PS5 at 1080p ("Lumen in the Land of Nanite"), 1.68 ms in the Matrix Awakens demo.
- **Hardware RT:** UE's current docs say rays read the surface cache by default there too, and "Hit Lighting" is an expensive opt-in.
- **Known limits:** coverage holes; foliage with many layers; walls under 10 cm leak; no single-mesh rooms.

**How it would map onto our platform.**
- **Generation:** at load, in the scene copy's worker. Start with six box cards per mesh, Lumen's own fallback, and add surfel clustering only if coverage needs it. Big single meshes, like Sponza's or Bistro's buildings, are the "single-mesh room" case Lumen can't do. They would need splitting into pieces first. RC5.4 already splits meshes for its bench, so that's reusable.
- **Capture:** a raster pass in the core over the scene copy, with each material's real textures bound per draw. Rasterising needs no bindless, so this is how cards close D2 without it. The platform keeps the textures it averages today, and GPUTex can compress them if memory needs it.
- **Lighting:** the same budget and histogram as lever 1.
  - Direct light: through the platform's light choice and our BVH shadow rays (no shadow maps in the core).
  - Indirect light: from the hash grid or DDGI, not new probe rays.
- **Lookup:** from a hit's instance and triangle to its mesh's cards, projection, then Lumen's blend. A miss or an uncovered hit falls back to the hash grid, so moving and deforming meshes keep working.
- **Memory:** uncompressed 2048² channels are about 50 MB, plus about 32 MB of lighting (`rgba16float`). At 4096² it's four times that. Uncompressed BC isn't writable in WebGPU, so compression would be GPUTex's job.
- **Brixelizer:** checked 2026-11-11 (below). It gives no material or triangle at a hit, so if its distance-field tracing is used for far rays, cards are one way to give those hits a material, which is Lumen's reason for cards.

**Effort and gain against lever 1.**
- **Effort: XL** (weeks): card generation and splitting, a core raster pass with textures, a page allocator, two lighting passes, the lookup and its fallback. Lever 1 is L and reuses Capsaicin's passes.
- **Gain:**
  - **Image:** real albedo and emission at hits (D2), the same thing lever 27 gives with bindless.
  - **Speed:** cards don't make ray tracing cheaper, and tracing is our cost (RC5.1: 12.5 of Sponza's 16.6 ms on the M5). Lumen is fast because cards come with distance-field tracing. For us, that's cards plus Brixelizer.
  - **Lighting:** it's budgeted, as in lever 1, so lever 1 already gets most of that part.

**The lead's recommendation.** Build lever 1 (RC-L1) and bindless (RC6.3) first, then measure D2's image gain on Sponza and Bistro with bindless on Windows. Build cards only if bindless can't cover the platforms we care about, macOS above all, or if Brixelizer's distance-field tracing arrives and needs a material source for its hits. That's Lumen's case: distance-field hits don't know their triangle, so they need cards. A smaller first step: six box cards on the Cornell box and two-room, which have no single-mesh-room problem, to measure the lookup and the image before taking on the rest.

### Brixelizer: what it is and where it fits (lead, 2026-11-11)

Dennis's port is at `/Users/dex/Documents/Codex/2026-11-09/wha/outputs/brixelizer-webgpu/` (work in progress; read-only review, HEAD d24a181).
- **What it is:** an engine-free distance-field library. It builds 8³ R8 bricks from triangles, with an octree, scrolling cascades, incremental updates and a ray march. It borrows the host's `GPUDevice` and needs no optional features. Its three adapter pins r186, against our r187.
- **What it isn't:** a GI system. Its own decision DEC-010 excludes GI, so there's no radiance cache to plug into our slot.
- **What a hit gives:** position, distance, a normal from the field's gradient, and a brick id (`FieldRayHit`, 64 bytes). There's no triangle, instance or material, by design (its CAP-Q14).
- **Licence:** MIT (Dennis), with AMD's MIT headers kept for reference and Inigo Quilez's triangle-distance function noted in its THIRD_PARTY_NOTICES. Any code we take carries those notices.

**Where it fits for us: a tracing backend for long rays (lever 9). There are three uses, cheapest first:**
1. **Occlusion only:** cache shadow rays and long any-hit rays through the field. No material is needed. Speed is the question.
2. **Far field read from the hash grid.** A probe ray past a few metres continues in the field, and its hit's position and normal look up an existing hash-grid cell (`worldCacheLookup(position, normal, direction)`, already in the interface). No triangle or material is needed. A hit with no lit cell yet falls back to the BVH, or counts as empty.
3. **Far field with material:** only with cards, which is Lumen's arrangement (RC-X3).

**Measured by the port (2026-11-11, its own bench: one Apple GPU, a 5,840-triangle procedural room, single runs; its docs/COMPARISON_PLAN.md):**
- **Speed:** short AO rays (499k, ≤ 0.5 m, any-hit) run at about 135 Mrays/s through the coarse field and 77 through the fine one, against about 2.4 Grays/s for pmndrs/bvh. So the field is 10–30× slower on a small scene.
- **Agreement:** it matches the BVH on 97.5–99% of rays, with no false misses.
- **Memory:** 37–103 MiB, against 0.44 MiB for the BVH.
- **Not measured:** large or dynamic scenes, or long rays, where a field could win.
- **Expected fix:** its hierarchy walk restarts from the root at every step (its DIV-012).
- **So for now:** the BVH stays the long-ray default. Lever 9 waits for DIV-012 and numbers on Sponza and Bistro.
- **Its query:** `traceFieldRay` / `traceClipmapRay(origin, dir, tMin, tMax, maxSteps, hitEpsilon)`, composed into our shader. It binds 3 storage buffers, a uniform, a 3D texture and a sampler at group 0, so it needs a group-index option before we can compose it. StepLimit, Pending and InvalidField mean "unknown", never "miss". A start within the hit tolerance of a surface hits immediately, so the field suits continuing a ray far from surfaces, not starting one on them.

**What's missing on our side:** the platform has one tracer. `createTraceModule` (`platform/trace/wgsl.ts:282-300`) wires the BVH walk in directly, so a second backend needs a seam first. It's a platform option, `longRays: 'bvh' | 'field'`, with the BVH as the default and DDGI unchanged.

## World-cache memory

Capsaicin's defaults allocate about 1 GiB for the hash cache (2^14 buckets × 16 tiles × 85 cells, two value arrays and two per-cell scratch arrays; `source-map.md` §4.2). We allocate the same (since RC3.2's review in two buffers: the cache words, 718,274,816 bytes, and the cell values, 356,515,840), under the adapter-maximum limits the platform already asks for; never a cap from WebGPU's default 128 MiB binding (DECISIONS 2026-11-09). A device that can't fit it throws `GILimitError` naming the limit. Capacity is a setting chosen by measured bucket overflow and quality. Sizing the update scratch per updated tile instead of per cell is a lever (it frees several hundred MB). The technique reports its GPU memory in stats. RC3.3 adds 2,764,800 bytes of per-ray lists at 1280 × 720 (the second-bounce entries' brdf and pdf, `max_ray_count` of them as in Capsaicin, and D40's bypass flags): the lists are 35,020,800 bytes. The second-bounce cells live in the same tiles: on Sponza the live tiles rose from about 18,500 to about 50,600 of 262,144 (Bistro exterior 23,300 to 58,500), still with no refused insert over 300 frames (`radiance-cache-overflow`, M5).

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

- *Passes,* in `encodeScreen`, each a WGSL kernel named after Capsaicin's entry point: `ClearCounters`; `ClearProbeMask` every frame (Capsaicin runs it at allocation; here it left every tile as `ReprojectScreenProbes` leaves a tile with no previous probe, `gi1.comp:457`; **retired by RC4.1**: it now runs when the masks are made and on `resetHistory`, as Capsaicin's, and `ReprojectScreenProbes` writes the mask every frame); `SpawnScreenProbes`; the platform's stream compaction (Capsaicin's `gfxCommandScanSum`) and `CompactScreenProbes`, which runs a thread per kept tile through RC1.3's indirect arguments and writes the same list in the same order; `SampleScreenProbes`; `PopulateScreenProbesMain`; the platform's `gi/direct`; `BlendScreenProbes`. Since the RC2.2 and RC3.2 merge the order is Capsaicin's (`gi1.cpp:2296-2636`): spawn, compact, sample, trace, the world cache's passes in `gi/direct`'s place, blend, then RC2.2's `FilterProbeMask`, `ProjectScreenProbes` and `InterpolateScreenProbes`. Workgroup sizes are Capsaicin's. The per-ray passes are dispatched over every possible ray with an early return past the probe count, as Capsaicin dispatches them (`gi1.cpp:2323-2542`): they are not sized on the GPU there.
- *Sampling with no history* (**retired by RC4.1**, which ports the taps, the probe cache, the CDF and the history write): with no previous probe, Capsaicin's 3 × 3 taps and cached tiles find nothing, every cell's previous radiance is (0, 0, 0, −1), the luminance CDF is zero and ray guiding samples each thread's own cell (`gi1.comp:972`): 64 rays, one uniform point in each hemi-octahedral cell. Phase 4 adds the taps, the CDF and the history write. The specular branch is D16.
- *Lighting every hit on the cache-bypass path* (**retired by RC3.2**: the world cache now lights the hits, and only Capsaicin's bypass cases, emissive front faces and rays shorter than their cell, skip its filtered values, lit inside its own passes; see "The world-cache interface"): until the world cache, every probe-ray hit is lit as Capsaicin lights a hit that bypasses it. A miss reads the sky (`PopulateScreenProbesHandleMiss`; D20). An emissive front face gives its emission and nothing more (`GenerateReservoirs`, `gi1.comp:2439-2452`). Any other hit gives the light it reflects toward the probe, `albedo / π × E_direct` from one light sample and its shadow ray (`GenerateReservoirs` then `PopulateCells`, `gi1.comp:2500-2598, 2040-2106`, which adds it to the probe ray when the bypass flag is set); here the platform's `gi/direct` traces that pass over the hits the populate kernel lists, and adds into the ray's result (light selection as D4). Phase 3 keeps this path for emissive hits and rays shorter than a cell and sends the rest to the cache.
- *Blending without history* (**retired by RC4.1**: the hysteresis and the probe cache's writes are in): `BlendScreenProbes` as Capsaicin, without the biased hysteresis against the previous probe and the off-screen probe cache (phase 4). A cell no ray reached takes Capsaicin's backup, the mean of the reached cells divided by the number of empty cells (`gi1.comp:1226-1233`). With no history the only empty cells are those whose ray half precision moved across an edge (the direction is stored as three halves, `packHalf3`, as in Capsaicin), so the next cell has two rays.
- *The probe rays for the world cache:* one record per ray (origin, flags, direction as three halves, the hit record in `GiHitRecord`'s order; 48 bytes) and the ray count on the GPU, written by `SampleScreenProbes` (direction) and `PopulateScreenProbes` (origin, hit record and flags), so phase 3's world cache gets what "The world-cache interface" asks for. The radiance slot is the platform ray scratch's result for the same ray. Every phase 2 ray carried the bypass flag; since RC3.2 none does.
- *Debug views:* the probe atlas and the probe mask, drawn by an engine-free kernel into an image the host shows (`RadianceCacheTechnique.encodeDebugView`). The contract's debug views go through the shading module's `giDebugColor`, which has no screen position; so the viewer's entry, with the three composite, is RC2.2's.

### Phase 3: world cache, faithful

The hash grid exactly as Capsaicin: descriptor and hashes in one shared WGSL function, CAS insert into 16-slot buckets, 4-level tile MIPs, decay and purge, filtered lookup, per-hit population with one sampled light and shadow ray, then GI-1.2's multi-bounce cells. The world-cache interface is designed here. Done when: Capsaicin's default capacity runs under adapter-maximum limits with memory in stats; bucket overflow is measured; multi-bounce visibly fills two-room's closed room compared to phase 2; GPU tests cover insert, find, collision, overflow and decay.

**Capsaicin's second bounce reads high, and the port keeps it (RC3.3, measured on the M5).** `UpdateMultibounceCells` divides each second-bounce sample by 1 - 0.7 to make up for the discarded rays (gi1.comp:1966), but the cell's indirect value is the mean of the samples that arrived (`UpdateTiles`, gi1.comp:2182-2199): discarded rays and rays that miss add no sample at all, so the discard is made up for twice. In a closed room, where nearly every ray hits, a cell's second bounce is 1 / 0.3 = 3.3 times the light it stands for; where rays miss, more. On the sunlit corner (`radiance-cache-multibounce`) every cell's indirect value reads albedo × L / 0.3 = 1.67 L as the estimator says (within 0.23%), where the physical second bounce is at most albedo × L / 2 = 0.25 L; a probe ray that bypasses the cache, whose own sample is added straight to it (D40) so the discarded and missed rays count as zero, reads 1.25 L on average, which is about right. Two-room's dark room is 3.9 times brighter with the multibounce than without. This is Capsaicin's behaviour, not a difference, so it has no register row; the lever is below.

### Phase 4: temporal quality, faithful

Reprojection with the plane and normal tests, adaptive patching, luminance-CDF ray guiding, the off-screen LRU probe cache, the probe filter, SSGI near-field and bent-normal occlusion, and the image-space GI denoiser (built as a `pmndrs/denoiser` contribution). Done when: Dennis judges the image steady by eye in the viewer on Cornell and Sponza, and a camera fly-through shows no probe popping.

**What RC4.1 built** (2026-11-10). The screen probes' history, in Capsaicin's order (`radiance-cache-technique.ts`, `wgsl/screen-probe-history.ts`, `wgsl/screen-probes.ts`, `probe-cache.ts`): `ReprojectScreenProbes`; the probe cache's `CountScreenProbes`, the platform scan and `ScatterScreenProbes`; the spawn, compaction and `CompactScreenProbes` with its override list; `PatchScreenProbes`; `SampleScreenProbes` with the 3 × 3 taps, the cached probes and the luminance CDF; `BlendScreenProbes` with the biased hysteresis and the cache's eviction and update; the scan of the LRU flags and `ReorderScreenProbes`. Register D45-D54. `resetHistory` forgets both caches; `resetScreenProbes` only the screen probes (a camera cut).

**What history does to the image, measured on the Apple M5 Pro** (`radiance-cache-temporal`, `radiance-cache-three`), **and why: Capsaicin's own rules, explained by a model.** With history the faithful port does not converge to its no-history image: under a constant sky E reads 0.6896 πL (no history 1.021 πL); Cornell's mean E is ×1.54 its no-history mean, Sponza ×0.89; the GI term over one-bounce DDGI goes from 1.013 / 0.940 / 0.822 to 1.564 / 0.818 / 0.820 (Cornell / two-room / Sponza). Four of Capsaicin's rules together cause it (RC4.1 review): (1) the **backup** for a cell no ray reached is all the rays' summed radiance over the reached cells R, over the empty cells, `64 / (R (64 − R))` under a sky of 1 (gi1.comp:1213-1232), 0.06-0.15 for typical R; (2) the **hysteresis** takes a darker value at once and holds the history only against a value more than twice it (gi1.comp:1266); (3) **CDF feedback**: a dark cell's history gives it almost no weight in the next frame's CDF, so it stays empty (gi1.comp:917-919, 944-947); (4) `SampleScreenProbes`' **3 × 3 taps** (gi1.comp:689-757) average a cell with its neighbours' same cell, and one probe in four is re-spawned each frame. A CPU model of these rules (the reviewer's) gives 0.690 πL against the GPU's 0.6896; without the taps it gives 0.52-0.54. Cornell's ×1.54 comes from the same backup rule: guided rays crowd into the light's cells, and the backup spreads their summed radiance over the empty cells (the model gives ×1.23 to ×1.59 with one or two bright cells, ×0.73 at low contrast, which fits Sponza). Capsaicin's default build also sends 5-30% of rays down the unguided specular branch (D16); the model still gives 0.69-0.71. So it is not a port bug. `FilterScreenProbes` (RC4.2) is not expected to undo it. **Still open:** native Capsaicin has not been run; phase 5 confirms it with a constant-sky furnace.

**What RC4.2 built** (2026-11-10). `FilterScreenProbes` (gi1.comp:1356-1443) in `wgsl/screen-irradiance.ts`, run as Capsaicin runs it (gi1.cpp:2588-2614): after the blend's copy of the atlas into the previous atlas, across the screen from the atlas into it, then down it from there back into the atlas, radius 3, with the plane, hemisphere and 3.6° reprojected-angle tests and the depth weight to the eighth power; the filtered atlas is what the shading module, the projection and the next frame read: it is copied into the previous atlas at the end of the frame, which is what Capsaicin's flip does (D45). Capsaicin's SSGI technique (ssgi.comp, ssgi.cpp) as a module of the radiance cache (`ssgi.ts`, `wgsl/ssgi.ts`), run first in the screen stage, its settings Capsaicin's defaults (1 slice, 2 steps, 0.2 m, falloff over the last 5%); `InterpolateScreenProbes` now runs Capsaicin's `HAS_OCCLUSION` branch: the SH about the bent normal through the cone the occlusion narrows, added to the near-field light. SSGI reads last frame's lit colour through a new optional `FrameScreen` channel, `previousLighting`, asked for like RC1.1's; the three adapter captures the picture `encodeBefore` is given (D56). SSGI's two debug views (occlusion, bent normal) are in both debug paths and the viewer's menu. Every new kernel fits WebGPU's defaults: `FilterScreenProbes` 1 storage buffer and 1 storage texture, `SSGI` 0 and 2, `InterpolateScreenProbes` 0 and 1 (`radiance-cache.test.ts`); neither new kernel reduces anything, so no subgroup path or fallback is needed. CPU twins of both (`probe-filter.ts`, `ssgi-cpu.ts`) are what the new pages check against. Register D55-D58.

**What the filter and SSGI do to the image, measured on the Apple M5 Pro** (`radiance-cache-three`, `-temporal`, `-image`, `-ssgi`). SSGI darkens: with no screen-probe history the GI term over one-bounce DDGI goes from 1.013 / 0.940 / 0.822 (RC4.1) to 0.852 / 0.819 / 0.635 (Cornell / two-room / Sponza), and the interpolated pixel over the path-traced one-bounce reference at probe seeds reads 0.845 / 0.869 / 0.798 while the probe rays still read 1.015 / 1.023 / 1.016. A floor under a constant sky reads 0.729 πL with no history (1.021 before SSGI) and 0.473 πL with history (0.69 before). Most of it is Capsaicin's default of one SSGI slice (`ssgi_slice_count_ = 1`, ssgi.h:47): one slice's occlusion and bent normal go into the cone SH, which is not linear in them, so even an open plane reads dark on average (the review's f64 twin, floor from 3 m at a 15° pitch: 0.735 πL with 1 slice, 0.969 with 16); rebuilding samples at their own uv costs only 2-4 points more (0.753 and 1.011 on texel centres; the list for AMD above). Its near-field light, at about E / π, adds little back. Flicker (T1.12's measure on E, still camera, information) rises with SSGI's one random slice per pixel per frame: floor 48.1% / 122% (mean / p95; RC4.1 3.0% / 7.7%), Cornell 43.7% / 191% (21.6% / 81.7%), Sponza 43.5% / 146% (12.6% / 43.1%). With SSGI's inputs switched off in a local build (not kept) the flicker is 0.39% / 1.09%, 11.5% / 48.9% and 9.1% / 36.7%: the filter lowers RC4.1's, and SSGI's single random slice is the rise. Capsaicin's GI denoiser (RC4.3) runs after this: with SSGI in, converged (frames 300+), Cornell reads 0.74% / 3.16% and Sponza 0.90% / 2.91% after it (0.36% / 1.32% and 0.27% before SSGI was merged); lever 16 is the first thing to try. The by-eye check is Dennis's. The filtered history also holds light longer: after the multibounce is switched off, two-room's dark room needs more than 30 frames to come back (120 frames: 0.0539 against 0.0540).
**What RC4.3 built** (2026-11-10). Capsaicin's image-space GI denoiser as a package of its own, `packages/denoiser-gi1` (`@pmndrs/denoiser-gi1`; README there): `ReprojectGI`, `FilterGI` horizontally and vertically, and `KeepGuides` (ours, the previous depth and normals), with Capsaicin's two history colours and colour deltas swapped every frame and the **filtered** result fed back as history (gi1.cpp:2882-2927, 3102). It imports nothing outside its folder (lint `local/self-contained`), takes plain textures and `@pmndrs/denoiser`'s camera and guide shapes, and has a CPU reference of its kernels. The technique runs it after `InterpolateScreenProbes` (step 8b, `gi-denoiser.ts`), on the interpolation's irradiance with Capsaicin's hint in alpha, and the shading module reads its output; `useGIDenoiser` (ours, default on) turns it off for comparisons. The blur mask and colour deltas are Capsaicin's `r8snorm` and `r16float` where the device has `texture-formats-tier1`, with one declared fallback (`rgba8snorm`, `r32float`) tested beside them. Register D65-D75.

**What the denoiser does to the image, measured on the Apple M5 Pro** (`radiance-cache-denoiser`, still camera, T1.12's measure beside the interpolated E of the same frames). On converged history (frames 300-331 after a reset): Cornell 21.6% / 81.6% (mean / p95) → 0.36% / 1.32%, Sponza 12.6% / 43.1% → 0.27% / 0.88%, about 60 and 45 times steadier, still above DDGI's 0.1% / 0.5%; the denoised mean is the input's within 0.1% (Cornell 0.9993, Sponza 0.9995). In RC4.1's window (frames 32-63) it read 0.49% / 1.82% and 0.33% / 1.02%, and **darker: Cornell 0.959 of the input's mean, Sponza 0.982. That is warm-up, not the port** (RC4.3 review): the history is a running mean of up to 64 samples, then fades with a time constant of about 64 frames (gi1.comp:4085-4097), so the cold-start frames after a reset, when the world cache is still empty and the input dark (Cornell's E at frame 8 is 0.253 against 0.28 settled), stay in it; the ratio over the input reaches 0.998-1.002 only from about frame 192 (the reviewer's measurement). The three page's picture is taken at frame 16 (GI term with history over one-bounce DDGI 1.403 on Cornell, against the interpolation's 1.564 × 0.905 = 1.415). Two more properties of Capsaicin's denoiser worth knowing: the history colour is irradiance × samples in halves, stored every frame, so the store's rounding is fed back (a constant settles 0.32% low on the M5, which truncates half stores; check 1); and the history is spread spatially: the first 8 frames after a reset are blurred over up to 8 pixels (the mask is 8 minus the history's length), and a still camera's 3 × 3 tent (weights 1 and 0.29) spreads it a little more each frame.

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
11. **SH projection with each cell's mean basis** (quality): `ProjectScreenProbes` takes the SH basis at each cell's centre, as Capsaicin, which leans towards the normal, so a constant sky reads 2.1% bright and a varying sky's coefficients are off by up to 4.2% of the largest (CPU, `screen-irradiance.test.ts`). Averaging the basis over 4 × 4 points per cell brings that to within 0.2% and 1.7%; the points' cost (16× the basis evaluations, in a pass that is cheap to begin with) was never measured. `probeCellSH(cell, normal, 4)` on the CPU is the reference (DECISIONS 2026-11-10).
12. **An unbiased second bounce** (quality): count the discarded and missed second-bounce rays as zero samples of their first-bounce cell (or drop the 1 / (1 - 0.7) weight, which is right only then), so a cell's indirect value is albedo × the light it stands for instead of about 3.3 times that in a closed room (phase 3's note; `radiance-cache-multibounce` measures both: 1.67 L cached against 0.25 L at most physically on the corner).
13. **An empty cell's backup that doesn't drift with history** (quality): Capsaicin's backup for a cell no ray reached, `64 / (R (64 − R))` of the mean under a constant sky (gi1.comp:1213-1232), with its hysteresis and CDF feedback, takes a constant sky to 0.69 πL once history runs (phase 4 notes). Candidates: a backup that doesn't depend on the rays' sum (the reached cells' mean, or the cell's own history), or hysteresis that keeps an empty cell's history instead of taking the backup at once. Measured against the baseline on the constant-sky furnace and Cornell; a lever beside the faithful path, not a fix to it.
14. **SSGI samples on texel centres, at least a pixel away** (quality, small): Capsaicin rebuilds each SSGI sample at its own uv with its texel's depth and lets the first step fall within the centre's own texel (ssgi.comp:191-196). Snapping samples to texel centres and starting a pixel out, as GTAO implementations do, is worth 2-4 points of E on an open floor (the review's f64 twin: 0.735 → 0.753 πL with 1 slice, 0.969 → 1.011 with 16). Measured on the constant-sky floor and Sponza against the baseline; worth doing with lever 16, not alone.
15. **SSGI's near-field light in irradiance units** (quality): × π, or the samples' light as radiance over their solid angle, so it adds to the probes' E like for like (the list for AMD above). With 14, measured against a path-traced reference at corners.
16. **More SSGI slices** (quality; Capsaicin's own setting, `ssgi_slice_count_`, ssgi.h:47, default 1): one slice per pixel per frame darkens open surfaces on average (the cone SH isn't linear in one slice's occlusion and bent normal) and is the most of the flicker before the denoiser. The review's f64 twin on a floor from 3 m at a 15° pitch: E = 0.735 πL with 1 slice, 0.969 with 16 (1.021 without SSGI). Costs a slice's samples each (not timed). Measured against the baseline on the constant-sky floor, Sponza and the flicker page; the denoiser's temporal accumulation may recover part of it at 1 slice.

## Extensions after the baseline (ranked)

1. DDGI as the third cache level (screen cache, world cache, then the DDGI volume), through the `BounceSource` slot.
2. Glossy reflections (GI-1.1), once materials carry roughness and metalness (D8).
3. Brixelizer and surface-cache world caches through the slot.
4. Object-attached cache entries for moving objects (EA surfels; Webgiya, MIT, is a WebGPU precedent).
5. Lumen-style improvements: product importance sampling, a GPU priority queue for cache updates.
6. Split Radiance Cascades (arXiv 2607.20384) as a world-cache experiment.

## Siblings: what goes back

- **`@pmndrs/upscaler`:** used as-is. A GI-history path for its issue #7 if phase 5 runs into it.
- **`@pmndrs/denoiser`:** GI-1.0's image-space denoiser (phase 4; built by RC4.3 as `packages/denoiser-gi1` in this repo, in that library's shapes, to move there with Dennis's go-ahead: its README says how), hit distance and cache confidence as guides, the "coverage" input mode (its issue #23), a bilateral upsample node for half-resolution GI. Probe-space filtering and hysteresis stay in the technique.
- **Textures:** GPUTex where real hit textures (D2) or big scenes need the memory.

## Risks

| Risk | Mitigation |
| --- | --- |
| The faithful baseline is far slower than native in tracing | Expected and measured: RC1.5 predicts it, phase 5 lists it, BVH work and levers follow |
| Getting three's buffers into compute | RC1.1 first; the smallest pinned step if three needs one, in [the divergences](radiance-cache-divergences.md) |
| Flicker the numbers miss | Dennis's look by eye is a phase 4 done-when |
| Temporal accumulation fighting the upscaler's jitter | Placement decided in phase 5, with upscaler issue #7 |
| World cells stale after objects move | Dirty-region invalidation in phase 6; object-attached entries later |
| Clashing with M2/M3 work | A separate branch; platform changes as separate, small merges |
| The denoiser branch is unmerged and Apple-only so far | Our contribution includes an RTX 3060 run |
