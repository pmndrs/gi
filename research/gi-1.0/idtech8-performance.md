# id Tech 8 performance thinking, mined for the GI-1.0 WebGPU port

Written 2026-11-09. Purpose: treat id Software's GI talk as a source of performance levers to try after the faithful GI-1.0 port. Nothing here is tested on our code.

## How to read the citations

- **S#** = printed slide number in `sources/SOUSA_SIGGRAPH_2025_Final.pdf` (Tiago Sousa, "idTech 8 Global Illumination", SIGGRAPH 2025 Advances). Speaker notes sit on the same slide as the slide text. Slide 5 has its notes continue onto a second PDF page, so from S6 on, **PDF page = S + 1**.
- **DE p#** = PDF page in `sources/GeffroyEtAl_SIGGRAPH2020_DoomEternal.pdf` (Geffroy, Gneiting, Wang, "Rendering the Hellscape of DOOM Eternal", SIGGRAPH 2020).
- **PLAN** = the "Native baseline" table in `PLAN.md` (Capsaicin GI-1.2, Sponza, 1280x720, RTX 3060 Laptop, hardware RT, one 120-frame run, 5.9 ms for the GI block of which about 4.6 ms is diffuse). Groups used below, in ms: tracing and cell population 0.98; cache update and resolve 1.11; screen probes 1.63; GI denoise and resolve 0.78; glossy 1.28 (not in our first port).
- "Derived" means my arithmetic from slide numbers, not something id states.
- I read all 38 slides (text extraction of the PDF, including notes). The hotspot table on S31 came out garbled in plain extraction, so I re-extracted it in content-stream order and checked the sums. I could not render slide images (no PDF rasteriser on this machine), so I did not see the figures, charts or screenshots; anything that exists only as a picture is not covered.

## 1. What I found and what I could not

| Item | Result |
|---|---|
| Sousa, idTech 8 GI, SIGGRAPH 2025 | Downloaded and fully read. |
| Geffroy et al., DOOM Eternal, SIGGRAPH 2020 | Downloaded (11.5 MB) and searched. It is about hybrid light binning, geometry caches, decals, materials and water. It has **no GI content**. Useful only for light-culling tricks (section 5). |
| Vahl 2025, "Pushing 60hz" (Indiana Jones) | Not found anywhere public (web searches return only news pages). Sousa cites it as a reference (S37). |
| GDC 2025 "Rip & Tear: Breaking Down the Rendering of DOOM: The Dark Ages" | Session page returns HTTP 403; no slides located. |
| NVIDIA / id posts on The Dark Ages | NVIDIA GeForce news post exists but is marketing level: it says the path-tracing update uses NVIDIA SHaRC and DLSS Ray Reconstruction. The NVIDIA developer blog "How id Software Used Neural Rendering and Path Tracing in DOOM: The Dark Ages" is a client-rendered page; my fetch got no article text. Not used. |
| id Tech 7 "Devil is in the Details" (2016) | Not downloaded; it predates GI and is not relevant here. |

Details and URLs are in `sources/SOURCES.md`.

## 2. Frame structure, hardware and per-pass cost

### Order of work in a frame (S12 and notes)

1. Update the cascaded light grid (an async-compute dispatch, S14).
2. Sample world visibility: irradiance-volume probes trace rays and record hits; each hit marks a world radiance cache entry (S16 to S19).
3. Shade all marked cache entries and store them in the world radiance cache (S20).
4. Update the irradiance volumes from the cache (S21).
5. Final gather: one ray per pixel into the caches, zero shading (S22).
6. Denoise, then upscale (S23).
7. A froxel irradiance volume serves transparencies (S29).

### Measured costs (S31, ms, "common hotspot", mission 4, a large vista with many enemies, vegetation, particles)

These are the values id shipped with. Resolution is the output resolution; the notes say the PC machine had an RTX 4080 (about 4x the console TFLOPS). All hardware ray tracing.

| Pass | Series S 900p | Series X 1440p | PS5 1440p | PS5 Pro 1800p | PC 4K |
|---|---|---|---|---|---|
| World sampling (visibility rays) | 0.38 | 0.27 | 0.45 | 0.083 | 0.091 |
| Radiance cache (shade entries) | 0.2 | 0.21 | 0.109 | 0.11 | 0.109 |
| Irradiance volumes | 0.08 | 0.08 | 0.046 | 0.035 | 0.07 |
| Final gather | 0.6 | 0.535 | 0.489 | 0.385 | 0.5 |
| Denoise | 0.208 | 0.18 | 0.266 | 0.22 | 0.52 |
| Upscale | 0.65 | 0.59 | 0.69 | 0.72 | 0.445 |
| Cost, serial (as printed) | ~2.11 | ~1.9 | ~2.05 | ~1.923 | ~1.7 |
| Cost, async (as printed) | ~1.71 | ~1.4 | ~1.55 | ~1.82 | n/a |

Checks and caveats (derived unless stated):
- Rows sum to the printed serial cost for Series S (2.118), Series X (1.865), PS5 (2.05) and PC (1.735). The **PS5 Pro column does not**: the rows sum to 1.553 but print 1.923 serial and 1.82 async. Either a cell or a total is wrong on the slide. I would not rely on the Pro totals.
- Async saved about 0.5 ms on Series X and PS5, about 0.4 ms on Series S, and little on PS5 Pro because "the async dispatch cases actually run fairly fast" (S31 notes). The notes add: "Our async workload is actually pretty full... Don't think we wouldn't have been able to achieve our performance target without." So async is about 20 to 25 percent of the GI time on consoles.
- The table does not say which quality tier each platform ran, how many rays per probe, or the final gather resolution. Final gather resolution is half or quarter per axis depending on the quality setting (S22). The light grid update (about 0.2 ms on consoles, 0.1 ms on PC, S14), directional occlusion (S30), froxel work and reflections are not in the table. Whether "cost" is GI-only is my reading, not stated.
- Where time goes: final gather plus denoise plus upscale is 1.3 to 1.5 ms out of 1.7 to 2.1 ms (about 70 to 80 percent). The tracing of world visibility and the cache shading together are 0.3 to 0.55 ms. That matches the GI-1.0 paper's finding that caching and sampling dominate, not traversal (paper.md section 9).
- id's irradiance-volume update is tiny: 0.035 to 0.08 ms for the whole octahedral atlas (S20, S31).
- A benchmarking lesson from the notes (S31): at CEDEC the capture comparison was wrong because "didn't compare exact same cascades/volumes being updated, there can be perf variations". With interleaved updates (one cascade and one local volume per frame, S15), frame time varies by which volume is updating. Our benches should average over a full update cycle.

Not comparable with our baseline: id's numbers are consoles and a 4080 with hardware RT, at 900p to 4K output; ours is a laptop 3060, 720p, and a software BVH later. Treat them as design evidence, not targets.

### Derived ray budget

Volume probes: 16x16x16 per cascade, 6 cascades, 10x10x10 local volumes, one cascade and one local volume updated per frame, 64/32/16 rays per probe by platform (S15). At the 64-ray setting that is up to about (4,096 + 1,000) x 64 = about 326k visibility rays per frame; at 16 rays it is about 82k. Final gather adds 1 ray per pixel at half or quarter resolution per axis (S22): at 1440p that is about 920k or 230k rays. Capsaicin's defaults at 1080p, per PLAN, are about 522k probe rays plus 160k multibounce rays plus a shadow ray per hit. So id traces fewer and cheaper rays (visibility only, no shading at hit), and moves the expensive work to a bounded cache shading step.

## 3. Techniques, with the GI-1.0 pass each maps to

Columns: **HW RT** = does it need hardware ray tracing; **WebGPU** = anything missing; **Output** = Free (bit-identical or equal in expectation) or Trade (changes the image or lag).

| # | Technique (source) | What id does | Applies to Capsaicin pass | Expected effect | HW RT | WebGPU | Output |
|---|---|---|---|---|---|---|---|
| T1 | Visibility-only probe rays; shade later (S15 to S17) | Rays record a 128-bit hit record (SBT index + 17-bit primitive id, instance, barycentrics, distance) and no shading happens at the hit. | `PopulateScreenProbesCells`, `PopulateMultibounceCells`, `PopulateRadianceCache` (0.98 ms group) | Cheaper hit stage; turns per-hit shadow rays into per-entry work. | No | None. Our BVH returns a hit record. SBT/bindless not needed (our `giSurfaceAt`). | Free |
| T2 | Fixed budget of hash entries shaded per frame, reused for N frames (S19 to S20: "~20k cache entries updated per frame. Re-use for N frames") | Active-entry list built with atomics; shading is bounded by the list, not the ray count. | `UpdateRadianceCache`, `ResolveCells`, `UpdateMultibounceCells` (1.11 ms group) | Largest lever; the same idea PLAN already ranks #1. 20k is id's number for large levels, our value must be tuned. | No | None | Trade (lighting lags by up to N frames) |
| T3 | Active-cell counter + indirect dispatch, one dispatch per shader type, async (S19 to S20) | Atomic add into a per-dispatch list, then `DispatchIndirect`; "Not that many shaders (less than 10)". | The ~15 indirect passes in the Capsaicin chain | Avoids dispatching over empty space. | No | `dispatchWorkgroupsIndirect` exists. Split-by-material needs only one shader for us. No async queue. | Free |
| T4 | 1D spatial hash, 25 cm cells, camera-distance LOD (cell size doubles, S18 to S19 code), 14 MB, linear probing over the next N slots; results dropped on overflow (S17 notes) | `lod = exp2(floor(log2(1 + dist/LOD_DIST)))`. | Hash cache (`hash_grid_cache`) | Confirms Capsaicin's design family. 14 MB is two orders of magnitude below Capsaicin's roughly 1 GiB default, so memory is not required to be large. | No | None | Trade if cell size is raised |
| T5 | Multiple bounces from last frame's irradiance volumes instead of more rays (S20: "Previous frame irradiance volumes for multiple bounces") | Shading reads the irradiance volumes updated last frame. | `PopulateMultibounceCells`, `UpdateMultibounceCells` | Removes the multibounce ray pass (about 160k of about 680k rays in Capsaicin's 1080p defaults, per PLAN). Our DDGI can provide the volumes. | No | None | Trade (volume resolution, one frame of lag) |
| T6 | Interleaved volume updates: one cascade and one local volume per frame (S15) | Rotates which volumes trace. | World cache refresh and any DDGI volume we reuse | Cost proportional to one cascade, not all. | No | None | Trade (slow response in far cascades) |
| T7 | Irradiance volume atlas: octahedral tiles 8x8 (10x10 with border), RGB9E5 color + RG16F visibility, 2560x1585, 64 MB (30 MB low spec), 8x8 thread group = 64 rays, 0.08 ms (S20) | One workgroup per tile, results in groupshared. | DDGI volume if used as last-resort cache (not a GI-1.0 pass) | Update costs almost nothing, so volumes are a cheap fallback. | No | RGB9E5 is a sampled-only texture format in WebGPU as far as I know [verify]; pack manually in a `u32` storage buffer. | Free |
| T8 | Fall-through lookup order in the gather: screen cache if the hit is in the frustum and unoccluded, else world radiance cache if valid, else irradiance volumes (S22) | Zero shading in the gather. | Screen-probe sampling and `ResolveCells` lookups | Fewer holes without extra rays; DDGI becomes the safety net. | No | None | Trade (changes how holes fill) |
| T9 | Gather at reduced resolution: 1 cosine-weighted blue-noise ray per pixel at half or quarter per axis, set by quality (S22, S24 low quality "bit softer", "Loses some directional lighting details") | Quality tiers. | `InterpolateScreenProbes`-side work and `ReprojectGI`, `FilterGI`, `ResolveGI1` (screen probes 1.63 + denoise 0.78 ms) | Proportional saving on the per-pixel stages: a quarter-per-axis gather has 1/16 the pixels. | No | None | Trade |
| T10 | Per-pixel 2-band SH output (3 x RGBA16F, R = L0, GBA = L1) so normal maps keep detail (S22) | Each pixel is an irradiance probe. | Final interpolation and `ResolveGI1` | GI-1.0 already projects to SH; this confirms the choice. | No | Storage RGBA16F textures fine. | Free |
| T11 | Denoise at gather resolution: separable "bilateral" Gaussian, inputs preloaded to groupshared in FP16, weights from normal, depth and hit-distance deltas; the notes admit it is "Not really separable, but good for perf" (S22) | Spatial then upscale then temporal. | `FilterGI` (inside 0.78 ms group) | Cheap; id's PC denoise is 0.18 to 0.52 ms. | No | WGSL `f16` is optional (`shader-f16`); fallback f32. | Trade (slight bias) |
| T12 | Upscale: 4-tap Poisson bilateral (normal/depth weights), then temporal filter with a per-pixel valid-frame counter capping history, `a_n = min(n+1, max)` (S22) | Replaces full-resolution filtering. | `ResolveGI1` / `FilterGI` | Replaces many-tap filters with 4 taps. id's upscale is still 0.445 to 0.72 ms, so it is not free. | No | None | Trade |
| T13 | RGB9E5 for temporal history and 32-bit color; avoid R11G11B10F (S22, S25: "green/yellow shift/no pure white", banding); RGBA16F where RGB9E5 is unsupported | 32 bits per pixel instead of 64. | GI history and radiance cache values | Halves history bandwidth. | No | See T7 on format support; manual packing costs ALU. | Free if packed well; R11G11B10 is visibly worse |
| T14 | Quality tiers by platform: rays per probe 64/32/16; cascade size 16^3 vs 12^3; atlas 64 vs 30 MB; froxel volume 50 vs 25 MB (S15, S20, S28) | Same code, smaller constants. | All tracing and cache sizing | Direct knobs; we should expose them as settings. | No | None | Trade |
| T15 | Async compute: light grid, cache shading dispatches and others overlap graphics work (S14, S19, S31) | Saves about 0.4 to 0.5 ms of about 2 ms on consoles. | Cache update passes in principle | About 20 to 25 percent on id's hardware. | No | **No async queue in WebGPU.** No equivalent. | Free (absent for us) |
| T16 | Directional occlusion pass added to composite: ray-march along the hemisphere, store average unoccluded direction, denoise and temporal filter (S30) | Restores high-frequency detail missing because BLASes do not contain every LOD/detail geometry. | Not in GI-1.0 | This is a quality tool that lets the traced scene be coarse. If our software BVH uses simplified geometry for GI, this is the compensation. Cost not given. | No | None | Adds cost, adds quality |
| T17 | Cascaded world-space light grid: 16^3 x 8 cascades, 64 ids per cell, hierarchical gather with groupshared bit arrays, about 0.2 ms on consoles and 0.1 ms on PC, async, 30 MB (S13 to S14) | Lets shading of RT hits find lights without screen binning. | Shading in `UpdateRadianceCache` | We use the platform light grid; this is a reference design only. | No | `subgroupBallot` type ops are available when the subgroups feature is on. | Free |
| T18 | Hybrid cluster + tile binning with a budget under 500 us; subgroup-ballot bitfield-to-list compaction; hash scalarization of light lists (DE p9, p15, p22) | From the 2020 talk. | Light sampler and light culling | Pattern for subgroup-first compaction. Small for GI. | No | Subgroup ballot, prefix and elect are exposed. | Free |
| T19 | Froxel irradiance volume reused for all transparencies (S29) | Stores SH at the light-scattering froxel dispatch; 50 MB (25 MB low). | Not in GI-1.0 | Avoids per-object GI lookups. | No | 3D storage textures OK. | Free |

## 4. What id says did not work or was too slow

- **Hardware RT cannot trace many rays on low-end parts.** "Can't trace many rays (low end hw can be orders of magnitude slower than faster hw)" (S10 notes). The whole design (visibility-only rays, bounded shading, 1 ray per pixel at reduced resolution) follows from this. This is the closest statement to our software-BVH situation.
- **Glossy ray-traced reflections were disabled on consoles for performance.** "Usually fairly fast (~2 ms)", "Ran out of time to optimize further", and only run on very smooth surfaces above a smoothness threshold; SSR plus probes was always the fallback (S27). This supports leaving GI-1.1 out of our first port (PLAN: glossy 1.28 ms).
- **Light grid has the clustered-binning weakness.** Overdraw grows with distance; 16^3 per cascade was the "sweet spot for memory/perf"; the notes say the culling is "Fairly fast for amount of data it processes, but could be improved" (S14 to S15).
- **R11G11B10F color storage.** Color shift, no pure white, banding (S25). They want RGB9E5 supported everywhere.
- **Hash collisions.** Handled by linear search over the next N entries; "Fingers crossed we fit, else we have to drop some results" (S17 notes). Overflow is silent loss, so measure it.
- **Detail loss.** BLASes lack upper-LOD detail, which is why directional occlusion is needed (S29).
- **Baked lightmaps (the thing GI replaced).** Full resolution bakes took too long, so they dropped to half resolution per axis (S6), and a ten-times larger level set would have been 27 to 68 days turnaround (S9). Context only; not a runtime cost.
- **Low quality is softer**, loses some directional lighting detail (S24), but they call the two settings "Relatively close".
- I found no slide giving a failed experiment for the cache itself (for example, rejected hash sizes or update fractions). Absence from the slides, not proof.

## 5. Levers to try after the faithful port, ranked

Ranking is by expected saving against the PLAN baseline per pass group, then by how little quality we give up and how little new machinery is needed. All of these are untested on GI-1.0; id's evidence is for its own pipeline, which differs (its probes live in volumes, not screen tiles, and its screen cache is applied inside the gather).

| Rank | Lever | Source | Capsaicin group it hits (PLAN ms, 720p) | Expected effect | Output |
|---|---|---|---|---|---|
| 1 | **Fixed per-frame budget of world-cache entries shaded, chosen by priority (new, stale, near a change), reused for N frames; shade from stored hit records** | T1, T2, T3 (S17 to S20) | Cache update and resolve 1.11 plus the shadow-ray part of population 0.98. On our software BVH PLAN estimates about 10 ms of shadow rays per frame at 1080p defaults, so the saving is potentially larger than on the native run. | Biggest single lever. id shades about 20k entries per frame for a whole game level. | Trade (lag) |
| 2 | **Replace the multibounce tracing pass with a lookup of last frame's irradiance volumes (our DDGI)** | T5, T7, T8 (S20, S22) | `PopulateMultibounceCells`, `UpdateMultibounceCells`, part of the 0.98 + 1.11 groups. Removes about 160k of about 680k rays at 1080p defaults. | Saves rays and cell updates; also gives a fallback where the hash cache is empty. Depends on DDGI already being resident. | Trade |
| 3 | **Internal-resolution tiers for the per-pixel stages (half and quarter per axis), with a 4-tap bilateral upscale and temporal accumulation** | T9, T11, T12 (S22, S24) | Screen probes 1.63 and denoise 0.78 (together 2.41 ms of about 4.6). | A quarter-per-axis tier cuts per-pixel work to about 1/16; id still pays 0.4 to 0.7 ms for the upscale on its own hardware, so budget for that. Overlaps with PLAN's "half-res + upscaler" lever. | Trade (soft on low tier) |
| 4 | **Do the denoise in groupshared at gather resolution in f16 (with f32 fallback), with a frame-counter history cap** | T11, T12 (S22) | `FilterGI` and `ReprojectGI` inside the 0.78 ms group. | Bandwidth and ALU reduction. Free-ish if f16 is available. | Free to small trade |
| 5 | **Compact data: 16-byte hit records, RGB9E5-style 32-bit color for history and cache values, 3 x RGBA16F SH** | T1, T10, T13 (S17, S22) | All groups through memory traffic; most felt in screen probes (1.63) and reprojection. | Halves history bandwidth; also shrinks the cache. | Free if packing is correct |
| 6 | **Interleaved/amortised refresh of far or low-priority cache regions** (one cascade per frame style) | T6, T14 (S15) | Cache update and resolve (1.11) | Proportional to the share of the cache refreshed per frame. Needs a full-cycle average in benches (S31 notes). | Trade |
| 7 | **Quality tiers as settings**: rays per probe, cell size, resolution (64/32/16 rays; 16^3 vs 12^3; half vs quarter) | T14 (S15, S20) | All tracing and cache stages | Gives a ladder to hit 60 fps on weaker GPUs. Do this to ship, not as a speedup of the best tier. | Trade |
| 8 | **Directional occlusion composite so the traced scene can be coarser** | T16 (S30) | Not a pass in GI-1.0; could let the BVH use lower-detail meshes for GI rays. | Only if we choose to simplify the traced geometry. Cost unknown. | Adds cost, adds quality |
| 9 | **Subgroup ballot compaction for active-cell and light lists** | T3, T18 (DE p15) | Prefix scans (4 per frame) and active-cell lists | Small; consistent with the subgroups-first rule. | Free |
| 10 | **Async compute** | T15 (S31) | n/a | About 20 to 25 percent on id's consoles. **Not available in WebGPU**; the only mitigation is keeping independent passes adjacent so the driver can overlap them, which is not guaranteed. | n/a |

## 6. Notes on fit and open questions

- id's pipeline puts visibility probes in volumes and shades world cache entries those rays touch. GI-1.0 puts probes on screen tiles and also feeds a hash cache. The caches are the same family, so levers 1, 2 and 5 transfer well; levers 3 and 4 are about the per-pixel stages, where GI-1.0 differs more.
- Because id shades about 20k entries per frame, I would check how many cells Capsaicin updates per frame in the native run (the UpdateRadianceCache dispatch size) before choosing our budget. I did not measure that.
- I could not verify from the slides how id decides which hash entries to refresh (priority or simple recency); the slides only say entries are updated per frame and reused for N frames (S20).
- RGB9E5 support as a storage format in WebGPU, and `shader-f16` availability on Chrome 156 on our two laptop GPUs, should be checked on the target machines before relying on levers 4 and 5.
- Neither id deck gives a ray-tracing cost per ray, so there is no direct number to scale to a compute BVH. Their 0.083 to 0.45 ms world-sampling range is hardware RT for at most about 326k rays (derived).
