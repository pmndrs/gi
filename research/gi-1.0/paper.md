# AMD GI-1.0: implementation-grade summary

Sources: `sources/gi10-paper.pdf` (arXiv 2310.19855, called "paper" below), `sources/SA2023_RealTimeReflection.pdf` ("GI-1.1 paper"), `sources/GDC2024_Brixelizer_GI.pdf` ("Brixelizer slides"). Details of each are in `sources/SOURCES.md`.

Citation notes:
- Paper citations use section, figure, algorithm or table number. I did not map PDF page numbers.
- There is no separate GDC 2023 slide deck. The GPUOpen talk page links the same technical report. So "slides" below means the 2024 Brixelizer deck, cited as "Brixelizer slide N" (the number printed on the slide).
- Anything marked **(Capsaicin source)** comes from AMD's MIT-licensed reference code (gi1.h, gi1.cpp, hash_grid_cache.hlsl, read 2026-11-09, current main, tagged up to v1.3), not from the paper. It can differ from the 2022 paper. Anything marked **(inferred)** is my reading.
- The text extraction of the paper lost Greek symbols and the bodies of Algorithms 1 to 6. Parameter names inside the algorithms (for example the "tau" depth/plane tolerance and the "alpha" blend factor) could not be recovered. I describe those from the prose and flag the gaps.

## 1. Summary

- Problem: only about 1 sample per pixel is affordable for ray-traced diffuse GI at 1080p. Probe grids (DDGI) are blurry and slow to react. Per-pixel reservoir methods (ReSTIR GI/PT) are noisy and too slow for consoles (abstract, §1).
- Idea: two radiance caches, each doing the job it is good at (§2, Fig. 2).
  - Screen cache: one 8x8 octahedral *incoming* radiance probe per 8x8 pixel tile, placed on visible surfaces, filled over several frames, reused through reprojection.
  - World cache: a spatial hash grid of *outgoing* radiance at secondary hit points, with distance-based LOD cells and a decay-based eviction.
- Screen probe rays use ray guiding from the reprojected probe. A ray that hits something reads the world cache. The world cache is filled with direct lighting from a world-space ReSTIR variant plus a light grid. Reading last frame's lit result gives approximate multi-bounce (§2.2.4, §2.3).
- Per pixel, SH-projected probes are interpolated, then denoised. A short-range horizon-based screen-space GI recovers fine detail (§2.4, §3).
- Results (Table 1): 1.93 to 3.12 ms total on RX 6900 XT, 1.91 to 3.43 ms on RTX 3080, 1080p, 1 spp, 99k to 9M triangles. Fig. 1 quotes 3.5 ms (Kitchen) and 4.2 ms (Sponza).
- Hardware RT (DXR 1.1) is used for all tracing. The authors say any closest-hit method works, and they tried a hybrid distance-field and BVH tracer (§4).

## 2. Per-frame pipeline

The paper never gives one ordered pass list. This list merges the paper's section order with the **(Capsaicin source)** timed-section order in gi1.cpp, so pass names are Capsaicin's.

| # | Pass (Capsaicin name) | Inputs | Outputs | Shape / notes |
|---|---|---|---|---|
| 0 | Clear/purge world cache: ClearBucketOverflowCount, PurgeRadianceCache | decay buffer | freed tiles | Cells with decay at 0 are deallocated (§2.2.1). The paper does not say which pass does this. The pass name is **(Capsaicin source)**. |
| 1 | ReprojectScreenProbes | depth, normals, motion vectors, previous probe buffers, previous SH | new probe mask (packed pixel coords), reprojected radiance and SH | One 8x8 thread group per tile, LDS used to pick the best pixel (Algorithm 1, §2.1.1). Runs at full-resolution pixel rate (§2.4.1). |
| 2 | Cached-tile lookup (LookupScreenProbes, with InitCachedTileLRU) | LRU side cache | per-8x8-tile lists of cached probes | Tile-deferred-style scatter (§2.1.8). |
| 3 | SpawnScreenProbes, with Count/Scatter/Compact | probe masks | list of tiles to spawn | Halton jitter per spawn tile (§2.1.1). Scan and compaction passes are Capsaicin-only detail. |
| 4 | PatchScreenProbes | empty-tile queue, override-tile queue | patched spawn list | Moves rays from successfully reprojected tiles to empty tiles (§2.1.2, Algorithm 2). |
| 5 | SampleScreenProbes (ray guiding) | reprojected radiance in LDS | ray directions per probe, ray count | Luminance prefix sum into a CDF, then pick a cell by CDF (§2.1.3, Fig. 5). Parallax-corrected scatter of reprojected radiance (Fig. 6). |
| 6 | PopulateScreenProbesCells | rays | per-ray radiance payload and world-cache cell requests | Closest-hit trace. A miss adds environment light. A hit looks up or allocates a cell by descriptor (§2.1.3, §2.2.1). Dispatch size is `max_ray_count` threads (Capsaicin source). |
| 7 | PopulateMultibounceCells (only if multibounce is on) | cell visibility list | more cell requests | **(Capsaicin source)** option `gi1_use_multibounce`, default true. Not in the paper (see §10). |
| 8 | World-space ReSTIR: ClearReservoirs, GenerateReservoirs, (GenerateMultibounceReservoirs), CompactReservoirs, ResampleReservoirs | hit positions, light grid | reservoirs per hit cell | RIS plus WRS, temporal reuse only, one shadow ray per reservoir after resampling (§2.3.1). |
| 9 | PopulateRadianceCache (shadow rays) and UpdateRadianceCache | reservoirs | direct lighting written into cells | The shadow trace uses an any-hit kernel for world cache rays (§5). |
| 10 | ResolveCells and UpdateTiles | cell accumulators | filtered MIP chain per tile | EMA at MIP 0, box-filtered upper MIPs (§2.2.3). Reads the previous frame's lit pixel when reprojection succeeds (§2.2.4). |
| 11 | BlendScreenProbes | ray payload, reprojected radiance | new probe radiance | Biased shadow-preserving hysteresis (Algorithm 3, §2.1.4) and untraced-cell averaging (Fig. 7). Also updates the LRU cache. |
| 12 | ReorderScreenProbes (LRU/MRU), FilterProbeMask | cached probe list | reordered queue, MIP chain of the probe mask | §2.1.8, Fig. 8, Algorithm 4. |
| 13 | FilterScreenProbes | probes | filtered probes | Separable 7x7 sparse blur in probe space, with depth-plane and angle rejection (Algorithm 5, Fig. 9, §2.1.6). |
| 14 | ProjectScreenProbes | probes | SH (3 bands) per probe | Done only for newly spawned probes. Reprojection carries the SH forward (§2.4.2). |
| 15 | InterpolateScreenProbes | SH probes, depth, normals | irradiance per pixel, alpha flag for relaxed interpolation | Full-rate pass. 4 neighbours found with the sparse directional search, plus pixel jitter (§2.4.1). |
| 16 | ReprojectGI, then FilterGI | irradiance, history, disocclusion mask | denoised irradiance | Temporal accumulation plus adaptive spatial filter (§2.4.3, Fig. 19). Capsaicin also reads a "Gradients" input (names only). |
| 17 | Near-field SSGI (separate technique "ssgi") | depth, normals, history | AO, bent normal, near-field GI mask | One horizon slice per frame, denoised by the loop in #16 (§3). In Capsaicin it is a separate render technique feeding "OcclusionAndBentNormal" and "NearFieldGlobalIllumination". |
| 18 | ResolveGI1 | all of the above | final GI texture | Combines far-field irradiance and near-field GI with the bent-cone mask (§3.1). |
| (glossy) | TraceReflections, MarkFireflies, CleanupFireflies, ResolveReflections (split X/Y, A-trous), ReprojectReflections | see §7 | reflection texture | GI-1.1. |

Resource formats and sizes the paper states:
- Probe grid: a 2D texture the size of the render target, rounded up to a multiple of 8. One 8x8 octahedral probe per 8x8 tile, alpha channel = ray travel distance (§2.1.1, §2.1.3). The RGB format is not stated.
- Probe mask: 2D texture, one 32-bit integer per tile holding the pixel coordinates of the probe, with a sentinel for empty tiles. Plus a MIP chain keeping the first valid probe of each 2x2 (§2.1.5).
- LRU side cache: an array of probe indices plus a 2D texture with the same dimensions as the probe grid. Each cache entry stores a 128-bit record holding single-float position and packed normal (§2.1.8).
- Hash cell tile: 8x8 cells per tile, MIP-mapped, linear layout (§2.2.3, Fig. 13).
- Light grid: uniform voxel grid around the AABB of all hit points (§2.3.2).
- **(Capsaicin source)** ScreenProbes has double-buffered probe, mask and SH buffers; `probe_size_ = 8`; `sampling_mode_ = kSamplingMode_QuarterSpp`, which makes the spawn tile 16x16 (probe_size_ << 1). The paper's own default is one probe per 8x8 spawn tile unless temporal upscale is enabled. The paper's Figure 3 example uses a 2x2 temporal upscale (a quarter of the probes per frame, full grid after 4 frames). The `max_ray_count` formula is `max_probe_spawn_count * 8 * 8`, i.e. one ray per octahedral cell at maximum.

## 3. Screen cache

- Placement. The probe stores incoming radiance over the oriented hemisphere as an 8x8 octahedral map (§2.1). A "spawn tile" is 8 x (lambda_x, lambda_y) pixels where lambda is the temporal upscale factor per axis. A Halton 2D jitter chooses the pixel inside each spawn tile where the new probe goes. With a 2x2 upscale, a quarter of the grid is filled per frame and all of it after 4 frames (§2.1.1, Fig. 3). Per-frame full-resolution work is limited to interpolation and reprojection (§2.4.1).
- Reprojection. Each lane of an 8x8 group tests one pixel of the old grid. Lanes use a depth-bit-packed atomic max in LDS (Algorithm 1) to pick the best pixel for reuse. A plane-distance heuristic (called "cell size" in §2.1.7) decides if reuse is allowed, plus a normal-agreement test (0.95 threshold visible in the pseudo-code; the symbol around it is lost). Sky pixels are skipped. Inputs: depth, normals (G-buffer) and per-pixel motion vectors (§2.1.1).
- Adaptive sampling. Reprojection leaves holes on disocclusion. Rather than add rays, the paper moves rays from tiles whose reprojection succeeded to empty tiles using two queues (empty tiles; override tiles), keeping the ray count constant (§2.1.2, Algorithm 2, Fig. 4).
- Ray guiding (importance sampling). The luminance of the reprojected radiance goes to LDS, a parallel prefix scan makes a CDF, a cell is picked in proportion to its luminance and a random sample taken inside it (§2.1.3, Fig. 5). Reconstruction scatters radiance from the 3x3 tile neighbourhood of reprojected probes, and from cached probes (§2.1.8), into the new probe with a parallax correction using the stored ray distance (Fig. 6). The same plane-distance heuristic rejects far probes.
- Blending. New radiance is blended with the reconstructed estimate by a *biased temporal hysteresis* (Algorithm 3). The blend factor is a function of the normalised difference between new and reconstructed values, with the result clamped and remapped to [0.0, 0.95] (§2.1.4). Plain EMA lost too much detail. The hysteresis preserves shadows, darkens the image slightly, and removes fireflies smaller than the cell's cone. Cells with no rays take the average radiance of populated cells (Fig. 7).
- Probe filtering: a separable 7x7 sparse blur, neighbours found by the mask-MIP directional search (Algorithm 4); depth rejection plus an angle check to keep small occlusion (Algorithm 5, Fig. 9, §2.1.6).
- Adaptive cell size. The plane-distance tolerance grows with distance from the camera (Algorithm 6, Fig. 10), computed from vertical FOV and a target cell size in pixels. Target value 8.0 pixels, roughly the spacing between probes (§2.1.7). The constant can be precomputed on the CPU.
- LRU side cache (§2.1.8, Fig. 11). Thin geometry makes reprojection fail every frame. Evicted probes go to a persistent queue so they can be matched several frames later. Three update cases are listed. An atomic compare-and-swap guards concurrent updates. An MRU reordering pass every frame keeps active entries from being evicted. The cache size is not given.
- Interpolation (§2.4.1). Per pixel, find the 4 nearest probes via the directional search, weight with an edge-aware function of depth and normal, weight 0 if the probe lies farther than a plane tolerance from the pixel's plane. If all weights are 0, "relaxed interpolation" uses equal weights and flags the pixel in the alpha channel as a denoiser hint (Fig. 18). Pixel position is jittered before the neighbour search, and the jitter is cancelled when it leaves the pixel's plane (§2.4.1). Irradiance comes from SH: the radiance of each new probe is projected to 3 bands (L2), the cosine lobe is projected analytically, and the irradiance is their dot product (§2.4.2). SH is reprojected along with the probe so only new probes need projection.

## 4. World cache

- What it stores. Outgoing radiance at secondary path vertices, keyed by a *descriptor* = quantised world position + quantised ray direction (§2.2.1). The same structure is also fed by the primary-hit vertices **(Capsaicin source: comment "primary (same as screen probes) and secondary path vertices")**.
- Hashing and collisions. Open addressing with linear probing (Fig. 12). Hash 1 gives a bucket, hash 2 gives a "fingerprint" used for the probing inside the bucket. Both hashes are fast ones selected from Jarzynski and Olano 2020 (§2.2.1). **(Capsaicin source)**: bucket = `pcgHash(...) % num_buckets`, fingerprint = `max(1, xxHash(...))`, chained over LOD level, 3D cell coordinate, quantised direction and a 'ray shorter than cell' flag.
- LOD. Position quantisation grows with distance so the sample count per cell stays about constant; the quantisation level is part of the descriptor, giving radiance LODs (§2.2.1). **(Capsaicin source)**: `cell_size_step = max(distance(eye, position) * cell_size, min_cell_size)`; level = floor(log2(1000 * cell_size_step)); cell size = 1e-3 * 2^level. So cell size is a power of two step of (camera distance x `gi1_hash_grid_cache_cell_size`), floored at `min_cell_size`.
- Cell and tile layout (§2.2.3, Fig. 13). A two-level structure: buckets hold tiles, tiles hold cells. A tile is a fixed-size volume. Cells inside a tile are a 2D 8x8 grid obtained by projecting along the main axis of the *outgoing direction* (largest component) and dropping that coordinate. 2D tiles give higher occupancy than 3D. The linear layout includes MIP levels used for pre-filtering. Tile size 8x8 gave the best performance. Filtering across tiles is not done; the screen cache hides tile-edge artifacts.
- Light-leak fix. Quantisation can merge a dark-region hit and a bright event. The fix is to add the boolean "ray length < cell size" to the hashed descriptor (§2.2.2, Figs. 14 and 15).
- Allocation, update, eviction. Each cell has a decay counter reset on access and left to decay otherwise; at zero the entry is freed (§2.2.1). Only one hash lookup is done per vertex and the cell index is cached to memory, because atomics on a global hash map are the main bottleneck. **(Capsaicin source)** `kHashGridCache_TileDecay = 50`. Float values are accumulated as integers: `kHashGridCache_FloatQuantize = 1e3`. Buffers are ping-ponged.
- Filtering. Contributions are EMA-accumulated in MIP 0. Higher MIPs are box filtered from the level above in one pass. A query uses the finest MIP level that has "enough samples" (§2.2.3). **(Capsaicin source)** `gi1_hash_grid_cache_max_sample_count = 16`.
- Query path. A ray from a screen probe hits geometry, the descriptor is hashed, the cell index is found or allocated, and the radiance read from the filtered MIP. If the cell is new, an update is queued (§2.1.3, §2.2.1). On reprojection success the previous frame's lit pixel is used instead of re-evaluating lighting (§2.2.4).
- Multi-bounce. "Temporal radiance feedback" (§2.2.4, Fig. 16). The reprojected previous-frame colour contains direct plus indirect light. Writing it into the cache gives an unbounded number of bounces over frames, for free, and noise-free input. Limitation: only works if the reflector was visible in the previous frame (§6.1). **(Capsaicin source)** has `gi1_use_temporal_feedback = false` and `gi1_use_multibounce = true` as defaults, with `gi1_hash_grid_cache_discard_multibounce_ray_probability = 0.7` and `gi1_hash_grid_cache_max_multibounce_sample_count = 16`; the shipped code appears to have replaced feedback with a traced multibounce path (inferred; the paper does not describe it).
- Memory. The paper gives no size. **(Capsaicin source)** defaults: `num_buckets` exponent 14 (2^14 buckets), `num_tiles_per_bucket` exponent 4 (16), `tile_cell_ratio` 8, four MIP levels per tile (inferred from the `size_tile_mip0..3` fields). That would be 2^18 tiles x 85 cells = about 22M cells; I did not verify the value record size, so the total memory is an open question.

## 5. Tracing

- Screen-space tracing is not used for the main rays. Only the near-field GI in §3 is screen space.
- Rays are traced with DXR 1.1 closest-hit queries for screen-probe rays; world-cache populate rays use an any-hit kernel for shadow rays. The paper says hit-point cost is "vastly dominated by ray traversal" (§5, Table 1 footnote).
- Ray budget. 1 sample per pixel overall (§5). Capsaicin default: one ray per octahedral cell of each probe spawned that frame (max rays = max spawn count x 64), and the default spawn tile of 16x16 pixels gives about 1/4 rays per pixel **(Capsaicin source, inferred)**. The shadow rays: one per hit-cell reservoir (§2.3.1).
- Hit shading needs: hit position (to hash the cell), the ray direction, the shading normal (stored in the reservoir stream for bilateral reuse), and the albedo/material to evaluate direct lighting for the world cache. For direct-lit pixels the paper reuses previous-frame lighting where possible (§2.2.4). A miss returns environment light into the payload (§2.1.3).
- Alternative tracers. §4 says any method that returns the hit point works. They built a hybrid hardware BVH + distance-field tracer (Bartels and Harada 2022) that used BVH where the SDF error is large. It was not fast enough to be the default because the algorithm already traces few rays.
- The DXR 1.0 path (RayGen/Miss/AnyHit/ClosestHit shader tables) and a DXR 1.1 inline path both exist **(Capsaicin source, option `gi1_use_dxr10`, default false)**.

## 6. Direct lighting and light sampling

- Visibility techniques such as shadow maps are supported but do not scale to many lights, so the paper adds ray-traced stochastic light sampling (§2.3).
- World-space ReSTIR (§2.3.1, after Boisse 2021). One reservoir per hit position via RIS and WRS. Reservoirs live in a spatial hash grid keyed by the adaptively quantised hit position only. The normal is stored in a side stream for bilateral thresholding. Differences from standard ReSTIR: no visibility ray during candidate selection, one shadow ray after resampling, no spatial reuse, one temporal pass. This adds a darkening bias (Fig. 17).
- Light grid (§2.3.2). Bounds are the AABB of all hit points; the grid is uniform with square cells and a user-set maximum cells per axis; each cell holds a fixed-size list of the most important lights (min of a user value and the number of lights). Importance = relative luminance of light deposited over the cell volume, approximated by sampling radiance at the 8 cell corners and trilinear interpolation, spot cones treated as point lights. A culling test uses each light's area of effect (user max distance for point/spot, a threshold-defined distance for area lights, hemisphere above the surface for area lights), with the cell AABB approximated as a bounding sphere. Types supported: point, spot, directional, area, environment. Candidates are chosen by a random index into the cell's list.
- Number of lights: no figure is given for the tests. The paper notes the clamped importance adds bias (§6.3).
- Primary vertices: the paper does not do direct lighting at primary vertices through this; it suggests the light grid could (§6.3). Emissive surfaces and environment maps contribute through ray guiding (§6).
- **(Capsaicin source)** `gi1_use_resampling` defaults to false, so ReSTIR is off by default; `gi1_reservoir_cache_cell_size = 16`.

## 7. Denoising and filtering

| Filter | Where | What it does |
|---|---|---|
| Biased temporal hysteresis | Probe blend (§2.1.4, Alg. 3) | Variable blend factor from new vs reconstructed difference; keeps shadows, kills small fireflies. |
| Untraced-cell fill | Probe blend (§2.1.4, Fig. 7) | Fills cells without rays with the average of traced cells. |
| Probe-space sparse blur | After blend (§2.1.6, Alg. 5, Fig. 9) | Separable 7x7 over probes, depth plane and angle rejection. |
| World-cache MIP prefilter | Cell resolve (§2.2.3) | EMA at MIP 0 and box-filtered upper MIPs; read the finest MIP with enough samples. |
| Light-grid / reservoir temporal reuse | Light sampling (§2.3) | Replaces spatial reuse. |
| Irradiance denoiser | After interpolation (§2.4.3, Fig. 19) | Temporal accumulation plus a spatial filter whose radius shrinks as history length grows. Edge-aware by depth and normal. A dilated disocclusion "blur mask" selects where to blur. Relaxed-interpolation pixels with no history use the relaxed result rather than black. |
| Near-field mask denoise | §3.2 | The AO/bent-normal mask is denoised by the same spatiotemporal loop. |
| Reflection A-trous (GI-1.1) | §2.4.1 of the GI-1.1 paper | Bilateral A-trous with Gaussian-sum kernels instead of spline kernels, 5x5 footprint at upscaled resolution, 4 iterations, edge weight = product of reflection sampling PDF, scaled plane distance and a hole function. Combines denoising with 2x upscale; samples are jittered. Last iteration at full resolution. |
| Dual-source temporal accumulation (GI-1.1) | §2.4.2 | Two reprojection points (surface and hit), blended by brightness difference, clamped to neighbourhood. Samples follow a Z-curve cycle across the 2x2 upscale. Reflection mean and variance come from the A-trous filter. |
| Reflection cleanup (GI-1.1) | §2.4.3 | Marks fireflies by counting darker and brighter neighbours; replaces marked and missing samples with a 3x3 bilateral filter of unmarked neighbours. Runs before the denoiser. |

## 8. Specular / glossy

The GI-1.0 paper only sketches it (§6.2: GGX VNDF sampling from Heitz 2018; screen probes too blurry for low roughness, so trace a ray and read the environment or the world cache). The GI-1.1 paper (Eto et al. 2023) gives the working design:
- Split-sum: `L = BRDF_LUT(...) x denoised accumulated radiance` (§2.1). The radiance is accumulated over frames, then denoised.
- Per pixel one GGX VNDF direction (§2.2). If roughness is above the high threshold, fall back to diffuse. Above `low` threshold, use *specular interpolation*, i.e. read the 8x8 texel of the screen probe in that direction. Below it, use *ray-traced reflection*.
- Ray-traced reflection runs at half resolution and is upscaled 2x by the denoiser. The hit reads the previous frame's pixel colour when visible there (higher quality than the cache, and gives multi-bounce), else area light or the hash cell with jittered hit position, else environment on a miss.
- Hash cell placement for glossy hits. Cells must exist at reflection hit points or they come out black (Fig. 4). So the shading point stochastically picks diffuse or specular by BRDF weight, ray-guides the diffuse case with the screen probe, GGX-samples the specular case, traces, and allocates the cell at the hit every frame.
- Parameters used in the paper: roughness thresholds 0.2 (low) and 0.6 (high) (the text shows "0.2" and "0.6" next to the lost symbols; Capsaicin defaults `low_roughness_threshold = 0.2`, `high_roughness_threshold = 0.6` agree). **(Capsaicin source)**: half-res true, A-trous pass count 4, radius 11 (full and half), firefly mark radius 3 (half) and 2 (full), cleanup radius 2 (half) and 1 (full), denoiser mode 1 = "Atrous Ratio Estimator".

## 9. Performance

GI-1.0 paper, 1080p, 1 spp, DXR 1.1, AMD internal testing Sept 2022, Ryzen 7 5800X, 32 GB (Table 1, ms; columns RX 6900 XT then RTX 3080):

| Scene (triangles) | Raytracing | Caching and sampling | Interpolate and denoise | Total |
|---|---|---|---|---|
| Gas station (99k) | 0.235 / 0.132 | 1.032 / 1.220 | 0.665 / 0.554 | 1.932 / 1.906 |
| Flying world (267k) | 0.532 / 0.292 | 1.226 / 1.460 | 0.705 / 0.574 | 2.463 / 2.326 |
| Kitchen #1 (1.4M) | 0.336 / 0.198 | 1.633 / 2.122 | 0.765 / 0.656 | 2.734 / 2.976 |
| Kitchen #2 (9.0M) | 0.676 / 0.343 | 1.693 / 2.327 | 0.755 / 0.756 | 3.124 / 3.426 |

- The text extraction scrambled the raytracing column. I reassigned the values so every row sums to its total (each AMD and NVIDIA column adds up exactly), but the assignment is inferred, not read from the table layout. Raytracing is 0.13 to 0.68 ms, so it is the smallest part; it is lower on the RTX 3080 in every scene.
- "Raytracing" is mostly traversal; "Caching and sampling" is everything else (guiding, reconstruction, filtering, prefiltering, reprojection); "Interpolate and denoise" includes SH projection and the disocclusion mask (§5).
- Fig. 1: 3.5 ms (Kitchen) and 4.2 ms (Sponza) with direct and indirect light, 1080p, RX 6900 XT.
- UE5 comparison (Fig. 21): GI-1.0 with HW traversal 2.4 ms vs Lumen software traversal 2.6 ms. The UE5 plugin was incomplete (materials approximated).
- Takeaway: caching and sampling dominate (about 50 to 60 percent), not ray traversal. That matters for a compute-BVH port: tracing is the smaller part.
- GI-1.1 paper (Table 1, RX 7900 XTX, 1080p): ray-traced reflection 0.31 to 0.37 ms, denoise and upscale 0.18 to 0.28 ms, GI-1.0 diffuse part 0.46 to 0.96 ms. Totals about 4.1 to 5.1 ms in four scenes (the table text lost the diffuse/specular interpolation row; the 3.09 to 3.66 row looks like GI-1.0 without reflection). About 1 ms overhead for reflection (§3).
- Brixelizer GI (follow-up with no HW RT, Brixelizer slide 54): about 2 ms at 4K on RX 7900 XTX, 2.2 ms RTX 4080, 1.7 ms at 1080p on RX 7600 XT; default is half-resolution internally. Memory ~110 MB to 500 MB depending on scale and resolution (slide 55).

## 10. Limitations stated by the authors

- Temporal radiance feedback only works when the reflector was visible in the previous frame, which hurts interior scenes where bounce light dominates. Planned: path continuation in hash space (§6.1).
- Screen probes lose details smaller than the probe spacing, hence the near-field SSGI (§3). Screen-space reuse fails on thin geometry (Fig. 11), hence the LRU cache.
- Light-grid importance clamps intensity and so adds bias; an unbiased version is future work (§6.3).
- ReSTIR variant is darkening-biased (Fig. 17); hysteresis darkens too (§2.1.4). Averaging untraced cells is an approximation (§2.1.4).
- No filtering between world-cache tiles; artifacts are hidden by the screen cache (§2.2.3).
- UE5 plugin incomplete (§5). Glossy support was "in progress" (§6.2).
- GI-1.1 paper: the hash cache holds only direct lighting at the hit, so reflections of multi-bounce light come out darker than path tracing (GI-1.1 paper §3, Fig. 12). Grid artifacts appear when roughness detail is near the A-trous hole size (Fig. 9).
- Brixelizer GI (derived technique): the radiance cache is filled from screen space, so surfaces need to be seen before they get bounce light (slide 56).

## 11. Parameters

Values labelled paper are from the paper text; values labelled Capsaicin are code defaults (may differ from the paper's test builds).

| Name | Default | Meaning |
|---|---|---|
| Probe size | 8 (paper §2.1; Capsaicin `probe_size_`) | Octahedral map is 8x8 texels per probe. |
| Spawn tile | 8 x (lambda_x, lambda_y); Capsaicin 16x16 (QuarterSpp) | Temporal upscale; 1 probe per tile per frame, jittered by Halton. |
| Reprojection normal test | 0.95 (appears in Alg. 1) | Normal agreement to accept reuse (symbol lost; reading is inferred). |
| Target cell size (pixels) | 8.0 (§2.1.7) | Target screen-space probe spacing for the adaptive plane tolerance. |
| Hysteresis clamp | 0.0 to 0.95 (§2.1.4, Alg. 3) | Range of the temporal blend factor. |
| Probe filter | 7x7 separable, 5 taps per side listed in Alg. 5 | Sparse blur. |
| SH bands | 3 (§2.4.2) | Irradiance projection. |
| World tile | 8x8 cells (§2.2.3; Capsaicin `tile_cell_ratio` = 8) | Tile layout. |
| `gi1_hash_grid_cache_cell_size` | 32.0 | Scale applied to camera distance for cell size. |
| `gi1_hash_grid_cache_min_cell_size` | 0.1 | Minimum cell size. |
| `gi1_hash_grid_cache_num_buckets` | 14 (2^14) | Bucket count exponent. |
| `gi1_hash_grid_cache_num_tiles_per_bucket` | 4 (2^4 = 16) | Tiles per bucket. |
| `gi1_hash_grid_cache_max_sample_count` | 16 | Samples before a cell is trusted / EMA cap. |
| `gi1_hash_grid_cache_discard_multibounce_ray_probability` | 0.7 | Multibounce ray culling. |
| `gi1_hash_grid_cache_max_multibounce_sample_count` | 16 | Multibounce sample cap. |
| `kHashGridCache_TileDecay` | 50 | Frames before an untouched tile decays. |
| `kHashGridCache_FloatQuantize` | 1e3 | Fixed-point scale for integer atomics. |
| `gi1_reservoir_cache_cell_size` | 16.0 | Reservoir hash cell scale. |
| `gi1_use_resampling` | false | World-space ReSTIR on/off. |
| `gi1_use_temporal_feedback` | false | Paper's multi-bounce feedback. |
| `gi1_use_multibounce` | true | Traced multibounce cells. |
| `gi1_use_direct_lighting` | true | Direct light at hits. |
| `gi1_use_dxr10` | false | DXR 1.0 shader tables vs inline queries. |
| Roughness thresholds (glossy) | 0.2 low, 0.6 high | Ray-traced vs probe vs diffuse fallback. |
| A-trous iterations (glossy) | 4 | Filter passes; 5x5 footprint. |
| Light grid size / list length / area light threshold | user-set (§2.3.2) | No default given in the paper. |

## 12. Hardware RT dependencies

What needs ray tracing (paper §4, §5):
1. Screen-probe rays: closest hit. Returns hit position (and enough to find/allocate a hash cell and read the normal), or a miss.
2. Shadow rays for world-cache populate: one per reservoir, any-hit style occlusion.
3. Reflection rays (GI-1.1): closest hit, half resolution.
4. Multibounce cell rays **(Capsaicin source)**.

A compute BVH must provide:
- Closest-hit with hit position, geometry normal and a way to get the shading normal or albedo (for lighting), or at least instance/triangle id plus barycentrics.
- Any-hit / occlusion query for shadow rays.
- Ray batches that are 1D indirect dispatches of up to `max_ray_count` rays (screen probe rays), and indirect dispatches driven by counters (cell lists). No ray-gen shader table is needed because the Capsaicin default path already uses inline queries.
- Opaque alpha-test handling is optional (`gi1_disable_alpha_testing` exists).
- Dynamic geometry: a refit/rebuild fast enough per frame; the paper has no BVH-update requirement of its own.
- Not needed: ray sorting, shader tables, recursion. The paper says the tracer only has to "return the hit point of a ray" (§4).
- Distance-field tracing is an acceptable partial substitute (paper §4; Brixelizer slides 9 and 14 show a full no-HW-RT variant built on the same two-level idea, with a brick-atlas radiance cache instead of a hash grid).

Everything else (reprojection, guiding, blending, filtering, hash cache, light grid, reservoirs, SH, interpolation, denoisers) is ordinary compute (LDS, parallel scans, atomics, compaction), so it is WGSL-feasible. LDS/subgroup use: workgroup-shared 8x8 maps and prefix scans. Integer atomics on `u32`/`u32` pairs for the hash grid and float accumulation as fixed-point (Capsaicin's 1e3 scale) fit WebGPU's atomic limits (inferred).

## 13. Open questions (for the source or experiments)

1. Exact formulas of Algorithms 1 to 6 (symbols lost in extraction; read the PDF figures or the HLSL in screen_probes.hlsl / gi1.comp).
2. Probe texture formats (radiance, SH, distance), LRU cache capacity, and per-frame ray counts at each setting.
3. World cache record size and total VRAM; what happens on bucket overflow (linear probing past the bucket?). Capsaicin has `gi1_hash_grid_cache_debug_max_bucket_overflow = 64`.
4. How the paper's temporal feedback differs from Capsaicin's multibounce path, and which one the quoted timings used.
5. Whether primary-hit vertices are cached in the world cache in the paper's build (Capsaicin comments suggest so).
6. Light grid defaults (max cell count, list size, threshold) and the number of lights in the benchmark scenes.
7. Sizes of reservoirs and the reservoir hash grid; how many candidates per RIS pass.
8. Exact interpolation weights (edge-aware function) and denoiser filter radius schedule.
9. How the SSGI mask and the bent-cone combination are tuned (horizon steps, radius).
10. How rays are distributed to probes when ray guiding yields cells with 0 or many rays (the CDF pick is per ray; is the budget always 64 per probe?).
11. Material/BRDF data the hit shader needs in Capsaicin (full glTF material or simplified).
12. Exact per-pass times (Table 1 only gives three groups; Capsaicin has per-pass timers worth running).
