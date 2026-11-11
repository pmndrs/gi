# radiance-cache: the speed plan

Written 2026-11-11 by the lead. This covers where the radiance cache's time goes, every speed idea we have (the "levers"), how Lumen and id Tech 8 stay fast without hardware ray tracing, and the order to try things in. The divergences from Capsaicin themselves are in [radiance-cache-divergences.md](radiance-cache-divergences.md).

**The rule stays the same:** the faithful port is the baseline. Every lever is a switch beside the faithful path. It is measured against that path on the same scene and camera, and adopted with a register row and a reason. Nothing replaces the faithful path.

## 1. Where the time goes

### What we can measure today: a gap in the timer

The technique already labels its passes for GPU timing. Capsaicin's section names are used for the world-cache passes (`radiance-cache/PopulateScreenProbesCells`, `UpdateRadianceCache`, `ResolveCells` and so on), so a per-pass table can line up with native Capsaicin's CSV pass for pass. The results land in `GIStats.passes` when the platform's `gpuTiming` option is on, and the viewer turns it on.

**But on 2026-11-11 every radiance-cache pass read 0 ms.** This was a snapshot on the Apple M5 Pro, viewer path, Sponza and Cornell, 300 warm-up frames then 120 measured, through the bench lock. The cause is in `packages/ddgi-core/src/platform/gi-platform.ts`. `encode()` resolves the frame's timestamps at its end (around line 376). Every radiance-cache pass runs later, in `encodeScreen()` (line 387), so their timestamps are written after the resolve. DDGI is timed correctly because its passes run inside `encode()`. Only `platform/top-level refit` gave a real number.

*Fixed in RC5.1 (2026-11-11):* the platform resolves again after the screen stage, every Capsaicin section is its own timed pass, and `radiance-cache-timing` writes the first M5 per-pass table to `bench/results/radiance-cache-m5-passes.csv`.

So **fixing this is the first speed task** (lever 17). It is small: resolve the timer after the screen stage when a technique has one, and add a test that a screen-stage pass reads more than 0. Next to it goes a `radiance-cache-timing` page shaped like `ddgi-timing`: per-pass medians over a warmed run, on Sponza, the two Bistros and two-room, written to `bench/results/`. Phase 5 needs that page anyway.

What the snapshot could give is wall time per frame, including the wait for the GPU, on the M5 at 1280 × 720. Sponza was 33.2 ms and Cornell 16.55 ms (`bench/results/radiance-cache-m5-passes.csv`). Both look tied to the display rate (one and two 60 Hz frames), so they only say that Sponza takes between about 17 and 33 ms. They are not GPU times.

### The tracing prediction (RC1.5, RTX 3060 Laptop, pinned 1,200 MHz, 3 VALID runs)

RC1.5 traced ray sets shaped like Capsaicin's at 720p through our compute BVH. The frame's tracing alone:

| Ray set (Sponza) | Rays per frame | ms | Mrays/s |
| --- | --- | --- | --- |
| Screen-probe rays | 230,400 | 6.92 | 33.3 |
| Multi-bounce rays (GI-1.2) | 69,120 | 2.73 | 25.3 |
| Shadow rays of probe hits | 103,662 | 1.85 | 56.1 |
| Shadow rays of bounce hits | 37,263 | 0.87 | 43.0 |
| **Tracing total** | | **12.36** | |

Bistro interior comes to 21.91 ms and Bistro exterior to 23.82 ms. That is 2.6 to 3.2 times native Capsaicin's whole "probe tracing and cell population" group. It is a floor: tracing only, with no shading, filtering or denoising.

One finding matters a lot. **Probe rays leave from surfaces, and they trace at 17 to 35 Mrays/s. Rays from a grid of points in the air trace at about 87.** Whatever makes surface rays slow is the biggest gap we have measured (lever 20). *RC5.4 (M5) measured why:* overlapping instance boxes at the top level, which one merged tree removes (Sponza's surface rays 1.86× faster, Bistro interior's 1.24×), and the rays themselves, which start in the cluttered part of the view and go farther; the grid's rate is not reachable for them (lever 20's row).

### Native Capsaicin GI-1.2 (RTX 3060 Laptop, 1280 × 720, frames 101-300, hardware RT, clock not locked)

From the research notes (`ddgi-research/research/gi-1.0/native-capsaicin.md`):

| Group | Sponza | Bistro interior | Bistro exterior | Two-room |
| --- | --- | --- | --- | --- |
| Probe tracing and cell population | 4.75 | 7.81 | 7.56 | 1.31 |
| Cache update and resolve | 0.74 | 0.67 | 0.80 | 0.65 |
| Screen probes | 1.38 | 1.18 | 1.31 | 1.51 |
| Denoise and resolve | 1.08 | 0.71 | 0.95 | 0.56 |
| **Diffuse subtotal** | **7.95** | **10.38** | **10.62** | **4.03** |
| SSGI | 0.36 | 0.31 | 0.32 | 0.43 |

Native is already 8 to 11 ms for the diffuse part on this laptop. Even native GI-1.2 doesn't fit a 60 Hz frame next to the rest of a renderer at these settings. Matching native isn't enough. We need levers that cut work, not only levers that make the same work faster.

## 2. The lever catalogue

Gains are for the RTX 3060 on Sponza at 720p unless stated. "Predicted" means arithmetic from RC1.5's measured ray costs. "Guess" means nobody has measured it. Levers 1-16 keep their numbers from the design doc. New ones start at 17.

| # | Lever | Source | Pass it targets | Expected gain | Image risk | Effort | Needs | Start today on the Mac? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Shade cells, not hits: rays store a hit record, a fixed budget of cells is lit per frame by priority | id Tech 8 (S17-S20) | Population, shadow rays, `UpdateRadianceCache` | About 2-2.5 ms predicted (the 2.7 ms of per-hit shadow rays becomes a fixed budget, e.g. 20k cells ≈ 0.4 ms), plus a cost that stops growing with ray count | Medium: lighting lags by N frames | L | Timing on Windows for the verdict | Yes, build |
| 2 | DDGI supplies the second bounce instead of GI-1.2's multi-bounce rays | id Tech 8 (S20, previous-frame volumes) | Multi-bounce rays and their shadow rays | Up to 3.6 ms predicted, minus DDGI's own update (about 2 ms budget) | Medium: DDGI's resolution, one frame of lag; it also removes Capsaicin's 3.3× bias | M | `BounceSource` wiring | Yes |
| 3 | Half- and quarter-resolution tiers for the per-pixel passes (interpolation, SSGI, denoiser), or `@pmndrs/upscaler` | id Tech 8 (S22), Lumen (`IntegrateDownsampleFactor`) | Screen probes, SSGI, denoiser | Guess: up to 3/4 of the per-pixel passes at half resolution; native's equivalent groups are about 2.8 ms | Medium: softer normals | M | Timing on Windows | Yes |
| 4 | Denoise in workgroup memory with f16 | id Tech 8 (S22) | `ReprojectGI`, `FilterGI` | Guess: small, native's whole denoise group is about 1 ms | Low | M | Timing | Yes |
| 5 | Compact formats: shared-exponent history, 16-byte hit records, SH in three `rgba16float` | id Tech 8 (S17, S22, S25) | Memory traffic everywhere | Guess: small to medium | Low if packed correctly | M | Timing | Yes |
| 6 | Interleaved refresh of distant cache regions | id Tech 8 (S15) | `UpdateRadianceCache`, tile mips | Guess, in proportion to the share skipped | Medium: slow response far away | M | Timing over a full cycle | Yes |
| 7 | Short probe rays that end in the world cache | Lumen | Probe rays | Unknown: our tracer gains little from short rays (NOTES), measure first | Medium | M | A quick ray-length test first | Yes, the test |
| 8 | Scratch memory per updated tile instead of per cell | ours | Memory | Memory only | None | S | Nothing | Yes |
| 9 | Brixelizer's SDF tracing for long rays | AMD Brixelizer (ported elsewhere) | Probe and bounce rays | Guess: large for long rays | Medium | XL | The other workstream | No |
| 10 | Priority order for passes, e.g. key probes first | ours, id Tech 8 | Screen probes | Guess | Low | M | Timing | Yes |
| 11 | SH projection with each cell's mean basis | ours | `ProjectScreenProbes` | **Quality lever.** Constant sky 2.1% bright becomes within 0.2% | None, fixes a bias | S | Nothing | Yes |
| 12 | An unbiased second bounce | ours | `UpdateMultibounceCells` | **Quality lever.** Removes Capsaicin's 3.3× second-bounce excess | Changes the image (towards correct) | S | Nothing | Yes |
| 13 | An empty-cell backup that doesn't drift with history | ours | `BlendScreenProbes` | **Quality lever.** Constant sky 0.69 πL becomes about 1.0 πL | Changes the image (towards correct) | S-M | Nothing | Yes |
| 14 | SSGI samples on texel centres, at least a pixel away | GTAO practice | SSGI | **Quality lever,** 2-4 points of E on an open floor | Low | S | Nothing | Yes |
| 15 | SSGI's near-field light in irradiance units | ours | SSGI | **Quality lever** | Low | S | A path-traced corner reference | Yes |
| 16 | More SSGI slices (`ssgi_slice_count_`; ours `ssgiSliceCount`) | Capsaicin setting | SSGI | **Quality lever,** costs time: most of the flicker before the denoiser; E 0.735 → 0.969 πL at 16 slices | None, costs about 1 SSGI pass per slice (guess) | S, the setting exists | Timing for the cost | Yes |
| 17 | **Time the screen stage:** resolve timestamps after `encodeScreen`, plus a `radiance-cache-timing` page | ours (the gap found 2026-11-11) | All passes (tooling) | None by itself. It unblocks every measurement and phase 5 | None | S | Nothing | **Yes, first** |
| 18 | Multi-bounce off (`useMultibounce: false`, Capsaicin's `gi1_use_multibounce`) | Capsaicin setting (D42) | Multi-bounce rays and their passes | 3.6 ms predicted (RC1.5's bounce and bounce-shadow sets) | Medium: loses the second bounce, which reads 3.3× high anyway | S, already built | Timing | Yes, measure only |
| 19 | `SixteenthSpp` spawn rate: one probe per 32 × 32 tile instead of 16 × 16 | Capsaicin setting (`ScreenProbes::SamplingMode`, gi1.h:174-198; shaders read the tile size as data) | Probe rays, their shadow rays, sample and blend | About 6.5 ms predicted: probe rays and their shadows cut to a quarter (8.8 → 2.2 ms) | Medium: more reliance on history and reprojection, more lag | S-M (layout sizes, the spawn tile) | Timing; native Capsaicin can be rebuilt in the same mode for a fair comparison | Yes |
| 20 | **Fix slow surface rays:** probe rays trace at 17-35 Mrays/s against about 87 for grid rays. *Measured in RC5.4 (M5):* not the origin offset, self-hits, directions, ray order or divergence (each ≤ 12% of the walk's steps; tie-breaks for boxes holding the origin buy nothing). Two causes: overlapping instance boxes at the top level (a Sponza surface ray enters 9.8 instances, 8.1 with no hit), and the rays themselves (they start in the cluttered part of the view and go farther). Fix for the BVH lane: merge never-moving, used-once instances into one bottom-level tree (a), or partial re-braiding (b), which keeps the instance-entry cost and so gains less | ours (RC1.5 finding, RC5.4) | Every probe and bounce ray, and their shadow rays | With (a), if the M5's paired ratios hold on the RTX: Sponza's RC1.5 tracing 12.36 → about 7.1 ms (closest 1.86×, any-hit 1.39×), Bistro interior 21.9 → about 18.2 ms (1.24×, 1.13×). The grid rate (the old "about 6 ms") is out of reach: the rest of the gap is the rays (levers 7, 19, 21, 24) | None: same rays, same hits | M-L (fix, BVH lane) | The RTX run of `trace-bench-surface-*`; the BVH lane's build | Investigated; fix not started |
| 21 | **Screen trace first:** march the depth buffer for the first stretch of each probe ray, and send only rays that leave the screen or pass behind geometry to the BVH | Lumen (screen traces first) | Probe rays | Guess: up to a third or half of probe rays on interior views | Medium: misses thin objects and anything hidden behind them; Lumen pays the same | M | Timing | Yes |
| 22 | **One frame budget for the whole technique:** the platform's work budget (DDGI's 2 ms controller) sets the spawn rate, cell updates and refresh share | id Tech 8 (fixed budgets), Lumen (`NumProbesToTraceBudget`), ours (`work-budget.ts`) | Everything | Holds a ceiling by construction; the image pays when over budget | Medium: lag and softness under load | M-L | Levers 1, 6 and 19 as its dials | Design today |
| 23 | Right-size the hash cache: today 1 GiB (Capsaicin's default); peak live tiles 47.7k on Sponza and 54.8k on Bistro exterior of 262,144 | id Tech 8 (14 MB cache, S17) | Memory, clears and purge | Memory: roughly 4× smaller at the measured peak. Speed: guess, small | None while nothing overflows (the overflow counter shows it) | S | Nothing | Yes (memory lever) |
| 24 | Fewer rays per probe: 32 or 16 instead of 64 | id Tech 8 (64/32/16 by platform, S15), Lumen (`TracingOctahedronResolution`) | Probe rays | About 4.4 / 6.6 ms predicted at 32 / 16 rays (the probe set and its shadows, 8.8 ms, halved or quartered) | High: changes the 8 × 8 hemi-octahedral layout every pass reads | L | Timing | Later |
| 25 | **Spread heavy passes across frames:** the LRU reorder, the tile mips and purge every other frame; DDGI-style round-robin for whatever doesn't need every frame | id Tech 8 (one cascade per frame), Lumen (amortised surface-cache and radiance-cache updates) | Bookkeeping passes | Guess: small to medium, unknown until lever 17 shows their share | Low to medium | S-M per pass | Lever 17's table first | After 17 |
| 26 | Async compute | id Tech 8 (20-25% of GI time on consoles), Lumen | n/a | **Not available:** WebGPU has one queue (D12). Keeping independent passes next to each other is the only thing to try, and nothing guarantees overlap | n/a | n/a | A WebGPU feature | No |

Machine-readable copy for the explainer page: the lead's scratchpad `levers.json`.

## 3. How Lumen and id Tech 8 stay fast without (or before) hardware ray tracing

Both engines rest on one idea: **trace few, cheap rays; shade a bounded amount; reuse everything over space and time.** Capsaicin's defaults do the opposite of the first part. They trace about 230k probe rays at 720p and light every hit with its own shadow ray. Hardware RT pays for that. Our compute BVH can't.

**Lumen (UE5, software ray tracing).** Sources: Wright, Narkowicz, Kelly, SIGGRAPH 2022 Advances, <https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-Lumen-Wright%20et%20al.pdf>; UE docs, <https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-technical-details-in-unreal-engine> and <https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-performance-guide-for-unreal-engine>.

- **A trace hierarchy, cheapest first.** Screen trace first, then mesh distance fields for the first 2 m, then the global distance field for the rest (the default is global-only, "the fastest traces"), then the sky. For us: lever 21 (screen trace first, reusing D3/D17's `FrameScreen`), lever 9 (Brixelizer's SDF for long rays) and lever 7 (short rays). Our compute BVH (D1) is the middle tier.
- **A surface cache, lit on a budget.** A ray hit reads lighting from cards instead of shading the material. Card lighting updates are "amortized over multiple frames" and set by update factors (`r.LumenScene.DirectLighting.UpdateFactor`, `r.LumenScene.Radiosity.UpdateFactor`). For us, Capsaicin's hash grid is already the surface cache. Lever 1 makes it lit on a budget instead of per hit. A card-style cache would also undo D2 (texture-average albedo at hits), because the material is captured by rasterising, not looked up at every hit.
- **Screen probes, sparse and guided.** About one probe per 16 × 16 pixels, 8 × 8 octahedral rays, rays aimed by last frame's lighting, and filtering in probe space (3 × 3 probes ≈ a 48 × 48 pixel filter). Capsaicin has the same pieces: QuarterSpp spawn, the luminance CDF and `FilterScreenProbes`. Lumen's dials are probe spacing (`r.Lumen.ScreenProbeGather.DownsampleFactor`) and rays per probe (`TracingOctahedronResolution`), which are levers 19 and 24.
- **A world radiance cache with a trace budget.** Distant light (beyond about 2 m) comes from a clipmap of probes. A GPU priority queue picks a fixed number to re-trace per frame (`r.Lumen.ScreenProbeGather.RadianceCache.NumProbesToTraceBudget`); set it too low and lighting pops during fast camera moves. For us: levers 6, 22 and 25. This touches D30, the world-cache interface, which today takes every traced ray.
- **Budgets:** Lumen targets 4 ms (60 fps) and 8 ms (30 fps) at 1080p on consoles, for GI and reflections together.

**id Tech 8 (Sousa, SIGGRAPH 2025; notes in `ddgi-research/research/gi-1.0/idtech8-performance.md`).** Hardware RT is used, but designed for weak hardware RT: "Can't trace many rays (low end hw can be orders of magnitude slower)".

- **Visibility-only rays.** A ray records a hit and marks a cache entry. Shading happens later, in one pass over the marked entries. This is lever 1. It replaces most per-hit shadow rays (D4's light draw per hit) with a fixed count.
- **A fixed shading budget,** about 20k cache entries a frame, reused for N frames. Levers 1 and 22.
- **Interleaving.** One cascade and one local volume are updated per frame. id warns that benches must average over a full cycle. Levers 6 and 25.
- **Previous-frame volumes for extra bounces** instead of more rays. Lever 2, with DDGI as the volume. This is extension 1 in the design doc.
- **Reduced-resolution final gather** (half or quarter per axis), denoised at that resolution in f16 shared memory, then upscaled with 4 taps. Levers 3 and 4.
- **Small caches.** 14 MB of hash cache against Capsaicin's 1 GiB. Lever 23.
- **Async compute** saves them 20-25%. We don't have it (D12, lever 26).

**What neither engine does:** trace hundreds of thousands of shaded rays a frame in software. Capsaicin's defaults only make sense with hardware RT. So the faithful port is a correctness reference, not a shipping configuration. The shipping configuration is the faithful port plus the levers that pass their image checks.

## 4. Proposed order

### Today on the Mac (no Windows timing)

1. **Lever 17:** resolve timestamps after the screen stage, plus the `radiance-cache-timing` page. Review it, then take a first per-pass table on the M5. The M5 shares are a hint, not the target: Apple's GPU differs from the RTX in tracing especially.
2. **The switches Capsaicin already has:** 18 (multi-bounce off) needs only a timing-page column. 19 (`SixteenthSpp`) needs the spawn tile and layout sizes as settings. 16 (SSGI slices) already exists. Correctness pages on the M5 for each.
3. **Lever 20, the investigation:** why surface-placed rays are 2.5 to 5 times slower than grid rays. Try the origin offset, ray order and per-probe ray grouping on `trace-bench-rc-sponza`. The M5 can show the ratio, though the RTX verdict comes tomorrow. *Done in RC5.4:* the fix goes to the BVH lane (lever 20's row).
4. **Design levers 1 and 22** (shade cells on a budget; one frame budget), and the phase 5 TASKS rows.
5. The quality levers 11-15 can go any time, because they need no timing. They come after the above, since today's bottleneck is speed.

### Tomorrow on the Windows laptop (RTX 3060 timing, native Capsaicin; this is phase 5)

1. Re-run all twelve RC correctness pages on the RTX 3060 and the Intel UHD. Nothing has run there since phase 2.
2. `radiance-cache-timing` on Sponza, both Bistros and two-room under the bench lock, VALID runs only, at Capsaicin's cameras. Native Capsaicin per pass at the same cameras: the 2026-11-09 runs exist, so re-run them only if the camera or settings changed. This gives the per-pass table and a cause for every pass over native (phase 5).
3. Time levers 18, 19 and 20 (if a fix exists) against the faithful path. For 19, also rebuild native Capsaicin with `kSamplingMode_SixteenthSpp`, so the comparison is like for like.
4. Never time anything while `scene_viewer.exe` is running.

### After

The order follows the phase 5 table, but the expected order is:
- Lever 1 (shade cells on a budget) and 22 (the frame budget), the structural change both engines make.
- Lever 2 (DDGI's second bounce).
- Lever 3 (resolution tiers) with the upscaler.
- Levers 21 and 7 (screen trace and short rays), once lever 20 says what a ray really costs.
- Then 4, 5, 6, 23 and 25 (formats, interleaving, memory).
- Lever 9 (Brixelizer) when the other workstream lands.
