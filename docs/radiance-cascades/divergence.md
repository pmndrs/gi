# Where our Split RC differs from Chandra, and why

Written 2026-11-27 by the lead (Mac). The reference is the authors' own implementation, Chandra, at [`9af00e8`](https://github.com/entropylost/chandra/tree/9af00e8de135a4126cb137d1ec62b451bd6b8e37), the same build our [Sponza comparison](chandra.md) runs. It needs an NVIDIA GPU, so it can't run on the Mac as it stands. The second reference is the Three.js port the gallery ships as `?backend=three` ([`examples/cascades/11-radiance-cascades/RealtimeRadianceCascades.js`](../../examples/cascades/11-radiance-cascades/RealtimeRadianceCascades.js), from zalo's three.js PR at `5425061`). Behind both is the Split RC paper (Freeman, Sannikov and Pitt-Francis, [rev. 2](https://arxiv.org/abs/2607.20384v2)). I didn't re-read the paper for this file. Our [method notes](../../research/radiance-cascades/foundations/method.md) summarise it.

**The rule we work under:** Chandra is a guide, not a target. Our solver is judged by browser frame time, memory and how the image looks in motion at the display size (DECISIONS 2026-11-19). A difference from Chandra is fine when it's measured and wins on those terms. This file shows which differences have been measured and which are still untested.

**How this was built:** by reading Chandra's source next to ours, plus our own records (PROGRESS, DECISIONS, NOTES). No new measurements were taken. Numbers cite where they were recorded. Impacts without numbers are the lead's estimate and say so.

**Paths:** `CH/` = Chandra's `src/` at `9af00e8`, `RC/` = `packages/core/src/techniques/radiance-cascades/`, `GAL/` = `examples/cascades/11-radiance-cascades/`.

**Columns:** the same as the [DDGI file](../divergence.md). **Quality** and **Speed** say *better*, *worse* or *same* for the picture and for GPU time or memory. **Effort** runs from XS (an hour or two) to L (a milestone-sized piece). **WebGPU** says whether Chandra's way can be done in WebGPU today.

## 1. Side by side

The last column says why ours differs. Where there is no recorded reason, it says so. The whole native solver came in one commit on 2026-11-19 (`065fd03`) as a bounded-volume prototype. Its PROGRESS entry records what it achieved (the analytic checks, 27 MB of probes against the Three.js arm's 50 MB, a quicker shutter), not why it is laid out this way. A reason marked *inferred* is the lead's reading of the code, not a record.

| | Chandra | Three.js port (gallery settings) | Ours, native | Why ours is different |
| --- | --- | --- | --- | --- |
| Probe placement | Sparse hash table, only cells that visible pixels land in (`CH/cache.rs`) | Sparse hash tables rebuilt each frame, up to 4,096 probes a pool | Dense grid over a bounded box the host sets (`RC/native.ts`) | **No recorded reason.** *Inferred:* a fixed grid is the simplest layout in raw WGSL, with fixed addresses, no hash table, no allocation pass and fixed memory. It was never compared with sparse storage (RCD-01) |
| Probe size | World grid, 6 scene units at the base, doubling each time the distance to the camera doubles past 1,200 units (`CH/cache.rs:297`, `:400`) | 0.10 m out to 1.5 m from the camera, doubling with each further level (10 levels) | 0.5 m everywhere in the gallery (10 × 6.1 × 23 m box, 20 × 12 × 46 cells) | **Set by the dense grid's memory, not by image quality.** 0.5 m fills the gallery box with 26 MiB, and each halving costs 8×. SRC.2 later found cell size is the main source of error against Chandra (RCD-01) |
| Levels | 4 (`CH/algorithm/rc.rs:46`) | 4 cascades in each level of detail | 3 | **No recorded reason, and it breaks RC's scaling rule at the last level.** Memory isn't the reason: a fourth level would add about 3 MB (RCD-07) |
| Directions | Equal-area grid on the sphere: 32, 128, 512, 2,048 (`CH/cache.rs:259`) | The same equal-area grid, 32 at the finest | Squares on the six faces of a cube, like a cube map: 2 × 2 a face (24), 4 × 4 (96), 8 × 8 (384) (`RC/native.ts:59`) | **A fair engineering choice, though not recorded.** The bin is the largest axis plus two coordinates, and each bin splits into exactly 4 at the next level. The bins aren't equal in solid angle, so the merge weights children by solid angle (`RC/shaders.ts:186`) |
| Interval lengths | Scale with probe size: about 1.6 × spacing, then ×4 a level (`CH/main.rs:377`) | Follow probe size | Fixed in metres: 0.6, 2.4, then to 28 m (gallery) | **Follows from the fixed probe size.** RC's rule is that each level doubles probe spacing, has 4× the directions and covers a 4× longer interval. Ours keeps it from the first level to the second (0.5 m cells to 0.6 m, 1 m cells to 2.4 m). The third level then runs to 28 m where the rule stops at 9.6 m. Chandra keeps the rule at every distance because its probes and intervals grow together |
| Rays | 1 a pixel (237,506 a frame at 640 × 360, PROGRESS 2026-11-26) | 1 a 2 × 2 block | 1 a 2 × 2 block (57,600 a frame at 640 × 360) | **Measured** (RCK-01) |
| History | Sum of rays and of weights, both × 0.97 each frame, but only once a direction has more than 2 rays' worth (`CH/algorithm/rc.rs:299`) | Same scheme at 0.94, held below 2 rays | Running average, keeps 0.94 each time a direction gets rays (`RC/shaders.ts:139`) | **Measured** (RCK-02, DECISIONS 2026-11-27) |
| Unused probes | Freed after 1 frame unseen, 10 for the second bounce (`CH/algorithm/rc.rs:51`) | Kept only while pixels land in them (tables rebuilt each frame) | Direction forgotten after 120 frames unseen (`RC/shaders.ts:145`) | **Measured as harmless** (RCK-03). A dense grid has no memory to free |
| Merge (far into near) | Blends the 8 surrounding parent probes, each weighted by how many rays it has seen (`CH/algorithm/rc.rs:260`) | Probes weighted by rays seen | Reads the 1 parent cell that contains the child (`RC/shaders.ts:176`) | **No recorded reason.** *Inferred:* 1 read instead of 8. Not compared (RCD-04) |
| Gather (lighting a pixel) | Per-probe irradiance map looked up by the normal, 8 probes blended, each weighted by rays seen, two LODs blended (`CH/algorithm/splat.rs:187`) | Per-probe octahedral irradiance maps, probes weighted by rays seen | Six-direction ambient cube per cell, 8 cells blended, every cell with data at full weight (`RC/shaders.ts:201`, `:222`) | **No recorded reason.** *Inferred:* one projection per cell and six numbers to read, cheaper than a map. Neither the cube nor the unweighted blend was compared (RCD-02, RCD-05) |
| Bounces | One by default; a second is a runtime toggle (`CH/main.rs:320`) | Second bounce on, from secondary probes | One: emission and shadowed direct light at the hit | **Matches Chandra's default.** The Three.js arm's second bounce wasn't ported (RCD-03) |
| Tracer | OptiX hardware ray tracing, static scene | three-mesh-bvh, static, refit for the shutter | Compute-shader BVH (`@pmndrs/raytracer`), moving instances | **Forced:** WebGPU has no hardware ray tracing. Moving geometry was a goal (SPEC) |

**In short:** three rows were measured (rays, history, unused probes), two are forced or match Chandra (tracer, bounces), and one is a fair choice (cube-face directions). Six have no recorded reason: placement, probe size, three levels, fixed intervals (which follow from the fixed probe size), the one-parent merge and the gather. Those six are what makes the native solver a simplified RC. It keeps RC's interval composition and merge, but not the distance-sized probes and enough levels that let RC scale.

Already the same: the interval composition (near radiance plus near transmittance times far), rays over the hemisphere around the normal from a low-discrepancy sequence, a ray counting as "seen through" for every interval before its hit and adding nothing beyond it, and misses landing in the last interval.

## 2. Worth trying, ranked

The things Chandra does that we don't, ranked by likely gain for the effort. Each one is an experiment first: build it behind a setting, run the checks named, then record the call in DECISIONS.

| Rank | ID | What | Quality | Speed | Effort | WebGPU | Checks that decide it |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | RCD-01 | **Smaller probes near the camera.** Sparse storage with probes sized by distance, as Chandra and the Three.js port do. Our 0.5 m cells straddle surfaces and mix light from both sides of them | Better. The Sponza error against Chandra follows cell size: 13.4 / 19.8 / 39.7% relative RMS over 40-pixel blocks at 0.25 / 0.5 / 1.0-unit cells. The emitter's own light reaches 0.86 / 0.71 / 0.62 of Chandra's (PROGRESS 2026-11-26). That the lamp alcove's flicker also drops is an estimate | Dense memory grows 8× each time the cell size halves: about 26 MiB now, about 206 MiB at 0.25 m in the gallery. Sparse storage avoids that but adds an allocation pass | L | Yes, core. The radiance cache already has a GPU hash grid | Sponza against Chandra (`rc-chandra`), gallery flicker at `camera=receiver&cue=amber`, `rc-door-response`, probe bytes, the polished-rail timing pairs |
| 2 | RCD-07 | **A fourth level**, so the intervals keep RC's ×4 rule: 0.6, 2.4, 9.6 m, then to the end (gallery 28 m) | Far light gets 1,536 directions from 2 m cells over 9.6–28 m, instead of 384 directions over 2.4–28 m. The gain is unmeasured (estimate) | About 55,000 more bins, about 3 MB, and one more merge pass | S: the deposit loop, merge passes, layout and uniforms assume three levels | Yes | `rc-native-core`, `rc-chandra`, gallery flicker, the polished-rail timing pairs |
| 3 | RCD-02 | **Weight each probe by the rays it has seen** when blending probes in the gather and the merge, instead of full weight for any probe with data | Probably better near the lamp, where a cell seen by a few rays pulls pixels around (estimate) | Same: one extra number a probe | S–M | Yes | Gallery flicker, the door's T90, `rc-native-core` |
| 4 | RCD-03 | **A second bounce.** The gallery's Three.js arm has one; Chandra can toggle one | Brighter, closer to the reference. The gallery reads 57 against Three.js's 85 in mean luma (PROGRESS 2026-11-27). At the door receiver native gives 0.135 and Three.js 0.282. A CPU integral of the single bounce gives 0.170 (SPEC), so a bounce isn't the whole gap: native also sits about 21% under its own one-bounce answer | Worse: a second trace or a cache read at each hit | M | Yes | `rc-door-response` against the one-bounce integral, gallery luma, the timing pairs |
| 5 | RCD-04 | **Blend the 8 surrounding parents** in the merge, not the one containing parent | Smoother far light. Seams at cell faces were small on Sponza (ratio 0.95–1.03, PROGRESS 2026-11-26), so the gain may be modest (estimate) | Worse: 8 parent reads for each partly open interval instead of 1 | S | Yes | `rc-chandra` ratio maps, `rc-native-core` |
| 6 | RCD-05 | **A per-probe irradiance map** instead of the six-direction ambient cube | Better for normals between the axes (estimate) | Some more memory and one more texture read a pixel | M | Yes | `rc-chandra` by normal, `rc-native-core` constant-radiance cases |
| 7 | RCD-06 | **Interval lengths that follow probe size** (comes with RCD-01) | Small on its own. Sponza: a first interval of 0.3 gave 25.1%, 1.2 gave 18.9%, against 19.8% at 0.6 (PROGRESS 2026-11-26) | Same | XS once RCD-01 exists | Yes | As RCD-01 |

## 3. Tried, and we keep ours

Each was measured. Revisit only if a later check shows otherwise.

| ID | Chandra | Ours | Why, with the evidence |
| --- | --- | --- | --- |
| RCK-01 | 1 ray a pixel | 1 ray a 2 × 2 block | One ray a pixel cost 4×: 15.7 against 3.8 ms of cascade compute on the Three.js arm at 960 × 540, RTX 3060 (PROGRESS 2026-11-19). It didn't fix what matters: on Sponza the error stayed at 19.8% (PROGRESS 2026-11-26), and in the gallery it lowered the average flicker but not the worst pixels (PROGRESS 2026-11-27). |
| RCK-02 | Ray-weighted sums, held until a direction has 2 rays' worth, × 0.97 a frame | Running average at 0.94 | Porting the ray weighting measured the same as the plain average at 0.94 and needed 7 MB more (PROGRESS 2026-11-27). 0.94 against our old 0.8 cut the flicker by three quarters, and the shutter still settles in 41 frames (DECISIONS 2026-11-27). |
| RCK-03 | Probe freed after 1 unseen frame | Direction forgotten after 120 | A dense grid frees no memory either way. Turning forgetting off changed nothing in the gallery's flicker (PROGRESS 2026-11-27). Revisit with RCD-01, where freeing probes saves memory. |

## 4. Forced by WebGPU or the browser

| What Chandra does | What we do instead | What it costs | Could it change? |
| --- | --- | --- | --- |
| Hardware ray tracing through OptiX | A compute-shader BVH walk (`@pmndrs/raytracer`), which also takes moving instances | Trace speed. The shutter refits each frame | Only if WebGPU gains ray tracing |
| Float atomic adds for deposits | Sums in 32-bit fixed point (`RC/shaders.ts:34`). Each ray's radiance is capped at 64 a channel and a direction takes at most 4,096 rays a frame, so the sum can't overflow | Light brighter than 64 is clipped before it is averaged. Precision is 1/2048 a ray | Only if WGSL gains float atomics |
| Warp shuffles in the merge and gather | Plain per-thread code | Speed only | Yes, behind the optional `subgroups` feature |

## 5. Things we have that Chandra doesn't

- **Moving geometry.** The tracer updates moving instances, so the shutter and a moving panel darken and recover (`rc-door-response`).
- **A raw WebGPU API with no engine.** The caller owns the device, encoder and submission (SPEC).
- **Temporal upscaling.** The gallery renders GI at a lower resolution and the upscaler restores the display size, checked by `rc-polished-quality`.

## 6. Doc errors found while writing this

- **PROGRESS 2026-11-26 (SRC.2) says Chandra puts its probes on 6-pixel screen tiles.** Chandra's `quantize` (`CH/cache.rs:357`) is a world-space grid: `floor(position / spacing)`, spacing 6 scene units at the base. At the Sponza capture's scale (0.0038, `tools/gpu/pages/rc-chandra/main.ts:25`) that is 0.023 of our units, against our 0.5-unit cells, about 22× smaller. Near the capture camera a 6-unit cell spans about two pixels (estimate from the camera's field of view). Left in PROGRESS as written, since PROGRESS is a log. The finding that error follows probe size stands either way.
- **In chat on 2026-11-27 I said Chandra's probes get many rays each frame.** By the estimate above, a base Chandra probe gets about 4 rays a frame over 32 directions, fewer per direction than ours. Chandra's steadiness, as far as the code shows, comes from many small probes, holding rarely seen directions, and the ray-weighted blending in RCD-02, not from rays per direction. Nobody has measured Chandra's flicker.

## 7. Sources

- Chandra settings and history: `CH/algorithm/rc.rs:15-60` (defaults), `:158` (quality weight), `:236-330` (merge and decay)
- Probe grid and directions: `CH/cache.rs:259` (directions), `:297-330` (LOD), `:357-410` (grid)
- Rays: `CH/algorithm/trace.rs:150` (one a pixel), `:258` (deposit into the hit's level)
- Gather: `CH/algorithm/splat.rs:187` (`shade_optimized_32`)
- Interval length and bounces: `CH/main.rs:316-392`
- Sponza settings: `scenes/sponza.json` (`base_grid_size` 6, `base_probes` 131,072)
- Ours: `RC/native.ts`, `RC/shaders.ts`, `GAL/native-adapter.ts`, `GAL/main.ts:118-123` (the Three.js arm's settings)
- Our records: PROGRESS entries of 2026-11-19, 2026-11-26 and 2026-11-27, DECISIONS 2026-11-19 and 2026-11-27, [SPEC](SPEC.md), [Chandra comparison](chandra.md)
