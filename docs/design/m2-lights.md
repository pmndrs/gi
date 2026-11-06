# M2 design draft: dynamic lights

Status: **approved by Dennis 2026-11-06 and merged into main; M2's board is in [TASKS.md](../../TASKS.md).** Written 2026-11-05, revised after an Opus review the same day and again after Dennis's answers (time of day rebuilt so nothing takes minutes to build; memory sized for mid- to high-end GPUs; the Intel UHD for correctness only). Where it changes a default or a settled choice from M1, it says so; those are DECISIONS entries.

It builds on [ddgi.md](ddgi.md) (the field and update loop), [platform.md](../api/platform.md) (the contract), [three.md](../api/three.md) (the adapter) and the M1 entries in [DECISIONS](../DECISIONS.md). The M3 draft, [m3-moving.md](m3-moving.md), reuses the change-driven scheduler designed here for moved geometry.

## In one paragraph

Lights already live in a growable table, and adding one never recompiles anything. M2 makes a light change show up quickly and cheaply. Three changes do most of the work:

1. **The platform tells DDGI what changed:** the old and the new light, not just a box. A small GPU kernel turns that into "how much did this probe's lighting change", worked out from the light's own numbers, so ray noise can't fake a change.
2. **Each probe's history** (T1.12's sample count) **is cut when its lighting changes.** It is shortened in proportion to the change. A GPU scheduler puts the most changed probes near the camera at the front of the update list.
3. **Shadow rays move into their own pass in platform hit shading.** Today they sit inside the trace kernel, where Sponza loses half its update time. The new pass picks lights from a world-space grid of pre-chosen candidates, so a thousand candles cost about what a few lamps do.

Light groups (one irradiance layer per group, added up into the atlas the renderer reads) make a dimmer or a colour change across a group instant.

For time of day, the sun's colour and brightness are instant and exact, because the sun gets a layer of its own. The sky is nudged in the same frame for what each probe sees of it directly. That uses a first-order sky transfer, built from the field's own rays no slower than the field converges (about 1.2 s into Sponza's warm-up). The sun's direction, and whatever the nudge leaves, are re-traced nearest the camera first, blending rather than popping. Nothing takes minutes to build (Dennis, 2026-11-05).

Dennis answered this draft's first questions on 2026-11-05; they're listed at the end.

## Words used here

- **Probe update:** one probe's rays traced and blended into its stored values, once.
- **History length (n):** T1.12's effective sample count: how many past updates a probe's stored value is worth. The blend's weight on the old value is `n / (n + 1)` (at most 0.995), and `n` fades with the time since the probe's last update.
- **Key probes:** the one probe in 8 whose grid indices are all even; after a change they're served first and pass their change on to the probes between them (4.5).
- **Response window:** SPEC's 0.5 s, 30 frames at 60 Hz.
- **ε (tolerance)** = 5%, SPEC's "within 5% of settled". (**τ** is T1.12's 5 s memory, ddgi.md.) **K** = how many updates a changed probe is promised inside the window (8 here).
- **Owed updates:** how many of those K a probe still waits for.
- **Shadow ray:** a ray from a lit point toward a light, to see if anything blocks it.
- **Subgroup:** the group of GPU threads (8–64) that run in lockstep. "Subgroups first" means kernels use subgroup operations where they help, with one declared fallback (DECISIONS 2026-11-04).
- **Compacted list:** the work items written one after another with no gaps, so every thread has something to do.
- **Indirect dispatch:** the GPU decides how many workgroups run, from a count it wrote itself, so the CPU never reads it back.
- **Reservoir** (from ReSTIR/ReGIR): a slot that keeps one light chosen at random from a stream of candidates, each with a probability that depends on how much it would contribute, plus a weight that keeps the result unbiased.
- **CDF:** a running sum of light powers, used to pick lights in proportion to their power.
- **SH patterns and first-order transfer:** spherical harmonics (SH) are fixed patterns over the sphere, and any smooth sky is close to a weighted sum of the first 9. A probe's *first-order transfer* for a pattern is how much of that pattern the probe sees directly, through openings, with no bounces.
- **GPU-bound:** measured with the GPU kept busy and its clocks up, not at 60 Hz under light load, where the laptop GPU lowers its clocks.
- **Linearity (superposition):** twice the light gives twice the bounce, and two lights give the sum of what each gives alone.

## What M2 must deliver (SPEC)

| Target | RTX 3060 and up, Apple M5 Pro (1440p) |
| --- | --- |
| Light change visible | next frame |
| Light change within 5% of settled | ≤ 0.5 s |
| GI update cost | ≤ 2.0 ms at the default budget |
| GI sampling cost | ≤ 1.0 ms (no regression from T1.6's 0.905 ms) |
| Lights | ≥ 32; adding or removing one never recompiles |
| Adapter cost | ≤ 0.2 ms still scene, ≤ 0.5 ms one moving thing |
| Sweep | looks right to Dennis next to native doing the same motion |

M1's accuracy, leak, furnace and stability checks must still pass on still scenes.

- **Hardware** (Dennis, 2026-11-05; SPEC and DECISIONS already revised on `m1`): the targets are for the RTX 3060 and up and an Apple M5 Pro. The Intel UHD is a **correctness device only**, with no performance targets ("we arent building for the intel"). It runs the oracle, furnace, accuracy and leak checks; its timings aren't gated.
- **Timings are GPU-bound** (Dennis: "Running at 60hz seems like a fail"): with the GPU loaded and its clocks up.

## The update rates this plan works with

Everything about response time depends on how many probes are updated each frame.

**Budget v2** (accepted by Dennis, 2026-11-05; ddgi.md "Budget v2") sets that number by work, not by milliseconds:

- The platform hands DDGI a fixed number of probes a frame. It's calibrated once at load, with the clocks up, so that the batch costs about the budget (2 ms) back to back.
- **A floor:** never fewer than a full sweep in 4 s at the host's frame rate (81 probes a frame on Sponza at 60 Hz).
- Milliseconds only cut the number when frames are actually late.

**Measured with the GPU loaded** (T1.9b, Budget v2's calibration at 2 ms, 3 runs under the clock rule, medians; PROGRESS 2026-11-05). T1.9a's old numbers (Sponza 30–43) were mostly the GPU's low-power state.

| Scene | Probes | Probes per frame (RTX, 2 ms) | Runs |
| --- | --- | --- | --- |
| Cornell | 1,728 | **1,574** | 1,563–1,821 |
| Two-room | 1,071 | **707** (a full sweep every 1.5 frames) | 706–715 |
| Furnished room | 4,998 | **not measured** | |
| Sponza | 19,425 | **268** (a full sweep in 1.2 s; run 3's fit: 4.6 µs per probe plus 0.76 ms fixed) | 255–272 |
| Bistro exterior | (its fixed volume) | **132** | 124–160 |

- Bistro's camera-following volume (12,288 probes, m3-moving.md) is the same scene, so this doc uses 132 for it too.
- Section 2's deferred shadow pass could raise these. Nothing below assumes it does.

## What others do, and what we copy

How other systems schedule probe updates under change (classification, cascades, priorities, feedback) is collected in [research/ddgi-dynamic-updates.md](../../research/ddgi-dynamic-updates.md). The table below is what M2 takes.

| Who | How light changes reach the GI | What we take |
| --- | --- | --- |
| **RTXGI-DDGI** (Majercik et al. 2019, 2021; the SDK in `native/rtxgi-ddgi`) | Every probe is updated every frame with hysteresis 0.97. The 2021 paper (§4.3) cuts hysteresis per texel when the estimate moves: by 0.15 above a 25% change, to 0 above 80%, plus cuts per event (for time of day, −50% for 10 frames). The SDK is **asymmetric** (`ProbeBlendingCS.hlsl:516-558`): when the texel gets darker by more than `probeIrradianceThreshold` (0.25), hysteresis drops by 0.75 and each step is at least 1/1024; when it gets brighter by more than `probeBrightnessThreshold` (0.10), the step is cut to a quarter. So darkening is fast and brightening damped. It blends in a gamma-5 encoding. A "probe variability" reduction can stop updating a converged volume. Every shaded hit casts one shadow ray per light, with no light sampling (`Lighting.hlsl`). | The field and its hysteresis, and the idea of cutting it on a change. **Not** the asymmetry: ours is symmetric (4.1), because a light switched on should arrive as fast as one switched off. **Not** the change test on the noisy estimate: T1.5b measured every variant of it biasing the field (+8% to +22% on Cornell; DECISIONS 2026-11-05). We take the change from the scene instead. |
| **Lumen** (Wright et al., "Lumen: Real-time Global Illumination in Unreal Engine 5", SIGGRAPH 2022 Advances) | Lights its surface cache within a fixed per-frame budget, choosing what to refresh by priority; indirect light lags, direct light is instant. Epic's docs say plainly that "global lighting changes, like disabling the Sun, can take multiple seconds to propagate". | **A fixed budget with a priority order,** camera distance included. And a yardstick: in shipped engines a change shows within a few frames and settles in about 1–3 s, the range 6.4's times aim for. |
| **Unity Adaptive Probe Volumes** (`ProbeVolume.hlsl:711-735`) and PRT practice | Each probe stores sky occlusion without colour; at run time it's multiplied by the current ambient probe, so a sky change is instant. | **The sun layer and the first-order sky transfer** (6.3): stored without colour, weighted by the current colour and intensity. |
| **Godot SDFGI**, and RTXGI | Store real colour and re-trace when lighting changes. | What our re-tracing does for the sun's direction; we take the instant weighting where it's exact (6.1). |
| **MegaLights** (Narkowicz and Costa, SIGGRAPH 2025 Advances; [slides](https://www.advances.realtimerendering.com/s2025/content/MegaLights_Stochastic_Direct_Lighting_2025.pdf)) | Direct light by importance-sampling lights and tracing a *fixed* number of shadow rays per pixel, in a pass of its own, then denoising. Cost stays constant as lights are added. | **A fixed number of shadow rays per hit, in their own pass.** Our "denoiser" is the field's blend over time. |
| **ReSTIR** (Bitterli et al., SIGGRAPH 2020) and **ReGIR** (Boksansky, Jukarainen, Wyman, *Ray Tracing Gems II* ch. 23, 2021; [chapter](https://boksajak.github.io/files/RTG2_ReGIRGridBasedReservoirs.pdf)) | ReSTIR picks lights per pixel on screen. ReGIR builds a few reservoirs per cell of a **world-space grid** each frame, and any ray that hits inside a cell draws from them, which is what rays that don't start at the camera need. | **ReGIR's world-space cells** for many lights. DDGI's rays hit surfaces all over the world, not on screen. |
| **Light trees** (Conty Estevez and Kulla 2018; Moreau, Pharr, Clarberg, "Dynamic Many-Light Sampling for Real-Time Ray Tracing", HPG 2019) | A tree over the lights, walked at random toward the bright parts. Moreau keeps it dynamic with two levels and refits: under 1 ms for thousands of lights. | The fallback if the grid is too noisy (risk 3). Our BVH package could build it. |
| **Wavefront tracing** (Laine, Karras, Aila, "Megakernels Considered Harmful", HPG 2013) | Split one big kernel into small ones joined by compacted work lists, so threads stay busy. | **The deferred shadow pass** is this, applied to our trace kernel. |
| **A-SVGF** (Schied, Peters, Dachsbacher, HPG 2018) | Shortens the history of a temporal filter where it detects that lighting changed. | **History length per probe, shortened by the size of the change.** We know the change from the scene, so we skip their noisy detector. |
| **Enlighten** (Martin and Einarsson, "A Real-Time Radiosity Architecture for Video Games", SIGGRAPH 2010 Advances) | Static geometry with precomputed links between surfaces; lights and time of day relit at run time, a little each frame. | Confirms incremental, budgeted updates are the norm. We stay ray-traced, because geometry moves in M3. |
| **Relighting by superposition** (Dorsey, Sillion, Greenberg, SIGGRAPH 1991; Nimeroff, Simoncelli, Dorsey, EGWR 1994) and HomeFig's light groups (SPEC) | A lighting change that only scales a group is a weighted sum of per-group solutions. HomeFig measured it linear to 0.006%. | **Light groups as layers,** with HomeFig's three rules (SPEC "Lessons"). |
| **Far Cry 3** (Gilabert and Stefanov, "Deferred Radiance Transfer Volumes", GDC 2012) and PRT (Sloan, Kautz, Snyder 2002) | Probes store how they respond to each of a few sky "basis" patterns, so the sky and sun relight instantly. | **The instant sky** (section 6.3), cut down to what a probe sees of the sky directly, so it builds from the field's own rays in seconds. Re-tracing near the camera stays the main path. |

## 1. The light buffer

**Today (T1.2):** lights are 64-byte records in the scene copy's table buffer, packed, directional lights first. Capacity starts at 32 and grows by reallocating the table buffer, with no shader changes. `updateLight` takes per-type partial changes (`LightChanges`). The change log turns a light change into a box (`position ± range`) or, for the sun or a light with `range: 0`, a global flag.

**M2 changes:**

- **Capacity** starts at 64 and grows by doubling, as now. Tested to 4,096 lights (256 KB).
- **The change log keeps the old and the new light,** not just boxes. `SceneChanges` gains `lights: readonly LightChange[]`, each `{ index, before, after }` in the platform's units. This is technique-facing; hosts don't see it. Up to 1,024 light changes per frame are kept one by one; past that, the frame is treated as a global change (section 4.2).
- **An influence radius** for each light, computed on the CPU, bounds which probes a change can touch: the distance where the light's unwindowed irradiance falls below `1/1,000 × E_ref`. `E_ref` is the scene's reference irradiance: the sun's, or, with no sun, the brightest local light's at 1 m. So a dim lamp gets a small radius and a bright one a large radius. A light's `range`, when set, caps it. The radius is used only for this, never in shading.
- **Adapter:** lights are checked every frame today. That's fine for a few lights. With hundreds it could break the 0.2 ms still-scene target (an estimate: about 0.2 µs per light check; T1.7 only timed scenes with a few lights). Lights beyond the first 64 get the same budgeted walk and hot list as objects (three.md, "How changes are found"). `markChanged(light)` still sends a change the next frame.
- **The manifest's `motions` format** (section 7) is defined here too, because the native harness (T2.7) and the viewer (T2.8) both read it.

**As built (T2.1, 2026-11-06).** Where it differs from the sketch above:

- **The light table starts at 64** and doubles; 4,096 lights made and destroyed in random order with DDGI running build 0 pipelines (`platform-recompiles`, RTX).
- **The adapter's light walk:** the first 64 lights and every directional light every frame; the rest `lightsPerFrame` (64) a frame, round robin, with a hot list (60 still frames) and `markChanged` (three.md, "How changes are found"). `userData.giCastsShadows` lets GI shadow lights three draws without shadows (candles under three's `ClusteredLighting`).
- **Many lights with no range changing at once** (`ddgi/mark`, mark-table.ts): up to 8 a frame are marked over every probe, as T2.4b's rule says; past 8 each is marked exactly within 2 probe spacings of where it was and is, and one entry over every probe carries a bound on what any of them changes farther away (the largest relative change of its own unshadowed irradiance there: its colour and intensity change, a move's bounded share, or a whole change when a light switches on or off, a spot turns, or a range is set). On `sponza-candles` with all 1,024 candles changing in one frame: 0.074 ms, against about 5.8 ms for the per-light rule (DECISIONS 2026-11-06).
- **Motions** (section 7): `motions: [{ id, label, target: 'light:<label>', interpolation?: 'linear' | 'step', loop, keyframes: [{ t, position?, intensity?, color? }] }]` in `scenes/src/manifest.ts`, with `motionState(motion, seconds)` for players. `step` interpolation holds each keyframe, which the step motion needs. Every scene with a lamp has `light-sweep` and `light-step`; Sponza has an `arcade-lamp`, off (intensity 0) until a motion plays it, along the north arcade. Lights at intensity 0 are left out of the scene by the viewer's loader and the native harness, because an off lamp still cost GI's hit shading a light loop.
- **`sponza-candles`** (section 3): night, lit only by the candles; a `light-set` part (one candle and Intel's 10,000 light positions, made by a Node convert step) and a `candles` scene setting (default 1,024, up to 10,000). The selection rule is in DECISIONS.

## 2. The deferred shadow-ray pass (platform hit shading)

**Why:** T1.9a measured shadow rays at 53% of Sponza's `ddgi/trace` (at 2,048 probes) and at no more than 1/7 of the trace bench's any-hit rate. The likely cause, until T2.0 measures it, is that the shadow-ray walk runs inside a kernel that already holds the closest-hit walk, the bounce lookup and the shading, and only some threads in a subgroup have a front-face hit to light.

**What changes:**

- `ddgi/trace` stops calling `giDirectIrradiance`. For each front-face hit of a blended ray, it writes a **surface record** and appends the ray to a compacted work list.
- A new **platform** kernel, `gi/direct`, runs over that list only. It picks lights, traces their shadow rays, and adds `albedo / π × direct` into the ray's radiance.
- `ddgi/blend` is unchanged: it reads finished radiances.

**Surface record** (`GiSurfaceRecord`, 24 bytes; the name `GiHitRecord` is already the batch tracer's record in `contract/queries.ts`):

| Field | Bytes | Why |
| --- | --- | --- |
| position, f32 × 3 | 12 | Shadow-ray origin; the push off the surface is applied in `gi/direct` |
| shading normal, octahedral 16 + 16 bits | 4 | The cosine term |
| ray-side geometric normal, octahedral | 4 | "Light from behind can't reach it", and the push-off direction |
| albedo, 10 + 10 + 10 bits unorm | 4 | So `gi/direct` never reads the material table. Albedo is at most 1 after `bounceAlbedoCap` (default 1.0); a cap above 1 is stored as a scale in the 2 spare bits |

**Memory:** the ray buffer grows on demand up to `maxProbesPerFrame` (default 4,096), and the viewer's warm-up uses 4,096-probe batches. Uncapped, surface records could reach 22 MB. So M2 splits any batch above 1,024 probes into rounds of 1,024 inside the same `encode` (trace, direct, blend, write per round). The ray scratch (ray results, surface records and the work-list counter, one buffer with three sections, the scene copy's pattern) is then at most 1,024 × (256 × 8 × layers + 224 × 24) bytes: **7.6 MB** with one layer, 11.8 MB with three (lights, sun and sky, with the instant sky on). `ddgi/trace` stays at 7 storage buffers.

**The work list** is filled with one `atomicAdd` per subgroup, with each thread's offset from `subgroupExclusiveAdd`. The declared fallback without subgroups is one atomic per thread. A 1-thread kernel turns the final count into indirect dispatch arguments.

**One code path, not two.** Light selection (section 3) becomes `giDirectLightSamples(surface, groupMask, random) -> GiDirectSamples` in the hit-shading module. `gi/direct` calls it and traces the samples. The existing `giDirectIrradiance` becomes a short wrapper that calls the same function and traces inline; the test views and the oracle keep using it. Both read the same selection code.

**Contract additions** (technique-facing; added to platform.md's "Revisions" when built):

- `HitShading.encodeDirect(encoder, { scratch, maxRays, groupLayers, gridBounds })` records `gi/direct` for a technique's ray scratch. `gridBounds` tells the platform where to build the light grid (section 3), because the platform doesn't know a technique's volumes.
- `GiSurfaceRecord` and its WGSL writer `giWriteSurfaceRecord(...)`, so techniques never depend on the byte layout.

**What it might buy.** Two things limit Sponza at 2 ms:

- **The per-probe cost.** This is where the 53% was measured. If shadow rays reach even half the bench's any-hit rate once they're coherent, full-subgroup work, the per-probe cost of trace plus direct drops to about 0.6× today's.
- **The fixed cost.** On Sponza a batch of 20 probes already costs 1.55 ms (T1.9a). Most likely that's the latency of the longest walks in a small batch. A cheaper per-probe cost buys few extra probes while the fixed part dominates. Shorter kernels may cut that latency too, but nobody has measured it.

So the gain in probes per frame is unknown, somewhere between "little" and "about 1.6×". **The spike (T2.0) measures both parts before anything else is built.** Nothing below counts on the gain: the rates table above is the plan.

Bindings: `gi/direct` = tracer 3 + scene copy and shading 2 + ray scratch 1 + light grid 1 = **7**.

**As built (T2.2, 2026-11-06; `platform/shade/direct.ts`, contract in `contract/shade.ts`).** Where it differs from the sketch above:

- **The work item is 32 bytes:** position (f32 × 3), the two octahedral normals, albedo as unorm16 per channel over a power-of-two scale (so a cap above 1 fits), and the ray's index. 10:10:10 albedo rounded by up to 0.75% of a dark channel (T2.0's fixed 0.03% offset); unorm16 rounds by at most 0.012% of it (DECISIONS 2026-11-06).
- **Rounds cover the trace and `gi/direct` only.** Batches above 1,024 probes trace in rounds (one `ddgi/trace` and one `gi/direct` pass each, the round's slots in a uniform picked with a dynamic offset), so the work list holds one round (7.3 MB). Ray results stay sized for the whole batch, as in M1, and `ddgi/blend` and `ddgi/write` run once.
- **The scratch's layout:** a 256-byte header (the count, this frame's number and light groups, the frame's counts of hits lit and dropped), the ray results, then the work list. `ddgi/blend` binds the results alone, as before. The indirect dispatch arguments live in a small platform buffer, since `gi/direct` binds the scratch as writable storage and a buffer can't be that and the indirect buffer of one dispatch.
- **Only hits a light can reach are listed** (`giDirectReaches`, the unshadowed test, no rays): 39 of Sponza's 81 blended front-face hits a probe.
- **`gi/direct` rounds its sum to halves stochastically.** The result goes through half floats twice on this path, and `pack2x16float` truncates on the RTX 3060 (−0.035% a store).
- **Light selection is still `giDirectIrradiance`,** called by `gi/direct` on the rebuilt surface; `giDirectLightSamples` comes with T2.3's grid. The inline path draws its numbers with `giDirectRandom` as `gi/direct` does, so the two fields compare texel by texel.
- Storage buffers: `ddgi/trace` 7 (unchanged), `gi/direct` 6 (7 with T2.3's grid).

## 3. Many lights: a world-space grid of light candidates (ReGIR-style)

**Today:** `giDirectIrradiance` loops over every local light twice: first to find which lights reach the hit and how much, then to pick `maxShadowedLightsPerHit` (4) of them by systematic sampling. It decides *per hit* whether to sample, from how many lights reach that hit. The cost per hit grows with the light count. With 1,024 candles, 512 probes × 224 rays × 2 × 1,024 is 235 M light evaluations per frame, which doesn't fit.

**Proposed:** two regimes, chosen by a uniform (the scene's total local-light count, which the CPU knows):

1. **At most `exactLightLimit` local lights (default 8):** today's code, unchanged. All of them are looped over per hit and K = 4 are sampled when more than 4 reach the hit. Scenes with a few lamps behave exactly as in M1.
2. **More than that:** draw from a **light grid** built each frame:
   - **Cells:** a uniform grid over `gridBounds`, one cell per 2 × 2 × 2 probe cells, at most 32,768 cells (Sponza: about 2,400).
   - **Candidates:** each cell fills `R = 8` reservoirs from `C = 16` candidates each. Candidates are drawn from all local lights in proportion to their power, using a CDF rebuilt by a scan when lights change (the `gpu-scan` from B1, on `bvh-b1`, which DECISIONS merges into `m1`).
   - **How a candidate is weighted in a cell:** an upper bound of the light's unshadowed irradiance over the cell's box, using the nearest point of the box for the falloff and range window, and a cone test against the box's bounding sphere for spots. There's no cosine, because the surface normal isn't known per cell. *Why an upper bound over the box and not the cell centre:* a hit near a cell's edge can be lit by a light the centre can't see (outside its range or its cone). The weight must be above zero wherever the true contribution is, or the estimate is biased.
   - **At a hit:** `gi/direct` draws `maxShadowedLightsPerHit` of its cell's 8 reservoirs (default 4, the same setting as today), re-evaluates each light at the real hit with the real cosine, and traces one shadow ray each. Directional lights are always evaluated (one shadow ray each). So a hit costs at most K + (directional lights) shadow rays, whatever the light count. T2.3 measures K = 1, 2 and 4 for noise against cost.
   - **Outside the grid** (hits beyond every volume): candidates straight from the power CDF.

Data: `cells × R × 8 bytes` (light index, weight): 2,400 × 8 × 8 = 154 KB on Sponza, 2 MB at the 32,768-cell cap. Build cost: cells × R × C light evaluations, 2,400 × 128 = 0.3 M per frame on Sponza. Estimate: about 0.05 ms on the RTX.

**Why not ReSTIR's reuse over time and between neighbours too:** the field already averages hundreds of rays over many frames, so a fair sample per hit is enough. Reuse would add per-hit state that probes revisit only every few frames. Revisit if candles look noisy (risk 3).

**Why not a light tree first:** constant cost per hit matters more for us than the tree's better quality on tight clusters of lights. The grid is rebuilt from scratch every frame, so it needs no refit or rebuild policy under motion, and it lines up with the probe grid we already have. The light tree (Moreau 2019) is the fallback if the grid fails its noise check on `sponza-candles`.

**`sponza-candles`:** a stated subset of Intel's 10,000 candles (DECISIONS 2026-11-05). Dennis: "1024 is fine for now". The count is a scene setting (`candles`, default 1,024, up to 10,000), so it can be raised as a stress test. Only the chosen candles' geometry is loaded (the full candle mesh is 19.5 M triangles). The light-grid checks run at 1,024; higher counts are recorded, not gated.

## 4. Change-driven updates: per-probe history and a GPU scheduler

This is how light changes get fast. It replaces the big-change rule (off by default since T1.5b) in that role, and it is the "priority list in front of the update cursor, in the same buffer" that ddgi.md left room for.

### 4.1 Per-probe history: T1.12's sample count, cut by changes

**What M1 already has (T1.12 phase 2a, ddgi.md "Hysteresis that knows the time between updates").** Each probe keeps an effective sample count `n`, the time `t` of its last update and a count `k` of its updates (`DdgiProbeHistory`, 12 bytes). On each update:

- `n` fades with the time since the last update, `n ← n · exp(−Δt / τ)` with `τ` = 5 s;
- the warm-up ramp caps it for the first 16 updates;
- the blend uses `h = n / (n + 1)`, then `n ← min(n + 1, 200)`.

So a probe updated every frame (Cornell) reaches the cap, `h` ≈ 0.995, a memory of 200 updates (about 3.3 s at 60 Hz), and a Sponza probe reached once a sweep (about every 1.2 s) settles at `n` ≈ 3.7 (`h` ≈ 0.79), a memory of about 5 s. Measured with no change signal at all, Sponza with the sun halved gets within 5% in 12.5 s and within 1% in 20.5 s (T1.12, with its ramp).

**What M2 adds: changes cut `n` at once.** Unreported, a change waits for the 5 s fade. Reported from the scene (4.2), it cuts `n` in the probe's next update. M2 widens the same record, so there's one history and one blend rule, not two:

| Word | Field | Meaning |
| --- | --- | --- |
| 0, 1 | `n`, `t` (f32) | T1.12's sample count and last update time, unchanged |
| 2 | `k` (16 bits), `owed` (8 bits), transfer count `m` (8 bits) | T1.12's update count (saturating); updates owed at priority; the sky transfer's own count (6.3) |
| 3 | `pendingChange` (f32) | Relative change since this probe's last update, added up over marks; reset by the blend. f32, so many small marks don't round away |
| 4 | `urgency` (f32) | The largest change since the last update, for the scheduler |
| 5 | `scheduledFrame` (24 bits), flags (8 bits) | The last frame it was put in the update list, so the fill never adds it twice (24 bits only need to differ from the current frame); flags: marked before its last update (the steady-change rule), awaiting a change (4.5) |
| 6, 7 | mean irradiance (rgb, halves), noise (half) | Written by the blend, which already computes the probe's mean. The mean is read by `ddgi/mark`; the noise by the feedback rule (4.6) |
| 8 | `wakeUntil` | M3 (m3-moving.md 5.2) |
| 9 | keys received (8 bits), spare | Which of its up to 8 key neighbours have passed it their change since its mark (4.5) |
| 10 | change-clock stamp (f32) | The volume's change clock at this probe's last update (4.5) |
| 11 | spare | |

48 bytes a probe: 0.93 MB at 19,425 probes. Halves are packed with `pack2x16float`, so `shader-f16` isn't needed.

**How short the history becomes after a change.** Say a probe's lighting changed by a relative amount `r`, and the scheduler promises it `K` updates soon. The error left after those K updates is `r × n / (n + K)`. Setting that equal to the tolerance ε (5%) gives

> `n ← min(n, εK / (r − ε))` when `r > ε`; untouched when `r ≤ ε`.

That's applied to T1.12's `n` before the blend's own rule, which then fades and grows it as usual.

*As built (T2.4b, DECISIONS 2026-11-06):* the cut **holds** for the promised updates and is solved with a bounce share ρ (`changeBounceShare`, 0.8): `h = (c − ρ)/(1 − ρ)` with `c = (ε/r)^(1/K)`, and a change promises `max(K, ⌈ln(ε/e)/ln ρ⌉)` updates. The formula above takes each estimate as unbiased, but estimates bounce off a field that still holds the old light; with the count growing back by one an update, two-room's lamp halved was 45% of the step away after 30 frames. A darkening's r is measured against the light that's left. ddgi.md "Change-driven history" has the details.

**The steady-change rule, for changes that keep going** (not to be confused with T1.12's warm-up ramp). A slow, steady change (a lamp gliding, a day passing) moves a probe only a little between two of its updates, so the step rule leaves its history alone, and T1.12's 5 s memory would trail the change by about 5 s of it: about 7° of sun at 1.5° a second. So when a probe was marked before its previous update too (a flag in word 5), the blend also applies

> `n ← min(n, ε / r)`, with `r` the change since its last update,

which keeps the lag near ε. A one-off small step (marked once) isn't affected.

With ε = 5% and K = 8:

| Relative change r | New n | First update's weight on the old value | Left after 8 updates |
| --- | --- | --- | --- |
| 1.0 (a light switched on right next to it) | 0.42 | 0.30 | 5% |
| 0.5 | 0.89 | 0.47 | 5% |
| 0.2 | 2.7 | 0.73 | 5% |
| 0.1 | 8 | 0.89 | 5% |
| ≤ 0.05 | unchanged | `n / (n + 1)` | n/a |

- *Why cut the count rather than switch to a fixed fast hysteresis:* a fixed `h = 0.5` forgets in about 5 updates, but its stored value then carries the noise of only about 3 updates' worth of rays, for good. A cut count gets less noisy with every update and climbs back by itself, under T1.12's rule.
- *Why the change comes from the scene, not from the probe's own estimate:* the estimate is noisy, and every rule that cut hysteresis from it biased the field (T1.5b). The estimate in 4.2 uses only light parameters and the field's stored data, so a still scene can never trip it. (4.6 uses the probe's own change only to schedule it sooner, never to cut its history.)
- **Symmetric on purpose.** RTXGI cuts hysteresis only when a texel darkens and damps brightening to a quarter per step. Ours cuts `n` by the size of the change, whichever way it goes, because a lamp switched on must meet the same 0.5 s as one switched off.
- **The time between updates is T1.12's business.** The fade makes a probe's memory about 5 s whatever its update interval, so a probe the round robin reaches every 1.2 s doesn't hold a 40 s history. M2's cut works on top of that and doesn't need its own time rule.
- The same history applies to the distance atlas. For light changes it hardly matters, since distances don't change; M3 uses it for geometry.
- **The opt-in big-change setting** (`irradianceChangeThreshold`) is left as decided: off by default, and when enabled it still applies its own 0.75 factor on top of `h`.

### 4.2 The change estimate (`ddgi/mark`)

It runs only in frames with changes. The CPU writes a small table of changed lights (from `SceneChanges.lights`) and dispatches one thread per (changed light, probe inside that light's influence box clipped to the volume). Lights are walked one by one, never through the merged boxes, which lose which light changed.

**A local light moved, was recoloured, dimmed or toggled:**

- At the probe, for the light before and after, take its unshadowed **irradiance vector** per colour channel: `E_c × direction to the light`. This uses three's falloff, range window and cone (the same WGSL as hit shading, `giShadeLightSample`, without a surface cosine).
- `Δ = max over r, g, b of |E_after,c × d_after − E_before,c × d_before|`. *Why a vector, per channel:* a light that keeps its distance but moves to the other side of the probe changes the vector, not its length, and a colour change at equal brightness changes a channel, not the luminance. A luminance difference would miss both.
- `r = lightChangeScale × Δ / (mean_c + ε)`, where `mean_c` is the probe's stored mean irradiance in the same channel (history words 6–7). `lightChangeScale` (default 0.25, tuned in T2.4a) accounts for the probe seeing the light only through the surfaces it lights.
  - *As built (T2.4a, DECISIONS 2026-11-06):* the default is **0.5**, the measured median of true change over the unscaled mark on two-room's lamp step (quartiles 0.25–0.73). At 0.25 the right probes were cut (recall 99.8%) but too gently, and the old lamp's bounce light lingered: 0.73 s to within 5% instead of 2 frames. The environment's r is 1 until T2.6a brings the sky's SH to the CPU. ddgi.md "Change-driven history" has the details.
- **Visibility filter:** one Chebyshev test from the probe's distance moments toward the light (as in sampling) scales `r` by `max(chebyshev, 0.25)`. A probe behind a wall gets a smaller share but never zero, because it may see lit surfaces through a door.

**Global changes** (the sun's direction, the sky; more than 1,024 light changes in one frame) mark every probe. On big volumes, far probes are owed 2 updates instead of 8 (6.4). The exceptions:

- a sun change of colour or intensity alone marks nothing: it becomes the sun layer's weight (6.1);
- a sky change marks probes with a built transfer at `r × skyResidualShare` (default 0.3), because the nudge has already moved most of it, unless the nudge's brightness ratio was clamped (6.3).

For a sky change, `r = max over channels of (|Δc₀| + |Δc₁…₃|) / c₀` (the change of the sky's band-0 and band-1 radiance SH, relative to its mean), times the same openness factor as below.

- `r` is the size of the change: the sun's angle moved (`2 sin(θ / 2)`) or `|1 − new / old|` for an intensity or colour ratio, per channel.
- It is multiplied by `0.25 + 0.75 × openness`, where openness comes from the probe's distance moments toward the old and new sun directions (a probe that sees open sky sees the change directly).

**Each marked probe:** `pendingChange += r`; `n ← min(n, εK / (pendingChange − ε))`; `owed ← max(owed, K)`; `urgency ← max(urgency, pendingChange)`; and the steady-change flag is set if it was marked since its previous update.

**Except fine probes that will be served by propagation** (4.5: not a key, camera weight under 0.5 at the mark): their cut is deferred to their own blend, which uses what propagation hasn't brought them, and their urgency is `pendingChange × keyResidualShare`. They're flagged as awaiting a change.

Continuous motion (the sweep) marks a little every frame. Because `pendingChange` adds up until the probe is next updated, the history settles where it matches the rate of change: short near the moving light, unchanged far away. No special case is needed.

### 4.3 The GPU scheduler

The update list moves from the CPU (`UpdateCursor` plus `writeBuffer`) to the GPU, because the urgencies live there. It is still one buffer, filled with exactly `min(units, probes)` entries. So the trace, blend and write dispatches keep sizes the CPU knows, and the cost model still sees the units it handed out. It takes four small dispatches, every frame:

1. **`ddgi/schedule-count`** (one thread per probe): a probe with `owed > 0` gets the priority `urgency × cameraWeight`, sorted into one of 32 buckets, each twice as urgent as the one below. A histogram of the buckets is built with one atomic per subgroup (fallback: one per thread).
   - `cameraWeight = 1 / (1 + (distance to camera / (4 × spacing))²)`, times 0.25 outside the view frustum. *Why:* SPEC's response target is judged on the image, and nearby, visible probes are what the image shows. Lumen's priority does the same.
2. **`ddgi/schedule-select`** (one workgroup): finds the bucket where the list fills up, so that at most `P = priorityShare × units` probes are chosen. `priorityShare` defaults to 0.75, so the round robin always keeps a quarter of the budget and no probe is starved.
3. **`ddgi/schedule-write`** (one thread per probe): probes above that bucket, and as many as fit from it, go into the list and are stamped with the frame number.
4. **`ddgi/schedule-fill`** (one workgroup): the rest of the list comes from the existing scrambled order, `order[i] = (i × stride) mod N`. A window of `units` consecutive entries holds at most `P` stamped probes, so it always has enough unstamped ones. A running sum over the window picks them, and the cursor (now one GPU word) moves past the entries used.

Cost: four dispatches of a few µs each, plus one thread per probe. Estimate: under 0.05 ms on the RTX at 20,000 probes. They run every frame: the CPU can't see on the GPU whether probes are still owed, and skipping them would leave owed probes stranded (a Sponza-wide change keeps probes owed for hundreds of frames).

*As built (T2.4b):* one pass `ddgi/schedule` of the four dispatches, 0.020–0.027 ms at 20,000 probes on the RTX (VALID); the fill's scan is subgroups first when the device lays subgroups out in order (checked in the kernel), else in workgroup memory. Every listed probe is stamped, the round robin's too. A probe's urgency stays while it is owed updates.

**What the budget buys.** With `U` probes per frame (the rates section), a window of `W` frames and K owed updates, at most `P × W / K` changed probes get all their updates in time. With W = 30 and K = 8:

| Scene | U | P (75%) | Changed probes fully served within 0.5 s |
| --- | --- | --- | --- |
| Cornell | 1,574 | ~1,180 | every probe, 8 updates in about 9 frames |
| Two-room | 707 | ~530 | every probe, 8 updates in about 12 frames |
| Sponza | 268 | ~201 | ~750 |

A lamp whose influence box holds more probes than that is served nearest-and-most-changed first. The visible image meets the target, while far, weakly changed probes finish later.

### 4.4 Light groups make some changes need no updates at all

See section 5: a dimmer or colour change across a whole group skips sections 4.2 and 4.3 entirely.

### 4.5 Key probes: a coarse level inside the same grid

Dennis (2026-11-06) asked about "subdividing probe grids and giving priority to 'key probes' that then feed into sub groups". This is that idea, fitted to what we have. The research note ([research/ddgi-dynamic-updates.md](../../research/ddgi-dynamic-updates.md)) lists what shipped systems do; none of the DDGI-like ones pass changes between levels, so this is ours to prove.

**The problem it solves.** After a global change (the sun's direction, the sky), every probe is owed updates. At Sponza's 268 a frame, 8 updates for each of 19,425 probes take about 9.7 s, which Dennis calls a fail. Owing far probes only 2 (6.4) brings it to about 2.5 s, but leaves them noisy.

**Key probes.** The probes whose grid index is even, or the last one, on every axis: about one in 8.

- Sponza (37 × 21 × 25) has 19 × 11 × 13 = 2,717; two-room (17 × 7 × 9) has 9 × 4 × 5 = 180.
- "Or the last one" matters on even-sized grids (Cornell's 12³, Bistro's 32 × 12 × 32), which would otherwise have no keys on their far faces. There the last key cell is one spacing wide instead of two.
- In a camera-following volume, parity is taken from the **world lattice index**, not the wrapped storage slot. So a probe stays a key as the volume scrolls.
- Key probes are ordinary probes in the same atlas, traced and blended by the same kernels. Only the scheduler and one new pass treat them differently. There's no second grid, no second atlas and no second sampling path.
- **After a change they go first.** A marked key probe's urgency is `r × keyProbeBoost` (default 4). A marked fine probe **out of the view's core** (camera weight under 0.5) gets `r × keyResidualShare` (default 0.3), because propagation will bring it most of the way. Fine probes with a camera weight of 0.5 or more (near and in view) keep their full urgency and their 8 owed updates, so they still come before far keys: a key 10 spacings away scores about 0.03 × 4 = 0.14, a view probe at least 0.5. It's an order of preference, not a strict one.
- They're the coarse level of a **virtual two-level grid** at twice the spacing, like a coarser cascade (Godot's SDFGI, Majercik et al. 2021's cascaded volumes) without its memory or sampling cost. Those cascades are traced independently and never pass changes between levels; this one does.

**Propagation: a fine probe takes its key neighbours' change until it's traced itself.**

- **What a key writes.** When `ddgi/blend` updates a key probe that is owed updates because of a change, it also writes the change it made, `Δ = new − old` per irradiance texel and layer, to a delta buffer: 288 bytes per update-list slot and layer, under 0.3 MB at 268 slots and 3 layers. Beside it goes the key's change-clock stamp from before this update, `G_prev` (history word 10, below).
- **Who receives it.** Fine probes that are **awaiting a change**: the flag `ddgi/mark` sets and the probe's own blend clears.
- **The pass.** `ddgi/propagate` runs after `ddgi/write`, over the fine probes that are awaiting and have a key neighbour updated this frame. Each adds, per texel and layer,

  > `Σⱼ wⱼ · sⱼ · Δⱼ` over its key neighbours `j` updated this frame,

  where:
  - **`wⱼ`** is the fine probe's trilinear weight in the key lattice, times a visibility test (below), normalised over its active, visible key neighbours, whether or not they were updated this frame. Off keys count 0. So each key's share is fixed, and the shares add up to 1.
  - **`sⱼ`** is the part of the key's change that came after the fine probe's own last update. *Why:* a key's `Δ` covers everything since its own last update, which can be a second back; if the fine probe was traced in between, it already has the earlier part, and a change that keeps going (a day animation) would otherwise be counted twice.
  - **It's measured in change, not in seconds.** The volume keeps a **change clock** `G`: every frame `ddgi/mark` adds the largest `r` it marked. Every probe stamps `G` at its update (history word 10), and a key's delta entry carries its stamp from before this update, `G_prev`. Then `sⱼ = clamp((G − max(G_f, G_prev,j)) / (G − G_prev,j), 0, 1)`, with `G_f` the fine probe's stamp, and `sⱼ = 1` when `G = G_prev,j`: a key's later owed updates after a step add no clock, and the awaiting flag already keeps out fine probes traced since their mark, so that can't double-count. A step lands in one frame, so on a key's first owed update `sⱼ` is exactly 1 (the fine probe was traced before the step) or 0 (after it); a steady change is split in proportion.
  - **The clock is a `u32` in fixed point** (1/4096 of a unit of `r`), wrapping like T1.12's clock, with differences taken modulo 2³². An f32 that grows for hours would stop advancing once its step passed the size of small marks. *Why not seconds:* a step lands entirely just before the key's first owed update, so a time ratio would pass on only part of it.
  - The clock is per volume, so two unrelated local changes at once share it. Then `sⱼ` is only approximate, which costs some propagation, never a double count.
- **The visibility test, from both ends.** Probe distances are clamped at 1.5 × the length of the spacing vector, about 2.6 spacings on a cubic grid (ddgi.md). Within one grid a fine probe is at most √3 ≈ 1.73 spacings from its keys, so either end's moments can vouch for the whole segment. Across nested volumes they can't (below). So the rule is: each probe's distance moments toward the other must clear its own end of the segment (a Chebyshev test, as sampling does, up to its own clamp), and the two clamps together must cover the distance. A fine probe that can see none of its keys gets nothing and waits for its own update.
- **How much it receives, and what's left.** The probe records which of its keys have passed it their change since its mark (8 bits in history word 9). `received` is the sum of those keys' weights, each counted once. A key moves about 80% of the way on its first owed update for a large change (K = 4 and r ≈ 1 give `h` ≈ 0.17; at r = 0.5 it's about 69%), so 0.8 is an estimate, so the blend cuts the fine probe's history for `r × max(keyResidualShare, 1 − 0.8 × received)` when it's finally traced, instead of the full `r`. *Why:* a fine probe the round robin traces before its keys are done (about 4,000 on Sponza in the first second) has received little, and cutting it only for the residual would leave it lagging.
- **Exact only in the simple case.** For one step of change, with no clamp acting, a fine probe receives exactly the interpolated total change its keys made while it waited. Two things break that:
  - the clamp that keeps irradiance ≥ 0 biases it a little bright where a large darkening meets noise;
  - a key's changes after its owed updates run out aren't passed on (a still-changing key is marked again, so that's only the tail of a change).

  Both are left for the fine probe's own update.
- **Noise.** A key's `Δ` carries most of one update's noise, because its history was cut. A fine probe averages that from its keys:
  - inside a key cell (8 keys) about 35% of one key's noise;
  - on an edge of the lattice (2 keys) about 70%.

  It then fades with T1.12's rate as the fine probe is traced. That's less than "far probes owe 2" leaves, but not nothing. T2.4c measures it.
- **Its own write and compose.** Propagation writes every layer, so it has its own scratch (288 bytes per layer for each fine probe it touches) and a compacted list. A `ddgi/write` dispatch over that list copies the tiles in, and `ddgi/compose` covers the list as well as the update list, so the change reaches the image the same frame.
  - It's the same write and compose kernels as the instant sky's nudge (6.3), run over a different list, not a second path. The sky's nudge runs before the trace; propagation runs after the write.
- **With the instant sky:** the nudge has already moved the sky layer of every probe with a built transfer, so a key's sky-layer `Δ` is only the residual. The sky layer is therefore propagated only between probes in the same transfer state (both built, or both not), with the weights renormalised over the keys in that state. A fine probe with no key in its own state gets nothing on its sky layer and waits for its own update.
- **Distances aren't propagated.** For a light change they don't move. For moved geometry, the probes near it are traced themselves (below).
- *Why a change and not a value:* a fine probe keeps its own detail (a sunlit patch's edge, a corner its keys can't see) and gets only the smooth part of the change. Copying the coarse value would wipe the detail it had.

**What it buys after a global change, on Sponza** (from T1.9b's median, 268 probes a frame: a priority share of 201, a round robin of 67):

| | K = 8 for every probe | Far probes owe 2 (6.4) | **Key probes and propagation** |
| --- | --- | --- | --- |
| What the camera sees (the ~200 nearest probes, all levels, owe 8: 1,600 updates) | ~0.13 s | ~0.13 s | ~0.13 s |
| Every key within 5% (the ~2,690 outside the view owe 4: 10,760 updates) | | | **~1.0 s** |
| Every probe shows the change | ~9.7 s | ~2.5 s | **~1.0 s**, at the key level, through propagation |
| Every fine probe traced once since the change (the round robin's 67 a frame meanwhile, then the whole budget) | | | **~1.8 s** |
| Far probes' noise afterwards | normal | about 2 updates' rays | 35–70% of one key update's noise, fading as they're traced |

- **Off probes owe nothing.** Probes switched off by classification (inside walls) aren't marked for light changes. Native's classification switches off 59% of Sponza's probes (NOTES); if ours is similar, every time in this table about halves. The table counts every probe, to be safe.
- **How good the propagated value is** depends on how smooth the change is. A sky colour change or a sun dimming is nearly uniform across two spacings, so propagation gets most of it. A sun-direction move shifts sunlit patches, whose edges the key level blurs, so fine probes keep part of the error until their own update, within about 1.8 s. T2.4c measures how much propagation removes on its own.
- **Its cost.** Each key has up to 26 fine neighbours. With the ~200 keys a frame spread out, that's up to about 5,000 fine probes touched, each reading up to 8 tiles per layer, with no rays. Estimated at under 0.1 ms on the RTX. It runs only in frames where keys were updated because of a change.
  - During a long change (a day animation) it runs every frame, together with the sky's nudge (about 0.1 ms with far quarters). That uses most of Budget v2's +10% margin (0.2 ms). Sponza's third calibration run was already at +4.3%. T2.4c measures it. If the margin breaks, the technique uses a second calibrated count while changes continue, lower by the measured cost of these passes.
- **Local changes too:** a light moved in a large room marks a few thousand probes; its keys go first and propagate, then the rest follow.
- **Moved geometry (M3):** the probes near the object (its near set, m3-moving.md 5.1) are all traced themselves, because their distances change too. Probes farther off (the far set), whose light changes through a shadow or a closed door, use keys and propagation.
- **Nested volumes** (M3's 0.25 m cloth volumes inside Sponza's 1 m volume): the parent volume's probes act as the nested volume's keys, with the same pass (the volumes share one atlas, m3-moving.md 4.1). The two-ended visibility test is what makes it work: the parent's distances reach about 2.6 m, more than the 0.87 m to any nested probe, while the nested probe's own reach only about 0.65 m.
- **Scrolling (M3):** entering planes are new probes with no distance data, so propagation can't help them. Of each pair of entering planes, the even one holds keys (a quarter of its probes) and is warmed first; the odd one has none.
- **The fallback** is "far probes owe 2" (6.4). If T2.4c shows propagation removing too little, keys stay as a scheduling order and propagation is switched off.

*As built (T2.4c, PROGRESS and DECISIONS 2026-11-06; ddgi.md "Key probes and propagation"):* the decision checks passed, so key probes and propagation are the main path (`keyProbes`, on by default) on volumes where K updates for every probe wouldn't fit in 120 frames (the same rule as 6.4), for local changes too; "far probes owe 2" stays behind the setting. A fine probe out of the view's core owes 1 update. The visibility test multiplies the two ends' Chebyshev values and a key counts when that is at least 0.5. Propagation's passes use no indirect dispatch (each cost 0.022 ms empty in Chrome). On Sponza at 316 a frame after a 30° sun turn: the view within 5% from frame 4, the field from frame 24 (fallback 48), every fine probe traced by frame 60; propagation alone removes 63.1% of the sun turn's change at the fine probes and 91.4% of a sky dimming's.

### 4.6 Feedback: revisit probes that changed more than their noise

Not every change is reported: an emissive texture a host animates without telling us, a material edit, or what's left after propagation and the instant sky. So the blend checks each probe's own update.

- The blend already computes the probe's mean irradiance. It looks at how far this update's estimate sits from the stored mean, `|Δmean| / (1 − h)` (the blend moves the mean by `1 − h` of that), and compares it with the probe's **noise** `σ`, the root of a running average of the same quantity squared over its quiet updates (history word 7). Dividing by `1 − h` keeps the test fair when `h` changes, after a cut for example.
- **Marked probes are skipped:** their change is already known and scheduled, and after a global change every probe would otherwise ask for its neighbours too.
- **If it's above 4σ and above 2% of the mean,** the probe asks for one more update at an urgency of `|Δmean| / mean`, and so do its 6 face neighbours, at half that. So a change nobody reported spreads to the probes around it within a few frames, nearest the camera first.

  - *As built (T2.4b, DECISIONS 2026-11-06):* **6σ**, because DDGI's ray noise has heavy tails (4σ fired on 0.1–0.2% of still updates); the noise word is relative to the mean and averaged over 64 quiet updates; the test skips probes in their warm-up ramp and probes still taking in a reported change.
- **No race.** The requests go to a separate buffer, one `u32` per probe, written with `atomicMax` on the urgency's bits. The blend never writes another probe's history record (T1.12's blend is race-free because each workgroup writes only its own probe). `ddgi/schedule-count` reads the requests and clears them. Blend goes from 5 storage buffers to 6.
- **It never touches hysteresis.** Every rule that cut hysteresis from a probe's own noisy estimate biased the field (T1.5b), because bright outliers won. Scheduling an extra update doesn't change how any update is blended. Because the choice depends on the value, a small bias over time is still possible, so T2.4b checks the still-scene mean with feedback on and off.
- **False triggers:** at 4σ, a Gaussian noise trips about once in 16,000 updates; bright outliers make it more. T2.4b counts it on a still scene; the bar is under 0.1%.

### 4.7 What triggers work, and in what order (the approach for M2 and M3)

| Trigger | Probes it marks | How they're served |
| --- | --- | --- |
| A local light changes (move, colour, intensity, on or off) | inside the light's influence box, by the vector change estimate (4.2) | near and in view first; keys before fine probes; propagation |
| A light group's weight | none | `ddgi/compose`, the same frame (section 5) |
| The sun's colour or intensity | none | the sun layer's weight, the same frame (6.1) |
| The sky | every probe, at the residual share where the transfer is built | the instant sky's nudge, the same frame (6.3); then as below |
| The sun's direction | every probe | near and in view first (owe 8); keys (owe 4) and propagation; fine probes at the residual share |
| An object moves (M3) | near set: within its old and new bounds plus a margin; far set: probes that see it, by the entry-distance test and the mover mask (m3-moving.md 5.1) | the near set traced itself, every probe; the far set through keys and propagation |
| A mesh deforms (M3) | the same, from `deformBounds` and the largest vertex movement | the same |
| The camera moves | none; a camera-following volume scrolls (m3-moving.md 4.2) | entering planes warmed, the even plane's keys first; the camera weight raises nearby probes every frame |
| Nothing reported, but a probe changed more than its noise | that probe and its 6 neighbours (4.6) | one update each, by urgency |

**The order in each frame**, all inside Budget v2's fixed probe count:

1. **The priority share** (75%), by `urgency × cameraWeight`, with the urgency boosted for keys and reduced for fine probes awaiting propagation (4.5). Owed probes near the camera and in view come first, then keys, then the rest.
2. **The round robin** (25%): the scrambled order, as in M1, so nothing is starved, and T1.12's fade keeps every probe's memory short (about 5 s on Sponza; Cornell, updated every frame, reaches T1.12's cap of 200 samples, about 3.3 s).
3. **After the blend and write:** `ddgi/propagate`, then its write and compose.

### 4.8 How it sits with Budget v2 and T1.12

- **Budget v2 sets how many probes a frame** (calibrated at load with the clocks up; T1.9b's medians: Cornell 1,574, two-room 707, Sponza 268, Bistro exterior 132 at 2 ms). The scheduler only decides **which**. A change never adds probes to the frame; it reorders them.
  - The passes M2 adds (mark, schedule, propagate, nudge) run only in frames that need them. During a long change they can use most of Budget v2's +10% margin (4.5).
- **T1.12 sets how each update blends** (the faded count). M2 cuts that count when a change is reported, and never otherwise. Feedback and propagation don't cut it at all; propagation only decides how big the cut is when the fine probe is finally traced.
- **Where the time goes after a big change on Sponza:**
  - the view in ~0.13 s;
  - the whole volume visibly changed in ~1.0 s (keys);
  - every probe traced once in ~1.8 s;
  - after that, T1.12's normal settling.

  Native RTXGI updates every active probe every frame, in about 13 ms on this RTX. We spend 2 ms and get the view in 8 frames.

## 5. Light groups

By linearity, the field for "group g at weight w" is w × the field for group g at weight 1, bounces included, as long as each group's bounce reads only its own layer (HomeFig rule 1).

**Layers:** one irradiance layer per light group in use (contract groups 0–3), plus an **environment layer** for the sky and emissive surfaces. HomeFig rule 2 applies: the sky's shape changes with time of day, so it gets a layer of its own and is re-traced, never just scaled. With groups off there is one layer, and everything below reduces to M1's kernels with a loop count of 1.

| What | Layers | Notes |
| --- | --- | --- |
| Irradiance, per group | G + 1 (with the environment) | Stored at the lights' real intensity and colour (HomeFig rule 3) |
| Composed irradiance | 1 (its own texture) | What the renderer samples. Not needed when there's only one layer |
| Distance, probe data | 1 | Geometry only, shared |

- **Where the layers live:** as blocks side by side inside each array layer of the irradiance atlas (6.3, "Layout"), not as more array layers. The atlas already uses one array layer per probe row, and the default limit is 256. The composed atlas is a texture of its own, because one dispatch can't read and write the same texture. So `ddgi/write` keeps its 4 storage textures, and the shading module still binds one irradiance texture.
- **Trace:** one closest hit per ray, as now. The bounce lookup works out the 8 probes' weights (distance, Chebyshev, crush) once and fetches each layer's irradiance with them. Each extra layer costs only its irradiance fetches, about a quarter of the lookup.
- **Direct light:** the hit's light samples are shared; each goes into its own group's layer. Each layer's estimate stays unbiased; groups with few lights get noisier.
- **Ray results:** 8 bytes per ray per layer, in the ray scratch.
- **Blend:** loops the 36 irradiance texels per layer; the 196 distance texels only once.
  - **One layer at a time in workgroup memory.** The blend sizes its arrays for `DDGI_MAX_RAYS` (512): 8 KB of ray directions and distances, 4 KB of radiance per layer, and about 1.2 KB of texel estimates, 13.5 KB of the 16 KB default today. A second and third radiance layer side by side would need about 21.7 KB. So the directions are loaded once, then each layer's radiance is loaded into the same 4 KB, blended, and replaced by the next layer after a barrier. Workgroup memory stays at 13.5 KB for any number of layers, at the cost of two barriers per extra layer (after its texels are blended, and after the next layer is loaded).
- **`ddgi/compose`:** composed = Σ weight_g × layer_g per texel, borders included. It runs for the tiles updated this frame, and for every tile in a frame where a weight changes. **A group weight change shows in the same frame,** with no lag and no noise.
- **Weights** are technique settings, `lightGroups: { count, weights: Color[], environmentWeight: Color }`: linear RGB multipliers on top of the lights' own intensities. The three adapter exposes `threeGI.setGroupWeight(group, color)`.

**Memory** (an irradiance tile is 8 × 8 texels × 8 bytes = 512 bytes per probe per layer):

| Scene (probes) | Groups off | 2 groups + environment + composed | 4 groups + environment + composed |
| --- | --- | --- | --- |
| Two-room (1,071) | 0.5 MB | 2.2 MB | 3.3 MB |
| Furnished room (4,998) | 2.6 MB | 10.2 MB | 15.4 MB |
| Sponza (19,425) | 9.9 MB | 39.8 MB | 59.7 MB |

Distance (1 KB per probe) doesn't change. Four groups fit the furnished room's 64 MB budget, but they are expensive on Sponza. So **groups are opt-in and off by default,** and the host picks how many (Dennis agreed, 2026-11-05). Time of day (6.1, 6.3) uses the same layers for the sun and the sky; those are on by default and have their own memory.

**Left open on purpose** (Dennis: "I dont want to close the door on common use cases"): in three.js, people dim a group the usual way, by scaling each light's `intensity`. The API keeps that possible later. Group weights are separate from light intensities, and the adapter already sees every intensity change, so a later detector could spot "every light of group g scaled by the same factor this frame" and turn it into a weight change. It's on the Later list, not in M2.

**Cost estimate** at 4 groups:

- trace about +10% (the extra irradiance fetches only);
- blend about +60% (the irradiance texel work × 5);
- compose about 0.1 ms on Sponza when a weight changes;
- in total about +25–35% per probe updated; with 3 layers (lights, sun and sky), about +15–20%; with 2 (lights and sun), about +8–10%.

Shading cost doesn't change, because the renderer samples one composed atlas. *Why not sample every layer in the fragment shader:* irradiance fetches were about 0.21 ms of T1.6's sampling cost (a run at a locked 1200 MHz clock, NOTES), and five layers would push sampling past 1.0 ms.

## 6. Sun and sky with time of day

Dennis (2026-11-05): time of day should be **"instant with a delayed fallback"**, and nothing may take minutes to build: "anything taking minutes to build is a fail and we should not go that direction." So each part below either needs no build, or builds from the field's own rays no slower than the field itself converges (on Sponza, about 1.2 s into the warm-up).

### 6.1 What changes with time of day, and how each part follows

| What changes | How it follows | What it has to build |
| --- | --- | --- |
| The sun's colour and brightness | **instant and exact.** The sun gets a layer of its own (section 5's layers), traced with the sun at a fixed white reference (colour 1, intensity 1). The layer's weight is the sun's actual colour × intensity, so a change is a new weight, with no marking. HomeFig's rule 3 (bake in real colours) is about groups of several differently coloured lights; for one light, white × its colour is the same thing | nothing |
| The sky's colour, brightness and shape | **instant for what each probe sees of the sky directly**, through openings (the first-order sky transfer, 6.3), with the sky's bounces rescaled by its brightness; **re-traced** for what that leaves | the transfer, from the field's own rays: built after a probe's first 8 updates |
| The sun's direction (the patch of sunlight moves, and its bounce with it) | **re-traced**, nearest the camera first (6.4) | nothing |

The last row is the largest. On open ground the sky is only 11–14% of daylight (T1.1's sky-to-sun ratios: Sponza 11.0%, furnished room 13.2%, open ground 14.3%), and in a sunlit interior the sun's bounce is often most of the GI. So how time of day looks on Sponza depends mostly on how fast re-tracing is (6.4), with or without the instant sky. An instant sun direction needs a different idea; it's on the Later list.

### 6.2 Re-baking the sky (needed by every path)

**Today:** a `procedural` environment is built on the CPU (Preetham, calibrated with the Ineichen–Perez ratio; `calibrated-sky.ts`) and uploaded. One workgroup then resamples it to the 256 × 128 radiance map and projects it to 18 SH coefficients. That runs only when `setEnvironment` is called. The projection kernel takes about 1 s to compile (T1.9a), and the CPU build isn't meant to run every frame.

**M2:**

- **A GPU procedural source.** The resample kernel evaluates Preetham directly (a new `SKY_SOURCE_KIND`). The calibration scale is computed on the CPU per sun direction (a few scalar formulas) and passed as a uniform. The CPU code stays: it writes the shared sky files for Cycles and native, and a test holds the GPU map to it (mean within 0.1%, every texel within 1%).
- **The SH projection works in two levels:** many workgroups write partial sums, and one adds them up. Subgroups first, with the existing workgroup-memory fallback. On a procedural sky it integrates over the 256 × 128 map itself. It also writes the sky's **radiance** SH, bands 0–2, which the instant sky needs. Those are the same sums before the cosine weights are applied (`projection-wgsl.ts`), so they cost nothing extra.
- **When it re-bakes:** when the sun moves more than 0.05°, or turbidity, intensity or `sunIlluminance` change; at most once per frame. Estimated GPU cost under 0.1 ms per re-bake; CPU under 0.05 ms.
- **The slow compile:** T2.6a finds why the projection takes about 1 s to compile (perhaps the unrolled 18-term basis inside the loop) and gets it under 0.2 s. That also helps startup (DECISIONS: "find the slow sky kernel").
- **The sun** is a directional light, a separate object (light buffer, shadow rays). Keeping the sun light and the sky's sun in step is the host's job; the viewer's slider moves both. The three adapter already maps pmndrs/sky and `SkyMesh` (three.md, "The sky").

### 6.3 The instant sky: a first-order sky transfer

**The idea** (PRT, Sloan, Kautz, Snyder 2002; Far Cry 3's probes, Gilabert and Stefanov, GDC 2012), kept to the part that's cheap to build:

- Any sky is close to a weighted sum of 9 fixed patterns, the SH basis functions of bands 0–2: `sky(ω) ≈ Σₖ cₖ Yₖ(ω)`, with colour weights `cₖ` (rgb). 6.2's projection computes them.
- What a probe texel sees **directly** of the sky, through openings, is `Σₖ cₖ Tₖ`. A texel's **first-order transfer** `Tₖ` is the pattern `Yₖ`, seen through the probe's visibility, with the same cosine weighting the texel uses for irradiance.
- `Tₖ` has no colour: nothing sits between the probe and the sky. That makes it 9 numbers per texel, not 9 colours, and it needs no bounces, so it doesn't depend on any other probe.
- The sky's **bounces** (sky light off the courtyard floor, the walls) aren't in `Tₖ`. They're handled by a rescale and by re-tracing (below).

**Building it, from the field's own rays.** No extra rays and no new kernel:

- In `ddgi/blend`, each texel thread already loops over the probe's rays with their directions and distances (in workgroup memory). For every ray that **missed** (escaped to the sky), it also adds `Yₖ(direction)` for the 9 patterns, with the same texel weight it uses for irradiance. Rays that hit anything add 0. The 9 polynomial terms are computed inline from the direction, so the transfer adds no workgroup memory (section 5 keeps the layers at 13.5 KB).
- **It must count rays exactly as the irradiance does,** or `Σₖ cₖ Tₖ` won't match the sky part of `S`:
  - it's normalised by the same sum of weights over all blended rays (`blend.ts`), so back-face hits count, with 0;
  - the 32 fixed rays are left out, as they are from irradiance;
  - "missed" uses the trace's own test (`signedDistance` equal to the miss value, `trace.ts`), so a ray that ends at a finite `maxRayDistance` counts as sky in both;
  - the firefly clamp applies to `S` and not to `T`. In steady state that changes nothing; right after a bright change it can make `S` lag `Σₖ cₖ Tₖ` for a few updates.
- The transfer keeps its own count `m` (8 bits in history word 2, beside `owed`) and blends with its own `h = min(0.97, m / (m + 1))`. It goes on updating with every probe update: that costs only the blend's extra work, and it follows moved geometry by itself. Light changes never shorten `m`. It's reset to 0 when geometry near the probe moves (M3, below), when the probe goes from off to on (state 3, "not updated yet", where `S` restarts too), and when relocation moves the probe by more than 10% of a spacing.
- A probe's transfer counts as **built** once `m ≥ 8` (about 1,800 rays). T2.6b measures its noise at 8 updates against 64 and raises the count if it has to.

**How long building takes.** A transfer is built after a probe's 8th update, so it is never later than the probe's own irradiance.

- **With the warm-up,** every probe gets 16 updates. Today the viewer runs it (up to 16 large passes). T2.4b moves it into the technique, **on by default**, so every host gets it.
  - Sponza's warm-up took about 2.3 s on the RTX (T1.9's cold start: first GI at 10.8 s, settled at 13.1 s). So transfers are built about halfway through it, around **1.2 s**.
  - Two-room's whole warm-up took 0.08 s.
- **With the warm-up turned off, at the default budget,** an unbuilt probe counts as owed one update, at a low urgency (0.05, below any real change) times its camera weight. So the view builds first.
  - The view's ~200 nearest probes need `200 × 8 / (0.75 × 268)` ≈ 8 frames, about **0.13 s** on Sponza.
  - **The whole volume takes about 10 s** on Sponza: 8 sweeps of 1.2 s. That's the same time the field itself takes to converge without a warm-up, which is why the warm-up is on by default. A host that turns it off gets an unconverged far end of Sponza for about 10 s, with or without the instant sky. The view is fine after 0.13 s.

**The instant step is a nudge to the ordinary sky layer.** The probe's sky layer `S` (section 5's environment layer: what the probe stores from the sky) stays the real field. With the instant sky on, emissive surfaces go in the lights layer, so `S` holds only the sky. When the sky's weights change from `c` to `c'`, a pass computes, per texel and channel:

1. **the direct part:** `Δdirect = Σₖ (c'ₖ − cₖ) Tₖ`;
2. **everything else:** `B = max(0, S − Σₖ cₖ Tₖ)`, what `S` holds beyond the 9-pattern sky seen directly. That's mostly the sky's bounces, but it also holds the part of the directly seen sky that bands 0–2 miss: miss rays read the full 256 × 128 radiance map (`giSkyRadiance`), not the SH;
3. **the sky's brightness change:** `ρ = c'₀ / c₀` (band 0, the sky's mean, per channel), clamped to [0.25, 4] for one nudge;
4. **the nudged sky layer:** `S' = max(0, S + Δdirect + B × (ρ − 1))`. The clamp at 0 is there because a truncated SH sum can dip below 0 where the sky is dark.

- *Why rescaling `B` is a fair guess:* bounced sky light is the sky's light blurred over the surfaces around the probe. When the sky reddens, brightens or dims as a whole, its bounce follows nearly in proportion.
- *What the rescale gets wrong, left for re-tracing:*
  - the bright side of the sky moving with the sun. SPEC's HomeFig rule 2 says it: time of day changes the sky's shape, not just its brightness. That's why `S` is still re-traced and the rescale is only the instant guess;
  - the directly seen detail that bands 0–2 miss (near the sun's glow, behind a window), which changes with the sky's shape too. Where it's negative, `B`'s clamp at 0 drops it. On open ground bands 0–2 already missed irradiance by up to 0.87% in M1 (DECISIONS: the platform's sky added band 4); behind windows it's more.
- T2.6b measures the nudge with the direct part alone and with the rescale, and keeps the rescale only if it helps.
- *Why clamp `ρ`:* near night `c₀` is close to 0, and a large ratio would multiply the noise in `B`. Once nudged, a probe's sky weights become `c'`, so a clamped rescale is never made up later. `ρ` is the same for every probe (it's the sky's), but the sky's SH is computed on the GPU (6.2). So `ddgi/mark` reads the old and new SH itself and, when `ρ` was clamped, marks every probe at the full `r` instead of the residual share below. No readback.

**Everything else about the nudge:**

- **`S` stays the real field.** Normal updates keep blending the traced sky into `S` with the probe's history and the firefly clamp, as today. So after the sky stops changing, `S` converges to exactly what it would without the instant sky. The nudge only decides how much of a change shows at once.
- **Bounces read `S`.** A ray hit reads `S` for the sky's bounce, as any layer reads its own (section 5). That is one fetch, as today.
- **The fallback is "no nudge".** A probe whose transfer isn't built (or a volume with the instant sky off) simply isn't nudged and follows by re-tracing (6.4). Nothing switches and nothing jumps.
- **Marking after a sky change:** every probe is marked as for a global change (6.4). A probe with a built transfer gets `r × skyResidualShare` (default 0.3 until T2.6b measures what the nudge leaves), so the budget goes first to probes the nudge couldn't cover.
  - *This matters less than it looks during time of day.* Every sun move re-bakes the sky too, and the sun's direction marks every probe in full. History `n` is per probe and shared by its layers, so the sky layer's history is shortened anyway. The share helps only for sky-only changes (turbidity, a host's own sky).
  - *The shared history has a cost:* a sun mark also makes the next updates of the unchanged lights layer noisier. If T2.4a's two-room step shows that, history becomes per layer (one more f32 per layer per probe).
- **Layout.** Layers aren't more array layers of the atlas: it already uses one array layer per probe row (ddgi.md), and the default limit is 256. So **layers are blocks side by side inside each array layer.** Layer `L`'s tile for probe `(x, z)` sits at `((L × nx + x) × 8, z × 8)`. Light groups (section 5) use the same layout, with no new textures and no new storage bindings. `prepare` checks `nx × 8 × blocks ≤ 8192` and throws `GILimitError` past it. Bistro exterior's camera volume (32 wide, 6 blocks) is 1,536 texels wide.
- **The transfer's 9 numbers per texel** sit in 3 `rgba16float` blocks (12 channels, 3 spare). Transfer values can be negative; halves hold that.
- **The composed atlas** the renderer samples is its own texture. A texture can't be read and written by the same dispatch, and the shading module still binds just one irradiance texture.
- **Writing the nudge** has the same problem (it reads `T` and `S` and writes `S`), so it works like blend and write. `ddgi/sky-nudge` writes the nudged `S` tiles to a nudge scratch buffer (288 bytes per probe). Then a second `ddgi/write` dispatch, in "sky" mode, copies them into the atlas for every probe whose transfer is built (one workgroup per probe; the others exit at once).
- **The order in a frame:** the nudge and its write run **before** `ddgi/trace`. So a probe that is also updated this frame blends from its already-nudged `S`, and the nudge is never lost to the blend writing an older `S`.

**Showing it.** In a frame where the sky changed by more than the re-bake threshold:

1. `ddgi/sky-nudge` handles every probe with a built transfer (before `ddgi/trace`);
2. a `ddgi/write` dispatch in sky mode copies the nudged tiles;
3. `ddgi/compose` writes the composed atlas from the layers:
   `composed = Σ light-group layers × weights + sun layer × sun weight + S × environmentWeight`.

The change shows in the same frame. In other frames compose writes only the tiles that were updated.

**The sun layer at a white reference departs from SPEC's HomeFig rule 3** ("bake each group in its real colors … not in white"). For a single light the result is the same, because each colour channel is linear on its own. Baking in the sun's real colour would break it: probes updated after a colour change would hold the new colour and the rest the old one, so one layer weight couldn't serve both. The one non-linear step, the firefly clamp, needs no change: it limits each update relative to the stored value (`maxBrightnessChange × old`, `blend.ts`), so it clamps the same at any scale. Its only absolute term is a tiny floor (the smallest normal half, about 6e-5), which is negligible. - **What others do** (for the DECISIONS exception): Unity's Adaptive Probe Volumes store sky occlusion without colour and multiply it by the current ambient probe at run time (`ProbeVolume.hlsl:711-735`), as PRT does. RTXGI and Godot's SDFGI store real colour and re-trace instead. Dennis: "go with what we need to get to the improvement passes but i would look at what other godot systems do". So the sun layer stays white, with three conditions:
  1. **the sun layer's bounces read only the sun layer** (HomeFig rule 1), so it stays one light's response, scaled as a whole;
  2. **the weight is applied in linear space.** Our atlas stores linear `rgba16float` (no gamma), so composing is a plain multiply. If an encoding is ever added (RTXGI blends with gamma 5), the weight goes on after decoding;
  3. **a change of the sun's direction still re-traces** (6.4). Only colour and intensity are a weight.
- This is recorded in DECISIONS as an exception to rule 3 (Dennis, 2026-11-05).

**Memory.** Per probe: the transfer's 3 blocks, the sun layer, the sky layer and the composed atlas (512 bytes each: `rgba16float`, 6 × 6 texels plus border), plus the nudge scratch (288 bytes): 3,360 bytes. The blend's scratch tile grows with the transfer too, by about 1.4 KB per probe in a round (about 1.5 MB at 1,024-probe rounds). That is less than half of the earlier design's, which needed a colour per pattern because it carried bounces.

| Scene (probes) | Extra GPU memory |
| --- | --- |
| Two-room (1,071) | 3.6 MB |
| Furnished room (4,998) | 16.8 MB |
| Bistro exterior's camera volume (12,288, m3-moving.md) | 41 MB |
| Sponza (19,425) | 65 MB |

- **Sized for the target machines** (Dennis: "mid to high end machines are the goal"). The RTX 3060 Laptop has 6 GB of its own, a desktop RTX 3060 12 GB, and the M5 Pro test Mac shares 64 GB with its GPU. Sponza's 65 MB is about 1% of the laptop's 6 GB.
- So the instant sky is **on by default.** A guard turns it off, with a line in the stats, only past `skyTransferMaxMB` (default 512 MB), which no test scene comes near. The earlier 128 MB and 48 MB caps are gone.
- **The furnished room** (everything the core owns): T1.9 measured 40.7 MB. With the instant sky (16.8 MB), the ray scratch for three layers (up to +4.2 MB) and the blend scratch (+1.5 MB), that's about 63 MB. SPEC now keeps its 64 MB as a guide, not a gate (revised 2026-11-05), so T2.6b reports it and nothing turns off.

**What happens when geometry moves (M3).** A moved object changes what the sky can reach: a door closing cuts the dark room off from the window. M3's `ddgi/mark` (m3-moving.md section 5) marks probes near moved geometry. Their transfer count `m` is reset to 0 with their history. Their transfer re-accumulates from the updates they're owed anyway, and is built again after 8 of them. Until then they aren't nudged, and sky changes reach them by re-tracing. `S` itself is updated by the normal blend all along, so nothing jumps. **In a camera-following volume** entering probes get their 2 warming updates first; the other 6 come at the low building urgency. With the camera still, the priority share is free and they come within about 1 s. During a fly-through the entering planes take part of the priority share (about 43 of Bistro's 132 a frame at 5 m/s, m3-moving.md), so the other 6 come mostly from the round robin (a quarter of 132, about 33 a frame, one pass per 6.2 s): about 37 s. Most entering probes reach the camera before their transfer is built, so a sky change during a fast fly-through reaches them by re-tracing. That's accepted, because their irradiance is already right; only the instant step is missing.

**Cost** (RTX 3060, estimates, to be measured GPU-bound):

| | GPU cost |
| --- | --- |
| Accumulating the transfer (always, while the instant sky is on) | the blend does about +50–80% more work. On Sponza, back to back, the blend is about 3.5% of a probe update (T1.9, 20 probes: 0.05 of 1.44 ms), so about **+2–3% per probe update**; more on small scenes, where the blend is a larger share |
| Its three lighting layers (lights, sun, sky) | each probe update about +15–20% (section 5's estimate for 3 layers) |
| A frame with a sky change: nudge, write and compose of every tile | Sponza ~0.3 ms. It moves about 4.3 KB per probe, 83 MB in all, so at the laptop's 336 GB/s it can't be under 0.25 ms. Furnished room ~0.08 ms |
| The same, with far probes a quarter per frame | Sponza ~0.1 ms |
| Sky still | the accumulation only |

- **Far probes a quarter per frame:** on volumes above 8,000 probes, probes with a camera weight under 0.25 are nudged a quarter per frame (probe index mod 4). Each quarter keeps the sky weights it was last nudged with, so its nudges still add up exactly, and the far end of the volume shows a sky change within 4 frames. Near probes are nudged every frame.
- **Under Budget v2** the probe count doesn't shrink while frames are on time, so the nudge adds to the update's GPU time in the frames where the sky changes. On Sponza the calibrated batch (about 272 probes) is already near 2 ms, so a sky-change frame relies on Budget v2's +10% margin (0.2 ms), which the far-quarter nudge fits and the every-tile nudge doesn't. T2.6b checks it at the calibrated count.

**Accuracy: measured first** (T2.6b's first check, before the rest is built).

- Right after a 30° move of the sun, with the sun layer's weight at 0 so only the sky counts, how much of the change between the old and the settled field (GI only) does the nudge remove in the same frame? Measured on the furnished room's window bookmark and Sponza's bookmark 1, with the direct part alone and with the rescale.
- **Decision rule, set before measuring:** keep the instant sky if it removes at least 70% on both bookmarks. Otherwise drop it. The sun layer and re-tracing don't depend on it, so nothing else changes.
- If bands 0–2 fall short, try bands 0–4 (band 4 is what fixed the platform's sky irradiance in M1): 25 numbers per texel, 7 blocks, about +40 MB on Sponza. For radiance seen directly, band 3 matters too, so it's 0–4, not 0–2 plus 4.

### 6.4 Re-tracing, nearest the camera first

It's today's field, driven by M2's change-driven updates (section 4). It's what runs:

- for the sun's direction, always;
- for what the sky's nudge leaves, and for probes whose transfer isn't built.

A change of the sun's direction (or of the sky) marks every probe (section 4.2). For a big step, all N probes need K updates, about `N × K / U` frames of the whole budget. Camera-near probes come first: with the camera weight, "near" means within about 4 spacings and in view, roughly the 100–300 probes that dominate the image.

**Far probes owe fewer updates after a global change.** (The main path is now key probes and propagation, 4.5; this is its fallback, and the table below shows both.) At K = 8 for every probe, the whole of Sponza takes about 10 s, which Dennis calls a fail for a real-time game ("8s to update a scene in a realtime game would be considered a fail"). So after a global change:

- probes with a camera weight of 0.5 or more (near and in view) owe 8;
- the rest owe 2.

This applies only to volumes where 8 updates for every probe wouldn't fit in 2 s at the current rate (`N × 8 / U > 120` frames): Sponza and Bistro's camera volume, not the rooms.

*As built (T2.4b):* with the bounces counted (4.1), two updates leave far probes about ρ² of their error (64% at ρ = 0.8); on Sponza after a 30° sun turn the field's light-weighted error is within 5% in 0.8 s and 1.7% at 3 s, while each far probe stays about 4 points noisier than before the step for a while (12.6% against 8.4%, per-probe relative).

- Two updates bring a far probe within 5% (that's what the history rule in 4.1 gives for K = 2), but it's noisier: it holds about 2 updates' rays instead of 8. Its history then grows by itself with the round robin.
- When the camera turns toward it, its camera weight raises its priority as usual.

| Scene | Probes | U (RTX, the rates section) | View within 5% | Whole volume, K = 8 everywhere | Whole volume, far probes owe 2 |
| --- | --- | --- | --- | --- | --- |
| Cornell | 1,728 | 1,574 | ~0.1 s | ~0.15 s | not needed |
| Two-room | 1,071 | 707 | ~0.13 s | ~0.2 s | not needed |
| Furnished room | 4,998 | not measured | not measured | not measured | not measured |
| Bistro exterior's camera volume | 12,288 | 132 | ~0.27 s | ~12.4 s | **~3.3 s** |
| Sponza | 19,425 | 268 | ~0.13 s | ~9.7 s | **~2.5 s** |

- **What the camera sees settles in 0.1–0.27 s** on every scene, well inside 2 s.
- **The whole volume with far probes owing 2:** about 2.5 s on Sponza and 3.3 s on Bistro's camera volume, over 2 s, though only out of view. **With 8 updates for every probe it's 10–12 s, which fails Dennis's bar.** Key probes and propagation (4.5) bring Sponza to about 1.0 s, and Bistro's camera volume (17 × 7 × 17 = 2,023 keys) to about 1.6 s.
- **Far probes owing 2 are noisier** for a while: about 448 rays each instead of about 1,800. Their history grows back with each sweep (every 4 s).
- **These are for a big jump** (dragging the slider across hours). Dennis agreed (2026-11-05) to judge global changes on the view.
- **A slow time-of-day animation** (an hour of sun in 10 s, 1.5° per second) changes each probe only a little between two of its updates, so the step rule in 4.1 wouldn't shorten its history, and with 32 updates of history it would lag far behind. Section 4.1's **steady-change rule** handles this: a probe that keeps changing gets a history short enough that its lag stays near 5%.
  - In two-room (every probe updated about every 1.5 frames) T1.12's count would reach its cap of 200 updates, about 5 s of lag. The steady-change rule holds it near 70 updates, about 1.8 s, which keeps the error near 5%.
  - Far from the camera on Sponza, out of view, a probe is reached only by the round robin's quarter of the budget (about 67 probes a frame), so about every 5 s. The sun moves about 7° in that time, so the history drops to about 1 update: those probes step from one estimate to the next about 5 s apart, with more noise. Near the camera the scheduler updates probes far more often.
- **It blends, it never goes black.** Every probe moves from its old value toward its new one through its history (section 4.1). In view, that shows as the old lighting fading into the new. The exception is the one above: during a fast day animation, far, out-of-view Sponza probes have so little history that they step from one estimate to the next.

### 6.5 Re-tracing alone, or with the instant sky: the comparison

On Sponza, on the RTX, at T1.9b's 268 probes a frame:

| | Re-tracing near the camera, with the sun layer | The same, plus the first-order sky transfer |
| --- | --- | --- |
| Sun colour and brightness | same frame, exact (the sun layer) | same |
| Sun direction | view within 5% in ~0.13 s; whole volume ~1 s at the key level (4.5), every probe traced in ~2 s | the same |
| Sky change, in view | within 5% in ~0.13 s | most of it in the same frame (T2.6b measures how much); the rest ~0.13 s |
| Sky change, whole volume | ~1 s at the key level, ~2 s every probe | most of it within 4 frames; the rest as on the left, at lower priority |
| Build | none | from the field's rays: ~1.2 s into the warm-up; with the warm-up off, the view in ~0.13 s and the whole volume in ~10 s, as the field itself converges |
| Extra memory | 20 MB (the sun layer and the composed atlas) | 65 MB |
| Extra GPU cost | the sun layer (about +8–10% per probe update) | also +2–3% per probe update, and ~0.1 ms in frames where the sky changed |
| New code | the layers (shared with light groups) | also the blend's accumulation, `ddgi/sky-nudge` and `ddgi/write`'s sky mode |

**Recommendation: both, in that order.**

1. **Re-tracing nearest the camera is the main path,** with far probes owing 2 updates after a global change, and the warm-up on by default. The sun's direction is most of time of day, and only re-tracing moves it. T2.4b and T2.6a build it.
2. **The sun layer goes with it:** exact, instant, and nothing to build.
3. **The first-order sky transfer is added in T2.6b,** on by default, as the instant part of the sky. It meets Dennis's bar: it builds from the field's own rays, never later than the field itself, about 1.2 s into Sponza's warm-up (on by default). Its memory is small for the target machines.
4. **It stays only if its first check passes** (it removes at least 70% of a sky change in the same frame). If it doesn't, it's dropped, and the rest stands unchanged.

**With the measured rate, the instant sky buys less than it did.** Re-tracing alone now settles a sky change in view in about 0.13 s (8 frames), so the transfer's gain in view is those 8 frames. Its larger gain is the far end of the volume, out of view (same frame against ~2.5 s). T2.6b's first check should weigh that against its 65 MB and its code; if the gain is too small to see, drop it.

This keeps "instant with a delayed fallback" where instant is cheap. It drops the full transfer with bounces from the earlier draft. That transfer needed 64 updates per probe because each probe's bounce read its neighbours' transfers: 17–25 minutes on Sponza at the old controller's rate (still about 3 minutes at 272 probes a frame), and 125 MB.

*Why not a separate pass of extra visibility rays to build the transfer faster:* it isn't needed. The field's own miss rays give the same transfer for free, and the field needs those updates anyway.

### 6.6 Native

The native harness's miss shader sees one flat colour (DECISIONS 2026-11-04). M2 gives it the texture sky and an animated sun (T2.7), so time-of-day comparisons are like for like. Native has no instant sky; it re-traces every probe every frame with hardware ray tracing.

## 7. The viewer

- **Gizmos:** three's `TransformControls` on light helpers: drag to move, rotate spots. Selecting a light opens its panel.
- **Sliders and toggles:** intensity (log scale), colour, range, cone and penumbra for spots, shadows on or off, group. Toggling sets intensity to 0, so three doesn't recompile its materials (NOTES: removing a three light recompiles).
- **Group dimmers:** a weight slider and a colour per group, plus the environment layer. The hover help says "instant: no update needed".
- **Time of day:** hour and day-of-year sliders. They drive the sun's direction (a simple solar-position formula), the scene's sun light and the GI sky together. A play button runs a 10-second day.
- **Scripted motions** from the manifest: `motions: [{ id, target: 'light:<id>', keyframes: [{ t, position, intensity, color }], loop }]`. Two come built in for each lit scene:
  - **light sweep:** a lamp moving along a path at 1 m/s (two-room: across the lit room and through the doorway; Sponza: along the lower colonnade);
  - **step:** the same lamp jumping between two keyframes every 3 s, for measuring response.
- **Response meter,** measured in **GI-only mode** (so three's own direct light, which changes instantly, doesn't count). Pick a screen rectangle. The viewer plots its mean per frame, marks "first frame that changed by more than 1%" and "first frame within 5% of settled", and writes the curve to CSV. "Settled" is the mean after 4 s with the budget raised to 16 ms (a test mode, labelled as such).
- **Native side by side:** the native harness plays the same manifest motion (T2.7) and records the meter's curve and one frame in four. A compare page plays both recordings in sync, with both curves on one plot. Showing native live inside the viewer isn't attempted.
  - Dennis (2026-11-05): **the look is judged by eye.** Our response time is measured and recorded next to native's, but matching native is **not** a requirement: native traces with hardware ray tracing ("we know we wont be anything near native without the hardware access").
- **Stats** gain: probes updated by priority and by round robin, probes owed, shadow rays per frame, `gi/direct` ms, compose ms, sky re-bake ms, light count.

## 8. Later (after M2, not abandoned)

**Area lights** (carried from M1; Dennis: after M2, "lets not abandon them").

- *Why:* so an emitter's direct light can be analytic and DDGI holds only its bounce (DECISIONS 2026-11-05, the viewer entry).
- *Why not in M2:* three's `RectAreaLight` casts no shadows, and SPEC's HomeFig lessons say to avoid area lights without shadows. Until then M1's split stays: DDGI lights the emitter.
- *How they'd fit:*
  - **The light record:** a `rect` type in the same 64 bytes. The position, the normal in the direction word, the intensity, the half-width and half-height in the two cone words, and the in-plane axis, octahedral-packed into the radius word.
  - **Shadow rays** aim at a uniform random point on the rectangle, with a cosine at both ends. `gi/direct` and the light grid treat it like any local light, bounded by its rectangle instead of a point.
  - **No double counting:** the emitter's surface is flagged "light carried by a light", so GI rays that hit it see no emission.
  - **The host** needs shadowed direct light for it (three would need a shadowed area light), or it keeps emissive-only lighting as now.

**Spotting three.js-style group dimming** (section 5): turn "every light of a group scaled by the same factor" into a group weight change.

**An instant sun direction** (6.1): keep every probe's ray hits (surface records) between updates, and on a sun move relight them with new shadow rays instead of re-tracing them. HomeFig relit 512 cached probes for the sun in about 1 ms (RTX 4090, SPEC). It costs the surface records for every probe (24 bytes × 224 rays, about 5 KB per probe, 104 MB on Sponza), and the cached ray directions stop rotating, which needs care.

## How it fits the existing kernels (no parallel GPU paths)

| Kernel | M1 | M2 |
| --- | --- | --- |
| `ddgi/mark` | none | new; only in frames with changes |
| `ddgi/schedule-*` | update list from the CPU cursor | the GPU scheduler writes the same buffer; the CPU cursor is removed, not kept beside it |
| `ddgi/trace` | trace, shade, direct light inline | trace, shade, bounce per layer, surface records; **no `giDirectIrradiance` call** |
| `gi/direct` (platform) | none | new; takes over direct light for every technique that traces |
| light grid build (platform) | none | new; only above `exactLightLimit` local lights |
| `ddgi/blend` | one global hysteresis | per-probe `h` from the history record; loops layers; writes the probe's mean; accumulates the first-order sky transfer from its miss rays |
| `ddgi/compose` | none | whenever there's more than one layer (the sun layer is on by default); over the update list and propagation's list |
| `ddgi/propagate` | none | new; only in frames where key probes were updated because of a change (4.5); its tiles go through `ddgi/write` and `ddgi/compose` over its own list |
| `ddgi/sky-nudge` | none | new; only in frames where the sky changed, with the instant sky on |
| `ddgi/write` | one block per probe | loops the layer blocks (the transfer's 3 included); a second dispatch in sky mode writes nudged sky tiles |
| sky resample and projection | CPU-built procedural sky, one-workgroup SH | GPU procedural source, two-level SH, plus radiance SH for the instant sky |

The transfer is built inside `ddgi/blend` from rays the field traces anyway, so there's no transfer kernel and no extra rays.

The instant sky is a nudge to the ordinary sky layer, and its fallback is simply no nudge. So there's no second sky path.

Every new kernel that reduces or compacts is subgroups first, with one declared fallback (per-thread or workgroup atomics), and a test runs both on the same input (the M1 rule).

**Storage buffers per kernel** (the default limit is 8):

| Kernel | Count |
| --- | --- |
| `ddgi/trace` | tracer 3 + scene copy and shading 2 + update list + ray scratch = **7** |
| `gi/direct` | tracer 3 + scene copy and shading 2 + ray scratch + light grid = **7** |
| `ddgi/blend` | update list, ray scratch, scratch tile, census, history, feedback requests = **6** (plus the delta buffer when keys propagate: **7**) |
| `ddgi/propagate` | history, delta buffer, update-list stamps, its list, its scratch = **5** (the atlases as textures) |
| `ddgi/write` | update list + scratch tile = 2; in sky mode, history + nudge scratch = 2 |
| `ddgi/mark` | light-change table, history = 2 (atlases as textures) |
| `ddgi/schedule-*` | history, update list, schedule state, feedback requests (read and cleared) = 4 |
| `ddgi/sky-nudge` | history + nudge scratch = 2 (the atlas as a texture; the sky weights as uniforms) |
| `ddgi/compose` | update list, propagation's list = 2 (the atlas as a texture, the composed atlas as 1 storage texture, the weights as uniforms) |

## Cost against the budgets (estimates, RTX 3060, to be measured GPU-bound)

| Item | When | GPU ms |
| --- | --- | --- |
| `ddgi/trace` + `gi/direct` on Sponza | always | per-probe part about 0.6× of M1's `ddgi/trace`; the fixed part unknown (T2.0) |
| Scheduler (4 dispatches) | every frame | < 0.05 |
| `ddgi/mark` | frames with changes | < 0.02 for a few lights; < 0.05 for a global change at 20k probes |
| Light grid build | above `exactLightLimit` local lights | ~0.05 on Sponza with 1,024 lights |
| `ddgi/compose` | groups on | updated tiles only: < 0.01; a weight change on Sponza: ~0.1 |
| Sky re-bake | the sky changed | < 0.1 |
| Instant sky: nudge + write + compose | the sky changed | Sponza ~0.3 for every tile (about 83 MB moved; at least 0.25 ms at the laptop's 336 GB/s), ~0.1 with far probes a quarter per frame; furnished room ~0.08 |
| Instant sky: accumulating the transfer in the blend | while it's on | each probe update ~+2–3% on Sponza |
| Time of day's lighting layers (lights, sun, sky) | on by default | each probe update ~+15–20% (section 5's estimate for 3 layers) |

Under Budget v2 the probe count is set by the calibration at load, which measures the `ddgi/*` passes and `gi/direct` together with whatever is on (layers, the transfer). So the costs that run every frame are paid for in a smaller calibrated batch. Costs that come and go (`ddgi/mark`, a sky change's nudge) don't change the count while frames are on time; they add to that frame's GPU time, and T2.4b and T2.6b check that the update stays within its 2 ms ceiling. Shading cost doesn't change.

Memory: history 32 bytes per probe, ray scratch at most 7.6 MB (11.8 MB with three layers), light grid at most 2 MB, light groups and the instant sky as tabled (Sponza +65 MB for the instant sky, on by default).

## Proposed task breakdown

Board format. IDs are proposals; T1.11 assigns the real ones. Every GPU check says which adapter and uses the bench lock (with Dennis's GPU apps closed for p95 and timings, DECISIONS 2026-11-05).

- **Speed checks** run on the RTX 3060, GPU-bound (the method the M1 lead's re-measure settles), and on the M5 Pro once it's set up.
- **The Intel UHD runs the correctness checks only** (oracle, furnace, accuracy, leaks, the subgroup fallback). Its timings may be recorded, never gated.

| ID | Task | Agent | Size | Depends on |
| --- | --- | --- | --- | --- |
| T2.0 | Spike: deferred shadow-ray pass on Sponza | implementer | S | M1 done |
| T2.1 | Light buffer v2, manifest `motions`, `sponza-candles` | implementer + wrangler | M | none |
| T2.2 | Deferred direct light (`gi/direct`) in platform hit shading | implementer | M | T2.0 |
| T2.3 | Many-light sampling: the light grid | implementer | M | T2.1, T2.2 |
| T2.4a | The change-driven history on T1.12's record, and `ddgi/mark` | implementer | M | T2.1, T1.12 |
| T2.4b | GPU scheduler with camera priority, the feedback rule (4.6); far probes owe 2 as the fallback; the warm-up moves into the technique | implementer | M | T2.4a |
| T2.4c | Key probes and propagation (4.5), with the instant sky's nudge pair shared | implementer | M | T2.4b |
| T2.5 | Light groups: layers, compose, weights, adapter API | implementer | M | T2.2, T2.4a |
| T2.6a | Sky re-bake: GPU procedural source, two-level SH with radiance SH, compile fix | implementer | M | T2.4b |
| T2.6b | Instant sky: the first-order transfer in the blend, the nudge, fallback (the sun layer comes with T2.5's layers) | implementer | M | T2.4c, T2.5, T2.6a |
| T2.7 | Native harness: manifest motions, texture sky, animated sun | native-runner | M | T2.1 |
| T2.8 | Viewer: gizmos, sliders, groups, time of day, motions, response meter, native compare page | implementer | M | T2.4b, T2.5, T2.6b, T2.7 |
| T2.10 | M2 measurements against targets | gpu-tester, native-runner | M | all above |
| T2.11 | M2 note and M3 breakdown | main session | S | T2.10 |

T2.9 is left free: area lights are on the Later list (section 8).

**Order (Dennis, 2026-11-06: dynamic updates first).** T2.4a → T2.4b → T2.4c come first, prototyped on Sponza's sun step and two-room's lamp before anything else in M2 is built on them. T2.0 (the deferred shadow pass) runs beside them; it changes how many probes a frame fit, not which. Time of day (T2.6a, T2.6b) follows T2.4c. If Dennis wants moving objects sooner, M3's change feed (T3.1) and geometry marks (T3.4, rigid objects only) can run right after T2.4c, because they reuse the same scheduler.

The startup items DECISIONS moved to M2 ("fast first GI", the scene copy off the main thread, compiling less) are designed in [m3-moving.md](m3-moving.md), because they share the BVH and scene-copy machinery. Their rows (T2.12–T2.14) are listed there. The target is ≤ 2 s from the hand-off to GI with a cold compile (Dennis, 2026-11-05); file loading is the host's, and is on m3-moving.md's Later list.

### Draft done-whens

- **T2.0 Spike: deferred shadows.** A throwaway branch splits Sponza's `ddgi/trace` into trace + `gi/direct` with surface records.
  - Done when: the fixed and per-probe cost of the pair (from a batch-size curve, 16 to 512 probes, GPU-bound) and the shadow-ray rate against the bench's any-hit rate are recorded on the RTX (3 runs); and the field matches M1's within 1% mean on Cornell and Sponza after 300 frames, on the RTX and the Intel UHD.
  - Decision rule, set before measuring: go ahead with T2.2 if Sponza fits ≥ 1.25× the probes at 2 ms on the RTX. Otherwise record why, and the main session decides.
  - Show: the cost curves. Record: `timing.csv` `batch_cost/` rows.
- **T2.1 Light buffer v2.**
  - Done when:
    1. creating 4,096 lights and destroying them in random order creates 0 pipelines (counter);
    2. a test round-trips every `LightChanges` field for each light type through `SceneChanges.lights`;
    3. the adapter's still-frame CPU time on `sponza-candles` is ≤ 0.2 ms (median of 300 frames, RTX laptop);
    4. `sponza-candles` fetches and verifies with the subset Dennis picks (licence, size, triangle and light counts, checksum).
  - Show: `sponza-candles` in the viewer. Record: light and triangle counts in PROGRESS.
- **T2.2 Deferred direct light.**
  - Done when:
    1. direct irradiance through `gi/direct` matches the inline `giDirectIrradiance` on T1.4's oracle points (mean within 0.5%), and sun-on-plane matches E·cos θ within 0.5%;
    2. the subgroup and fallback work lists give the same field (mean within 0.1% after 300 frames);
    3. Sponza's probes per frame at 2 ms reach T2.0's measured gain within 10%;
    4. the furnace, Cornell and two-room checks of T1.5 still pass.
  - Record: `timing.csv` rows; the shadow-ray rate.
- **T2.3 Many-light sampling.**
  - Done when, on a test scene with 1,024 point lights:
    1. direct irradiance from `gi/direct`, averaged over 1,024 frames, matches an exhaustive evaluation within 1% mean, with 99% of 1,000 test points within 5%;
    2. `sponza-candles` costs ≤ 1.3× per updated probe compared with `sponza` lit by the sun only;
    3. the grid build takes ≤ 0.1 ms on the RTX.
  - Record: noise and cost at K = 1, 2, 4.
- **T2.4a Per-probe history and `ddgi/mark`.**
  - Done when:
    1. on a still scene, after 600 frames the field matches M1's (T1.12's rule) within 0.5% mean, and flicker is within 10% of T1.12's;
    2. on two-room in a test mode that updates every probe every frame (budget 16 ms), moving the lamp 2 m (the step) gets within 5% of settled in ≤ 0.5 s (response meter, GI only, `lit-room` bookmark), so this tests the history without the scheduler;
    3. a still scene marks 0 probes over 600 frames (counter);
    4. marked probes against probes whose true change was > 5% (from a re-converged field) are recorded, with a lit-room recall of at least 95%.
  - Record: `lightChangeScale` chosen, and the counts.
- **T2.4b GPU scheduler.**
  - Done when, at the default budget, in GI-only mode, on the RTX:
    1. two-room and Cornell (point light) step changes show the next frame (ROI changes by > 1%) and are within 5% of settled in ≤ 0.5 s;
    2. Sponza's step change at bookmark 1 meets the same;
    3. no probe goes un-updated for more than 2 full round-robin passes during a 60 s sweep (census);
    4. the scheduler costs ≤ 0.05 ms at 20,000 probes, and the subgroup and fallback histograms agree exactly (on the Intel UHD too);
    5. after a 30° sun step on Sponza, the mean error over all probes against a re-converged field is within 5% in ≤ 3 s at the calibrated count, and the bookmark view's ROI in ≤ 0.5 s (far probes owing 2; re-set from T1.9's GPU-loaded rates); the p95 and the far probes' extra noise are recorded;
    6. the warm-up runs in the technique and is on by default: a host that does nothing gets the same first frames as the viewer does today;
    7. feedback (4.6): on two-room at a test budget of 64 probes a frame, an emissive texture changed without telling the platform has its nearest probes revisited within 0.5 s; on a still scene it fires on under 0.1% of updates over 600 frames (counter), and the image mean with feedback on is within 0.2% of feedback off.
  - Record: response curves (CSV), per GPU.
- **T2.4c Key probes and propagation.** The first check runs before the rest of M2 builds on it.
  - Done when, on the RTX at the default budget, GI only:
    1. after a 30° sun step on Sponza, the bookmark view is within 5% of settled in ≤ 15 frames (0.25 s), and the mean over all probes is within 5% in ≤ 2 s, when every fine probe has been traced once since the step (~1.8 s); the p95 is recorded. If check 2 passes but this misses by up to 50% (≤ 3 s), it's recorded and kept; beyond that, the main session decides with Dennis;
    2. in a test mode that holds fine probes back from their own updates, propagation alone removes at least 60% of the change at the fine probes (mean over the volume), for that sun step and for a sky dimming. If it removes less, propagation is switched off and "far probes owe 2" stays (the decision rule, set before measuring);
    3. no light crosses a wall through propagation: on two-room at a test budget of 64 probes a frame (so propagation does the work), with fine probes held back, switching the lamp on in the lit room raises the dark room by no more than SPEC's 1% leak bar in any frame;
    4. noise: 3 s after the sun step, the out-of-view fine probes' noise (the standard deviation of each probe's mean irradiance over 30 frames, averaged) is no worse than with the "far probes owe 2" fallback at the same moment;
    5. on a still scene propagation never runs (counter 0); in a frame with a global change `ddgi/propagate` costs ≤ 0.1 ms; and during a 10-second day the update stays within Budget v2's +10% margin, back to back.
  - Show: the sun step side by side with and without propagation, at the moment the view has settled.
- **T2.5 Light groups.**
  - Done when:
    1. on two-room with 2 groups, a weight change shows in the same frame, and the image matches a field re-converged from scratch at that weight within 1% mean (linearity);
    2. with groups off, the field is within 0.1% mean of T2.4b's and the cost within 3%;
    3. with 4 groups, the cost per updated probe is ≤ 1.4× groups off;
    4. GI sampling at 1440p stays ≤ 1.0 ms (T1.6's view).
  - Record: memory per layer per scene.
- **T2.6a Sky re-bake.**
  - Done when:
    1. the GPU procedural map matches the CPU sky within 0.1% mean and 1% per texel, and `giSkyIrradiance` still meets ±0.5% on open ground;
    2. a re-bake costs ≤ 0.1 ms GPU and ≤ 0.05 ms CPU;
    3. the projection compiles in ≤ 0.2 s cold on the RTX ("cold": a fresh browser profile with nothing cached, one scene per run, as T1.9a measured startup);
    4. by re-tracing alone (instant sky off), a 30° sun step on the furnished room settles the bookmark view (GI only) within 5% in ≤ 0.5 s on the RTX (global changes are judged on the view, Dennis 2026-11-05); the whole-volume time is recorded.
- **T2.6b Instant sky.** The first check runs before the rest is built.
  - Done when, on the RTX:
    1. right after a 30° sun move, with the sun layer's weight at 0 (GI only), the nudge removes at least 70% of the difference between the old and the settled field in the same frame, on the furnished room's window bookmark and Sponza's bookmark 1, with bands 0–2. Recorded with the direct part alone and with the bounce rescale. If it falls short, stop: the instant sky is dropped and the main session tells Dennis;
    2. transfers are built: on Sponza ≤ 1.5 s after the warm-up starts; with the warm-up off, the bookmark view's probes in ≤ 1 s;
    3. on the furnished room's window bookmark, a sky change moves the GI-only mean of the sunlit floor rectangle by more than 1% in the frame it happens;
    4. on two-room and the furnished room, 1,800 frames (at a 16 ms test budget) after a sky change, the field with the instant sky on matches a run with it off within 0.5% mean: the nudge only makes the change sooner, it doesn't move where the field settles;
    5. µs per updated probe (fixed batch of 512, GPU-bound) is within 5% of the same three layers with no transfer; and on Sponza, at the calibrated count, the update plus a sky change's nudge, write and compose stays within Budget v2's "≤ gpuMs (+10%)", back to back (the nudge's own ms recorded, every tile and far quarters);
    6. checks 3 and 4 also pass on the Intel UHD (correctness, not speed).
  - Record: build times per scene, extra memory, the furnished room's total core memory (SPEC's 64 MB is a guide), nudge and compose ms, the transfer's noise at 8 against 64 updates, and bands 0–2 against 0–4.
- **T2.7 Native harness.**
  - Done when:
    1. `play.ps1 -Scene two-room -Motion light-sweep` plays the manifest motion at the viewer's speed (positions within 1 cm at matching times);
    2. the miss shader reads the shared sky file;
    3. the sun can follow a time-of-day motion;
    4. the default config renders as before (by eye plus one capture).
  - Record: native response curves for the step (3 runs).
- **T2.8 Viewer.**
  - Done when:
    1. the sweep and step motions run on two-room, Cornell and Sponza;
    2. the response meter's CSV for a two-room step matches the T2.4b test page within 2 frames;
    3. the native compare page plays both recordings in sync;
    4. every control has hover help, and Dennis can drive it without instructions.
  - Show: the viewer URL with two-room at `lit-room`.
- **T2.10 M2 measurements.** The table at the top on the RTX (GPU-bound) and the M5 Pro if it's set up, the correctness checks on the Intel UHD, plus our and native's response times for the same motions (3 runs; recorded as numbers, not required to match native). Done when each target is met or its miss is explained and agreed with Dennis, and he has judged the sweep's look next to native by eye.

## Risks, and what to spike first

1. **The deferred pass may buy little** at Sponza's operating point, where the fixed cost dominates. Then Sponza stays at about 268 probes a frame (T1.9b), which already meets the response target in view. This is the largest lever and the largest uncertainty. *Spike first: T2.0.*
2. **Short histories flicker.** At `n` = 0.42 the first update is 70% one noisy estimate. If that shows as a flash, raise the floor on `n`, or give marked probes 2× rays for their first update (at a cost). T2.4a's two-room step shows it.
3. **Light-grid noise with many small lights.** Candles have short ranges, and a cell 2 probe spacings wide may hold many lights that matter only at its corners. The options are smaller cells near lights, more reservoirs, or Moreau's light tree. `sponza-candles` decides.
4. **The change estimate is a stand-in.** `lightChangeScale` and the visibility floor are guesses until T2.4a compares marked probes against probes that really changed. Too generous wastes budget; too stingy leaves stale probes until the round robin reaches them.
5. **Global changes on big scenes** can't meet 0.5 s everywhere at 2 ms by re-tracing (section 6.4). Dennis agreed to judge them on the view.
6. **Native motion** needs fork changes, because the harness has no animation. Two-room first, Sponza second.
7. **The instant sky may cover too little.** It moves only what a probe sees of the sky directly, plus a rescaled guess at the bounce. Deep inside Sponza's colonnade, where sky light arrives mostly by bounce, the same-frame share may fall under 70%, and then the transfer is dropped (T2.6b's first check). Re-tracing still carries time of day, so the plan doesn't depend on it.
8. **Re-tracing after a big jump, far from the camera, on Sponza and Bistro:** about 2.5–3.3 s for the whole volume with far probes owing 2 (6.4), 10–12 s with 8 everywhere, which fails Dennis's bar. Key probes and propagation (4.5) are the plan to bring it to 1.0–1.6 s; T2.4c's first check decides. The deferred shadow pass (T2.0) would help both.
9. **The steady-change rule's short histories are noisy** far from the camera on Sponza during a day animation (the history drops to about 1 update there, 6.4). If that flickers, the floor on `n` rises and those probes lag more.
10. **The rates move with the GPU's clocks.** T1.9b's Sponza runs spread 255–272 probes a frame, and its third run was already at +4.3% of the 2 ms budget. The times here are from the medians; T2.4c checks that the passes M2 adds keep the update inside Budget v2's +10% margin (4.5).
11. **Propagation spreads key noise into fine probes** (35–70% of one key update's noise, 4.5) and can be a little biased by its clamp. If T2.4c's noise check fails, `keyResidualShare` rises (fine probes traced sooner) or propagation is switched off.

## Decided by Dennis (2026-11-05)

1. **Time of day:** "instant with a delayed fallback". Instant where it's cheap: the sun's colour and brightness (the sun layer) and the sky seen directly (6.3). Re-tracing nearest the camera (6.4) carries the sun's direction and the rest; see 7.
2. **`sponza-candles`:** "1024 is fine for now". The count is a setting, for raising later as a stress test.
3. **Light groups:** an explicit group weight, off by default. "Go with stability but also flexibility and expansion later. I dont want to close the door on common use cases": spotting three.js-style dimming stays possible and is on the Later list.
4. **Global changes** (the sun, the sky) are judged on the view.
5. **Native:** the look is judged by eye. Response time is measured as a number, not as a requirement to match native, which has hardware ray tracing.
6. **Area lights:** after M2, "lets not abandon them". On the Later list, with how they'd fit (section 8).
7. **No minutes-long builds:** "anything taking minutes to build is a fail and we should not go that direction." The full sky transfer with bounces (17–25 minutes on Sponza) is dropped; the first-order transfer (6.3) builds from the field's own rays in seconds.
8. **Memory** is sized for mid- to high-end GPUs: "Why are we memory capping so low? this machine has like 6-8gb of GPU memory and 64gb of DDR5 CPU memory. mid to high end machines are the goal." The 128 MB and 48 MB caps are gone; the instant sky is on by default. SPEC now keeps the furnished room's 64 MB as a guide, not a gate (revised 2026-11-05).
9. **Hardware:** the RTX 3060 and up and an Apple M5 Pro are the targets. The Intel UHD is a correctness test only ("we arent building for the intel").
10. **Measurement:** benchmarks and estimates assume a loaded GPU with its clocks up ("Running at 60hz seems like a fail"). Budget v2 is accepted (work per frame, calibrated with the clocks up, a 4 s sweep floor); this doc's rates are its measured medians (T1.9b: Sponza 268 a frame).
11. **Sun layer in white**, weighted by colour × intensity: kept, as an exception to SPEC's rule 3, after comparing what Unity APV, RTXGI and Godot SDFGI do ("go with what we need to get to the improvement passes but i would look at what other godot systems do"). The three conditions are in 6.3.

## Decided by Dennis (2026-11-06)

1. **Key probes and propagation are the main path for global changes** (4.5), with "far probes owe 2" as the fallback. On Sponza: the view in ~0.13 s, the whole volume visibly changed in ~1 s, every probe traced in ~1.8 s, against ~2.5 s with noisier far probes for the fallback. It's new (no shipped DDGI system propagates changes between levels this way, research note), so T2.4c's first check decides.
2. **Rigid moving objects come into M2,** right after T2.4c and ahead of the instant sky (board row T2.M1), since they reuse the same scheduler. Deforming meshes and the BVH work stay in M3.

Nothing is left open in this doc.

The times in sections 4.3–4.8, 6.3, 6.4 and 6.5 come from T1.9b's measured medians (the rates section).
