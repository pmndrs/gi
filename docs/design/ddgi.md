# DDGI field and update loop (T1.5 design)

Status: **design by the main session, 2026-11-04.** The implementer builds from this page. Changing a default listed here is fine during tuning if PROGRESS says why; changing a done-when number is a DECISIONS entry.

Implemented from the papers, never from RTXGI SDK code (DECISIONS 2026-10-30):
- Majercik, Guertin, Nowrouzezahrai, McGuire 2019, "Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields", JCGT 8(2). The field, the Chebyshev test, hysteresis.
- Majercik, Marrs, Spjut, McGuire 2021, "Scaling Probe-Based Real-Time Dynamic Global Illumination for Production", JCGT 10(2). Relocation, classification, the fixed-ray split, biases, the weight crush.
- Cigolle et al. 2014 (octahedral maps); Keinert et al. 2015 (spherical Fibonacci points).

The native harness is a comparison tool only. Its settings are listed where we match them on purpose.

## In one paragraph

A box of probes on a regular grid. Each probe stores, in two texture atlases, the irradiance arriving from every direction (6×6 texels, octahedral) and the mean and mean-squared distance to the nearest surface in every direction (14×14). Each frame a budget of probes is updated: one workgroup per probe traces its rays inline, shades the hits with the platform's hit shading plus last frame's field for bounce light, and blends the result into the atlases with hysteresis. The first 32 rays of each probe point in fixed directions and only feed relocation and classification. A renderer reads the field through one WGSL function, which the update kernel also uses for bounce light, so there is exactly one sampling code path.

## Where things live

- `packages/ddgi-core/src/techniques/ddgi/`: the technique (`DDGITechnique implements GITechnique<DDGISettings>`), its WGSL, its tests.
- It sees only `TechniqueContext` (scene copy, ray queries, hit shading, kernel builder, timer). Never the host, never three.js.
- Test pages: `tools/gpu/pages/ddgi-*`.

## The volume

- **One axis-aligned box per technique in M1.** Settings give `center`, `size` and `spacing` (metres, one number or per axis). Probe counts are `floor(size / spacing) + 1` per axis, and the grid is centred in the box. Camera-following volumes and several volumes are M3.
- **Probe index:** `p = x + nx * (z + nz * y)`.
- **Limits** (default device, `maxTextureDimension2D` 8192, `maxTextureArrayLayers` 256): `nx * 16 ≤ 8192`, `nz * 16 ≤ 8192`, `ny ≤ 256`. `prepare` throws `GILimitError` with the numbers past that.

## The textures

All three are `texture_2d_array`s with **one layer per probe row `y`**. Inside a layer, probe `(x, z)` owns tile `(x, z)`. That keeps every dimension inside default limits for all our scenes, and a plane of probes is one layer for debugging.

| Texture | Tile (interior + 1-texel border) | Format | Contents |
| --- | --- | --- | --- |
| irradiance | 6×6 + border = 8×8 | `rgba16float` | irradiance E (not E/π), rgb |
| distance | 14×14 + border = 16×16 | `rg16float` with the `texture-formats-tier1` feature; otherwise `rgba16float` (declared fallback) | mean distance, mean squared distance |
| probe data | 1 texel per probe | `rgba16float` | xyz: relocation offset ÷ spacing (within ±0.45); w: state |

- **Why these formats:** fragments sample irradiance and distance with bilinear filtering, so they must be filterable without `float32-filterable`. `rg16float` is not a storage format in core WebGPU; `texture-formats-tier1` adds it. Without the feature, distance uses `rgba16float` and twice the memory. A test covers that path.
- **fp16 range:** values up to 65,504. Scenes in three.js-style units (sun intensity around 1–10) are far below that. A host using real lux (around 100,000) would overflow; that's recorded as a known limit, not handled in M1.
- **Borders:** the border texels copy the octahedral neighbour (the mirrored texel across the edge; corners take the diagonally opposite corner), so bilinear filtering is seamless across octahedral seams.
- **Sizes and atlas sizes are fixed at `prepare`.** Changing counts or texel sizes needs a new `prepare`, as the technique contract says.

### Probe states (the `w` channel)

| Value | Meaning | Traces | Blended | Used when shading |
| --- | --- | --- | --- | --- |
| 0 | active | all rays | yes | yes |
| 1 | off: inside geometry | 32 fixed rays | no | no |
| 2 | off: no surface nearby | 32 fixed rays | no | no |
| 3 | not updated yet | all rays | yes, hysteresis 0 | no |

State 3 is the starting state, so the first frames show partial GI instead of black probes. Off probes keep tracing their fixed rays so they can come back on when the scene changes (M3).

## Rays

- **Count:** `raysPerProbe`, default 256, a multiple of 32, from 64 to 512. The first 32 are the **fixed rays**.
- **Fixed rays:** 32 spherical-Fibonacci directions, never rotated. Used for relocation and classification only, never blended. That's what native does too, so ray counts match (NOTES "Native ray counts").
- **Blended rays:** `raysPerProbe − 32` spherical-Fibonacci directions, rotated by one random rotation per frame (shared by all probes that frame). The rotation comes from the CPU: a uniform random quaternion from a seeded generator, seeded by the frame number. Not byte-identical across runs, and nothing should require it.
- **Range:** rays are traced to `maxRayDistance`, default unlimited (`tMax` = 1e27), so a miss sees the sky. Distances written to the distance atlas are clamped to `1.5 × length(spacing)`.

## Revision after T1.5a (2026-11-05): trace and blend are separate dispatches

T1.5a built the one-kernel design below and, as the board asked, timed it once against a split version. The split won clearly, so it replaces the single kernel (DECISIONS 2026-11-05):

- **`ddgi/trace`:** workgroups of 64 threads, one ray each. It calls `giTraceClosest` inline (tracing is still inline in the technique's own kernel, not the platform's `encodeBatch`), shades the hit exactly as phase A below, and writes each ray's radiance (rgb as halves) and signed distance to a ray buffer (about 4 MB at 512 probes × 256 rays).
- **`ddgi/blend`:** one workgroup per probe. It loads that probe's rays from the buffer into workgroup memory, runs phase B (classification and relocation, T1.5b) and phase C (blend) as below, and writes the scratch tile.
- **`ddgi/write`:** unchanged.
- *Why:* the single kernel is register-bound. Tracing, shading and blending in one 256-thread workgroup leaves few waves resident. Measured at 512 probes, 256 rays: RTX 3060 2.48 ms one kernel against 1.22 ms split; Intel UHD 11.6 against 7.8 ms, with the same field.
- The "rays never touch device memory" goal below is dropped; the ray buffer is small and is read once.
- Bindings: `ddgi/trace` uses the tracer 3, the scene copy and hit shading 2, plus the update list and the ray buffers, within 8.

## The update kernel (`ddgi/update`): trace, shade and accumulate in one kernel (superseded by the revision above; the per-ray and per-texel rules still apply)

One workgroup per probe in this frame's update list. Workgroup size 256 (the default `maxComputeInvocationsPerWorkgroup`).

**Phase A: trace and shade.** Thread `t` handles rays `t, t + 256, ...`.
- Ray origin: the probe's grid position plus its relocation offset. Off probes trace only the fixed rays.
- **Fixed rays** need only the hit distance and whether it was a front face. They are not shaded.
- **Blended rays:**
  - **Miss:** radiance = `giSkyRadiance(direction)`, distance = the clamp value.
  - **Back-face hit:** radiance 0. Distance = `−0.2 × t`. The negative sign marks a back face for classification; the distance atlas uses `0.2 × t`, which pulls the probe's mean distance short in that direction so the Chebyshev test sees it as blocked. This is the "double-sided distance" leak fix, measured as the largest single fix (84×, NOTES).
  - **Front-face hit:** `s = giSurfaceAt(ray, hit)`, then
    `radiance = s.emission + s.albedo / π × (giDirectIrradiance(s, groupMask, random) + bounceEnergy × ddgiSampleIrradiance(s.position, s.normal, ray.direction, ...))`,
    where `ddgiSampleIrradiance` reads the **current** atlases (the field from earlier updates) for unlimited bounces. Hit-shading's `bounceAlbedoCap` applies to `s.albedo`.
- Results go to workgroup memory: rgb radiance and signed distance per ray (16 bytes × 256 = 4 KB). **Rays never touch device memory.**

**Phase B: classification and relocation**, over the 32 fixed rays. Workgroup reductions:
- the number of back-face hits;
- the closest back-face hit (distance and direction);
- the closest and the farthest front-face hits (distance and direction).

**Subgroups first:** reduce inside each subgroup with `subgroupAdd` / `subgroupMin` / `subgroupMax`, then combine subgroups with workgroup atomics. **Declared fallback** (no `subgroups` feature): every thread does the workgroup atomics itself. Same result, same structure, chosen when the kernel is built; a test runs both on the same probes and requires identical states and offsets. Never assume a subgroup size.

Thread 0 then decides, from the paper's rules (Majercik 2021, section 5):
- **Relocation** (setting `relocation`, default on):
  - more than 25% back faces among the fixed rays: the probe is inside something; move it through the closest back face, to just past that surface (`closest back-face distance + 0.5 × minFrontFaceDistance`);
  - otherwise, closest front face nearer than `minFrontFaceDistance` (default 0.02 × the smallest spacing since T1.5b; the first design said 0.1): move away from it, along the farthest front-face direction, but only if that doesn't head back toward the closest one;
  - otherwise, ease back toward the grid point.
  - A new offset is accepted only if it stays within ±0.45 of the spacing on every axis, and (since T1.5b) only if the moved probe stays inside the volume's probe box. Otherwise the old offset stays.
- **Classification** (setting `classification`, default on):
  - **off, inside geometry (1):** more than 25% back faces among the fixed rays, or the nearest thing the fixed rays see is a back face (added in T1.5b), after relocation had its chance;
  - **off, no surface nearby (2):** no fixed ray hit a front face within reach, so the probe affects no surface's interpolation. The paper's rule uses the probe's own cell; T1.5b measured that 32 fixed rays miss slivers near the cell's corners and switched off 25 probes Cornell needs, so the reach is 1.5 spacings (DECISIONS 2026-11-05). It's what makes probes outside a building switch off;
  - otherwise **active (0)**, or stays **3** until its first blended update.
- The new offset and state go to the scratch buffer.

**Phase C: blend** (active and not-yet-updated probes only). Threads `0..35` each own an irradiance texel and threads `36..231` each own a distance texel (for the default 6 and 14; other sizes loop). Each loops over the blended rays in workgroup memory:
- **irradiance texel**, direction `n` from the texel centre: the stored value is `π × Σ L·max(0, n·ω) / Σ max(0, n·ω)`. *Why π:* with rays spread evenly over the sphere, the cosine-weighted mean radiance over the hemisphere is E/π, and the contract's `giIrradiance` returns E. Check against the furnace: constant L gives E = πL exactly.
- **distance texel:** weight `max(0, n·ω)^distanceSharpness` (default 50), accumulating `d = min(|distance|, clamp)` and `d²`.
- **Hysteresis:** `new = mix(estimate, old, h)`, with `h = hysteresis` (default 0.97). The old value is read from the atlas with `textureLoad` (the atlas is bound for sampling in this kernel). `h = 0` for a probe in state 3.
  - **Big changes:** if the probe's **mean** irradiance (the average over all its interior texels) changed by more than `irradianceChangeThreshold` (default 0.25 of the old mean), lower `h` by a factor of 0.75 for the whole probe, so a light switch shows up quickly (the paper's "probe irradiance threshold").
    - **Off by default since T1.5b (DECISIONS 2026-11-05):** every variant measured biased the field; M2 lowers hysteresis from `SceneChanges` instead. When enabled, the verdict comes from the probe's mean in the previous update.
    - *Revised after the T1.5a review:* the first version tested each texel. A single texel's estimate from about 224 rays is noisy, and bright outliers are the common kind, so they won the lower hysteresis and the field drifted high: Cornell read +9.5% instead of +4.4% with the rule off. Averaging over the probe's texels averages the noise out, while a real light change still trips it.
  - **Firefly clamp:** the change applied in one update is clamped to `maxBrightnessChange` (default 2.0, in irradiance units relative to the old value; a setting) so one lucky ray can't flash a probe.
- Results go to the scratch buffer.

**Why the blend doesn't write the atlases directly:** this kernel samples the atlases (bounce light, the old value). WebGPU won't let one dispatch both sample a texture and write it as a storage texture, and core read-write storage textures exist only for 32-bit single-channel formats. So:

## The write kernel (`ddgi/write`)

One workgroup per updated probe. It copies the scratch tile into the irradiance and distance atlases (write-only storage textures), fills the borders, and writes the probe-data texel. In a compute pass each dispatch is its own usage scope, so `update` sampling the atlases and `write` storing into them can share one pass.

Scratch buffer: `maxProbesPerFrame × (36 × 8 + 196 × 4 + 8)` bytes, about 1 MB at 1,000 probes. Packed as `pack2x16float` pairs, the same precision as the atlases.

## Bindings (the default 8 storage buffers per stage)

| Kernel | Storage buffers | Textures |
| --- | --- | --- |
| trace | tracer 3 + scene copy and hit shading 2 + update list and ray buffer = **7** | irradiance, distance, probe data (sampled), the sky's radiance map |
| blend | update list, ray buffer, scratch and its own state = **4** | irradiance, distance, probe data (sampled) |
| write | update list 1 + scratch 1 = **2** | irradiance, distance, probe data (write-only storage) |

(Counts as built in T1.5b; the single `update` kernel of the first design used 7.)

One uniform buffer holds the volume (origin, spacing, counts), the frame rotation, the settings and the update-list length.

## Budget v2: work first, milliseconds as a ceiling (accepted by Dennis 2026-11-05; as built in T1.9b)

**Why.** A GPU-millisecond budget plus GPU power management is a positive feedback loop: less work lets the GPU idle, its clocks drop, the same work takes longer, so the controller cuts work again. On a quiet laptop at 60 Hz the RTX sat in its low-power state and 2 ms bought 2–3 Sponza probes a frame (a sweep every 162 s). The budget exists to protect frame time, and a GPU that is idling has time to spare. RTXGI itself uses fixed work per frame.

The code is `packages/ddgi-core/src/platform/work-budget.ts`; every threshold below is a named constant there (`CALIBRATION`, `SATURATION`) with its reason, and recorded in DECISIONS.

**1. The control is work: probes per frame (`units`).**
- Each frame the platform hands a technique a fixed number of units: the calibrated batch (point 2), never below the floor (point 3), stepped down only when frames are late (point 4).
- `budget.units` stays a hard cap over everything, the calibration burst and the floor included.
- Without GPU timings (`gpuTiming` off, or no `timestamp-query`) `gpuMs` can't be measured: the hand-out is `budget.units`, or 512.

**2. The calibration: once, GPU-bound, timings that agree.**
- **When:** at the first frame with `gpuMs` after `prepare` (it waits while the mesh trees are still being built), and when the host calls `GIPlatform.recalibrate()`. Nothing else starts one: a burst during play makes frames heavier for a moment.
- **How the host runs it:** behind its loading screen, GPU-bound, with two frames in flight so the GPU never waits (`deltaTime` 0). Waiting for every frame left the RTX about 40% busy and its clocks wandered by up to 2× over seconds; a light 60 Hz frame does the same. The viewer and the `ddgi-batch-cost` page end their warm-up this way.
- **The burst** (`CalibrationBurst`): about 20–170 frames on the test scenes.
  1. 1,024 probes, until the size's timings agree: its last 3 within 5% of each other, and their median within 2% of the 3 before. Timings that still move are the GPU changing its clocks.
  2. Down (or up) by ×4 until two settled sizes sit on either side of the technique's share of `gpuMs` (shared evenly between techniques).
  3. The batch on the line between them is measured itself and accepted within 5%; otherwise the size on the other side is dropped as suspect (measured while the clocks moved) and the ladder moves on, at most 3 times.
  - A size whose timings never agree takes the median of its last 5 after 24 timings.
  - A burst that hasn't finished after 400 frames gives up: the hand-out is the best estimate from the sizes that settled, or the floor with none, and the counters say "uncalibrated".
- **Filler:** only in the platform's first calibration, and only when the host leaves time between frames (a host at 60 Hz). Small batches then alternate with a filler batch of at most half the frame interval, at most 1 s of GPU time in all, so the GPU stays loaded. A technique added at runtime calibrates without filler. Counted in the stats.
- **The confirm pass:** after the burst, while the host still runs frames back to back, the chosen batch is timed for 300 more frames; if the median of the last 100 is off the curve by more than 10%, the curve is scaled by that factor and the batch solved again. It ends unconfirmed at the first frame with time between frames (the host left its loading screen), or after 600 frames, so it never stays open.
- **The stored curve:** the settled sizes are kept. A new `gpuMs` or `budget.units` is solved from them with no new burst (`units` = the largest batch on the curve within the share, extrapolated at most one ×4 step beyond the largest size measured).
- **A geometry change** (or mesh trees still building) marks the calibration **stale**: reported in `stats.counters['<technique> budget'].stale`, the batch unchanged until the host's next `prepare` or `recalibrate()`. A change during a burst leaves the new calibration stale. The viewer's "Start the probes over" re-calibrates.
- **Reported:** the line around the batch (fixed ms + ms per probe, between the two measured sizes on either side of it), the burst's frames, the spread of the timings it settled on, filler frames and ms, the confirm factor. The GPU's clock can't be read from WebGPU.
- **Per tier:** the host's `gpuMs` (default 2 ms) gives the number on each device. A host that wants a fixed number per scene sets `units` and skips the calibration.

**3. A floor bounds convergence on every machine.**
- DDGI asks for at least `minProbesPerFrame` probes (`GITechnique.minUnitsPerFrame`). Default "auto": enough for a full sweep within 4 s at the frame rate the host runs (`targetFps`, else the observed rate; Sponza 81 at 60 Hz, Cornell 8).
- The floor wins over the millisecond ceiling, so a weak device runs slower, never frozen. A host can lower it explicitly (a number; 0 turns it off). `budget.units` still caps it.
- **The floor adapts to the device (added after measuring the Intel UHD):** "auto" never asks for more than twice the calibrated batch. Measured fixed batches (60 Hz / back to back): Intel UHD, Sponza 16 probes 1.8 / 2.1 ms, 40 probes 3.3 / 3.0 ms, 81 probes 5.3 / 4.8 ms, 160 probes 9.3 / 6.7 ms; Cornell 8 probes 0.6 ms. RTX, Sponza 81 probes 4.0 / 1.4 ms. So on the Intel the cap gives Sponza about 32–40 probes (about 3 ms, a sweep every 8–10 s), while on the RTX the cap doesn't bind. No tier detection is needed: the calibration already knows the device.

**4. Late frames step the batch down; on-time frames bring it back.**
- The platform watches `FrameInput.deltaTime` in windows of about 1 s. Frames are late when the window's median frame time is more than 10% over the target interval.
- **The target interval:** `1 / budget.targetFps` when the host gives it. Otherwise the display's interval, **learned and kept while frames are late** (the learned-target rule): a vsync-like window, within 3% of a common display refresh rate (24 to 165 Hz) *and* steady (frame times from the 10th to the 90th percentile within 10% of the median, as a display paces them), teaches it; only a later vsync-like window that is on time can change it, to a faster rate. A slowdown never replaces it: a target that followed recent windows forgot 60 Hz a few seconds into a 3× slowdown, took the 30 Hz frames as on time and climbed back to the full batch while every frame stayed late (budget v2 re-review). The floor's frame rate follows the learned target too, so a slowdown doesn't raise the floor.
  - Without a vsync-like window the host counts as uncapped and has **no ceiling**: an uncapped frame rate rises and falls with the view, which says nothing about being late.
  - A host must give `targetFps` when it is late from its first frame (a vsync host that never makes 60 Hz looks like an on-time 30 Hz host), and when it changes its display rate for good (120 Hz to 60 Hz looks exactly like a 2× slowdown, and the slowdown wins).
  - A frame that misses now and then leaves the window's median at the display interval: not late, no step-down. One late frame in many isn't saturation.
- **Step down:** each late window multiplies the batch by 0.8, never below the floor. There is no clock test (DECISIONS, after the budget v2 re-review): a thermally throttled RTX reads about 3× its calibrated cost, and a test on DDGI's timings blocked the step-down in exactly that hot state. "Late" already needs the frame to miss its interval, so an idle, low-clocked GPU is never late. The cost: a CPU-bound host that misses its interval steps DDGI down too, which doesn't help it; the floor bounds that.
- **Recover (no oscillation):** after a run of late windows, recovery waits for 3 s on time, then grows ×1.1 a second up to a ceiling of 0.9 × the batch that was late first. At the ceiling, after 10 s on time, the ceiling rises ×1.05, back to the calibrated batch.
- *Why this avoids the loop:* while frames are on time, a slow-looking DDGI pass is low clocks, not a real cost, and nothing changes.

**5. What gets reported.** `stats.counters['<technique> budget']`: units this frame, calibrating, confirming, uncalibrated, calibrated units, floor units, stepped down, ceiling units, frames late, stale, frames per second, and the calibration's line, frames, spread, filler and confirm factor. DDGI's own counters add the seconds per full sweep. The viewer shows them in plain language ("per frame", "measured cost").

**Defined measurements (DECISIONS):**
- The calibrated batch's GPU time, measured GPU-bound (two frames in flight) with the clocks up (from below about 70 °C, clocks ≥ 1,500 MHz, 10–15 s timed), is within ±10% of `gpuMs`; no lower bound when every probe fits.
- A full sweep takes ≤ 4 s at the floor on every device. On a GPU-bound frame, the whole frame with GI on against off adds ≤ 3 ms p50 (SPEC). 60 Hz light-load numbers are information only.
- It behaves at 30, 60 and 120 Hz on the RTX and the Intel: a stable number of units, the floor respected, no step-down while frames are on time, and a saturation step-down that recovers. The last is the `saturation` section of `ddgi-batch-cost`: at 60 Hz a synthetic GPU load makes frames late for 10 s; the batch must step down within 4 s, never below the floor, and get back to 0.85 × the calibrated batch within 40 s of the load ending. The Intel UHD is a correctness target: its numbers are reported, not judged.

## Which probes update (the budget)

- **Unit = one probe update.** `TechniqueFrame.units` comes from the platform's budget v2 (above): the calibrated batch, at least the floor, stepped down only when frames are late. `FrameBudget.units` stays a hard cap. Without GPU timings, `FrameBudget.units`, default 512 on Tier A.
- **Order:** a fixed scrambled order of all probes, `order[i] = (i × stride) mod N` with `stride` coprime with `N` and near `0.618 N`, so each frame's set is spread across the volume, not one slab. A CPU cursor walks it. Each frame's list is `units` consecutive entries, written into the update-list buffer with `queue.writeBuffer` inside `encode` (no readback; probe states stay on the GPU).
- Off probes in the list still cost a workgroup, but only 32 rays and no blend, so they're cheap. The calibration measures the mix as it is.
- **Probes near a change first** (SceneChanges) is M2. The design leaves room: a priority list in front of the cursor, in the same buffer.
- `maxProbesPerFrame` (default 4,096) sizes the update list and the scratch buffer. Dispatches over 65,535 workgroups use two dimensions.
- `FrameBudget.paused` skips every DDGI kernel.

## Sampling the field (`ddgiSampleIrradiance`): one function, two users

The update kernel (for bounce light) and the shading module (T1.6) call the same WGSL. It takes every resource as a parameter, as the shading-module contract requires, and uses `textureSampleLevel(…, 0)` so it works in compute and fragment stages alike.

Given a surface point `P`, its normal `N` and the direction `V` it's seen from:
1. **Bias:** `P' = P + N × normalBias + (−V) × viewBias`. Defaults: `normalBias = 0.1 × min spacing`, `viewBias = 0.1 × min spacing`. Tune with the leak and accuracy checks, not by eye: NOTES says too little bias made leaks worse (up to 34×), and a large view bias caused black patches under furniture.
2. **The 8 probes around `P'`**, trilinear weights from `P'`'s position in the cell.
3. Per probe, using its relocated position:
   - skip it unless its state is 0 (active);
   - **direction weight:** `((dot(dirToProbe, N) + 1) / 2)² + 0.2`, so probes behind the surface count for little but never zero. `dirToProbe` is taken from the **unbiased** point `P`, as in the 2019 paper (T1.5a measured about 1% better on Cornell's walls than from `P'`);
   - **visibility (Chebyshev):** sample the distance atlas in direction `−dirToProbe`: mean `μ`, mean squared `μ2`. If the probe's distance to `P'` is greater than `μ`: `variance = |μ2 − μ²|`, `chebyshev = variance / (variance + (d − μ)²)`, weight × `max(chebyshev³, 0.05)`;
   - **weight crush:** a combined weight below 0.2 is scaled by `(w / 0.2)²`, so small weights fade fast instead of adding a faint wrong colour.
     - HomeFig's trap: a crush threshold equal to the direction-weight floor (both 0.2) let fully back-facing exterior probes leak, because their weight sat exactly at the threshold and was never crushed.
     - Here a fully back-facing probe also carries the Chebyshev factor, which is at most 1 and drops to 0.05 when it's occluded, so its weight lands below 0.2 and gets crushed. The leak tests (two-room, thin wall, closed door) watch this.
   - multiply by the trilinear weight;
   - sample irradiance bilinearly at `octahedral(N)` in the probe's tile.
4. Return `Σ w·E / Σ w`, or 0 when `Σ w` is 0.

**Coverage** (`giCoverage`, T1.6): 1 out to half a spacing beyond the outermost probes, falling to 0 over the next half spacing, times `saturate(Σw / 0.02)`, so a point no probe can see fades to the host's environment instead of going black (DECISIONS 2026-11-05).

## Settings (`DDGISettings`)

```ts
interface DDGISettings {
    volume: { center: Vec3; size: Vec3; spacing: number | Vec3 }; // new prepare when changed
    raysPerProbe: number;            // 256
    irradianceTexels: number;        // 6 (interior), new prepare when changed
    distanceTexels: number;          // 14
    hysteresis: number;              // 0.97
    irradianceChangeThreshold: number; // Infinity (off; DECISIONS 2026-11-05). A value turns on the lagged probe-mean rule
    maxBrightnessChange: number;     // 2.0
    distanceSharpness: number;       // 50
    normalBias: number | 'auto';     // auto = 0.1 × min spacing
    viewBias: number | 'auto';
    maxRayDistance: number;          // Infinity
    bounceEnergy: number;            // 1.0. Native RTXGI dims bounces (0.95); set it only to match native
    relocation: boolean;             // true
    minFrontFaceDistance: number | 'auto'; // auto = 0.02 × min spacing (T1.5b; was 0.1)
    classification: boolean;         // true
    maxProbesPerFrame: number;       // 4096, new prepare when changed
}
```

`bounceEnergy` defaults to 1, not 0.95: with 0.95 the furnace room settles 4.8% low (0.25 / (1 − 0.95 × 0.5) = 0.476 instead of 0.5), which would fail its 2% check. Light groups (one layer per group) are M2; the kernel takes a `groupMask` now and passes all bits.

## Debug data and counters

- **Points:** the probe-data texture already holds offset and state; a `points` debug view publishes world position and state in an `rgba32float` texture (written by the write kernel for updated probes). Legend: the state table above.
- **Counters:** probes updated, rays traced, probes active, probes off (each reason), probes not yet updated.
- **Timings:** `ddgi/update` and `ddgi/write` through the timer scope.

## Done when (T1.5, from the board), and how each is measured

1. **Furnace room converges to its analytic value within 2%.** Albedo 0.5, emission 0.25: E/π = 0.5 everywhere. After enough frames for the hysteresis to settle (state the count), sample `ddgiSampleIrradiance` at a grid of points on every wall, floor and ceiling (a compute test kernel, read back), and require `|E/π − 0.5| / 0.5 ≤ 2%` for the mean and report the worst point.
2. **Cornell indirect light matches Cycles within the M1 accuracy target** (SPEC: mean within ±10%; FLIP and 8×8 block error reported, the FLIP threshold is set in T1.9).
   - **Use the point-light Cornell** (`light: 'point'`). DDGI's field counts light seen straight from an emitter as GI, but the Cycles "indirect" pass doesn't, so the emissive Cornell can't be compared indirect-to-indirect. With a point light, DDGI's field holds exactly the bounced light, the same as Cycles' Diffuse Indirect × Diffuse Colour. Render that Cycles reference at bookmark 1 first (2048 samples, through the bench lock).
   - DDGI image: an engine-free test view. Per pixel, a primary ray through the platform tracer finds the surface; the pixel is `albedo / π × ddgiSampleIrradiance(...)`. Primary rays here are a test tool only, never the shading path. Written as EXR, compared with `tools/compare`.
   - Also report the emissive Cornell's full image against Cycles full (not gating).
   - **What the Cycles side is, in every scene (set 2026-11-05 at T1.9, DECISIONS "What 'indirect light' means against Cycles in a scene with a sky"):** the GI composite, `cycles-gi.exr` (`cycles-untextured-gi.exr` for an untextured render) = Cycles indirect + Cycles direct − a lights-only direct render. DDGI's field holds every light a probe ray brings back: bounces, and also the sky and emitters seen straight. Only the manifest's analytic lights (sun, point, spot) are drawn outside the field, so only their direct light is taken away. The lights-only render switches off the world and every emitting shader input, and `tools/reference/cycles/run.ts` refuses to write the composite if a known-answer check on it fails. In a scene with no sky and no emitter (point-light Cornell, two-room) the composite equals the indirect pass, so the rule above is the same; in a scene with a sky (Sponza) the raw indirect pass would leave the sky light out (+101% instead of +26% on Sponza before the decal fix). `tools/reference/README.md` ("The GI composite") has the details.
3. **Probes inside walls are classified off.** Cornell (the two blocks are solid) and the cabinet row: every probe whose grid point lies inside a closed solid, and that can't relocate out within 0.45 spacing, reads state 1; probes outside the room read 1 or 2; none inside the room's air reads off unless it has no surface in its cell. Checked against the geometry by a test.
4. **Two-room leak ≤ 1%.** Leak = DDGI's indirect light in the dark room *beyond* the reference's (the doorway lets real light in), as a share of the lit room's indirect light, both measured as mean pixel values of the test view at the `dark-room-leak-view` and `lit-room` bookmarks against Cycles renders of the same views. Also report thin-wall and closed-door, where the true dark-side light is zero (reference-free), without gating M1 on them.

Plus: the subgroup fallback test (above), `pnpm lint && pnpm typecheck && pnpm test`, and every GPU check on both GPUs, saying which adapter. If the update cost comes near the 2 ms budget, time inline (this design) against a batch version once, as the board asks.

## Not in T1.5

- The shading module packaging (`giIrradiance`, `giCoverage`, `globalsCode`), the hello example and its cost target: T1.6.
- Light groups and change-driven priority: M2. Camera-following and several volumes: M3.
- A perceptual encoding of irradiance (the paper's gamma) and R10G10B10A2 storage: only if banding or flicker shows up in T1.9's stability check.
