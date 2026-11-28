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

- `packages/ddgi/src/`: the technique (`DDGITechnique implements GITechnique<DDGISettings>`), its WGSL, its tests.
- It sees only `TechniqueContext` (scene copy, ray queries, hit shading, kernel builder, timer). Never the host, never three.js.
- Test pages: `tools/gpu/pages/ddgi-*`.

## The volume

- **One axis-aligned box per technique in M1.** Settings give `center`, `size` and `spacing` (metres, one number or per axis). Probe counts are `floor(size / spacing) + 1` per axis, and the grid is centred in the box. Several volumes are in since T3.5, camera-following volumes since T3.6 (the next two sections).
- **Probe index:** `p = x + nx * (z + nz * y)`.
- **Limits** (default device, `maxTextureDimension2D` 8192, `maxTextureArrayLayers` 256): `nx * 16 ≤ 8192`, `nz * 16 ≤ 8192`, `ny ≤ 256`. `prepare` throws `GILimitError` with the numbers past that.

### Several volumes (T3.5, m3-moving.md §4.1, as built)

- **Settings:** `volumes: DDGIVolumeSettings[]` (at most 8), each with an optional `priority` (default 0; higher first, equal priorities in list order). Without `volumes`, `volume` is the one-element case; with it, `volume` reads as the first of them. New `prepare` when changed. Code: volumes.ts.
- **One atlas set.** Volume v owns the array layers `[layerBase_v, layerBase_v + ny_v)`, in priority order; every layer is as large as the largest footprint (largest nx by largest nz tiles). Past `maxTextureArrayLayers` in total, `prepare` throws `GILimitError`. One volume lays out exactly as before (layer base 0, the atlas its own grid).
- **The volume table:** an `rgba32float` texture, one row per volume in priority order, each row the volume's four sampling `vec4f`s: origin and normal bias, spacing and view bias, counts and **layer base** (`volumeCounts.w`), texel sizes and **the atlas's tiles across and down** (`volumeTexels.zw`); since T3.6 four more texels, the window (lattice coordinates of its first probe), the fade's width per axis and the valid extent's low and high sides (`VOLUME_TABLE_TEXEL`, next section). The kernels get the first four as their first uniforms. The CPU writes the first six after `prepare`, when a bias setting changes and when a window moves; the valid extent only after `prepare` (a fixed volume's is its first and last probes), then `ddgi/scroll` copies a camera-following volume's in every frame.
- **Sampling** (`ddgiSampleVolumes`, wgsl/sample.ts): volumes in table order; each read volume adds its normalised irradiance with weight `coverage_v × (1 − W)` (coverage: its edge fade × `saturate(Σw / 0.02)`); stop at W ≥ 0.99 or once 2 volumes have added weight (a volume that covers the point but whose probes all have weight 0 there doesn't count, so a third nested volume still fills in). A volume whose edge coverage is 0 at the point is skipped before any probe is read; when none was read, the **nearest** volume (by distance to its valid extent; on a tie the lower priority) is read with no weight, so a point outside every volume gets the nearest cell as before, and volumes side by side each answer on their own side (T3.6; T3.5 read the lowest-priority one). The irradiance is `Σ w_v E_v / W` (the one volume's own value when only one added weight), the coverage W. So **one volume gives exactly the M2 value**, through the same code. Renderers (`giIrradiance`, `giCoverage`) and the update's bounce light (`ddgi/trace`, the layered lookup too) read it, so light bounces between volumes.
- **The shading module** is version 2: its four `vec4f` uniforms became the volume table texture (4 textures in all), and it has no uniforms now; `globalsCode` writes no uniform buffer then.
- **The update:** each volume runs the same kernels with its own uniforms, update list, scratch, history, scheduler, census, propagation, mark table and ray scratch, one volume after another in a frame (`PreparedResources` per volume, the shared textures and kernels the same objects). The kernels see one volume (`ddgiUniformVolume()`) and add its layer base where they touch the atlases (`ddgiProbeLayer`, `ddgiAtlasLayer`). Each volume gets Budget v2's floor of its own (a sweep within 4 s, never 0 probes a frame), and the rest of the frame's probes is shared in proportion to probe counts, each volume's fraction carried over frames (`UnitSharer`), so every volume sweeps at the same rate and a small one is never starved; the technique's floor is the volumes' sum. The clock, the hot movers, the layer weights and the warm-up (done when every volume has made its passes) are the technique's. Counters add up over volumes ("volumes" says how many; each volume's change clock is listed too). The GPU timer takes 160 passes a frame (was 16), room for 8 volumes.
- **Not in T3.5:** a global probe numbering across volumes (a scheduler per volume, so priorities don't compete across volumes: DECISIONS 2026-11-09); the camera-following volume (T3.6, next section).

### The camera-following volume (T3.6, m3-moving.md §4.2, as built)

- **Settings:** `follow: 'camera'` on a volume (`'fixed'` by default). Its probes sit on a lattice fixed in world space whose probe 0 is the box's first probe as the settings place it; the box is a window of that size, starting there. Needs 4 probes or more on every axis. Code: scroll.ts (CPU), wgsl/scroll.ts (`ddgi/scroll`).
- **Slots:** lattice probe `L` keeps its data (atlas tiles, probe-data texel, history record, update-list index) in slot `L mod n` per axis, so a move reuses the planes left behind and nothing else moves. Kernels work on slots as before; positions and neighbours are worked out in window probes (`ddgiWindowProbe`, `ddgiProbeSlot`, `ddgiNeighbourSlot`, wgsl/sample.ts and common.ts). A fixed volume's window stays at 0, so its slot is its window probe: the same code. The uniforms carry the window (`volumeWindow`), the table too.
- **Key probes** go by the lattice's parity (even lattice index, or the window's first or last), so a one-plane move changes only the planes at the window's ends; for a fixed volume this is the grid's parity as before.
- **The window rule** (CPU, once a frame before any volume's passes): the window moves by whole planes when the camera is more than 0.75 spacings from its centre on an axis, then centres on the camera. A move of more than half the window is a teleport: every slot resets and the warm-up runs again.
- **`ddgi/scroll`** (one timed pass per camera-following volume, before every volume's passes): `ddgiScrollReset`, in a frame where the window moved, resets the entering slots: state 3 with offset 0, the history record cleared, owing 2 updates (the warming ones, not K) at urgency 1, the scheduler's stamp kept. `ddgiScrollExtent` moves the valid extent: one workgroup per side; outward by at most 1/15 spacing a frame and only over planes with at most 1/32 of their probes in state 3 (a stricter rule let one late probe stall it); inward at once. The extent lives in a 32-byte storage buffer, copied into the table row after the pass.
- **The fade is 10 spacings wide** on a long axis (`CAMERA_FADE_SPACINGS`; narrower on a short one, keeping a spacing of full coverage around the camera: 3.75 on the 12-probe vertical axis); a fixed volume's stays half a spacing. *Why:* the band sweeps at the camera's speed, so a point fades in the time the band takes to pass it; half a spacing at 5 m/s and 1.5 m changed the band 0.59% a frame on Bistro, 10 spacings (3 s) 0.074%, under the stability bar (DECISIONS 2026-11-10). Coverage is the lowest of the axes' fades (with half a spacing on every axis, exactly the old rule).
- **The valid extent** keeps the fade's whole reach (the 0.5 margin plus the fade) inside the warm planes: ahead of the camera it grows over warm planes (a plane in about a quarter second once warm); behind the camera it follows the camera continuously, the reach plus 0.25 inside the camera's half-window, so planes about to be reused have faded out before the window moves and the trailing edge never jumps either. With the fly-through's window and the camera centred, the camera volume answers alone within about 5 spacings of it on x and z and 1.5 vertically, and blends with what lies behind out to the window's ends.
- **Warming is served from the whole frame:** a camera-following volume's floor adds the warming updates its entering planes owed lately (2 per probe, averaged over 30 frames; DECISIONS 2026-11-09), so a share by probe count beside a big fixed volume doesn't starve them.
- **Entering probes are never read before 2 blended updates:** state 3 ends after the second (T2.M1's `warmedOnce`), and sampling skips every probe that isn't active.
- **Counters:** "window moves", "planes cleared (total)", "probes cleared (total)" and "teleports", with a camera-following volume. `DDGIField.volumes[i]` gives `followsCamera`, `window` and the `validExtent` buffer.
- **Not in T3.6:** a viewer setting or scenario for it (T3.8); a fade width from the camera's speed (it is a constant).

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
| 3 | not updated yet (fewer than 2 blended updates) | all rays | yes: the first with hysteresis 0, the second as usual | no |

State 3 is the starting state, so the first frames show partial GI instead of black probes. Off probes keep tracing their fixed rays so they can come back on when the scene changes (M3). Since T2.M1 a probe leaves state 3 after its **second** blended update, not its first (m3-moving.md §5.2: one noisy estimate is a pop); the history flag `warmedOnce` says the first has happened, and the scratch's tiles-written word tells `ddgi/write` to copy the tiles of both.

## Rays

- **Count:** `raysPerProbe`, default 256, a multiple of 32, from 64 to 512. The first 32 are the **fixed rays**.
- **Fixed rays:** 32 spherical-Fibonacci directions, never rotated. Used for relocation and classification only, never blended. That's what native does too, so ray counts match (NOTES "Native ray counts"). As in native, they are set aside only while `relocation` or `classification` is on; with both off there are no fixed rays and all `raysPerProbe` rays turn and are blended (DV-07, 2026-11-19; `ddgiSetAsideRays`).
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
    `radiance = s.emission + s.albedo / π × giDirectIrradiance(s, groupMask, random) + giBounceAlbedo(s) / π × bounceEnergy × ddgiSampleIrradiance(s.position, s.normal, ray.direction, ...)`,
    where `ddgiSampleIrradiance` reads the **current** atlases (the field from earlier updates) for unlimited bounces. Hit-shading's `bounceAlbedoCap` applies to the bounce term only (`giBounceAlbedo`), as native RTXGI's sample does (DV-08; it applied to both terms before 2026-11-19).
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
- **Hysteresis:** `new = mix(estimate, old, h)`. The old value is read from the atlas with `textureLoad` (the atlas is bound for sampling in this kernel). `h = 0` for a probe in state 3. *Since T1.12* `h` is per probe and knows the time since the probe's last update ("Hysteresis that knows the time between updates" below); the fixed `h = hysteresis` (0.97) of the first design stays as `adaptiveHysteresis: false`. Irradiance and distance texels use the same `h`.
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
| blend | update list, ray buffer, scratch, its own state (census) and the probe history (T1.12) = **5** | irradiance, distance, probe data (sampled) |
| write | update list 1 + scratch 1 = **2** | irradiance, distance, probe data (write-only storage) |

(Counts as built in T1.5b; the single `update` kernel of the first design used 7.)

One uniform buffer holds the volume (origin, spacing, counts), the frame rotation, the settings and the update-list length.

## Budget v2: work first, milliseconds as a ceiling (accepted by Dennis 2026-11-05; as built in T1.9b)

**Why.** A GPU-millisecond budget plus GPU power management is a positive feedback loop: less work lets the GPU idle, its clocks drop, the same work takes longer, so the controller cuts work again. On a quiet laptop at 60 Hz the RTX sat in its low-power state and 2 ms bought 2–3 Sponza probes a frame (a sweep every 162 s). The budget exists to protect frame time, and a GPU that is idling has time to spare. RTXGI itself uses fixed work per frame.

The code is `packages/core/src/platform/work-budget.ts`; every threshold below is a named constant there (`CALIBRATION`, `SATURATION`) with its reason, and recorded in DECISIONS.

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
- **The target interval:** `1 / budget.targetFps` when the host gives it. Otherwise the display's interval, **learned and kept while frames are late** (the learned-target rule): a vsync-like window, within 3% of a common display refresh rate (24 to 165 Hz) *and* steady (frame times from the 10th to the 90th percentile within 10% of the median, as a display paces them), teaches it; only a later vsync-like window that is on time can change it, to a faster rate. A slowdown doesn't replace it at once: a target that followed recent windows forgot 60 Hz a few seconds into a 3× slowdown, took the 30 Hz frames as on time and climbed back to the full batch while every frame stayed late (budget v2 re-review). The floor's frame rate follows the learned target too, so a slowdown doesn't raise the floor.
  - **The slower-rate rule:** a display rate can drop for good (Chrome's Energy Saver caps at 30 fps; the window moves to a 60 Hz screen), and the viewer and ddgi-three don't pass `targetFps`. When the batch has been **at the floor** and the frames steady and vsync-like at one slower refresh rate for 10 s, that rate is adopted, and the batch recovers against it. At the floor DDGI's work is as small as it gets, so it isn't what holds the frames at the slower rate. 10 s is several times what reaching the floor takes, so an overload that stops on its own isn't taken for a new display, and short enough that DDGI doesn't sit at the floor for long. A real GPU overload makes unsteady late frames, which aren't vsync-like, so it never adopts a rate. A steady slowdown that lasts (a GPU throttled to exactly half rate) can't be told from a new display rate and is adopted the same way; if DDGI's recovery then makes frames miss the new rate, it steps down again against it. `GIPlatform.recalibrate()` forgets the learned rate.
  - Without a vsync-like window the host counts as uncapped and has **no ceiling**: an uncapped frame rate rises and falls with the view, which says nothing about being late.
  - A host must give `targetFps` when it is late from its first frame (a vsync host that never makes 60 Hz looks like an on-time 30 Hz host). Hosts may pass it in any case; nothing depends on it.
  - A frame that misses now and then leaves the window's median at the display interval: not late, no step-down. One late frame in many isn't saturation.
- **Step down:** each late window multiplies the batch by 0.8, never below the floor. There is no clock test (DECISIONS, after the budget v2 re-review): a thermally throttled RTX reads about 3× its calibrated cost, and a test on DDGI's timings blocked the step-down in exactly that hot state. "Late" already needs the frame to miss its interval, so an idle, low-clocked GPU is never late. The cost: a CPU-bound host that misses its interval steps DDGI down too, which doesn't help it; the floor bounds that.
- **Recover (no oscillation):** after a run of late windows, recovery waits for 3 s on time, then grows ×1.1 a second up to a ceiling of 0.9 × the batch that was late first. At the ceiling, after 10 s on time, the ceiling rises ×1.05, back to the calibrated batch.
- *Why this avoids the loop:* while frames are on time, a slow-looking DDGI pass is low clocks, not a real cost, and nothing changes.

**5. What gets reported.** `stats.counters['<technique> budget']`: units this frame, calibrating, confirming, uncalibrated, calibrated units, floor units, stepped down, ceiling units, frames late, stale, frames per second, and the calibration's line, frames, spread, filler and confirm factor. DDGI's own counters add the seconds per full sweep. The viewer shows them in plain language ("per frame", "measured cost").

**Defined measurements (DECISIONS):**
- The calibrated batch's GPU time, measured GPU-bound (two frames in flight) with the clocks up (from below about 70 °C, clocks ≥ 1,500 MHz, 10–15 s timed), is within ±10% of `gpuMs`; no lower bound when every probe fits.
- A full sweep takes ≤ 4 s at the floor on every device. On a GPU-bound frame, the whole frame with GI on against off adds ≤ 3 ms p50 (SPEC). 60 Hz light-load numbers are information only.
- It behaves at 30, 60 and 120 Hz on the RTX and the Intel: a stable number of units, the floor respected, no step-down while frames are on time, and a saturation step-down that recovers. The last is the `saturation` section of `ddgi-batch-cost`: at 60 Hz a synthetic GPU load makes frames late for 10 s; the batch must step down within 4 s, never below the floor, and get back to 0.85 × the calibrated batch within the time the controller's own constants allow from the lowest batch under the load: 3 s on time, then one ×1.1 step a second, plus 10 s of margin, at most 90 s. The check catches a recovery that stalls (a ceiling that ratchets down, frames that never count as on time again), not a slow one; how fast recovery is comes from those constants. The Intel UHD is a correctness target: its numbers are reported, not judged.

## Hysteresis that knows the time between updates (T1.12 phase 2a; the lead's design, 2026-11-05)

**Why.** With a fixed hysteresis (0.97) every update keeps the same share of the old value, whatever the time since the probe's last one.
- Cornell updates every probe every frame, so the wobble from small-emitter ray noise (σ about 30% per update, NOTES) shows: 0.56% per pixel per frame, and Dennis: "it is not steady". A higher fixed h is quieter but slower everywhere.
- Under budget v2 a Sponza probe is updated about once every 73 frames (1.2 s at 60 Hz). At 0.97 a change takes about 75 updates to settle 90%: about 90 s. A fixed 0.995 would make that minutes.

**The rule.** Each probe keeps an effective sample count `n`, the time `t` of its last blended update and its count of blended updates `k`. On an update at time `T`:
1. `n ← n · exp(−(T − t) / τ)`: what it remembers fades with the time since its last update.
2. `n ← min(n, ramp(k))`, the ramp (below, added after the first review): `ramp(k) = k / a` for `k ≤ K`, then `K / a + (k − K)`.
3. `h = n / (n + 1) = 1 − 1 / (n + 1)` blends this update: the new estimate counts as one sample among `n + 1`.
4. `n ← min(n + 1, nMax)`, `t ← T`, `k ← k + 1`.

- **First update** (state 3: new, after `resetHistory()`, or back on after being off): `n = 0` and `k = 0`, so `h = 0`, as before.
- **The warm-up** is the same rule with no time between updates: since T2.4b it is the technique's own ("The warm-up" below), whose clock doesn't advance while it runs; the viewer's `settings.hysteresis` override is gone with the viewer's copy of it.
- **The ramp (a = 8 for a probe's first K = 16 updates; `hysteresisUpdatesPerSample`, `hysteresisRampUpdates`).** Without it, the warm-up was a plain running average, `h = k/(k+1)`. Its bounce light reads the field it is building, so the first estimates are too dark, and a plain mean keeps them at full weight: the error shrinks only like `k^−(1−ρ)` (ρ: the share of a probe's light that comes back through the field; Cornell fits ρ ≈ 0.65). Cornell came out of the 16-pass warm-up 25% low and took 19 s to get within 1%. Under the ramp, every-frame updates get `h = k/(k+a)`: a mean weighted towards later updates (weights grow like `k^(a−1)`), whose error shrinks like `k^−a(1−ρ)`, at `a/√(2a−1)` times the noise of a plain mean. It is limited to the first K updates because under the cap an every-frame probe reaches nMax only after `a · nMax` updates (1,600) and stays noisy until then; after the ramp it gets there about `K/a + nMax` updates in. 16 is the warm-up's own pass count. A fixed small cap on `n` instead (n ≤ 2 to 16 during the warm-up) does worse by the same model: a cap of 8 leaves Cornell about 23% low, a cap of 2 about 13%.
- **Where it settles.** A probe updated every Δ seconds settles at `n = e/(1 − e)` with `e = exp(−Δ/τ)`, so `h = e`: an exponential average with a memory of τ seconds, whatever the frame rate. Cornell (Δ = 1/60 s) would reach `n = 60τ`, so it stops at `nMax`; Sponza at 60 Hz (Δ ≈ 1.2 s) settles at a few samples.
- **nMax** sets the quietest a probe can get: `h ≤ nMax / (nMax + 1)`. 200 gives 0.995, the hold the flicker decision asked for (DECISIONS, Cornell's flicker).
- **The big-change rule** is unchanged and still off by default: when on, it multiplies this `h` by 0.75 for the probe's next update. The firefly clamp is unchanged too.
- **Distances** blend with the same `h` as irradiance.
- **The fixed rule** stays as `adaptiveHysteresis: false` (uses `hysteresis`). The history is kept under both rules, so switching mid-run carries on from the right count.

**The clock.** The technique's own, in seconds: the sum of every frame's `deltaTime` since `prepare`, each frame counting at most 0.25 s (`MAX_CLOCK_STEP_SECONDS`). A longer frame is a stall (a hidden tab, a debugger, a hitch). Counted in full, a minute in a background tab would make every probe forget its history at once, and the whole field would flash noisy as it updated again. Paused frames (`budget.paused`) don't call the technique and don't advance it.
- **Wrapped every 4096 s (`CLOCK_WRAP_SECONDS`, about 68 min).** Each probe's `t` is an f32, whose step is 0.49 ms below 2^12 s but 7.8 ms after a day and 62.5 ms after a week of running. Unwrapped, the average fade would stay right (the rounding errors of successive elapsed times cancel), but single gaps between updates would be measured in steps that coarse. Wrapped, every time stays under 4096 s; the blend adds the period back when `T − t` comes out negative. Only a probe left unblended for more than a whole period would be misread. An active probe is blended every sweep (seconds), and an off probe starts again from state 3 when it comes back on.

**Storage.** `DdgiProbeHistory` per probe (`n` and `t` as f32, `k` as u32, saturating) in a storage buffer that only `ddgi/blend` reads and writes (12 bytes a probe; Sponza 233 KB). Each workgroup owns one probe and a probe is listed at most once a frame, so there are no races. New buffers start at zero, and state 3 resets `n` and `k`, so `resetHistory()` needs no clear. Blend now binds 5 storage buffers.
- *Since T2.4a* the record is m2-lights.md §4.1's 48 bytes (`PROBE_HISTORY_WORD` in layout.ts; Sponza 0.93 MB): `k` moved to the low 16 bits of word 2 (saturating at 65,535), and `ddgi/mark` writes the change words with atomics in its own dispatch before the trace (next section).

**What it trades.** For a texel updated with hysteresis h and per-update noise σ, the wobble is `σ·√((1−h)/(1+h))` and the change per update about `σ·(1−h)·√(2/(1+h))`. Bias from bounce light that hasn't settled shrinks per update by about `(1−h)(1−ρ)` (ρ: how much of a probe's light comes back through the field). So a quieter probe is also slower to settle bounce light: on the accuracy pages, where every probe is updated every frame for 600 frames, the field is still a little low (numbers below). And a running average whose estimates use the field it is building (h = k/(k+1)) settles bounce light only like `k^−(1−ρ)`, so early estimates made with a dark field linger until the clock fades them.

**Measured (T1.12 phase 2a; RTX 3060 Laptop, Chrome 156, DDGI's clock at 60 Hz; pages `ddgi-flicker`, `ddgi-settle`, `ddgi-furnace`, `ddgi-leak`; numbers in PROGRESS).**

| | fixed 0.97 | τ = 2 s | **τ = 5 s** | τ = 10 s |
| --- | --- | --- | --- | --- |
| Cornell per-pixel change a frame, mean / p95 (every probe each frame) | 0.56% / 2.30% | 0.155% / 0.64% | **0.095% / 0.39%** | 0.093% / 0.38% |
| Sponza per-pixel change a frame (291 probes a frame) | 0.0038% / 0.014% | 0.058% / 0.22% | **0.027% / 0.10%** | 0.0145% / 0.054% |
| Intel Cornell per-pixel change (90 probes a frame) | 0.046% / 0.24% | | **0.090% / 0.46%** | 0.048% / 0.26% |
| Sponza, sun halved: within 5% / 1% of the new answer | 91 / 162.5 s | 5.5 s / noisy | **13 / 22.5 s** | 25.5 / 44 s |
| Sponza, restart with no warm-up: within 5% / 1% | 1 / 44 s | 0.5 / 12 s | **1 / 3 s** | 0.5 / 3.5 s |
| Sponza, after the warm-up: image mean | within 0.2% | within 0.6% | **within 0.3%** | within 0.4% |
| Cornell, after the warm-up (25% low): within 5% / 1% | 2 / 4.5 s (noise to 49 s) | 3.5 / 10 s | **4.5 / 17.5 s** | 6 / 18.5 s |

**The ramp, measured** (RTX 3060, τ 5 s, nMax 200; Cornell 1,574 probes a frame, Sponza 291; `ddgi-settle`, and `ddgi-flicker` with `VITE_DDGI_FLICKER_AFTER_WARMUP`). The error right after the warm-up is one image, about ±1.5 points from run to run (a = 8 for 16 updates ran three times: −4.6%, −6.3%, −7.4%).

| | no ramp (a = 1) | a = 2 | a = 3 | a = 4 | a = 8 (endless) | **a = 8, K = 16** |
| --- | --- | --- | --- | --- | --- | --- |
| Cornell, right after the warm-up | −26% | −15.6% | −10.5% | −9.2% | −4.6% | **−7.4% / −6.3%** |
| Cornell after the warm-up: within 5% / 1% | 4.5 / 19 s | 1 / 5.5 s | 0.5 / 2 s | 0.5 / 1.5 s | 0 / 0.5 s | **0.5 / 1 s** |
| Cornell after a restart: within 5% / 1% | 4.5 / 16.5 s | 1.5 / 6.5 s | 1 / 2.5 s | 0.5 / 2 s | 0.5 / 0.5 s | **0.5 / 2 s** |
| Cornell change a frame, the 240 frames after the warm-up (mean / p95) | 0.30% / 1.18% (the image climbs 21%) | | 0.63% / 2.63% | | 1.52% / 6.33% | **0.39% / 1.62%** |
| Cornell change a frame, 400 frames later | 0.092% (still climbing 1.7%) | | 0.109% | | 0.283% | **0.089%** |
| Sponza right after the warm-up; within 1% after a restart | −0.5%; 3 s | | +0.5%; 3.5 s | | −0.3%; 6 s | **+0.3%; 3.5 s** |

Settled flicker with the ramp is unchanged (it no longer binds): Cornell 0.094% / 0.38% a frame at `gpuMs:2` (0.094% / 0.38% without the ramp), Sponza 0.025% / 0.091% at 291 probes a frame (0.026% / 0.094%). A light change on Sponza: within 5% / 1% in 12.5 / 20.5 s (13 / 22.5 s without).

- **Defaults:** τ = 5 s and nMax = 200 (`ADAPTIVE_MEMORY_SECONDS`, `ADAPTIVE_MAX_SAMPLES` in settings.ts). Any τ above about 3.4 s lets Cornell reach the cap, so 5 and 10 are equally quiet there. 5 s follows a light change on Sponza in half the time of 10 s, at about twice the per-frame change on sparsely updated probes (still a fifth of Cornell's adaptive number). 2 s is noisier everywhere and no faster to a steady answer. nMax = 100 (h ≤ 0.99) was measured as well: Cornell 0.185% / 0.76%, furnace −0.80% after 600 frames.
- **Sparse probes are noisier than before.** Where a probe's updates are more than about 0.15 s apart (τ·ln(1/0.97)), its h is below the old 0.97, so each update moves it more. That is the point (Sponza follows changes in seconds). The cost: Sponza changes about 7× more per frame than at 0.97, and Intel Cornell (a sweep every 0.32 s) 2×. Both stay well under the RTX Cornell's adaptive number.
- **Every-frame probes settle bounce light more slowly.** On the accuracy pages (every probe every frame), 600 frames left the furnace −1.95% / −2.05% (RTX / Intel; the Intel over the 2% limit) and the two-room lit room 5.4% below its settled value. The pages now run 1800 (furnace) and 3000 (leak) frames, and the settled values match the fixed rule's: furnace −0.24% / −0.36% (fixed 0.97: −0.16% / −0.28%), two-room leak 0.81% (fixed 0.80%). With the ramp (its table above) the furnace settles in 600 frames again (−0.19% at a = 3), the two-room scene still needs more (0.74% at 600 frames with a = 3, settled 0.81%), so the counts stay. Measured with the ramp defaults: furnace (1800) −0.15%, leak (3000) 0.83%, Cornell test view (600) +5.63% (+4.52% without the ramp, which hadn't settled). M2's change-driven reset is what makes light changes fast on every-frame probes.

## Change-driven history (M2, T2.4a; m2-lights.md §4.1, §4.2, as built)

**What it does.** When `SceneChanges` reports a light or sky change, `ddgi/mark` (wgsl/mark.ts) works out from the lights' own numbers how much each probe's light changed, `r`, and adds it to the probe's record (`pendingChange`). At the probe's next blend, `n ← min(n, εK / (r − ε))` when `r > ε` (ε = `changeTolerance` 0.05, K = `changeUpdates` 8), and, for a probe marked before its previous update too, `n ← min(n, ε / r)` (the steady-change rule). Then T1.12's fade, ramp and growth run as before. A still scene runs the mark only once, for the scene's creation, and marks 0 probes, so it is exactly T1.12's rule (checked: `ddgi-change-still` gives M1's field to the last digit on the RTX).

**The mark** (one thread per changed light and probe in its influence box; the table is built on the CPU, mark-table.ts):
- A point or spot light: per colour channel, the change of its unshadowed irradiance vector at the probe (`E_c × direction`, three's falloff from the platform's light-falloff module, no surface cosine), over the probe's stored mean in that channel (history words 6-7, written by every blend), the largest channel, times `lightChangeScale` and `max(Chebyshev visibility toward the light, 0.25)`. Probes that are off or not updated yet are skipped.
- A directional light: every probe, `r = max_c |E_new,c d_new − E_old,c d_old| / E_old,c` (a turn alone gives `2 sin(θ/2)`, a dimming alone `|1 − new/old|`), times `0.25 + 0.75 × openness` toward the old and new direction. Until the sun has a layer of its own (T2.5, T2.6b), a colour or intensity change marks this way too; the exception's place is marked in mark-table.ts.
- The environment: every probe, `r` from the sky's radiance SH (T2.6a), times the openness over six directions. The kernel compares the SH now with the SH when the field last took a sky change (the platform's sky-change module, `HitShading.skyChange`; no readback): per channel `(|Δc₀| + |Δ(c₁, c₂, c₃)|) / c₀`, against what is left where the sky got darker, the largest channel. A sky switched on from black is `MAX_GLOBAL_CHANGE`; a `setEnvironment` that changed nothing, or turned a procedural sun too little to re-bake (0.05°), marks nothing. Until T2.6a this was `r = 1`.
- More than 1,024 light changes in a frame (`lightsOverflowed`): every probe, `r = 1`.
- The marks are counted subgroups first (fallback: an atomic per marking thread; `ddgi-mark-reduction` checks both agree) into the census: "probes marked" and "probes marked (total)".

**`lightChangeScale` is 0.5, not the design's 0.25** (DECISIONS 2026-11-06). On two-room's 2 m lamp step (`ddgi-change-step`, every probe every frame), a probe's true change was 0.48 × the mark's unscaled value at the median (quartiles 0.25-0.73), so 0.5 makes `r` about the true change. The scale sets how deep a probe's cut is, not only whether it is cut: at 0.25 every probe that really changed by over 5% was still cut (lit-room recall 99.8%), but too gently, and the view took 0.73 s to get within 5%. The lingering error is the old lamp's bounce light: right after the step a probe's new estimates bounce off neighbours that still hold the old light, and probes cut too gently hold it for seconds. At 0.5 the view is within 5% from the first update (+3.9% at most, +2.0% after 2 s); M1's rule alone takes 1.42 s. *The step is small* (the view changes by 6.9%), so that overshoot is 56% of the step and still 29% of it after 2 s; T2.4b re-checks with a larger step (TASKS, M2 done-when changes).

**Not done, and measured not to help:** moving a cut probe back along the warm-up ramp too (so its count would grow back by 1/8 an update instead of 1). It made the transient overshoot larger (0.30 s instead of 0.03 s at scale 0.5; 1.5 s instead of 0.73 s at 0.25), so the count grows back by one an update, as the design has it.

### Revised in T2.4b: the cut holds, and counts the bounces

T2.4b's larger steps (TASKS, M2 done-when changes) showed why the old lamp's bounce lingered. A cut probe's next estimates bounce off a field that still holds the old light, so each estimate's error is ρ times the field's (ρ: the share of a probe's light that comes back through the field). With the count growing back by one an update the probe becomes a plain average of those estimates, whose error shrinks only like k^−(1−ρ). On two-room at its calibrated 880 probes a frame, the lamp halved was still 45% of the step away after 30 frames.

What changed (wgsl/blend.ts `ddgiChangeLimit`, `ddgiHysteresisStep`; wgsl/mark.ts):
- **The cut holds for the updates the change promised.** While a probe is owed updates from a mark (the `takingChange` flag), its count doesn't grow back; every promised update replaces as much as the first. After the last one it grows by one an update again.
- **The cut is solved with the bounce share.** Each held update shrinks the error by `c = h + (1 − h)ρ`. Within ε after K updates needs `c^K r ≤ ε`, so `h = (c − ρ)/(1 − ρ)` with `c = (ε/r)^(1/K)`, `n = h/(1 − h)`, and 0 when even h = 0 can't do it. ρ is `changeBounceShare`, 0.8 by default: two-room's, measured from that halving (the error shrank 0.84 a frame with 82% of the probes updated at h ≈ 0, so ρ ≈ 0.80). With ρ = 0 the rule is the unbiased case, `h = (ε/r)^(1/K)`.
- **A change promises the updates its bounces need:** at best (h = 0) the error shrinks by ρ an update, so `ddgi/mark` promises `max(K, ⌈ln(ε/e) / ln ρ⌉)`, with e the error against the new light. A light halved on two-room needs 14, not 8. Far probes after a global change still owe `farProbeUpdates` (2), and are left noisier and later for it.
- **A darkening is measured against the light that's left.** ε is a share of the settled (new) value, so halving a light leaves an error of 100% of it, not 50%: where the light at a probe got weaker, the mark's `r` is `r/(1 − r)` (for a directional light, `old/new − 1`). A brightening stays against the old light, so both directions are cut at least by the change's size; for the promised count a brightening's error against the new light is `r/(1 + r)`.
- **No firefly clamp on a reported change.** The clamp lets a channel grow by at most 2× its old value an update. A probe that was dark when a lamp came on near it then needed about 8 updates just to climb; the change is known and real, so updates taking it in aren't clamped.
- **A light with no range is marked at every probe** (mark-table.ts). Its influence radius is set against the sun, but a probe's change is measured against its own light, and inside sunlit Sponza a weak lamp's light reached far past that radius (recall 59% for a lamp of 0.3, a tenth of the sun's number). A light with a range keeps its range as the box. Cost: one mark thread per probe for each such light changing in a frame.

Measured on the RTX 3060 (`ddgi-schedule-step`, `ddgi-change-step`, `ddgi-schedule-sponza`; PROGRESS 2026-11-06, T2.4b): two-room's lamp switched on is within 5% from frame 9 and halved from frame 17, Cornell's from frames 5 and 8, with no overshoot to speak of (0.2-0.3% of the step); T2.4a's 2 m move no longer overshoots (at most +0.8%, against +3.9% before).

### The feedback rule (T2.4b, m2-lights.md §4.6, as built)

`ddgi/blend` tests each quiet update (no change reported, none being taken in, past the probe's warm-up ramp): when the estimate lands more than 6 times the probe's noise from its stored value and more than 2% of it, the probe asks for one more update at the urgency `|estimate − stored| / stored`, and its active face neighbours at half that, through a buffer of its own (`atomicMax` on the urgency's bits); `ddgi/schedule-count` takes the requests into the records as one owed update. It never changes how an update blends.
- **6σ, not the design's 4σ** (DECISIONS 2026-11-06): DDGI's ray noise has heavy tails. On still Cornell and two-room after the warm-up 4σ fired on about 0.1-0.2% of updates, 6σ on 0.07-0.09% in steady state (bar 0.1%).
- **The noise word is relative** (its root-mean-square deviation over the stored mean, over 64 quiet updates, the updates that fire included). Ray noise grows with the light, so an absolute noise read too small for dozens of updates after a probe got brighter. Only the feedback rule reads it.
- An unreported change (two-room's crate made to glow through `updateMaterial`, which DDGI's mark doesn't read) had every probe within a spacing of it revisited within 0.5 s at 64 probes a frame (median 6 frames, against 24 without feedback).

### Key probes and propagation (T2.4c, m2-lights.md §4.5, as built)

The main path for big changes on a big volume (the rule above: K updates for every probe wouldn't fit in `FAR_RULE_FRAMES` at this batch, 30 frames since T2.4d, 120 before; Sponza and the furnished room yes, two-room and Cornell no), on by default (`keyProbes`); "far probes owe 2" is the fallback.

- **Key probes:** grid index even or the last on every axis (`isKeyProbe`, `ddgiIsKeyProbe`): Sponza 2,717 of 19,425, two-room 180, Cornell 343. Ordinary probes otherwise.
- **The mark** (wgsl/mark.ts, `ddgiServing`; as revised in T2.4d, below): a key owes the full count at `keyProbeBoost` (4) × its change; away from the view's core (camera weight under 0.5) at most `keyProbeUpdates` (4), unless every key's full count fits in `FAR_RULE_FRAMES` at the priority share (then the full count: the furnished room yes, Sponza no). Every fine probe, near the view too, owes 1 (`FINE_PROBE_UPDATES`) at `keyResidualShare` (0.3) × its change and is flagged as awaiting a change, with the frame (word 13). Local light changes take the same path on such a volume.
- **The change clock** G (census word, u32 fixed point, 1/4096 of r): each mark frame a one-thread dispatch adds the frame's largest r; every blended update stamps G in the probe's record (word 10).
- **What a key passes on** (`ddgi/blend`): a key's update that takes in a reported change (owed updates since a mark, not its first update) writes new − old per irradiance texel, as halves, into the key-delta buffer at its update-list slot, with its stamp from before this update, and flags itself (`passedChange`, word 11 the slot; valid in the frame of its scheduled stamp).
- **`ddgi/propagate`**, one pass after `ddgi/write`, four dispatches: find (one thread per probe: awaiting, active fine probes with a key neighbour that passed its change this frame, into a compacted list, subgroups first or one atomic per thread; each workgroup stops at once when the census says no key passed one), args (clamps the list's length), propagate (256 workgroups looping over the list: per fine probe, 8 threads weigh its keys, both ends' Chebyshev tests multiplied, a key counts when that is at least 0.5, weights normalised over the keys that count; then per texel old + Σ w s Δ, clamped at 0, into its own scratch; the probe's mean follows; which keys passed it theirs and their summed weight go in word 9), and the write (write.ts's irradiance-tile entry point, the same tile code as `ddgi/write`).
- **The fine probe's own update** cuts its history for r × max(0.3, 1 − 0.8 × received) instead of r, then clears the awaiting flag, `waitingForKeys` and word 9.
- **Fine probes wait for their keys** (T2.4d): each frame `ddgi/schedule-count` flags an awaiting fine probe `waitingForKeys` while any of its key neighbours is still taking the change in (their `takingChange` flag, in word 5, which no thread of that dispatch writes, so the read can't race), for at most `KEY_WAIT_FRAMES` (30) since its mark. A waiting probe gets no priority bucket and the round robin skips it (the fill goes on past its window, and takes waiting probes only in a second pass when nothing else is left, so the list is always full). Meanwhile propagation passes it its keys' change; it is traced once they are done, so its own update reads a field that has taken the change in.
- **No indirect dispatches:** in Chrome on the RTX 3060 each `dispatchWorkgroupsIndirect` cost about 0.022 ms even when empty (Dawn checks its arguments with a dispatch of its own), so a version with three of them cost 0.07 ms on every still frame. Now a still frame's `ddgi/propagate` costs 0.010 ms.
- **Storage buffers per kernel:** blend 7, propagate 6, find 4, the tile write 3.
- **The test mode** `holdFineProbes`: awaiting fine probes are traced but not blended (their tiles written back as they were), and give up their owed updates so they don't take priority slots from the keys. It measures propagation alone.

Measured on the RTX 3060 (PROGRESS 2026-11-06, T2.4c; `ddgi-key-probes-sponza`, `-leak`, `-cost`, `ddgi-propagate-reduction`), Sponza at 316 probes a frame, GI only:
- 30° sun turn: the view within 5% from frame 4; the field's light-weighted error within 5% from frame 24 (far probes owe 2: 48), 1.98% at 2 s; every fine probe traced by frame 60.
- Sun halved: the view within 5% from frame 11 (T2.4b: 19-34).
- Propagation alone, fine probes held: 63.1% of the change removed for the sun turn, 91.4% for the sky at a quarter (the decision rule: 60%).
- No leak through the two-room wall with fine probes held (the dark room's fine probes at most 0.18-0.24% of the lit room's light past their settled values, three runs).
- Out-of-view fine probes' noise at 3 s: 0.15% against 0.18% for far probes owe 2.
- Cost (VALID): `ddgi/propagate` 0.010 ms on still frames, median 0.046 ms after the sun turn; a 10-second day's update median 2.17 ms at the calibrated 311 (Budget v2's bar 2.2; that day went over the zenith, NOTES).

### Revised in T2.4d: the coarse level first, and the batch while changes continue

**Why.** On the furnished room with its lamp off (4,998 probes, 2,941 active, 448 active keys, 571 a frame), the sun 30° lower left the view 23-25% off at 0.5 s and 10% at 3 s. Tracked by class (`ddgi-sky-sun-step`'s `byClass`): the probes the view samples were traced in the first frames at h = 0 while the rest of the room still held the old light, so their estimates bounced off stale light, and once their promises were spent their history averaged those estimates in. The room isn't short of updates (5.8 per active probe in 0.5 s), it is short of updates *in the right order*.

**What changed** (each measured on its own first, PROGRESS 2026-11-06, T2.4d):
1. The key path applies from `N × K / units` > 30 frames (was 120), so the furnished room (70) takes it; two-room (10) and Cornell (9) still don't.
2. Every key owes the full count where all the keys' full counts fit in those 30 frames at the priority share (the furnished room: 792 × 8 / 428 = 15 frames); else keys away from the view owe `keyProbeUpdates` (Sponza: 2,717 keys; owing the full count there made its view settle at frame 31-55).
3. Fine probes near the view wait for their keys too, and no fine probe is traced while its keys are still taking the change in (above).

On the furnished room the keys then converge at about one round a frame (7% light-weighted error at frame 10), the fine probes follow by propagation, and are traced at frames 16-22 reading the converged field: the bookmarks are within 5% from frames 11 and 17 (bar 30), per pixel 2.8% / 3.8% at 0.5 s and 1.3% / 1.9% at 3 s. Steps 2 and 3 each alone left 8-13% at 0.5 s.

**The batch while changes continue** (continuing-change.ts, m2-lights.md §4.5's second count). Budget v2 calibrates on a still scene, where the round robin's list holds the volume's mix of active and off probes (Sponza: half are off, and an off probe traces 32 rays, not 256). While a change keeps going, the priority share lists only active probes, so the same batch costs more: on Sponza through a real day (elevation 5°-75°-5°), 3.29 ms against 2.04 still at the calibrated 586, with the rays traced a frame 1.58× (`ddgi/trace` +0.96 ms, `gi/direct` +0.09, `ddgi/blend` +0.12, propagate and mark +0.10). So once `ddgi/mark` has run in `CONTINUING_CHANGE_FRAMES` (30) frames in a row (gaps of up to 4 unmarked frames included, T2.8), DDGI lists the hand-out times what a listed probe cost on still frames over what it costs now (from the census, a few frames late; at least `CHANGE_BATCH_MIN_SHARE`, 0.5). Without GPU timings there is no census, and the batch is the hand-out.

**Revised in T2.18: single steps on volumes that take a change in slowly** (change-intake-batch.ts; DECISIONS 2026-11-08, T2.18). After one 30° sun turn on Sponza at the calibrated batch, the 60 frames after it ran at 2.44 ms median, p95 3.12, with 51 of 52 over 2 ms (T2.10, about 800 MHz): the same list-mix effect, for about two seconds of a full priority share.
- *When:* a single change also gets the smaller batch, from the frame `ddgi/mark` runs until a census from at or after its latest mark shows nothing owed ("taking in a change" in the counters), on a volume where a change outlasts `FAR_RULE_FRAMES` anyway: the key path with keys away from the view owing `keyProbeUpdates` (Sponza, Bistro), or "far probes owe 2". Where a change fits the window (two-room, Cornell, and the furnished room, whose keys all owe their full count), a step is still served at the hand-out. *Why:* the furnished room's keys converge a round a frame only while the priority share holds all of them. With the batch cut, the most urgent keys spent their owed updates on estimates bounced off keys that had not moved yet, and its sun step settled at frames 162 / 124 instead of 11 / 13.
- *The cost a probe:* rays traced, plus `PROPAGATED_PROBE_RAYS` (2.5) for each fine probe `ddgi/propagate` passed a change to (Sponza after the turn: +0.053 ms for 926 propagated, about 2.5 rays' trace time each).
- *Before the census arrives* (its first readings after the step are from before the step), the list is taken to cost the most it can: every active probe at its full rays, off probes at their 32 fixed rays only once every active probe is listed. So the step's first frames are cut too, and a hand-out of the whole volume is not cut.
- *Limits* (T2.18 review): never under Budget v2's floor (`minProbesPerFrame`, at the platform's last rate, or the observed frame rate under a units budget; Sponza 81 at 60 Hz). A single change counts as taken in for at most `CHANGE_INTAKE_MAX_FRAMES` (120) after its latest mark, census or not, so without GPU timings the state ends on its own.
- *Measured* (RTX 3060, about 780 MHz, VALID, 3c93a29; `ddgi-key-probes-cost`, `ddgi-key-probes-sponza`): Sponza's 60 frames after the turn at a median **1.99 ms**, p95 **2.38** (still 2.02; before 2.44 / 3.12), a batch of 81 of 134 (median), taken in until past frame 89; the day 2.07 / 2.35. The first ~12 frames still run at 2.15-2.49 ms. They trace fewer rays than a still frame, but `gi/direct` lights more hits (6,600-7,700 against 4,850) and the trace takes longer per ray. The census's rays don't see that (a follow-up for M3).
- *The check* (lead ruling, DECISIONS 2026-11-08): the post-step median within 5% of the same run's still median; the p95 is reported beside the still p95. After the review fixes (27a63e3, VALID): 2.083 against 2.112 still (0.986×), p95 2.50 against 2.37 (1.05×), first frames 85-99 of 152. The view within 5% from frame 15 (bar 15; was 13), the field from frame 60 (was 36; bar 120), every fine probe traced by frame 90 (was 60) at 316 a frame. Two-room 9 / 18 and Cornell 5 / 8 frames, the furnished room 11 / 13: unchanged.

## Moving geometry (T2.M1, m3-moving.md §1 and §5, rigid objects, as built)

The platform's change feed says what moved and how far: a moved instance's two dirty regions (its old and new world box) carry `moved`, the farthest any corner of its mesh's box travelled, and its slot. `setTransforms` sends many in one call. `ddgi/mark` turns each geometry region (moved, added, removed, shown, hidden) into one entry over the probes within 8 spacings of it (mark-table.ts, wgsl/mark.ts):

- **The near set** (within 1.5 spacings of the box): `r = min(1, moved / spacing)`, or 1 for a whole change. Off probes are included, since the change may free them, and go first (`FREED_PROBE_URGENCY_SHARE` 16: they show nothing until they're back and warmed, and until then they cost 32 rays). Probes not updated yet are left alone. Every near probe is traced itself, with the full count of updates: its distances change too.
- **The far set** (active probes up to 8 spacings away that see the box): the probe's stored mean distance toward the box's centre reaches where that direction enters the box, less half a spacing (past the distance clamp, "nothing nearer than the clamp" counts); or its mover mask has the object's bit. Since T3.4, `r = min(1, Ω / 2π × contrast) × min(1, moved / spacing)`, with `Ω` the box's projected area seen from the probe over the distance to its centre squared, and `contrast` the brightest stored mean among the active probes within 1.5 spacings of the box's centre over the probe's own, at least 1 (`FAR_SET_REFERENCE_SOLID_ANGLE` says why; until T3.4 it was `Ω / 0.5 sr` with `Ω ≈ π radius² / distance²`). Served like a local light's probes: the full count, or keys and propagation on the key-probe path. **For an object that gives off light** (`DirtyRegion.emits`, T3.4) the far set is every active probe in reach, seen or not: its light reaches the others through the surfaces it lights. *Why:* `three-deforming-gi`'s emissive panel, the room's only light, unfolded with its exact box: 392 of about 1,434 active probes failed the visibility test though their light changed by 66% on average, so they were never updated again, and the probes that were marked bounced off them. The room stayed 6% (floor) and 15% (wall) under rest; with every probe in reach it comes back within 0.5%.
- **Woken air probes** (T3.4, m3-moving.md §5.3): a receiver (`SceneChanges.receivers`) gives a `receiver` entry, and a move or a pose wakes probes too. Probes within 2 spacings of the box are awake for 300 frames (history word 8, `wakeUntil`); one off for "no surface nearby" is owed 2 updates at the freed probes' urgency, comes back on at its next (fixed-ray) update and warms over 2 blended updates. `ddgi/blend` doesn't switch an awake probe off for "no surface nearby", and clears the wake at the first update after it runs out. Objects created, destroyed, shown or hidden don't wake (a scene's first frame creates everything).
- **Deforming meshes** (T3.1's poses): each pose logs the mesh's `deformBounds` box in world space with `moved`, the largest vertex travel, and the instance; `ddgi/mark` treats it as a move (near set, far set, mover mask, wake).
- **The mover mask** (history word 12): the technique keeps up to 32 hot movers (mover-table.ts: objects that moved in the last 300 frames, each with a bit and its current box) in a uniform buffer for `ddgi/blend`, which tests each blended ray's hit point (the probe's position plus direction times stored distance) against their boxes and keeps the bits hit. A hit on the mover always lands in its box, so nothing is missed; a hit on the floor under it over-marks. *Why in the blend and not the trace:* no instance id has to travel from the trace, whose storage buffers are full (T2.2 needs the last one).
- A move that slides a little every frame marks a little every frame: `pendingChange` adds up until the probe's next update, and the steady-change rule shortens the history, as for a light sweep.

Measured on the RTX 3060 (PROGRESS 2026-11-06, T2.M1; `ddgi-door-swing`, `ddgi-move-cabinet`, `ddgi-warm-two-updates`): the closed door swung shut over 1 s at 880 probes a frame leaves the dark room at 0.20% of the lit room's GI the frame after it closes and 0.17–0.18% 1 s later over two runs (open: 7.5%); a 1 m cabinet slide in the furnished room at 358 probes a frame has the pixels it changed within 5% of settled from frame 15-29 (580 and 358 probes a frame, two runs); probes freed by a move are active again 3 frames later, after exactly 2 blended updates. With T3.4 (PROGRESS 2026-11-09; `ddgi-wake-air-probes`, `ddgi-deforming-marks`): the cabinet's slide marks 645 probes instead of 993 (recall 97.4%), the door swing's leak is 0.157% at 1 s; a ball rays don't see, moved through Cornell's open middle, wakes the air probes around it and reads within 1.2% of a field with every probe on; a deforming sheet's pose marks its near probes with exactly `min(1, travel / spacing)`.

## Which probes update (the budget, and the GPU scheduler since T2.4b)


- **Unit = one probe update.** `TechniqueFrame.units` comes from the platform's budget v2 (above): the calibrated batch, at least the floor, stepped down only when frames are late. `FrameBudget.units` stays a hard cap. Without GPU timings, `FrameBudget.units`, default 512 on Tier A.
- **The round robin:** a fixed scrambled order of all probes, `order[i] = (i × stride) mod N` with `stride` coprime with `N` and near `0.618 N`, so each frame's set is spread across the volume, not one slab (schedule.ts).
- **The GPU scheduler** (wgsl/schedule.ts, m2-lights.md §4.3; M1's CPU cursor is gone) fills the update list on the GPU, one pass `ddgi/schedule` of four dispatches: `count` (each probe owed updates gets the priority `urgency × camera weight`, in one of 32 buckets each twice the one below; the histogram subgroups first, one atomic per bucket present in a subgroup, or the declared fallback, one per thread), `select` (the bucket where `priorityShare` × units, 75%, fills up), `write` (those probes reserve list slots, subgroups first or per thread, and are stamped with the frame), `fill` (the rest from the round robin, from the cursor, skipping probes stamped this frame; a workgroup scan, subgroups first when the device lays subgroups out in order, checked in the kernel, else in workgroup memory). The cursor moves past the entries used, so the round robin still reaches every probe once a pass. With nothing owed the list is exactly M1's order. The list holds exactly `min(units, N)` entries, so the trace, blend and write dispatches keep sizes the CPU knows.
- **Owed updates and urgency:** `ddgi/mark` promises updates (owed, 8 bits in history word 2; `takingChange` flag) once a probe's change passes ε, and its urgency is the change; each blend takes one off, and the urgency stays until the last. A feedback request is one owed update.
- **The camera weight:** `1 / (1 + (d / (4 × spacing))²)`, times 0.25 outside the view (the frustum's planes from `projection × view`, a cell's diagonal of margin; camera.ts and `ddgiCameraWeight`).
- **On a big volume** (K updates for every probe wouldn't fit in `FAR_RULE_FRAMES`, 30 frames, at this batch; Sponza: about 250, the furnished room 70), a change takes the key-probe path ("Key probes and propagation" above; `keyProbes`, on by default), or with `keyProbes` off **far probes owe 2** (m2-lights.md §6.4): after a global change, probes with a camera weight under 0.5 owe `farProbeUpdates` (2) instead of K.
- Off probes in the list still cost a workgroup, but only 32 rays and no blend, so they're cheap; a listed probe that isn't blended forgets what it was owed. The calibration measures the mix as it is.
- Cost: 0.020-0.027 ms at 20,000 probes on the RTX 3060 (`ddgi-schedule-reduction`, VALID), 0.025 ms in Sponza's frames.
- `maxProbesPerFrame` (default 4,096) sizes the update list and the scratch buffer. Dispatches over 65,535 workgroups use two dimensions.
- `FrameBudget.paused` skips every DDGI kernel.

## The warm-up (in the technique since T2.4b)

After `prepare` and after `resetHistory`, the technique does no probe work while the ray-tracing trees are still being built (`queries.rays.building`), then makes `warmUpPasses` (16) full passes over every probe, `maxProbesPerFrame` a frame whatever the frame's units, with no time between them for the history (the clock doesn't advance), stopping after `warmUpMaxSeconds` (8) once `warmUpMinPasses` (2) are made. `warmingUp` is true meanwhile (`GITechnique.warmingUp`), and the platform hands it no budget and leaves those frames out of the calibration, which starts at the next frame. It is what the viewer did itself before (now through `volume.start`): a host that only runs frames gets the same field after the same frames, value for value (`ddgi-warmup`). `warmUpPasses: 0` switches it off; GI then fills in at the budget. `volume.start` runs the frames back to back behind the host's loading screen, and `GIVolume.restart` runs them again.

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
   - with `samplingWeightFloors` (off by default; DV-06, DECISIONS 2026-11-19), RTXGI's floors: the weight before the crush is at least 1e-6, and each axis of the trilinear weight at least 0.001 (`Irradiance.hlsl`), so a point on a probe plane gives the far plane's probes a tiny weight;
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
    adaptiveHysteresis: boolean;     // true (T1.12): per probe, from its history and the time since its last update
    hysteresis: number;              // 0.97, used only with adaptiveHysteresis false
    hysteresisMemorySeconds: number; // 5 (τ); Infinity never forgets
    hysteresisMaxSamples: number;    // 200 (nMax; h at most 0.995)
    hysteresisUpdatesPerSample: number; // 8 (a, the ramp)
    hysteresisRampUpdates: number;   // 16 (K, how long the ramp lasts)
    irradianceChangeThreshold: number; // Infinity (off; DECISIONS 2026-11-05). A value turns on the lagged probe-mean rule
    changeDrivenHistory: boolean;    // true (T2.4a): reported light and sky changes cut the history ("Change-driven history")
    changeTolerance: number;         // 0.05 (epsilon)
    changeUpdates: number;           // 8 (K), or more when the bounces need more (T2.4b)
    changeBounceShare: number;       // 0.8 (rho, T2.4b): the share of a probe's light taken to come back through the field, for the cut
    lightChangeScale: number;        // 0.5 (T2.4a; the design's first guess was 0.25)
    priorityShare: number;           // 0.75 (T2.4b): the share of each frame's probes for probes owed updates
    farProbeUpdates: number;         // 2 (T2.4b): what far probes owe after a global change on a big volume, with keyProbes off
    keyProbes: boolean;              // true (T2.4c): key probes and propagation on a big volume; false: far probes owe 2
    keyProbeBoost: number;           // 4 (T2.4c): a marked key's urgency is its change times this
    keyResidualShare: number;        // 0.3 (T2.4c): a fine probe waiting for its keys: urgency share, and the least of its change it is cut for
    keyProbeUpdates: number;         // 4 (T2.4c): what a key away from the view's core owes
    holdFineProbes: boolean;         // false: a test mode (fine probes waiting for their keys aren't blended), for measuring propagation alone
    feedback: boolean;               // true (T2.4b): the feedback rule
    warmUpPasses: number;            // 16 (T2.4b): the warm-up's passes; 0 switches it off
    warmUpMaxSeconds: number;        // 8
    warmUpMinPasses: number;         // 2
    maxBrightnessChange: number;     // 2.0
    samplingWeightFloors: boolean;   // false (DV-06): RTXGI's floors on the sampling weights; live, through the volume table
    distanceSharpness: number;       // 50
    normalBias: number | 'auto';     // auto = 0.1 × min spacing
    viewBias: number | 'auto';
    maxRayDistance: number;          // Infinity
    bounceEnergy: number;            // 1.0. Native RTXGI has no such scale (its sample caps bounce albedo at 0.9)
    relocation: boolean;             // true
    minFrontFaceDistance: number | 'auto'; // auto = 0.02 × min spacing (T1.5b; was 0.1)
    classification: boolean;         // true
    maxProbesPerFrame: number;       // 4096, new prepare when changed
}
```

`bounceEnergy` defaults to 1 (corrected 2026-11-18: native RTXGI has no 0.95 bounce scale; its sample caps the bounce albedo at 0.9, which is `bounceAlbedoCap`, docs/divergence.md §6). At 0.95 the furnace room would settle 4.8% low (0.25 / (1 − 0.95 × 0.5) = 0.476 instead of 0.5), which would fail its 2% check. Light groups (one layer per group) are M2; the kernel takes a `groupMask` now and passes all bits.

## Debug data and counters

- **Points:** the probe-data texture already holds offset and state; a `points` debug view publishes world position and state in an `rgba32float` texture (written by the write kernel for updated probes). Legend: the state table above.
- **Counters:** probes updated, rays traced, probes active, probes off (each reason), probes not yet updated. Since T2.4a also: frames with changes marked, changes marked this frame, probes marked (this frame) and probes marked (total since `prepare`; the census keeps it and never clears it). Since T2.4b: warming up, warm-up passes, probes owed updates, probes updated at priority, feedback requests (this frame and total), probes blended (total).
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
