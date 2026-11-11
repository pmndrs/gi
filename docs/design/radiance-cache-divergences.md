# radiance-cache: differences from Capsaicin

Every way our radiance-cache port differs from AMD's Capsaicin v1.3 (GI-1.2, default settings), why, what it does to the image and to speed, how hard it would be to match Capsaicin, and what we recommend. The technique's design is in [radiance-cache.md](radiance-cache.md); this file was split out of it on 2026-11-11, merging its divergence register and its challenge list into one entry per difference.

Use it to decide what to fix. The summary table gives one line per difference; the full entries below it keep every citation and measurement. The shortlist at the end collects the rows worth acting on now.

## How to read a row

- **Area:** the part of the technique it touches.
- **Why (cause):** what made us differ.
  - *Platform can't:* WebGPU, WGSL, the browser or three r187 really can't do it Capsaicin's way, shown by a citation or a measurement.
  - *Platform can't? (unchecked):* claimed as forced, but only assumed. It stays on the list until someone checks.
  - *Our rule:* one of our own platform rules causes it (the rule is named). WebGPU could do it Capsaicin's way; the rule is the assumption to challenge.
  - *Quick setup:* it was easier to build this way. Nothing requires it.
  - *Bug fix:* Capsaicin goes wrong here (on our GPUs, or in its own logic), and we fixed it.
  - *GPU quirk:* a workaround for how one of our GPUs behaves, such as half-float rounding.
  - *Same result:* the code differs, the result doesn't (or it only affects measuring).
  - *Speed/memory:* done for speed or memory. No row is in this group yet: speed levers get a row only when one is adopted.
  - *License:* the license won't let us copy it.
- **Evidence:** measured (a GPU check; the GPU is named), cited (a spec section, a three line or a Capsaicin line), or assumed (nobody has checked).
- **Image impact:** how much it changes the picture: High, Medium, Low or None. For a bug fix, the impact is what reverting the fix would do, noted in the full entry.
- **Speed impact:** what it costs or saves, if known.
- **Effort to match Capsaicin:** S (under a day, contained), M (a few days, one module), L (a week or more, crosses the platform or a contract), XL (a platform or engine change, or research), n/a (nothing to remove, or forced).
- **Recommendation:** *Fix now* (cheap and it matters), *Fix with lever N* (folds into a planned lever in the design doc), *Measure first* (impact unknown; the measurement is named), *Keep* (same result, or not worth it), *Wait for platform* (truly forced), *Dennis decides* (a policy or scope call).

The original vocabulary, kept in the full entries:

- **Kind** (from the old register). P platform (WebGPU or our platform lacks something), L license, Q affects quality, S speed lever (only after the baseline). A row's "way back" says how to remove the difference, including what others do.
- **Cause** (from the old challenge list):
  - **Forced:** WebGPU, WGSL, the browser or three r187 can't do what Capsaicin does. It goes away only when the platform changes, or through a heavier workaround.
  - **Rule:** one of our own rules causes it, such as shared hit shading, diffuse-only platform materials, the `FrameScreen` contract or the world-cache interface. WebGPU could do it Capsaicin's way. The rule is the assumption to challenge.
  - **Shortcut:** it was easier to build. Neither the platform nor a rule requires it.
  - **Same:** the code is different but the result is the same, or the difference only affects measuring.
  - **Fix:** we fixed something that goes wrong in Capsaicin on our GPUs.
  - **License:** the license won't let us copy it.
- **Evidence** for the cause: **measured** (a GPU check on our devices; the GPU is named), **cited** (a spec section, a three source line or a Capsaicin line), or **assumed** (nobody has checked). A "Forced" that is only assumed doesn't count as forced: it stays on the list until someone checks it.
- **Challenge:** how much the difference changes the image (high, medium, low, none), and what removing it would take.

Added 2026-11-10 at Dennis's request. The register's "kind" says what a difference touches. This list asks a different question: **did anything make us do it?** Agents often pick the easier path, or repeat "WebGPU can't do X" from memory when it can. Once the port reaches parity, or only these differences are left, this list is where the focused work on challenging those assumptions starts.

Speed levers (kind S) get a row only when one is adopted. Phase 1 adds rows for any gap RC1.1 or RC1.4 finds (a three backend step, a missing feature), numbered from D13. D13 is held for RC1.1's three backend step, if one is needed; RC1.4's rows are D14 and D15. Numbers D43, D44 and D60 to D64 are unused.

## Summary

| ID | Area | What Capsaicin does (short) | What we do (short) | Why (cause) | Evidence | Image impact | Speed impact | Effort to match | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [D1](#d1) | Tracing | DXR inline ray queries on ray-tracing hardware | Our compute BVH tracer in WGSL | Platform can't: WebGPU has no ray tracing | Cited | None | Large: RC1.5 measured tracing alone at 12.4 / 21.9 / 23.8 ms (Sponza / Bistro interior / exterior, RTX 3060), 2.6-3.2× native's tracing | n/a: needs ray tracing in WebGPU; the speed gap is BVH work and levers 1, 7 and 9 | Wait for platform: the speed gap is worked on through the BVH and the levers |
| [D2](#d2) | Materials & lights | Samples the material's textures at every ray hit | One average colour per material; constant emission | Our rule: shared hit shading, one record per material (no bindless is forced; averaging isn't) | Cited for bindless; the average being close enough is assumed | High | Matching adds texture fetches at every hit (unknown) | L: texture arrays or an atlas with UVs at hits, in the platform's hit record (shared with DDGI) | Measure first: the image on Sponza and Bistro with real albedo at hits, against the Cycles references |
| [D3](#d3) | Screen inputs | Mesh-shader visibility buffer | three's MRT screen buffers through FrameScreen | Our rule: the FrameScreen contract (no mesh shaders is forced; the visibility buffer isn't) | Cited | Medium | unknown | L: a visibility channel in FrameScreen, from three r187's VisibilityBuffer pipeline or a raster pass | Measure first: D17's grazing-angle check on a normal-mapped scene decides whether it is worth it |
| [D4](#d4) | Materials & lights | Streamed light-sampler grid; 8-candidate reservoir per hit | The platform light grid; one light draw per hit | Our rule: one light system, shared hit shading | Assumed (that the light grid's image is close) | High | unknown; one draw is likely cheaper than eight candidates | L: port LightSamplerGridStream as a platform option (subgroups in the build) | Measure first: phase 5's image comparison with native Capsaicin on Bistro, where many lights matter |
| [D5](#d5) | Memory & bindings | Read-write half-float textures | texture-formats-tier2 read-write; ping-pong fallback | Same result | Measured (RTX 3060, Intel UHD) | None | none | n/a | Keep |
| [D6](#d6) | Memory & bindings | About 30 hash-cache buffers in unbounded arrays | A few packed buffers at fixed offsets, 8 per pass | Platform can't: no buffer arrays; storage buffers per stage | Cited and measured (RTX 3060, Intel UHD, M5) | None | none known | n/a | Keep |
| [D7](#d7) | Infrastructure | Ships blue-noise and Sobol tables | Regenerated tables (not built yet) | License: no license header on the tables | Cited | Low | none | M: generate the tables and serve them as a platform texture (this also fixes D27 and D55) | Keep: build the regenerated tables with D27 and D55; ask AMD only if they behave differently |
| [D8](#d8) | Materials & lights | GGX BRDF at hits; glossy reflections on by default | Lambert only; no reflections | Our rule: diffuse-only platform materials | Cited | High | Matching adds GGX at hits and a whole reflection pass (GI-1.1) | XL: roughness and metalness in the platform's material descriptor, GGX in hit shading, then GI-1.1's reflections | Dennis decides: a platform scope call (extension 2) |
| [D9](#d9) | Infrastructure | FSR 2/3 upscaling | @pmndrs/upscaler | Our rule: sibling libraries | Assumed equivalent | Low | unknown | n/a | Keep: check upscaler issue #7 (jitter against GI history) once GI runs upscaled |
| [D10](#d10) | Infrastructure | D3D12 per-pass timestamps | WebGPU timestamp-query per pass | Same result: measuring only | Cited | None | none | n/a | Keep |
| [D11](#d11) | Precision & GPU quirks | f32tof16 half packing | pack2x16float / unpack2x16float | Same result | Cited | None | none | n/a | Keep |
| [D12](#d12) | Infrastructure | One queue (id Tech 8 uses async compute) | One WebGPU queue | Platform can't: WebGPU has no async compute | Cited | None | unknown: no overlap of passes as async compute would give | n/a | Wait for platform: WebGPU multi-queue proposals |
| [D13](#d13) | Screen inputs | Reads its own visibility texture in place | Reads three's MRT textures through one version-pinned backend field | Platform can't: three r187 has no public API for a target's GPUTexture | Cited (three source) | None | Saves a full-screen copy per frame | n/a: a public three API (upstream) | Wait for platform: a public three API |
| [D14](#d14) | Memory & bindings | A device with the hardware's limits (about 1 GiB of cache) | Our device asks for the adapter's limits; three's own device has WebGPU's defaults | Platform can't: three's default device | Measured (RTX 3060, Intel UHD) | None | none | n/a | Keep: hosts pass our device or requiredLimits; a three change upstream would remove it |
| [D15](#d15) | Precision & GPU quirks | f32 math, halves only in storage | The same; any f16 math must stay under 65,504 | GPU quirk: WGSL allows f16 overflow to differ (§15.7.3) | Measured (RTX 3060 against Intel UHD) | None | none | n/a | Keep: a rule for future f16 speed work |
| [D16](#d16) | Screen probes | Some probe rays sample the GGX lobe | Every probe ray takes the diffuse branch | Our rule: follows D8 (diffuse-only materials) | Cited | Medium | none | M: comes back with D8, plus a roughness channel in FrameScreen | Dennis decides: comes with D8 |
| [D17](#d17) | Screen inputs | Exact triangle point and geometric normal from its visibility buffer | Point rebuilt from depth; three's shading normal | Our rule: FrameScreen has no visibility buffer or geometric normal | Measured (M5): no self-shadowing seen on Sponza or two-room; Cornell inconclusive | Low | none | L: a visibility or geometric-normal channel in FrameScreen (D3) | Measure first: a scene with strong normal maps at grazing angles |
| [D18](#d18) | Infrastructure | PCG seeded from a CPU table of random numbers | PCG seeded by a hash | Quick setup | Argued, not measured | Low | none | S: upload a seed table | Keep |
| [D19](#d19) | Precision & GPU quirks | A miss stores infinity (or 1e9) | A miss stores 65,504, the largest half | Platform can't: WGSL's half range | Cited | None | none | n/a | Keep |
| [D20](#d20) | Materials & lights | Missed rays sample the environment cube map | The platform sky, baked to 256 × 128 | Our rule: shared sky bake | Assumed (that the bake is detailed enough) | Low | none (the bake size is a setting) | S: raise the bake size, or read the host's environment map | Measure first: a bigger bake on an outdoor scene; compare the probes' E |
| [D21](#d21) | Materials & lights | Emissive triangles are lights | Only point, spot and directional lights plus the sky | Our rule: platform lights are punctual only | Measured (cornell-box, RC2.1) | High | Matching adds light sampling over emissive triangles (unknown) | L: emissive triangles as platform lights, sampled in giDirectIrradiance and gi/direct (shared with DDGI) | Dennis decides: a platform task, also carried for DDGI (area lights, M2) |
| [D22](#d22) | World cache | Strong compare-exchange to claim a slot | Weak compare-exchange, retried | Platform can't: WGSL has only the weak exchange | Cited | None | negligible | n/a | Keep |
| [D23](#d23) | World cache | Bucket overflow counted only in debug builds | One always-on counter | Same result: a counter | Cited | None | none (one atomic on the refusal path) | n/a | Keep |
| [D24](#d24) | World cache | Lookup with the ray's length and a jitter | Lookup without them | Our rule: the world-cache lookup signature; partly a quick setup | Cited | Low | none | S: add the fields when GI-1.1's reflections are ported | Keep: add them with GI-1.1 |
| [D25](#d25) | Screen probes | SH weight 1/probe_size: GI reads about 30% bright | The cell's solid angle: 1.021 πL | Bug fix | Cited and measured (CPU; M5) | None | none | n/a | Keep: tell AMD |
| [D26](#d26) | Memory & bindings | SH in a storage buffer | SH in an rg32uint texture | Our rule: fragment-readable data lives in textures | Cited | None | none | S: a read-only storage buffer if the rule goes | Keep |
| [D27](#d27) | Screen probes | Blue-noise jitter in the interpolation | White noise | Quick setup: the platform has no blue noise yet | Cited | Low | none | M: blue-noise tables as a platform texture (D7), shared with D55 | Measure first: the flicker after the denoiser with blue noise against white |
| [D28](#d28) | Precision & GPU quirks | SH packed; overflow gives infinity | Clamped to ±65,504 before packing | Platform can't: WGSL's pack2x16float range | Cited | None | none | n/a | Keep |
| [D29](#d29) | Screen probes | Linear depth by the reversed-Z formula | View-axis distance of the rebuilt point | Same result | Measured (CPU) | None | none | n/a | Keep |
| [D30](#d30) | World cache | Insert inside the probe trace; 4-word visibility entry | A separate insert kernel; 10-word entry | Our rule: the world-cache interface | Cited | Low | An extra dispatch and 10-word entries where Capsaicin has 4 (not timed) | M: fuse the insert into the trace when the hash grid is the cache; 4-word entries once D37 goes | Fix with lever 5: compact formats (16-byte hit records) |
| [D31](#d31) | Precision & GPU quirks | f32tof16 rounds to nearest | pack2x16float plus a rounding correction | GPU quirk: WGSL leaves the rounding to the GPU | Measured (RTX 3060 and Intel UHD truncate; M5 rounds) | None | negligible | n/a | Keep |
| [D32](#d32) | World cache | A new tile can be listed twice, and two workgroups race | A per-tile word lets one thread list it | Bug fix | Measured (Intel UHD) | None | 1 MiB of memory | n/a | Keep: tell AMD |
| [D33](#d33) | Materials & lights | The sky is a light that cache hits sample | Closed by RC6.1: the sky is a candidate of the platform's light choice at cache hits; left: always a candidate (Capsaicin's competes with more than 8 lights), weighed at the bake's level 0, not a blurred level | Our rule (D4's one light system) and Quick setup (no mips in the bake) | Measured (M5, `radiance-cache-world-cache`) and cited | Low (noise only) | One sky read per hit; a shadow ray where the sky is drawn | S for the mips; the rest is D4's L | Keep; revisit with D4 |
| [D34](#d34) | Precision & GPU quirks | A ray slightly past the normal gives NaN and lands in a random cell | Clamped: the ray maps to the pole cell | Platform can't: WGSL gives Capsaicin's line no defined result; the pole is our fix | Cited and measured (M5) | Low | none | n/a | Keep |
| [D35](#d35) | World cache | Cache purge at the start of the frame | Purge after the screen trace, before the insert | Same result: order only | Cited | None | none | n/a | Keep |
| [D36](#d36) | World cache | Shadow-ray cosine on the geometric normal | The platform's light ray: cosine on the shading normal | Our rule: shared hit shading (one light evaluation) | Cited | Low | none | S: a geometric-normal option in giDirectLightRay | Keep |
| [D37](#d37) | World cache | Short-ray test from the depth point | From the pushed ray origin | Our rule: the world-cache interface; follows D17 | Assumed | Low | none | S: hand the cache the unpushed depth point | Measure first: count the rays that change path against the unpushed point |
| [D38](#d38) | Screen probes | Reads a missing neighbour probe and multiplies by 0 | Skips it | Same result: a guard | Cited | None | none | n/a | Keep |
| [D39](#d39) | Multi-bounce | offsetPosition push for the bounce ray | The platform's push | Quick setup | Measured only for shadow rays | Low | none | S: port offsetPosition for this ray (8 lines) | Fix now: cheap and restores the faithful port; count the second-bounce rays that change hit |
| [D40](#d40) | Multi-bounce | Wrong index: light lands on unrelated probe rays | Correct indexing with a bypass list of our own | Bug fix | Cited and measured (M5) | None | 921,600 B list at 720p | n/a | Keep: tell AMD |
| [D41](#d41) | Multi-bounce | Vertex normals not transformed by the instance | World-space normals | Bug fix | Cited; the effect is assumed | Low | none | n/a | Keep: measure on a scene with rotated instances when one exists |
| [D42](#d42) | Multi-bounce | Multi-bounce is a compile switch; its memory is freed when off | A live setting; 535 MB stays allocated when off | Quick setup: the memory (the setting itself is the same result) | Cited | None | 535 MB held with multi-bounce off | S: size the buffers from useMultibounce at prepare | Fix now: cheap; matters on memory-tight devices |
| [D45](#d45) | Screen probes | Two history textures, swapped each frame | One texture plus a copy into a previous one | Same result: three re-wraps a texture whose identity changes | Cited and measured (M5) | None | Copies about 8.5 MB a frame at 720p (57 KB mask, 1 MB SH, 7.4 MB atlas) | M: double-buffered textures in the shading contract | Keep: unless phase 5 shows the copies cost |
| [D46](#d46) | Screen probes | Reads and writes the atlas in place | A third atlas carries one value between passes | Our rule: the platform doesn't require texture-formats-tier2 | Cited | None | 7.4 MB at 720p | S: use the tier-2 binding where the device has it | Keep |
| [D47](#d47) | Precision & GPU quirks | Float texture holding raw bits | rgba32uint holding the same bits | Platform can't? (unchecked): subnormal flushing, assumed | Assumed | None | none | n/a | Keep: re-read the WGSL section |
| [D48](#d48) | Screen probes | A sky cell's hit-distance sum wraps around | Capped so it can't wrap | Bug fix | Measured (M5) | Low | none | n/a | Keep |
| [D49](#d49) | Screen inputs | The framework keeps the previous camera, depth and normals | The technique keeps its own copies | Our rule: the FrameCamera and FrameScreen contract | Cited | None | A full-screen copy per frame | L: previous buffers in the platform contract, shared by every screen technique | Keep |
| [D50](#d50) | Screen inputs | uv motion, y down | NDC motion, converted on read | Same result: a unit conversion | Cited | None | none | n/a | Keep |
| [D51](#d51) | Precision & GPU quirks | Relies on D3D for out-of-range reads | Explicit guards | Same result: guards | Cited | None | negligible | n/a | Keep |
| [D52](#d52) | Precision & GPU quirks | f32tof16 rank key | pack2x16float, or infinity's bits past the range | Platform can't: WGSL's half range | Cited | None | none | n/a | Keep |
| [D53](#d53) | Screen probes | 64 threads race to clear a flag | One thread clears it with atomics | Same result | Cited | None | none | n/a | Keep |
| [D54](#d54) | Memory & bindings | Probe-cache buffers sized for the screen | Scan buffers sized for the largest possible screen | Quick setup | Cited | None | About 16 MB of memory | S: make the scans per screen size | Keep |
| [D55](#d55) | Filter & SSGI | Blue-noise slice angles | White noise | Quick setup: the platform has no blue noise yet | Cited | Low | none | M: blue-noise tables (D7), shared with D27 | Measure first: the flicker after the denoiser (SSGI is most of it) |
| [D56](#d56) | Filter & SSGI | Last frame's combined lighting | The host's last picture through a new FrameScreen channel | Our rule: FrameScreen: the host draws the picture | Cited and measured (M5) | Low | One extra full-screen pass, 8 bytes a pixel (not timed) | L: a platform combine pass, if the platform ever draws direct light itself | Keep |
| [D57](#d57) | Filter & SSGI | SSGI is a technique of its own, sharing textures | A module inside the radiance cache | Quick setup | Cited | None | none | M: a platform service for shared screen textures | Keep |
| [D58](#d58) | Filter & SSGI | Relies on D3D's NaN rules | Guards that give the same answer | Same result: guards | Cited and measured (M5) | None | none | n/a | Keep |
| [D59](#d59) | Precision & GPU quirks | Half-float texture stores round to nearest (D3D) | The same stores; the M5 rounds them down | Platform can't? (unchecked): forced if WGSL leaves the store's rounding to the GPU (not re-read) | Measured (M5); Windows GPUs not run | Low | none | S: store through pack2x16float into rg32uint, or round stochastically before the store | Fix now: already the agreed follow-up; measure the Windows GPUs too |
| [D65](#d65) | Denoiser | R8_SNORM and R16_FLOAT textures | The same with texture-formats-tier1; a declared fallback without | Same result: the fallback alone is forced | Measured (M5) | Low | Fallback only: 7 bytes a pixel more (6.5 MB at 720p) | n/a | Keep |
| [D66](#d66) | Denoiser | Writes its input buffer in place | Its own input and history textures | Our rule: the module takes the caller's texture | Cited | None | One more screen texture | n/a | Keep |
| [D67](#d67) | Denoiser | Sky told by a zero normal | Sky told by far-plane depth | Our rule: the FrameScreen and FrameGuides contract | Cited; the sky neighbour's weight assumed | None | none | n/a | Keep |
| [D68](#d68) | Denoiser | The framework keeps the previous depth and normals | The denoiser keeps its own copy | Our rule: the module is self-contained (D49) | Cited | None | 11 MB at 720p and a copy | M: optional previous guides in the module's inputs | Keep |
| [D69](#d69) | Denoiser | World-space inverse matrices | Camera-relative matrices in f64 on the CPU | Same result | Cited and measured | None | none | n/a | Keep |
| [D70](#d70) | Denoiser | Reversed-Z linear depth | View-axis distance | Same result | Cited and measured | None | none | n/a | Keep |
| [D71](#d71) | Denoiser | NaN removal through D3D's saturate | A bit test for non-finite values | Platform can't: WGSL's NaN rules | Cited and measured (M5) | None | none | n/a | Keep |
| [D72](#d72) | Denoiser | Point-sampled reads | textureLoad of the same texel | Same result | Assumed | None | none | n/a | Keep |
| [D73](#d73) | Denoiser | Always on; history starts empty only when made | A switch, and resets also clear its history | Same result | Cited | None | none | n/a | Keep |
| [D74](#d74) | Denoiser | uv motion, TAA-jittered matrices, fovY | @pmndrs/denoiser's conventions, converted | Same result | Cited | None | none | n/a | Keep |
| [D75](#d75) | Denoiser | HLSL's integer sign | Sign by comparisons | Platform can't: WGSL's sign(NaN) is undefined | Cited | None | none | n/a | Keep |

**Totals.** By cause: Our rule 20, Same result 18, Platform can't 12, Quick setup 8, Bug fix 5, GPU quirk 2, Platform can't? (unchecked) 2, License 1.
By recommendation: Keep 49, Measure first 8, Fix now 4, Wait for platform 3, Dennis decides 3, Fix with lever 5 1.

## Disputed causes

Rows whose stated cause looks wrong on reading the evidence (2026-11-11). The cause above is kept as stated until the lead decides.

- **D12:** Capsaicin also uses one queue, so this isn't a difference from Capsaicin at all; it is a gap against id Tech 8 (async compute). It belongs with the speed levers.
- **D26:** The rule's reason (compatibility mode may allow no fragment storage buffers) doesn't buy anything here: the row's own evidence says `rg32uint` is refused as a storage texture in compatibility mode too.
- **D59:** Called forced, but only if WGSL leaves the rounding open (not re-read), and a known workaround exists (D31's correction, or stochastic rounding). It is better described as a GPU quirk with a fix.
- **D18:** Called a shortcut, but the evidence says the seed table is random data that can't be matched anyway, so the result is the same in distribution: closer to "Same result".
- **D13:** Called forced, but the backend read is a choice over a per-frame copy, which three does allow; what is forced is only the missing public API.

## What the list shows so far

*As of 2026-11-10, after RC3.2's review and RC2.2, with later additions.* The differences that are truly forced (D1, D6, D12 to D15, D19, D22, D28, D31, D34) all give the same image, or only cost speed. Every difference marked high or medium for the image comes from **our own rules**: D2, D3, D4, D8, D16, D17 and D21. The one exception was D33 (no sky light at cache hits, high outdoors), a **shortcut**; RC6.1 closed it (2026-11-11), and what is left of it changes only the noise. Most of the rules are DDGI's shared hit shading and diffuse-only materials, which the platform carried over. RC2.2 adds one fix (D25, Capsaicin's SH projection weight, high if reverted) and one shortcut (D27, white-noise jitter until the platform has blue noise, low); its other rows (D26, D28, D29, D38) give the same image. RC3.2's small differences (D35 to D37) are low or none. RC3.3 (GI-1.2's multibounce) adds two fixes of Capsaicin's own indexing (D40, medium near walls; D41, low), a shortcut on the bounce ray's push (D39, low) and the option as data (D42, none). The multibounce's biggest effect on the image isn't a difference at all: Capsaicin's own second-bounce estimator reads high (see the design doc's phase 3 notes), and the port keeps it. The challenge work is therefore mainly platform work (materials, area and sky lights, real textures at hits, a visibility channel), not a fight with WebGPU. RC4.1 (D45-D54) adds one fix (D48, the capped hit-distance sum), one shortcut (D54, scan buffers for the largest screen, memory only), two contract rules (D46, D49) and six rows that give the same result (D45, D47, D50-D53); none changes the image measurably. The large change history brings (a constant sky reads 0.69 πL with it, phase 4 notes) is not a difference from Capsaicin: a model of Capsaicin's own backup, hysteresis, CDF feedback and taps gives the same 0.690; native Capsaicin is still to confirm it in phase 5. RC4.2 (D55-D59, `FilterScreenProbes` and Capsaicin's SSGI) adds the M5's half-float texture stores rounding down (D59, forced if WGSL allows it, low), two shortcuts (D55, white noise for SSGI's blue noise, low; D57, SSGI as a module, none), one contract rule (D56, last frame's lit colour from the host through a new `FrameScreen` channel, low) and guards that give Capsaicin's answers (D58); it widens D17, D29, D45 and D51 to the new passes. Again the large change to the image is Capsaicin's own, not a difference: its SSGI, at its default of one slice, darkens open surfaces, and adds near-field light at about E / π (the design doc's phase 4 notes, and the list for AMD below). RC4.3 (the image-space GI denoiser, D65-D75) adds two forced rows (D71, NaN removal, measured; D75, the sign of a NaN, cited), one difference only on devices without `texture-formats-tier1` (D65, a declared fallback since the review), three rules from the module being self-contained or the contract (D66-D68) and five that give the same result. D75 is outside RC4.3's range D65-D74: the lead may renumber it at the merge.

## Each difference in full

### D1

*Tracing.* Capsaicin traces rays on the GPU's ray-tracing hardware. WebGPU has no ray tracing, so we trace with our own BVH in compute shaders. The image is the same; the cost is speed, and it is the biggest speed gap we know of.

- **Capsaicin:** DXR 1.1 inline ray queries; hardware traversal; alpha test in the loop
- **Ours:** The platform's compute BVH (`giTraceClosest`, `giTraceAny`); cutouts from the material
- **Kind:** P
- **Why:** WebGPU has no ray tracing
- **Cause:** Forced
- **Evidence:** Cited: WebGPU has no ray-tracing API
- **What is really behind it:** Hardware traversal is out of reach
- **Challenge:** None for the image. Speed is the open question: phase 5's per-pass table shows how far our compute BVH is from DXR
- **Way back / others' workarounds:** Our BVH work (wide nodes, H-PLOC, any-hit fixes); Brixelizer's SDF backend (in progress elsewhere); any WebGPU ray-query proposal; compare against three-mesh-bvh's WebGPU walk and Webgiya
- **Assessment (2026-11-11):** Platform can't: WebGPU has no ray tracing. Image impact none. Speed: Large: RC1.5 measured tracing alone at 12.4 / 21.9 / 23.8 ms (Sponza / Bistro interior / exterior, RTX 3060), 2.6-3.2× native's tracing. Effort to match: n/a: needs ray tracing in WebGPU; the speed gap is BVH work and levers 1, 7 and 9. Recommendation: Wait for platform: the speed gap is worked on through the BVH and the levers.

### D2

*Materials & lights.* Capsaicin reads the actual texture colour where a ray hits. We use one average colour per material, because that is how the platform's shared hit shading works for DDGI. WebGPU has no bindless textures, but texture arrays or an atlas work today, so this is our choice, not a hard limit.

- **Capsaicin:** Bindless material textures sampled at every hit; an emissive hit's emission is its emissive texture at the hit's UV, scaled by alpha, through `evaluateAreaLight` (`emissiveAlphaScaled`, `gi1.comp:2441-2445`)
- **Ours:** Texture-average albedo through `giSurfaceAt` (as DDGI); an emissive hit's emission is the material's constant `surface.emission` (RC3.2's `GenerateReservoirs`)
- **Kind:** P, Q
- **Why:** WebGPU has no bindless textures; the platform's hit record is per material
- **Cause:** Rule (missing bindless is forced; averaging is not)
- **Evidence:** Cited for bindless; assumed that the average is close enough
- **What is really behind it:** Texture-average albedo is how DDGI's `giSurfaceAt` works, and the material's constant emission stands in for Capsaicin's alpha-scaled emissive texture at the hit (`gi1.comp:2441-2445`). Texture arrays or an atlas, with UVs at hits, work in WebGPU today
- **Challenge:** **High** on textured scenes (Sponza, Bistro). Real albedo and emission at hits through an array or atlas
- **Way back / others' workarounds:** Texture arrays or an atlas with real UVs at hits; compressed textures (GPUTex, T3.12) to afford it; the WebGPU bindless proposal
- **Assessment (2026-11-11):** Our rule: shared hit shading, one record per material (no bindless is forced; averaging isn't). Image impact high. Speed: Matching adds texture fetches at every hit (unknown). Effort to match: L: texture arrays or an atlas with UVs at hits, in the platform's hit record (shared with DDGI). Recommendation: Measure first: the image on Sponza and Bistro with real albedo at hits, against the Cycles references.

### D3

*Screen inputs.* Capsaicin draws a visibility buffer that says which triangle each pixel sees. We take the depth, normals and other buffers three already draws. WebGPU lacks mesh shaders, but a visibility buffer doesn't need them; we chose the simpler contract.

- **Capsaicin:** Mesh-shader visibility buffer (instance, primitive, barycentrics)
- **Ours:** Host-provided screen buffers through `FrameScreen` (depth, normal, velocity, albedo, instance id); three fills them from its MRT scene pass
- **Kind:** P
- **Why:** WebGPU has no mesh shaders; three already renders these
- **Cause:** Rule (missing mesh shaders is forced; the visibility buffer is not)
- **Evidence:** Cited
- **What is really behind it:** A visibility buffer doesn't need mesh shaders: three r187 has a VisibilityBuffer pipeline, and a raster pass over the scene copy would also do. We chose MRT outputs through `FrameScreen`
- **Challenge:** Medium, mostly through D17. A visibility channel in `FrameScreen`
- **Way back / others' workarounds:** three's own VisibilityBuffer pipeline; a platform raster pass over the scene copy for hosts that can't provide them
- **Assessment (2026-11-11):** Our rule: the FrameScreen contract (no mesh shaders is forced; the visibility buffer isn't). Image impact medium. Speed: unknown. Effort to match: L: a visibility channel in FrameScreen, from three r187's VisibilityBuffer pipeline or a raster pass. Recommendation: Measure first: D17's grazing-angle check on a normal-mapped scene decides whether it is worth it.

### D4

*Materials & lights.* Capsaicin chooses which light each cache cell tests with its own sampler, weighing 8 candidates. We reuse the platform's light grid, which DDGI also uses, and draw one light. Nothing in WebGPU forces this; it keeps one light system on the platform.

- **Capsaicin:** `LightSamplerGridStream` (64 streamed reservoirs per cell, wave intrinsics in the build)
- **Ours:** The platform light grid (16 picks per cell weighed at the hit, N shadow rays per hit). The radiance cache's `GenerateReservoirs` draws one light per hit from the same candidates (`giDirectChooseLight`: the directional lights, then the point and spot lights or the grid's picks for the hit's cell, each weighed by the luminance of its unshadowed irradiance times its share), with weight share / probability in the role of the reservoir's W; `PopulateCells` traces its one shadow ray (`giDirectLightRay`, `giTraceAny`) (RC3.2). `GenerateMultibounceReservoirs` draws the same way for the second-bounce hits (RC3.3). That one draw replaces Capsaicin's streamed reservoir of `kReservoir_SampleCount` = 8 candidates (`sampleLightsListCone<8>`, `world_space_restir.hlsl:30`, `gi1.comp:2513`). The insert's `requestLightSampleLocation` (`gi1.comp:1019`), which bounds Capsaicin's light sampler grid to the frame's hits, is dropped (also in `PopulateMultibounceCells`, gi1.comp:1796): the platform grid covers the scene's bounds
- **Kind:** P, Q
- **Why:** The platform rule "hit shading is shared"; no second light system
- **Cause:** Rule (one light system, shared hit shading)
- **Evidence:** Assumed that the light grid's image is close
- **What is really behind it:** WebGPU has subgroups now, so the stream sampler's build can be ported. The cache's one light draw also stands in for Capsaicin's 8-candidate reservoir (`world_space_restir.hlsl:30`), and `requestLightSampleLocation` is dropped
- **Challenge:** **High**: it is the light sampling behind every cache cell. Port `LightSamplerGridStream` as a platform option and compare the images
- **Way back / others' workarounds:** Port the stream sampler as a platform option (subgroups in the build) if the light grid's image differs measurably from Capsaicin's
- **Assessment (2026-11-11):** Our rule: one light system, shared hit shading. Image impact high. Speed: unknown; one draw is likely cheaper than eight candidates. Effort to match: L: port LightSamplerGridStream as a platform option (subgroups in the build). Recommendation: Measure first: phase 5's image comparison with native Capsaicin on Bistro, where many lights matter.

### D5

*Memory & bindings.* Capsaicin reads and writes half-float textures in place. WebGPU needs an optional feature for that; both Windows GPUs have it, and the fallback works too.

- **Capsaicin:** Read-write `RGBA16F` textures
- **Ours:** `texture-formats-tier2` read-write storage textures
- **Kind:** P
- **Why:** Core WebGPU allows read-write storage only on 32-bit one-channel formats
- **Cause:** Same
- **Evidence:** Measured (RTX 3060, Intel UHD; RC1.4)
- **What is really behind it:** The feature is there; the fallback also works
- **Challenge:** None. M5 not checked yet
- **Way back / others' workarounds:** Declared fallback: ping-pong (a sampled read plus a write-only storage texture). RC1.4: the feature works on both GPUs, on our device and three's, and the fallback was run and works too
- **Assessment (2026-11-11):** Same result. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D6

*Memory & bindings.* WebGPU can't bind arrays of buffers, and it limits how many buffers a pass can see. We pack the cache into a few buffers at fixed offsets. Same data, same result.

- **Capsaicin:** Unbounded buffer arrays (about 30 hash-cache buffers)
- **Ours:** Packed storage buffers at fixed offsets, one per element type: the cache words (counters, per-tile hash, decay, packed lists and update-listed frame, then the per-cell update scratch) and the cell values; the per-ray lists in a third, made for the screen. Each pass binds only the buffers it uses (`POPULATION_BINDINGS`): `GenerateReservoirs` and `PopulateCells` bind the cache words, the radiance slots and the per-ray lists beside hit shading's 5 (the tracer's 3, the scene copy's 2), exactly the platform's 8
- **Kind:** P
- **Why:** WebGPU has no buffer arrays, and a stage binds at most `maxStorageBuffersPerShaderStage` storage buffers. The platform budgets WebGPU's default of 8 (`MIN_STORAGE_BUFFERS_PER_STAGE`); the adapters offer 16 (RTX 3060, Intel UHD) and 10 (M5, Chrome 157). A unit test builds every hash-grid kernel through the kernel builder at 8, and the radiance-cache GPU pages run on a device held at 8
- **Cause:** Forced (no buffer arrays in WebGPU; storage buffers per stage)
- **Evidence:** Cited (WebGPU's default `maxStorageBuffersPerShaderStage` is 8, the platform's budget) and measured: the adapters offer 16 (RTX 3060, Intel UHD) and 10 (M5); every hash-grid kernel is built at 8 in a unit test and on the M5
- **What is really behind it:** Same data in fewer buffers; each pass binds only what it uses
- **Challenge:** None
- **Way back / others' workarounds:** None needed; same data
- **Assessment (2026-11-11):** Platform can't: no buffer arrays; storage buffers per stage. Image impact none. Speed: none known. Effort to match: n/a. Recommendation: Keep.

### D7

*Infrastructure.* Capsaicin's noise tables carry no license, so we can't copy them. The plan is to generate our own; they haven't been built yet, which is why D27 and D55 use white noise.

- **Capsaicin:** Shipped blue-noise and Sobol tables
- **Ours:** Regenerated tables (or the platform's blue noise)
- **Kind:** L
- **Why:** The tables have no license header
- **Cause:** License
- **Evidence:** Cited: no license header on the tables
- **Challenge:** Low. Ask AMD if the noise behaves differently
- **Way back / others' workarounds:** Ask AMD for terms if the regenerated noise behaves differently
- **Assessment (2026-11-11):** License: no license header on the tables. Image impact low. Speed: none. Effort to match: M: generate the tables and serve them as a platform texture (this also fixes D27 and D55). Recommendation: Keep: build the regenerated tables with D27 and D55; ask AMD only if they behave differently.

### D8

*Materials & lights.* Capsaicin shades hits with a full metal and roughness BRDF and traces glossy reflections by default. The platform's materials are diffuse only, so we do neither. WebGPU doesn't stop us; it is a platform scope decision, and it is the biggest look difference on shiny scenes.

- **Capsaicin:** GGX BRDF at hits from a packed material; glossy reflections on by default
- **Ours:** Lambert at hits; no reflections until the platform has roughness and metalness; GI-1.2's second-bounce ray is drawn and weighed as Lambert's (brdf / pdf = albedo), where Capsaicin forces the second vertex to roughness 1 and evaluates its GGX with the diffuse compensation (gi1.comp:1909-1914; RC3.3)
- **Kind:** P, Q
- **Why:** Platform materials are diffuse-only (`contract/scene.ts`)
- **Cause:** Rule (diffuse-only platform materials, `contract/scene.ts`)
- **Evidence:** Cited
- **What is really behind it:** Nothing in WebGPU stops GGX at hits
- **Challenge:** **High**. It also blocks GI-1.1 reflections, which are on by default in Capsaicin. Roughness and metalness in the material descriptor
- **Way back / others' workarounds:** Roughness and metalness in the material descriptor (also needed by planned RT reflections)
- **Assessment (2026-11-11):** Our rule: diffuse-only platform materials. Image impact high. Speed: Matching adds GGX at hits and a whole reflection pass (GI-1.1). Effort to match: XL: roughness and metalness in the platform's material descriptor, GGX in hit shading, then GI-1.1's reflections. Recommendation: Dennis decides: a platform scope call (extension 2).

### D9

*Infrastructure.* Capsaicin upscales with AMD's FSR; we use our sibling library. They should be equivalent; the open question is how its jitter meets the GI history.

- **Capsaicin:** FSR 2/3 in the frame chain
- **Ours:** `@pmndrs/upscaler`
- **Kind:** P
- **Why:** Our sibling library
- **Cause:** Rule (siblings)
- **Evidence:** Assumed equivalent
- **What is really behind it:** Our own upscaler instead of FSR
- **Challenge:** Low. Check upscaler issue #7 (jitter against GI history)
- **Way back / others' workarounds:** Equivalent; watch upscaler issue #7 (jitter against GI history)
- **Assessment (2026-11-11):** Our rule: sibling libraries. Image impact low. Speed: unknown. Effort to match: n/a. Recommendation: Keep: check upscaler issue #7 (jitter against GI history) once GI runs upscaled.

### D10

*Infrastructure.* Both time each pass on the GPU; only the API differs.

- **Capsaicin:** D3D12 per-pass timestamps
- **Ours:** WebGPU `timestamp-query` per pass
- **Kind:** P
- **Why:** Measurement only
- **Cause:** Same
- **Evidence:** Cited
- **What is really behind it:** Measuring only
- **Challenge:** None
- **Way back / others' workarounds:** The platform's `GpuTimer` already does this
- **Assessment (2026-11-11):** Same result: measuring only. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D11

*Precision & GPU quirks.* The same half-float packing under WGSL's names. Rounding differences are D31.

- **Capsaicin:** Half-precision packing via `f32tof16`
- **Ours:** `pack2x16float` / `unpack2x16float`; `shader-f16` where it helps
- **Kind:** P
- **Why:** WGSL equivalents
- **Cause:** Same
- **Evidence:** Cited
- **What is really behind it:** WGSL names for the same packing
- **Challenge:** None (D31 covers rounding)
- **Way back / others' workarounds:** None needed
- **Assessment (2026-11-11):** Same result. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D12

*Infrastructure.* Capsaicin runs everything on one queue, as we do. WebGPU has no async compute, so id Tech 8's way of overlapping work isn't available to us.

- **Capsaicin:** One queue (id Tech 8 uses async compute)
- **Ours:** One WebGPU queue
- **Kind:** P
- **Why:** WebGPU has no async compute
- **Cause:** Forced
- **Evidence:** Cited: one queue in WebGPU
- **What is really behind it:** Speed only
- **Challenge:** None for the image
- **Way back / others' workarounds:** WebGPU multi-queue proposals; order passes to keep the GPU busy
- **Assessment (2026-11-11):** Platform can't: WebGPU has no async compute. Image impact none. Speed: unknown: no overlap of passes as async compute would give. Effort to match: n/a. Recommendation: Wait for platform: WebGPU multi-queue proposals.
- **Disputed:** Capsaicin also uses one queue, so this isn't a difference from Capsaicin at all; it is a gap against id Tech 8 (async compute). It belongs with the speed levers.

### D13

*Screen inputs.* three r187 has no public way to hand us its render targets. We read one internal field, pinned to r187, instead of copying every frame.

- **Capsaicin:** The visibility buffer is Capsaicin's own texture, read in place
- **Ours:** `ddgi-three` hands three's MRT targets to `FrameScreen` without a copy (except the instance id: blending needs an 8-bit target, which a small pass turns into the `r32uint` texture each frame) by reading one field of three r187's WebGPU backend, `renderer.backend.get(texture).texture`, in one function (`three-gpu-texture.ts`) that refuses any other three revision
- **Kind:** P
- **Why:** three r187 has no public way to get a render target's `GPUTexture` (only the other way: `ExternalTexture` wraps ours). A copy into our own textures would cost bandwidth every frame; the backend field is what three's own XR path writes
- **Cause:** Forced (three r187 API)
- **Evidence:** Cited (three source)
- **What is really behind it:** A version-pinned read of three's backend; same data
- **Challenge:** None
- **Way back / others' workarounds:** A public three API returning a texture's `GPUTexture` (none asked for upstream as of 2026-11-09: no issue or PR found; #31653 added the reverse); or the host renders into `GPUTexture`s we own, wrapped as `ExternalTexture`s in the pass's render target, which needs resizes handled by us; or a copy (`copyTextureToTexture`) per frame
- **Assessment (2026-11-11):** Platform can't: three r187 has no public API for a target's GPUTexture. Image impact none. Speed: Saves a full-screen copy per frame. Effort to match: n/a: a public three API (upstream). Recommendation: Wait for platform: a public three API.
- **Disputed:** Called forced, but the backend read is a choice over a per-frame copy, which three does allow; what is forced is only the missing public API.

### D14

*Memory & bindings.* When three makes its own WebGPU device it asks for the minimum limits, too small for the 1 GiB cache. Hosts pass our device, which asks for the adapter's maximum. The image is unchanged.

- **Capsaicin:** A D3D12 device with the hardware's limits: about 1 GiB of hash cache in about 30 buffers
- **Ours:** When three r187 makes its own device (`new WebGPURenderer()` with no `device`), it has WebGPU's default limits: a 128 MiB storage binding, a 256 MiB buffer and 8 storage buffers per stage, on the RTX 3060 and the Intel UHD (Chrome 157, RC1.4). Our device (`requestGIDevice`) gets 2 GiB bindings and 16 storage buffers per stage on both. The same holds for `maxColorAttachmentBytesPerSample`: three's own device keeps 32 bytes, while the screen inputs with the composite need up to 56 (RC1.1)
- **Kind:** P
- **Why:** three requests every feature it knows that the adapter has (all five RC1.4 features work on its device), but no limits: `requiredLimits` defaults to `{}` (`three/src/renderers/webgpu/WebGPUBackend.js:97, 245-248`)
- **Cause:** Forced (three's default device)
- **Evidence:** Measured (RTX 3060, Intel UHD)
- **What is really behind it:** Gone when the host passes our device or the adapter's limits
- **Challenge:** None
- **Way back / others' workarounds:** Declared fallback: none at the default limits. The technique throws `GILimitError` naming `maxStorageBufferBindingSize` ("World-cache memory"). The host passes our device (`createGIRenderer`, or `requestGIDevice` then `new WebGPURenderer({ device })`) or passes the adapter's limits as `new WebGPURenderer({ requiredLimits })` (three's own parameter, `WebGPUBackend.js:71`). Upstream: a three change that asks for the adapter's storage limits when it makes its own device. A smaller hash-cache capacity would fit 128 MiB, but isn't Capsaicin's default
- **Assessment (2026-11-11):** Platform can't: three's default device. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep: hosts pass our device or requiredLimits; a three change upstream would remove it.

### D15

*Precision & GPU quirks.* Half-float overflow behaves differently on our two Windows GPUs, which WGSL allows. The baseline does f32 math like Capsaicin, so nothing changes; it's a rule for future f16 speed work.

- **Capsaicin:** f32 math, half precision only in storage (`f32tof16`)
- **Ours:** f16 math must stay below 65,504; don't rely on overflow giving infinity (the Intel UHD keeps a wider intermediate). Measured: `300h * 300h / 1000h` is infinity on the RTX 3060 and 90 on the Intel UHD (Chrome 157, RC1.4)
- **Kind:** P
- **Why:** WGSL makes a run-time f16 overflow an indeterminate value (§15.7.3), so both answers are allowed; and radiance easily passes 65,504
- **Cause:** Forced (WGSL §15.7.3)
- **Evidence:** Measured (both GPUs disagree, as the spec allows)
- **What is really behind it:** The baseline does f32 math like Capsaicin
- **Challenge:** None
- **Way back / others' workarounds:** Declared fallback: f32 math with `pack2x16float` storage (D11), which is also Capsaicin's way. An f16 kernel clamps or pre-scales its inputs, and is checked on both GPUs. Not a gap for the faithful baseline, which doesn't do f16 math
- **Assessment (2026-11-11):** GPU quirk: WGSL allows f16 overflow to differ (§15.7.3). Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep: a rule for future f16 speed work.

### D16

*Screen probes.* Capsaicin sends a few percent of probe rays toward shiny reflections even on dull materials. With diffuse-only materials we send them all diffuse. It returns with D8.

- **Capsaicin:** `SampleScreenProbes` picks a ray's lobe by the surface's BRDF: with probability `calculateBRDFProbability(F0, ...)` it samples GGX, otherwise the diffuse cell (`gi1.comp:952-978`; `gi1_disable_specular_materials` is false by default)
- **Ours:** Every probe ray takes the diffuse branch, as Capsaicin built with `DISABLE_SPECULAR_MATERIALS`. The random draws keep Capsaicin's order (the GGX pair is drawn and dropped)
- **Kind:** P, Q
- **Why:** The platform's materials are diffuse-only (D8), and `FrameScreen` has no roughness or F0. Note (RC2.1 review): with Lambert-only materials Capsaicin still sends a few percent of probe rays down the GGX branch, since F0 = lerp(0.04, albedo, metallicity) (`materials.hlsl:186`) is never 0; this row drops that share. Capsaicin's default build also draws one more `rand()` (`gi1.comp:960`) than the `DISABLE_SPECULAR` build our order follows, which only matters under D18
- **Cause:** Rule (follows D8)
- **Evidence:** Cited
- **What is really behind it:** Capsaicin sends a few percent of rays down GGX even for Lambert materials; we drop that share
- **Challenge:** Medium. Comes back with D8
- **Way back / others' workarounds:** With D8's way back: roughness and metalness in the material descriptor, and a roughness channel in `FrameScreen` (RC1.1's channel requests), then the branch as Capsaicin has it
- **Assessment (2026-11-11):** Our rule: follows D8 (diffuse-only materials). Image impact medium. Speed: none. Effort to match: M: comes back with D8, plus a roughness channel in FrameScreen. Recommendation: Dennis decides: comes with D8.

### D17

*Screen inputs.* Capsaicin starts probe rays from the exact triangle point with the true surface normal. We rebuild the point from depth and use the normal three draws, which normal maps can bend. Checks on three scenes found no harm; normal-mapped surfaces at grazing angles are untested.

- **Capsaicin:** The probes read Capsaicin's own visibility buffer: the ray origin is the exact point on the triangle moved off it by `offsetPosition` (`gi1.comp:1135-1150`), the probe frame is the geometry normal, and a sky pixel is one with a zero geometry normal (`gi1.comp:570-571`). `InterpolateScreenProbes` uses the geometry normal for its plane tests and weights, the pixel's and each probe's (`gi1.comp:1518, 1598`), and the shading normal only for the SH (`gi1.comp:1620`)
- **Ours:** The probes read the host's `FrameScreen`: the position comes from `depth` through the frame's inverse view-projection (camera-relative, f64 on the CPU), the origin is pushed along the normal by the platform's rounding offset plus what four steps of the stored depth move the point; the frame is `FrameScreen.normal` (three's MRT gives the shading normal, not the geometric one); a sky pixel is one at the far plane's depth (the SH debug view's too, and since RC4.1 `ReprojectScreenProbes`' and `PatchScreenProbes`' sky tests). The interpolation uses that one normal everywhere: plane tests, weights and the SH (RC2.2). Since RC4.2 `FilterScreenProbes` takes it for its plane and hemisphere tests too (Capsaicin's geometry normal, gi1.comp:1380, 1409-1412), and SSGI reads it as Capsaicin's `ShadingNormal` (ssgi.comp:141, 212), which is what three draws; SSGI's sky test is the far plane's depth too (Capsaicin: a zero shading normal, ssgi.comp:142)
- **Kind:** P, Q
- **Why:** No visibility buffer on WebGPU (D3); `FrameScreen` carries depth and normals, and its normal is whatever the host draws
- **Cause:** Rule (`FrameScreen` has no visibility buffer or geometric normal)
- **Evidence:** Measured on the M5 (RC2.2, `radiance-cache-three`): at probes whose surface the camera sees at a grazing angle (|n · v| < 0.25), no more rays end within 1 cm than elsewhere, and the probes' E against a path-traced reference started 2 cm further out reads two-room 0.962, Sponza 0.989, Cornell 0.950 (non-grazing 0.996, 0.988, 1.057). Re-review: self-shadowing **not seen on Sponza or two-room; Cornell inconclusive** (149 probes; its grazing probes have 7× the short-hit share, 0.72% against 0.10%, and lifting the reference changed nothing, so it isn't contact occlusion either). The unlifted ratio can't detect self-shadowing at all, since the rays and the reference share the origin. After the RC3.2 merge (hits lit through the world cache, 16 warm-up frames; rc2-2 merge, M5) grazing probes read two-room 1.034, Sponza 1.182, Cornell 0.967 over the lifted reference (non-grazing 1.010, 1.057, 1.020): still no low reading at grazing angles. Not measured: a scene with strong normal maps at grazing angles. **Origins outside a closed room at its edges** (RC6.1, M5, a throwaway page over `radiance-cache-world-cache`'s scenario): in an 8 × 4 × 10 m box of one mesh, the camera 7 m from the back wall with the edges in view, one probe in 3 of 24 still frames started outside: 0.02 mm above the ceiling, 0.1 mm behind the back wall, on an edge pixel. The depth's rounding puts the rebuilt point a hair past the neighbouring face, and the push along the pixel's own normal runs parallel to that face. Its 64 rays hit the box's back faces from outside or missed (the sky, straight into the probe; that path existed before RC6.1). Since RC6.1 the back-face hits also see the sky, and the cells they land in are shared with the inside near the edge (the tile key's direction bins are coarse, hash_grid_cache.hlsl:146): 4 to 8 of about 61,000 cells were lit, within about 10 cm of a corner, the brightest (a few samples) as bright as a cell under the open sky. Capsaicin's origin is a point on the triangle, so it stays inside
- **What is really behind it:** Origin rebuilt from depth; shading normal instead of geometric, now also for the interpolation's plane tests and weights
- **Challenge:** Low on the scenes checked; medium in principle where shading and geometric normals differ a lot; a little light into closed rooms at their edges (seen in a test box). Comes back with D3's visibility channel
- **Way back / others' workarounds:** A visibility channel in `FrameScreen` (instance, triangle, barycentrics; three r187's VisibilityBuffer pipeline could fill it) or a geometric-normal channel. RC2.2's grazing check (M5): no self-shadowing seen (below)
- **Assessment (2026-11-11):** Our rule: FrameScreen has no visibility buffer or geometric normal. Image impact low. Speed: none. Effort to match: L: a visibility or geometric-normal channel in FrameScreen (D3). Recommendation: Measure first: a scene with strong normal maps at grazing angles.

### D18

*Infrastructure.* Capsaicin seeds its random numbers from a table of random values. We hash the seed instead. The table is random anyway, so the numbers can't match either way.

- **Capsaicin:** `MakeRandom(seed, frame)` starts its PCG from `g_RandomSeedBuffer[seed % size]`, a table of `mt19937` numbers the CPU makes at start-up (`random_number_generator.hlsl:108-116`; from `std::random_device` unless deterministic)
- **Ours:** The same PCG and the same increments, started from `pcgHash(seed)` (`math/hash.hlsl:33-39`) instead of a table entry
- **Kind:** P
- **Why:** No table to make and bind; the table is random data, so its numbers can't be matched anyway (by default they differ run to run in Capsaicin too)
- **Cause:** Shortcut
- **Evidence:** Argued, not measured: the table is random data anyway
- **What is really behind it:** Uploading a seed table would be easy
- **Challenge:** Low
- **Way back / others' workarounds:** Upload a seed table if a statistic ever differs; none expected
- **Assessment (2026-11-11):** Quick setup. Image impact low. Speed: none. Effort to match: S: upload a seed table. Recommendation: Keep.
- **Disputed:** Called a shortcut, but the evidence says the seed table is random data that can't be matched anyway, so the result is the same in distribution: closer to "Same result".

### D19

*Precision & GPU quirks.* WGSL can't reliably store infinity in a half float, so a miss stores the largest half value. It still reads as far away everywhere.

- **Capsaicin:** A missed ray's hit distance is `FLT_MAX` packed with `f32tof16` (`gi1.comp:1094, 1156`); a cell no ray reached gets `MAX_HIT_DISTANCE` (1e9) in the probe atlas (`gi1.comp:1232`)
- **Ours:** Both store 65,504, the largest finite half
- **Kind:** P
- **Why:** WGSL's `pack2x16float` and stores to `rgba16float` give an indeterminate value outside the half range (§16.9, D15's rule)
- **Cause:** Forced (WGSL half range)
- **Evidence:** Cited
- **What is really behind it:** 65,504 reads as "far" everywhere
- **Challenge:** None
- **Way back / others' workarounds:** None needed: 65,504 m reads as "far" wherever a hit distance is read (phase 4's reprojection and filter)
- **Assessment (2026-11-11):** Platform can't: WGSL's half range. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D20

*Materials & lights.* Missed rays read the platform's baked sky, shared with DDGI, instead of Capsaicin's cube map. The bake might be too coarse for detailed skies; nobody has checked. (Image impact: low to medium.)

- **Capsaicin:** A missed probe ray samples the environment cube map (`g_EnvironmentBuffer`, `gi1.comp:1082`)
- **Ours:** `giSkyRadiance`: the platform's sky, baked to a 256 × 128 radiance map (shared hit shading, as DDGI)
- **Kind:** P, Q
- **Why:** The platform rule: hit shading, the sky included, is shared between techniques
- **Cause:** Rule (shared sky bake, 256 × 128)
- **Evidence:** Assumed that the bake is detailed enough
- **What is really behind it:** WebGPU can sample a cube map
- **Challenge:** Low to medium. A bigger bake or the host's environment map
- **Way back / others' workarounds:** A larger sky bake if fine sky detail shows in the probes; the bake's size is a platform setting
- **Assessment (2026-11-11):** Our rule: shared sky bake. Image impact low. Speed: none (the bake size is a setting). Effort to match: S: raise the bake size, or read the host's environment map. Recommendation: Measure first: a bigger bake on an outdoor scene; compare the probes' E.

### D21

*Materials & lights.* Capsaicin treats glowing surfaces as lights it can aim shadow rays at. The platform only knows point, spot and directional lights, so glowing surfaces light the scene only when a ray happens to hit them. The Cornell box can't match Capsaicin without this.

- **Capsaicin:** Emissive triangles are lights: `LightBuilder` gathers them every frame (`gather_area_lights.comp`) into the light list the sampler draws from, so a hit beside an emissive surface is lit by it through one light sample and a shadow ray
- **Ours:** The platform's lights are the host's punctual lights (directional, point, spot) and the sky; an emissive surface reaches a probe only when a probe ray hits it (its emission, `gi1.comp:2439-2452`; the material's constant `surface.emission`, not Capsaicin's `emissiveAlphaScaled` through `evaluateAreaLight`, `gi1.comp:2441-2445`, see D2). Seen in RC2.1 on `cornell-box`, lit by its emissive ceiling alone: the walls' probe cells hold no direct light, only the cells whose ray hit the ceiling are bright
- **Kind:** P, Q
- **Why:** Hit shading is shared and has no area lights; DDGI gets emissive light only through its field's bounces too
- **Cause:** Rule (platform lights are punctual only)
- **Evidence:** Measured: seen on `cornell-box` in RC2.1
- **What is really behind it:** Hit shading has no area lights; WebGPU could gather emissive triangles as Capsaicin does. A hit's own emission is the material's constant (D2)
- **Challenge:** **High**: the Cornell box can't match without it. Emissive triangles as platform lights
- **Way back / others' workarounds:** Emissive triangles as platform lights: gathered into the light table (or the light grid) with their area and radiance, sampled in `giDirectIrradiance` and `gi/direct`. A platform task, needed before RC2.2's Cornell image can match Capsaicin's
- **Assessment (2026-11-11):** Our rule: platform lights are punctual only. Image impact high. Speed: Matching adds light sampling over emissive triangles (unknown). Effort to match: L: emissive triangles as platform lights, sampled in giDirectIrradiance and gi/direct (shared with DDGI). Recommendation: Dennis decides: a platform task, also carried for DDGI (area lights, M2).

### D22

*World cache.* WGSL only offers the weak atomic exchange, which can fail for no reason. We retry, which gives the same result.

- **Capsaicin:** `HashGridCache_InsertCell` claims a bucket slot with `InterlockedCompareExchange`, which is strong (`hash_grid_cache.hlsl:259`)
- **Ours:** `atomicCompareExchangeWeak`, retried on the same slot while the exchange fails and the slot still reads 0 (`HashGridCache_CompareExchangeEmptySlot`, `world-cache/hash-grid-wgsl.ts`)
- **Kind:** P
- **Why:** WGSL has only the weak exchange, which may fail spuriously; without the retry a spurious failure would move the tile to the next slot and could put one tile in two slots
- **Cause:** Forced (WGSL has only the weak exchange)
- **Evidence:** Cited
- **What is really behind it:** Retry gives the same result
- **Challenge:** None
- **Way back / others' workarounds:** None needed: same result (RC4.1's claim of a probe-cache slot, `rcClaimCachedTile`, retries the same way). RC3.1's page checks 65,536 threads inserting one tile get one slot, but that can't tell the retry from no retry: D3D12's weak exchange is likely strong underneath, so a spurious failure may never happen here
- **Assessment (2026-11-11):** Platform can't: WGSL has only the weak exchange. Image impact none. Speed: negligible. Effort to match: n/a. Recommendation: Keep.

### D23

*World cache.* We always count refused cache inserts, where Capsaicin counts them only in debug builds. It costs nothing on the normal path.

- **Capsaicin:** A refused insert (bucket full) is counted per bucket only with `DEBUG_HASH_STATS` (`hash_grid_cache.hlsl:268-275`), cleared by a separate kernel when debug stats are on
- **Ours:** One counter, always on, in the cache's counter block; `ClearCounters` zeroes it each frame (`HashGridCache.bucketOverflowCount`)
- **Kind:** none (a counter)
- **Why:** RC3.2 measures bucket overflow on Sponza and Bistro by counting (its done-when); one `atomicAdd` on the refusal path costs nothing on the normal path
- **Cause:** Same
- **Evidence:** Cited
- **What is really behind it:** A debug counter that is always on
- **Challenge:** None
- **Way back / others' workarounds:** Capsaicin's per-bucket histogram (`DebugStatsBucketOverflowCountBuffer`) if a per-bucket view is wanted
- **Assessment (2026-11-11):** Same result: a counter. Image impact none. Speed: none (one atomic on the refusal path). Effort to match: n/a. Recommendation: Keep.

### D24

*World cache.* Only reflections read the cache this way in Capsaicin, and we don't have reflections yet. The lookup can get the extra fields when they arrive.

- **Capsaicin:** Reflections read the cache with the hit's real ray length (the key's short-ray flag) and a position jittered by a random within one cell (`gi1.comp:2866-2899`)
- **Ours:** `worldCacheLookup(position, normal, direction)` takes no ray length, so it looks up as a ray longer than a tile (flag 0), and no jitter (it has no random numbers)
- **Kind:** Q
- **Why:** The slot's lookup signature (design doc, "The world-cache interface") carries no length or random state
- **Cause:** Rule (the world-cache lookup signature), partly a shortcut
- **Evidence:** Cited
- **What is really behind it:** The interface left out ray length and jitter
- **Challenge:** Low until GI-1.1's reflections read the cache. Add the fields then
- **Way back / others' workarounds:** Add the ray length (and a random pair, or let the caller jitter `position`) to the lookup when the first reader (GI-1.1's reflections) is ported
- **Assessment (2026-11-11):** Our rule: the world-cache lookup signature; partly a quick setup. Image impact low. Speed: none. Effort to match: S: add the fields when GI-1.1's reflections are ported. Recommendation: Keep: add them with GI-1.1.

### D25

*Screen probes.* Capsaicin's probe projection weighs each cell by a constant that makes the GI about 30% too bright, a units slip as far as we can tell. We use the correct weight. One constant brings Capsaicin's back. (High if reverted.)

- **Capsaicin:** `ProjectScreenProbes` weighs each cell by `1 / probe_size` (`gi1.comp:1466`), 4/π times the cell's solid angle, with the SH basis at the cell centre (`gi1.comp:1467-1474`). Nothing downstream divides the 4/π out: `FilterScreenProbes` (`gi1.comp:1442`), the GI denoiser (`ReprojectGI`, `FilterGI`, `gi1.comp:3997-4154`) and the probe interpolation all normalise by their own weights, `ResolveGI1` multiplies by `evaluateLambert` = albedo/π (`gi1.frag:127, 140`, `material_evaluation.hlsl:157-160`) and `combine.comp` adds the result. So a constant sky of radiance L reads E = 1.30 πL
- **Ours:** The cell's solid angle, 2π/64, with the same cell-centre basis: E = 1.021 πL under a constant sky (the 2.1% is the centre point leaning towards the normal, Capsaicin's own discretisation)
- **Kind:** Q
- **Why:** A 30% brighter GI term than the light that arrives would show in every comparison with a path tracer; the weight is a units slip, not a choice we can find a reason for
- **Cause:** Fix
- **Evidence:** Cited and measured on the CPU: Capsaicin's chain has no factor that undoes `1 / probe_size` (`gi1.comp:1442, 1466, 3997-4154`, `gi1.frag:127, 140`, `combine.comp`); its projection reads a constant sky as 1.30 πL and ours as 1.021 πL (`screen-irradiance.test.ts`); ours measured 1.020 πL on the M5 (`radiance-cache-image`)
- **What is really behind it:** A units slip in Capsaicin's projection: 4/π too bright
- **Challenge:** **High** on the GI term (+27%) if reverted. One constant; worth telling AMD
- **Way back / others' workarounds:** Capsaicin's temporal radiance feedback (`gi1.comp:2458`, off by default) feeds the previous image back into hits, so there the 4/π excess would build up over frames. One constant: `SH_CELL_SOLID_ANGLE` back to `1 / 8`. Ask AMD whether it is meant
- **Assessment (2026-11-11):** Bug fix. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep: tell AMD.

### D26

*Memory & bindings.* Only the SH debug view reads these in a fragment shader, and our rule puts such data in textures. Same values, different container.

- **Capsaicin:** The SH live in a structured buffer of `uint2`, `g_ScreenProbes_ProbeSHBuffer[9 × index + i]` (`gi1.comp:1510`)
- **Ours:** The same packed values in an `rg32uint` texture, each probe tile's nine in a 3 × 3 block
- **Kind:** none (same values)
- **Why:** The shading module's SH debug view reads them in a fragment shader, and our rule is that fragment-readable data lives in textures
- **Cause:** Rule (fragment-readable data lives in textures; DECISIONS, three integration)
- **Evidence:** Cited: core WebGPU allows a read-only storage buffer in a fragment shader; the rule comes from three r187 (no supported way to bind an outside buffer) and compatibility mode (may allow no fragment storage buffers; NOTES). Note `rg32uint` is itself refused as a storage texture in compatibility mode (NOTES), so the texture doesn't buy compatibility as built
- **What is really behind it:** Same data, another container
- **Challenge:** None. Only the SH debug view reads it in a fragment shader; a storage buffer would do if the rule changes
- **Way back / others' workarounds:** A read-only storage buffer if the rule goes; core WebGPU allows one in a fragment shader
- **Assessment (2026-11-11):** Our rule: fragment-readable data lives in textures. Image impact none. Speed: none. Effort to match: S: a read-only storage buffer if the rule goes. Recommendation: Keep.
- **Disputed:** The rule's reason (compatibility mode may allow no fragment storage buffers) doesn't buy anything here: the row's own evidence says `rg32uint` is refused as a storage texture in compatibility mode too.

### D27

*Screen probes.* Capsaicin jitters with blue noise, which leaves noise the denoiser removes more easily. The platform has no blue noise yet, so we use white noise. It may be part of the remaining flicker.

- **Capsaicin:** The interpolation's jitter comes from `MakeBlueNoiseSampler(did, g_FrameIndex, 1)` (`gi1.comp:1540-1543`), Capsaicin's blue-noise tables
- **Ours:** The probes' PCG (`rcMakeRandom`, D18's), white noise
- **Kind:** Q
- **Why:** The platform has no blue-noise source yet: D7's regenerated tables were never built, and no technique uses one (DDGI doesn't)
- **Cause:** Shortcut
- **Evidence:** Cited: `gi1.comp:1540-1543`; the platform has no blue-noise source to use (searched, RC2.2; D7's tables were never regenerated)
- **What is really behind it:** White-noise jitter where Capsaicin's is blue
- **Challenge:** Low until phase 4's denoiser; blue noise leaves less structured noise for it. Build D7's tables as a platform texture
- **Way back / others' workarounds:** Regenerate blue-noise tables (D7) as a platform texture and use them here; the phase 4 denoiser hides most of the difference
- **Assessment (2026-11-11):** Quick setup: the platform has no blue noise yet. Image impact low. Speed: none. Effort to match: M: blue-noise tables as a platform texture (D7), shared with D55. Recommendation: Measure first: the flicker after the denoiser with blue noise against white.

### D28

*Precision & GPU quirks.* WGSL gives no defined answer for packing a too-large value, so we clamp first. A coefficient that large is wrong either way.

- **Capsaicin:** SH coefficients are packed with `f32tof16` (`packHalf4`, `screen_probes.hlsl:169-172`), which turns a value past the half range into infinity
- **Ours:** Clamped to ±65,504 before `pack2x16float`
- **Kind:** P
- **Why:** WGSL's `pack2x16float` gives an indeterminate value outside the half range (WGSL §17.9.9; §16.9 in older drafts, as D19 cites)
- **Cause:** Forced (WGSL `pack2x16float` range)
- **Evidence:** Cited: WGSL §17.9.9
- **What is really behind it:** A clamp where Capsaicin gets infinity
- **Challenge:** None
- **Way back / others' workarounds:** None needed: a coefficient that large is wrong either way
- **Assessment (2026-11-11):** Platform can't: WGSL's pack2x16float range. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D29

*Screen probes.* three uses normal depth, not reversed-Z, so we compute the same distance another way.

- **Capsaicin:** `InterpolateScreenProbes` linearises depth with `toLinearDepth(depth, g_NearFar)` (`gi1.comp:1597`, `transform.hlsl:144-148`), which assumes reversed-Z; so do `FilterScreenProbes`' depth weight (gi1.comp:1434) and SSGI's slice radius (ssgi.comp:150, 159)
- **Ours:** `dot(p − eye, forward)` of the point rebuilt from depth, in all three (the filter and SSGI since RC4.2)
- **Kind:** none (same value)
- **Why:** `FrameScreen`'s depth convention is the host's (three draws standard 0-to-1 depth, not reversed-Z) so `toLinearDepth`'s reversed-Z formula would be wrong for it (`FrameCamera` does carry the planes, `contract/frame.ts:57-58`); the view-axis distance is the same number for any perspective projection
- **Cause:** Same
- **Evidence:** Measured on the CPU: equal to 9 digits on three points (`screen-irradiance.test.ts`)
- **What is really behind it:** The view-axis distance from the rebuilt point instead of from reversed-Z depth and the planes
- **Challenge:** None
- **Way back / others' workarounds:** Port `toLinearDepth` with the planes taken from the projection, for a reversed-Z host
- **Assessment (2026-11-11):** Same result. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D30

*World cache.* Capsaicin adds rays to the cache inside the trace. Our world-cache slot takes finished probe rays, so the insert is its own pass with bigger records. Same result; it is a speed lever for later.

- **Capsaicin:** The cache half of `PopulateScreenProbes` runs inside the probe trace (`PopulateScreenProbesHandleHit`, `gi1.comp:985-1076`, over every possible ray); a visibility entry copies the hit (`HashGridCache_PackVisibility`, four words: instance with the back-face bit, geometry, primitive, barycentrics as halves; `gi1.comp:1003-1015`, `hash_grid_cache.hlsl:379-384`), and `GenerateReservoirs` stores each shadow ray's origin, hit point and packed material for `PopulateCells` (`gi1.comp:2594-2598`)
- **Ours:** The world-cache slot gets the traced probe rays, so the insert is a kernel of its own after the screen cache's trace, dispatched on the probe-ray count. As in Capsaicin, it copies each hit into the visibility list (`g_HashGridCache_VisibilityBuffer`), with its probe ray's index and its cell; an entry is ten words: the hit (instance with the back-face bit, triangle, barycentrics as f32 as the probe ray has them; no geometry index) and the ray (origin, direction, length). `GenerateReservoirs` and `PopulateCells` rebuild the surface from the entry with `giSurfaceAt` and keep no per-shadow-ray surface or material; a reservoir is three words (light, weight, point on the light). The lists live in one packed buffer (D6)
- **Kind:** P
- **Why:** The world-cache interface (design doc): a cache other than the hash grid must be able to take the probe rays as they are. The ray is copied because the short-ray test measures from its origin (D37) and the shading passes can't also bind the probe rays within 8 storage buffers (D6). Rebuilding the surface needs no material packing (diffuse materials, D8)
- **Cause:** Rule (the world-cache interface)
- **Evidence:** Cited (the interface, D6's budget); the short-ray part is now D37 (assumed)
- **What is really behind it:** The insert is its own kernel because the cache gets traced rays; the visibility entry copies the hit as Capsaicin's does, plus the ray. The surface is rebuilt with `giSurfaceAt`, which also leans on D8
- **Challenge:** Low (a speed lever and ten words where Capsaicin has four). `radiance-cache-world-cache` passes on the M5 with it (the lit plane within 0.23% per ray)
- **Way back / others' workarounds:** A speed lever: fuse the insert into `PopulateScreenProbesMain` when the hash grid is the cache; same results. Capsaicin's four-word entry once D37 goes back to the depth point (the ray words go)
- **Assessment (2026-11-11):** Our rule: the world-cache interface. Image impact low. Speed: An extra dispatch and 10-word entries where Capsaicin has 4 (not timed). Effort to match: M: fuse the insert into the trace when the hash grid is the cache; 4-word entries once D37 goes. Recommendation: Fix with lever 5: compact formats (16-byte hit records).

### D31

*Precision & GPU quirks.* Two of our GPUs round half floats down, which would slowly darken the cache. A small correction restores Capsaicin's rounding and changes nothing on a GPU that already rounds.

- **Capsaicin:** `f32tof16` rounds to the nearest half, and gives infinity above 65,504
- **Ours:** `HashGridCache_PackRadiance` (also the multibounce brdf and pdf, RC3.3) and the probe rays' `ScreenProbes_AccumulateRadiance` pack with `pack2x16float`, then take the next half out when it is nearer (`HashGridCache_PackHalf2`), after clamping to 65,504: a value past the half range packs to 65,504, where Capsaicin's gives infinity
- **Kind:** P
- **Why:** WGSL leaves `pack2x16float`'s rounding to the implementation, and its result past the half range is indeterminate (D15). It truncates on the RTX 3060 and the Intel UHD (NOTES) and rounds to nearest-even on the M5 (Chrome 157, Metal). `UpdateTiles` re-packs every cell's value each frame as a running mean, where truncation would pull a cell down by up to 16 half steps (0.8%). The correction only ever moves a half that isn't the nearest, so on a rounding GPU it changes nothing: `radiance-cache-hash-grid` section 9 on the M5, 4,096 values, none moved
- **Cause:** Forced (WGSL leaves `pack2x16float`'s rounding to the implementation; its result past the half range is indeterminate)
- **Evidence:** Measured on three GPUs: two truncate (RTX 3060, Intel UHD), one rounds to nearest-even (M5); on the M5 the correction moves no half (`radiance-cache-hash-grid` section 9)
- **What is really behind it:** The correction restores Capsaicin's rounding on a truncating GPU and changes nothing on a rounding one. The clamp gives 65,504 where Capsaicin gives infinity
- **Challenge:** None for the image (a value past 65,504 is far outside indoor radiance)
- **Way back / others' workarounds:** None needed; drop the correction if WGSL ever defines rounding to nearest
- **Assessment (2026-11-11):** GPU quirk: WGSL leaves the rounding to the GPU. Image impact none. Speed: negligible. Effort to match: n/a. Recommendation: Keep.

### D32

*World cache.* Capsaicin's rule can put a new cache tile on the update list twice, and the two updates then race. We saw it on the Intel GPU and fixed it with one extra word per tile.

- **Capsaicin:** A new tile goes on `UpdateTiles`' list from the thread that claimed its slot (`is_new_tile`) and from any thread that touched it first this frame (`gi1.comp:1043-1048`, also `1819-1824`). A thread that finds the tile and stamps its decay before the claimer does is both, so the tile is listed twice; two `UpdateTiles` workgroups then update it at once and race on its scratch
- **Ours:** The same rule, plus a word per tile (the frame plus one) that lets exactly one of those threads append it (`HashGridCache_ListTileForUpdate`; 1 MiB)
- **Kind:** Q
- **Why:** Seen on the Intel UHD in RC3.2's lit-plane test: a cell read scratch another workgroup had half cleared (red and green 1/16 low) on 150 of 107,709 probe rays and 42 cells; the RTX 3060 hid it (its workgroup reads the four words before any clear)
- **Cause:** Fix
- **Evidence:** Measured: our port of Capsaicin's rule raced on the Intel UHD, hidden on the RTX 3060. Capsaicin itself was not run
- **What is really behind it:** Capsaicin's tile list can list a tile twice
- **Challenge:** None. Worth telling AMD once seen in Capsaicin itself
- **Way back / others' workarounds:** Capsaicin's rule alone (drop the word) if Capsaicin fixes the race another way
- **Assessment (2026-11-11):** Bug fix. Image impact none. Speed: 1 MiB of memory. Effort to match: n/a. Recommendation: Keep: tell AMD.

### D33

*Materials & lights.* In Capsaicin the sky counts as a light that cache cells sample, so skylight bounces show outdoors. **Closed by RC6.1 (2026-11-11):** the cache's hits now draw the sky as one more light, as Capsaicin's do, through a platform helper over the baked sky. What remains differs in noise only, not in the expected light.

- **Capsaicin:** The environment map is the first light of the light builder's list (`environment_light_enable`, default true, light_builder.h:63; light_builder.cpp:131-134, 200-207), so the light sampler streams it into the first reservoir of every grid cell (light_sampler_grid_stream_build.comp:120, reservoir `lightIndex % 64`), weighed by `luminance(4π × the map's mean radiance)` (`sampleLightVolume`, light_sampling_volume.hlsl:312-331). `GenerateReservoirs` and `GenerateMultibounceReservoirs` resample `kReservoir_SampleCount` = 8 candidates (`sampleLightsListCone<8>`, gi1.comp:2513, 2659; world_space_restir.hlsl:30) by the default fast merge (`light_grid_stream_merge_type` 1, light_sampler_grid_stream.h:65; light_sampler_grid_stream.hlsl:342-420): with 8 lights or fewer in the list each light is one candidate. The sky's candidate is one direction uniform over the shading normal's hemisphere, pdf 1 / 2π (the default `EnvironmentSamplingMode::Uniform`, light_builder.h:54-65; `sampleEnvironmentLight`, light_sampling.hlsl:164-180; `mapToHemisphere`, sampling.hlsl:230-248), weighed by `luminance(brdf × cos × L) / luminance(albedo)` with L read at the blurred level of solid angle 4π / 12 (`updateReservoirCone`, reservoir.hlsl:305-320; `Reservoir_EvaluateTargetPdfNormalized`, 131-135; `evaluateEnvironmentLightCone`, light_evaluation.hlsl:293-304; gi1.comp:2504). The reservoir keeps the two random numbers as halves (`packReservoir`, reservoir.hlsl:545-551). `PopulateCells` rebuilds the direction about the interpolated vertex normal (`evaluateLightConeSampledAM`, light_evaluation.hlsl:1039-1051; gi1.comp:2063, 2071-2073), reads L at the sharpest level its solid angle 4π / 96,000 gives, traces an unbounded shadow ray from the point offset along the geometric normal (`MakeRayInfoShadow` with no light position, then `MakeRayInfoShadowDir`, trace_ray.hlsl:95-101, 147-170), and on a miss adds `brdf × L × W` (`PopulateCellsHandleMiss`, gi1.comp:2006-2020)
- **Ours (since RC6.1):** `GenerateReservoirs` and `GenerateMultibounceReservoirs` draw through `giDirectChooseLightWithSky` (platform hit shading, `platform/shade/wgsl.ts`): the sky is one candidate ahead of the lights (Capsaicin's first light), its direction from `giSkyLightSample` (uniform over the shading normal's hemisphere, pdf 1 / 2π, through the platform's tangent basis: the same distribution), weighed by the luminance of that direction's irradiance estimate `L cos / pdf` from the platform's baked sky (`giSkyRadiance`, the 256 × 128 map, D20); drawn with probability weight / total, its weight total / its weight, as D4's lights. The reservoir keeps the two numbers as unorm16, quantised before the sky is weighed. `PopulateCells` calls `giDirectLightRay` for `GI_DIRECT_SKY_LIGHT`, which rebuilds the same direction and traces an unbounded shadow ray from the platform's offset origin. The setting `environmentLightEnable` is the light builder's option (default true). DDGI's `gi/direct` doesn't change: there the sky comes through probe rays that miss
- **Kind:** Q (noise only, since RC6.1)
- **Why (what remains):** (a) **The candidate set with more than 8 lights.** Capsaicin's sky then shares the first of 8 segments with other reservoirs and is a candidate only when drawn there, by its build weight; ours is always a candidate. The platform's light grid (D4) has no slot for the sky, and one light system is the platform's rule. With 8 lights or fewer the sets are the same: Sponza has the sun, the arcade lamp and the sky (3), Bistro exterior the sun and the sky (2). Both are unbiased, so only the noise differs. (b) **The weight.** Capsaicin's reads the sky at a blurred level and multiplies by the albedo's colour (normalised); ours reads level 0 of the bake, which has no mip chain, and weighs irradiance alone, as D4 does for every light. Again only the noise differs: the light added is the sharp L in both. (c) **The rebuilt direction.** Capsaicin rebuilds it from halves about an untransformed, unnormalised vertex normal, a hair off the one it weighed (the D36 and D41 normals); ours rebuilds exactly the weighed one. (d) The cosine at the end is on the shading normal (D36), and a direction below the geometric side weighs 0 (the platform's side test; Capsaicin traces it, and it hits the surface it leaves or slips past)
- **Cause:** (a) Our rule (D4: one light system, the platform light grid); (b) Quick setup (the bake has no mips) and our rule (D4's weight); (c) Bug fix (D36, D41); (d) Our rule (D36)
- **Evidence:** Cited (the lines above). Measured on the Apple M5 Pro, `radiance-cache-world-cache` sections 6 and 7: under a constant sky of (1, 0.8, 0.6) and no lights, the open floor's cells read 1.0049 of albedo × L × V (V ≥ 0.99, mean 0.998: E = πL within 0.5%), the whole floor 1.0018, the wall 1.0076 (V from the exact view factor, `sky-view.ts`), every channel; every hit drew the sky. BOXRESULT
- **What is really behind it:** WebGPU samples the sky as Capsaicin does; the remaining differences are the platform's one light system (D4) and a bake without mips
- **Challenge:** **Low** (was high): the expected light at every hit is Capsaicin's; the noise differs in scenes with more than 8 lights and wherever the sky's brightness varies a lot within the blur Capsaicin's weight uses. Sponza's GI E rises SPONZARESULT, Bistro exterior's BISTRORESULT (`radiance-cache-sky-light`)
- **Way back / others' workarounds:** (a) comes with D4 (a stream light sampler on the platform, the sky in it); (b) a mip chain on the sky bake (S) and the albedo in the weight (with D4); (c) nothing to undo (a fix)
- **Assessment (2026-11-11, after RC6.1):** Mostly our rule (D4) and a quick setup (no mips in the bake). Image impact low (noise only). Speed: one sky texture read per hit in the light choice; one more shadow ray only where the sky is drawn instead of a light (the timing page measures it). Effort to match: S for the mips; the rest is D4's L. Recommendation: Keep; revisit with D4.

### D34

*Precision & GPU quirks.* A rounding effect can push a ray direction a hair past the normal, and Capsaicin's formula then gives NaN. We clamp, so the ray lands next to where it was aimed. About 1 ray in 15,000 to 32,000.

- **Capsaicin:** `mapToHemiOctahedronInverse` takes `sqrt(1 - abs(z))` and expects a unit direction (`sampling.hlsl:90-96`), but `BlendScreenProbes` passes it the probe ray's direction read back from halves (`ScreenProbes_UnpackSample`, `gi1.comp:1197-1201`). Where `f32tof16` rounds to nearest, that direction can be up to about 3e-4 longer than 1, so a ray near the normal has z past 1: the root is NaN and `uint2(NaN)` puts the ray in a cell the GPU chooses (cell 0, a grazing corner, on the M5)
- **Ours:** `rcMapToHemiOctahedronInverse` takes `sqrt(max(1 - abs(z), 0))`, so such a ray maps to the pole (the square's centre, cell 36), next to the cell it was sampled in (27, 28, 35 or 36). The CPU twin does the same. A unit direction maps as before
- **Kind:** Q (small)
- **Why:** WGSL gives no defined result: `sqrt`'s domain is [0, +∞] (§17.5.58), a NaN converted to `u32` is an indeterminate value (§15.7.6), and an implementation may assume no NaN at run time (§15.7.2), so a NaN test after the root could be optimised away. Seen on the Apple M5 (Chrome 157), whose `pack2x16float` rounds to nearest: 4 of 129,600 rays on `cornell-box` and 15 of 230,400 on Sponza had z past 1, and the GPU put them all in cell 0 (`radiance-cache-probes` readback, 2026-11-10). The RTX 3060 and Intel UHD truncate when packing (NOTES), so the halves are no longer than 1 and none was seen there
- **Cause:** Forced (WGSL gives Capsaicin's line no defined result); sending the ray to the pole is our choice, a fix
- **Evidence:** Cited: WGSL §17.5.58 (`sqrt`'s domain), §15.7.6 (NaN to `u32`), §15.7.2 (no NaN assumed). Measured on the Apple M5: 19 rays in two scenes went to cell 0 before, cell 36 after, on the GPU and the CPU alike. Assumed, not checked: Capsaicin on D3D12 puts them in cell 0 too (D3D converts a NaN to 0) wherever `f32tof16` rounds to nearest
- **What is really behind it:** Capsaicin's map expects a unit direction and gets one read back from halves. A faithful port could only copy "some cell"; the pole is the nearest point of the map
- **Challenge:** Low: 1 ray in about 15,000 (Sponza) to 32,000 (Cornell) a frame on the M5, which went to a grazing corner cell instead of beside the normal. Nothing to remove
- **Way back / others' workarounds:** None needed: the clamp only acts where Capsaicin's own result is NaN. Normalising the unpacked direction before the map would also work, but would move rays near every cell edge too, and a dot of unit vectors in f32 can still pass 1
- **Assessment (2026-11-11):** Platform can't: WGSL gives Capsaicin's line no defined result; the pole is our fix. Image impact low. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D35

*World cache.* The purge runs at a different point in the frame, but nothing in between touches the cache.

- **Capsaicin:** `PurgeRadianceCache` (`ClearCounters`, `PurgeTiles`) runs at the start of the frame, before the screen probes are spawned and traced (`gi1.cpp:2238-2253`)
- **Ours:** It runs first in the world cache's `encodeFrame`, after the screen cache's trace and before the insert
- **Kind:** none (order)
- **Why:** The world-cache interface: all of the cache's per-frame work is recorded at the seam, after the probe rays are traced. Same result: no pass between Capsaicin's purge and our place for it reads or writes the hash grid (the screen cache's passes bind none of its buffers, and the screen half of `ClearCounters` is the screen cache's own)
- **Cause:** Same (order only)
- **Evidence:** Cited: no pass between Capsaicin's purge and ours binds a hash-grid buffer
- **What is really behind it:** The world-cache interface records all cache work after the trace
- **Challenge:** None
- **Way back / others' workarounds:** None needed. A `beginFrame` step in the interface if a cache ever needs work before the screen probes
- **Assessment (2026-11-11):** Same result: order only. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D36

*World cache.* The platform's shared light code uses the smooth shading normal where Capsaicin uses the triangle's flat normal. They differ only on smooth curved surfaces.

- **Capsaicin:** `PopulateCells` offsets the shadow ray along the triangle's geometric normal (`MakeRayInfoShadow` with `offsetPosition`, `gi1.comp:2062, 2076`; `trace_ray.hlsl:131-137`) and takes the light's cosine on the geometric normal too (`evaluateBRDF` with `payload.normal`, `gi1.comp:2018`); the interpolated shading normal only picks environment-light directions (`evaluateLightConeSampledAM`, `gi1.comp:2063, 2073`). Its shading normal is interpolated from vertex normals it doesn't transform (gi1.comp:2054, 2063: only the positions and the geometric normal go through the instance's transform), so on a rotated instance the environment-light directions are picked about the wrong normal
- **Ours:** The platform's `giDirectLightRay`: the shadow ray leaves along the geometric normal turned to the ray's side, by `giShadeOffsetRayOrigin` (256 rounding steps of the point), and the cosine is on `giSurfaceAt`'s shading normal (with the geometric side check); `giSurfaceAt`'s shading normal is in world space (a fix of the missing transform, as D41)
- **Kind:** Q
- **Why:** Hit shading is shared: the platform has one light evaluation, which DDGI uses too, and it takes the cosine on the shading normal. The two agree wherever the shading normal is the geometric one (flat-shaded and planar surfaces)
- **Cause:** Rule (shared hit shading: one light evaluation)
- **Evidence:** Cited (`gi1.comp:2018, 2062-2063, 2073-2076`; `giDirectLightRay`)
- **What is really behind it:** Capsaicin takes the cosine on the geometric normal; the platform on the shading normal. Nothing in WebGPU stops either
- **Challenge:** Low: only smooth-shaded curved surfaces differ, by the angle between the normals. The untransformed shading normal (gi1.comp:2054, 2063) is a fix on our side, cited; its effect is assumed, not measured (no rotated instances run). Since RC6.1 Capsaicin's environment light sampling, which reads that normal, is ported (D33): ours rebuilds the sky's direction about `giSurfaceAt`'s world-space normal. A geometric-normal option in the platform's light ray
- **Way back / others' workarounds:** A geometric-normal option in `giDirectLightRay`, or the port computes the cosine itself from the light's direction
- **Assessment (2026-11-11):** Our rule: shared hit shading (one light evaluation). Image impact low. Speed: none. Effort to match: S: a geometric-normal option in giDirectLightRay. Recommendation: Keep.

### D37

*World cache.* A test of whether a ray is short measures from a point pushed slightly off the surface instead of the surface itself. Only rays right at the threshold can change; nobody has counted them.

- **Capsaicin:** `GenerateReservoirs`' short-ray test measures from the probe's pixel on the depth buffer, the point not pushed off the surface (`gi1.comp:2429-2436, 2500, 2558-2561`)
- **Ours:** It measures from the probe ray's origin (kept in the visibility entry, D30): that depth point pushed off the surface by D17's offset
- **Kind:** Q
- **Why:** The world-cache interface hands the cache the probe rays, not the screen's depth; the probe ray's origin is the depth point plus the push. A ray whose length is within the push of its cell's size can take the other path; nobody has measured how many
- **Cause:** Rule (the world-cache interface; follows D17)
- **Evidence:** Assumed: small and unmeasured
- **What is really behind it:** The cache gets the probe ray's pushed origin, not the screen's depth point
- **Challenge:** Low: only rays within the push of their cell's size can change path. Count them against the unpushed point (both are known in the screen cache)
- **Way back / others' workarounds:** Hand the cache the unpushed depth point (or the push), then measure from it; that also brings back Capsaicin's four-word visibility entry
- **Assessment (2026-11-11):** Our rule: the world-cache interface; follows D17. Image impact low. Speed: none. Effort to match: S: hand the cache the unpushed depth point. Recommendation: Measure first: count the rays that change path against the unpushed point.

### D38

*Screen probes.* WGSL doesn't promise a missing probe reads as zero, so we skip it instead of multiplying by zero.

- **Capsaicin:** The interpolation evaluates all four neighbour probes and multiplies a missing one by its weight 0 (`gi1.comp:1629-1634`); the missing probe's tile is far past the SH buffer, which D3D12 reads as 0
- **Ours:** A neighbour with weight 0 is skipped, not read
- **Kind:** none (same value)
- **Why:** WGSL leaves an out-of-bounds `textureLoad` to the implementation: a texel from inside the texture, or zero (WGSL §17.7.4). A texel from inside is finite (D28), so 0 × it is 0; but a NaN coefficient anywhere would spread into pixels that don't use that probe
- **Cause:** Same (a guard)
- **Evidence:** Cited: WGSL §17.7.4 (an out-of-bounds `textureLoad` gives a texel from inside or zero), D3D12 reads past a structured buffer as 0
- **What is really behind it:** Ported as Capsaicin it gives the same image unless a coefficient somewhere is NaN; the review called this a fix, but nothing goes wrong on our GPUs without it, so it is "same"
- **Challenge:** None
- **Way back / others' workarounds:** None needed
- **Assessment (2026-11-11):** Same result: a guard. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D39

*Multi-bounce.* The second-bounce ray starts from a point pushed off the surface with the platform's method instead of Capsaicin's. It was easier to reuse ours; porting Capsaicin's is about 8 lines.

- **Capsaicin:** `PopulateMultibounceCells` traces from the first-bounce hit pushed off by `offsetPosition` (Wächter and Binder's per-coordinate integer step, `geometry.hlsl:35-47`) along the interpolated normal (`MakeRayInfoClosest`, gi1.comp:1933), and rebuilds the probe's depth point for the view direction its GGX lobe reads (gi1.comp:1919-1930)
- **Ours:** The platform's push (`giShadeOffsetRayOrigin`, 256 rounding steps along the same normal); no view direction (Lambert, D8, doesn't read one)
- **Kind:** Q (small)
- **Why:** Easier: the platform's push is what every other ray in the port uses (D17, D36)
- **Cause:** **Shortcut**
- **Evidence:** Measured only for shadow rays: the platform's push replaced Wächter and Binder's after their tilt carried 8 corner points past the neighbouring face on the trace oracle's thin box (NOTES, `giShadeOffsetRayOrigin`); not measured for bounce rays
- **What is really behind it:** Reusing the platform's push was easier; WebGPU can do `offsetPosition`
- **Challenge:** Low. Port it and count the second-bounce rays that change hit
- **Way back / others' workarounds:** Port `offsetPosition` for this ray (8 lines); the view direction comes back with D8
- **Assessment (2026-11-11):** Quick setup. Image impact low. Speed: none. Effort to match: S: port offsetPosition for this ray (8 lines). Recommendation: Fix now: cheap and restores the faithful port; count the second-bounce rays that change hit.

### D40

*Multi-bounce.* Capsaicin's multi-bounce code uses the wrong kind of index in two places, adding light to unrelated rays and dropping some second bounces. We fixed it. (Medium near walls if reverted.)

- **Capsaicin:** Two passes patch a probe ray with a second-bounce entry's index. `PopulateCells` adds a bypassing shadow ray's light to the probe ray `VisibilityQueryBuffer[visibility_index]` (gi1.comp:2100-2106), which for a second-bounce entry is the first-bounce entry it came from (gi1.comp:1794), not a probe ray. `UpdateMultibounceCells` takes its bypass flag from `VisibilityRayBuffer[did2]` (a list indexed by shadow ray, not by visibility entry) and adds to the same wrong "probe ray" (gi1.comp:1977-1981). So light lands on unrelated probe rays, and a probe ray that bypasses the cache never gets its second bounce (`ResolveCells` skips it)
- **Ours:** `PopulateCells` patches first-bounce entries only. `GenerateReservoirs` writes each first-bounce entry's bypass flag to a list of its own (`visibility_bypass_buffer`, one word per probe ray, 921,600 B at 1280 × 720); `UpdateMultibounceCells` reads the first-bounce entry's flag and adds the sample to that entry's own probe ray. A second-bounce ray shorter than its cell has no effect
- **Kind:** Q
- **Why:** A fix. Index domains, cited; measured on the M5 (`radiance-cache-multibounce`, one frame of the corner): Capsaicin's indices would add light to 1,077 unrelated probe rays in each pass and drop the second bounce of 370 bypassing ones, of 107,892 hits
- **Cause:** Fix
- **Evidence:** Cited (gi1.comp:1794, 1977-1981, 2100-2106: the indices' domains) and measured on the Apple M5 (1,077 + 1,077 misrouted patches and 370 dropped second bounces a frame on the corner). Capsaicin itself was not run
- **What is really behind it:** Capsaicin's multibounce indexes two lists with the wrong kind of index
- **Challenge:** Medium near walls: the misrouted light is noise on unrelated probe rays, and short first-bounce rays lose their second bounce. None to remove; report to AMD
- **Way back / others' workarounds:** Capsaicin's indices (drop the list and the first-bounce test). Tell AMD
- **Assessment (2026-11-11):** Bug fix. Image impact none. Speed: 921,600 B list at 720p. Effort to match: n/a. Recommendation: Keep: tell AMD.

### D41

*Multi-bounce.* Capsaicin forgets to rotate normals for rotated objects in one place. We use the correct normal. No test scene has rotated objects yet.

- **Capsaicin:** `GenerateMultibounceReservoirs` interpolates the second-bounce hit's vertex normals without transforming them (gi1.comp:2617-2626; `PopulateMultibounceCells` does transform them, gi1.comp:1898-1900)
- **Ours:** `giSurfaceAt`'s world-space normal
- **Kind:** Q (small, assumed)
- **Why:** A fix: the untransformed normal is wrong on any rotated instance. Cited; its effect is not measured (no scene with rotated instances was run)
- **Cause:** Fix
- **Evidence:** Cited (gi1.comp:2617-2626 against 1898-1900); the impact is assumed, not measured: no scene with rotated instances was run
- **What is really behind it:** A missing `transformNormal`
- **Challenge:** Low, assumed: identity-transform scenes are unchanged (the only ones checked), and the normal only weighs the light choice. Measure on a scene with rotated instances
- **Way back / others' workarounds:** Leave the normal untransformed (only the light choice reads it, D4)
- **Assessment (2026-11-11):** Bug fix. Image impact low. Speed: none. Effort to match: n/a. Recommendation: Keep: measure on a scene with rotated instances when one exists.

### D42

*Multi-bounce.* Capsaicin frees 535 MB when multi-bounce is off. We made it a live setting and never freed the buffers. Easy to fix, and worth it on memory-tight devices.

- **Capsaicin:** `gi1_use_multibounce` is a compile define (`USE_MULTI_BOUNCE`, gi1.cpp:1428-1431); off, `ResolveCells` reads the shadow-ray list and the visibility lists are half the size (gi1.cpp:640); with it off, the indirect value array and the indirect update scratch are freed (gi1.cpp:497-552), about 535 MB at the default capacity
- **Ours:** A word in the constant block the passes read, so changing the option needs no recompile; off, the three passes don't run and `GenerateReservoirs` lists only the hits that drew a light for `ResolveCells` (the same entries); the lists stay sized for two bounces. Both indirect arrays stay allocated with the option off: 534,773,760 bytes (178,257,920 of values, 356,515,840 of scratch) Capsaicin doesn't hold then
- **Kind:** none for the image; memory
- **Why:** Same result; the option is a setting the viewer can flip. Keeping the memory is a shortcut: the buffers are made once at `prepare`, and nobody resized them for the option
- **Cause:** Same (the option as data); **Shortcut** (the memory)
- **Evidence:** Cited (gi1.cpp:497-552, 1428-1431)
- **What is really behind it:** The option as data instead of a define; with it off, 535 MB of indirect arrays Capsaicin frees stay allocated, because the buffers are sized once
- **Challenge:** None for the image. 535 MB with the option off: size the buffers by the option at `prepare`
- **Way back / others' workarounds:** Size the cell values and the cache words from `useMultibounce` at `prepare` (and prepare again when it changes) to free the 535 MB
- **Assessment (2026-11-11):** Quick setup: the memory (the setting itself is the same result). Image impact none. Speed: 535 MB held with multi-bounce off. Effort to match: S: size the buffers from useMultibounce at prepare. Recommendation: Fix now: cheap; matters on memory-tight devices.

### D45

*Screen probes.* Swapping textures every frame would make three re-bind them every frame, so we copy instead. Same data; a small copy cost.

- **Capsaicin:** Two of each history texture, swapped every frame: probe atlas, probe mask (with its mips) and SH (`probe_buffers_[2]`, `probe_mask_buffers_[2]`, `probe_sh_buffers_[2]`, gi1.h:202-204, flipped at gi1.cpp:3097); the atlas is also copied current → previous after the blend (gi1.cpp:2540-2541)
- **Ours:** One current texture of each, which the shading module and the debug views read, plus a "previous" texture filled by a copy at the end of the frame: the atlas by Capsaicin's own copy after the blend (which `FilterScreenProbes`' second pass reads, RC4.2) and again at the end of the frame, after the filter, so the next frame's history is the filtered atlas as Capsaicin's flip makes it (the second pass writes `probe_buffers_[probe_buffer_index_]`, gi1.cpp:2606-2607, which the flip at 3097 makes the next frame's previous); the mask's level 0 and the SH by `copyTextureToTexture` after the interpolation. The previous mask has one level (the reprojection reads level 0 only)
- **Kind:** none (same data)
- **Why:** The shading module hands three r187 its textures through `ShadingParameter.texture()`, and `ddgi-three` re-wraps a texture as a new `ExternalTexture` whenever the identity changes (`gi-light.ts:380-391`); swapping every frame would re-wrap and rebind every frame
- **Cause:** Same
- **Evidence:** Cited: `gi-light.ts:380-391` re-wraps on a new texture; the copy after the blend is Capsaicin's own. Measured on the Apple M5 (`radiance-cache-temporal` check 1): a still camera keeps every reprojected probe's seed and SH bit for bit across 12 frames; and (`radiance-cache-filter` check 4, RC4.2) after a frame the previous atlas equals the filtered atlas bit for bit, and a still camera's 7,350 reprojected tiles keep its cells
- **What is really behind it:** Copies at the end of the frame instead of a swap
- **Challenge:** None for the image. A small copy cost
- **Way back / others' workarounds:** Swap with a flip index (two bind groups per pass) if the shading contract learns double-buffered textures; costs nothing but the copies (57 KB mask, 1 MB SH, 7.4 MB atlas at 720p)
- **Assessment (2026-11-11):** Same result: three re-wraps a texture whose identity changes. Image impact none. Speed: Copies about 8.5 MB a frame at 720p (57 KB mask, 1 MB SH, 7.4 MB atlas). Effort to match: M: double-buffered textures in the shading contract. Recommendation: Keep: unless phase 5 shows the copies cost.

### D46

*Screen probes.* Writing a half-float texture in place needs an optional feature the platform doesn't require, so one value travels through an extra texture. Same result, 7.4 MB more.

- **Capsaicin:** `BlendScreenProbes` reads the reprojected radiance of the probe it replaces from `g_ScreenProbes_ProbeBuffer[pos]` and writes its own result to the same texel (read-write `RWTexture2D<float4>`, gi1.comp:1309, 1329)
- **Ours:** `SampleScreenProbes` copies that texel into a third atlas, the evicted-probe atlas, as it writes `PreviousProbeBuffer[pos]`; `BlendScreenProbes` reads it from there and writes the atlas write-only. The same value: nothing writes the atlas between the two passes
- **Kind:** none (same value)
- **Why:** Core WebGPU allows read-write storage textures only for the one-channel 32-bit formats; `rgba16float` needs `texture-formats-tier2` (WebGPU spec, texture format capabilities), which the platform doesn't require (D5's fallback is the core path)
- **Cause:** Rule (the platform doesn't require `texture-formats-tier2`)
- **Evidence:** Cited (WebGPU's read-write storage formats; D5)
- **What is really behind it:** A third atlas carries one value from `SampleScreenProbes` to `BlendScreenProbes`
- **Challenge:** None for the image; 7.4 MB at 720p. Use the tier-2 binding where the device has it
- **Way back / others' workarounds:** A read-write binding where the device has `texture-formats-tier2` (RC1.4: the RTX 3060 and Intel UHD do); 7.4 MB less at 720p
- **Assessment (2026-11-11):** Our rule: the platform doesn't require texture-formats-tier2. Image impact none. Speed: 7.4 MB at 720p. Effort to match: S: use the tier-2 binding where the device has it. Recommendation: Keep.

### D47

*Precision & GPU quirks.* Storing raw bits in a float texture could be corrupted if the GPU flushes tiny floats to zero, so we use an integer texture. Same bits; the WGSL rule behind it wasn't re-read.

- **Capsaicin:** `g_ScreenProbes_ProbeCachedTileIndexBuffer` is `R32G32B32A32_FLOAT`: the cached probe's position, and its normal's `packNormal` word stored with `asfloat` in w (gi1.comp:1306, 853)
- **Ours:** `rgba32uint`, every channel the bits (`bitcast`)
- **Kind:** none (same bits)
- **Why:** `packNormal`'s word for a normal with a small z (a horizontal one: (0, 1, 0) packs to 523,264) is a subnormal as an f32, and WGSL is believed to let an implementation flush subnormals to zero (assumed: the section, thought to be §15.7.2, was not re-read), which would turn the cached normal into (0, 0, 0)
- **Cause:** Forced if WGSL lets subnormals flush, which is **assumed**: the section (believed to be §15.7.2, "Differences from IEEE-754") was not re-read, and nobody measured whether a GPU flushes a stored subnormal
- **Evidence:** Assumed
- **What is really behind it:** The bits are the same; only the texture's format differs
- **Challenge:** None
- **Way back / others' workarounds:** None needed
- **Assessment (2026-11-11):** Platform can't? (unchecked): subnormal flushing, assumed. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep: re-read the WGSL section.

### D48

*Screen probes.* Summing many 'far away' distances overflows Capsaicin's counter. We cap each sample, so it can't wrap.

- **Capsaicin:** A probe cell's radiance and hit distance are summed as `uint(round(1e4 × x))` with 32-bit atomics (`ScreenProbes_QuantizeRadiance`, screen_probes.hlsl:231-234). A missed ray's hit distance is `f32tof16(FLT_MAX)` (D19), infinity, which D3D converts to `0xFFFFFFFF`, so a cell's distance sum wraps from the second missed ray
- **Ours:** The quantized hit distance of one sample is capped at ⌊(2³² − 1) / 64⌋ = 67,108,863 (6,710.9 m), so a whole probe's 64 rays in one cell can't wrap the sum; radiance is not capped
- **Kind:** Q (small)
- **Why:** Our miss stores 65,504 (D19), 655,040,000 quantized: seven misses in one cell wrapped. Ray guiding sends a probe's rays to its brightest cells, often the sky, and the reprojected hit distance sets where a cell's radiance is re-aimed. Measured on the M5 (`radiance-cache-temporal`, check 3): every ray of 2,043 of 2,400 probes in one sky cell, its hit distance 6,708 against the cap (wrapped it would read 5,106)
- **Cause:** Fix
- **Evidence:** Measured on the Apple M5 (check 3 above); Capsaicin's own wrap is assumed from D3D's float-to-uint rule, not run
- **What is really behind it:** Capsaicin's distance sum wraps for sky cells; ours can't for one probe's rays
- **Challenge:** Low: a sky cell's distance is "far" either way; the cap moves real hits beyond 6.7 km only
- **Way back / others' workarounds:** Any cap that keeps 64 samples below 2³²; a 64-bit sum (two words) if long real hits must keep their length. `SampleScreenProbes` can bring more than 64 samples into a cell (9 taps and the cached probes), which the cap doesn't bound
- **Assessment (2026-11-11):** Bug fix. Image impact low. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D49

*Screen inputs.* Our contract doesn't carry last frame's buffers, so the technique copies them itself each frame. Same data; one copy per frame.

- **Capsaicin:** The previous view-projection comes from Capsaicin's camera (`view_proj_inv_prev`, gi1.cpp:1817-1821), and its framework copies the visibility depth and geometry normals into `PrevVisibilityDepth` and `PrevGeometryNormal` at the start of each frame (gi1.cpp:1368-1369, capsaicin_internal.cpp:879-889)
- **Ours:** The technique keeps the latest frame's `FrameCamera` (jitter included) and, at the end of its screen stage, copies the host's depth (`r32float`) and decoded normals (`rgba16float`) with a small kernel of its own (`CopyPreviousScreen`)
- **Kind:** none (same data)
- **Why:** `FrameCamera` has no previous camera and `FrameScreen` no previous buffers; the host's targets (three's MRT) are overwritten each frame and may lack `COPY_SRC`
- **Cause:** Rule (the `FrameCamera` and `FrameScreen` contract)
- **Evidence:** Cited
- **What is really behind it:** The technique keeps its own history of the camera and the host's depth and normals
- **Challenge:** None for the image; a full-screen copy per frame. The contract could carry previous buffers for every screen technique
- **Way back / others' workarounds:** A previous camera and previous screen buffers in the platform contract, shared by every screen technique (and the denoiser, RC4.3)
- **Assessment (2026-11-11):** Our rule: the FrameCamera and FrameScreen contract. Image impact none. Speed: A full-screen copy per frame. Effort to match: L: previous buffers in the platform contract, shared by every screen technique. Recommendation: Keep.

### D50

*Screen inputs.* Motion vectors arrive in different units; we convert them.

- **Capsaicin:** `g_VelocityBuffer` holds the uv motion, y down: `(current_uv − previous_uv) × (1, −1)` with uv = ½ ndc (geometry.hlsl:104-110)
- **Ours:** `FrameScreen.velocity` holds the NDC motion (current − previous), converted on read: × (0.5, −0.5)
- **Kind:** none (same value)
- **Why:** The contract's channel is the NDC delta three's MRT velocity gives (`contract/frame.ts`)
- **Cause:** Same
- **Evidence:** Cited (geometry.hlsl:104-110; `contract/frame.ts`)
- **What is really behind it:** A unit conversion
- **Challenge:** None
- **Way back / others' workarounds:** None needed
- **Assessment (2026-11-11):** Same result: a unit conversion. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D51

*Precision & GPU quirks.* D3D gives defined answers for some out-of-range reads; WGSL doesn't, so we guard them.

- **Capsaicin:** Out-of-range reads and writes left to D3D: a cell index from `mapToHemiOctahedronInverse` × 8 can be 8 on the square's edge (an index past the 64-cell workgroup arrays), `normalize` of a zero-length reprojected ray gives NaN, out-of-screen previous positions and invalid seeds are loaded (D3D returns 0)
- **Ours:** The remapped cell is clamped to 7 (since RC4.2 `FilterScreenProbes`' neighbour cell too, gi1.comp:1417, whose 8 on the edge reads the next tile's cell 0, or past the atlas, in Capsaicin); a zero-length reprojected sample is skipped; only valid seeds and on-screen pixels are read; a previous position far off screen is clamped before it becomes an integer. `GIDenoiser_RemoveNaNs` in the filter (gi_denoiser.hlsl:75-82) keeps Capsaicin's formula, which removes a NaN only through D3D's `saturate`; in WGSL it bounds large values and leaves a NaN's result to the implementation
- **Kind:** none (guards)
- **Why:** WGSL makes an out-of-bounds workgroup access, a NaN conversion and an out-of-bounds load indeterminate (WGSL §6.5.7 out-of-bounds access, §15.7.6), where D3D's answers are defined
- **Cause:** Same (guards)
- **Evidence:** Cited (WGSL §6.5.7, §15.7.6)
- **What is really behind it:** Defined answers where Capsaicin relies on D3D's
- **Challenge:** None
- **Way back / others' workarounds:** None needed: each guard only acts where Capsaicin's result is undefined or unused
- **Assessment (2026-11-11):** Same result: guards. Image impact none. Speed: negligible. Effort to match: n/a. Recommendation: Keep.

### D52

*Precision & GPU quirks.* A sort key past the half-float range is written by hand. It only matters for ties.

- **Capsaicin:** `ReprojectScreenProbes` ranks candidates by `f32tof16(distance / cell_size) << 16 | lane` (gi1.comp:343)
- **Ours:** `pack2x16float` of the ratio, or the bits of infinity (0x7c00) past 65,504
- **Kind:** none (ties only)
- **Why:** `pack2x16float`'s result past the half range is indeterminate in WGSL (D15, D28); its rounding is the implementation's (D31), which only matters between candidates within a half step of each other
- **Cause:** Forced (WGSL half range)
- **Evidence:** Cited (D15, D28)
- **What is really behind it:** Infinity's bits by hand past the half range
- **Challenge:** None
- **Way back / others' workarounds:** None needed
- **Assessment (2026-11-11):** Platform can't: WGSL's half range. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D53

*Screen probes.* Many threads writing the same value at once is undefined in WGSL, so one thread does it.

- **Capsaicin:** Every thread of a `BlendScreenProbes` workgroup clears a newly allocated slot's flag bit and LRU flag, then reads the slot (gi1.comp:1281-1287)
- **Ours:** The cell-0 thread clears them with `atomicAnd` and `atomicStore`; every thread reads the slot with `atomicLoad` and masks the bit off itself
- **Kind:** none (same result)
- **Why:** Sixty-four plain writes and reads of one word race; WGSL gives a racing non-atomic read no defined value
- **Cause:** Same
- **Evidence:** Cited (WGSL's memory model: racing plain accesses)
- **What is really behind it:** One writer with atomics instead of 64 racing writers
- **Challenge:** None
- **Way back / others' workarounds:** None needed
- **Assessment (2026-11-11):** Same result. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D54

*Memory & bindings.* The platform scan's API made per-size buffers awkward, so they're sized for the biggest screen. About 16 MB more; no image change.

- **Capsaicin:** The probe cache's buffers are sized for the render size (`max_probe_count`, gi1.cpp:298-335)
- **Ours:** The four arrays the platform's scans read and write (the LRU flags and their scan, the list counts and theirs) are made in `prepare` for the largest screen the device can draw (`maxTextureDimension2D`², about 1 M probe tiles at 8192: 16.8 MB); the rest per screen size
- **Kind:** none (memory)
- **Why:** The platform's scans take their buffers when they are made, asynchronously, and `encodeScreen` (where a resize is seen) can't wait for them: building them per size was harder, so this is the easy path
- **Cause:** **Shortcut**
- **Evidence:** Cited: nothing in WebGPU stops per-size scans; the platform scan's API made it easier this way
- **What is really behind it:** Scan buffers for the largest screen
- **Challenge:** None for the image; about 16 MB of memory
- **Way back / others' workarounds:** Make the scans per screen size from `prepare` when the host announces its size, or let the scan take a buffer per encode; frees about 16 MB at 720p
- **Assessment (2026-11-11):** Quick setup. Image impact none. Speed: About 16 MB of memory. Effort to match: S: make the scans per screen size. Recommendation: Keep.

### D55

*Filter & SSGI.* As D27, for SSGI: white noise where Capsaicin uses blue. SSGI is most of the remaining flicker, so this is worth measuring.

- **Capsaicin:** SSGI draws its slice angle and step offset from `MakeBlueNoiseSampler(did, frame_index).rand2()` (ssgi.comp:156-157), Capsaicin's blue-noise tables
- **Ours:** The probes' PCG (`rcMakeRandom`, D18's), white noise, seeded with the pixel index plus the pixel count so its numbers aren't the interpolation's (Capsaicin keeps the two apart by blue-noise dimension 0 and 1, gi1.comp:1540)
- **Kind:** Q
- **Why:** As D27: the platform has no blue-noise source yet
- **Cause:** **Shortcut**
- **Evidence:** Cited: ssgi.comp:156-157; as D27, the platform has no blue-noise source (D7's tables never regenerated)
- **What is really behind it:** White noise where Capsaicin's is blue
- **Challenge:** Low until the denoiser: one slice a pixel, so the occlusion's noise is the pattern it leaves. Build D7's tables once for D27 and D55
- **Way back / others' workarounds:** D7's tables as a platform texture, for both; the denoiser (RC4.3) hides most of it
- **Assessment (2026-11-11):** Quick setup: the platform has no blue noise yet. Image impact low. Speed: none. Effort to match: M: blue-noise tables (D7), shared with D27. Recommendation: Measure first: the flicker after the denoiser (SSGI is most of it).

### D56

*Filter & SSGI.* SSGI needs last frame's lit picture. three draws the picture, so we capture three's own, which also includes specular and the sky.

- **Capsaicin:** SSGI reads last frame's lit colour, `PrevCombinedIllumination` (ssgi.cpp:81, 149), which Capsaicin's `Combine` pass writes as direct light plus GI plus emission (combine.comp:50-74; `R16G16B16A16_FLOAT`, combine.cpp:62-66), render-sized
- **Ours:** A new optional `FrameScreen` channel, `previousLighting` (last frame's HDR picture before tone mapping, read by screen uv, any size), asked for through `TechniqueRequirements.screen` as RC1.1's channels are. The three adapter draws the node `encodeBefore` is given (the composite) into a half-float target of its own (three's `rtt`) and hands the next frame's stage that target before it is drawn again: one more full-screen pass and 8 bytes a pixel; so it is three's whole picture, its sky background and specular included. The engine-free pages write a stand-in (albedo × a scale plus emission, or black)
- **Kind:** Q (small)
- **Why:** The platform has no lighting of its own on screen; the host draws the picture, and `FrameScreen` is how hosts hand screen buffers over
- **Cause:** Rule (the `FrameScreen` contract: the host draws the picture)
- **Evidence:** Cited: ssgi.cpp:81, 149, combine.comp:50-74. Measured on the Apple M5 (`radiance-cache-three`, RC4.2 review): the near-field light through three is non-zero and a frame behind: the frame the scene's lights go off it holds (×0.96-1.00 of before), the frame after it drops (×0.12, 0, 0 on Cornell, two-room, Sponza)
- **What is really behind it:** The host's own picture instead of Capsaicin's combine output; a channel request, not a side door
- **Challenge:** Low: three's picture is direct + GI + emission like Capsaicin's, plus specular and the sky background (weight 0 there). The capture costs a full-screen pass (not timed)
- **Way back / others' workarounds:** A platform combine pass if the platform ever draws the direct light itself. The sky pixels don't matter: a sample there is at the far plane, past the 0.2 m radius, and gets weight 0
- **Assessment (2026-11-11):** Our rule: FrameScreen: the host draws the picture. Image impact low. Speed: One extra full-screen pass, 8 bytes a pixel (not timed). Effort to match: L: a platform combine pass, if the platform ever draws direct light itself. Recommendation: Keep.

### D57

*Filter & SSGI.* Capsaicin runs SSGI as a separate technique. The platform has no shared screen textures, so it lives inside the radiance cache. Same result.

- **Capsaicin:** SSGI is a technique of its own, run before GI-1 (gi1_renderer.cpp:60-61), writing shared textures (`OcclusionAndBentNormal`, `NearFieldGlobalIllumination`, ssgi.cpp:74-77) that GI-1 binds when they exist (`HAS_OCCLUSION`, gi1.cpp:1420-1423)
- **Ours:** A module of the radiance cache (`ssgi.ts`), run first in its screen stage, its two outputs owned by the technique and bound by `InterpolateScreenProbes`; its options are the technique's settings (`ssgiSliceCount` and the rest, ssgi.h:46-52). Capsaicin's `ssgi_unroll_kernel_` (default off; a variant with the counts fixed at 1 and 2) is not ported
- **Kind:** none
- **Why:** The platform has no textures shared between techniques; building SSGI into the one technique that reads it was easier
- **Cause:** **Shortcut**
- **Evidence:** Cited: gi1_renderer.cpp:60-61, ssgi.cpp:74-77; nothing in WebGPU stops a separate technique, the platform just has no shared screen textures
- **What is really behind it:** One module where Capsaicin has a technique and shared textures
- **Challenge:** None for the image
- **Way back / others' workarounds:** A platform service for screen textures one technique writes and another reads, if a second reader (DDGI, the denoiser) wants SSGI
- **Assessment (2026-11-11):** Quick setup. Image impact none. Speed: none. Effort to match: M: a platform service for shared screen textures. Recommendation: Keep.

### D58

*Filter & SSGI.* Where Capsaicin relies on D3D quietly dropping NaNs, we guard explicitly.

- **Capsaicin:** SSGI divides a sample's offset by its length (ssgi.comp:199-200), which is 0 for a sample at the pixel's own point, and decodes every sample's normal with `normalize(2 n − 1)` (ssgi.comp:212), which a sky pixel's cleared normal turns into −(1, 1, 1)/√3; D3D's `max` and `>=` then drop the zero-length sample's NaN
- **Ours:** A sample of zero length is skipped (it still becomes the previous sample, as in Capsaicin); a sky sample (far-plane depth) gets −(1, 1, 1)/√3 as its normal
- **Kind:** none (guards)
- **Why:** WGSL gives a NaN no defined result and lets the implementation assume there are none (NOTES, §15.7.2), and a host with signed-float normals clears the sky to 0, which `normalize` turns into NaN
- **Cause:** Same (guards)
- **Evidence:** Cited: ssgi.comp:199-200, 212; D3D's NaN rule for `max` from memory (not re-read); WGSL's NaN rule as NOTES records it (§15.7.2). Measured on the M5: 0 zero-length samples over 921,600 pixels on the corner
- **What is really behind it:** Defined answers where Capsaicin's rest on D3D's NaN rules
- **Challenge:** None
- **Way back / others' workarounds:** None needed: each guard gives Capsaicin's D3D result. Measured: no zero-length sample on `radiance-cache-ssgi` (the twin counts them); a sky sample's weight is 0 either way
- **Assessment (2026-11-11):** Same result: guards. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D59

*Precision & GPU quirks.* The Mac's GPU rounds half-float texture stores down, so probe hit distances drift a little low (about 0.4%; much more for probes kept only by reprojection). A D31-style correction fixes it.

- **Capsaicin:** Stores to `R16G16B16A16_FLOAT` textures (the probe atlas, written by `BlendScreenProbes` and both `FilterScreenProbes` passes every frame, gi1.comp:1329, 1442) convert with D3D's rounding, to nearest (the review's reading of D3D; not re-read here)
- **Ours:** The same `textureStore`s to `rgba16float`; on the Apple M5 they round toward zero (`radiance-cache-filter` check 5: 4,096 values that aren't halves, every one stored toward zero, 2,017 of them where nearest would differ, ties included). So each pass can lose up to a half step, and the filter's `min` keeps a lower hit distance: a constant 40 m reads 39.9375 after one filter, 29.7 m after 300 filters with no new rays; in the frame's loop the sky's hit distance falls 27 m (0.4%) over the first 100 frames and then holds (3 m more by frame 300); radiance moves under a half step. The M5's `pack2x16float` rounds to nearest (NOTES), so it is the texture store, not the conversion
- **Kind:** Q (small)
- **Why:** WGSL leaves the conversion's rounding to the implementation (believed; the section was not re-read), and the M5 rounds toward zero; the RTX 3060's stores also read low (NOTES: the sky bake, T2.6a)
- **Cause:** Forced if WGSL leaves the store's rounding to the implementation (**assumed**: not re-read)
- **Evidence:** Measured on the Apple M5 (`radiance-cache-filter` checks 5, 6); D3D's rounding to nearest taken from the review, not re-read; Windows GPUs not run
- **What is really behind it:** The same stores; the M5's hardware rounds them down
- **Challenge:** Low: the hit distance settles 0.4% low in the frame's loop; radiance under a half step. Without new rays (a probe kept only by reprojection for long) it would fall further: −26% after 300 filters. Measure on the RTX 3060 and Intel UHD
- **Way back / others' workarounds:** Store through `pack2x16float` into an `rg32uint` atlas (the M5 rounds that to nearest; the Windows GPUs truncate, D31) or round stochastically before the store, as `gi/direct` does
- **Assessment (2026-11-11):** Platform can't? (unchecked): forced if WGSL leaves the store's rounding to the GPU (not re-read). Image impact low. Speed: none. Effort to match: S: store through pack2x16float into rg32uint, or round stochastically before the store. Recommendation: Fix now: already the agreed follow-up; measure the Windows GPUs too.
- **Disputed:** Called forced, but only if WGSL leaves the rounding open (not re-read), and a known workaround exists (D31's correction, or stochastic rounding). It is better described as a GPU quirk with a fix.

### D65

*Denoiser.* With the usual feature we use Capsaicin's formats. On a device without it, wider formats stand in. (Image impact low and assumed, fallback only.)

- **Capsaicin:** The GI denoiser's blur mask is `R8_SNORM` and its colour deltas `R16_FLOAT` (gi1.cpp:1200, 1222)
- **Ours:** The same formats where the device has `texture-formats-tier1` (the default, `textureFormats: 'auto'`); on a device without it, the one declared fallback: `rgba8snorm` (r used: the same snorm steps) and `r32float` (f32 instead of half)
- **Kind:** P (fallback only)
- **Why:** Core WebGPU refuses `r8snorm` and `r16float` as storage textures without `texture-formats-tier1` (measured on the M5: a device without the feature refuses both, accepts `rgba8snorm`, `r32float`, `rgba16float`). The host owns the device, so the module takes what it was given (DDGI's distance atlas does the same, `rg16float` or `rgba16float`)
- **Cause:** Same with `texture-formats-tier1`; **Forced** without it (the fallback)
- **Evidence:** Measured on the Apple M5 (`radiance-cache-denoiser` check 0: refused without the feature). Both paths run checks 1-4 on the M5 and pass; their numbers agree within 0.003% (check 1's settled mean 0.99682 against 0.99683)
- **What is really behind it:** Capsaicin's formats where the device allows them, one declared fallback where it doesn't (until RC4.3's review the fallback was the only path: a shortcut)
- **Challenge:** Low, **assumed**: the fallback's delta is f32 where Capsaicin's is a half stored back every frame, which only the constant and noisy scenes of check 1-3 measured, not a real scene
- **Way back / others' workarounds:** Ask the host for `texture-formats-tier1` (the platform's `optionalFeatures` does). The fallback costs 7 bytes a pixel (6.5 MB at 720p)
- **Assessment (2026-11-11):** Same result: the fallback alone is forced. Image impact low. Speed: Fallback only: 7 bytes a pixel more (6.5 MB at 720p). Effort to match: n/a. Recommendation: Keep.

### D66

*Denoiser.* The denoiser is a self-contained module, so it doesn't overwrite its caller's texture.

- **Capsaicin:** `InterpolateScreenProbes` writes into `g_GIDenoiser_ColorBuffer[i]`, which `ReprojectGI` reads and overwrites in place (`RWTexture2D<float4>`, gi1.comp:1638, 4004, 4100)
- **Ours:** The interpolation keeps its own irradiance texture (as since RC2.2); the denoiser reads it as its input and writes its history to a texture of its own
- **Kind:** none (same values)
- **Why:** The denoiser is a module that takes the caller's texture (`@pmndrs/denoiser`'s `DiffuseInputs`); in place would also need a read-write `rgba16float` binding (`texture-formats-tier2`, D5)
- **Cause:** Rule (the module takes the caller's texture)
- **Evidence:** Cited (gi1.comp:1638, 4004, 4100; D5)
- **What is really behind it:** The input and the history in two textures instead of one
- **Challenge:** None
- **Way back / others' workarounds:** None needed. One more screen texture than Capsaicin, which the radiance cache had anyway
- **Assessment (2026-11-11):** Our rule: the module takes the caller's texture. Image impact none. Speed: One more screen texture. Effort to match: n/a. Recommendation: Keep.

### D67

*Denoiser.* Our contract marks sky by depth, not by an empty normal. A sky neighbour weighs about zero either way.

- **Capsaicin:** The denoiser tells sky by a zero shading normal (gi1.comp:4011) and a previous sky pixel by a zero previous normal (4041); in `FilterGI` a sky neighbour's normal decodes to (−1, −1, −1) (4131)
- **Ours:** Sky by depth at the far plane (as D17); `KeepGuides` stores 0 as the previous normal of a sky pixel, so the previous test is Capsaicin's; a sky neighbour's normal in `FilterGI` is whatever the host wrote there
- **Kind:** none (sky); Q (tiny, assumed)
- **Why:** `FrameGuides` and `FrameScreen` say sky by depth; the host's normal at a sky pixel means nothing
- **Cause:** Rule (the `FrameScreen` and `FrameGuides` contract)
- **Evidence:** Cited; the sky neighbour's weight is assumed, not measured
- **What is really behind it:** Sky by depth, as D17
- **Challenge:** None for surfaces; a sky neighbour weighs about 0
- **Way back / others' workarounds:** None needed: a sky neighbour next to a surface weighs `exp2(−200 × |1 − d / far|)`, about 0
- **Assessment (2026-11-11):** Our rule: the FrameScreen and FrameGuides contract. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D68

*Denoiser.* Because the denoiser is self-contained it keeps its own copy of last frame's depth and normals, a second copy alongside the radiance cache's.

- **Capsaicin:** Capsaicin's framework keeps last frame's depth and shading normals for every technique (`PrevVisibilityDepth`, `PrevShadingNormal`, gi1.cpp:2010-2014)
- **Ours:** The denoiser keeps its own copy (`KeepGuides`, after its passes), so the radiance cache holds two (its own for the screen probes, D49, and the denoiser's)
- **Kind:** none (same values)
- **Why:** The module is self-contained, and the platform contract carries no previous buffers (D49)
- **Cause:** Rule (the module is self-contained; D49)
- **Evidence:** Cited
- **What is really behind it:** A second copy of last frame's depth and normals
- **Challenge:** None for the image; 11 MB at 720p
- **Way back / others' workarounds:** Optional previous guides in the module's inputs, or previous buffers in the contract (D49's way back): 12 bytes a pixel (11 MB at 720p)
- **Assessment (2026-11-11):** Our rule: the module is self-contained (D49). Image impact none. Speed: 11 MB at 720p and a copy. Effort to match: M: optional previous guides in the module's inputs. Recommendation: Keep.

### D69

*Denoiser.* Positions are rebuilt relative to the camera for float precision. Same values.

- **Capsaicin:** Positions are rebuilt with `inverse(view_projection)` of this frame and the previous one in world space (gi1.comp:4023, 4049), the eye at `g_Eye`
- **Ours:** Both inverses are taken relative to this frame's eye on the CPU in f64 (`frameConstants`), so points come out as offsets from the eye
- **Kind:** none (same values)
- **Why:** f32 precision far from the origin (as RC2.1's camera-relative matrices)
- **Cause:** Same
- **Evidence:** Cited; measured: the GPU matches the CPU reference on `radiance-cache-denoiser` checks 1-3
- **What is really behind it:** Camera-relative matrices
- **Challenge:** None
- **Way back / others' workarounds:** None needed
- **Assessment (2026-11-11):** Same result. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D70

*Denoiser.* As D29, in the denoiser.

- **Capsaicin:** `FilterGI`'s linear depth is `toLinearDepth(depth, g_NearFar)`, reversed-Z (transform.hlsl:144-148; gi1.comp:4120, 4130)
- **Ours:** The rebuilt point's distance along the view axis, for either depth convention
- **Kind:** none (same value)
- **Why:** As D29: `FrameGuides` carry the host's depth convention
- **Cause:** Same
- **Evidence:** Cited (D29); measured as D69
- **What is really behind it:** A unit conversion
- **Challenge:** None
- **Way back / others' workarounds:** None needed
- **Assessment (2026-11-11):** Same result. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D71

*Denoiser.* Capsaicin's NaN cleanup only works because of how D3D treats NaN. On the Mac it let NaNs through and they spread, so we test the bits directly.

- **Capsaicin:** `GIDenoiser_RemoveNaNs` is `x / (1 + x)`, saturated, mapped back (gi_denoiser.hlsl:64-92): on D3D `saturate` takes NaN to 0, so NaN and both infinities come out 0
- **Ours:** The same formula for finite values; a non-finite value is told by its bits and gives 0
- **Kind:** none (same result)
- **Why:** WGSL gives `saturate(NaN)` no defined value: an implementation may assume no NaN or infinity at run time (WGSL §15.7.2, "Differences from IEEE-754"). Measured on the M5 with the formula as written: a NaN and an infinity in the input came through, the history kept them and they spread (with them in the input three frames running, 54 non-finite values by the third); with the bit test they are gone the next frame, as Capsaicin's are on D3D. Not shown whether the M5's `saturate` or the compiler's folding under that assumption let them through
- **Cause:** Forced
- **Evidence:** Cited (WGSL §15.7.2: implementations may assume no NaN, so `saturate(NaN)` is undefined) and measured on the Apple M5 (`radiance-cache-denoiser` check 4, before and after the bit test); which of `saturate` and the compiler's folding is at fault is not shown
- **What is really behind it:** WGSL doesn't promise D3D's NaN flush; the bit test restores D3D's result
- **Challenge:** None (it matches Capsaicin on D3D); without it a NaN in the input would spread through the history
- **Way back / others' workarounds:** None needed
- **Assessment (2026-11-11):** Platform can't: WGSL's NaN rules. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D72

*Denoiser.* Loads the texel a point sampler would pick, so no sampler binding is needed.

- **Capsaicin:** Taps and the velocity are read with `SampleLevel(g_NearestSampler, uv)` (gi1.comp:4015, 4038-4056)
- **Ours:** `textureLoad` of the texel a clamped point sampler picks: `clamp(floor(uv × size))`
- **Kind:** none (same values)
- **Why:** No sampler binding needed (the module binds textures only)
- **Cause:** Same
- **Evidence:** Assumed: Capsaicin's nearest sampler clamps (gfx's default address mode; `third_party/gfx` isn't checked out here), and D3D's sub-texel snapping is ignored
- **What is really behind it:** A load instead of a point sample
- **Challenge:** None
- **Way back / others' workarounds:** None needed
- **Assessment (2026-11-11):** Same result. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D73

*Denoiser.* An on/off switch for comparisons, and a camera cut clears the history. Defaults match Capsaicin.

- **Capsaicin:** The denoiser always runs, and its history starts empty only when it is made (gi1.cpp:1193-1227, 2882-2927)
- **Ours:** `useGIDenoiser` (default on; off, the shading module reads the interpolation); `resetHistory` and `resetScreenProbes` also empty the denoiser's history
- **Kind:** none (default behaviour)
- **Why:** A switch for comparisons; a camera cut should drop the screen's history
- **Cause:** Same (an option and a reset Capsaicin hasn't)
- **Evidence:** Cited
- **What is really behind it:** Defaults as Capsaicin
- **Challenge:** None
- **Way back / others' workarounds:** None needed
- **Assessment (2026-11-11):** Same result. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D74

*Denoiser.* Unit conversions to the sibling denoiser's conventions.

- **Capsaicin:** The velocity buffer holds uv motion with y down (geometry.hlsl:104-110); the camera matrices are jittered with TAA on (gi1.cpp:1816-1820); `cell_size` uses the camera's `fovY` (gi1.cpp:1839-1843)
- **Ours:** `FrameGuides.motion` is NDC motion, converted × (0.5, −0.5) (as D50); the jitter comes in pixels (`@pmndrs/denoiser`'s `FrameCamera`), and the module remembers the previous frame's; the field of view comes from `projection[5]`
- **Kind:** none (same values)
- **Why:** `@pmndrs/denoiser`'s conventions
- **Cause:** Same
- **Evidence:** Cited
- **What is really behind it:** Unit conversions
- **Challenge:** None
- **Way back / others' workarounds:** None needed
- **Assessment (2026-11-11):** Same result. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

### D75

*Denoiser.* WGSL doesn't define the sign of NaN, so we compute the sign with comparisons, giving Capsaicin's answer.

- **Capsaicin:** `ReprojectGI` restores the colour delta's sign with HLSL's `sign` (gi1.comp:4065), which returns an int: 0 for a NaN delta, whose `RemoveNaNs` then gives 0
- **Ours:** The sign by comparisons, `select(select(0, −1, x < 0), 1, x > 0)`: 0 for NaN
- **Kind:** none (same result)
- **Why:** WGSL's `sign` returns a float and gives `sign(NaN)` no defined value (WGSL §15.7.2), so a NaN delta could stay NaN (`0 × NaN`) and spread through the 3 × 3 taps; Metal gives 0, so the M5 shows no difference (`radiance-cache-denoiser` check 4 reads the delta: clean the frame after)
- **Cause:** Forced
- **Evidence:** Cited (WGSL §15.7.2; HLSL's `sign` returns int); not measured on a GPU whose `sign(NaN)` isn't 0
- **What is really behind it:** A float `sign` where HLSL's is an int
- **Challenge:** None
- **Way back / others' workarounds:** None needed
- **Assessment (2026-11-11):** Platform can't: WGSL's sign(NaN) is undefined. Image impact none. Speed: none. Effort to match: n/a. Recommendation: Keep.

## Worth telling AMD

Things that look wrong in Capsaicin v1.3 itself, found while porting (none was run in Capsaicin):

- **D32:** a new tile can go on `UpdateTiles`' list twice (gi1.comp:1043-1048, 1819-1824), and the two workgroups race on its scratch. Seen in our port on the Intel UHD.
- **The second bounce reads about 3.3 times high in a closed room** (RC3.3): `PopulateMultibounceCells` drops 70% of the rays (gi1.comp:1879) and adds no entry for a ray that misses (gi1.comp:1857-1861); `UpdateMultibounceCells` divides each surviving sample by 1 - 0.7 (gi1.comp:1966) and counts it as one sample (gi1.comp:1975); `UpdateTiles` and `ResolveCells` then take the mean over the samples that arrived (gi1.comp:2189-2190, 2381-2382). So the discard is made up for twice, and misses don't count as zero. Measured on our port: 1.67 L where the physical value is at most 0.25 L on the sunlit corner.
- **D40:** for a second-bounce entry, `VisibilityQueryBuffer` holds a visibility index (gi1.comp:1794), but `PopulateCells` (gi1.comp:2100-2106) and `UpdateMultibounceCells` (gi1.comp:1977-1981) add light to it as if it were a probe ray, and `UpdateMultibounceCells` reads its bypass flag from the shadow-ray list by visibility index.
- **D41:** `GenerateMultibounceReservoirs` (gi1.comp:2617-2626) and `PopulateCells` (gi1.comp:2054, 2063) interpolate vertex normals without the instance's transform.
- **Portability, not a bug (RC4.3):** `GIDenoiser_RemoveNaNs` (gi_denoiser.hlsl:64-92) removes NaN only because D3D's `saturate` takes NaN to 0; on Metal through WebGPU it doesn't (measured on the M5, D71), and a NaN then stays in the denoiser's history and spreads. A bit test, or `isnan`, would make it hold on every API.
- **A question (RC4.3): half stores fed back through the denoiser's history.** The history is `R16G16B16A16_FLOAT` holding irradiance × samples (up to 64), stored every frame: each store's rounding is fed back. On a GPU that truncates f32 to half (the M5 here, measured; the RTX 3060 under D3D12 in NOTES) a constant input settles 0.32% low on average and 0.88% at worst; round-to-nearest halves give up to 0.41% either way (`radiance-cache-denoiser` check 1, the CPU reference matches the GPU frame by frame). Small, but it grows with the history's length.
- **A question (RC4.3): `FilterGI` counts the centre pixel twice** (`weight = 1` and the centre's colour before the loop, then the loop's `r = 0` tap again, gi1.comp:4117-4141), so the centre weighs about twice a neighbour. It can't darken (line 4144 divides by the total weight); it only sharpens the kernel. Intended?
- **A question, not a bug (RC4.1):** with ray guiding on, the screen probes don't converge to their own no-history image: a constant sky reads about 0.69 πL, a scene with a small bright light reads brighter (Cornell ×1.54). It follows from the empty-cell backup `64 / (R (64 − R))` (gi1.comp:1213-1232) with the hysteresis that takes darker values at once (1266) and the CDF's feedback (917-919). Is the backup meant to depend on the sum of the rays, and is that bias expected, perhaps made up for by the denoiser? Modelled and seen in our port, not run in Capsaicin.
- **SSGI's horizon truncation intersects from the world's origin** (RC4.2; ssgi.comp:214-221): `IntersectDirPlaneOneSided(prev_sample_world_pos, ...)` is h3r2tic's test from rtoy-samples (`ssgi.glsl`, `process_sample`), where `prev_sample_vs` and `center_vs` are view-space, relative to the eye; Capsaicin's positions are world space, so the ray starts at the world's origin. Its result is clamped between two horizons, so it stays bounded, but the 1e-5 clamp on its denominator can multiply an input's rounding by up to 1e5: it is where our f32 GPU and the f64 twin differ most (`radiance-cache-ssgi`).
- **SSGI scales its near-field light once per side, and the sum so far each time** (RC4.2; ssgi.comp:239): `global_lighting *= proj_world_normal_len` inside the side loop scales the first side's light twice (and an earlier slice's again in each later slice); h3r2tic's scales a slice's whole result once (`col *= slice_contrib_weight`).
- **A question: SSGI's near-field light is about E / π, added to the probes' E** (RC4.2; ssgi.comp:224-228, gi1.comp:1625, 1633): each sample's light is weighted by its share of the cosine-weighted hemisphere (`IntegrateHalfArc`, which sums to 1 over an open slice), so the sum is radiance times a share, irradiance / π, and `InterpolateScreenProbes` adds it to E from the SH, which carries π. On `radiance-cache-ssgi`, floor pixels within 5 cm of a wall lit at 1.5 read a median 0.38 of (1 − occlusion) × 1.5 (a physical irradiance would be about π times the full share). h3r2tic's original has no π either. Is it meant?
- **An observation, small: SSGI rebuilds a sample at its own uv with its texel's depth** (RC4.2; ssgi.comp:191-196): with no minimum step the first sample often falls within a pixel of the centre, and on a surface whose depth changes across a texel it sits off the plane and raises a horizon. It costs 2-4 points of E on an open floor (the review's f64 twin: 0.735 against 0.753 πL with samples on texel centres). Most of SSGI's darkening is its default of one slice (phase 4 notes), which is a setting, not a fault. Also, SSGI reads last frame's lit colour at this frame's uv with no reprojection (ssgi.comp:211), where h3r2tic's reads a reprojected lighting texture.

## Keeping it honest

- Every new difference gets a summary row and a full entry here in the same commit, with the cause, evidence, impact, effort and recommendation filled in. The implementer fills it in, and the reviewer checks the cause against the Capsaicin source and the platform. A row that calls itself forced needs a citation or a measurement.
- The reviewer also checks the diff for differences from Capsaicin that have no row at all. Those are the ones that matter most.
- Staging differences, such as phase 2's cache-bypass path, aren't rows. They go in the phase notes and must be gone by the end of their phase.
- When a row changes cause (for example, a WebGPU feature arrives, or someone shows a rule isn't needed), update the line and say so in PROGRESS.
- Rows measured only on Windows say "M5 not checked yet" until someone runs them on the Mac.

## Shortlist

The rows to act on now (*Fix now* and *Measure first*), roughly by impact over effort.

1. ~~**[D33](#d33)**~~ Closed by RC6.1 (2026-11-11); what remains is noise only (Keep; revisit with D4).
2. **[D20](#d20)** (Materials & lights; image low; effort S): Measure first: a bigger bake on an outdoor scene; compare the probes' E.
3. **[D2](#d2)** (Materials & lights; image high; effort L): Measure first: the image on Sponza and Bistro with real albedo at hits, against the Cycles references.
4. **[D4](#d4)** (Materials & lights; image high; effort L): Measure first: phase 5's image comparison with native Capsaicin on Bistro, where many lights matter.
5. **[D59](#d59)** (Precision & GPU quirks; image low; effort S): Fix now: already the agreed follow-up; measure the Windows GPUs too.
6. **[D42](#d42)** (Multi-bounce; image none; effort S): Fix now: cheap; matters on memory-tight devices.
7. **[D27](#d27)** (Screen probes; image low; effort M): Measure first: the flicker after the denoiser with blue noise against white.
8. **[D55](#d55)** (Filter & SSGI; image low; effort M): Measure first: the flicker after the denoiser (SSGI is most of it).
9. **[D39](#d39)** (Multi-bounce; image low; effort S): Fix now: cheap and restores the faithful port; count the second-bounce rays that change hit.
10. **[D37](#d37)** (World cache; image low; effort S): Measure first: count the rays that change path against the unpushed point.
11. **[D17](#d17)** (Screen inputs; image low; effort L): Measure first: a scene with strong normal maps at grazing angles.
12. **[D3](#d3)** (Screen inputs; image medium; effort L): Measure first: D17's grazing-angle check on a normal-mapped scene decides whether it is worth it.

High-impact rows that wait on a decision from Dennis rather than on work: **D8** (diffuse-only materials; also blocks reflections), **D21** (emissive surfaces as lights; Cornell can't match without it) and **D16** (follows D8).
