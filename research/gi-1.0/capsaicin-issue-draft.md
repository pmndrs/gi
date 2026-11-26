# Draft: GitHub issue for GPUOpen-LibrariesAndSDKs/Capsaicin

Status: draft only, not posted. For Dennis to review and post himself.
Source checked: Capsaicin v1.3, commit `914b91596cd119eda85fbc1d3c7ee6ac391b1452` (all line numbers verified against that commit; only unmodified files cited).

---

## Title

GI-1.2: multibounce cache estimate is scaled by 1/(1-p), and two bypass paths mix up probe-ray and visibility indices

## Body

We are doing a line-by-line WebGPU port of GI-1.2 for a research project, and while validating it against analytic test scenes we ran into two things in the multibounce path that look unintended. Apologies if we have misread the intent; happy to be corrected.

### 1. Multibounce estimate is biased by 1/(1-p)

**What happens.** `PopulateMultibounceCells` drops a first-bounce entry when `rand() <= p` (p = `discard_multibounce_ray_probability`, default 0.7). A kept ray that hits adds `L * brdf / pdf / (1 - p)` with sample weight 1 to the first-bounce cell. Dropped rays add nothing and no count; misses also add nothing (empty miss shader, and the inline path only handles hits). `UpdateTiles` then divides by the count and blends, and `ResolveCells` divides by the weight again. So the stored value is `E[L f cos / pdf | kept, hit] / (1 - p)`: the Russian-roulette weight is applied, but the mean is taken over kept rays only, so the 1/(1-p) is never cancelled. That is a 3.33x scale on the second bounce in a closed scene, and an extra 1/P(hit) in open ones.

**Where.**
- Default p: [gi1.h#L64](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/gi1.h#L64)
- Discard: [gi1.comp#L1879-L1882](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/gi1.comp#L1879-L1882)
- Hit-only handling: [gi1.comp#L1854-L1866](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/gi1.comp#L1854-L1866), empty miss shader [gi1.rt#L120-L123](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/gi1.rt#L120-L123)
- 1/(1-p) and weight 1: [gi1.comp#L1964-L1975](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/gi1.comp#L1964-L1975), [hash_grid_cache.hlsl#L301-L305](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/hash_grid_cache.hlsl#L301-L305)
- Per-frame mean and blend: [gi1.comp#L2184-L2199](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/gi1.comp#L2184-L2199)
- Readback divides by weight: [gi1.comp#L2379-L2383](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/gi1.comp#L2379-L2383)

**Why it matters.** In our port, a closed "furnace" (floor and ceiling, albedo 0.5, sun from above) gives a second bounce of 1.061 against the analytic 0.318 (ratio 3.335). Removing the 1/(1-p) factor gives 0.319. Changing p also changes brightness, which a Russian-roulette weight should not do.

**Suggested fix.** Either (a) count every first-bounce entry in the sample weight (kept or dropped, hit or miss), keeping the 1/(1-p); or (b) drop the 1/(1-p) factor, since the mean is already over kept samples only. Option (a) also removes the 1/P(hit) term in open scenes.

### 2. Index mix-up in two cache-bypass paths

**What happens.** A first-bounce visibility entry stores its probe ray index in `VisibilityQueryBuffer`; a second-bounce entry stores the visibility index of its first-bounce entry. `UpdateMultibounceCells` correctly treats that word as a visibility index (L1957), but `PopulateCells` and `UpdateMultibounceCells` also pass it straight to `ScreenProbes_AccumulateRadiance`, which indexes probe rays. For second-bounce entries the light therefore lands on an unrelated probe ray.

Separately, `VisibilityRayBuffer` is written at the compacted shadow-ray index, but `UpdateMultibounceCells` reads it at `did2`, a visibility index, so the bypass flag it tests belongs to some other entry and can be a stale word from an earlier frame.

**Where.**
- First-bounce store (probe ray index): [gi1.comp#L1016](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/gi1.comp#L1016)
- Second-bounce store (visibility index): [gi1.comp#L1794](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/gi1.comp#L1794)
- Read as visibility index: [gi1.comp#L1957](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/gi1.comp#L1957)
- Read as probe ray index: [gi1.comp#L2100-L2106](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/gi1.comp#L2100-L2106), [gi1.comp#L1977-L1982](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/gi1.comp#L1977-L1982), into [screen_probes.hlsl#L249-L259](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/screen_probes.hlsl#L249-L259)
- `VisibilityRayBuffer` writes at compacted index: [gi1.comp#L2525](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/gi1.comp#L2525), [#L2578-L2583](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/gi1.comp#L2578-L2583), [#L2669-L2678](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/gi1.comp#L2669-L2678)

**Why it matters.** It adds light to the wrong screen-probe rays (a small, scene-dependent error that is hard to spot by eye). In our port's furnace and corner tests about 780 to 1,216 rays per frame at 1280x720 are affected, a count that a CPU model of the index flow predicts exactly.

**Suggested fix.** For second-bounce entries, resolve the probe ray with a second lookup (`VisibilityQueryBuffer[VisibilityQueryBuffer[v]]`), or skip the direct probe patch for them in `PopulateCells`. Store the bypass flag keyed by visibility index (or recompute it) instead of reading `VisibilityRayBuffer[did2]`.

Thanks for releasing Capsaicin and GI-1.x; the code has been a great reference.

---

## Notes for Dennis (not part of the issue)

- **Word count.** The issue body (title through "Thanks") is about 580 words (excluding link URLs).
- **Issue 2 intent is uncertain.** The bypass logic exists to avoid leaks when a segment is shorter than a hash cell. For `UpdateMultibounceCells`, the flag that would make sense is the *first-bounce* entry's bypass flag, but that flag is only stored compacted in `ResolveBuffer` (L2530, L2566), not keyed by visibility index, so there is no one-line fix; hence the neutral wording. For `PopulateCells` (L2104), even with the right probe ray, `payload.lighting` at a second-bounce vertex has not been multiplied by the first-bounce brdf/pdf, so simply fixing the index would add an unweighted second-bounce term; skipping the patch for second-bounce entries may be the cleaner fix. AMD may prefer to decide this.
- **Furnace numbers** (1.061 vs 0.318, 0.319 after the change) and the 780 to 1,216 rays/frame are from our WebGPU port and CPU model, not native Capsaicin. If you want a native number before posting, the Windows RTX laptop can run the same furnace scene through the bench lock.
- **Deliberately excluded:** second-bounce hits get no emission term. Our review judged this correct by design, so it is deliberately not in the issue.
- **Possible third issue (not included).** The screen-probe SH projection weights each hemi-octahedral cell by 1/8 (`radiance / probe_size`, [gi1.comp#L1466](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin/blob/914b91596cd119eda85fbc1d3c7ee6ac391b1452/src/core/src/render_techniques/gi1/gi1.comp#L1466)) instead of a solid-angle weight; on native Capsaicin a constant sky reads 1.30 πL at frame 1 and 0.87 πL at steady state (D25 in `native-capsaicin.md`). Two-line draft if you want it:

  > **3. Screen-probe SH projection is not normalised to the hemisphere.** `ProjectScreenProbes` weights each of the 64 cells by 1/8 (gi1.comp L1466) rather than its solid angle, so a constant sky of radiance L reads 1.30 πL irradiance at frame 1 and 0.87 πL at steady state (expected πL); weighting each cell by its solid angle reads 1.00 πL in our port.
