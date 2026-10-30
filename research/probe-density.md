# Probe density: counts, spacing and the fixed comparison workload

Clarification,29 October2026. No probe-count change was requested or made during G4.

A regular volume has nx×ny×nz candidate probes. Per-axis spacing is extent/(count−1). Density should follow the spatial variation of indirect lighting and geometry, not the number of lamps. Small rooms, corners, doorways, occlusion changes and volume coverage affect useful density. Rays per probe instead affect directional sampling/noise. Halving spacing on all three axes approaches eight times the probes. Updating every active probe costs roughly activeProbes×raysPerProbe×passes, plus visibility/shading/integration work.

| Reference | Grid | Allocated |
| --- | --- | --- |
| Original Homefig Cornell kitchen |8×5×8|320|
| Original Homefig two-room leak fixture |17×5×9|765|
| three.js r186 Cornell light-probe demo |6×6×6|216|
| three.js r186 Sponza light-probe demo |10×7×7|490|
| Shared NVIDIA two-room fixture used for our comparison |32×6×32|6,144|

Homefig source is in browser/homefig-bench/app/src/scenes/{cornellKitchen,twoRoomLeak}.ts, originally extracted from homefig/homefig@bc7cbdaa615fa9da72a4ff4123ac51b593a2be67. The upstream r186 defaults are [Cornell](https://github.com/mrdoob/three.js/blob/r186/examples/webgpu_lightprobes.html) and [Sponza](https://github.com/mrdoob/three.js/blob/r186/examples/webgpu_lightprobes_sponza.html). These upstream examples store SH without directional visibility moments; their counts are not proof of equivalent DDGI leak quality.

The shared native fixture inherited spacing[11,11,11] in unscaled native asset units, not an asserted meter scale. It is a fixed reference/performance workload, not a recommended density for two rooms or a house. We have not established the minimum grid meeting the intended interior-quality limits. NVIDIA notes that high counts are usually unnecessary with properly configured wall geometry: [DDGIVolume reference](https://github.com/NVIDIAGameWorks/RTXGI-DDGI/blob/main/docs/DDGIVolume.md).

Native and browser allocate the same6,144 lattice positions. In the saved native reference, ray-based classification has3,200 ACTIVE and2,944 INACTIVE probes, including1,024 relocated active probes. Our different geometry-based eligibility mask retains5,120 active positions. These classifiers/relocation policies are not equivalent; the mismatch is preserved and documented rather than silently changing workload.

The native1024-history experiment means1,024 update frames, not1,024 total probes. Another1,024 figure is the original raster baker's default integration sample count per texel. Probe count, active count, relocated count, rays/samples and update frames must be labeled separately. [Native field manifest](../artifacts/native/rtxgi-ddgi/parity/variability-disabled-legacy-codec-camera0-frame1024/field-manifest.json), [history explanation](../artifacts/native/rtxgi-ddgi/parity/HISTORY_AND_FIRST_BOUNCE.md).

The G4 full workload traces5,120×512×3=7,864,320 primary rays over a complete bake, plus light visibility rays. Reducing active probes reduces this work, but a production density choice needs a quality/cost comparison after the current foundation gate. No such choice is implied by passing the fixed workload.
