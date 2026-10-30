# DDGI research landscape and experiment candidates

Initial source review: 2026-10-27. This is a planning memo, not an exhaustive state-of-the-art survey or a claim that any implementation has been reproduced locally. Sources below are papers, author presentations, official engine documentation, or implementation repositories. Performance claims from their machines are not project targets.

## Recommended foundation

Use the 2019 DDGI algorithm, the 2021 production extensions, and a pinned RTXGI-DDGI implementation as the first reference family. Keep the probe representation, visibility reconstruction, ray provider, hit shading, and engine integration separable. This is a project recommendation: it favors a testable foundation over choosing a newer cache before establishing a baseline.

DDGI is an irradiance/visibility caching algorithm; hardware ray tracing is an implementation choice for producing probe samples, not an intrinsic requirement of its interpolation equations. The cited SDK uses native ray tracing. A WebGPU implementation can supply samples through compute-based software triangle/BVH traversal, or investigate SDF traversal later. Such substitutions require independent correctness and cost validation; the native SDK's performance does not transfer automatically. Precomputed probe data is also a useful diagnostic input because it isolates reconstruction from tracing. NVIDIA explicitly documents loading precomputed probes for platforms without GPU ray tracing. [SDK algorithm and limitations](https://github.com/NVIDIAGameWorks/RTXGI-DDGI/blob/main/docs/Algorithms.md)

## Reading sequence and reproducible baseline

| Work | Verified contribution / reason to read | Planning use |
| --- | --- | --- |
| [McGuire et al., *Real-Time Global Illumination using Precomputed Light Field Probes*, 2017](https://mail.casual-effects.com/research/McGuire2017LightField/index.html) | Visibility-aware probe data and world-space light-field queries; author page provides paper, slides, and reference implementation. | Background for why plain irradiance interpolation leaks; optional precomputed reconstruction fixture. |
| [Majercik et al., *Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields*, JCGT 2019](https://jcgt.org/published/0008/02/01/) | Dynamic irradiance fields with moment-based visibility-aware interpolation and reference GLSL. | First algorithm specification: directional irradiance and distance moments, temporal updates, reconstruction, diffuse transport. |
| [Majercik et al., *Scaling Probe-Based Real-Time Dynamic Global Illumination for Production*, JCGT 2021](https://jcgt.org/published/0010/02/01/) | Bias controls, faster transitions, probe state management, cascaded volumes, and other production extensions. | Establish controlled ablations for baseline versus relocation/classification/temporal improvements; postpone large-world cascades. |
| [NVIDIA RTXGI-DDGI repository](https://github.com/NVIDIAGameWorks/RTXGI-DDGI) | SDK, sample application, documentation, and native integration paths. | Candidate first executable reference. Pin commit, configuration, shaders, and assets before measuring. No local build verified in this memo. |

Important naming distinction: the current [RTXGI repository](https://github.com/NVIDIA-RTX/RTXGI) presents **Neural Radiance Cache (NRC)** and **Spatial Hash Radiance Cache (SHaRC)**. It links DDGI v1.x to the separate RTXGI-DDGI repository. A checkout called RTXGI is therefore not sufficient evidence that a benchmark is running DDGI.

## Follow-ups worth investigating after baseline parity

| Family / primary source | Verified status | Question for this project |
| --- | --- | --- |
| [Dynamic Diffuse Global Illumination Resampling, 2021](https://arxiv.org/abs/2108.05263) | Author paper combining DDGI proxies with spatiotemporal resampling ideas. | Can better sampling reduce probe update cost without increasing leakage or response latency? This is more machinery than the first baseline needs. |
| [Importance-Based Ray Strategies for DDGI, 2023](https://doi.org/10.1145/3585500) | ACM abstract describes importance-based allocation, MIS, and temporal reuse. Full text was not fetched successfully during this pass; code reproducibility remains unverified. | Test against equal total ray budgets, including scheduling and PDF bookkeeping cost. Do not adopt published speedups as expected WebGPU results. |
| [Adaptive Dynamic Global Illumination, 2023](https://arxiv.org/abs/2301.05125) | Author preprint describes allocating samples to changes in lighting and geometry. | Static interior workloads may benefit from sparse updates, but cold rooms and newly opened doors need a starvation/latency test. |
| [EA SEED GIBS, 2021](https://www.ea.com/seed/news/siggraph21-global-illumination-surfels) | Official presentation describes surfel lighting caches, hardware tracing, and dynamic content. | Compare surface-attached caches with probes in empty space. Account for spawn, lookup, coverage, and maintenance cost. |
| [Shipping Dynamic Global Illumination in Frostbite, 2024](https://advances.realtimerendering.com/s2024/content/EA-GIBS2/Apers_Advances-s2024_Shipping-Dynamic-GI.pdf) | Production update: GIBS retains both surfels and probes; probe interpolation moved to a DDGI-like octahedral representation with depth attenuation. | Surfels and probes can complement one another. Study visibility detail versus irradiance resolution and update prioritization; do not assume replacing probes is the only direction. No public Frostbite implementation verified. |
| [Crassin et al., Voxel Cone Tracing, 2011](https://www.icare3d.org/research-cat/publications/interactive-indirect-illumination-using-voxel-cone-tracing.html) | Foundational voxel cone tracing paper and author materials. | Establish a deliberately different geometry/transport approximation for the same thin-wall scenes. |
| [Godot VoxelGI documentation](https://docs.godotengine.org/en/latest/tutorials/3d/global_illumination/using_voxel_gi.html) | Official docs explain tuning and leakage with thin/unsealed geometry. | Useful native comparator; voxel resolution and geometry preprocessing must be recorded. Thickening walls is a content change, not proof that the original scene passed. Pin a Godot version rather than relying on changing `latest` docs. |
| [Godot SDFGI documentation](https://docs.godotengine.org/en/stable/tutorials/3d/global_illumination/using_sdfgi.html) | Separate system from VoxelGI: camera-following cascades; dynamic lights supported, dynamic occluders/emissive surfaces unsupported in the documented implementation. | Compare static interior lighting first. Dynamic-door occlusion is a capability limitation to label, not a like-for-like pass/fail test. Occlusion rays and feedback controls offer useful ablations. |
| [AMD GI-1.0 technical report](https://gpuopen.com/download/GPUOpen2022_GI1_0.pdf) and [GI-1.1 update, 2024](https://gpuopen.com/learn/gi-1-1-glossy-reflection-rendering/) | Two-level screen/world radiance caching; GI-1.1 adds glossy reflections. | Screen/world hybrid comparator with different visibility and disocclusion tradeoffs. Keep glossy quality outside initial diffuse scoring. |
| [AMD GI-1.2, November 2025](https://gpuopen.com/learn/gi-1-2-multibounce-indirect-rendering/) and [Capsaicin v1.3](https://gpuopen.com/capsaicin/) | GI-1.2 extends hash-grid transport for additional bounce energy; official framework includes a reference path tracer. Capsaicin is Direct3D 12. | Strong native research candidate for multi-room energy propagation; it is not a Vulkan sample. Audit bounce semantics before comparing images. |
| [AMD Brixelizer/GI](https://gpuopen.com/fidelityfx-brixelizer/) | Compute-based dynamic GI using sparse distance fields; official page lists Vulkan and DX12. Simplified GI-1.0-derived approach. | Relevant software-traversal alternative. Evaluate sparse SDF construction, thin-wall fidelity, and total update cost before considering a WebGPU port. |
| [NVIDIA SHaRC library](https://github.com/NVIDIA-RTX/SHARC) | Shader-source world-space radiance cache, with examples in RTXGI. | Later native cache comparator; cache resolution, bias, update cost, and path termination policy need separate measurement. |

## Current-research scan through October 2026

These are verified leads, not replacements for the reproduction baseline:

- [EA ORCA, SIGGRAPH 2026 Advances](https://advances.realtimerendering.com/s2026/index.html): the official abstract describes an instantaneous radiance cache with no temporal history; the course page links presentation materials. Worth reading for responsiveness and fully dynamic scenes. Implementation availability and browser suitability have not been established.
- [Split Radiance Cascades, Freeman, Sannikov, Pitt-Francis, 2026 preprint v2 dated October 11](https://arxiv.org/abs/2607.20384): sparse world-space probe hashing plus ray splitting; author abstract reports both single-frame and temporal variants. A relevant later experiment for spatial detail. Peer-reviewed status, implementation maturity, and WebGPU feasibility were not established in this pass.
- [NVIDIA, *A Generalizable Light Transport 3D Embedding for Global Illumination*, SIGGRAPH 2026](https://research.nvidia.com/labs/rtr/publication/xu2026generalizable/): learned 3D scene embedding demonstrated on diffuse indoor illumination. Distinct training/inference architecture; keep as a research watchlist item, not a dependency for a transparent DDGI foundation.
- [AMD, *Lightweight attention-based indirect illumination*, August 2026](https://gpuopen.com/learn/lightweight-attention-based-indirect-illumination/): official research summary uses reflective-shadow-map inputs alongside G-buffers for learned indirect illumination. Potentially relevant to few-light interiors, but trained-model availability and WebGPU costs remain unverified.

Search also surfaced Mobile-DDGI (I3D Companion 2026, DOI `10.1145/3807895.3807920`). Its publisher page could not be accessed and only a secondary bibliographic listing was found; keep it **unverified** until the primary paper and any implementation are inspected. This scan is enough to seed research assignments, not to declare a single universal 2026 GI baseline.

## Interior failure modes to make measurable

The following are proposed engineering tests, inferred from the representations and documented limitations rather than guaranteed fixes from the papers:

1. **Separate intersection leaks from reconstruction leaks.** Compare traced hit distance/normal/primitive ID against a trusted triangle oracle before evaluating probe interpolation. Use identical ray inputs and dump ray buffers. Perfect triangle hits do not by themselves prove the reconstructed irradiance is correct.
2. **Use two sealed rooms before a beautiful scene.** One room illuminated, adjacent room dark, initially black environment and no connecting aperture. Sweep wall thickness, probe spacing, grid offset, surface bias, depth-map resolution, and distance variance. Report indirect-only linear-HDR leakage at fixed receiver masks. Also score dark bias: turning GI off must not qualify as a leak fix.
3. **Add a doorway and then a moving door.** A real aperture changes the correct transport. Compare door-closed, open, and transition references; measure settling time, stale history, and occupancy changes separately. Label methods without dynamic occlusion support.
4. **Guard physical energy accounting.** Match point/spot definitions, attenuation, shadowing, material reflectance, emission, units, environment, and bounce interpretation. Do not count the same direct term in both final shading and GI. Freeze tone mapping/exposure for display and score the scene-linear buffers.
5. **Check temporal feedback deliberately.** Compare one-bounce mode, recursive feedback, and high-sample reference. Use plausible diffuse albedos; log overshoot, convergence, and light-off decay. A brighter image is not evidence of more accurate multibounce transport.
6. **Measure irradiance and visibility budgets independently.** Depth moments may need more angular resolution than smooth diffuse irradiance. Vary them independently before spending more rays everywhere.
7. **Compare equal quality and equal budget.** Record total GI time, traversal, hit shading/shadows, cache update, lookup/composite, memory, and response latency. Include cold shader setup separately from steady-state GPU timing. Native hardware tracing versus browser software traversal is a system comparison; a shared software traversal path is needed to isolate API/runtime overhead.

A useful success gate is “no visible leakage on an agreed fixture suite within stated numeric thresholds and geometry assumptions,” not an unrestricted promise of zero leaks for all scenes. Numeric tolerances should be chosen after establishing reference noise and light/scene scale, then frozen before tuning candidates.

## Questions the next research/build phase must resolve

- Which pinned RTXGI-DDGI revision builds and captures repeatably on available hardware? Which API path is simplest locally?
- Which raw buffers can be extracted without rewriting the whole sample? Can probe data be replayed into WebGPU unchanged?
- What is the minimal WebGPU triangle/BVH traversal and hit-shading contract for static interiors? How are point/spot shadows evaluated at probe hits?
- How do relocation/classification interact with thin walls, sparse room coverage, probes inside geometry, and volume boundaries?
- Is one common neutral scene manifest sufficient to match materials, lights, cameras, animation, and metric masks across native and browser tools?
- Which techniques have suitable code and asset licenses for local comparison, redistribution, and derivative implementations? An available paper or public repository does not settle all three.
- What are target browser/GPU tiers and acceptable memory, startup, GI frame time, and lighting-change response budgets? Keep these configurable until the first measurements.

Source confidence: high for the baseline lineage, official implementation identity, documented Godot capability limits, and existence of the listed papers/presentations. Medium for practical port suitability until source/build audits. No local renderer correctness or performance results are claimed here.
