# WebGPU and three.js foundation audit

Research snapshot: 2026-10-27. This is a source audit and proposed experiment sequence. No renderer was run, cloned, benchmarked, or changed for this memo. No applicable AGENTS.md was found in the workspace or its parent path.

## What r186 actually supplies

The upstream annotated r186 tag resolves to commit `9b4a2ac29c63ccb43fd51c5661f2f873ac2c39b8`, tagged 2026-09-24. This was checked through GitHub's tag and ref APIs, rather than inferred from current documentation. The release includes a WebGPU LightProbeGrid, incremental probe baking, and VXGINode voxel cone tracing. These are valuable comparators for the proposed system. [Release](https://github.com/mrdoob/three.js/releases/tag/r186), [tag object](https://api.github.com/repos/mrdoob/three.js/git/tags/48dd7be01263658aad497fd72e09d1e4c6213461).

### LightProbeGrid: capture and SH integration baseline

The inspected r186 implementation renders cubemaps at probe positions, projects them to nine L2 spherical-harmonic coefficients, and packs them into seven RGBA sub-volumes in one 3D texture. Baking stays on the GPU. Additional passes capture indirect bounces using a previous-pass snapshot. Incremental updates expose `start`, `count`, and `pass`, allowing a bounded number of probes per frame. These facilities provide an immediate baseline for capture cost, probe layout, atlas handling, and material integration. [Pinned capture source](https://github.com/mrdoob/three.js/blob/9b4a2ac29c63ccb43fd51c5661f2f873ac2c39b8/examples/jsm/lighting/LightProbeGrid.js), [incremental-bake PR](https://github.com/mrdoob/three.js/pull/34486).

The sampler offsets the shading position along the surface normal by half the probe spacing, then uses hardware trilinear interpolation of the SH atlas. It does not store or test directional depth moments, perform DDGI visibility-weighted interpolation, relocate probes, or classify probes in the inspected sampler. Atlas padding prevents interpolation between packed SH sub-volumes; that padding does not stop lighting from crossing scene walls. This is a baked/progressively rebaked irradiance grid, not evidence that three.js ships the desired visibility-aware DDGI system. [Pinned sampler](https://github.com/mrdoob/three.js/blob/9b4a2ac29c63ccb43fd51c5661f2f873ac2c39b8/examples/jsm/tsl/lighting/LightProbeGridNode.js).

The integration is especially useful: a custom Light is associated with an AnalyticLightNode via the renderer's light library, and its node adds irradiance to the lighting context. This is a plausible first adapter seam for the new GI system, subject to a small integration test. [WebGPU integration PR](https://github.com/mrdoob/three.js/pull/33913).

### VXGI: an existing alternative to measure

r186 also includes voxel cone tracing based on a dense voxel volume. It can update lighting without a full probe rebake, while geometry changes require re-voxelization. The API exposes directional radiance filtering, cone controls, bounce settings, and temporal filtering. The docs explicitly identify resolution, memory, leaking, and re-voxelization tradeoffs; this should be tested in the same interiors rather than treated as a presumed solution. It requires the actual WebGPU backend. [Pinned VXGI docs](https://github.com/mrdoob/three.js/blob/9b4a2ac29c63ccb43fd51c5661f2f873ac2c39b8/docs/pages/VXGINode.html.md), [upstream PR](https://github.com/mrdoob/three.js/pull/34402).

## Platform facts affecting the plan

Hardware ray queries and acceleration structures must not be requirements of the browser baseline. On this research date the upstream acceleration-structure and WGSL ray-query proposals are open and unmerged. Native wgpu or custom-browser experiments must be recorded separately from portable browser WebGPU. [Acceleration proposal](https://github.com/gpuweb/gpuweb/pull/11054), [ray-query proposal](https://github.com/gpuweb/gpuweb/pull/11056).

Consequently, keep ray acquisition replaceable. A browser implementation can explore software triangle-BVH traversal, raster probe capture with distances, or other geometry representations without replacing probe storage and interpolation. DDGI's cache/update strategy should not become coupled to DXR/Vulkan ray-tracing calls.

Query device features and limits at runtime, and write them into results. Storage-binding size, workgroup limits, texture dimensions/formats, and binding pressure affect viable atlas and BVH layouts. GPU timestamp queries are optional; mark unavailable GPU timings as unavailable and retain separate CPU/whole-frame measurements. [WebGPU specification](https://www.w3.org/TR/webgpu/).

The r186 WebGPU backend accepts an application-provided GPUDevice. This enables investigation of a shared-device adapter, but passing a device alone does not prove that arbitrary external GPU textures can be wrapped and sampled through TSL without private APIs. Prove texture import, ownership, submission order, disposal, and loss recovery before committing to that route. [Pinned backend options](https://github.com/mrdoob/three.js/blob/9b4a2ac29c63ccb43fd51c5661f2f873ac2c39b8/src/renderers/webgpu/WebGPUBackend.js).

## Proposed architecture to validate

This is a design proposal, not an existing API commitment.

1. **Engine-neutral scene and experiment contract.** Explicit meters, handedness/up axis, world transforms, triangle winding, geometric and shading normals, material subset, point/spot light units and falloff, emissive convention, environment, and scene revision IDs. Initially support opaque Lambertian surfaces and deterministic cameras; expand material coverage after numeric parity.
2. **Transport provider.** Given probe origins and deterministic ray directions, return hit distance, hit flags/backface policy, normal/material ID, and incoming radiance. Separate hit acquisition from hit shading where practical. Implementations may use raster capture, software BVH, native Vulkan RT, or native CPU tracing.
3. **Portable GI core.** CPU configuration/scheduling plus canonical WGSL compute passes for projection/integration, temporal updates, directional distance data, relocation/classification experiments, and probe-state debugging. Core dependencies are GPUDevice, explicit resources, and documented dispatch order; they should not include three scene objects or TSL nodes.
4. **Sampling contract.** Versioned volume bounds, probe indexing/spacing, encoding, radiance-versus-irradiance semantics, directional visibility representation, texture layout, validity, and revision. A raw WGSL evaluator is the reference. A small TSL evaluator for three must agree numerically with it; if shader-source reuse is possible without private APIs, test it rather than presuppose it.
5. **three adapter first.** Extract supported scene/lights, schedule updates, bind result resources, and add indirect diffuse through the material lighting seam. Keep direct lighting, environment contribution, albedo, and the Lambertian `1/pi` factor explicit to prevent double counting.
6. **Vanilla WebGPU harness alongside it.** A minimal viewer/evaluator consumes the same fixtures and core outputs. Add Babylon/custom-engine adapters after this demonstrates that no three dependency escaped into the core.

Native and browser implementations should share fixture bytes and intermediate-output formats. Their shader source can differ. Compare identical stored ray samples or probe atlases before comparing independently traced final images; otherwise traversal, material, and GI differences become entangled.

## Small experiments and decision gates

| Experiment | Controlled question | Evidence required to move on |
| --- | --- | --- |
| W0: reproduce upstream | What do r186 LightProbeGrid and VXGI actually cost and fail on? | Pinned checkout/runtime revision; same one-room and sealed-two-room scenes; direct-only and GI-only HDR captures; actual WebGPU backend verified; no quality claims from screenshots alone. |
| W1: resource seam | Can one core output feed raw WebGPU and three safely? | A known probe field sampled on both paths; agreement within a declared numeric tolerance; no readback/upload in steady state; documented texture/device ownership; two independent instances and disposal tested. A small compatibility shim is acceptable if version-pinned and isolated. |
| W2: interpolation first | Is SH/irradiance sampling correct before live transport? | Constant environment gives the expected diffuse response; directional synthetic field, atlas-edge probes, corners, normals, and transformed scenes match CPU/reference samples. Include explicit irradiance/radiance and `pi` checks. |
| W3: visibility separately | Does directional distance weighting stop cross-wall blending? | Paired scenes with/without an opening and a wall-thickness sweep; compare identical radiance with visibility off/on; publish leak-region error and contact-darkening error. Do not hide leaks with ambient occlusion or tonemapping. |
| W4: browser transport | Which acquisition path fits the first interiors? | Raster capture and software BVH prototypes return independently checked distances on analytic scenes; measure build/update time, rays or probes per update, memory, and per-pass cost. Choose using measured budgets. |
| W5: dynamic update | How rapidly does the cache react to point/spot changes? | Scripted light toggle, camera move, and door move; settle frames/ms, overshoot, flicker, stale-lit region, and update-budget sweeps; compare with the same static reference. |
| W6: combined adapter | Does three integration preserve the core's result? | Same linear indirect buffers and fixture revision in raw WebGPU/three; direct/indirect layers checked independently; material and environment contributions compose once; no unexpected shader variants during supported uniform-only controls. |

Pick numeric tolerances and frame/memory budgets after W0/W1 establish device behavior. State targets as provisional until then. The central quality gate is sealed multi-room isolation with thin-wall and corner cases, while preserving light transfer through actual doorways. A fixed room-ID mask may be a useful ablation, but it must not silently become the general algorithm.

## Benchmark instrumentation from the compile-cost skill

Use the local `webgpu-tsl-compile` skill as operational guidance, and re-check each version-specific claim against the pinned renderer. Its case-study milliseconds and shader-size thresholds are not project budgets or measurements of this system.

- Record browser/OS/GPU/driver, backend, device limits/features, viewport, resolution, scene/config hashes, three revision, and cache state.
- Separate asset/BVH preparation, first preview, first complete GI image, convergence, steady-state CPU/GPU passes, and frame-gap percentiles. Capture upload and readback costs explicitly.
- Count unique shader modules, generated WGSL bytes/hashes, sync/async pipeline creations, and actual compiler events when available. API return duration alone is not evidence of compiler duration.
- Measure cold and warm separately with isolated profiles/origins; name which browser and driver caches remain. Use at least three alternating paired runs on the same machine. Schedule one GPU benchmark at a time across agents.
- Use asynchronous raster/compute preparation where supported. Keep shared TSL helpers laid out, control loop expansion, and inspect generated WGSL for duplication. Avoid deriving shader capacities or identifiers from fixture data or frame timing.
- Treat zero unexpected variants for supported uniform-only changes as a goal. Zero warm compiler events is a useful cache target, not proof that startup is fast.
- Verify pixel/intermediate-output parity after compilation optimizations, including a dark interior dominated by the point/spot lighting under test. Run no instrumentation that silently changes the viewed backend or accidentally measures a background tab.

## Outstanding audit work

- The separate [Homefig audit](homefig-audit.md) records the inspected branches and reuse candidates.
- No tested performance or API-interop result exists yet. W0/W1 are intended to resolve those unknowns.
- Revisit ray-query standardization during future planning reviews; portable baseline acceptance must remain independent of experimental support.
- Inspect r186 VXGI source in detail and run it before deciding how much code can be reused. Documentation and PR claims establish candidate capabilities, not local correctness.

Skill feedback: none. No empirical run was performed, so no compile-skill rules or timings were validated or contradicted.
