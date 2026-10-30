# Pluggable transport providers

Decision recorded 29 October 2026 from the user's direction: reuse established ray-tracing/BVH libraries before writing a custom acceleration structure or traversal. The DDGI field owns integration, history, classification, relocation and reconstruction; a provider supplies transport samples. Preserve the raster provider as a frozen baseline. Under the user’s 29 October direction, parity refinement is deferred; the active execution order and dependencies are in [PLAN.md](../PLAN.md).

## First candidates

- [three-mesh-bvh WebGPU API](https://github.com/gkjohnson/three-mesh-bvh/blob/master/WEBGPU_API.md): first arbitrary-ray intersection candidate. Its documented GPU data model includes TLAS/BLAS buffers, TSL traversal and world-space hit information. The API requires three.js r185+ and is marked unstable, so pin exact revisions and contain its types within the adapter.
- [three-gpu-pathtracer](https://github.com/gkjohnson/three-gpu-pathtracer/blob/main/README.md): first browser path-traced reference candidate, using its WebGPU renderer. Its camera-oriented rendering API needs an integration spike before it can be treated as an arbitrary probe-ray transport provider. The documented material/light restrictions must be mapped explicitly to our diffuse fixture.
- The user mentioned “three-mesh-gpu”; the verified repositories above are the initial candidates. Do not claim an exact package match for that name or install an assumed package.

These observations establish suitability for an experiment, not measured compatibility. No dependency has been added to the validated bench. Read-only audit has resolved candidate source revisions: three-mesh-bvh 26fc97c5f007b8bd905ec31df9deac5d9d668782 (source package0.9.17) and three-gpu-pathtracer 149c7f57b16e6a5a99fc9d182316a446d41ea6a2 (source package0.0.27). These are candidate pins, not an installed/tested dependency set; resolve transitive dependencies in the separate spike.

The [source-pinned library audit](transport-library-audit.md) identifies public BVHComputeData and its raycastFirstHit traversal as the first adapter boundary. A full path tracer’s internal ray kernel/hit buffers are not assumed to be public exports. The supported BVH interface is Three/TSL-specific; plain WebGPU needs a versioned packing/binding adapter and remains untested. Avoid deprecated raw WGSL helpers that upstream replaced for cross-platform pointer limitations.

## Boundary to prove

Use two composable capabilities rather than coupling the field to a full camera renderer:

| Capability | Inputs | Outputs |
| --- | --- | --- |
| Scene acceleration / intersection | Versioned scene snapshot; batched world-space ray origins, unit directions, tMin/tMax and sidedness policy | Hit/miss, physical distance, hit position, geometric normal, front/back flag, primitive/material IDs and barycentrics |
| Probe transport | Same rays plus explicit lighting/material snapshot, bounce policy, seed and previous immutable field generation | Linear directional radiance, physical hit distance, hit validity/backface flags and generation metadata |
| Reference rendering | Shared fixture, camera, light/material mapping, sample budget and seed | Independent scene-linear image, convergence/error evidence and renderer settings |

The interface may expose GPU-resident buffers or shader-callable functions as appropriate; do not require per-ray CPU readback. Specify GPU device/resource ownership, synchronization, cancellation, disposal and stable publication. A provider reports supported geometry, transparency, instancing, transforms/refit, light types and backface semantics. Missing capabilities fail explicitly or select a named fallback.

three.js objects and TSL helper types stay in the three.js provider. The core contract describes packed data and sample semantics so plain WebGPU, Babylon, a custom engine or a future vendor implementation can supply the same information. Backend-specific acceleration buffers need not share one binary layout. Hardware/vendor support remains an implementation capability, not an assumed WebGPU ray-tracing feature.

## Bounded G3 spike after G1 tests and G2 profiling

G3 does not start until the G2 cost baseline and optimization budget pass. Scope only this provider phase:

1. Pin three.js and three-mesh-bvh in an isolated dependency set. Keep the existing bench lockfile and frozen captures unchanged. The path-tracer source audit is retained for G6; it is not a G3 installation requirement.
2. Build/refit the exact two-room fixture with the public library API. Consume shared deterministic world-space ray batches and return GPU-resident hit/miss, physical distance, original face orientation, barycentrics and stable primitive/material identifiers.
3. Compare rays to analytic geometry and the existing independent triangle oracle: sealed wall, doorway, grazing exclusions, backface, max distance, nonuniform/mirrored transforms and instance IDs. Distance tolerance is absolute1e-4 plus relative1e-5 on nondegenerate cases; G1 suites must remain green.
4. Verify transform refit, topology rebuild, independent provider instances, capacity changes and disposal. A fixed ray capacity batches work; increasing probes within a batch must not introduce a JS dispatch per probe/ray. Readback is only for validation.
5. Compare counts/cost with the G2 baseline under its declared workload and cache state; record shader/backend/resource evidence and keep output correctness separate from performance.

Acceptance is the complete G3 contract in [PLAN.md](../PLAN.md). Do not replace field integration, add recursive shading or build a reference path tracer while G3 is incomplete. G4 later consumes these GPU samples and proves the DDGI update/portable-buffer contract. G6 later calibrates a path-traced reference with explicit material/light/clamp settings and convergence evidence. A custom production BVH requires a measured unmet library requirement and a separate scope decision.

## Relation to deferred parity refinement

The source-frozen native/browser diagnosis batch is complete. The acquisition residual is preserved as a later G6 investigation; it does not authorize more parity sweeps now. G1 tests must close, then G2 measures cost and sets budgets, then this G3 provider spike begins. The full path-traced reference adapter waits behind the GPU foundation/interior-suite gates. Trace semantics learned from diagnosis become contract tests without requiring the old percentage difference to be eliminated first.
