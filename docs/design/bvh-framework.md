# BVH framework (proposal, 2026-11-04)

Status: **package and H-PLOC target decided by Dennis 2026-11-04** (see "Decided" below). The milestone fit ("How the BVH track fits the milestones") is a recommendation waiting for his answer. It is built from three research notes:
- [research/bvh-gpu-build.md](../../research/bvh-gpu-build.md) (algorithms);
- [research/bvh-libraries.md](../../research/bvh-libraries.md) (code we can port);
- [research/webgpu-compute-for-bvh.md](../../research/webgpu-compute-for-bvh.md) (what WebGPU allows).

It extends [trace.md](trace.md), which covers the T0.10 spike. Decisions it relies on are in [DECISIONS](../DECISIONS.md) (2026-11-04):
- BVH work on the GPU where possible;
- subgroups first, with one declared fallback;
- three.js is the only adapter until v1.

## What we're building

An engine-free BVH module inside the platform, with its own tests and benchmark. Builders are interchangeable. A new method from a paper becomes one more builder, run against the same scenes, oracle and benchmark. DDGI is its first user. Picking, path tracing and other GI techniques can use it later.

```
            ┌─────────────── Builder contract ───────────────┐
 triangles ─▶ CPU: tinybvh (WASM, worker)  ─┐                 │
 (arrays or  │ three adapter: three-mesh-bvh ├─▶ our GPU layout ─▶ WGSL tracer (giTraceClosest / giTraceAny)
  GPU buffer)│ GPU: LBVH → PLOC → H-PLOC  ─┘   (nodes, tris)   │
            └────────────────────────────────────────────────┘
 every frame:  BLAS refit (deforming) · TLAS rebuild/refit (moving instances)  — all on the GPU
```

**Builder contract**:
- **In:** positions and indices, as CPU arrays or a GPU buffer, plus options such as leaf size and quality.
- **Out:** GPU buffers in our layout (nodes, tree-ordered triangles, parent indices for refit) and stats (build ms, SAH cost, depth, bytes).
- **Rule:** every builder must pass the same ray oracle.

## What changes from the T0.10 design, and why

| Part | Today (textbook) | Proposed | Evidence |
| --- | --- | --- | --- |
| Mesh build | Our binned SAH on the CPU, 28 s for Sponza | **Now:** tinybvh compiled to WASM in a worker, as the CPU path and the high-quality "bake". **Then:** a GPU build, LBVH first, then PLOC or H-PLOC | H-PLOC builds Sponza in 0.74 ms (7900 XT) and 0.44 ms (RTX 4080 SUPER), with trees within 9% of the best builder. LBVH trees are 33% worse. |
| Top level | CPU rebuild, 1–10 ms per move, 86–121 ms for 10k instances | GPU LBVH or PLOC every frame when instances move | H-PLOC: 1M instances in 2.2 ms. Our estimate for 10k on the 3060 is under 0.2 ms. |
| Deforming meshes | Planned GPU refit (M3) | Same refit, done bottom-up with a parent counter, so no float atomics. Rebuild on the GPU when SAH cost drifts past a limit. | Standard practice in production engines |
| Node layout | 64-byte two-child (Aila–Laine) | **Keep.** tinybvh's `BVH_GPU` is byte-for-byte ours. CWBVH (compressed 8-wide) gets its own spike later. | CWBVH: 1.9–2.1× faster, but those numbers use CUDA-specific instructions and an old baseline. |
| Sort and scan | None yet | Reduce-then-scan radix sort, subgroup version first. One-sweep decoupled look-back only as an optional fast path behind a self-test. | WebGPU doesn't guarantee forward progress. Gridwise's own docs say look-back can fail on Apple and ARM. |
| Trace | Stack walk, watertight test | Part C fixes as planned (row vectors, any-hit early exit). Cross-check our walk against tinybvh's `traverse_bvh2.cl`. | Watertight costs 45% on NVIDIA versus 20% on Intel, which points at how the code is written. |

## Code we port (all MIT or Apache-2.0)

1. **tinybvh** (MIT, active): the WASM CPU builder, `BVH_GPU` layout, traversal reference, and `BuildHPLOC` as the CPU oracle for our GPU H-PLOC. NexusBVH reports identical trees from it.
   - Without cross-origin isolation, WASM threads need SharedArrayBuffer, so the drop-in default is single-threaded. Measure that.
2. **HIPRT** (MIT): `LbvhBuilderKernels.h` (186 lines), `PlocBuilderKernels.h` (311 lines) and `MortonCode.h`, ported to WGSL with subgroup operations. (B2 ported tinybvh's `BuildHPLOC` instead of the PLOC kernels: H-PLOC needs no separate PLOC builder.)
   - **Port rule:** never assume a 32- or 64-wide warp; read `subgroup_size`.
3. **NexusBVH** (Apache-2.0, CUDA): the H-PLOC reference. It is heavy on warp ballots.
   - **Morton codes:** WGSL has no u64, so its 64-bit codes would be emulated as `vec2<u32>`. That costs twice the radix-sort passes (8 instead of 4).
   - **Default:** 30-bit codes over each mesh's own bounds, tie-broken by index.
   - **Option:** emulated 64-bit codes as a builder setting, measured on Bistro exterior (SAH cost and build ms). Turn it on only where it clearly pays.
   - **Atomics:** 64-bit atomics can't be emulated this way. None of these builders need them.
4. **Gridwise** (Apache-2.0, WGSL): sort and scan to vendor or depend on, minus the look-back default.
5. **three-mesh-bvh** (MIT): in the three adapter only, a converter of about 60 lines into our layout. This is part C's experiment.
6. **Not copied:** repos with no license (Meister's PLOC and PRBVH, AlanIWBFT/CWBVH, kishimisu's sort). Their papers are fine to build from.

## Stages, each with a done-when

| Stage | What | Done when (RTX 3060 Laptop, Chrome; Intel UHD must also pass, with no speed bar) | Where |
| --- | --- | --- | --- |
| **B0** | tinybvh WASM builder in a worker, behind the Builder contract (SIMD Bonsai builder, emsdk 6.0.13 pinned). Our CPU builder is deleted once the checks pass. Status 2026-11-04: Sponza 2.2-2.7 s; B0b moves the layout conversion into the WASM. | Sponza builds in ≤ 2 s single-threaded, with the time recorded. The oracle passes. Trace speed on tinybvh trees is ≥ our current trees. | T0.10 part C, with the three-mesh-bvh experiment |
| **B1** | GPU LBVH, subgroups first: Morton codes, sort, Apetrei build, SAH leaf collapse. GPU top-level rebuild every frame, and BLAS refit. Status 2026-11-05: in review. The top level is B1a's GPU refit. Mesh trees are built in batches (every mesh with the same dispatches), Morton codes use cubic cells (stretched cells made flat meshes 2.4-2.6x worse), the climb runs in two rounds; `packages/bvh/README.md` has the design and PROGRESS (B1) the numbers. | Sponza GPU build ≤ 50 ms. 10k-instance top level ≤ 0.5 ms of GPU time. The oracle passes on both GPUs. No subgroups: the BVH falls back to B0, and a test covers that. | M1 |
| **B2** | GPU PLOC, then H-PLOC. Status 2026-11-05: in review. H-PLOC only: its PLOC merge is the PLOC step, so no separate PLOC builder was built. tinybvh's `BuildHPLOC` ported (with the WASM's own `BuildHPLOC` as the oracle), the merge done by a subgroup with cluster slots for any subgroup size, exact ties paired up instead of chaining the tree, nodes numbered without a counter, B1's cubic 30-bit codes by default; `packages/bvh/README.md` has the design and PROGRESS (B2) the numbers. | H-PLOC trees match tinybvh `BuildHPLOC` (same SAH cost ±1%). SAH cost ≤ 1.15× tinybvh's best CPU build. Sponza build ≤ 50 ms. Trace ≥ 0.9× of B0 trees. | M1 or M2, by speed need |
| **B3** | CWBVH spike: compressed 8-wide nodes and traversal, with tinybvh's builder and kernels as reference | Adopt if closest-hit Mrays/s ≥ 1.3× BVH2 on Sponza on both GPUs. Otherwise record the result and drop it. | After B2 |
| **Ongoing** | Paper watch: a new method is a new builder or layout behind the contract | Runs the same oracle and `bench/results/trace.csv` rows | Any time |

The speed limits above are planning numbers. Like any threshold, changing one is a DECISIONS entry.

## Decided (Dennis, 2026-11-04)

1. **Its own package, in this repo for now.**
   - `packages/bvh`, published later as `@pmndrs/bvh`, probably under pmndrs/bvh.
   - It is engine-free, has no DDGI or GI dependencies, and ddgi-core depends on it.
   - It ships through `/ddgi` until it's split out.
   - Dennis: "if its useful enough I might even say to make it pmndrs/bvh... I think that will be the plan, but lets keep it in the repo for now for simplicity sake."
2. **The target builder is H-PLOC.** LBVH (B1) is the stepping stone, because H-PLOC is built from the same pieces: Morton codes, sort, and the Apetrei bottom-up pass.

## How the BVH track fits the milestones (recommendation)

The BVH runs as its **own lane, alongside the DDGI milestones**. It isn't a gate in front of them. Two reasons:

- **Build speed and trace speed are different levers.**
  - B1 and B2 make builds fast. That matters for startup, deforming meshes and moving instances.
  - A good CPU SAH tree (B0, tinybvh) traces as fast as, or faster than, a GPU-built one. GPU builders aim to *approach* SAH quality.
  - So for M1's static scenes, B0 trees are already the fastest trees to trace. Waiting for H-PLOC wouldn't make M1's DDGI faster.
- **The DDGI loop only talks to the tracer through `giTraceClosest` and `giTraceAny`.** Swapping builders or layouts later changes no DDGI code. M1's numbers get re-run on each new BVH stage, and `trace.csv` and the DDGI benchmarks record which stage was used.

Where the BVH does gate:

| BVH stage | Needed by | Why |
| --- | --- | --- |
| B0 | M1 start | Sponza and Bistro must load in seconds, not 28 s |
| B1 GPU top level and refit | M3 (moving objects), with the GPU top-level refit in M1 | Dragging furniture and the animated tree |
| B2 H-PLOC | M3 sign-off, and the ≤ 2 s startup target at M5 | Fast rebuilds for deforming meshes and large scenes |
| B3 CWBVH | The M2 or M3 performance review | The only stage aimed at *trace* speed. If DDGI's update budget is limited by rays per second, B3 is the lever. |

If M1 shows that tracing is the bottleneck, B3 moves forward.
