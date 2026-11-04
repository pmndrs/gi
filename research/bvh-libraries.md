# BVH libraries and implementations: survey for webgpu-ddgi

Date: 2026-11-04. Research only. Method: cloned and read tinybvh, three-mesh-bvh (v0.9.18), NexusBVH, HIPRT and Gridwise sources; queried GitHub for license and last-push dates; read READMEs for the rest. Items I did not open the code for are marked "unverified".

Our target: BVH2, 64-byte two-child nodes (Aila-Laine), watertight triangles, top-level over instances, WGSL tracing. The CPU binned-SAH build takes 28 s for Sponza (262k triangles) and must move off the main path.

## Best candidates to port (ranking)

| # | Candidate | What we take | Source language | License | Port effort | Why |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **tinybvh** (`BVH_GPU` layout, `kernels/traverse_bvh2.cl`, `BVH::Build` + Emscripten) | Node layout (identical to ours), traversal kernel to compare against our WGSL; the builder itself compiled to WASM and run in a worker | C++14 header-only; OpenCL C kernels | MIT | **S** for layout and traversal; **M** for a WASM build with worker plumbing | The `BVH_GPU` node is byte-for-byte our node: `lmin,left / lmax,right / rmin,triCount / rmax,firstTri`. Fastest CPU SAH builder in its own comparison with Embree and madmann91 (README graph, no numbers in text). Built-in Emscripten support. Used for WebGPU by EA's Gigi and for WebGL by Wonderland Engine. Fixes the 28 s problem without writing a GPU builder. |
| 2 | **HIPRT builders** (`PlocBuilderKernels.h`, `LbvhBuilderKernels.h`, `MortonCode.h`) | GPU LBVH and PLOC kernels, the ones AMD ships in production | HIP C++ (CUDA-like) | MIT (AMD `license.txt`) | **M** for LBVH (186 lines); **M-L** for PLOC (311 lines) plus sort and scan | Permissive license, small, readable, written for a GPU that has 32 or 64 lane warps. The "second arrival continues upward" pattern uses `atomicExch` on a per-node counter, so no spin wait. |
| 3 | **NexusBVH** (`src/Cuda/BinaryBuilder.cu`, `BuilderUtils.h`, `WideConverter.cu`) | H-PLOC: best quality-per-ms GPU BVH2 builder in the literature; official-style reference | CUDA C++17 | Apache-2.0 | **L** | Sponza (0.3M triangles) builds in 0.44 ms total on an RTX 4080 SUPER (0.20 ms for the BVH2 kernel). The README says tinybvh's `useHPLOC` produces identical trees, which gives us a CPU oracle to test a WGSL port against. Heavy on warp ballot and shuffle, so it is a second-phase item. |
| 4 | **Gridwise** (+ `wgpu_sort` as backup) | WGSL OneSweep radix sort and decoupled-lookback scan, needed for Morton ordering in options 2 and 3 | WGSL in JS template literals | Apache-2.0 (Gridwise); BSD-2 (wgpu_sort, per GitHub) | **S** to depend on it; **M** to vendor | Already WGSL, already published to npm. Caveat: forward progress (see below). |
| 5 | **obvhs / madmann91 reinsertion optimizers** (`obvhs/src/bvh2/reinsertion.rs`, `madmann91/src/bvh/v2/reinsertion_optimizer.h`) | Algorithm reference for a post-pass that improves LBVH/PLOC tree quality (parallel reinsertion) | Rust; C++ | Apache-2.0 or MIT (obvhs dual); MIT | **M** CPU, **L** GPU | Only needed if the GPU-built tree traces worse than binned SAH. Lowest priority. |

Not recommended for copying: Meister's original PLOC and PRBVH repos and AlanIWBFT/CWBVH (no license on GitHub, so treat as all rights reserved); CWBVH in general (80-byte 8-wide nodes, heavy bit tricks, a large retrace of our traversal).

## Recommended path in one paragraph

Short term, keep the node layout and replace the build: compile tinybvh to WASM and run it in a worker (or, in the three adapter, use three-mesh-bvh; see the conversion section). Medium term, port a GPU build: Morton codes, Gridwise sort, then HIPRT LBVH, then HIPRT PLOC. Long term, port H-PLOC from NexusBVH if PLOC's single-warp window is not enough.

## Details per candidate

### 1. tinybvh (Jacco Bikker)
- Repo: https://github.com/jbikker/tinybvh. License MIT. Last push 2026-11-02, version 1.9.1. Active.
- Core files: `tiny_bvh_base.h` (builders, layouts, `BVH_GPU::BVHNode` at about line 1156, `BuildHPLOC` at about line 7352), `tiny_bvh_x86_float.h` and `tiny_bvh_arm_float.h` (SIMD builders), `kernels/traverse_bvh2.cl` (244 lines), `traverse_cwbvh.cl` (581), `traverse_tlas.cl` (190), `wavefront.cl`.
- Builders: Bonsai (default), Bonsai HQ and SBVH (spatial splits), binned SAH (SIMD), full sweep, H-PLOC, midpoint (`BuildQuick`). Layouts: `BVH` (32-byte), `BVH_Verbose`, `MBVH<M>`, `BVH_GPU` (64-byte Aila-Laine), `BVH4_GPU`, `BVH4_CPU`, `BVH8_CWBVH`, `BVH4_AMD_HW`. TLAS and instancing are included.
- Speed: README graphs only (CPU build faster than Embree and madmann91; up to 5 billion rays/s on an RTX 5080 laptop in Sponza with the binary format). I found no Sponza build milliseconds in text; **measure it ourselves**. Threaded by default via a `std::thread` pool; `NO_THREADED_BUILDS` drops threading (C++14).
- Use: (a) WASM builder in a worker, output `BVH_GPU` nodes straight into our buffer after a small triangle-gather step; (b) copy `traverse_bvh2.cl` logic as a cross-check of our WGSL walk; (c) `BuildHPLOC` as the CPU oracle for a GPU port.
- Doesn't map / caveats: WASM threads need SharedArrayBuffer (cross-origin isolation) or fall back to a single thread; the pool is `std::thread`, so Emscripten needs `-pthread` and a pool size. Leaf size and triangle order differ from ours, so we convert. Its traversal uses `triCount > 0` for leaf, whereas ours stores child counts in the parent; trivial. Whether a WASM build of `tiny_bvh.h` is already published to npm is unverified (the README only says "WASM / Emscripten: supported").
- Port effort: S (layout, traversal), M (WASM build).

### 2. HIPRT (AMD)
- Repo: https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT. License: MIT text in `license.txt` (GitHub reports NOASSERTION because the file is named `license.txt`). Last push 2026-10-21. Active.
- Core files: `hiprt/impl/PlocBuilderKernels.h` (311 lines), `LbvhBuilderKernels.h` (186), `MortonCode.h` (286), `RadixSort.cpp` (73; uses Orochi, MIT), `BvhBuilderKernels.h` (1,931; shared setup, bounds, wide conversion), `SbvhBuilder*` (spatial splits). Also `BatchBuilder*` for many small meshes at once, which matches our many-mesh case.
- Language: HIP C++ (CUDA-like, one codebase for AMD and NVIDIA).
- Builders: LBVH, PLOC, SBVH, batch builder, plus an Embree-based offline path in `contrib/embree`.
- Doesn't map to WGSL:
  - PLOC works per warp with `hiprt::ballot` (returns a 64-bit mask), `popcll`, `sync_warp` and shared-memory staging of `WarpSize` clusters. WGSL has `subgroupBallot` (a `vec4<u32>`, 128 bits) and `subgroupShuffle`, but subgroup size is 4 to 128 and not known at compile time; the kernel assumes 32 or 64. Either request a fixed size (`@subgroup_size` is not in WGSL yet) or emulate the warp with workgroup memory and barriers.
  - 64-bit atomics: none seen in the PLOC kernel; 64-bit ballot arithmetic becomes `vec4<u32>`.
  - No recursion, no pointers into storage buffers across functions (a WGSL limit), and `atomicExch`/`atomicMin` are fine (WGSL has `atomicExchange`, `atomicMin` on `u32`).
  - Forward progress: the parent-walk uses `atomicExch(updateCounters)`, and the loop `while ((k = atomicExch(...)) != InvalidValue)` ends after at most one extra trip. It is not a spin wait on another thread.
- Port effort: LBVH M, PLOC M-L, SBVH L (skip).

### 3. NexusBVH (H-PLOC)
- Repo: https://github.com/StokastX/NexusBVH. License Apache-2.0. Last push 2026-09-11. 36 stars; started as a university course project but cited as matching the paper.
- Core files: `NexusBVH/src/Cuda/BinaryBuilder.cu` (279 lines, the H-PLOC kernel), `BuilderUtils.h` (376), `Setup.cu` (scene bounds, Morton codes), `WideConverter.cu` (442, BVH2 to compressed BVH8), `Eval.cu` (SAH cost).
- Language: CUDA C++17.
- Speed (README, RTX 4080 SUPER, i9-14900K): Sponza 0.3M tris 0.44 ms total (bounds 0.03, Morton 0.01, sort 0.06, BVH2 0.20, BVH8 0.14); Buddha 1.1M 1.12 ms; Hairball 2.9M 2.95 ms; Bistro 3.9M about 4 ms. 64-bit Morton codes make the sort about 3 times slower. Paper reference: Benthin et al. 2024, https://dl.acm.org/doi/10.1145/3675377.
- Doesn't map to WGSL:
  - Warp intrinsics everywhere: `__ballot_sync`, `__shfl_sync`, `__ffs`, `__popc`, full-mask warp loops (lines 55 to 268). Needs subgroups with size >= 32; on Intel (8/16/32), Apple (32) and AMD (32/64) the code needs a size-generic rewrite or a fixed-size workgroup-memory fallback.
  - Atomics are `atomicAdd` (cluster counter) and `atomicExch` (parent index handoff). Both exist in WGSL on `u32`. No 64-bit atomics seen.
  - 64-bit Morton option: skip it; use 30-bit codes (or two-pass 2x30) since WGSL has no u64.
  - No spin waits seen in `BinaryBuilder.cu` (the `while` loops are warp-uniform ballot loops).
- Port effort: L. Do it after PLOC.

### 4. Sort and scan primitives (WGSL, needed for LBVH/PLOC)
- **Gridwise** https://github.com/gridwise-webgpu/gridwise, Apache-2.0, last push 2026-10-27, npm `gridwise`. `DLDFScan` (decoupled lookback / fallback), `OneSweepSort` (u32 keys, key-only, key-value variants), `HierarchicalScan` (no carry chain). Uses `subgroups` where available, with emulation otherwise. Its own docs (`docs/scan-and-reduce.md`) say decoupled lookback needs a forward-progress guarantee, that Apple and ARM GPUs do not provide it, and that on Apple hardware it locked up the machine in their testing. Emulated subgroups spin-wait on workgroup flags and "can stall" on implementations without forward progress. For us: fine on NVIDIA and Intel desktops, **risky on Mac/iOS**, so we must test on the Mac and keep a non-lookback fallback (reduce-then-scan, e.g. `HierarchicalScan`). Performance charts live in `docs/performance.md`; I did not extract numbers. Needs `maxComputeWorkgroupStorageSize: 32768` for the sort.
- **wgpu_sort** https://github.com/KeKsBoTer/wgpu_sort, BSD-2 per GitHub (no LICENSE file found at root; verify before copying), last push 2024-07-15. Port of the Fuchsia Vulkan radix sort (reduce-then-scan style, no decoupled lookback, so no forward-progress dependency). Bench (README): 1M pairs 0.32 ms, 8M 1.64 ms, 20M 3.98 ms on RTX A5000; 20M takes 46 ms on a Radeon R9 380. Weakness: guesses subgroup size by trial because it predates WebGPU subgroups. Sorting 262k Morton keys is well under 1 ms on a desktop GPU.
- **kishimisu/WebGPU-Radix-Sort**: 4-way radix, no license on GitHub (npm page says MIT; conflicting, treat as unusable until clarified). Last push 2024-08.
- Others seen, not evaluated: `jaesung-cs/webgpu_radix_sort` (MIT, WIP), `AICL-Lab/webgpu-sorting` (MIT, bitonic and radix), Rust crate `wgpu-algorithms` (reports about 365 M items/s sort, 5.2 G items/s scan on an M3 Max, author's numbers).

### 5. Existing WGSL LBVH examples (reference only)
- `nff747/splat-bvh-core` (Apache-2.0, 2026-11-01, TypeScript + WGSL): Morton, GPU radix sort, Karras LBVH, ray cast, for Gaussian splats. Claims millisecond builds for millions of splats; I saw no measured numbers. Useful as a WGSL skeleton for Karras's `clz`-based topology kernel. Splats, not triangles.
- `Skyepulse/Fast-Parallel-BVH-Construction-WebGPU` (MIT, 1 star, 2026-03): Karras LBVH in WebGPU. Small, unverified quality.
- Babylon.js forum thread, https://forum.babylonjs.com/t/i-am-writing-webgpu-ray-tracer-is-anyone-interested-in-this-topic/42188 : a pure WGSL LBVH build and traversal; the author says a WGSL radix sort is required and that the Babylon port ran 5 to 7 times slower than native WebGPU because of bind-group recreation. License unknown; treat as an anecdote, not code.
- `ycl-berserk/quarry-gpu` (Apache-2.0): CUDA, not WebGPU; ignore.
- `lisyarus/webgpu-raytracer` (MIT), `Domenicobrz/C2-Renderer` (MIT), `codedhead/webrtx` (MIT): WebGPU tracers; per the Babylon thread webrtx has no GPU BVH build. Not checked in code.

### 6. Other native libraries (algorithm references)
- **Embree** https://github.com/embree/embree, Apache-2.0, last push 2026-11-03. Builders in `kernels/builders/` (Morton `bvh_builder_morton.h`, SAH `bvh_builder_sah.h`, binning heuristics, spatial split, multi-segment). C++ with TBB; GPU path is SYCL/Level Zero for Intel (not WebGPU-portable). Use as the correctness and quality reference: SAH cost and leaf rules. Not worth porting.
- **madmann91/bvh** https://github.com/madmann91/bvh, MIT, 2026-10-16. C++ CPU only: `binned_sah_builder.h`, `sweep_sah_builder.h`, `top_down_sah_builder.h`, `mini_tree_builder.h`, `reinsertion_optimizer.h`, `node.h`. The README of tinybvh benchmarks against it. The Rust `bvh` crate (svenstaro/bvh, MIT, 2026-10-12) is a simple SAH BVH with no GPU layout; reference only.
- **obvhs** https://github.com/DGriffin91/obvhs, Apache-2.0/MIT, 2026-10-05. Rust: PLOC BVH2 + parallel reinsertion (`src/ploc/`, `src/bvh2/reinsertion.rs`), CWBVH (`src/cwbvh/`). Benchmarks in https://github.com/DGriffin91/tray_racing (Apache-2.0). Readable CPU PLOC reference to prototype the algorithm in TypeScript before touching WGSL.
- **Radeon Rays 4 (RadeonRays_SDK)** https://github.com/GPUOpen-LibrariesAndSDKs/RadeonRays_SDK, MIT, last push 2024-03. HLSL/Vulkan, has LBVH and a bottom-up refit; older sibling of HIPRT. Reference only; HIPRT supersedes it.
- **HIP-BVH-Construction** https://github.com/Niravaana/HIP-BVH-Construction, MIT, 2024-10: single readable HIP codebase with two-pass LBVH, single-pass LBVH, PLOC++, H-PLOC (`HplocKernel.h`, `Ploc++Kernel.h`, `SinglePassLbvhKernel.h`) and a traversal kernel. A good second reading alongside NexusBVH, since all variants share one structure. Single-pass LBVH needs a forward-progress-free rewrite check before porting.
- **Intel GPU builders**: Embree's GPU builders are part of the main Embree repo (SYCL); not examined in depth. No standalone open Intel GPU builder found.
- **Kajiya** (h3r2tic/kajiya, Apache-2.0) and **pbrt-v4** (Apache-2.0): CPU/Vulkan RT references only.

### 7. Web: three-mesh-bvh and friends
- **three-mesh-bvh** https://github.com/gkjohnson/three-mesh-bvh, MIT, v0.9.18 (2026-11-03). Active.
  - Builds on the CPU: `strategy` CENTER (default, fastest), AVERAGE, SAH (8 bins per axis). `targetLeafSize` default 10, `maxDepth` default 40. Worker builds: `GenerateMeshBVHWorker` and `ParallelMeshBVHWorker` (needs SharedArrayBuffer), both under `three-mesh-bvh/worker`.
  - WebGPU path `three-mesh-bvh/webgpu`: `BVHComputeData` packs a TLAS plus BLAS into storage buffers and exposes WGSL `raycastFirstHit`, `raycastAnyHit`, `closestPointToPoint`; runs only inside three's node system. Stack depth 60. `ClusteredBVH` groups instances/skinned meshes under a top level. WGSL source is in `src/webgpu/bvh_ray_functions.wgsl.js`.
  - Speed: no build-time numbers in the repo that I found. From our own spike (`docs/design/trace.md`, part B): our CPU build is about 4 times slower than three-mesh-bvh's with SAH, so its SAH build of Sponza is roughly 7 s by that ratio (derived, not measured). Trace: 87 Mrays/s closest-hit on Sponza on the RTX 3060 and 5.3 on the Intel UHD (same document).
  - Use: dependency of the **three adapter only**; port its packing, not its traversal.
- **three-gpu-pathtracer** https://github.com/gkjohnson/three-gpu-pathtracer, MIT, active (2026-11-04). WebGL/three, built on three-mesh-bvh; not usable in an engine-free core. Reference for BVH texture packing and for how tmb handles multi-mesh scenes.
- **WASM BVH builders**: tinybvh (Emscripten supported, see above); Embree has no official WASM build (unverified; I did not find one). `diogovtx/com.diogovtx.tinybvh` is a C# port for Unity, not relevant.

## Check: per-mesh trees from three-mesh-bvh, converted to our layout

How tmb stores a tree (`src/core/Constants.js`, `src/core/utils/nodeBufferUtils.js`, `src/core/build/buildTree.js`):
- `bvh._roots` is an array of `ArrayBuffer`, one per index group (usually one). Serialized form is `MeshBVH.serialize(bvh)` with `roots`, `index` and `indirectBuffer`.
- Nodes are 32 bytes (`BYTES_PER_NODE = 6*4+4+4`), stored **depth-first**: the left child of node `i` is `i+1`.
  - Words 0 to 5 (f32): the node's own AABB min and max.
  - Internal node: u32[6] = right child offset **relative** to this node (in nodes); u32[7] = split axis.
  - Leaf node: u32[6] = first primitive (triangle) offset; u16[14] = count; u16[15] = `0xFFFF` (leaf flag).
- Triangle order: by default tmb reorders `geometry.index` in place so leaf ranges are contiguous index runs; with `indirect: true` it leaves the index alone and keeps a separate `_indirectBuffer`. Empty-leaf bounds are `[+inf, -inf]`.
- Needs a leaf size near 4: pass `targetLeafSize: 4` (default is 10); the SAH may still go below or above as a soft limit unless `_strictLeafSize` is set (private).

How hard is the conversion? **Easy; tmb already does most of it.** `packBVHBufferUtils.js` (`appendBVHData`, `appendBVHSubtree`) writes a 64-byte GPU node with both children's bounds in words 0 to 11, `u32[12]` = right-child offset or triangle offset, `u32[13]` = split axis or (`count | 0xFFFF << 16`). That is the same two-child-bounds idea as our `GiBvhNode`. Differences to handle in our own converter (about 60 lines of JS, one linear pass, cheap next to the build):
1. tmb keeps leaves as separate nodes and stores a leaf's bounds in its parent's slot; ours inlines leaf children (count > 0 on the child side). So walk the DFS array and, for each internal node, emit one node whose left/right child fields are either an inner-node index (remapped, since leaf nodes disappear and indices shrink) or `(firstTri, count)`.
2. Our indices are absolute; tmb's are relative. Compute a remap table in the same pass.
3. Triangles: gather from tmb's index order into our 48-byte tree-order triangles with the watertight vertex layout (`v0, v1, v2` plus original triangle index). If tmb ran with `indirect: false`, original triangle ids need `indirectBuffer` or a saved copy of the original index; use `indirect: true` to keep the id mapping.
4. Single-triangle meshes: tmb gives one leaf root; we need an inner root with an empty right child.
5. Extra outputs we need: a parent index array (for refit) and max depth, both computed in the same pass.
6. Depth: tmb `maxDepth` 40 is within our 64 stack, but top-level plus mesh depth must still fit.
7. tmb can run in a worker (`GenerateMeshBVHWorker`) and returns serialized buffers that can be transferred; the converter then runs in the same worker or on the main thread.
Risk: the SAH quality and leaf choices differ from our binned SAH (tmb's cost constants are marked TODO in `Constants.js`), so trace speed needs re-measuring; and the build is still CPU time (CENTER builds are much faster but make worse trees).

Also: the **TLAS** could reuse tmb's `ClusteredBVH`-style logic or stay as our own CPU binned SAH; no reason to change.

## Cross-cutting WebGPU caveats

- **Subgroups:** WebGPU `subgroups` is an optional feature; size varies by GPU (Intel 8/16/32, NVIDIA 32, AMD 32/64, Apple 32, others 4 to 128) and cannot be fixed in WGSL. Warp-style kernels (HIPRT PLOC, H-PLOC) assume 32 or 64. We build subgroups-first, but need either a size-generic implementation or a workgroup-memory fallback for devices without the feature.
- **64-bit atomics and integers:** not in WGSL. Use 30-bit Morton codes in `u32`; packed 64-bit keys need two `u32` and a two-pass sort.
- **Forward progress:** avoid anything that spins on another workgroup's flag. Decoupled-lookback scan and OneSweep (Gridwise) and single-pass LBVH variants need it; the evidence is that Apple and ARM GPUs may hang. Prefer reduce-then-scan sort and "last arrival continues" parent-walk with `atomicExchange`.
- **Pointers/recursion:** WGSL forbids recursion and storage-buffer pointer parameters across function calls (tmb's own release notes cite this), so ported kernels must use index-based loops.
- **Storage buffer count:** 8 per stage on a default device; sort, scan and build passes each need several buffers, so batch them into few structured buffers.

## Table of all libraries

| Name | License | Layout | Builders | Speed (source) | How we'd use it | Link |
| --- | --- | --- | --- | --- | --- | --- |
| tinybvh | MIT | `BVH` 32 B; `BVH_GPU` 64 B (= ours); `BVH4_GPU`; `BVH8_CWBVH` 80 B; MBVH; TLAS | Bonsai, SBVH, binned SAH, full sweep, H-PLOC, midpoint | Faster CPU SAH than Embree and madmann91 (README graph, no ms); up to 5 Grays/s traced on RTX 5080 laptop Sponza (README) | WASM builder in worker; port `traverse_bvh2.cl`; HPLOC as oracle | https://github.com/jbikker/tinybvh |
| HIPRT (AMD) | MIT (`license.txt`) | Own BVH2/wide nodes | LBVH, PLOC, SBVH, batch builder | No numbers read | Port LBVH and PLOC kernels to WGSL | https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT |
| NexusBVH | Apache-2.0 | BVH2, compressed BVH8 (80 B) | H-PLOC | Sponza 0.44 ms, Bistro about 4 ms on RTX 4080 SUPER (README) | Port H-PLOC later; oracle via tinybvh `useHPLOC` | https://github.com/StokastX/NexusBVH |
| Radeon Rays 4 | MIT | BVH2/BVH4 | LBVH, refit (HLSL/Vulkan) | Not read | Reference only | https://github.com/GPUOpen-LibrariesAndSDKs/RadeonRays_SDK |
| Niravaana HIP-BVH-Construction | MIT | BVH2, N-wide collapse | Two-pass and single-pass LBVH, PLOC++, H-PLOC, binned SAH | Not read | Second reference for PLOC and H-PLOC kernels | https://github.com/Niravaana/HIP-BVH-Construction |
| Meister PLOC / PRBVH (official) | No license (all rights reserved) | n/a | PLOC, parallel reinsertion (CUDA) | Paper only | Read the paper; do not copy code | https://github.com/meistdan/ploc |
| AlanIWBFT CWBVH | No license | CWBVH 80 B | n/a | n/a | Do not copy; use the paper (Ylitie 2017) | https://github.com/AlanIWBFT/CWBVH |
| Embree | Apache-2.0 | BVH4/BVH8 (CPU), SYCL GPU | Morton, SAH, spatial split, multi-segment | Industry reference | Algorithm reference only | https://github.com/embree/embree |
| madmann91/bvh | MIT | BVH2 | Binned/sweep/top-down SAH, mini-tree, reinsertion (CPU) | Slower than tinybvh per tinybvh README | Algorithm reference | https://github.com/madmann91/bvh |
| Rust `bvh` (svenstaro) | MIT | BVH2 | SAH (CPU) | Not read | Reference only | https://github.com/svenstaro/bvh |
| obvhs (+ tray_racing) | Apache-2.0 / MIT (dual per README badge) | BVH2, CWBVH 80 B | PLOC with reinsertion, CWBVH | Benchmarks in tray_racing | Readable CPU PLOC to prototype in TS; reinsertion reference | https://github.com/DGriffin91/obvhs |
| three-mesh-bvh | MIT | 32 B CPU nodes (DFS, relative right offset); 64 B GPU nodes in `BVHComputeData`; TLAS+BLAS | CENTER, AVERAGE, SAH (8 bins); worker and parallel worker | No numbers in repo; our spike: tmb 4 times faster to build than our CPU SAH; 87 and 5.3 Mrays/s on Sponza (RTX 3060, Intel UHD), `docs/design/trace.md` | Three adapter dependency; convert to our nodes (see section) | https://github.com/gkjohnson/three-mesh-bvh |
| three-gpu-pathtracer | MIT | tmb textures (WebGL) | via tmb | Not read | Reference only | https://github.com/gkjohnson/three-gpu-pathtracer |
| Gigi (EA) tinybvh WebGPU demo | BSD-3-style (EA `LICENSE.txt`) | tinybvh layouts | tinybvh | README claim "as fast as DXR in DX12" (via tinybvh README) | Look for the WGSL traversal; not located | https://github.com/electronicarts/gigi |
| Gridwise | Apache-2.0 | n/a (scan, reduce, sort) | OneSweep sort, decoupled lookback scan, hierarchical scan | `docs/performance.md` (not extracted) | Depend on it or vendor; test Apple forward progress | https://github.com/gridwise-webgpu/gridwise |
| wgpu_sort | BSD-2 (per GitHub; verify) | n/a | Radix key-value sort (Fuchsia port) | 8M pairs 1.64 ms, RTX A5000 (README) | Vendor as sort fallback | https://github.com/KeKsBoTer/wgpu_sort |
| kishimisu WebGPU-Radix-Sort | none on GitHub (npm says MIT) | n/a | 4-way radix | Not read | Avoid until license clarified | https://github.com/kishimisu/WebGPU-Radix-Sort |
| splat-bvh-core | Apache-2.0 | LBVH over splats | Morton, radix sort, Karras LBVH (WGSL) | No numbers found | Reference for WGSL Karras kernel | https://github.com/nff747/splat-bvh-core |
| Fast-Parallel-BVH-Construction-WebGPU | MIT | LBVH | Karras LBVH (WebGPU) | None | Reference only | https://github.com/Skyepulse/Fast-Parallel-BVH-Construction-WebGPU |
| Babylon forum WGSL LBVH | Unknown | LBVH | WGSL LBVH + radix sort | Anecdote: Babylon port 5 to 7 times slower than raw WebGPU | Cautionary note | https://forum.babylonjs.com/t/i-am-writing-webgpu-ray-tracer-is-anyone-interested-in-this-topic/42188 |
| Wonderland Engine | Proprietary engine | tinybvh | tinybvh (WebGL ray tracing) | n/a | Evidence tinybvh works on the web | https://wonderlandengine.com/ |
| Intel GPU builders | Part of Embree (Apache-2.0) | SYCL | n/a | n/a | Not usable for WebGPU | https://github.com/embree/embree |

No GPL or LGPL libraries turned up in this survey. The unusable-to-copy items are the license-less repos (Meister PLOC/PRBVH, AlanIWBFT CWBVH, kishimisu radix sort).

## Recommendations

1. **Fix the load-time problem now with a CPU build that is already fast.** Spike tinybvh compiled with Emscripten in a worker (single-threaded first), emit `BVH_GPU` nodes, and compare against our 28 s Sponza figure. In parallel, in the three adapter only, time three-mesh-bvh with `{strategy: SAH, targetLeafSize: 4, indirect: true}` plus the ~60-line converter. Either gives a faster path with no GPU risk.
2. **Do the GPU build in stages, each a separate kernel:** Morton codes, then a sort (Gridwise first, wgpu_sort as the no-forward-progress fallback), then HIPRT's LBVH, then HIPRT's PLOC. Test each against tinybvh output on CPU (and tinybvh `useHPLOC` for the PLOC stage) and measure trace speed on top of the resulting tree, not just build time.
3. **Defer H-PLOC and CWBVH.** H-PLOC (NexusBVH) is the best build, but its warp-heavy code is the biggest port; CWBVH adds an 80-byte node and a new traversal for an unproven gain in a DDGI workload with short rays and 256 rays per probe.
4. **Decide the subgroup policy before porting:** require `subgroups` for the GPU builder (feature-detect, fall back to the WASM CPU builder), use 30-bit Morton codes, and put the Mac in the test matrix for any kernel that waits on another workgroup.
5. **Keep tmb out of the core.** It needs three's node system for traversal; use it only as the adapter's CPU builder and as a performance baseline, and copy its ideas (DFS layout, relative offsets, empty-leaf handling) rather than its code paths.

## Added sources (from Dennis, 2026-11-04)

- **Alain Galvan, "Ray Tracing Acceleration Structures"** (2022, since updated): https://alain.xyz/blog/ray-tracing-acceleration-structures. An overview of builder families (top-down, bottom-up, insertion), Morton codes, refit, two-level TLAS/BLAS, and short-stack wide traversal. Papers it adds to our list:
  - **Meister et al. 2021, "A Survey on Bounding Volume Hierarchies for Ray Tracing":** https://meistdan.github.io/publications/bvh_star/paper.pdf. The standard survey; read it first.
  - **Benthin et al. 2017, "Improved Two-Level BVHs Using Partial Re-Braiding":** http://www.sven-woop.de/papers/2017-HPG-openmerge.pdf. Better top-level trees when instances overlap. Relevant to furniture packed in a room.
  - **Vaidyanathan et al. 2019, "Wide BVH Traversal with a Short Stack":** https://www.intel.com/content/www/us/en/developer/articles/technical/wide-bvh-traversal-with-a-short-stack.html. Relevant to B3.
  - **Karras, "Thinking Parallel, Part III: Tree Construction on the GPU":** https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
  - **Bikker 2022, "How to build a BVH" series:** https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
  - **Libmorton** (MIT): https://github.com/Forceflow/libmorton

## License status (Dennis, 2026-11-04)

Dennis cleared every source listed in this file for use: "So far all of the ones you listed we can use". That includes the repos with no license file (meistdan/ploc, AlanIWBFT/CWBVH, kishimisu/WebGPU-Radix-Sort, and wgpu_sort).

- Any code ported from them is attributed in `THIRD_PARTY.md`, with repo, commit and what was taken.
- Gigi's WebGPU output is generated code, so it's dropped as a source.
- The original PLOC is allowed "if needed". Prefer the newer methods (H-PLOC).
