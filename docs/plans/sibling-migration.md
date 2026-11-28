# Moving gi onto the published sibling libraries

Plan written 2026-11-19 for Dennis's decision of that day (DECISIONS: gi uses the published `@pmndrs/bvh` and `@pmndrs/downsampler`; fixes go upstream as PRs or issues). Read against gi `main` 0d55a09, `radiance-cache` 7c80225, bvh 6163232 (0.1.0) and downsampler 78d5407 (0.1.0). Nothing here is done yet: it runs after the radiance-cache merge and the workspace rename (#32).


Read on 2026-11-19: gi `origin/main` 0d55a09, `origin/radiance-cache` 7c80225, bvh 6163232, downsampler 78d5407. Nothing was committed or filed.

## 1. Inventory (every import is in `packages/ddgi-core/src/platform/`)

The trace imports are the same on both branches. 0.1.0 exports only `.`, `/worker`, `/three` and `/babylon`, so **every deep subpath import breaks**. Each one has to move to the root import.

| Import (used by) | 0.1.0 |
|---|---|
| root: `createTinyBvhBuilder`, `TinyBvhBuilder`, `TOPOLOGY_*`, `MeshTreeTopology`, `GPU_LAYOUT_*` (mesh-bvh-build, -topology, 6 tests) | root ✓ |
| `/gpu-layout` constants, `GpuLayoutMeshBvh` (layout, gpu-mesh-trees, deforming-trees) | root ✓ (layout v2) |
| `/gpu-hploc` `GpuHplocBuilder`, `planGpuHplocBatches` (gpu-mesh-trees) | root ✓ |
| `/gpu-lbvh` types `GpuLbvhBufferGeometry/MeshTree/Trees` | **renamed** (`GpuMeshBuffers`/`GpuMeshGeometry`, `GpuFirstTrees` result types) |
| `/gpu-mesh-refit` `GpuMeshRefit(+Kernels)` (deforming-trees) | root ✓; `setTree` with an offset now needs `nodeFormat` |
| `/gpu-refit` `GpuBvhRefit`, `BVH_REFIT_STAT*` (ray-queries, top-level-gpu) | root ✓ |
| `/gpu-instance-box`, `/refit-cpu` (`refitOnCpu`, `relativeSahCost`, `halfArea`, slot helpers, `PRIMITIVE_BOX_FLOATS`, `BVH_FLOAT_MAX`) | root ✓ (`refitOnCpu` refuses quantized nodes) |
| radiance-cache only: `/gpu-scan` (`createScanPipelines` with `{reduction, io, layout}`, `encodeScan`, `ScanIo`, `SCAN_*`) in compute/gpu-compute.ts, wgsl.ts | **gap**: not exported, and it has no io and no fallback |
| radiance-cache only: `/wgsl-names` `wgslNameProblems` (compute.test) | **gap**: not exported |

Behind those imports, gi's own `platform/trace` (about 7,900 lines: walk, tree storage, GPU top level, deforming trees) is the pre-split code that bvh now owns (`src/trace`, `src/scene`). P3 deletes most of it.

## 2. Differences that matter

- **Layout:** `GPU_LAYOUT_VERSION` 1→2. The instance flags carry the mask in bits 8–15 and the material in bits 16–31. With masks on, gi's gather (`top-level-gpu.ts`) must write `0xff << 8`, or no ray hits anything. Defaults: 32-byte quantized nodes for scenes and walks (builders still write f32), static instances merged (only those marked `static: true`), a 40-byte hit record, and `BvhHit` with 7 fields.
- **Who owns the walk:** bvh, through `traceModuleWgsl({ traceSlot, triangleFilter: null, rayMasks, nodeFormat })`. gi keeps `giTraceClosest` and `giTraceAny` as its *platform contract* (contract/queries.ts, plus the gi* naming rule) as thin wrappers over `bvhTrace*`. gi also keeps a `bvhTrianglePasses` wrapper around `giMaterialPasses`, its alpha-aware `reference-tracer` (bvh's reference can't flag alpha-edge ambiguity), and `triangle-flags`.
- **Tie rule:** since 2026-11-17, when two boxes share an entry plane past `tMin`, the closest-hit walk enters the right one first (gi: left). On exact ties, hits may name another triangle, and the trace-bench numbers move.
- **Builders:** H-PLOC is capped at 256 MiB per batch (gi builds one batch; `gpuMaxBatchBytes: Infinity` restores that), `gpuQuantizeOnEmit` is on, and tinybvh runs from the `@pmndrs/bvh/worker` pool (with its Vite fix). `gpuMaxDepth` now also covers tinybvh trees.
- **The swap (#26, 2026-11-28):** gi no longer swaps tinybvh's trees in over the GPU's; tinybvh only builds what the GPU can't (DECISIONS 2026-11-28). bvh 0.1.1's `SceneTrees` swaps by default (`refine: true`); `refine: false` is gi's rule, so step D passes it. One gap: under `refine: false` a deforming mesh whose layout readback failed never gets a refit (gi refuses such a tree and falls back to tinybvh). pmndrs/bvh#8 asks for `refine: false` as the default and for that fallback.
- **Refit:** a deforming mesh refits an f32 copy and is re-quantized.
- **Dirty uploads:** BVH1/BVH2 upload only changed records, and a settled update costs about 5 µs. This is a win over gi's full rewrite.
- **Scan and sort:** subgroups only and internal (`scanTakeOrder()` takes no argument).
- **`SceneTrees` gaps:** gi's `requestTriangleFlags` install hook has no twin, and `update` throws differently.

## 3. Upstream items

1. **bvh issue: any-hit misses a hit at exactly `tMax`.** The docs promise `[tMin, tMax]` inclusive, and the triangle test is inclusive. Suspected cause: the slab test's entry distance (computed through `inverseDirection`) or `t = scaledT * (1/det)` rounds past a `tMax` that the caller computed exactly, and the any-hit walk has no slack the way the closest-hit walk's `farthest` does. Repro: pathtracer's interval boundary rays. Suggested fix: pad the `tMax` side of the slab test as the exit side already is, and pin it with an oracle case. gi's own walk has the same slab test, so the move adds no regression.
2. **bvh PR for #3.** Turn the four walk `loop {}`s in `wgsl.ts` and `query-wgsl.ts` into constant-bounded `for` loops, add the final returns, and fix the `walk-counters` anchor. This matters most to gi: GI-1.0's kernels are large, and pathtracer measured 3.3x on D3D12. Prove it with trace-oracle, query-oracle and trace-inline.
3. **Scan: move it into gi's `platform/compute`; do not PR it.** Move `ScanIo` and the workgroup fallback (gpu-scan plus workgroup-scan, about 400 lines) into gi. Reasons: bvh's declared no-subgroups fallback is tinybvh, so it never needs the workgroup path; exporting a general scan widens bvh past BVHs; and gi already owns compaction, indirect args and `rebind`. This is gi's code, not a frozen sibling copy. Copy `wgslNameProblems` too (about 20 lines).
4. **bvh PR (or issue): a triangle-flags hook on placement in `SceneTrees`**, the twin of `requestTriangleFlags` (MIGRATING §8). The other route is `meshStatus(mesh).placement`.
5. **downsampler issue or PR: an ordered "first valid" uint fold with floor footprints** (see §4). DIVERGES l.164 already lists it as "no user has asked"; gi is that user.

## 4. Downsampler

| gi pass | Fit |
|---|---|
| `FilterProbeMask` (r32uint, about 8 levels, one dispatch per level) | **Needs a sibling change.** "First valid of the 2×2 children in raster order" isn't commutative, and Capsaicin drops the odd last row and column. downsampler's folds must be commutative, `'exact'` folds use the `2x…2x+2` footprint, and `'spd'` rejects folds. A `min` fold would be a divergence, so keep the hand-written pass (faithful port first) until item 5 lands. Small gain. |
| Hash-grid tile mips (4 levels) | **No.** Packed buffer cells, summed in workgroup memory, fused into `UpdateTiles`. Keep. |
| Hi-Z | **Drop-in, and new work:** RC4.2's SSGI needs Capsaicin's HZB. `{ source: depth, reduction: 'min' | 'max' }` under `'exact'` matches GPUMip DepthMin. Build it on downsampler from the start. |
| main: texture-store mips (alpha-weighted, sRGB through buffer copies, 9 levels) | **No.** The transform hook is planned but not built; `Reducer` `'weight'`/`'weighted'` covers 6 levels at most. Load time only. Keep. |
| main: sky SH, 19 sums to a scalar | **No.** `ScalarReduction` takes at most 8 channels. |

downsampler's SPEC §14 lists only these as gi's users. DDGI's per-frame mark, schedule and propagate reductions run over buffers, not pyramids.

## 5. Steps

**Order (DECISIONS 2026-11-19):** the radiance-cache merge (RC3.4, RC4.2–4.4), then #32 (`ddgi`→`gi`), then this migration. File items 1–5 now so that fixes ship in bvh 0.1.x. Work on one branch, `sibling-libs`, cut from main after the rename, in a worktree, as one PR with one commit per step.

| Step | Work | Size | Risk |
|---|---|---|---|
| A | Scan into `platform/compute` | S | low |
| B | Exact pin `@pmndrs/bvh@0.1.0`; root imports; renamed types; delete `packages/bvh` and its pages (bvh-\*, tlas-\*); `verify-packed`: bvh becomes a runtime dependency, not bundled; worker under Vite | M | low |
| C | Walk: `traceModuleWgsl` (f32 nodes, `rayMasks: false` first), `gi*` wrappers, `bvhTrianglePasses`, CPU tracer options | L | med |
| D | `SceneTrees` replaces gpu-mesh-trees, growable-storage, top-level-gpu, deforming-trees and the tree half of ray-queries; no `static` marks yet | L | high |
| E | Measured levers, each a register row with RTX and M5 numbers: quantized nodes, merged statics, `hitNormal: false` for shadow batches, masks | M | med |
| F | Motion: gi's own motion handling moves onto `@pmndrs/motion-vectors` (0.1.0; DECISIONS 2026-11-27). The three adapter may keep three's `velocity` node as its source. What gi does with motion itself takes the package: the velocity contract the screen inputs promise, cut detection for history resets (screen-probe history, the GI denoiser, DDGI's change intake), camera motion from depth where an engine gives no velocity, and the package's oracle audit of three's velocity as a GPU page. Behind a toggle against today's path, judged by ghosting first (Dennis: ghosting is bad, latency is acceptable) | M | med |

**Checks after each step:** lint, typecheck, unit tests. From C on, also: `trace-oracle` (0 mismatches), `trace-materials`, `trace-inline`, `trace-bench-quick` (all tracers agree), `trace-runtime-trees`, `ddgi-gpu-trees` (was `ddgi-tree-swap`), `ddgi-trees-no-subgroups`, `scene-*`, `three-composite`, `three-neutrality`, `three-deforming-gi`, `platform-compute` (both reduction paths), every `radiance-cache-*` page (hash-grid, image, overflow, probes, scenes, stability, temporal, three, viewer, world-cache), `examples-site`. Then the full `pnpm test:gpu`; accuracy pages within noise.

**Mac (M5 Pro):** the same correctness set after C and D. It is the deciding machine for E: quantized nodes were 3–7% slower there. Intel UHD gets correctness only.

**Overall:** about 1.5–2 weeks of agent time, with a net deletion of about 6–8k lines.
