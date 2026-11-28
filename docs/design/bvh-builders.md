# BVH builders: how they work

How each builder in `packages/bvh` works inside. To learn what the package is and how to use it, read [packages/bvh/README.md](../../packages/bvh/README.md) first. The plan and the stages are in [bvh-framework.md](bvh-framework.md), and the measured numbers are in PROGRESS (B0, B0b, B1a, B1, B2, B3).

Paths below are relative to `packages/bvh/`.

The sections run in build order: tinybvh (B0) first, then the GPU refit (B1a), then the GPU LBVH (B1), then H-PLOC (B2), which reuses most of the LBVH.

## tinybvh in WebAssembly (stages B0 and B0b)

`src/tinybvh.ts` runs [tinybvh](https://github.com/jbikker/tinybvh) 1.9.1 (MIT, Jacco Bikker), compiled to WebAssembly. The `.wasm` is committed, base64-encoded, in `src/tinybvh-wasm.ts`, so it loads the same way in a worker, on a page, in Node and in any bundler. It is single-threaded; gi-core runs it in a worker (`mesh-bvh-async.ts`).

It uses tinybvh's default builder, which with SIMD is Bonsai (Ganestam et al., 2015). From one tree it gives either output:

- `build(geometry)` returns the tree's shape (`MeshTreeTopology`, `src/topology.ts`).
- `buildLayout(geometry)` (stage B0b) returns the tracer's GPU layout, written inside the WASM by `native/tinybvh-wasm.cpp`: 64-byte nodes, 48-byte triangles in tree order, parents, and the mesh's box.
  - The bytes are exactly what gi-core's `buildMeshBvhFromTopology` makes from `build`'s shape. A test there checks it (`mesh-bvh-tinybvh-layout.test.ts`).
  - As there, the boxes are recomputed from the triangles, not taken from tinybvh.
  - It is several times faster than doing the conversion in JavaScript. The JavaScript walks every node and triangle through typed arrays and is bound by memory traffic; the WASM packs each leaf's triangles and computes its box in one pass, with SIMD compares instead of branches.
- `buildLayout(geometry, { builder: 'hploc' })` runs tinybvh's scalar `BuildHPLOC` instead (leaves of one triangle). It is slow, and is there only as the oracle for the GPU H-PLOC (B2).

`tools/bvh/layout-identity.ts` rebuilds every mesh of Sponza and both Bistro scenes both ways and reports how many differ (it should be none):

```sh
node tools/bvh/layout-identity.ts            # all three scenes
node tools/bvh/layout-identity.ts sponza     # one scene
```

The layout itself is defined in `packages/core/src/platform/trace/layout.ts`; `src/gpu-layout.ts` holds its sizes and word positions, which layout.ts imports. The C++ repeats it as structs with `static_assert`s on every offset; change both together.

The build needs WebAssembly SIMD (SIMD128), which every current Chrome, Edge, Firefox and Safari has (Chrome 91, Firefox 89 and Safari 16.4 onwards), so there is no scalar fallback build.

### Rebuilding the .wasm

```sh
pnpm bvh:wasm           # install the pinned Emscripten if needed, then build
pnpm bvh:wasm --check   # build, and fail unless it matches the committed .wasm
```

`native/build-wasm.ts` does everything:

1. **Toolchain.** It uses emsdk **6.0.13** (emsdk commit `35659964bc2762fffec0d257997fb61475f765b7`) in `tools/emsdk/`, which is gitignored.
   - If that folder is missing it clones emsdk there and runs `emsdk install 6.0.13` and `emsdk activate 6.0.13` inside it, which downloads about 700 MB.
   - It never uses or touches any other Emscripten on the machine: it drops every `EM*` variable from the environment and sets its own config and cache, both inside `tools/emsdk/`.
2. **Sources.** It downloads tinybvh's headers at commit `c9be3103cc9693b456b0a0353927c11cac7d710f` and checks each one's SHA-256.
3. **Build.** It compiles `native/tinybvh-wasm.cpp` (our small C interface) with the flags listed, with a reason for each, in `BUILD_FLAGS`: `-O3`, no threads, and `-msimd128 -mavx`.
   - Emscripten turns tinybvh's AVX code into WASM SIMD.
   - AVX turns on tinybvh's fast builders. Without it tinybvh falls back to its scalar reference builder, which is about 3 times slower on Sponza.
   - The build runs in a temporary folder with relative paths only, so no folder name ends up in the output.

Needs: Node 24, git and network access. On Linux and macOS it also needs Python 3, because emsdk brings its own Python only on Windows.

**Reproducibility.** On Windows (2026-11-04), repeated builds from different temporary folders gave the same bytes as the committed `.wasm` (`--check` passes). A Linux or macOS build has not been compared yet.

### How the licences travel with the .wasm

`licenses/tinybvh.txt` is tinybvh's licence; the build checks it against tinybvh's own `LICENSE` file. `licenses/emscripten-runtime.txt` covers the parts of Emscripten's runtime that the `.wasm` contains: Emscripten's own code and musl (MIT), dlmalloc (public domain), and LLVM's libc++ and compiler-rt (Apache 2.0 with LLVM Exceptions).

The generated `src/tinybvh-wasm.ts` starts with both texts in a `/*! */` comment, which bundlers keep. So every built file that contains the `.wasm` also contains its licences. `packages/gi/scripts/verify-artifacts.mjs` fails the package build if one doesn't.

## GPU refit (stage B1a)

`src/gpu-refit.ts` (`GpuBvhRefit`) refits a tree in the GPU layout on the GPU, keeping its shape.

- One thread per leaf climbs to the root. At each node the second thread to arrive (a per-node counter, no float atomics) takes the union of the two child boxes.
- Boxes travel as 32-bit atomic words, so no plain store has to be visible across workgroups. A bounded re-read covers WGSL's relaxed atomic order.
- If a read still gives up, a one-workgroup repair pass refits the whole tree level by level in the same compute pass, so a wrong box is never kept. The stats (read back only with `diagnostics: true`) count it.
- Sibling leaves held by neighbouring subgroup lanes finish their node in registers. The check compares node and side, never lane numbers, so it works on any subgroup size.
- The climb also sums every inner node's half-area: the tree-quality signal (`relativeSahCost`) a caller uses to decide when to rebuild.
- It needs the `subgroups` feature (`GpuBvhRefit.isSupported`).

`src/gpu-instance-box.ts` is the WGSL for an instance's padded world box (Arvo's method). `src/refit-cpu.ts` holds the CPU twins: leaf and parent slots, the cost signal, and a plain refit the tests compare the GPU with (box unions are exact, so a correct GPU refit matches it bit for bit).

The tracer's top level (gi-core `trace/top-level-gpu.ts`) uses all three: a gather kernel writes each instance's world box, the refit runs every frame something moved, and a worker builds a new shape on the CPU now and then (instances added or removed, or the refit tree's cost past a limit).

## GPU LBVH (stage B1)

`src/gpu-lbvh.ts` (`GpuLbvhBuilder`) builds mesh trees on the GPU, in exactly the layout tinybvh's `buildLayout` writes (`src/gpu-layout.ts`), so the tracer reads them unchanged. The kernels are in `src/gpu-lbvh-wgsl.ts`, ported from HIPRT's LBVH (MIT, `licenses/hiprt.txt`):

1. Triangle boxes and each mesh's centroid box (subgroup, then workgroup, then one atomic per workgroup).
2. 30-bit Morton codes in cubic cells over each mesh's own centroid box.
   - Stretching every axis to 1024 cells, as HIPRT's plain code does, made flat and long meshes cost 2.4–2.6 times tinybvh's SAH. Cubic cells cost 1.02–1.22 times.
3. A stable, segmented radix sort (`src/gpu-radix-sort.ts`: 5-bit digits, reduce-then-scan, no look-back), on the scan in `src/gpu-scan.ts`.
   - No decoupled look-back, because WebGPU doesn't promise forward progress between workgroups.
4. Apetrei's bottom-up climb with HIPRT's index tie-break. The SAH leaf collapse is decided on the way up. The climb runs in two rounds; the second is an indirect dispatch.
5. A scan numbers the nodes that stay; triangles, nodes and parents are written in the tracer's layout.

**Batches.** It builds every mesh of a batch with the same dispatches (a mesh table and a tile table say which workgroup works on which mesh), so a scene of many small meshes doesn't pay per mesh.

- The trees come back in three shared GPU buffers with an offset per mesh (`GpuLbvhTrees`).
- `readLayouts` copies them into CPU memory.
- `planGpuLbvhBatches` splits a scene into batches that fit the device's buffer limits.

**Handing words between threads.** Every word one climbing thread hands to another goes through atomics with a bounded wait, as in the B1a refit. A tree whose wait ran out is refused (`problem`), never used.

**Subgroup size.** Workgroup scans number threads in subgroup order (`src/gpu-workgroup-scan.ts`), so nothing depends on the subgroup size (Intel's UHD runs 8 or 16 wide) or the lane layout. The builder needs `subgroups`.

**Fallback.** `src/mesh-builder.ts` (`createMeshLayoutBuilder`) picks a GPU builder when the device has subgroups and tinybvh (B0) otherwise. It also sends single meshes to tinybvh when they are too big for the device or the GPU refused them. The result says which builder made each tree.

**Deforming meshes.** `src/gpu-mesh-refit.ts` (`GpuMeshRefit`) refits any builder's mesh tree from a vertex buffer the adapter supplies.

- One kernel rewrites the tree-ordered triangles from the moved vertices and writes their boxes; then the B1a refit climbs.
- The shape comes from a CPU copy of the layout once (`setTree`), not every frame.

**Checking a tree.** `src/layout-check.ts` checks any layout against its mesh: triangles bit for bit, one tree from node 0, every box exactly the box below it, parents, depth and counts. `treeSahCost` gives its SAH cost. `src/lbvh-cpu.ts` is the same LBVH written plainly on the CPU, for the unit tests and as a cross-check (not a builder to ship).

GPU pages: `bvh-lbvh` (scan, sort, trees), `bvh-blas-refit`, `bvh-lbvh-fallback`, `bvh-lbvh-profile` (manual, step-by-step timing), the `watertight-lbvh` variant of `trace-oracle`, and `trace-bench-lbvh-sponza` / `-quick` (build time and trace speed against tinybvh's trees).

## GPU H-PLOC (stage B2)

`src/gpu-hploc.ts` (`GpuHplocBuilder`) is the target GPU builder: H-PLOC (Benthin et al., HPG 2024), behind the same contract as the LBVH and with the same output (`GpuLbvhTrees`, the tracer's layout).

- It reuses the LBVH's batches, triangle boxes, Morton codes, sort, scans and readback, and climbs the same radix tree.
- On the way up it merges clusters with PLOC, so its trees are close to a full SAH build's.
- The kernels are in `src/gpu-hploc-wgsl.ts`, ported from tinybvh's `BuildHPLOC` (MIT, `licenses/tinybvh.txt`).

How the climb works:

1. A node of at most 16 triangles holds them as they are. A bigger node (and the root) holds its children's clusters, at most 32, and merges mutual nearest neighbours (search radius 8, by merged surface area) until 16 are left (the root: one).
2. The whole subgroup merges one node at a time, with each cluster in a lane and slot: `ceil(32 / smallest subgroup size)` slots per lane, so one on 32-wide GPUs and four on the Intel UHD. It shuffles by cluster number, so the tree is the same on every subgroup size; the RTX and the Intel build identical trees.
3. The SAH leaf collapse (B1's rule) is decided as each node is made. A scan numbers the nodes that stay, and a second gives each kept node's leaves their run of triangle positions (each leaf's triangles together).

As in the LBVH, everything one climbing thread hands to another is an atomic, written once and read with a bounded wait. A node's stored clusters reuse positions on the way up, so each entry carries its writer's radix-tree level.

The smallest subgroup size comes from `device.adapterInfo.subgroupMinSize`. If the browser doesn't report it, the builder assumes 4 (eight slots per lane): correct, but slower.

**Options** (`HplocOptions`): the LBVH's, plus:

- `mortonCodes`: `'cubic-30'` (B1's codes, the default) or `'tinybvh-63'` (tinybvh's 63-bit codes, two sort words).
- `tieBreak`: `'pairs'` (the default) or `'tinybvh'`. On exactly equal merged areas tinybvh's rule chains the tree: 3000 identical triangles give depth 128, twice the tracer's stack. `'pairs'` pairs ties up and keeps the tree balanced.

**The oracle.** `src/hploc-cpu.ts` is H-PLOC on the CPU, step for step, with the same options.

- With tinybvh's codes and tie rule it builds exactly the tree of tinybvh's `BuildHPLOC` (`buildLayout(geometry, { builder: 'hploc' })`). The unit tests check that.
- The GPU's trees match the twin's (same SAH cost and node count).
- With tinybvh's options the GPU's trees match tinybvh's: SAH cost within 1% on the `bvh-hploc` page, and on whole scenes in the trace bench.

`tools/bvh/tree-quality.ts` compares the CPU builders and twins on real scenes without a GPU (`node tools/bvh/tree-quality.ts sponza`).

GPU pages: `bvh-hploc`, `bvh-hploc-profile` (manual), the `watertight-hploc` variant of `trace-oracle`, and `trace-bench-hploc-sponza` / `-quick`.

`createMeshLayoutBuilder` picks H-PLOC by default when the device has subgroups (`prefer: 'gpu-lbvh'` gives the LBVH), with the same fallback to tinybvh.
