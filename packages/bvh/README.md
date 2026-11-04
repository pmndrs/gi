# @pmndrs/bvh

Engine-free BVH builders for the ddgi platform (docs/design/bvh-framework.md).
A builder produces a plain tree description (`MeshTreeTopology`, in
`src/topology.ts`), which the tracer in `@pmndrs/ddgi-core` turns into its GPU
layout, or writes that layout itself (`GpuLayoutMeshBvh`, in `src/gpu-layout.ts`):
tinybvh in WebAssembly (B0) on the CPU, the LBVH (B1) on the GPU.
This package is bundled into `@pmndrs/ddgi` and never published on its own.

## GPU LBVH (stage B1)

`src/gpu-lbvh.ts` (`GpuLbvhBuilder`) builds mesh trees on the GPU, in exactly
the layout tinybvh's `buildLayout` writes (`src/gpu-layout.ts`), so the tracer
reads them unchanged. The kernels are in `src/gpu-lbvh-wgsl.ts`, ported from
HIPRT's LBVH (MIT, `licenses/hiprt.txt`):

1. triangle boxes and each mesh's centroid box (subgroup, then workgroup, then
   one atomic per workgroup);
2. 30-bit Morton codes in cubic cells over each mesh's own centroid box
   (stretching every axis to 1024 cells, as HIPRT's plain code does, made flat
   and long meshes cost 2.4-2.6 times tinybvh's SAH; cubic cells 1.02-1.22);
3. a stable, segmented radix sort (`src/gpu-radix-sort.ts`: 5-bit digits,
   reduce-then-scan, no look-back), on the scan in `src/gpu-scan.ts`;
4. Apetrei's bottom-up climb with HIPRT's index tie-break, the SAH leaf
   collapse decided on the way up, in two rounds (the second an indirect
   dispatch);
5. a scan numbers the nodes that stay; triangles, nodes and parents are
   written in the tracer's layout.

It builds every mesh of a **batch** with the same dispatches (a mesh table and
a tile table say which workgroup works on which mesh), so a scene of many
small meshes doesn't pay per mesh. The trees come back in three shared GPU
buffers with an offset per mesh (`GpuLbvhTrees`); `readLayouts` copies them
into CPU memory. `planGpuLbvhBatches` splits a scene into batches that fit the
device's buffer limits. Every word one climbing thread hands to another goes
through atomics with a bounded wait, as in the B1a refit; a tree whose wait ran
out is refused (`problem`), never used.

Workgroup scans number threads in subgroup order (`src/gpu-workgroup-scan.ts`),
so nothing depends on the subgroup size (Intel's UHD runs 8 or 16 wide) or lane
layout. The builder needs `subgroups`.

**Fallback.** `src/mesh-builder.ts` (`createMeshLayoutBuilder`) picks the GPU
LBVH when the device has subgroups and tinybvh (B0) otherwise, and also sends
single meshes to tinybvh when they are too big for the device or the GPU
refused them. The result says which builder made each tree.

**Deforming meshes.** `src/gpu-mesh-refit.ts` (`GpuMeshRefit`) refits any
builder's mesh tree from a vertex buffer the adapter supplies: one kernel
rewrites the tree-ordered triangles from the moved vertices and writes their
boxes, then the B1a refit climbs. The shape comes from a CPU copy of the layout
once (`setTree`), not every frame.

**Checking a tree.** `src/layout-check.ts` checks any layout against its mesh:
triangles bit for bit, one tree from node 0, every box exactly the box below
it, parents, depth and counts; `treeSahCost` gives its SAH cost.
`src/lbvh-cpu.ts` is the same LBVH written plainly on the CPU, for the unit
tests and as a cross-check (not a builder to ship).

GPU pages: `bvh-lbvh` (scan, sort, trees), `bvh-blas-refit`, `bvh-lbvh-fallback`,
`bvh-lbvh-profile` (manual, step-by-step timing), the `watertight-lbvh` variant
of `trace-oracle`, and `trace-bench-lbvh-sponza` / `-quick` (build time and trace
speed against tinybvh's trees).

## GPU refit (stage B1a)

`src/gpu-refit.ts` refits a tree in the GPU layout on the GPU, keeping its
shape: one thread per leaf climbs to the root, and at each node the second
thread to arrive (a per-node counter, no float atomics) takes the union of the
two child boxes. Boxes travel as 32-bit atomic words, so no plain store has to
be visible across workgroups; a bounded re-read covers WGSL's relaxed atomic
order. If a read still gives up, a one-workgroup repair pass refits the whole
tree level by level in the same compute pass, so a wrong box is never kept;
the stats (read back only with `diagnostics: true`) count it. Sibling
leaves held by neighbouring subgroup lanes finish their node in registers. The
climb also sums every inner node's half-area, the tree-quality signal
(`relativeSahCost`) a caller uses to decide when to rebuild. It needs the
`subgroups` feature (`GpuBvhRefit.isSupported`).

`src/gpu-instance-box.ts` is the WGSL for an instance's padded world box
(Arvo's method), and `src/refit-cpu.ts` holds the CPU twins: leaf and parent
slots, the cost signal, and a plain refit the tests compare the GPU with.
The tracer's top level (ddgi-core `trace/top-level-gpu.ts`) uses all three.

## tinybvh in WebAssembly (stage B0)

`src/tinybvh.ts` runs [tinybvh](https://github.com/jbikker/tinybvh) 1.9.1
(MIT, Jacco Bikker), compiled to WebAssembly. The `.wasm` is committed,
base64-encoded, in `src/tinybvh-wasm.ts`, so it loads the same way in a worker,
on a page, in Node and in any bundler. It is single-threaded; the tracer runs it
in a worker.

It uses tinybvh's default builder, which with SIMD is Bonsai (Ganestam et al.,
2015). From one tree it gives either output:

- `build(geometry)` returns the tree's shape (`MeshTreeTopology`).
- `buildLayout(geometry)` (stage B0b) returns the tracer's GPU layout, written
  inside the WASM by `native/tinybvh-wasm.cpp`: 64-byte nodes, 48-byte
  triangles in tree order, parents and the mesh's box. The bytes are exactly
  what ddgi-core's `buildMeshBvhFromTopology` makes from `build`'s shape; a
  test there checks it (`mesh-bvh-tinybvh-layout.test.ts`). As there, the
  boxes are recomputed from the triangles, not taken from tinybvh. It is
  several times faster than doing the conversion in JavaScript, which walks
  every node and triangle through typed arrays and is bound by memory traffic;
  the WASM packs each leaf's triangles and computes its box in one pass, with
  SIMD compares instead of branches. The timings are in PROGRESS.md (B0b). The
  tracer's tinybvh path uses it.

`tools/bvh/layout-identity.ts` rebuilds every mesh of Sponza and both Bistro
scenes both ways and reports how many differ (it should be none):

```sh
node tools/bvh/layout-identity.ts            # all three scenes
node tools/bvh/layout-identity.ts sponza     # one scene
```

The layout itself is defined in `packages/ddgi-core/src/platform/trace/layout.ts`.
The C++ repeats it as structs with `static_assert`s on every offset; change both
together. The build needs WebAssembly SIMD (SIMD128), which every current Chrome,
Edge, Firefox and Safari has (Chrome 91, Firefox 89 and Safari 16.4 onwards), so
there is no scalar fallback build.

### Rebuilding the .wasm

```sh
pnpm bvh:wasm           # install the pinned Emscripten if needed, then build
pnpm bvh:wasm --check   # build, and fail unless it matches the committed .wasm
```

`native/build-wasm.ts` does everything:

1. **Toolchain.** It uses emsdk **6.0.13** (emsdk commit
   `35659964bc2762fffec0d257997fb61475f765b7`) in `tools/emsdk/`, which is
   gitignored. If that folder is missing it clones emsdk there and runs
   `emsdk install 6.0.13` and `emsdk activate 6.0.13` inside it, which downloads
   about 700 MB. It never uses or touches any other Emscripten on the machine:
   it drops every `EM*` variable from the environment and sets its own config
   and cache, both inside `tools/emsdk/`.
2. **Sources.** It downloads tinybvh's headers at commit
   `c9be3103cc9693b456b0a0353927c11cac7d710f` and checks each one's SHA-256.
3. **Build.** It compiles `native/tinybvh-wasm.cpp` (our small C interface) with
   the flags listed, with a reason for each, in `BUILD_FLAGS`: `-O3`, no threads,
   and `-msimd128 -mavx`. Emscripten turns tinybvh's AVX code into WASM SIMD.
   AVX turns on tinybvh's fast builders. Without it tinybvh falls back to its
   scalar reference builder, which is about 3 times slower on Sponza.
   The build runs in a temporary folder with relative paths only, so no folder
   name ends up in the output.

Needs: Node 24, git and network access. On Linux and macOS it also needs
Python 3, because emsdk brings its own Python only on Windows.

**Reproducibility.** On Windows (2026-11-04), repeated builds from different
temporary folders gave the same bytes as the committed `.wasm` (`--check`
passes). A Linux or macOS build has not been compared yet.

### Licences

`licenses/tinybvh.txt` is tinybvh's licence. The build checks it against
tinybvh's own `LICENSE` file. `licenses/emscripten-runtime.txt` covers the
parts of Emscripten's runtime that the `.wasm` contains: Emscripten's own code
and musl (MIT), dlmalloc (public domain), and LLVM's libc++ and compiler-rt
(Apache 2.0 with LLVM Exceptions). The generated `src/tinybvh-wasm.ts` starts
with both texts in a `/*! */` comment, which bundlers keep. As a result, every
built file that contains the `.wasm` also contains its licences.
`packages/ddgi/scripts/verify-artifacts.mjs` fails the package build if one
doesn't.
