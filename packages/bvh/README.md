# @pmndrs/bvh

Engine-free BVH builders for the ddgi platform (docs/design/bvh-framework.md).
A builder produces a plain tree description (`MeshTreeTopology`, in
`src/topology.ts`), which the tracer in `@pmndrs/ddgi-core` turns into its GPU
layout, or writes that layout itself (`GpuLayoutMeshBvh`, in `src/gpu-layout.ts`).
This package is bundled into `@pmndrs/ddgi` and never published on its own.

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
  boxes are recomputed from the triangles, not taken from tinybvh. On Sponza
  this takes about 0.2 s, against 0.65 s for the conversion in JavaScript. The
  tracer's tinybvh path uses it.

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
