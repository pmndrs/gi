# Third-party code in @pmndrs/gi

@pmndrs/gi is MIT licensed (`LICENSE`). It bundles the code below, which comes under its own licences. Each licence's full text is in `licenses/`, and the embedded BVH WebAssembly also carries its notices in the JavaScript bundle.

| What | Where in this package | Licence | Full text |
| --- | --- | --- | --- |
| [tinybvh](https://github.com/jbikker/tinybvh) 1.9.1 by Jacco Bikker (commit `c9be3103cc9693b456b0a0353927c11cac7d710f`), compiled unchanged to WebAssembly. Builds the ray-tracing trees for meshes. | The WebAssembly module, embedded as base64 in `dist/core.js` and in the tree-building worker `dist/assets/mesh-bvh-worker-*.js` | MIT | `licenses/tinybvh.txt` |
| The Emscripten 6.0.13 runtime linked into that WebAssembly module: Emscripten's system code, musl libc, dlmalloc, LLVM libc++ / libc++abi and compiler-rt | Inside the same WebAssembly module | Emscripten: MIT or NCSA. musl: MIT. dlmalloc: public domain (CC0). libc++, libc++abi, compiler-rt: Apache 2.0 with LLVM Exceptions | `licenses/emscripten-runtime.txt` |
| tinybvh's `BuildHPLOC`, ported step for step to WGSL and TypeScript (the GPU H-PLOC tree builder) | `dist/core.js` | MIT (tinybvh's) | `licenses/tinybvh.txt` |
| [HIPRT](https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT) by Advanced Micro Devices (commit `0629d08568fff39f9a36bf9ee800fc7bac5bb096`): its Morton codes and its LBVH build kernels, ported to WGSL (the GPU LBVH tree builder) | `dist/core.js` | MIT | `licenses/hiprt.txt` |
| Engine-free interval query and shader assembly from [pmndrs/pathtracer](https://github.com/pmndrs/pathtracer/tree/2489a09ab1844a7effe6e845ecccd05876580c9f/packages/raytracer), pinned at `2489a09ab1844a7effe6e845ecccd05876580c9f`. | RC layer in `dist/chunks/` | MIT, Dennis Smolek | `licenses/pathtracer-MIT.txt` |
| The raytracer's WGSL implementation of the equal-area square-to-sphere mapping, adapted from [pbrt-v4](https://github.com/mmp/pbrt-v4). Copyright 1998–2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys. | RC interval query shader source | Apache 2.0 | `licenses/pbrt-v4-Apache-2.0.txt` |

The RC layer imports the engine-free entry of `@pmndrs/bvh@0.1.0` under the npm alias `@pmndrs/rc-bvh`. Its distribution includes its own third-party notices.

three.js is not bundled: it is a peer dependency, installed by you, under its own MIT licence.

The repository's [THIRD_PARTY.md](https://github.com/pmndrs/gi/blob/main/THIRD_PARTY.md) also lists the test scenes and tools used to build and check this package. None of those ship in it.
