# Third-party code in @pmndrs/gi

@pmndrs/gi is MIT licensed (`LICENSE`). It bundles the code below, which comes under its own licences. Each licence's full text is in `licenses/`.

| What | Where in this package | Licence | Full text |
| --- | --- | --- | --- |
| Engine-free interval query and shader assembly from [@pmndrs/raytracer 0.1.0](https://www.npmjs.com/package/@pmndrs/raytracer/v/0.1.0). | RC layer in `dist/chunks/` | MIT, Dennis Smolek | `licenses/pathtracer-MIT.txt` |
| The raytracer's WGSL implementation of the equal-area square-to-sphere mapping, adapted from [pbrt-v4](https://github.com/mmp/pbrt-v4). Copyright 1998–2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys. | RC interval query shader source | Apache 2.0 | `licenses/pbrt-v4-Apache-2.0.txt` |
| [AMD Capsaicin](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin) v1.3 by Advanced Micro Devices (commit `914b91596cd119eda85fbc1d3c7ee6ac391b1452`): its GI-1.0/1.2 technique (screen probes, the hash-grid world cache) and math helpers, ported to WGSL and TypeScript (the radiance cache); its SSGI technique, ported to WGSL (the platform's screen pass) | The radiance-cache layer in `dist/chunks/` (`/radiance-cache`); the SSGI pass in the platform layer | MIT | `licenses/capsaicin-MIT.txt` |
| [rtoy-samples](https://github.com/h3r2tic/rtoy-samples) by Tomasz Stachowiak (commit `29597c168b2515ad242d5be55f78c597f6f5f62c`): `IntersectDirPlaneOneSided` and the near-field light, which Capsaicin's SSGI takes from its `ssgi.glsl` | The SSGI pass in the platform layer | MIT | `licenses/rtoy-samples-MIT.txt` |
| [Open Asset Import Library](https://github.com/assimp/assimp) (assimp): `FromToMatrix`, which Capsaicin's SSGI takes as `RotFromToMatrix` | The SSGI pass in the platform layer | BSD 3-Clause | `licenses/assimp-BSD-3-Clause.txt` |
| The hash-grid radiance cache of [pmndrs/pathtracer](https://github.com/pmndrs/pathtracer) (commit `dd771c815dbd80bd1c7dc78673746a7742c51618`), ported to WGSL and TypeScript (the radiance cache's `worldCache: 'sharc'` backend) | The radiance-cache layer in `dist/chunks/` | MIT, Dennis Smolek | `licenses/pathtracer-MIT.txt` |

The platform builds and refits its ray-tracing trees with `@pmndrs/bvh@0.1.1` (MIT, pmndrs; its engine-free entry, and `@pmndrs/bvh/worker` for the tinybvh workers), which the RC layer's interval tracer traces over too, and the radiance cache imports `Gi1Denoiser` from `denoiser/ffx` (`denoiser@2.1.0`, MIT, AMD FidelityFX ports). Both are npm dependencies, installed with this package, not bundled; their distributions include their own third-party notices. `@pmndrs/bvh` carries tinybvh 1.9.1 (Jacco Bikker, MIT) compiled to WebAssembly with its Emscripten runtime, the GPU H-PLOC builder ported from tinybvh and the GPU LBVH ported from AMD's HIPRT (MIT), with their notices in its `THIRD_PARTY.md` and their texts in its `licenses/`; none of that code is bundled here (pmndrs/gi#63).

The radiance cache's optional neural tail (`neuralCache: 'tail'`) loads `@pathtracer/mlp` (pmndrs/pathtracer, MIT) at run time if it can be found. It is neither bundled nor a dependency, and it isn't on npm.

three.js is not bundled: it is a peer dependency, installed by you, under its own MIT licence.

The repository's [THIRD_PARTY.md](https://github.com/pmndrs/gi/blob/main/THIRD_PARTY.md) also lists the test scenes and tools used to build and check this package. None of those ship in it.
