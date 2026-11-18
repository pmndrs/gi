# From Chandra to the upscaler

| Source | What to inspect |
| --- | --- |
| [entropylost/chandra at 9af00e8](https://github.com/entropylost/chandra/tree/9af00e8de135a4126cb137d1ec62b451bd6b8e37/src) | Author reference. Under `algorithm/`, `trace.rs` generates and traces rays, `rc.rs` splits and merges intervals, and `splat.rs` evaluates irradiance. `cache.rs` and `table.rs` implement probe storage. `algorithm/ddgi.rs` is the authors’ comparison path. |
| [zalo/three.js example at 5425061](https://github.com/zalo/three.js/blob/54250617f2724d034909080e576cfe3c6ab1a8b3/examples/webgpu_radiance_cascades.html) | Johnathon Selstad’s WebGPU example after Chandra. The PR describes one world-space ray per 2×2 pixel block by default, four cascades per world LOD, a static three-mesh-bvh, temporal probe accumulation and a coarse-to-fine merge. |
| [pmndrs/upscaler S5 at fa9c64f](https://github.com/pmndrs/upscaler/tree/fa9c64f2d996cbeadcfa8fefc670d3f332b3a204/examples/s5-radiance-cascades) | S5 vendors the Three.js addon and renders its G-buffer and GI at a selected internal resolution. Its upscale resolve uses depth and velocity from that G-buffer. It also provides a native TRAA comparison and switches between probe spacing in render or display pixels. |

The Three.js PR supports a moving camera, sun and material colors. Its BVH is built by `setScene()`, and hit shading reads material colors and emissive values. The demo uses Chandra’s flat-material Sponza conversion, credited to Crytek and Frank Meinl in the PR. A textured replacement needs corresponding ray-hit material sampling. A moving raster mesh needs a traceable representation at its current transform. The upstream PR reports 539 MB of probes in its default 1600×900 benchmark and up to nine storage buffers in one pass. Query adapter limits and measure actual allocations before adopting its resource layout. [Three.js PR and author measurements](https://github.com/mrdoob/three.js/pull/35349)

### Published S5 baseline

The S5 PR reports headless Chrome on Apple Metal at 1600×900, fixed GI quality, still camera and vsync disabled. Its comparison keeps the same output size. These are external measurements on that setup. [S5 benchmark and integration notes](https://github.com/pmndrs/upscaler/pull/106)

| Resolve | Internal size | Probe-spacing units | GI GPU ms | Upscale GPU ms |
| --- | --- | --- | ---: | ---: |
| Native TRAA | 1600×900 | Native pixels | 11.9 | — |
| Upscaler 2× | 800×450 | Display pixels | 5.4 | 0.6 |
| Upscaler 2× | 800×450 | Render pixels | 3.1 | 0.6 |

Use the display-pixel setting when testing reduced render resolution at comparable probe spacing. Use the render-pixel setting for a second, lower-cost configuration, recording the coarser sampling. The PR reports GI and upscale costs separately. Compare either full-frame GPU times or matching sets of passes.
