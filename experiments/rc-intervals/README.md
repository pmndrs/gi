# Split RC interval tracer experiment

This page runs six surface rays through a triangle BVH using the pathtracer's public `RtIntervalTracer`. A second GPU pass reads the first-hit distances and classifies contributions to four consecutive intervals. Both passes share one command encoder and one submission. A readback at the end compares the GPU results with the known fixture.

The scene includes a triangle at the ray origin, hits exactly at 1 and 4, a second blocker behind a first hit, an interior hit, a far hit and an environment escape. Each ray uses the open-low convention `(0, tMax]`. For the cascade ends 1, 4, 16 and the largest finite float, `T` means the ray crossed an interval, `H` means the interval owns the hit, and `-` means the ray stopped before reaching it. A miss contributes transmittance to all four. This experiment checks the boundary between tracing and deposition. Probe bins, irradiance, shading and history run in the separate demo.

## Run

Use a WebGPU-enabled browser and Vite from the GI development install. Set `RC_PATHTRACER_ROOT` to a checkout with interval queries, and `RC_BVH_ROOT` to a built BVH checkout. The pathtracer package also depends on `@pmndrs/downsampler`. If using an extracted checkout without its installed dependencies, set `RC_DOWNSAMPLER_ROOT` to a built downsampler checkout.

From the GI repository root in PowerShell, replace the example paths with your checkouts.

```powershell
$env:RC_PATHTRACER_ROOT = 'D:/src/pathtracer'
$env:RC_BVH_ROOT = 'D:/src/bvh'
$env:RC_DOWNSAMPLER_ROOT = 'D:/src/downsampler'
pnpm exec vite --config experiments/rc-intervals/vite.config.mjs
```

On macOS or Linux, export the same variables before running the Vite command. Open the localhost address Vite prints. The page checks all six cases and prints the failing ray, distance and cascade states on a mismatch. Use `pnpm exec vite build --config experiments/rc-intervals/vite.config.mjs` to build the standalone page.

## Verification

The production bundle passed on Windows in headless Chrome on an NVIDIA Ampere WebGPU adapter. The GPU returned the following distances and interval contributions.

| First hit | States |
| ---: | --- |
| 1 | `H---` |
| 1.5 | `TH--` |
| 4 | `TH--` |
| 6 | `TTH-` |
| 18 | `TTTH` |
| −1 (miss) | `TTTT` |

Tested sources were pathtracer `2489a09`, BVH `6163232` and downsampler `5a656c3`. The fixture only needs GPU distance output. Timing a six-ray dispatch does not characterize the cost of a full scene. [Pathtracer interval query implementation](https://github.com/pmndrs/pathtracer/blob/2489a09ab1844a7effe6e845ecccd05876580c9f/packages/raytracer/src/trace/interval-batch.ts)
