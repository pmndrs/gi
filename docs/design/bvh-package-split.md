# Splitting out @pmndrs/bvh (outline)

Status: **an outline, not a decision** (2026-11-06). What is decided (DECISIONS 2026-11-04): `packages/bvh` is its own engine-free package, kept in this repo for now and shipped inside `@pmndrs/ddgi`; Dennis's plan is to publish it later as `@pmndrs/bvh`, probably as `pmndrs/bvh`. This page lists what would have to be true first, what would move, and how ddgi would depend on it. The user-facing README is [packages/bvh/README.md](../../packages/bvh/README.md).

## Already engine-free

- `packages/bvh` imports nothing from gi-core, GI code, the adapters or any engine. Lint enforces it (`eslint.config.mjs`), and gi-core depends on it (`"@pmndrs/bvh": "workspace:*"`).
- It needs only WebGPU and WebAssembly: the `.wasm` is inlined as base64, so it loads the same way in a worker, on a page, in Node and in any bundler. It starts no workers itself.
- It has its own unit tests, GPU pages (`tools/gpu/pages/bvh-*`), licence texts (`licenses/`) and a reproducible `.wasm` build (`pnpm bvh:wasm --check`).

## Still in gi-core, though not DDGI-specific

A standalone package would be thinner than what users of a BVH expect, because these pieces live in gi-core today:

- **The layout's WGSL**: the `GiBvhNode` and `GiBvhTriangle` structs in `platform/trace/layout.ts`. The sizes and word positions already come from `packages/bvh/src/gpu-layout.ts`.
- **The ray walk**: `giTraceClosest` and `giTraceAny`, with the watertight triangle test. The oracle every builder must pass runs through it.
- **Shape to layout**: `buildMeshBvhFromTopology`, which turns a `MeshTreeTopology` into the GPU layout.
- **The top level's CPU shape builder** (`top-level-shape.ts`, in a worker), and the worker wrapper around tinybvh (`mesh-bvh-async.ts`).
- **The bench tooling**: `trace-bench`, `trace-oracle`, the bench lock and the CSVs.

The instance gather in `top-level-gpu.ts` reads DDGI's scene copy, so it stays.

## What must be true before the split

1. **A stable public API, reviewed after M2 wires it in.** M2's "fast first GI" item is the first real use of the GPU builders, and it will change the API. Known rough edges:
   - H-PLOC returns types named after the LBVH (`GpuLbvhTrees`, `GpuLbvhGeometry`);
   - `TinyBvhInput` doubles as the general mesh input;
   - the `./*` export exposes internals such as `USAGE` and `placeMeshes`;
   - the refits need a CPU copy of the tree (`setTree`), so a GPU-built tree has to be read back once.
2. **At least one user that isn't DDGI**, such as picking or a path tracer, built against the package in this repo.
3. **A comparison against three-mesh-bvh** on the package's own terms: build time, tree quality, trace speed, memory, and features. three-mesh-bvh also offers CPU raycasts, shapecasts and closest-point queries. What exists so far is from T0.10 and T1.3, on the tracer:
   - closest-hit 1.20× three-mesh-bvh on Sponza (RTX 3060);
   - any-hit 0.93–1.22×;
   - 302 MB of memory against 153 MB.
4. **Its own CI and test GPUs.** This repo has no CI today. The package needs, at least:
   - unit tests on every change;
   - `bvh:wasm --check` on Linux and macOS (only Windows builds have been compared);
   - the GPU pages run on an NVIDIA, an Intel and an Apple GPU before each release (a self-hosted runner or a written checklist).
5. **The M5 Pro run.** It is a performance target that hasn't been measured. It is also the first device where the browser may not report `subgroupMinSize`, which makes H-PLOC fall back to its slower setting (assuming 4-wide subgroups).
6. **Licences checked again.** All ported code is MIT (tinybvh, HIPRT). DECISIONS says to ask the authors of the unlicensed repos for a licence before shipping anything ported from them. Nothing is ported from them today; keep it that way.

## What moves and what stays

| Moves to `@pmndrs/bvh` | Stays in ddgi |
| --- | --- |
| Everything in `packages/bvh` | Probes, the GI platform, the scene copy and mesh table |
| The layout's WGSL structs | The instance gather (it reads the scene copy) |
| `buildMeshBvhFromTopology` | The update budget, the three.js adapter and the viewer |
| A generic top-level shape builder over boxes | DDGI benchmarks and references |
| Optional worker helpers for tinybvh | |
| The bvh GPU pages, `layout-identity` and `tree-quality` | |
| **To decide:** the ray walk and the oracle | |

**The open question is the ray walk.** If it moves, `@pmndrs/bvh` becomes "build, refit and trace", which is what picking and path tracers want, and the oracle travels with the builders it judges. gi-core would then import its walk from the package. If it stays, every user writes its own walk against the layout, and the oracle stays in ddgi. Moving it looks better, but it is Dennis's call, best made after M2.

## Versioning, and how ddgi depends on it

- **Until the split:** a private workspace package with source exports, bundled into `@pmndrs/ddgi`. Nothing changes for ddgi users.
- **After the split:** `@pmndrs/bvh` is published with semver, as `0.x` while the API moves. `@pmndrs/ddgi` takes it as a normal dependency (not a peer), so users install one package. Before 1.0 it pins a minor version (`~0.y.z`).
- **The GPU layout is part of the API.** Any change to it is a breaking change. Export a `GPU_LAYOUT_VERSION` constant, so ddgi can check at startup that its WGSL matches the package it loaded.
- **Repo:** either its own repository (`pmndrs/bvh`, with `packages/bvh`'s history carried over) or a second package in this one. Developing ddgi against a local checkout works the same either way (pnpm overrides or links).

## A possible order

1. M2 wires the GPU builders into gi-core (fast first GI), and the API settles.
2. API review: trim the exports, rename the shared types, and move the layout's WGSL (and the walk, if decided) into `packages/bvh` **inside this repo first**, so the new boundary is proven before any split.
3. A second user (picking) built on the package in this repo.
4. CI and test GPUs, the three-mesh-bvh comparison, and the M5 Pro run.
5. Split, publish `0.1`, and switch `@pmndrs/ddgi` to the published package.
