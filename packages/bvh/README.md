# @pmndrs/bvh

BVH builders for WebGPU that don't depend on any engine. You give it triangles, and it gives you a bounding volume hierarchy in one fixed GPU layout that a WGSL ray walk can read: built on the CPU (tinybvh in WebAssembly) or on the GPU (LBVH or H-PLOC), and refit on the GPU when things move.

DDGI (`@pmndrs/ddgi`) is its first user. The package imports nothing from DDGI, from GI code or from any engine, so picking, a path tracer or another GI technique can use it too.

**Status.** For now it lives in this repo and ships bundled inside `@pmndrs/ddgi`; it is planned as a standalone package later ([docs/design/bvh-package-split.md](../../docs/design/bvh-package-split.md)). ddgi-core uses the CPU builder, the GPU refit, and, since M2's "fast first GI" (T2.12), the GPU H-PLOC for the first trees of a scene as it loads: one batch for all meshes, its trees copied into the tracer's buffers after a small stats readback, then replaced mesh by mesh by tinybvh's from a worker (ddgi-core's `trace/gpu-mesh-trees.ts` and `trace/ray-queries.ts`). The LBVH isn't used by ddgi-core.

How the builders work inside: [docs/design/bvh-builders.md](../../docs/design/bvh-builders.md). The plan and its stages: [docs/design/bvh-framework.md](../../docs/design/bvh-framework.md).

## The builder contract

Every builder follows the same contract, so one can replace another without changing the code that traces rays.

- **In:** one mesh, or many: vertex positions (`x, y, z` floats) and, optionally, three indices per triangle and a flags word per triangle. CPU arrays (`TinyBvhInput`) work for every builder. The GPU builders also take buffers already on the GPU (`GpuLbvhBufferGeometry`), for example the output of a skinning pass.
- **Options:** the largest leaf (1 to 8 triangles, default 8) and, for the GPU builders, the SAH costs. H-PLOC has two more (see [bvh-builders.md](../../docs/design/bvh-builders.md#gpu-h-ploc-stage-b2)).
- **Out:** the tree in the shared GPU layout (`GpuLayoutMeshBvh` on the CPU, `GpuLbvhTrees` on the GPU):
  - 64-byte nodes with two child boxes each (the Aila–Laine layout), node 0 the root;
  - 48-byte triangles in tree order, each with its original index and flags;
  - one parent index per node, for refits;
  - the mesh's box, its depth and its counts.

  The word positions are in `src/gpu-layout.ts`.
- **Rule:** every builder must pass the same ray oracle (the `trace-oracle` GPU page, on the RTX 3060 and the Intel UHD) before it is used. `checkGpuLayout` also checks any tree against its mesh: every triangle bit for bit, every box exactly the box below it, parents, depth and counts.

## Which builder to use

| Path | What it is | Use it when | Needs |
| --- | --- | --- | --- |
| **tinybvh** (`createTinyBvhBuilder`) | tinybvh 1.9.1 compiled to WebAssembly, on the CPU. Its default builder (Bonsai, SIMD) makes the best trees here. | You want the best trees and can wait about a second or two (Sponza: 1.6–1.9 s). Run it in a worker: it is synchronous. It is also the fallback for everything else. | WebAssembly SIMD (every current browser) |
| **GPU H-PLOC** (`GpuHplocBuilder`) | H-PLOC (Benthin et al., 2024) in WGSL, ported from tinybvh's `BuildHPLOC`. **The default GPU builder.** | You need trees fast: at startup, or to rebuild a mesh that changed a lot. Its trees trace about as fast as tinybvh's. | `subgroups` |
| **GPU LBVH** (`GpuLbvhBuilder`) | A linear BVH in WGSL, ported from HIPRT. | The fastest build when tree quality matters less. Its trees trace about 20% slower. | `subgroups` |
| **Either, chosen for you** (`createMeshLayoutBuilder`) | H-PLOC when the device has subgroups, otherwise tinybvh. Meshes too big for the device, or refused by the GPU, go to tinybvh too. | The default entry point. Each result says which builder made it and why. | nothing |
| **GPU refit** (`GpuBvhRefit`) | Keeps a tree's shape and recomputes every box on the GPU from one box per primitive. | A top level over instances that move every frame. ddgi-core builds the instance tree's shape on the CPU in a worker and refits it with this. | `subgroups` |
| **BLAS refit** (`GpuMeshRefit`) | Refits one mesh's tree from a vertex buffer on the GPU. Works on any builder's tree. | Deforming meshes (skinned, morphed) whose triangles keep their connectivity. Rebuild when the tree's quality drifts. | `subgroups` |

**Without subgroups.** Every GPU path needs the WebGPU `subgroups` feature. Without it there is one fallback: build on the CPU with tinybvh. `createMeshLayoutBuilder` does that for you, and says so in `fallbackReason`. A test page (`bvh-lbvh-fallback`) covers it. The GPU code never assumes a subgroup size: it reads it from the device, and gives the same trees at 8, 16 or 32 lanes.

## Examples

These show the shape of each path. The full signatures are in the source files' comments.

**A device with subgroups when there are any:**

```ts
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
    requiredFeatures: adapter.features.has('subgroups') ? ['subgroups'] : [],
});
```

**Let the package choose** (H-PLOC on the GPU, or tinybvh):

```ts
import { createMeshLayoutBuilder } from '@pmndrs/bvh';

const builder = await createMeshLayoutBuilder(device); // compiles the GPU kernels once
console.log(builder.kind, builder.fallbackReason);       // 'gpu-hploc' and null, or 'tinybvh' and why

const results = await builder.buildLayouts([
    { positions: floorPositions, indices: floorIndices },
    { positions: chairPositions, indices: chairIndices },
]);
for (const { layout, builtBy, fallbackReason } of results) {
    // layout.nodes and layout.triangles are ArrayBuffers in the GPU layout: upload them.
}
```

This path copies the trees back to the CPU. To keep GPU-built trees on the GPU, use the builder directly (below).

**CPU, tinybvh** (in a worker):

```ts
import { createTinyBvhBuilder } from '@pmndrs/bvh';

const tinybvh = await createTinyBvhBuilder();
const { layout, buildMs } = tinybvh.buildLayout({ positions, indices }, { maxLeafPrimitives: 8 });
```

**GPU H-PLOC, trees left on the GPU:**

```ts
import { GpuHplocBuilder, planGpuHplocBatches } from '@pmndrs/bvh';

const hploc = new GpuHplocBuilder(device);
await hploc.prepare(); // compiles the kernels, asynchronously

// One batch builds many meshes with the same dispatches. A big scene may not fit
// the device's buffers in one batch: planGpuHplocBatches splits it.
const { batches, tooBig } = planGpuHplocBatches(device, meshes);
for (const batch of batches) {
    const trees = await hploc.buildMany(batch.map((index) => meshes[index]));
    // trees.nodes, trees.triangles and trees.parents are GPU buffers shared by the batch.
    for (const mesh of trees.meshes) {
        if (mesh.problem) continue; // refused: build this mesh with tinybvh instead
        // mesh.nodeOffset and mesh.triangleOffset say where its tree sits in those buffers.
    }
    // ... bind them to your ray walk; trees.destroy() frees them.
}
// Meshes in tooBig go to tinybvh.
```

A mesh can also come from a GPU buffer: `{ positionBuffer, vertexCount, indexBuffer, indexFormat: 'uint32', triangleCount }`. The buffers need `COPY_SRC` usage. `GpuLbvhBuilder` has the same API.

**GPU refit of a tree** (a top level over moving instances):

```ts
import { GpuBvhRefit, leafSlotsOf, parentSlotsOf } from '@pmndrs/bvh';

const refit = new GpuBvhRefit(device);
await refit.prepare();

// Once per tree shape, from a CPU copy of the tree:
const words = new Uint32Array(layout.nodes);
refit.setShape({
    nodeWordOffset: 0,
    nodeCount: layout.nodeCount,
    parentSlots: parentSlotsOf(words, 0, layout.nodeCount, layout.parents),
    leafSlots: leafSlotsOf(words, 0, layout.nodeCount),
    generation: 1,
});

// Every frame, after a pass wrote one box per primitive (8 floats: min x, y, z, unused,
// max x, y, z, unused), in the order the tree's leaves index them:
refit.encode(computePass, treeBuffer, primitiveBoxes);
```

**BLAS refit of a deforming mesh:**

```ts
import { GpuMeshRefit } from '@pmndrs/bvh';

const meshRefit = new GpuMeshRefit(device);
await meshRefit.prepare();
// Once, from a CPU copy of the tree (any builder's). For a tree inside shared buffers,
// pass where it sits: setTree(layout, { nodeWordOffset, triangleWordOffset }).
meshRefit.setTree(layout);

// Every frame, after the pass that writes the moved vertices:
meshRefit.encode(
    computePass,
    { nodes: nodeBuffer, triangles: triangleBuffer },
    { positionBuffer: skinnedPositions, vertexCount, indexBuffer, indexFormat: 'uint32' },
);
```

Both refits can report the tree's quality (`latestStats.relativeCost`, with `diagnostics: true`), so you know when to rebuild. Stats are read back a frame or two late; nothing else needs a readback.

**Check a tree:**

```ts
import { checkGpuLayout, treeSahCost } from '@pmndrs/bvh';

const problems = checkGpuLayout(layout, { positions, indices }); // [] when the tree is right
const cost = treeSahCost(layout);
```

## Measured numbers

Speed targets are for the RTX 3060 Laptop (and the M5 Pro, not measured yet). The Intel UHD runs every check for correctness only, with no speed target. All numbers are on headless Chrome 156 under Windows. The details, and how each run was taken, are in PROGRESS (stages B0b, B1, B2) and `bench/results/trace.csv`.

**Sponza:** 5,744,384 triangles in 132 meshes.

| What | Number | Conditions |
| --- | --- | --- |
| GPU H-PLOC build, whole scene | **46.6 ms** (target ≤ 50) | Median of the 2 runs that were valid under the clock rule (cool start, under 5% of samples thermally throttled, enough samples), 2026-11-05. Loaded SM clock about 1,000 MHz, power-limited. |
| CPU tinybvh build, whole scene | **1.6–1.9 s** | In a browser worker, single-threaded; medians of 3 runs per session, 2026-11-04. |
| H-PLOC tree quality | SAH cost **1.032×** tinybvh's | Lower is better. The LBVH's is 1.217×. |
| Trace speed on H-PLOC trees | closest-hit **0.95×**, any-hit **1.02×** of tinybvh's trees | Same rays, both tracers in each run, median of 3. The runs were flagged as low-clock, so the ratio is more reliable than the rays per second. |
| GPU LBVH build, whole scene | 31.2 ms | Median of 3, measured before the clock rule existed. Its trees trace at 0.82× (closest) and 0.80× (any) of tinybvh's. |
| GPU top-level refit, 10,000 moving instances | about **0.09 ms** of GPU time (0.086–0.091) | B1a's page, across two measurements. |

**Intel UHD (correctness only):** the oracle passes for tinybvh, LBVH and H-PLOC trees, with no rays slipping through shared edges. H-PLOC builds exactly the same trees on the Intel (8–16 lanes) as on the RTX (32 lanes).

## Licences

The package is MIT. It contains or ports code from:

- **tinybvh** 1.9.1 by Jacco Bikker (MIT): compiled into the `.wasm`, and its `BuildHPLOC` ported to WGSL and TypeScript. Text: `licenses/tinybvh.txt`.
- **HIPRT** by AMD (MIT): the LBVH kernels and Morton codes, ported to WGSL. Text: `licenses/hiprt.txt`.
- **Emscripten's runtime**, linked into the `.wasm`: Emscripten's code and musl (MIT), dlmalloc (public domain), LLVM libc++ and compiler-rt (Apache 2.0 with LLVM Exceptions). Text: `licenses/emscripten-runtime.txt`.

The `.wasm` carries its licence texts in a `/*! */` comment that bundlers keep, and the `@pmndrs/ddgi` build fails if a built file loses them. Sources and commits are in [THIRD_PARTY.md](../../THIRD_PARTY.md).

## Working on the package

- Unit tests: `pnpm test` (the CPU twins, the layout checker, the batch planner, the WGSL names).
- GPU pages: `pnpm test:gpu --filter bvh-` and `--filter trace-oracle`, on both GPUs.
- Rebuild the `.wasm` with `pnpm bvh:wasm`, or check it with `pnpm bvh:wasm --check`; [bvh-builders.md](../../docs/design/bvh-builders.md#rebuilding-the-wasm) has the toolchain.
- Port only MIT or Apache-2.0 code, and add its licence to `licenses/` and THIRD_PARTY.md.
