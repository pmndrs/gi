# Tracing design (T0.10 spike)

Status: **design for the spike.** The spike builds this, measures it against three-mesh-bvh, and the decision goes in `docs/DECISIONS.md`. Plain-language summary first, then details.

## In one paragraph

Each mesh gets its own bounding-volume hierarchy (a "BVH": a tree of boxes that lets a ray skip most triangles). It is built once, in a worker, using the standard "binned SAH" method. A second, small tree over all object instances is rebuilt on the CPU whenever something moves, which is cheap because there are only thousands of instances, not millions of triangles. On the GPU, a WGSL function walks the small tree, and for each object the ray might hit, it moves the ray into that object's local space and walks the object's tree. There are two versions of the walk: "closest hit" (what does this ray hit first?) and "any hit" (is anything in the way?). The second stops at the first blocker, which makes shadow rays cheap.

## What the spike must answer

1. **Is it correct?** The ported oracle cases pass on NVIDIA and Intel: thin walls, mirrored and non-uniform transforms, instancing, plus rays aimed exactly at shared triangle edges (they must not slip through).
2. **Is it fast enough?** Rays per second, closest-hit and any-hit, on Sponza and Bistro, ours vs the latest three-mesh-bvh, on NVIDIA, Intel and the Mac.
3. **What does it cost to build and hold?** Build time per mesh and per scene, top-level rebuild time, and GPU memory.

### How much speed we need (planning number, not yet a target)

An update budget of about 2 ms on the RTX 3060, with roughly half of it spent tracing, has to cover something like 200 probes × 256 rays, plus one shadow ray per hit for the sun. That's about 100k closest-hit rays and 100k any-hit rays per millisecond: **≈ 100 M closest-hit rays/s plus 100 M any-hit rays/s on Sponza**. If we're far below that, DDGI updates fewer probes per frame and settles more slowly. That is acceptable, but it changes the M1 plan, so the spike reports how far we are from this number.

## Data layout

Everything is plain arrays in a few storage buffers, so it survives a worker hand-off (transferable `ArrayBuffer`s) and an upload without conversion.

### Per-mesh tree ("bottom level")

- **Node: 64 bytes, two child boxes per node** (the Aila–Laine layout used by most GPU BVH2 tracers). One node read gives both children's boxes, so the walk tests both and visits the nearer one first without a second memory fetch.

  ```
  struct GiBvhNode {          // 64 bytes
      leftMin:  vec3f, leftIndex:  u32,
      leftMax:  vec3f, rightIndex: u32,
      rightMin: vec3f, leftCount:  u32,   // count > 0: child is a leaf with `count` triangles starting at index
      rightMax: vec3f, rightCount: u32,
  }
  ```
  A child with count 0 is an inner node at `index`. A leaf child stores the first triangle and the number of triangles.
- **Triangles: 48 bytes each, stored in tree order** (leaves point at contiguous runs, so no index indirection): `v0` + original triangle index, `v1` + flags, `v2` + pad. The three corners are stored as they are, so a vertex shared by two triangles has identical bits in both; the watertight test needs that (part A replaced the corner-plus-two-edges layout Möller–Trumbore used). The original index maps a hit back to the mesh's UVs and material group. The flags bit 0 marks "needs alpha test" for M4.
- **Leaves:** at most 4 triangles, fewer where the SAH cost says so.
- **Builder: binned SAH,** 16 bins per axis over centroid bounds, all three axes tried, leaf when splitting costs more than intersecting. It runs in a worker and returns node and triangle arrays. Large meshes must not freeze the page.
- **Deforming meshes (M3)** keep this tree's shape. Refitting bottom-up needs each node's parent, so the builder also writes a parent index per node (4 bytes per node, kept in a separate array that only the refit pass uses).

All meshes' nodes go in one node buffer, and all triangles in one triangle buffer, at per-mesh offsets.

### Instances and the top level

- **Instance record: 64 bytes.** It holds the world-to-object transform as a 3×4 matrix (48 bytes), the mesh's root node offset, its triangle offset, the platform instance index, and flags (contributes to GI, visible). Rays are moved into object space, so moving an object never touches its tree. Mirrored and non-uniform scales work because the hit distance is measured on the object-space ray, whose direction we deliberately *don't* renormalize, so `t` stays in world units.
- **Top-level tree:** the same 64-byte node layout, over instance world boxes. Leaves point at instance records. It is rebuilt on the CPU with binned SAH whenever a transform changes, and uploaded with `writeBuffer`. This is fast because instance counts are in the thousands. The spike measures it on Bistro and on a synthetic 10,000-instance scene. If it gets too slow, a GPU build is the follow-up.
- **Buffers:** instance records and top-level nodes share one buffer (records first). That gives three storage buffers for tracing: nodes, triangles, and top level. This matters because a default device allows only 8 per shader stage (see docs/api/platform.md).

### Size limits

A default device caps one storage binding at 128 MB. At 48 bytes per triangle that's about 2.7 M triangles in one binding. Sponza (about 262k triangles) fits easily. Bistro exterior (about 2.8 M) doesn't. For the spike, request a larger `maxStorageBufferBindingSize` through `requiredLimits` (the RTX 3060 and the Mac allow far more) and record what was needed. Splitting across bindings, so default limits still work, is M1 work if the spike shows we need it.

## The walk (WGSL)

- **One loop, one small stack per thread** (`array<u32, 64>` in private memory; three-mesh-bvh uses 60), shared by both levels. Entering an instance pushes a marker, switches the ray to object space and continues with that mesh's root. Popping the marker switches back to the world-space ray.
- **Nearer child first:** both child boxes are tested (slab test with a precomputed inverse direction). Both hit means push the farther one and go to the nearer one. Boxes beyond the current closest hit are skipped.
- **Closest hit** keeps the nearest `t`, instance, triangle and barycentrics, and whether the hit was on the front face. Back-face hits count as hits, because DDGI uses them to find probes inside geometry.
- **Any hit** returns at the first triangle hit within `[tMin, tMax]`.
- **Triangle test: watertight** (Woop, Benthin and Wald 2013), with per-ray shear rows computed once per ray. The design started with Möller–Trumbore, but it let up to 12% of edge-aimed rays slip (part A), so it was deleted in part C (DECISIONS 2026-11-04).
- **Stack overflow:** the builder reports the maximum depth, and the spike asserts that the top-level and mesh depths together fit in the stack. If 32 would always fit, the spike measures whether the smaller stack is faster, since private arrays cost registers.

## What part A found (2026-11-04)

- **Möller–Trumbore leaks at shared edges; the watertight test doesn't.** Two million rays were aimed exactly at shared triangle edges. With Möller–Trumbore, 11.6% slipped through on the RTX 3060 and 3.0% on the Intel GPU. Rays into the corners of a sealed thin box also leaked (354 of 6,000 on NVIDIA). The watertight test (Woop, Benthin and Wald 2013) let none through on either GPU.
  - Watertight needs bit-identical shared vertices, so it stores `v0, v1, v2` (still 48 bytes) instead of `v0` plus edges.
  - Both tests are a build option for the spike. Part B times both, and the slower one is deleted after the decision; production shaders keep one.
- **Buffers can't hold `bool`,** so batches write `GiHitRecord` with a `u32` front-face flag (contract updated).
- **Other small changes:**
  - Mesh triangle indices are mesh-relative.
  - The top-level buffer starts with a 64-byte header.
  - Instances carry a `mirrored` flag, so front-face stays correct under mirroring.
  - A single-triangle tree has an empty right child.
- **Budget:** the trace module binds 3 storage buffers, and a batch kernel 5 of the 8 allowed. The worst-case stack depth is 19 of 64 on the oracle scenes; the busiest ray used 12.

## What part B found (2026-11-04)

Measured on the RTX 3060 Laptop and Intel UHD, Chrome 156, GPU timestamps. Rows are in `bench/results/trace.csv`; the summary is in PROGRESS.

- **Speed, closest hit on Sponza** (ours Möller–Trumbore / ours watertight / three-mesh-bvh):
  - RTX 3060: 111 / 61 / 87 Mrays/s.
  - Intel: 10.0 / 8.0 / 5.3 Mrays/s.
  - Möller–Trumbore beats three-mesh-bvh on both GPUs. Watertight does too on Intel, but on the RTX 3060 it is 0.70×, below the bar.
- **Watertight costs too much on NVIDIA.** It costs 45% of the speed there but only 20% on Intel, which points at how it is written rather than the method. The likely cause is that the walk indexes vectors by a per-ray axis (`a[kx]`, `a[kz]`). Indexing a vector by a runtime value can push it into slow per-thread scratch memory. The fix is to build three per-ray "row" vectors once, so each corner is transformed with dot products and no runtime indexing. That's the same arithmetic, so it stays watertight.
- **Any hit:** three-mesh-bvh beats ours on Bistro interior and the furnished room. Our any-hit walk probably still visits the nearer child first and keeps narrowing the ray, which it doesn't need to do.
- **Build and memory:** our build is about 4× slower and uses about 2× the memory. Sponza takes 28 s, which is too slow for a scene load. The memory comes from the 64-byte nodes and the 48-byte triangles, which copy vertices instead of indexing them.
- **Top level:** it rebuilds in 1–10 ms on the real scenes and 86–121 ms for 10,000 instances. That's fine while objects move rarely, but a scene with thousands of moving instances would need an incremental or GPU build (M3).
- **Correctness under load:** with probe-style rays on all four scenes, no ray hit in one tracer and missed in another.

## What part C found (2026-11-04)

- The NVIDIA gap was one thing: indexing a vector with a per-ray value (`a[kx]`) in the watertight test. Building per-ray rows once with `select()` took Sponza from 61 to 101 Mrays/s on the RTX 3060.
- Reading nodes and triangles as `vec4f` arrays added 3% on the RTX 3060 and 23% on Intel.
- An unordered any-hit walk and a 32-entry stack bought nothing.
- Final (median of 3, Sponza closest-hit): 104.3 Mrays/s on the RTX 3060 (1.20× three-mesh-bvh), 12.8 on Intel (1.97×). The rule below picks our tracer (DECISIONS 2026-11-04).
- three-mesh-bvh trees converted to our layout trace at 96.7 Mrays/s and take 7.4 s to build, so tinybvh (101.6 Mrays/s, 5.3 s) is the better CPU builder.

## Benchmark

- **Rays:** probe-style rays. A grid of probes fills the scene bounds, and each probe fires 256 rays in random directions from a fixed seed. Closest-hit rays use those directions. Any-hit rays go from each closest hit toward the sun (or a fixed direction), which is what DDGI's shadow rays look like. The rays are generated once and stored in a buffer, so every tracer gets identical input.
- **Timing:** GPU timestamps around the trace dispatch, median of 20 runs after 5 warm-ups. On devices without timestamps, use a fence-based wall time over 20 dispatches, labelled as such.
- **three-mesh-bvh:** v0.9.18 (3 Nov 2026), `three-mesh-bvh/webgpu` (`BVHComputeData`). It only runs inside three's node system, not on a bare device. So its side of the benchmark is a three `WebGPURenderer` compute node that reads the same ray buffer through a storage node, with its `raycastFirstHit` and `raycastAnyHit` functions timed by GPU timestamps around that compute pass. The extra three overhead is outside the timed pass, but it gets noted. Build its per-mesh trees with `{ strategy: SAH }` explicitly, because its default is CENTER, which would make it look slower than it is. Both sides use normalized directions.
- **Record:** one row per run in `bench/results/trace.csv`, plus build time, top-level rebuild time and memory per scene.

## Decision rule (set before measuring)

- **Use our tracer as designed** if it passes every oracle case and its closest-hit throughput on Sponza is at least 0.8× three-mesh-bvh's on both GPUs. three-mesh-bvh does have a two-level structure (clusters under a top level), any-hit and nearest-first. But it needs three's node system, so the engine-free core can't use it. It also bakes skinned meshes on the CPU and rebuilds fully on geometry change, so it has no GPU refit. A small speed gap is worth keeping the core engine-free and having refit.
- **Below 0.8×:** adopt three-mesh-bvh's node packing and traversal ideas into our tracer (it is MIT), re-measure, and say which ideas made the difference. SPEC already settled "learn from it, don't depend on it", so this is about layout, not about taking a dependency.
- **Far below the planning number** (under 30 M rays/s on the RTX 3060): flag it to Dennis before M1, since it changes probes-per-frame and settling time.
