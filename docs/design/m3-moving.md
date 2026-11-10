# M3 design draft: moving objects and the camera

Status: **approved by Dennis 2026-11-06 and merged into main.** Written 2026-11-05, revised after an Opus review and again after Dennis's answers the same day (listed at the end). It is the input for M3's breakdown (the last task of M2). Rigid moving objects (T3.1 and T3.4, rigid parts) and the startup lane (T2.12–T2.14) run in M2; see [TASKS.md](../../TASKS.md). Dennis also approved the H-PLOC rebuild on drift (section 3.1; DECISIONS 2026-11-06).

It builds on [ddgi.md](ddgi.md), [bvh-framework.md](bvh-framework.md), [platform.md](../api/platform.md), [three.md](../api/three.md), the B1 hand-over (branch `bvh-b1`: `packages/bvh/README.md` and its PROGRESS entry) and the M2 draft, [m2-lights.md](m2-lights.md). M3 reuses M2's per-probe history and GPU scheduler (m2-lights.md section 4) for moved geometry; if M2 changes them, this page follows.

## In one paragraph

Moving geometry already works at the tracing level. The top-level tree is refit on the GPU every frame and rebuilt in a worker (B1a), and nothing recompiles when instances come and go. M3 makes the rest of the chain fast and complete:

- **Trees.** New and edited meshes get trees built on the GPU (B1's LBVH, later B2's H-PLOC) in the frame they arrive, and tinybvh's better trees replace them when the worker finishes. The same machinery gives first GI about 2 s after a scene is handed over.
- **Deforming meshes** keep their tree's shape and are refit on the GPU from positions the adapter supplies.
- **Probes.** Probes near a moved object get their history shortened and go to the front of the update list, exactly as for a light change. The size of the change comes from how far the object moved and how much of the probe's view it covers. Probes that were switched off come back warmed, never black.
- **The camera-following volume** keeps its probes fixed in world space and reuses the planes left behind for the planes entering ahead (RTXGI's "infinite scrolling"). New planes fade in, so the edge never pops. Several volumes share one atlas and blend at their edges.
- **The three adapter** finds additions and removals from three's own events, and sends many transforms in one call.

## Words used here

- **Mesh tree / top-level tree:** each mesh has its own BVH (a tree of boxes around its triangles); a small top-level tree over all instances points into them. Other engines call these BLAS and TLAS.
- **Refit:** recompute a tree's boxes for new vertex positions, keeping its shape. **Rebuild:** make a new shape.
- **SAH cost:** a standard estimate of how expensive a tree is to trace; lower is better.
- **Solid angle (Ω, in steradians, sr):** how much of a probe's view an object covers. The whole sphere is 4π ≈ 12.6 sr.
- **Toroidal addressing:** storage that wraps around. When the window moves one step, the slot that falls off one end is reused for the new step at the other end, so nothing else moves. Clipmaps (Tanner et al. 1998) use it for textures.
- **History length, ε, K, owed updates, the scheduler:** as defined in m2-lights.md ("Words used here", section 4).

## What M3 must deliver (SPEC)

| Target | RTX 3060 and up, Apple M5 Pro (1440p) |
| --- | --- |
| Object move response (a cabinet move, a door closing) | within 5% of settled in ≤ 1 s |
| Closed door | no leak: dark-room light ≤ 1% of the lit room's, and none visible |
| Recompiles | the counter stays at 0 through every scenario |
| Adapter cost | ≤ 0.2 ms a frame still, ≤ 0.5 ms with one object moving |
| Camera-following volume | scrolling never shows as a seam or a pop |
| Deforming meshes | cast moving occlusion into the GI |

M1's still-scene checks and M2's light checks still pass.

## What others do, and what we copy

| Who | How moved geometry reaches the GI | What we take |
| --- | --- | --- |
| **RTXGI-DDGI** (Majercik et al. 2021; SDK `DDGIVolume.md`) | Rays see moved geometry at once (the app rebuilds or refits its hardware trees). Probes aren't told anything moved: the 0.97 hysteresis and the per-texel change test catch up. Classification and relocation re-run from the fixed rays each update. **Infinite scrolling volumes** keep probes fixed in world space; when the active area moves, only the plane of probes that wrapped to the far edge is cleared (`DDGIClearScrolledPlane`, `ProbeBlendingCS.hlsl`). | Fixed rays re-checking probe states every update (we do this already, T1.5b). **Scrolling with plane resets.** Blending volumes by priority. |
| **Lumen** (Wright et al., SIGGRAPH 2022 Advances) | Rigid instances carry precomputed distance fields and surface-cache "cards" that move with them; caches near changes are refreshed within a budget, by priority. Deforming meshes need hardware ray tracing. | Trees that move with their instance (our two levels). **Budgeted, prioritized refresh near changes** (M2's scheduler). |
| **Godot SDFGI** ([docs](https://docs.godotengine.org/en/stable/tutorials/3d/global_illumination/using_sdfgi.html)) | Camera-following cascades; dynamic objects don't block or bounce light. | Confirms camera-following is the norm. We go further on movers. |
| **EA SEED GIBS** (Halén et al., SIGGRAPH 2021) and **GIBS2** (Apers, SIGGRAPH 2024 Advances) | Surfels are attached to objects (and their skin), so they move with them; probes with DDGI-like octahedral data cover the rest. | Movers must be *found*, not waited out. No surfels in v1. |
| **AMD Brixelizer** ([GPUOpen](https://gpuopen.com/fidelityfx-brixelizer/)) | Sparse distance-field bricks around the camera, rebuilt within a budget for dynamic instances. | Camera-centred, budgeted rebuilds. |
| **Enlighten** (Martin and Einarsson, SIGGRAPH 2010 Advances) | Geometry is static; dynamic objects only receive light from probes. | What we have to beat: SPEC's door and cabinet cases need movers to block light. |
| **Clipmaps** (Tanner, Migdal, Jones, SIGGRAPH 1998) | A fixed-size window over a larger virtual texture, addressed modulo its size, updated only at the edge it moves toward. | The addressing for the scrolling grid. |
| **BVH upkeep** (Wald, Boulos, Shirley, "Ray Tracing Deformable Scenes Using Dynamic Bounding Volume Hierarchies", TOG 2007; LBVH: Karras 2012, Apetrei 2014; H-PLOC: Benthin et al., HPG 2024) | Refit while the shape stays similar; rebuild when quality drifts; GPU builders make rebuilds cheap. | Refit for deforming meshes (SPEC's tracing decision: keep the shape). GPU builds for new meshes. |

## 1. The change feed (platform side)

What already exists: `setTransform`, `createInstance` and `destroyInstance`, `createMesh`, `updateMesh` and `destroyMesh`, and `setDeformedPositions` (which throws "arrives in M3"). The change log writes two regions per moved instance (old box and new box), with the instance as `source`. Tables have fixed capacity and grow by reallocating; nothing recompiles.

M3 adds:

- **`setTransforms(instances: Uint32Array, matrices: Float32Array)`** (a host-facing addition; Dennis: yes, 2026-11-05). B1a measured 10,000 `setTransform` calls at 18–20 ms, about 2 µs each, all bookkeeping per call. A batch call writes straight into the instance mirror and logs one region list. Target: 10,000 transforms in ≤ 2 ms.
- **The move distance per instance:** `SceneChanges` regions from a move carry how far the box's corners moved (`moved`, in metres). `ddgi/mark` uses it (section 5.1). Technique-facing.
- **`SceneChanges.receivers: readonly Aabb[]`** in place of the `movedReceivers` count (technique-facing). Moving objects that don't contribute to GI (characters, small props) still need nearby probes awake (section 5.3).
- **`deformBounds`** on a `deforming: true` mesh: a box, in mesh space, the deformed shape stays inside. The platform can't read GPU-supplied positions on the CPU, so this is what it logs as the mesh's region each frame.
- **`setDeformedPositions`** as the contract says: a same-device `GPUBuffer`. Also a CPU `Float32Array`, which the platform uploads into its own buffer (Dennis: yes, 2026-11-05). *Why the CPU form:* three r187 gives no supported way to get at a GPU buffer it owns (section 7). It's a host-facing addition beside SPEC's tracing decision, which describes positions arriving "as a GPU buffer, so there's no rebuild and no CPU work", so the M1 lead records it in DECISIONS.
- **A recompile counter** in `gi.stats`: compute pipelines created after `isReady`. The viewer shows it next to three's own program count. With T2.14, every kernel a configuration needs is compiled in `prepare`, so nothing compiles on first use and the counter means what it says.

*As built (T2.M1, rigid parts):* `setTransforms`, `DirtyRegion.moved` (the farthest corner's travel, from a per-instance world frame) and `.instance` (the slot), and `GIStats.recompiles` (counted at a view of the host's device that the platform's own code and its techniques use, while `isReady`). `receivers`, `deformBounds` and `setDeformedPositions` stay M3. ddgi.md "Moving geometry" has the details and numbers.

*As built (T3.1, prototype):* `SceneChanges.receivers` (boxes where receivers are now: created, shown, moved or deformed; 256 a frame, then one merged box), `MeshDescriptor.deformBounds`, and `setDeformedPositions` with a `Float32Array` (`platform/scene/deformed-positions.ts`). Until T3.3, a pose reaches rays by a tree rebuild, as `updateMesh` does: the positions go into the mesh's scene-copy range in place (no layout change), the mesh is reported uploaded, and new poses wait until the trees being built are placed (`GIPlatformImpl.treesWaitingAfterUpdate`), so a mesh posed every frame isn't starved of trees. Its region carries `moved` (the largest vertex travel × the instance's largest scale) and `instance`. The GPU-buffer form throws until T3.3.

## 2. Per-mesh trees: GPU first, tinybvh as the refinement

### 2.1 What exists

- **B0 (tinybvh, WASM worker):** the best trees (lowest SAH cost). Sponza takes 3.9 s in the worker.
- **B1 (GPU LBVH, branch `bvh-b1`, being merged into `m1`):** Sponza's 5.74 M triangles in 31.2 ms on the RTX. Its trees cost 1.22× tinybvh's SAH and trace at 0.82× (closest hit) and 0.80× (any hit). It builds every mesh of a batch with the same dispatches, in exactly the tracer's layout. It needs subgroups; without them `createMeshLayoutBuilder` uses tinybvh.
- **What B1 still needs from the CPU:** node counts, depth and "refused" flags come back through a stats readback, and `GpuMeshRefit` takes a tree's shape from a CPU copy of its layout (`setTree`, via `readLayouts`). **It isn't wired into `ddgi-core` yet;** the platform builds with tinybvh only.
- **B1a top level:** refit on the GPU every frame; the shape rebuilt in a worker when an instance is created or quality drifts (DECISIONS 2026-11-04). New instances appear when the worker's shape arrives (about 55 ms at 10,000 instances, less for small scenes).
- **B2 (H-PLOC):** target trees within 1.15× tinybvh's SAH and trace speed ≥ 0.9× B0's. Dennis (2026-11-05): B2 is done **before M3 sign-off, not before M3 starts**. It must not block anything, and the fallbacks (tinybvh, LBVH) always stay. (DECISIONS 2026-11-04 had "B1–B2 before M3"; the M1 lead records the change.)

### 2.2 The plan: GPU trees, with one small readback

The platform's habit is **"prefer GPU-resident; read back when it clearly pays"**. Dennis (2026-11-05): "This isnt a rule but a preference". Here it pays. A GPU build's node count, depth and "refused" flag are a few bytes per mesh, and knowing them on the CPU saves a large temporary memory pool and a GPU-side bookkeeping scheme.

- **Build:** B1's LBVH builds a batch of meshes into its own output buffers, as it does now.
- **Read back the stats only** (asynchronously, `mapAsync`, a few frames later; never the trees themselves): each mesh's node count, depth, and whether its tree was refused. It's what B1's `finish` reads today.
- **Then, on the CPU:**
  - a tree that's usable (not refused, depth within the tracer's 64-entry stack) gets an exact range in the tracer's node and triangle buffers;
  - one `copyBufferToBuffer` per mesh moves it there;
  - the mesh table's root offset flips to it.

  A refused or too-deep tree is never used; that mesh waits for tinybvh.
- **The cost of the readback** is latency, not frame time: a GPU tree reaches the tracer 1–3 frames after it's built. For first GI that's about 50 ms more.
- **tinybvh still builds every mesh in its worker,** as now. When its tree arrives, it's uploaded into a new range, the mesh table flips to it, and the LBVH range is freed a frame later (after the GPU is done with it). The tracer sees a different tree over the same triangles; watertight hits agree up to float rounding, so the field doesn't notice. During a swap one mesh briefly has two trees.
- **When B2 lands** and its trees trace at ≥ 0.95× tinybvh's, the swap can become optional (`meshTreeRefinement: 'tinybvh' | 'none'`), which saves the worker's copy and time.
- **Without subgroups:** tinybvh only, as in M1 (the declared fallback). First GI waits for the worker on those devices.
- **The top level** stays as B1a decided. If "add a cabinet" shows the worker's delay, the GPU LBVH could build the top level in the same frame from the instance boxes. It's the same code, but not planned.
- *As built (T3.2):* the builder is H-PLOC (B2), not the LBVH, at load and at runtime alike: T2.12's path, with the "scene as first loaded" limit removed. Each frame's uploaded meshes go in one batch with its own submit and stats readback (DECISIONS 2026-11-09). The GPU tree's node and triangle ranges are released in the frame that places tinybvh's tree, and the batch's own buffers one update after the copy. tinybvh builds one version of a mesh at a time; a mesh replaced again meanwhile keeps only its newest version waiting. Page: `trace-runtime-trees`.

### 2.3 First GI from GPU trees (scheduled in M2)

DECISIONS 2026-11-05 (provisional): first GI from B1's GPU trees, swapping to tinybvh's when the worker finishes, plus the other startup levers (the scene copy off the main thread, the cold compile; file loading). These are rows T2.12–T2.14 below, run in M2.

**The target** (Dennis, 2026-11-05): **≤ 2 s from the moment the host hands the scene to GI, with a cold shader compile.** File loading is the host's and isn't part of it, so the viewer's file loading moves to the Later list.

Sponza from load today (T1.9a, cold compile, first GI drawn): 10.8 s, of which files 4.2 s, mesh trees 3.3 s, main-thread scene copy 1.2 s and shader compile 1.2 s. The remaining 0.9 s (the first frame, the first warm-up step, three's own work) isn't broken down yet.

| Part | Today | After T2.12–T2.14 (estimate) | How |
| --- | --- | --- | --- |
| Files | 4.2 s | 4.2 s | The host's; outside the target |
| Mesh trees | 3.3 s | ~0.3 s | GPU LBVH: 31 ms GPU; B1's first build took 302 ms wall, including uploads and its stats readback; plus 1–3 frames until the trees are copied in |
| Scene copy | 1.2 s, main thread | ~0.15 s | Raw upload plus GPU packing (section 2.4); done in one go at load, not spread over frames |
| Compile | 1.2 s | ~0.7 s | Only what's configured; the sky kernel fix (T2.6); the LBVH's own kernels added (not yet timed) |
| Not broken down | 0.9 s | 0.9 s | T2.12 breaks it down |
| **From load** | **10.8 s** | **~6.2 s** | |
| **From "scene handed to GI" (the target)** | **~6.6 s** | **~2.0 s** | |

So Sponza lands near the 2 s target, with no margin: the 0.9 s not yet broken down and the LBVH's compile time decide it. T2.12 measures both first.

### 2.4 The scene copy off the main thread

The 1.2 s (Sponza) and 4.1 s (Bistro exterior) are JS loops on the main thread, one mesh at a time: octahedral normals, 16-bit indices, interleaving. A worker would still need the arrays copied in and the results copied back, followed by a `writeBuffer` from the main thread (the device lives there; the host owns it). The GPU can do the packing instead:

- **Upload the raw arrays** (positions, normals, uvs and indices, as the host gave them) with `writeBuffer` into a staging buffer. At load this happens in one go. For meshes added later it happens a few meshes per frame, at most `uploadMsPerFrame` (default 4 ms) of main-thread time. *As built (T2.13):* there is no per-frame budget; a mesh added later is staged in its `createMesh` call and packed by the next `encode`.
- **`gi/pack-mesh`** (compute, one thread per vertex, then one per triangle) writes octahedral normals, narrows indices to 16 bits where they all fit, and lays them out as `mesh-packing.ts` does today.
- **The CPU packer stays as the test oracle.** Positions and indices must match it byte for byte; normals within one octahedral step (the GPU encodes in f32, the CPU in f64).
- **The floor** is the raw copy itself. NOTES: reading 300 MB in JS takes about 110 ms on this laptop, so Sponza's ~180 MB of raw arrays costs about 60–100 ms of main-thread copying however it's done. The tinybvh worker's own copy adds to that.

**As built (T2.13).** `gi/pack-mesh` (`platform/scene/pack-mesh-wgsl.ts`, `gpu-mesh-packing.ts`):

- *When the copy happens:* `createMesh` and `updateMesh` write the host's arrays into staging buffers straight away, once `prepare` has asked for the kernel (`PlatformScene.prepareGpuPacking`, the first compile `prepare` asks for). So the host may still reuse its arrays after the call, and a scene copied in while preparing uploads alongside the compiles (DECISIONS 2026-11-08's revisit note). The next `encode` packs.
- *One dispatch per staging buffer, not per mesh:* the CPU writes a job table, one job per part (copy, normals, narrow or widen), one thread per output word, and a thread finds its job by binary search. Parts already in packed form (positions, uvs, slots, indices of the right width) are copy jobs.
- *The CPU packer stays,* split into `planMesh` (checks, box, layout and slot words, always on the CPU) and `packMeshWords` (the reference, and the fallback for a scene that never prepares and for a mesh too large to stage). The vertex buffer's layout is unchanged.
- *Tree builders* get the scene's own copies of positions and indices (`UploadedMesh.geometry`, in the host's index type), made at `createMesh`, instead of views of packed words.
- *What the estimate missed:* packing was not most of the main thread's copy time. Most of it was the three adapter's texture averages (texture-average.ts), and what is left after packing moves to the GPU is close to the floor above plus the checks. PROGRESS (T2.13) has the breakdown.

## 3. Deforming meshes (mesh-tree refit)

SPEC's tracing decision: deforming meshes "keep their BVH shape", refit every frame on the GPU, "no rebuild and no CPU work". The plan follows it.

- **Shape:** a deforming mesh's tree is tinybvh's tree of the rest pose. It's built on the CPU, so `GpuMeshRefit.setTree` gets its layout exactly as B1 built it.
- **Per frame:** `GpuMeshRefit` (B1) rewrites the mesh's tree-ordered triangle records from the positions and refits its boxes bottom-up. The top-level refit, which already runs every frame, picks up the new mesh box. Everything stays on the GPU.
- **The scene copy's positions too.** Hit shading rebuilds the hit point from the scene copy's positions (`giSurfaceAt` → `giInstanceTriangle`), not from the tracer's records. So the same kernel also writes that mesh's positions into the scene copy's vertex buffer.
- **Normals:** the scene copy's octahedral normals belong to the rest pose. A deforming mesh shades with each triangle's geometric normal, worked out from the fresh positions. At probe resolution that's invisible on foliage; an optional normals buffer can come later.
- **How much it moved:** the triangle kernel also writes the frame's largest vertex movement since the last frame (an `atomicMax`) into a GPU word. `ddgi/mark` reads it for the change size (section 5.1). The CPU logs `deformBounds` as the region.
- **Quality:** refits drift from the shape the tree was built for. A sway stays near its rest pose; a bend held for a minute doesn't. Section 3.1 proposes rebuilding with H-PLOC when the refit's cost passes 1.25× (a DECISIONS entry, because it departs from SPEC's "keep the shape"). T3.0 measures trace-speed decay on the wind tree.
- **Without subgroups:** B1a's refit uses subgroups for its pairing step. On such devices `isSupported` reports deforming meshes as unsupported, and they're traced in their rest pose with a one-time warning (the declared fallback; no slower second path).
- **Cost estimate:** B1a's top-level refit is 0.086 ms for 10,000 instances. A mesh refit also rewrites triangles and positions, so a 100,000-triangle tree should be about 0.1–0.2 ms on the RTX (not yet timed). Planning number: ≤ 0.3 ms a frame for the wind tree.

*As built (T3.3, prototype):* each deforming mesh has a `GpuMeshRefit` (shared kernels, `GpuMeshRefitKernels`; its own parameters and topology), in `trace/deforming-trees.ts`. A pose (CPU `writeBuffer`, or one GPU copy by `gi/pose-copy` for a buffer or texture) lands in the scene copy's positions, and the refit reads the scene copy (positions and indices) in the same encoder, before anything traces; a tree placed (GPU first tree, tinybvh's swap, or a drift rebuild) is refit at once. The refit writes only the corners' xyz words, so the flags word (7) survives. The top level keeps the mesh's deform box. The deforming meshes' GPU first trees are built in a batch of their own and read back (their layout; the refit needs the shape). Normals: a deforming mesh is stored without normals, so `giInstanceNormalAt` falls back to the posed triangle's geometric normal. The movement word was not built: GPU poses report the deform box's diagonal as `moved` (an upper bound), CPU poses the CPU's exact travel; a GPU word read by `ddgi/mark` is left for when a GPU pose's change size matters. Held drift: the refit's stats (relative SAH) are copied at most once every 15 frames per posed mesh and read back a few frames late (DECISIONS 2026-11-10); above 1.1× the built tree's for 60 frames, H-PLOC rebuilds from the scene copy (its own submit, then its stats and layout read back), at most one a frame, never while the mesh's first trees are still coming.

### 3.1 The BVH under motion: what's built, and the gaps

**Built (on `m1`):**

- **Rigid movers: the top level is refit on the GPU every frame** (B1a, in `ddgi-core`'s tracer, `ray-queries.ts` and `top-level-gpu.ts`): 0.084 ms for 10,000 moving instances on the RTX. Its quality is tracked without a GPU readback: every 250 ms a worker computes the same relative SAH cost (`refitOnCpu`) from transforms the main thread packs for it (about 1.3 ms for 10,000 instances), and past 1.25× the built cost the top level is rebuilt in the worker (`top-level-async.ts`) and swapped in. The counters show `quality rebuilds`. 1.25× suits the top level because it's a small share of trace time.
- **Deforming meshes: the mesh-tree refit** (`GpuMeshRefit`, B1, in `packages/bvh`): it rewrites a mesh's triangle records from a vertex buffer and refits its boxes, with the same relative-cost signal (`latestStats.relativeCost`, behind `diagnostics: true`). Section 3 above plans its use.
- **GPU builders:** the LBVH (B1) and H-PLOC (B2, the default GPU builder: Sponza's 5.74 M triangles in 46.6 ms, SAH 1.03× tinybvh's, trace speed 0.95–1.02×). Neither is wired into `ddgi-core` yet; that's the M2 item "fast first GI" (DECISIONS 2026-11-06 keeps it, as a lower item).

**Gaps for moving objects:**

1. **Rebuilding a deforming mesh's tree when its refit drifts.** SPEC says deforming meshes keep their tree's shape, and section 3 planned no rebuilds. A sway stays near its rest pose, but a character's arm raised for a minute, or a cloth falling, doesn't. Approved by Dennis 2026-11-06 (DECISIONS, `dyn-blas-rebuild`, recorded as an exception to SPEC):
   - **The signal:** with the CPU form of `setDeformedPositions` the CPU has the positions and can compute the relative cost itself, like the top level. With the GPU-buffer form it needs the refit's stats: one small asynchronous readback a few frames late. Dennis called readbacks a preference, not a rule ("prefer GPU-resident; read back when it clearly pays"), but DECISIONS still says no GPU readbacks by default, so this readback goes in the same DECISIONS entry.
   - **The rebuild:** past **1.1×** the built cost (a mesh tree is most of a ray's walk, unlike the top level, so its cost tracks trace speed closely; 1.1× matches the done-when below), rebuild that mesh's tree with H-PLOC on the GPU from the current positions, and swap it in when it's done, as the top level does. Scaling Sponza's 46.6 ms by triangle count, a 100,000-triangle mesh is roughly 1–2 ms with H-PLOC's fixed passes. At most one such rebuild starts per frame.
   - The swap needs the new tree's layout on the CPU for the refit (`GpuMeshRefit.setTree`). `readLayouts` gives it, also a readback, once per rebuild.
2. **Wiring the GPU builders for deforming meshes.** Today a deforming mesh's rest-pose tree comes from tinybvh on the CPU (section 3). With H-PLOC wired in (the M2 item), the rest-pose tree is built on the GPU too, and gap 1 reuses the same builder.
3. **Rigid objects that are added or removed** need their mesh tree before they're traced. That's 2.2's GPU-first plan (an object traces as soon as its LBVH or H-PLOC tree exists, and tinybvh's replaces it later).

**What "done" means for the BVH side:** a 60 s wind-tree run and a 60 s character-like bend (a test mesh folding its arm through 90° and holding it) keep trace speed within 10% of a fresh build, with at most one rebuild per second (counter), and the oracle passing mid-swap.

## 4. Several volumes, and the camera-following volume

### 4.1 Several volumes in one technique

- **Settings:** `volumes: VolumeSettings[]` (at most 8), each `{ center, size, spacing, follow: 'fixed' | 'camera', priority }`. M1's single `volume` becomes the one-element case.
- **One atlas set for all volumes:** the irradiance, distance and probe-data textures stay three 2D arrays. Volume `v` owns layers `[base_v, base_v + ny_v)` and the tiles `(x, z)` inside them. Each layer is as large as the largest volume's footprint. *Why shared:* a shading module may use at most 6 textures (platform.md gap 1); one set per volume breaks that at the second volume. Past 256 layers in total, `prepare` throws `GILimitError`.
- **The volume table** is the small `rgba32float` texture the contract already plans: 4 texels per volume (origin and spacing; counts and layer base; scroll offset and valid extent; priority, fade width and flags). The GPU writes it (section 4.2), so it is also a storage texture.
- **Probe numbering:** one global index across volumes (`volumeBase + local`), so the scheduler, history and update list don't change.
- **Shading:** for a point, volumes are visited in priority order (the CPU sorts the table). Each contributes with weight `w_v = coverage_v × (1 − W)`, where coverage is the existing edge fade times `saturate(Σw / 0.02)` (DECISIONS 2026-11-05), and `W` is the weight gathered so far. Stop at `W ≥ 0.99`, or after 2 volumes. Inside a volume that's one sample, as now; only in fade bands is it two. RTXGI blends volumes the same way.
- **Bounce light** in `ddgi/trace` uses the same function, so light bounces between volumes.

### 4.2 The camera-following volume: scrolling

The volume is a fixed-size window of probes over a lattice fixed in world space (RTXGI's infinite scrolling, clipmap addressing):

- **Lattice and slots:** a probe's lattice coordinates are `L = round((P − anchor) / spacing)`, and its storage slot is `L mod n` on each axis. Probes never move in world space, so their irradiance and distance stay valid.
- **The window** is `[windowMin, windowMin + n)` in lattice units, centred on the camera. It moves by whole planes when the camera is more than **0.75 spacing** from the window's centre plane on an axis. *Why 0.75, not 0.5:* a camera resting on a plane boundary would otherwise make the window flip back and forth, clearing the same planes every frame.
- **On a move:** the slots of the planes that leave become the planes that enter at the far edge. A small kernel, `ddgi/scroll`, resets them: state 3 (not updated yet), relocation offset 0, history `n = 0`, owed K, high urgency. M2's scheduler then updates them first.
- **Fading the new planes in.** Coverage is 1 out to half a spacing beyond the outermost probes (DECISIONS 2026-11-05), so a new plane at full coverage the moment the window moves would be a pop. The volume table therefore holds a **valid extent** next to the window. Coverage fades at the valid extent, not at the window's edge. `ddgi/scroll` advances the valid extent toward the window by `spacing / 15` a frame, and only over planes whose probes have finished warming (from the census). A new plane fades in over about a quarter of a second instead of appearing at once.
- **Sampling** maps a point to lattice coordinates, then to slots with the same modulo; the window and valid extent come from the volume table. For a fixed volume the offset is 0 and the modulo changes nothing: the same code, not a second path. Cost: one integer modulo per axis per probe sample.
- **The ends of the scroll:** a plane that enters next to a fixed, higher-priority volume is covered by that volume meanwhile; elsewhere the host's environment covers the fade band, as at any volume edge.
- **Speed:** entering probes need their 2 warming updates before they're shown (section 5.2).
  - With a 32 × 12 × 32 window at spacing `s`, each plane on the x or z axis holds 384 probes. At `v` m/s that's `384 × v / s / 60` entering probes a frame, and twice that in warming updates: `8.5 v` a frame at 1.5 m spacing.
  - **Measured:** Bistro exterior fits **20–27** probe updates a frame at 2 ms (T1.9a, with the fixed controller). An earlier estimate here, about 90, came from its cost curve and was wrong.
  - **Budget v2, with the GPU loaded** (T1.9b, medians of 3 runs): Bistro exterior updates **132** probes a frame at 2 ms (124–160), against T1.9a's 20–27; the old numbers were mostly the GPU's low-power state. The camera volume is the same scene, so 132 is used for it.
  - At 132, **5 m/s at 1.5 m needs about 43 a frame, a third of the budget**, inside the scheduler's priority share (0.75 × 132 ≈ 99). Half the budget allows about 7.8 m/s.
  - (At T1.9a's old rate of 20–27, half the budget allowed only about 1.4 m/s.)
  - Levers:
    - **coarser spacing:** at 3 m (a 96 × 36 × 96 m window, the same memory), 5 m/s needs about 21 a frame;
    - **a larger budget while moving;**
    - **M2's deferred shadow pass,** if it raises the rate.
  - Dennis picked 5 m/s at 1.5 m on the wrong estimate, so this goes back to him (open question below).
- **Teleports** (the camera jumps more than half the window): every slot resets, and the viewer's warm-up mode (DECISIONS 2026-11-05) runs, as after a scene load.
- **Memory:** 32 × 12 × 32 = 12,288 probes at about 1.5 KB each (irradiance 512 B, distance 1 KB, probe data, history) is **19 MB**, covering 48 × 18 × 48 m at 1.5 m spacing. M2's instant sky, on by default, adds 41 MB (m2-lights.md 6.3), so **60 MB** in all.

*As built (T3.6; ddgi.md, "The camera-following volume"):* as above, with these differences.
- **Entering probes owe 2 updates, not K:** they start from nothing, and the first update replaces the history outright, so the two warming updates are what they need before they are shown; K for each would cost 4× the budget this section's arithmetic gives.
- **The valid extent's trailing side follows the camera** continuously (1.25 planes inside its half-window), not only the window: otherwise the planes about to be reused would leave the coverage at once, a pop at the trailing edge. And the extent keeps a whole plane between it and any cold plane, because coverage reaches a spacing past it.
- **The table has 8 texels a row,** not 4: the four sampling `vec4f`s (T3.5), the window, the fade's width per axis, and the extent's low and high sides. No flags.
- **The fade is 10 spacings wide, not half a spacing** (DECISIONS 2026-11-10): the band sweeps at the camera's speed, and half a spacing made the edge change 0.59% a frame on Bistro at 5 m/s; 10 spacings (3 s to pass a point) 0.074%. The camera volume then answers alone within about 5 spacings of the camera.
- **Key probes go by the lattice's parity,** so a move changes the role of the planes at the window's ends only.
- **A point outside every volume reads the nearest one** (T3.5 read the lowest-priority one, wrong for volumes side by side).
- **Teleports** restart the technique's own warm-up (T2.4b moved it into the technique).

## 5. Probes near moved geometry

### 5.1 How a moved object becomes a set of probes

M2's `ddgi/mark` (m2-lights.md 4.2) gets geometry entries. For each geometry region (the old and the new box of a moved, added, removed or deforming object):

- **Near set:** probes within the box grown by 1.5 spacings. Their rays may now start inside the object, or see a different surface right next to them. The change size is:
  - added or removed: `r = 1`;
  - moved: `r = min(1, moved / spacing)`, from the region's corner movement (section 1);
  - deforming: `r = min(1, largest vertex movement / spacing)`, the GPU word from section 3.

  A drag or a sway therefore marks a little every frame, and `pendingChange` adds it up until the probe's next update. That's the same behaviour as a light sweep in M2, not a reset every frame.
- **Far set:** probes within 8 spacings (clipped to the volume) that can see the box. The test: in the direction of the box centre, the probe's stored mean distance (from its distance moments) must reach at least the distance to where that direction **enters** the box, minus half a spacing. A test against the box centre would fail for the old box, because the object hid its own centre: the stored distance stops at its near face.
  - The change size is the share of the view the object covers: `r = min(1, Ω / Ω_ref)`, with `Ω ≈ π × boxRadius² / distance²` and `Ω_ref = 0.5 sr` (an object covering about 1/25 of the sphere counts as a full change).
  - For a move, `r` is also scaled by `min(1, moved / spacing)`, so a slow drag doesn't count as a full change every frame.
- **Probes that saw it last update** (an addition to the geometric tests, so a thin or odd-shaped object isn't missed or over-counted): each probe keeps a 32-bit **mover mask**. When one of its blended rays hits an instance on the hot list of movers (section 6), the trace sets bit `moverSlot mod 32` with one `atomicOr`. `ddgi/mark` then adds to the far set every probe within 8 spacings whose mask has the moved object's bit. A shared bit only over-marks, never misses. 4 bytes a probe. The trace collects the bits in a workgroup atomic while its rays run, and one thread stores the result at the end, so each mask holds exactly its latest update's hits and needs no storage atomics. It's `ddgi/trace`'s 8th storage buffer, the default limit; if anything else needs one, the mask moves into the history record and the blend sets it from the ray scratch.
- Then the same history rule as for lights: `n ← min(n, εK / (pendingChange − ε))`, owed K, urgency from `pendingChange` and the camera weight.
- **How they're served** (m2-lights.md 4.5, 4.7): the near set is traced itself, every probe, because its distances change. The far set, whose light changes through a shadow or a closed door, goes through key probes and propagation.

**Example: a door between two rooms** (1 × 2 m, radius about 1.1 m), closed in one step. A probe 4 m into the dark room that saw the open doorway has `Ω ≈ 0.24 sr` and `r ≈ 0.47`. Its history drops to `n ≈ 0.95`, and it is owed 8 updates. A probe behind a wall doesn't pass the far-set test and isn't marked.

**What the budget buys** (the same arithmetic as m2-lights.md 4.3): the door's near and far sets are a few hundred probes on two-room, where 707 probes are updated a frame (T1.9b), so all of them get their 8 updates within about 12 frames, 0.2 s. A cabinet moved in the furnished room touches fewer. On Sponza-sized scenes the probes nearest the camera are served first.

**With M2's instant sky on** (m2-lights.md 6.3), marked probes' first-order sky transfers are reset with their history. They re-accumulate from the updates the probes are owed anyway, and count as built again after 8, under a second on two-room for the door's few hundred probes. Until then, sky changes reach them by re-tracing. The sky layer itself keeps updating, so nothing jumps.

**The distance atlas** uses the same per-probe history. For geometry that matters most: until a probe near the door learns the door's distance, light can leak through it. The short history and the priority keep that to the first few updates (risk 3).

### 5.2 Probes that were off come back warmed

Off probes (inside geometry, or no surface nearby) trace their 32 fixed rays whenever they're updated, and classification can turn them back on (T1.5b). Today `blend.ts` already moves a probe coming back on to state 3 (not updated yet, not used in shading), and it becomes active after its first blended update.

M3 changes one thing: **state 3 ends after 2 blended updates, not 1,** so a probe is used in shading only once it has at least two estimates. One noisy estimate is the pop SPEC forbids. Both off states are in the near set of any change that can free them, so they're scheduled at once.

On → off (a cabinet pushed over a probe) stays immediate, as now; neighbouring probes fill in through the normal sampling weights.

*As built (T2.M1):* state 3 ends after the second blended update (the history flag `warmedOnce`); a probe coming back on keeps the updates its move promised it, and off probes in a move's near set go first (`FREED_PROBE_URGENCY_SHARE` 16), so on two-room a probe freed by a moved crate is active again 3 frames later.

### 5.3 Moving receivers in empty space

TASKS carries this from M1: classification turns off probes in empty space (347 of Cornell's 1,728), so an object moving through mid-air gets no GI. RTXGI has the same limit. Proposed:

- **Woken air probes:** a probe that is off for "no surface nearby" and lies within 2 spacings of a receiver region (`SceneChanges.receivers`) or a geometry region is **woken** for 300 frames (5 s) after the last touch. While awake it traces all rays, blends, and is used in shading once warmed. Then it goes back to sleep.
- *Why not keep every air probe awake at a slow rate:* that would change M1's still-scene cost and results. Waking keeps a still scene exactly as M1 has it.
- Data: a `wakeUntil` frame, in word 8 of the history record (m2-lights.md 4.1).

## 6. Change tracking in the three adapter

**Today (T1.7, three.md, `scene-sync.ts`):** a walk of 64 objects per frame plus a hot set for movers, `markChanged(object)` for no delay, 8 geometry and 8 material edit checks per frame, and every light every frame.

- A still frame costs 0.06–0.10 ms on Sponza and 0.11–0.15 ms on Bistro interior.
- Each object costs about **0.7 µs**, because its data is cold in the cache right after three renders. A full pass over Bistro interior every frame would be 1.2 ms. That's why the walk is budgeted.
- An unmarked first move on Bistro interior can take 41 frames (0.7 s); an unmarked geometry edit up to about 260.

What three r187 offers, and what M3 uses:

| Change | How it's found in M3 | Latency |
| --- | --- | --- |
| Added or removed objects | three's `childadded` and `childremoved` events (they exist in r187, `Object3D.js`), on every tracked parent | the same frame |
| Moved objects | the walk and the hot set stay (cache misses, not code, set their cost). New: the walk skips objects with `matrixAutoUpdate` and `matrixWorldAutoUpdate` both false (three's own idiom for static objects), leaving them to `markChanged`; that shortens a pass in scenes that use the idiom | `ceil(movable objects / objectsPerFrame)` frames, or the next frame with `markChanged` |
| Moved objects, faster (T3.0 spike) | compare the matrix right after three computes it, while it's still in the cache: override `updateMatrixWorld` on each tracked object (not on the prototype) | the next frame, if the spike shows it's cheap and safe |
| `InstancedMesh` | `instanceMatrix.version`, and only the ranges in `instanceMatrix.updateRanges`. three clears those ranges when it uploads during `render`, and `ThreeGI.update` runs before `render` (the documented order), so they're still there | the next frame |
| Geometry edits | each position and index attribute's `version`, more per frame than now (an integer compare each) | a few frames |
| Material edits | three bumps `material.version` only on `needsUpdate`, so value edits (a colour change) still need the snapshot compare, budgeted as now | as now |
| Anything else | the walk, as a safety net | one pass |

- **Sending:** moved objects go out in one `setTransforms` batch per frame.
- **Deforming meshes:** a mesh with `userData.giDeforming = true` is created `deforming`. When its position attribute's `version` changes, the adapter sends the CPU array (section 7). A `SkinnedMesh` skins on the GPU, so its position attribute never changes; skinned meshes stay in their bind pose (a stretch goal after v1, Dennis 2026-11-05).
- **Moves still reach GI one frame late** by default (DECISIONS 2026-11-05: the adapter reads the matrices three's last render made).
- **For interactive tools,** `markChanged` stays the recommended path; the viewer's drag gizmo calls it.

*As built (T3.7, prototype):* `childadded` and `childremoved` are heard on every object of the scene (`change-watch.ts`), so added and removed objects reach GI in the next update. The `updateMatrixWorld` hook is on each mirrored mesh only (three calls it on every child, so groups need none), compares against the matrix last sent and puts a moved mesh in the hot set; it is removed by assignment. The walk keeps running but skips the transforms of hooked meshes while `scene.matrixWorldAutoUpdate` is on. Each `InstancedMesh`'s `instanceMatrix.version` is compared every update, because a hook sees an edit made after the render only at the next render, which also clears the update ranges. Deforming meshes: skinned (when they contribute to GI), morphed and `userData.giDeforming` meshes are created `deforming`, one platform mesh per three mesh, with a `deformBounds` (`userData.giDeformBounds`, or an estimate: exact for morphs), and send CPU poses through `setDeformedPositions` when their inputs change (`deforming.ts`). Skinned meshes are posed on the CPU with three's formulas, not left in the bind pose as the table above planned; `PoseSource` is the seam for T3.3's `ExternalTexture` form. Bistro interior: an unmarked first move is sent 2 frames after it happens (update, render, update), against up to 41 before.

## 7. Getting deformed positions out of three

The hard part of the wind tree isn't the refit. It's that three r187 gives no supported way to read a GPU buffer it owns (NOTES: no external buffers; and we don't reach into `renderer.backend`). A wind sway written as a TSL `positionNode` exists only inside three's vertex shader.

Three routes, in the order I'd try them:

1. **CPU positions** (works today): the host animates the position attribute on the CPU (three uploads it as well), and the adapter passes the same `Float32Array` to `setDeformedPositions`. Cost: one `writeBuffer` a frame, about 1.2 MB for a 100,000-vertex tree. This is the plan for the viewer's tree (Dennis: yes).
2. **A TSL compute writing into our texture** (T3.0 spike): if three can bind an `ExternalTexture` as a *storage* texture in a TSL compute, the host's GPU wind writes positions into a texture we own, three's vertex shader reads them back through `positionNode`, and our refit reads the same texture. No backend access and no upstream change. Unknown whether r187 allows it.
3. **An upstream `ExternalBufferAttribute`** (the small PR SPEC already lists): three would read our buffer as a vertex attribute. This is the cleanest, but it depends on review upstream. The draft is in section 7.1; Dennis sees it before anything is submitted, and nothing is posted until he says so.

Skinned characters as light blockers stay a stretch goal after v1 (SPEC; Dennis 2026-11-05). If they're wanted later, a platform skinning kernel (linear blend skinning from bone matrices, engine-neutral) is the route.

### 7.1 Draft: the `ExternalBufferAttribute` PR to three.js (not submitted)

Dennis sees this text before anything is posted to three.js, and nothing is submitted until he says so. The code below was checked against three r187's `WebGPUAttributeUtils.js` (`createAttribute`, `updateAttribute`, `destroyAttribute`) and `StorageBufferAttribute.js`. It is a sketch for the PR, not tested code.

**Title:** WebGPURenderer: add `ExternalBufferAttribute` (use a `GPUBuffer` from the same device as an attribute)

**Description:**

> `ExternalTexture` (r187) lets an app hand three a `GPUTexture` it created on the renderer's device, and three binds it without allocating or destroying it. There's no equivalent for buffers. Apps that compute vertex data with their own WebGPU code (cloth or physics solvers, GPU skinning outside three, GI or ray-tracing libraries that refit their own copy of a mesh) must read the data back to the CPU, or copy it into a three-owned attribute each frame.
>
> This adds `ExternalBufferAttribute`, the buffer counterpart of `ExternalTexture`. It wraps an existing `GPUBuffer`. three uses that buffer for vertex input and for storage bindings in TSL, and never uploads into it, resizes it or destroys it. The caller owns its contents and its lifetime.
>
> - WebGPU backend only; the WebGL backend throws a clear error.
> - The buffer must come from `renderer.backend.device` (or the device passed to `WebGPURenderer`) and have the usages it's used with (`VERTEX`, `STORAGE`).
> - The data layout is the caller's: tightly packed items of `itemSize` components of the given type. No padding is applied.
> - There's no CPU copy, so `computeBoundingBox` / `computeBoundingSphere` can't run; set `geometry.boundingBox` / `boundingSphere` yourself (frustum culling needs them).
>
> Example: positions written by an app's own compute pass, drawn by three:
>
> ```js
> const positions = new THREE.ExternalBufferAttribute( gpuBuffer, vertexCount, 3 ); // Float32 by default
> geometry.setAttribute( 'position', positions );
> geometry.boundingSphere = new THREE.Sphere( center, radius );
> ```
>
> Also works with TSL storage nodes, so a TSL compute can write into an app-owned buffer: `storage( positions, 'vec3', vertexCount )`.

**Code sketch:**

```js
// src/renderers/common/ExternalBufferAttribute.js
import { BufferAttribute } from '../../core/BufferAttribute.js';

class ExternalBufferAttribute extends BufferAttribute {

	constructor( buffer, count, itemSize, typeClass = Float32Array ) {

		// An empty array of the right type: three reads the vertex format from its constructor.
		super( new typeClass( 0 ), itemSize );

		this.isExternalBufferAttribute = true;
		this.buffer = buffer; // GPUBuffer, owned by the caller
		this.count = count;

	}

}

// src/renderers/webgpu/utils/WebGPUAttributeUtils.js
createAttribute( attribute, usage ) {
	const bufferAttribute = backend.getBufferAttribute( attribute );
	const bufferData = backend.get( bufferAttribute );
	if ( bufferAttribute.isExternalBufferAttribute ) {
		bufferData.buffer = bufferAttribute.buffer; // use it as it is: no allocation, no upload
		return;
	}
	// ... unchanged
}

updateAttribute( attribute ) {
	if ( this.backend.getBufferAttribute( attribute ).isExternalBufferAttribute ) return; // contents are the caller's
	// ... unchanged
}

destroyAttribute( attribute ) {
	const bufferAttribute = backend.getBufferAttribute( attribute );
	const data = backend.get( bufferAttribute );
	if ( bufferAttribute.isExternalBufferAttribute !== true ) data.buffer.destroy(); // never destroy the caller's buffer
	backend.delete( bufferAttribute );
}
```

Plus: the export from `Three.WebGPU.js`; a WebGL backend check that throws; a unit test that the buffer is used without being written to or destroyed; a short example (an app-side compute pass that moves a grid's vertices). It goes to three's `dev` branch, as small as the maintainers prefer (NOTES and SPEC: they want small PRs).

**What it gives us:** a host's GPU-animated mesh (TSL wind, morph targets computed by the host) writes positions into a buffer we own. three draws from it, and our refit reads the same buffer. No CPU copy, and no reaching into three's backend.

## 8. The viewer scenarios

| Scenario | Scene | What it shows | Measured |
| --- | --- | --- | --- |
| **Drag furniture** | furnished room | `TransformControls` on any object; shadows and colour bleeding follow | time to within 5% of settled after a 1 m cabinet move (M2's response meter, GI only) |
| **Animated door** | two-room with a hinged door | a keyframed door swing closing on the lit side | dark-room leak at `dark-room-leak-view` over time (DECISIONS' leak definition) |
| **Add and duplicate cabinets** | cabinet row | "add" makes a new mesh (a new tree, built on the GPU that frame); "duplicate" shares the mesh (a new instance only); up to 100 | frames until GI sees it; the recompile counter |
| **Wind tree** | courtyard (small outdoor scene) with a game-ready tree (8.1) | a CPU wind sway on the tree, casting moving occlusion | refit ms; trace speed over 60 s; the GI under the tree following the sway |
| **Fly-through** | Bistro exterior | a manifest camera path longer than one volume, with the camera-following volume | frame-to-frame change in the edge band; probes per frame; memory |

- Every scenario is a manifest motion (m2-lights.md section 7), so it replays the same way in tests, and next to native where native can follow.
- The response meter and the recompile counter are on screen during each.
- **Alpha cutouts are M4.** Until they land, the tree's leaf cards block light as solid quads and over-darken. M3 checks that the occlusion *moves*; the look is judged after M4 (Dennis: fine, 2026-11-05).
- **Native:** the harness has no moving geometry. The side-by-side for M3 covers the door and cabinet scenes only if T2.7's motion support can be extended to instance transforms; otherwise M3 compares against Cycles renders of the end states only.

### 8.1 The wind tree's asset

Dennis (2026-11-05): `sponza-full`'s 2 M-triangle trees are too heavy. Plan for a **game-ready tree**:

- **What:** a single tree of about 20,000–100,000 triangles (trunk, branches, leaf cards with alpha), real-world size, with its source and licence recorded like every asset (T0.4).
- **Where from, in order:**
  1. a CC0 or CC BY tree from Sketchfab or Poly Haven;
  2. one generated in Blender with a tree add-on (Blender's bundled Sapling Tree Gen, for example), saved as glTF; the output is ours, but the add-on's own licence is checked and recorded;
  3. a Sponza tree, only if one is easy to cut out and small enough.
- **The sway:** done by the viewer on the CPU. Each vertex moves by a small sine offset that grows with its height above the trunk base, plus a faster flutter on leaf-card vertices. At 100,000 vertices that's about 0.3–0.5 ms of JS a frame (an estimate), and it's the viewer's cost (the host's), not the adapter's.
- **Leaves** block light as solid cards until M4's alpha cutouts (fine for M3).
- A wrangler task (T3.11) finds and converts it.

## How it fits the existing kernels (no parallel GPU paths)

| Piece | M1 | After M2's startup lane (T2.12–T2.15) | M3 |
| --- | --- | --- | --- |
| Mesh trees at load | tinybvh in the worker | GPU LBVH, copied in after its stats readback; tinybvh swapped in; tinybvh only without subgroups | unchanged |
| Meshes added or replaced later | tinybvh in the worker | tinybvh in the worker | GPU LBVH in the frame, tinybvh swapped in |
| Deforming meshes | refused | refused | `GpuMeshRefit` every frame on the rest-pose tree; rest pose without subgroups |
| Top level | GPU refit + worker rebuild (B1a) | unchanged | unchanged |
| Scene copy packing | CPU, main thread | `gi/pack-mesh` on the GPU; the CPU packer is the test oracle only | + time-sliced uploads for meshes added at run time |
| `ddgi/mark` | none | light entries (M2) | + geometry and receiver entries |
| Probe states | back on after 1 update | unchanged | back on after 2 updates; air probes woken near movers |
| Sampling | one volume | one volume | several volumes in priority order with wrapping slots; one fixed volume is the same code |
| `ddgi/scroll` | none | none | new, tiny, only on window moves and while new planes fade in |

Storage buffers stay as in M2 (`ddgi/trace` 7). `gi/pack-mesh` binds the staging buffer and the scene copy's vertex buffer (2). `ddgi/scroll` binds the history (1) and writes the probe-data and volume-table textures.

## Cost against the budgets (estimates, RTX 3060, to be measured)

| Item | When | Cost |
| --- | --- | --- |
| Top-level refit | every frame | 0.086 ms at 10k instances (B1a, measured) |
| Mesh refit, wind tree (~100k triangles) | every frame | ~0.1–0.2 ms |
| A new mesh tree (a cabinet, a few thousand triangles) | on add | well under 1 ms GPU, in the same frame |
| Tree depth pass | after each GPU build | small; Sponza's 5.7 M leaves × depth 33 at load |
| `ddgi/mark` geometry entries | frames with moves | < 0.02 ms for a few objects |
| `ddgi/scroll` | window moves and fades | < 0.01 ms |
| Sampling with two volumes | fade bands only | ≤ 2× there; the whole view must stay ≤ 1.0 ms at 1440p (SPEC) |
| Adapter, still / one mover | every frame | ≤ 0.2 / ≤ 0.5 ms (targets) |
| GPU mesh packing | at load | ~10–20 ms GPU for Sponza, once |

Memory: the camera-following volume is 19 MB, 60 MB with M2's instant sky. LBVH output buffers live until their trees are copied in (a few frames). During a swap a mesh briefly holds two trees. Deforming meshes on the CPU route keep a buffer the size of their positions.

## Proposed task breakdown

### Startup lane, run in M2 (DECISIONS 2026-11-05, provisional)

| ID | Task | Agent | Size | Depends on |
| --- | --- | --- | --- | --- |
| T2.12 | Fast first GI: GPU LBVH trees at load (stats readback, exact ranges, copy in); tinybvh swapped in per mesh | implementer | M | B1 merged into `m1` |
| T2.13 | Scene copy off the main thread: raw upload, GPU packing | implementer | M | none |
| T2.14 | Compile only what's configured: `encodeBatch` kernels only when a technique asks for batches; compile while uploading | implementer | S | T2.6 (which owns the sky kernel's compile) |

- **T2.12.** Done when:
  1. Sponza from "scene handed to GI" to the first frame drawn with GI is ≤ 2.0 s on the RTX (cold compile, median of 3), with the 0.9 s now unexplained broken down in PROGRESS;
  2. across the swap frame the field's mean changes by ≤ 0.5%, and trace speed afterwards is within 5% of M1's on tinybvh trees;
  3. the oracle passes in mid-swap (some meshes on LBVH trees, some on tinybvh's) and with a forced refused tree (that mesh waits for tinybvh), on both GPUs;
  4. a device without subgroups still gets tinybvh trees and GI (test).
  - Record: startup by stage in `timing.csv`.
- **T2.13.** Done when the scene copy's main-thread time is ≤ 150 ms on Sponza and ≤ 200 ms on Bistro exterior (from 1.2 s and 4.1 s), and the GPU-packed buffers match the CPU packer (positions and indices byte for byte, normals within one octahedral step) on every procedural scene and Sponza.
- **T2.14.** Done when Cornell from load to the first frame with GI is ≤ 1.5 s cold on the RTX (from 2.34 s), and the recompile counter stays at 0 through a viewer session on every scene.

### M3

| ID | Task | Agent | Size | Depends on |
| --- | --- | --- | --- | --- |
| T3.0 | Spike: deformed positions from three (storage `ExternalTexture`); the `updateMatrixWorld` override's cost and safety on Bistro interior; refit decay on a swaying tree | implementer | S | M2 done |
| T3.1 | Change feed: `setTransforms`, `moved`, `receivers`, `deformBounds`, `setDeformedPositions`, recompile counter | implementer | M | none |
| T3.2 | Runtime mesh trees: GPU builds for added and updated meshes, tinybvh refinement | implementer | M | T2.12 |
| T3.3 | Deforming meshes: `GpuMeshRefit` wired, scene-copy positions, geometric normals, movement word; H-PLOC rebuild when the refit drifts (3.1) | implementer | M | T3.1, T3.2 |
| T3.4 | Probes near moved geometry: geometry entries in `ddgi/mark`, the mover mask, far sets through key probes, the 2-update warm-up, woken air probes | implementer | M | T3.1, M2's T2.4b |
| T3.5 | Several volumes: shared atlas, volume table, priority blending | implementer | M | none |
| T3.6 | Camera-following volume: wrapping slots, `ddgi/scroll`, window rule, fade-in | implementer | M | T3.5 |
| T3.7 | three adapter change tracking v2 | implementer | M | T3.0, T3.1 |
| T3.8 | Viewer scenarios and their manifest motions (furniture, door, cabinets, wind tree, fly-through) | implementer | M | T3.3, T3.4, T3.6, T3.7 |
| T3.9 | M3 measurements | gpu-tester | M | all above |
| T3.10 | M3 note and M4/M5 breakdown | main session | S | T3.9 |
| T3.11 | The wind tree's asset (8.1): find or generate, convert, record licence | wrangler | S | none |

B1 is done before M3 starts. B2 (H-PLOC) is done before M3 sign-off, and doesn't block any M3 task (Dennis, 2026-11-05).

### Draft done-whens

- **T3.0 Spike.** Done when each question has a written yes/no answer with numbers, as M0's spikes did:
  1. can a TSL compute write an `ExternalTexture` in r187, and can our refit read it;
  2. ns per object for the `updateMatrixWorld` override on Bistro interior, and whether three's own tests and our adapter tests still pass with it;
  3. trace speed on a swaying 100,000-triangle tree after 600 frames of refits only, against a fresh build.
- **T3.1 Change feed.** Done when:
  1. 10,000 transforms through `setTransforms` cost ≤ 2 ms of CPU (from 18–20 ms);
  2. a moved receiver produces a `receivers` box, and a moved instance carries `moved`;
  3. `setDeformedPositions` passes the oracle on a deforming test mesh;
  4. creating and destroying 1,000 instances and 100 meshes in random order leaves the recompile counter at 0.
- **T3.2 Runtime mesh trees.** Done when:
  1. a replaced mesh (`updateMesh` on an existing instance) is hit by rays within 4 frames (oracle page; the stats readback);
  2. a new instance of a new mesh on the cabinet row is hit within 4 frames (the worker's top-level rebuild);
  3. after the tinybvh swap, trace speed equals tinybvh-only within 5%, and the LBVH range is freed within 2 frames;
  4. without subgroups, meshes get tinybvh trees (test).
- **T3.3 Deforming meshes.** Done when:
  1. the oracle passes on a deforming mesh over 60 frames on both GPUs, hit points included (scene-copy positions);
  2. the refit costs ≤ 0.3 ms a frame on the RTX for a 100,000-triangle mesh;
  3. with the tree held at its two extreme poses, the converged GI in a ground rectangle under it differs by ≥ 2%, and while it sways that rectangle's mean follows the pose within 0.5 s (from the cross-correlation of the two curves);
  4. a device without subgroups shows the rest pose and one warning.
- **T3.4 Probes near moved geometry.** Done when, at the default budget, GI only:
  1. on the furnished room, a 1 m cabinet move is within 5% of settled in ≤ 1 s on the RTX;
  2. on two-room with a door, the dark room's leak (DECISIONS' definition) is ≤ 1% within 1 s of the door closing and stays there;
  3. no probe is used in shading before 2 blended updates (test);
  4. a still scene's field and cost match M2's (mean within 0.1%, cost within 3%): air probes stay asleep.
- **T3.5 Several volumes.** Done when:
  1. a dense fixed volume inside a coarse one shows no seam: the image difference across the fade band, frame to frame, stays within the stability threshold set in T1.9;
  2. sampling at 1440p on a view half inside the fade band costs ≤ 1.0 ms on the RTX;
  3. one-volume scenes match M2 (field mean within 0.1%, cost within 3%).
- **T3.6 Camera-following volume.** Done when, on Bistro exterior's fly-through at the speed and spacing settled in the open question below (5 m/s at 1.5 m as decided, unless that changes):
  1. the frame-to-frame change in the edge band stays within T1.9's stability threshold;
  2. entering probes are never used before warming (test);
  3. memory stays fixed;
  4. a camera resting on a plane boundary for 10 s clears 0 planes.
- **T3.7 Adapter change tracking v2.** Done when, on Bistro interior (RTX laptop, median of 300 frames):
  1. added and removed objects reach GI in the next update without `markChanged`;
  2. a still frame costs ≤ 0.2 ms, and one mover ≤ 0.5 ms;
  3. an unmarked first move reaches GI in ≤ 10 frames if T3.0's override is adopted, or within one pass of the (shortened) walk otherwise;
  4. `InstancedMesh` update ranges send only those instances (test).
- **T3.8 Viewer scenarios.** Done when every scenario plays from its manifest motion with hover help, shows the response meter and the recompile counter, and Dennis can run each without instructions.
- **T3.11 The wind tree's asset.** Done when the tree loads in the courtyard scene at real-world size against the 1 m reference cube, has 20,000–100,000 triangles, and its source, licence and checksum are in its manifest and `THIRD_PARTY.md`.
- **T3.9 M3 measurements.** The SPEC table above on the RTX 3060 (GPU-bound) and the M5 Pro if it's set up, the correctness checks on the Intel UHD (no speed targets there, Dennis 2026-11-05), plus the door and cabinet response curves against Cycles renders of their end states (3 runs), with the recompile counter at 0 throughout. Done when each target is met or its miss is explained and agreed with Dennis.

## Risks, and what to spike first

1. **Deformed positions from three** (T3.0). If neither a storage `ExternalTexture` nor the upstream PR works, GPU-animated meshes can't contribute without reaching into three's backend; only CPU-animated ones can. *Spike first.*
2. **Change tracking at scale.** Cache misses, not code, set the walk's cost (0.7 µs per object). If the `updateMatrixWorld` override is unsafe or no cheaper, unmarked moves on Bistro-sized scenes keep a latency of tens of frames, and `markChanged` stays the answer for interactive moves.
3. **The door's leak window.** Until the near probes' distance tiles learn the closed door, light leaks for a few frames. With `n` near 0.4 and the door's probes updated every frame, that should be 2–4 frames. If it shows, those probes' first update could use 2× rays.
4. **The far-set estimate** (`Ω_ref`, the 8-spacing radius, the entry-distance test) is a guess. T3.4 compares marked probes against probes that really changed by more than 5%.
5. **Scrolling speed.** At 132 probes a frame (T1.9b), 5 m/s at 1.5 m uses about a third of the budget, and half allows about 7.8 m/s. Faster flights show the environment longer at the edge (section 4.2).
6. **The stats readback adds 1–3 frames** before a GPU tree is used. That's harmless at load. For an object added at run time it adds to the top level's worker delay; if it shows in the cabinet scenario, the top level can move to the GPU (section 2.2).
7. **Refit quality** on large deformations (cloth, fan blades) may need rebuilds, which SPEC's tracing decision rules out today.

## Later

- **The viewer's file loading** (Sponza's 4.2 s): outside the startup target, which counts from the hand-off to GI. Worth breaking down later, because it's what a viewer user waits for.
- **The GPU building the top level** in the same frame (section 2.2), if the worker's delay shows.
- **A platform skinning kernel** for skinned characters as light blockers (a stretch goal after v1).
- **Rebuilds for strongly deforming meshes** (cloth, fan blades), if T3.0 shows refits alone degrade too far. That would be a DECISIONS entry against SPEC's "keep the shape".

## Decided by Dennis (2026-11-05)

1. **Deformed positions:** `setDeformedPositions` also takes a CPU array. The `ExternalBufferAttribute` PR is drafted in section 7.1 for him to see; nothing is posted.
2. **`setTransforms`:** yes.
3. **Readbacks** aren't a rule but a preference: "prefer GPU-resident; read back when it clearly pays". GPU trees use one small asynchronous stats readback (section 2.2); the provisional pool is gone.
4. **The fly-through:** Bistro exterior, a 32 × 12 × 32 camera-following volume at 1.5 m (19 MB; 60 MB with M2's instant sky), up to 5 m/s. (It rested on a wrong rate at first; at T1.9b's measured 132 probes a frame it fits.)
5. **Startup:** measured from the hand-off to GI, ≤ 2 s with a cold shader compile.
6. **The wind tree:** a game-ready tree (section 8.1), not a 2 M-triangle one. Solid leaf cards until M4 are fine.
7. **B2 (H-PLOC):** before M3 sign-off, not before M3 starts. It must not block; the fallbacks always stay.
8. **Skinned characters** as light blockers: a stretch goal after v1.

## Still open

1. **The fly-through's speed and spacing: settled by the measured rate, pending T3.9.** Dennis's pick (5 m/s at 1.5 m) first rested on a wrong estimate (about 90 a frame), then T1.9a's low-power 20–27 didn't fit. T1.9b measured 132 with the GPU loaded: 5 m/s at 1.5 m needs about 43 of them, a third of the budget. So the pick stands; T3.9 measures the fly-through itself.

Apart from this, nothing is left open in this doc. The M1 lead records 1, 3, 5 and 7 in DECISIONS: they change or add to SPEC's tracing decision, the B1a review's no-readback wording, the startup target and the B-lane gate.
