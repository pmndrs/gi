// Engine-free BVH builders (docs/design/bvh-framework.md). A builder produces a
// MeshTreeTopology (the tree's shape), or, for tinybvh, the tracer's GPU layout
// written inside the WASM (GpuLayoutMeshBvh).
export * from './topology.ts';
export * from './gpu-layout.ts';
export * from './tinybvh.ts';
