// Engine-free BVH builders (docs/design/bvh-framework.md). A builder produces a
// MeshTreeTopology (the tree's shape), or, for tinybvh, the tracer's GPU layout
// written inside the WASM (GpuLayoutMeshBvh). The GPU refit (B1a) keeps a
// built tree's shape and recomputes its boxes on the GPU. The GPU LBVH (B1)
// builds mesh trees on the GPU in the same layout; createMeshLayoutBuilder
// picks it when the device has subgroups and tinybvh otherwise.
export * from './topology.ts';
export * from './gpu-layout.ts';
export * from './tinybvh.ts';
export * from './refit-cpu.ts';
export * from './gpu-refit.ts';
export * from './gpu-instance-box.ts';
export * from './gpu-lbvh.ts';
export { LBVH_STAT } from './gpu-lbvh-wgsl.ts';
export * from './mesh-builder.ts';
export * from './layout-check.ts';
export * from './gpu-mesh-refit.ts';
