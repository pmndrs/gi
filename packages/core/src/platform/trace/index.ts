// The `rays` query backend: two-level BVH, CPU builders, WGSL tracer (T0.10).
// The oracle's helpers (reference-tracer.ts, oracle-compare.ts, oracle-fixtures.ts)
// are test code: they stay out of this index, so out of the published package,
// and tests import them by path.
export * from './layout.ts';
export * from './sah-builder.ts';
export * from './mesh-bvh.ts';
export * from './mesh-bvh-topology.ts';
export * from './mesh-bvh-build.ts';
export * from './mesh-bvh-async.ts';
export * from './top-level.ts';
export * from './scene-assembly.ts';
export * from './wgsl.ts';
export * from './trace-gpu.ts';
export * from './cpu-traversal.ts';
