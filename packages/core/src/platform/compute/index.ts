// The platform's GPU utilities (RC1.3): prefix scan, stream compaction and
// indirect dispatch arguments. Techniques reach them through the contract
// (`TechniqueContext.compute`, platform/contract/compute.ts); platform code,
// tests and pages import this folder by path.
export { PlatformGpuCompute, type PlatformGpuComputeOptions } from './gpu-compute.ts';
export { compactOnCpu, exclusiveScanOnCpu, indirectArgsOnCpu } from './cpu-reference.ts';
export { INDIRECT_ARGS_BYTES, INDIRECT_ARGS_COUNT_WORD, INDIRECT_ARGS_MODULE } from './wgsl.ts';
