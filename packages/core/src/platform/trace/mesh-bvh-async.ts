/**
 * Builds mesh BVHs off the main thread: `@pmndrs/bvh/worker`'s builder, a
 * pool of one tinybvh module worker, which builds on the calling thread where
 * there is no `Worker` (Node tests) or the worker can't start or load.
 *
 * gi started a worker of its own here until it moved onto the published
 * `@pmndrs/bvh` (pmndrs/gi#63). The package's worker is shipped by the user's
 * bundler from the package itself, so `@pmndrs/gi` bundles no tinybvh; under
 * Vite's dev server, keep `@pmndrs/bvh` out of `optimizeDeps` (README).
 */

export { createMeshBvhBuilder, type MeshBvhBuilder, type MeshBvhBuilderConfig } from '@pmndrs/bvh/worker';
