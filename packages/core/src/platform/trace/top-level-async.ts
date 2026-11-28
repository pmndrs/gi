/**
 * Builds and checks top-level tree shapes off the main thread:
 * `@pmndrs/bvh/worker`'s builder, which works on the calling thread (on a
 * later turn) where there is no `Worker` (Node tests) or the worker can't
 * start or load. Its shapes are `TopLevelShape`s (top-level-shape.ts holds the
 * same types, and gi's inline twin of the builder for tools).
 *
 * gi started a worker of its own here until it moved onto the published
 * `@pmndrs/bvh` (pmndrs/gi#63), as mesh-bvh-async.ts did.
 */

export { createTopLevelBuilder, type TopLevelBuilder, type TopLevelBuilderConfig, type TopLevelResult } from '@pmndrs/bvh/worker';
