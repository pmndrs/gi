/**
 * The platform contract (T0.11 draft): types only, plus two error classes and
 * the change-event tracker techniques share (`change-events.ts`).
 * Explained in plain language in docs/api/platform.md.
 */

export * from './change-events';
export * from './errors';
export type * from './compute';
export type * from './frame';
export type * from './handles';
export type * from './math';
export type * from './platform';
export type * from './queries';
export type * from './scene';
export type * from './scene-copy';
export type * from './shade';
export type * from './shading-module';
export type * from './stats';
export type * from './technique';
export type * from './timing';
export type * from './wgsl';
