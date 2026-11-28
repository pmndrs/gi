/**
 * The library as the viewer uses it: the published package's entry points
 * `@pmndrs/gi/three`, `@pmndrs/gi/core` and `@pmndrs/gi/ddgi`, read from source.
 *
 * Why a relative path instead of the package name: the viewer's own vite
 * config aliases `@pmndrs/gi/*` to source, but the GPU test runner serves
 * viewer modules (the baseline benchmark page) from a vite server with no
 * config, where the package name would resolve to an unbuilt `dist/`. These
 * are the same files the aliases point at, so only the public surface of the
 * package is used either way.
 */

export * from '../../../../packages/gi/src/three';
export * from '../../../../packages/gi/src/core';
export * from '../../../../packages/gi/src/ddgi';
