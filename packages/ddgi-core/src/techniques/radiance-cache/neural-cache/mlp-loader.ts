/**
 * Loads `@pathtracer/mlp` (pmndrs/pathtracer's network, MIT) at run time, for
 * the neural tail only (P6.4). The package is an optional sibling from a
 * private repository (root package.json `linkedSiblings`): nothing imports it
 * statically, so a checkout or a bundle without it builds and runs, and the
 * tail turns off with one warning. Where it is missing, or its clone isn't at
 * the pinned commit, vite and vitest resolve the import to a module that
 * throws the reason (tools/optional-siblings.ts), so this rejects with it.
 */

import type * as Mlp from '@pathtracer/mlp';

export type MlpModule = typeof Mlp;

export function loadMlp(): Promise<MlpModule> {
    return import('@pathtracer/mlp');
}
