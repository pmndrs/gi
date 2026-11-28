/**
 * Loads `@pathtracer/mlp` (pmndrs/pathtracer's network, MIT) at run time, for
 * the neural tail only (P6.4). The package is an optional sibling from a
 * private repository (root package.json `linkedSiblings`): nothing imports it
 * statically, so a checkout or a bundle without it builds and runs, and the
 * tail turns off with one warning. Where it is missing, or its clone isn't at
 * the pinned commit, vite and vitest resolve the import to a module that
 * throws the reason (tools/optional-siblings.ts), so this rejects with it.
 *
 * In the published `@pmndrs/gi/radiance-cache` the package stays external and
 * isn't a dependency: the build writes this `import()` so a consumer's bundler
 * leaves it alone (packages/ddgi/vite.config.ts), and it rejects at run time
 * where the package can't be resolved. Its types are inlined from
 * `mlp-unavailable.d.ts` (packages/ddgi/tsconfig.build.json).
 */

import type * as Mlp from '@pathtracer/mlp';

export type MlpModule = typeof Mlp;

export function loadMlp(): Promise<MlpModule> {
    return import('@pathtracer/mlp');
}
