/**
 * A type-level check that `mlp-unavailable.d.ts` (the declarations a checkout
 * without the optional sibling type-checks against) still matches
 * `@pathtracer/mlp` (P6.4's review). Where the sibling is cloned, tsconfig
 * `paths` resolves `@pathtracer/mlp` to the real package and `pnpm typecheck`
 * fails here if it has drifted from the declarations; without the clone both
 * imports are the declarations and it holds trivially. Types only: nothing
 * imports this file, and it emits nothing.
 */

import type * as Package from '@pathtracer/mlp';
import type * as Declared from './mlp-unavailable';

/** `true` only when `A` is assignable to `B`. */
type Assignable<A, B> = [A] extends [B] ? true : false;

/**
 * The real module's values fit the declared ones (what the tail calls exists
 * with those signatures), the shapes agree both ways, and the options the
 * tail builds against the declarations are options the real engine takes.
 */
export type MlpDeclarationsMatch = [
    // The engine apart: its private constructor makes the two class types nominal, so its factory and instances are compared.
    Assignable<Omit<typeof Package, 'MlpEngine'>, Omit<typeof Declared, 'MlpEngine'>>,
    Assignable<Pick<typeof Package.MlpEngine, 'create'>, Pick<typeof Declared.MlpEngine, 'create'>>,
    Assignable<Package.MlpEngine, Declared.MlpEngine>,
    Assignable<Package.MlpShape, Declared.MlpShape>,
    Assignable<Declared.MlpShape, Package.MlpShape>,
    Assignable<Declared.TrainSettings, Package.TrainSettings>,
    Assignable<Declared.MlpEngineOptions, Package.MlpEngineOptions>,
];

export const MLP_DECLARATIONS_MATCH: MlpDeclarationsMatch = [true, true, true, true, true, true, true];
