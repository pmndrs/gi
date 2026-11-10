/**
 * Vitest setup for running the three adapter's tests with the T3.0 matrix
 * watch on every object added to a parent (vitest.watch.config.ts).
 */

import { Object3D } from 'three';
import { afterAll, expect } from 'vitest';
import { watchEverythingAdded } from './watch-everything.ts';

const everything = watchEverythingAdded(Object3D);

afterAll(() => {
    // A suite that builds scenes must have exercised the override; the count shows it did.
    if (process.env.T30_WATCH_REPORT) console.log(`matrix watch: ${everything.installs()} objects watched in ${expect.getState().testPath ?? 'this file'}`);
});
