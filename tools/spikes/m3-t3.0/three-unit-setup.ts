/**
 * Runs three.js's own unit tests (a checkout of three's repo at tag r187, not
 * in this repo) with the T3.0 matrix watch on every object added to a parent.
 * Loaded with `node --import` before QUnit, so the patch is in place before
 * any test module runs, on the same `src/core/Object3D.js` module instance
 * the tests import:
 *
 *   THREE_REPO=<three checkout> T30_WATCH=1 node --import ./three-unit-setup.ts <qunit cli> <three>/test/unit/utils/console-wrapper.js <test files...>
 *
 * Without T30_WATCH=1 it does nothing, for the baseline run.
 */

import { pathToFileURL } from 'node:url';
import { watchEverythingAdded } from './watch-everything.ts';

const repo = process.env.THREE_REPO;
if (process.env.T30_WATCH === '1') {
    if (!repo) throw new Error('set THREE_REPO to a three.js checkout at r187');
    const { Object3D } = (await import(pathToFileURL(`${repo}/src/core/Object3D.js`).href)) as { Object3D: Parameters<typeof watchEverythingAdded>[0] };
    const everything = watchEverythingAdded(Object3D);
    process.on('exit', () => console.log(`matrix watch: ${everything.installs()} objects watched`));
}
