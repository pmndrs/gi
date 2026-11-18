/**
 * Puts a `MatrixWatch` on every object that is added to a parent (and on the
 * parent), by wrapping `Object3D.prototype.add`. That is roughly what the
 * three adapter would watch (everything in a tracked scene), so test suites
 * that build scenes run with the override in place: three's own unit tests
 * (three-unit-setup.ts) and our adapter's (vitest-setup.ts).
 *
 * Spike code for T3.0; nothing in `packages/` uses it.
 */

import type { Object3D } from 'three';
import { MatrixWatch } from './matrix-watch.ts';

interface Object3DClass {
    prototype: Object3D;
}

export interface WatchEverything {
    readonly watch: MatrixWatch;
    /** Objects that got the override, over the whole run. */
    installs(): number;
}

export function watchEverythingAdded(object3D: Object3DClass): WatchEverything {
    const watch = new MatrixWatch();
    let installs = 0;
    const install = (object: Object3D): void => {
        if (watch.isWatched(object)) return;
        watch.install(object);
        installs++;
    };
    const originalAdd = object3D.prototype.add;
    object3D.prototype.add = function (this: Object3D, ...objects: Object3D[]): Object3D {
        const result = originalAdd.apply(this, objects);
        install(this);
        for (const object of objects) if (object.parent === this) object.traverse(install);
        // Nobody hands the queue over in a test suite; drop it so it doesn't hold every moved object.
        watch.takeMoved();
        return result;
    };
    return { watch, installs: () => installs };
}
