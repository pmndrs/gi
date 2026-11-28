/**
 * How the scene sync hears about changes the moment three makes them, rather
 * than when its budgeted walk gets round to the object (T3.7;
 * docs/design/m3-moving.md §6). Two mechanisms, both plain three r187 API:
 *
 * - **Structure:** three dispatches `childadded` and `childremoved` on the
 *   parent whenever `add`, `remove`, `attach`, `clear` or `removeFromParent`
 *   changes its children (`Object3D.js`). {@link StructureWatch} listens on
 *   every object of the tracked scene, so an added or removed object is known
 *   in the update after it happens, wherever it is in the tree.
 * - **World matrices:** three keeps no version for `matrixWorld`. Its render
 *   calls `scene.updateMatrixWorld()`, which calls `updateMatrixWorld` on
 *   every object, parents first. {@link hookMatrixWorld} replaces that method
 *   on one object (an own property; the prototype is untouched) with one that
 *   calls the original and then a callback. The callback compares the fresh
 *   matrix while it is still in the CPU cache. T3.0 measured about 50 ns an
 *   object on Bistro interior, with three's own unit tests unchanged
 *   (DECISIONS 2026-11-09).
 *
 * Why only meshes need the hook: three's `updateMatrixWorld` always calls it
 * on every child, whether or not the parent changed, so a moved group's
 * meshes each get their own call. Hooking the groups too would add cost and
 * find nothing more.
 */

import type { Object3D, Object3DEventMap } from 'three';

type UpdateMatrixWorld = (this: Object3D, force?: boolean) => void;

/** One object's hook: what was there before, and what we put there. */
export interface MatrixHook {
    readonly object: Object3D;
    /** The method the object had (three's, or a subclass's such as `SkinnedMesh`'s), restored by {@link unhookMatrixWorld}. */
    readonly original: UpdateMatrixWorld;
    readonly replacement: UpdateMatrixWorld;
}

/**
 * Makes `object.updateMatrixWorld` call `afterUpdate` each time it has run.
 * The callback runs inside three's render (or wherever the host updates
 * matrices), so it must only compare and queue, never send.
 */
export function hookMatrixWorld(object: Object3D, afterUpdate: () => void): MatrixHook {
    const original = object.updateMatrixWorld as UpdateMatrixWorld;
    const replacement: UpdateMatrixWorld = function (this: Object3D, force?: boolean): void {
        original.call(this, force);
        afterUpdate();
    };
    object.updateMatrixWorld = replacement;
    return { object, original, replacement };
}

/**
 * Gives the object its own method back, by assignment. Never `delete` the
 * property: after deleting it on Bistro interior's objects, three's own
 * `updateMatrixWorld` ran 3× slower (V8 turns an object with a deleted
 * property into a slow dictionary object; T3.0, NOTES). The own property then
 * points at the method the prototype has, which costs nothing.
 *
 * If someone wrapped our hook since, theirs stays in place and still calls
 * ours; the caller's callback must then ignore a record it has destroyed.
 */
export function unhookMatrixWorld(hook: MatrixHook): void {
    if (hook.object.updateMatrixWorld === hook.replacement) hook.object.updateMatrixWorld = hook.original;
}

/** Whether the hook is still what three will call (nobody replaced it since). */
export function matrixHookInPlace(hook: MatrixHook): boolean {
    return hook.object.updateMatrixWorld === hook.replacement;
}

/**
 * Listens for children added to or removed from any object of a tree, and
 * reports the child (the root of what moved in or out). The listeners follow
 * the tree: an added subtree gets them, a removed one loses them, so nothing
 * outside the tracked scene keeps a reference to the sync.
 *
 * One listener function per kind is shared by every object, so three's own
 * duplicate test (`addEventListener` ignores a listener already there) keeps
 * each object at one listener of each kind.
 */
export class StructureWatch {
    private root: Object3D | null = null;
    private readonly changed: (child: Object3D) => void;

    private readonly onChildAdded = (event: Object3DEventMap['childadded']): void => {
        this.listenTree(event.child);
        this.changed(event.child);
    };

    private readonly onChildRemoved = (event: Object3DEventMap['childremoved']): void => {
        this.unlistenTree(event.child);
        this.changed(event.child);
    };

    /** `changed` receives every child added or removed (also a child moved between parents, once for each event). */
    constructor(changed: (child: Object3D) => void) {
        this.changed = changed;
    }

    /** Listens on `root` and everything under it; stops listening on the previous root. */
    attach(root: Object3D): void {
        if (root === this.root) return;
        this.detach();
        this.root = root;
        this.listenTree(root);
    }

    detach(): void {
        if (this.root) this.unlistenTree(this.root);
        this.root = null;
    }

    private listenTree(tree: Object3D): void {
        tree.traverse((object) => {
            object.addEventListener('childadded', this.onChildAdded);
            object.addEventListener('childremoved', this.onChildRemoved);
        });
    }

    private unlistenTree(tree: Object3D): void {
        // three's removeEventListener splices its array; it never deletes a property (see unhookMatrixWorld).
        tree.traverse((object) => {
            object.removeEventListener('childadded', this.onChildAdded);
            object.removeEventListener('childremoved', this.onChildRemoved);
        });
    }
}
