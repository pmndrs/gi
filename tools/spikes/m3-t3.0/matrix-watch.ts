/**
 * T3.0 spike, question 2 (docs/design/m3-moving.md §6): find moved objects by
 * comparing each world matrix right after three computes it, while it is still
 * in the cache, instead of walking the scene later.
 *
 * `install(object)` replaces `updateMatrixWorld` on that object only (an own
 * property; the prototype is untouched). The replacement calls the method the
 * object had (three's, or a subclass's such as `SkinnedMesh`'s or `Camera`'s),
 * then compares the fresh `matrixWorld` with the copy taken when it last
 * changed. A difference copies it and queues the object; `takeMoved()` hands
 * the queue over (the adapter's next `update`).
 *
 * Why compare against our own copy, not "before and after": three also writes
 * `matrixWorld` outside `updateMatrixWorld` (`updateWorldMatrix`, from
 * `getWorldPosition`, `lookAt`, `Box3.setFromObject`, ...). Then the matrix
 * is already new when `updateMatrixWorld` runs and before == after, but it
 * still differs from the copy we hold.
 *
 * Not a package API: spike code. T3.7 decides whether the adapter adopts it.
 */

import type { Object3D } from 'three';

const MATRIX_FLOATS = 16;

type UpdateMatrixWorld = (this: Object3D, force?: boolean) => void;

interface Watched {
    readonly slot: number;
    /** The method the object had before `install`, restored by `uninstall`. */
    readonly original: UpdateMatrixWorld;
    readonly replacement: UpdateMatrixWorld;
}

export class MatrixWatch {
    /** Each watched object's last seen world matrix, 16 doubles per slot (three keeps doubles). */
    private snapshots = new Float64Array(MATRIX_FLOATS * 256);
    /** 1 while a slot's object is in `moved`, so it is queued once per hand-over. */
    private queued = new Uint8Array(256);
    private readonly freeSlots: number[] = [];
    private nextSlot = 0;
    private readonly watched = new Map<Object3D, Watched>();
    private moved: Object3D[] = [];

    get watchedCount(): number {
        return this.watched.size;
    }

    isWatched(object: Object3D): boolean {
        return this.watched.has(object);
    }

    /** Watches `object` (not its children). Its current matrix counts as already seen. */
    install(object: Object3D): void {
        if (this.watched.has(object)) return;
        const slot = this.takeSlot();
        this.snapshots.set(object.matrixWorld.elements, slot * MATRIX_FLOATS);
        this.queued[slot] = 0;
        const original = object.updateMatrixWorld as UpdateMatrixWorld;
        // eslint-disable-next-line @typescript-eslint/no-this-alias -- the replacement runs with three's `this` (the object) and needs the watch too
        const watch = this;
        const replacement: UpdateMatrixWorld = function (this: Object3D, force?: boolean): void {
            original.call(this, force);
            watch.compare(this, slot);
        };
        object.updateMatrixWorld = replacement;
        this.watched.set(object, { slot, original, replacement });
    }

    /** Watches `root` and everything under it. */
    installTree(root: Object3D): void {
        root.traverse((object) => this.install(object));
    }

    /** Stops watching `object` and gives it back its own method. */
    uninstall(object: Object3D): void {
        const entry = this.watched.get(object);
        if (!entry) return;
        // Put the original back by assignment, not `delete`: deleting the property left Bistro
        // interior's objects about 3x slower in three's own updateMatrixWorld afterwards (V8 turns
        // an object with a deleted property into a slow dictionary object; T3.0 spike). The own
        // property then points at the same method the prototype has, which costs nothing.
        // If someone wrapped ours since, theirs stays (and still calls ours, which then does nothing harmful).
        if (object.updateMatrixWorld === entry.replacement) object.updateMatrixWorld = entry.original;
        this.watched.delete(object);
        this.freeSlots.push(entry.slot);
        if (this.queued[entry.slot]) this.moved = this.moved.filter((candidate) => candidate !== object);
        this.queued[entry.slot] = 0;
    }

    /** The objects whose world matrix changed since the last call, each once. */
    takeMoved(): Object3D[] {
        const moved = this.moved;
        this.moved = [];
        for (const object of moved) {
            const entry = this.watched.get(object);
            if (entry) this.queued[entry.slot] = 0;
        }
        return moved;
    }

    dispose(): void {
        for (const object of [...this.watched.keys()]) this.uninstall(object);
    }

    /** The hot path: 16 compares on a matrix three just wrote. */
    private compare(object: Object3D, slot: number): void {
        const elements = object.matrixWorld.elements;
        const snapshots = this.snapshots;
        const base = slot * MATRIX_FLOATS;
        let changed = false;
        for (let index = 0; index < MATRIX_FLOATS; index++) {
            if (snapshots[base + index] !== elements[index]) {
                snapshots[base + index] = elements[index];
                changed = true;
            }
        }
        if (changed && this.queued[slot] === 0) {
            this.queued[slot] = 1;
            this.moved.push(object);
        }
    }

    private takeSlot(): number {
        const reused = this.freeSlots.pop();
        if (reused !== undefined) return reused;
        const slot = this.nextSlot++;
        if (slot >= this.queued.length) {
            const snapshots = new Float64Array(this.snapshots.length * 2);
            snapshots.set(this.snapshots);
            this.snapshots = snapshots;
            const queued = new Uint8Array(this.queued.length * 2);
            queued.set(this.queued);
            this.queued = queued;
        }
        return slot;
    }
}
