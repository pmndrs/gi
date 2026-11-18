/**
 * Collects what changed between two `encode` calls, as the contract's
 * `SceneChanges`: dirty world-space boxes with a reason, changes that have no
 * useful box (the sun, the environment, light groups), and every changed
 * light with its old and new state (docs/design/m2-lights.md §1).
 */

import type { InstanceHandle, LightHandle } from '../contract/handles.ts';
import type { Aabb } from '../contract/math.ts';
import type { LightDescriptor } from '../contract/scene.ts';
import type { ChangeReason, DirtyRegion, LightChange, SceneChanges } from '../contract/scene-copy.ts';

/**
 * Past this many regions in one frame, the regions of each reason are merged
 * into one box. A technique gains little from hundreds of small boxes, and
 * the list stays cheap to walk.
 */
export const MAX_DIRTY_REGIONS = 256;

/**
 * Up to this many light changes a frame are listed one by one in
 * `SceneChanges.lights`; past it the frame counts as a global light change
 * (m2-lights.md §1). 1,024 is `sponza-candles`' light count (Dennis,
 * 2026-11-05), so that scene's lights can all change in one frame and still
 * be handled light by light.
 */
export const MAX_LISTED_LIGHT_CHANGES = 1024;

/**
 * A light's influence ends where its irradiance (without three's range
 * window) falls below this share of the scene's reference irradiance
 * (m2-lights.md §1).
 */
export const INFLUENCE_IRRADIANCE_SHARE = 1 / 1000;

type GlobalChange = SceneChanges['global'][number];

/** `[minX, minY, minZ, maxX, maxY, maxZ]` to the contract's box. */
export function toAabb(box: ArrayLike<number>): Aabb {
    return { min: [box[0], box[1], box[2]], max: [box[3], box[4], box[5]] };
}

/** What the change log needs to know about the lights at `take` time. */
export interface LightLookup {
    /** A live light's index in the packed table and its state now; undefined once destroyed. */
    current(light: LightHandle): { readonly index: number; readonly descriptor: LightDescriptor } | undefined;
    /** Every live light's state now. */
    all(): Iterable<LightDescriptor>;
}

/** The brightest colour channel times the intensity: irradiance at normal incidence (directional) or at 1 m (point, spot). */
function strongestChannel(light: LightDescriptor): number {
    return Math.max(light.color[0], light.color[1], light.color[2]) * light.intensity;
}

/**
 * The scene's reference irradiance E_ref (m2-lights.md §1): the brightest
 * directional light's, or, with none (or none above 0), the brightest local
 * light's at 1 m. Per colour channel, the strongest one. 0 when every light is
 * black.
 */
export function referenceIrradiance(lights: Iterable<LightDescriptor>): number {
    let directional = 0;
    let local = 0;
    for (const light of lights) {
        const strength = strongestChannel(light);
        if (light.type === 'directional') directional = Math.max(directional, strength);
        else local = Math.max(local, strength);
    }
    return directional > 0 ? directional : local;
}

/**
 * How far from a light a change to it can matter, in metres (m2-lights.md
 * §1): where `I / d²` drops below `INFLUENCE_IRRADIANCE_SHARE × E_ref`, so
 * `d = sqrt(I / (share × E_ref))`, capped by the light's `range` when it has
 * one. Spot lights use their on-axis intensity (their cone only narrows it).
 * Infinity for directional lights, 0 for no light or a black one.
 */
export function influenceRadius(light: LightDescriptor | null, reference: number): number {
    if (!light) return 0;
    if (light.type === 'directional') return Infinity;
    const intensity = strongestChannel(light);
    if (!(intensity > 0)) return 0;
    // A lit local light makes the reference at least its own strength, so this only guards the arithmetic.
    const radius = reference > 0 ? Math.sqrt(intensity / (INFLUENCE_IRRADIANCE_SHARE * reference)) : Infinity;
    return light.range > 0 ? Math.min(radius, light.range) : radius;
}

/** A copy that shares no arrays with the host's objects, so a host editing its vectors in place can't rewrite history. */
export function copyLight(light: LightDescriptor): LightDescriptor {
    const copy: Record<string, unknown> = { ...light };
    for (const key of ['color', 'position', 'direction']) {
        const value = copy[key];
        if (Array.isArray(value)) copy[key] = [...value];
    }
    return copy as unknown as LightDescriptor;
}

function sameValue(a: unknown, b: unknown): boolean {
    if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((value, index) => Object.is(value, b[index]));
    return Object.is(a, b);
}

/** Two states of a light are the same when every field is (labels included: harmless, and simpler than a list of the fields that light). */
export function sameLight(a: LightDescriptor, b: LightDescriptor): boolean {
    const first = a as unknown as Record<string, unknown>;
    const second = b as unknown as Record<string, unknown>;
    const keys = new Set([...Object.keys(first), ...Object.keys(second)]);
    for (const key of keys) {
        if (!sameValue(first[key], second[key])) return false;
    }
    return true;
}

const REASONS: readonly ChangeReason[] = ['geometry', 'material', 'light'];
/** A reason's index in REASONS. A switch, not an object lookup by string: this runs twice per moved instance. */
function reasonCode(reason: ChangeReason): number {
    switch (reason) {
        case 'geometry':
            return 0;
        case 'material':
            return 1;
        case 'light':
            return 2;
    }
}
/** Stands for "none" in the typed columns: no move distance, no instance slot, no source. */
const NO_VALUE = -1;
/**
 * A frame's regions in flat typed columns, so logging one costs no
 * allocation: `setTransforms` logs two per moved instance, and 10,000
 * instances moved in a frame made 20,000 region objects one by one (B1a
 * measured the old `setTransform` at about 2 µs a call). The contract's
 * `DirtyRegion` objects are made at `take`, at most MAX_DIRTY_REGIONS of
 * them. Past that many in a frame the regions of each reason are merged
 * anyway, so from then on each new one is folded into a running box per
 * reason as it arrives, and nothing more is stored.
 */
class RegionColumns {
    /** Regions stored, up to MAX_DIRTY_REGIONS. */
    private count = 0;
    private readonly boxes = new Float32Array(MAX_DIRTY_REGIONS * 6);
    private readonly reasons = new Uint8Array(MAX_DIRTY_REGIONS);
    private readonly moved = new Float32Array(MAX_DIRTY_REGIONS);
    private readonly instances = new Int32Array(MAX_DIRTY_REGIONS);
    /** 1 where the region's instance gives off light (`DirtyRegion.emits`). */
    private readonly emits = new Uint8Array(MAX_DIRTY_REGIONS);
    /** Handles are positive integers (handles.ts); NO_VALUE when there is none. */
    private readonly sources = new Float64Array(MAX_DIRTY_REGIONS);
    /** Past MAX_DIRTY_REGIONS: one running box per reason (min then max), the largest move, and whether every region was a move. */
    private merging = false;
    private readonly mergedBoxes = new Float64Array(REASONS.length * 6);
    private readonly mergedLargestMove = new Float64Array(REASONS.length);
    private readonly mergedAllMoves = new Uint8Array(REASONS.length);
    private readonly mergedPresent = new Uint8Array(REASONS.length);
    private readonly mergedEmits = new Uint8Array(REASONS.length);

    /** A region whose box is the 6 numbers from `boxAt` in `box`. */
    add(box: ArrayLike<number>, boxAt: number, reason: ChangeReason, source: number | undefined, moved: number | undefined, instance: number | undefined, emits: boolean): void {
        const code = reasonCode(reason);
        const movedValue = moved ?? NO_VALUE;
        if (!this.merging && this.count === MAX_DIRTY_REGIONS) this.startMerging();
        if (this.merging) {
            this.fold(box, boxAt, code, movedValue, emits);
            return;
        }
        const index = this.count++;
        const at = index * 6;
        for (let value = 0; value < 6; value++) this.boxes[at + value] = box[boxAt + value];
        this.reasons[index] = code;
        this.moved[index] = movedValue;
        this.instances[index] = instance ?? NO_VALUE;
        this.sources[index] = source ?? NO_VALUE;
        this.emits[index] = emits ? 1 : 0;
    }

    /** The regions as the contract has them, merged by reason when there were more than MAX_DIRTY_REGIONS. */
    take(): DirtyRegion[] {
        const regions = this.merging ? this.merged() : this.each();
        this.count = 0;
        this.merging = false;
        return regions;
    }

    private each(): DirtyRegion[] {
        const regions: DirtyRegion[] = [];
        for (let index = 0; index < this.count; index++) {
            const source = this.sources[index];
            const moved = this.moved[index];
            const instance = this.instances[index];
            regions.push({
                bounds: toAabb(this.boxes.subarray(index * 6, index * 6 + 6)),
                reason: REASONS[this.reasons[index]],
                ...(source !== NO_VALUE ? { source: source as InstanceHandle | LightHandle } : {}),
                ...(moved >= 0 ? { moved } : {}),
                ...(instance !== NO_VALUE ? { instance } : {}),
                ...(this.emits[index] ? { emits: true } : {}),
            });
        }
        return regions;
    }

    /** Switches to merging: the regions stored so far go into the running boxes. */
    private startMerging(): void {
        this.merging = true;
        for (let reason = 0; reason < REASONS.length; reason++) {
            this.mergedBoxes.fill(Infinity, reason * 6, reason * 6 + 3);
            this.mergedBoxes.fill(-Infinity, reason * 6 + 3, reason * 6 + 6);
        }
        this.mergedLargestMove.fill(0);
        this.mergedAllMoves.fill(1);
        this.mergedPresent.fill(0);
        this.mergedEmits.fill(0);
        for (let index = 0; index < this.count; index++) this.fold(this.boxes, index * 6, this.reasons[index], this.moved[index], this.emits[index] === 1);
        this.count = 0;
    }

    private fold(box: ArrayLike<number>, at: number, reason: number, moved: number, emits: boolean): void {
        const base = reason * 6;
        for (let axis = 0; axis < 3; axis++) {
            if (box[at + axis] < this.mergedBoxes[base + axis]) this.mergedBoxes[base + axis] = box[at + axis];
            if (box[at + 3 + axis] > this.mergedBoxes[base + 3 + axis]) this.mergedBoxes[base + 3 + axis] = box[at + 3 + axis];
        }
        this.mergedPresent[reason] = 1;
        if (emits) this.mergedEmits[reason] = 1;
        if (moved >= 0) {
            if (moved > this.mergedLargestMove[reason]) this.mergedLargestMove[reason] = moved;
        } else {
            this.mergedAllMoves[reason] = 0;
        }
    }

    /**
     * One box per reason. A merged geometry region carries the largest move
     * when every region in it was a move, and none (a whole change) when any
     * wasn't; it names no instance, and emits if any region in it did.
     */
    private merged(): DirtyRegion[] {
        const regions: DirtyRegion[] = [];
        for (let reason = 0; reason < REASONS.length; reason++) {
            if (!this.mergedPresent[reason]) continue;
            const box = this.mergedBoxes.subarray(reason * 6, reason * 6 + 6);
            regions.push({
                bounds: toAabb(box),
                reason: REASONS[reason],
                ...(this.mergedAllMoves[reason] ? { moved: this.mergedLargestMove[reason] } : {}),
                ...(this.mergedEmits[reason] ? { emits: true } : {}),
            });
        }
        return regions;
    }
}

/**
 * Past this many receiver boxes in one frame, they are merged into one box
 * (`SceneChanges.receivers`). A crowd of characters moving every frame would
 * otherwise list thousands, while a technique only needs to know roughly
 * where receivers are.
 */
export const MAX_RECEIVER_BOXES = 256;

/**
 * A frame's receiver boxes in one flat typed array, so logging one allocates
 * nothing: `setTransforms` can move thousands of receivers a frame. Past
 * MAX_RECEIVER_BOXES the boxes are folded into one running box.
 */
class ReceiverBoxes {
    private count = 0;
    private readonly boxes = new Float32Array(MAX_RECEIVER_BOXES * 6);
    private merging = false;
    /** Past MAX_RECEIVER_BOXES: the running box, min then max. */
    private readonly mergedBox = new Float64Array(6);

    /** A box: the 6 numbers from `boxAt` in `box`, min then max. */
    add(box: ArrayLike<number>, boxAt: number): void {
        if (!this.merging && this.count === MAX_RECEIVER_BOXES) this.startMerging();
        if (this.merging) {
            this.fold(box, boxAt);
            return;
        }
        const at = this.count++ * 6;
        for (let value = 0; value < 6; value++) this.boxes[at + value] = box[boxAt + value];
    }

    take(): Aabb[] {
        const boxes: Aabb[] = [];
        if (this.merging) {
            boxes.push(toAabb(this.mergedBox));
        } else {
            for (let index = 0; index < this.count; index++) boxes.push(toAabb(this.boxes.subarray(index * 6, index * 6 + 6)));
        }
        this.count = 0;
        this.merging = false;
        return boxes;
    }

    private startMerging(): void {
        this.merging = true;
        this.mergedBox.fill(Infinity, 0, 3);
        this.mergedBox.fill(-Infinity, 3, 6);
        for (let index = 0; index < this.count; index++) this.fold(this.boxes, index * 6);
        this.count = 0;
    }

    private fold(box: ArrayLike<number>, at: number): void {
        for (let axis = 0; axis < 3; axis++) {
            if (box[at + axis] < this.mergedBox[axis]) this.mergedBox[axis] = box[at + axis];
            if (box[at + 3 + axis] > this.mergedBox[3 + axis]) this.mergedBox[3 + axis] = box[at + 3 + axis];
        }
    }
}

/** Receiver boxes merged into one, as a frame's are past MAX_RECEIVER_BOXES. */
function mergeReceivers(boxes: readonly Aabb[]): Aabb[] {
    const merged = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (const box of boxes) {
        for (let axis = 0; axis < 3; axis++) {
            merged[axis] = Math.min(merged[axis], box.min[axis]);
            merged[axis + 3] = Math.max(merged[axis + 3], box.max[axis]);
        }
    }
    return [toAabb(merged)];
}

export class ChangeLog {
    private readonly regions = new RegionColumns();
    private readonly global = new Set<GlobalChange>();
    private readonly receivers = new ReceiverBoxes();
    /** Each light touched this frame, with its state before the frame's first change (null: created this frame). In first-touched order. */
    private readonly lightsBefore = new Map<LightHandle, LightDescriptor | null>();

    /** A region; `moved` and `instance` as `DirtyRegion` has them, its box the 6 numbers from `boxAt` in `box` (positional, so logging one allocates nothing). */
    addRegion(box: ArrayLike<number>, reason: ChangeReason, source?: InstanceHandle | LightHandle, moved?: number, instance?: number, boxAt = 0, emits = false): void {
        this.regions.add(box, boxAt, reason, source, moved, instance, emits);
    }

    addGlobal(change: GlobalChange): void {
        this.global.add(change);
    }

    /** A receiver's world box where it is now (`SceneChanges.receivers`): the 6 numbers from `boxAt` in `box`. */
    addReceiver(box: ArrayLike<number>, boxAt = 0): void {
        this.receivers.add(box, boxAt);
    }

    /**
     * Call before a light is created (`before` null), changed or destroyed
     * (`before` its state until now). Only the first call in a frame keeps its
     * state, so several edits arrive as one change from where the frame began.
     */
    noteLight(light: LightHandle, before: LightDescriptor | null): void {
        if (!this.lightsBefore.has(light)) this.lightsBefore.set(light, before ? copyLight(before) : null);
    }

    /** Everything since the last call, merged; clears the log. */
    take(lights: LightLookup): SceneChanges {
        const lightChanges = this.lightChanges(lights);
        const lightsOverflowed = lightChanges.length > MAX_LISTED_LIGHT_CHANGES;
        for (const change of lightChanges) this.logLightChange(change);
        if (lightsOverflowed) this.global.add('lights');

        const changes: SceneChanges = {
            regions: this.regions.take(),
            global: [...this.global],
            receivers: this.receivers.take(),
            lights: lightsOverflowed ? [] : lightChanges,
            lightsOverflowed,
        };
        this.global.clear();
        this.lightsBefore.clear();
        return changes;
    }

    /** The frame's light changes with their influence radii, leaving out lights that ended where they began. */
    private lightChanges(lights: LightLookup): LightChange[] {
        if (this.lightsBefore.size === 0) return [];
        // The reference counts the lights as they were too, so a light switched off is measured
        // against a scene that still had it (otherwise turning off the only lamp would leave no reference).
        const everyState: LightDescriptor[] = [...lights.all()];
        for (const before of this.lightsBefore.values()) if (before) everyState.push(before);
        const reference = referenceIrradiance(everyState);

        const changes: LightChange[] = [];
        for (const [light, before] of this.lightsBefore) {
            const now = lights.current(light);
            const after = now ? copyLight(now.descriptor) : null;
            if (!before && !after) continue; // created and destroyed in one frame
            if (before && after && sameLight(before, after)) continue;
            changes.push({
                light,
                index: now ? now.index : -1,
                before,
                after,
                influenceRadius: { before: influenceRadius(before, reference), after: influenceRadius(after, reference) },
            });
        }
        return changes;
    }

    /**
     * The boxes and global flags a light change stands for: directional lights
     * are global ('sun'); a local light dirties a box of its influence radius
     * around where it was and where it is. A group change is global ('lights').
     */
    private logLightChange(change: LightChange): void {
        const sides: Array<[LightDescriptor | null, number]> = [
            [change.before, change.influenceRadius.before],
            [change.after, change.influenceRadius.after],
        ];
        for (const [light, radius] of sides) {
            if (!light) continue;
            if (light.type === 'directional') {
                this.global.add('sun');
            } else if (radius > 0) {
                const [x, y, z] = light.position;
                this.addRegion([x - radius, y - radius, z - radius, x + radius, y + radius, z + radius], 'light', change.light);
            }
        }
        if (change.before && change.after && (change.before.group ?? 0) !== (change.after.group ?? 0)) this.global.add('lights');
    }
}

/**
 * Two frames' changes as one, `earlier` first: for frames a technique didn't
 * see (the budget was paused), so it gets them all at its next frame instead
 * of losing them. A light changed in both keeps its state from before the
 * earlier frame and its state after the later one; if that comes back to
 * where it began, it is left out, as within a frame. Past
 * MAX_LISTED_LIGHT_CHANGES the merged list overflows like a single frame's.
 */
export function mergeSceneChanges(earlier: SceneChanges, later: SceneChanges): SceneChanges {
    const lights = new Map<LightHandle, LightChange>();
    for (const change of earlier.lights) lights.set(change.light, change);
    for (const change of later.lights) {
        const first = lights.get(change.light);
        if (!first) {
            lights.set(change.light, change);
            continue;
        }
        const merged: LightChange = { ...change, before: first.before, influenceRadius: { before: first.influenceRadius.before, after: change.influenceRadius.after } };
        const unchanged = (!merged.before && !merged.after) || (merged.before && merged.after && sameLight(merged.before, merged.after));
        if (unchanged) lights.delete(change.light);
        else lights.set(change.light, merged);
    }
    const lightsOverflowed = earlier.lightsOverflowed || later.lightsOverflowed || lights.size > MAX_LISTED_LIGHT_CHANGES;
    const global = new Set([...earlier.global, ...later.global]);
    if (lightsOverflowed) global.add('lights');
    let regions = [...earlier.regions, ...later.regions];
    if (regions.length > MAX_DIRTY_REGIONS) regions = mergeByReason(regions);
    let receivers = [...earlier.receivers, ...later.receivers];
    if (receivers.length > MAX_RECEIVER_BOXES) receivers = mergeReceivers(receivers);
    return {
        regions,
        global: [...global],
        receivers,
        lights: lightsOverflowed ? [] : [...lights.values()],
        lightsOverflowed,
    };
}

/** One box per reason, as RegionColumns merges a frame's: the largest move when every region merged was a move, no instance. */
function mergeByReason(regions: readonly DirtyRegion[]): DirtyRegion[] {
    const merged = new Map<ChangeReason, { box: number[]; largestMove: number; allMoves: boolean }>();
    for (const region of regions) {
        const entry = merged.get(region.reason) ?? { box: [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity], largestMove: 0, allMoves: true };
        for (let axis = 0; axis < 3; axis++) {
            entry.box[axis] = Math.min(entry.box[axis], region.bounds.min[axis]);
            entry.box[axis + 3] = Math.max(entry.box[axis + 3], region.bounds.max[axis]);
        }
        if (region.moved === undefined) entry.allMoves = false;
        else entry.largestMove = Math.max(entry.largestMove, region.moved);
        merged.set(region.reason, entry);
    }
    return [...merged].map(([reason, entry]) => ({ bounds: toAabb(entry.box), reason, ...(entry.allMoves ? { moved: entry.largestMove } : {}) }));
}
