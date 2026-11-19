/**
 * Scene changes as events over time, for techniques that forget history
 * where the scene changed (the radiance cache's world cells, E8 and E9, and
 * its screen probes, D46; later the GI denoiser's change reset). Engine-free,
 * CPU only, no GPU resources: a technique feeds it each frame's
 * `SceneChanges` and reads the events back.
 *
 * Why events, not each frame's boxes: a change that goes on (a moving object,
 * an animated sun) dirties the same place every frame. Resetting history
 * there every frame keeps it at one frame's samples, as noisy as no history
 * at all; and a list of each frame's boxes overflows after a few frames into
 * "everything changed". Here a change is an event with a start frame and a
 * box that grows to cover it (swept, merged), and a technique resets a cell
 * (or probe) at most once per epoch of the event: at its start, again every
 * `rearmFrames` while it goes on (so a long change is followed), and once
 * more when it stops (so the final state isn't mixed with the last frames of
 * the change). An event is forgotten `quietFrames` after it stops.
 *
 * Events are matched frame to frame by their source (the light or instance a
 * dirty region names); regions with no source join an event their box
 * overlaps. Past `maxEvents`, the two events whose union grows least are
 * merged (the later start kept, so their cells reset again): there is never
 * an escalation to "everything" from a count. A change with no bounds
 * (`SceneChanges.global`: the sun, the sky, a light group) is one event of
 * its own with no box.
 */

import type { Aabb } from './math';
import type { SceneChanges } from './scene-copy';

/** One change event as a technique reads it at a frame. */
export interface ChangeEvent {
    /** Where it is: the union of every box it dirtied; null for a change with no bounds (everywhere). */
    readonly box: Aabb | null;
    /** The first frame it dirtied. */
    readonly start: number;
    /** The latest frame it dirtied. */
    readonly last: number;
    /**
     * The start of its current epoch: a cell or probe that last forgot its
     * history before this frame forgets it now (once). `start` and every
     * `rearmFrames` after it while the change goes on; `last + 1` once it
     * stopped.
     */
    readonly epoch: number;
}

export interface ChangeEventOptions {
    /** Frames between re-arms of a change that goes on (default 8). */
    readonly rearmFrames?: number;
    /** Frames an event is kept after it stops (default 16). */
    readonly quietFrames?: number;
    /** The most events kept; past it the cheapest pair is merged (default 16). */
    readonly maxEvents?: number;
}

interface TrackedEvent {
    /** The sources it answers to: `light:<handle>`, `object:<handle>`, `global`; none for regions with no source. */
    keys: Set<string>;
    box: Aabb | null;
    start: number;
    last: number;
}

function union(a: Aabb, b: Aabb): Aabb {
    return {
        min: [Math.min(a.min[0], b.min[0]), Math.min(a.min[1], b.min[1]), Math.min(a.min[2], b.min[2])],
        max: [Math.max(a.max[0], b.max[0]), Math.max(a.max[1], b.max[1]), Math.max(a.max[2], b.max[2])],
    };
}

function overlaps(a: Aabb, b: Aabb): boolean {
    return a.min[0] <= b.max[0] && b.min[0] <= a.max[0] && a.min[1] <= b.max[1] && b.min[1] <= a.max[1] && a.min[2] <= b.max[2] && b.min[2] <= a.max[2];
}

function volume(box: Aabb): number {
    return Math.max(box.max[0] - box.min[0], 0) * Math.max(box.max[1] - box.min[1], 0) * Math.max(box.max[2] - box.min[2], 0);
}

export class ChangeEventTracker {
    readonly rearmFrames: number;
    readonly quietFrames: number;
    readonly maxEvents: number;
    private tracked: TrackedEvent[] = [];
    private frame = 0;

    constructor(options: ChangeEventOptions = {}) {
        this.rearmFrames = options.rearmFrames ?? 8;
        this.quietFrames = options.quietFrames ?? 16;
        this.maxEvents = options.maxEvents ?? 16;
        if (!(this.rearmFrames >= 1 && this.quietFrames >= 1 && this.maxEvents >= 1)) throw new RangeError('@pmndrs/gi: change events need rearmFrames, quietFrames and maxEvents of at least 1');
    }

    /** Adds frame `frame`'s changes (call once per frame, frames increasing; a frame with no changes may be skipped). */
    update(frame: number, changes: Pick<SceneChanges, 'regions' | 'global'>): void {
        this.frame = frame;
        if (changes.global.length > 0) this.touch('global', null, frame);
        for (const region of changes.regions) {
            const key = region.source === undefined ? null : `${region.reason === 'light' ? 'light' : 'object'}:${region.source}`;
            this.touch(key, region.bounds, frame);
        }
        this.tracked = this.tracked.filter((event) => frame - event.last < this.quietFrames + 1);
        while (this.tracked.filter((event) => event.box).length > this.maxEvents) this.mergeCheapest();
    }

    /** The events at `frame` (default: the latest updated), with their current epochs. */
    events(frame = this.frame): ChangeEvent[] {
        return this.tracked
            .filter((event) => frame - event.last <= this.quietFrames && event.start <= frame)
            .map((event) => ({ box: event.box, start: event.start, last: event.last, epoch: this.epochOf(event, frame) }));
    }

    /** The events whose epoch starts at `frame`: what a technique with no per-cell memory resets this frame. */
    armedEvents(frame = this.frame): ChangeEvent[] {
        return this.events(frame).filter((event) => event.epoch === frame);
    }

    /** Forgets every event (a technique's history reset). */
    clear(): void {
        this.tracked = [];
    }

    private activeEpoch(event: TrackedEvent, frame: number): number {
        return event.start + this.rearmFrames * Math.floor((frame - event.start) / this.rearmFrames);
    }

    private epochOf(event: TrackedEvent, frame: number): number {
        if (frame <= event.last) return this.activeEpoch(event, frame);
        // Stopped: once more after its last frame, unless its latest epoch began on that frame (a
        // one-frame change, or one that re-armed as it stopped), whose samples are the final state's.
        const latest = this.activeEpoch(event, event.last);
        return latest < event.last ? event.last + 1 : latest;
    }

    private touch(key: string | null, box: Aabb | null, frame: number): void {
        let event = key === null ? undefined : this.tracked.find((candidate) => candidate.keys.has(key));
        // A region with no source joins a sourceless event its box overlaps.
        if (!event && key === null && box) event = this.tracked.find((candidate) => candidate.keys.size === 0 && candidate.box !== null && overlaps(candidate.box, box));
        if (!event) {
            this.tracked.push({ keys: new Set(key === null ? [] : [key]), box, start: frame, last: frame });
            return;
        }
        // A change that stopped and starts again is a new event (it was quiet for more than a frame).
        if (frame - event.last > 1) event.start = frame;
        event.last = frame;
        if (event.box && box) event.box = union(event.box, box);
    }

    /** Merges the two boxed events whose union grows least; the merged one starts at the later start, so its cells reset again. */
    private mergeCheapest(): void {
        const boxed = this.tracked.filter((event) => event.box !== null);
        let best: [TrackedEvent, TrackedEvent] | null = null;
        let bestCost = Infinity;
        for (let i = 0; i < boxed.length; i++) {
            for (let j = i + 1; j < boxed.length; j++) {
                const a = boxed[i].box as Aabb;
                const b = boxed[j].box as Aabb;
                const cost = volume(union(a, b)) - volume(a) - volume(b);
                if (cost < bestCost) {
                    bestCost = cost;
                    best = [boxed[i], boxed[j]];
                }
            }
        }
        if (!best) return;
        const [keep, drop] = best;
        keep.box = union(keep.box as Aabb, drop.box as Aabb);
        keep.start = Math.max(keep.start, drop.start);
        keep.last = Math.max(keep.last, drop.last);
        // It answers to both events' sources, so they keep extending it rather than starting anew.
        for (const key of drop.keys) keep.keys.add(key);
        this.tracked = this.tracked.filter((event) => event !== drop);
    }
}
