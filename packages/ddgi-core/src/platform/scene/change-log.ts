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

export class ChangeLog {
    private regions: DirtyRegion[] = [];
    private readonly global = new Set<GlobalChange>();
    private movedReceivers = 0;
    /** Each light touched this frame, with its state before the frame's first change (null: created this frame). In first-touched order. */
    private readonly lightsBefore = new Map<LightHandle, LightDescriptor | null>();

    addRegion(box: ArrayLike<number>, reason: ChangeReason, source?: InstanceHandle | LightHandle): void {
        this.regions.push({ bounds: toAabb(box), reason, ...(source !== undefined ? { source } : {}) });
    }

    addGlobal(change: GlobalChange): void {
        this.global.add(change);
    }

    addMovedReceiver(): void {
        this.movedReceivers++;
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

        let regions = this.regions;
        if (regions.length > MAX_DIRTY_REGIONS) regions = mergeByReason(regions);
        const changes: SceneChanges = {
            regions,
            global: [...this.global],
            movedReceivers: this.movedReceivers,
            lights: lightsOverflowed ? [] : lightChanges,
            lightsOverflowed,
        };
        this.regions = [];
        this.global.clear();
        this.movedReceivers = 0;
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

function mergeByReason(regions: readonly DirtyRegion[]): DirtyRegion[] {
    const boxes = new Map<ChangeReason, number[]>();
    for (const region of regions) {
        const box = boxes.get(region.reason) ?? [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
        for (let axis = 0; axis < 3; axis++) {
            box[axis] = Math.min(box[axis], region.bounds.min[axis]);
            box[axis + 3] = Math.max(box[axis + 3], region.bounds.max[axis]);
        }
        boxes.set(region.reason, box);
    }
    return [...boxes].map(([reason, box]) => ({ bounds: toAabb(box), reason }));
}
