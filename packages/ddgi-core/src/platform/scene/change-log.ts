/**
 * Collects what changed between two `encode` calls, as the contract's
 * `SceneChanges`: dirty world-space boxes with a reason, plus changes that
 * have no useful box (the sun, the environment, light groups).
 */

import type { InstanceHandle, LightHandle } from '../contract/handles.ts';
import type { Aabb } from '../contract/math.ts';
import type { ChangeReason, DirtyRegion, SceneChanges } from '../contract/scene-copy.ts';

/**
 * Past this many regions in one frame, the regions of each reason are merged
 * into one box. A technique gains little from hundreds of small boxes, and
 * the list stays cheap to walk.
 */
export const MAX_DIRTY_REGIONS = 256;

type GlobalChange = SceneChanges['global'][number];

/** `[minX, minY, minZ, maxX, maxY, maxZ]` to the contract's box. */
export function toAabb(box: ArrayLike<number>): Aabb {
    return { min: [box[0], box[1], box[2]], max: [box[3], box[4], box[5]] };
}

export class ChangeLog {
    private regions: DirtyRegion[] = [];
    private readonly global = new Set<GlobalChange>();
    private movedReceivers = 0;

    addRegion(box: ArrayLike<number>, reason: ChangeReason, source?: InstanceHandle | LightHandle): void {
        this.regions.push({ bounds: toAabb(box), reason, ...(source !== undefined ? { source } : {}) });
    }

    addGlobal(change: GlobalChange): void {
        this.global.add(change);
    }

    addMovedReceiver(): void {
        this.movedReceivers++;
    }

    /** Everything since the last call, merged; clears the log. */
    take(): SceneChanges {
        let regions = this.regions;
        if (regions.length > MAX_DIRTY_REGIONS) regions = mergeByReason(regions);
        const changes: SceneChanges = {
            regions,
            global: [...this.global],
            movedReceivers: this.movedReceivers,
        };
        this.regions = [];
        this.global.clear();
        this.movedReceivers = 0;
        return changes;
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
