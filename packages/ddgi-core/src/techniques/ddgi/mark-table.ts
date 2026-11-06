/**
 * `ddgi/mark`'s table, built on the CPU from a frame's `SceneChanges`
 * (docs/design/m2-lights.md §4.2; the kernel is wgsl/mark.ts, the layout
 * MARK_* in layout.ts).
 *
 * Lights are walked one by one, never through the merged dirty boxes, which
 * lose which light changed and how. Each changed point or spot light gives
 * one entry over the probes its change can reach; each global change gives
 * one entry over the whole volume.
 *
 * Geometry (T2.M1, docs/design/m3-moving.md §5.1): each geometry region (an
 * object's old or new box after a move, or its box when it was added,
 * removed, shown or hidden) gives one entry over the probes within
 * FAR_SET_SPACINGS of it: the kernel sorts them into the near set (within
 * NEAR_SET_SPACINGS) and the far set (those that can see the box, or whose
 * mover mask has the object's bit).
 */

import type { LightDescriptor } from '../../platform/contract/scene.ts';
import type { DirtyRegion, LightChange, SceneChanges } from '../../platform/contract/scene-copy.ts';
import { MARK_ENTRY_WORD, MARK_ENTRY_WORDS, MARK_GEOMETRY_WORD, MARK_HEADER_WORDS, MARK_KIND, MARK_LIGHT_KIND, MARK_LIGHT_WORD } from './layout.ts';
import type { ProbeGrid } from './settings.ts';

/**
 * The near set of a geometry change (m3-moving.md §5.1): probes within this
 * many spacings of the region's box. Their rays may now start inside the
 * object or see a different surface right next to them, so their distances
 * change too and each is traced itself.
 */
export const NEAR_SET_SPACINGS = 1.5;
/** The far set's reach: probes within this many spacings of the box that can see it (m3-moving.md §5.1). */
export const FAR_SET_SPACINGS = 8;
/**
 * The solid angle (steradians) an object must cover in a far probe's view to
 * count as a whole change there (m3-moving.md §5.1): r = min(1, Ω / Ω_ref)
 * with Ω about π × radius² / distance². 0.5 sr is about 1/25 of the sphere.
 * The design's guess; T2.M1 records how it does (the far-set check).
 */
export const FAR_SET_REFERENCE_SOLID_ANGLE = 0.5;

/**
 * How much more urgent an off probe in a geometry change's near set is than
 * an active one with the same change (m3-moving.md §5.2: off probes "are
 * scheduled at once"). An off probe shows nothing until it is back on and
 * has had its 2 warming updates, which is worse than a stale value, and
 * until then it is cheap (32 fixed rays, no blend). 16 puts it ahead of keys
 * (`keyProbeBoost` 4) and near probes at the same camera weight. It keeps
 * the urgency through its warming updates (they are owed with it).
 */
export const FREED_PROBE_URGENCY_SHARE = 16;

/** What `buildMarkTable` needs to know about the hot movers: an instance's mask bit, 0 when it isn't one. */
export interface MoverBits {
    bitFor(slot: number): number;
}

const NO_MOVERS: MoverBits = { bitFor: () => 0 };

/**
 * The largest relative change a global entry carries. A sun switched on from
 * nothing is an infinite change relative to the old light; any r this large
 * cuts a probe's history to almost nothing (n = εK / (r − ε) ≈ 0.004), which
 * is what "replace it" means, and keeps the sums finite.
 */
export const MAX_GLOBAL_CHANGE = 100;

/**
 * The environment entry's factor on the sky's own change. Since T2.6a the
 * kernel works out the sky's change itself, on the GPU, from its radiance SH
 * now and when the field last took a sky change (m2-lights.md §4.2:
 * bands 0 and 1 relative to the sky's mean; wgsl/mark.ts,
 * ddgiMarkSkyChange), and multiplies it by this. 1: the sky's change as it
 * is (until T2.6a every sky change counted as r = 1).
 */
export const ENVIRONMENT_CHANGE = 1;

/** More light changes than are listed: every probe, as a full change (the lights' numbers are gone). */
export const OVERFLOW_CHANGE = 1;

/**
 * How far past a light's influence box a probe's grid point may lie and
 * still be marked, in spacings: relocation moves probes up to 0.45 spacing
 * off their grid points (DDGI_MAX_OFFSET in wgsl/blend.ts), so half a
 * spacing covers every relocated probe whose real position is inside.
 */
const RELOCATION_MARGIN_SPACINGS = 0.5;

export interface MarkTable {
    /** Header and entries, as the kernel reads them; `Float32Array` view of the same bytes in `floats`. */
    readonly words: Uint32Array<ArrayBuffer>;
    readonly floats: Float32Array<ArrayBuffer>;
    readonly entryCount: number;
    /** One thread per (entry, probe in its box). */
    readonly threadCount: number;
}

type Vec3 = readonly [number, number, number];

interface Entry {
    readonly kind: number;
    readonly boxMin: Vec3;
    readonly boxSize: Vec3;
    readonly globalChange: number;
    readonly before: LightRecord;
    readonly after: LightRecord;
    readonly geometry: GeometryRecord;
}

/** The geometry block of an entry (MARK_GEOMETRY_WORD). */
interface GeometryRecord {
    readonly regionMin: Vec3;
    readonly regionMax: Vec3;
    /** Metres; negative for a whole change (added, removed, shown, hidden). */
    readonly moved: number;
    readonly moverBit: number;
}

const NO_GEOMETRY: GeometryRecord = { regionMin: [0, 0, 0], regionMax: [0, 0, 0], moved: 0, moverBit: 0 };

interface LightRecord {
    readonly kind: number;
    readonly position: Vec3;
    readonly direction: Vec3;
    readonly range: number;
    readonly intensity: Vec3;
    readonly cosInnerCone: number;
    readonly cosOuterCone: number;
}

const DARK: Vec3 = [0, 0, 0];
const DOWN: Vec3 = [0, -1, 0];
const NO_LIGHT: LightRecord = { kind: MARK_LIGHT_KIND.point, position: DARK, direction: DOWN, range: 0, intensity: DARK, cosInnerCone: -1, cosOuterCone: -1 };

function normalized(vector: Vec3): Vec3 {
    const length = Math.hypot(vector[0], vector[1], vector[2]);
    return length > 0 ? [vector[0] / length, vector[1] / length, vector[2] / length] : DOWN;
}

function colourTimesIntensity(light: LightDescriptor): Vec3 {
    return [light.color[0] * light.intensity, light.color[1] * light.intensity, light.color[2] * light.intensity];
}

/** A light as the kernel reads it; cones as the platform's light table writes them (point and directional: every angle inside). */
function lightRecord(light: LightDescriptor): LightRecord {
    const intensity = colourTimesIntensity(light);
    if (light.type === 'directional') {
        return { ...NO_LIGHT, kind: MARK_LIGHT_KIND.directional, direction: normalized(light.direction), intensity };
    }
    if (light.type === 'point') {
        return { ...NO_LIGHT, kind: MARK_LIGHT_KIND.point, position: light.position, range: light.range, intensity };
    }
    return {
        kind: MARK_LIGHT_KIND.spot,
        position: light.position,
        direction: normalized(light.direction),
        range: light.range,
        intensity,
        cosInnerCone: Math.cos(light.innerConeAngle),
        cosOuterCone: Math.cos(light.outerConeAngle),
    };
}

/** A side that doesn't exist (created or destroyed this frame): dark, where the other side is, so the vector change is the other side's whole light. */
function darkLike(other: LightRecord): LightRecord {
    return { ...other, intensity: DARK };
}

function bothSides(change: LightChange): [LightRecord, LightRecord] {
    const before = change.before ? lightRecord(change.before) : null;
    const after = change.after ? lightRecord(change.after) : null;
    if (before && after) return [before, after];
    if (before) return [before, darkLike(before)];
    if (after) return [darkLike(after), after];
    return [NO_LIGHT, NO_LIGHT];
}

/**
 * A directional light's r (m2-lights.md §4.2): the change of its irradiance
 * vector relative to the old irradiance, per channel, the largest; where the
 * channel got weaker, relative to the new one (T2.4b). For a turn alone that
 * is the angle's chord, 2 sin(θ / 2); for a brightening alone new / old − 1,
 * for a dimming alone old / new − 1, per channel. A channel that was black
 * and isn't any more is MAX_GLOBAL_CHANGE.
 */
function directionalChange(before: LightRecord, after: LightRecord): number {
    let largest = 0;
    for (let channel = 0; channel < 3; channel++) {
        const old = before.intensity[channel];
        const next = after.intensity[channel];
        const difference = Math.hypot(
            next * after.direction[0] - old * before.direction[0],
            next * after.direction[1] - old * before.direction[1],
            next * after.direction[2] - old * before.direction[2],
        );
        // Where the channel got weaker, against the new light: the tolerance is a share of it,
        // and a sun halved leaves an error of 100% of it (T2.4b). A dimming alone gives old / new − 1.
        const against = next < old ? Math.max(next, old / MAX_GLOBAL_CHANGE) : old;
        const relative = old > 0 ? difference / against : next > 0 ? MAX_GLOBAL_CHANGE : 0;
        largest = Math.max(largest, relative);
    }
    return Math.min(largest, MAX_GLOBAL_CHANGE);
}

/** The probes (grid index box, inclusive) whose grid points lie within `margin` spacings of a world box; null when none. */
function probeBox(grid: ProbeGrid, min: Vec3, max: Vec3): { min: Vec3; size: Vec3 } | null {
    const low: number[] = [];
    const size: number[] = [];
    for (let axis = 0; axis < 3; axis++) {
        const spacing = grid.spacing[axis];
        const first = Math.max(0, Math.ceil((min[axis] - grid.origin[axis]) / spacing - RELOCATION_MARGIN_SPACINGS));
        const last = Math.min(grid.counts[axis] - 1, Math.floor((max[axis] - grid.origin[axis]) / spacing + RELOCATION_MARGIN_SPACINGS));
        if (!(last >= first)) return null;
        low.push(first);
        size.push(last - first + 1);
    }
    return { min: [low[0], low[1], low[2]], size: [size[0], size[1], size[2]] };
}

/**
 * The probe box of a local light's change: its range around where it was and
 * where it is, or every probe for a light with no range.
 *
 * *Why every probe for a light with no range (T2.4b, DECISIONS 2026-11-06):*
 * the change log's influence radius is set against the scene's reference
 * irradiance, the sun's when there is one, but a probe's change is measured
 * against its own light, and inside sunlit Sponza many probes see much less
 * than the sun. A lamp of 0.3 (a tenth of Sponza's sun) got a 10 m radius,
 * and the probes beyond it that its light reached through the colonnade
 * changed by far more than 5% (dark ones by thousands of percent): recall
 * 59%. Over the whole volume the mark measures each probe against its own
 * stored mean, as it already does. A light with a range can't reach past it,
 * so that box stays. Cost: one thread per probe for each such light that
 * changes in a frame (Sponza: 19,425 each).
 */
function localBox(change: LightChange, grid: ProbeGrid): { min: Vec3; size: Vec3 } | null {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    const sides: Array<[LightDescriptor | null, number]> = [
        [change.before, change.influenceRadius.before],
        [change.after, change.influenceRadius.after],
    ];
    for (const [light, radius] of sides) {
        if (!light || light.type === 'directional' || !(radius > 0)) continue;
        if (!(light.range > 0)) return { min: [0, 0, 0], size: [grid.counts[0], grid.counts[1], grid.counts[2]] };
        for (let axis = 0; axis < 3; axis++) {
            min[axis] = Math.min(min[axis], light.position[axis] - radius);
            max[axis] = Math.max(max[axis], light.position[axis] + radius);
        }
    }
    if (!(min[0] <= max[0])) return null;
    return probeBox(grid, [min[0], min[1], min[2]], [max[0], max[1], max[2]]);
}

/**
 * The table for one frame, or null when nothing changed that marks probes
 * (then `ddgi/mark` doesn't run at all).
 *
 * - A point or spot light: one entry, over its influence box.
 * - A directional light: one entry over every probe with its r.
 *   **The sun-layer exception goes here.** Once the sun has a layer of its
 *   own (T2.5 with T2.6b), a change of its colour or intensity alone marks
 *   nothing: it becomes that layer's weight (m2-lights.md §4.2, §6.1). Until
 *   then such a change is in the field's light, so it marks every probe by
 *   its change per channel (old / new − 1 for a dimming), like a turn.
 * - The environment: one entry over every probe; its r comes from the sky's
 *   SH on the GPU (ENVIRONMENT_CHANGE). The sky transfer's exceptions
 *   (`skyResidualShare`) are T2.6b's.
 * - More light changes than are listed: one entry over every probe
 *   (OVERFLOW_CHANGE).
 * - A light's group changing marks nothing by itself: every group is in the
 *   field's one layer until light groups (T2.5).
 */
export function buildMarkTable(changes: SceneChanges, grid: ProbeGrid, movers: MoverBits = NO_MOVERS): MarkTable | null {
    const entries: Entry[] = [];
    const everywhere = { boxMin: [0, 0, 0] as Vec3, boxSize: [grid.counts[0], grid.counts[1], grid.counts[2]] as Vec3, geometry: NO_GEOMETRY };
    for (const change of changes.lights) {
        const [before, after] = bothSides(change);
        const directional = change.before?.type === 'directional' || change.after?.type === 'directional';
        if (directional) {
            const globalChange = directionalChange(before, after);
            // A missing side faces the same way as the other, for the openness test.
            const beforeFacing = change.before ? before : { ...before, direction: after.direction };
            const afterFacing = change.after ? after : { ...after, direction: before.direction };
            if (globalChange > 0) entries.push({ kind: MARK_KIND.directionalLight, ...everywhere, globalChange, before: beforeFacing, after: afterFacing });
            continue;
        }
        const box = localBox(change, grid);
        if (!box) continue;
        entries.push({ kind: MARK_KIND.localLight, boxMin: box.min, boxSize: box.size, globalChange: 0, before, after, geometry: NO_GEOMETRY });
    }
    for (const region of changes.regions) {
        const entry = geometryEntry(region, grid, movers);
        if (entry) entries.push(entry);
    }
    if (changes.global.includes('environment')) {
        entries.push({ kind: MARK_KIND.environment, ...everywhere, globalChange: ENVIRONMENT_CHANGE, before: NO_LIGHT, after: NO_LIGHT });
    }
    if (changes.lightsOverflowed) {
        entries.push({ kind: MARK_KIND.everywhere, ...everywhere, globalChange: OVERFLOW_CHANGE, before: NO_LIGHT, after: NO_LIGHT });
    }
    if (entries.length === 0) return null;

    const buffer = new ArrayBuffer((MARK_HEADER_WORDS + entries.length * MARK_ENTRY_WORDS) * 4);
    const words = new Uint32Array(buffer);
    const floats = new Float32Array(buffer);
    let threadCount = 0;
    entries.forEach((entry, index) => {
        const base = MARK_HEADER_WORDS + index * MARK_ENTRY_WORDS;
        writeGeometry(words, floats, base + MARK_ENTRY_WORD.geometry, entry.geometry);
        words.set(entry.boxMin, base + MARK_ENTRY_WORD.boxMin);
        words[base + MARK_ENTRY_WORD.kind] = entry.kind;
        words.set(entry.boxSize, base + MARK_ENTRY_WORD.boxSize);
        words[base + MARK_ENTRY_WORD.firstThread] = threadCount;
        floats[base + MARK_ENTRY_WORD.globalChange] = entry.globalChange;
        writeLight(words, floats, base + MARK_ENTRY_WORD.before, entry.before);
        writeLight(words, floats, base + MARK_ENTRY_WORD.after, entry.after);
        threadCount += entry.boxSize[0] * entry.boxSize[1] * entry.boxSize[2];
    });
    words[0] = entries.length;
    words[1] = threadCount;
    return { words, floats, entryCount: entries.length, threadCount };
}

/**
 * A geometry region's entry, or null when it can't mark anything: not a
 * geometry change, it moved nothing, or no probe lies within its far reach.
 * Its probe box is the region grown by FAR_SET_SPACINGS (the kernel tells
 * the near set from the far set); `moved` is the region's move distance, or
 * −1 for a whole change.
 */
function geometryEntry(region: DirtyRegion, grid: ProbeGrid, movers: MoverBits): Entry | null {
    if (region.reason !== 'geometry') return null;
    if (region.moved !== undefined && !(region.moved > 0)) return null;
    const { min, max } = region.bounds;
    const reach = grid.spacing.map((spacing) => FAR_SET_SPACINGS * spacing);
    const box = probeBox(grid, [min[0] - reach[0], min[1] - reach[1], min[2] - reach[2]], [max[0] + reach[0], max[1] + reach[1], max[2] + reach[2]]);
    if (!box) return null;
    return {
        kind: MARK_KIND.geometry,
        boxMin: box.min,
        boxSize: box.size,
        globalChange: 0,
        before: NO_LIGHT,
        after: NO_LIGHT,
        geometry: {
            regionMin: [min[0], min[1], min[2]],
            regionMax: [max[0], max[1], max[2]],
            moved: region.moved ?? -1,
            moverBit: region.instance === undefined ? 0 : movers.bitFor(region.instance),
        },
    };
}

function writeGeometry(words: Uint32Array, floats: Float32Array, base: number, geometry: GeometryRecord): void {
    floats.set(geometry.regionMin, base + MARK_GEOMETRY_WORD.regionMin);
    floats[base + MARK_GEOMETRY_WORD.moved] = geometry.moved;
    floats.set(geometry.regionMax, base + MARK_GEOMETRY_WORD.regionMax);
    words[base + MARK_GEOMETRY_WORD.moverBit] = geometry.moverBit;
}

function writeLight(words: Uint32Array, floats: Float32Array, base: number, light: LightRecord): void {

    floats.set(light.position, base + MARK_LIGHT_WORD.position);
    words[base + MARK_LIGHT_WORD.kind] = light.kind;
    floats.set(light.direction, base + MARK_LIGHT_WORD.direction);
    floats[base + MARK_LIGHT_WORD.range] = light.range;
    floats.set(light.intensity, base + MARK_LIGHT_WORD.intensity);
    floats[base + MARK_LIGHT_WORD.cosInnerCone] = light.cosInnerCone;
    floats[base + MARK_LIGHT_WORD.cosOuterCone] = light.cosOuterCone;
}
