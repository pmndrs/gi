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
 * mover mask has the object's bit; every active one for an object that gives
 * off light, T3.4, flagged MARK_ENTRY_FLAG.emitter).
 *
 * Receivers (T3.4, m3-moving.md §5.3): each box in `SceneChanges.receivers`
 * gives one entry over the probes within WAKE_SPACINGS of it, which the
 * kernel wakes (so do geometry entries of a move or a pose, within the same
 * distance). A still scene has neither, so its air probes stay asleep as in
 * M2.
 */

import type { LightHandle } from '../../platform/contract/handles.ts';
import type { Aabb } from '../../platform/contract/math.ts';
import type { LightDescriptor } from '../../platform/contract/scene.ts';
import type { DirtyRegion, LightChange, SceneChanges } from '../../platform/contract/scene-copy.ts';
import { MARK_ENTRY_FLAG, MARK_ENTRY_WORD, MARK_ENTRY_WORDS, MARK_GEOMETRY_WORD, MARK_HEADER_WORDS, MARK_KIND, MARK_LIGHT_KIND, MARK_LIGHT_WORD } from './layout.ts';
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
 * The far set's change (m3-moving.md §5.1, as T3.4 sized it to the mover):
 * r = min(1, Ω / Ω_ref × contrast), with
 * - Ω the object's box as the probe sees it: the area of the box's faces
 *   facing it (each by how squarely), over the distance to its centre
 *   squared. So a thin door seen edge-on or a small box counts for its real
 *   size, where the design's π × radius² of the box's bounding sphere
 *   counted a 0.9 m cabinet as about 2.3 times the area its face has;
 * - contrast: how much brighter the light around the object is than the
 *   probe's own, from the stored means (the brightest active probe within
 *   1.5 spacings of the box's centre, over this probe's), at least 1;
 * - Ω_ref this solid angle: 2π, a hemisphere.
 *
 * *Why:* a probe's light (the mean over its texels) changes by about
 * Ω / 4π times how much the object's radiance differs from the average the
 * probe sees. An object about as bright as the room changes far probes
 * little; a doorway into a lit room seen from a dark one changes them a lot,
 * because most of their light comes through it. With the design's 0.5 sr
 * and the sphere, every far probe that saw the 0.9 m cabinet counted as a
 * whole change: its 1 m slide marked 826 far probes, 13% of them really
 * changed by more than 5% (T2.M1 item 8). Sized and weighed by contrast
 * (T3.4, measured on the same dump), the cabinet's far set is 442 with the
 * same far recall (106 of 112), and the door's swing keeps its recall
 * (about 90%), because the dark room is far darker than the doorway.
 */
export const FAR_SET_REFERENCE_SOLID_ANGLE = 2 * Math.PI;
/**
 * The far set's contrast looks for the light around the object in the
 * probes within this many spacings of its box's centre (per axis): the
 * 3 × 3 × 3 probes around it, so the centre's own probe may be inside the
 * object (switched off) and its neighbours still answer.
 */
export const FAR_SET_CONTRAST_SPACINGS = 1.5;

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

/**
 * Woken air probes (T3.4, m3-moving.md §5.3). Classification switches off
 * probes with no surface within reach ("no surface nearby": 347 of
 * Cornell's 1,728), because no surface needs them. A moving object rays
 * don't see (a receiver: a character, a ball) or a moved object in mid-air
 * does need them, or it gets no GI there. So probes within this many
 * spacings of a receiver's box, or of a moved or posed object's, are woken: kept on
 * whatever classification says. 2: a point inside the box reads the 8
 * probes of its cell, which lie within 1 spacing of it; one more covers the
 * next frames' movement, so the probes ahead are awake before the object
 * reaches them.
 */
export const WAKE_SPACINGS = 2;
/**
 * How long a probe stays awake after the last receiver or change near it, in
 * frames: 300, 5 s at 60 Hz (m3-moving.md §5.3). Then classification may
 * switch it off again at its next update, and a still scene ends up exactly
 * as it was, with its air probes asleep and costing their 32 fixed rays.
 */
export const WAKE_FRAMES = 300;
/**
 * Updates a woken probe that was off is owed, at FREED_PROBE_URGENCY_SHARE,
 * so they come at once: the blended two it needs before shading uses it
 * (m3-moving.md §5.2). Its first update, with the 32 fixed rays it traces
 * while off, switches it back on and doesn't count: only blends take an
 * owed update off. So it is in shading three frames after it is woken, at
 * any budget that has room for the woken probes.
 */
export const WAKE_OWED_UPDATES = 2;

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
 * Up to this many changes of lights with no range in one frame are each
 * marked over every probe, against each probe's own light (T2.4b's rule).
 * Past it the frame's unranged changes are folded (T2.1, DECISIONS
 * 2026-11-06): each light is marked exactly only within
 * FOLDED_NEAR_SPACINGS of where it was and where it is, and one entry over
 * every probe carries a bound on what any of them changes farther away
 * ({@link foldedFarChange}). So the cost is about one global change plus a
 * small box per light, instead of one thread per probe per light: 1,024
 * candles changing at once on Sponza would otherwise be 1,024 x 19,425 =
 * 20 M threads. 8 is `exactLightLimit`'s count (m2-lights.md §3): a few
 * lamps stay exact.
 */
export const MAX_UNRANGED_LIGHT_ENTRIES = 8;

/**
 * How far around a folded light (in probe spacings, the largest axis) its
 * change is marked probe by probe. Beyond it a move of d changes the
 * light's own irradiance at a probe by a bounded share, about 3d / R, so
 * the far entry's bound stays small for small moves (a flame's flicker, a
 * candle drifting), while the probes right next to it, where a move or a
 * switch matters most, are judged exactly.
 */
export const FOLDED_NEAR_SPACINGS = 2;

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
    /** The object gives off light (`DirtyRegion.emits`): MARK_ENTRY_FLAG.emitter. */
    readonly emitter: boolean;
}

const NO_GEOMETRY: GeometryRecord = { regionMin: [0, 0, 0], regionMax: [0, 0, 0], moved: 0, moverBit: 0, emitter: false };

/** A light as the kernel reads it (MARK_LIGHT_WORD). */
export interface LightRecord {
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
export function lightRecord(light: LightDescriptor): LightRecord {
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

/** Whether a light change is one of a light with no range (a point or spot light with range 0 on either side): T2.4b marks those over every probe. */
function isUnrangedLocal(change: LightChange): boolean {
    for (const light of [change.before, change.after]) {
        if (light && light.type !== 'directional' && !(light.range > 0)) return true;
    }
    return false;
}

function sameVector(a: Vec3, b: Vec3): boolean {
    return a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
}

/**
 * A folded light's change beyond `radius` of both its positions (T2.1): an
 * upper bound on the relative change of its own unshadowed irradiance
 * vector there, per colour channel, the largest. Where a channel got weaker
 * it is measured against what is left, as the kernel does (T2.4b). A
 * probe's light is this light's plus others', so the relative change of
 * the probe's whole light is at most the largest such bound over the
 * changed lights (a weighted average of their changes): one number covers
 * them all.
 *
 * - Colour or intensity alone: |new / old - 1| per channel (old / new - 1
 *   for a dimming), the same at every distance.
 * - A point light moved by d < radius: a probe at least `radius` from both
 *   positions sees its distance change by at most d, so the irradiance by a
 *   factor between (R / (R + d))^2 and ((R + d) / R)^2, and the direction
 *   by at most asin(d / R). The vector's change is at most the magnitudes'
 *   difference plus the smaller magnitude times that angle.
 * - Anything that can switch the light on or off at a far probe (on from
 *   black, off to black, a spot's cone or aim changed, a range window set
 *   or changed, a move of `radius` or more): MAX_GLOBAL_CHANGE.
 *
 * Only the direct light is bounded. Its bounce off surfaces near the light
 * can change more, as with the per-light rule, which also measures direct
 * light only.
 */
export function foldedFarChange(before: LightRecord, after: LightRecord, radius: number): number {
    const reshaped = before.kind !== after.kind || before.range !== after.range || before.cosInnerCone !== after.cosInnerCone || before.cosOuterCone !== after.cosOuterCone;
    const moved = Math.hypot(after.position[0] - before.position[0], after.position[1] - before.position[1], after.position[2] - before.position[2]);
    const reaimed = after.kind === MARK_LIGHT_KIND.spot && (!sameVector(before.direction, after.direction) || moved > 0);
    if (reshaped || reaimed || before.range > 0 || !(moved < radius)) return MAX_GLOBAL_CHANGE;
    // The irradiance at the new position over that at the old: its smallest and largest, and the largest turn.
    const nearer = ((radius + moved) / radius) ** 2;
    const farther = (radius / (radius + moved)) ** 2;
    const turn = Math.asin(moved / radius);
    let largest = 0;
    for (let channel = 0; channel < 3; channel++) {
        const old = before.intensity[channel];
        const next = after.intensity[channel];
        if (old <= 0 && next <= 0) continue;
        if (old <= 0 || next <= 0) return MAX_GLOBAL_CHANGE;
        for (const magnitude of [farther, nearer]) {
            const now = next * magnitude;
            const difference = Math.abs(now - old) + Math.min(now, old) * turn;
            const against = now < old ? now : old;
            largest = Math.max(largest, difference / against);
        }
    }
    return Math.min(largest, MAX_GLOBAL_CHANGE);
}

/** A folded light's own box: `radius` around where it was and where it is (T2.1). */
function foldedNearBox(change: LightChange, grid: ProbeGrid, radius: number): { min: Vec3; size: Vec3 } | null {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (const light of [change.before, change.after]) {
        if (!light || light.type === 'directional') continue;
        for (let axis = 0; axis < 3; axis++) {
            min[axis] = Math.min(min[axis], light.position[axis] - radius);
            max[axis] = Math.max(max[axis], light.position[axis] + radius);
        }
    }
    if (!(min[0] <= max[0])) return null;
    return probeBox(grid, [min[0], min[1], min[2]], [max[0], max[1], max[2]]);
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

/** Every light group in the one layer (groups off). */
const ONE_LAYER: readonly number[] = [0, 0, 0, 0];

/**
 * The sun layer, as `ddgi/mark` needs it (T2.6b; light-groups.ts): which
 * light the layer holds this frame and which it held in the frame before
 * (`SceneCopy.firstDirectionalLight`'s handle; null with no directional light).
 */
export interface SunLayerMarks {
    readonly sun: LightHandle | null;
    readonly previousSun: LightHandle | null;
}

/** The sun as its layer holds it: at the white reference (colour 1, intensity 1). */
function atWhite(light: LightRecord): LightRecord {
    return { ...light, intensity: [1, 1, 1] };
}

/**
 * The frame's light changes as `ddgi/mark` treats them: a light that moved to
 * another irradiance layer (its group changed, and the groups' layers differ)
 * becomes two changes, its removal from the old layer (before → none) and
 * its arrival in the new one (none → after). Why not one: each layer changes
 * by the light's whole irradiance, though the field's sum doesn't (with
 * every weight 1), and the history is cut for what changed in a layer.
 */
function byLayer(changes: readonly LightChange[], groupLayers: readonly number[], sunInItsLayer: LightHandle | null): LightChange[] {
    const layerOf = (light: LightDescriptor): number => groupLayers[light.group ?? 0] ?? groupLayers[groupLayers.length - 1] ?? 0;
    const result: LightChange[] = [];
    for (const change of changes) {
        // The sun in its own layer stays there whatever its group: a group change only changes its weight.
        const moves = change.light !== sunInItsLayer && change.before && change.after && layerOf(change.before) !== layerOf(change.after);
        if (moves && change.before && change.after) {
            result.push({ ...change, after: null, influenceRadius: { before: change.influenceRadius.before, after: 0 } });
            result.push({ ...change, before: null, influenceRadius: { before: 0, after: change.influenceRadius.after } });
        } else {
            result.push(change);
        }
    }
    return result;
}

/**
 * The table for one frame, or null when nothing changed that marks probes
 * (then `ddgi/mark` doesn't run at all).
 *
 * - A point or spot light: one entry, over its influence box (every probe
 *   for a light with no range).
 * - More than MAX_UNRANGED_LIGHT_ENTRIES lights with no range changing in
 *   one frame (T2.1): each gets one entry over FOLDED_NEAR_SPACINGS around
 *   it, and one entry over every probe carries the largest of their bounds
 *   beyond that ({@link foldedFarChange}), served as a global change.
 * - A directional light: one entry over every probe with its r. A change of
 *   colour or intensity is in the field's light, so it marks every probe by
 *   its change per channel (old / new − 1 for a dimming), like a turn.
 * - **The sun-layer exception** (T2.6b, `sunLayer`; m2-lights.md §4.2,
 *   §6.1): the sun in a layer of its own is traced at a white reference, so
 *   a change of its colour, intensity or group alone marks nothing: it is
 *   the layer's weight, composed in its frame. Its r is worked out with both
 *   sides at white, so a turn marks by its angle alone, whatever its
 *   brightness did in the same frame (a turn of a sun at intensity 0 too: its
 *   layer must follow for when it comes back). When the sun becomes another
 *   light (the sun destroyed, or a first directional light created), the
 *   layer holds the wrong light's response: every probe is marked as a whole
 *   change (MAX_GLOBAL_CHANGE). Rare: it takes destroying or creating
 *   directional lights.
 * - The environment: one entry over every probe; its r comes from the sky's
 *   SH on the GPU (ENVIRONMENT_CHANGE). The sky transfer's exceptions
 *   (`skyResidualShare`) are T2.6b's.
 * - More light changes than are listed: one entry over every probe
 *   (OVERFLOW_CHANGE).
 * - A receiver where it is now: one entry over the probes within
 *   WAKE_SPACINGS of its box, which wakes them (receiverEntry).
 * - A light whose group changes so that it moves to another irradiance
 *   layer (light groups, T2.5; `groupLayers`) leaves its old layer and joins
 *   the new one: it is marked as two changes, switched off where it was and
 *   switched on where it is (byLayer), so the probes of both layers settle as
 *   after a light switch. With groups off (one layer) a group change alone
 *   marks nothing: the light's numbers didn't change.
 */
export function buildMarkTable(
    changes: SceneChanges,
    grid: ProbeGrid,
    movers: MoverBits = NO_MOVERS,
    groupLayers: readonly number[] = ONE_LAYER,
    sunLayer: SunLayerMarks | null = null,
): MarkTable | null {
    const entries: Entry[] = [];
    const everywhere = { boxMin: [0, 0, 0] as Vec3, boxSize: [grid.counts[0], grid.counts[1], grid.counts[2]] as Vec3, geometry: NO_GEOMETRY };
    // The sun keeps its layer this frame: its changes are judged at the white reference.
    const steadySun = sunLayer && sunLayer.sun !== null && sunLayer.sun === sunLayer.previousSun ? sunLayer.sun : null;
    const lightChanges = byLayer(changes.lights, groupLayers, steadySun);
    let unranged = 0;
    for (const change of lightChanges) if (isUnrangedLocal(change)) unranged++;
    const fold = unranged > MAX_UNRANGED_LIGHT_ENTRIES;
    const foldedRadius = FOLDED_NEAR_SPACINGS * Math.max(grid.spacing[0], grid.spacing[1], grid.spacing[2]);
    let foldedChange = 0;
    for (const change of lightChanges) {
        const [before, after] = bothSides(change);
        const directional = change.before?.type === 'directional' || change.after?.type === 'directional';
        if (directional) {
            const isSun = change.light === steadySun;
            // The sun in its own layer: both sides at white, so only a turn (or a switch on or off) counts.
            // (A side that doesn't exist stays dark; for a sun that keeps its layer both sides exist.)
            const globalChange = isSun
                ? directionalChange(change.before ? atWhite(before) : before, change.after ? atWhite(after) : after)
                : directionalChange(before, after);
            // A missing side faces the same way as the other, for the openness test.
            const beforeFacing = change.before ? before : { ...before, direction: after.direction };
            const afterFacing = change.after ? after : { ...after, direction: before.direction };
            if (globalChange > 0) entries.push({ kind: MARK_KIND.directionalLight, ...everywhere, globalChange, before: beforeFacing, after: afterFacing });
            continue;
        }
        if (fold && isUnrangedLocal(change)) {
            foldedChange = Math.max(foldedChange, foldedFarChange(before, after, foldedRadius));
            const near = foldedNearBox(change, grid, foldedRadius);
            if (near) entries.push({ kind: MARK_KIND.localLight, boxMin: near.min, boxSize: near.size, globalChange: 0, before, after, geometry: NO_GEOMETRY });
            continue;
        }
        const box = localBox(change, grid);
        if (!box) continue;
        entries.push({ kind: MARK_KIND.localLight, boxMin: box.min, boxSize: box.size, globalChange: 0, before, after, geometry: NO_GEOMETRY });
    }
    if (foldedChange > 0) {
        entries.push({ kind: MARK_KIND.everywhere, ...everywhere, globalChange: foldedChange, before: NO_LIGHT, after: NO_LIGHT });
    }
    if (sunLayer && sunLayer.sun !== sunLayer.previousSun) {
        // The sun layer now holds another light (or none): a whole change everywhere.
        entries.push({ kind: MARK_KIND.everywhere, ...everywhere, globalChange: MAX_GLOBAL_CHANGE, before: NO_LIGHT, after: NO_LIGHT });
    }
    for (const region of changes.regions) {
        const entry = geometryEntry(region, grid, movers);
        if (entry) entries.push(entry);
    }
    for (const box of changes.receivers) {
        const entry = receiverEntry(box, grid);
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
        words[base + MARK_ENTRY_WORD.flags] = entry.geometry.emitter ? MARK_ENTRY_FLAG.emitter : 0;
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
            emitter: region.emits === true,
        },
    };
}

/**
 * A receiver's entry (T3.4, m3-moving.md §5.3): the probes within
 * WAKE_SPACINGS of its box, which the kernel wakes. Its box goes in the
 * geometry block; it marks no change (rays don't see a receiver), so it
 * carries no move and no mover bit. Null when no probe is that near.
 */
function receiverEntry(box: Aabb, grid: ProbeGrid): Entry | null {
    const { min, max } = box;
    const reach = grid.spacing.map((spacing) => WAKE_SPACINGS * spacing);
    const probes = probeBox(grid, [min[0] - reach[0], min[1] - reach[1], min[2] - reach[2]], [max[0] + reach[0], max[1] + reach[1], max[2] + reach[2]]);
    if (!probes) return null;
    return {
        kind: MARK_KIND.receiver,
        boxMin: probes.min,
        boxSize: probes.size,
        globalChange: 0,
        before: NO_LIGHT,
        after: NO_LIGHT,
        geometry: { regionMin: [min[0], min[1], min[2]], regionMax: [max[0], max[1], max[2]], moved: 0, moverBit: 0, emitter: false },
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
