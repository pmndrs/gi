import { describe, expect, it } from 'vitest';
import type { LightHandle } from '../../platform/contract/handles.ts';
import type { LightDescriptor, PointLightDescriptor } from '../../platform/contract/scene.ts';
import type { LightChange, SceneChanges } from '../../platform/contract/scene-copy.ts';
import { MARK_ENTRY_WORD, MARK_ENTRY_WORDS, MARK_GEOMETRY_WORD, MARK_GEOMETRY_WORDS, MARK_HEADER_WORDS, MARK_KIND, MARK_LIGHT_KIND, MARK_LIGHT_WORD, MARK_LIGHT_WORDS } from './layout.ts';
import { buildMarkTable, ENVIRONMENT_CHANGE, FAR_SET_SPACINGS, FOLDED_NEAR_SPACINGS, foldedFarChange, lightRecord, MAX_GLOBAL_CHANGE, MAX_UNRANGED_LIGHT_ENTRIES } from './mark-table.ts';
import { probeGrid } from './settings.ts';
import { markKernelWgsl } from './wgsl/mark.ts';

// Two-room's volume: 17 x 7 x 9 probes at 0.5 m, origin (0.1, -0.1, 0).
const grid = probeGrid({ center: [4.1, 1.4, 2.0], size: [8.4, 3.0, 4.2], spacing: 0.5 });

const NONE: SceneChanges = { regions: [], global: [], receivers: [], lights: [], lightsOverflowed: false };

function lamp(position: [number, number, number], intensity = 12): LightDescriptor {
    return { type: 'point', color: [1, 0.96, 0.88], intensity, position, range: 14 };
}

function change(before: LightDescriptor | null, after: LightDescriptor | null, radius: number): LightChange {
    return { light: 1 as LightHandle, index: 0, before, after, influenceRadius: { before: before ? radius : 0, after: after ? radius : 0 } };
}

function entry(table: NonNullable<ReturnType<typeof buildMarkTable>>, index: number) {
    const base = MARK_HEADER_WORDS + index * MARK_ENTRY_WORDS;
    const words = table.words;
    const floats = table.floats;
    const light = (offset: number) => ({
        kind: words[base + offset + MARK_LIGHT_WORD.kind],
        position: [...floats.subarray(base + offset + MARK_LIGHT_WORD.position, base + offset + MARK_LIGHT_WORD.position + 3)],
        direction: [...floats.subarray(base + offset + MARK_LIGHT_WORD.direction, base + offset + MARK_LIGHT_WORD.direction + 3)],
        intensity: [...floats.subarray(base + offset + MARK_LIGHT_WORD.intensity, base + offset + MARK_LIGHT_WORD.intensity + 3)],
    });
    return {
        kind: words[base + MARK_ENTRY_WORD.kind],
        boxMin: [...words.subarray(base + MARK_ENTRY_WORD.boxMin, base + MARK_ENTRY_WORD.boxMin + 3)],
        boxSize: [...words.subarray(base + MARK_ENTRY_WORD.boxSize, base + MARK_ENTRY_WORD.boxSize + 3)],
        firstThread: words[base + MARK_ENTRY_WORD.firstThread],
        globalChange: floats[base + MARK_ENTRY_WORD.globalChange],
        before: light(MARK_ENTRY_WORD.before),
        after: light(MARK_ENTRY_WORD.after),
        geometry: {
            regionMin: [...floats.subarray(base + MARK_ENTRY_WORD.geometry + MARK_GEOMETRY_WORD.regionMin, base + MARK_ENTRY_WORD.geometry + MARK_GEOMETRY_WORD.regionMin + 3)],
            regionMax: [...floats.subarray(base + MARK_ENTRY_WORD.geometry + MARK_GEOMETRY_WORD.regionMax, base + MARK_ENTRY_WORD.geometry + MARK_GEOMETRY_WORD.regionMax + 3)],
            moved: floats[base + MARK_ENTRY_WORD.geometry + MARK_GEOMETRY_WORD.moved],
            moverBit: words[base + MARK_ENTRY_WORD.geometry + MARK_GEOMETRY_WORD.moverBit],
        },
    };
}

describe("ddgi/mark's table", () => {
    it('is empty for a still frame, so the kernel never runs', () => {
        expect(buildMarkTable(NONE, grid)).toBeNull();
        // With groups off (one layer) a group change alone marks nothing.
        const regrouped = change(lamp([2, 2.45, 2]), { ...lamp([2, 2.45, 2]), group: 1 }, 14);
        const table = buildMarkTable({ ...NONE, lights: [regrouped], global: ['lights'] }, grid);
        // The light is still listed (its numbers didn't change, so the kernel finds no change), but nothing global is added.
        expect(table?.entryCount).toBe(1);
        expect(entry(table!, 0).kind).toBe(MARK_KIND.localLight);
    });

    it('marks a light that moves to another layer as switched off in the old one and on in the new one (light groups)', () => {
        const before = lamp([2, 2.45, 2]);
        const regrouped = change(before, { ...before, group: 1 }, 1);
        // 2 groups: group 0 in layer 0, groups 1-3 in layer 1.
        const table = buildMarkTable({ ...NONE, lights: [regrouped], global: ['lights'] }, grid, undefined, [0, 1, 1, 1])!;
        expect(table.entryCount).toBe(2);
        const off = entry(table, 0);
        const on = entry(table, 1);
        // Off: the lamp before, nothing after; on: nothing before, the lamp after. The same box (it didn't move).
        expect(off.before.intensity[0]).toBeGreaterThan(0);
        expect(off.after.intensity).toEqual([0, 0, 0]);
        expect(on.before.intensity).toEqual([0, 0, 0]);
        expect(on.after.intensity[0]).toBeGreaterThan(0);
        expect(off.boxMin).toEqual(on.boxMin);
        expect(off.boxSize).toEqual(on.boxSize);
        // Between two groups that share a layer it is no change of layer: one entry, as with groups off.
        const sameLayer = change({ ...before, group: 1 }, { ...before, group: 3 }, 1);
        expect(buildMarkTable({ ...NONE, lights: [sameLayer] }, grid, undefined, [0, 1, 1, 1])!.entryCount).toBe(1);
        // Moved as well: each side over its own box.
        const movedAndRegrouped = change(before, { ...lamp([6, 2.45, 2]), group: 1 }, 1);
        const both = buildMarkTable({ ...NONE, lights: [movedAndRegrouped] }, grid, undefined, [0, 1, 1, 1])!;
        expect(entry(both, 0).boxMin[0]).toBeLessThan(entry(both, 1).boxMin[0]);
    });

    it('gives a moved lamp one entry over its influence box, clipped to the volume, with both states', () => {
        // A 1 m radius around (2, 2.45, 2) and (4, 2.45, 2).
        const moved = change(lamp([2, 2.45, 2]), lamp([4, 2.45, 2]), 1);
        const table = buildMarkTable({ ...NONE, lights: [moved] }, grid)!;
        expect(table.entryCount).toBe(1);
        const first = entry(table, 0);
        expect(first.kind).toBe(MARK_KIND.localLight);
        // x from 1 m to 5 m on a grid from 0.1 m, with half a spacing for relocated probes: probes 1.3 .. 10.3 -> 2 .. 10.
        expect(first.boxMin[0]).toBe(2);
        expect(first.boxSize[0]).toBe(9);
        // y from 1.45 m to 3.45 m on a grid from -0.1 m: probes 2.6 .. 7.6 -> 3 .. 6 (the last row).
        expect(first.boxMin[1]).toBe(3);
        expect(first.boxSize[1]).toBe(4);
        expect(table.threadCount).toBe(first.boxSize[0] * first.boxSize[1] * first.boxSize[2]);
        expect(first.before.position[0]).toBeCloseTo(2, 6);
        expect(first.after.position[0]).toBeCloseTo(4, 6);
        expect(first.after.intensity[1]).toBeCloseTo(12 * 0.96, 5);
        expect(first.after.kind).toBe(MARK_LIGHT_KIND.point);
    });

    it('gives a light switched on a dark side where it is, and skips a light whose box misses the volume', () => {
        const switchedOn = change(null, lamp([2, 2.45, 2]), 14);
        const far = change(lamp([100, 0, 0]), lamp([101, 0, 0]), 1);
        const table = buildMarkTable({ ...NONE, lights: [switchedOn, far] }, grid)!;
        expect(table.entryCount).toBe(1);
        const only = entry(table, 0);
        expect(only.before.intensity).toEqual([0, 0, 0]);
        expect(only.before.position[0]).toBeCloseTo(2, 6);
        // The whole volume is within 14 m.
        expect(only.boxSize).toEqual([...grid.counts]);
    });

    it("gives a lamp with no range every probe: its change is judged against each probe's own light (T2.4b)", () => {
        // A 1 m influence radius, as the change log sets it against a bright sun; with no range the box is the volume anyway.
        const unranged = change({ type: 'point', color: [1, 1, 1], intensity: 12, position: [2, 2.45, 2], range: 0 }, { type: 'point', color: [1, 1, 1], intensity: 6, position: [2, 2.45, 2], range: 0 }, 1);
        const table = buildMarkTable({ ...NONE, lights: [unranged] }, grid)!;
        expect(entry(table, 0).boxMin).toEqual([0, 0, 0]);
        expect(entry(table, 0).boxSize).toEqual([...grid.counts]);
        expect(table.threadCount).toBe(grid.probeCount);
    });

    describe('many lights with no range changing at once (T2.1)', () => {
        const candle = (position: [number, number, number], intensity = 1, color: [number, number, number] = [1, 0.234, 0]): PointLightDescriptor => ({ type: 'point', color, intensity, position, range: 0 });
        // Candles spread through the volume, as in sponza-candles.
        const positions = Array.from({ length: 1024 }, (_, index): [number, number, number] => [0.3 + ((index * 0.618) % 1) * 7.8, 0.2 + ((index * 0.382) % 1) * 2.6, 0.2 + ((index * 0.137) % 1) * 3.8]);
        const radius = FOLDED_NEAR_SPACINGS * 0.5;

        it(`marks up to ${MAX_UNRANGED_LIGHT_ENTRIES} of them over every probe, as T2.4b does`, () => {
            const few = positions.slice(0, MAX_UNRANGED_LIGHT_ENTRIES).map((position) => change(candle(position), candle(position, 0.5), 30));
            const table = buildMarkTable({ ...NONE, lights: few }, grid)!;
            expect(table.entryCount).toBe(MAX_UNRANGED_LIGHT_ENTRIES);
            expect(table.threadCount).toBe(MAX_UNRANGED_LIGHT_ENTRIES * grid.probeCount);
        });

        it('folds more of them into a small box each and one entry over every probe with their bound', () => {
            const flicker = positions.map((position, index) => change(candle(position), candle(position, index % 2 === 0 ? 1.1 : 0.95), 30));
            const table = buildMarkTable({ ...NONE, lights: flicker }, grid)!;
            expect(table.entryCount).toBe(positions.length + 1);
            const last = entry(table, positions.length);
            expect(last.kind).toBe(MARK_KIND.everywhere);
            expect(last.boxSize).toEqual([...grid.counts]);
            // Brighter by 10%: 0.1; dimmer by 5%, against what is left: 0.05 / 0.95.
            expect(last.globalChange).toBeCloseTo(0.1, 5);
            // Each candle's box: 1 m (2 spacings) around it, half a spacing more for relocation, so at most 6 probes a side.
            for (let index = 0; index < positions.length; index++) {
                const near = entry(table, index);
                expect(near.kind).toBe(MARK_KIND.localLight);
                for (const size of near.boxSize) expect(size).toBeLessThanOrEqual(6);
            }
            // About a global change plus 1,024 small boxes, against 1,024 times every probe.
            expect(table.threadCount).toBeLessThan(positions.length * 6 ** 3 + grid.probeCount);
            expect(table.threadCount).toBeLessThan((positions.length * grid.probeCount) / 4);
        });

        it('bounds a switch or a reshape as a whole change, and still lists ranged lights one by one', () => {
            const switchedOn = positions.map((position) => change(null, candle(position), 30));
            const on = buildMarkTable({ ...NONE, lights: switchedOn }, grid)!;
            expect(entry(on, on.entryCount - 1).globalChange).toBe(MAX_GLOBAL_CHANGE);
            const ranged = positions.map((position) => change(lamp(position), lamp(position, 6), 1));
            const listed = buildMarkTable({ ...NONE, lights: ranged }, grid)!;
            expect(listed.entryCount).toBe(positions.length);
            expect([...Array(listed.entryCount).keys()].every((index) => entry(listed, index).kind === MARK_KIND.localLight)).toBe(true);
            // A range window set on a lamp that had none changes the far light from all to nothing.
            const windowed = foldedFarChange(lightRecord(candle([1, 1, 1])), lightRecord({ ...candle([1, 1, 1]), range: 3 }), radius);
            expect(windowed).toBe(MAX_GLOBAL_CHANGE);
            // A spot turned, or moved.
            const spot: LightDescriptor = { type: 'spot', color: [1, 1, 1], intensity: 1, position: [1, 1, 1], direction: [0, -1, 0], range: 0, innerConeAngle: 0.2, outerConeAngle: 0.4 };
            expect(foldedFarChange(lightRecord(spot), lightRecord({ ...spot, direction: [0, -0.9, Math.sqrt(0.19)] }), radius)).toBe(MAX_GLOBAL_CHANGE);
            expect(foldedFarChange(lightRecord(spot), lightRecord({ ...spot, position: [1, 1, 1.01] }), radius)).toBe(MAX_GLOBAL_CHANGE);
            // A spot dimmed alone is bounded like a point light: 0.2 / 0.8.
            expect(foldedFarChange(lightRecord(spot), lightRecord({ ...spot, intensity: 0.8 }), radius)).toBeCloseTo(0.25, 6);
        });

        it("bounds a moved candle's change at every point beyond the radius (checked at random points)", () => {
            let seed = 7;
            const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
            for (const [moved, scale] of [
                [0.01, 1],
                [0.1, 1.2],
                [0.3, 0.7],
                [0.9, 1],
            ] as const) {
                const from: [number, number, number] = [2, 1.5, 2];
                const direction = [random() - 0.5, random() - 0.5, random() - 0.5];
                const length = Math.hypot(...direction);
                const to: [number, number, number] = [from[0] + (moved * direction[0]) / length, from[1] + (moved * direction[1]) / length, from[2] + (moved * direction[2]) / length];
                const before = candle(from, 1, [1, 0.5, 0.25]);
                const after = candle(to, scale, [1, 0.5, 0.25]);
                const bound = foldedFarChange(lightRecord(before), lightRecord(after), radius);
                expect(bound).toBeLessThan(MAX_GLOBAL_CHANGE);
                // The relative change of the unshadowed irradiance vector at a point, largest channel, a dimming against what is left.
                const vectors = (position: readonly number[], intensity: number, point: number[]) => {
                    const offset = position.map((value, axis) => value - point[axis]);
                    const d = Math.hypot(...offset);
                    return [1, 0.5, 0.25].map((channel) => offset.map((value) => (channel * intensity * value) / (d * d * d)));
                };
                let largest = 0;
                for (let sample = 0; sample < 4000; sample++) {
                    // A point at least `radius` from both positions, out to 10 m.
                    const angle = [random() - 0.5, random() - 0.5, random() - 0.5];
                    const norm = Math.hypot(...angle);
                    const distance = radius + moved + random() * 10;
                    const point = angle.map((value, axis) => from[axis] + (distance * value) / norm);
                    if (Math.hypot(point[0] - to[0], point[1] - to[1], point[2] - to[2]) < radius) continue;
                    const old = vectors(from, 1, point);
                    const next = vectors(to, scale, point);
                    for (let channel = 0; channel < 3; channel++) {
                        const difference = Math.hypot(...next[channel].map((value, axis) => value - old[channel][axis]));
                        const against = Math.min(Math.hypot(...old[channel]), Math.hypot(...next[channel]));
                        largest = Math.max(largest, difference / against);
                    }
                }
                expect(largest).toBeLessThanOrEqual(bound);
                // And not wildly above it: the bound is the worst case, met near the radius.
                expect(bound).toBeLessThan(largest * 2.5 + 1e-6);
            }
        });
    });

    it('marks every probe for a sun change, by the change of its irradiance vector', () => {
        const sun = (intensity: number, direction: [number, number, number]): LightDescriptor => ({ type: 'directional', color: [1, 1, 1], intensity, direction });
        const turned = change(sun(3, [0, -1, 0]), sun(3, [Math.sin(Math.PI / 6), -Math.cos(Math.PI / 6), 0]), Infinity);
        const halved = change(sun(3, [0, -1, 0]), sun(1.5, [0, -1, 0]), Infinity);
        const switchedOn = change(null, sun(3, [0, -1, 0]), Infinity);
        const table = buildMarkTable({ ...NONE, lights: [turned, halved, switchedOn], global: ['sun'] }, grid)!;
        expect(table.entryCount).toBe(3);
        const [turn, half, on] = [entry(table, 0), entry(table, 1), entry(table, 2)];
        expect(turn.kind).toBe(MARK_KIND.directionalLight);
        expect(turn.boxSize).toEqual([...grid.counts]);
        // 30 degrees: the chord 2 sin(15 degrees).
        expect(turn.globalChange).toBeCloseTo(2 * Math.sin(Math.PI / 12), 5);
        // Half the sun, against the light that is left: old / new - 1 = 1 (T2.4b). (Without a sun layer; with one, below.)
        expect(half.globalChange).toBeCloseTo(1, 6);
        expect(on.globalChange).toBe(MAX_GLOBAL_CHANGE);
        // The missing side faces the same way, for the openness test.
        expect(on.before.direction).toEqual(on.after.direction);
        expect(table.threadCount).toBe(3 * grid.probeCount);
        expect(half.firstThread).toBe(grid.probeCount);
    });

    describe('the sun in a layer of its own (T2.6b)', () => {
        const sunHandle = 7 as LightHandle;
        const sun = (intensity: number, direction: [number, number, number], color: [number, number, number] = [1, 0.95, 0.85], group = 0): LightDescriptor => ({
            type: 'directional',
            color,
            intensity,
            direction,
            group,
        });
        const sunChange = (before: LightDescriptor | null, after: LightDescriptor | null): LightChange => ({ ...change(before, after, Infinity), light: sunHandle });
        const steady = { sun: sunHandle, previousSun: sunHandle };
        const down: [number, number, number] = [0, -1, 0];
        const turned: [number, number, number] = [Math.sin(Math.PI / 6), -Math.cos(Math.PI / 6), 0];

        it('marks nothing for a change of its colour, intensity or group alone: that is its weight', () => {
            const recoloured = sunChange(sun(3, down), sun(0.5, down, [1, 0.5, 0.2]));
            const off = sunChange(sun(3, down), sun(0, down));
            const backOn = sunChange(sun(0, down), sun(3, down));
            const regrouped = sunChange(sun(3, down), sun(3, down, undefined, 1));
            for (const only of [recoloured, off, backOn, regrouped]) {
                expect(buildMarkTable({ ...NONE, lights: [only], global: ['sun'] }, grid, undefined, [0, 1, 1, 1], steady)).toBeNull();
            }
            // Without a sun layer the same colour change marks every probe, as before.
            expect(buildMarkTable({ ...NONE, lights: [recoloured], global: ['sun'] }, grid)?.entryCount).toBe(1);
        });

        it('marks a turn by its angle alone, whatever its brightness did, and a turn of a sun at intensity 0 too', () => {
            for (const [before, after] of [
                [sun(3, down), sun(3, turned)],
                [sun(3, down), sun(0.3, turned, [1, 0.4, 0.1])],
                [sun(0, down), sun(0, turned)],
            ] as const) {
                const table = buildMarkTable({ ...NONE, lights: [sunChange(before, after)], global: ['sun'] }, grid, undefined, undefined, steady)!;
                expect(table.entryCount).toBe(1);
                expect(entry(table, 0).kind).toBe(MARK_KIND.directionalLight);
                expect(entry(table, 0).globalChange).toBeCloseTo(2 * Math.sin(Math.PI / 12), 5);
            }
        });

        it('treats other directional lights as before, and a sun that became another light as a whole change', () => {
            const otherLight = change(sun(3, down), sun(1.5, down), Infinity);
            const table = buildMarkTable({ ...NONE, lights: [otherLight], global: ['sun'] }, grid, undefined, undefined, steady)!;
            expect(entry(table, 0).globalChange).toBeCloseTo(1, 6);
            // The sun destroyed; another directional light (or none) is the sun now.
            for (const next of [8 as LightHandle, null]) {
                const replaced = buildMarkTable({ ...NONE, lights: [sunChange(sun(3, down), null)], global: ['sun'] }, grid, undefined, undefined, { sun: next, previousSun: sunHandle })!;
                const kinds = Array.from({ length: replaced.entryCount }, (_, index) => entry(replaced, index));
                expect(kinds.some((row) => row.kind === MARK_KIND.everywhere && row.globalChange === MAX_GLOBAL_CHANGE)).toBe(true);
            }
            // A first directional light created where there was none.
            const created = buildMarkTable({ ...NONE, lights: [sunChange(null, sun(3, down))], global: ['sun'] }, grid, undefined, undefined, { sun: sunHandle, previousSun: null })!;
            expect(Array.from({ length: created.entryCount }, (_, index) => entry(created, index).globalChange)).toContain(MAX_GLOBAL_CHANGE);
        });
    });

    it('marks every probe for an environment change and for more light changes than are listed', () => {
        const table = buildMarkTable({ ...NONE, global: ['environment', 'lights'], lightsOverflowed: true }, grid)!;
        expect([entry(table, 0).kind, entry(table, 1).kind]).toEqual([MARK_KIND.environment, MARK_KIND.everywhere]);
        expect(entry(table, 0).globalChange).toBe(ENVIRONMENT_CHANGE);
    });

    it('gives each geometry region one entry over the probes within the far reach, with its box, move and mover bit (T2.M1)', () => {
        // A 1 m crate in room A moved 0.25 m, and one hidden (no move: a whole change); the first is a hot mover.
        const before = { min: [2.5, 0, 2.5] as [number, number, number], max: [3.5, 1, 3.5] as [number, number, number] };
        const after = { min: [2.75, 0, 2.5] as [number, number, number], max: [3.75, 1, 3.5] as [number, number, number] };
        const regions = [
            { bounds: before, reason: 'geometry' as const, moved: 0.25, instance: 7 },
            { bounds: after, reason: 'geometry' as const, moved: 0.25, instance: 7 },
            { bounds: after, reason: 'geometry' as const, instance: 9 },
            // Material changes and moves of nothing mark no geometry.
            { bounds: after, reason: 'material' as const, instance: 7 },
            { bounds: after, reason: 'geometry' as const, moved: 0, instance: 7 },
        ];
        const movers = { bitFor: (slot: number) => (slot === 7 ? 1 << 3 : 0) };
        const table = buildMarkTable({ ...NONE, regions }, grid, movers)!;
        expect(table.entryCount).toBe(3);
        const [old, now, hidden] = [entry(table, 0), entry(table, 1), entry(table, 2)];
        expect(old.kind).toBe(MARK_KIND.geometry);
        expect(old.geometry.regionMin).toEqual(before.min);
        expect(old.geometry.regionMax).toEqual(before.max);
        expect(old.geometry.moved).toBeCloseTo(0.25, 6);
        expect([old.geometry.moverBit, now.geometry.moverBit, hidden.geometry.moverBit]).toEqual([8, 8, 0]);
        expect(hidden.geometry.moved).toBe(-1);
        // The probe box: the region grown by 8 spacings (4 m) and the relocation margin, clipped to the 17 x 7 x 9 grid.
        // x: (2.5 - 4 - 0.1) / 0.5 - 0.5 = -3.7 -> 0; (3.5 + 4 - 0.1) / 0.5 + 0.5 = 15.3 -> 15.
        expect(FAR_SET_SPACINGS).toBe(8);
        expect(old.boxMin).toEqual([0, 0, 0]);
        expect(old.boxSize).toEqual([16, 7, 9]);
        expect(table.threadCount).toBe(old.boxSize[0] * 7 * 9 + now.boxSize[0] * 7 * 9 + hidden.boxSize[0] * 7 * 9);
    });

    it("matches the kernel's structs word for word", () => {
        const wgsl = markKernelWgsl('workgroup');
        const members = (name: string) => {
            const body = new RegExp(`struct ${name} \\{([\\s\\S]*?)\\n\\}`).exec(wgsl)?.[1] ?? '';
            return [...body.matchAll(/^\s*(\w+): ([\w<>]+),/gm)].map((match) => ({ name: match[1], type: match[2] }));
        };
        const words = (type: string) => (type.startsWith('vec3') ? 3 : type === 'DdgiMarkLight' ? MARK_LIGHT_WORDS : type === 'DdgiMarkGeometry' ? MARK_GEOMETRY_WORDS : 1);
        // vec3 members are followed by a scalar here, so each vec3 + scalar pair fills one 16-byte slot.
        const offsets = (list: { name: string; type: string }[]) => {
            let offset = 0;
            const result: Record<string, number> = {};
            for (const member of list) {
                result[member.name] = offset;
                offset += words(member.type);
            }
            return { result, total: offset };
        };
        const light = offsets(members('DdgiMarkLight'));
        expect(light.total).toBe(MARK_LIGHT_WORDS);
        for (const [name, word] of Object.entries(MARK_LIGHT_WORD)) expect(light.result[name], name).toBe(word);
        const entryLayout = offsets(members('DdgiMarkEntry'));
        expect(entryLayout.total).toBe(MARK_ENTRY_WORDS);
        for (const [name, word] of Object.entries(MARK_ENTRY_WORD)) expect(entryLayout.result[name], name).toBe(word);
        const geometry = offsets(members('DdgiMarkGeometry'));
        expect(geometry.total).toBe(MARK_GEOMETRY_WORDS);
        for (const [name, word] of Object.entries(MARK_GEOMETRY_WORD)) expect(geometry.result[name], name).toBe(word);
        expect(offsets(members('DdgiMarkTable').filter((member) => member.name !== 'entries')).total).toBe(MARK_HEADER_WORDS);
    });
});
