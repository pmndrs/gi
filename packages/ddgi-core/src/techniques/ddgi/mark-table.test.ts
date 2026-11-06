import { describe, expect, it } from 'vitest';
import type { LightHandle } from '../../platform/contract/handles.ts';
import type { LightDescriptor } from '../../platform/contract/scene.ts';
import type { LightChange, SceneChanges } from '../../platform/contract/scene-copy.ts';
import { MARK_ENTRY_WORD, MARK_ENTRY_WORDS, MARK_HEADER_WORDS, MARK_KIND, MARK_LIGHT_KIND, MARK_LIGHT_WORD, MARK_LIGHT_WORDS } from './layout.ts';
import { buildMarkTable, ENVIRONMENT_CHANGE, MAX_GLOBAL_CHANGE } from './mark-table.ts';
import { probeGrid } from './settings.ts';
import { markKernelWgsl } from './wgsl/mark.ts';

// Two-room's volume: 17 x 7 x 9 probes at 0.5 m, origin (0.1, -0.1, 0).
const grid = probeGrid({ center: [4.1, 1.4, 2.0], size: [8.4, 3.0, 4.2], spacing: 0.5 });

const NONE: SceneChanges = { regions: [], global: [], movedReceivers: 0, lights: [], lightsOverflowed: false };

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
    };
}

describe("ddgi/mark's table", () => {
    it('is empty for a still frame, so the kernel never runs', () => {
        expect(buildMarkTable(NONE, grid)).toBeNull();
        // A group change alone marks nothing until light groups (T2.5).
        const regrouped = change(lamp([2, 2.45, 2]), { ...lamp([2, 2.45, 2]), group: 1 }, 14);
        const table = buildMarkTable({ ...NONE, lights: [regrouped], global: ['lights'] }, grid);
        // The light is still listed (its numbers didn't change, so the kernel finds no change), but nothing global is added.
        expect(table?.entryCount).toBe(1);
        expect(entry(table!, 0).kind).toBe(MARK_KIND.localLight);
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
        // Half the sun, against the light that is left: old / new - 1 = 1 (T2.4b). (Until the sun layer exists, T2.5/T2.6b.)
        expect(half.globalChange).toBeCloseTo(1, 6);
        expect(on.globalChange).toBe(MAX_GLOBAL_CHANGE);
        // The missing side faces the same way, for the openness test.
        expect(on.before.direction).toEqual(on.after.direction);
        expect(table.threadCount).toBe(3 * grid.probeCount);
        expect(half.firstThread).toBe(grid.probeCount);
    });

    it('marks every probe for an environment change and for more light changes than are listed', () => {
        const table = buildMarkTable({ ...NONE, global: ['environment', 'lights'], lightsOverflowed: true }, grid)!;
        expect([entry(table, 0).kind, entry(table, 1).kind]).toEqual([MARK_KIND.environment, MARK_KIND.everywhere]);
        expect(entry(table, 0).globalChange).toBe(ENVIRONMENT_CHANGE);
    });

    it("matches the kernel's structs word for word", () => {
        const wgsl = markKernelWgsl('workgroup');
        const members = (name: string) => {
            const body = new RegExp(`struct ${name} \\{([\\s\\S]*?)\\n\\}`).exec(wgsl)?.[1] ?? '';
            return [...body.matchAll(/^\s*(\w+): ([\w<>]+),/gm)].map((match) => ({ name: match[1], type: match[2] }));
        };
        const words = (type: string) => (type.startsWith('vec3') ? 3 : type === 'DdgiMarkLight' ? MARK_LIGHT_WORDS : 1);
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
        expect(offsets(members('DdgiMarkTable').filter((member) => member.name !== 'entries')).total).toBe(MARK_HEADER_WORDS);
    });
});
