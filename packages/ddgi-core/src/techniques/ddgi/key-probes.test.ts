import { describe, expect, it } from 'vitest';
import { isKeyProbe, keyNeighbourOffset, keyProbeCount, KEY_NEIGHBOURS, PROPAGATE_WORKGROUPS, propagationListCapacity, PROBE_HISTORY_FLAG, scratchLayout } from './layout.ts';
import { blendSettings, withDefaults } from './settings.ts';
import { blendKernelWgsl } from './wgsl/blend.ts';
import { propagateKernelWgsl } from './wgsl/propagate.ts';
import { writeKernelWgsl, WRITE_ENTRY_POINTS } from './wgsl/write.ts';

type Vec3 = [number, number, number];

/** Every grid coordinate of a grid. */
function* coordinates(counts: Vec3): Generator<Vec3> {
    for (let y = 0; y < counts[1]; y++) for (let z = 0; z < counts[2]; z++) for (let x = 0; x < counts[0]; x++) yield [x, y, z];
}

describe('key probes (T2.4c, m2-lights.md §4.5)', () => {
    it("counts the design's keys: Sponza 2,717, two-room 180, Cornell 343", () => {
        // Even indices plus the last on even-sized axes (Cornell's 12: 0, 2, ..., 10 and 11).
        expect(keyProbeCount([37, 21, 25])).toBe(19 * 11 * 13);
        expect(keyProbeCount([17, 7, 9])).toBe(9 * 4 * 5);
        expect(keyProbeCount([12, 12, 12])).toBe(7 * 7 * 7);
        for (const counts of [[37, 21, 25], [17, 7, 9], [12, 12, 12], [1, 2, 3]] as Vec3[]) {
            expect([...coordinates(counts)].filter((probe) => isKeyProbe(probe, counts)).length).toBe(keyProbeCount(counts));
        }
    });

    it('puts keys on the far faces of even-sized grids', () => {
        expect(isKeyProbe([11, 0, 11], [12, 12, 12])).toBe(true);
        expect(isKeyProbe([9, 0, 0], [12, 12, 12])).toBe(false);
    });

    it("gives every fine probe 2, 4 or 8 key neighbours one step away on each axis where it isn't a key", () => {
        for (const counts of [[9, 7, 5], [12, 12, 12], [17, 7, 9]] as Vec3[]) {
            for (const probe of coordinates(counts)) {
                if (isKeyProbe(probe, counts)) continue;
                const offsets = Array.from({ length: KEY_NEIGHBOURS }, (_, neighbour) => keyNeighbourOffset(neighbour, probe, counts)).filter((offset) => offset !== null);
                const fineAxes = probe.filter((value, axis) => !(value % 2 === 0 || value === counts[axis] - 1)).length;
                expect(offsets.length).toBe(2 ** fineAxes);
                for (const offset of offsets) {
                    const key = probe.map((value, axis) => value + offset[axis]) as Vec3;
                    expect(key.every((value, axis) => value >= 0 && value < counts[axis])).toBe(true);
                    expect(isKeyProbe(key, counts)).toBe(true);
                    expect(offset.every((step) => Math.abs(step) <= 1)).toBe(true);
                }
                // No two neighbours are the same key.
                expect(new Set(offsets.map((offset) => offset.join(','))).size).toBe(offsets.length);
            }
        }
    });

    it('sizes the list for every fine probe, or the fine neighbours of a full update list', () => {
        expect(propagationListCapacity([37, 21, 25], 4096)).toBe(37 * 21 * 25 - 2717);
        expect(propagationListCapacity([37, 21, 25], 16)).toBe(26 * 16);
    });

    it('has key probes on by default, the hold test mode off, and the fallback behind the setting', () => {
        const settings = blendSettings(withDefaults({ volume: { center: [0, 0, 0], size: [1, 1, 1], spacing: 1 } }));
        expect(settings.keyProbes).toBe(true);
        expect(settings.holdFineProbes).toBe(false);
        expect(settings.keyProbeBoost).toBe(4);
        expect(settings.keyResidualShare).toBe(0.3);
        expect(settings.keyProbeUpdates).toBe(4);
        expect(blendSettings(withDefaults({ volume: { center: [0, 0, 0], size: [1, 1, 1], spacing: 1 }, keyProbes: false })).keyProbes).toBe(false);
    });

    it('keeps the flags apart', () => {
        const flags = Object.values(PROBE_HISTORY_FLAG);
        expect(new Set(flags).size).toBe(flags.length);
        expect(flags.every((flag) => flag < 256)).toBe(true);
    });
});

describe('the propagation kernels', () => {
    it('build subgroups first, with the declared fallback free of subgroup calls, binding only group 1', () => {
        const subgroups = propagateKernelWgsl({ irradianceTexels: 6, reduction: 'subgroups', listCapacity: 1000 });
        const fallback = propagateKernelWgsl({ irradianceTexels: 6, reduction: 'workgroup', listCapacity: 1000 });
        expect(subgroups.trimStart().startsWith('enable subgroups;')).toBe(true);
        expect(subgroups).toMatch(/subgroupExclusiveAdd\(/);
        expect(fallback).not.toMatch(/subgroup[A-Z_]|enable subgroups/);
        expect(subgroups.match(/@group\((\d)\)/g)?.every((match: string) => match === '@group(1)')).toBe(true);
    });

    it('use no indirect dispatch: their workgroups loop over the list', () => {
        const code = propagateKernelWgsl({ irradianceTexels: 6, reduction: 'workgroup', listCapacity: 1000 });
        expect(code).toMatch(/entry \+= groups\.x/);
        expect(PROPAGATE_WORKGROUPS).toBeGreaterThan(0);
    });

    it("write propagation's tiles with the write kernel's own tile code", () => {
        const code = writeKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch: scratchLayout(6, 14), distanceFormat: 'rg16float' });
        expect(code).toContain(`fn ${WRITE_ENTRY_POINTS.irradianceTiles}(`);
        // Both entry points call the one tile writer.
        expect(code.match(/ddgiStoreIrradianceTile\(probe,/g)?.length).toBe(2);
    });

    it('stay within 8 storage buffers per kernel (blend 7, propagate 6)', () => {
        const blend = blendKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch: scratchLayout(6, 14), reduction: 'workgroup' });
        expect(blend.match(/var<storage/g)?.length).toBe(7);
        const propagate = propagateKernelWgsl({ irradianceTexels: 6, reduction: 'workgroup', listCapacity: 1000 });
        // Declared for the module; ddgi/propagate binds 6 of them (not the writable state, binding 3).
        expect(propagate.match(/var<storage/g)?.length).toBe(7);
    });
});
