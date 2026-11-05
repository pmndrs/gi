import { describe, expect, it } from 'vitest';
import { GILimitError } from '../../platform/contract/errors.ts';
import { checkFieldLimits, distanceFormatFor, fieldBytes, fieldLayout, scratchLayout, UNIFORM_BYTES, UNIFORM_WORD } from './layout.ts';
import { borderSource, octDecode, octEncode, texelDirection } from './octahedral.ts';
import { frameRayRotation, quaternionToMatrix, seededRandom, uniformRandomQuaternion } from './ray-rotation.ts';
import { blendSettings, CHANGE_RULE_OFF, clockStep, FIXED_RAYS, MAX_CLOCK_STEP_SECONDS, distanceClamp, effectiveMaxRayDistance, effectiveRaysPerProbe, probeGrid, resolveBias, UNBOUNDED_RAY_DISTANCE, withDefaults } from './settings.ts';
import { scrambleStride, UpdateCursor } from './update-order.ts';
import { DDGI_SAMPLE_WGSL } from './wgsl/sample.ts';
import { blendKernelWgsl, KEY_INDEX_BITS } from './wgsl/blend.ts';
import { DDGI_CENSUS_WGSL } from './wgsl/census.ts';
import { DDGI_COMMON_WGSL } from './wgsl/common.ts';
import { DDGI_TRACE_WGSL } from './wgsl/trace.ts';
import { writeKernelWgsl } from './wgsl/write.ts';

const DEFAULT_LIMITS = { maxTextureDimension2D: 8192, maxTextureArrayLayers: 256 };

describe('the probe grid', () => {
    it('has floor(size / spacing) + 1 probes per axis, centred in the box', () => {
        // The furnace room's manifest volume.
        const grid = probeGrid({ center: [2.5, 1.4, 2.0], size: [5.2, 3.0, 4.2], spacing: 0.5 });
        expect(grid.counts).toEqual([11, 7, 9]);
        expect(grid.probeCount).toBe(693);
        expect(grid.origin[0]).toBeCloseTo(0, 12);
        expect(grid.origin[1]).toBeCloseTo(-0.1, 12);
        expect(grid.origin[2]).toBeCloseTo(0, 12);
    });

    it('does not lose a probe to rounding when the size is a whole number of spacings', () => {
        expect(probeGrid({ center: [0, 0, 0], size: [3, 0.3, 0.7], spacing: [0.1, 0.1, 0.1] }).counts).toEqual([31, 4, 8]);
    });

    it('takes a spacing per axis', () => {
        const grid = probeGrid({ center: [0, 0, 0], size: [4, 2, 1], spacing: [1, 0.5, 2] });
        expect(grid.counts).toEqual([5, 5, 1]);
        expect(grid.origin).toEqual([-2, -1, 0]);
    });

    it('refuses a spacing that is not positive', () => {
        expect(() => probeGrid({ center: [0, 0, 0], size: [1, 1, 1], spacing: 0 })).toThrow(/spacing/);
    });
});

describe('settings', () => {
    it('fills defaults from the design', () => {
        const settings = withDefaults({ volume: { center: [0, 0, 0], size: [1, 1, 1], spacing: 1 } });
        expect(settings.raysPerProbe).toBe(256);
        expect(settings.adaptiveHysteresis).toBe(true);
        expect(settings.hysteresis).toBe(0.97);
        expect(settings.hysteresisMaxSamples).toBe(200);
        expect(settings.bounceEnergy).toBe(1);
        expect(settings.maxProbesPerFrame).toBe(4096);
    });

    it('leaves the big-change rule off by default, and on when a threshold is set', () => {
        const settings = withDefaults({ volume: { center: [0, 0, 0], size: [1, 1, 1], spacing: 1 } });
        expect(blendSettings(settings).irradianceChangeThreshold).toBe(CHANGE_RULE_OFF);
        expect(blendSettings({ ...settings, irradianceChangeThreshold: 0.25 }).irradianceChangeThreshold).toBe(0.25);
        expect(blendSettings({ ...settings, irradianceChangeThreshold: Number.NaN }).irradianceChangeThreshold).toBe(CHANGE_RULE_OFF);
    });

    it('gives the kernel 1 / tau for the memory time, 0 for never forgetting, and the default for nonsense', () => {
        const settings = withDefaults({ volume: { center: [0, 0, 0], size: [1, 1, 1], spacing: 1 } });
        expect(blendSettings({ ...settings, hysteresisMemorySeconds: 4 }).inverseMemorySeconds).toBe(0.25);
        expect(blendSettings({ ...settings, hysteresisMemorySeconds: Infinity }).inverseMemorySeconds).toBe(0);
        const fallback = 1 / settings.hysteresisMemorySeconds;
        expect(blendSettings({ ...settings, hysteresisMemorySeconds: 0 }).inverseMemorySeconds).toBe(fallback);
        expect(blendSettings({ ...settings, hysteresisMemorySeconds: Number.NaN }).inverseMemorySeconds).toBe(fallback);
        expect(blendSettings({ ...settings, hysteresisMaxSamples: -3 }).maxSamples).toBe(0);
        expect(blendSettings({ ...settings, adaptiveHysteresis: false }).adaptiveHysteresis).toBe(false);
        expect(blendSettings(settings).updatesPerSample).toBe(settings.hysteresisUpdatesPerSample);
        expect(blendSettings({ ...settings, hysteresisUpdatesPerSample: 0 }).updatesPerSample).toBe(1);
        expect(blendSettings({ ...settings, hysteresisUpdatesPerSample: Number.NaN }).updatesPerSample).toBe(settings.hysteresisUpdatesPerSample);
        expect(blendSettings({ ...settings, hysteresisRampUpdates: 16 }).rampUpdates).toBe(16);
        // An endless ramp reaches the kernel as a large finite number.
        const endless = blendSettings({ ...settings, hysteresisRampUpdates: Infinity }).rampUpdates;
        expect(Number.isFinite(endless) && endless >= 1e9).toBe(true);
        expect(blendSettings({ ...settings, hysteresisRampUpdates: Number.NaN }).rampUpdates).toBe(endless);
    });

    it('advances the clock by deltaTime, at most MAX_CLOCK_STEP_SECONDS a frame, never backwards', () => {
        expect(clockStep(1 / 60)).toBeCloseTo(1 / 60, 12);
        expect(clockStep(30)).toBe(MAX_CLOCK_STEP_SECONDS);
        expect(clockStep(0)).toBe(0);
        expect(clockStep(-1)).toBe(0);
        expect(clockStep(Number.NaN)).toBe(0);
    });

    it('keeps raysPerProbe a multiple of 32 between 64 and 512', () => {
        expect(effectiveRaysPerProbe(256)).toBe(256);
        expect(effectiveRaysPerProbe(100)).toBe(96);
        expect(effectiveRaysPerProbe(10)).toBe(64);
        expect(effectiveRaysPerProbe(4000)).toBe(512);
        expect(effectiveRaysPerProbe(Number.NaN)).toBe(256);
    });

    it('resolves auto biases to 0.1 of the smallest spacing', () => {
        const grid = probeGrid({ center: [0, 0, 0], size: [4, 4, 4], spacing: [1, 0.5, 2] });
        expect(resolveBias('auto', grid, 0.1)).toBeCloseTo(0.05, 12);
        expect(resolveBias(0.3, grid, 0.1)).toBe(0.3);
        expect(resolveBias(-1, grid, 0.1)).toBe(0);
    });

    it('clamps distances at 1.5 cell diagonals and treats Infinity as unbounded rays', () => {
        const grid = probeGrid({ center: [0, 0, 0], size: [1, 1, 1], spacing: 1 });
        expect(distanceClamp(grid)).toBeCloseTo(1.5 * Math.sqrt(3), 12);
        expect(effectiveMaxRayDistance(Infinity)).toBe(UNBOUNDED_RAY_DISTANCE);
        expect(effectiveMaxRayDistance(12)).toBe(12);
        expect(effectiveMaxRayDistance(0)).toBe(UNBOUNDED_RAY_DISTANCE);
    });
});

describe('the atlases', () => {
    const grid = probeGrid({ center: [0, 0, 0], size: [5.5, 5.5, 5.5], spacing: 0.5 });

    it('put one probe row per layer and a bordered tile per probe', () => {
        const layout = fieldLayout(grid, 6, 14, 'rg16float');
        expect(layout.irradiance).toEqual({ width: 96, height: 96, layers: 12, tile: 8 });
        expect(layout.distance).toEqual({ width: 192, height: 192, layers: 12, tile: 16 });
        expect(layout.probeData).toEqual({ width: 12, height: 12, layers: 12, tile: 1 });
    });

    it('use rg16float for distances only with texture-formats-tier1', () => {
        expect(distanceFormatFor(new Set(['texture-formats-tier1']))).toBe('rg16float');
        expect(distanceFormatFor(new Set(['subgroups']))).toBe('rgba16float');
    });

    it('cost twice the distance memory in the fallback format', () => {
        const compact = fieldBytes(fieldLayout(grid, 6, 14, 'rg16float'));
        const fallback = fieldBytes(fieldLayout(grid, 6, 14, 'rgba16float'));
        expect(fallback - compact).toBe(192 * 192 * 12 * 4);
    });

    it('throw GILimitError past the default texture limits, naming the limit', () => {
        // 513 probes along x need a 513 x 16 = 8208 texel wide distance atlas.
        const wide = fieldLayout(probeGrid({ center: [0, 0, 0], size: [512, 1, 1], spacing: 1 }), 6, 14, 'rg16float');
        expect(() => checkFieldLimits(wide, DEFAULT_LIMITS)).toThrow(GILimitError);
        try {
            checkFieldLimits(wide, DEFAULT_LIMITS);
        } catch (error) {
            expect((error as GILimitError).limit).toBe('maxTextureDimension2D');
            expect((error as GILimitError).needed).toBe(8208);
        }
        const tall = fieldLayout(probeGrid({ center: [0, 0, 0], size: [1, 256, 1], spacing: 1 }), 6, 14, 'rg16float');
        expect(() => checkFieldLimits(tall, DEFAULT_LIMITS)).toThrow(/maxTextureArrayLayers = 257/);
        const fits = fieldLayout(probeGrid({ center: [0, 0, 0], size: [511, 255, 511], spacing: 1 }), 6, 14, 'rg16float');
        expect(() => checkFieldLimits(fits, DEFAULT_LIMITS)).not.toThrow();
    });

    it('size the scratch slot as the design says (36 x 8 + 196 x 4 + 8 bytes)', () => {
        const scratch = scratchLayout(6, 14);
        expect(scratch.strideWords * 4).toBe(36 * 8 + 196 * 4 + 8);
        expect(scratch.distanceOffset).toBe(72);
        expect(scratch.probeDataOffset).toBe(72 + 196);
    });

    it('keep the uniform buffer a whole number of vec4s', () => {
        expect(UNIFORM_BYTES % 16).toBe(0);
    });
});

describe('the octahedral map', () => {
    it('round-trips directions', () => {
        const random = seededRandom(7);
        for (let index = 0; index < 1000; index++) {
            const z = random() * 2 - 1;
            const angle = random() * 2 * Math.PI;
            const radius = Math.sqrt(1 - z * z);
            const direction: [number, number, number] = [radius * Math.cos(angle), radius * Math.sin(angle), z];
            const back = octDecode(octEncode(direction));
            for (let axis = 0; axis < 3; axis++) expect(back[axis]).toBeCloseTo(direction[axis], 9);
        }
    });

    /**
     * A border texel must hold the sphere's neighbour across the seam: the
     * texel it copies and the interior texel next to it must lie at the same
     * angle from the seam point between them (mirror images across the seam),
     * and close to each other. Checked on the sphere, not on the map, so it
     * tests the rule rather than restating it.
     */
    it('fills borders with the octahedral neighbour across the seam', () => {
        const angle = (a: readonly number[], b: readonly number[]): number => Math.acos(Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]));
        for (const interior of [6, 14]) {
            const square = (tile: number): number => ((tile - 1 + 0.5) / interior) * 2 - 1;
            for (let row = 0; row < interior + 2; row++) {
                for (let column = 0; column < interior + 2; column++) {
                    const isBorder = row === 0 || column === 0 || row === interior + 1 || column === interior + 1;
                    if (!isBorder) continue;
                    const u = square(column);
                    const v = square(row);
                    const seamPoint = octDecode([Math.max(-1, Math.min(1, u)), Math.max(-1, Math.min(1, v))]);
                    const adjacentColumn = Math.min(interior - 1, Math.max(0, column - 1));
                    const adjacentRow = Math.min(interior - 1, Math.max(0, row - 1));
                    const adjacent = texelDirection(adjacentColumn, adjacentRow, interior);
                    const [sourceColumn, sourceRow] = borderSource(column, row, interior);
                    const copied = texelDirection(sourceColumn, sourceRow, interior);
                    expect(angle(seamPoint, copied)).toBeCloseTo(angle(seamPoint, adjacent), 9);
                    // Not the adjacent texel itself (with an even texel count no texel sits on an edge's middle)...
                    expect(sourceColumn !== adjacentColumn || sourceRow !== adjacentRow).toBe(true);
                    // ...but its neighbour on the sphere, not some far texel.
                    expect(angle(copied, adjacent)).toBeLessThan((2.5 * Math.PI) / interior);
                }
            }
        }
    });

    it('copies interior texels unchanged', () => {
        expect(borderSource(3, 4, 6)).toEqual([2, 3]);
    });
});

describe('the update order', () => {
    it('visits every probe once per pass, for many probe counts', () => {
        for (const count of [1, 2, 3, 7, 64, 693, 1728, 1000, 4096]) {
            const cursor = new UpdateCursor(count);
            const list = new Uint32Array(count);
            expect(cursor.next(count, list)).toBe(count);
            expect(new Set(list).size).toBe(count);
        }
    });

    it('uses a coprime stride near 0.618 N', () => {
        const stride = scrambleStride(1728);
        expect(Math.abs(stride / 1728 - 0.618)).toBeLessThan(0.01);
        let a = stride;
        let b = 1728;
        while (b) [a, b] = [b, a % b];
        expect(a).toBe(1);
    });

    it('spreads one frame of updates across the volume, not one slab', () => {
        // 12 x 12 x 12 probes, 144 per frame: a slab order would give one probe row.
        const cursor = new UpdateCursor(1728);
        const list = new Uint32Array(144);
        cursor.next(144, list);
        const rows = new Set(Array.from(list, (probe) => Math.floor(probe / 144)));
        expect(rows.size).toBe(12);
    });

    it('walks on across frames and counts probes not updated yet', () => {
        const cursor = new UpdateCursor(10);
        const list = new Uint32Array(4);
        expect(cursor.notUpdatedYet).toBe(10);
        cursor.next(4, list);
        const first = Array.from(list);
        cursor.next(4, list);
        expect(cursor.notUpdatedYet).toBe(2);
        expect(Array.from(list).some((probe) => first.includes(probe))).toBe(false);
        cursor.next(4, list);
        expect(cursor.notUpdatedYet).toBe(0);
        cursor.resetHistory();
        expect(cursor.notUpdatedYet).toBe(10);
    });

    it('never hands out more than the probes or the list hold', () => {
        const cursor = new UpdateCursor(5);
        expect(cursor.next(100, new Uint32Array(100))).toBe(5);
        expect(cursor.next(100, new Uint32Array(3))).toBe(3);
        expect(cursor.next(-2, new Uint32Array(3))).toBe(0);
        expect(cursor.next(Number.NaN, new Uint32Array(3))).toBe(0);
        expect(cursor.next(Infinity, new Uint32Array(5))).toBe(5);
    });
});

describe('the ray rotation', () => {
    it('is a proper rotation (orthonormal, determinant 1)', () => {
        for (let frame = 0; frame < 50; frame++) {
            const m = frameRayRotation(frame);
            const column = (index: number): number[] => m.slice(index * 3, index * 3 + 3);
            const dot = (a: number[], b: number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            for (let a = 0; a < 3; a++) {
                for (let b = 0; b < 3; b++) expect(dot(column(a), column(b))).toBeCloseTo(a === b ? 1 : 0, 9);
            }
            const [x, y, z] = [column(0), column(1), column(2)];
            const determinant = dot(x, [y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0]]);
            expect(determinant).toBeCloseTo(1, 9);
        }
    });

    it('differs from frame to frame and repeats for the same frame', () => {
        expect(frameRayRotation(1)).not.toEqual(frameRayRotation(2));
        expect(frameRayRotation(5)).toEqual(frameRayRotation(5));
    });

    it('spreads rotated directions evenly (no preferred axis)', () => {
        // The mean of R * e_z over many uniform rotations tends to 0.
        const random = seededRandom(123);
        const mean = [0, 0, 0];
        const samples = 20000;
        for (let index = 0; index < samples; index++) {
            const m = quaternionToMatrix(uniformRandomQuaternion(random));
            for (let axis = 0; axis < 3; axis++) mean[axis] += m[6 + axis] / samples;
        }
        for (const value of mean) expect(Math.abs(value)).toBeLessThan(0.02);
    });
});

describe('the WGSL', () => {
    const scratch = scratchLayout(6, 14);

    it('keeps the sampling function free of bindings, so engines can wrap it', () => {
        expect(DDGI_SAMPLE_WGSL).not.toMatch(/@group|@binding/);
        expect(DDGI_SAMPLE_WGSL).toMatch(/fn ddgiSampleIrradiance\(/);
        // textureSample (implicit derivatives) is fragment-only; the function must also run in compute.
        expect(DDGI_SAMPLE_WGSL).not.toMatch(/textureSample\(/);
    });

    it('writes the distance format it was built for', () => {
        expect(writeKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch, distanceFormat: 'rg16float' })).toMatch(/texture_storage_2d_array<rg16float, write>/);
        expect(writeKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch, distanceFormat: 'rgba16float' })).toMatch(/texture_storage_2d_array<rgba16float, write>/);
    });

    it('uses only technique groups (1) for its own bindings', () => {
        const kernels = [
            DDGI_TRACE_WGSL,
            DDGI_CENSUS_WGSL,
            blendKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch, reduction: 'subgroups' }),
            writeKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch, distanceFormat: 'rg16float' }),
        ];
        for (const code of kernels) expect(code.match(/@group\((\d)\)/g)?.every((match: string) => match === '@group(1)')).toBe(true);
    });

    it('builds phase B subgroups first, with the declared fallback free of subgroup calls', () => {
        const subgroups = blendKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch, reduction: 'subgroups' });
        const fallback = blendKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch, reduction: 'workgroup' });
        // The enable directive must come before everything else in the module.
        expect(subgroups.trimStart().startsWith('enable subgroups;')).toBe(true);
        expect(subgroups).toMatch(/subgroupMin\(/);
        expect(fallback).not.toMatch(/subgroup[A-Z]|enable subgroups/);
    });
});

describe('the uniform block', () => {
    /** Size and alignment of the WGSL types DdgiUniforms uses (WGSL spec, "Alignment and Size"). */
    const TYPES: Record<string, { size: number; align: number }> = {
        f32: { size: 4, align: 4 },
        u32: { size: 4, align: 4 },
        vec4f: { size: 16, align: 16 },
        vec4u: { size: 16, align: 16 },
        mat3x3f: { size: 48, align: 16 },
    };

    it('has every field where UNIFORM_WORD says, under WGSL layout rules', () => {
        const body = /struct DdgiUniforms \{([\s\S]*?)\n\}/.exec(DDGI_COMMON_WGSL)?.[1];
        expect(body).toBeDefined();
        const members = [...(body ?? '').matchAll(/^\s*(\w+): (\w+),/gm)].map((match) => ({ name: match[1], type: match[2] }));
        let offset = 0;
        let structAlign = 0;
        const words: Record<string, number> = {};
        for (const member of members) {
            const type = TYPES[member.type];
            expect(type, `unknown type ${member.type}`).toBeDefined();
            offset = Math.ceil(offset / type.align) * type.align;
            words[member.name] = offset / 4;
            offset += type.size;
            structAlign = Math.max(structAlign, type.align);
        }
        expect(words).toEqual(UNIFORM_WORD);
        expect(Math.ceil(offset / structAlign) * structAlign).toBe(UNIFORM_BYTES);
    });
});

describe("phase B's reduction keys", () => {
    it('reserve enough low bits for every fixed ray index, and few enough to keep distances precise', () => {
        expect(2 ** KEY_INDEX_BITS).toBeGreaterThanOrEqual(FIXED_RAYS);
        expect(KEY_INDEX_BITS).toBeLessThanOrEqual(8);
        expect(blendKernelWgsl({ irradianceTexels: 6, distanceTexels: 14, scratch: scratchLayout(6, 14), reduction: 'workgroup' })).toContain(`const DDGI_KEY_INDEX_MASK: u32 = ${2 ** KEY_INDEX_BITS - 1}u;`);
    });
});
