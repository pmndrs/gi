import { describe, expect, it } from 'vitest';
import { buildCalibratedSky } from './calibrated-sky.ts';
import { liftSunAboveMinimum, proceduralSkyNeedsRebake, proceduralSkyParameters, REBAKE_SUN_DEGREES, type GpuPerez, type ProceduralSkyParameters } from './procedural-sky.ts';
import { createPreethamSky, preethamSkyXyY, xyYToLinearRgb } from './preetham.ts';
import { PEREZ_WORDS, PROJECTION_PARAMS_BYTES, PROJECTION_PARAMS_WORD } from './projection-wgsl.ts';

type Vec3 = [number, number, number];

function sunAt(elevationDeg: number, azimuthDeg: number): Vec3 {
    const e = (elevationDeg * Math.PI) / 180;
    const a = (azimuthDeg * Math.PI) / 180;
    return [Math.cos(e) * Math.cos(a), Math.sin(e), Math.cos(e) * Math.sin(a)];
}

/** The GPU's per-direction formula (projection-wgsl.ts, giSkyPreetham), written again in double precision. */
function gpuFormula(sky: ProceduralSkyParameters, direction: Vec3): Vec3 {
    if (direction[1] <= 0) return [0, 0, 0];
    const sun = sky.sunDirection;
    const cosGamma = direction[0] * sun[0] + direction[1] * sun[1] + direction[2] * sun[2];
    const cross = [direction[1] * sun[2] - direction[2] * sun[1], direction[2] * sun[0] - direction[0] * sun[2], direction[0] * sun[1] - direction[1] * sun[0]];
    const gamma = Math.atan2(Math.hypot(cross[0], cross[1], cross[2]), cosGamma);
    const distribution = (perez: GpuPerez) =>
        perez.zenithScale * (1 + perez.a * Math.exp(perez.b / Math.max(direction[1], 1e-6))) * (1 + perez.c * Math.exp(perez.d * gamma) + perez.e * cosGamma * cosGamma);
    return xyYToLinearRgb({ luminance: distribution(sky.luminance), x: distribution(sky.chromaticityX), y: distribution(sky.chromaticityY) });
}

describe('the GPU procedural sky (T2.6a)', () => {
    it("gives the kernel the numbers that make Preetham's sky, direction by direction", () => {
        for (const [elevation, turbidity] of [[5.5, 2], [20, 3], [45, 6], [80, 10]] as const) {
            const environment = { type: 'procedural' as const, sunDirection: sunAt(elevation, 30), turbidity };
            const sky = proceduralSkyParameters(environment);
            const preetham = createPreethamSky(environment.sunDirection, turbidity);
            for (const direction of [[0, 1, 0], sunAt(elevation, 30), sunAt(10, 200), sunAt(0.5, -60), sunAt(60, 95)] as Vec3[]) {
                const expected = xyYToLinearRgb(preethamSkyXyY(preetham, direction));
                const got = gpuFormula(sky, direction);
                for (let channel = 0; channel < 3; channel++) expect(got[channel]).toBeCloseTo(expected[channel], 9);
            }
        }
    });

    it("calibrates to the CPU sky's target, and clamps the sun and turbidity as the CPU sky does", () => {
        const environment = { type: 'procedural' as const, sunDirection: sunAt(35, 10), turbidity: 4, sunIlluminance: 3, intensity: 0.5 };
        const cpu = buildCalibratedSky({ sunDirection: environment.sunDirection, turbidity: 4, sunIlluminance: 3, intensity: 0.5, width: 64, height: 32 });
        expect(proceduralSkyParameters(environment).targetHorizontalLuminance).toBeCloseTo(cpu.calibration.targetSkyHorizontalIlluminance, 12);
        const low = proceduralSkyParameters({ type: 'procedural', sunDirection: sunAt(1, 50), turbidity: 40 });
        expect(low.turbidity).toBe(20);
        liftSunAboveMinimum(sunAt(1, 50)).forEach((value, axis) => expect(low.sunDirection[axis]).toBeCloseTo(value, 12));
        expect(proceduralSkyParameters({ type: 'procedural', sunDirection: [0, 2, 0] }).turbidity).toBe(2);
    });

    it('re-bakes for a turn of more than 0.05 degrees since the last bake, or any other change', () => {
        const baked = proceduralSkyParameters({ type: 'procedural', sunDirection: sunAt(30, 0), turbidity: 3 });
        const turned = (degrees: number) => proceduralSkyParameters({ type: 'procedural', sunDirection: sunAt(30 + degrees, 0), turbidity: 3 });
        expect(proceduralSkyNeedsRebake(baked, turned(0))).toBe(false);
        expect(proceduralSkyNeedsRebake(baked, turned(REBAKE_SUN_DEGREES * 0.9))).toBe(false);
        expect(proceduralSkyNeedsRebake(baked, turned(REBAKE_SUN_DEGREES * 1.1))).toBe(true);
        expect(proceduralSkyNeedsRebake(baked, proceduralSkyParameters({ type: 'procedural', sunDirection: sunAt(30, 0), turbidity: 3.001 }))).toBe(true);
        expect(proceduralSkyNeedsRebake(baked, proceduralSkyParameters({ type: 'procedural', sunDirection: sunAt(30, 0), turbidity: 3, intensity: 1.001 }))).toBe(true);
        expect(proceduralSkyNeedsRebake(baked, proceduralSkyParameters({ type: 'procedural', sunDirection: sunAt(30, 0), turbidity: 3, sunIlluminance: 0.999 }))).toBe(true);
    });

    it("lays its parameter block out like the WGSL struct (uniform rules: a struct member starts on 16 bytes)", () => {
        for (const offset of [PROJECTION_PARAMS_WORD.perezLuminance, PROJECTION_PARAMS_WORD.perezX, PROJECTION_PARAMS_WORD.perezY]) expect(offset % 4).toBe(0);
        expect(PROJECTION_PARAMS_WORD.perezX - PROJECTION_PARAMS_WORD.perezLuminance).toBe(PEREZ_WORDS);
        expect(PROJECTION_PARAMS_WORD.perezY - PROJECTION_PARAMS_WORD.perezX).toBe(PEREZ_WORDS);
        // The vec3f sun direction starts on 16 bytes, with the source kind in its fourth word.
        expect(PROJECTION_PARAMS_WORD.sunDirection % 4).toBe(0);
        expect(PROJECTION_PARAMS_WORD.sourceKind).toBe(PROJECTION_PARAMS_WORD.sunDirection + 3);
        expect(PROJECTION_PARAMS_BYTES).toBe((PROJECTION_PARAMS_WORD.perezY + PEREZ_WORDS) * 4);
    });
});
