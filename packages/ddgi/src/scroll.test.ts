import { describe, expect, it } from 'vitest';
import { probeGrid } from './settings.ts';
import {
    CAMERA_FADE_SPACINGS,
    CameraWindow,
    cameraFadeSpacings,
    coverageReach,
    enteringProbes,
    extentInset,
    MIN_FULL_REACH_SPACINGS,
    slotOf,
    VALID_EXTENT_STEP_SPACINGS,
    WINDOW_MOVE_SPACINGS,
    windowProbeOf,
} from './scroll.ts';
import { volumeEdgeCoverage } from './volumes.ts';
import { DDGI_SCROLL_WGSL, SCROLL_UNIFORM_BYTES, SCROLL_UNIFORM_WORD } from './wgsl/scroll.ts';
import { DDGI_COMMON_WGSL } from './wgsl/common.ts';
import { wgslNameProblems } from '@pmndrs/gi-core/platform/testing/wgsl-names';

// The fly-through's volume (m3-moving.md §4.2, Dennis 2026-11-05): 32 x 12 x 32 probes at 1.5 m.
const SPACING = 1.5;
const FLY_THROUGH = { center: [0, 8.25, 0], size: [46.5, 16.5, 46.5], spacing: SPACING, follow: 'camera' } as const;

function flyThroughWindow(): CameraWindow {
    return new CameraWindow(probeGrid(FLY_THROUGH));
}

/** The window's centre on each axis, in metres. */
function windowCentre(window: CameraWindow): [number, number, number] {
    const grid = window.grid;
    return [0, 1, 2].map((axis) => window.windowOrigin[axis] + ((grid.counts[axis] - 1) / 2) * grid.spacing[axis]) as [number, number, number];
}

describe('the camera-following window (m3-moving.md §4.2)', () => {
    it('starts where the settings put the box, with lattice 0 its first probe', () => {
        const window = flyThroughWindow();
        expect(window.grid.counts).toEqual([32, 12, 32]);
        expect(window.window).toEqual([0, 0, 0]);
        expect(window.windowOrigin).toEqual(window.grid.origin);
        expect(windowCentre(window)).toEqual([0, 8.25, 0]);
    });

    it('stays while the camera is within 0.75 spacings of its centre, then moves a plane and centres again', () => {
        const window = flyThroughWindow();
        const centre = windowCentre(window);
        expect(window.follow([centre[0] + 0.74 * SPACING, centre[1], centre[2]]).moved).toEqual([0, 0, 0]);
        const move = window.follow([centre[0] + 0.76 * SPACING, centre[1], centre[2]]);
        expect(move.moved).toEqual([1, 0, 0]);
        expect(move.teleport).toBe(false);
        // One plane of 12 x 32 probes entered.
        expect(move.probesEntering).toBe(12 * 32);
        expect(window.window).toEqual([1, 0, 0]);
        // The camera is now a quarter spacing short of the new centre.
        expect(windowCentre(window)[0] - (centre[0] + 0.76 * SPACING)).toBeCloseTo(0.24 * SPACING, 10);
    });

    it('clears 0 planes while the camera rests on a plane boundary for 10 s (done-when 4)', () => {
        // Two boundaries: halfway between two planes (where a 0.5 rule would flip), and right at the 0.75 rule's
        // own threshold, each with a millimetre of jitter (a hand on a mouse) for 600 frames.
        for (const offset of [0.5, WINDOW_MOVE_SPACINGS]) {
            const window = flyThroughWindow();
            const centre = windowCentre(window);
            const boundary = centre[0] + offset * SPACING;
            // Arriving there (both ends of the jitter) may move the window once: at the 0.75 threshold it does.
            window.follow([boundary + 0.001, centre[1], centre[2]]);
            window.follow([boundary - 0.001, centre[1], centre[2]]);
            const movesOnArrival = window.moves;
            expect(movesOnArrival).toBeLessThanOrEqual(1);
            const planesOnArrival = window.planesCleared;
            for (let frame = 0; frame < 600; frame++) {
                const jitter = 0.001 * Math.sin(frame * 1.7);
                window.follow([boundary + jitter, centre[1] + jitter, centre[2] - jitter]);
            }
            expect(window.moves - movesOnArrival).toBe(0);
            expect(window.planesCleared - planesOnArrival).toBe(0);
        }
    });

    it('moves on every axis, both ways, entering the planes at the far end', () => {
        const window = flyThroughWindow();
        const centre = windowCentre(window);
        const move = window.follow([centre[0] - 2.2 * SPACING, centre[1] + 1.1 * SPACING, centre[2]]);
        expect(move.moved).toEqual([-2, 1, 0]);
        expect(move.probesEntering).toBe(enteringProbes([32, 12, 32], [-2, 1, 0]));
        // Every probe less the 30 x 11 x 32 that stayed.
        expect(move.probesEntering).toBe(32 * 12 * 32 - 30 * 11 * 32);
        expect(window.planesCleared).toBe(3);
    });

    it('calls a jump of more than half the window a teleport, entering every slot', () => {
        const window = flyThroughWindow();
        const centre = windowCentre(window);
        const move = window.follow([centre[0] + 20 * SPACING, centre[1], centre[2]]);
        expect(move.teleport).toBe(true);
        expect(move.probesEntering).toBe(32 * 12 * 32);
        expect(window.teleports).toBe(1);
        expect(window.window[0]).toBe(20);
    });

    it('averages the warming updates entering planes owe into a demand a frame, which a teleport leaves alone', () => {
        const window = flyThroughWindow();
        const centre = windowCentre(window);
        // One plane of 384 probes every 18 frames: 5 m/s at 1.5 m along x. The demand settles near 2 x 384 / 18.
        for (let frame = 0; frame < 600; frame++) window.follow([centre[0] + (frame * 5) / 60, centre[1], centre[2]]);
        expect(window.warmingDemand).toBeGreaterThan(20);
        expect(window.warmingDemand).toBeLessThan(80);
        const before = window.warmingDemand;
        window.follow([centre[0] + 1000, centre[1], centre[2]]);
        expect(window.teleports).toBe(1);
        expect(window.warmingDemand).toBeLessThan(before);
    });

    it('leaves the window where it is for a camera that is not a point', () => {
        const window = flyThroughWindow();
        expect(window.follow([Number.NaN, 0, 0]).moved).toEqual([0, 0, 0]);
        expect(window.moves).toBe(0);
    });

    it('fades over 10 spacings on the long axes and less on the short one, keeping full coverage around the camera', () => {
        const window = flyThroughWindow();
        expect(window.fadeSpacings[0]).toBe(CAMERA_FADE_SPACINGS);
        expect(window.fadeSpacings[2]).toBe(CAMERA_FADE_SPACINGS);
        // 12 probes up: half-width 5.5 holds 1 of full reach, the 0.5 margin, the 0.25 the window moves by, and a 3.75 fade.
        expect(window.fadeSpacings[1]).toBeCloseTo(3.75, 10);
        expect(cameraFadeSpacings(4)).toBe(0.5);
        // About 3 s for the band to pass a point at 5 m/s and 1.5 m: a share step of about 0.0056 a frame.
        expect((CAMERA_FADE_SPACINGS * SPACING) / 5).toBeCloseTo(3, 6);
        // A fixed volume's fade (0.5) keeps T3.5's inset of 1.25.
        expect(extentInset(0.5)).toBeCloseTo(1.25, 10);
        // Centred on the camera, the camera volume answers alone within MIN_FULL_REACH (1.5 with the margin) or more of it.
        const centre = windowCentre(window);
        const targets = window.extentTargets(centre);
        const cameraLattice = window.cameraLattice(centre);
        for (let axis = 0; axis < 3; axis++) expect(cameraLattice[axis] - targets.min[axis]).toBeGreaterThanOrEqual(MIN_FULL_REACH_SPACINGS - 1e-9);
        // Full coverage at the camera, and none at the window's ends on the long axes.
        expect(volumeEdgeCoverage(window.grid, centre, targets, window.fadeSpacings)).toBe(1);
        const atEnd: [number, number, number] = [window.windowOrigin[0], centre[1], centre[2]];
        expect(volumeEdgeCoverage(window.grid, atEnd, targets, window.fadeSpacings)).toBeLessThan(0.05);
    });

    it("keeps the fade's reach inside the window, following the camera behind it without a jump when the window moves", () => {
        const window = flyThroughWindow();
        const centre = windowCentre(window);
        const reach = coverageReach(window.fadeSpacings[0]);
        // Centred: the inset inside the camera's half-window on each side.
        const centred = window.extentTargets(centre);
        expect(centred.min[0]).toBeCloseTo(15.5 - (15.5 - extentInset(window.fadeSpacings[0])), 10);
        expect(centred.max[0]).toBeCloseTo(31 - extentInset(window.fadeSpacings[0]), 10);
        // Just before and just after the move, the trailing side's target is continuous, and the fade's whole reach inside the new window.
        const before = window.extentTargets([centre[0] + 0.7499 * SPACING, centre[1], centre[2]]);
        window.follow([centre[0] + 0.7501 * SPACING, centre[1], centre[2]]);
        expect(window.window[0]).toBe(1);
        const after = window.extentTargets([centre[0] + 0.7501 * SPACING, centre[1], centre[2]]);
        expect(after.min[0] - before.min[0]).toBeCloseTo(0.0002, 6);
        expect(after.min[0] - reach).toBeGreaterThanOrEqual(window.window[0] - 1e-9);
        // The leading side's fade can't reach past the window's last plane.
        expect(after.max[0] + reach).toBeLessThanOrEqual(window.window[0] + 31 + 1e-9);
    });

    it('moves the extent out over a plane in about a quarter of a second, and keeps up with 5 m/s at 1.5 m', () => {
        expect(1 / VALID_EXTENT_STEP_SPACINGS / 60).toBeCloseTo(0.25, 6);
        expect(VALID_EXTENT_STEP_SPACINGS).toBeGreaterThan(5 / 60 / SPACING);
    });

    it('maps window probes to wrapping slots and back', () => {
        for (const window of [0, 5, 31, 32, 33, -1, -40, 1000]) {
            const slots = new Set<number>();
            for (let probe = 0; probe < 32; probe++) {
                const slot = slotOf(probe, window, 32);
                expect(windowProbeOf(slot, window, 32)).toBe(probe);
                slots.add(slot);
            }
            expect(slots.size).toBe(32);
        }
        // A lattice probe keeps its slot while the window moves over it.
        expect(slotOf(10 - 3, 3, 32)).toBe(slotOf(10 - 4, 4, 32));
    });

    it('refuses fewer than 4 probes on an axis', () => {
        expect(() => new CameraWindow(probeGrid({ center: [0, 0, 0], size: [3, 3, 3], spacing: 1.5 }))).toThrow(/at least 4/);
    });
});

describe("the camera-following volume's WGSL", () => {
    it('declares no reserved word and no name twice (target broke the first GPU run)', () => {
        expect(wgslNameProblems(DDGI_SCROLL_WGSL)).toEqual([]);
        expect(wgslNameProblems(DDGI_COMMON_WGSL)).toEqual([]);
    });
});

describe("ddgi/scroll's uniforms", () => {
    const TYPES: Record<string, { size: number; align: number }> = {
        f32: { size: 4, align: 4 },
        u32: { size: 4, align: 4 },
        i32: { size: 4, align: 4 },
        vec4f: { size: 16, align: 16 },
        vec4i: { size: 16, align: 16 },
    };

    it('has every field where SCROLL_UNIFORM_WORD says, under WGSL layout rules', () => {
        const body = /struct DdgiScroll \{([\s\S]*?)\n\}/.exec(DDGI_SCROLL_WGSL)?.[1];
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
        expect(words).toEqual(SCROLL_UNIFORM_WORD);
        expect(Math.ceil(offset / structAlign) * structAlign).toBe(SCROLL_UNIFORM_BYTES);
    });
});
