/**
 * T1.5b: light leaks, measured as DECISIONS 2026-11-04 defines them.
 *
 * - **Two-room (gating, ≤ 1%):** the leak is DDGI's indirect light in the dark
 *   room beyond the Cycles reference's (the doorway lets real light in), as a
 *   share of the lit room's indirect light: mean pixel values of the
 *   engine-free test view at the `dark-room-leak-view` and `lit-room`
 *   bookmarks, against the means of the Cycles indirect renders of the same
 *   views. Lights without a range window, as in Cycles.
 * - **Thin wall and closed door (reported, not gating):** the dark side gets no
 *   light at all, so the leak is DDGI's dark-view mean as a share of its own
 *   lit-view mean, with no reference.
 *
 * Each scene: DDGI on the manifest's volume with default settings, every
 * probe updated every frame for FRAMES frames, then the test views.
 */

import { assert, runGpuTest } from '../../client.ts';
import { createDdgiScene, imageMean, runFrames, TestView } from '../ddgi-shared/harness.ts';

const FRAMES = 600;
const WIDTH = 640;
const HEIGHT = 360;
/** SPEC's leak target, as a share of the lit room's indirect light. */
const LEAK_LIMIT = 0.01;

/**
 * Means of the Cycles indirect references (Cycles 4.5, 2048 samples,
 * 640 × 360, all pixels and channels): `bench/references/two-room/
 * lit-room/cycles-indirect.exr` and `.../dark-room-leak-view/cycles-indirect.exr`.
 */
const TWO_ROOM_CYCLES_LIT_MEAN = 0.676064;
const TWO_ROOM_CYCLES_DARK_MEAN = 0.060163;

interface LeakCase {
    readonly scene: string;
    readonly litCamera: string;
    readonly darkCamera: string;
    /** Cycles means of the lit and dark views, when the leak is measured against a reference. */
    readonly reference: { readonly lit: number; readonly dark: number } | null;
}

const CASES: readonly LeakCase[] = [
    { scene: 'two-room', litCamera: 'lit-room', darkCamera: 'dark-room-leak-view', reference: { lit: TWO_ROOM_CYCLES_LIT_MEAN, dark: TWO_ROOM_CYCLES_DARK_MEAN } },
    { scene: 'thin-wall', litCamera: 'lit-side', darkCamera: 'dark-side-leak-view', reference: null },
    { scene: 'closed-door', litCamera: 'lit-room', darkCamera: 'dark-room-leak-view', reference: null },
];

await runGpuTest(
    async ({ device, log: logLine }) => {
        // Also to the console, so a failing run still shows every scene's numbers (--verbose).
        const log = (line: string): void => {
            logLine(line);
            console.log(line);
        };
        const results: Record<string, unknown> = {};
        let twoRoomLeak = Number.NaN;
        for (const leakCase of CASES) {
            const scene = await createDdgiScene(device, leakCase.scene, {}, { ignoreLightRange: true });
            const field = scene.ddgi.field;
            assert(field, 'no field after prepare');
            await runFrames(device, scene.platform, FRAMES, field.grid.probeCount);
            const view = await TestView.create(device, scene.platform);
            const camera = (name: string) => {
                const found = scene.manifest.cameras.find((bookmark) => bookmark.name === name);
                assert(found, `${leakCase.scene} has no camera '${name}'`);
                return found;
            };
            const lit = imageMean((await view.render(field, camera(leakCase.litCamera), WIDTH, HEIGHT)).indirect);
            const dark = imageMean((await view.render(field, camera(leakCase.darkCamera), WIDTH, HEIGHT)).indirect);
            const counters = scene.ddgi.counters;
            scene.platform.dispose();

            if (leakCase.reference) {
                const leak = (dark - leakCase.reference.dark) / leakCase.reference.lit;
                twoRoomLeak = leak;
                results[leakCase.scene] = {
                    litMean: lit,
                    darkMean: dark,
                    cyclesLitMean: leakCase.reference.lit,
                    cyclesDarkMean: leakCase.reference.dark,
                    leak,
                    litBias: lit / leakCase.reference.lit - 1,
                    darkBias: dark / leakCase.reference.dark - 1,
                    counters,
                };
                log(`${leakCase.scene}: leak ${(leak * 100).toFixed(3)}% of the lit room (dark ${dark.toFixed(5)} against Cycles ${leakCase.reference.dark}; lit ${lit.toFixed(5)} against ${leakCase.reference.lit})`);
            } else {
                results[leakCase.scene] = { litMean: lit, darkMean: dark, leak: dark / lit, counters };
                log(`${leakCase.scene}: dark side ${((dark / lit) * 100).toFixed(3)}% of the lit side (dark ${dark.toFixed(6)}, lit ${lit.toFixed(5)})`);
            }
        }
        assert(twoRoomLeak <= LEAK_LIMIT, `the two-room leak is ${(twoRoomLeak * 100).toFixed(3)}% of the lit room, over the ${LEAK_LIMIT * 100}% limit`);
        return results;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
