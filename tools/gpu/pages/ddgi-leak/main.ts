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
 *
 * For comparing hysteresis rules (T1.12), two optional variables:
 * `VITE_DDGI_HYSTERESIS` (one rule from ddgi-shared/hysteresis-modes.ts, for
 * example `fixed:0.97`) and `VITE_DDGI_FRAMES` (instead of FRAMES).
 */

import { assert, runGpuTest } from '../../client.ts';
import { createDdgiScene, imageMean, runFrames, TestView, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { parseHysteresisMode } from '../ddgi-shared/hysteresis-modes.ts';

const env = import.meta.env as Record<string, unknown>;

/**
 * Frames every probe is updated: 50 s of DDGI's clock at 60 Hz. Why 3000:
 * the adaptive hysteresis (T1.12, the default) climbs to about 0.995 when
 * every probe is updated every frame, so bounce light settles far more
 * slowly than at the old fixed 0.97, for which 600 was enough; the two rooms
 * bounce a lot of their light. Two-room on the RTX: adaptive 600 frames leak
 * -0.22% (lit room 5.4% below its settled value), 1800 frames 0.67%, 3000
 * frames 0.81%; fixed 0.97 at 600 0.80%. The settled value is the same.
 */
const FRAMES = Number(env.VITE_DDGI_FRAMES) > 0 ? Math.round(Number(env.VITE_DDGI_FRAMES)) : 3000;
/** The hysteresis rule, when a run names one; otherwise the technique's defaults. */
const HYSTERESIS = typeof env.VITE_DDGI_HYSTERESIS === 'string' && env.VITE_DDGI_HYSTERESIS !== '' ? parseHysteresisMode(env.VITE_DDGI_HYSTERESIS) : null;
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
        releaseDeviceOnPageHide(device);
        // Also to the console, so a failing run still shows every scene's numbers (--verbose).
        const log = (line: string): void => {
            logLine(line);
            console.log(line);
        };
        const results: Record<string, unknown> = {};
        let twoRoomLeak = Number.NaN;
        for (const leakCase of CASES) {
            const scene = await createDdgiScene(device, leakCase.scene, HYSTERESIS?.settings ?? {}, { ignoreLightRange: true });
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
        return { frames: FRAMES, hysteresis: HYSTERESIS?.name ?? 'default', ...results };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
