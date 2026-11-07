/**
 * Motions for the native RTXGI harness (T2.7, docs/design/m2-lights.md §7):
 * writes the harness's motion file from a manifest motion or a time-of-day
 * sun path, and checks a played motion's per-frame log against the manifest.
 *
 *   node tools/native/motion.ts write --scene two-room --motion light-sweep --out <file>
 *   node tools/native/motion.ts write --scene sponza --day --out <file> [--seconds 10] [--low 5] [--high 75] [--sweep 160] [--sky-step 0.25]
 *   node tools/native/motion.ts check --scene two-room --motion light-sweep --log <frames.csv> [--tolerance 0.01]
 *
 * The harness is a comparison tool under NVIDIA's licence; this script only
 * writes its input. The motion's state at a time comes from `motionState`
 * (scenes/src/manifest.ts), the same function the viewer plays, so the file's
 * keys are the manifest's keys: the harness interpolates them the same way
 * (linear, or held for `step`; looping on the last key's time).
 *
 * The motion file (plain text, one item a line, `#` starts a comment):
 *
 *   track <target> <linear|step> <loop|once> <field> [<field> ...]
 *   key <t seconds> <values, in field order>
 *
 * Targets: `light:<name>`, a harness light by its ini name (play.ps1 names
 * each light after its manifest label), or `sky`, a sequence of sky images.
 * Fields: `position` (3 numbers, metres), `direction` (3, the way the light
 * travels; interpolated and normalised), `intensity` (1), `color` (3),
 * `sky` (a path to a half-float EXR in the shared sky file's format,
 * relative to the motion file; the harness blends neighbouring frames).
 *
 * The time-of-day sun (`--day`) is T2.4d's real day (ddgi-key-probes-cost):
 * over `seconds`, the sun's elevation rises from `low` to `high` degrees and
 * sets again (a sine), while its azimuth moves steadily through `sweep`
 * degrees, centred on the manifest's sun. The sun's direction gets a key
 * every 1/60 s (the harness's fixed step), so it is exact at every frame;
 * the sky gets one image every `sky-step` seconds, made by the shared sky
 * file's own recipe (`buildSkyImage`) for that sun, and blended between.
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { deflateSync } from 'node:zlib';
import { motionState, parseManifest, type ManifestMotion, type SceneManifest, type Vec3 } from '../../scenes/src/manifest.ts';
import { encodeHalfExr } from '../../scenes/src/sky/exr-half.ts';
import { buildSkyImage } from '../../scenes/src/sky/sky-image.ts';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');

/** The harness's fixed motion step (app.motionTimeStep): DDGI's 60 Hz clock on the web side. */
export const FRAME_SECONDS = 1 / 60;

/** T2.4d's real day (tools/gpu/pages/ddgi-key-probes-cost: DAY_FRAMES at 60 Hz, DAY_LOW/HIGH/AZIMUTH_DEGREES). */
const DAY_DEFAULTS = { seconds: 10, low: 5, high: 75, sweep: 160, skyStep: 0.25 };

export function readManifest(sceneId: string): SceneManifest {
    const path = join(repoRoot, 'scenes', 'manifests', `${sceneId}.json`);
    return parseManifest(readFileSync(path, 'utf8'), `${sceneId}.json`);
}

function findMotion(manifest: SceneManifest, motionId: string): ManifestMotion {
    const motion = (manifest.motions ?? []).find((candidate) => candidate.id === motionId);
    if (!motion) {
        const known = (manifest.motions ?? []).map((candidate) => candidate.id).join(', ') || 'none';
        throw new Error(`${manifest.id} has no motion "${motionId}" (its motions: ${known})`);
    }
    return motion;
}

/** The light a motion targets: its label, which is the harness light's name. */
export function motionTargetLabel(manifest: SceneManifest, motion: ManifestMotion): string {
    const label = /^light:(.+)$/.exec(motion.target)?.[1];
    const lights = manifest.lights.filter((light) => label !== undefined && light.label === label);
    // The manifest validator (T2.8) rejects these too; the harness must never guess which light moves.
    if (!label || lights.length !== 1) throw new Error(`${manifest.id} motion ${motion.id}: target "${motion.target}" must name exactly one light by its label (found ${lights.length})`);
    return label;
}

/** Numbers as the harness reads them: plain decimals, enough digits for 1e-6 m. */
function number(value: number): string {
    return Number(value.toPrecision(9)).toString();
}

function vector(values: readonly number[]): string {
    return values.map(number).join(' ');
}

/** The motion file for a manifest motion: its keyframes, as they are. */
export function manifestMotionFile(manifest: SceneManifest, motion: ManifestMotion): string {
    const label = motionTargetLabel(manifest, motion);
    const first = motion.keyframes[0];
    const fields = (['position', 'intensity', 'color'] as const).filter((field) => first[field] !== undefined);
    const lines = [
        `# webgpu-ddgi native motion: ${manifest.id} ${motion.id} ("${motion.label}"), written by tools/native/motion.ts from scenes/manifests/${manifest.id}.json`,
        `track light:${label} ${motion.interpolation ?? 'linear'} ${motion.loop ? 'loop' : 'once'} ${fields.join(' ')}`,
    ];
    for (const keyframe of motion.keyframes) {
        const values = fields.map((field) => {
            const value = keyframe[field];
            return typeof value === 'number' ? number(value) : vector(value as Vec3);
        });
        lines.push(`key ${number(keyframe.t)} ${values.join(' ')}`);
    }
    return `${lines.join('\n')}\n`;
}

/**
 * A still motion file holding one keyframe of a motion: its light stays where
 * that key puts it. Recorded after a long warm-up, it gives the field
 * converged in that state, the "settled" level a step's response is held
 * against (response.ts).
 */
export function settleMotionFile(manifest: SceneManifest, motion: ManifestMotion, keyIndex: number): string {
    if (!(Number.isInteger(keyIndex) && keyIndex >= 0 && keyIndex < motion.keyframes.length)) throw new Error(`${motion.id} has keys 0 to ${motion.keyframes.length - 1}`);
    const held: ManifestMotion = { ...motion, interpolation: 'step', loop: false, keyframes: [{ ...motion.keyframes[keyIndex], t: 0 }, { ...motion.keyframes[keyIndex], t: 1 }] };
    return manifestMotionFile(manifest, held).replace(/^# (.*)$/m, `# $1; held at key ${keyIndex} (a settled level)`);
}

/** The way a sun's light travels for an elevation and azimuth (degrees; azimuth from +X toward +Z), as sponza-response.ts's sunFromAngles. */
function sunTravel(elevationDeg: number, azimuthDeg: number): Vec3 {
    const e = (elevationDeg * Math.PI) / 180;
    const a = (azimuthDeg * Math.PI) / 180;
    return [-Math.cos(e) * Math.cos(a), -Math.sin(e), -Math.cos(e) * Math.sin(a)];
}

function travelAzimuth(travel: readonly number[]): number {
    return (Math.atan2(-travel[2], -travel[0]) * 180) / Math.PI;
}

export interface DayOptions {
    seconds: number;
    low: number;
    high: number;
    sweep: number;
    skyStep: number;
}

/** The sun's travel direction at `t` seconds into the day. */
export function daySun(options: DayOptions, startAzimuthDeg: number, t: number): Vec3 {
    const share = Math.min(1, Math.max(0, t / options.seconds));
    return sunTravel(options.low + (options.high - options.low) * Math.sin(Math.PI * share), startAzimuthDeg + options.sweep * (share - 0.5));
}

/**
 * The motion file for a time-of-day day on a scene with a sun (its first
 * directional light) and a procedural sky, and the sky images it names
 * (written next to it, in `<name>-sky/`).
 */
export function dayMotionFile(manifest: SceneManifest, options: DayOptions, outFile: string): { text: string; skyFrames: number } {
    const sunIndex = manifest.lights.findIndex((light) => light.type === 'directional');
    if (sunIndex < 0) throw new Error(`${manifest.id} has no directional light to move`);
    if (manifest.environment.type !== 'procedural') throw new Error(`${manifest.id} has no procedural sky (its environment is "${manifest.environment.type}")`);
    const sun = manifest.lights[sunIndex];
    if (sun.type !== 'directional') throw new Error('unreachable');
    if (!sun.label) throw new Error(`${manifest.id}: the sun (lights[${sunIndex}]) needs a label, the harness light's name`);
    const environment = manifest.environment;
    const startAzimuth = travelAzimuth(sun.direction);
    const frames = Math.round(options.seconds / FRAME_SECONDS);

    const lines = [
        `# webgpu-ddgi native motion: ${manifest.id} time of day, written by tools/native/motion.ts`,
        `# T2.4d's day: ${options.seconds} s, elevation ${options.low} -> ${options.high} -> ${options.low} degrees (a sine), azimuth through ${options.sweep} degrees around the manifest sun's ${number(startAzimuth)}`,
        `track light:${sun.label} linear once direction`,
    ];
    for (let frame = 0; frame <= frames; frame++) {
        const t = frame * FRAME_SECONDS;
        lines.push(`key ${number(t)} ${vector(daySun(options, startAzimuth, t))}`);
    }

    // The sky follows the sun: one image per sky step, by the sky file's own recipe.
    const skyDirectory = `${outFile.replace(/\.[^./\\]+$/, '')}-sky`;
    mkdirSync(skyDirectory, { recursive: true });
    lines.push('track sky linear once sky');
    const skyFrames = Math.round(options.seconds / options.skyStep);
    for (let index = 0; index <= skyFrames; index++) {
        const t = Math.min(options.seconds, index * options.skyStep);
        const travel = daySun(options, startAzimuth, t);
        const towardSun: Vec3 = [-travel[0], -travel[1], -travel[2]];
        const moved = {
            environment: { ...environment, sunDirection: towardSun },
            lights: manifest.lights.map((light, lightIndex) => (lightIndex === sunIndex ? { ...light, direction: travel } : light)),
        };
        const sky = buildSkyImage(moved);
        const file = join(skyDirectory, `sky-${String(index).padStart(4, '0')}.exr`);
        writeFileSync(file, encodeHalfExr(sky.image, (bytes) => deflateSync(bytes, { level: 6 })));
        lines.push(`key ${number(t)} ${relative(dirname(outFile), file).replace(/\\/g, '/')}`);
    }
    return { text: `${lines.join('\n')}\n`, skyFrames: skyFrames + 1 };
}

/** One frame of the harness's motion log (frames.csv). */
export interface LoggedFrame {
    frame: number;
    seconds: number;
    values: Record<string, number>;
}

export function readFramesCsv(path: string): LoggedFrame[] {
    const [header, ...rows] = readFileSync(path, 'utf8').trim().split(/\r?\n/);
    const columns = header.split(',');
    return rows.map((row) => {
        const cells = row.split(',');
        const values: Record<string, number> = {};
        columns.forEach((column, index) => (values[column] = Number(cells[index])));
        return { frame: values.frame, seconds: values.motion_time_s, values };
    });
}

/** Largest distance (metres) between the logged light position and `motionState` at the logged time. */
export function checkPositions(manifest: SceneManifest, motion: ManifestMotion, frames: readonly LoggedFrame[]): { worst: number; worstFrame: number; checked: number } {
    const label = motionTargetLabel(manifest, motion);
    let worst = 0;
    let worstFrame = -1;
    let checked = 0;
    for (const frame of frames) {
        const x = frame.values[`${label}.position.x`];
        if (x === undefined || Number.isNaN(x)) continue;
        const logged = [x, frame.values[`${label}.position.y`], frame.values[`${label}.position.z`]];
        const expected = motionState(motion, frame.seconds).position;
        if (!expected) throw new Error(`${motion.id} sets no position`);
        const distance = Math.hypot(logged[0] - expected[0], logged[1] - expected[1], logged[2] - expected[2]);
        checked++;
        if (distance > worst) {
            worst = distance;
            worstFrame = frame.frame;
        }
    }
    return { worst, worstFrame, checked };
}

/** Largest angle (degrees) between the logged sun direction and the day's path at the logged time. */
export function checkDay(manifest: SceneManifest, options: DayOptions, frames: readonly LoggedFrame[]): { worst: number; worstFrame: number; checked: number } {
    const sun = manifest.lights.find((light) => light.type === 'directional');
    if (!sun || sun.type !== 'directional' || !sun.label) throw new Error(`${manifest.id} has no labelled sun`);
    const startAzimuth = travelAzimuth(sun.direction);
    let worst = 0;
    let worstFrame = -1;
    let checked = 0;
    for (const frame of frames) {
        const x = frame.values[`${sun.label}.direction.x`];
        if (x === undefined || Number.isNaN(x)) continue;
        const logged = [x, frame.values[`${sun.label}.direction.y`], frame.values[`${sun.label}.direction.z`]];
        const expected = daySun(options, startAzimuth, frame.seconds);
        const cos = (logged[0] * expected[0] + logged[1] * expected[1] + logged[2] * expected[2]) / Math.hypot(logged[0], logged[1], logged[2]);
        const degrees = (Math.acos(Math.min(1, Math.max(-1, cos))) * 180) / Math.PI;
        checked++;
        if (degrees > worst) {
            worst = degrees;
            worstFrame = frame.frame;
        }
    }
    return { worst, worstFrame, checked };
}

async function main(): Promise<void> {
    const { values, positionals } = parseArgs({
        allowPositionals: true,
        options: {
            scene: { type: 'string' },
            motion: { type: 'string' },
            day: { type: 'boolean', default: false },
            out: { type: 'string' },
            log: { type: 'string' },
            tolerance: { type: 'string', default: '0.01' },
            seconds: { type: 'string', default: String(DAY_DEFAULTS.seconds) },
            low: { type: 'string', default: String(DAY_DEFAULTS.low) },
            high: { type: 'string', default: String(DAY_DEFAULTS.high) },
            sweep: { type: 'string', default: String(DAY_DEFAULTS.sweep) },
            'sky-step': { type: 'string', default: String(DAY_DEFAULTS.skyStep) },
            'settle-key': { type: 'string' },
        },
    });
    const command = positionals[0];
    if (!values.scene || (command !== 'write' && command !== 'check')) {
        console.log('usage: node tools/native/motion.ts write --scene <id> (--motion <id> | --day) --out <file>\n       node tools/native/motion.ts check --scene <id> --motion <id> --log <frames.csv>');
        process.exitCode = 2;
        return;
    }
    const manifest = readManifest(values.scene);

    if (command === 'write') {
        if (!values.out) throw new Error('write needs --out');
        const out = resolve(values.out);
        mkdirSync(dirname(out), { recursive: true });
        if (values.day) {
            const options: DayOptions = { seconds: Number(values.seconds), low: Number(values.low), high: Number(values.high), sweep: Number(values.sweep), skyStep: Number(values['sky-step']) };
            const { text, skyFrames } = dayMotionFile(manifest, options, out);
            writeFileSync(out, text);
            console.log(`wrote ${out}: ${manifest.id} time of day, ${options.seconds} s, ${skyFrames} sky images`);
        } else {
            if (!values.motion) throw new Error('write needs --motion <id> or --day');
            const motion = findMotion(manifest, values.motion);
            if (values['settle-key'] !== undefined) {
                const key = Number(values['settle-key']);
                writeFileSync(out, settleMotionFile(manifest, motion, key));
                console.log(`wrote ${out}: ${manifest.id} ${motion.id} held at key ${key}`);
                return;
            }
            writeFileSync(out, manifestMotionFile(manifest, motion));
            const duration = motion.keyframes[motion.keyframes.length - 1].t;
            console.log(`wrote ${out}: ${manifest.id} ${motion.id}, ${motion.keyframes.length} keys over ${duration} s${motion.loop ? ', looping' : ''}`);
        }
        return;
    }

    if (!values.log) throw new Error('check needs --log');
    if (values.day) {
        const options: DayOptions = { seconds: Number(values.seconds), low: Number(values.low), high: Number(values.high), sweep: Number(values.sweep), skyStep: Number(values['sky-step']) };
        const tolerance = 0.01;
        const result = checkDay(manifest, options, readFramesCsv(resolve(values.log)));
        const verdict = result.checked > 0 && result.worst <= tolerance;
        console.log(`${verdict ? 'ok' : 'FAIL'}: ${result.checked} frames of ${manifest.id}'s day; largest angle from the day's sun at the same time ${result.worst.toFixed(5)} degrees (frame ${result.worstFrame}; tolerance ${tolerance})`);
        if (!verdict) process.exitCode = 1;
        return;
    }
    if (!values.motion) throw new Error('check needs --motion or --day');
    const motion = findMotion(manifest, values.motion);
    const tolerance = Number(values.tolerance);
    const result = checkPositions(manifest, motion, readFramesCsv(resolve(values.log)));
    const verdict = result.checked > 0 && result.worst <= tolerance;
    console.log(
        `${verdict ? 'ok' : 'FAIL'}: ${result.checked} frames of ${manifest.id} ${motion.id}; largest distance from the manifest motion at the same time ${(result.worst * 100).toFixed(4)} cm` +
            ` (frame ${result.worstFrame}; tolerance ${(tolerance * 100).toFixed(2)} cm)`,
    );
    if (!verdict) process.exitCode = 1;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    main().catch((error: unknown) => {
        console.error(error instanceof Error ? error.message : error);
        process.exitCode = 1;
    });
}
