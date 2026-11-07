import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { motionState } from '../../scenes/src/manifest.ts';
import { checkPositions, checkTimes, daySun, manifestMotionFile, readManifest, type LoggedFrame } from './motion.ts';

const DAY = { seconds: 10, low: 5, high: 75, sweep: 160, skyStep: 0.25 };

function elevation(travel: readonly number[]): number {
    return (Math.asin(-travel[1]) * 180) / Math.PI;
}

describe('native motion files (T2.7)', () => {
    it('writes a manifest motion as its own keys, targeting the light by its label', () => {
        const manifest = readManifest('two-room');
        const motion = manifest.motions!.find((candidate) => candidate.id === 'light-step')!;
        const lines = manifestMotionFile(manifest, motion).trim().split('\n').filter((line) => !line.startsWith('#'));
        expect(lines[0]).toBe('track light:room-a-lamp step loop position');
        expect(lines.slice(1)).toEqual(motion.keyframes.map((key) => `key ${key.t} ${key.position!.join(' ')}`));
    });

    it('finds no distance between a log that follows motionState and the motion', () => {
        const manifest = readManifest('two-room');
        const motion = manifest.motions!.find((candidate) => candidate.id === 'light-sweep')!;
        const frames: LoggedFrame[] = Array.from({ length: 700 }, (_, frame) => {
            const seconds = frame / 60;
            const [x, y, z] = motionState(motion, seconds).position!;
            return { frame, seconds, values: { 'room-a-lamp.position.x': x, 'room-a-lamp.position.y': y, 'room-a-lamp.position.z': z } };
        });
        expect(checkPositions(manifest, motion, frames)).toMatchObject({ worst: 0, checked: 700 });
        // A log 2 cm off at one frame is caught
        frames[300].values['room-a-lamp.position.x'] += 0.02;
        expect(checkPositions(manifest, motion, frames).worst).toBeCloseTo(0.02, 6);
    });

    it('checks a fixed-clock log at frame / 60, not at the time the harness logged', () => {
        const manifest = readManifest('two-room');
        const motion = manifest.motions!.find((candidate) => candidate.id === 'light-sweep')!;
        // A harness clock that runs one frame late: the light is where the motion is at the logged time, so a check
        // at the logged time would pass; checked at frame / 60 it is 1/60 m (1.67 cm at 1 m/s) off
        const rows = Array.from({ length: 300 }, (_, frame) => {
            const logged = (frame - 1) / 60;
            const [x, y, z] = motionState(motion, logged).position!;
            return `${frame},${logged},0,,0,0,0,0,-1,0,${x},${y},${z}`;
        });
        const folder = mkdtempSync(join(tmpdir(), 'motion-check-'));
        const log = join(folder, 'frames.csv');
        writeFileSync(log, `frame,motion_time_s,wall_time_s,image,gi_mean_r,gi_mean_g,gi_mean_b,gi_mean_luminance,sky_image,sky_share,room-a-lamp.position.x,room-a-lamp.position.y,room-a-lamp.position.z\n${rows.join('\n')}\n`);
        writeFileSync(join(folder, 'recording.json'), JSON.stringify({ clock: 'fixed', frameRate: 60 }));
        expect(checkPositions(manifest, motion, checkTimes(log)).worst).toBeCloseTo(1 / 60, 4);
        rmSync(folder, { recursive: true });
    });

    it("follows T2.4d's day: 5 degrees at both ends, 75 at noon, the azimuth through 160 degrees", () => {
        const start = 0;
        expect(elevation(daySun(DAY, start, 0))).toBeCloseTo(5, 6);
        expect(elevation(daySun(DAY, start, 5))).toBeCloseTo(75, 6);
        expect(elevation(daySun(DAY, start, 10))).toBeCloseTo(5, 6);
        const azimuth = (travel: readonly number[]) => (Math.atan2(-travel[2], -travel[0]) * 180) / Math.PI;
        expect(azimuth(daySun(DAY, start, 0))).toBeCloseTo(-80, 6);
        expect(azimuth(daySun(DAY, start, 10))).toBeCloseTo(80, 6);
    });
});
