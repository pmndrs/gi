import { describe, expect, it } from 'vitest';
import { curveValues, duration, frameAt, framesCsv, gridNote, imageAt, makeRecording, type RecordingInfo } from './recording';

const INFO: RecordingInfo = { format: 'test', scene: 'two-room', motion: 'light-sweep', camera: 'lit-room', width: 16, height: 9, view: 'full', clock: 'fixed', frameRate: 60, frames: 9, imageEvery: 4 };

/** Native's columns (tools/native/README.md), 9 frames, images every 4th. */
const NATIVE_CSV = [
    'frame,motion_time_s,wall_time_s,image,gi_mean_r,gi_mean_g,gi_mean_b,gi_mean_luminance,sky_image,sky_share,room-a-lamp.position.x',
    ...Array.from({ length: 9 }, (_, frame) => `${frame},${frame / 60},${frame * 0.1},${frame % 4 === 0 ? `frames/frame-${String(frame).padStart(6, '0')}.png` : ''},${0.3 + frame * 0.03},${0.3},${0.3},0.3,-1,0,1`),
].join('\n');

describe('motion recordings', () => {
    it("reads native's frames.csv: the brightness is the mean of red, green and blue", () => {
        const recording = makeRecording(INFO, NATIVE_CSV);
        expect(recording.frames).toHaveLength(9);
        expect(recording.imageFrames.map((frame) => frame.frame)).toEqual([0, 4, 8]);
        expect(recording.frames[3].value).toBeCloseTo((0.39 + 0.3 + 0.3) / 3, 9);
        expect(duration(recording)).toBeCloseTo(8 / 60, 9);
    });

    it('reads our frames.csv back as written', () => {
        const csv = framesCsv(
            [0, 1, 2, 3, 4].map((frame) => ({ frame, time: frame / 60, wallTime: frame * 0.017, image: frame % 4 === 0 ? `frames/frame-${frame}.jpg` : null, value: 0.5 + frame, probesUpdated: 400, light: [1, 1.8, 2] as const })),
            'room-a-lamp',
        );
        const recording = makeRecording({ ...INFO, frames: 5 }, csv);
        expect(recording.frames.map((frame) => frame.value)).toEqual([0.5, 1.5, 2.5, 3.5, 4.5]);
        expect(recording.imageFrames.map((frame) => frame.image)).toEqual(['frames/frame-0.jpg', 'frames/frame-4.jpg']);
    });

    it('shows, at each moment, the last image drawn by then, on both sides alike', () => {
        const recording = makeRecording(INFO, NATIVE_CSV);
        expect(imageAt(recording, 0)?.frame).toBe(0);
        expect(imageAt(recording, 3 / 60)?.frame).toBe(0);
        expect(imageAt(recording, 4 / 60)?.frame).toBe(4);
        expect(imageAt(recording, 7.99 / 60)?.frame).toBe(4);
        expect(imageAt(recording, 100)?.frame).toBe(8);
        expect(imageAt(recording, -1)?.frame).toBe(0);
        expect(frameAt(recording, 5 / 60)?.frame).toBe(5);
        expect(frameAt(recording, 100)?.frame).toBe(8);
    });

    it('says when the two sides used different probe grids', () => {
        const ours: RecordingInfo = { ...INFO, source: 'viewer', gridRule: 'ddgi', probes: 1071 };
        expect(gridNote(ours, { ...INFO, gridRule: 'ddgi', probes: 1071 })).toBeNull();
        // Native's recordings without a gridRule used its own cells grid.
        expect(gridNote(ours, { ...INFO, probes: 918 })).toBe('Different probe grids: native used its own ("cells", 918 probes), ours DDGI\'s ("ddgi", 1,071 probes).');
        expect(gridNote({ ...INFO, source: 'viewer' }, { ...INFO, gridRule: 'ddgi' })).toBeNull();
    });

    it('scales a curve to its first frame, or leaves it as measured', () => {
        const recording = makeRecording(INFO, NATIVE_CSV);
        const relative = curveValues(recording, 'relative');
        expect(relative[0]).toBe(1);
        expect(relative[8]).toBeCloseTo(recording.frames[8].value / recording.frames[0].value, 9);
        expect(curveValues(recording, 'measured')[8]).toBe(recording.frames[8].value);
    });
});
