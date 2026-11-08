import { describe, expect, it } from 'vitest';
import { pairRecordings, type ListedRecording } from './pairs';

function listed(source: 'native' | 'viewer', name: string, motion: string, clock = 'fixed'): ListedRecording {
    return { source, name, url: `/@fs/x/${name}/`, scene: 'two-room', motion, clock };
}

describe('recording pairs', () => {
    it('pairs ours with native by scene and motion, our probe grid first, without settle runs or logs', () => {
        const pairs = pairRecordings([
            listed('native', 'two-room-light-step-run1', 'light-step'),
            listed('native', 'two-room-light-step-ddgi-run1', 'light-step'),
            listed('native', 'two-room-light-step-ddgi-settle0', 'light-step'),
            listed('native', 'two-room-light-sweep', 'light-sweep'),
            listed('native', 'two-room-light-sweep-log', 'light-sweep', 'wall'),
            listed('native', 'sponza-time-of-day', 'time-of-day'),
            listed('viewer', 'two-room-light-step', 'light-step'),
            listed('viewer', 'two-room-light-sweep', 'light-sweep'),
        ]);
        expect(pairs.map((pair) => pair.key)).toEqual(['two-room-light-step', 'two-room-light-sweep']);
        expect(pairs[0].native.map((recording) => recording.name)).toEqual(['two-room-light-step-ddgi-run1', 'two-room-light-step-run1']);
        expect(pairs[1].native.map((recording) => recording.name)).toEqual(['two-room-light-sweep']);
        expect(pairs[1].ours.map((recording) => recording.name)).toEqual(['two-room-light-sweep']);
    });
});
