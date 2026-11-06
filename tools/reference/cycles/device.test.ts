import { describe, expect, it } from 'vitest';
import { benchLockFor, renderDevice } from './device.ts';

describe('the render device and its bench lock', () => {
    it('reads --device cpu and --device=cpu as the CPU, with the CPU lock', () => {
        for (const args of [['--scene', 'sponza', '--device', 'cpu'], ['--scene', 'sponza', '--device=cpu', '--threads', '12']]) {
            expect(renderDevice(args)).toBe('cpu');
            expect(benchLockFor(renderDevice(args))).toBe('cpu');
        }
    });

    it('defaults to the GPU, with the GPU lock, as render.py does', () => {
        for (const args of [['--scene', 'sponza'], ['--device', 'gpu'], ['--device=gpu']]) {
            expect(renderDevice(args)).toBe('gpu');
            expect(benchLockFor(renderDevice(args))).toBe('gpu');
        }
    });

    it('takes the last --device, as argparse does', () => {
        expect(renderDevice(['--device', 'gpu', '--device=cpu'])).toBe('cpu');
        expect(renderDevice(['--device=cpu', '--device', 'gpu'])).toBe('gpu');
    });
});
