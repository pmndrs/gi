import { describe, expect, it } from 'vitest';
import { browserCandidates, buildChromeArgs, findGpuLuid, parseGpuLuidLines } from './chrome.ts';

const DEVICES = [
    'PCI\\VEN_10DE&DEV_2520&SUBSYS_130C1043&REV_A1\\4&1200BD89&0&0008|80901',
    'PCI\\VEN_8086&DEV_9A60&SUBSYS_130C1043&REV_01\\3&11583659&0&10|66912',
].join('\r\n');

describe('GPU LUID lookup', () => {
    it('finds each vendor by PCI vendor id', () => {
        const entries = parseGpuLuidLines(DEVICES);
        expect(findGpuLuid(entries, 'nvidia')).toEqual({ high: 0, low: 80901 });
        expect(findGpuLuid(entries, 'intel')).toEqual({ high: 0, low: 66912 });
    });

    it('splits a 64-bit LUID into high and low halves', () => {
        const luid = (3n << 32n) | 7n;
        const entries = parseGpuLuidLines(`PCI\\VEN_8086&DEV_9A60\\x|${luid}`);
        expect(findGpuLuid(entries, 'intel')).toEqual({ high: 3, low: 7 });
    });

    it('finds nothing when the vendor is absent or the line is malformed', () => {
        expect(findGpuLuid(parseGpuLuidLines(''), 'intel')).toBeNull();
        expect(findGpuLuid(parseGpuLuidLines(DEVICES.split('\r\n')[0] ?? ''), 'intel')).toBeNull();
        expect(parseGpuLuidLines('PCI\\VEN_8086&DEV_9A60\\x|not-a-number')).toEqual([]);
    });
});

describe('buildChromeArgs', () => {
    const base = { profileDir: 'C:/tmp/profile', luid: null, withoutGpu: false };

    it('is headless with a throwaway profile and a free DevTools port', () => {
        const args = buildChromeArgs(base);
        expect(args).toContain('--headless=new');
        expect(args).toContain('--user-data-dir=C:/tmp/profile');
        expect(args).toContain('--remote-debugging-port=0');
    });

    it('pins the adapter only when given a LUID', () => {
        expect(buildChromeArgs(base).some((arg) => arg.startsWith('--use-adapter-luid'))).toBe(false);
        expect(buildChromeArgs({ ...base, luid: { high: 0, low: 66912 } })).toContain('--use-adapter-luid=0,66912');
    });

    it('disables the GPU only when asked', () => {
        expect(buildChromeArgs(base)).not.toContain('--disable-gpu');
        expect(buildChromeArgs({ ...base, withoutGpu: true })).toContain('--disable-gpu');
    });
});

describe('browserCandidates', () => {
    it('prefers installed Chrome over Edge on Windows', () => {
        const candidates = browserCandidates('win32', { ProgramFiles: 'C:/Program Files' });
        expect(candidates[0]).toMatch(/Google[\\/]Chrome[\\/]Application[\\/]chrome\.exe$/);
        expect(candidates.at(-1)).toMatch(/msedge\.exe$/);
    });
});
