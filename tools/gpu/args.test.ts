import { describe, expect, it } from 'vitest';
import { parseRunnerArgs, UsageError } from './args.ts';

describe('parseRunnerArgs', () => {
    it('defaults to any GPU and every page', () => {
        const options = parseRunnerArgs([]);
        expect(options.gpu).toBeNull();
        expect(options.filters).toEqual([]);
        expect(options.withoutWebGpu).toBe(false);
        expect(options.timeoutMs).toBeNull();
        expect(options.maxStorageBuffers).toBeNull();
    });

    it('reads --max-storage-buffers as a positive whole number (pmndrs/gi#49)', () => {
        expect(parseRunnerArgs(['--max-storage-buffers', '10']).maxStorageBuffers).toBe(10);
        expect(() => parseRunnerArgs(['--max-storage-buffers', '0'])).toThrow(UsageError);
        expect(() => parseRunnerArgs(['--max-storage-buffers', 'ten'])).toThrow(UsageError);
    });

    it('reads --gpu', () => {
        expect(parseRunnerArgs(['--gpu', 'intel']).gpu).toBe('intel');
        expect(parseRunnerArgs(['--gpu=nvidia']).gpu).toBe('nvidia');
    });

    it('rejects an unknown GPU', () => {
        expect(() => parseRunnerArgs(['--gpu', 'amd'])).toThrow(UsageError);
    });

    it('splits comma-separated and repeated filters', () => {
        const options = parseRunnerArgs(['--filter', 'smoke,bvh', '--filter', 'timing']);
        expect(options.filters).toEqual(['smoke', 'bvh', 'timing']);
    });

    it('requires a positive whole number for --timeout', () => {
        expect(parseRunnerArgs(['--timeout', '1500']).timeoutMs).toBe(1500);
        expect(() => parseRunnerArgs(['--timeout', '0'])).toThrow(UsageError);
        expect(() => parseRunnerArgs(['--timeout', 'soon'])).toThrow(UsageError);
    });

    it('rejects unknown flags and stray words instead of ignoring them', () => {
        expect(() => parseRunnerArgs(['--gpus', 'intel'])).toThrow(UsageError);
        expect(() => parseRunnerArgs(['intel'])).toThrow(UsageError);
    });

    it('reads --json and --out paths', () => {
        const options = parseRunnerArgs(['--json', 'run.json', '--out', 'shots']);
        expect(options.jsonPath).toBe('run.json');
        expect(options.outDir).toBe('shots');
        expect(parseRunnerArgs([]).jsonPath).toBeNull();
    });

    it('reads the boolean flags', () => {
        const options = parseRunnerArgs(['--no-webgpu', '--verbose']);
        expect(options.withoutWebGpu).toBe(true);
        expect(options.verbose).toBe(true);
    });
});
