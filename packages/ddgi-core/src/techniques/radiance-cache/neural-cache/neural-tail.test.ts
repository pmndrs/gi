import { describe, expect, it } from 'vitest';
import { hashGridTailRuns } from '../world-cache/hash-grid-cache.ts';
import { hashGridCacheRayListLayout, MULTIBOUNCE_DIRECT_WORDS, NEURAL_TAIL_WORDS, RAY_LISTS_UNIFORM_BYTES } from '../world-cache/hash-grid-population-wgsl.ts';
import { HASH_GRID_CONSTANTS_BYTES, HASH_GRID_CONSTANTS_WORD } from '../world-cache/hash-grid-wgsl.ts';
import { loadMlp, type MlpModule } from './mlp-loader.ts';
import { furnaceCeiling, skyStripsCeiling, neuralTailAngles, neuralTailBox, neuralTailGateOpen, neuralTailNeedsRecentre, neuralTailPosition, neuralTailQueryCapacity, neuralTailSetCapacity, neuralTailStepsThisFrame } from './neural-tail-cpu.ts';
import { DEFAULT_NEURAL_TAIL_SETTINGS, neuralTailSettings } from './neural-tail-settings.ts';
import {
    gatherCellRecordsWgsl,
    gatherQueriesWgsl,
    gatherScreenRecordsWgsl,
    NEURAL_TAIL_ENTRY_POINT,
    NEURAL_TAIL_PARAMS_BYTES,
    NEURAL_TAIL_PARAMS_WORD,
    neuralTailInferHooks,
    neuralTailTrainHooks,
} from './neural-tail-wgsl.ts';
import { hashGridCacheBufferLayout, hashGridCacheCapacity, DEFAULT_HASH_GRID_CACHE_OPTIONS } from '../world-cache/capacity.ts';

/** The optional sibling, where this checkout has it at its pin (tools/optional-siblings.ts); null otherwise. */
const mlp: MlpModule | null = await loadMlp().catch(() => null);

describe('the neural tail (P6.4, register E11)', () => {
    it('reads the furnace answers the GPU page judges against', () => {
        // ρ = 0.5, E = 4: the floor's light ρE/π = 0.6366.
        expect(furnaceCeiling(0.5, 4, 'second-bounce')).toBeCloseTo(0.31831, 5);
        expect(furnaceCeiling(0.5, 4, 'every-bounce')).toBeCloseTo(0.42441, 5);
        expect(furnaceCeiling(0.5, 4, 'second-bounce', 0.5)).toBeCloseTo(0.15915, 5);
        expect(furnaceCeiling(0.5, 4, 'every-bounce', 0.5)).toBeCloseTo(0.18189, 5);
        // The every-bounce answer is the fixed point L = ρ fill (L_f + ρ L).
        const ceiling = furnaceCeiling(0.3, 2, 'every-bounce', 0.7);
        expect(ceiling).toBeCloseTo(0.3 * 0.7 * ((0.3 * 2) / Math.PI + 0.3 * ceiling), 12);
    });

    it('reads the strips under a sky the GPU page judges against (RC3.4)', () => {
        expect(skyStripsCeiling(0.5, 1, 'second-bounce')).toBeCloseTo(0.125, 12);
        expect(skyStripsCeiling(0.5, 1, 'every-bounce')).toBeCloseTo(1 / 7, 12);
        // The fixed point: L_c = ρ L_f with L_f = ρ ((1 − fill) L + fill L_c).
        const ceiling = skyStripsCeiling(0.3, 2, 'every-bounce', 0.4);
        expect(ceiling).toBeCloseTo(0.3 * 0.3 * (0.6 * 2 + 0.4 * ceiling), 12);
    });

    it('gives each direction two angles in [0, 1]', () => {
        expect(neuralTailAngles([0, 1, 0])[0]).toBeCloseTo(0, 6);
        expect(neuralTailAngles([0, -1, 0])[0]).toBeCloseTo(1, 6);
        expect(neuralTailAngles([1, 0, 0])).toEqual([0.5, 0.5]);
        expect(neuralTailAngles([0, 0, 1])[1]).toBeCloseTo(0.75, 6);
        expect(neuralTailAngles([0, 0, -1])[1]).toBeCloseTo(0.25, 6);
        for (const direction of [[0.3, -0.2, -0.9], [-1, 0, -1e-9], [0.2, 0.9, 0.4]] as const) {
            const [theta, phi] = neuralTailAngles(direction);
            expect(theta).toBeGreaterThanOrEqual(0);
            expect(theta).toBeLessThanOrEqual(1);
            expect(phi).toBeGreaterThanOrEqual(0);
            expect(phi).toBeLessThanOrEqual(1);
        }
    });

    it('maps the box to [−1, 1]³ and clamps outside it', () => {
        const box = neuralTailBox({ min: [-2, 0, 10], max: [2, 4, 10] });
        expect(box.scale).toEqual([0.5, 0.5, 1]); // a flat axis keeps scale 1
        expect(neuralTailPosition([-2, 0, 10], box)).toEqual([-1, -1, -1]);
        expect(neuralTailPosition([2, 4, 10], box)).toEqual([1, 1, -1]);
        expect(neuralTailPosition([0, 2, 10], box)).toEqual([0, 0, -1]);
        expect(neuralTailPosition([9, -9, 99], box)).toEqual([1, -1, 1]);
    });

    it('trains fast until the gate opens, then at the steady rate', () => {
        const settings = { warmupSteps: 256, warmupStepsPerFrame: 16, stepsPerFrame: 2 };
        let steps = 0;
        let frames = 0;
        while (!neuralTailGateOpen(steps, settings.warmupSteps)) {
            steps += neuralTailStepsThisFrame(steps, settings);
            frames++;
        }
        expect(frames).toBe(16);
        expect(neuralTailStepsThisFrame(steps, settings)).toBe(2);
        expect(neuralTailStepsThisFrame(0, { ...settings, warmupStepsPerFrame: 0 })).toBe(2);
        // After a change with no bounds: the warm-up rate again.
        expect(neuralTailStepsThisFrame(steps, settings, true)).toBe(16);
        expect(neuralTailGateOpen(0, 0)).toBe(true);
    });

    it('checks its settings', () => {
        expect(neuralTailSettings()).toEqual(DEFAULT_NEURAL_TAIL_SETTINGS);
        expect(() => neuralTailSettings({ positionBands: 13 })).toThrow(/positionBands/);
        expect(() => neuralTailSettings({ batch: 100 })).toThrow(/multiple of 64/);
        expect(() => neuralTailSettings({ learningRate: 0 })).toThrow(/learningRate/);
        expect(() => neuralTailSettings({ boundsHalfSize: -1 })).toThrow(/boundsHalfSize/);
        expect(DEFAULT_NEURAL_TAIL_SETTINGS.ema).toBe('off');
    });

    it.skipIf(!mlp)('uses the compact network of the pathtracer (with the optional sibling)', () => {
        // 64 → 3 × 32 → 3: 4,192 weights.
        expect(mlp!.weightCount(mlp!.MLP_SHAPES[DEFAULT_NEURAL_TAIL_SETTINGS.network])).toBe(64 * 32 + 32 * 32 * 2 + 32 * 3);
    });

    it('sizes its buffers by what a frame gathers, not by the screen', () => {
        // The training set: both kinds of record, a quarter to spare, the same at any screen size.
        expect(neuralTailSetCapacity({ cellRecords: 16384, screenRecords: 16384, screenStride: 8 })).toBe(40960);
        expect(neuralTailSetCapacity({ cellRecords: 16384, screenRecords: 16384, screenStride: 0 })).toBe(20480);
        // The queries: (1 − p) of the probe rays with half again to spare (4K: 2,073,600 rays at p = 0.7).
        expect(neuralTailQueryCapacity(2073600, 0.7)).toBe(933120);
        expect(neuralTailQueryCapacity(1000, 0)).toBe(1000);
        // The default box is centred again once the camera leaves its middle half.
        expect(neuralTailNeedsRecentre([31, 0, 0], [0, 0, 0], 64)).toBe(false);
        expect(neuralTailNeedsRecentre([0, 0, -33], [0, 0, 0], 64)).toBe(true);
    });

    it('lays its parameter words out as the WGSL struct does', () => {
        // Parsed from the kernel's own struct: each field's word (vec4f 4, mat4x4f 16, u32 1), named as the TypeScript key in snake case.
        const layout = hashGridCacheBufferLayout(hashGridCacheCapacity(DEFAULT_HASH_GRID_CACHE_OPTIONS));
        const struct = /struct NeuralTailParams \{([^}]*)\}/.exec(gatherScreenRecordsWgsl(layout))?.[1] ?? '';
        const sizes: Record<string, number> = { vec4f: 4, mat4x4f: 16, u32: 1, f32: 1 };
        const offsets = new Map<string, number>();
        let word = 0;
        for (const line of struct.split(/\r?\n/).map((text) => text.replace(/\/\/.*$/, '').trim()).filter((text) => text.endsWith(','))) {
            const [name, type] = line.slice(0, -1).split(':').map((part) => part.trim());
            expect(sizes[type], `${name}: ${type}`).toBeDefined();
            offsets.set(name, word);
            word += sizes[type];
        }
        for (const [key, offset] of Object.entries(NEURAL_TAIL_PARAMS_WORD)) {
            const field = key.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
            expect(offsets.get(field), `${key} → ${field}`).toBe(offset);
        }
        expect(word * 4).toBe(NEURAL_TAIL_PARAMS_BYTES);
    });

    it('gives the hash grid a tail list only with a tail and its own second-bounce rays', () => {
        const without = hashGridCacheRayListLayout(4096, 22282240);
        const withTail = hashGridCacheRayListLayout(4096, 22282240, true, true);
        expect(without.neuralTailBuffer).toBe(0);
        expect(withTail.neuralTailBuffer).toBe(without.multibounceDirectBuffer + MULTIBOUNCE_DIRECT_WORDS * 4096);
        expect(withTail.bytes - without.bytes).toBe(NEURAL_TAIL_WORDS * 4096 * 4);
        // A tail with no second-bounce rays (the bounce from a fallback, E10): no list, no bytes.
        const noRays = hashGridCacheRayListLayout(4096, 22282240, false);
        const noRaysWithTail = hashGridCacheRayListLayout(4096, 22282240, false, true);
        expect(noRaysWithTail.neuralTailBuffer).toBe(0);
        expect(noRaysWithTail.bytes).toBe(noRays.bytes);
    });

    it('runs the tail, and sets use_neural_tail, only with GI-1.2 rays that have settled', () => {
        expect(HASH_GRID_CONSTANTS_WORD.useNeuralTail * 4).toBeLessThan(HASH_GRID_CONSTANTS_BYTES);
        expect(hashGridTailRuns(true, 'gi1-rays', false)).toBe(true);
        expect(hashGridTailRuns(false, 'gi1-rays', false)).toBe(false); // multibounce off: no second bounce
        expect(hashGridTailRuns(true, 'fallback', false)).toBe(false); // E10: the fallback's field has the later bounces
        expect(hashGridTailRuns(true, 'gi1-rays', true)).toBe(false); // 'auto' while the fallback's pass builds
    });

    it('lays the hash grid ray-list uniform out in its bytes', () => {
        const layout = hashGridCacheBufferLayout(hashGridCacheCapacity(DEFAULT_HASH_GRID_CACHE_OPTIONS));
        const struct = /struct HashGridCacheRayLists \{([^}]*)\}/.exec(gatherCellRecordsWgsl(layout))?.[1] ?? '';
        const fields = struct.split(/\r?\n/).map((line) => line.replace(/\/\/.*$/, '').trim()).filter((line) => line.endsWith(','));
        expect(fields.every((field) => field.endsWith(': u32,'))).toBe(true);
        // Words 16 and 17, as hash-grid-cache.ts writes them.
        expect(fields[16]).toMatch(/^neural_tail_buffer:/);
        expect(fields.at(-1)).toMatch(/^reset_frames_buffer:/);
        expect(fields).toHaveLength(18);
        expect(RAY_LISTS_UNIFORM_BYTES).toBe(Math.ceil((fields.length * 4) / 16) * 16);
    });

    it('builds its kernels and the network hooks', () => {
        const layout = hashGridCacheBufferLayout(hashGridCacheCapacity(DEFAULT_HASH_GRID_CACHE_OPTIONS));
        expect(gatherCellRecordsWgsl(layout)).toContain(`fn ${NEURAL_TAIL_ENTRY_POINT.gatherCellRecords}(`);
        expect(gatherCellRecordsWgsl(layout)).toContain('HashGridCache_FilteredRadianceIndirect(cell_index)');
        expect(gatherQueriesWgsl(layout)).toContain(`fn ${NEURAL_TAIL_ENTRY_POINT.gatherQueries}(`);
        expect(gatherScreenRecordsWgsl(layout)).toContain(`fn ${NEURAL_TAIL_ENTRY_POINT.gatherScreenRecords}(`);
        expect(gatherScreenRecordsWgsl(layout)).toContain('worldCacheLookup(');
        expect(neuralTailTrainHooks('// encoding')).toContain('fn nnTrainLoad(r: u32, x: ptr<function, array<f32, NN_IN>>) -> vec3f');
        const infer = neuralTailInferHooks('// encoding');
        expect(infer).toContain('fn nnInferLoad(r: u32) -> array<f32, NN_IN>');
        expect(infer).toContain('fn nnInferStore(r: u32, y: vec3f)');
    });
});
