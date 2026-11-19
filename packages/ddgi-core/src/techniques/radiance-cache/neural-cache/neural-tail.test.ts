import { describe, expect, it } from 'vitest';
import { MLP_SHAPES, weightCount } from '@pathtracer/mlp';
import { hashGridCacheRayListLayout, MULTIBOUNCE_DIRECT_WORDS, NEURAL_TAIL_WORDS } from '../world-cache/hash-grid-population-wgsl.ts';
import { HASH_GRID_CONSTANTS_BYTES, HASH_GRID_CONSTANTS_WORD } from '../world-cache/hash-grid-wgsl.ts';
import { furnaceCeiling, neuralTailAngles, neuralTailBox, neuralTailGateOpen, neuralTailPosition, neuralTailStepsThisFrame } from './neural-tail-cpu.ts';
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
        expect(neuralTailGateOpen(0, 0)).toBe(true);
    });

    it('checks its settings', () => {
        expect(neuralTailSettings()).toEqual(DEFAULT_NEURAL_TAIL_SETTINGS);
        expect(() => neuralTailSettings({ positionBands: 13 })).toThrow(/positionBands/);
        expect(() => neuralTailSettings({ batch: 100 })).toThrow(/multiple of 64/);
        expect(() => neuralTailSettings({ learningRate: 0 })).toThrow(/learningRate/);
        expect(() => neuralTailSettings({ boundsHalfSize: -1 })).toThrow(/boundsHalfSize/);
        // The compact network is the pathtracer's 64 → 3 × 32 → 3: 4,096 weights.
        expect(weightCount(MLP_SHAPES[DEFAULT_NEURAL_TAIL_SETTINGS.network])).toBe(64 * 32 + 32 * 32 * 2 + 32 * 3);
    });

    it('lays its parameter words out as the WGSL struct does', () => {
        // box_lo, box_scale (vec4 each), a mat4x4f, eye (vec4), then eight words.
        expect(NEURAL_TAIL_PARAMS_WORD.inverseViewProjection).toBe(8);
        expect(NEURAL_TAIL_PARAMS_WORD.eye).toBe(8 + 16);
        expect(NEURAL_TAIL_PARAMS_WORD.frameIndex).toBe(28);
        expect(NEURAL_TAIL_PARAMS_BYTES).toBe((NEURAL_TAIL_PARAMS_WORD.screenHeight + 1) * 4);
    });

    it('gives the hash grid a tail list and a constant only with a tail', () => {
        const without = hashGridCacheRayListLayout(4096, 22282240);
        const withTail = hashGridCacheRayListLayout(4096, 22282240, true, true);
        expect(without.neuralTailBuffer).toBe(0);
        expect(withTail.neuralTailBuffer).toBe(without.multibounceDirectBuffer + MULTIBOUNCE_DIRECT_WORDS * 4096);
        expect(withTail.bytes - without.bytes).toBe(NEURAL_TAIL_WORDS * 4096 * 4);
        expect(HASH_GRID_CONSTANTS_WORD.useNeuralTail * 4).toBeLessThan(HASH_GRID_CONSTANTS_BYTES);
    });

    it('builds its kernels and the network hooks', () => {
        const layout = hashGridCacheBufferLayout(hashGridCacheCapacity(DEFAULT_HASH_GRID_CACHE_OPTIONS));
        expect(gatherCellRecordsWgsl(layout)).toContain(`fn ${NEURAL_TAIL_ENTRY_POINT.gatherCellRecords}(`);
        expect(gatherCellRecordsWgsl(layout)).toContain('HashGridCache_FilteredRadianceIndirect(cell_index)');
        expect(gatherQueriesWgsl(layout)).toContain(`fn ${NEURAL_TAIL_ENTRY_POINT.gatherQueries}(`);
        expect(gatherScreenRecordsWgsl()).toContain(`fn ${NEURAL_TAIL_ENTRY_POINT.gatherScreenRecords}(`);
        expect(neuralTailTrainHooks(12)).toContain('fn nnTrainLoad(r: u32, x: ptr<function, array<f32, NN_IN>>) -> vec3f');
        const infer = neuralTailInferHooks(4);
        expect(infer).toContain('fn nnInferLoad(r: u32) -> array<f32, NN_IN>');
        expect(infer).toContain('fn nnInferStore(r: u32, y: vec3f)');
    });
});
