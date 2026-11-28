/**
 * Settings of the neural tail (phase 6 experiment P6.4, register E11): a
 * small network from pmndrs/pathtracer's `@pathtracer/mlp` that learns the
 * hash grid's indirect values and answers the second bounce's tail, so
 * GI-1.2's two bounces become every bounce in expectation while the grid
 * stays the fast-reacting near cache (docs/design/radiance-cache.md,
 * "Neural tail").
 *
 * The network's defaults are the pathtracer's (`bench/nrc/cache.ts`,
 * `DEFAULT_NRC_SETTINGS` at 0e4435b): its compact shape (64 → 3 × 32 → 3,
 * `phase10Settings('mlp')`), f16 where the device has `shader-f16`, batches
 * of 16,384, Adam at 0.002 and a zero output layer at the start. Ours: what
 * the training set holds, how many steps open the gate, the inference weights
 * (the trained ones, not the EMA: the same flicker and less lag, P6.4's
 * review), and a burst of training after a change with no bounds.
 */

import type { MlpNetwork } from '@pathtracer/mlp';
import type { Aabb } from '@pmndrs/gi-core/platform/contract/math';

/** `neuralCache`: off (GI-1.2's two bounces, the default) or the network as the second bounce's tail. */
export type NeuralCacheMode = 'off' | 'tail';

export interface NeuralTailSettings {
    /** The network's shape (`MLP_SHAPES`): `compact` (64 → 3 × 32 → 3) or `default` (64 → 5 × 64 → 3, VkNRC's). */
    network: MlpNetwork;
    /** `auto`: f16 where the device has `shader-f16`, else f32. */
    precision: 'auto' | 'f16' | 'f32';
    /** Triangle-wave bands per position axis (1-12; `encodeNrc`'s `positionBands`). */
    positionBands: number;
    /** Records per training step, drawn at random from the frame's training set. */
    batch: number;
    /** Training steps a frame once the gate is open. */
    stepsPerFrame: number;
    /** Training steps a frame while the gate is closed (the pathtracer's `warmupStepsPerFrame`). */
    warmupStepsPerFrame: number;
    /** The gate: the tail is used once the network has taken this many training steps since it was (re)started. */
    warmupSteps: number;
    /** About this many first-bounce cells' indirect values go into the training set each frame (a random share of the hits). */
    cellRecords: number;
    /**
     * Screen records come from blocks of `screenStride` × `screenStride`
     * pixels (one random pixel each); 0: none.
     */
    screenStride: number;
    /** About this many screen records a frame: a random share of the blocks, so the share is the same anywhere on the screen at any size. */
    screenRecords: number;
    /**
     * Frames of training at `warmupStepsPerFrame` after a change with no
     * bounds (the sun, the sky: the platform's change events), so the network
     * follows the new light as fast as the grid does.
     */
    changeBoostFrames: number;
    /** Adam's learning rate. */
    learningRate: number;
    /** Inference weights: the trained weights (default), the warm-up EMA (min(0.99, (1 + t) / (10 + t)), the pathtracer's default) or the plain EMA (0.99). */
    ema: 'warmup' | 'plain' | 'off';
    /** Seed of the network's first weights. */
    seed: number;
    /**
     * The box the network's position input spans ([−1, 1]³; positions outside
     * are clamped). The technique's context has no scene bounds, so a host
     * that knows them passes them here (`gi.scene.bounds`). Null: a cube
     * `boundsHalfSize` metres each way around the camera, centred again (and
     * the network started over) when the camera leaves its middle half.
     */
    bounds: Aabb | null;
    /** Half the side of the default box, in metres. */
    boundsHalfSize: number;
}

export const DEFAULT_NEURAL_TAIL_SETTINGS: Readonly<NeuralTailSettings> = Object.freeze({
    network: 'compact',
    precision: 'auto',
    positionBands: 12,
    batch: 16384,
    stepsPerFrame: 2,
    warmupStepsPerFrame: 16,
    warmupSteps: 256,
    cellRecords: 16384,
    screenStride: 8,
    screenRecords: 16384,
    changeBoostFrames: 16,
    learningRate: 0.002,
    ema: 'off',
    seed: 1,
    bounds: null,
    boundsHalfSize: 64,
});

/** The settings over the defaults, checked. */
export function neuralTailSettings(overrides: Partial<NeuralTailSettings> = {}): NeuralTailSettings {
    const settings = { ...DEFAULT_NEURAL_TAIL_SETTINGS, ...overrides };
    const whole = (name: keyof NeuralTailSettings, value: number, min: number, max = Number.MAX_SAFE_INTEGER): void => {
        if (!Number.isInteger(value) || value < min || value > max) throw new RangeError(`@pmndrs/gi: neural tail ${name} must be a whole number from ${min} to ${max}, got ${value}`);
    };
    whole('positionBands', settings.positionBands, 1, 12);
    whole('batch', settings.batch, 64);
    if (settings.batch % 64 !== 0) throw new RangeError(`@pmndrs/gi: neural tail batch must be a multiple of 64, got ${settings.batch}`);
    whole('stepsPerFrame', settings.stepsPerFrame, 0, 64);
    whole('warmupStepsPerFrame', settings.warmupStepsPerFrame, 0, 64);
    whole('warmupSteps', settings.warmupSteps, 0);
    whole('cellRecords', settings.cellRecords, 0);
    whole('screenStride', settings.screenStride, 0, 256);
    whole('screenRecords', settings.screenRecords, 0);
    whole('changeBoostFrames', settings.changeBoostFrames, 0);
    whole('seed', settings.seed, 0);
    if (!(settings.boundsHalfSize > 0)) throw new RangeError(`@pmndrs/gi: neural tail boundsHalfSize must be above 0, got ${settings.boundsHalfSize}`);
    if (!(settings.learningRate > 0)) throw new RangeError(`@pmndrs/gi: neural tail learningRate must be above 0, got ${settings.learningRate}`);
    return settings;
}
