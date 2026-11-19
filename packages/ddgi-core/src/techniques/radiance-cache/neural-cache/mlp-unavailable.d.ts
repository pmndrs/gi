/**
 * The part of `@pathtracer/mlp`'s API the neural tail uses (P6.4), as
 * declarations only: tsconfig `paths` falls back to this file when the
 * optional sibling isn't cloned (a public checkout), so the code still
 * type-checks. With the clone, TypeScript reads the package itself. Nothing
 * loads this at run time: the tail imports the package dynamically, and a
 * missing one turns the tail off (tools/optional-siblings.ts).
 *
 * Mirrors pmndrs/pathtracer `packages/mlp/src` at the pinned commit (root
 * package.json `linkedSiblings`); keep it in step when the pin moves.
 */

export interface MlpShape {
    inputs: number;
    width: number;
    hidden: number;
    outputs: number;
}

export interface TrainSettings {
    learningRate: number;
    beta1: number;
    beta2: number;
    epsilon: number;
    emaDecay: number;
    emaWarmup?: boolean;
    lossEpsilon: number;
}

export type Precision = 'f32' | 'f16';
export type MlpNetwork = 'default' | 'compact';

export declare const MLP_SHAPES: { readonly default: MlpShape; readonly compact: MlpShape };
export declare const DEFAULT_TRAIN: TrainSettings;

export declare function initWeights(shape: MlpShape, seed: number): Float32Array;
export declare function layerOffset(shape: MlpShape, l: number): number;
export declare function weightCount(shape: MlpShape): number;
export declare function defaultForwardRecords(precision: Precision): number;
export declare function forwardWorkgroupBytes(o: { shape: MlpShape; precision: Precision }, f: { records: number }): number;
export declare function inferTiledWorkgroupBytes(o: { shape: MlpShape; precision: Precision }, f: { records: number }): number;
export declare function nrcEncodingWgsl(positionBands?: number, spatialWgsl?: string): string;

export interface MlpEngineOptions {
    shape?: MlpShape;
    precision: Precision;
    capacity: number;
    inferCapacity?: number;
    train?: TrainSettings;
    useEma?: boolean;
    trainHooks: string;
    inferHooks: string;
    infer?: 'perRecord' | 'tiled';
    inferOptions?: { records?: number };
    forwardOptions?: { records?: number };
    gradientOptions?: { splits?: number; chunk?: number };
}

export declare class MlpEngine {
    static create(device: GPUDevice, options: MlpEngineOptions): Promise<MlpEngine>;
    readonly shape: MlpShape;
    readonly precision: Precision;
    readonly weightCount: number;
    readonly trainState: GPUBuffer;
    readonly inferBackend: 'portable' | 'portable-tiled' | 'sgm' | 'sgm-emulated';
    readonly trainLayout: GPUBindGroupLayout;
    readonly inferLayout: GPUBindGroupLayout;
    readonly bufferBytes: number;
    uploadWeights(weights: Float32Array): void;
    resetOptimizer(): void;
    encodeTrainStep(encoder: GPUCommandEncoder, counts: GPUBuffer, countWord: number, hooks: GPUBindGroup, timestampWrites?: GPUComputePassTimestampWrites): void;
    encodeInfer(encoder: GPUCommandEncoder, counts: GPUBuffer, countWord: number, hooks: GPUBindGroup, timestampWrites?: GPUComputePassTimestampWrites): void;
    destroy(): void;
}
