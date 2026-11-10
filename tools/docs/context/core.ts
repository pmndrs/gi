/**
 * The names the engine-free docs' examples use without declaring them
 * (docs/guides/adapter.md, docs/api/platform.md): each with its type and no
 * value. Generated snippet files (tools/docs/snippets/) import from here;
 * they are typechecked, never run. See tools/docs/doc-snippets.ts.
 */

import type {
    CompressedPixelLevel,
    DDGITechnique,
    DDGIVolumeSettings,
    FrameCamera,
    GIPlatform,
    InstanceHandle,
    LightHandle,
    Mat4,
    MaterialHandle,
    MeshHandle,
    TextureHandle,
    Vec3,
} from '@pmndrs/ddgi/core';
import type { ProceduralScene } from '../../../scenes/src/scene-data.ts';

// The device and the platform (the guide's steps 1, 2 and 7).
export declare const adapter: GPUAdapter;
export declare const device: GPUDevice;
export declare const gi: GIPlatform;
export declare const ddgi: DDGITechnique;
export declare const encoder: GPUCommandEncoder;

// The hello example's scene data and what the guide made from it.
export declare const data: ProceduralScene;
export declare const meshes: MeshHandle[];
export declare const materials: MaterialHandle[];
export declare const textures: TextureHandle[];
export declare const compressedLevels: CompressedPixelLevel[];
export declare const imageBlob: Blob;
export declare const bricks: TextureHandle;
export declare const wall: MaterialHandle;
export declare const steel: MaterialHandle;
export declare const lamp: LightHandle;
export declare const sun: LightHandle;
export declare const instance: InstanceHandle;
export declare const matrix: Mat4;
export declare const nextMatrix: Mat4;
export declare const movedInstances: Uint32Array;
export declare const movedMatrices: Float32Array;

// Deforming meshes.
export declare const restPositions: Float32Array;
export declare const posedPositions: Float32Array;
export declare const flagIndices: Uint32Array;
export declare const skinnedVertices: GPUBuffer;
export declare const posedPositionTexture: GPUTexture;
export declare const flag: MeshHandle;

// The frame.
export declare const time: number;
export declare const previousTime: number;
export declare const deltaTime: number;
/** Hours since midnight, for the time-of-day example. */
export declare const timeOfDay: number;
export declare const camera: FrameCamera;
export declare const matrices: { position: Vec3; view: Mat4; projection: Mat4; near: number; far: number };
export declare const renderer: { render(encoder: GPUCommandEncoder, matrices: object, view: object): void };
export declare const view: { mode: 'full' };
export declare const myFragmentShader: string;

// DDGI's volume, as the platform docs use it.
export declare const volume: DDGIVolumeSettings;
