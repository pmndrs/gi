/**
 * The names the radiance cache's docs' examples use without declaring them
 * (docs/api/radiance-cache.md): each with its type and no value. Generated
 * snippet files (tools/docs/snippets/) import from here; they are
 * typechecked, never run. See tools/docs/doc-snippets.ts.
 */

import type { PerspectiveCamera, Scene } from 'three/webgpu';
import type { FrameCamera } from '@pmndrs/gi/core';

// The three.js examples.
export declare const scene: Scene;
export declare const camera: PerspectiveCamera;
export declare const device: GPUDevice;

// The engine-free example: the host's adapter, frame and screen textures.
export declare const adapter: GPUAdapter;
export declare const encoder: GPUCommandEncoder;
export declare const deltaTime: number;
export declare const frameCamera: FrameCamera;
export declare const width: number;
export declare const height: number;
export declare const depth: GPUTexture;
export declare const normal: GPUTexture;
export declare const velocity: GPUTexture;
