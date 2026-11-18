/**
 * The names the three.js docs' examples use without declaring them
 * (README.md, docs/api/three.md): each with its type and no value.
 * Generated snippet files (tools/docs/snippets/) import from here; they are
 * typechecked, never run. See tools/docs/doc-snippets.ts.
 */

import type { DirectionalLight, Mesh, Object3D, PerspectiveCamera, Scene, WebGPURenderer } from 'three/webgpu';
import type { GIPlatform } from '@pmndrs/gi/core';
import type { DDGIVolume, ThreeGI } from '@pmndrs/gi/three';

export declare const renderer: WebGPURenderer;
export declare const scene: Scene;
export declare const camera: PerspectiveCamera;
export declare const volume: DDGIVolume;
export declare const sun: DirectionalLight;
export declare const gi: GIPlatform;
export declare const threeGI: ThreeGI;
export declare const door: Object3D;
export declare const flag: Mesh;
export declare const character: Object3D;
export declare const lamp: Object3D;
/** Hours since midnight, for the time-of-day example. */
export declare const hours: number;
