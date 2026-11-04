import { Color, CubeTexture, DataTexture, DirectionalLight, EquirectangularReflectionMapping, FloatType, RGBAFormat, Scene, Texture } from 'three';
import { describe, expect, it, vi } from 'vitest';
import type { EnvironmentDescriptor } from '@pmndrs/ddgi-core';
import { chooseEnvironment, environmentDescriptor, sameChoice, sunIlluminance, type PmndrsSkyLike, type SkyMeshLike } from './environment.ts';

const FAKE_GPU_TEXTURE = { label: 'uploaded' } as unknown as GPUTexture;
const upload = () => FAKE_GPU_TEXTURE;

function equirect(): DataTexture {
    const texture = new DataTexture(new Float32Array(8 * 4 * 4), 8, 4, RGBAFormat, FloatType);
    texture.mapping = EquirectangularReflectionMapping;
    return texture;
}

function cube(): CubeTexture {
    const face = { data: new Uint8Array(4 * 4), width: 2, height: 2 };
    return new CubeTexture([face, face, face, face, face, face]);
}

function describeScene(scene: Scene, warn = vi.fn()): EnvironmentDescriptor {
    return environmentDescriptor(chooseEnvironment(scene, {}), upload, warn);
}

describe('environment from the scene', () => {
    it('passes environmentRotation.y and environmentIntensity with an equirect', () => {
        const scene = new Scene();
        scene.environment = equirect();
        scene.environmentRotation.set(0, 0.7, 0);
        scene.environmentIntensity = 2.5;
        const warn = vi.fn();
        expect(describeScene(scene, warn)).toEqual({ type: 'texture', texture: FAKE_GPU_TEXTURE, layout: 'equirect', intensity: 2.5, rotationY: 0.7 });
        expect(warn).not.toHaveBeenCalled();
    });

    it('warns that x and z rotations are dropped', () => {
        const scene = new Scene();
        scene.environment = equirect();
        scene.environmentRotation.set(0.2, 0.7, 0);
        const warn = vi.fn();
        const descriptor = describeScene(scene, warn);
        expect(descriptor.type === 'texture' && descriptor.rotationY).toBe(0.7);
        expect(warn).toHaveBeenCalledOnce();
    });

    it('marks a cube texture as a cube (the upload mirrors it)', () => {
        const scene = new Scene();
        scene.environment = cube();
        const descriptor = describeScene(scene);
        expect(descriptor.type === 'texture' && descriptor.layout).toBe('cube');
    });

    it('falls back to the background texture, with its own intensity and rotation', () => {
        const scene = new Scene();
        scene.background = cube();
        scene.backgroundIntensity = 0.5;
        scene.backgroundRotation.set(0, -1.2, 0);
        expect(describeScene(scene)).toMatchObject({ type: 'texture', layout: 'cube', intensity: 0.5, rotationY: -1.2 });
    });

    it('skips textures it can\'t read, and says why', () => {
        const scene = new Scene();
        const renderTarget = equirect() as DataTexture & { isRenderTargetTexture?: boolean };
        renderTarget.isRenderTargetTexture = true;
        scene.environment = renderTarget;
        scene.background = new Color(0.1, 0.2, 0.3);
        const skipped = vi.fn();
        const choice = chooseEnvironment(scene, {}, skipped);
        expect(choice.kind).toBe('color');
        expect(skipped).toHaveBeenCalledWith('scene.environment', expect.stringContaining('render-target'));
    });

    it('ignores a plain 2D texture background (three draws it flat on screen, not as a sky)', () => {
        const scene = new Scene();
        scene.background = new Texture();
        expect(describeScene(scene)).toEqual({ type: 'none' });
    });

    it('uses a background colour as a uniform sky, and none without anything', () => {
        const scene = new Scene();
        expect(describeScene(scene)).toEqual({ type: 'none' });
        scene.background = new Color(0.25, 0.5, 1);
        expect(describeScene(scene)).toEqual({ type: 'color', color: [0.25, 0.5, 1] });
    });
});

describe('skies', () => {
    const pmndrsSky: PmndrsSkyLike = { baker: { sky: { luminanceScale: { value: 40 }, sunDirection: { value: { x: 0, y: 3, z: 4 } } } } };

    it('maps a pmndrs/sky exposure to sunIlluminance', () => {
        const descriptor = environmentDescriptor(chooseEnvironment(new Scene(), { sky: pmndrsSky }), upload, vi.fn());
        expect(descriptor).toEqual({ type: 'procedural', sunDirection: [0, 0.6, 0.8], turbidity: undefined, sunIlluminance: 40 });
    });

    it('takes a SkyMesh\'s sunIlluminance from the sun light', () => {
        const sun = new DirectionalLight(new Color(1, 1, 1), 80000);
        const skyMesh: SkyMeshLike = { isSkyMesh: true, turbidity: { value: 3 }, sunPosition: { value: { x: 1, y: 1, z: 0 } } };
        const descriptor = environmentDescriptor(chooseEnvironment(new Scene(), { skyMesh, sun }), upload, vi.fn());
        expect(descriptor.type).toBe('procedural');
        if (descriptor.type !== 'procedural') return;
        expect(descriptor.sunIlluminance).toBeCloseTo(80000);
        expect(descriptor.turbidity).toBe(3);
        expect(descriptor.sunDirection[0]).toBeCloseTo(Math.SQRT1_2);
    });

    it('weights the sun\'s colour by luminance', () => {
        expect(sunIlluminance(new DirectionalLight(new Color(1, 0, 0), 10))).toBeCloseTo(2.126);
        expect(sunIlluminance(null)).toBe(1);
    });

    it('prefers an explicit descriptor, then pmndrs/sky, then the scene', () => {
        const scene = new Scene();
        scene.environment = equirect();
        const override: EnvironmentDescriptor = { type: 'color', color: [1, 1, 1] };
        expect(chooseEnvironment(scene, { override, sky: pmndrsSky }).kind).toBe('override');
        expect(chooseEnvironment(scene, { sky: pmndrsSky }).kind).toBe('procedural');
        expect(chooseEnvironment(scene, {}).kind).toBe('texture');
    });
});

describe('change detection', () => {
    it('sees intensity, rotation and texture version changes, and nothing else', () => {
        const scene = new Scene();
        const texture = equirect();
        scene.environment = texture;
        const first = chooseEnvironment(scene, {});
        expect(sameChoice(first, chooseEnvironment(scene, {}))).toBe(true);
        scene.environmentIntensity = 2;
        const second = chooseEnvironment(scene, {});
        expect(sameChoice(first, second)).toBe(false);
        scene.environmentRotation.y = 1;
        const third = chooseEnvironment(scene, {});
        expect(sameChoice(second, third)).toBe(false);
        texture.needsUpdate = true;
        expect(sameChoice(third, chooseEnvironment(scene, {}))).toBe(false);
    });
});
