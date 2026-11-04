/**
 * The test scene shared by every T0.8 page: a white plane lit by nothing but
 * our GI light, seen from above, rendered into a float render target so the
 * pixel values can be read back and compared with a number worked out by hand.
 *
 * For a white Lambert surface lit only by irradiance E, the outgoing radiance
 * is E * albedo / PI. With albedo 1 that is E / PI per channel.
 */

import {
    FloatType,
    MeshLambertNodeMaterial,
    Mesh,
    PerspectiveCamera,
    PlaneGeometry,
    RenderTarget,
    Scene,
    Vector3,
    type Material,
    type WebGPURenderer,
} from 'three/webgpu';
import { assert } from '../../client.ts';

export const RENDER_SIZE = 128;

/** The GI volume box: 3 x 3 x 3 around the origin. The plane (8 x 8) is larger. */
export const VOLUME_CENTER = new Vector3(0, 0, 0);
export const VOLUME_SIZE = new Vector3(3, 3, 3);

/**
 * Relative tolerance of the pixel checks, plus a small absolute floor for
 * channels near zero. Stored values are exact in half float and the shader math
 * is plain float arithmetic, so the real error is well below this; the stated
 * tolerance leaves room for different GPUs' filtering and rounding.
 */
export const RELATIVE_TOLERANCE = 0.01;
export const ABSOLUTE_TOLERANCE = 0.001;

export interface TestScene {
    scene: Scene;
    camera: PerspectiveCamera;
    plane: Mesh;
    target: RenderTarget;
}

export interface TestSceneOptions {
    /** The material for the plane. Default: white `MeshLambertNodeMaterial`. */
    material?: Material;
    /** 16 x 16 segments by default, so the meshlet pipelines get a few hundred triangles to cluster. */
    segments?: number;
}

export function createTestScene(options: TestSceneOptions = {}): TestScene {
    const scene = new Scene();
    const segments = options.segments ?? 16;
    const plane = new Mesh(
        new PlaneGeometry(8, 8, segments, segments),
        options.material ?? new MeshLambertNodeMaterial({ color: 0xffffff }),
    );
    plane.rotation.x = -Math.PI / 2;
    scene.add(plane);

    // Straight down from 6 units. At that height the view is 5.6 units wide, so
    // the middle of the image sees the origin (inside the volume) and the
    // corners see about +-2.3 (outside the 3-wide volume).
    const camera = new PerspectiveCamera(50, 1, 0.1, 50);
    camera.position.set(0, 6, 0);
    camera.up.set(0, 0, -1);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();

    const target = new RenderTarget(RENDER_SIZE, RENDER_SIZE, { type: FloatType, depthBuffer: true });
    return { scene, camera, plane, target };
}

/** Renders the scene into the float target. */
export function renderToTarget(renderer: WebGPURenderer, test: TestScene): void {
    renderer.setRenderTarget(test.target);
    renderer.render(test.scene, test.camera);
    renderer.setRenderTarget(null);
}

export interface PixelBlock {
    /** Mean rgb over the block. */
    mean: [number, number, number];
    /** Largest deviation from the mean in any channel: 0 means a flat block. */
    spread: number;
}

/** Mean colour of a square block of a float render target. */
export async function readBlock(
    renderer: WebGPURenderer,
    target: RenderTarget,
    centreX: number,
    centreY: number,
    halfSize = 4,
    faceIndex = 0,
): Promise<PixelBlock> {
    const size = halfSize * 2;
    const pixels = (await renderer.readRenderTargetPixelsAsync(
        target,
        centreX - halfSize,
        centreY - halfSize,
        size,
        size,
        0,
        faceIndex,
    )) as Float32Array;
    const sum = [0, 0, 0];
    for (let index = 0; index < size * size; index++) {
        for (let channel = 0; channel < 3; channel++) sum[channel] += pixels[index * 4 + channel];
    }
    const mean = sum.map((total) => total / (size * size)) as [number, number, number];
    let spread = 0;
    for (let index = 0; index < size * size; index++) {
        for (let channel = 0; channel < 3; channel++) {
            spread = Math.max(spread, Math.abs(pixels[index * 4 + channel] - mean[channel]));
        }
    }
    return { mean, spread };
}

/** The centre of the image (the origin of the plane) and a corner (outside the volume). */
export async function readCentreAndCorner(
    renderer: WebGPURenderer,
    target: RenderTarget,
): Promise<{ centre: PixelBlock; corner: PixelBlock }> {
    const centre = await readBlock(renderer, target, RENDER_SIZE / 2, RENDER_SIZE / 2);
    const corner = await readBlock(renderer, target, 14, 14);
    return { centre, corner };
}

/** Outgoing radiance of a white Lambert surface lit by irradiance E. */
export function lambertRadiance(irradiance: readonly number[], intensity = 1, albedo = 1): [number, number, number] {
    return irradiance.map((value) => (value * intensity * albedo) / Math.PI) as [number, number, number];
}

export function withinTolerance(actual: number, expected: number): boolean {
    return Math.abs(actual - expected) <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * Math.abs(expected);
}

export function formatRgb(value: readonly number[]): string {
    return `(${value.map((channel) => channel.toFixed(4)).join(', ')})`;
}

/** Fails with a message naming the check, the measured and expected rgb. */
export function assertRgbClose(
    what: string,
    actual: readonly number[],
    expected: readonly number[],
): { what: string; actual: number[]; expected: number[]; ok: true } {
    const ok = actual.every((value, channel) => withinTolerance(value, expected[channel]));
    assert(ok, `${what}: read ${formatRgb(actual)}, expected ${formatRgb(expected)} (tolerance ${RELATIVE_TOLERANCE * 100}% + ${ABSOLUTE_TOLERANCE})`);
    return { what, actual: [...actual], expected: [...expected], ok: true };
}
