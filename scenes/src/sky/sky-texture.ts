/**
 * Turns a decoded sky file into a `GPUTexture` on the host's device, for
 * `setEnvironment({ type: 'texture', layout: 'equirect', texture })`.
 * Plain WebGPU: no engine.
 *
 * The texture is `rgba16float`: the file's half floats are copied as they are
 * (no rounding on the way), alpha is 1, and `rgba16float` is filterable on
 * every WebGPU device (`rgba32float` needs the optional `float32-filterable`).
 *
 * Rows are in the platform's equirect texture order (`contract/scene.ts`,
 * T1.4): the first row in memory is the nadir and the last row the zenith,
 * the reverse of the file's image order. That is how three.js stores its own
 * equirect textures, so its `equirectUV` reads this texture the right way up
 * too. {@link SKY_EQUIRECT_WGSL} is the matching lookup.
 */

import { halfImageToTextureRgba } from '../../../packages/ddgi-core/src/platform/sky/half-float.ts';
import type { HalfRgbImage } from './exr-half.ts';

/** Texture format of the sky. */
export const SKY_TEXTURE_FORMAT: GPUTextureFormat = 'rgba16float';

/** RGB halves to RGBA halves (alpha 1) in texture row order, as the `rgba16float` texture wants them. */
export function skyImageToRgbaHalves(image: HalfRgbImage): Uint16Array {
    return halfImageToTextureRgba(image);
}

/**
 * Creates and fills the sky texture. The caller owns it (and destroys it).
 * @param extraUsage more usage flags than TEXTURE_BINDING | COPY_DST, if the host needs them (COPY_SRC for a readback, say)
 */
export function createSkyTexture(device: GPUDevice, image: HalfRgbImage, label = 'shared sky', extraUsage: GPUTextureUsageFlags = 0): GPUTexture {
    const texture = device.createTexture({
        label,
        size: { width: image.width, height: image.height },
        format: SKY_TEXTURE_FORMAT,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | extraUsage,
    });
    const bytesPerTexel = 8;
    device.queue.writeTexture(
        { texture },
        skyImageToRgbaHalves(image) as Uint16Array<ArrayBuffer>,
        { bytesPerRow: image.width * bytesPerTexel, rowsPerImage: image.height },
        { width: image.width, height: image.height },
    );
    return texture;
}

/**
 * WGSL: world direction (+Y up, need not be normalised) to the sky texture's
 * coordinates. u wraps at -X (use `repeat` addressing on u); v is 0 at the
 * nadir (the first row in memory) and 1 at the zenith. Exactly three.js's
 * `equirectUV`, and the platform's equirect contract.
 */
export const SKY_EQUIRECT_WGSL = /* wgsl */ `
const SKY_INV_TWO_PI: f32 = 0.15915494309189535;
const SKY_INV_PI: f32 = 0.3183098861837907;

fn skyEquirectUv(direction: vec3f) -> vec2f {
    let d = normalize(direction);
    let u = atan2(d.z, d.x) * SKY_INV_TWO_PI + 0.5;
    let v = asin(clamp(d.y, -1.0, 1.0)) * SKY_INV_PI + 0.5;
    return vec2f(u, v);
}
`;
