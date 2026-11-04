/**
 * Turns a decoded sky file into a `GPUTexture` on the host's device, for
 * `setEnvironment({ type: 'texture', layout: 'equirect', texture })`.
 * Plain WebGPU: no engine.
 *
 * The texture is `rgba16float`: the file's half floats are copied as they are
 * (no rounding on the way), alpha is 1, and `rgba16float` is filterable on
 * every WebGPU device (`rgba32float` needs the optional `float32-filterable`).
 * Row 0 of the texture is the zenith row of the file. {@link SKY_EQUIRECT_WGSL}
 * is the matching lookup; `equirect.ts` describes the layout.
 */

import type { HalfRgbImage } from './exr-half.ts';

/** binary16 1.0, the alpha of every texel. */
const HALF_ONE = 0x3c00;

/** Texture format of the sky. */
export const SKY_TEXTURE_FORMAT: GPUTextureFormat = 'rgba16float';

/** RGB halves to RGBA halves (alpha 1), as `rgba16float` wants them. */
export function skyImageToRgbaHalves(image: HalfRgbImage): Uint16Array {
    const texelCount = image.width * image.height;
    const rgba = new Uint16Array(texelCount * 4);
    for (let texel = 0; texel < texelCount; texel++) {
        rgba[texel * 4] = image.halves[texel * 3];
        rgba[texel * 4 + 1] = image.halves[texel * 3 + 1];
        rgba[texel * 4 + 2] = image.halves[texel * 3 + 2];
        rgba[texel * 4 + 3] = HALF_ONE;
    }
    return rgba;
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
 * zenith row. The same mapping as three.js's `equirectUV` and as
 * `directionToEquirectUv` in `equirect.ts`.
 */
export const SKY_EQUIRECT_WGSL = /* wgsl */ `
const SKY_INV_TWO_PI: f32 = 0.15915494309189535;
const SKY_INV_PI: f32 = 0.3183098861837907;

fn skyEquirectUv(direction: vec3f) -> vec2f {
    let d = normalize(direction);
    let u = atan2(d.z, d.x) * SKY_INV_TWO_PI + 0.5;
    let v = acos(clamp(d.y, -1.0, 1.0)) * SKY_INV_PI;
    return vec2f(u, v);
}
`;
