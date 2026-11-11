/**
 * The radiance cache's image-space GI denoiser (RC4.3): GI-1's `ReprojectGI`
 * and two `FilterGI` passes from `@pmndrs/denoiser-gi1`, run where Capsaicin
 * runs them, after `InterpolateScreenProbes` and before the resolve
 * (`GI1::render`, gi1.cpp:2882-2927). Capsaicin has no switch for it; the
 * technique's `useGIDenoiser` (default on) is ours, for comparisons.
 *
 * The denoiser is a separate package that knows nothing of the platform, so
 * this file is the whole seam: it hands over the interpolated irradiance
 * (with Capsaicin's denoiser hint in alpha, as `InterpolateScreenProbes`
 * writes it), the host's screen buffers and this frame's and the previous
 * frame's camera. The denoiser keeps its own history (Capsaicin's two colour
 * buffers, the filtered result fed back) and its own copy of last frame's
 * depth and normals (register D68).
 */

import { GI1Denoiser, type FrameCamera as DenoiserCamera } from '@pmndrs/denoiser-gi1';
import type { FrameCamera, FrameScreen } from '../../platform/contract/frame.ts';
import type { TechniqueContext } from '../../platform/contract/technique.ts';

/** Its passes are timed under this name: `radiance-cache/gi-denoiser/ReprojectGI` and `.../FilterGI`. */
export const GI_DENOISER_LABEL = 'radiance-cache/gi-denoiser';

export function createGIDenoiser(context: TechniqueContext): Promise<GI1Denoiser> {
    return GI1Denoiser.create(context.device, { label: GI_DENOISER_LABEL, timer: context.timer });
}

/**
 * Records this frame's denoise of `irradiance` (the interpolation's output)
 * and returns the denoised irradiance, which the shading module then reads.
 * `previousCamera` is the camera of the frame before (this frame's after a
 * history reset).
 */
export function encodeGIDenoiser(
    denoiser: GI1Denoiser,
    encoder: GPUCommandEncoder,
    screen: FrameScreen,
    camera: FrameCamera,
    previousCamera: FrameCamera,
    irradiance: GPUTexture,
): GPUTexture {
    const motion = screen.velocity;
    if (!motion) throw new Error('@pmndrs/ddgi: the radiance cache denoises with FrameScreen.velocity, which this screen has not');
    denoiser.configure({ width: screen.width, height: screen.height });
    return denoiser.encode(
        encoder,
        { radiance: irradiance },
        { depth: screen.depth, normal: screen.normal, motion, normalEncoding: screen.normalEncoding === 'unorm' ? 'unorm' : 'signed' },
        denoiserCamera(camera, previousCamera, screen),
    );
}

/**
 * The platform's camera as the denoiser takes it: the same matrices, and
 * the jitter in pixels (y down) where `FrameCamera.jitter` is an NDC offset
 * (y up). The denoiser remembers the previous frame's jitter itself.
 */
export function denoiserCamera(camera: FrameCamera, previousCamera: FrameCamera, screen: Pick<FrameScreen, 'width' | 'height' | 'reversedDepth'>): DenoiserCamera {
    const [jitterX, jitterY] = camera.jitter ?? [0, 0];
    return {
        view: camera.view,
        projection: camera.projection,
        prevView: previousCamera.view,
        prevProjection: previousCamera.projection,
        jitter: [(jitterX * screen.width) / 2, (-jitterY * screen.height) / 2],
        near: camera.near,
        far: camera.far,
        position: camera.position,
        reversedDepth: screen.reversedDepth === true,
    };
}
