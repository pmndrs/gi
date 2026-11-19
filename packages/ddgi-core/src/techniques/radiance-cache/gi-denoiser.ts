/**
 * The radiance cache's image-space GI denoiser (RC4.3): AMD GI-1.0's `ReprojectGI` and
 * `FilterGI` × 2 (Capsaicin v1.3, gi1.comp:3997-4154, dispatched at gi1.cpp:2881-2926), run
 * on `InterpolateScreenProbes`' irradiance before the composite reads it, as Capsaicin runs
 * them before `ResolveGI1`.
 *
 * The denoiser itself lives in the sibling library `pmndrs/denoiser` (`Gi1Denoiser` in
 * `@pmndrs/denoiser-ffx`, next to its other AMD ports; DECISIONS 2026-11-19). This file only
 * maps the platform's frame onto its input contract. During development the package is linked
 * from a local clone of the denoiser repo (root package.json, `link:`); it is not published,
 * so nothing here is part of `@pmndrs/gi` yet.
 *
 * What it reads: the irradiance with Capsaicin's denoiser hint in alpha (screen-irradiance.ts),
 * `FrameScreen`'s depth, normal and velocity, and the platform's screen history for the
 * previous depth and normals (`GpuScreenHistory`, the copy `ReprojectScreenProbes` reads too),
 * and the frame's scene changes (`SceneChanges`, as the probes' running means take them, D46):
 * inside a changed region, and everywhere for a global change (the sun, the sky), the
 * denoiser keeps a short history for a few frames (its soft reset, register D50).
 */

import { Gi1Denoiser } from '@pmndrs/denoiser-ffx';
import type { ScreenHistoryTextures } from '../../platform/contract/compute.ts';
import type { FrameCamera, FrameScreen } from '../../platform/contract/frame.ts';
import type { HistoryChanges } from './screen-probe-history.ts';

/** The camera the denoiser saw last frame: its `prevView` and `prevProjection` next frame. */
interface PreviousCamera {
    readonly view: ArrayLike<number>;
    readonly projection: ArrayLike<number>;
}

export class RadianceCacheDenoiser {
    private readonly denoiser: Gi1Denoiser;
    private width = 0;
    private height = 0;
    private previousCamera: PreviousCamera | null = null;
    private latestOutput: GPUTexture | null = null;

    constructor(device: GPUDevice) {
        // Capsaicin's InterpolateScreenProbes writes its denoiser hint in alpha (gi1.comp:1636-1638).
        this.denoiser = new Gi1Denoiser(device, { hintInAlpha: true, timestamps: false });
    }

    /** The denoised irradiance of the latest frame (rgba16float, rgb E), or null before the first. */
    get output(): GPUTexture | null {
        return this.latestOutput;
    }

    /** 48 bytes a pixel: the platform's screen history stands in for the denoiser's own previous depth and normals. */
    get memoryBytes(): number {
        return this.denoiser.memoryBytes;
    }

    /** Forget the history at the next frame (a cut, or the screen probes' history reset). */
    reset(): void {
        this.denoiser.resetHistory();
        this.previousCamera = null;
    }

    /**
     * Records the denoiser's passes into `encoder`, after `InterpolateScreenProbes` and before
     * the screen history copies this frame's depth and normals (it reads last frame's).
     */
    encode(
        encoder: GPUCommandEncoder,
        irradiance: GPUTexture,
        screen: FrameScreen,
        history: ScreenHistoryTextures,
        camera: FrameCamera,
        changes: HistoryChanges,
        options: { readonly historyClamp: 'neighbourhood' | 'none'; readonly changeReset: boolean },
    ): GPUTexture {
        this.denoiser.historyClamp = options.historyClamp;
        this.denoiser.changeReset = options.changeReset;
        if (!screen.velocity) throw new Error('@pmndrs/gi: the GI denoiser reads FrameScreen.velocity');
        if (screen.width !== this.width || screen.height !== this.height) {
            this.denoiser.configure({ width: screen.width, height: screen.height });
            this.width = screen.width;
            this.height = screen.height;
            this.previousCamera = null;
        }
        // No previous frame of this size in the history (first frame, resize): start over.
        if (!history.valid || !this.previousCamera) this.denoiser.resetHistory();
        this.denoiser.normalEncoding = screen.normalEncoding === 'unorm' ? 'unorm' : 'signed';
        const previous = this.previousCamera ?? camera;
        // The platform's jitter is an NDC offset; the denoiser takes @pmndrs/upscaler's render
        // pixels, y down (FrameCamera.jitter: NDC = (-2x / width, +2y / height)).
        const jitter = camera.jitter ?? [0, 0];
        const output = this.denoiser.encode(
            encoder,
            {
                radiance: irradiance,
                changedRegions: changes.everything ? 'everything' : changes.boxes.length > 0 ? changes.boxes.map((box) => ({ min: box.min, max: box.max })) : undefined,
            },
            { depth: screen.depth, normal: screen.normal, motion: screen.velocity, previousDepth: history.depth, previousNormal: history.normal },
            {
                view: camera.view,
                projection: camera.projection,
                prevView: previous.view,
                prevProjection: previous.projection,
                jitter: [(-jitter[0] * screen.width) / 2, (jitter[1] * screen.height) / 2],
                near: camera.near,
                far: camera.far,
                position: [camera.position[0], camera.position[1], camera.position[2]],
                reversedDepth: screen.reversedDepth ?? false,
            },
        );
        this.previousCamera = { view: Array.from(camera.view), projection: Array.from(camera.projection) };
        this.latestOutput = output;
        return output;
    }

    dispose(): void {
        this.denoiser.dispose();
        this.latestOutput = null;
    }
}
