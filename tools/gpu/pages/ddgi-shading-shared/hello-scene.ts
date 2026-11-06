/**
 * Shared by the DDGI shading pages (T1.6): a platform with a manifest scene
 * described by the hello example's own `describeScene`, DDGI on the
 * manifest's volume, and the hello example's renderer, with the core loaded
 * straight from source (the hello app itself imports the built package name,
 * which the test site doesn't alias).
 *
 * What main.ts in examples/webgpu/hello does, minus the canvas loop.
 *
 * Not a page itself (no index.html).
 */

import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import type { DDGISettingsInput } from '../../../../packages/ddgi-core/src/techniques/ddgi/settings.ts';
import { describeScene, loadManifest, type DescribedScene, type SceneOptions } from '../../../../examples/webgpu/hello/src/scene.ts';
import { runFrame } from '../ddgi-shared/harness.ts';

export interface HelloScene {
    readonly platform: GIPlatformImpl;
    readonly ddgi: DDGITechnique;
    readonly scene: DescribedScene;
}

/** Platform, scene and DDGI, prepared, with the mesh trees built and placed. */
export async function createHelloScene(
    device: GPUDevice,
    sceneId: string,
    options: SceneOptions & { readonly gpuTiming?: boolean; readonly settings?: Partial<DDGISettingsInput> } = {},
): Promise<HelloScene> {
    const manifest = loadManifest(sceneId);
    const platform = new GIPlatformImpl(device, { label: `hello ${sceneId}`, gpuTiming: options.gpuTiming });
    const scene = describeScene(platform.scene, manifest, options);
    const volume = { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing };
    const ddgi = platform.addTechnique(new DDGITechnique({ warmUpPasses: 0, ...options.settings, volume }));
    platform.setOutput(ddgi);
    await platform.prepare();
    // A frame with no probe updates places the scene; then wait for the mesh trees and place them.
    runFrame(device, platform, 0);
    await platform.rays.whenBuilt();
    runFrame(device, platform, 0);
    return { platform, ddgi, scene };
}

/** An rgba32float texture read back as rgb floats, row 0 at the top. */
export async function readRgba32FloatAsRgb(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
    const bytesPerTexel = 16;
    const rowBytes = texture.width * bytesPerTexel;
    const paddedRowBytes = Math.ceil(rowBytes / 256) * 256;
    const buffer = device.createBuffer({ label: 'hello readback', size: paddedRowBytes * texture.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder({ label: 'hello readback' });
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: paddedRowBytes }, { width: texture.width, height: texture.height });
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const padded = new Float32Array(buffer.getMappedRange());
    const rgb = new Float32Array(texture.width * texture.height * 3);
    const floatsPerPaddedRow = paddedRowBytes / 4;
    for (let row = 0; row < texture.height; row++) {
        for (let column = 0; column < texture.width; column++) {
            const source = row * floatsPerPaddedRow + column * 4;
            const target = (row * texture.width + column) * 3;
            rgb[target] = padded[source];
            rgb[target + 1] = padded[source + 1];
            rgb[target + 2] = padded[source + 2];
        }
    }
    buffer.unmap();
    buffer.destroy();
    return rgb;
}
