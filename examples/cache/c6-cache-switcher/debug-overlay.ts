/**
 * The radiance cache's own debug views (`cache.debugViews`: the probe atlas,
 * the probe mask and the probes' SH), drawn in three.js. The technique draws
 * a view engine-free into a screen-sized `rgba8unorm` storage texture
 * (`cache.encodeDebugView`); three shows that texture full screen, wrapped
 * with `wrapGpuTexture`, so nothing is read back.
 *
 * The view is drawn after the frame's screen stage and shown by the next
 * frame, so it is one frame behind the picture. Its colours are already
 * tone-mapped and gamma-encoded, so the pipeline shows it without its own
 * colour transform (main.ts).
 */

import type { Node, TextureNode } from 'three/webgpu';
import { texture, uv } from 'three/tsl';
import type { RadianceCacheDebugViewId, RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';
import { wrapGpuTexture } from '@pmndrs/gi/three';

const USAGE = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;

export class CacheDebugOverlay {
    /** The view to draw, or null for none. */
    view: RadianceCacheDebugViewId | null = null;
    /** The latest view, full screen: put it in the pipeline's output node. */
    readonly node: Node;
    private readonly picture: TextureNode;
    private target: GPUTexture;

    constructor(
        private readonly device: GPUDevice,
        private readonly cache: RadianceCacheTechnique,
    ) {
        this.target = device.createTexture({ label: 'c6: debug view', size: [1, 1], format: 'rgba8unorm', usage: USAGE });
        this.picture = texture(wrapGpuTexture(this.target, { dimension: '2d', sampleType: 'float' }));
        this.node = this.picture.sample(uv());
    }

    /** Draws the view of this frame's probes. Call after the frame renders. */
    draw(): void {
        const layout = this.cache.probes?.layout;
        if (this.view === null || !layout) return;
        if (this.target.width !== layout.width || this.target.height !== layout.height) {
            // A new screen size: a new texture, which three binds at its next draw.
            this.target.destroy();
            this.target = this.device.createTexture({ label: 'c6: debug view', size: [layout.width, layout.height], format: 'rgba8unorm', usage: USAGE });
            this.picture.value = wrapGpuTexture(this.target, { dimension: '2d', sampleType: 'float' });
        }
        const encoder = this.device.createCommandEncoder({ label: 'c6: debug view' });
        this.cache.encodeDebugView(encoder, this.view, this.target);
        this.device.queue.submit([encoder.finish()]);
    }
}
