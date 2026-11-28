import { upscale } from '@pmndrs/upscaler';
import { traa } from 'three/addons/tsl/display/TRAANode.js';
import { convertToTexture, Fn, OnBeforeObjectUpdate, velocity } from 'three/tsl';
import { RenderPipeline, type Node, type PerspectiveCamera, type Scene, type TextureNode, type WebGPURenderer } from 'three/webgpu';

export type ReconstructionMode = 'native' | 'taa' | 'upscale';

export interface ReconstructionOptions {
    mode: ReconstructionMode;
    /** Final drawing-buffer dimensions in pixels. The gallery uses pixel ratio 1. */
    width: number;
    height: number;
    /** Display/render ratio for the upscaler. Both native modes render at display resolution. */
    ratio: number;
    /** RCAS amount from 0 (off) to 1. */
    sharpness: number;
}

/** Presentation inputs supplied by either gallery backend's Three adapter. */
export interface CascadeSource {
    velocity: boolean;
    renderSize: { x: number; y: number } | null;
    getCompositeNode(): Node;
    getTextureNode(name: 'depth' | 'velocity'): TextureNode;
    update(renderer: WebGPURenderer, scene: Scene, camera: PerspectiveCamera): void;
}

/** S5's render graph, with the RC G-buffer rendered under the active resolve's camera jitter. */
export function createReconstruction(renderer: WebGPURenderer, scene: Scene, camera: PerspectiveCamera, rc: CascadeSource) {
    const pipeline = new RenderPipeline(renderer);
    let mode: ReconstructionMode = 'native';
    let displayWidth = 0, displayHeight = 0, renderWidth = 0, renderHeight = 0;
    let ratio = 1, sharpness = 0;
    let lastComposite: Node | null = null;
    let taa: ReturnType<typeof traa> | null = null;
    let fsr: ReturnType<typeof upscale> | null = null;
    let color: ReturnType<typeof convertToTexture> | null = null;
    let needsCascadeUpdate = false;
    let updates = 0, historyResets = 0;
    let jitterX = 0, jitterY = 0;

    function releaseGraph() {
        taa?.beautyNode.dispose();
        taa?.dispose();
        fsr?.dispose();
        color?.dispose();
        taa = null;
        fsr = null;
        color = null;
        lastComposite = null;
        // The temporal upscaler pins three's velocity singleton to its unjittered matrix.
        // Release ownership before switching to native or TRAA.
        velocity.setProjectionMatrix(null);
    }

    function updateCascade() {
        if (!needsCascadeUpdate) return;
        needsCascadeUpdate = false;
        jitterX = camera.view?.enabled ? camera.view.offsetX : 0;
        jitterY = camera.view?.enabled ? camera.view.offsetY : 0;
        rc.update(renderer, scene, camera);
        updates++;
    }

    function build(composite: Node) {
        releaseGraph();
        lastComposite = composite;
        if (mode === 'native') {
            pipeline.outputNode = composite;
        } else {
            // Run inside the beauty draw, after the pipeline's jitter callback. The
            // guard also covers multiple uses of the node in one presentation.
            const frame = Fn(() => {
                OnBeforeObjectUpdate(updateCascade);
                return composite;
            })();
            const depth = rc.getTextureNode('depth');
            const motion = rc.getTextureNode('velocity');
            if (mode === 'taa') {
                // r187 TRAA adopts the input depth format when building its history.
                taa = traa(frame, depth, motion, camera);
                pipeline.outputNode = taa;
            } else {
                color = convertToTexture(frame, renderWidth, renderHeight);
                fsr = upscale(color, depth, motion, camera, { path: 'temporal', ratio, jitter: true });
                pipeline.outputNode = fsr;
            }
        }
        pipeline.needsUpdate = true;
    }

    return {
        configure(options: ReconstructionOptions) {
            const nextRatio = options.mode === 'upscale' ? options.ratio : 1;
            if (!(Number.isFinite(nextRatio) && nextRatio >= 1)) throw new RangeError('Upscale ratio must be finite and at least 1');
            if (![options.width, options.height].every(n => Number.isInteger(n) && n > 0)) throw new RangeError('Display dimensions must be positive integers');
            if (!(options.sharpness >= 0 && options.sharpness <= 1)) throw new RangeError('Sharpness must be in [0, 1]');
            const resized = displayWidth !== options.width || displayHeight !== options.height;
            if (resized || mode !== options.mode || ratio !== nextRatio) lastComposite = null;
            mode = options.mode;
            ratio = nextRatio;
            sharpness = options.sharpness;
            displayWidth = options.width;
            displayHeight = options.height;
            renderWidth = Math.max(1, Math.floor(displayWidth / ratio));
            renderHeight = Math.max(1, Math.floor(displayHeight / ratio));
            if (resized) {
                renderer.setSize(displayWidth, displayHeight, false);
                camera.aspect = displayWidth / displayHeight;
                camera.updateProjectionMatrix();
            }
            rc.velocity = mode !== 'native';
            rc.renderSize = mode === 'upscale' ? { x: renderWidth, y: renderHeight } : null;
        },
        render() {
            if (!displayWidth) throw new Error('Configure the RC presentation size before rendering');
            needsCascadeUpdate = true;
            if (mode === 'native') updateCascade();
            const composite = rc.getCompositeNode();
            if (composite !== lastComposite) build(composite);
            if (fsr?.upscaler) fsr.upscaler.settings.sharpness = sharpness;
            pipeline.render();
        },
        resetHistory() {
            fsr?.upscaler?.resetHistory();
            // TRAA seeds history from the current image when its dimensions change.
            // setSize is public on r187, but omitted from @types/three.
            if (taa) (taa as typeof taa & { setSize(w: number, h: number): void }).setSize(renderWidth === 1 ? 2 : 1, 1);
            if (taa || fsr?.upscaler) historyResets++;
        },
        get ready() { return mode !== 'upscale' || (fsr?.upscaler?.isReady ?? false); },
        async prepare() {
            if (mode !== 'upscale') return;
            if (!fsr?.upscaler) throw new Error('Render the first gallery frame before preparing temporal reconstruction');
            await fsr.upscaler.prepare();
        },
        get renderWidth() { return renderWidth; },
        get renderHeight() { return renderHeight; },
        get updates() { return updates; },
        get historyResets() { return historyResets; },
        get jitterX() { return jitterX; },
        get jitterY() { return jitterY; },
        dispose() {
            releaseGraph();
            pipeline.dispose();
        },
    };
}
