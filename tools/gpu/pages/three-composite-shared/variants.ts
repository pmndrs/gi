/**
 * Measurement-only variants of the GI composite (T1.14 cost study, for
 * Dennis's decision on GTAO and the upscaler). Nothing here is product: the
 * defaults and exports of `packages/ddgi-three` don't change.
 *
 * - **Through `@pmndrs/upscaler`** (Dennis: "I would plan on using the
 *   upscaler with whatever we do, AO included"): the upscaler's documented
 *   composable path (its `UpscalerNode.ts`, `upscale()`): the scene pass at
 *   1 / ratio of the output with `velocity` in its MRT, as `upscaleScene`
 *   builds its own pass, plus the composite's `giAlbedo` and normal; the
 *   composite (and its own GTAO, which sizes itself from the pass) at that reduced resolution, made a texture by
 *   three's `rtt()` at the same scale; then `upscale(color, depth, velocity,
 *   camera, { quality })` to the output. The viewer's toggle uses
 *   `upscaleScene` (apps/viewer `scene-pass-technique.ts`), which can't take
 *   the composite, so this is the same package's next step down.
 * - **Cheaper occlusion on GI:** GTAO with its own temporal pattern plus an
 *   exponential average of its output over frames (static camera only:
 *   nothing reprojects the history, see `AccumulatedOcclusion`); three's
 *   SSAO (`SSAONode`, its own blur, half resolution by default).
 *
 * The upscaler is a devDependency of the workspace root, for these pages.
 */

import { HalfFloatType, Node, NodeMaterial, NodeUpdateType, QuadMesh, RenderTarget, type Camera, type NodeFrame, type PassNode, type Texture, type TextureNode, type WebGPURenderer } from 'three/webgpu';
import { mix, mrt, normalView, output, packNormalToRGB, pass, passTexture, rtt, sample, texture, uniform, unpackRGBToNormal, uv, velocity, vec4 } from 'three/tsl';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import type GTAONode from 'three/addons/tsl/display/GTAONode.js';
import { ssao } from 'three/addons/tsl/display/SSAONode.js';
import { QualityMode, getQualityModeRatio, upscale, upscaleScene, type UpscalerNode } from '@pmndrs/upscaler';
import { DDGICompositeNode, giAlbedo, DEFAULT_GI_OCCLUSION_RADIUS, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT, type CompositeGISource, type GIOcclusionSource } from '../../../../packages/ddgi-three/src/composite.ts';

export { QualityMode };

/** The upscaler presets the cost study uses, by name, with the render scale they imply (1 / ratio). */
export const UPSCALE_PRESETS = {
    quality: QualityMode.Quality,
    performance: QualityMode.Performance,
} as const;
export type UpscalePreset = keyof typeof UPSCALE_PRESETS;

export function renderScale(preset: UpscalePreset): number {
    return 1 / getQualityModeRatio(UPSCALE_PRESETS[preset]);
}

/** three's GTAO on a pass's depth and packed normal (the composite's own way, composite.ts `makeOcclusion`). */
export function gtaoOn(scenePass: PassNode, settings: { radius?: number; resolutionScale?: number; samples?: number } = {}): GTAONode {
    const normalTexture = scenePass.getTextureNode(GI_NORMAL_OUTPUT);
    const viewNormal = sample((coordinate) => unpackRGBToNormal(normalTexture.sample(coordinate).rgb));
    const node = ao(scenePass.getTextureNode('depth'), viewNormal, scenePass.camera);
    node.radius.value = settings.radius ?? DEFAULT_GI_OCCLUSION_RADIUS;
    if (settings.resolutionScale !== undefined) node.resolutionScale = settings.resolutionScale;
    if (settings.samples !== undefined) node.samples.value = settings.samples;
    return node;
}

/** three's SSAO on a pass's depth and packed normal: radius 1 m, 16 samples, its default half resolution and blur. */
export function ssaoOn(scenePass: PassNode, settings: { radius?: number; resolutionScale?: number } = {}): GIOcclusionSource {
    const normalTexture = scenePass.getTextureNode(GI_NORMAL_OUTPUT);
    const viewNormal = sample((coordinate) => unpackRGBToNormal(normalTexture.sample(coordinate).rgb));
    const node = ssao(scenePass.getTextureNode('depth'), viewNormal, scenePass.camera);
    node.radius.value = settings.radius ?? DEFAULT_GI_OCCLUSION_RADIUS;
    if (settings.resolutionScale !== undefined) node.resolutionScale = settings.resolutionScale;
    return node as unknown as GIOcclusionSource;
}

/**
 * GTAO averaged over frames: GTAO's own temporal pattern (`useTemporalFiltering`,
 * a new rotation and offset each frame) and an exponential average of its output,
 * reaching a plain mean of the last `frames` frames. **Static camera only:**
 * nothing reprojects the history, so a moving camera would smear it (three's
 * `TemporalReprojectNode` is built around SSR's inputs, and TRAA works on the
 * whole image, not the GI term alone).
 *
 * Built like GTAONode: `getTextureNode()` is a pass texture of this node, so
 * reading it builds this node, whose setup builds GTAO's texture first; GTAO
 * therefore draws before this node averages it, every frame.
 */
export class AccumulatedOcclusion extends Node {
    static get type(): string {
        return 'AccumulatedOcclusion';
    }

    private readonly targets: [RenderTarget, RenderTarget];
    private readonly quad: QuadMesh;
    private readonly history: TextureNode;
    private readonly blend = uniform(1);
    private readonly output: TextureNode;
    private count = 0;
    private current = 0;

    constructor(
        readonly source: GTAONode,
        readonly frames: number,
    ) {
        super('vec4');
        source.useTemporalFiltering = true;
        this.updateBeforeType = NodeUpdateType.FRAME;
        this.targets = [new RenderTarget(1, 1, { type: HalfFloatType, depthBuffer: false }), new RenderTarget(1, 1, { type: HalfFloatType, depthBuffer: false })];
        this.history = texture(this.targets[1].texture) as unknown as TextureNode;
        const material = new NodeMaterial();
        material.name = 'accumulated occlusion';
        this.quad = new QuadMesh(material);
        this.output = passTexture(this as never, this.targets[0].texture as Texture) as unknown as TextureNode;
    }

    getTextureNode(): TextureNode {
        return this.output;
    }

    updateBefore(frame: NodeFrame): boolean | undefined {
        const renderer = frame.renderer as unknown as WebGPURenderer;
        const source = (this.source.getTextureNode() as unknown as { value: Texture }).value;
        const image = source.image as { width?: number; height?: number } | undefined;
        const width = image?.width ?? 1;
        const height = image?.height ?? 1;
        for (const target of this.targets) target.setSize(width, height);
        // A plain mean of the frames so far, up to `frames`, then an exponential average of that length.
        this.count = Math.min(this.count + 1, this.frames);
        this.blend.value = 1 / this.count;
        const write = this.targets[this.current];
        const read = this.targets[1 - this.current];
        (this.history as unknown as { value: Texture }).value = read.texture;
        const previous = renderer.getRenderTarget();
        renderer.setRenderTarget(write);
        this.quad.render(renderer);
        renderer.setRenderTarget(previous);
        (this.output as unknown as { value: Texture }).value = write.texture;
        this.current = 1 - this.current;
        return undefined;
    }

    setup(): Node {
        const current = this.source.getTextureNode().sample(uv()).r;
        (this.quad.material as NodeMaterial).fragmentNode = vec4(mix(this.history.sample(uv()).r, current, this.blend), 0, 0, 1) as unknown as Node;
        return this.output as unknown as Node;
    }
}

/** A composite scene pass at `scale` of the output: the composite's MRT, plus `velocity` when an upscaler follows. */
export function compositePassAt(scene: Parameters<typeof pass>[0], camera: Camera, scale: number, withVelocity: boolean): PassNode {
    const scenePass = pass(scene, camera);
    scenePass.setMRT(
        mrt(
            withVelocity
                ? { output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), velocity }
                : { output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView) },
        ),
    );
    if (scale !== 1) scenePass.setResolutionScale(scale);
    return scenePass;
}

export interface UpscaledComposite {
    readonly scenePass: PassNode;
    readonly composite: DDGICompositeNode;
    /** The pipeline's output: the upscaled colour (or GI term). */
    readonly node: UpscalerNode;
}

/**
 * The composite through the upscaler (file comment). `occlusion` is the
 * composite's own GTAO, its `resolutionScale` relative to the reduced pass (the
 * composite's GTAO sizes itself from the pass), or false for none. `giTermOnly`
 * feeds the upscaler the composite minus the pass's own colour, for the
 * accuracy check of the GI term at output resolution.
 */
export function upscaledComposite(
    scene: Parameters<typeof pass>[0],
    camera: Camera,
    source: CompositeGISource,
    preset: UpscalePreset,
    options: { readonly occlusion: false | { readonly resolutionScale: number; readonly samples?: number; readonly radius?: number }; readonly giTermOnly?: boolean },
): UpscaledComposite {
    const scale = renderScale(preset);
    const scenePass = compositePassAt(scene, camera, scale, true);
    const composite = new DDGICompositeNode(scenePass, source, { occlusion: options.occlusion === false ? false : { ...options.occlusion } });
    const passColor = scenePass.getTextureNode('output');
    const color = options.giTermOnly ? vec4((composite as unknown as ReturnType<typeof vec4>).rgb.sub(passColor.rgb), 1) : composite;
    const atRenderScale = rtt(color as never, null, null, { type: HalfFloatType, resolutionScale: scale } as never);
    const node = upscale(atRenderScale as never, scenePass.getTextureNode('depth') as never, scenePass.getTextureNode('velocity') as never, camera as never, {
        quality: UPSCALE_PRESETS[preset],
        gpuTiming: true,
    });
    return { scenePass, composite, node };
}

/** The scene through the upscaler with no GI, as the viewer's toggle does it (`upscaleScene`). */
export function upscaledScene(scene: Parameters<typeof pass>[0], camera: Camera, preset: UpscalePreset): UpscalerNode {
    return upscaleScene(scene, camera as never, { quality: UPSCALE_PRESETS[preset], gpuTiming: true });
}

/** The upscaler's own GPU time in its latest timed frame (ms), or null before it has one. */
export function upscalerMs(node: UpscalerNode): number | null {
    const timings = node.upscaler?.gpuTimings;
    if (!timings || timings.size === 0) return null;
    let total = 0;
    for (const ms of timings.values()) total += ms;
    return total;
}
