/**
 * Baseline 2: three.js r187's VXGI (voxel cone tracing), wired the way
 * three's own `webgpu_vxgi_sponza` example does it:
 *
 *   pre-pass (normals + velocity) -> VXGI gather (AO + irradiance)
 *   -> scene pass with the GI injected into the materials' lighting
 *   -> TRAA, which resolves the cone-tracing noise over several frames.
 *
 * Differences from the example, all on purpose, so the wipe against the probe
 * grid is fair (DECISIONS: a fair wipe):
 *  - The lights and the sky come from the manifest, shared with every other
 *    technique, not from the example's sun slider.
 *  - No ambient fill. The example adds a white `AmbientLight` of 0.5 that the
 *    GI pass's AO darkens. Nothing else gets such a fill, so it is left out.
 *    VXGI injects only direct lights and has no sky term, so in the open
 *    scenes the sky's light is missing on this side. That is a finding about
 *    VXGI, not something to tune away.
 *  - GI strength 1, the physical value (the example uses 1.5).
 *
 * "GI off" keeps the same pipeline (pre-pass, scene pass, TRAA) and only
 * drops the VXGI gather and its injection, so the benchmark's GI cost of
 * VXGI is the gather and nothing else.
 *
 * Set-up is the voxelization. In the viewer it happens inside the first frame,
 * as in three's example. `encodePrepareStep` runs it outside a frame, which is
 * how the benchmark times a re-voxelization.
 */

import { UnsignedByteType } from 'three';
import { builtinGIContext, mrt, normalView, packNormalToRGB, pass, sample, screenUV, unpackRGBToNormal, velocity } from 'three/tsl';
import type { Node, Scene, TextureNode, WebGPURenderer } from 'three/webgpu';
import { vxgi } from 'three/addons/lighting/vxgi/VXGINode.js';
import { traa } from 'three/addons/tsl/display/TRAANode.js';
import { applyViewLayers, type GiMode } from '../scene/layers';
import type { GiTechnique, TechniqueContext, TechniqueId, UpscaleSettings } from './types';
import { copyViewerCamera, createViewCamera } from './view-camera';

export interface VxgiSettings {
    /** Cached indirect bounces in the voxel volume, 0 to 2. */
    bounces: number;
    /** Voxels along the longest axis of the scene. Three's default is 128; its limit is 256. */
    resolution: number;
}

export const DEFAULT_VXGI_SETTINGS: VxgiSettings = { bounces: 1, resolution: 128 };

/** The physical strength: the gathered irradiance as it is. */
const GI_STRENGTH = 1;

/** What VXGINode exposes that this file uses (three ships no types for the addons). */
interface VxgiNodeShape {
    giIntensity: { value: number };
    bounces: number;
    volume: { resolution: number; needsUpdate: boolean; update(renderer: WebGPURenderer, scene: Scene): void };
    getAONode(): Node;
    getGINode(): Node;
    dispose(): void;
}

interface Disposable {
    dispose(): void;
}

export class VxgiTechnique implements GiTechnique {
    readonly id: TechniqueId = 'vxgi';
    readonly label = 'VXGI (three.js)';
    readonly camera = createViewCamera();
    readonly supportsUpscale = false;

    private giMode: GiMode = 'on';
    private readonly giPass: VxgiNodeShape;
    private readonly prePassDepth: TextureNode;
    private readonly prePassVelocity: TextureNode;
    private readonly withGi: Node;
    private withoutGi: Node | null = null;
    private readonly ownedNodes: Disposable[] = [];
    private settings: VxgiSettings;
    private generation = 0;
    /** True after `restartPrepare` until the next `encodePrepareStep`. */
    private voxelizePending = false;

    constructor(
        private readonly context: TechniqueContext,
        settings: VxgiSettings = DEFAULT_VXGI_SETTINGS,
    ) {
        this.settings = { ...settings };
        const { scene } = context.loaded;
        const camera = this.camera;
        this.applyLayers();

        // Pre-pass: view-space normals (packed to 8 bits to save bandwidth) and velocity for TRAA.
        const prePass = pass(scene, camera);
        prePass.name = 'VXGI pre-pass';
        prePass.transparent = false;
        prePass.setMRT(mrt({ output: packNormalToRGB(normalView), velocity }));
        prePass.getTexture('output').type = UnsignedByteType;
        const prePassNormal = sample((uv) => unpackRGBToNormal(prePass.getTextureNode().sample(uv)));
        this.prePassDepth = prePass.getTextureNode('depth');
        this.prePassVelocity = prePass.getTextureNode('velocity');
        this.ownedNodes.push(prePass as unknown as Disposable);

        const giPass = vxgi(this.prePassDepth, prePassNormal, scene, camera, settings.resolution) as unknown as VxgiNodeShape;
        this.giPass = giPass;
        giPass.giIntensity.value = GI_STRENGTH;
        giPass.bounces = settings.bounces;

        // Scene pass: AO and indirect irradiance are injected into the materials' lighting.
        const ao = giPass.getAONode();
        const gi = giPass.getGINode();
        const scenePass = pass(scene, camera);
        scenePass.contextNode = builtinGIContext(
            (ao as unknown as { sample(uv: unknown): { r: Node } }).sample(screenUV).r,
            (gi as unknown as { sample(uv: unknown): { rgb: Node } }).sample(screenUV).rgb,
        );
        this.ownedNodes.push(scenePass as unknown as Disposable);
        this.withGi = this.addTraa(scenePass as unknown as Node);
    }

    get outputNode(): Node {
        return this.giMode === 'off' ? this.graphWithoutGi() : this.withGi;
    }

    get ready(): boolean {
        return !this.voxelizePending;
    }

    get progress(): number {
        return this.voxelizePending ? 0 : 1;
    }

    get prepareGeneration(): number {
        return this.generation;
    }

    setGiMode(mode: GiMode): void {
        this.giMode = mode;
        this.applyLayers();
    }

    /** VXGI's graph is built from several passes of its own; `upscaleScene` can't wrap it. */
    setUpscale(_settings: UpscaleSettings | null): void {}

    syncCamera(viewerCamera: ReturnType<typeof createViewCamera>): void {
        copyViewerCamera(viewerCamera, this.camera);
    }

    describe(): string {
        const voxelizing = this.voxelizePending ? ', voxelizing' : '';
        return `Voxel grid ${this.settings.resolution} along the longest axis, ${this.settings.bounces} bounce${this.settings.bounces === 1 ? '' : 's'}${voxelizing}`;
    }

    restartPrepare(): void {
        this.generation++;
        this.voxelizePending = true;
    }

    /** Voxelizes the scene and injects its light now, outside any frame. */
    encodePrepareStep(): boolean {
        if (!this.voxelizePending) return true;
        this.voxelize();
        this.voxelizePending = false;
        return true;
    }

    notePrepareFinished(): void {}

    /** Takes new settings. A bounce change only re-injects the light (cheap); the voxels stay. */
    configure(settings: VxgiSettings): void {
        if (settings.bounces !== this.settings.bounces) this.giPass.bounces = settings.bounces;
        this.settings = { ...settings, resolution: this.settings.resolution };
    }

    get currentSettings(): Readonly<VxgiSettings> {
        return this.settings;
    }

    describeSettings(): Record<string, number | string> {
        return { voxel_resolution: this.settings.resolution, bounces: this.settings.bounces, gi_strength: GI_STRENGTH };
    }

    /** One throwaway voxelization, so the voxelizer's shaders are compiled before a timed one. */
    async warmUp(): Promise<void> {
        this.voxelize();
        await this.context.device.queue.onSubmittedWorkDone();
    }

    dispose(): void {
        this.giPass.dispose();
        for (const node of this.ownedNodes) node.dispose();
    }

    private voxelize(): void {
        this.giPass.volume.needsUpdate = true;
        this.giPass.volume.update(this.context.renderer, this.context.loaded.scene);
    }

    private addTraa(scenePass: Node): Node {
        const resolved = traa(scenePass, this.prePassDepth, this.prePassVelocity, this.camera);
        this.ownedNodes.push(resolved as unknown as Disposable);
        return resolved as unknown as Node;
    }

    /** The same pipeline without the VXGI gather: built the first time "GI off" is asked for. */
    private graphWithoutGi(): Node {
        if (!this.withoutGi) {
            const plainScenePass = pass(this.context.loaded.scene, this.camera);
            this.ownedNodes.push(plainScenePass as unknown as Disposable);
            this.withoutGi = this.addTraa(plainScenePass as unknown as Node);
        }
        return this.withoutGi;
    }

    private applyLayers(): void {
        applyViewLayers(this.camera.layers, this.giMode, []);
    }
}
