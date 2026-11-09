/**
 * Screen inputs (RC1.1, docs/design/radiance-cache.md "Screen inputs: the
 * flexible route"): the platform's `FrameScreen` filled from the host's own
 * MRT scene pass, the T1.14 pattern, every frame and through resizes.
 *
 * ```ts
 * const scenePass = pass(scene, camera);
 * const screen = screenInputs(scenePass, threeGI);
 * scenePass.setMRT(mrt({ output, giAlbedo, normal: packNormalToRGB(normalView), ...screen.outputs }));
 * pipeline.outputNode = screen.encodeBefore(ddgi(scenePass, threeGI));
 * ```
 *
 * **What the pass writes.** `outputs` holds the MRT outputs for the channels
 * the platform's techniques read (`GIPlatform.screenNeeds`, at the time
 * `screenInputs` is called) plus any the host asks for (`channels`):
 * - always: `giNormal`, the world-space shading normal (three's
 *   `normalWorld`, normal maps included), signed, half float;
 * - `velocity`: three's `velocity` (NDC current − previous), half float,
 *   under the name `@pmndrs/upscaler` reads, so one target feeds both;
 * - `albedo`: `giDiffuseAlbedo`, the diffuse albedo (three's
 *   `diffuseContribution`, base colour × (1 − metalness), for Standard and
 *   Physical materials; `diffuseColor` for the rest), linear, 8-bit;
 * - `instance`: `giInstance`, `r32uint`, the platform instance index + 1
 *   (0 for objects GI doesn't mirror), from {@link InstanceIdTable}.
 * Depth is the pass's own depth buffer. Each output is the last surface
 * drawn at the pixel, like depth (the outputs don't blend).
 *
 * **When it runs.** `encodeBefore(node)` returns `node` with the screen
 * stage attached: once per three frame, before `node` is drawn, it makes
 * sure the pass has drawn this frame (it asks three to run the pass's own
 * once-per-frame update, which does nothing if the pass already drew), takes
 * three's targets as `GPUTexture`s, and records `gi.encodeScreen` (and the
 * debug view, if one is shown) into an encoder it submits at once. three
 * submits a pass's commands when the pass finishes drawing, and the node
 * graph's own draw is submitted after every update-before, so the order on
 * the GPU queue is: the scene pass, the screen stage, then `node`. Put
 * anything that reads a screen technique's results inside `node`.
 *
 * **No copy.** The textures are three's own render targets, read through
 * the version-pinned backend step in three-gpu-texture.ts (D13). They are
 * fetched again every frame, because three replaces them when the pass is
 * resized. Only single-sampled passes (three's `pass()` default) are
 * supported: with MSAA the targets are resolved per sample, not per pixel.
 */

import {
    DataTexture,
    FloatType,
    HalfFloatType,
    Node,
    NodeUpdateType,
    RedFormat,
    RedIntegerFormat,
    RGFormat,
    UnsignedByteType,
    UnsignedIntType,
    type NodeBuilder,
    type NodeFrame,
    type Object3D,
    type PassNode,
    type Texture,
    type TextureNode,
    type WebGPURenderer,
} from 'three/webgpu';
import { diffuseColor, diffuseContribution, instanceIndex, normalWorld, texture, textureLoad, uint, uniform, uv, uvec2, velocity, vec4 } from 'three/tsl';
import { ScreenInputsView, type FrameScreen, type GIPlatform, type ScreenChannel, type ScreenChannels, type ScreenNeeds } from '@pmndrs/ddgi-core';
import { wrapGpuTexture } from './external-texture.ts';
import type { SceneSync } from './scene-sync.ts';
import { threeGpuTexture } from './three-gpu-texture.ts';

/** MRT output names (`mrt({ ...screen.outputs })` uses them; three-composite's `giAlbedo` and `normal` are separate). */
export const GI_SCREEN_NORMAL_OUTPUT = 'giNormal';
export const GI_SCREEN_ALBEDO_OUTPUT = 'giDiffuseAlbedo';
export const GI_SCREEN_VELOCITY_OUTPUT = 'velocity';
export const GI_SCREEN_INSTANCE_OUTPUT = 'giInstance';

type TslNode = ReturnType<typeof vec4>;
type UintNode = ReturnType<typeof uint>;

/** Where the screen stage finds the platform and the scene sync's instance map: a `ThreeGI` fits. */
export interface ScreenInputsSource {
    readonly gi: GIPlatform;
    readonly sync: SceneSync;
}

export interface ScreenInputsOptions {
    /**
     * Channels to render whether or not a technique reads them (a debug view,
     * a check). The channels the platform's techniques ask for are always
     * rendered.
     */
    readonly channels?: ScreenChannels;
}

/**
 * The diffuse albedo per pixel (the file comment): what `FrameScreen.albedo`
 * holds. Use one instance, {@link giDiffuseAlbedo}.
 */
export class GIDiffuseAlbedoNode extends Node {
    static get type(): string {
        return 'GIDiffuseAlbedoNode';
    }

    constructor() {
        super('vec4');
    }

    setup(builder: NodeBuilder): Node {
        // Standard and Physical materials fold metalness in (MeshStandardNodeMaterial assigns
        // diffuseContribution = diffuseColor × (1 − metalness)); others have no metalness.
        // Node materials copy the classic flags (NOTES), so the flag covers both kinds.
        const standard = (builder.material as { isMeshStandardMaterial?: boolean } | null)?.isMeshStandardMaterial === true;
        const albedo = standard ? diffuseContribution : diffuseColor.rgb;
        return vec4(albedo, 1) as unknown as Node;
    }
}

/** The diffuse albedo, for `mrt({ ..., giDiffuseAlbedo })`; `screen.outputs` includes it when asked for. */
export const giDiffuseAlbedo = new GIDiffuseAlbedoNode();

/** Marks a run of the instance table that belongs to no mirrored object: the pixel writes 0. */
const NO_RUN = 0xffffffff;
/** Width of the instance table texture; rows are added as it fills. */
const TABLE_WIDTH = 1024;

/**
 * Per object, the platform instance index + 1 of each of its instances, in
 * a `r32uint` texture the scene pass reads (fragment-readable data lives in
 * textures; NOTES). Each mirrored mesh gets a run of entries, one per
 * instance (a plain mesh one, an `InstancedMesh` one per instance, in
 * order), and a per-object uniform with where its run starts; the pixel
 * reads entry `start + instanceIndex`. Rebuilt when the scene sync creates
 * or destroys an instance (`SceneSync.instanceVersion`).
 *
 * A run keeps its start while its object keeps its instance count, so most
 * objects' uniforms never change (three may skip an unchanged object's
 * uniform update). Runs of destroyed objects are not reused: the table only
 * grows, by one entry per instance ever created.
 */
export class InstanceIdTable {
    /** The node a pass's MRT output uses: `giInstance`. */
    readonly node: TslNode;
    private table: DataTexture;
    private readonly tableNode: TextureNode;
    private readonly runs = new Map<Object3D, { start: number; count: number }>();
    private used = 0;
    private seenVersion = -1;

    constructor() {
        this.table = makeTable(1);
        this.tableNode = texture(this.table) as unknown as TextureNode;
        // Each object's own value: three keeps object uniforms per draw.
        const runStart = uniform(NO_RUN, 'uint');
        runStart.onObjectUpdate(({ object }) => {
            runStart.value = (object && this.runs.get(object)?.start) ?? NO_RUN;
        });
        // All unsigned: the run's start plus the draw's instance, then the table's texel in rows of TABLE_WIDTH.
        const start = runStart as unknown as UintNode;
        const entry = start.add(instanceIndex as unknown as UintNode);
        const texel = uvec2(entry.mod(uint(TABLE_WIDTH)), entry.div(uint(TABLE_WIDTH)));
        const id = (textureLoad(this.tableNode, texel as unknown as Node) as unknown as { x: UintNode }).x;
        this.node = start.equal(uint(NO_RUN)).select(uint(0), id) as unknown as TslNode;
    }

    /** Brings the table up to date with the sync's instances (cheap when nothing was created or destroyed). */
    refresh(sync: SceneSync, gi: GIPlatform): void {
        if (sync.instanceVersion === this.seenVersion) return;
        this.seenVersion = sync.instanceVersion;
        const live = new Set<Object3D>();
        const writes: Array<{ start: number; ids: number[] }> = [];
        for (const [object, instances] of sync.mirroredInstances()) {
            live.add(object);
            let run = this.runs.get(object);
            if (!run || run.count !== instances.length) {
                run = { start: this.used, count: instances.length };
                this.used += instances.length;
                this.runs.set(object, run);
            }
            writes.push({ start: run.start, ids: instances.map((instance) => gi.scene.instanceIndex(instance) + 1) });
        }
        for (const object of [...this.runs.keys()]) if (!live.has(object)) this.runs.delete(object);
        const rows = Math.max(1, Math.ceil(this.used / TABLE_WIDTH));
        if (rows !== this.table.image.height) {
            this.table.dispose();
            this.table = makeTable(rows);
            this.tableNode.value = this.table;
        }
        const data = this.table.image.data as Uint32Array;
        data.fill(0);
        for (const write of writes) data.set(write.ids, write.start);
        this.table.needsUpdate = true;
    }

    dispose(): void {
        this.table.dispose();
    }
}

function makeTable(rows: number): DataTexture {
    const table = new DataTexture(new Uint32Array(TABLE_WIDTH * rows), TABLE_WIDTH, rows, RedIntegerFormat, UnsignedIntType);
    table.name = 'ddgi-three instance ids';
    table.needsUpdate = true;
    return table;
}

/**
 * WebGPU's "render target pixel byte cost" of a pass's colour targets, the
 * sum `maxColorAttachmentBytesPerSample` limits (default 32): each target's
 * cost, after rounding the running total up to its component size. Note
 * that `rgba8unorm` costs 8, like `rgba16float` (the spec's table).
 */
export function colorBytesPerSample(textures: readonly Texture[]): number {
    let bytes = 0;
    for (const texture of textures) {
        const channels = texture.format === RedIntegerFormat || texture.format === RedFormat ? 1 : texture.format === RGFormat ? 2 : 4;
        const componentBytes = texture.type === FloatType || texture.type === UnsignedIntType ? 4 : texture.type === HalfFloatType ? 2 : 1;
        const cost = channels === 4 && componentBytes === 1 ? 8 : channels * componentBytes;
        bytes = Math.ceil(bytes / componentBytes) * componentBytes + cost;
    }
    return bytes;
}

/** The channels a pass renders: the platform's techniques' needs plus the host's. */
function channelsFor(needs: ScreenNeeds, extra: ScreenChannels | undefined): Required<ScreenChannels> {
    return {
        velocity: needs.velocity || extra?.velocity === true,
        albedo: needs.albedo || extra?.albedo === true,
        instance: needs.instance || extra?.instance === true,
    };
}

/**
 * The screen stage on a scene pass (file comment). Made by
 * {@link screenInputs}. It is itself a node: `encodeBefore` and `view`
 * attach it to the node graph.
 */
export class ScreenInputsNode extends Node {
    static get type(): string {
        return 'ScreenInputsNode';
    }

    readonly source: ScreenInputsSource;
    /** The optional channels this pass renders (fixed when the outputs are made). */
    readonly channels: Required<ScreenChannels>;
    // Node-valued fields start with '_': three treats an own property holding a node as a
    // child to build in the graph the stage joins (Node._getChildren skips '_' names), and
    // the pass and the MRT outputs belong to the scene pass's graph, not the quad's.
    private readonly _scenePass: PassNode;
    private readonly _outputs: Readonly<Record<string, Node>>;
    /** Screen stages run (one per three frame the stage was part of). */
    stages = 0;

    private readonly instanceIds: InstanceIdTable | null;
    private latest: FrameScreen | null = null;
    /** The MRT node whose clear colours were set (configureTargets). */
    private configuredMrt: ReturnType<PassNode['getMRT']> = null;
    private debugView: ScreenInputsView | null = null;
    private debugChannel: ScreenChannel | null = null;
    private readonly debugNodes = new Map<ScreenChannel, TextureNode>();
    private debugPlaceholder: GPUTexture | null = null;
    private readonly warned = new Set<string>();

    constructor(scenePass: PassNode, source: ScreenInputsSource, options: ScreenInputsOptions = {}) {
        super('void');
        this._scenePass = scenePass;
        this.source = source;
        this.channels = channelsFor(source.gi.screenNeeds, options.channels);
        // Once per three frame (NodeFrame.updateBeforeNode keys FRAME updates by frame id).
        this.updateBeforeType = NodeUpdateType.FRAME;

        const outputs: Record<string, Node> = { [GI_SCREEN_NORMAL_OUTPUT]: vec4(normalWorld, 1) as unknown as Node };
        if (this.channels.velocity) outputs[GI_SCREEN_VELOCITY_OUTPUT] = velocity as unknown as Node;
        if (this.channels.albedo) outputs[GI_SCREEN_ALBEDO_OUTPUT] = giDiffuseAlbedo;
        this.instanceIds = this.channels.instance ? new InstanceIdTable() : null;
        if (this.instanceIds) outputs[GI_SCREEN_INSTANCE_OUTPUT] = this.instanceIds.node as unknown as Node;
        this._outputs = outputs;

        // The targets now, with their formats; the pass's default is half float rgba.
        scenePass.getTexture(GI_SCREEN_NORMAL_OUTPUT);
        // Motion has two components: rg16float, 4 bytes per pixel instead of rgba16float's 8.
        if (this.channels.velocity) scenePass.getTexture(GI_SCREEN_VELOCITY_OUTPUT).format = RGFormat;
        // Albedo is 0..1: 8 bits, as Capsaicin's G-buffer stores it, halving its bandwidth.
        if (this.channels.albedo) scenePass.getTexture(GI_SCREEN_ALBEDO_OUTPUT).type = UnsignedByteType;
        if (this.channels.instance) {
            const ids = scenePass.getTexture(GI_SCREEN_INSTANCE_OUTPUT);
            ids.type = UnsignedIntType;
            ids.format = RedIntegerFormat;
        }
    }

    get scenePass(): PassNode {
        return this._scenePass;
    }

    /**
     * The MRT outputs to spread into the pass's `mrt({...})`. Their targets
     * exist from construction on (three draws an MRT output only once its
     * texture exists; NOTES), with the formats the file comment lists.
     */
    get outputs(): Readonly<Record<string, Node>> {
        return this._outputs;
    }

    /** Nothing to build: the stage is an update-before only, never part of the shader. */
    setup(): null {
        return null;
    }

    /** The screen as of the latest stage: three's textures this frame. Null before the first. */
    get screen(): FrameScreen | null {
        return this.latest;
    }

    /**
     * `node`, with the screen stage run once per frame before it is drawn
     * (file comment). Use it on the node that reads what screen techniques
     * produce, usually the pipeline's output node.
     */
    encodeBefore(node: Node): Node {
        return new ScreenStageNode(this, node) as unknown as Node;
    }

    /**
     * A debug view of one channel (the core's `ScreenInputsView`), as a node
     * that draws it full screen in display values (show it after tone
     * mapping, or with none). The screen stage runs before it. One channel
     * at a time: showing another switches what the stage draws.
     */
    view(channel: ScreenChannel): Node {
        if (channel !== 'depth' && channel !== 'normal' && !this.channels[channel]) {
            throw new Error(`@pmndrs/ddgi/three: this pass doesn't render the '${channel}' screen channel; ask for it with screenInputs(pass, gi, { channels: { ${channel}: true } })`);
        }
        this.debugChannel = channel;
        let textureNode = this.debugNodes.get(channel);
        if (!textureNode) {
            textureNode = texture(wrapGpuTexture(this.placeholder(), { dimension: '2d', sampleType: 'float' })) as unknown as TextureNode;
            this.debugNodes.set(channel, textureNode);
        }
        const shown = new DebugViewNode(this, channel, textureNode.sample(uv()) as unknown as Node);
        return this.encodeBefore(shown as unknown as Node);
    }

    /** The texture the debug view last drew (`view`), or null; the platform's, replaced on resize. */
    get viewTexture(): GPUTexture | null {
        return this.debugView?.texture ?? null;
    }

    /** Called by the view nodes each frame: the channel whose view the stage should draw. */
    showChannel(channel: ScreenChannel): void {
        this.debugChannel = channel;
    }

    updateBefore(frame: NodeFrame): boolean | undefined {
        const renderer = frame.renderer as WebGPURenderer | null;
        if (!renderer) return false;
        const gi = this.source.gi;
        if (!this.configureTargets()) return undefined;
        // The table must be right before the pass draws (it reads it).
        if (this.instanceIds && gi.isReady) this.instanceIds.refresh(this.source.sync, gi);
        // Draws the scene pass now if it hasn't drawn this frame; otherwise nothing.
        frame.updateBeforeNode(this.scenePass as unknown as Node);
        const screen = this.resolve(renderer);
        if (!screen) return undefined;
        this.latest = screen;
        this.stages++;

        const needs = gi.isReady ? gi.screenNeeds : null;
        const runTechniques = needs !== null && needs.screen && this.covers(needs);
        const showChannel = this.debugChannel;
        // A shown view says so again each frame (DebugViewNode.update), so one taken off screen stops costing.
        this.debugChannel = null;
        if (!runTechniques && showChannel === null) return undefined;
        const device = gi.device;
        const encoder = device.createCommandEncoder({ label: 'ddgi-three screen stage' });
        if (runTechniques) gi.encodeScreen(encoder, screen);
        if (showChannel !== null) {
            this.debugView ??= new ScreenInputsView(device);
            // The pass's camera planes let the depth view show distance (ScreenInputsView); only a perspective camera's.
            const camera = this._scenePass.camera as unknown as { isPerspectiveCamera?: boolean; near: number; far: number };
            this.debugView.settings.depthPlanes = camera.isPerspectiveCamera === true ? { near: camera.near, far: camera.far } : null;
            const shown = this.debugView.encode(encoder, screen, showChannel);
            const node = this.debugNodes.get(showChannel);
            const wrapped = node?.value as unknown as { sourceTexture?: GPUTexture } | undefined;
            // A new view texture (a resize): wrap it; three reads the node's value at the next bind.
            if (node && wrapped?.sourceTexture !== shown) node.value = wrapGpuTexture(shown, { dimension: '2d', sampleType: 'float' });
        }
        device.queue.submit([encoder.finish()]);
        return undefined;
    }

    dispose(): void {
        this.instanceIds?.dispose();
        this.debugView?.dispose();
        this.debugPlaceholder?.destroy();
        super.dispose();
    }

    /**
     * Checks that the host spread `outputs` into the pass's MRT, and clears
     * our targets to 0 where nothing is drawn: three clears every MRT target
     * to the scene background otherwise (NOTES), which would read as a
     * normal, an albedo or an instance there. Once per MRT node.
     */
    private configureTargets(): boolean {
        const outputs = this._scenePass.getMRT();
        if (!outputs || !Object.keys(this._outputs).every((name) => outputs.has(name))) {
            this.warnOnce("the scene pass's MRT lacks the screen inputs' outputs: scenePass.setMRT(mrt({ output, ...screen.outputs })). Nothing is handed to the platform until it has them.");
            return false;
        }
        if (this.configuredMrt === outputs) return true;
        this.configuredMrt = outputs;
        for (const name of Object.keys(this._outputs)) outputs.setClearColor(name, 0x000000, 0);
        const bytes = colorBytesPerSample(this._scenePass.renderTarget.textures);
        const limit = this.source.gi.device.limits.maxColorAttachmentBytesPerSample;
        if (bytes > limit) {
            this.warnOnce(
                `the scene pass writes about ${bytes} bytes per pixel, past this device's maxColorAttachmentBytesPerSample of ${limit}, so three can't draw it. ` +
                    "Make the device with requestGIDevice/createGIRenderer (they ask for the adapter's maximum), or ask for maxColorAttachmentBytesPerSample yourself.",
            );
        }
        return true;
    }

    /** The platform's techniques read only channels this pass renders; warns once if not. */
    private covers(needs: ScreenNeeds): boolean {
        const missing = (['velocity', 'albedo', 'instance'] as const).filter((channel) => needs[channel] && !this.channels[channel]);
        if (missing.length === 0) return true;
        this.warnOnce(
            `a GI technique reads the screen's ${missing.join(', ')}, which this pass doesn't render (screenInputs was made before that technique was added): make the screen inputs again and rebuild the pipeline. Screen techniques are skipped until then.`,
        );
        return false;
    }

    /** three's targets for this frame as the platform's `FrameScreen`, or null if three hasn't made them yet. */
    private resolve(renderer: WebGPURenderer): FrameScreen | null {
        const scenePass = this.scenePass;
        const gpu = (target: Texture) => threeGpuTexture(renderer, target);
        const depthTexture = (scenePass as unknown as { renderTarget: { depthTexture: Texture | null; samples: number } }).renderTarget;
        if (depthTexture.samples > 1) {
            this.warnOnce('the screen inputs need a single-sampled scene pass (no MSAA): with MSAA the targets hold per-sample data. Screen techniques are skipped.');
            return null;
        }
        const depth = depthTexture.depthTexture ? gpu(depthTexture.depthTexture) : null;
        const normal = gpu(scenePass.getTexture(GI_SCREEN_NORMAL_OUTPUT));
        if (!depth || !normal) return null;
        const optional = (enabled: boolean, name: string) => (enabled ? (gpu(scenePass.getTexture(name)) ?? undefined) : undefined);
        return {
            width: normal.width,
            height: normal.height,
            depth,
            normal,
            normalEncoding: 'snorm-float',
            velocity: optional(this.channels.velocity, GI_SCREEN_VELOCITY_OUTPUT),
            albedo: optional(this.channels.albedo, GI_SCREEN_ALBEDO_OUTPUT),
            instance: optional(this.channels.instance, GI_SCREEN_INSTANCE_OUTPUT),
            reversedDepth: (renderer as { reversedDepthBuffer?: boolean }).reversedDepthBuffer === true,
        };
    }

    /** A 1 × 1 texture the debug view nodes show until the first stage draws (three needs a texture at build time). */
    private placeholder(): GPUTexture {
        this.debugPlaceholder ??= this.source.gi.device.createTexture({ label: 'ddgi-three screen view placeholder', size: { width: 1, height: 1 }, format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING });
        return this.debugPlaceholder;
    }

    private warnOnce(message: string): void {
        if (this.warned.has(message)) return;
        this.warned.add(message);
        console.warn(`@pmndrs/ddgi/three: ${message}`);
    }
}

/**
 * `inner` with the screen stage attached (`ScreenInputsNode.encodeBefore`).
 * Building the stage first puts its once-per-frame update ahead of `inner`'s
 * (three runs update-befores in the order their nodes finished building),
 * so `inner`'s own passes, if any, see this frame's screen results.
 */
class ScreenStageNode extends Node {
    static get type(): string {
        return 'ScreenStageNode';
    }

    constructor(
        private readonly stage: ScreenInputsNode,
        private readonly inner: Node,
    ) {
        super();
    }

    getNodeType(builder: NodeBuilder): string {
        return this.inner.getNodeType(builder);
    }

    setup(builder: NodeBuilder): Node {
        this.stage.build(builder);
        return this.inner;
    }
}

/** A debug view's picture; tells the stage, each render, which channel to draw. */
class DebugViewNode extends Node {
    static get type(): string {
        return 'ScreenInputsDebugViewNode';
    }

    constructor(
        private readonly stage: ScreenInputsNode,
        private readonly channel: ScreenChannel,
        private readonly picture: Node,
    ) {
        super('vec4');
        this.updateType = NodeUpdateType.FRAME;
    }

    update(): boolean | undefined {
        this.stage.showChannel(this.channel);
        return undefined;
    }

    setup(): Node {
        return this.picture;
    }
}

/**
 * The screen inputs on a scene pass (file comment): `source` is the
 * `ThreeGI` driving the platform. Spread `outputs` into the pass's MRT and
 * attach the stage with `encodeBefore`.
 */
export function screenInputs(scenePass: PassNode, source: ScreenInputsSource, options: ScreenInputsOptions = {}): ScreenInputsNode {
    return new ScreenInputsNode(scenePass, source, options);
}
