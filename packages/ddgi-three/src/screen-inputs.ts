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
 * - `velocity`: three's `velocity` (NDC current − previous) in rg, half
 *   float, under the name `@pmndrs/upscaler` reads, so one target feeds both;
 * - `albedo`: `giDiffuseAlbedo`, the diffuse albedo (three's
 *   `diffuseContribution`, base colour × (1 − metalness), for Standard and
 *   Physical materials; `diffuseColor` for the rest), linear, 8-bit;
 * - `instance`: `giInstance`, the platform instance index + 1 (0 for
 *   objects GI doesn't mirror, {@link InstanceIdTable}), written as three
 *   bytes and turned into `r32uint` by a small pass (instance-id-resolve.ts).
 * Depth is the pass's own depth buffer.
 *
 * **`previousLighting`** (last frame's lit colour, RC4.2) is no MRT output:
 * it is the picture `encodeBefore(node)` is given, `node` being the host's
 * HDR result before tone mapping (the composite). With the channel on,
 * `encodeBefore` draws `node` into a half-float target of its own (three's
 * `rtt`, the drawing buffer's size) and shows that, and the next frame's
 * stage hands the target to the platform before it is drawn again: so the
 * techniques read the previous frame's picture with GI in it, as Capsaicin's
 * SSGI reads `PrevCombinedIllumination`. It costs one full-screen pass and
 * 8 bytes a pixel. Until the target has been drawn the stage hands over a
 * 1 × 1 black texture.
 *
 * **The opaque surface under the depth sample** (RC1.1 review). three gives
 * every MRT output but `output` no blending, so a transparent surface would
 * overwrite the channels while the depth buffer keeps the surface behind it
 * (glTF's BLEND materials don't write depth). So each output carries a
 * coverage in alpha, 1 for an opaque surface and 0 for a blended one, and
 * takes the material's own blending (`MaterialBlending`, as the composite's
 * `giAlbedo` does): an opaque material doesn't blend, so it writes; a
 * blended one is drawn with a source its blend equation leaves the target
 * unchanged under (0 for normal and additive blending, premultiplied or
 * not, and for premultiplied subtractive; 1 for premultiplied multiply).
 * Not covered: subtractive and multiply without `premultipliedAlpha` (three
 * r187 logs an error and gives them no blend state, so they overwrite the
 * channels), and custom blending (warned once). three's pipeline cache key
 * leaves out MRT blend modes and the stage sets them on its first run, so
 * the pass must not draw before the first screen stage (see docs/api/three.md).
 * Alpha-tested materials write depth and the channels. A blended material
 * that writes depth (three's default for `transparent`) puts its own depth
 * over the opaque surface's channels: turn its `depthWrite` off.
 *
 * **When it runs.** `encodeBefore(node)` returns `node` with the screen
 * stage attached: once per three frame, before `node` is drawn, it makes
 * sure the pass has drawn this frame (it asks three to run the pass's own
 * once-per-frame update, which does nothing if the pass already drew), takes
 * three's targets as `GPUTexture`s, resolves the instance ids, and records
 * `gi.encodeScreen` (and the debug view, if one is shown) into an encoder
 * it submits at once. three submits a pass's commands when the pass
 * finishes drawing, and the node graph's own draw is submitted after every
 * update-before, so the order on the GPU queue is: the scene pass, the
 * screen stage, then `node`. Put anything that reads a screen technique's
 * results inside `node`.
 *
 * **After the frame's `encode`.** A screen stage rebuilds positions with the
 * camera of the platform's latest `encode` (`ThreeGI.update`), so the host
 * calls `update` before rendering, every frame (with `updateOnRender` the GI
 * light does it while the pass draws, which is also before). The stage
 * checks that a GI frame was encoded since the previous stage and warns once
 * if not (`ScreenInputsSource.encodedFrames`).
 *
 * **No copy** of depth, normal, velocity and albedo: they are three's own
 * render targets, read through the version-pinned backend step in
 * three-gpu-texture.ts (D13), fetched again every frame because three
 * replaces them when the pass is resized. Only single-sampled passes
 * (three's `pass()` default) are supported.
 *
 * **Colour bytes.** The pass's targets must fit the device's
 * `maxColorAttachmentBytesPerSample` (WebGPU's default is 32; with the
 * composite's outputs and every channel it is 56). The platform's
 * `requiredLimits` asks for the adapter's maximum; a device without enough
 * throws `GILimitError` at the first stage. three's own device (a
 * `WebGPURenderer` made without `device`) keeps the default 32 (D14).
 */

import {
    BlendMode,
    CustomBlending,
    DataTexture,
    FloatType,
    HalfFloatType,
    MaterialBlending,
    MultiplyBlending,
    NoBlending,
    Node,
    NodeUpdateType,
    NormalBlending,
    RedFormat,
    RedIntegerFormat,
    RGFormat,
    UnsignedByteType,
    UnsignedIntType,
    type Material,
    type NodeBuilder,
    type NodeFrame,
    type Object3D,
    type PassNode,
    type Texture,
    type TextureNode,
    type WebGPURenderer,
} from 'three/webgpu';
import { diffuseColor, diffuseContribution, instanceIndex, normalWorld, rtt, texture, textureLoad, uint, uniform, uv, uvec2, velocity, type vec2, vec3, vec4 } from 'three/tsl';
import { GILimitError, ScreenInputsView, type FrameScreen, type GIPlatform, type ScreenChannel, type ScreenChannelRequest, type ScreenNeeds } from '@pmndrs/ddgi-core';
import { wrapGpuTexture } from './external-texture.ts';
import { InstanceIdResolve, MAX_ENCODED_INSTANCE_ID } from './instance-id-resolve.ts';
import type { SceneSync } from './scene-sync.ts';
import { threeGpuTexture } from './three-gpu-texture.ts';

/** MRT output names (`mrt({ ...screen.outputs })` uses them; three-composite's `giAlbedo` and `normal` are separate). */
export const GI_SCREEN_NORMAL_OUTPUT = 'giNormal';
export const GI_SCREEN_ALBEDO_OUTPUT = 'giDiffuseAlbedo';
export const GI_SCREEN_VELOCITY_OUTPUT = 'velocity';
export const GI_SCREEN_INSTANCE_OUTPUT = 'giInstance';

type Vec3Node = ReturnType<typeof vec3>;
type UintNode = ReturnType<typeof uint>;

/** Where the screen stage finds the platform and the scene sync's instance map: a `ThreeGI` fits. */
export interface ScreenInputsSource {
    readonly gi: GIPlatform;
    readonly sync: SceneSync;
    /** GI frames encoded so far (`ThreeGI.encodedFrames`): when given, the stage checks one ran before it (file comment). */
    readonly encodedFrames?: number;
}

export interface ScreenInputsOptions {
    /**
     * Channels to render whether or not a technique reads them (a debug view,
     * a check). The channels the platform's techniques ask for are always
     * rendered.
     */
    readonly channels?: ScreenChannelRequest;
}

/**
 * Whether three draws this material with blending: the same test three
 * r187 makes for a pipeline's material blending (WebGPUPipelineUtils).
 */
export function drawsBlended(material: Pick<Material, 'blending' | 'transparent'> | null): boolean {
    if (!material) return false;
    return material.blending !== NoBlending && (material.blending !== NormalBlending || material.transparent);
}

/**
 * One screen channel's MRT output (the file comment): `value` with coverage
 * 1 on an opaque surface; on a blended one, the source its material's
 * blend equation leaves the target unchanged under.
 */
export class GIScreenChannelNode extends Node {
    static get type(): string {
        return 'GIScreenChannelNode';
    }

    // '_': three would otherwise treat a node-valued own property as a child (Node._getChildren).
    private readonly _value: (builder: NodeBuilder) => Vec3Node;

    constructor(value: (builder: NodeBuilder) => Vec3Node) {
        super('vec4');
        this._value = value;
    }

    setup(builder: NodeBuilder): Node {
        const material = builder.material as Material | null;
        if (material && drawsBlended(material)) {
            if (material.blending === CustomBlending) warnCustomBlendingOnce(material);
            // Multiply keeps destination × source: 1 leaves it. Normal, additive and subtractive
            // (premultiplied or not) keep the destination when the source colour and alpha are 0.
            const unchanged = material.blending === MultiplyBlending ? 1 : 0;
            return vec4(unchanged, unchanged, unchanged, unchanged) as unknown as Node;
        }
        return vec4(this._value(builder), 1) as unknown as Node;
    }
}

let customBlendingWarned = false;

/**
 * A custom blend equation may not leave the target unchanged under the
 * "no change" source, so its material may overwrite the screen channels. Said
 * once, so a host sees why a custom-blended object shows in them.
 */
function warnCustomBlendingOnce(material: Material): void {
    if (customBlendingWarned) return;
    customBlendingWarned = true;
    console.warn(`@pmndrs/ddgi/three: material "${material.name || material.type}" uses CustomBlending; the screen inputs can't promise it leaves their channels unchanged, so it may overwrite the opaque surface's normal, albedo, velocity and instance id.`);
}

/**
 * The diffuse albedo per pixel (the file comment): what `FrameScreen.albedo`
 * holds. Standard and Physical materials fold metalness in
 * (MeshStandardNodeMaterial assigns diffuseContribution = diffuseColor × (1 −
 * metalness)); others have no metalness. Node materials copy the classic
 * flags (NOTES), so the flag covers both kinds.
 */
export const giDiffuseAlbedo = new GIScreenChannelNode((builder) => {
    const standard = (builder.material as { isMeshStandardMaterial?: boolean } | null)?.isMeshStandardMaterial === true;
    return (standard ? diffuseContribution : diffuseColor.rgb) as unknown as Vec3Node;
});

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
    /** The id per pixel (uint), before it is encoded for the target. */
    readonly id: UintNode;
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
        this.id = start.equal(uint(NO_RUN)).select(uint(0), id) as unknown as UintNode;
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
            const ids = instances.map((instance) => gi.scene.instanceIndex(instance) + 1);
            for (const id of ids) {
                if (id > MAX_ENCODED_INSTANCE_ID) throw new Error(`@pmndrs/ddgi/three: the screen inputs encode instance ids in 24 bits; instance index ${id - 1} is past it`);
            }
            writes.push({ start: run.start, ids });
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

/** The id's three low bytes as unorm values, red the lowest (instance-id-resolve.ts reads them back). */
function idBytes(id: UintNode): Vec3Node {
    const byte = (shift: number) => (id.shiftRight(uint(shift)).bitAnd(uint(0xff)) as unknown as UintNode).toFloat().div(255);
    return vec3(byte(0), byte(8), byte(16)) as unknown as Vec3Node;
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
function channelsFor(needs: ScreenNeeds, extra: ScreenChannelRequest | undefined): Required<ScreenChannelRequest> {
    return {
        velocity: needs.velocity || extra?.velocity === true,
        albedo: needs.albedo || extra?.albedo === true,
        instance: needs.instance || extra?.instance === true,
        previousLighting: needs.previousLighting || extra?.previousLighting === true,
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
    readonly channels: Required<ScreenChannelRequest>;
    // Node-valued fields start with '_': three treats an own property holding a node as a
    // child to build in the graph the stage joins (Node._getChildren skips '_' names), and
    // the pass and the MRT outputs belong to the scene pass's graph, not the quad's.
    private readonly _scenePass: PassNode;
    private readonly _outputs: Readonly<Record<string, Node>>;
    /** Screen stages run (one per three frame the stage was part of). */
    stages = 0;
    /** three's frame id at the latest stage, for checks that it runs once per frame. */
    lastStageFrame = -1;

    private readonly instanceIds: InstanceIdTable | null;
    private instanceResolve: InstanceIdResolve | null = null;
    private latest: FrameScreen | null = null;
    /** The MRT node whose blending and clear colours were set (configureTargets). */
    private configuredMrt: ReturnType<PassNode['getMRT']> = null;
    private debugView: ScreenInputsView | null = null;
    private debugChannel: ScreenChannel | null = null;
    private readonly debugNodes = new Map<ScreenChannel, TextureNode>();
    private debugPlaceholder: GPUTexture | null = null;
    /** The picture `encodeBefore` draws into a target of its own when the `previousLighting` channel is on. */
    private lightingCapture: { readonly target: Texture } | null = null;
    /** The 1 × 1 black `previousLighting` handed over until the capture's target exists. */
    private lightingPlaceholder: GPUTexture | null = null;
    /** `source.encodedFrames` at the previous stage that ran the techniques. */
    private encodedAtLastStage = -1;
    private readonly warned = new Set<string>();

    constructor(scenePass: PassNode, source: ScreenInputsSource, options: ScreenInputsOptions = {}) {
        super('void');
        this._scenePass = scenePass;
        this.source = source;
        this.channels = channelsFor(source.gi.screenNeeds, options.channels);
        // Once per three frame (NodeFrame.updateBeforeNode keys FRAME updates by frame id).
        this.updateBeforeType = NodeUpdateType.FRAME;

        const outputs: Record<string, Node> = { [GI_SCREEN_NORMAL_OUTPUT]: new GIScreenChannelNode(() => normalWorld as unknown as Vec3Node) };
        if (this.channels.velocity) outputs[GI_SCREEN_VELOCITY_OUTPUT] = new GIScreenChannelNode(() => vec3(velocity as unknown as ReturnType<typeof vec2>, 0) as unknown as Vec3Node);
        if (this.channels.albedo) outputs[GI_SCREEN_ALBEDO_OUTPUT] = giDiffuseAlbedo;
        const ids = this.channels.instance ? new InstanceIdTable() : null;
        this.instanceIds = ids;
        if (ids) outputs[GI_SCREEN_INSTANCE_OUTPUT] = new GIScreenChannelNode(() => idBytes(ids.id));
        this._outputs = outputs;

        // The targets now (three draws an MRT output only once its texture exists), with their
        // formats; the pass's default is rgba16float, and every output needs alpha for its coverage.
        scenePass.getTexture(GI_SCREEN_NORMAL_OUTPUT);
        if (this.channels.velocity) scenePass.getTexture(GI_SCREEN_VELOCITY_OUTPUT);
        // Albedo is 0..1: 8 bits, as Capsaicin's G-buffer stores it, halving its bandwidth.
        if (this.channels.albedo) scenePass.getTexture(GI_SCREEN_ALBEDO_OUTPUT).type = UnsignedByteType;
        // Ids as three bytes plus coverage (instance-id-resolve.ts says why not r32uint).
        if (this.channels.instance) scenePass.getTexture(GI_SCREEN_INSTANCE_OUTPUT).type = UnsignedByteType;
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
        if (!this.channels.previousLighting) return new ScreenStageNode(this, node) as unknown as Node;
        // The picture drawn into a target of its own, shown from there; the next frame's stage
        // reads it as last frame's lighting before it is drawn again (the file comment).
        // No name: three r187 uses a texture node's name as its WGSL variable's.
        const capture = rtt(node);
        this.lightingCapture = { target: capture.value as Texture };
        return new ScreenStageNode(this, capture as unknown as Node) as unknown as Node;
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
        // A debug picture is not the scene's lighting: never captured as previousLighting.
        return new ScreenStageNode(this, shown as unknown as Node) as unknown as Node;
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
        const targets = this.resolveTargets(renderer);
        if (!targets) return undefined;

        const device = gi.device;
        const encoder = device.createCommandEncoder({ label: 'ddgi-three screen stage' });
        let instance: GPUTexture | undefined;
        if (targets.encodedInstance) {
            this.instanceResolve ??= new InstanceIdResolve(device);
            instance = this.instanceResolve.encode(encoder, targets.encodedInstance);
        }
        const screen: FrameScreen = { ...targets.screen, instance };
        this.latest = screen;
        this.stages++;
        this.lastStageFrame = frame.frameId;

        const needs = gi.isReady ? gi.screenNeeds : null;
        const runTechniques = needs !== null && needs.screen && this.covers(needs);
        const showChannel = this.debugChannel;
        // A shown view says so again each frame (DebugViewNode.update), so one taken off screen stops costing.
        this.debugChannel = null;
        if (runTechniques) {
            this.checkEncodedBefore();
            gi.encodeScreen(encoder, screen);
        }
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
        this.instanceResolve?.dispose();
        this.debugView?.dispose();
        this.debugPlaceholder?.destroy();
        this.lightingPlaceholder?.destroy();
        super.dispose();
    }

    /**
     * Checks that the host spread `outputs` into the pass's MRT and that the
     * pass fits the device's colour bytes (throws `GILimitError` if not),
     * then gives our outputs the material's blending (the file comment) and
     * clears them to 0 where nothing is drawn: three clears every MRT target
     * to the scene background otherwise (NOTES). Once per MRT node, before
     * the pass first draws with it.
     */
    private configureTargets(): boolean {
        const outputs = this._scenePass.getMRT();
        if (!outputs || !Object.keys(this._outputs).every((name) => outputs.has(name))) {
            this.warnOnce("the scene pass's MRT lacks the screen inputs' outputs: scenePass.setMRT(mrt({ output, ...screen.outputs })). Nothing is handed to the platform until it has them.");
            return false;
        }
        if (this.configuredMrt === outputs) return true;
        const bytes = colorBytesPerSample(this._scenePass.renderTarget.textures);
        const limit = this.source.gi.device.limits.maxColorAttachmentBytesPerSample;
        if (bytes > limit) throw new GILimitError('maxColorAttachmentBytesPerSample', bytes, limit);
        this.configuredMrt = outputs;
        for (const name of Object.keys(this._outputs)) {
            outputs.setBlendMode(name, new BlendMode(MaterialBlending));
            outputs.setClearColor(name, 0x000000, 0);
        }
        return true;
    }

    /**
     * Screen stages use the camera of the latest `encode`, so one must have
     * run since the previous stage (file comment). Warns once if not; the
     * stage still runs, with the camera it has.
     */
    private checkEncodedBefore(): void {
        const encoded = this.source.encodedFrames;
        if (encoded === undefined) return;
        if (encoded === this.encodedAtLastStage) {
            this.warnOnce(
                'the screen stage ran with no GI frame encoded since the previous one, so screen techniques rebuild positions with an old camera: call ThreeGI.update(camera) before rendering every frame (gi.encode before gi.encodeScreen).',
            );
        }
        this.encodedAtLastStage = encoded;
    }

    /** The platform's techniques read only channels this pass renders; warns once if not. */
    private covers(needs: ScreenNeeds): boolean {
        const missing = (['velocity', 'albedo', 'instance', 'previousLighting'] as const).filter((channel) => needs[channel] && !this.channels[channel]);
        if (missing.length === 0) return true;
        this.warnOnce(
            `a GI technique reads the screen's ${missing.join(', ')}, which this pass doesn't render (screenInputs was made before that technique was added): make the screen inputs again and rebuild the pipeline. Screen techniques are skipped until then.`,
        );
        return false;
    }

    /**
     * three's targets for this frame: the `FrameScreen` but its instance ids,
     * and the encoded id target to resolve. Null if three hasn't made them yet.
     */
    private resolveTargets(renderer: WebGPURenderer): { screen: Omit<FrameScreen, 'instance'>; encodedInstance: GPUTexture | null } | null {
        const scenePass = this.scenePass;
        const gpu = (target: Texture) => threeGpuTexture(renderer, target);
        const renderTarget = (scenePass as unknown as { renderTarget: { depthTexture: Texture | null; samples: number } }).renderTarget;
        if (renderTarget.samples > 1) {
            this.warnOnce('the screen inputs need a single-sampled scene pass (no MSAA): with MSAA the targets hold per-sample data. Screen techniques are skipped.');
            return null;
        }
        const depth = renderTarget.depthTexture ? gpu(renderTarget.depthTexture) : null;
        const normal = gpu(scenePass.getTexture(GI_SCREEN_NORMAL_OUTPUT));
        if (!depth || !normal) return null;
        const optional = (enabled: boolean, name: string) => (enabled ? (gpu(scenePass.getTexture(name)) ?? undefined) : undefined);
        return {
            screen: {
                width: normal.width,
                height: normal.height,
                depth,
                normal,
                normalEncoding: 'snorm-float',
                velocity: optional(this.channels.velocity, GI_SCREEN_VELOCITY_OUTPUT),
                albedo: optional(this.channels.albedo, GI_SCREEN_ALBEDO_OUTPUT),
                previousLighting: this.channels.previousLighting ? this.previousLighting(renderer) : undefined,
                reversedDepth: (renderer as { reversedDepthBuffer?: boolean }).reversedDepthBuffer === true,
            },
            encodedInstance: optional(this.channels.instance, GI_SCREEN_INSTANCE_OUTPUT) ?? null,
        };
    }

    /**
     * Last frame's picture (`encodeBefore`'s capture, not yet drawn this
     * frame), or a 1 × 1 black texture before it has been drawn, or when the
     * host never attached a picture with `encodeBefore` (warned once).
     */
    private previousLighting(renderer: WebGPURenderer): GPUTexture {
        const capture = this.lightingCapture;
        if (!capture) {
            this.warnOnce(
                "a GI technique reads last frame's lighting (previousLighting), which this stage captures from the node given to encodeBefore: attach the pipeline's HDR picture with screen.encodeBefore(node). Black until then.",
            );
        }
        const drawn = capture ? threeGpuTexture(renderer, capture.target) : null;
        if (drawn) return drawn;
        this.lightingPlaceholder ??= this.source.gi.device.createTexture({
            label: 'ddgi-three previous lighting placeholder',
            size: { width: 1, height: 1 },
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING,
        });
        return this.lightingPlaceholder;
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
