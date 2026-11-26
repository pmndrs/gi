/**
 * A debug view of the host's screen inputs (RC1.1): one `FrameScreen`
 * channel drawn into a texture the host can put on screen, so a person can
 * see what a screen-space technique is given. Engine-free: plain WebGPU, one
 * small compute kernel per channel.
 *
 * ```ts
 * const view = new ScreenInputsView(device);
 * // each frame, after the host's scene pass and before it shows the view:
 * const texture = view.encode(encoder, screen, 'normal');
 * ```
 *
 * The texture is `rgba8unorm` holding **display values**: show it as it is,
 * with no tone mapping or colour conversion on top. Per channel:
 * - `depth`: grey, white near and black far; black where nothing was drawn.
 *   With the camera's planes (`depthPlanes`) it shows the distance along the
 *   view axis, `1 / (1 + distance / depthMidDistance)`: mid grey at
 *   `depthMidDistance`. Without them, a log scale over `depthDecades`
 *   decades of `1 − depth` (about near ÷ distance for perspective depth,
 *   which crowds a scene's range into a few greys);
 * - `normal`: the world-space normal as `n × 0.5 + 0.5` (the usual
 *   normal-map look: +x red, +y green, +z blue); black where nothing was drawn;
 * - `velocity`: NDC motion as `0.5 + 0.5 × clamp(v × velocityScale)` in red
 *   and green, blue 0.5: mid grey is still;
 * - `albedo`: the diffuse albedo, sRGB-encoded so it looks like the colour;
 * - `instance`: each instance a colour from a hash of its id
 *   ({@link instanceDebugColor}), black for none.
 *
 * Kernels are compiled the first time a channel is shown (one synchronous
 * `createComputePipeline` each), on the host's device, so they don't count
 * as platform recompiles. It's a debug view: nothing compiles until it is used.
 */

import type { FrameScreen } from '../contract/frame.ts';

/** Every `FrameScreen` channel the view can show. */
export type ScreenChannel = 'depth' | 'normal' | 'velocity' | 'albedo' | 'instance';

export const SCREEN_CHANNELS: readonly ScreenChannel[] = ['depth', 'normal', 'velocity', 'albedo', 'instance'];

/** Live settings: change them any time. */
export interface ScreenInputsViewSettings {
    /** NDC motion that reaches full colour is `1 / velocityScale`. Default 20 (a tenth of the screen's half-width). */
    velocityScale: number;
    /** Without `depthPlanes`: decades of `1 − depth` the depth view spans, near to far. Default 5. */
    depthDecades: number;
    /**
     * The camera's near and far planes (a perspective camera, 0..1 depth as
     * WebGPU and three use), so the depth view shows distance. Null: the log
     * fallback.
     */
    depthPlanes: { readonly near: number; readonly far: number } | null;
    /** With `depthPlanes`: the distance shown as mid grey, in the scene's units. Default 10. */
    depthMidDistance: number;
}

export const DEFAULT_SCREEN_INPUTS_VIEW_SETTINGS: Readonly<ScreenInputsViewSettings> = { velocityScale: 20, depthDecades: 5, depthPlanes: null, depthMidDistance: 10 };

/** The view's format: display values, 8 bits are plenty, and core WebGPU allows it as a write-only storage texture. */
export const SCREEN_INPUTS_VIEW_FORMAT: GPUTextureFormat = 'rgba8unorm';

const WORKGROUP_SIZE = 8;
/** The settings uniform (GiScreenViewSettings): eight four-byte words. */
const SETTINGS_BYTES = 32;

/**
 * A 32-bit hash (PCG, Jarzynski and Olano 2020) of an instance id, the same
 * as the kernel's `giScreenViewHash`, so a check can predict the colour.
 */
export function instanceDebugHash(id: number): number {
    const state = (Math.imul(id >>> 0, 747796405) + 2891336453) >>> 0;
    const word = Math.imul((state >>> ((state >>> 28) + 4)) ^ state, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
}

/**
 * The colour the instance view gives id `id` (FrameScreen's value: index + 1),
 * 0..1 before 8-bit storage: black for 0, otherwise three hash bytes lifted
 * to 0.25..1, so no instance is drawn as black.
 */
export function instanceDebugColor(id: number): [number, number, number] {
    if (id === 0) return [0, 0, 0];
    const hash = instanceDebugHash(id);
    const lift = (byte: number) => 0.25 + 0.75 * (byte / 255);
    return [lift(hash & 255), lift((hash >>> 8) & 255), lift((hash >>> 16) & 255)];
}

/** How a channel's texture is bound: its WGSL type and the matching layout entry. */
interface SourceKind {
    readonly key: string;
    readonly wgslType: string;
    readonly sampleType: GPUTextureSampleType;
}

const DEPTH_SOURCE: SourceKind = { key: 'depth', wgslType: 'texture_depth_2d', sampleType: 'depth' };
// Unfilterable binds every float format, 32-bit ones included; the view only loads texels.
const FLOAT_SOURCE: SourceKind = { key: 'float', wgslType: 'texture_2d<f32>', sampleType: 'unfilterable-float' };
const UINT_SOURCE: SourceKind = { key: 'uint', wgslType: 'texture_2d<u32>', sampleType: 'uint' };

function sourceKind(channel: ScreenChannel, texture: GPUTexture): SourceKind {
    if (channel === 'instance') return UINT_SOURCE;
    if (channel === 'depth' && texture.format.startsWith('depth')) return DEPTH_SOURCE;
    return FLOAT_SOURCE;
}

/** The body of `giScreenViewColor(pixel) -> vec3f` for a channel. */
function channelBody(channel: ScreenChannel, source: SourceKind): string {
    switch (channel) {
        case 'depth': {
            // A depth texture loads a plain f32; a colour texture holds depth in red.
            const load = source === DEPTH_SOURCE ? 'textureLoad(giScreenSource, pixel, 0)' : 'textureLoad(giScreenSource, pixel, 0).r';
            return `
    let depth = ${load};
    // Standard depth: 1 - depth is about near / distance; reversed depth holds that already.
    let towardNear = select(1.0 - depth, depth, giScreenViewSettings.reversedDepth == 1u);
    if (towardNear <= 0.0) {
        return vec3f(0.0);
    }
    let near = giScreenViewSettings.near;
    let far = giScreenViewSettings.far;
    if (near > 0.0) {
        // Perspective depth d = far / (far - near) × (1 - near / z), so with towardNear = 1 - d:
        // z = near × far / (near + towardNear × (far - near)).
        let distance = near * far / (near + towardNear * (far - near));
        return vec3f(1.0 / (1.0 + distance / giScreenViewSettings.depthMidDistance));
    }
    // log10(towardNear) / decades + 1: 1 at the near plane, 0 at decades decades further.
    let shade = clamp(1.0 + log2(towardNear) / (giScreenViewSettings.depthDecades * log2(10.0)), 0.0, 1.0);
    return vec3f(shade);`;
        }
        case 'normal':
            return `
    var normal = textureLoad(giScreenSource, pixel, 0).rgb;
    if (giScreenViewSettings.normalSigned == 0u) {
        normal = normal * 2.0 - 1.0;
    }
    // Hosts clear the target where nothing is drawn; no unit normal is that short.
    if (dot(normal, normal) < 0.01) {
        return vec3f(0.0);
    }
    return normalize(normal) * 0.5 + 0.5;`;
        case 'velocity':
            return `
    let motion = textureLoad(giScreenSource, pixel, 0).rg;
    let shown = clamp(motion * giScreenViewSettings.velocityScale, vec2f(-1.0), vec2f(1.0));
    return vec3f(shown * 0.5 + 0.5, 0.5);`;
        case 'albedo':
            return `
    return giLinearToSrgb(clamp(textureLoad(giScreenSource, pixel, 0).rgb, vec3f(0.0), vec3f(1.0)));`;
        case 'instance':
            return `
    let id = textureLoad(giScreenSource, pixel, 0).r;
    if (id == 0u) {
        return vec3f(0.0);
    }
    let hash = giScreenViewHash(id);
    let bytes = vec3f(f32(hash & 255u), f32((hash >> 8u) & 255u), f32((hash >> 16u) & 255u)) / 255.0;
    return 0.25 + 0.75 * bytes;`;
    }
}

function viewKernel(channel: ScreenChannel, source: SourceKind): string {
    return /* wgsl */ `
struct GiScreenViewSettings {
    velocityScale: f32,
    depthDecades: f32,
    normalSigned: u32,
    reversedDepth: u32,
    // The camera's planes; near 0 means unknown (the log fallback).
    near: f32,
    far: f32,
    depthMidDistance: f32,
    unused: f32,
}

@group(0) @binding(0) var giScreenSource: ${source.wgslType};
@group(0) @binding(1) var giScreenView: texture_storage_2d<${SCREEN_INPUTS_VIEW_FORMAT}, write>;
@group(0) @binding(2) var<uniform> giScreenViewSettings: GiScreenViewSettings;

// PCG hash (Jarzynski and Olano 2020); instanceDebugHash in TypeScript is the same.
fn giScreenViewHash(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn giLinearToSrgb(linear: vec3f) -> vec3f {
    let low = linear * 12.92;
    let high = 1.055 * pow(linear, vec3f(1.0 / 2.4)) - 0.055;
    return select(high, low, linear <= vec3f(0.0031308));
}

fn giScreenViewColor(pixel: vec2u) -> vec3f {${channelBody(channel, source)}
}

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(giScreenView);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    textureStore(giScreenView, id.xy, vec4f(giScreenViewColor(id.xy), 1.0));
}
`;
}

interface ChannelKernel {
    readonly pipeline: GPUComputePipeline;
    readonly layout: GPUBindGroupLayout;
    readonly settings: GPUBuffer;
}

export class ScreenInputsView {
    readonly settings: ScreenInputsViewSettings;
    private readonly device: GPUDevice;
    private readonly kernels = new Map<string, ChannelKernel>();
    private target: GPUTexture | null = null;
    private readonly settingsData = new ArrayBuffer(SETTINGS_BYTES);

    constructor(device: GPUDevice, settings: Partial<ScreenInputsViewSettings> = {}) {
        this.device = device;
        this.settings = { ...DEFAULT_SCREEN_INPUTS_VIEW_SETTINGS, ...settings };
    }

    /** The texture the last `encode` wrote (it is replaced when the screen's size changes), or null before the first. */
    get texture(): GPUTexture | null {
        return this.target;
    }

    /**
     * Records the view of `channel` into `encoder` and returns the texture it
     * writes, the screen's size. One channel per submitted encoder (the
     * channel's settings are written with `queue.writeBuffer`). Throws if the
     * screen doesn't have the channel.
     */
    encode(encoder: GPUCommandEncoder, screen: FrameScreen, channel: ScreenChannel): GPUTexture {
        const source = screen[channel];
        if (!source) throw new Error(`@pmndrs/gi: the screen has no '${channel}' channel to show`);
        const kind = sourceKind(channel, source);
        const kernel = this.kernelFor(channel, kind);
        const target = this.targetFor(screen.width, screen.height);

        // GiScreenViewSettings, word by word.
        const floats = new Float32Array(this.settingsData);
        const words = new Uint32Array(this.settingsData);
        const planes = this.settings.depthPlanes;
        floats[0] = this.settings.velocityScale;
        floats[1] = this.settings.depthDecades;
        words[2] = screen.normalEncoding === 'snorm-float' ? 1 : 0;
        words[3] = screen.reversedDepth === true ? 1 : 0;
        floats[4] = planes?.near ?? 0;
        floats[5] = planes?.far ?? 0;
        floats[6] = this.settings.depthMidDistance;
        floats[7] = 0;
        this.device.queue.writeBuffer(kernel.settings, 0, this.settingsData);

        // Depth formats with stencil need the depth aspect named for a sampled view.
        const sourceView = source.format.startsWith('depth') ? source.createView({ aspect: 'depth-only' }) : source.createView();
        const bindGroup = this.device.createBindGroup({
            label: `gi/screen-view ${channel}`,
            layout: kernel.layout,
            entries: [
                { binding: 0, resource: sourceView },
                { binding: 1, resource: target.createView() },
                { binding: 2, resource: { buffer: kernel.settings } },
            ],
        });
        const pass = encoder.beginComputePass({ label: `gi/screen-view ${channel}` });
        pass.setPipeline(kernel.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(screen.width / WORKGROUP_SIZE), Math.ceil(screen.height / WORKGROUP_SIZE));
        pass.end();
        return target;
    }

    dispose(): void {
        this.target?.destroy();
        this.target = null;
        for (const kernel of this.kernels.values()) kernel.settings.destroy();
        this.kernels.clear();
    }

    private kernelFor(channel: ScreenChannel, source: SourceKind): ChannelKernel {
        const key = `${channel}:${source.key}`;
        const existing = this.kernels.get(key);
        if (existing) return existing;
        const layout = this.device.createBindGroupLayout({
            label: `gi/screen-view ${key}`,
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: source.sampleType } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: SCREEN_INPUTS_VIEW_FORMAT } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
        const pipeline = this.device.createComputePipeline({
            label: `gi/screen-view ${key}`,
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            compute: { module: this.device.createShaderModule({ label: `gi/screen-view ${key}`, code: viewKernel(channel, source) }), entryPoint: 'main' },
        });
        const settings = this.device.createBuffer({ label: `gi/screen-view ${key} settings`, size: SETTINGS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const kernel = { pipeline, layout, settings };
        this.kernels.set(key, kernel);
        return kernel;
    }

    private targetFor(width: number, height: number): GPUTexture {
        const current = this.target;
        if (current && current.width === width && current.height === height) return current;
        current?.destroy();
        this.target = this.device.createTexture({
            label: 'gi/screen-view',
            size: { width, height },
            format: SCREEN_INPUTS_VIEW_FORMAT,
            // Sampled by the host to show it, copied by checks that read it back.
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        return this.target;
    }
}
