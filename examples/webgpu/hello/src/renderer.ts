/**
 * The hello renderer: a plain WebGPU forward renderer, no engine. It is what
 * an engine adapter does, in the fewest lines:
 *
 * 1. It owns its meshes, materials and passes; the GI platform only lends it
 *    a shading module (textures, samplers, a few `vec4f` uniforms and WGSL).
 * 2. It binds the module's resources as globals at group 1 with
 *    `globalsCode(1)`, re-reads `texture()` each frame (the textures change
 *    only when the platform prepares again), and refills the module's
 *    uniforms with `writeUniforms`.
 * 3. Its fragment shader calls `giIrradianceAt(...)` and shades
 *    `emission + albedo / π × (direct + GI)`.
 *
 * A depth prepass comes first, so the forward pass shades every pixel once
 * and GI is sampled once per pixel, not once per overlapping surface.
 *
 * Direct light from the scene's one point light is the renderer's own job,
 * with a cube shadow map; GI never includes light seen straight from a light
 * (it does include light from emissive surfaces).
 *
 * Only type imports from the library here, so GPU test pages can reuse this
 * file with the core loaded straight from source.
 */

import type { ShadingModule, ShadingParameter } from '@pmndrs/gi/core';
import { cameraMatrices, lookAt, multiply, normalMatrix, perspective, type CameraMatrices, type Vec3 } from './camera.ts';
import type { DescribedScene } from './scene.ts';
import { DEPTH_PREPASS_SHADER, DISPLAY_SHADER, FORWARD_BINDING, forwardShader, GI_GROUP, SHADOW_BINDING, SHADOW_FACES, SHADOW_SHADER, VIEW_MODE, type GiSource } from './shaders.ts';

/** Float offsets of `FrameUniforms` (shaders.ts). */
const FRAME_FLOAT = {
    viewProjection: 0,
    cameraPosition: 16,
    viewMode: 19,
    lightPosition: 20,
    lightRange: 23,
    lightIntensity: 24,
    lightEnabled: 27,
    fallbackIrradiance: 28,
    shadowNormalOffset: 31,
    flatIrradiance: 32,
    debugView: 35,
    inverseViewportSize: 36,
    // vec2f, then 8 bytes of padding up to the matrices' 16-byte alignment.
    shadowViewProjection: 40,
} as const;
const FRAME_FLOATS = FRAME_FLOAT.shadowViewProjection + 16 * SHADOW_FACES.length;

/** Float offsets of `DrawUniforms` (shaders.ts). */
const DRAW_FLOAT = {
    model: 0,
    // mat3x3f: three columns of vec3f, each padded to 16 bytes.
    normalMatrix: 16,
    albedo: 28,
    mirrored: 31,
    emission: 32,
} as const;
const DRAW_FLOATS = DRAW_FLOAT.emission + 4;
/** Each draw's uniforms sit at a multiple of the dynamic-offset alignment (256 on every device). */
const UNIFORM_SLOT_BYTES = 256;

const DEPTH_FORMAT: GPUTextureFormat = 'depth32float';
const SHADOW_FORMAT: GPUTextureFormat = 'depth32float';
/** Cube faces see 90°, so together they cover every direction. */
const SHADOW_FACE_FOV = Math.PI / 2;
const SHADOW_NEAR = 0.02;
const SHADOW_FAR = 100;

/** What the forward pass shows. */
export type RenderView =
    | { readonly mode: 'full' }
    /** albedo / π × GI only. */
    | { readonly mode: 'gi' }
    /** One of the GI technique's screen debug views, by id (`technique.debugViews`). */
    | { readonly mode: 'debug'; readonly debugView: number };

export interface ForwardRendererOptions {
    /**
     * Where irradiance comes from: the platform's shading module, or `null`
     * for a constant (`flatIrradiance`), which is GI off, or the baseline
     * when timing what GI sampling costs.
     */
    readonly shading: ShadingModule | null;
    /** The constant irradiance when `shading` is null. Default 0. */
    readonly flatIrradiance?: readonly [number, number, number];
    /** The HDR image's format. Default `rgba16float`; tests read back `rgba32float`. */
    readonly hdrFormat?: GPUTextureFormat;
    /** Pixels per side of each shadow cube face. Default 1024. */
    readonly shadowMapSize?: number;
    /** How far shadow lookups move off the surface, metres. Default 0.02. */
    readonly shadowNormalOffset?: number;
}

interface GpuMesh {
    readonly positions: GPUBuffer;
    readonly normals: GPUBuffer;
    readonly indices: GPUBuffer;
}

/** One run of a mesh's triangles with one material: the whole mesh, or one of its groups. */
interface Draw {
    readonly mesh: GpuMesh;
    readonly firstIndex: number;
    readonly indexCount: number;
    readonly uniformOffset: number;
}

/**
 * The module's bindings at group 1, as `globalsCode` numbers them: textures
 * and samplers from 0, then one uniform buffer when the module has `vec4f`
 * parameters (DDGI's has none since its version 2).
 */
function giBindGroupLayoutEntries(parameters: readonly ShadingParameter[]): GPUBindGroupLayoutEntry[] {
    const entries: GPUBindGroupLayoutEntry[] = [];
    for (const parameter of parameters) {
        if (parameter.kind === 'texture') {
            entries.push({ binding: entries.length, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: parameter.sampleType, viewDimension: parameter.dimension } });
        } else if (parameter.kind === 'sampler') {
            entries.push({ binding: entries.length, visibility: GPUShaderStage.FRAGMENT, sampler: { type: parameter.filtering === 'linear' ? 'filtering' : 'non-filtering' } });
        }
    }
    if (uniformFloatCount(parameters) > 0) entries.push({ binding: entries.length, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } });
    return entries;
}

function uniformFloatCount(parameters: readonly ShadingParameter[]): number {
    return parameters.filter((parameter) => parameter.kind === 'vec4f').length * 4;
}

/** The GI module's group: layout, samplers, the uniform buffer, and a bind group rebuilt when a texture changes. */
class GiBindings {
    readonly layout: GPUBindGroupLayout;
    private readonly samplers = new Map<string, GPUSampler>();
    private readonly uniformBuffer: GPUBuffer;
    private readonly uniformData: Float32Array<ArrayBuffer>;
    private group: GPUBindGroup | null = null;
    private boundTextures: GPUTexture[] = [];

    constructor(
        private readonly device: GPUDevice,
        readonly shading: ShadingModule,
    ) {
        this.layout = device.createBindGroupLayout({ label: 'hello gi', entries: giBindGroupLayoutEntries(shading.parameters) });
        for (const parameter of shading.parameters) {
            if (parameter.kind !== 'sampler') continue;
            this.samplers.set(
                parameter.name,
                device.createSampler({
                    label: `hello ${parameter.name}`,
                    magFilter: parameter.filtering,
                    minFilter: parameter.filtering,
                    addressModeU: parameter.addressMode,
                    addressModeV: parameter.addressMode,
                    addressModeW: parameter.addressMode,
                }),
            );
        }
        this.uniformData = new Float32Array(Math.max(uniformFloatCount(shading.parameters), 4));
        this.uniformBuffer = device.createBuffer({ label: 'hello gi uniforms', size: this.uniformData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }

    /** This frame's bind group: uniforms refreshed, textures re-read (new group only if one changed). */
    update(): GPUBindGroup {
        this.shading.writeUniforms(this.uniformData);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);
        const textures = this.shading.parameters.flatMap((parameter) => (parameter.kind === 'texture' ? [parameter.texture()] : []));
        const changed = textures.length !== this.boundTextures.length || textures.some((texture, index) => texture !== this.boundTextures[index]);
        if (this.group && !changed) return this.group;

        const entries: GPUBindGroupEntry[] = [];
        for (const parameter of this.shading.parameters) {
            if (parameter.kind === 'texture') {
                // Explicit dimension: a one-layer array texture's default view would be 2d.
                entries.push({ binding: entries.length, resource: parameter.texture().createView({ dimension: parameter.dimension }) });
            } else if (parameter.kind === 'sampler') {
                entries.push({ binding: entries.length, resource: this.samplers.get(parameter.name)! });
            }
        }
        if (uniformFloatCount(this.shading.parameters) > 0) entries.push({ binding: entries.length, resource: { buffer: this.uniformBuffer } });
        this.group = this.device.createBindGroup({ label: 'hello gi', layout: this.layout, entries });
        this.boundTextures = textures;
        return this.group;
    }

    destroy(): void {
        this.uniformBuffer.destroy();
    }
}

export class ForwardRenderer {
    readonly hdrFormat: GPUTextureFormat;
    /** The forward pass's output, linear HDR (COPY_SRC, so tests can read it). */
    hdrTexture: GPUTexture | null = null;
    width = 0;
    height = 0;

    private readonly device: GPUDevice;
    private readonly scene: DescribedScene;
    private readonly options: ForwardRendererOptions;
    private readonly meshes: GpuMesh[];
    private readonly draws: Draw[];
    private readonly drawUniforms: GPUBuffer;
    private readonly frameUniforms: GPUBuffer;
    private readonly frameData = new Float32Array(FRAME_FLOATS);
    private readonly frameWords = new Uint32Array(this.frameData.buffer);
    private readonly forwardLayout: GPUBindGroupLayout;
    private readonly forwardGroup: GPUBindGroup;
    private readonly gi: GiBindings | null;
    private forwardPipeline: GPURenderPipeline;
    private readonly depthPipeline: GPURenderPipeline;
    private giVersion: number;
    private readonly shadowMap: GPUTexture;
    private readonly shadowFaceUniforms: GPUBuffer;
    private readonly shadowGroup: GPUBindGroup;
    private readonly shadowPipeline: GPURenderPipeline;
    private shadowsDrawn = false;
    private depthTexture: GPUTexture | null = null;
    private display: { pipeline: GPURenderPipeline; format: GPUTextureFormat; uniforms: GPUBuffer; group: GPUBindGroup | null } | null = null;

    constructor(device: GPUDevice, scene: DescribedScene, options: ForwardRendererOptions) {
        this.device = device;
        this.scene = scene;
        this.options = options;
        this.hdrFormat = options.hdrFormat ?? 'rgba16float';

        // Geometry: one vertex buffer per attribute, 32-bit indices.
        const buffer = (label: string, data: Float32Array | Uint32Array, usage: number): GPUBuffer => {
            const created = device.createBuffer({ label, size: Math.max(data.byteLength, 4), usage: usage | GPUBufferUsage.COPY_DST });
            device.queue.writeBuffer(created, 0, data as Float32Array<ArrayBuffer>);
            return created;
        };
        this.meshes = scene.data.meshes.map((mesh) => ({
            positions: buffer(`hello ${mesh.name} positions`, mesh.positions, GPUBufferUsage.VERTEX),
            normals: buffer(`hello ${mesh.name} normals`, mesh.normals, GPUBufferUsage.VERTEX),
            indices: buffer(`hello ${mesh.name} indices`, mesh.indices, GPUBufferUsage.INDEX),
        }));

        // One draw per run of triangles with one material: the whole mesh, or
        // each of its groups with the material of its slot (a wall and its
        // glass pane). Each draw has a uniform slot: transform, normal matrix
        // and material colour. The scene is static, so written once.
        const runs = scene.data.instances.flatMap((instance) => {
            const mesh = scene.data.meshes[instance.mesh];
            const slots = instance.materials ?? [instance.material];
            const groups = mesh.groups ?? [{ start: 0, count: mesh.indices.length, materialSlot: 0 }];
            return groups.map((group) => ({ instance, group, material: scene.rendererMaterials[slots[group.materialSlot]] }));
        });
        const drawData = new Float32Array((runs.length * UNIFORM_SLOT_BYTES) / 4);
        const drawWords = new Uint32Array(drawData.buffer);
        this.draws = runs.map(({ instance, group, material }, index) => {
            const base = (index * UNIFORM_SLOT_BYTES) / 4;
            const normals = normalMatrix(instance.transform);
            drawData.set(instance.transform, base + DRAW_FLOAT.model);
            drawData.set(normals.columns, base + DRAW_FLOAT.normalMatrix);
            drawData.set(material.albedo, base + DRAW_FLOAT.albedo);
            drawWords[base + DRAW_FLOAT.mirrored] = normals.mirrored ? 1 : 0;
            drawData.set(material.emission, base + DRAW_FLOAT.emission);
            return { mesh: this.meshes[instance.mesh], firstIndex: group.start, indexCount: group.count, uniformOffset: index * UNIFORM_SLOT_BYTES };
        });
        this.drawUniforms = buffer('hello draw uniforms', drawData, GPUBufferUsage.UNIFORM);
        this.frameUniforms = device.createBuffer({ label: 'hello frame uniforms', size: this.frameData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

        // The point light's cube shadow map: six layers of one depth texture, each with its own view-projection.
        const shadowSize = options.shadowMapSize ?? 1024;
        this.shadowMap = device.createTexture({
            label: 'hello shadow map',
            size: { width: shadowSize, height: shadowSize, depthOrArrayLayers: SHADOW_FACES.length },
            format: SHADOW_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        const lightPosition: Vec3 = scene.pointLight?.position ?? [0, 0, 0];
        const faceMatrices = SHADOW_FACES.map((face) => {
            const target: Vec3 = [lightPosition[0] + face.direction[0], lightPosition[1] + face.direction[1], lightPosition[2] + face.direction[2]];
            return multiply(perspective(SHADOW_FACE_FOV, 1, SHADOW_NEAR, SHADOW_FAR), lookAt(lightPosition, target, face.up));
        });
        const faceData = new Float32Array((SHADOW_FACES.length * UNIFORM_SLOT_BYTES) / 4);
        faceMatrices.forEach((matrix, face) => {
            faceData.set(matrix, (face * UNIFORM_SLOT_BYTES) / 4);
            this.frameData.set(matrix, FRAME_FLOAT.shadowViewProjection + face * 16);
        });
        this.shadowFaceUniforms = buffer('hello shadow faces', faceData, GPUBufferUsage.UNIFORM);

        const shadowLayout = device.createBindGroupLayout({
            label: 'hello shadow',
            entries: [
                { binding: SHADOW_BINDING.face, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } },
                { binding: SHADOW_BINDING.draw, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: DRAW_FLOATS * 4 } },
            ],
        });
        this.shadowGroup = device.createBindGroup({
            label: 'hello shadow',
            layout: shadowLayout,
            entries: [
                { binding: SHADOW_BINDING.face, resource: { buffer: this.shadowFaceUniforms, size: 64 } },
                { binding: SHADOW_BINDING.draw, resource: { buffer: this.drawUniforms, size: DRAW_FLOATS * 4 } },
            ],
        });
        const shadowModule = device.createShaderModule({ label: 'hello shadow', code: SHADOW_SHADER });
        this.shadowPipeline = device.createRenderPipeline({
            label: 'hello shadow',
            layout: device.createPipelineLayout({ bindGroupLayouts: [shadowLayout] }),
            vertex: { module: shadowModule, entryPoint: 'helloShadowVertex', buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
            // Slope-scaled bias keeps lit surfaces from shadowing themselves at grazing angles.
            depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2 },
        });

        this.forwardLayout = device.createBindGroupLayout({
            label: 'hello forward',
            entries: [
                { binding: FORWARD_BINDING.frame, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: FORWARD_BINDING.draw, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: DRAW_FLOATS * 4 } },
                { binding: FORWARD_BINDING.shadowMap, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
                { binding: FORWARD_BINDING.shadowSampler, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
            ],
        });
        this.forwardGroup = device.createBindGroup({
            label: 'hello forward',
            layout: this.forwardLayout,
            entries: [
                { binding: FORWARD_BINDING.frame, resource: { buffer: this.frameUniforms } },
                { binding: FORWARD_BINDING.draw, resource: { buffer: this.drawUniforms, size: DRAW_FLOATS * 4 } },
                { binding: FORWARD_BINDING.shadowMap, resource: this.shadowMap.createView({ dimension: '2d-array' }) },
                {
                    binding: FORWARD_BINDING.shadowSampler,
                    // Linear comparison filtering: each lookup averages a 2x2 block of depth tests.
                    resource: device.createSampler({ label: 'hello shadow', compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' }),
                },
            ],
        });

        const depthModule = device.createShaderModule({ label: 'hello depth prepass', code: DEPTH_PREPASS_SHADER });
        this.depthPipeline = device.createRenderPipeline({
            label: 'hello depth prepass',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.forwardLayout] }),
            vertex: { module: depthModule, entryPoint: 'helloDepthVertex', buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
            primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' },
            depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
        });

        this.gi = options.shading ? new GiBindings(device, options.shading) : null;
        this.giVersion = options.shading?.version ?? 0;
        this.forwardPipeline = this.createForwardPipeline();
    }

    /** Recompiles the forward pass when the shading module's code or parameters change shape. */
    private createForwardPipeline(): GPURenderPipeline {
        const device = this.device;
        const source: GiSource = this.gi ? { kind: 'module', globalsCode: this.gi.shading.globalsCode(GI_GROUP) } : { kind: 'flat' };
        const module = device.createShaderModule({ label: `hello forward (${source.kind} GI)`, code: forwardShader(source) });
        const bindGroupLayouts = this.gi ? [this.forwardLayout, this.gi.layout] : [this.forwardLayout];
        return device.createRenderPipeline({
            label: `hello forward (${source.kind} GI)`,
            layout: device.createPipelineLayout({ bindGroupLayouts }),
            vertex: {
                module,
                entryPoint: 'helloVertex',
                buffers: [
                    { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
                    { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] },
                ],
            },
            fragment: { module, entryPoint: 'helloFragment', targets: [{ format: this.hdrFormat }] },
            // Both sides of every surface are drawn, as rays see both; the fragment shader turns the normal towards the eye.
            primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' },
            // Only the surface the depth prepass kept passes, so each pixel is shaded once.
            depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'equal' },
        });
    }

    /** (Re)creates the HDR and depth targets. */
    resize(width: number, height: number): void {
        if (width === this.width && height === this.height && this.hdrTexture) return;
        this.hdrTexture?.destroy();
        this.depthTexture?.destroy();
        this.width = width;
        this.height = height;
        this.hdrTexture = this.device.createTexture({
            label: 'hello hdr',
            size: { width, height },
            format: this.hdrFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        this.depthTexture = this.device.createTexture({ label: 'hello depth', size: { width, height }, format: DEPTH_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT });
        if (this.display) this.display.group = null;
    }

    /** The camera for a bookmark at the current size. */
    camera(pose: Parameters<typeof cameraMatrices>[0]): CameraMatrices {
        return cameraMatrices(pose, this.width / this.height);
    }

    /**
     * Records the depth prepass and the forward pass (and, the first time,
     * the shadow map) into `encoder`. Call after `gi.encode` in the same
     * encoder, so this frame's GI updates land before the pixels read them.
     * One call per renderer per submit: the frame uniforms are written with
     * `queue.writeBuffer`, which lands before the whole submit.
     * `timestampWrites` times the forward pass alone.
     */
    render(encoder: GPUCommandEncoder, camera: CameraMatrices, view: RenderView, timestampWrites?: GPURenderPassTimestampWrites): void {
        if (!this.hdrTexture || !this.depthTexture) throw new Error('call resize(width, height) before render');
        if (!this.shadowsDrawn) {
            this.renderShadows(encoder);
            this.shadowsDrawn = true;
        }
        if (this.gi && this.gi.shading.version !== this.giVersion) {
            this.giVersion = this.gi.shading.version;
            this.forwardPipeline = this.createForwardPipeline();
        }
        this.writeFrameUniforms(camera, view);
        const giGroup = this.gi?.update() ?? null;

        const depthView = this.depthTexture.createView();
        const prepass = encoder.beginRenderPass({
            label: 'hello depth prepass',
            colorAttachments: [],
            depthStencilAttachment: { view: depthView, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
        });
        prepass.setPipeline(this.depthPipeline);
        for (const draw of this.draws) {
            prepass.setBindGroup(0, this.forwardGroup, [draw.uniformOffset]);
            prepass.setVertexBuffer(0, draw.mesh.positions);
            prepass.setIndexBuffer(draw.mesh.indices, 'uint32');
            prepass.drawIndexed(draw.indexCount, 1, draw.firstIndex);
        }
        prepass.end();

        const pass = encoder.beginRenderPass({
            label: 'hello forward',
            colorAttachments: [{ view: this.hdrTexture.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
            // Load and store rather than depthReadOnly: Safari 26 doesn't show a
            // read-only depth attachment's contents to the pass, so the 'equal'
            // test failed everywhere and the picture was black.
            depthStencilAttachment: { view: depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
            timestampWrites,
        });
        pass.setPipeline(this.forwardPipeline);
        if (giGroup) pass.setBindGroup(GI_GROUP, giGroup);
        for (const draw of this.draws) {
            pass.setBindGroup(0, this.forwardGroup, [draw.uniformOffset]);
            pass.setVertexBuffer(0, draw.mesh.positions);
            pass.setVertexBuffer(1, draw.mesh.normals);
            pass.setIndexBuffer(draw.mesh.indices, 'uint32');
            pass.drawIndexed(draw.indexCount, 1, draw.firstIndex);
        }
        pass.end();
    }

    /** Tone-maps the HDR image into `target` (a canvas texture, say), at 2^`exposure` EV. */
    present(encoder: GPUCommandEncoder, target: GPUTexture, exposure = 0): void {
        if (!this.hdrTexture) throw new Error('call resize(width, height) and render before present');
        const device = this.device;
        if (!this.display || this.display.format !== target.format) {
            const module = device.createShaderModule({ label: 'hello display', code: DISPLAY_SHADER });
            this.display?.uniforms.destroy();
            this.display = {
                pipeline: device.createRenderPipeline({
                    label: 'hello display',
                    layout: 'auto',
                    vertex: { module, entryPoint: 'displayVertex' },
                    fragment: { module, entryPoint: 'displayFragment', targets: [{ format: target.format }] },
                }),
                format: target.format,
                uniforms: device.createBuffer({ label: 'hello display', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
                group: null,
            };
        }
        const display = this.display;
        device.queue.writeBuffer(display.uniforms, 0, new Float32Array([2 ** exposure, 0, 0, 0]));
        display.group ??= device.createBindGroup({
            layout: display.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.hdrTexture.createView() },
                { binding: 1, resource: { buffer: display.uniforms } },
            ],
        });
        const pass = encoder.beginRenderPass({ label: 'hello display', colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store' }] });
        pass.setPipeline(display.pipeline);
        pass.setBindGroup(0, display.group);
        pass.draw(3);
        pass.end();
    }

    destroy(): void {
        for (const mesh of this.meshes) for (const gpuBuffer of [mesh.positions, mesh.normals, mesh.indices]) gpuBuffer.destroy();
        for (const gpuBuffer of [this.drawUniforms, this.frameUniforms, this.shadowFaceUniforms]) gpuBuffer.destroy();
        this.display?.uniforms.destroy();
        this.shadowMap.destroy();
        this.hdrTexture?.destroy();
        this.depthTexture?.destroy();
        this.gi?.destroy();
    }

    private renderShadows(encoder: GPUCommandEncoder): void {
        // Without a point light the map is never read, but it must exist for the bind group.
        for (let face = 0; face < SHADOW_FACES.length; face++) {
            const pass = encoder.beginRenderPass({
                label: `hello shadow ${SHADOW_FACES[face].name}`,
                colorAttachments: [],
                depthStencilAttachment: {
                    view: this.shadowMap.createView({ dimension: '2d', baseArrayLayer: face, arrayLayerCount: 1 }),
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                    depthClearValue: 1,
                },
            });
            if (this.scene.pointLight) {
                pass.setPipeline(this.shadowPipeline);
                for (const draw of this.draws) {
                    pass.setBindGroup(0, this.shadowGroup, [face * UNIFORM_SLOT_BYTES, draw.uniformOffset]);
                    pass.setVertexBuffer(0, draw.mesh.positions);
                    pass.setIndexBuffer(draw.mesh.indices, 'uint32');
                    pass.drawIndexed(draw.indexCount, 1, draw.firstIndex);
                }
            }
            pass.end();
        }
    }

    private writeFrameUniforms(camera: CameraMatrices, view: RenderView): void {
        const data = this.frameData;
        const words = this.frameWords;
        const light = this.scene.pointLight;
        data.set(camera.viewProjection, FRAME_FLOAT.viewProjection);
        data.set(camera.position, FRAME_FLOAT.cameraPosition);
        words[FRAME_FLOAT.viewMode] = VIEW_MODE[view.mode];
        words[FRAME_FLOAT.debugView] = view.mode === 'debug' ? view.debugView : 0;
        data.set(light?.position ?? [0, 0, 0], FRAME_FLOAT.lightPosition);
        data[FRAME_FLOAT.lightRange] = light?.range ?? 0;
        data.set(light?.intensity ?? [0, 0, 0], FRAME_FLOAT.lightIntensity);
        words[FRAME_FLOAT.lightEnabled] = light ? 1 : 0;
        data.set(this.scene.fallbackIrradiance, FRAME_FLOAT.fallbackIrradiance);
        data[FRAME_FLOAT.shadowNormalOffset] = this.options.shadowNormalOffset ?? 0.02;
        data.set(this.options.flatIrradiance ?? [0, 0, 0], FRAME_FLOAT.flatIrradiance);
        data.set([1 / this.width, 1 / this.height], FRAME_FLOAT.inverseViewportSize);
        this.device.queue.writeBuffer(this.frameUniforms, 0, data);
    }
}
