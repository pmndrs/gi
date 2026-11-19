import { nativeRcShader } from './shaders.ts';

type Triple = readonly [number, number, number];
export interface NativeRcOptions {
    bounds: { min: Triple; max: Triple };
    /** Fine grid dimensions. The next grids ceil-divide each axis by two. */
    grid: Triple;
    /** The three world-space intervals are [0,r), [r,4r), [4r,maxDistance]. */
    firstInterval: number;
    maxDistance: number;
    rayStride?: 1 | 2;
    /** Fraction of the previous estimate retained when a bin receives new rays. */
    history?: number;
    /** World-space origin offset along the surface normal, in metres. */
    normalBias?: number;
}
export interface NativeRcScreen {
    /** Unfiltered float textures, at the exact output size. xyz are world units, w>0 marks a surface. */
    worldPosition: GPUTextureView;
    /** Signed world-space unit normal in xyz. */
    worldNormal: GPUTextureView;
}
export interface NativeRcTransport {
    beginFrame(): void;
    /** Rays: two vec4f (origin.xyz, min distance; unit direction.xyz, max distance).
     * Results: vec4f (scene-linear outgoing radiance, hit distance; -1 on a sky miss). */
    encode(encoder: GPUCommandEncoder, job: { rays: GPUBuffer; results: GPUBuffer; count: number }): void;
}

const entries = ['begin', 'generate', 'deposit', 'resolveHistory', 'mergeFar', 'mergeMiddle', 'mergeNear', 'project', 'gather'] as const;
type Entry = typeof entries[number];
type Pipelines = Record<Entry, GPUComputePipeline>;

/** Counts and offsets of tightly packed directional bins, ordered near to far. */
export function nativeRcLayout(grid: Triple) {
    if (grid.some(value => !Number.isSafeInteger(value) || value < 1)) throw new RangeError('RC grid dimensions must be positive integers');
    let offset = 0;
    const levels = [0, 1, 2].map(level => {
        const dims = grid.map(size => Math.ceil(size / (2 ** level)));
        const cells = dims[0] * dims[1] * dims[2];
        const directions = 6 * (2 << level) ** 2;
        const bins = cells * directions;
        const base = offset;
        offset += bins;
        return { dims, cells, directions, bins, base };
    });
    if (!Number.isSafeInteger(offset) || offset > 0xffffffff) throw new RangeError('RC bin offsets exceed 32-bit addressing');
    return { levels, bins: offset };
}

/**
 * A renderer-independent bounded-volume Split RC prototype. The host owns
 * the GPUDevice, surface inputs, transport, encoder and queue submission.
 * Call resize() before the first encode(). The output is diffuse irradiance E
 * in linear HDR; a Lambertian material reflects albedo * E / pi.
 */
export class NativeRadianceCascades {
    readonly layout: ReturnType<typeof nativeRcLayout>;
    readonly rayStride: number;
    readonly probeBytes: number;
    private readonly persistent: GPUBuffer[] = [];
    private readonly uniforms: GPUBuffer;
    private readonly frame: GPUBuffer;
    private readonly observations: GPUBuffer;
    private readonly history: GPUBuffer;
    private readonly seen: GPUBuffer;
    private readonly merged: GPUBuffer;
    private readonly cube: GPUBuffer;
    private frameData: { width: number; height: number; count: number; rays: GPUBuffer; results: GPUBuffer; output: GPUTexture | null; view: GPUTextureView | null } | null = null;
    private bindings: { worldPosition: GPUTextureView; worldNormal: GPUTextureView; output: GPUTextureView; group: GPUBindGroup } | null = null;
    private clearHistory = true;
    private disposed = false;

    static async create(device: GPUDevice, options: NativeRcOptions): Promise<NativeRadianceCascades> {
        const layout = nativeRcLayout(options.grid);
        const { min, max } = options.bounds;
        if (min.some((v, i) => !Number.isFinite(v) || !Number.isFinite(max[i]) || max[i] <= v)) throw new RangeError('RC bounds must have positive finite extent');
        if (!(options.firstInterval > 0 && Number.isFinite(options.maxDistance) && options.maxDistance > 4 * options.firstInterval)) {
            throw new RangeError('RC maxDistance must be greater than four times firstInterval');
        }
        if (options.rayStride !== undefined && options.rayStride !== 1 && options.rayStride !== 2) throw new RangeError('RC rayStride must be 1 or 2');
        const retention = options.history ?? 0.8;
        if (!(retention >= 0 && retention < 1)) throw new RangeError('RC history must be in [0,1)');
        if (!Number.isFinite(options.normalBias ?? 0.004) || (options.normalBias ?? 0.004) < 0) throw new RangeError('RC normalBias must be finite and nonnegative');
        const maxBinding = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
        if (20 * layout.bins > maxBinding || Math.ceil(layout.bins / 64) > device.limits.maxComputeWorkgroupsPerDimension) {
            throw new RangeError('RC volume exceeds this adapter’s buffer or dispatch limits');
        }
        const visibility = GPUShaderStage.COMPUTE;
        const resources: GPUBindGroupLayoutEntry[] = [
            { binding: 0, visibility, buffer: { type: 'uniform' } },
            ...Array.from({ length: 8 }, (_, i): GPUBindGroupLayoutEntry => ({ binding: i + 1, visibility, buffer: { type: 'storage' } })),
            { binding: 9, visibility, texture: { sampleType: 'unfilterable-float' } },
            { binding: 10, visibility, texture: { sampleType: 'unfilterable-float' } },
            { binding: 11, visibility, storageTexture: { access: 'write-only', format: 'rgba16float' } },
        ];
        const groupLayout = device.createBindGroupLayout({ label: 'RC solver resources', entries: resources });
        const pipelineLayout = device.createPipelineLayout({ label: 'RC solver layout', bindGroupLayouts: [groupLayout] });
        const module = device.createShaderModule({ label: 'Bounded-volume Split RC', code: nativeRcShader });
        const programs = await Promise.all(entries.map(entryPoint => device.createComputePipelineAsync({
            label: `RC ${entryPoint}`, layout: pipelineLayout, compute: { module, entryPoint },
        })));
        const pipelines = Object.fromEntries(entries.map((entry, i) => [entry, programs[i]])) as Pipelines;
        return new NativeRadianceCascades(device, options, layout, groupLayout, pipelines);
    }

    private constructor(private readonly device: GPUDevice, private readonly options: NativeRcOptions,
        layout: ReturnType<typeof nativeRcLayout>, private readonly groupLayout: GPUBindGroupLayout, private readonly pipelines: Pipelines) {
        this.layout = layout;
        this.rayStride = options.rayStride ?? 2;
        const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        const allocate = (label: string, size: number, usage = storage) => {
            const buffer = device.createBuffer({ label: `RC ${label}`, size, usage });
            this.persistent.push(buffer);
            return buffer;
        };
        try {
            this.uniforms = allocate('parameters', 112, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            this.frame = allocate('frame', 4);
            this.observations = allocate('deposits', 20 * layout.bins);
            this.history = allocate('directional history', 16 * layout.bins);
            this.seen = allocate('last observation', 4 * layout.bins);
            this.merged = allocate('merged radiance', 16 * layout.bins);
            this.cube = allocate('diffuse irradiance per cell', 96 * layout.levels[0].cells);
        } catch (error) {
            for (const buffer of this.persistent) buffer.destroy();
            throw error;
        }
        this.probeBytes = this.observations.size + this.history.size + this.seen.size + this.merged.size + this.cube.size;
    }

    get outputTexture(): GPUTexture {
        const data = this.frameData;
        if (!data) throw new Error('Call resize() before accessing native RC output');
        if (!data.output) {
            data.output = this.device.createTexture({
                label: 'RC linear diffuse irradiance', size: [data.width, data.height], format: 'rgba16float',
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
            });
        }
        return data.output;
    }
    /** Zero for a host that always supplies its own storage view. */
    get ownedOutputBytes(): number { return this.frameData?.output ? this.frameData.width * this.frameData.height * 8 : 0; }
    get rayScratchBytes(): number { return this.frameData ? this.frameData.rays.size + this.frameData.results.size : 0; }

    resize(width: number, height: number) {
        if (this.disposed) throw new Error('Native RC has been disposed');
        if (this.frameData?.width === width && this.frameData.height === height) return;
        const limit = this.device.limits;
        if (![width, height].every(value => Number.isInteger(value) && value >= 1 && value <= limit.maxTextureDimension2D)) throw new RangeError('Invalid native RC screen dimensions');
        const count = Math.ceil(width / this.rayStride) * Math.ceil(height / this.rayStride);
        if (count * 32 > Math.min(limit.maxStorageBufferBindingSize, limit.maxBufferSize) || Math.ceil(count / 64) > limit.maxComputeWorkgroupsPerDimension) {
            throw new RangeError('Native RC ray batch exceeds this adapter’s buffer or dispatch limits');
        }
        const resources: GPUBuffer[] = [];
        try {
            const batch = (label: string, bytes: number) => {
                const b = this.device.createBuffer({ label, size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
                resources.push(b);
                return b;
            };
            const rays = batch('RC rays', count * 32), results = batch('RC hits', count * 16);
            const data = new ArrayBuffer(112);
            const f = new Float32Array(data), u = new Uint32Array(data);
            f.set(this.options.bounds.min);
            f.set(this.options.bounds.min.map((v, i) => 1 / (this.options.bounds.max[i] - v)), 4);
            u.set([width, height, this.rayStride, count], 8);
            f.set([this.options.firstInterval, this.options.maxDistance, this.options.history ?? 0.8, this.options.normalBias ?? 0.004], 12);
            this.layout.levels.forEach((level, i) => u.set([...level.dims, level.base], 16 + 4 * i));
            this.device.queue.writeBuffer(this.uniforms, 0, data);
            this.releaseScreen();
            this.frameData = { width, height, count, rays, results, output: null, view: null };
        } catch (error) {
            for (const resource of resources) resource.destroy();
            throw error;
        }
    }

    /** Invalidates probe history at the start of the next recorded frame. */
    reset() { this.clearHistory = true; }

    encode(encoder: GPUCommandEncoder, screen: NativeRcScreen, transport: NativeRcTransport, outputView?: GPUTextureView, timing?: GPUComputePassTimestampWrites) {
        const data = this.frameData;
        if (!data) throw new Error('Native RC encode requires a live, resized solver');
        const output = outputView ?? (data.view ??= this.outputTexture.createView());
        if (!this.bindings || this.bindings.worldPosition !== screen.worldPosition || this.bindings.worldNormal !== screen.worldNormal || this.bindings.output !== output) {
            const buffers = [this.uniforms, this.frame, data.rays, data.results, this.observations, this.history, this.seen, this.merged, this.cube];
            const entries: GPUBindGroupEntry[] = buffers.map((buffer, binding) => ({ binding, resource: { buffer } }));
            entries.push({ binding: 9, resource: screen.worldPosition }, { binding: 10, resource: screen.worldNormal }, { binding: 11, resource: output });
            const group = this.device.createBindGroup({ label: 'RC frame bindings', layout: this.groupLayout, entries });
            this.bindings = { ...screen, output, group };
        }
        if (this.clearHistory) {
            encoder.clearBuffer(this.frame);
            encoder.clearBuffer(this.history);
            encoder.clearBuffer(this.seen);
            this.clearHistory = false;
        }
        encoder.clearBuffer(this.observations);
        const dispatch = (pass: GPUComputePassEncoder, entry: Entry, groups: number, rows = 1) => {
            pass.setPipeline(this.pipelines[entry]);
            pass.dispatchWorkgroups(groups, rows);
        };
        const groups = (count: number) => Math.ceil(count / 64);
        transport.beginFrame();
        const pre = encoder.beginComputePass({ label: 'RC surface rays',
            timestampWrites: timing?.beginningOfPassWriteIndex === undefined ? undefined : { querySet: timing.querySet, beginningOfPassWriteIndex: timing.beginningOfPassWriteIndex } });
        pre.setBindGroup(0, this.bindings.group);
        dispatch(pre, 'begin', 1);
        dispatch(pre, 'generate', groups(data.count));
        pre.end();
        transport.encode(encoder, { rays: data.rays, results: data.results, count: data.count });
        const post = encoder.beginComputePass({ label: 'RC deposits, merge and irradiance',
            timestampWrites: timing?.endOfPassWriteIndex === undefined ? undefined : { querySet: timing.querySet, endOfPassWriteIndex: timing.endOfPassWriteIndex } });
        post.setBindGroup(0, this.bindings.group);
        dispatch(post, 'deposit', groups(data.count));
        dispatch(post, 'resolveHistory', groups(this.layout.bins));
        const levels = this.layout.levels;
        dispatch(post, 'mergeFar', groups(levels[2].bins));
        dispatch(post, 'mergeMiddle', groups(levels[1].bins));
        dispatch(post, 'mergeNear', groups(levels[0].bins));
        dispatch(post, 'project', groups(levels[0].cells));
        dispatch(post, 'gather', Math.ceil(data.width / 8), Math.ceil(data.height / 8));
        post.end();
    }

    private releaseScreen() {
        this.frameData?.rays.destroy();
        this.frameData?.results.destroy();
        this.frameData?.output?.destroy();
        this.frameData = null;
        this.bindings = null;
    }
    dispose() {
        if (this.disposed) return;
        this.releaseScreen();
        for (const buffer of this.persistent) buffer.destroy();
        this.disposed = true;
    }
}
