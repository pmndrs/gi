/**
 * `DDGITechnique.resetProbes` (DV-12, docs/divergence.md): separate resets
 * for the probes' relocation offsets and their classification states, which
 * keep what the probes have learned (irradiance, distances, history). The
 * CPU half: what a request means, the request waiting for the next frame,
 * the per-volume uniforms, and the pass that records `ddgi/reset-probes`
 * (wgsl/probe-reset.ts, which says what the kernel does to each probe).
 *
 * `resetHistory()` is the reset that forgets everything; it stays as the
 * platform's engine-neutral contract has it. Offsets and states are DDGI's,
 * so these resets are the DDGI technique's own.
 */

import type { ComputeKernel } from '@pmndrs/gi-core/platform/contract/wgsl';
import type { GpuTimerScope } from '@pmndrs/gi-core/platform/contract/timing';
import { PROBE_DATA_FORMAT, PROBE_STATE } from './layout.ts';
import { PROBE_RESET_UNIFORM_BYTES, PROBE_RESET_UNIFORM_WORD, PROBE_RESET_UNIFORM_WORDS, PROBE_RESET_WORKGROUP_SIZE } from './wgsl/probe-reset.ts';

/**
 * What `resetProbes` resets. Each flag defaults to false, so `{}` (or no
 * argument) resets nothing.
 */
export interface ProbeResetRequest {
    /** Every probe's relocation offset back to 0: relocation finds them again over its usual updates. */
    readonly offsets?: boolean;
    /** Every probe classification switched off comes back on (through "not updated yet"): classification decides again. */
    readonly states?: boolean;
}

/** A request with both flags settled. */
export interface ProbeResetFlags {
    readonly offsets: boolean;
    readonly states: boolean;
}

/** The flags a request asks for, or null when it asks for nothing (`{}`, both false, or no argument). */
export function probeResetFlags(what: ProbeResetRequest | undefined): ProbeResetFlags | null {
    const offsets = what?.offsets === true;
    const states = what?.states === true;
    return offsets || states ? { offsets, states } : null;
}

/**
 * The reset waiting for the next `encode`, as `resetHistory` waits with its
 * flag. Several calls before a frame add up: `{ offsets }` then `{ states }`
 * is both.
 */
export class PendingProbeReset {
    private offsets = false;
    private states = false;

    /** Adds a request; one that asks for nothing changes nothing. */
    request(what: ProbeResetRequest | undefined): void {
        const flags = probeResetFlags(what);
        if (!flags) return;
        this.offsets ||= flags.offsets;
        this.states ||= flags.states;
    }

    get pending(): boolean {
        return this.offsets || this.states;
    }

    /** The flags to record this frame (null when nothing waits), and nothing waits after. */
    take(): ProbeResetFlags | null {
        const flags = this.pending ? { offsets: this.offsets, states: this.states } : null;
        this.clear();
        return flags;
    }

    /** Forgets a waiting request (a new field: `prepare`, `resetHistory`, `loadField`). */
    clear(): void {
        this.offsets = false;
        this.states = false;
    }
}

/**
 * One probe-data texel after a reset: the CPU twin of the kernel's rule
 * (wgsl/probe-reset.ts), for tests and the GPU check. `state` as stored
 * (PROBE_STATE).
 */
export function resetProbeDataTexel(
    texel: { readonly offset: readonly [number, number, number]; readonly state: number },
    flags: ProbeResetFlags,
): { offset: [number, number, number]; state: number } {
    const offset: [number, number, number] = flags.offsets ? [0, 0, 0] : [...texel.offset];
    const off = texel.state === PROBE_STATE.offInsideGeometry || texel.state === PROBE_STATE.offNoSurfaceNearby;
    const state = flags.states && off ? PROBE_STATE.notUpdatedYet : texel.state;
    return { offset, state };
}

/** What the kernel needs of one volume: where its probes are and where its data sits. */
export interface ProbeResetVolume {
    /** Probes per axis. */
    readonly counts: readonly [number, number, number];
    readonly spacing: readonly [number, number, number];
    /** World position of the window's first probe (a fixed volume's first probe). */
    readonly origin: readonly [number, number, number];
    /** Lattice coordinates of the window's first probe (0, 0, 0 for a fixed volume; scroll.ts). */
    readonly window: readonly [number, number, number];
    /** The volume's first layer in the shared atlases, and its first point in the points view (volumes.ts). */
    readonly layerBase: number;
    readonly probeBase: number;
}

/** Fills one volume's `DdgiProbeReset` uniform block (`words` and `data` view the same buffer). */
export function writeProbeResetUniforms(data: Float32Array, words: Uint32Array, volume: ProbeResetVolume, flags: ProbeResetFlags): void {
    data.fill(0);
    data.set(volume.origin, PROBE_RESET_UNIFORM_WORD.origin);
    data.set(volume.spacing, PROBE_RESET_UNIFORM_WORD.spacing);
    // counts, window and layerBase are signed in the WGSL; they are never negative here.
    words.set(volume.counts, PROBE_RESET_UNIFORM_WORD.counts);
    words.set(volume.window, PROBE_RESET_UNIFORM_WORD.window);
    words[PROBE_RESET_UNIFORM_WORD.layerBase] = volume.layerBase;
    words[PROBE_RESET_UNIFORM_WORD.probeBase] = volume.probeBase;
    words[PROBE_RESET_UNIFORM_WORD.resetOffsets] = flags.offsets ? 1 : 0;
    words[PROBE_RESET_UNIFORM_WORD.resetStates] = flags.states ? 1 : 0;
}

/** WebGPU usage and stage bits; the globals don't exist in Node, where tests import this file. */
const TEXTURE_USAGE = { copyDst: 0x2, textureBinding: 0x4 } as const;
const BUFFER_USAGE = { copyDst: 0x8, uniform: 0x40 } as const;
const COMPUTE = 0x4;

/** `ddgi/reset-probes` group 1 (wgsl/probe-reset.ts): uniforms, the probe data before (read), the probe data and the points (written). */
export const PROBE_RESET_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi/reset-probes group 1',
    entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 2, visibility: COMPUTE, storageTexture: { access: 'write-only', format: PROBE_DATA_FORMAT, viewDimension: '2d-array' } },
        { binding: 3, visibility: COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float', viewDimension: '2d' } },
    ],
};

/** Bytes per probe-data texel (rgba16float). */
const PROBE_DATA_TEXEL_BYTES = 8;

/**
 * The pass, with what it needs made the first time a reset is recorded (a
 * copy of the probe-data texture, and a uniform buffer and bind group per
 * volume), kept until the field is released. One per `prepare`, as the
 * textures it binds are.
 */
export class ProbeResetPass {
    private copy: GPUTexture | null = null;
    private readonly uniforms: GPUBuffer[] = [];
    private readonly groups: GPUBindGroup[] = [];
    private readonly data = new Float32Array(PROBE_RESET_UNIFORM_WORDS);
    private readonly words = new Uint32Array(this.data.buffer);

    constructor(
        private readonly device: GPUDevice,
        private readonly kernel: ComputeKernel,
        private readonly textures: { readonly probeData: GPUTexture; readonly points: GPUTexture },
    ) {}

    /** GPU memory it holds: nothing until the first reset, then the copy and the uniforms. */
    get bytes(): number {
        const copy = this.copy;
        const copyBytes = copy ? copy.width * copy.height * copy.depthOrArrayLayers * PROBE_DATA_TEXEL_BYTES : 0;
        return copyBytes + this.uniforms.length * PROBE_RESET_UNIFORM_BYTES;
    }

    /**
     * Records the reset for every volume: the probe data copied, then one
     * timed pass with a dispatch per volume. Its writes land before the
     * frame's other DDGI passes, which are recorded after it.
     */
    encode(encoder: GPUCommandEncoder, timer: GpuTimerScope, volumes: readonly ProbeResetVolume[], flags: ProbeResetFlags): void {
        const { probeData } = this.textures;
        const copy = this.ensureResources(volumes.length);
        const size = { width: probeData.width, height: probeData.height, depthOrArrayLayers: probeData.depthOrArrayLayers };
        encoder.copyTextureToTexture({ texture: probeData }, { texture: copy }, size);
        volumes.forEach((volume, index) => {
            writeProbeResetUniforms(this.data, this.words, volume, flags);
            this.device.queue.writeBuffer(this.uniforms[index], 0, this.data);
        });
        const maxPerDimension = this.device.limits.maxComputeWorkgroupsPerDimension;
        const pass = encoder.beginComputePass({ label: 'ddgi/reset-probes', timestampWrites: timer.computePass('ddgi/reset-probes') });
        this.kernel.bind(pass);
        volumes.forEach((volume, index) => {
            const probes = volume.counts[0] * volume.counts[1] * volume.counts[2];
            const groups = Math.ceil(probes / PROBE_RESET_WORKGROUP_SIZE);
            const groupsX = Math.min(groups, maxPerDimension);
            pass.setBindGroup(1, this.groups[index]);
            pass.dispatchWorkgroups(groupsX, Math.ceil(groups / groupsX));
        });
        pass.end();
    }

    destroy(): void {
        this.copy?.destroy();
        this.copy = null;
        for (const buffer of this.uniforms) buffer.destroy();
        this.uniforms.length = 0;
        this.groups.length = 0;
    }

    private ensureResources(volumeCount: number): GPUTexture {
        const { probeData, points } = this.textures;
        const device = this.device;
        const copy =
            this.copy ??
            device.createTexture({
                label: 'ddgi reset-probes: probe data before',
                size: { width: probeData.width, height: probeData.height, depthOrArrayLayers: probeData.depthOrArrayLayers },
                format: PROBE_DATA_FORMAT,
                usage: TEXTURE_USAGE.copyDst | TEXTURE_USAGE.textureBinding,
                // Compatibility mode fixes the binding dimension at creation (see the atlases in ddgi-technique.ts).
                textureBindingViewDimension: '2d-array',
            });
        this.copy = copy;
        for (let index = this.uniforms.length; index < volumeCount; index++) {
            const uniforms = device.createBuffer({ label: `ddgi reset-probes uniforms (volume ${index})`, size: PROBE_RESET_UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
            this.uniforms.push(uniforms);
            this.groups.push(
                device.createBindGroup({
                    label: `ddgi/reset-probes group 1 (volume ${index})`,
                    layout: this.kernel.ownLayouts[0],
                    entries: [
                        { binding: 0, resource: { buffer: uniforms } },
                        { binding: 1, resource: copy.createView({ dimension: '2d-array' }) },
                        { binding: 2, resource: probeData.createView({ dimension: '2d-array' }) },
                        { binding: 3, resource: points.createView() },
                    ],
                }),
            );
        }
        return copy;
    }
}
