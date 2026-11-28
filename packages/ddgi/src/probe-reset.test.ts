import { describe, expect, it } from 'vitest';
import type { ComputeKernel } from '@pmndrs/gi-core/platform/contract/wgsl';
import type { GpuTimerScope } from '@pmndrs/gi-core/platform/contract/timing';
import { DDGITechnique } from './ddgi-technique.ts';
import { PROBE_STATE } from './layout.ts';
import { PendingProbeReset, probeResetFlags, ProbeResetPass, resetProbeDataTexel, type ProbeResetVolume } from './probe-reset.ts';
import { DDGI_PROBE_RESET_WGSL, PROBE_RESET_UNIFORM_BYTES, PROBE_RESET_UNIFORM_WORD, PROBE_RESET_WORKGROUP_SIZE } from './wgsl/probe-reset.ts';
import { wgslNameProblems } from '@pmndrs/bvh/wgsl-names';

describe('what resetProbes asks for (DV-12)', () => {
    it('asks for nothing with {}, no argument, or both flags false: each flag defaults to false', () => {
        expect(probeResetFlags(undefined)).toBeNull();
        expect(probeResetFlags({})).toBeNull();
        expect(probeResetFlags({ offsets: false, states: false })).toBeNull();
    });

    it('asks for each flag on its own, or both', () => {
        expect(probeResetFlags({ offsets: true })).toEqual({ offsets: true, states: false });
        expect(probeResetFlags({ states: true })).toEqual({ offsets: false, states: true });
        expect(probeResetFlags({ offsets: true, states: true })).toEqual({ offsets: true, states: true });
        expect(probeResetFlags({ offsets: true, states: false })).toEqual({ offsets: true, states: false });
    });
});

describe('the reset waiting for the next frame', () => {
    it('stays empty after a request for nothing', () => {
        const pending = new PendingProbeReset();
        pending.request({});
        pending.request(undefined);
        pending.request({ offsets: false });
        expect(pending.pending).toBe(false);
        expect(pending.take()).toBeNull();
    });

    it('adds up the calls before a frame, and is empty once taken', () => {
        const pending = new PendingProbeReset();
        pending.request({ offsets: true });
        pending.request({ states: true });
        pending.request({});
        expect(pending.take()).toEqual({ offsets: true, states: true });
        expect(pending.pending).toBe(false);
        expect(pending.take()).toBeNull();
    });

    it('is forgotten by clear (prepare, resetHistory, loadField)', () => {
        const pending = new PendingProbeReset();
        pending.request({ states: true });
        pending.clear();
        expect(pending.take()).toBeNull();
    });
});

describe("a probe-data texel after a reset (the kernel's CPU twin)", () => {
    const offset: [number, number, number] = [0.25, -0.125, 0.4];
    const states = [PROBE_STATE.active, PROBE_STATE.offInsideGeometry, PROBE_STATE.offNoSurfaceNearby, PROBE_STATE.notUpdatedYet];

    it('offsets: every offset goes to 0 and every state stays', () => {
        for (const state of states) expect(resetProbeDataTexel({ offset, state }, { offsets: true, states: false })).toEqual({ offset: [0, 0, 0], state });
    });

    it('states: off probes come back on through "not updated yet"; active and not-updated-yet stay; offsets stay', () => {
        const flags = { offsets: false, states: true };
        expect(resetProbeDataTexel({ offset, state: PROBE_STATE.active }, flags)).toEqual({ offset, state: PROBE_STATE.active });
        expect(resetProbeDataTexel({ offset, state: PROBE_STATE.offInsideGeometry }, flags)).toEqual({ offset, state: PROBE_STATE.notUpdatedYet });
        expect(resetProbeDataTexel({ offset, state: PROBE_STATE.offNoSurfaceNearby }, flags)).toEqual({ offset, state: PROBE_STATE.notUpdatedYet });
        expect(resetProbeDataTexel({ offset, state: PROBE_STATE.notUpdatedYet }, flags)).toEqual({ offset, state: PROBE_STATE.notUpdatedYet });
    });

    it('both: offsets 0 and off probes back on', () => {
        expect(resetProbeDataTexel({ offset, state: PROBE_STATE.offInsideGeometry }, { offsets: true, states: true })).toEqual({ offset: [0, 0, 0], state: PROBE_STATE.notUpdatedYet });
        expect(resetProbeDataTexel({ offset, state: PROBE_STATE.active }, { offsets: true, states: true })).toEqual({ offset: [0, 0, 0], state: PROBE_STATE.active });
    });

    it('the WGSL uses the same states', () => {
        expect(DDGI_PROBE_RESET_WGSL).toContain(`DDGI_RESET_STATE_OFF_INSIDE_GEOMETRY: u32 = ${PROBE_STATE.offInsideGeometry}u`);
        expect(DDGI_PROBE_RESET_WGSL).toContain(`DDGI_RESET_STATE_OFF_NO_SURFACE_NEARBY: u32 = ${PROBE_STATE.offNoSurfaceNearby}u`);
        expect(DDGI_PROBE_RESET_WGSL).toContain(`DDGI_RESET_STATE_NOT_UPDATED_YET: u32 = ${PROBE_STATE.notUpdatedYet}u`);
    });
});

describe("ddgi/reset-probes' WGSL", () => {
    it('declares no reserved word and no name twice', () => {
        expect(wgslNameProblems(DDGI_PROBE_RESET_WGSL)).toEqual([]);
    });

    it('has every uniform field where PROBE_RESET_UNIFORM_WORD says, under WGSL layout rules', () => {
        const TYPES: Record<string, { size: number; align: number }> = {
            u32: { size: 4, align: 4 },
            i32: { size: 4, align: 4 },
            vec4f: { size: 16, align: 16 },
            vec4i: { size: 16, align: 16 },
        };
        const body = /struct DdgiProbeReset \{([\s\S]*?)\n\}/.exec(DDGI_PROBE_RESET_WGSL)?.[1];
        expect(body).toBeDefined();
        const members = [...(body ?? '').matchAll(/^\s*(\w+): (\w+),/gm)].map((match) => ({ name: match[1], type: match[2] }));
        let offset = 0;
        let structAlign = 0;
        const words: Record<string, number> = {};
        for (const member of members) {
            const type = TYPES[member.type];
            expect(type, `unknown type ${member.type}`).toBeDefined();
            offset = Math.ceil(offset / type.align) * type.align;
            words[member.name] = offset / 4;
            offset += type.size;
            structAlign = Math.max(structAlign, type.align);
        }
        expect(words).toEqual(PROBE_RESET_UNIFORM_WORD);
        expect(Math.ceil(offset / structAlign) * structAlign).toBe(PROBE_RESET_UNIFORM_BYTES);
    });
});

/** What the pass did to the fake device and encoder, in order. */
interface Recording {
    readonly calls: string[];
    readonly uniforms: Uint32Array[];
    readonly dispatches: [number, number][];
    texturesMade: number;
    buffersMade: number;
    destroyed: number;
}

function fakeTexture(width: number, height: number, layers: number, recording: Recording): GPUTexture {
    return { width, height, depthOrArrayLayers: layers, createView: () => ({}), destroy: () => recording.destroyed++ } as unknown as GPUTexture;
}

function fakeGpu(maxComputeWorkgroupsPerDimension = 65535): { device: GPUDevice; encoder: GPUCommandEncoder; recording: Recording } {
    const recording: Recording = { calls: [], uniforms: [], dispatches: [], texturesMade: 0, buffersMade: 0, destroyed: 0 };
    const device = {
        limits: { maxComputeWorkgroupsPerDimension },
        createTexture: (descriptor: GPUTextureDescriptor) => {
            recording.texturesMade++;
            const size = descriptor.size as GPUExtent3DDict;
            return fakeTexture(size.width, size.height ?? 1, size.depthOrArrayLayers ?? 1, recording);
        },
        createBuffer: () => {
            recording.buffersMade++;
            return { destroy: () => recording.destroyed++ };
        },
        createBindGroup: () => ({}),
        queue: {
            writeBuffer: (_buffer: GPUBuffer, _offset: number, data: Float32Array) => {
                recording.calls.push('writeBuffer');
                recording.uniforms.push(new Uint32Array(data.buffer.slice(0)));
            },
        },
    } as unknown as GPUDevice;
    const pass = {
        setBindGroup: () => recording.calls.push('setBindGroup'),
        dispatchWorkgroups: (x: number, y = 1) => {
            recording.calls.push('dispatch');
            recording.dispatches.push([x, y]);
        },
        end: () => recording.calls.push('end'),
    };
    const encoder = {
        copyTextureToTexture: (_source: unknown, _destination: unknown, size: GPUExtent3DDict) => recording.calls.push(`copy ${size.width}x${size.height}x${size.depthOrArrayLayers}`),
        beginComputePass: (descriptor: GPUComputePassDescriptor) => {
            recording.calls.push(`pass ${descriptor.label}`);
            return pass;
        },
    } as unknown as GPUCommandEncoder;
    return { device, encoder, recording };
}

const KERNEL = { ownLayouts: [{}], bind: () => {} } as unknown as ComputeKernel;
const TIMER = { computePass: () => undefined } as unknown as GpuTimerScope;
const VOLUMES: ProbeResetVolume[] = [
    { counts: [10, 4, 8], spacing: [1, 1, 1], origin: [-4.5, 0, -3.5], window: [0, 0, 0], layerBase: 0, probeBase: 0 },
    { counts: [4, 4, 4], spacing: [2, 2, 2], origin: [-3, 0, -3], window: [1, 0, 3], layerBase: 4, probeBase: 320 },
];

describe('the reset pass', () => {
    it('copies the probe data, then dispatches every probe of every volume in one pass, with the flags in the uniforms', () => {
        const { device, encoder, recording } = fakeGpu();
        const textures = { probeData: fakeTexture(10, 8, 8, recording), points: fakeTexture(512, 1, 1, recording) };
        const pass = new ProbeResetPass(device, KERNEL, textures);
        pass.encode(encoder, TIMER, VOLUMES, { offsets: true, states: false });
        expect(recording.calls).toEqual(['copy 10x8x8', 'writeBuffer', 'writeBuffer', 'pass ddgi/reset-probes', 'setBindGroup', 'dispatch', 'setBindGroup', 'dispatch', 'end']);
        // 320 probes and 64 probes, 64 a workgroup.
        expect(recording.dispatches).toEqual([
            [320 / PROBE_RESET_WORKGROUP_SIZE, 1],
            [1, 1],
        ]);
        const [first, second] = recording.uniforms;
        expect(first[PROBE_RESET_UNIFORM_WORD.resetOffsets]).toBe(1);
        expect(first[PROBE_RESET_UNIFORM_WORD.resetStates]).toBe(0);
        expect(Array.from(second.subarray(PROBE_RESET_UNIFORM_WORD.window, PROBE_RESET_UNIFORM_WORD.window + 3))).toEqual([1, 0, 3]);
        expect(second[PROBE_RESET_UNIFORM_WORD.layerBase]).toBe(4);
        expect(second[PROBE_RESET_UNIFORM_WORD.probeBase]).toBe(320);
        expect(new Float32Array(second.buffer)[PROBE_RESET_UNIFORM_WORD.spacing]).toBe(2);
    });

    it('writes each flag as asked, and makes its texture and buffers once', () => {
        const { device, encoder, recording } = fakeGpu();
        const textures = { probeData: fakeTexture(10, 8, 8, recording), points: fakeTexture(512, 1, 1, recording) };
        const pass = new ProbeResetPass(device, KERNEL, textures);
        expect(pass.bytes).toBe(0);
        pass.encode(encoder, TIMER, VOLUMES, { offsets: false, states: true });
        pass.encode(encoder, TIMER, VOLUMES, { offsets: true, states: true });
        expect(recording.texturesMade).toBe(1);
        expect(recording.buffersMade).toBe(2);
        expect(pass.bytes).toBe(10 * 8 * 8 * 8 + 2 * PROBE_RESET_UNIFORM_BYTES);
        const flags = recording.uniforms.map((words) => [words[PROBE_RESET_UNIFORM_WORD.resetOffsets], words[PROBE_RESET_UNIFORM_WORD.resetStates]]);
        expect(flags).toEqual([
            [0, 1],
            [0, 1],
            [1, 1],
            [1, 1],
        ]);
        pass.destroy();
        expect(recording.destroyed).toBe(3);
        expect(pass.bytes).toBe(0);
    });

    it('spreads a big volume over two dispatch dimensions', () => {
        const { device, encoder, recording } = fakeGpu(4);
        const textures = { probeData: fakeTexture(10, 8, 8, recording), points: fakeTexture(512, 1, 1, recording) };
        new ProbeResetPass(device, KERNEL, textures).encode(encoder, TIMER, [VOLUMES[0]], { offsets: true, states: false });
        expect(recording.dispatches).toEqual([[4, 2]]);
    });
});

describe('DDGITechnique.resetProbes', () => {
    it('does nothing before prepare, whatever it is asked', () => {
        const technique = new DDGITechnique({ volume: { center: [0, 0, 0], size: [2, 2, 2], spacing: 1 } });
        expect(() => technique.resetProbes()).not.toThrow();
        expect(() => technique.resetProbes({})).not.toThrow();
        expect(() => technique.resetProbes({ offsets: true, states: true })).not.toThrow();
    });
});
