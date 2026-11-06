/**
 * The `ddgi/census` kernel: counts the probes in each state, for the
 * technique's counters ("probes active", "probes off: ...", "probes not
 * updated yet"). One thread per probe of the volume; each workgroup adds up
 * its own probes in workgroup atomics, then one thread per state adds that
 * into the census buffer. The buffer is copied to a staging buffer and read
 * back a few frames later, like GPU timings; nothing waits for it.
 *
 * Bindings (group 1): uniforms, the census buffer (1 storage buffer), and
 * the probe-data atlas (sampled).
 */

import { DDGI_COMMON_WGSL } from './common.ts';

export const CENSUS_WORKGROUP_SIZE = 64;
/** States are 0..3 (layout.ts, PROBE_STATE). */
const STATE_COUNT = 4;

export const DDGI_CENSUS_WGSL = /* wgsl */ `
${DDGI_COMMON_WGSL}

// ---- DDGI census kernel (techniques/ddgi) ----------------------------------------------

const DDGI_CENSUS_WORKGROUP_SIZE: u32 = ${CENSUS_WORKGROUP_SIZE}u;
const DDGI_STATE_COUNT: u32 = ${STATE_COUNT}u;

@group(1) @binding(0) var<uniform> ddgi: DdgiUniforms;
@group(1) @binding(1) var<storage, read_write> ddgiCensus: array<atomic<u32>>;
@group(1) @binding(2) var ddgiProbeData: texture_2d_array<f32>;

var<workgroup> ddgiStateCounts: array<atomic<u32>, DDGI_STATE_COUNT>;

@compute @workgroup_size(DDGI_CENSUS_WORKGROUP_SIZE)
fn ddgiCountStates(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    let counts = ddgiProbeCount();
    let probeIndex = (group.x + group.y * groups.x) * DDGI_CENSUS_WORKGROUP_SIZE + thread;
    if (probeIndex < counts.x * counts.y * counts.z) {
        let probe = ddgiProbeCoordinates(probeIndex, counts);
        let state = min(ddgiReadProbeData(ddgiProbeData, probe).state, DDGI_STATE_COUNT - 1u);
        atomicAdd(&ddgiStateCounts[state], 1u);
    }
    workgroupBarrier();
    if (thread < DDGI_STATE_COUNT) {
        atomicAdd(&ddgiCensus[thread], atomicLoad(&ddgiStateCounts[thread]));
    }
}
`;
