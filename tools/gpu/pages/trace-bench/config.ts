/**
 * Settings of the tracer benchmark (T0.10 part B), in one place.
 *
 * The page is chosen and tuned with query parameters (see `parseBenchConfig`),
 * but every default and every number the design fixed lives here.
 */

/**
 * The scenes the design benchmarks. Ids are scene manifest ids, or one of
 * {@link BENCH_SCENE_SUBSETS}.
 */
export const BENCH_SCENE_IDS = ['sponza', 'bistro-exterior', 'bistro-interior', 'furnished-room'] as const;
export type BenchSceneId = (typeof BENCH_SCENE_IDS)[number];

/**
 * Bench scenes that load only some parts of a manifest, so rows stay
 * comparable when a manifest grows. `sponza-base` is Intel's base package
 * alone (3.75 M triangles): what the `sponza` rows measured until T1.10 added
 * the curtains to the manifest. From then on `sponza` rows mean base plus
 * curtains; compare older `sponza` rows with `sponza-base`.
 */
export const BENCH_SCENE_SUBSETS: Readonly<Record<string, { readonly manifest: string; readonly assets: readonly string[] }>> = {
    'sponza-base': { manifest: 'sponza', assets: ['sponza-intel'] },
};

/** Pseudo-scene for the top-level rebuild timing: many instances of one small mesh. */
export const SYNTHETIC_SCENE_ID = 'synthetic-10k';
export const SYNTHETIC_INSTANCE_COUNT = 10_000;

/**
 * - `ours-tinybvh`: tinybvh (WASM, in our worker) builds each mesh's tree, traced by our watertight walk.
 *   Our own SAH mesh builder (`ours-wt` in rows before T1.3) was deleted once B0's checks passed.
 * - `ours-tmbuild`: three-mesh-bvh builds each mesh's tree, converted to our layout, traced by our watertight walk.
 * - `platform`: the same as `ours-tinybvh`, but through the platform API (`createGI`, its scene API, `queries.rays`).
 * - `tmbvh`: three-mesh-bvh's own trees and its own WebGPU walk.
 * - `ours-lbvh`: the GPU LBVH (B1, `@pmndrs/bvh`) builds every mesh's tree on the GPU in one batch, timed
 *   with GPU timestamps; the trees are read back and traced by our watertight walk. Needs subgroups.
 * - `ours-hploc`: the same with the GPU H-PLOC (B2). Also builds the scene once more with tinybvh's
 *   codes, tie rule and one-triangle leaves, and compares its SAH cost with tinybvh's `BuildHPLOC`.
 */
export const TRACER_IDS = ['ours-tinybvh', 'ours-tmbuild', 'platform', 'tmbvh', 'ours-lbvh', 'ours-hploc'] as const;
export type TracerId = (typeof TRACER_IDS)[number];

/** What runs when the URL names no tracer. `ours-lbvh` and `ours-hploc` are asked for by their own pages (trace-bench-lbvh*, trace-bench-hploc*). */
export const DEFAULT_TRACER_IDS: readonly TracerId[] = ['ours-tinybvh', 'ours-tmbuild', 'platform', 'tmbvh'];

/** GPU builds (LBVH, H-PLOC) timed per scene after one warm-up (median, fastest and slowest are reported). */
export const GPU_BUILD_TIMED_RUNS = 5;

/**
 * Then the loaded section: builds submitted back to back for about this long,
 * so the GPU stays busy and its clock is the loaded one (DECISIONS f3e3bba:
 * pass/fail timing is taken GPU-bound, over sections of 10-15 s, from a cool
 * card). The single builds above wait for each result, so the GPU idles
 * between them and the bench lock's once-a-second clock sampler barely sees them.
 */
export const GPU_BUILD_LOADED_MS = 10_000;
/** Bounds on the loaded section's build count, whatever one build takes. */
export const GPU_BUILD_LOADED_RUNS = { min: 10, max: 400 } as const;

/**
 * How big a run is.
 *
 * - `full`: what the decision uses. 256 rays per probe, about 2M rays per dispatch,
 *   median of 20 timed dispatches after 5 warm-ups (docs/design/trace.md, "Benchmark").
 * - `quick`: a few seconds, for checking that the page and the tracers work.
 */
export interface BenchProfile {
    raysPerProbe: number;
    /** The probe grid is sized so the total ray count lands near this. */
    targetRays: number;
    warmupDispatches: number;
    timedDispatches: number;
    /**
     * Ray-set mode `rc` only: after the timed dispatches, each ray set is
     * submitted back to back for about this long per tracer (the loaded
     * section, as the GPU builds have; see {@link RC_LOADED_DISPATCHES}).
     */
    rcLoadedMs: number;
}

export const BENCH_PROFILES = {
    full: { raysPerProbe: 256, targetRays: 2_097_152, warmupDispatches: 5, timedDispatches: 20, rcLoadedMs: 1_000 },
    quick: { raysPerProbe: 64, targetRays: 131_072, warmupDispatches: 2, timedDispatches: 5, rcLoadedMs: 150 },
} as const satisfies Record<string, BenchProfile>;
export type BenchProfileName = keyof typeof BENCH_PROFILES;

/** Fixed seed for the ray directions, so every tracer and every run shoots the same rays. */
export const RAY_DIRECTION_SEED = 0x5eed_d151;

/** "No limit" for a ray's far end. Large but finite so the float math stays well behaved. */
export const RAY_FAR = 1e30;

/**
 * Shadow rays start this far (as a fraction of the scene diagonal) off the
 * surface along its normal, so they do not hit the triangle they start on.
 */
export const SHADOW_RAY_OFFSET_FRACTION = 1e-5;

/** Sun direction (pointing toward the sun) for scenes whose manifest has no directional light. */
export const FALLBACK_TOWARD_SUN: readonly [number, number, number] = [0.4, 0.8, 0.45];

/**
 * The agreement check: two closest hits are the same when both miss, or both
 * hit with t within this relative tolerance (docs/design/trace.md).
 */
export const AGREEMENT_RELATIVE_T = 1e-3;
/** Absolute slack (metres) so two very short hits are not failed by float noise. */
export const AGREEMENT_ABSOLUTE_T = 1e-5;
/** Below this, a closest-hit disagreement rate means a bug and fails the page. */
export const MIN_AGREEMENT_RATE = 0.98;

/**
 * three-mesh-bvh per-mesh trees. Its default build strategy is CENTER, which would make
 * it look slower than it is, so SAH is always requested. `targetLeafSize` 10 is the
 * library's own default; override it with `?tmLeaf=`.
 */
export const TMBVH_DEFAULT_LEAF_SIZE = 10;

/**
 * three-mesh-bvh target leaf size for the `ours-tmbuild` trees. 4 matches our own
 * builder's leaves; override with `?tmbuildLeaf=` (leaves over 8 are cut in half).
 */
export const TMBUILD_DEFAULT_LEAF_SIZE = 4;

/**
 * One dispatch must stay well under the OS's GPU watchdog (Windows resets a GPU
 * after about 2 s), and the Intel iGPU is roughly 10x slower than the RTX 3060.
 * Before the real run, a small sample of the rays is timed, and if the full
 * batch would take longer than this the probe grid is thinned so it fits. The
 * results say when that happened. 0 turns the check off.
 */
export const DEFAULT_MAX_DISPATCH_MS = 400;

/** Probe count of the sample used for that estimate (enough rays to time, few enough to be quick). */
export const CALIBRATION_SAMPLE_RAYS = 16_384;

/** Top-level rebuilds timed per scene (median and worst are reported). */
export const TOP_LEVEL_REBUILD_RUNS = 20;

/**
 * Which rays the bench traces.
 *
 * - `probe-grid` (the default, T0.10 on): DDGI-shaped. A grid of probes
 *   fills the scene's volume, every probe fires the same sphere of
 *   directions, and the hits facing the sun send one shadow ray each.
 * - `rc` (RC1.5): shaped like Capsaicin GI-1.2's defaults (the radiance-cache
 *   track's baseline) from the scene's first camera bookmark: screen probes
 *   placed by primary rays, 64 cosine rays per probe, multi-bounce rays, and
 *   one shadow ray per hit toward one sampled light. See rc-rays.ts.
 */
export const RAY_SET_MODES = ['probe-grid', 'rc'] as const;
export type RaySetMode = (typeof RAY_SET_MODES)[number];

/**
 * Ray-set mode `rc`: Capsaicin GI-1.2's defaults (`gi1.h`, AMD): the view is
 * 1280x720, one screen probe per 16x16-pixel tile (80 x 45 = 3,600 tiles),
 * 64 rays per probe (Capsaicin's 8x8 hemi-octahedral cells), and multi-bounce
 * rays about 0.3x the probe rays. These match the design doc's native
 * yardstick (docs/design/radiance-cache.md), measured at 1280x720.
 */
export const RC_VIEW = { width: 1280, height: 720 } as const;
export const RC_TILE_SIZE = 16;
export const RC_RAYS_PER_PROBE = 64;
export const RC_MULTI_BOUNCE_RATIO = 0.3;
/** Fixed seed for the `rc` sets' random choices (per-probe rotation and jitter, bounce subset, light picks). */
export const RC_SEED = 0x0c4b_5a1c;
/**
 * Probe placement jitters each tile's primary ray by the Halton (2, 3)
 * point of index `tile + RC_HALTON_FIRST_INDEX` (index 0 is the corner
 * (0, 0), so it is skipped). Capsaicin jitters with Halton per frame; here
 * one frame's worth is spread over the tiles, so neighbouring tiles don't all
 * pick the same pixel.
 */
export const RC_HALTON_FIRST_INDEX = 1;
/**
 * The loaded section's dispatch count is `rcLoadedMs` over one dispatch's
 * median time, within these bounds. The upper bound keeps a very short
 * dispatch from submitting tens of thousands (the placement set, 3,600 rays,
 * takes about 0.42 ms on Sponza on the RTX 3060, so it stays well under it).
 */
export const RC_LOADED_DISPATCHES = { min: 10, max: 4_000 } as const;
/**
 * Rays per set checked against the brute-force reference tracer (every
 * triangle, in doubles). Small, because each closest-hit ray tests all of
 * Sponza's 5.7 M triangles on the CPU.
 */
export const RC_ORACLE_SAMPLE_RAYS = 16;

export interface BenchConfig {
    scenes: string[];
    tracers: TracerId[];
    profile: BenchProfileName;
    raysPerProbe: number;
    targetRays: number;
    warmupDispatches: number;
    timedDispatches: number;
    tmbvhLeafSize: number;
    /** See {@link TMBUILD_DEFAULT_LEAF_SIZE}. */
    tmbuildLeafSize: number;
    /** See {@link DEFAULT_MAX_DISPATCH_MS}. */
    maxDispatchMs: number;
    /** Fill the manifest's probe volume (default) or the geometry's own bounds. */
    volume: 'manifest' | 'scene';
    /** See {@link RAY_SET_MODES}. */
    raySet: RaySetMode;
    /** See {@link BenchProfile.rcLoadedMs}. 0 skips the loaded section. */
    rcLoadedMs: number;
}

export class BenchConfigError extends Error {}

function readPositiveInteger(params: URLSearchParams, name: string, fallback: number): number {
    const raw = params.get(name);
    if (raw === null) return fallback;
    const value = Number(raw);
    if (!Number.isInteger(value) || value <= 0) throw new BenchConfigError(`?${name}= must be a positive whole number, got "${raw}"`);
    return value;
}

function readNonNegativeInteger(params: URLSearchParams, name: string, fallback: number): number {
    const raw = params.get(name);
    if (raw === null) return fallback;
    const value = Number(raw);
    if (!Number.isInteger(value) || value < 0) throw new BenchConfigError(`?${name}= must be a whole number, 0 or more, got "${raw}"`);
    return value;
}

function readList(params: URLSearchParams, name: string): string[] | null {
    const raw = params.get(name);
    if (raw === null || raw.trim() === '') return null;
    return raw
        .split(',')
        .map((entry) => entry.trim())
        .filter((entry) => entry.length > 0);
}

/**
 * Query parameters (all optional):
 *
 *   ?scene=sponza,bistro-interior   scene manifest ids, or `all` (default: what the page folder says)
 *   &tracer=ours-tinybvh,tmbvh      any of TRACER_IDS (default: DEFAULT_TRACER_IDS)
 *   &rays=256                       rays per probe
 *   &targetRays=2097152             total rays per dispatch, about
 *   &profile=quick|full             defaults for the numbers above and the timing counts
 *   &iterations=20&warmups=5        timed and warm-up dispatches per tracer and mode
 *   &tmLeaf=10                      three-mesh-bvh target leaf size
 *   &tmbuildLeaf=4                  three-mesh-bvh target leaf size for the ours-tmbuild trees
 *   &maxDispatchMs=400              thin the probe grid if a batch would take longer (0 = never)
 *   &volume=manifest|scene          which box the probe grid fills
 *   &raySet=probe-grid|rc           which rays to trace (RAY_SET_MODES)
 *   &loadedMs=1000                  rc only: back-to-back section per ray set and tracer (0 = none)
 */
export function parseBenchConfig(
    search: string,
    defaults: { scenes: string[]; profile: BenchProfileName },
): BenchConfig {
    const params = new URLSearchParams(search);

    const profileName = (params.get('profile') ?? defaults.profile) as BenchProfileName;
    if (!(profileName in BENCH_PROFILES)) {
        throw new BenchConfigError(`?profile= must be ${Object.keys(BENCH_PROFILES).join(' or ')}, got "${profileName}"`);
    }
    const profile = BENCH_PROFILES[profileName];

    let scenes = readList(params, 'scene') ?? defaults.scenes;
    if (scenes.includes('all')) scenes = [...BENCH_SCENE_IDS];

    const tracerNames = readList(params, 'tracer') ?? [...DEFAULT_TRACER_IDS];
    const tracers = tracerNames.map((name) => {
        if (!(TRACER_IDS as readonly string[]).includes(name)) {
            throw new BenchConfigError(`?tracer= entries must be from ${TRACER_IDS.join(', ')}, got "${name}"`);
        }
        return name as TracerId;
    });

    const volume = params.get('volume') ?? 'manifest';
    if (volume !== 'manifest' && volume !== 'scene') throw new BenchConfigError(`?volume= must be manifest or scene, got "${volume}"`);

    const raySet = params.get('raySet') ?? 'probe-grid';
    if (!(RAY_SET_MODES as readonly string[]).includes(raySet)) {
        throw new BenchConfigError(`?raySet= must be ${RAY_SET_MODES.join(' or ')}, got "${raySet}"`);
    }

    return {
        scenes,
        tracers,
        profile: profileName,
        raysPerProbe: readPositiveInteger(params, 'rays', profile.raysPerProbe),
        targetRays: readPositiveInteger(params, 'targetRays', profile.targetRays),
        warmupDispatches: readPositiveInteger(params, 'warmups', profile.warmupDispatches),
        timedDispatches: readPositiveInteger(params, 'iterations', profile.timedDispatches),
        tmbvhLeafSize: readPositiveInteger(params, 'tmLeaf', TMBVH_DEFAULT_LEAF_SIZE),
        tmbuildLeafSize: readPositiveInteger(params, 'tmbuildLeaf', TMBUILD_DEFAULT_LEAF_SIZE),
        maxDispatchMs: readNonNegativeInteger(params, 'maxDispatchMs', DEFAULT_MAX_DISPATCH_MS),
        volume,
        raySet: raySet as RaySetMode,
        rcLoadedMs: readNonNegativeInteger(params, 'loadedMs', profile.rcLoadedMs),
    };
}
