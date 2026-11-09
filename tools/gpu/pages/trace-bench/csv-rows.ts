/**
 * Turns the benchmark page's `data` into `(scene, metric, value)` rows for
 * `bench/results/trace.csv`. Pure functions, no Node or browser APIs, so the
 * converter script and its test share them.
 *
 * Metric names are `<group>/<what>_<unit>`, with the tracer and mode in the
 * group, for example `ours-tinybvh/closest/mrays_per_s` or `tmbvh/build_ms_total`.
 * Ray-set mode `rc` (RC1.5) puts `rc/<set>/` in front, for example
 * `rc/probe/platform/closest/mrays_per_s`, adds `rc/<set>/rays` and the like,
 * the per-frame `rc/total/<tracer>/predicted_ms`, and `note/ray_set_rc,1`.
 */

import { RC_FRAME_SETS, type AgreementResult, type ModeResult, type RcSetName, type SceneResult, type SyntheticResult, type TraceBenchData } from './results.ts';
import { median, sum } from './statistics.ts';

export interface MetricRow {
    scene: string;
    metric: string;
    value: number;
}

/** `group` goes before the tracer: empty for the probe-grid bench, `rc/<set>/` for ray-set mode `rc`. */
function modeRows(scene: string, mode: ModeResult, group = ''): MetricRow[] {
    const prefix = `${group}${mode.tracer}/${mode.mode}`;
    return [
        { scene, metric: `${prefix}/mrays_per_s`, value: mode.mraysPerSecond },
        { scene, metric: `${prefix}/gpu_ms_median`, value: mode.timing.medianMs },
        { scene, metric: `${prefix}/gpu_ms_min`, value: mode.timing.minMs },
        { scene, metric: `${prefix}/gpu_ms_max`, value: mode.timing.maxMs },
        { scene, metric: `${prefix}/rays`, value: mode.rays },
        // 1 when the time came from GPU timestamps, 0 for the wall-clock fallback.
        { scene, metric: `${prefix}/timestamp_timed`, value: mode.timingSource === 'timestamp-query' ? 1 : 0 },
    ];
}

function agreementRows(scene: string, agreement: AgreementResult): MetricRow[] {
    const pair = agreement.pair.replace(' vs ', '_vs_');
    const rows: MetricRow[] = [
        { scene, metric: `agreement/${pair}/closest_rate`, value: agreement.closestRate },
        { scene, metric: `agreement/${pair}/closest_hit_miss_mismatches`, value: agreement.hitMissMismatch },
        { scene, metric: `agreement/${pair}/closest_distance_mismatches`, value: agreement.distanceMismatch },
    ];
    if (agreement.anyRate !== null) rows.push({ scene, metric: `agreement/${pair}/any_rate`, value: agreement.anyRate });
    return rows;
}

/** Probes the profile asked for, before any thinning. */
function probesRequested(result: SceneResult): number {
    return result.probesRequested ?? Math.round(result.closestRaysRequested / Math.max(1, result.raysPerProbe));
}

/** Whether the probe grid was thinned. Null for results from before the page recorded it. */
function isThinned(result: SceneResult): boolean | null {
    return typeof result.thinned === 'boolean' ? result.thinned : null;
}

/** The probe-grid bench's ray counts. Ray-set mode `rc` has its own (`rcRows`). */
function probeGridRows(result: SceneResult): MetricRow[] {
    const scene = result.scene;
    const rows: MetricRow[] = [
        { scene, metric: 'rays/probes', value: result.probeCount },
        { scene, metric: 'rays/per_probe', value: result.raysPerProbe },
        { scene, metric: 'rays/closest', value: result.closestRays },
        { scene, metric: 'rays/closest_requested', value: result.closestRaysRequested },
        { scene, metric: 'rays/closest_hits', value: result.closestHits },
        { scene, metric: 'rays/any', value: result.shadowRays },
        // The probe grid: how many probes were traced, how many the profile asked for, and whether the
        // grid was thinned to keep one dispatch under the GPU watchdog (then the rays differ between devices).
        { scene, metric: 'probes/count', value: result.probeCount },
        { scene, metric: 'probes/requested', value: probesRequested(result) },
    ];
    const thinned = isThinned(result);
    if (thinned !== null) rows.push({ scene, metric: 'probes/thinned', value: thinned ? 1 : 0 });
    if (result.estimatedNsPerRay !== null) rows.push({ scene, metric: 'rays/estimated_ns_per_ray', value: result.estimatedNsPerRay });
    return rows;
}

/** One set's predicted time: its rays over the tracer's rate on that set. */
export interface RcSetPrediction {
    set: RcSetName;
    rays: number;
    mraysPerSecond: number;
    ms: number;
}

export interface RcPrediction {
    tracer: string;
    /** `loaded` when every set had a loaded section (the GPU-bound number), else `single` dispatches. */
    source: 'loaded' | 'single';
    /** The per-frame sets (RC_FRAME_SETS), each traced. */
    sets: RcSetPrediction[];
    /** Their sum: the predicted tracing time of one frame. */
    totalMs: number;
}

/**
 * Per tracer, the predicted tracing time of one frame of the radiance-cache
 * baseline: for each per-frame set (probe, bounce and their shadow rays;
 * placement isn't one, as Capsaicin places probes from its visibility
 * buffer), rays / rate, and their sum. The rate is the loaded section's when
 * there is one. Each set was traced as one dispatch of exactly its rays, so a
 * set's prediction is that dispatch's median time; the sum is the new number.
 * A tracer missing a per-frame set that has rays gets no prediction.
 */
export function rcPredictions(result: SceneResult): RcPrediction[] {
    if (!result.rc) return [];
    const tracers = [...new Set(result.modes.filter((mode) => mode.raySet !== undefined).map((mode) => mode.tracer))];
    const predictions: RcPrediction[] = [];
    for (const tracer of tracers) {
        const sets: RcSetPrediction[] = [];
        let allLoaded = true;
        let complete = true;
        for (const name of RC_FRAME_SETS) {
            const rays = result.rc.sets.find((entry) => entry.set === name)?.rays ?? 0;
            if (rays === 0) continue;
            const mode = result.modes.find((entry) => entry.tracer === tracer && entry.raySet === name);
            if (!mode) {
                complete = false;
                break;
            }
            if (!mode.loaded) allLoaded = false;
            const mraysPerSecond = mode.loaded?.mraysPerSecond ?? mode.mraysPerSecond;
            sets.push({ set: name, rays, mraysPerSecond, ms: rays / (mraysPerSecond * 1000) });
        }
        if (!complete) continue;
        predictions.push({ tracer, source: allLoaded ? 'loaded' : 'single', sets, totalMs: sum(sets.map((entry) => entry.ms)) });
    }
    return predictions;
}

/** Ray-set mode `rc` (RC1.5): what was placed, each set's rays, hits and checks, each tracer's rates and the predictions. */
function rcRows(result: SceneResult): MetricRow[] {
    const rc = result.rc;
    if (!rc) return [];
    const scene = result.scene;
    const rows: MetricRow[] = [
        // Which rays these are: Capsaicin GI-1.2-shaped sets from the first bookmark, not the probe grid.
        { scene, metric: 'note/ray_set_rc', value: 1 },
        { scene, metric: 'rc/view_width', value: rc.width },
        { scene, metric: 'rc/view_height', value: rc.height },
        { scene, metric: 'rc/view_fov_y_deg', value: rc.camera.fovYDeg },
        { scene, metric: 'rc/tiles', value: rc.tiles },
        { scene, metric: 'rc/probes_placed', value: rc.probesPlaced },
        { scene, metric: 'rc/tiles_missed', value: rc.tilesMissed },
        { scene, metric: 'rc/rays_per_probe', value: rc.raysPerProbe },
        { scene, metric: 'rc/multi_bounce_ratio', value: rc.multiBounceRatio },
        { scene, metric: 'rc/light_is_sun', value: rc.lightKind === 'sun' ? 1 : 0 },
        { scene, metric: 'rc/light_count', value: rc.lightCount },
    ];
    for (const entry of rc.sets) {
        const prefix = `rc/${entry.set}`;
        rows.push(
            { scene, metric: `${prefix}/rays`, value: entry.rays },
            { scene, metric: `${prefix}/hit_fraction`, value: entry.hitFraction },
            { scene, metric: `${prefix}/oracle_rays`, value: entry.oracleRays },
            { scene, metric: `${prefix}/oracle_mismatches`, value: entry.oracleMismatches },
            { scene, metric: `${prefix}/oracle_ambiguous`, value: entry.oracleAmbiguous },
        );
        if (entry.facingAway !== undefined) rows.push({ scene, metric: `${prefix}/facing_away`, value: entry.facingAway });
        if (entry.agreementRate !== null) rows.push({ scene, metric: `${prefix}/agreement_ours-tinybvh_vs_platform`, value: entry.agreementRate });
    }
    for (const mode of result.modes) {
        if (mode.raySet === undefined) continue;
        const prefix = `rc/${mode.raySet}/${mode.tracer}/${mode.mode}`;
        rows.push(...modeRows(scene, mode, `rc/${mode.raySet}/`));
        if (mode.loaded) {
            rows.push(
                { scene, metric: `${prefix}/loaded_mrays_per_s`, value: mode.loaded.mraysPerSecond },
                { scene, metric: `${prefix}/loaded_gpu_ms_median`, value: mode.loaded.medianMs },
                { scene, metric: `${prefix}/loaded_gpu_ms_min`, value: mode.loaded.minMs },
                { scene, metric: `${prefix}/loaded_gpu_ms_max`, value: mode.loaded.maxMs },
                { scene, metric: `${prefix}/loaded_dispatches`, value: mode.loaded.dispatches },
                { scene, metric: `${prefix}/loaded_seconds`, value: mode.loaded.seconds },
            );
        }
        const mraysPerSecond = mode.loaded?.mraysPerSecond ?? mode.mraysPerSecond;
        rows.push({ scene, metric: `${prefix}/predicted_ms`, value: mode.rays / (mraysPerSecond * 1000) });
    }
    for (const prediction of rcPredictions(result)) {
        rows.push(
            { scene, metric: `rc/total/${prediction.tracer}/predicted_ms`, value: prediction.totalMs },
            // 1 when every set's rate came from its loaded section, 0 when from single dispatches.
            { scene, metric: `rc/total/${prediction.tracer}/predicted_from_loaded`, value: prediction.source === 'loaded' ? 1 : 0 },
        );
    }
    const frameRays = sum(rc.sets.filter((entry) => RC_FRAME_SETS.includes(entry.set)).map((entry) => entry.rays));
    rows.push({ scene, metric: 'rc/total/rays', value: frameRays });
    return rows;
}

function sceneRows(result: SceneResult): MetricRow[] {
    const scene = result.scene;
    const rows: MetricRow[] = [
        { scene, metric: 'scene/meshes', value: result.meshCount },
        { scene, metric: 'scene/instances', value: result.instanceCount },
        { scene, metric: 'scene/triangles_unique', value: result.uniqueTriangles },
        { scene, metric: 'scene/triangles_instanced', value: result.instancedTriangles },
        ...(result.rc ? rcRows(result) : probeGridRows(result)),
    ];
    const ours = result.ours;
    if (ours) {
        rows.push(
            { scene, metric: 'ours/build_ms_total', value: ours.buildMsTotal },
            { scene, metric: 'ours/build_ms_max_mesh', value: ours.buildMsMaxMesh },
            { scene, metric: 'ours/assemble_ms', value: ours.assembleMs },
            { scene, metric: 'ours/top_level_rebuild_ms_median', value: ours.topLevelRebuildMsMedian },
            { scene, metric: 'ours/top_level_rebuild_ms_max', value: ours.topLevelRebuildMsMax },
            { scene, metric: 'ours/top_level_instances', value: ours.topLevelInstances },
            { scene, metric: 'ours/nodes', value: ours.nodeCount },
            { scene, metric: 'ours/max_mesh_depth', value: ours.maxMeshDepth },
            { scene, metric: 'ours/stack_entries_needed', value: ours.requiredStackSize },
            { scene, metric: 'ours/leaf_size', value: ours.leafSize },
            { scene, metric: 'ours/memory_bytes_nodes', value: ours.bytesNodes },
            { scene, metric: 'ours/memory_bytes_triangles', value: ours.bytesTriangles },
            { scene, metric: 'ours/memory_bytes_top_level', value: ours.bytesTopLevel },
            { scene, metric: 'ours/memory_bytes', value: ours.bytesTotal },
        );
    }
    const tinybvh = result.oursTinybvh;
    if (tinybvh) {
        rows.push(
            { scene, metric: 'ours-tinybvh/build_ms_total', value: tinybvh.buildMsTotal },
            { scene, metric: 'ours-tinybvh/build_ms_max_mesh', value: tinybvh.buildMsMaxMesh },
            { scene, metric: 'ours-tinybvh/nodes', value: tinybvh.nodeCount },
            { scene, metric: 'ours-tinybvh/max_mesh_depth', value: tinybvh.maxMeshDepth },
            { scene, metric: 'ours-tinybvh/stack_entries_needed', value: tinybvh.requiredStackSize },
            { scene, metric: 'ours-tinybvh/memory_bytes_nodes', value: tinybvh.bytesNodes },
            { scene, metric: 'ours-tinybvh/memory_bytes_triangles', value: tinybvh.bytesTriangles },
            { scene, metric: 'ours-tinybvh/memory_bytes', value: tinybvh.bytesTotal },
        );
    }
    const tmbuild = result.oursTmbuild;
    if (tmbuild) {
        rows.push(
            { scene, metric: 'ours-tmbuild/build_ms_total', value: tmbuild.buildMsTotal },
            { scene, metric: 'ours-tmbuild/build_ms_tmbvh', value: tmbuild.tmbvhBuildMsTotal },
            { scene, metric: 'ours-tmbuild/build_ms_convert', value: tmbuild.convertMsTotal },
            { scene, metric: 'ours-tmbuild/build_ms_max_mesh', value: tmbuild.buildMsMaxMesh },
            { scene, metric: 'ours-tmbuild/target_leaf_size', value: tmbuild.targetLeafSize },
            { scene, metric: 'ours-tmbuild/nodes', value: tmbuild.nodeCount },
            { scene, metric: 'ours-tmbuild/max_mesh_depth', value: tmbuild.maxMeshDepth },
            { scene, metric: 'ours-tmbuild/stack_entries_needed', value: tmbuild.requiredStackSize },
            { scene, metric: 'ours-tmbuild/memory_bytes_nodes', value: tmbuild.bytesNodes },
            { scene, metric: 'ours-tmbuild/memory_bytes_triangles', value: tmbuild.bytesTriangles },
            { scene, metric: 'ours-tmbuild/memory_bytes', value: tmbuild.bytesTotal },
        );
    }
    for (const [tracer, built] of [
        ['ours-lbvh', result.oursLbvh],
        ['ours-hploc', result.oursHploc],
    ] as const) {
        if (!built) continue;
        rows.push(
            { scene, metric: `${tracer}/build_gpu_ms_median`, value: built.gpuBuildMsMedian },
            { scene, metric: `${tracer}/build_gpu_ms_min`, value: built.gpuBuildMsMin },
            { scene, metric: `${tracer}/build_gpu_ms_max`, value: built.gpuBuildMsMax },
            { scene, metric: `${tracer}/build_gpu_runs`, value: built.gpuBuildRuns },
            { scene, metric: `${tracer}/build_timestamp_timed`, value: built.gpuBuildTimestampTimed },
            { scene, metric: `${tracer}/build_gpu_loaded_ms_median`, value: built.gpuBuildLoadedMsMedian },
            { scene, metric: `${tracer}/build_gpu_loaded_ms_min`, value: built.gpuBuildLoadedMsMin },
            { scene, metric: `${tracer}/build_gpu_loaded_ms_max`, value: built.gpuBuildLoadedMsMax },
            { scene, metric: `${tracer}/build_gpu_loaded_runs`, value: built.gpuBuildLoadedRuns },
            { scene, metric: `${tracer}/build_gpu_loaded_seconds`, value: built.gpuBuildLoadedSeconds },
            { scene, metric: `${tracer}/build_ms_wall`, value: built.wallBuildMs },
            { scene, metric: `${tracer}/readback_ms`, value: built.readbackMs },
            { scene, metric: `${tracer}/sah_cost_vs_tinybvh`, value: built.sahCostVsTinybvh },
            { scene, metric: `${tracer}/meshes_fallen_back`, value: built.meshesFallenBack },
            { scene, metric: `${tracer}/nodes`, value: built.nodeCount },
            { scene, metric: `${tracer}/max_mesh_depth`, value: built.maxMeshDepth },
            { scene, metric: `${tracer}/stack_entries_needed`, value: built.requiredStackSize },
            { scene, metric: `${tracer}/memory_bytes_nodes`, value: built.bytesNodes },
            { scene, metric: `${tracer}/memory_bytes_triangles`, value: built.bytesTriangles },
            { scene, metric: `${tracer}/memory_bytes`, value: built.bytesTotal },
        );
        if (built.sahCostVsTinybvhHploc !== undefined) {
            rows.push(
                { scene, metric: `${tracer}/sah_cost_vs_tinybvh_hploc`, value: built.sahCostVsTinybvhHploc },
                { scene, metric: `${tracer}/hploc_match_meshes_refused`, value: built.hplocMatchMeshesRefused ?? 0 },
            );
        }
    }
    const platform = result.platform;
    if (platform) {
        rows.push(
            { scene, metric: 'platform/describe_ms', value: platform.describeMs },
            { scene, metric: 'platform/ready_ms', value: platform.readyMs },
            { scene, metric: 'platform/memory_bytes_scene_copy', value: platform.bytesSceneCopy },
            { scene, metric: 'platform/memory_bytes_acceleration', value: platform.bytesAcceleration },
            { scene, metric: 'platform/memory_bytes_triangles', value: platform.bytesTriangles },
        );
    }
    const tmbvh = result.tmbvh;
    if (tmbvh) {
        rows.push(
            { scene, metric: 'tmbvh/build_ms_total', value: tmbvh.buildMsTotal },
            { scene, metric: 'tmbvh/build_ms_max_mesh', value: tmbvh.buildMsMaxMesh },
            { scene, metric: 'tmbvh/data_update_ms', value: tmbvh.dataUpdateMs },
            { scene, metric: 'tmbvh/leaf_size', value: tmbvh.leafSize },
            { scene, metric: 'tmbvh/memory_bytes_nodes', value: tmbvh.bytesNodes },
            { scene, metric: 'tmbvh/memory_bytes_index', value: tmbvh.bytesIndex },
            { scene, metric: 'tmbvh/memory_bytes_attributes', value: tmbvh.bytesAttributes },
            { scene, metric: 'tmbvh/memory_bytes_transforms', value: tmbvh.bytesTransforms },
            { scene, metric: 'tmbvh/memory_bytes', value: tmbvh.bytesTotal },
        );
    }
    // Ray-set mode `rc`'s timings are in rcRows, under their set.
    for (const mode of result.modes) if (mode.raySet === undefined) rows.push(...modeRows(scene, mode));
    for (const agreement of result.agreement) rows.push(...agreementRows(scene, agreement));
    return rows;
}

function syntheticRows(result: SyntheticResult): MetricRow[] {
    const scene = result.scene;
    return [
        { scene, metric: 'scene/instances', value: result.instanceCount },
        { scene, metric: 'scene/triangles_unique', value: result.meshTriangles },
        { scene, metric: 'ours/top_level_rebuild_ms_median', value: result.topLevelRebuildMsMedian },
        { scene, metric: 'ours/top_level_rebuild_ms_max', value: result.topLevelRebuildMsMax },
        { scene, metric: 'ours/top_level_nodes', value: result.topLevelNodes },
        { scene, metric: 'ours/top_level_max_depth', value: result.topLevelMaxDepth },
        { scene, metric: 'ours/memory_bytes_top_level', value: result.topLevelBytes },
    ];
}

export function traceBenchRows(data: TraceBenchData): MetricRow[] {
    return [
        ...data.scenes.flatMap(sceneRows),
        ...data.synthetic.flatMap(syntheticRows),
        // The limits the run needed, so the CSV says what a default device could not do.
        ...(data.scenes.length > 0
            ? [
                  { scene: 'device', metric: 'limit/max_storage_buffer_binding_size_bytes', value: data.device.maxStorageBufferBindingSize },
                  { scene: 'device', metric: 'limit/max_buffer_size_bytes', value: data.device.maxBufferSize },
              ]
            : []),
    ];
}

/**
 * Rows for several runs of the same page on the same device: every run's rows
 * with the metric prefixed `run1/`, `run2/`, ..., then a `median/` row per
 * metric that every run reported, and `median/runs` with the run count.
 * The median rows are the result; the per-run rows show the spread.
 */
export function repeatedRunRows(runs: readonly (readonly MetricRow[])[]): MetricRow[] {
    const rows: MetricRow[] = [];
    runs.forEach((runRows, index) => {
        for (const row of runRows) rows.push({ scene: row.scene, metric: `run${index + 1}/${row.metric}`, value: row.value });
    });
    const valuesByKey = new Map<string, { scene: string; metric: string; values: number[] }>();
    runs.forEach((runRows) => {
        for (const row of runRows) {
            const key = `${row.scene}|${row.metric}`;
            const entry = valuesByKey.get(key) ?? { scene: row.scene, metric: row.metric, values: [] };
            entry.values.push(row.value);
            valuesByKey.set(key, entry);
        }
    });
    const scenes = new Set<string>();
    for (const entry of valuesByKey.values()) {
        if (entry.values.length !== runs.length) continue;
        rows.push({ scene: entry.scene, metric: `median/${entry.metric}`, value: median(entry.values) });
        scenes.add(entry.scene);
    }
    for (const scene of scenes) rows.push({ scene, metric: 'median/runs', value: runs.length });
    return rows;
}

/** Six significant digits, no exponent for ordinary sizes, integers untouched. */
export function formatValue(value: number): string {
    if (!Number.isFinite(value)) return '';
    if (Number.isInteger(value)) return String(value);
    return String(Number(value.toPrecision(6)));
}

/** "Chrome 156.0.8078.74 headless" from a user-agent string, or the raw string when it is not Chrome. */
export function browserFromUserAgent(userAgent: string, fullVersion: string | null = null): string {
    const match = /(Headless)?Chrome\/([\d.]+)/.exec(userAgent);
    if (!match) return userAgent;
    return `Chrome ${fullVersion ?? match[2]}${match[1] ? ' headless' : ''}`;
}

/**
 * Chrome shows no adapter description in headless mode, so the two GPUs of this
 * project's test laptop are named from their architecture (same names as
 * bench/results/timing.csv). Pass `--device` to the converter for any other machine.
 */
const TEST_LAPTOP_GPUS: Readonly<Record<string, string>> = {
    'nvidia/ampere': 'RTX 3060 Laptop',
    'intel/gen-12lp': 'Intel UHD',
};

/** A short GPU name for the `device` column: "RTX 3060 Laptop", "Intel UHD". */
export function deviceNameFromAdapter(adapter: { vendor: string; architecture: string; description: string } | null): string {
    if (!adapter) return 'unknown';
    const description = adapter.description.trim();
    const known = TEST_LAPTOP_GPUS[`${adapter.vendor}/${adapter.architecture}`];
    if (!description && known) return known;
    if (adapter.vendor === 'nvidia' && description) {
        return description.replace(/^NVIDIA\s+/i, '').replace(/^GeForce\s+/i, '').replace(/\s+GPU$/i, '');
    }
    if (adapter.vendor === 'intel') return /UHD/i.test(description) ? 'Intel UHD' : description || `Intel ${adapter.architecture}`;
    return description || `${adapter.vendor} ${adapter.architecture}`;
}
