/**
 * Turns the benchmark page's `data` into `(scene, metric, value)` rows for
 * `bench/results/trace.csv`. Pure functions, no Node or browser APIs, so the
 * converter script and its test share them.
 *
 * Metric names are `<group>/<what>_<unit>`, with the tracer and mode in the
 * group, for example `ours-wt/closest/mrays_per_s` or `tmbvh/build_ms_total`.
 */

import type { AgreementResult, ModeResult, SceneResult, SyntheticResult, TraceBenchData } from './results.ts';

export interface MetricRow {
    scene: string;
    metric: string;
    value: number;
}

function modeRows(scene: string, mode: ModeResult): MetricRow[] {
    const prefix = `${mode.tracer}/${mode.mode}`;
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

function sceneRows(result: SceneResult): MetricRow[] {
    const scene = result.scene;
    const rows: MetricRow[] = [
        { scene, metric: 'scene/meshes', value: result.meshCount },
        { scene, metric: 'scene/instances', value: result.instanceCount },
        { scene, metric: 'scene/triangles_unique', value: result.uniqueTriangles },
        { scene, metric: 'scene/triangles_instanced', value: result.instancedTriangles },
        { scene, metric: 'rays/probes', value: result.probeCount },
        { scene, metric: 'rays/per_probe', value: result.raysPerProbe },
        { scene, metric: 'rays/closest', value: result.closestRays },
        { scene, metric: 'rays/closest_requested', value: result.closestRaysRequested },
        { scene, metric: 'rays/closest_hits', value: result.closestHits },
        { scene, metric: 'rays/any', value: result.shadowRays },
    ];
    if (result.estimatedNsPerRay !== null) rows.push({ scene, metric: 'rays/estimated_ns_per_ray', value: result.estimatedNsPerRay });
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
    for (const mode of result.modes) rows.push(...modeRows(scene, mode));
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
