/**
 * Turns the benchmark page's `data` into `(scene, metric, value)` rows for
 * `bench/results/trace.csv`. Pure functions, no Node or browser APIs, so the
 * converter script and its test share them.
 *
 * Metric names are `<group>/<what>_<unit>`, with the tracer and mode in the
 * group, for example `ours-wt/closest/mrays_per_s` or `tmbvh/build_ms_total`.
 */

import type { AgreementResult, ModeResult, SceneResult, SyntheticResult, TraceBenchData } from './results.ts';
import { median } from './statistics.ts';

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

/** Probes the profile asked for, before any thinning. */
function probesRequested(result: SceneResult): number {
    return result.probesRequested ?? Math.round(result.closestRaysRequested / Math.max(1, result.raysPerProbe));
}

/** Whether the probe grid was thinned. Null for results from before the page recorded it. */
function isThinned(result: SceneResult): boolean | null {
    return typeof result.thinned === 'boolean' ? result.thinned : null;
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
        // The probe grid: how many probes were traced, how many the profile asked for, and whether the
        // grid was thinned to keep one dispatch under the GPU watchdog (then the rays differ between devices).
        { scene, metric: 'probes/count', value: result.probeCount },
        { scene, metric: 'probes/requested', value: probesRequested(result) },
    ];
    const thinned = isThinned(result);
    if (thinned !== null) rows.push({ scene, metric: 'probes/thinned', value: thinned ? 1 : 0 });
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
