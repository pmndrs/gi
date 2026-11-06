/**
 * The shape of what the benchmark page reports in `data`, shared by the page,
 * the CSV converter (`to-csv.ts`) and their tests. Plain JSON only.
 */

import type { TracerId } from './config.ts';

export type TraceMode = 'closest' | 'any';

/** How a GPU time was taken. Wall-clock is the labelled fallback for devices without timestamps. */
export type TimingSource = 'timestamp-query' | 'wall-clock';

export interface TimingSummary {
    medianMs: number;
    minMs: number;
    maxMs: number;
    /** Every timed dispatch, in order. */
    samplesMs: number[];
}

export interface ModeResult {
    tracer: TracerId;
    mode: TraceMode;
    /** Rays in the dispatch. */
    rays: number;
    timing: TimingSummary;
    timingSource: TimingSource;
    /** Million rays per second at the median time. */
    mraysPerSecond: number;
}

export interface OursBuildResult {
    /** Whether the trees were built in a worker (the page falls back to the main thread without one). */
    usesWorker: boolean;
    meshCount: number;
    triangles: number;
    /** Wall time to build every mesh's tree, one after another. */
    buildMsTotal: number;
    /** The slowest single mesh. */
    buildMsMaxMesh: number;
    maxMeshTriangles: number;
    /** Copying the trees into the shared buffers and building the top level once. */
    assembleMs: number;
    /** Top-level rebuild, median and worst of several runs. */
    topLevelRebuildMsMedian: number;
    topLevelRebuildMsMax: number;
    topLevelRebuildRuns: number;
    topLevelInstances: number;
    topLevelNodes: number;
    nodeCount: number;
    maxMeshDepth: number;
    requiredStackSize: number;
    leafSize: number;
    bytesNodes: number;
    bytesTriangles: number;
    bytesTopLevel: number;
    bytesTotal: number;
}

/** Our layout built from three-mesh-bvh's trees (`ours-tmbuild`): the usual numbers plus the two halves of the build. */
export interface TmbuildBuildResult extends OursBuildResult {
    /** three-mesh-bvh's SAH build of every mesh (`new MeshBVH`), summed. */
    tmbvhBuildMsTotal: number;
    /** Converting every tree to our layout (nodes, boxes, triangles), summed. */
    convertMsTotal: number;
    /** three-mesh-bvh's `targetLeafSize`. */
    targetLeafSize: number;
}

/** A GPU builder's trees (`ours-lbvh`, B1; `ours-hploc`, B2): the usual numbers plus the GPU build time and the tree quality. */
export interface GpuBuildResult extends OursBuildResult {
    /** GPU time of building every mesh's tree in one batch (one compute pass, timestamps): median, fastest, slowest. */
    gpuBuildMsMedian: number;
    gpuBuildMsMin: number;
    gpuBuildMsMax: number;
    gpuBuildRuns: number;
    /** 1 when the build was timed with GPU timestamps, 0 for the wall-clock fallback. */
    gpuBuildTimestampTimed: number;
    /**
     * The loaded section (config.ts, GPU_BUILD_LOADED_MS): builds submitted back to back, each timed on its
     * own. The pass/fail number; median, fastest, slowest, how many, and the section's wall time in seconds.
     */
    gpuBuildLoadedMsMedian: number;
    gpuBuildLoadedMsMin: number;
    gpuBuildLoadedMsMax: number;
    gpuBuildLoadedRuns: number;
    gpuBuildLoadedSeconds: number;
    /** Wall time of the first build: uploading the meshes, building, and reading the stats back. */
    wallBuildMs: number;
    /** Reading every tree back into CPU memory for the tracer's scene assembly (not part of a GPU-resident build). */
    readbackMs: number;
    /** The scene's summed SAH cost on these trees over tinybvh's, mesh by mesh (each mesh's cost scaled by its box's area). */
    sahCostVsTinybvh: number;
    /** Meshes the GPU refused and tinybvh built instead. Should be 0. */
    meshesFallenBack: number;
    /**
     * H-PLOC only: the scene built again with tinybvh's codes, tie rule and one-triangle leaves, its
     * summed SAH cost over that of tinybvh's `BuildHPLOC` trees of the same meshes (B2: within 1%).
     */
    sahCostVsTinybvhHploc?: number;
    /** H-PLOC only: meshes that comparison left out because the GPU refused them. */
    hplocMatchMeshesRefused?: number;
}

/** The `platform` tracer: the scene described through `createGI` and its scene API (T1.3). */
export interface PlatformBuildResult {
    /** Creating every mesh, material and instance through the scene API (CPU: checking and packing). */
    describeMs: number;
    /** From the first scene call until every tree is placed and the top level is built. */
    readyMs: number;
    usesWorker: boolean;
    /** GPU bytes of the platform's scene copy (vertices and tables). */
    bytesSceneCopy: number;
    /** GPU bytes of the tracer's buffers (allocated, so with room to grow). */
    bytesAcceleration: number;
    bytesNodes: number;
    bytesTriangles: number;
    bytesTopLevel: number;
}

export interface TmbvhBuildResult {
    meshCount: number;
    /** Wall time to build every mesh's tree on the main thread (SAH), one after another. */
    buildMsTotal: number;
    buildMsMaxMesh: number;
    /** `BVHComputeData.update()`: the clustered top level and packing for the GPU. */
    dataUpdateMs: number;
    leafSize: number;
    strategy: 'SAH';
    bytesNodes: number;
    bytesIndex: number;
    bytesAttributes: number;
    bytesTransforms: number;
    bytesTotal: number;
}

export interface AgreementResult {
    /** For example `ours-tinybvh vs tmbvh`. */
    pair: string;
    closestRate: number;
    closestRays: number;
    hitMissMismatch: number;
    distanceMismatch: number;
    /** Share of shadow rays with the same blocked/clear answer. */
    anyRate: number | null;
}

export interface SceneResult {
    scene: string;
    meshCount: number;
    instanceCount: number;
    uniqueTriangles: number;
    instancedTriangles: number;
    probeBoxSource: 'manifest' | 'scene';
    probeBox: { min: number[]; max: number[] };
    probeDimensions: number[];
    probeCount: number;
    raysPerProbe: number;
    closestRays: number;
    /** Rays the profile asked for, before the grid was thinned to keep one dispatch short; equal to `closestRays` when it was not. */
    closestRaysRequested: number;
    /** Probes the profile asked for. Missing in results from before T0.10 part C. */
    probesRequested?: number;
    /** True when the grid was thinned to keep one dispatch short, so the rays differ from the full set. Missing before T0.10 part C. */
    thinned?: boolean;
    /** The estimate that decided it: GPU nanoseconds per closest-hit ray on a sample, or null when the check was off. */
    estimatedNsPerRay: number | null;
    /** Closest-hit rays that hit something (by ours-tinybvh). */
    closestHits: number;
    /** Shadow rays traced; hits facing away from the sun get none. */
    shadowRays: number;
    towardSun: number[];
    sunSource: 'manifest' | 'fallback';
    modelFiles: string[];
    loadMs: number;
    /** Our trees (tinybvh's since T1.3; our own SAH builder's before): build times, top-level rebuilds, memory. */
    ours: OursBuildResult | null;
    /** Missing in results from before T0.10 part C. */
    oursTmbuild?: TmbuildBuildResult | null;
    /** Trees built by tinybvh (WASM) in the worker. Missing in results from before T0.10 part C. */
    oursTinybvh?: OursBuildResult | null;
    /** The GPU LBVH's trees (B1). Missing before B1. */
    oursLbvh?: GpuBuildResult | null;
    /** The GPU H-PLOC's trees (B2). Missing before B2. */
    oursHploc?: GpuBuildResult | null;
    /** The same scene through the platform API. Missing before T1.3. */
    platform?: PlatformBuildResult | null;
    tmbvh: TmbvhBuildResult | null;
    modes: ModeResult[];
    agreement: AgreementResult[];
    /** Things that could not run, for example a buffer over the device's limit. */
    skipped: string[];
}

/** The synthetic many-instance scene: only the top-level rebuild is timed. */
export interface SyntheticResult {
    scene: string;
    instanceCount: number;
    meshTriangles: number;
    topLevelRebuildMsMedian: number;
    topLevelRebuildMsMax: number;
    topLevelRebuildRuns: number;
    topLevelNodes: number;
    topLevelMaxDepth: number;
    topLevelBytes: number;
}

export interface DeviceRecord {
    /** Limits requested above the defaults and what the device granted. */
    maxStorageBufferBindingSize: number;
    maxBufferSize: number;
    maxStorageBuffersPerShaderStage: number;
    maxComputeWorkgroupsPerDimension: number;
    timestampQuery: boolean;
}

export interface TraceBenchData {
    benchmark: 'trace-bench';
    profile: string;
    raysPerProbe: number;
    targetRays: number;
    warmupDispatches: number;
    timedDispatches: number;
    tracers: TracerId[];
    userAgent: string;
    /** Full Chrome version (the user agent string only shows 156.0.0.0), or null when the browser will not say. */
    browserVersion: string | null;
    device: DeviceRecord;
    /** How three-mesh-bvh's pass was timed. */
    tmbvhTimingMethod: string;
    scenes: SceneResult[];
    synthetic: SyntheticResult[];
}
