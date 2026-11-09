/**
 * Ray-set mode `rc` of the trace bench (RC1.5): Capsaicin-shaped ray sets
 * (rc-rays.ts) from the scene's first camera bookmark, timed through the
 * `platform` tracer (what the radiance-cache technique will use) and
 * `ours-tinybvh` (the same trees and walk without the platform around them).
 *
 * Per scene, in order:
 *   1. load the geometry and build our trees (tinybvh, in the worker), as the
 *      probe-grid bench does;
 *   2. make the sets one after another, each aimed from the hits of the one
 *      before (placement, probe, bounce, then the two shadow sets), tracing
 *      each once, untimed, on our trees;
 *   3. check every ray (finite, unit direction, a real interval) and the
 *      probe count (more than 0, at most one per tile);
 *   4. time every set with each tracer: warm-ups, single dispatches, then a
 *      loaded section of back-to-back dispatches;
 *   5. compare the two tracers ray by ray, and a small random sample of each
 *      set against the brute-force reference tracer (every triangle, in
 *      doubles). A mismatch fails the page.
 *
 * Loading, tracers, timing, statistics and CSV rows are the bench's own; only
 * the rays and this order are new.
 */

import { RAY_FLOATS, HIT_RECORD_BYTES, HIT_RECORD_WORD, type MeshBvhBuilder } from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import { addToTally, compareAny, compareClosest, newTally, type ComparisonTally } from '../../../../packages/ddgi-core/src/platform/trace/oracle-compare.ts';
import { createRandom, toFloat32Ray } from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';
import {
    buildReferenceScene,
    traceAnyReference,
    traceClosestReference,
    type ReferenceScene,
} from '../../../../packages/ddgi-core/src/platform/trace/reference-tracer.ts';
import {
    BENCH_SCENE_SUBSETS,
    RC_HALTON_FIRST_INDEX,
    RC_MULTI_BOUNCE_RATIO,
    RC_ORACLE_SAMPLE_RAYS,
    RC_RAYS_PER_PROBE,
    RC_SEED,
    RC_TILE_SIZE,
    RC_VIEW,
    type BenchConfig,
} from './config.ts';
import { rcPredictions } from './csv-rows.ts';
import type { DispatchTimer } from './dispatch-timer.ts';
import {
    buildOursScene,
    extractHitDistances,
    openOursPrepareTracer,
    runOursTracerOnSets,
    uploadRays,
    yieldToBrowser,
    type RaySetBuffer,
    type RaySetRun,
    type TimingPlan,
} from './ours-bench.ts';
import { runPlatformTracerOnSets } from './platform-bench.ts';
import {
    hitSurfaces,
    makeBounceRays,
    makeLightShadowRays,
    makePlacementRays,
    makeProbeRays,
    multiBounceRayCount,
    placeProbes,
    rayProblems,
    rcLightsFromManifest,
    tileGrid,
} from './rc-rays.ts';
import { RC_SET_NAMES, type RcSceneResult, type RcSetName, type RcSetResult, type SceneResult } from './results.ts';
import { findManifest, loadBenchScene, type BenchScene } from './scene-source.ts';
import { compareBlocked, compareClosestDistances } from './statistics.ts';

/** The tracers this mode times. The others' trees aren't what the technique traces. */
const RC_TRACERS = ['platform', 'ours-tinybvh'] as const;
type RcTracer = (typeof RC_TRACERS)[number];

export interface RcBenchContext {
    device: GPUDevice;
    timer: DispatchTimer;
    builder: MeshBvhBuilder;
    config: BenchConfig;
    plan: TimingPlan;
    progress(message: string): void;
}

/** One set's rays on the CPU and the GPU. */
interface RcSet {
    name: RcSetName;
    mode: 'closest' | 'any';
    rays: Float32Array;
    count: number;
    buffer: GPUBuffer | null;
    facingAway?: number;
}

/** A ray of a packed set, as the reference tracer takes it. */
function rayAt(rays: Float32Array, index: number): ReturnType<typeof toFloat32Ray> {
    const base = index * RAY_FLOATS;
    return toFloat32Ray(
        [rays[base], rays[base + 1], rays[base + 2]],
        [rays[base + 4], rays[base + 5], rays[base + 6]],
        rays[base + 3],
        rays[base + 7],
    );
}

/**
 * Checks `sampleSize` randomly chosen rays of a set against the brute-force
 * reference (every triangle, in doubles): closest hits by triangle,
 * distance, side and barycentrics; any hits by blocked or clear. A ray whose
 * answer sits within rounding of an edge or the interval's end counts as
 * ambiguous, not a mismatch (oracle-compare.ts). Yields between rays, as one
 * Sponza ray tests 5.7 M triangles.
 */
async function checkAgainstReference(
    reference: ReferenceScene,
    set: RcSet,
    results: ArrayBuffer,
    sampleSize: number,
    seed: number,
): Promise<ComparisonTally> {
    const tally = newTally();
    const random = createRandom(seed);
    const floats = new Float32Array(results);
    const words = new Uint32Array(results);
    const recordWords = HIT_RECORD_BYTES / 4;
    for (let sample = 0; sample < Math.min(sampleSize, set.count); sample++) {
        const index = Math.floor(random() * set.count);
        const ray = rayAt(set.rays, index);
        if (set.mode === 'closest') {
            const record = index * recordWords;
            const t = floats[record + HIT_RECORD_WORD.t];
            const traced =
                t >= 0
                    ? {
                          hit: true,
                          t,
                          instanceIndex: words[record + HIT_RECORD_WORD.instance],
                          triangleIndex: words[record + HIT_RECORD_WORD.triangle],
                          u: floats[record + HIT_RECORD_WORD.barycentrics],
                          v: floats[record + HIT_RECORD_WORD.barycentrics + 1],
                          frontFace: words[record + HIT_RECORD_WORD.frontFace] !== 0,
                      }
                    : null;
            addToTally(tally, compareClosest(traced, traceClosestReference(reference, ray), ray), `${set.name} ray ${index}`);
        } else {
            addToTally(tally, compareAny(words[index] !== 0, traceAnyReference(reference, ray)), `${set.name} ray ${index}`);
        }
        await yieldToBrowser();
    }
    return tally;
}

function emptySceneResult(sceneId: string, scene: BenchScene): SceneResult {
    return {
        scene: sceneId,
        meshCount: scene.meshes.length,
        instanceCount: scene.instances.length,
        uniqueTriangles: scene.uniqueTriangles,
        instancedTriangles: scene.instancedTriangles,
        probeBoxSource: scene.probeBoxSource,
        probeBox: { min: scene.probeBox.min, max: scene.probeBox.max },
        probeDimensions: [],
        probeCount: 0,
        raysPerProbe: RC_RAYS_PER_PROBE,
        closestRays: 0,
        closestRaysRequested: 0,
        estimatedNsPerRay: null,
        closestHits: 0,
        shadowRays: 0,
        towardSun: scene.towardSun,
        sunSource: scene.sunSource,
        modelFiles: scene.modelFiles,
        loadMs: scene.loadMs,
        ours: null,
        oursTinybvh: null,
        tmbvh: null,
        modes: [],
        agreement: [],
        skipped: [],
    };
}

/** Closest: rays that hit geometry. Any: rays that were blocked. */
function countHits(set: RcSet, results: ArrayBuffer): number {
    let hits = 0;
    if (set.mode === 'closest') {
        for (const t of extractHitDistances(results)) if (t >= 0) hits++;
    } else {
        for (const blocked of new Uint32Array(results)) if (blocked !== 0) hits++;
    }
    return hits;
}

export async function runRcScene(context: RcBenchContext, sceneId: string): Promise<SceneResult> {
    const { device, config, progress } = context;
    const label = `${sceneId} (rc)`;
    progress(`${label}: loading geometry`);
    const scene = await loadBenchScene(sceneId, config.volume);
    const manifest = findManifest(BENCH_SCENE_SUBSETS[sceneId]?.manifest ?? sceneId);
    const camera = manifest.cameras[0];
    progress(
        `${label}: ${scene.meshes.length} meshes, ${scene.instances.length} instances, ${scene.instancedTriangles} triangles; ` +
            `camera "${camera.name}" at ${camera.position.join(', ')} toward ${camera.target.join(', ')}, ${camera.fovYDeg} deg vertical, ${RC_VIEW.width}x${RC_VIEW.height}`,
    );
    const result = emptySceneResult(sceneId, scene);
    await yieldToBrowser();

    progress(`${label}: building our trees (tinybvh, in the worker)`);
    const ours = await buildOursScene(context.builder, scene.meshes, scene.instances);
    result.ours = ours.result;
    result.oursTinybvh = ours.result;
    await yieldToBrowser();

    // ---- The sets, each aimed from the hits of the one before. ----
    const view = { width: RC_VIEW.width, height: RC_VIEW.height, tileSize: RC_TILE_SIZE };
    const tiles = tileGrid(view);
    const lights = rcLightsFromManifest(manifest.lights);
    const random = createRandom(RC_SEED);
    const sets: RcSet[] = [];
    const gpuBuffers: GPUBuffer[] = [];
    let probesPlaced = 0;
    let tilesMissed = 0;
    try {
        const prepare = await openOursPrepareTracer(device, ours.data);
        try {
            /** Uploads a set; a closest-hit set is also traced once on our trees, and its hit records returned. */
            const addSet = async (set: Omit<RcSet, 'buffer'>): Promise<ArrayBuffer | null> => {
                const buffer = set.count > 0 ? uploadRays(device, `rc ${set.name} rays`, set.rays) : null;
                if (buffer) gpuBuffers.push(buffer);
                sets.push({ ...set, buffer });
                return buffer && set.mode === 'closest' ? prepare.trace(set.mode, buffer, set.count) : null;
            };

            const placementRays = makePlacementRays({ position: camera.position, target: camera.target, fovYDeg: camera.fovYDeg }, view, RC_HALTON_FIRST_INDEX);
            const placementHits = await addSet({ name: 'placement', mode: 'closest', rays: placementRays, count: tiles.count });
            if (!placementHits) throw new Error('no placement rays');
            const probes = placeProbes(placementRays, placementHits, scene.meshes, scene.instances);
            probesPlaced = probes.count;
            tilesMissed = probes.missCount;
            progress(`${label}: ${probes.count} of ${tiles.count} tiles placed a probe (${probes.missCount} saw sky or nothing, ${probes.degenerateCount} hit a collapsed triangle)`);
            if (probes.count <= 0 || probes.count > tiles.count) {
                throw new Error(`${probes.count} probes placed for ${tiles.count} tiles: there must be at least one and at most one per tile`);
            }

            const probeRays = makeProbeRays(probes, RC_RAYS_PER_PROBE, random);
            const probeCount = probes.count * RC_RAYS_PER_PROBE;
            const probeHits = await addSet({ name: 'probe', mode: 'closest', rays: probeRays, count: probeCount });
            const probeSurfaces = probeHits ? hitSurfaces(probeRays, probeHits, scene.meshes, scene.instances) : null;
            await yieldToBrowser();

            const bounceCount = probeSurfaces && probeSurfaces.count > 0 ? multiBounceRayCount(probeCount, RC_MULTI_BOUNCE_RATIO) : 0;
            const bounceRays = probeSurfaces ? makeBounceRays(probeSurfaces, bounceCount, random) : new Float32Array(0);
            const bounceHits = await addSet({ name: 'bounce', mode: 'closest', rays: bounceRays, count: bounceCount });
            const bounceSurfaces = bounceHits ? hitSurfaces(bounceRays, bounceHits, scene.meshes, scene.instances) : null;
            await yieldToBrowser();

            const noSurfaces = { positions: new Float64Array(0), normals: new Float64Array(0), count: 0 };
            const shadowOfProbe = makeLightShadowRays(probeSurfaces ?? noSurfaces, lights, random);
            await addSet({ name: 'shadow-probe', mode: 'any', rays: shadowOfProbe.rays, count: shadowOfProbe.count, facingAway: shadowOfProbe.facingAwayCount });
            const shadowOfBounce = makeLightShadowRays(bounceSurfaces ?? noSurfaces, lights, random);
            await addSet({ name: 'shadow-bounce', mode: 'any', rays: shadowOfBounce.rays, count: shadowOfBounce.count, facingAway: shadowOfBounce.facingAwayCount });
            progress(
                `${label}: ${probeCount} probe rays (${probeSurfaces?.count ?? 0} hit), ${bounceCount} multi-bounce rays (${bounceSurfaces?.count ?? 0} hit), ` +
                    `${shadowOfProbe.count} + ${shadowOfBounce.count} shadow rays toward ${lights.kind === 'sun' ? 'the sun' : `one of ${lights.positions.length} lights`} ` +
                    `(${shadowOfProbe.facingAwayCount} + ${shadowOfBounce.facingAwayCount} hits face away from theirs)`,
            );
        } finally {
            prepare.dispose();
        }

        // ---- Every ray must be usable before any number means anything. ----
        const problems = sets.flatMap((set) => rayProblems(set.name, set.rays));
        if (problems.length > 0) throw new Error(`bad rays: ${problems.join('; ')}`);

        // ---- Time each set with each tracer. ----
        const timedSets: RaySetBuffer[] = sets.flatMap((set) =>
            set.buffer ? [{ raySet: set.name, mode: set.mode, rays: set.buffer, count: set.count }] : [],
        );
        for (const set of sets) if (!set.buffer) result.skipped.push(`rc ${set.name}: no rays (nothing to time)`);
        const runs = new Map<RcTracer, RaySetRun[]>();
        for (const tracer of config.tracers) {
            if (!(RC_TRACERS as readonly string[]).includes(tracer)) {
                result.skipped.push(`${tracer}: ray-set mode rc times only ${RC_TRACERS.join(' and ')}`);
                continue;
            }
            try {
                progress(`${label}: timing ${tracer} on ${timedSets.length} sets`);
                let tracerRuns: RaySetRun[];
                if (tracer === 'platform') {
                    const run = await runPlatformTracerOnSets(device, context.timer, scene.meshes, scene.instances, timedSets, context.plan);
                    result.platform = run.build;
                    tracerRuns = run.sets;
                } else {
                    tracerRuns = await runOursTracerOnSets(device, context.timer, 'ours-tinybvh', ours.data, timedSets, context.plan);
                }
                runs.set(tracer as RcTracer, tracerRuns);
                for (const run of tracerRuns) {
                    result.modes.push(run.mode);
                    const loaded = run.mode.loaded;
                    progress(
                        `${label}: ${tracer} ${run.mode.raySet} (${run.mode.mode}, ${run.mode.rays} rays): ${run.mode.timing.medianMs.toFixed(3)} ms single, ` +
                            `${run.mode.mraysPerSecond.toFixed(1)} Mrays/s` +
                            (loaded ? `; loaded ${loaded.medianMs.toFixed(3)} ms, ${loaded.mraysPerSecond.toFixed(1)} Mrays/s over ${loaded.dispatches} dispatches` : ''),
                    );
                }
            } catch (error) {
                result.skipped.push(`${tracer}: ${error instanceof Error ? error.message : String(error)}`);
                progress(`${label}: ${tracer} skipped (${error instanceof Error ? error.message : String(error)})`);
            }
            await yieldToBrowser();
        }
        if (runs.size === 0) throw new Error(`no tracer ran in ray-set mode rc (${result.skipped.join('; ')})`);

        // ---- Agreement, the reference sample, and the hit fractions. ----
        const answersOf = (tracer: RcTracer, name: RcSetName): ArrayBuffer | null =>
            runs.get(tracer)?.find((run) => run.mode.raySet === name)?.results ?? null;
        const primary: RcTracer = runs.has('platform') ? 'platform' : 'ours-tinybvh';
        progress(`${label}: checking ${RC_ORACLE_SAMPLE_RAYS} rays per set against the brute-force reference (CPU, every triangle)`);
        const reference = buildReferenceScene(
            scene.meshes,
            scene.instances.map((instance, index) => ({ meshIndex: instance.meshIndex, instanceIndex: index, objectToWorld: instance.objectToWorld })),
        );
        const setResults: RcSetResult[] = [];
        for (const [setNumber, set] of sets.entries()) {
            const answers = answersOf(primary, set.name);
            const other = answersOf('ours-tinybvh', set.name);
            const platformAnswers = answersOf('platform', set.name);
            let agreementRate: number | null = null;
            if (other && platformAnswers) {
                agreementRate =
                    set.mode === 'closest'
                        ? compareClosestDistances(extractHitDistances(other), extractHitDistances(platformAnswers)).rate
                        : compareBlocked(new Uint32Array(other), new Uint32Array(platformAnswers)).rate;
            }
            const tally = answers ? await checkAgainstReference(reference, set, answers, RC_ORACLE_SAMPLE_RAYS, RC_SEED + setNumber + 1) : newTally();
            if (tally.mismatches > 0) progress(`${label}: ${set.name} disagrees with the reference: ${tally.examples.join(' | ')}`);
            const hits = answers ? countHits(set, answers) : 0;
            setResults.push({
                set: set.name,
                mode: set.mode,
                rays: set.count,
                hits,
                hitFraction: set.count > 0 ? hits / set.count : 0,
                ...(set.facingAway === undefined ? {} : { facingAway: set.facingAway }),
                oracleRays: tally.rays,
                oracleMismatches: tally.mismatches,
                oracleAmbiguous: tally.ambiguous,
                agreementRate,
            });
        }
        const rc: RcSceneResult = {
            camera: { name: camera.name, position: [...camera.position], target: [...camera.target], fovYDeg: camera.fovYDeg },
            width: RC_VIEW.width,
            height: RC_VIEW.height,
            tileSize: RC_TILE_SIZE,
            tiles: tiles.count,
            probesPlaced,
            tilesMissed,
            raysPerProbe: RC_RAYS_PER_PROBE,
            multiBounceRatio: RC_MULTI_BOUNCE_RATIO,
            seed: RC_SEED,
            lightKind: lights.kind,
            lightCount: lights.kind === 'sun' ? 1 : lights.positions.length,
            sets: RC_SET_NAMES.flatMap((name) => setResults.filter((entry) => entry.set === name)),
        };
        result.rc = rc;
        for (const entry of rc.sets) {
            progress(
                `${label}: ${entry.set}: ${entry.rays} rays, ${(entry.hitFraction * 100).toFixed(1)}% ${entry.mode === 'closest' ? 'hit' : 'blocked'}` +
                    (entry.agreementRate === null ? '' : `, tracers agree on ${(entry.agreementRate * 100).toFixed(3)}%`) +
                    `, reference: ${entry.oracleRays - entry.oracleMismatches - entry.oracleAmbiguous} match, ${entry.oracleAmbiguous} ambiguous, ${entry.oracleMismatches} mismatch`,
            );
        }
        for (const prediction of rcPredictions(result)) {
            progress(
                `${label}: ${prediction.tracer} predicted tracing per frame ${prediction.totalMs.toFixed(3)} ms ` +
                    `(${prediction.sets.map((entry) => `${entry.set} ${entry.ms.toFixed(3)}`).join(', ')}; ${prediction.source} rates)`,
            );
        }
    } finally {
        for (const buffer of gpuBuffers) buffer.destroy();
    }
    return result;
}
