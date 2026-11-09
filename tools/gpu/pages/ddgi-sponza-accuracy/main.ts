/**
 * Sponza accuracy (T1.9): DDGI's indirect light on Sponza through the three
 * adapter, rendered with the engine-free test view, for comparison with an
 * *untextured* Cycles reference.
 *
 * Why untextured: in M1 the platform samples no textures. The adapter sends
 * each material one constant GI albedo (the albedo texture's average times
 * the metalness fold, DECISIONS 2026-11-05, T1.7). The fully textured Cycles
 * reference would mix that texture error with the GI error. So this page
 * also exports the GI material table the adapter actually sent, and
 * `render.py --materials-json` renders Cycles with exactly those constants.
 *
 * Steps:
 * 1. Load `sponza` (base + curtains) with three's GLTFLoader (the viewer's
 *    loader), and leave the sky dome out of GI.
 * 2. Sync it through `ThreeGI` into a platform with DDGI on the manifest's
 *    volume, the manifest's sun (from the scene's DirectionalLight) and the
 *    manifest's sky (`platformSkyEnvironment`, the same sky Cycles reads).
 * 3. Converge, by this rule (`SETTLE_RULE` below):
 *    - warm-up: `warmupPasses` full passes over every probe with a running
 *      average (pass k keeps k/(k+1) of the old value; the viewer's warm-up);
 *    - then blocks of `blockPasses` full passes at the normal hysteresis.
 *      After each block the test view is rendered; the field counts as
 *      settled when the indirect image's mean moved by less than
 *      `settleTolerance` since the previous block (after at least
 *      `minBlocks` blocks). The page fails if `maxBlocks` blocks pass without
 *      settling.
 * 4. Render the test view at the bookmark: indirect = albedo / π × E from
 *    the field (as the Cornell check), and the full image.
 *
 * The field holds sky light reaching a surface straight, which Cycles'
 * indirect pass counts as direct; run.ts therefore compares with Cycles'
 * "everything but the lights' direct light" image (`cycles-untextured-gi.exr`).
 *
 * Probe updates use a fixed `units` budget (whole passes split into equal
 * frames), not the millisecond budget, so the run doesn't depend on the
 * budget controller.
 *
 * Experiments (the Sponza +26% investigation) go through the job, e.g.
 * `--job '{"parts":["sponza-intel"],"pathTrace":{"paths":128,"bounces":12,"backFaces":"shaded"}}'`:
 * `ddgi` settings and volume over the defaults, `parts` to keep,
 * `excludeMaterials` to leave out of GI, `pathTrace` for the same view from
 * the page's path tracer (ddgi-shared/path-tracer.ts; it agrees with Cycles
 * to +0.15% on Sponza, so it is the reference for changed scenes), and
 * `probeCheck` for probes against path-traced irradiance at their own
 * positions. Probe-state counts are always reported, overall and for the
 * probes near the curtains (the curtain meshes' boxes grown by 0.5 m).
 *
 * The curtains investigation added `partsOutOfGI` (a part loaded but left out
 * of GI), `singleSidedFrontSide` (only glTF double-sided materials reach the
 * platform double-sided) and `diagnosticLookup` (the test view with exact
 * visibility rays, or times short-range occlusion). With `exportImages` the
 * page also returns the test view's albedo and position images; run.ts uses
 * the albedo to split the comparison by material.
 *
 * Run `node tools/gpu/pages/ddgi-sponza-accuracy/run.ts` for the EXRs, the
 * material table and the comparison. `pnpm test:gpu --filter
 * ddgi-sponza-accuracy` (manual: it needs the downloaded Sponza) runs the
 * same page and reports the means and the convergence only.
 */

import { Box3, FrontSide, PerspectiveCamera, Vector3, type Material, type Mesh, type Object3D } from 'three/webgpu';
import type { DDGISettingsInput, DDGIVolumeSettings } from '../../../../packages/ddgi-core/src/techniques/ddgi/settings.ts';
import { PathTracer, type PathTraceOptions } from '../ddgi-shared/path-tracer.ts';
import { ProbeValues } from '../ddgi-shared/probe-values.ts';
import { assert } from '../../client.ts';
import type { MaterialDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { ThreeGI } from '../../../../packages/ddgi-three/src/three-gi.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import type { CameraBookmark } from '../../../../scenes/src/manifest.ts';
import { floatsToBase64, imageMean, readProbeData, TestView, type TestViewLookup } from '../ddgi-shared/harness.ts';
import { runSteps, runThreePage } from '../three-shared/harness.ts';

export interface SponzaAccuracyJob {
    /** 1-based camera bookmark. */
    readonly camera: number;
    /**
     * The Cycles reference's stored camera (`shot.camera` in its json, T1.13),
     * used instead of the bookmark so both images show exactly the same view.
     */
    readonly view?: CameraBookmark;
    readonly width: number;
    readonly height: number;
    /** Full passes with a running average before the normal hysteresis. */
    readonly warmupPasses: number;
    /** Full passes between two checks of the image mean. */
    readonly blockPasses: number;
    /** Settled when the indirect mean moves by less than this share between two blocks. */
    readonly settleTolerance: number;
    readonly minBlocks: number;
    readonly maxBlocks: number;
    /** Return the images as base64 floats and the material table (for run.ts). */
    readonly exportImages: boolean;
    /**
     * Experiments (T1.9 Sponza investigation). DDGI settings over the
     * defaults; `volume` fields replace the manifest volume's.
     */
    readonly ddgi: Partial<Omit<DDGISettingsInput, 'volume'>> & { readonly volume?: Partial<DDGIVolumeSettings> };
    /** Model parts to load, by asset id (e.g. only `sponza-intel` for the base package); null loads all. */
    readonly parts: readonly string[] | null;
    /** Materials (glTF names) whose meshes are left out of GI entirely: not traced, not seen by the test view. */
    readonly excludeMaterials: readonly string[];
    /** Also render the test view's indirect image with the page's path tracer (ddgi-shared/path-tracer.ts). */
    readonly pathTrace: PathTraceOptions | null;
    /**
     * Compare probes with path-traced irradiance at their own positions:
     * `probes` active probes in front of the camera, each in the 6 axis
     * directions, `paths` paths per direction.
     */
    readonly probeCheck: (PathTraceOptions & { readonly probes: number }) | null;
    /**
     * Model parts (asset ids) loaded but left out of GI: not traced, not seen
     * by the test view. Unlike `parts`, their meshes still mark the region the
     * probe census looks at, so a run without the curtains can be compared
     * probe for probe with one with them.
     */
    readonly partsOutOfGI: readonly string[];
    /**
     * Experiment: mark every material three draws `FrontSide` single-sided for
     * GI (`userData.giSingleSided`), so only materials authored double-sided
     * (glTF `doubleSided`: the curtains, glass) reach the platform as
     * double-sided. By default the adapter sends every material double-sided.
     */
    readonly singleSidedFrontSide: boolean;
    /**
     * Also render the indirect image with one of the test view's diagnostic
     * lookups (ddgi-shared/harness.ts, `TestViewLookup`): `tracedVisibility`
     * (exact visibility rays instead of the Chebyshev test) or
     * `shortRangeOcclusion` (the field times short-range occlusion).
     */
    readonly diagnosticLookup: Exclude<TestViewLookup, 'field'> | null;
    /** `shortRangeOcclusion` only: one image per ray count in the list. */
    readonly diagnosticRays: readonly number[];
}

const DEFAULT_SPONZA_ACCURACY_JOB: SponzaAccuracyJob = {
    camera: 1,
    width: 640,
    height: 360,
    warmupPasses: 16,
    blockPasses: 32,
    settleTolerance: 0.005,
    minBlocks: 2,
    maxBlocks: 12,
    exportImages: false,
    ddgi: {},
    parts: null,
    excludeMaterials: [],
    pathTrace: null,
    probeCheck: null,
    partsOutOfGI: [],
    singleSidedFrontSide: false,
    diagnosticLookup: null,
    diagnosticRays: [32],
};

/** The model part whose probes the census looks at, and how far around its meshes' boxes it looks. */
const CENSUS_PART = 'sponza-intel-curtains';
const CENSUS_MARGIN_METRES = 0.5;

declare global {
    interface Window {
        ddgiSponzaAccuracyJob?: Partial<SponzaAccuracyJob>;
    }
}

const job: SponzaAccuracyJob = { ...DEFAULT_SPONZA_ACCURACY_JOB, ...window.ddgiSponzaAccuracyJob };

/** The DDGI probe updates allowed in one frame by the technique's default `maxProbesPerFrame`. */
const MAX_PROBES_PER_FRAME = 4096;

/**
 * One row of the GI material table: what the platform received for one
 * three material, keyed by the model part it came from and its glTF name
 * (Blender's importer keeps the same names, so render.py can find it).
 */
export interface GiMaterialEntry {
    /** The manifest part's asset id (several parts can have a material of the same name). */
    readonly part: string;
    /** The glTF material name. */
    readonly name: string;
    /** Diffuse albedo as sent: colour × the albedo texture's average × (1 − metalness), linear. */
    readonly albedo: [number, number, number];
    /** Emitted radiance as the platform stores it: emissive × the emissive texture's average × emissiveIntensity. */
    readonly emission: [number, number, number];
    /**
     * False for see-through materials (alpha-blended, glass, hidden). The M1
     * tracer doesn't honour it per triangle (every surface is opaque until
     * M4); instead the adapter leaves out of GI each mesh whose materials all
     * block no light (`leftOutMeshes`).
     */
    readonly blocksLight: boolean;
    /** Sent, but not yet honoured by the M1 tracer (see `blocksLight`). */
    readonly alphaCutoff: number | null;
    readonly doubleSided: boolean;
    /** Meshes in the part that use it. */
    readonly meshes: number;
    /**
     * Of those, the meshes the adapter left out of GI because all their
     * materials let light through (M1 to T4.2; always 0 since T4.3). render.py
     * removes this material's faces when it equals `meshes`, and stops when
     * it is between 0 and `meshes` (it can't tell those meshes apart).
     */
    readonly leftOutMeshes: number;
}

export interface GiMaterialTable {
    readonly scene: string;
    /** The model parts loaded (asset ids), so render.py loads the same ones. */
    readonly parts: readonly string[];
    readonly note: string;
    readonly materials: GiMaterialEntry[];
}

function tableEntry(part: string, name: string, descriptor: MaterialDescriptor): Omit<GiMaterialEntry, 'meshes' | 'leftOutMeshes'> {
    // The same defaults the platform's scene copy applies (platform-scene.ts, writeMaterialRecord).
    // Since T4.3 the adapter sends metalness apart and the platform samples the textures: the constant is colour × (1 − metalness).
    const diffuse = 1 - (descriptor.metalness ?? 0);
    const albedo = (descriptor.albedo ?? [1, 1, 1]).map((channel) => channel * diffuse);
    const emissive = descriptor.emissive ?? [0, 0, 0];
    const intensity = descriptor.emissiveIntensity ?? 1;
    return {
        part,
        name,
        albedo: [albedo[0], albedo[1], albedo[2]],
        emission: [emissive[0] * intensity, emissive[1] * intensity, emissive[2] * intensity],
        blocksLight: descriptor.blocksLight !== false,
        alphaCutoff: descriptor.alphaCutoff ?? null,
        doubleSided: descriptor.doubleSided !== false,
    };
}

/** The part an object was loaded from: load-scene.ts names each part's holder `<asset>/<path>`. */
function partOf(object: Object3D, holders: ReadonlyMap<Object3D, string>): string | null {
    for (let current: Object3D | null = object; current; current = current.parent) {
        const part = holders.get(current);
        if (part) return part;
    }
    return null;
}

/**
 * The GI material table for the scene's model parts. Materials that share a
 * part and a name (GLTFLoader clones a material for meshes with vertex
 * colours) must have been sent with the same values; otherwise the name
 * can't identify one constant and the page fails.
 */
function giMaterialTable(threeGI: ThreeGI, content: Object3D, holders: ReadonlyMap<Object3D, string>): GiMaterialEntry[] {
    const entries = new Map<string, GiMaterialEntry>();
    content.traverse((object) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh) return;
        const part = partOf(mesh, holders);
        assert(part, `mesh '${mesh.name}' is in no model part`);
        const materials: Material[] = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        // Since T4.3 the adapter leaves no mesh out: the platform lets light through see-through materials triangle by triangle.
        const leftOut = 0;
        for (const material of materials) {
            const descriptor = threeGI.sync.sentMaterial(material);
            assert(descriptor, `material '${material.name}' of mesh '${mesh.name}' was never sent to the platform`);
            const entry = tableEntry(part, material.name, descriptor);
            const key = `${part}\n${material.name}`;
            const existing = entries.get(key);
            if (existing) {
                const same = JSON.stringify({ ...existing, meshes: 0, leftOutMeshes: 0 }) === JSON.stringify({ ...entry, meshes: 0, leftOutMeshes: 0 });
                assert(same, `two materials named '${material.name}' in ${part} were sent with different GI values`);
                entries.set(key, { ...existing, meshes: existing.meshes + 1, leftOutMeshes: existing.leftOutMeshes + leftOut });
            } else {
                entries.set(key, { ...entry, meshes: 1, leftOutMeshes: leftOut });
            }
        }
    });
    return [...entries.values()];
}

function percent(value: number): string {
    return `${(value * 100).toFixed(2)}%`;
}

const AXIS_DIRECTIONS: readonly (readonly [number, number, number])[] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];
const AXIS_NAMES = ['+x', '-x', '+y (up)', '-y (down)', '+z', '-z'];

/** A small seeded generator (mulberry32), so the probe sample is the same run to run. */
function seededRandom(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let value = Math.imul(state ^ (state >>> 15), 1 | state);
        value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
}

interface ViewBookmark {
    readonly position: readonly [number, number, number];
    readonly target: readonly [number, number, number];
    readonly fovYDeg: number;
}

/** Probes nearer than this to the camera, or farther, are not checked. */
const PROBE_CHECK_RANGE = { near: 0.5, far: 30 } as const;

/**
 * Probes against path-traced irradiance at their own (relocated) positions,
 * per axis direction. Picks active probes in front of the camera (inside
 * a cone of the horizontal field of view at 16:9, within 30 m). Reports the
 * summed ratio overall, per direction and per height.
 */
async function checkProbes(
    device: GPUDevice,
    ddgi: DDGITechnique,
    tracer: PathTracer,
    bookmark: ViewBookmark,
    options: PathTraceOptions & { readonly probes: number },
    log: (message: string) => void,
    platform: GIPlatformImpl,
): Promise<Record<string, unknown>> {
    const field = ddgi.field;
    assert(field, 'DDGI has no field');
    const records = await readProbeData(device, field);
    const { counts, origin, spacing } = field.grid;
    const forward = bookmark.target.map((value, axis) => value - bookmark.position[axis]);
    const forwardLength = Math.hypot(...forward);
    const horizontalHalfFov = Math.atan(Math.tan((bookmark.fovYDeg * Math.PI) / 360) * (16 / 9));
    const coneCosine = Math.cos(horizontalHalfFov);
    const candidates: { index: number; position: [number, number, number] }[] = [];
    records.forEach((record, index) => {
        if (record.state !== 0) return;
        const x = index % counts[0];
        const z = Math.floor(index / counts[0]) % counts[2];
        const y = Math.floor(index / (counts[0] * counts[2]));
        const position = [x, y, z].map((cell, axis) => origin[axis] + (cell + record.offset[axis]) * spacing[axis]) as [number, number, number];
        const toProbe = position.map((value, axis) => value - bookmark.position[axis]);
        const distance = Math.hypot(...toProbe);
        if (distance > PROBE_CHECK_RANGE.far || distance < PROBE_CHECK_RANGE.near) return;
        const cosine = toProbe.reduce((sum, value, axis) => sum + value * forward[axis], 0) / (distance * forwardLength);
        if (cosine < coneCosine) return;
        candidates.push({ index, position });
    });
    const random = seededRandom(1234);
    for (let index = candidates.length - 1; index > 0; index--) {
        const other = Math.floor(random() * (index + 1));
        [candidates[index], candidates[other]] = [candidates[other], candidates[index]];
    }
    const chosen = candidates.slice(0, options.probes);
    const directionsPerProbe = AXIS_DIRECTIONS.length;
    const queries = chosen.flatMap((probe) => AXIS_DIRECTIONS.map((direction) => ({ probe: probe.index, direction })));
    const stored = await (await ProbeValues.create(device, platform)).read(field, queries);
    assert(
        stored.every((value) => value.state === 0),
        'a chosen probe is not active',
    );
    const traceStart = performance.now();
    const traced = await tracer.irradianceAt(
        chosen.flatMap((probe) => AXIS_DIRECTIONS.map((direction) => ({ position: probe.position, normal: direction }))),
        options,
    );
    const storedValues = stored.map((value) => value.irradiance);
    const sum = (values: readonly (readonly [number, number, number])[], pick: (index: number) => boolean): number =>
        values.reduce((total, value, index) => (pick(index) ? total + value[0] + value[1] + value[2] : total), 0);
    const ratio = (pick: (index: number) => boolean): number => sum(storedValues, pick) / sum(traced, pick);
    const byDirection: Record<string, number> = {};
    AXIS_NAMES.forEach((name, direction) => {
        byDirection[name] = ratio((index) => index % directionsPerProbe === direction);
    });
    const byHeight: Record<string, number> = {};
    for (const [name, low, high] of [
        ['below 2 m', -Infinity, 2],
        ['2 to 6 m', 2, 6],
        ['6 to 11 m', 6, 11],
        ['above 11 m', 11, Infinity],
    ] as const) {
        const probeInBand = (probe: { position: [number, number, number] }): boolean => probe.position[1] >= low && probe.position[1] < high;
        const probesInBand = chosen.filter(probeInBand).length;
        if (probesInBand > 0) byHeight[`${name} (${probesInBand} probes)`] = ratio((index) => probeInBand(chosen[Math.floor(index / directionsPerProbe)]));
    }
    // Per probe (summed over its directions), for the spread.
    const onProbe = (probe: number) => (index: number) => Math.floor(index / directionsPerProbe) === probe;
    const perProbeDetail = chosen.map((probe, index) => ({
        probe: probe.index,
        position: probe.position.map((value) => Number(value.toFixed(3))),
        stored: sum(storedValues, onProbe(index)),
        traced: sum(traced, onProbe(index)),
        ratio: ratio(onProbe(index)),
    }));
    const perProbe = perProbeDetail.map((detail) => detail.ratio).sort((a, b) => a - b);
    const quantile = (share: number): number => perProbe[Math.min(perProbe.length - 1, Math.floor(perProbe.length * share))];
    // Probes far brighter than the truth at their own position: where the field is wrong at the source.
    const outliers = [...perProbeDetail].sort((a, b) => b.ratio - a.ratio).slice(0, 25);
    const overTwice = perProbeDetail.filter((detail) => detail.ratio > 2);
    const storedTotal = sum(storedValues, () => true);
    const result = {
        candidates: candidates.length,
        probes: chosen.length,
        paths: options.paths,
        bounces: options.bounces,
        backFaces: options.backFaces,
        traceMs: Math.round(performance.now() - traceStart),
        storedOverTraced: ratio(() => true),
        byDirection,
        byHeight,
        perProbeQuantiles: { p10: quantile(0.1), median: quantile(0.5), p90: quantile(0.9) },
        probesOverTwice: overTwice.length,
        storedShareOfProbesOverTwice: overTwice.reduce((total, detail) => total + detail.stored, 0) / storedTotal,
        outliers,
    };
    log(
        `probe check: ${chosen.length} of ${candidates.length} active probes in view; stored / path traced ${result.storedOverTraced.toFixed(4)}; ` +
            `by direction ${JSON.stringify(byDirection)}; by height ${JSON.stringify(byHeight)}; per probe ${JSON.stringify(result.perProbeQuantiles)}`,
    );
    return result;
}

const SETTLE_RULE =
    'warm-up passes with a running average, then blocks of full passes at the normal hysteresis until the test view indirect mean moves by less than the tolerance between blocks';

await runThreePage(async ({ device, data, log }) => {
    const manifest = loadAllManifests().find((candidate) => candidate.id === 'sponza');
    assert(manifest, 'no sponza manifest');
    const bookmark = job.view ?? manifest.cameras[job.camera - 1];
    assert(bookmark, `sponza has no camera ${job.camera}`);
    data.job = job;
    data.camera = bookmark.name;

    const loadStart = performance.now();
    const keptParts = job.parts;
    const loadedManifest = keptParts ? { ...manifest, parts: manifest.parts.filter((part) => part.kind !== 'model' || keptParts.includes(part.asset)) } : manifest;
    data.parts = loadedManifest.parts.map((part) => (part.kind === 'model' ? part.asset : part.kind));
    const loaded = await loadScene(loadedManifest, () => {});
    data.loadMs = Math.round(performance.now() - loadStart);
    const scene = loaded.scene;

    // The model parts: load-scene.ts puts each under a Group named `<asset>/<path>`.
    const holders = new Map<Object3D, string>();
    for (const part of loadedManifest.parts) {
        if (part.kind !== 'model') continue;
        const holder = loaded.content.getObjectByName(`${part.asset}/${part.path}`);
        assert(holder, `no loaded object for part ${part.asset}`);
        holders.set(holder, part.asset);
    }
    // Only the model is GI geometry; the sky dome is the viewer's background (GI gets the sky as its environment).
    const content = new Set<Object3D>();
    loaded.content.traverse((object) => content.add(object));
    scene.traverse((object) => {
        if ((object as Mesh).isMesh && !content.has(object)) object.userData.giContributes = false;
    });
    // Experiments: meshes of excluded materials leave GI altogether.
    let excludedMeshes = 0;
    loaded.content.traverse((object) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh) return;
        const materials: Material[] = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        if (materials.some((material) => job.excludeMaterials.includes(material.name))) {
            mesh.userData.giContributes = false;
            excludedMeshes++;
        }
    });
    data.excludedMeshes = excludedMeshes;
    // Parts kept for the census but out of GI.
    let meshesOutOfGI = 0;
    loaded.content.traverse((object) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh) return;
        const part = partOf(mesh, holders);
        if (part && job.partsOutOfGI.includes(part)) {
            mesh.userData.giContributes = false;
            meshesOutOfGI++;
        }
    });
    data.meshesOutOfGI = meshesOutOfGI;
    // The census region: the curtain meshes' world boxes, grown by the margin.
    loaded.content.updateMatrixWorld(true);
    const censusBoxes: Box3[] = [];
    loaded.content.traverse((object) => {
        const mesh = object as Mesh;
        if (mesh.isMesh && partOf(mesh, holders) === CENSUS_PART) censusBoxes.push(new Box3().setFromObject(mesh).expandByScalar(CENSUS_MARGIN_METRES));
    });
    data.censusBoxes = censusBoxes.map((box) => ({ min: box.min.toArray(), max: box.max.toArray() }));
    if (job.singleSidedFrontSide) {
        const singleSided = new Set<string>();
        loaded.content.traverse((object) => {
            const mesh = object as Mesh;
            if (!mesh.isMesh) return;
            for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
                if (material.side === FrontSide) {
                    material.userData.giSingleSided = true;
                    singleSided.add(material.name);
                }
            }
        });
        data.singleSidedMaterials = [...singleSided];
    }

    const camera = new PerspectiveCamera(bookmark.fovYDeg, job.width / job.height, 0.05, loaded.diagonal * 2);
    camera.position.set(...bookmark.position);
    camera.lookAt(...bookmark.target);

    const platform = new GIPlatformImpl(device, { label: 'ddgi sponza accuracy' });
    const volume: DDGIVolumeSettings = { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing, ...job.ddgi.volume };
    const ddgi = platform.addTechnique(new DDGITechnique({ warmUpPasses: 0, ...job.ddgi, volume }));
    platform.setOutput(ddgi);
    await platform.prepare();
    const threeGI = new ThreeGI(platform, scene, { budget: { units: 0 }, environment: platformSkyEnvironment(manifest) });
    try {
        // The first update copies the scene; then wait for the mesh trees and place them.
        threeGI.update(camera, 1 / 60);
        await platform.rays.whenBuilt();
        threeGI.update(camera, 1 / 60);
        assert(!platform.rays.building, 'the platform still reports trees building');

        const table = giMaterialTable(threeGI, loaded.content, holders);
        data.materials = table.length;
        data.emissiveMaterials = table.filter((entry) => entry.emission.some((channel) => channel > 0)).map((entry) => `${entry.part}/${entry.name}`);
        data.seeThroughMaterials = table.filter((entry) => !entry.blocksLight).map((entry) => `${entry.part}/${entry.name}`);
        data.leftOutMeshes = table.filter((entry) => entry.leftOutMeshes > 0).map((entry) => `${entry.part}/${entry.name}: ${entry.leftOutMeshes} of ${entry.meshes}`);
        if (job.exportImages) {
            const exported: GiMaterialTable = {
                scene: manifest.id,
                parts: [...holders.values()],
                note:
                    'GI materials as the three adapter sent them to the platform (SceneSync.sentMaterial), albedo = colour × (1 − metalness). ' +
                    'Since T4.3 GI also samples the textures and honours blocksLight and alpha per triangle, which this constant table does not hold.',
                materials: table,
            };
            data.giMaterials = exported;
        }

        const field = ddgi.field;
        assert(field, 'DDGI has no field after prepare');
        const probeCount = field.grid.probeCount;
        // Whole passes split into equal frames, so pass boundaries fall on frame boundaries.
        const framesPerPass = Math.ceil(probeCount / MAX_PROBES_PER_FRAME);
        const unitsPerFrame = Math.ceil(probeCount / framesPerPass);
        threeGI.budget = { units: unitsPerFrame };
        data.probes = field.grid.counts;
        data.probeCount = probeCount;
        data.framesPerPass = framesPerPass;
        const normalHysteresis = ddgi.settings.hysteresis;
        const runPasses = (passes: number): Promise<void> => runSteps(device, passes * framesPerPass, () => threeGI.update(camera, 1 / 60));

        const convergeStart = performance.now();
        for (let pass = 0; pass < job.warmupPasses; pass++) {
            // Pass k keeps k/(k+1): the field is the plain mean of the passes so far.
            ddgi.settings.hysteresis = Math.min(normalHysteresis, pass / (pass + 1));
            await runPasses(1);
        }
        ddgi.settings.hysteresis = normalHysteresis;

        const view = await TestView.create(device, platform);
        const means: number[] = [];
        let images = await view.render(field, bookmark, job.width, job.height);
        let settled = false;
        for (let block = 1; block <= job.maxBlocks; block++) {
            await runPasses(job.blockPasses);
            const previous = imageMean(images.indirect);
            images = await view.render(field, bookmark, job.width, job.height);
            const mean = imageMean(images.indirect);
            means.push(mean);
            const change = mean / previous - 1;
            log(`block ${block} (${job.warmupPasses + block * job.blockPasses} passes): indirect mean ${mean.toFixed(6)}, ${percent(change)} since the last block`);
            if (block >= job.minBlocks && Math.abs(change) < job.settleTolerance) {
                settled = true;
                break;
            }
        }
        data.convergeMs = Math.round(performance.now() - convergeStart);
        data.blockMeans = means;
        data.passes = job.warmupPasses + means.length * job.blockPasses;
        data.settleRule = SETTLE_RULE;
        assert(settled, `the indirect mean did not settle within ±${percent(job.settleTolerance)} in ${job.maxBlocks} blocks`);

        let nonFinite = 0;
        for (const image of [images.indirect, images.full]) for (const value of image) if (!Number.isFinite(value)) nonFinite++;
        assert(nonFinite === 0, `${nonFinite} non-finite pixel values`);
        data.indirectMean = imageMean(images.indirect);
        data.fullMean = imageMean(images.full);
        assert((data.indirectMean as number) > 0, 'the test view is black');
        log(`sponza / ${bookmark.name}: ${String(data.passes)} passes over ${probeCount} probes in ${String(data.convergeMs)} ms; indirect mean ${(data.indirectMean as number).toFixed(6)}, full ${(data.fullMean as number).toFixed(6)}`);
        if (job.exportImages) {
            data.indirectBase64 = floatsToBase64(images.indirect);
            data.fullBase64 = floatsToBase64(images.full);
            data.albedoBase64 = floatsToBase64(images.albedo);
            data.positionBase64 = floatsToBase64(images.position);
        }
        if (job.diagnosticLookup) {
            // One image per ray count for the short-range term; the other lookups take no count.
            const rayCounts = job.diagnosticLookup === 'shortRangeOcclusion' ? job.diagnosticRays : [0];
            const means: Record<string, number> = {};
            const images: Record<string, string> = {};
            for (const rays of rayCounts) {
                const diagnosticView = await TestView.create(device, platform, job.diagnosticLookup, rays > 0 ? rays : undefined);
                const diagnostic = await diagnosticView.render(field, bookmark, job.width, job.height);
                const key = rays > 0 ? `r${rays}` : 'image';
                means[key] = imageMean(diagnostic.indirect);
                log(`indirect mean with the ${job.diagnosticLookup} lookup (${key}): ${means[key].toFixed(6)}`);
                if (job.exportImages) images[key] = floatsToBase64(diagnostic.indirect);
            }
            data.diagnosticMeans = means;
            if (job.exportImages) data.diagnosticImages = images;
        }

        // Probe states and relocations, for comparing experiments (layout.ts PROBE_STATE: 0 active, 1 off inside, 2 off no surface, 3 not updated).
        const probeRecords = await readProbeData(device, field);
        const stateCounts = [0, 0, 0, 0];
        let relocated = 0;
        for (const record of probeRecords) {
            stateCounts[record.state] = (stateCounts[record.state] ?? 0) + 1;
            if (record.offset.some((value) => value !== 0)) relocated++;
        }
        data.probeStates = { active: stateCounts[0], offInside: stateCounts[1], offNoSurface: stateCounts[2], notUpdated: stateCounts[3], relocated };
        log(`probe states: ${JSON.stringify(data.probeStates)}`);
        // The same counts for the probes whose grid point lies near the curtains
        // (CENSUS_PART's mesh boxes, grown by the margin), with the mean offset.
        if (censusBoxes.length > 0) {
            const { counts, origin, spacing } = field.grid;
            const near = [0, 0, 0, 0];
            let nearRelocated = 0;
            let offsetSum = 0;
            const gridPoint = new Vector3();
            probeRecords.forEach((record, index) => {
                const x = index % counts[0];
                const z = Math.floor(index / counts[0]) % counts[2];
                const y = Math.floor(index / (counts[0] * counts[2]));
                gridPoint.set(origin[0] + x * spacing[0], origin[1] + y * spacing[1], origin[2] + z * spacing[2]);
                if (!censusBoxes.some((box) => box.containsPoint(gridPoint))) return;
                near[record.state] = (near[record.state] ?? 0) + 1;
                if (record.offset.some((value) => value !== 0)) nearRelocated++;
                offsetSum += Math.hypot(record.offset[0] * spacing[0], record.offset[1] * spacing[1], record.offset[2] * spacing[2]);
            });
            const total = near.reduce((sum, value) => sum + value, 0);
            data.probeStatesNearCurtains = {
                probes: total,
                active: near[0],
                offInside: near[1],
                offNoSurface: near[2],
                notUpdated: near[3],
                relocated: nearRelocated,
                meanOffsetMetres: total > 0 ? offsetSum / total : 0,
                marginMetres: CENSUS_MARGIN_METRES,
            };
            log(`probe states near the curtains: ${JSON.stringify(data.probeStatesNearCurtains)}`);
        }

        if (job.pathTrace || job.probeCheck) {
            const tracer = await PathTracer.create(device, platform);
            if (job.pathTrace) {
                const traceStart = performance.now();
                const traced = await tracer.renderIndirect(bookmark, job.width, job.height, job.pathTrace);
                data.pathTraceMs = Math.round(performance.now() - traceStart);
                data.pathTracedMean = imageMean(traced);
                const ratio = (data.indirectMean as number) / (data.pathTracedMean as number);
                log(`path-traced indirect mean ${(data.pathTracedMean as number).toFixed(6)} (${job.pathTrace.paths} paths, ${job.pathTrace.bounces} rays, back faces ${job.pathTrace.backFaces}) in ${String(data.pathTraceMs)} ms; DDGI / path traced ${ratio.toFixed(4)}`);
                if (job.exportImages) data.pathTracedBase64 = floatsToBase64(traced);
            }
            if (job.probeCheck) data.probeCheck = await checkProbes(device, ddgi, tracer, bookmark, job.probeCheck, log, platform);
        }
    } finally {
        threeGI.dispose();
        platform.dispose();
    }
});
