/**
 * Storage buffers per shader stage in every pipeline gi builds
 * (pmndrs/gi#49). Chrome on Apple GPUs allows 10 (the M5 Pro), so no
 * pipeline any technique or the platform can make may bind more.
 *
 * Every pipeline is made on a fake device in Node and counted
 * (platform/testing/storage-inventory.ts), with and without subgroups (the
 * kernels' two paths):
 *
 * - the radiance cache in every combination of the settings that change what
 *   its pipelines bind. Those are found by trying every discrete setting
 *   (`worldCache`, `cellShading`, `secondBounce`, `neuralCache` with the
 *   network's own pipelines, SSGI, the denoiser and the probe filter among
 *   them) from several starting points, and must be the ones listed in
 *   CACHE_BINDING_AXES, which are crossed;
 * - DDGI the same way; Split RC (the solver and the transport it traces
 *   through); the platform's pipelines made after `prepare` (mesh packing,
 *   ray batches, the SH export);
 * - a guard: every source file in packages/ that makes a pipeline is listed
 *   here, as counted (its pipelines must turn up in an inventory) or as
 *   binding no storage buffers (its source must declare none). A new file
 *   that makes pipelines fails until it is added.
 *
 * `GI_STORAGE_INVENTORY=1 pnpm vitest run storage-buffers` prints the
 * inventory: per configuration the worst pipeline and every pipeline over 8
 * (three's own device), and the settings found to change bindings.
 */

import { readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { GIPlatformImpl } from '../platform/gi-platform.ts';
import { GpuMeshPacker } from '../platform/scene/gpu-mesh-packing.ts';
import { createInventoryGpu, describeOver, recordComputePipelines, worstPipeline, type InventoryPipeline } from '../platform/testing/storage-inventory.ts';
import { TraceBatchKernel } from '../platform/trace/trace-gpu.ts';
import { DDGITechnique, type DDGITechniqueOptions } from './ddgi/ddgi-technique.ts';
import { exportIrradianceSH } from './ddgi/irradiance-sh.ts';
import { DEFAULT_DDGI_SETTINGS, type DDGISettingsInput } from './ddgi/settings.ts';
import { loadMlp } from './radiance-cache/neural-cache/mlp-loader.ts';
import { RADIANCE_CACHE_STORAGE_BUFFERS_PER_STAGE, RadianceCacheTechnique } from './radiance-cache/radiance-cache-technique.ts';
import { DEFAULT_RADIANCE_CACHE_SETTINGS, type RadianceCacheSettings } from './radiance-cache/settings.ts';
import { NativeRadianceCascades } from './radiance-cascades/native.ts';
import { createRcTransport } from './radiance-cascades/transport.ts';

/** The most storage buffers per stage Chrome gives on Apple GPUs (M5 Pro, 2026-11-26). */
const APPLE_STORAGE_BUFFERS_PER_STAGE = 10;
/** What a device three.js makes itself has (WebGPU's default; NOTES). DDGI and Split RC fit it. */
const DEFAULT_STORAGE_BUFFERS_PER_STAGE = 8;

/** The optional sibling the neural tail runs on, where this checkout has it at its pin; null otherwise. */
const mlp = await loadMlp().catch(() => null);

const FEATURE_SETS = [
    { name: 'subgroups', features: ['subgroups', 'shader-f16', 'timestamp-query'] },
    { name: 'no subgroups', features: ['timestamp-query'] },
] as const;

const DDGI_VOLUME = { center: [0, 1.5, 0] as [number, number, number], size: [4, 3, 4] as [number, number, number], spacing: 1 };

const printing = process.env.GI_STORAGE_INVENTORY === '1';

/** Every pipeline label any inventory in this file recorded, for the guard at the end. */
const seenLabels = new Set<string>();

/** Every compute pipeline `build` makes on a fresh inventory device. */
async function inventory(features: readonly string[], build: (device: GPUDevice) => Promise<void>): Promise<InventoryPipeline[]> {
    const gpu = createInventoryGpu(features);
    const pipelines = recordComputePipelines(gpu.device);
    await build(gpu.device);
    for (const pipeline of pipelines) seenLabels.add(pipeline.label);
    return pipelines;
}

/** What two runs compare to tell whether a setting changed bindings: each pipeline and its storage buffers. */
function bindingSignature(pipelines: readonly InventoryPipeline[]): string {
    return pipelines
        .map((pipeline) => `${pipeline.label}|${pipeline.buffers.map((buffer) => buffer.name).join(',')}`)
        .sort()
        .join('\n');
}

function print(title: string, pipelines: readonly InventoryPipeline[]): void {
    if (!printing) return;
    const worst = worstPipeline(pipelines);
    const over8 = describeOver(pipelines, DEFAULT_STORAGE_BUFFERS_PER_STAGE);
    console.log([`== ${title}: ${pipelines.length} pipelines, worst ${worst?.storageBuffers ?? 0} (${worst?.label ?? 'none'})`, ...over8.map((line) => `   ${line}`)].join('\n'));
}

/** A discrete setting: its values, the default first. */
interface Axis<C> {
    readonly name: string;
    readonly values: readonly unknown[];
    readonly apply: (configuration: C, value: unknown) => C;
}

/** Every combination of the axes' values. */
function crossProduct<C>(start: C, axes: readonly Axis<C>[]): C[] {
    let configurations = [start];
    for (const axis of axes) configurations = configurations.flatMap((configuration) => axis.values.map((value) => axis.apply(configuration, value)));
    return configurations;
}

/**
 * The axes whose values change what some pipeline binds, from any of
 * `bases`: each value of each axis is tried on each base and compared with it.
 */
async function bindingAxes<C>(
    bases: readonly C[],
    axes: readonly Axis<C>[],
    run: (configuration: C) => Promise<InventoryPipeline[]>,
    allowed: (configuration: C) => boolean = () => true,
): Promise<Set<string>> {
    const found = new Set<string>();
    for (const base of bases) {
        const reference = bindingSignature(await run(base));
        for (const axis of axes) {
            if (found.has(axis.name)) continue;
            for (const value of axis.values) {
                const configuration = axis.apply(base, value);
                if (allowed(configuration) && bindingSignature(await run(configuration)) !== reference) {
                    found.add(axis.name);
                    break;
                }
            }
        }
    }
    return found;
}

/** An axis per boolean of `defaults`, named after its key. */
function booleanAxes<C>(defaults: object, set: (configuration: C, key: string, value: unknown) => C): Axis<C>[] {
    return Object.entries(defaults)
        .filter(([, value]) => typeof value === 'boolean')
        .map(([key, value]) => ({ name: key, values: [value, !value], apply: (configuration: C, next: unknown) => set(configuration, key, next) }));
}

// ---------------------------------------------------------------- the radiance cache

interface CacheConfiguration {
    readonly settings: Partial<RadianceCacheSettings>;
    /** DDGI stacked under the cache (what `secondBounce: 'auto'` and `'fallback'` read). */
    readonly ddgiBelow: boolean;
}

function cacheSetting(configuration: CacheConfiguration, key: string, value: unknown): CacheConfiguration {
    return { ...configuration, settings: { ...configuration.settings, [key]: value } };
}

function cacheSubSetting(group: 'sharc' | 'neural', key: string) {
    return (configuration: CacheConfiguration, value: unknown): CacheConfiguration => ({
        ...configuration,
        settings: { ...configuration.settings, [group]: { ...configuration.settings[group], [key]: value } },
    });
}

function cacheAxes(entries: readonly (readonly [string, readonly unknown[]])[], apply: (name: string) => Axis<CacheConfiguration>['apply'], prefix = ''): Axis<CacheConfiguration>[] {
    return entries.map(([name, values]) => ({ name: `${prefix}${name}`, values, apply: apply(name) }));
}

/** Every discrete setting of the cache, of its SHaRC world cache and of its neural tail, and DDGI below it. */
const CACHE_AXES: readonly Axis<CacheConfiguration>[] = [
    ...booleanAxes<CacheConfiguration>(DEFAULT_RADIANCE_CACHE_SETTINGS, cacheSetting),
    ...cacheAxes(
        [
            ['worldCache', ['gi1-hash-grid', 'sharc']],
            ['cellShading', ['per-hit', 'budgeted']],
            ['secondBounce', ['auto', 'gi1-rays', 'fallback', 'shared-cell']],
            ['neuralCache', ['off', 'tail']],
            ['denoiserHistoryClamp', ['neighbourhood', 'none']],
            ['probeBlend', ['running-mean', 'capsaicin-hysteresis']],
            ['emptyCellFill', ['keep-previous', 'capsaicin']],
            // Numbers whose 0 turns a part off.
            ['cellChangeResetHistory', [0, 8]],
            ['ssgiHistoryFrames', [DEFAULT_RADIANCE_CACHE_SETTINGS.ssgiHistoryFrames, 0]],
            ['fallbackIndirectSampleCount', [4, 0, 16]],
        ],
        (name) => (configuration, value) => cacheSetting(configuration, name, value),
    ),
    ...cacheAxes(
        [
            ['normalBins', [true, false]],
            ['pyramid', [true, false]],
            ['cacheTail', [true, false]],
            ['changeReset', [true, false]],
            ['fallbackLevels', [3, 0]],
            ['bounceRays', [2, 0, 1]],
        ],
        (name) => cacheSubSetting('sharc', name),
        'sharc.',
    ),
    ...cacheAxes(
        [
            ['network', ['compact', 'default']],
            ['precision', ['auto', 'f16', 'f32']],
            ['ema', ['off', 'warmup', 'plain']],
            ['screenStride', [8, 0]],
        ],
        (name) => cacheSubSetting('neural', name),
        'neural.',
    ),
    { name: 'ddgiBelow', values: [false, true], apply: (configuration, value) => ({ ...configuration, ddgiBelow: value as boolean }) },
];

/**
 * The settings that change the cache's bindings, as found on 2026-11-27.
 * The test finds them again and fails if the set moved, so a new one gets
 * crossed with the others.
 */
const CACHE_BINDING_AXES = ['ssgi', 'worldCache', 'secondBounce', 'neuralCache', 'ddgiBelow'];

async function cachePipelines(configuration: CacheConfiguration, features: readonly string[]): Promise<InventoryPipeline[]> {
    return inventory(features, async (device) => {
        const platform = new GIPlatformImpl(device);
        const technique = platform.addTechnique(new RadianceCacheTechnique(configuration.settings));
        if (configuration.ddgiBelow) {
            const ddgi = platform.addTechnique(new DDGITechnique({ volume: DDGI_VOLUME }));
            platform.setFallback(technique, ddgi);
        }
        await platform.prepare();
        platform.dispose();
    });
}

function describeCache(configuration: CacheConfiguration): string {
    const settings = Object.entries(configuration.settings).map(([key, value]) => `${key}=${typeof value === 'object' ? JSON.stringify(value) : String(value)}`);
    if (configuration.ddgiBelow) settings.push('DDGI below');
    return settings.length > 0 ? settings.join(', ') : 'defaults';
}

// ---------------------------------------------------------------- DDGI

interface DDGIConfiguration {
    readonly settings: DDGISettingsInput;
    readonly options: DDGITechniqueOptions;
}

function ddgiSetting(configuration: DDGIConfiguration, key: string, value: unknown): DDGIConfiguration {
    return { ...configuration, settings: { ...configuration.settings, [key]: value } as DDGISettingsInput };
}

function ddgiOption(key: keyof DDGITechniqueOptions) {
    return (configuration: DDGIConfiguration, value: unknown): DDGIConfiguration => ({ ...configuration, options: { ...configuration.options, [key]: value } });
}

const LIGHT_GROUPS = { count: 2, weights: [[1, 1, 1], [1, 1, 1]], environmentWeight: [1, 1, 1] };

function ddgiVolumes(configuration: DDGIConfiguration, value: unknown): DDGIConfiguration {
    const settings: Record<string, unknown> = { ...configuration.settings };
    delete settings.volume;
    delete settings.volumes;
    if (value === 'one') settings.volume = DDGI_VOLUME;
    else if (value === 'two') settings.volumes = [DDGI_VOLUME, { ...DDGI_VOLUME, center: [6, 1.5, 0], priority: 1 }];
    else settings.volume = { ...DDGI_VOLUME, follow: 'camera' };
    return { ...configuration, settings: settings as unknown as DDGISettingsInput };
}

/** Every discrete setting and option of DDGI. */
const DDGI_AXES: readonly Axis<DDGIConfiguration>[] = [
    ...booleanAxes<DDGIConfiguration>(DEFAULT_DDGI_SETTINGS, ddgiSetting),
    { name: 'lightGroups', values: [null, LIGHT_GROUPS], apply: (configuration, value) => ddgiSetting(configuration, 'lightGroups', value) },
    { name: 'volumes', values: ['one', 'two', 'camera'], apply: ddgiVolumes },
    { name: 'directLight', values: ['deferred', 'inline'], apply: ddgiOption('directLight') },
    { name: 'reduction', values: [undefined, 'workgroup'], apply: ddgiOption('reduction') },
    { name: 'directWorkList', values: [undefined, 'atomics'], apply: ddgiOption('directWorkList') },
];

/** What DDGI refuses at prepare: light groups and the sun layer need direct light deferred (settings.ts). */
function ddgiAllowed(configuration: DDGIConfiguration): boolean {
    const layers = configuration.settings.lightGroups != null || configuration.settings.sunLayer === true;
    return !(layers && configuration.options.directLight === 'inline');
}

/** The settings that change DDGI's bindings, as found on 2026-11-27 (the test finds them again). */
const DDGI_BINDING_AXES = ['lightGroups', 'sunLayer', 'volumes', 'directLight', 'reduction', 'directWorkList'];

async function ddgiPipelines(configuration: DDGIConfiguration, features: readonly string[]): Promise<InventoryPipeline[]> {
    return inventory(features, async (device) => {
        const platform = new GIPlatformImpl(device);
        platform.addTechnique(new DDGITechnique(configuration.settings, configuration.options));
        await platform.prepare();
        platform.dispose();
    });
}

// ---------------------------------------------------------------- Split RC

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

async function splitRadianceCascadesPipelines(roughSpecular: boolean, features: readonly string[]): Promise<InventoryPipeline[]> {
    return inventory(features, async (device) => {
        await NativeRadianceCascades.create(device, { bounds: { min: [-1, -1, -1], max: [1, 1, 1] }, grid: [8, 8, 8], firstInterval: 0.1, maxDistance: 4, roughSpecular });
        // The transport the solver traces through: its own kernels and @pmndrs/raytracer's interval tracers.
        await createRcTransport(device, {
            meshes: [{ positions: Float32Array.of(-1, -1, 0, 0, 1, 0, 1, -1, 0) }],
            instances: [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY }],
            materials: [{ albedo: [1, 1, 1], emission: [0, 0, 0] }],
            lights: [{ position: [0, 0, 2], intensity: [1, 1, 1] }],
        });
    });
}

// ---------------------------------------------------------------- the guard

/** packages/, from this file. */
const PACKAGES = fileURLToPath(new URL('../../../', import.meta.url));

/** A call that makes a pipeline (a mention in a comment has no `(` after the name). */
const MAKES_PIPELINE = /\bcreate(Compute|Render)Pipeline(Async)?\s*\(/;

/**
 * Every source file that makes pipelines, and how this test covers it: a
 * pattern its pipelines' labels match in some inventory above, or
 * `'no storage buffers'` when its pipelines bind only textures and uniforms
 * (checked: the file declares no storage buffer).
 */
const PIPELINE_SOURCES: Readonly<Record<string, RegExp | 'no storage buffers'>> = {
    'bvh/src/gpu-hploc.ts': /mesh trees hploc/,
    'bvh/src/gpu-lbvh.ts': /mesh trees lbvh/,
    'bvh/src/gpu-mesh-refit.ts': /^gi mesh refit$/,
    'bvh/src/gpu-radix-sort.ts': /mesh trees sort/,
    'bvh/src/gpu-refit.ts': /refit bvhRefit/,
    'bvh/src/gpu-scan.ts': /mesh trees scan/,
    'ddgi-core/src/platform/compute/gpu-compute.ts': /gi\/indirect-args|gi\/compaction/,
    'ddgi-core/src/platform/kernel-builder.ts': /radiance-cache\/|ddgi\//,
    'ddgi-core/src/platform/scene/gpu-mesh-packing.ts': /gi mesh pack/,
    'ddgi-core/src/platform/scene/gpu-pose-copy.ts': /gi\/pose-copy/,
    'ddgi-core/src/platform/screen/screen-history.ts': /gi\/screen-history/,
    'ddgi-core/src/platform/screen/screen-inputs-view.ts': 'no storage buffers',
    'ddgi-core/src/platform/screen/ssgi.ts': /ssgi/,
    'ddgi-core/src/platform/sky/platform-sky.ts': /gi sky /,
    'ddgi-core/src/platform/textures/texture-store.ts': /gi\/texture-store/,
    'ddgi-core/src/platform/trace/trace-gpu.ts': /batch (closest|any)/,
    'ddgi-core/src/techniques/ddgi/irradiance-sh.ts': /export-sh/,
    'ddgi-core/src/techniques/radiance-cascades/native.ts': /^RC (begin|gather)/,
    'ddgi-core/src/techniques/radiance-cascades/transport.ts': /^RC transport /,
    'ddgi-three/src/instance-id-resolve.ts': 'no storage buffers',
};

/** Pipelines from dependencies, made on our device through the files above: the labels they must show up under. */
const DEPENDENCY_PIPELINES: Readonly<Record<string, RegExp>> = {
    'denoiser@2.1.0 (Gi1Denoiser)': /^gi1 /,
    '@pmndrs/raytracer (RtIntervalTracer)': /^RC interval /,
    ...(mlp ? { '@pathtracer/mlp (the neural tail)': /^nn / } : {}),
};

function sourceFiles(directory: string): string[] {
    return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
        const path = join(directory, entry.name);
        if (entry.isDirectory()) return entry.name === 'node_modules' || entry.name === 'testing' ? [] : sourceFiles(path);
        return entry.name.endsWith('.ts') && !entry.name.endsWith('.test.ts') && !entry.name.endsWith('.d.ts') ? [path] : [];
    });
}

// ---------------------------------------------------------------- tests

describe('storage buffers per shader stage in every pipeline gi builds (pmndrs/gi#49)', () => {
    it(`the radiance cache, every combination of the settings that change its bindings: at most ${APPLE_STORAGE_BUFFERS_PER_STAGE}`, { timeout: 300_000 }, async () => {
        const start: CacheConfiguration = { settings: {}, ddgiBelow: false };
        const bases: CacheConfiguration[] = [
            start,
            cacheSetting(start, 'worldCache', 'sharc'),
            cacheSetting(start, 'cellShading', 'budgeted'),
            { settings: { secondBounce: 'fallback' }, ddgiBelow: true },
            ...(mlp ? [cacheSetting(start, 'neuralCache', 'tail')] : []),
        ];
        const found = await bindingAxes(bases, CACHE_AXES, (configuration) => cachePipelines(configuration, FEATURE_SETS[0].features));
        if (printing) console.log(`radiance cache settings that change bindings: ${[...found].join(', ')}`);
        // Without the optional mlp the tail stays off, so it changes nothing.
        expect([...found].sort()).toEqual(CACHE_BINDING_AXES.filter((name) => mlp || name !== 'neuralCache').sort());

        const crossed = CACHE_AXES.filter((axis) => CACHE_BINDING_AXES.includes(axis.name));
        let worst = 0;
        const over: string[] = [];
        for (const { name: featureName, features } of FEATURE_SETS) {
            for (const configuration of crossProduct(start, crossed)) {
                const pipelines = await cachePipelines(configuration, features);
                const title = `radiance cache, ${describeCache(configuration)}, ${featureName}`;
                print(title, pipelines);
                // The configuration's own passes were made (a pass left out would go uncounted).
                const labels = pipelines.map((pipeline) => pipeline.label).join('\n');
                expect(labels, title).toMatch(configuration.settings.worldCache === 'sharc' ? /Sharc/ : /PopulateCells/);
                expect(labels, title).toMatch(/^gi1 /m);
                if (configuration.settings.ssgi) expect(labels, title).toMatch(/ssgi/i);
                if (configuration.settings.neuralCache === 'tail' && mlp && configuration.settings.worldCache !== 'sharc') expect(labels, title).toMatch(/^nn /m);
                over.push(...describeOver(pipelines, APPLE_STORAGE_BUFFERS_PER_STAGE).map((line) => `${title}: ${line}`));
                worst = Math.max(worst, worstPipeline(pipelines)?.storageBuffers ?? 0);
            }
        }
        expect(over).toEqual([]);
        // What the technique declares is its real need: the most any pass of any configuration binds.
        expect(worst).toBe(RADIANCE_CACHE_STORAGE_BUFFERS_PER_STAGE);
    });

    it(`the radiance cache declares its ${APPLE_STORAGE_BUFFERS_PER_STAGE} in its requirements, so the platform reports it`, () => {
        expect(RADIANCE_CACHE_STORAGE_BUFFERS_PER_STAGE).toBeLessThanOrEqual(APPLE_STORAGE_BUFFERS_PER_STAGE);
        expect(new RadianceCacheTechnique().requirements.limits).toEqual({ maxStorageBuffersPerShaderStage: RADIANCE_CACHE_STORAGE_BUFFERS_PER_STAGE });
        const platform = new GIPlatformImpl(createInventoryGpu().device);
        platform.addTechnique(new RadianceCacheTechnique());
        expect(platform.limitNeeds).toEqual({ maxStorageBuffersPerShaderStage: RADIANCE_CACHE_STORAGE_BUFFERS_PER_STAGE });
        platform.dispose();
    });

    it(`DDGI, every combination of the settings that change its bindings: at most ${DEFAULT_STORAGE_BUFFERS_PER_STAGE}`, { timeout: 300_000 }, async () => {
        const start: DDGIConfiguration = { settings: { volume: DDGI_VOLUME }, options: {} };
        const bases = [start, ddgiSetting(start, 'lightGroups', LIGHT_GROUPS), ddgiOption('directLight')(start, 'inline')];
        const found = await bindingAxes(bases, DDGI_AXES, (configuration) => ddgiPipelines(configuration, FEATURE_SETS[0].features), ddgiAllowed);
        if (printing) console.log(`DDGI settings that change bindings: ${[...found].join(', ')}`);
        expect([...found].sort()).toEqual([...DDGI_BINDING_AXES].sort());

        const crossed = DDGI_AXES.filter((axis) => DDGI_BINDING_AXES.includes(axis.name));
        const over: string[] = [];
        for (const { name: featureName, features } of FEATURE_SETS) {
            for (const configuration of crossProduct(start, crossed).filter(ddgiAllowed)) {
                const title = `DDGI, ${JSON.stringify({ ...configuration.options, lightGroups: configuration.settings.lightGroups ? 2 : 0, sunLayer: configuration.settings.sunLayer ?? false, volumes: configuration.settings.volumes?.length ?? configuration.settings.volume?.follow ?? 1 })}, ${featureName}`;
                const pipelines = await ddgiPipelines(configuration, features);
                print(title, pipelines);
                expect(pipelines.map((pipeline) => pipeline.label).join('\n'), title).toMatch(/ddgi\/trace/);
                over.push(...describeOver(pipelines, DEFAULT_STORAGE_BUFFERS_PER_STAGE).map((line) => `${title}: ${line}`));
            }
        }
        expect(over).toEqual([]);
    });

    it(`Split RC, the solver (with and without rough specular) and its transport: at most ${DEFAULT_STORAGE_BUFFERS_PER_STAGE}`, async () => {
        for (const { name: featureName, features } of FEATURE_SETS) {
            for (const roughSpecular of [false, true]) {
                const pipelines = await splitRadianceCascadesPipelines(roughSpecular, features);
                print(`Split RC, roughSpecular ${roughSpecular}, ${featureName}`, pipelines);
                const labels = pipelines.map((pipeline) => pipeline.label).join('\n');
                expect(labels).toMatch(/^RC transport /m);
                expect(labels).toMatch(/^RC interval /m);
                expect(describeOver(pipelines, DEFAULT_STORAGE_BUFFERS_PER_STAGE)).toEqual([]);
            }
        }
    });

    it(`the platform's pipelines made after prepare (mesh packing, ray batches, the SH export): at most ${APPLE_STORAGE_BUFFERS_PER_STAGE}`, async () => {
        for (const { name: featureName, features } of FEATURE_SETS) {
            const pipelines = await inventory(features, async (device) => {
                await new GpuMeshPacker(device, { label: 'gi mesh pack' }).prepare();
                await TraceBatchKernel.create(device);
                const platform = new GIPlatformImpl(device);
                const ddgi = platform.addTechnique(new DDGITechnique({ volume: DDGI_VOLUME }));
                await platform.prepare();
                await exportIrradianceSH(ddgi);
                platform.dispose();
            });
            print(`platform after prepare, ${featureName}`, pipelines);
            const labels = pipelines.map((pipeline) => pipeline.label).join('\n');
            expect(labels).toMatch(/gi mesh pack/);
            expect(labels).toMatch(/batch closest/);
            expect(labels).toMatch(/export-sh/);
            expect(describeOver(pipelines, APPLE_STORAGE_BUFFERS_PER_STAGE)).toEqual([]);
        }
    });

    // Reads what the tests above counted, so it runs after them (alone, with -t, it reports everything uncounted).
    it('every source file that makes a pipeline is covered above (counted, or declares no storage buffer)', () => {
        const makers = ['bvh/src', 'ddgi-core/src', 'ddgi-three/src']
            .flatMap((directory) => sourceFiles(join(PACKAGES, directory)))
            .filter((path) => MAKES_PIPELINE.test(readFileSync(path, 'utf8')))
            .map((path) => relative(PACKAGES, path).replaceAll('\\', '/'))
            .sort();
        expect(makers).toEqual(Object.keys(PIPELINE_SOURCES).sort());
        const labels = [...seenLabels];
        const uncovered: string[] = [];
        for (const [file, coverage] of Object.entries(PIPELINE_SOURCES)) {
            if (coverage === 'no storage buffers') {
                if (/var<storage|'storage'|'read-only-storage'/.test(readFileSync(join(PACKAGES, file), 'utf8'))) uncovered.push(`${file}: declares a storage buffer`);
            } else if (!labels.some((label) => coverage.test(label))) uncovered.push(`${file}: no pipeline labelled ${String(coverage)} was counted`);
        }
        for (const [dependency, pattern] of Object.entries(DEPENDENCY_PIPELINES)) {
            if (!labels.some((label) => pattern.test(label))) uncovered.push(`${dependency}: no pipeline labelled ${String(pattern)} was counted`);
        }
        expect(uncovered).toEqual([]);
    });
});
