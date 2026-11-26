/**
 * The radiance cache's storage buffers per shader stage, for every setting
 * that changes what its kernels bind (pmndrs/gi#49): Chrome on Apple GPUs
 * allows 10 (the M5 Pro), so no pipeline of a configuration the cache can be
 * in may bind more. Every pipeline the platform makes for the configuration
 * is counted (the platform's own, the cache's, the denoiser's), on a fake
 * device in Node (platform/testing/storage-inventory.ts), with and without
 * subgroups (the kernels' two paths).
 *
 * `GI_STORAGE_INVENTORY=1 pnpm vitest run storage-buffers` prints the
 * inventory: per configuration the worst pipeline and every pipeline over 8
 * (three's own device), DDGI's and Split RC's included.
 */

import { describe, expect, it } from 'vitest';
import { GIPlatformImpl } from '../../platform/gi-platform.ts';
import { createInventoryGpu, describeOver, recordComputePipelines, worstPipeline, type InventoryPipeline } from '../../platform/testing/storage-inventory.ts';
import { DDGITechnique } from '../ddgi/ddgi-technique.ts';
import { NativeRadianceCascades } from '../radiance-cascades/native.ts';
import { loadMlp } from './neural-cache/mlp-loader.ts';
import { RadianceCacheTechnique } from './radiance-cache-technique.ts';
import type { RadianceCacheSettings } from './settings.ts';

/** The most storage buffers per stage Chrome gives on Apple GPUs (M5 Pro, 2026-11-26). */
const APPLE_STORAGE_BUFFERS_PER_STAGE = 10;
/** What a device three.js makes itself has (WebGPU's default; NOTES). */
const DEFAULT_STORAGE_BUFFERS_PER_STAGE = 8;

/** The optional sibling the neural tail runs on, where this checkout has it at its pin; null otherwise. */
const mlp = await loadMlp().catch(() => null);

interface Configuration {
    readonly name: string;
    readonly settings: Partial<RadianceCacheSettings>;
    /** DDGI stacked under the cache (what `secondBounce: 'auto'` and `'fallback'` read). */
    readonly ddgiBelow?: boolean;
    readonly needsMlp?: boolean;
}

/** Every setting that changes what the cache's kernels bind, one at a time. */
const CONFIGURATIONS: readonly Configuration[] = [
    { name: 'default', settings: {} },
    { name: "worldCache 'sharc'", settings: { worldCache: 'sharc' } },
    { name: "cellShading 'budgeted'", settings: { cellShading: 'budgeted' } },
    { name: "secondBounce 'gi1-rays'", settings: { secondBounce: 'gi1-rays' } },
    { name: "secondBounce 'fallback' (DDGI below)", settings: { secondBounce: 'fallback' }, ddgiBelow: true },
    { name: "secondBounce 'auto' (DDGI below)", settings: { secondBounce: 'auto' }, ddgiBelow: true },
    { name: "neuralCache 'tail'", settings: { neuralCache: 'tail' }, needsMlp: true },
    { name: 'ssgi on', settings: { ssgi: true } },
    { name: 'denoiser off', settings: { denoiser: false } },
];

const FEATURE_SETS = [
    { name: 'subgroups', features: ['subgroups', 'shader-f16', 'timestamp-query'] },
    { name: 'no subgroups', features: ['timestamp-query'] },
] as const;

const DDGI_VOLUME = { center: [0, 1.5, 0] as [number, number, number], size: [4, 3, 4] as [number, number, number], spacing: 1 };

/** Every compute pipeline a platform with the radiance cache at `configuration` makes in `prepare`. */
async function radianceCachePipelines(configuration: Configuration, features: readonly string[]): Promise<InventoryPipeline[]> {
    const gpu = createInventoryGpu(features);
    const pipelines = recordComputePipelines(gpu.device);
    const platform = new GIPlatformImpl(gpu.device);
    const technique = platform.addTechnique(new RadianceCacheTechnique(configuration.settings));
    if (configuration.ddgiBelow) {
        const ddgi = platform.addTechnique(new DDGITechnique({ volume: DDGI_VOLUME }));
        platform.setFallback(technique, ddgi);
    }
    await platform.prepare();
    platform.dispose();
    return pipelines;
}

async function ddgiPipelines(features: readonly string[], directLight: 'deferred' | 'inline'): Promise<InventoryPipeline[]> {
    const gpu = createInventoryGpu(features);
    const pipelines = recordComputePipelines(gpu.device);
    const platform = new GIPlatformImpl(gpu.device);
    platform.addTechnique(new DDGITechnique({ volume: DDGI_VOLUME }, { directLight }));
    await platform.prepare();
    platform.dispose();
    return pipelines;
}

async function splitRadianceCascadesPipelines(roughSpecular: boolean): Promise<InventoryPipeline[]> {
    const gpu = createInventoryGpu();
    const pipelines = recordComputePipelines(gpu.device);
    await NativeRadianceCascades.create(gpu.device, { bounds: { min: [-1, -1, -1], max: [1, 1, 1] }, grid: [8, 8, 8], firstInterval: 0.1, maxDistance: 4, roughSpecular });
    return pipelines;
}

const printing = process.env.GI_STORAGE_INVENTORY === '1';

function print(title: string, pipelines: readonly InventoryPipeline[]): void {
    if (!printing) return;
    const worst = worstPipeline(pipelines);
    const over8 = describeOver(pipelines, DEFAULT_STORAGE_BUFFERS_PER_STAGE);
    console.log([`== ${title}: ${pipelines.length} pipelines, worst ${worst?.storageBuffers ?? 0} (${worst?.label ?? 'none'})`, ...over8.map((line) => `   ${line}`)].join('\n'));
}

describe('storage buffers per shader stage (pmndrs/gi#49)', () => {
    for (const { name: featureName, features } of FEATURE_SETS) {
        for (const configuration of CONFIGURATIONS) {
            it.skipIf(configuration.needsMlp === true && !mlp)(`the radiance cache, ${configuration.name}, ${featureName}: at most ${APPLE_STORAGE_BUFFERS_PER_STAGE}`, async () => {
                const pipelines = await radianceCachePipelines(configuration, features);
                print(`radiance cache, ${configuration.name}, ${featureName}`, pipelines);
                // The configuration's own passes were made (a pass left out would go uncounted).
                const labels = pipelines.map((pipeline) => pipeline.label).join('\n');
                if (configuration.settings.worldCache === 'sharc') expect(labels).toMatch(/Sharc/);
                else expect(labels).toMatch(/PopulateCells/);
                if (configuration.ddgiBelow) expect(labels).toMatch(/SampleBounceSourceCells/);
                if (configuration.needsMlp) expect(labels).toMatch(/NeuralTail/);
                if (configuration.settings.ssgi) expect(labels).toMatch(/ssgi/i);
                // The gate on APPLE_STORAGE_BUFFERS_PER_STAGE comes with the packing that meets it (#49).
                expect(worstPipeline(pipelines)?.storageBuffers).toBeLessThanOrEqual(15);
            });
        }
    }

    it(`DDGI binds at most ${DEFAULT_STORAGE_BUFFERS_PER_STAGE} in every pass (deferred and inline direct light)`, async () => {
        for (const { name: featureName, features } of FEATURE_SETS) {
            for (const directLight of ['deferred', 'inline'] as const) {
                const pipelines = await ddgiPipelines(features, directLight);
                print(`DDGI, directLight '${directLight}', ${featureName}`, pipelines);
                expect(describeOver(pipelines, DEFAULT_STORAGE_BUFFERS_PER_STAGE)).toEqual([]);
            }
        }
    });

    it(`Split RC binds at most ${DEFAULT_STORAGE_BUFFERS_PER_STAGE} in every pass`, async () => {
        for (const roughSpecular of [false, true]) {
            const pipelines = await splitRadianceCascadesPipelines(roughSpecular);
            print(`Split RC, roughSpecular ${roughSpecular}`, pipelines);
            expect(pipelines.length).toBeGreaterThan(0);
            expect(describeOver(pipelines, DEFAULT_STORAGE_BUFFERS_PER_STAGE)).toEqual([]);
        }
    });
});
