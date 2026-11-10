import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { buildProceduralParts, mergeScenes } from './build-scene';
import { loadManifests } from './export';
import {
    checkManifestAgainstSources,
    type AssetOutputInfo,
    defaultExposureEv,
    isHeavyScene,
    isProceduralOnly,
    manifestLightToSceneLight,
    NO_RANGE_WINDOW,
    parseManifest,
    sceneSettingValues,
    type ModelTransform,
    validateManifest,
    type SceneManifest,
} from './manifest';
import { gltfWorldBounds, modelTransformMatrix, readModelJson } from './model-bounds';
import { buildProcedural, proceduralBuilders } from './procedural';
import { sceneBounds } from './procedural/geometry';

const knownBuilders = Object.keys(proceduralBuilders);
const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

/** The smallest manifest that passes, to mutate in each test. */
function validManifest(): SceneManifest {
    return {
        id: 'test-scene',
        title: 'Test scene',
        description: 'A scene used by the unit tests.',
        tags: ['interior'],
        units: 'meters',
        upAxis: 'y',
        parts: [{ kind: 'procedural', builder: 'cornell-box' }],
        cameras: [
            { name: 'one', position: [0, 1, 5], target: [0, 1, 0], fovYDeg: 60 },
            { name: 'two', position: [3, 1, 5], target: [0, 1, 0], fovYDeg: 60 },
        ],
        lights: [],
        environment: { type: 'none' },
        volume: { center: [0, 1, 0], size: [4, 3, 4], probeSpacing: 0.5 },
        nativeCapable: true,
        credits: [{ asset: null, license: 'MIT', attribution: 'MIT, this repo.' }],
    };
}

function errorsFor(change: (manifest: Record<string, unknown>) => void): string[] {
    const manifest = validManifest() as unknown as Record<string, unknown>;
    change(manifest);
    return validateManifest(manifest, { knownBuilders }).errors;
}

const identityTransform: ModelTransform = { position: [0, 0, 0], rotationDeg: [0, 0, 0], scale: 1 };

describe('validateManifest', () => {
    it('accepts a valid manifest', () => {
        expect(validateManifest(validManifest(), { knownBuilders })).toEqual({ ok: true, errors: [] });
    });

    it('rejects things that are not objects', () => {
        expect(validateManifest(null).ok).toBe(false);
        expect(validateManifest([]).ok).toBe(false);
    });

    it('requires a kebab-case id', () => {
        expect(errorsFor((m) => (m.id = 'Not Kebab'))).toEqual([expect.stringContaining('id:')]);
    });

    it('rejects unknown top-level fields so typos do not pass silently', () => {
        expect(errorsFor((m) => (m.camreas = []))).toEqual([expect.stringContaining('camreas')]);
    });

    it('accepts a default exposure in stops and refuses a nonsense one', () => {
        expect(errorsFor((m) => (m.exposureEv = 2))).toEqual([]);
        expect(errorsFor((m) => (m.exposureEv = -1.5))).toEqual([]);
        for (const bad of ['2', Number.NaN, 40, -9]) {
            expect(errorsFor((m) => (m.exposureEv = bad))).toEqual([expect.stringContaining('exposureEv')]);
        }
        expect(defaultExposureEv(validManifest())).toBe(0);
        expect(defaultExposureEv({ ...validManifest(), exposureEv: 2 })).toBe(2);
    });

    it('requires 2 to 4 cameras', () => {
        expect(errorsFor((m) => (m.cameras = []))).toEqual([expect.stringContaining('cameras')]);
        const five = Array.from({ length: 5 }, (_, i) => ({ name: `c${i}`, position: [i, 1, 5], target: [0, 1, 0], fovYDeg: 60 }));
        expect(errorsFor((m) => (m.cameras = five))).toEqual([expect.stringContaining('cameras')]);
    });

    it('rejects a camera that looks at itself and a silly field of view', () => {
        const errors = errorsFor((m) => {
            m.cameras = [
                { name: 'a', position: [1, 1, 1], target: [1, 1, 1], fovYDeg: 60 },
                { name: 'b', position: [0, 1, 5], target: [0, 1, 0], fovYDeg: 200 },
            ];
        });
        expect(errors).toHaveLength(2);
        expect(errors[1]).toContain('cameras[1].fovYDeg');
    });

    it('rejects duplicate camera names', () => {
        const same = { name: 'x', position: [0, 1, 5], target: [0, 1, 0], fovYDeg: 60 };
        expect(errorsFor((m) => (m.cameras = [same, { ...same }]))).toEqual([expect.stringContaining('unique')]);
    });

    it('names the unknown builder and lists the known ones', () => {
        const errors = errorsFor((m) => (m.parts = [{ kind: 'procedural', builder: 'nope' }]));
        expect(errors[0]).toContain('unknown builder "nope"');
        expect(errors[0]).toContain('cornell-box');
    });

    it('checks model parts: asset id, safe path, positive scale', () => {
        const errors = errorsFor((m) => {
            m.parts = [
                {
                    kind: 'model',
                    asset: 'Bad Id',
                    path: '../outside.glb',
                    transform: { position: [0, 0, 0], rotationDeg: [0, 0, 0], scale: 0 },
                },
            ];
        });
        expect(errors.join('\n')).toContain('parts[0].asset');
        expect(errors.join('\n')).toContain('parts[0].path');
        expect(errors.join('\n')).toContain('parts[0].transform.scale');
    });

    it('rejects absolute and drive-letter model paths', () => {
        for (const path of ['/etc/x.glb', 'C:/x.glb', '\\\\share\\x.glb']) {
            const errors = errorsFor((m) => {
                m.parts = [{ kind: 'model', asset: 'a', path, transform: identityTransform }];
                m.credits = [{ asset: 'a', license: 'CC0-1.0', attribution: 'x' }];
            });
            expect(errors.join('\n')).toContain('parts[0].path');
        }
    });

    it('validates each light type', () => {
        const errors = errorsFor((m) => {
            m.lights = [
                { type: 'directional', color: [1, 1, 1], intensity: 3, direction: [0, 0, 0] },
                { type: 'point', color: [1, 1, 1], intensity: 3, position: [0, 1, 0], range: -1 },
                {
                    type: 'spot',
                    color: [1, 1, 1],
                    intensity: 3,
                    position: [0, 1, 0],
                    direction: [0, -1, 0],
                    range: 5,
                    innerConeDeg: 50,
                    outerConeDeg: 30,
                },
                { type: 'area', color: [1, 1, 1], intensity: 3 },
            ];
        });
        expect(errors).toHaveLength(4);
        expect(errors[0]).toContain('lights[0].direction');
        expect(errors[1]).toContain('lights[1].range');
        expect(errors[2]).toContain('innerConeDeg');
        expect(errors[3]).toContain('lights[3].type');
    });

    it('allows range 0 on point and spot lights, meaning no range window, but not a missing range', () => {
        const point = { type: 'point', color: [1, 1, 1], intensity: 3, position: [0, 1, 0], range: NO_RANGE_WINDOW };
        const spot = { ...point, type: 'spot', direction: [0, -1, 0], innerConeDeg: 20, outerConeDeg: 30 };
        expect(errorsFor((m) => (m.lights = [point, spot]))).toEqual([]);
        const { range: _missing, ...noRange } = point;
        const errors = errorsFor((m) => (m.lights = [noRange, { ...spot, range: Number.NaN }]));
        expect(errors).toHaveLength(2);
        expect(errors[0]).toContain('lights[0].range: must be a distance in metres, or 0 for no range window');
        expect(errors[1]).toContain('lights[1].range');
    });

    it('rejects a probe spacing larger than the volume', () => {
        expect(errorsFor((m) => (m.volume = { center: [0, 0, 0], size: [2, 2, 2], probeSpacing: 3 }))).toEqual([
            expect.stringContaining('volume.probeSpacing'),
        ]);
    });

    it('needs a credit for every model asset and an MIT credit for procedural parts', () => {
        expect(errorsFor((m) => (m.credits = [{ asset: 'someone', license: 'CC0-1.0', attribution: 'x' }]))).toEqual([
            expect.stringContaining('procedural part needs a credit'),
        ]);
    });

    it('checks light-set parts and the settings they read (T2.1)', () => {
        const candles = { kind: 'light-set', asset: 'sponza-intel-candles', path: 'candles/sponza-candles.json', countSetting: 'candles' };
        const setting = { label: 'Candles', description: 'How many candles.', default: 1024, min: 1, max: 10000 };
        const withSet = (change: (manifest: Record<string, unknown>) => void) =>
            errorsFor((m) => {
                (m.parts as unknown[]).push({ ...candles });
                (m.credits as unknown[]).push({ asset: 'sponza-intel-candles', license: 'CC-BY-4.0+intel-preface', attribution: 'Intel' });
                m.settings = { candles: setting };
                change(m);
            });
        expect(withSet(() => {})).toEqual([]);
        expect(withSet((m) => delete m.settings)).toEqual([expect.stringContaining('"candles" is not one of the scene settings')]);
        expect(withSet((m) => (m.settings = { candles: { ...setting, default: 20000 } }))).toEqual([expect.stringContaining('min <= default <= max')]);
        expect(withSet((m) => (m.settings = { candles: { ...setting, max: 1.5 } }))).toEqual([expect.stringContaining('whole numbers')]);
        expect(withSet((m) => ((m.parts as Record<string, unknown>[])[1].path = '../x.json'))).toEqual([expect.stringContaining('relative path')]);
        expect(withSet((m) => (m.credits as unknown[]).pop())).toEqual([expect.stringContaining('no credit for asset "sponza-intel-candles" used by a light-set part')]);
    });

    it('reads settings from a link, clamped to their range, the default otherwise', () => {
        const manifest = { settings: { candles: { label: 'Candles', description: 'x', default: 1024, min: 1, max: 10000 } } };
        expect(sceneSettingValues(manifest)).toEqual({ candles: 1024 });
        expect(sceneSettingValues(manifest, { candles: '4096' })).toEqual({ candles: 4096 });
        expect(sceneSettingValues(manifest, { candles: 99999 })).toEqual({ candles: 10000 });
        expect(sceneSettingValues(manifest, { candles: 'many', other: 3 })).toEqual({ candles: 1024 });
        expect(sceneSettingValues({})).toEqual({});
    });

    it('parseManifest reports bad JSON and lists every problem', () => {
        expect(() => parseManifest('{', 'broken.json')).toThrow('broken.json: not valid JSON');
        expect(() => parseManifest('{"id": 1}', 'bad.json')).toThrow(/not a valid scene manifest[\s\S]*id:[\s\S]*title:/);
    });
});

describe('checkManifestAgainstSources', () => {
    const modelManifest: SceneManifest = {
        ...validManifest(),
        parts: [{ kind: 'model', asset: 'sponza-khronos', path: 'main.glb', transform: identityTransform }],
        credits: [{ asset: 'sponza-khronos', license: 'CC-BY-4.0', attribution: 'x' }],
    };

    it('passes when the asset and output exist', () => {
        expect(checkManifestAgainstSources(modelManifest, [{ id: 'sponza-khronos', outputs: [{ path: 'main.glb' }] }])).toEqual([]);
    });

    it('flags a missing asset and a missing output', () => {
        expect(checkManifestAgainstSources(modelManifest, [])[0]).toContain('not in tools/assets/sources.json');
        const wrongOutput = [{ id: 'sponza-khronos', outputs: [{ path: 'other.glb' }] }];
        expect(checkManifestAgainstSources(modelManifest, wrongOutput)[0]).toContain('not an output');
    });

    it('skips scenes that are still waiting for their asset', () => {
        expect(checkManifestAgainstSources({ ...modelManifest, status: 'awaiting-asset' }, [])).toEqual([]);
    });
});

describe('manifestLightToSceneLight', () => {
    it('converts spot cone angles from degrees to radians', () => {
        const light = manifestLightToSceneLight({
            type: 'spot',
            color: [1, 1, 1],
            intensity: 50,
            position: [0, 2, 0],
            direction: [0, -1, 0],
            range: 8,
            innerConeDeg: 20,
            outerConeDeg: 45,
        });
        if (light.type !== 'spot') throw new Error('expected a spot light');
        expect(light.innerConeAngle).toBeCloseTo(Math.PI / 9);
        expect(light.outerConeAngle).toBeCloseTo(Math.PI / 4);
    });

    it('passes a range of 0 (no range window) through unchanged', () => {
        const light = manifestLightToSceneLight({ type: 'point', color: [1, 1, 1], intensity: 50, position: [0, 2, 0], range: NO_RANGE_WINDOW });
        expect(light.type === 'point' && light.range).toBe(0);
    });
});

describe('the manifests in scenes/manifests', async () => {
    // loadManifests validates every file, checks the file name matches the id and
    // throws a readable message on the first invalid manifest.
    const manifests = await loadManifests();

    it('includes the six procedural scenes and the downloaded-model scenes', () => {
        const expected = [
            'bistro-exterior',
            'bistro-interior',
            'cabinet-row',
            'closed-door',
            'cornell-box',
            'cornell-box-point',
            'courtyard-tree',
            'furnace-room',
            'furnished-room',
            'open-ground',
            'sponza',
            'sponza-candles',
            'sponza-full',
            'thin-wall',
            'two-room',
        ];
        expect(manifests.map((m) => m.id).sort()).toEqual(expected);
    });

    it('marks only sponza-full and sponza-candles heavy, so tools that go through every scene leave them out unless named', () => {
        expect(manifests.filter(isHeavyScene).map((manifest) => manifest.id)).toEqual(['sponza-candles', 'sponza-full']);
    });

    it('has no placeholder scenes left: every scene has its assets confirmed', () => {
        expect(manifests.filter((manifest) => manifest.status === 'awaiting-asset').map((manifest) => manifest.id)).toEqual([]);
    });

    it('names real assets and outputs from tools/assets/sources.json, with matching licenses', async () => {
        const sources = JSON.parse(await readFile(join(repoRoot, 'tools', 'assets', 'sources.json'), 'utf8')) as {
            assets: Array<AssetOutputInfo & { license: string }>;
        };
        for (const manifest of manifests) {
            expect(checkManifestAgainstSources(manifest, sources.assets), manifest.id).toEqual([]);
            if (manifest.status === 'awaiting-asset') continue;
            for (const credit of manifest.credits) {
                const asset = sources.assets.find((candidate) => candidate.id === credit.asset);
                if (asset) expect(credit.license, `${manifest.id} credit for ${asset.id}`).toBe(asset.license);
            }
        }
    });

    // Where the real files are, the suggested probe volume must cover all of the scene's
    // geometry (models and procedural parts together) and not be wasteful about it.
    const downloadsDirectory = process.env.DDGI_DOWNLOADS_DIR ?? join(repoRoot, 'assets', 'downloads');
    for (const manifest of manifests.filter((candidate) => !isProceduralOnly(candidate))) {
        const files = manifest.parts.flatMap((part) => (part.kind === 'model' ? [join(downloadsDirectory, part.asset, part.path)] : []));
        it.skipIf(files.some((file) => !existsSync(file)))(`${manifest.id}: the volume covers the geometry without wasting space`, async () => {
            const min = [Infinity, Infinity, Infinity];
            const max = [-Infinity, -Infinity, -Infinity];
            const include = (bounds: { min: readonly number[]; max: readonly number[] } | null) => {
                for (let axis = 0; axis < 3; axis++) {
                    min[axis] = Math.min(min[axis], bounds?.min[axis] ?? Infinity);
                    max[axis] = Math.max(max[axis], bounds?.max[axis] ?? -Infinity);
                }
            };
            for (const part of manifest.parts) {
                if (part.kind === 'procedural') include(sceneBounds(buildProcedural(part.builder, part.options)));
                else if (part.kind === 'model') include(gltfWorldBounds(await readModelJson(join(downloadsDirectory, part.asset, part.path)), modelTransformMatrix(part.transform)));
            }
            for (let axis = 0; axis < 3; axis++) {
                const volumeMin = manifest.volume.center[axis] - manifest.volume.size[axis] / 2;
                const volumeMax = manifest.volume.center[axis] + manifest.volume.size[axis] / 2;
                expect(volumeMin, `axis ${axis} min`).toBeLessThanOrEqual(min[axis] + 1e-6);
                expect(volumeMax, `axis ${axis} max`).toBeGreaterThanOrEqual(max[axis] - 1e-6);
                // A margin of up to 1 m or 15% beyond the geometry is fine; more is wasted probes.
                expect(manifest.volume.size[axis], `axis ${axis} size`).toBeLessThanOrEqual((max[axis] - min[axis]) * 1.15 + 1);
            }
        });
    }

    for (const manifest of manifests.filter(isProceduralOnly)) {
        describe(manifest.id, () => {
            const merged = mergeScenes(buildProceduralParts(manifest));

            it('has the same lights as its builders, in the same units', () => {
                expect(manifest.lights.map(manifestLightToSceneLight)).toEqual(merged.lights);
            });

            it('has a probe volume whose centre is inside the geometry bounds', () => {
                const bounds = sceneBounds(merged);
                manifest.volume.center.forEach((value, axis) => {
                    expect(value).toBeGreaterThan(bounds.min[axis]);
                    expect(value).toBeLessThan(bounds.max[axis]);
                });
            });

            it('looks at something inside the bounds from every bookmark', () => {
                const bounds = sceneBounds(merged);
                for (const camera of manifest.cameras) {
                    camera.target.forEach((value, axis) => {
                        expect(value, `${camera.name} target axis ${axis}`).toBeGreaterThanOrEqual(bounds.min[axis] - 1e-6);
                        expect(value, `${camera.name} target axis ${axis}`).toBeLessThanOrEqual(bounds.max[axis] + 1e-6);
                    });
                }
            });

            it('puts the reference cube inside the volume', () => {
                const cube = manifest.referenceCube?.position;
                expect(cube).toBeDefined();
                cube?.forEach((value, axis) => {
                    const half = manifest.volume.size[axis] / 2;
                    expect(Math.abs(value - manifest.volume.center[axis])).toBeLessThan(half);
                });
            });
        });
    }
});
