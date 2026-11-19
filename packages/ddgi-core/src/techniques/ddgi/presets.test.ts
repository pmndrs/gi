import { describe, expect, it } from 'vitest';
import { STORE_LAYER_SIZE, storeLayout, TEXTURE_LAYER_SIZES } from '../../platform/textures/texture-store-wgsl.ts';
import { DDGI_PRESET_NAMES, ddgiPreset, isDDGIPresetName, recommendPreset } from './presets.ts';
import { DEFAULT_DDGI_SETTINGS, effectiveRaysPerProbe, MAX_RAYS_PER_PROBE, MIN_RAYS_PER_PROBE, withDefaults } from './settings.ts';

const VOLUME = { center: [0, 1, 0], size: [4, 2, 4], spacing: 0.5 } as const;
/** Both test laptops' GPUs offer 2 GiB buffers (NOTES). */
const DESKTOP_LIMITS = { maxBufferSize: 2 ** 31 };

describe('quality presets (m5-coverage.md §1)', () => {
    it('high is exactly the defaults M1 to M4 were checked with', () => {
        const high = ddgiPreset('high');
        // Every DDGI setting it names is the default's value...
        for (const [name, value] of Object.entries(high.ddgi)) expect(value).toBe(DEFAULT_DDGI_SETTINGS[name as keyof typeof DEFAULT_DDGI_SETTINGS]);
        // ...so the settings a technique gets with it are the defaults, field for field.
        expect(withDefaults({ ...high.ddgi, volume: VOLUME })).toEqual(withDefaults({ volume: VOLUME }));
        // The platform-side numbers: the adapters' 2 ms budget, the store's 256² layers, LightProbeGrid's one probe a metre.
        expect(high.budget).toEqual({ gpuMs: 2 });
        expect(high.textureLayerSize).toBe(STORE_LAYER_SIZE);
        expect(high.probeSpacing).toBe(1);
    });

    it('scales down from high to low, inside the settings’ own ranges', () => {
        const [low, medium, high] = DDGI_PRESET_NAMES.map((name) => ddgiPreset(name));
        for (const preset of [low, medium, high]) {
            expect(effectiveRaysPerProbe(preset.ddgi.raysPerProbe)).toBe(preset.ddgi.raysPerProbe);
            expect(preset.ddgi.raysPerProbe).toBeGreaterThanOrEqual(MIN_RAYS_PER_PROBE);
            expect(preset.ddgi.raysPerProbe).toBeLessThanOrEqual(MAX_RAYS_PER_PROBE);
            expect(TEXTURE_LAYER_SIZES).toContain(preset.textureLayerSize);
        }
        expect(low.ddgi.raysPerProbe).toBeLessThan(medium.ddgi.raysPerProbe);
        expect(medium.ddgi.raysPerProbe).toBeLessThan(high.ddgi.raysPerProbe);
        expect(low.ddgi.distanceTexels).toBeLessThan(medium.ddgi.distanceTexels);
        expect(medium.ddgi.distanceTexels).toBeLessThan(high.ddgi.distanceTexels);
        expect(low.budget.gpuMs).toBeLessThan(medium.budget.gpuMs);
        expect(medium.budget.gpuMs).toBeLessThan(high.budget.gpuMs);
        expect(low.probeSpacing).toBeGreaterThan(medium.probeSpacing);
        // The warm-up batch: smaller on low, so warm-up frames stay short on its devices (TU.1); medium keeps high's.
        expect(low.ddgi.maxProbesPerFrame).toBe(1024);
        expect(low.ddgi.maxProbesPerFrame).toBeLessThan(medium.ddgi.maxProbesPerFrame);
        expect(medium.ddgi.maxProbesPerFrame).toBe(high.ddgi.maxProbesPerFrame);
        expect(storeLayout(low.textureLayerSize).layerBytes).toBeLessThan(storeLayout(high.textureLayerSize).layerBytes);
    });

    it('is plain data: each call is a new object, so a host may change fields', () => {
        const first = ddgiPreset('low');
        first.ddgi.raysPerProbe = 512;
        first.budget.gpuMs = 8;
        expect(ddgiPreset('low').ddgi.raysPerProbe).toBe(128);
        expect(ddgiPreset('low').budget.gpuMs).toBe(1);
        expect(isDDGIPresetName('medium')).toBe(true);
        expect(isDDGIPresetName('ultra')).toBe(false);
        expect(() => ddgiPreset('ultra' as 'low')).toThrow(/no quality preset "ultra"/);
    });
});

describe('recommendPreset', () => {
    const desktop = { mobile: false };

    it('gives low for integrated and mobile GPUs', () => {
        expect(recommendPreset({ vendor: 'intel', architecture: 'gen-12lp' }, DESKTOP_LIMITS, desktop)).toBe('low');
        expect(recommendPreset({ vendor: 'intel', architecture: 'xe-lpg' }, DESKTOP_LIMITS, desktop)).toBe('low');
        expect(recommendPreset({ vendor: 'qualcomm', architecture: 'adreno-7xx' }, DESKTOP_LIMITS, desktop)).toBe('low');
        expect(recommendPreset({ vendor: 'arm', architecture: 'valhall' }, DESKTOP_LIMITS, desktop)).toBe('low');
        expect(recommendPreset({ vendor: 'amd', architecture: 'rdna-3', description: 'AMD Radeon(TM) Graphics' }, DESKTOP_LIMITS, desktop)).toBe('low');
        // An iPad's Apple GPU is the same family as a Mac's: being on a tablet decides.
        expect(recommendPreset({ vendor: 'apple', architecture: 'metal-3' }, DESKTOP_LIMITS, { mobile: true })).toBe('low');
        expect(recommendPreset({ vendor: 'apple', architecture: 'metal-3', description: 'Apple M2' }, DESKTOP_LIMITS, desktop)).toBe('low');
        expect(recommendPreset({ vendor: 'google', architecture: '', isFallbackAdapter: true }, DESKTOP_LIMITS, desktop)).toBe('low');
        expect(recommendPreset({ vendor: 'google', architecture: 'swiftshader' , description: 'SwiftShader Device' }, DESKTOP_LIMITS, desktop)).toBe('low');
        expect(recommendPreset({ vendor: 'nvidia', architecture: 'ampere' }, { maxBufferSize: 256 * 1024 * 1024 }, desktop)).toBe('low');
    });

    it('gives high otherwise, Apple’s Pro and Max chips included', () => {
        expect(recommendPreset({ vendor: 'nvidia', architecture: 'ampere' }, DESKTOP_LIMITS, desktop)).toBe('high');
        expect(recommendPreset({ vendor: 'amd', architecture: 'rdna-3', description: 'AMD Radeon RX 7800 XT' }, DESKTOP_LIMITS, desktop)).toBe('high');
        expect(recommendPreset({ vendor: 'intel', architecture: 'xe-hpg' }, DESKTOP_LIMITS, desktop)).toBe('high');
        // What Chrome reports on the M5 Pro: no chip name.
        expect(recommendPreset({ vendor: 'apple', architecture: 'metal-3', description: '' }, DESKTOP_LIMITS, desktop)).toBe('high');
        expect(recommendPreset({ vendor: 'apple', architecture: 'metal-3', description: 'Apple M5 Pro' }, DESKTOP_LIMITS, desktop)).toBe('high');
        expect(recommendPreset({ vendor: 'apple', architecture: 'metal-3', description: 'Apple M3 Max' }, DESKTOP_LIMITS, desktop)).toBe('high');
    });
});
