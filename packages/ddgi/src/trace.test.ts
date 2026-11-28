import { describe, expect, it } from 'vitest';
import { rayScratchTraceWgsl, RAY_RESULT_BYTES, WORK_ITEM_BYTES } from '@pmndrs/gi-core/platform/shade/direct';
import { RAY_RECORD_BYTES } from './layout.ts';
import { DEFAULT_DDGI_SETTINGS, FIXED_RAYS } from './settings.ts';
import { defaultLightGroups, layerPlan } from './light-groups.ts';
import { DDGI_LIGHT_GROUP_MASK, DDGI_TRACE_WGSL, deferredTraceKernelWgsl, DIRECT_ROUND_PROBES } from './wgsl/trace.ts';

describe('ddgi/trace with direct light deferred', () => {
    const deferred = deferredTraceKernelWgsl(rayScratchTraceWgsl('ddgiRayScratch', 'subgroups'), layerPlan(null));
    const layered = deferredTraceKernelWgsl(rayScratchTraceWgsl('ddgiRayScratch', 'subgroups'), layerPlan(defaultLightGroups(2)));

    it("stores its rays in the platform's ray results, which the blend reads as its own records", () => {
        expect(RAY_RECORD_BYTES).toBe(RAY_RESULT_BYTES);
    });

    it('takes its light groups from the one constant, in both versions', () => {
        const constant = `const DDGI_ALL_LIGHT_GROUPS: u32 = ${DDGI_LIGHT_GROUP_MASK}u;`;
        expect(DDGI_TRACE_WGSL).toContain(constant);
        expect(deferred).toContain(constant);
        // The deferred kernel hands its groups to gi/direct and never names a mask of its own.
        expect(deferred).not.toMatch(/giDirectIrradiance\(/);
        expect(deferred.match(/giNeedsDirect\(surface\)/g)).toHaveLength(1);
    });

    it('lights a ray inline with the random numbers gi/direct would use for it', () => {
        expect(DDGI_TRACE_WGSL).toContain('ddgiShadeRay(ray, hit, giDirectRandom(rayRecord, ddgi.frame))');
    });

    it('reaches the append from every thread: no return after the round check', () => {
        for (const code of [deferred, layered]) {
            const body = code.slice(code.indexOf('fn ddgiTrace('));
            const afterRoundCheck = body.slice(body.indexOf('slot >= ddgi.updateCount'));
            expect(afterRoundCheck.match(/\breturn\b/g)).toHaveLength(1);
            expect(afterRoundCheck.trimEnd().endsWith('giDeferDirect(deferred, record, rayRecord);\n}')).toBe(true);
        }
    });

    it("with groups off, reads the bounce through renderers' one sampling function and stores one result per ray", () => {
        expect(deferred).toContain('ddgiSampleIrradiance(');
        expect(deferred).toContain('giStoreRayResult(rayRecord, radiance, signedDistance);');
        expect(deferred).not.toContain('giStoreRayLayerResult(rayRecord');
    });

    it('with 2 groups, stores 3 layers per ray from one set of probe weights, the sky and emission in the environment layer', () => {
        for (const layer of [0, 1, 2]) expect(layered).toContain(`giStoreRayLayerResult(rayRecord, ${layer}u, radiance.layer${layer}`);
        expect(layered).not.toContain('radiance.layer3');
        // The weights come from the corner-weight function renderers use: in their loop, and once in the layered lookup.
        expect(layered.match(/ddgiCornerWeight\(corner, volume, cage/g)).toHaveLength(2);
        expect(layered).toContain('radiance.layer2 = giSkyRadiance(direction);');
        expect(layered).toContain('result.layer2 += surface.emission;');
    });

    it('multiplies the bounce by the capped albedo and the direct light by the full one, in every version (DV-08)', () => {
        for (const code of [DDGI_TRACE_WGSL, deferred, layered]) {
            expect(code).toContain('giBounceAlbedo(surface) * (1.0 / DDGI_PI)');
            expect(code).not.toMatch(/surface\.albedo \* \(1\.0 \/ DDGI_PI\) \* \(?ddgi\.bounceEnergy/);
        }
        // The inline kernel adds the direct light itself, with the surface's own albedo.
        expect(DDGI_TRACE_WGSL).toContain('surface.albedo * (1.0 / DDGI_PI) * direct');
    });

    it("sizes one round's work list at about 7.3 MB with the default rays per probe", () => {
        const bytes = DIRECT_ROUND_PROBES * (DEFAULT_DDGI_SETTINGS.raysPerProbe - FIXED_RAYS) * WORK_ITEM_BYTES;
        expect(bytes / 1e6).toBeCloseTo(7.34, 2);
    });
});
