import { describe, expect, it } from 'vitest';
import { rayScratchTraceWgsl, RAY_RESULT_BYTES, WORK_ITEM_BYTES } from '../../platform/shade/direct.ts';
import { RAY_RECORD_BYTES } from './layout.ts';
import { DEFAULT_DDGI_SETTINGS, FIXED_RAYS } from './settings.ts';
import { DDGI_LIGHT_GROUP_MASK, DDGI_TRACE_WGSL, deferredTraceKernelWgsl, DIRECT_ROUND_PROBES } from './wgsl/trace.ts';

describe('ddgi/trace with direct light deferred', () => {
    const deferred = deferredTraceKernelWgsl(rayScratchTraceWgsl('ddgiRayScratch', 'subgroups'));

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
        const body = deferred.slice(deferred.indexOf('fn ddgiTrace('));
        const afterRoundCheck = body.slice(body.indexOf('slot >= ddgi.updateCount'));
        expect(afterRoundCheck.match(/\breturn\b/g)).toHaveLength(1);
        expect(afterRoundCheck.trimEnd().endsWith('giDeferDirect(deferred, record, rayRecord);\n}')).toBe(true);
    });

    it("sizes one round's work list at about 7.3 MB with the default rays per probe", () => {
        const bytes = DIRECT_ROUND_PROBES * (DEFAULT_DDGI_SETTINGS.raysPerProbe - FIXED_RAYS) * WORK_ITEM_BYTES;
        expect(bytes / 1e6).toBeCloseTo(7.34, 2);
    });
});
