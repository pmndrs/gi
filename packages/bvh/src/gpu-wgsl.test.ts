import { describe, expect, it } from 'vitest';
import { lbvhWgsl } from './gpu-lbvh-wgsl.ts';
import { meshRefitTrianglesWgsl } from './gpu-mesh-refit.ts';
import { radixSortWgsl } from './gpu-radix-sort.ts';
import { scanWgsl } from './gpu-scan.ts';
import { wgslNameProblems } from './wgsl-names.ts';

describe('the B1 shaders', () => {
    const shaders = { lbvh: lbvhWgsl(), sort: radixSortWgsl(), scan: scanWgsl(), scanFallback: scanWgsl({ reduction: 'workgroup' }), meshRefit: meshRefitTrianglesWgsl() };

    it('use no reserved word as a name, and no function shares a name with a variable', () => {
        for (const [name, code] of Object.entries(shaders)) expect(wgslNameProblems(code), name).toEqual([]);
    });

    it('catch the names that broke them before', () => {
        expect(wgslNameProblems('fn f() { let active = 1u; }')).toHaveLength(1);
        expect(wgslNameProblems('var<storage> scanTileSums: array<u32>;\nfn scanTileSums() {}')).toHaveLength(1);
    });

    it('enable subgroups where they use them, and the plain ones do not', () => {
        for (const code of [shaders.lbvh, shaders.sort, shaders.scan]) expect(code.startsWith('enable subgroups;')).toBe(true);
        expect(shaders.meshRefit).not.toContain('subgroup');
        // The scan's declared fallback uses no subgroup operation or builtin (its comments may name them).
        expect(shaders.scanFallback).not.toMatch(/enable subgroups|subgroup[A-Z]\w*\(|subgroup_/);
    });
});
