import { describe, expect, it } from 'vitest';
import { hoistDirectives } from './kernel-builder.ts';

describe('hoistDirectives', () => {
    it('moves directives in a technique after the modules to the top, once each', () => {
        const code = ['// ---- module gi-light-falloff ----', 'fn giShadeDistanceAttenuation() {}', 'enable subgroups;', 'fn ddgiMark() {}', '  enable subgroups;  ', 'requires readonly_and_readwrite_storage_textures;'].join('\n');
        const hoisted = hoistDirectives(code);
        expect(hoisted.split('\n').slice(0, 2)).toEqual(['enable subgroups;', 'requires readonly_and_readwrite_storage_textures;']);
        expect(hoisted.match(/enable subgroups;/g)).toHaveLength(1);
        expect(hoisted).toContain('fn ddgiMark() {}');
    });

    it('leaves code without directives alone, and function attributes where they are', () => {
        const code = '@diagnostic(off, subgroup_uniformity)\nfn a() {}';
        expect(hoistDirectives(code)).toBe(code);
    });
});
