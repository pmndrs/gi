import { describe, expect, it } from 'vitest';
import { batchEntryWgsl, HIT_RECORD_BYTES, HIT_RECORD_WORD } from './wgsl.ts';

/** Size and alignment of the WGSL types a hit record uses (WGSL spec, "Alignment and size"). */
const TYPE_LAYOUT: Record<string, { size: number; align: number }> = {
    f32: { size: 4, align: 4 },
    u32: { size: 4, align: 4 },
    vec2f: { size: 8, align: 8 },
};

/** Byte offset of every member of a WGSL struct, and the struct's size, by the WGSL layout rules. */
function structLayout(wgsl: string, name: string): { offsets: Record<string, number>; size: number } {
    const start = wgsl.indexOf(`struct ${name} {`);
    const body = start < 0 ? undefined : wgsl.slice(wgsl.indexOf('{', start) + 1, wgsl.indexOf('}', start));
    if (!body) throw new Error(`struct ${name} not found`);
    const offsets: Record<string, number> = {};
    let offset = 0;
    let structAlign = 1;
    for (const match of body.matchAll(/(\w+)\s*:\s*(\w+)/g)) {
        const [, member, type] = match;
        const layout = TYPE_LAYOUT[type];
        if (!layout) throw new Error(`no layout for ${type}`);
        offset = Math.ceil(offset / layout.align) * layout.align;
        offsets[member] = offset;
        offset += layout.size;
        structAlign = Math.max(structAlign, layout.align);
    }
    return { offsets, size: Math.ceil(offset / structAlign) * structAlign };
}

describe('GiHitRecord', () => {
    it('has the layout the contract documents and the readers use', () => {
        const layout = structLayout(batchEntryWgsl('closest', 1), 'GiHitRecord');
        expect(layout.size).toBe(HIT_RECORD_BYTES);
        expect(layout.offsets).toEqual({
            t: HIT_RECORD_WORD.t * 4,
            instance: HIT_RECORD_WORD.instance * 4,
            triangle: HIT_RECORD_WORD.triangle * 4,
            frontFace: HIT_RECORD_WORD.frontFace * 4,
            barycentrics: HIT_RECORD_WORD.barycentrics * 4,
        });
    });
});
