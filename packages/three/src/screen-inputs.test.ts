import { AdditiveBlending, HalfFloatType, NoBlending, NormalBlending, RedIntegerFormat, RGFormat, Texture, UnsignedByteType, UnsignedIntType } from 'three/webgpu';
import { describe, expect, it } from 'vitest';
import { colorBytesPerSample, drawsBlended } from './screen-inputs.ts';

function target(type: number, format?: number): Texture {
    const texture = new Texture();
    texture.type = type as Texture['type'];
    if (format !== undefined) texture.format = format as Texture['format'];
    return texture;
}

describe('the screen inputs', () => {
    it("count a pass's colour bytes per pixel as WebGPU does", () => {
        // The set Chrome refused at the default limit of 32 on the RTX 3060 (RC1.1's first run): it said 36.
        const refused = [target(HalfFloatType), target(HalfFloatType), target(HalfFloatType), target(UnsignedByteType), target(UnsignedIntType, RedIntegerFormat)];
        expect(colorBytesPerSample(refused)).toBe(36);
        // rg16float costs 4.
        expect(colorBytesPerSample([target(HalfFloatType, RGFormat)])).toBe(4);
        // The composite's pass (output, giAlbedo, normal) with every screen channel: past 32, within 64.
        const composite = [target(HalfFloatType), target(HalfFloatType), target(HalfFloatType)];
        const screen = [target(HalfFloatType), target(HalfFloatType), target(UnsignedByteType), target(UnsignedByteType)];
        expect(colorBytesPerSample([...composite, ...screen])).toBe(56);
    });

    it('leave the channels to blended materials exactly when three draws them blended', () => {
        expect(drawsBlended({ blending: NormalBlending, transparent: false })).toBe(false);
        expect(drawsBlended({ blending: NormalBlending, transparent: true })).toBe(true);
        expect(drawsBlended({ blending: AdditiveBlending, transparent: false })).toBe(true);
        expect(drawsBlended({ blending: NoBlending, transparent: true })).toBe(false);
    });
});
