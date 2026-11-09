import { HalfFloatType, RedIntegerFormat, RGFormat, Texture, UnsignedByteType, UnsignedIntType } from 'three/webgpu';
import { describe, expect, it } from 'vitest';
import { colorBytesPerSample } from './screen-inputs.ts';

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
    });
});
