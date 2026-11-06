import { describe, expect, it } from 'vitest';
import { averageImages, checkParts, type PartMeta } from './combine.ts';

/** A part as render.py records it; `change` edits one copy. */
function part(seed: number, change: (meta: PartMetaDraft) => void = () => {}): PartMeta {
    const meta: PartMetaDraft = {
        seed,
        samplesPerPixel: 2048,
        maxBounces: 32,
        shot: {
            camera: { position: [1, 2, 3], target: [4, 5, 6], fovYDeg: 75, width: 1920, height: 1080 },
            scene: { manifest: 'sponza', manifestSha256: 'aaa', skySha256: 'bbb', giMaterialsSha256: 'ccc' },
            display: { exposureEv: 2, toneMapping: 'aces' },
            render: { pixelFilter: { type: 'BLACKMAN_HARRIS', widthPixels: 1.5 }, device: 'OPTIX: RTX', git: { commit: 'abc' } },
        },
    };
    change(meta);
    return meta as PartMeta;
}

interface PartMetaDraft {
    seed: number;
    samplesPerPixel: number;
    maxBounces: number;
    shot?: {
        camera: { position: number[]; target: number[]; fovYDeg: number; width: number; height: number };
        scene: { manifest: string; manifestSha256: string; skySha256: string; giMaterialsSha256: string | null };
        display: { exposureEv: number; toneMapping: string };
        render: { pixelFilter: { type: string; widthPixels: number }; device: string; git: { commit: string } | null };
    };
}

const names = ['-part0', '-part1'];

describe('combine.ts: which parts may be averaged', () => {
    it('accepts parts that differ only in their seed', () => {
        expect(() => checkParts(names, [part(0), part(1)])).not.toThrow();
    });

    const refusals: Array<[string, (meta: PartMetaDraft) => void, RegExp]> = [
        ['a different camera', (meta) => (meta.shot!.camera.fovYDeg = 70), /camera and size/],
        ['a different size', (meta) => (meta.shot!.camera.width = 640), /camera and size/],
        ['a different material table', (meta) => (meta.shot!.scene.giMaterialsSha256 = 'ddd'), /scene, sky or material table/],
        ['a different manifest', (meta) => (meta.shot!.scene.manifestSha256 = 'zzz'), /scene, sky or material table/],
        ['a different sample count', (meta) => (meta.samplesPerPixel = 1024), /sample count/],
        ['a different pixel filter', (meta) => (meta.shot!.render.pixelFilter = { type: 'BOX', widthPixels: 1 }), /pixel filter/],
        ['a different device', (meta) => (meta.shot!.render.device = 'CPU (16 logical cores)'), /device/],
        ['a different commit', (meta) => (meta.shot!.render.git = { commit: 'def' }), /commit/],
        ['a different bounce count', (meta) => (meta.maxBounces = 8), /bounce count/],
    ];
    for (const [what, change, message] of refusals) {
        it(`refuses ${what}`, () => {
            expect(() => checkParts(names, [part(0), part(1, change)])).toThrow(message);
        });
    }

    it('refuses parts that share a seed', () => {
        expect(() => checkParts(names, [part(3), part(3)])).toThrow(/different seeds/);
    });

    it('refuses a part rendered before shots were stored', () => {
        expect(() => checkParts(names, [part(0), part(1, (meta) => delete meta.shot)])).toThrow(/no stored shot/);
    });

    it('refuses a single part', () => {
        expect(() => checkParts(['-part0'], [part(0)])).toThrow(/at least two/);
    });
});

describe('combine.ts: averaging', () => {
    it('takes the pixel-by-pixel mean', () => {
        const a = { width: 1, height: 1, rgb: new Float32Array([1, 2, 3]) };
        const b = { width: 1, height: 1, rgb: new Float32Array([3, 4, 5]) };
        expect(Array.from(averageImages([a, b]).rgb)).toEqual([2, 3, 4]);
    });

    it('refuses images of different sizes', () => {
        const a = { width: 1, height: 1, rgb: new Float32Array(3) };
        const b = { width: 2, height: 1, rgb: new Float32Array(6) };
        expect(() => averageImages([a, b])).toThrow(/differ in size/);
    });
});
