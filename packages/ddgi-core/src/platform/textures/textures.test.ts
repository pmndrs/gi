/**
 * The texture store's pure parts in Node: format facts, the build layout
 * the kernel and the copies share, how sources are told apart and staged,
 * and how the scene copy hands textures to the store and rewrites the
 * materials using them. The GPU pages `textures-sources`,
 * `textures-uv-board` and `ddgi-textured-furnace` run the kernel and the
 * sampling on real GPUs.
 */

import { describe, expect, it } from 'vitest';
import type { TextureHandle } from '../contract/handles.ts';
import type { CompressedPixelData, MaterialDescriptor } from '../contract/scene.ts';
import { ALPHA_MODE, MATERIAL_RECORD_VEC4S, MATERIAL_WORD, NONE, TEXTURE_CHANNEL, UV_PACK_SHIFT, UV_WRAP } from '../scene/layout.ts';
import { PlatformScene } from '../scene/platform-scene.ts';
import { createFakeGpu, type FakeBuffer } from '../scene/testing/fake-gpu.ts';
import { sceneModuleWgsl } from '../scene/wgsl.ts';
import { DEFAULT_CONE_SPREAD, shadeModuleWgsl } from '../shade/wgsl.ts';
import { chooseCompressedLevel, compressedBlock, compressionFeature, decodesSrgb, readableAsFloat } from './texture-formats.ts';
import { sourceKind, stageSource } from './texture-sources.ts';
import type { TextureLayerRequest, TextureLayerResults, TextureLayerStore } from './texture-store.ts';
import { LAYER_BYTES } from './texture-store.ts';
import {
    BUILD_LAYERS_PER_PASS,
    BUILD_STEP_STRIDE,
    buildStepOffset,
    buildStepWords,
    LAYER_BUILD_BYTES,
    levelByteOffset,
    levelRowBytes,
    levelSize,
    LINEAR_SCRATCH_TEXELS,
    linearLevelTexel,
    STORE_LAYER_SIZE,
    STORE_MIP_LEVELS,
    TEXTURE_STORE_WGSL,
} from './texture-store-wgsl.ts';

describe('texture formats', () => {
    it('knows each compressed family, its feature and its blocks', () => {
        expect(compressionFeature('bc7-rgba-unorm-srgb')).toBe('texture-compression-bc');
        expect(compressionFeature('etc2-rgba8unorm')).toBe('texture-compression-etc2');
        expect(compressionFeature('eac-rg11unorm')).toBe('texture-compression-etc2');
        expect(compressionFeature('astc-8x6-unorm')).toBe('texture-compression-astc');
        expect(compressionFeature('rgba8unorm')).toBeNull();
        expect(compressedBlock('bc7-rgba-unorm')).toEqual({ width: 4, height: 4, bytes: 16 });
        expect(compressedBlock('bc1-rgba-unorm-srgb')).toEqual({ width: 4, height: 4, bytes: 8 });
        expect(compressedBlock('bc4-r-unorm')).toEqual({ width: 4, height: 4, bytes: 8 });
        expect(compressedBlock('bc5-rg-unorm')).toEqual({ width: 4, height: 4, bytes: 16 });
        expect(compressedBlock('etc2-rgb8unorm')).toEqual({ width: 4, height: 4, bytes: 8 });
        expect(compressedBlock('etc2-rgba8unorm')).toEqual({ width: 4, height: 4, bytes: 16 });
        expect(compressedBlock('astc-10x5-unorm-srgb')).toEqual({ width: 10, height: 5, bytes: 16 });
        expect(compressedBlock('rgba8unorm')).toBeNull();
    });

    it('says which formats the GPU decodes sRGB for, and which the kernel can read', () => {
        expect(decodesSrgb('rgba8unorm-srgb')).toBe(true);
        expect(decodesSrgb('bc7-rgba-unorm-srgb')).toBe(true);
        expect(decodesSrgb('rgba8unorm')).toBe(false);
        expect(readableAsFloat('rgba32float')).toBe(true);
        expect(readableAsFloat('bc7-rgba-unorm')).toBe(true);
        expect(readableAsFloat('rgba8uint')).toBe(false);
        expect(readableAsFloat('r32sint')).toBe(false);
        expect(readableAsFloat('depth24plus')).toBe(false);
    });

    it('uploads the smallest compressed level that still covers a layer, else the largest', () => {
        const chain = (sizes: readonly number[]) => sizes.map((size) => ({ width: size, height: size }));
        expect(chooseCompressedLevel(chain([2048, 1024, 512, 256, 128, 64]), 256)).toBe(3);
        expect(chooseCompressedLevel(chain([512, 256, 128]), 256)).toBe(1);
        expect(chooseCompressedLevel(chain([300, 150]), 256)).toBe(0);
        // Smaller than a layer from the start: the largest level, stretched by the kernel.
        expect(chooseCompressedLevel(chain([128, 64]), 256)).toBe(0);
        // Non-square: both sides must cover the layer.
        expect(chooseCompressedLevel([{ width: 1024, height: 512 }, { width: 512, height: 256 }, { width: 256, height: 128 }], 256)).toBe(1);
    });
});

describe('the build layout gi/texture-store and its copies share', () => {
    it('lays out 9 levels of 256² down to 1², rows padded to 256 bytes', () => {
        expect(STORE_MIP_LEVELS).toBe(9);
        expect(levelSize(0)).toBe(256);
        expect(levelSize(8)).toBe(1);
        for (let level = 0; level < STORE_MIP_LEVELS; level++) {
            expect(levelRowBytes(level) % 256).toBe(0);
            expect(levelRowBytes(level)).toBeGreaterThanOrEqual(levelSize(level) * 4);
            expect(levelByteOffset(level) % 4).toBe(0);
        }
        expect(levelRowBytes(0)).toBe(1024);
        expect(levelRowBytes(6)).toBe(256);
        expect(linearLevelTexel(1)).toBe(256 * 256);
        expect(LINEAR_SCRATCH_TEXELS).toBe(87381);
        // 0.33 MB a layer, as the design says.
        expect(LAYER_BYTES).toBe(87381 * 4);
        expect(LAYER_BUILD_BYTES).toBe(levelByteOffset(8) + 256);
    });

    it('writes one step per place and level, at offsets a dynamic uniform binding can use', () => {
        const words = buildStepWords();
        const stride = BUILD_STEP_STRIDE / 4;
        for (let place = 0; place < BUILD_LAYERS_PER_PASS; place++) {
            for (let level = 0; level < STORE_MIP_LEVELS; level++) {
                const base = buildStepOffset(place, level) / 4;
                expect(buildStepOffset(place, level) % 256).toBe(0);
                expect(words[base]).toBe(levelSize(level));
                expect(words[base + 2]).toBe(linearLevelTexel(level));
                if (level > 0) expect(words[base + 1]).toBe(linearLevelTexel(level - 1));
                expect(words[base + 3] * 4).toBe(place * LAYER_BUILD_BYTES + levelByteOffset(level));
                expect(words[base + 4] * 4).toBe(levelRowBytes(level));
            }
        }
        expect(words.length).toBe(BUILD_LAYERS_PER_PASS * STORE_MIP_LEVELS * stride);
        expect(TEXTURE_STORE_WGSL).toContain('fn giTextureStoreFillEncoded(');
        expect(TEXTURE_STORE_WGSL).toContain('fn giTextureStoreFillLinear(');
        expect(TEXTURE_STORE_WGSL).toContain('fn giTextureStoreMip(');
    });
});

/** A stand-in device for staging: textures and uploads are recorded, not made. */
function stagingDevice(features: string[] = []) {
    const textures: GPUTextureDescriptor[] = [];
    const writes: { size: GPUExtent3DDict; layout: GPUTexelCopyBufferLayout; bytes: number }[] = [];
    const device = {
        features: new Set(features),
        limits: { maxTextureDimension2D: 8192 },
        createTexture: (descriptor: GPUTextureDescriptor) => {
            textures.push(descriptor);
            return { ...descriptor, createView: () => ({}), destroy() {} };
        },
        queue: {
            writeTexture: (_destination: unknown, data: ArrayBufferView, layout: GPUTexelCopyBufferLayout, size: GPUExtent3DDict) => writes.push({ size, layout, bytes: data.byteLength }),
            copyExternalImageToTexture: () => {},
        },
    } as unknown as GPUDevice;
    return { device, textures, writes };
}

describe('texture sources', () => {
    const pixels = { data: new Uint8Array(4 * 4 * 4), width: 4, height: 4 };
    const bc7: CompressedPixelData = {
        format: 'bc7-rgba-unorm',
        levels: [
            { data: new Uint8Array((512 / 4) * (512 / 4) * 16), width: 512, height: 512 },
            { data: new Uint8Array((256 / 4) * (256 / 4) * 16), width: 256, height: 256 },
            { data: new Uint8Array((128 / 4) * (128 / 4) * 16), width: 128, height: 128 },
        ],
    };

    it('tells the four kinds apart by shape', () => {
        expect(sourceKind(pixels)).toBe('pixels');
        expect(sourceKind(bc7)).toBe('compressed');
        expect(sourceKind({ format: 'rgba8unorm', createView: () => ({}) } as unknown as GPUTexture)).toBe('gpu-texture');
        expect(sourceKind({ width: 8, height: 8, close() {} } as unknown as ImageBitmap)).toBe('image');
    });

    it('writes PixelData into an rgba8unorm staging texture, to be decoded when the descriptor says sRGB', () => {
        const { device, textures, writes } = stagingDevice();
        const staged = stageSource(device, pixels, true, STORE_LAYER_SIZE, 'test');
        expect('reason' in staged).toBe(false);
        expect(textures[0].format).toBe('rgba8unorm');
        expect(writes[0].layout.bytesPerRow).toBe(16);
        expect(!('reason' in staged) && staged.encoded).toBe(true);
        const linear = stageSource(device, pixels, false, STORE_LAYER_SIZE, 'test');
        expect(!('reason' in linear) && linear.encoded).toBe(false);
    });

    it('uploads one compressed level, the 256² one, in whole blocks', () => {
        const { device, textures, writes } = stagingDevice(['texture-compression-bc']);
        const staged = stageSource(device, bc7, true, STORE_LAYER_SIZE, 'test');
        expect('reason' in staged).toBe(false);
        expect(textures).toHaveLength(1);
        expect(textures[0].format).toBe('bc7-rgba-unorm');
        expect(textures[0].size).toEqual({ width: 256, height: 256 });
        // 64 blocks a row of 16 bytes: 64 KB in all.
        expect(writes[0].layout).toEqual({ bytesPerRow: 64 * 16, rowsPerImage: 64 });
        expect(writes[0].bytes).toBe(65536);
        // An -srgb format is decoded by the GPU, whatever the descriptor says.
        const srgbFormat = stageSource(device, { ...bc7, format: 'bc7-rgba-unorm-srgb' }, true, STORE_LAYER_SIZE, 'test');
        expect(!('reason' in srgbFormat) && srgbFormat.encoded).toBe(false);
    });

    it('refuses what it cannot read, saying why', () => {
        const { device } = stagingDevice();
        const noFeature = stageSource(device, bc7, true, STORE_LAYER_SIZE, 'test');
        expect('reason' in noFeature && noFeature.reason).toMatch(/texture-compression-bc/);
        const short = stageSource(device, { data: new Uint8Array(10), width: 4, height: 4 }, true, STORE_LAYER_SIZE, 'test');
        expect('reason' in short && short.reason).toMatch(/fewer than/);
        const integer = stageSource(device, { format: 'rgba8uint', usage: 0x4, dimension: '2d', createView: () => ({}) } as unknown as GPUTexture, false, STORE_LAYER_SIZE, 'test');
        expect('reason' in integer && integer.reason).toMatch(/rgba8uint/);
        const unbindable = stageSource(device, { format: 'rgba8unorm', usage: 0x2, dimension: '2d', createView: () => ({}) } as unknown as GPUTexture, false, STORE_LAYER_SIZE, 'test');
        expect('reason' in unbindable && unbindable.reason).toMatch(/TEXTURE_BINDING/);
    });
});

/** A store stand-in: ready when told, makes every layer it is asked for unless the layer is listed to fail. */
class StubStore implements TextureLayerStore {
    ready = true;
    readonly requests: TextureLayerRequest[][] = [];
    failLayers = new Set<number>();
    readonly memoryBytes = 0;

    encode(_encoder: GPUCommandEncoder, requests: readonly TextureLayerRequest[]): TextureLayerResults {
        this.requests.push([...requests]);
        return {
            built: requests.filter((request) => !this.failLayers.has(request.layer)).map((request) => request.layer),
            failed: requests.filter((request) => this.failLayers.has(request.layer)).map((request) => request.layer),
        };
    }

    dispose(): void {}
}

function sceneWithStore() {
    const gpu = createFakeGpu();
    const store = new StubStore();
    const scene = new PlatformScene(gpu.device, { textureStore: store });
    const encode = () => {
        const encoder = gpu.createEncoder();
        const upload = scene.encode(encoder);
        gpu.submit(encoder);
        return upload;
    };
    /** The material record's words as the GPU has them. */
    const materialWords = (material: Parameters<PlatformScene['materialSlot']>[0]): Uint32Array => {
        const tables = (scene.module.resources()[1] as GPUBufferBinding).buffer as unknown as FakeBuffer;
        const header = new Uint32Array(tables.bytes.buffer, 0, 32);
        const base = (header[6] + scene.materialSlot(material) * MATERIAL_RECORD_VEC4S) * 4;
        return new Uint32Array(tables.bytes.buffer, base * 4, MATERIAL_RECORD_VEC4S * 4);
    };
    return { gpu, store, scene, encode, materialWords };
}

const QUAD = {
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
};
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

describe('textures in the scene copy', () => {
    it('shades flat until the layer exists, then writes the layer and logs a material change where it is used', () => {
        const { store, scene, encode, materialWords } = sceneWithStore();
        const texture = scene.createTexture({ source: { data: new Uint8Array(16), width: 2, height: 2 }, srgb: true, label: 'checker' });
        const material = scene.createMaterial({ albedo: [0.5, 0.5, 0.5], albedoTexture: texture });
        const mesh = scene.createMesh(QUAD);
        const instance = scene.createInstance({ mesh, materials: [material], transform: IDENTITY });

        // Not ready (still compiling): nothing is handed over, the material shades flat.
        store.ready = false;
        encode();
        expect(store.requests).toHaveLength(0);
        expect(materialWords(material)[MATERIAL_WORD.albedoTexture]).toBe(NONE);
        expect(scene.textureCounts()).toEqual({ built: 0, pending: 1, failed: 0 });

        store.ready = true;
        const upload = encode();
        expect(store.requests[0]).toEqual([expect.objectContaining({ layer: 0, srgb: true, label: 'checker' })]);
        expect(materialWords(material)[MATERIAL_WORD.albedoTexture]).toBe(scene.textureLayer(texture));
        expect(upload.changes.regions).toEqual([expect.objectContaining({ reason: 'material', source: instance })]);
        expect(scene.textureCounts()).toEqual({ built: 1, pending: 0, failed: 0 });

        // Made once: the next frames hand over nothing.
        encode();
        expect(store.requests[1]).toEqual([]);
    });

    it('gives a texture the store refuses no layer, and its material keeps its flat colour', () => {
        const { store, scene, encode, materialWords } = sceneWithStore();
        const refused = scene.createTexture({ source: { data: new Uint8Array(16), width: 2, height: 2 }, srgb: true });
        store.failLayers.add(0);
        const material = scene.createMaterial({ albedoTexture: refused });
        encode();
        expect(scene.textureLayer(refused)).toBeNull();
        expect(materialWords(material)[MATERIAL_WORD.albedoTexture]).toBe(NONE);
        expect(scene.textureCounts().failed).toBe(1);
    });

    it('reuses a destroyed texture layer, and drops a texture destroyed before it was made', () => {
        const { store, scene, encode } = sceneWithStore();
        const first = scene.createTexture({ source: { data: new Uint8Array(16), width: 2, height: 2 }, srgb: true });
        scene.destroyTexture(first);
        const second = scene.createTexture({ source: { data: new Uint8Array(16), width: 2, height: 2 }, srgb: false });
        encode();
        expect(store.requests[0]).toHaveLength(1);
        expect(store.requests[0][0].srgb).toBe(false);
        expect(scene.textureLayer(second)).toBe(0);
    });

    it('stores metalness, its map and channel, opacity and the alpha mode in the record', () => {
        const { scene, encode, materialWords } = sceneWithStore();
        const metalMap = scene.createTexture({ source: { data: new Uint8Array(16), width: 2, height: 2 }, srgb: false });
        const material = scene.createMaterial({
            metalness: 0.75,
            metalnessTexture: { texture: metalMap, channel: 'g', uv: { channel: 1, wrapU: 'mirror', matrix: [2, 0, 0, 2, 0.5, 0.25] } },
            opacity: 0.4,
            alphaMode: 'stochastic',
        });
        encode();
        const words = materialWords(material);
        const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
        expect(floats[MATERIAL_WORD.metalness]).toBeCloseTo(0.75, 6);
        expect(floats[MATERIAL_WORD.opacity]).toBeCloseTo(0.4, 6);
        expect(words[MATERIAL_WORD.alphaMode]).toBe(ALPHA_MODE.stochastic);
        expect(words[MATERIAL_WORD.metalnessTexture]).toBe(scene.textureLayer(metalMap));
        const packed = words[MATERIAL_WORD.metalnessUvPacked];
        expect((packed >> UV_PACK_SHIFT.textureChannel) & 0xf).toBe(TEXTURE_CHANNEL.g);
        expect((packed >> UV_PACK_SHIFT.channel) & 0xf).toBe(1);
        expect((packed >> UV_PACK_SHIFT.wrapU) & 0xf).toBe(UV_WRAP.mirror);
        expect(Array.from(floats.subarray(MATERIAL_WORD.metalnessUvMatrix, MATERIAL_WORD.metalnessUvMatrix + 4))).toEqual([2, 0, 0, 2]);
        expect(Array.from(floats.subarray(MATERIAL_WORD.metalnessUvTranslation, MATERIAL_WORD.metalnessUvTranslation + 2))).toEqual([0.5, 0.25]);
        // A stochastic material has no cutoff in the record (only cutouts do).
        expect(floats[MATERIAL_WORD.alphaCutoff]).toBe(0);
    });

    it('resolves the alpha mode as the contract says', () => {
        const { scene, encode, materialWords } = sceneWithStore();
        const cases: [MaterialDescriptor, number, number][] = [
            [{}, ALPHA_MODE.opaque, 0],
            [{ alphaCutoff: 0.3 }, ALPHA_MODE.cutout, 0.3],
            [{ alphaMode: 'cutout' }, ALPHA_MODE.cutout, 0.5],
            [{ alphaMode: 'opaque', alphaCutoff: 0.3 }, ALPHA_MODE.opaque, 0],
        ];
        const materials = cases.map(([descriptor]) => scene.createMaterial(descriptor));
        encode();
        cases.forEach(([, mode, cutoff], index) => {
            const words = materialWords(materials[index]);
            expect(words[MATERIAL_WORD.alphaMode]).toBe(mode);
            expect(new Float32Array(words.buffer, words.byteOffset, words.length)[MATERIAL_WORD.alphaCutoff]).toBeCloseTo(cutoff, 6);
        });
        // Defaults: no metalness, fully opaque.
        const plain = new Float32Array(materialWords(materials[0]).slice().buffer);
        expect(plain[MATERIAL_WORD.metalness]).toBe(0);
        expect(plain[MATERIAL_WORD.opacity]).toBe(1);
    });

    it('checks the new material fields', () => {
        const { scene } = sceneWithStore();
        expect(() => scene.createMaterial({ metalness: 1.5 })).toThrow(/metalness/);
        expect(() => scene.createMaterial({ opacity: -0.1 })).toThrow(/opacity/);
        expect(() => scene.createMaterial({ alphaMode: 'blend' as never })).toThrow(/alphaMode/);
        expect(() => scene.createMaterial({ metalnessTexture: { texture: 999 as TextureHandle, channel: 'b' } })).toThrow(/texture/);
        const texture = scene.createTexture({ source: { data: new Uint8Array(4), width: 1, height: 1 }, srgb: false });
        expect(() => scene.createMaterial({ metalnessTexture: { texture, channel: 'x' as never } })).toThrow(/channel/);
        // A texture a material's metalness map uses can't be destroyed under it.
        scene.createMaterial({ metalnessTexture: { texture, channel: 'b' } });
        expect(() => scene.destroyTexture(texture)).toThrow(/still used/);
    });
});

describe('textures at hits (WGSL)', () => {
    it('defines giSurfaceAtCone, and giSurfaceAt as it at the default spread', () => {
        const code = shadeModuleWgsl({ group: 0, firstBinding: 0 });
        expect(code).toContain('fn giSurfaceAtCone(ray: GiRay, hit: GiHit, spread: f32) -> GiSurface');
        expect(code).toContain('return giSurfaceAtCone(ray, hit, GI_SHADE_DEFAULT_CONE_SPREAD);');
        expect(DEFAULT_CONE_SPREAD).toBeCloseTo(0.2216, 4);
        expect(sceneModuleWgsl({ group: 0, firstBinding: 0 })).toContain('fn giInstanceTriangleUvs(instance: u32, triangle: u32, channel: u32) -> GiTriangleUvs');
    });
});
