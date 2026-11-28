/**
 * M4, T4.1 done-whens 3 and 4: the texture store's sources and growth.
 *
 * 1. **Every source kind makes the same layer.** One 256² test image
 *    (4×4 blocks of one colour each, alpha 128-254) is given to the platform
 *    as `PixelData`, `ImageData`, an `ImageBitmap` (copyExternalImageToTexture),
 *    an `rgba8unorm` `GPUTexture` (the kernel decodes sRGB), an
 *    `rgba8unorm-srgb` `GPUTexture` (the GPU decodes it), and BC7 mips made
 *    here (mode 6 blocks of one colour, which decode exactly). The BC7 mips
 *    are 512², 256² and 128², and the 512² level is a different colour, so
 *    the store must upload the 256² level to pass. Every level of every
 *    layer must be within 2/255 of the `PixelData` layer, and level 0 of it
 *    within 1/255 of the image itself (an sRGB round trip through linear
 *    light), level 1 within 1/255 of a CPU 2×2 average in linear light.
 * 2. **500 more textures compile nothing.** After that, 500 textures of
 *    various sizes (32² to 512², non-square too) are added and built in one
 *    frame; the array grows from its first layers past 506, `stats.recompiles`
 *    stays where it was (0), every texture has its layer, and the first
 *    layer still reads the same after the growth copies.
 *
 * Added with T4.2's second half (T4.1's review notes):
 *
 * 3. **Values, not only sameness.** The image as `PixelData` with
 *    `srgb: false` (its values are linear light) and as `srgb: true`, and as
 *    `bc7-rgba-unorm-srgb` (the GPU decodes it), each against the CPU twin
 *    of the build (textures/texture-store-cpu.ts) from the same pixels:
 *    every level within 1/255 (BC7 sRGB within 2/255 of the sRGB layer).
 * 4. **Alpha-weighted mips.** A checkerboard of red opaque and blue clear
 *    8 × 8 blocks: from mip 4 on (each texel over 2 × 2 blocks) the colour
 *    stays red (sRGB red ≥ 254, blue ≤ 1), alpha about half; and the layer
 *    matches the CPU twin within 1/255.
 * 5. **A bad source fails alone.** In one frame with a good texture: a
 *    closed `ImageBitmap`, `PixelData` whose buffer was transferred away, a
 *    source that throws when read, and another device's `GPUTexture`. Every
 *    bad one ends `failed`, the good one is built, nothing stays pending,
 *    and no GPU error escapes (the other device's texture is checked inside
 *    an error scope before it is used).
 */

import { assert, runGpuTest } from '../../client.ts';
import type { TextureHandle } from '../../../../packages/core/src/platform/contract/handles.ts';
import type { TextureSource } from '../../../../packages/core/src/platform/contract/scene.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { frame } from '../hit-shading-shared/harness.ts';
import { buildStoreLayerOnCpu } from '../../../../packages/core/src/platform/textures/texture-store-cpu.ts';
import { alphaPatternImage } from '../../../../packages/core/src/platform/trace/oracle-fixtures.ts';
import { bc7Level, blockColour, blockImage, largestByteDifference, linearToSrgb, readLayer, srgbToLinear, type LayerLevels } from '../textures-shared/textures.ts';

const SIZE = 256;
/** Done-when 3: every source within 2/255 of the PixelData layer. */
const SAME_LAYER_TOLERANCE = 2;
/** Level 0 against the image, and level 1 against a CPU average: one step of rounding. */
const REFERENCE_TOLERANCE = 1;
const MANY_TEXTURES = 500;

/** Frames until no texture is pending (a host's GPUTexture is checked for a frame or so first). */
async function framesUntilBuilt(device: GPUDevice, platform: GIPlatformImpl): Promise<number> {
    for (let frames = 1; frames <= 20; frames++) {
        frame(device, platform);
        await device.queue.onSubmittedWorkDone();
        if (platform.sceneCopy.textureCounts().pending === 0) return frames;
    }
    throw new Error(`textures still pending after 20 frames: ${JSON.stringify(platform.sceneCopy.textureCounts())}`);
}

/** The largest difference, any level, between a layer read back and one the CPU made. */
function largestLayerDifference(gpu: LayerLevels, cpu: readonly Uint8Array[]): { difference: number; level: number; index: number } {
    let worst = { difference: 0, level: 0, index: -1 };
    gpu.forEach((bytes, level) => {
        const { difference, index } = largestByteDifference(bytes, cpu[level]);
        if (difference > worst.difference) worst = { difference, level, index };
    });
    return worst;
}

function pixelsTexture(device: GPUDevice, pixels: Uint8Array<ArrayBuffer>, format: GPUTextureFormat): GPUTexture {
    const texture = device.createTexture({ label: `textures-sources ${format}`, size: { width: SIZE, height: SIZE }, format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture }, pixels, { bytesPerRow: SIZE * 4, rowsPerImage: SIZE }, { width: SIZE, height: SIZE });
    return texture;
}

/** Level 1 of the image on the CPU: each 2×2 block averaged in linear light (alpha linear), encoded again. */
function cpuLevelOne(pixels: Uint8Array): Uint8Array {
    const half = SIZE / 2;
    const level = new Uint8Array(half * half * 4);
    for (let y = 0; y < half; y++) {
        for (let x = 0; x < half; x++) {
            for (let channel = 0; channel < 4; channel++) {
                let sum = 0;
                for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
                    const value = pixels[((2 * y + dy) * SIZE + 2 * x + dx) * 4 + channel] / 255;
                    sum += channel === 3 ? value : srgbToLinear(value);
                }
                const average = sum / 4;
                level[(y * half + x) * 4 + channel] = Math.round((channel === 3 ? average : linearToSrgb(average)) * 255);
            }
        }
    }
    return level;
}

await runGpuTest(
    async ({ device, log }) => {
        assert(device.features.has('texture-compression-bc'), 'this page needs texture-compression-bc (both test GPUs have it)');
        const platform = new GIPlatformImpl(device, { label: 'textures-sources' });
        await platform.prepare();
        const scene = platform.scene;
        const store = platform.textures;

        const image = blockImage(SIZE, blockColour);
        const imageData = new ImageData(new Uint8ClampedArray(image.pixels), SIZE, SIZE);
        const bitmap = await createImageBitmap(imageData, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
        const wrongColour = blockImage(512, () => [200, 0, 200, 254]);
        const sources: Record<string, TextureSource> = {
            pixelData: { data: image.pixels, width: SIZE, height: SIZE },
            imageData,
            imageBitmap: bitmap,
            gpuTextureUnorm: pixelsTexture(device, image.pixels, 'rgba8unorm'),
            gpuTextureSrgb: pixelsTexture(device, image.pixels, 'rgba8unorm-srgb'),
            bc7Mips: {
                format: 'bc7-rgba-unorm',
                levels: [
                    { data: bc7Level(wrongColour), width: 512, height: 512 },
                    { data: bc7Level(image), width: 256, height: 256 },
                    { data: bc7Level(blockImage(128, () => [0, 200, 0, 254])), width: 128, height: 128 },
                ],
            },
        };
        const handles = new Map<string, TextureHandle>();
        for (const [name, source] of Object.entries(sources)) handles.set(name, scene.createTexture({ source, srgb: true, label: name }));
        // The GPUTexture sources are checked for this device first, which takes a frame or two.
        const framesToBuild = await framesUntilBuilt(device, platform);
        log(`every source built after ${framesToBuild} frame(s)`);
        assert(store.texture, 'the store has no array after building');

        const layers = new Map<string, LayerLevels>();
        for (const [name, handle] of handles) {
            const layer = platform.sceneCopy.textureLayer(handle);
            assert(layer !== null, `${name} has no layer after a frame`);
            layers.set(name, await readLayer(device, store.texture, layer));
        }
        const reference = layers.get('pixelData')!;
        const levelZero = largestByteDifference(reference[0], image.pixels);
        const levelOne = largestByteDifference(reference[1], cpuLevelOne(image.pixels));
        log(`PixelData layer: level 0 within ${levelZero.difference}/255 of the image, level 1 within ${levelOne.difference}/255 of the CPU's linear-light average`);
        assert(levelZero.difference <= REFERENCE_TOLERANCE, `level 0 is ${levelZero.difference}/255 from the image (byte ${levelZero.index})`);
        assert(levelOne.difference <= REFERENCE_TOLERANCE, `level 1 is ${levelOne.difference}/255 from the CPU's linear-light average (byte ${levelOne.index})`);

        const sourceDifferences: Record<string, number> = {};
        for (const [name, levels] of layers) {
            let worst = 0;
            levels.forEach((bytes, level) => {
                const { difference, index } = largestByteDifference(bytes, reference[level]);
                if (difference > SAME_LAYER_TOLERANCE) throw new Error(`${name} level ${level} is ${difference}/255 from the PixelData layer (byte ${index})`);
                worst = Math.max(worst, difference);
            });
            sourceDifferences[name] = worst;
        }
        log(`largest difference to the PixelData layer, any level, in 1/255: ${JSON.stringify(sourceDifferences)}`);

        // 3. Values against the CPU twin of the build, from the same pixels.
        const linearHandle = scene.createTexture({ source: { data: image.pixels, width: SIZE, height: SIZE }, srgb: false, label: 'linear PixelData' });
        const bc7SrgbHandle = scene.createTexture({
            source: { format: 'bc7-rgba-unorm-srgb', levels: [{ data: bc7Level(image), width: SIZE, height: SIZE }] },
            srgb: true,
            label: 'bc7 sRGB',
        });
        const alphaImage = alphaPatternImage(SIZE, (x, y) => ((x >> 3) + (y >> 3)) % 2 === 0);
        const alphaHandle = scene.createTexture({ source: alphaImage, srgb: true, label: 'red opaque, blue clear' });
        await framesUntilBuilt(device, platform);
        const readBack = async (handle: TextureHandle) => readLayer(device, store.texture!, platform.sceneCopy.textureLayer(handle)!);
        const imagePixels = { width: SIZE, height: SIZE, data: image.pixels };
        const values = {
            srgbAgainstCpu: largestLayerDifference(reference, buildStoreLayerOnCpu(imagePixels, true).levels),
            linearAgainstCpu: largestLayerDifference(await readBack(linearHandle), buildStoreLayerOnCpu(imagePixels, false).levels),
            bc7SrgbAgainstSrgbLayer: largestLayerDifference(await readBack(bc7SrgbHandle), reference),
        };
        log(`values, largest difference in 1/255 (level, byte): ${JSON.stringify(values)}`);
        assert(values.srgbAgainstCpu.difference <= REFERENCE_TOLERANCE, `the sRGB PixelData layer is ${values.srgbAgainstCpu.difference}/255 from the CPU's build`);
        assert(values.linearAgainstCpu.difference <= REFERENCE_TOLERANCE, `the linear PixelData layer is ${values.linearAgainstCpu.difference}/255 from the CPU's build`);
        assert(values.bc7SrgbAgainstSrgbLayer.difference <= SAME_LAYER_TOLERANCE, `the bc7-rgba-unorm-srgb layer is ${values.bc7SrgbAgainstSrgbLayer.difference}/255 from the sRGB PixelData layer`);
        // Level 0 of the linear layer holds the image's values sRGB-encoded (the store is an sRGB texture).
        const linearLevelZero = (await readBack(linearHandle))[0];
        const encodedFirst = Math.round(linearToSrgb(image.pixels[4 * 5 * SIZE + 4 * 5] / 255) * 255);
        assert(Math.abs(linearLevelZero[4 * 5 * SIZE + 4 * 5] - encodedFirst) <= REFERENCE_TOLERANCE, `the linear layer's texel (5, 5) reads ${linearLevelZero[4 * 5 * SIZE + 4 * 5]}, expected ${encodedFirst}`);

        // 4. Alpha-weighted mips: red opaque blocks, blue clear ones.
        const alphaLayer = await readBack(alphaHandle);
        let reddest = 255;
        let bluest = 0;
        let alphaOff = 0;
        for (let level = 4; level < alphaLayer.length; level++) {
            const bytes = alphaLayer[level];
            for (let texel = 0; texel < bytes.length / 4; texel++) {
                reddest = Math.min(reddest, bytes[texel * 4]);
                bluest = Math.max(bluest, bytes[texel * 4 + 2]);
                alphaOff = Math.max(alphaOff, Math.abs(bytes[texel * 4 + 3] - 128));
            }
        }
        const alphaAgainstCpu = largestLayerDifference(alphaLayer, buildStoreLayerOnCpu(alphaImage, true).levels);
        const topTexel = Array.from(alphaLayer[alphaLayer.length - 1]);
        log(`alpha-weighted mips 4-8: red >= ${reddest}, blue <= ${bluest}, alpha within ${alphaOff} of 128; the 1 x 1 mip ${JSON.stringify(topTexel)}; against the CPU build ${JSON.stringify(alphaAgainstCpu)}`);
        assert(reddest >= 254 && bluest <= 1, `the coarse mips took in the clear texels' colour: red down to ${reddest}, blue up to ${bluest}`);
        assert(alphaOff <= 2, `the coarse mips' alpha is up to ${alphaOff}/255 from half`);
        assert(alphaAgainstCpu.difference <= REFERENCE_TOLERANCE, `the red/blue layer is ${alphaAgainstCpu.difference}/255 from the CPU's build`);

        // 5. Bad sources fail alone, in a frame with a good one.
        const closedBitmap = await createImageBitmap(imageData, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
        closedBitmap.close();
        const transferredPixels = new Uint8Array(image.pixels);
        structuredClone(transferredPixels.buffer, { transfer: [transferredPixels.buffer] });
        const throwing = {
            get width(): number {
                throw new Error('this source throws when read');
            },
            height: SIZE,
            data: image.pixels,
        };
        // A second adapter object: the first one is used up by the test's device.
        const otherAdapter = await navigator.gpu.requestAdapter();
        assert(otherAdapter, 'no second adapter for the other device');
        const otherDevice = await otherAdapter.requestDevice();
        const foreign = otherDevice.createTexture({ size: [SIZE, SIZE], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
        otherDevice.queue.writeTexture({ texture: foreign }, image.pixels, { bytesPerRow: SIZE * 4 }, [SIZE, SIZE]);
        const countsBefore = platform.sceneCopy.textureCounts();
        const good = scene.createTexture({ source: { data: image.pixels, width: SIZE, height: SIZE }, srgb: true, label: 'good, beside bad ones' });
        const bad: Record<string, TextureSource> = {
            closedImageBitmap: closedBitmap,
            transferredPixelData: { data: transferredPixels, width: SIZE, height: SIZE },
            throwingSource: throwing as unknown as TextureSource,
            otherDevicesGpuTexture: foreign,
        };
        for (const [name, source] of Object.entries(bad)) scene.createTexture({ source, srgb: true, label: name });
        device.pushErrorScope('validation');
        const framesForBad = await framesUntilBuilt(device, platform);
        const escaped = await device.popErrorScope();
        const countsAfter = platform.sceneCopy.textureCounts();
        const badOutcome = { built: countsAfter.built - countsBefore.built, failed: countsAfter.failed - countsBefore.failed, pending: countsAfter.pending, framesForBad, escapedError: escaped?.message ?? null };
        log(`one good and ${Object.keys(bad).length} bad sources: ${JSON.stringify(badOutcome)}`);
        assert(platform.sceneCopy.textureLayer(good) !== null, 'the good texture beside the bad ones was not built');
        assert(badOutcome.built === 1 && badOutcome.failed === Object.keys(bad).length && badOutcome.pending === 0, `bad sources: ${JSON.stringify(badOutcome)}`);
        assert(escaped === null, `a GPU error escaped while the bad sources were handled: ${escaped?.message}`);
        foreign.destroy();
        otherDevice.destroy();

        // 2. Five hundred more: the array grows, nothing compiles.
        const recompilesBefore = platform.stats.recompiles;
        const capacityBefore = store.layerCapacity;
        const many: TextureHandle[] = [];
        const sizes = [[32, 32], [64, 64], [128, 64], [256, 256], [300, 200], [512, 512]] as const;
        for (let index = 0; index < MANY_TEXTURES; index++) {
            const [width, height] = sizes[index % sizes.length];
            const data = new Uint8Array(width * height * 4).fill(index & 0xff);
            many.push(scene.createTexture({ source: { data, width, height }, srgb: index % 2 === 0, label: `many ${index}` }));
        }
        const start = performance.now();
        frame(device, platform);
        await device.queue.onSubmittedWorkDone();
        const buildMs = performance.now() - start;
        frame(device, platform);
        await device.queue.onSubmittedWorkDone();
        const recompiles = platform.stats.recompiles - recompilesBefore;
        const counts = platform.sceneCopy.textureCounts();
        log(`${MANY_TEXTURES} textures in one frame: ${buildMs.toFixed(0)} ms to submit and finish; layers ${capacityBefore} -> ${store.layerCapacity}; ${counts.built} built, ${counts.failed} refused; recompiles ${recompiles}; store ${(platform.stats.memory.textures / 1e6).toFixed(1)} MB`);
        assert(recompiles === 0, `adding ${MANY_TEXTURES} textures compiled ${recompiles} pipeline(s)`);
        assert(platform.stats.recompiles === 0, `stats.recompiles is ${platform.stats.recompiles}`);
        // Built so far: the sources, the three value textures, the good one of section 5 (its four bad ones failed).
        const builtBefore = handles.size + 4;
        assert(counts.built === builtBefore + MANY_TEXTURES && counts.failed === 4 && counts.pending === 0, `textures: ${JSON.stringify(counts)}`);
        assert(store.layerCapacity >= builtBefore + MANY_TEXTURES, `the array has ${store.layerCapacity} layers`);
        const afterGrowth = await readLayer(device, store.texture, platform.sceneCopy.textureLayer(handles.get('pixelData')!)!);
        const kept = Math.max(...afterGrowth.map((bytes, level) => largestByteDifference(bytes, reference[level]).difference));
        assert(kept === 0, `the first layer changed by ${kept}/255 when the array grew`);
        // A sample of the 500: each is one flat value, so every level holds it (sRGB or linear, encoded alike).
        const lastLayer = await readLayer(device, store.texture, platform.sceneCopy.textureLayer(many[MANY_TEXTURES - 1])!);
        const lastValue = (MANY_TEXTURES - 1) & 0xff;
        const expectedEncoded = (MANY_TEXTURES - 1) % 2 === 0 ? lastValue : Math.round(linearToSrgb(lastValue / 255) * 255);
        const lastDifference = Math.abs(lastLayer[0][0] - expectedEncoded);
        assert(lastDifference <= REFERENCE_TOLERANCE, `the last texture's layer reads ${lastLayer[0][0]}, expected ${expectedEncoded}`);

        const memory = platform.stats.memory;
        platform.dispose();
        bitmap.close();
        return {
            framesToBuild,
            values,
            alphaWeightedMips: { reddest, bluest, alphaOff, topTexel, againstCpu: alphaAgainstCpu },
            badSources: badOutcome,
            levelZeroAgainstImage: levelZero.difference,
            levelOneAgainstCpu: levelOne.difference,
            sourceDifferences,
            manyTextures: MANY_TEXTURES,
            buildMs: Math.round(buildMs),
            layersBefore: capacityBefore,
            layersAfter: store.layerCapacity,
            recompiles,
            textureBytes: memory.textures,
        };
    },
    { optionalFeatures: ['texture-compression-bc'], requiredLimits: { maxTextureArrayLayers: 2048 } },
);
