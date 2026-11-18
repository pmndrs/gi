/**
 * M4, T4.3 done-when 3c: a three `CompressedTexture` as a material's map
 * gives the same GI albedo as the same image uncompressed.
 *
 * A 256² test image of 4×4 blocks, each one colour with even channels, so
 * BC7 mode 6 holds it exactly (textures-shared: `blockImage`, `bc7Level`).
 * Four materials on four meshes in one three scene, mirrored by the
 * adapter's SceneSync:
 *
 * - `CompressedTexture` (`RGBA_BPTC_Format`, BC7) with three mips, 512²,
 *   256² and 128², in sRGB (`SRGBColorSpace`) and in linear
 *   (`NoColorSpace`). The 512² mip is the image at twice the size, so the
 *   store's pick (the smallest mip covering 256²) has to be the 256² one;
 * - a `DataTexture` of the 256² image, sRGB and linear.
 *
 * After one frame (the store builds layers in the next encode), each pair's
 * store layers, every mip, are compared: the layer is what GI's albedo
 * reads at a hit (albedo × the layer). Passing: within 2/255 everywhere,
 * and the pair's materials sent with the same albedo, uv mapping and
 * format family (bc7-rgba-unorm-srgb, bc7-rgba-unorm).
 */

import { BoxGeometry, CompressedTexture, DataTexture, Mesh, MeshStandardMaterial, NoColorSpace, RGBA_BPTC_Format, SRGBColorSpace, Scene, type Texture } from 'three';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import type { CompressedPixelData } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import { SceneSync } from '../../../../packages/ddgi-three/src/scene-sync.ts';
import { textureDescriptor } from '../../../../packages/ddgi-three/src/convert.ts';
import { assert, runGpuTest } from '../../client.ts';
import { frameInput } from '../ddgi-shared/harness.ts';
import { bc7Level, blockColour, blockImage, largestByteDifference, readLayer, type LayerLevels } from '../textures-shared/textures.ts';

const SIZE = 256;
const TOLERANCE = 2;

/** The image's BC7 mips: twice the size, the image, and half the size (each from the same block colours). */
function bc7Mips(): { data: Uint8Array; width: number; height: number }[] {
    return [
        { data: bc7Level(blockImage(SIZE * 2, (x, y) => blockColour(x >> 1, y >> 1))), width: SIZE * 2, height: SIZE * 2 },
        { data: bc7Level(blockImage(SIZE, blockColour)), width: SIZE, height: SIZE },
        { data: bc7Level(blockImage(SIZE / 2, (x, y) => blockColour(x * 2, y * 2))), width: SIZE / 2, height: SIZE / 2 },
    ];
}

function compressedMap(colorSpace: string): CompressedTexture {
    const texture = new CompressedTexture(bc7Mips(), SIZE * 2, SIZE * 2, RGBA_BPTC_Format);
    texture.colorSpace = colorSpace;
    texture.name = `bc7 ${colorSpace || 'linear'}`;
    texture.needsUpdate = true;
    return texture;
}

function uncompressedMap(colorSpace: string): DataTexture {
    const texture = new DataTexture(blockImage(SIZE, blockColour).pixels, SIZE, SIZE);
    texture.colorSpace = colorSpace;
    texture.flipY = false;
    texture.name = `pixels ${colorSpace || 'linear'}`;
    texture.needsUpdate = true;
    return texture;
}

await runGpuTest(
    async ({ device, log }) => {
        assert(device.features.has('texture-compression-bc'), 'this page needs texture-compression-bc (both test GPUs have it)');
        const platform = new GIPlatformImpl(device, { label: 'three-m4-compressed' });
        await platform.prepare();
        const warnings: string[] = [];
        const sync = new SceneSync(platform.scene, { device, warn: (message) => warnings.push(message) });
        try {
            const pairs = [
                { name: 'sRGB', compressed: compressedMap(SRGBColorSpace), uncompressed: uncompressedMap(SRGBColorSpace), format: 'bc7-rgba-unorm-srgb' },
                { name: 'linear', compressed: compressedMap(NoColorSpace), uncompressed: uncompressedMap(NoColorSpace), format: 'bc7-rgba-unorm' },
            ];
            const scene = new Scene();
            const materialOf = new Map<Texture, MeshStandardMaterial>();
            for (const pair of pairs) {
                for (const map of [pair.compressed, pair.uncompressed]) {
                    const material = new MeshStandardMaterial({ map, name: map.name });
                    materialOf.set(map, material);
                    scene.add(new Mesh(new BoxGeometry(), material));
                }
            }
            scene.updateMatrixWorld(true);
            sync.update(scene);
            const encoder = device.createCommandEncoder({ label: 'three-m4-compressed frame' });
            platform.encode(encoder, frameInput(0));
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            const store = platform.textures.texture;
            assert(store, 'the store has no array after a frame');

            const results: Record<string, unknown> = {};
            for (const pair of pairs) {
                const sourceFormat = (textureDescriptor(pair.compressed)!.source as CompressedPixelData).format;
                assert(sourceFormat === pair.format, `the ${pair.name} CompressedTexture became ${sourceFormat}, not ${pair.format}`);
                const sent = [pair.compressed, pair.uncompressed].map((map) => sync.sentMaterial(materialOf.get(map)!)!);
                assert(JSON.stringify(sent[0].albedo) === JSON.stringify(sent[1].albedo), `${pair.name}: albedo sent differs`);
                assert(JSON.stringify(sent[0].albedoUv) === JSON.stringify(sent[1].albedoUv), `${pair.name}: uv mapping sent differs (${JSON.stringify(sent[0].albedoUv)} against ${JSON.stringify(sent[1].albedoUv)})`);
                const layers: LayerLevels[] = [];
                for (const map of [pair.compressed, pair.uncompressed]) {
                    const handle = sync.sentTexture(map);
                    assert(handle !== undefined, `${map.name} was not sent`);
                    const layer = platform.sceneCopy.textureLayer(handle);
                    assert(layer !== null, `${map.name} has no store layer after a frame`);
                    layers.push(await readLayer(device, store, layer));
                }
                const perLevel = layers[0].map((bytes, level) => largestByteDifference(bytes, layers[1][level]).difference);
                const worst = Math.max(...perLevel);
                log(`${pair.name}: ${pair.format} mips against the DataTexture, largest difference per store mip (1/255): ${perLevel.join(', ')}`);
                assert(worst <= TOLERANCE, `${pair.name}: the compressed map's layer is ${worst}/255 from the uncompressed one's (limit ${TOLERANCE})`);
                results[pair.name] = { format: pair.format, largestDifferencePerLevel: perLevel };
            }
            assert(warnings.length === 0, `the adapter warned: ${warnings.join(' | ')}`);
            return { ...results, recompiles: platform.stats.recompiles };
        } finally {
            sync.dispose();
            platform.dispose();
        }
    },
    { optionalFeatures: ['texture-compression-bc', 'texture-formats-tier1', 'subgroups'] },
);
