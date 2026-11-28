/**
 * M4, T4.1 done-whens 1 and 5: the textured furnace.
 *
 * The furnace room (scenes/src/procedural/furnace-room.ts: every surface
 * emits 0.25 and reflects 0.5, so E/π = 0.5 everywhere) three times, each
 * on a platform of its own with DDGI updating every probe every frame:
 *
 * - **flat**: the room as it is, albedo 0.5;
 * - **textured**: albedo colour 1 × a checker texture whose two colours
 *   average 0.5 in linear light (squares about half a metre on the walls), so
 *   probe rays see a mean albedo of 0.5 through the texture store, the ray-cone
 *   mip and the uv mapping. Done-when 1: its field mean is within 1% of flat's;
 * - **textures off**: albedo colour 1 × the same checker, and
 *   `sampleTextures` off (T4.4: off reads every texture at its last mip, its
 *   mean, as the adapter's averages did before M4), so every hit sees the
 *   checker's stored mean `m`. The stored 1 × 1 mip is 8-bit sRGB, so `m`
 *   is 0.5 only to about 0.6% (NOTES), and the furnace answers
 *   E/π = 0.25 / (1 − m). Done-when 5 here: the field mean is within 0.5% of
 *   that closed form. (It was 0.1%, set on the RTX alone; the furnace's own
 *   error differs by about 0.3% between GPUs, flat reading −0.15% on the RTX,
 *   −0.22% on the M5 Pro and −0.28% on the Intel, so 0.5% is the tightest
 *   limit every device can meet; DECISIONS 2026-11-10. Against flat's field
 *   it would carry flat's own error, and not by the same amount on every
 *   GPU.) (The other half
 *   of done-when 5, flat against `m4` before T4.1, is the plain `ddgi-furnace`
 *   page run on both commits.)
 *
 * Field mean: `ddgiSampleIrradiance` at a grid of wall, floor and ceiling
 * points, as `ddgi-furnace` reads it, over every point and channel.
 *
 * `VITE_DDGI_FRAMES` replaces FRAMES. `VITE_DDGI_PRESET=low|medium|high`
 * runs it with that preset's rays, tile sizes and texture store (T5.1;
 * ddgi-shared/page-preset.ts): low's 128² store is checked here.
 */

import { assert, runGpuTest } from '../../client.ts';
import type { PixelData } from '../../../../packages/core/src/platform/contract/scene.ts';
import { FURNACE_SIZE_M } from '../../../../scenes/src/procedural/furnace-room.ts';
import { createDdgiScene, FieldProbe, releaseDeviceOnPageHide, runFrames, type SurfacePoint } from '../ddgi-shared/harness.ts';
import { readLayer, srgbToLinear } from '../textures-shared/textures.ts';
import { pagePreset, presetReport } from '../ddgi-shared/page-preset.ts';

/** The quality preset, when a run names one (T5.1). */
const PRESET = pagePreset();

const env = import.meta.env as Record<string, unknown>;
/** As ddgi-furnace: 30 s of DDGI's clock at 60 Hz, which its adaptive hysteresis needs to settle. */
const FRAMES = Number(env.VITE_DDGI_FRAMES) > 0 ? Math.round(Number(env.VITE_DDGI_FRAMES)) : 1800;
const TEXTURED_TOLERANCE = 0.01;
/** 0.5%: above the furnace's device-to-device spread (see the header; DECISIONS 2026-11-10). */
const TEXTURES_OFF_TOLERANCE = 0.005;
const POINTS_PER_METRE = 4;
/** The checker's squares: 2 × 2 in the texture, the texture tiled 4 × 4 on each face. */
const CHECKER_TILING = 4;
const CHECKER_SIZE = 256;

type Vec3 = [number, number, number];
type Variant = 'flat' | 'textured' | 'texturesOff';

/**
 * Two sRGB bytes near linear 0.25 and 0.75 whose decoded values average
 * closest to 0.5: the checker's dark and light squares.
 */
function checkerBytes(): [number, number] {
    let best: [number, number] = [0, 0];
    let bestError = Infinity;
    for (let dark = 120; dark < 150; dark++) {
        for (let light = 210; light < 240; light++) {
            const error = Math.abs((srgbToLinear(dark / 255) + srgbToLinear(light / 255)) / 2 - 0.5);
            if (error < bestError) {
                bestError = error;
                best = [dark, light];
            }
        }
    }
    return best;
}

function checker(): PixelData {
    const [dark, light] = checkerBytes();
    const data = new Uint8Array(CHECKER_SIZE * CHECKER_SIZE * 4);
    for (let y = 0; y < CHECKER_SIZE; y++) {
        for (let x = 0; x < CHECKER_SIZE; x++) {
            const value = (x < CHECKER_SIZE / 2) === (y < CHECKER_SIZE / 2) ? dark : light;
            data.set([value, value, value, 255], (y * CHECKER_SIZE + x) * 4);
        }
    }
    return { data, width: CHECKER_SIZE, height: CHECKER_SIZE };
}

/**
 * Points on the room's six inside faces, away from the blocks: ddgi-furnace's
 * grid (its helpers are page-local there), with the blocks' boxes taken from
 * the scene data.
 */
function surfacePoints(data: Awaited<ReturnType<typeof createDdgiScene>>['data']): SurfacePoint[] {
    const blocks: { min: Vec3; max: Vec3 }[] = [];
    for (const instance of data.instances) {
        if (instance.name.startsWith('furnace-')) continue;
        const positions = data.meshes[instance.mesh].positions;
        const m = instance.transform;
        const box = { min: [Infinity, Infinity, Infinity] as Vec3, max: [-Infinity, -Infinity, -Infinity] as Vec3 };
        for (let vertex = 0; vertex < positions.length; vertex += 3) {
            for (let axis = 0; axis < 3; axis++) {
                const world = m[axis] * positions[vertex] + m[4 + axis] * positions[vertex + 1] + m[8 + axis] * positions[vertex + 2] + m[12 + axis];
                box.min[axis] = Math.min(box.min[axis], world);
                box.max[axis] = Math.max(box.max[axis], world);
            }
        }
        blocks.push(box);
    }
    const size: Vec3 = [FURNACE_SIZE_M[0], FURNACE_SIZE_M[1], FURNACE_SIZE_M[2]];
    const covered = (point: Vec3) => blocks.some((box) => [0, 1, 2].every((axis) => point[axis] > box.min[axis] - 0.05 && point[axis] < box.max[axis] + 0.05));
    const points: SurfacePoint[] = [];
    for (let axis = 0; axis < 3; axis++) {
        for (const end of [0, 1]) {
            const [a, b] = [0, 1, 2].filter((other) => other !== axis);
            const stepsA = Math.round(size[a] * POINTS_PER_METRE);
            const stepsB = Math.round(size[b] * POINTS_PER_METRE);
            for (let i = 0; i < stepsA; i++) {
                for (let j = 0; j < stepsB; j++) {
                    const position: Vec3 = [0, 0, 0];
                    position[axis] = end * size[axis];
                    position[a] = ((i + 0.5) / stepsA) * size[a];
                    position[b] = ((j + 0.5) / stepsB) * size[b];
                    if (covered(position)) continue;
                    const normal: Vec3 = [0, 0, 0];
                    normal[axis] = end === 0 ? 1 : -1;
                    points.push({ position, normal, viewDirection: [-normal[0], -normal[1], -normal[2]] });
                }
            }
        }
    }
    return points;
}

interface VariantResult {
    readonly meanEOverPi: number;
    readonly pointSpread: number;
    readonly runMs: number;
    readonly textureMean?: number;
}

async function runVariant(device: GPUDevice, variant: Variant, log: (line: string) => void): Promise<VariantResult> {
    const scene = await createDdgiScene(device, 'furnace-room', { ...PRESET?.ddgi }, { textureLayerSize: PRESET?.textureLayerSize });
    const field = scene.ddgi.field;
    assert(field, 'DDGI has no field after prepare');
    let textureMean: number | undefined;
    if (variant !== 'flat') {
        // One material for every surface of the room.
        assert(scene.materials.length === 1, `the furnace has ${scene.materials.length} materials`);
        const texture = scene.platform.scene.createTexture({ source: checker(), srgb: true, label: 'furnace checker' });
        scene.platform.scene.updateMaterial(scene.materials[0], {
            albedo: [1, 1, 1],
            albedoTexture: texture,
            albedoUv: { matrix: [CHECKER_TILING, 0, 0, CHECKER_TILING, 0, 0] },
        });
        scene.platform.shade.settings.sampleTextures = variant === 'textured';
        await runFrames(device, scene.platform, 1, 0);
        const layer = scene.platform.sceneCopy.textureLayer(texture);
        assert(layer !== null && scene.platform.textures.texture, 'the checker has no layer');
        // The last mip (1 texel) is the texture's mean in linear light.
        const levels = await readLayer(device, scene.platform.textures.texture, layer);
        textureMean = srgbToLinear(levels[levels.length - 1][0] / 255);
    }
    const start = performance.now();
    await runFrames(device, scene.platform, FRAMES, field.grid.probeCount);
    const runMs = performance.now() - start;
    const points = surfacePoints(scene.data);
    const irradiance = await (await FieldProbe.create(device, scene.platform)).sample(field, points);
    let sum = 0;
    let lowest = Infinity;
    let highest = -Infinity;
    for (const rgb of irradiance) {
        for (const channel of rgb) sum += channel / Math.PI;
        const point = (rgb[0] + rgb[1] + rgb[2]) / (3 * Math.PI);
        lowest = Math.min(lowest, point);
        highest = Math.max(highest, point);
    }
    const meanEOverPi = sum / (irradiance.length * 3);
    log(`${variant}: mean E/pi ${meanEOverPi.toFixed(5)} over ${points.length} points (points from ${lowest.toFixed(4)} to ${highest.toFixed(4)})${textureMean !== undefined ? `, texture mean ${textureMean.toFixed(4)}` : ''}, ${FRAMES} frames in ${(runMs / 1000).toFixed(1)} s`);
    scene.platform.dispose();
    return { meanEOverPi, pointSpread: highest - lowest, runMs: Math.round(runMs), textureMean };
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const flat = await runVariant(device, 'flat', log);
        const textured = await runVariant(device, 'textured', log);
        const texturesOff = await runVariant(device, 'texturesOff', log);
        const texturedDifference = textured.meanEOverPi / flat.meanEOverPi - 1;
        // With textures off every hit reads the stored mean m, so the furnace's closed-form answer is Le / (1 - m), Le = 0.25.
        assert(texturesOff.textureMean !== undefined, 'the textures-off variant has no stored mean');
        const predictedOff = 0.25 / (1 - texturesOff.textureMean);
        const texturesOffDifference = texturesOff.meanEOverPi / predictedOff - 1;
        log(
            `textured against flat ${(texturedDifference * 100).toFixed(3)}%; textures off ${texturesOff.meanEOverPi.toFixed(5)} against the closed form ${predictedOff.toFixed(5)} ` +
                `for the stored mean ${texturesOff.textureMean.toFixed(5)}: ${(texturesOffDifference * 100).toFixed(4)}%`,
        );
        assert(Math.abs(texturedDifference) <= TEXTURED_TOLERANCE, `the textured furnace's mean is ${(texturedDifference * 100).toFixed(3)}% from the flat one's (limit ${TEXTURED_TOLERANCE * 100}%)`);
        assert(Math.abs(texturesOffDifference) <= TEXTURES_OFF_TOLERANCE, `with sampleTextures off the mean is ${(texturesOffDifference * 100).toFixed(4)}% from the closed form for the stored texture mean (limit ${TEXTURES_OFF_TOLERANCE * 100}%): flat ${flat.meanEOverPi.toFixed(5)}, off ${texturesOff.meanEOverPi.toFixed(5)}, stored mean ${texturesOff.textureMean.toFixed(5)}, textured ${textured.meanEOverPi.toFixed(5)}`);
        return { ...presetReport(PRESET), frames: FRAMES, flat, textured, texturesOff, texturedDifference, texturesOffDifference };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
