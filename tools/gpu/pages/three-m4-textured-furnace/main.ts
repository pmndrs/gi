/**
 * M4, T4.3 done-when 3a: the textured furnace through the three adapter.
 * Textures are applied once, not twice.
 *
 * The furnace room (every surface emits 0.25 and reflects 0.5, so E/π = 0.5
 * everywhere) built as a three scene and mirrored by the adapter's
 * SceneSync, twice, each on a platform of its own with DDGI updating every
 * probe every frame:
 *
 * - **flat**: `MeshStandardMaterial` colour 0.5;
 * - **textured**: colour 1 × `map`, a checker `DataTexture` (sRGB) whose two
 *   colours average 0.5 in linear light, tiled 4 × 4 on each face.
 *
 * Before T4.3 the adapter also multiplied the colour by the map's average
 * (0.5) and passed the map, so the textured room saw an albedo of 0.25 and
 * its field read far below 0.5. Passing: the textured field mean is within
 * 1% of the flat one's, as the core-only `ddgi-textured-furnace` (T4.1) is.
 *
 * Field mean: `ddgiSampleIrradiance` at a grid of wall, floor and ceiling
 * points away from the blocks, over every point and channel.
 *
 * `VITE_DDGI_FRAMES` replaces FRAMES.
 */

import { DataTexture, RepeatWrapping, SRGBColorSpace } from 'three';
import { FURNACE_SIZE_M } from '../../../../scenes/src/procedural/furnace-room.ts';
import type { ProceduralScene } from '../../../../scenes/src/scene-data.ts';
import { assert, runGpuTest } from '../../client.ts';
import { FieldProbe, releaseDeviceOnPageHide, type SurfacePoint } from '../ddgi-shared/harness.ts';
import { srgbToLinear } from '../textures-shared/textures.ts';
import { startThreeGi, threeProceduralScene } from '../three-m4-shared/three-scene.ts';

const env = import.meta.env as Record<string, unknown>;
/** As ddgi-textured-furnace: 30 s of DDGI's clock at 60 Hz, which its adaptive hysteresis needs to settle. */
const FRAMES = Number(env.VITE_DDGI_FRAMES) > 0 ? Math.round(Number(env.VITE_DDGI_FRAMES)) : 1800;
const TEXTURED_TOLERANCE = 0.01;
const POINTS_PER_METRE = 4;
const CHECKER_TILING = 4;
const CHECKER_SIZE = 256;

type Vec3 = [number, number, number];
type Variant = 'flat' | 'textured';

/** Two sRGB bytes near linear 0.25 and 0.75 whose decoded values average closest to 0.5. */
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

/** The checker as a three texture a host would make: sRGB, repeating, tiled by `repeat`. */
function checkerTexture(): { texture: DataTexture; linearMean: number } {
    const [dark, light] = checkerBytes();
    const data = new Uint8Array(CHECKER_SIZE * CHECKER_SIZE * 4);
    for (let y = 0; y < CHECKER_SIZE; y++) {
        for (let x = 0; x < CHECKER_SIZE; x++) {
            const value = (x < CHECKER_SIZE / 2) === (y < CHECKER_SIZE / 2) ? dark : light;
            data.set([value, value, value, 255], (y * CHECKER_SIZE + x) * 4);
        }
    }
    const texture = new DataTexture(data, CHECKER_SIZE, CHECKER_SIZE);
    texture.colorSpace = SRGBColorSpace;
    texture.wrapS = RepeatWrapping;
    texture.wrapT = RepeatWrapping;
    texture.repeat.set(CHECKER_TILING, CHECKER_TILING);
    texture.needsUpdate = true;
    return { texture, linearMean: (srgbToLinear(dark / 255) + srgbToLinear(light / 255)) / 2 };
}

/** Points on the room's six inside faces, away from the blocks (ddgi-textured-furnace's grid). */
function surfacePoints(data: ProceduralScene): SurfacePoint[] {
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

async function runVariant(device: GPUDevice, variant: Variant, log: (line: string) => void, warnings: string[]) {
    const built = threeProceduralScene('furnace-room');
    assert(built.materials.length === 1, `the furnace has ${built.materials.length} materials`);
    const material = built.materials[0];
    let textureMean: number | undefined;
    if (variant === 'textured') {
        const checker = checkerTexture();
        material.color.setRGB(1, 1, 1);
        material.map = checker.texture;
        textureMean = checker.linearMean;
    }
    const run = await startThreeGi(device, built.scene, built.manifest, `three-m4 furnace ${variant}`, (message) => warnings.push(message));
    try {
        const sent = run.sync.sentMaterial(material);
        assert(sent, 'the furnace material was not sent');
        if (variant === 'textured') {
            assert(sent.albedoTexture !== undefined, 'the checker was not sent as the albedo texture');
            assert(sent.albedo!.every((channel) => channel === 1), `the albedo colour sent is ${sent.albedo}, not the material's 1 (a texture average folded in?)`);
            const layer = run.platform.sceneCopy.textureLayer(sent.albedoTexture);
            assert(layer !== null, 'the checker has no store layer after the first frames');
        }
        const field = run.ddgi.field;
        assert(field, 'DDGI has no field after prepare');
        const start = performance.now();
        await run.frames(FRAMES, field.grid.probeCount);
        const runMs = performance.now() - start;
        const points = surfacePoints(built.data);
        const irradiance = await (await FieldProbe.create(device, run.platform)).sample(field, points);
        let sum = 0;
        for (const rgb of irradiance) for (const channel of rgb) sum += channel / Math.PI;
        const meanEOverPi = sum / (irradiance.length * 3);
        log(`${variant}: mean E/pi ${meanEOverPi.toFixed(5)} over ${points.length} points; albedo sent ${JSON.stringify(sent.albedo)}${textureMean !== undefined ? ` × checker (linear mean ${textureMean.toFixed(4)})` : ''}; ${FRAMES} frames in ${(runMs / 1000).toFixed(1)} s`);
        return { meanEOverPi, albedoSent: sent.albedo, textureMean, runMs: Math.round(runMs), recompiles: run.platform.stats.recompiles };
    } finally {
        run.dispose();
    }
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const warnings: string[] = [];
        const flat = await runVariant(device, 'flat', log, warnings);
        const textured = await runVariant(device, 'textured', log, warnings);
        const difference = textured.meanEOverPi / flat.meanEOverPi - 1;
        log(`textured against flat ${(difference * 100).toFixed(3)}% (limit ${TEXTURED_TOLERANCE * 100}%); adapter warnings: ${warnings.length}`);
        assert(warnings.length === 0, `the adapter warned: ${warnings.join(' | ')}`);
        assert(Math.abs(difference) <= TEXTURED_TOLERANCE, `the textured furnace's mean is ${(difference * 100).toFixed(3)}% from the flat one's (limit ${TEXTURED_TOLERANCE * 100}%)`);
        return { frames: FRAMES, flat, textured, texturedOverFlatMinusOne: difference };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
