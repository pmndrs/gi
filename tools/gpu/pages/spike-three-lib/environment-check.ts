/**
 * `scene.environment` plus our GI, on a Standard (physical) material.
 *
 * r187 adds the environment's diffuse light separately from probe GI, so a
 * fragment covered by both gets the diffuse twice, and there is no switch to
 * turn it off (PR #35017 adds `material.envMapDiffuseEnabled`). Our light node
 * has a workaround (`suppressEnvironmentDiffuse`, see gi-volume-light.ts). This
 * check measures both behaviours on a flat grey environment:
 *
 *   A  environment alone                      -> R_env
 *   B  our light alone, no environment        -> R_ours
 *   C  our light + environment, no workaround -> expect R_ours + R_env (double count)
 *   D  workaround light at intensity 0 + env  -> R_left: what the environment still adds inside the volume
 *   E  workaround light + environment         -> expect R_ours + R_left
 *
 * and that outside the volume the workaround leaves the environment untouched.
 *
 * Lambert and Phong materials never read `iblIrradiance`, so they cannot
 * double count; that is why this uses MeshStandardNodeMaterial.
 */

import {
    DataTexture,
    EquirectangularReflectionMapping,
    HalfFloatType,
    LinearFilter,
    MeshStandardNodeMaterial,
    RGBAFormat,
    type WebGPURenderer,
} from 'three/webgpu';
import { DataUtils } from 'three';
import { assert } from '../../client.ts';
import type { WrappedGiTextures } from './gi-textures.ts';
import { GiVolumeLight } from './gi-volume-light.ts';
import {
    VOLUME_CENTER,
    VOLUME_SIZE,
    createTestScene,
    formatRgb,
    readCentreAndCorner,
    renderToTarget,
    withinTolerance,
} from './test-scene.ts';

const FRAMES = 4;

/** An equirect environment of one flat colour, in half float so no float32-filterable feature is needed. */
function createFlatEnvironment(red: number, green: number, blue: number): DataTexture {
    const pixelCount = 8 * 4;
    const pixels = new Uint16Array(pixelCount * 4);
    for (let pixel = 0; pixel < pixelCount; pixel++) {
        pixels[pixel * 4] = DataUtils.toHalfFloat(red);
        pixels[pixel * 4 + 1] = DataUtils.toHalfFloat(green);
        pixels[pixel * 4 + 2] = DataUtils.toHalfFloat(blue);
        pixels[pixel * 4 + 3] = DataUtils.toHalfFloat(1);
    }
    const environment = new DataTexture(pixels, 8, 4, RGBAFormat, HalfFloatType);
    environment.mapping = EquirectangularReflectionMapping;
    environment.magFilter = LinearFilter;
    environment.minFilter = LinearFilter;
    environment.needsUpdate = true;
    return environment;
}

function sum(a: readonly number[], b: readonly number[]): number[] {
    return a.map((value, channel) => value + b[channel]);
}

function closeTo(actual: readonly number[], expected: readonly number[]): boolean {
    return actual.every((value, channel) => withinTolerance(value, expected[channel]));
}

export async function checkEnvironmentDiffuse(
    renderer: WebGPURenderer,
    wrapped: WrappedGiTextures,
): Promise<Record<string, unknown>> {
    const test = createTestScene({
        material: new MeshStandardNodeMaterial({ color: 0xffffff, roughness: 1, metalness: 0 }),
    });
    const environment = createFlatEnvironment(0.4, 0.4, 0.4);
    test.scene.environment = environment;

    const plainLight = new GiVolumeLight(wrapped, { center: VOLUME_CENTER, size: VOLUME_SIZE });
    const suppressingLight = new GiVolumeLight(wrapped, {
        center: VOLUME_CENTER,
        size: VOLUME_SIZE,
        suppressEnvironmentDiffuse: true,
    });

    const measure = async () => {
        for (let frame = 0; frame < FRAMES; frame++) renderToTarget(renderer, test);
        const reading = await readCentreAndCorner(renderer, test.target);
        return { centre: reading.centre.mean, corner: reading.corner.mean };
    };

    const environmentAlone = await measure(); // A
    assert(
        environmentAlone.centre.every((value) => value > 0.02),
        `environment alone lit the plane with ${formatRgb(environmentAlone.centre)}: nothing to double count`,
    );

    test.scene.environment = null; // B
    test.scene.add(plainLight);
    const oursAlone = await measure();
    test.scene.environment = environment;

    const oursPlusEnvironment = await measure(); // C
    test.scene.remove(plainLight);

    suppressingLight.intensity = 0; // D
    test.scene.add(suppressingLight);
    const environmentLeft = await measure();
    suppressingLight.intensity = 1; // E
    const withWorkaround = await measure();
    test.scene.remove(suppressingLight);

    const doubleCounted = sum(oursAlone.centre, environmentAlone.centre);
    assert(
        closeTo(oursPlusEnvironment.centre, doubleCounted),
        `ours + environment read ${formatRgb(oursPlusEnvironment.centre)}; expected the plain sum ${formatRgb(doubleCounted)} (ours ${formatRgb(oursAlone.centre)} + env ${formatRgb(environmentAlone.centre)})`,
    );
    const expectedWithWorkaround = sum(oursAlone.centre, environmentLeft.centre);
    assert(
        closeTo(withWorkaround.centre, expectedWithWorkaround),
        `with the workaround the plane read ${formatRgb(withWorkaround.centre)}; expected ours + remaining environment ${formatRgb(expectedWithWorkaround)}`,
    );
    // On a rough dielectric nearly all the environment light is diffuse.
    const removedFraction = (environmentAlone.centre[0] - environmentLeft.centre[0]) / environmentAlone.centre[0];
    assert(
        removedFraction > 0.5,
        `the workaround removed only ${(removedFraction * 100).toFixed(1)}% of the environment light inside the volume`,
    );
    // Outside the volume the environment must be untouched.
    test.scene.add(suppressingLight);
    const outside = (await measure()).corner;
    test.scene.remove(suppressingLight);
    assert(
        closeTo(outside, environmentAlone.corner),
        `outside the volume the workaround changed the environment: read ${formatRgb(outside)}, environment alone ${formatRgb(environmentAlone.corner)}`,
    );

    return {
        environmentAloneCentre: environmentAlone.centre,
        oursAloneStandardMaterialCentre: oursAlone.centre,
        oursPlusEnvironmentNoWorkaround: oursPlusEnvironment.centre,
        sumOfTheTwo: doubleCounted,
        environmentLeftInsideVolumeWithWorkaround: environmentLeft.centre,
        oursPlusEnvironmentWithWorkaround: withWorkaround.centre,
        environmentDiffuseRemovedFraction: removedFraction,
        cornerEnvironmentAlone: environmentAlone.corner,
        cornerWithWorkaround: outside,
    };
}
