/**
 * The GI composite of a Cycles reference, and the check on its lights-only
 * render. render.py renders the parts; run.ts calls these.
 *
 * DDGI's field holds every light that reaches a probe by a ray: bounces, and
 * also the sky and emitters seen straight. Only the manifest's lights (sun,
 * point, spot) are drawn by the renderer itself, outside the field. Cycles'
 * indirect pass leaves the sky and emitters seen straight out (they count as
 * direct), so the reference for DDGI is
 *
 *     GI = indirect + direct - lights-only direct
 *
 * (DECISIONS: "What 'indirect light' means against Cycles in a scene with a
 * sky"). The subtraction is only right if the lights-only render really has
 * the lights alone, so it is checked where the answer is known:
 *
 * - `equals-direct`: the manifest lights are the scene's only source (no sky,
 *   no emitter). Switching everything else off changes nothing, so the
 *   lights-only direct image must equal the normal direct image within noise.
 *   This catches a lights-only render that lost or changed the lights' own
 *   light (a wrong bounce setting, a light switched off with the emitters).
 * - `black`: the scene has no manifest lights. The lights-only image must be 0.
 *   This catches an emitter (or sky) the switch-off missed.
 * - `null`: both kinds of source; nothing known to check against.
 */

import type { FloatImage } from '../../compare/exr.ts';
import { compareImages, downsample } from '../../compare/metrics.ts';

/** render.py's `lights_only_expectation`. */
export type LightsOnlyExpectation = 'equals-direct' | 'black' | null;

/**
 * `equals-direct` tolerances. The two renders share seed and sample count, so
 * the camera rays and the first hit's light samples are the same and most of
 * the noise cancels; what is left is the second render's different bounce
 * settings (it stops after the first hit). Measured on cornell-box-point and
 * two-room (tools/reference/README.md, "The GI composite"): the differences
 * are far below these, so a failure means a real change in the lights' light,
 * not noise.
 */
export const LIGHTS_ONLY_TOLERANCE = {
    /** |sum(lights-only) - sum(direct)| / sum(direct). */
    bias: 0.005,
    /** Mean relative error after 8 x 8 block averaging (tools/compare's definition). */
    blockError: 0.02,
    blockSize: 8,
    /** `black`: sum(lights-only) / sum(direct) must stay below this. */
    blackShare: 1e-4,
} as const;

export interface LightsOnlyCheck {
    readonly expected: LightsOnlyExpectation;
    readonly passed: boolean;
    /** One line for the console and the error message. */
    readonly summary: string;
    readonly bias?: number;
    readonly blockError?: number;
    readonly share?: number;
}

function requireSameSize(images: readonly FloatImage[]): void {
    const [first] = images;
    for (const image of images) {
        if (image.width !== first.width || image.height !== first.height) {
            throw new Error(`the GI parts differ in size: ${first.width}x${first.height} and ${image.width}x${image.height}`);
        }
    }
}

function sum(rgb: Float32Array): number {
    let total = 0;
    for (const value of rgb) total += value;
    return total;
}

const percent = (value: number): string => `${(value * 100).toFixed(3)}%`;

/** Checks the lights-only direct image against what it must be (see the file header). */
export function checkLightsOnlyImage(expected: LightsOnlyExpectation, direct: FloatImage, lightsDirect: FloatImage): LightsOnlyCheck {
    requireSameSize([direct, lightsDirect]);
    if (expected === null) {
        return { expected, passed: true, summary: 'lights-only render not checked: the scene has both manifest lights and other sources' };
    }
    if (expected === 'black') {
        const directSum = sum(direct.rgb);
        const lightsSum = sum(lightsDirect.rgb);
        const share = directSum > 0 ? lightsSum / directSum : lightsSum > 0 ? Infinity : 0;
        const passed = Math.abs(share) <= LIGHTS_ONLY_TOLERANCE.blackShare;
        return {
            expected,
            passed,
            share,
            summary: `lights-only render, scene without manifest lights: ${percent(share)} of the direct light left (must be 0, limit ${percent(LIGHTS_ONLY_TOLERANCE.blackShare)})`,
        };
    }
    const { blockSize } = LIGHTS_ONLY_TOLERANCE;
    const pixels = compareImages(direct.rgb, lightsDirect.rgb);
    const directBlocks = downsample(direct.rgb, direct.width, direct.height, blockSize);
    const lightsBlocks = downsample(lightsDirect.rgb, lightsDirect.width, lightsDirect.height, blockSize);
    const blocks = compareImages(directBlocks.rgb, lightsBlocks.rgb);
    const passed = Math.abs(pixels.bias) <= LIGHTS_ONLY_TOLERANCE.bias && blocks.meanRelativeError <= LIGHTS_ONLY_TOLERANCE.blockError;
    return {
        expected,
        passed,
        bias: pixels.bias,
        blockError: blocks.meanRelativeError,
        summary:
            `lights-only render, scene lit by the manifest lights alone, against the normal direct image: ` +
            `bias ${percent(pixels.bias)} (limit ±${percent(LIGHTS_ONLY_TOLERANCE.bias)}), ` +
            `${blockSize}x${blockSize} block error ${percent(blocks.meanRelativeError)} (limit ${percent(LIGHTS_ONLY_TOLERANCE.blockError)})`,
    };
}

/** The sum of `add` minus `subtract`, pixel by pixel. */
export function composeGiImage(add: readonly FloatImage[], subtract: FloatImage): FloatImage {
    if (add.length === 0) throw new Error('nothing to add');
    requireSameSize([...add, subtract]);
    const [first, ...others] = add;
    const rgb = new Float32Array(first.rgb);
    for (let index = 0; index < rgb.length; index++) {
        let value = rgb[index];
        for (const image of others) value += image.rgb[index];
        rgb[index] = value - subtract.rgb[index];
    }
    return { width: first.width, height: first.height, rgb };
}
