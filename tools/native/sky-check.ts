/**
 * Checks that the native harness's texture sky (T2.7, fork option
 * scene.skyTexture) shows the shared sky file the right way round: for every
 * pixel where the camera's ray leaves the scene, the harness's G-buffer sky
 * colour is compared with the sky file sampled in the same direction by the
 * file's own layout (the sidecar's: u = atan2(z, x) / 2pi + 0.5,
 * v = acos(y) / pi, row 0 at the zenith; three.js's equirectUV, so the
 * viewer's sky dome reads the same texel). It also reports where the
 * brightest sky pixel is against the sun, and that the sky below the horizon
 * is black (the flat sky was not).
 *
 *   node tools/native/sky-check.ts --scene open-ground --capture <screenshots folder>
 *
 * The capture comes from `play.ps1 -Sky texture -CaptureFrames n -CaptureField`
 * (receiver.albedo-flags.bgra8.bin, native-frame.json).
 */

import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { inflateSync } from 'node:zlib';
import { decodeHalfExr, halfToFloat } from '../../scenes/src/sky/exr-half.ts';

const repoRoot = resolve(fileURLToPath(import.meta.url), '..', '..', '..');

/** The harness's LinearToSRGB (Common.hlsl), as written there. */
function harnessSrgb(value: number): number {
    const v = Math.min(1, Math.max(0, value));
    return v < 0.0031308 ? v * 12.92 : Math.pow(v * 1.055, 1 / 2.4) - 0.055;
}

interface NativeFrame {
    width: number;
    height: number;
    fov: number;
    aspect: number;
    position: number[];
    right: number[];
    up: number[];
    forward: number[];
}

async function main(): Promise<void> {
    const { values } = parseArgs({ options: { scene: { type: 'string' }, capture: { type: 'string' } } });
    if (!values.scene || !values.capture) throw new Error('usage: node tools/native/sky-check.ts --scene <id> --capture <folder>');
    const sky = await decodeHalfExr(new Uint8Array(readFileSync(join(repoRoot, 'scenes', 'sky', `${values.scene}.exr`))), (bytes) => inflateSync(bytes));
    const sidecar = JSON.parse(readFileSync(join(repoRoot, 'scenes', 'sky', `${values.scene}.json`), 'utf8')) as { sunDirection: number[] };
    const frame = JSON.parse(readFileSync(join(values.capture, 'native-frame.json'), 'utf8')) as NativeFrame;
    const gbuffer = readFileSync(join(values.capture, 'receiver.albedo-flags.bgra8.bin'));

    const texel = (x: number, y: number, channel: number) => halfToFloat(sky.halves[(y * sky.width + x) * 3 + channel]);
    // Bilinear, wrapping in u and clamped in v, as the harness's SampleSkyImage
    const sample = (direction: number[]): number[] => {
        const u = Math.atan2(direction[2], direction[0]) / (2 * Math.PI) + 0.5;
        const v = Math.acos(Math.min(1, Math.max(-1, direction[1]))) / Math.PI;
        const x = u * sky.width - 0.5;
        const y = v * sky.height - 0.5;
        const x0 = Math.floor(x);
        const y0 = Math.floor(y);
        const fx = x - x0;
        const fy = y - y0;
        const c0 = ((x0 % sky.width) + sky.width) % sky.width;
        const c1 = (c0 + 1) % sky.width;
        const r0 = Math.min(sky.height - 1, Math.max(0, y0));
        const r1 = Math.min(sky.height - 1, Math.max(0, y0 + 1));
        return [0, 1, 2].map((channel) => {
            const top = texel(c0, r0, channel) + (texel(c1, r0, channel) - texel(c0, r0, channel)) * fx;
            const bottom = texel(c0, r1, channel) + (texel(c1, r1, channel) - texel(c0, r1, channel)) * fx;
            return top + (bottom - top) * fy;
        });
    };

    const tanHalf = Math.tan(((frame.fov * Math.PI) / 180) * 0.5);
    let skyPixels = 0;
    let below = 0;
    let belowNotBlack = 0;
    let sumDifference = 0;
    let worst = 0;
    let within1 = 0;
    let mirroredSum = 0;
    let brightest = { value: -1, direction: [0, 0, 0] };
    for (let y = 0; y < frame.height; y += 2) {
        for (let x = 0; x < frame.width; x += 2) {
            const offset = (y * frame.width + x) * 4;
            const alpha = gbuffer[offset + 3];
            if (Math.abs(alpha - 128) > 1) continue; // COMPOSITE_FLAG_POSTPROCESS_PIXEL (0.5): the ray left the scene
            const px = ((x + 0.5) / frame.width) * 2 - 1;
            const py = ((y + 0.5) / frame.height) * -2 + 1;
            const d = [0, 1, 2].map((i) => px * frame.aspect * tanHalf * frame.right[i] + py * tanHalf * frame.up[i] + frame.forward[i]);
            const length = Math.hypot(d[0], d[1], d[2]);
            const direction = d.map((value) => value / length);
            const harness = [gbuffer[offset + 2], gbuffer[offset + 1], gbuffer[offset]]; // B, G, R in memory
            const expected = sample(direction).map((value) => harnessSrgb(value) * 255);
            // The same pixel if the map were mirrored left to right: what a wrong-handed lookup would show
            const mirrored = sample([direction[0], direction[1], -direction[2]]).map((value) => harnessSrgb(value) * 255);
            const difference = Math.max(...harness.map((value, i) => Math.abs(value - expected[i])));
            skyPixels++;
            sumDifference += difference;
            mirroredSum += Math.max(...harness.map((value, i) => Math.abs(value - mirrored[i])));
            worst = Math.max(worst, difference);
            if (difference <= 1.5) within1++;
            if (direction[1] < -0.01) {
                below++;
                if (Math.max(...harness) > 0) belowNotBlack++;
            }
            const luminance = 0.2126 * harness[0] + 0.7152 * harness[1] + 0.0722 * harness[2];
            if (luminance > brightest.value) brightest = { value: luminance, direction };
        }
    }
    const sun = sidecar.sunDirection;
    const sunLength = Math.hypot(sun[0], sun[1], sun[2]);
    const angleToSun = (Math.acos(brightest.direction.reduce((sum, value, i) => sum + (value * sun[i]) / sunLength, 0)) * 180) / Math.PI;
    console.log(
        `${values.scene}: ${skyPixels} sky pixels (every 2nd pixel each way); harness against the file in the same direction: mean ${(sumDifference / skyPixels).toFixed(2)} of 255, ` +
            `largest ${worst.toFixed(1)}, ${((within1 / skyPixels) * 100).toFixed(1)}% within 1.5; the same against a mirrored map: mean ${(mirroredSum / skyPixels).toFixed(2)}`,
    );
    console.log(`brightest sky pixel ${angleToSun.toFixed(1)} degrees from the sun direction; ${below} pixels below the horizon, ${belowNotBlack} of them not black`);
    const ok = skyPixels > 0 && sumDifference / skyPixels <= 1 && within1 / skyPixels >= 0.99 && belowNotBlack === 0;
    console.log(ok ? 'ok' : 'FAIL');
    if (!ok) process.exitCode = 1;
}

main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
});
