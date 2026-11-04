/**
 * Writes the shared sky files: one per scene whose manifest environment is
 * `procedural`, into `scenes/sky/<scene id>.exr` plus a `<scene id>.json`
 * sidecar. See `tools/sky/README.md`.
 *
 *   node tools/sky/generate.ts                 every scene with a procedural sky
 *   node tools/sky/generate.ts --scene sponza  one scene (repeatable, or comma-separated)
 *   node tools/sky/generate.ts --check         regenerate in memory and compare with the committed files; exit 1 if any differ
 *
 * The files are committed: each is about 50 kB, and committing them means
 * the viewer, Blender and the native scripts never need a generation step.
 * `--check` (also run by the unit tests) keeps them in step with the manifests
 * and the code. It compares decoded pixels, not file bytes, so a different
 * zlib build producing different compressed bytes is not a failure.
 */

import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { deflateSync, inflateSync } from 'node:zlib';
import { parseManifest, type SceneManifest } from '../../scenes/src/manifest.ts';
import { decodeHalfExr, encodeHalfExr, halfToFloat, type HalfRgbImage } from '../../scenes/src/sky/exr-half.ts';
import { SKY_DIRECTORY, skyFileName, skySidecarName, type SkySidecar } from '../../scenes/src/sky/sky-file.ts';
import { buildSkyImage, type SkyImage } from '../../scenes/src/sky/sky-image.ts';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
export const skyDirectory = join(repoRoot, ...SKY_DIRECTORY.split('/'));
const manifestsDirectory = join(repoRoot, 'scenes', 'manifests');

/**
 * Largest difference allowed by `--check` between a committed and a regenerated
 * texel, relative to the regenerated value: 2e-3, about two half-float steps
 * (one step is 2^-10, about 9.8e-4). Exact equality is expected on the same
 * Node version; this only forgives a value that rounds to a neighbouring half
 * after a last-bit change in Math.exp/acos between JavaScript engines.
 */
const CHECK_RELATIVE_TOLERANCE = 2e-3;

export async function loadSkyManifests(only: readonly string[] = []): Promise<SceneManifest[]> {
    const files = (await readdir(manifestsDirectory)).filter((file) => file.endsWith('.json')).sort();
    const manifests: SceneManifest[] = [];
    for (const file of files) {
        const manifest = parseManifest(await readFile(join(manifestsDirectory, file), 'utf8'), file);
        if (manifest.environment.type !== 'procedural') continue;
        if (only.length > 0 && !only.includes(manifest.id)) continue;
        manifests.push(manifest);
    }
    const missing = only.filter((id) => !manifests.some((manifest) => manifest.id === id));
    if (missing.length > 0) throw new Error(`no scene with a procedural sky named ${missing.join(', ')}`);
    return manifests;
}

function pixelSha256(image: HalfRgbImage): string {
    const bytes = new Uint8Array(image.halves.length * 2);
    const view = new DataView(bytes.buffer);
    image.halves.forEach((half, index) => view.setUint16(index * 2, half, true));
    return createHash('sha256').update(bytes).digest('hex');
}

/** Rounds for the JSON, so the sidecar stays readable and stable. */
function round(value: number, digits = 6): number {
    return Number(value.toPrecision(digits));
}

export function makeSidecar(manifest: SceneManifest, sky: SkyImage): SkySidecar {
    const { calibration, measured } = sky;
    const radians = (value: number) => (value * 180) / Math.PI;
    return {
        scene: manifest.id,
        file: skyFileName(manifest.id),
        generator: 'tools/sky/generate.ts',
        skyModel: 'Preetham, Shirley and Smits 1999 (no sun disc; black below the horizon)',
        width: sky.image.width,
        height: sky.image.height,
        layout:
            'equirectangular, +Y up; row 0 is the zenith; u = atan2(z, x) / (2 pi) + 0.5, v (down from the top) = acos(y) / pi; the same as three.js equirectUV',
        sunDirection: sky.sun.direction.map((value) => round(value)) as [number, number, number],
        sunElevationDeg: round(radians(sky.sun.elevationRad), 5),
        turbidity: sky.turbidity,
        sunIlluminance: round(sky.sun.illuminance),
        sunSource: sky.sun.source,
        calibration: {
            model: calibration.model,
            skyToSunHorizontalRatio: round(calibration.clearSky.skyToSunHorizontalRatio),
            diffuseFraction: round(calibration.clearSky.diffuseFraction),
            airMass: round(calibration.clearSky.airMass),
            intensity: calibration.intensity,
            sunHorizontalIlluminance: round(calibration.sunHorizontalIlluminance),
            targetSkyHorizontalIlluminance: round(calibration.targetSkyHorizontalIlluminance),
        },
        measured: {
            skyHorizontalIrradianceRgb: measured.rgb.map((value) => round(value)) as [number, number, number],
            skyHorizontalIlluminance: round(measured.luminance),
            calibrationError: round(measured.luminance / calibration.targetSkyHorizontalIlluminance - 1, 3),
        },
        flatEquivalentRadianceRgb: measured.rgb.map((value) => round(value / Math.PI)) as [number, number, number],
        pixelSha256: pixelSha256(sky.image),
    };
}

/** Differences between a committed sky and a fresh one, as readable lines (empty when they match). */
export async function compareWithCommitted(manifest: SceneManifest, sky: SkyImage): Promise<string[]> {
    const filePath = join(skyDirectory, skyFileName(manifest.id));
    const sidecarPath = join(skyDirectory, skySidecarName(manifest.id));
    if (!existsSync(filePath) || !existsSync(sidecarPath)) return [`${manifest.id}: ${SKY_DIRECTORY}/${skyFileName(manifest.id)} or its .json is missing`];

    const problems: string[] = [];
    const committed = await decodeHalfExr(new Uint8Array(await readFile(filePath)), (bytes) => inflateSync(bytes));
    if (committed.width !== sky.image.width || committed.height !== sky.image.height) {
        return [`${manifest.id}: committed sky is ${committed.width}x${committed.height}, expected ${sky.image.width}x${sky.image.height}`];
    }
    let worst = 0;
    for (let i = 0; i < committed.halves.length; i++) {
        const expected = halfToFloat(sky.image.halves[i]);
        const found = halfToFloat(committed.halves[i]);
        const difference = Math.abs(found - expected) / Math.max(Math.abs(expected), 1e-4);
        worst = Math.max(worst, difference);
    }
    if (worst > CHECK_RELATIVE_TOLERANCE) problems.push(`${manifest.id}: pixels differ from a fresh generation by up to ${(worst * 100).toFixed(2)}%`);

    const sidecar = JSON.parse(await readFile(sidecarPath, 'utf8')) as SkySidecar;
    const fresh = makeSidecar(manifest, sky);
    const comparable = (value: SkySidecar) => JSON.stringify({ ...value, pixelSha256: undefined, measured: undefined });
    if (comparable(sidecar) !== comparable(fresh)) problems.push(`${manifest.id}: the sidecar's settings differ from a fresh generation`);
    return problems;
}

async function main(): Promise<void> {
    const { values } = parseArgs({
        options: {
            scene: { type: 'string', multiple: true, default: [] },
            check: { type: 'boolean', default: false },
            help: { type: 'boolean', default: false },
        },
    });
    if (values.help) {
        console.log('usage: node tools/sky/generate.ts [--scene id[,id]] [--check]');
        return;
    }
    const only = values.scene.flatMap((value) => value.split(',')).filter(Boolean);
    const manifests = await loadSkyManifests(only);

    const problems: string[] = [];
    for (const manifest of manifests) {
        const sky = buildSkyImage(manifest);
        const sidecar = makeSidecar(manifest, sky);
        const summary =
            `${manifest.id}: sun ${sidecar.sunElevationDeg.toFixed(1)} deg, T ${sidecar.turbidity}, ` +
            `sky/sun on the ground ${sidecar.calibration.skyToSunHorizontalRatio.toFixed(4)} ` +
            `(diffuse fraction ${sidecar.calibration.diffuseFraction.toFixed(3)}), ` +
            `sky irradiance ${sidecar.measured.skyHorizontalIlluminance.toPrecision(5)} (error ${(sidecar.measured.calibrationError * 100).toFixed(3)}%)`;
        if (values.check) {
            const found = await compareWithCommitted(manifest, sky);
            problems.push(...found);
            console.log(`${found.length === 0 ? 'ok  ' : 'DIFF'} ${summary}`);
        } else {
            const file = encodeHalfExr(sky.image, (bytes) => deflateSync(bytes, { level: 9 }));
            await writeFile(join(skyDirectory, skyFileName(manifest.id)), file);
            await writeFile(join(skyDirectory, skySidecarName(manifest.id)), `${JSON.stringify(sidecar, null, 4)}\n`);
            console.log(`${summary}; wrote ${skyFileName(manifest.id)} (${(file.length / 1024).toFixed(0)} kB)`);
        }
    }

    if (values.check && only.length === 0) {
        // A sky file whose scene no longer has a procedural sky is stale.
        const expected = new Set(manifests.flatMap((manifest) => [skyFileName(manifest.id), skySidecarName(manifest.id)]));
        const extra = (await readdir(skyDirectory)).filter((file) => (file.endsWith('.exr') || file.endsWith('.json')) && !expected.has(file));
        for (const file of extra) problems.push(`${SKY_DIRECTORY}/${file} belongs to no scene with a procedural sky`);
    }
    if (problems.length > 0) {
        console.error(`\n${problems.join('\n')}\nRun "node tools/sky/generate.ts" to rewrite the sky files.`);
        process.exitCode = 1;
    }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    main().catch((error: unknown) => {
        console.error(error instanceof Error ? error.message : error);
        process.exit(1);
    });
}
