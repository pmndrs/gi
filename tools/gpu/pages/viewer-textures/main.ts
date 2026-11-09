/**
 * T3.12: a scene in the viewer's own code (`ViewerCore`, DDGI on, the scene's
 * exposure) with its model textures compressed at load or not
 * (apps/viewer/src/scene/compress-textures.ts), so Bistro exterior fits a 6 GB GPU.
 *
 * Checks (the page fails on any):
 * - with compression asked for, the device could compress (`texture-compression-bc`),
 *   and every texture the plan compresses was compressed (procedural scenes have none);
 * - every texture in the materials is what the plan says: a `CompressedTexture`
 *   with its whole mip chain where planned, BC5 in normal-map slots and BC7
 *   elsewhere; in `compressed` mode none of the maps GI reads colours from
 *   (`GI_AVERAGED_SLOTS`); in `uncompressed` mode none at all;
 * - no GPU error or device loss, DDGI starts and runs.
 *
 * Recorded, not judged (M3 is prototype level, DECISIONS 2026-11-09): what
 * compression did (counts, bytes, time), three's own texture bytes (it counts a
 * compressed texture as 1 byte, so this is the uncompressed ones), the load
 * time, and the wall time of GI-on frames after DDGI settles (each frame drawn
 * and waited for). The picture at the end is returned as a PNG data URL
 * (`data.picture`), to compare between modes; the GPU's memory is read
 * outside the page (nvidia-smi).
 *
 *   VITE_DDGI_SCENE=bistro-exterior   (default sponza)
 *   VITE_DDGI_TEXTURES=compressed     (or all-compressed, uncompressed; default compressed)
 *   VITE_DDGI_MAX_TEXTURE_SIZE=1024   (optional: T2.17's cap, for the capped comparison)
 *   VITE_DDGI_CAMERA=0                (the bookmark, by index; default 0)
 *   VITE_DDGI_FRAMES=300              (frames drawn after DDGI starts, before timing; default 300)
 *   VITE_DDGI_TIMED_FRAMES=60         (frames timed; default 60)
 *
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "T3.12 textures" -- pnpm test:gpu --filter viewer-textures --gpu nvidia --json <report.json>`
 */

import { CompressedTexture, RED_GREEN_RGTC2_Format, RGBA_BPTC_Format, type Material, type Mesh, type Texture } from 'three';
import { reportGpuTest } from '../../client.ts';
import { ViewerCore } from '../../../../apps/viewer/src/core/viewer-core.ts';
import { GI_AVERAGED_SLOTS, planTexture, textureSlots } from '../../../../apps/viewer/src/scene/compress-textures.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { TEXTURE_COMPRESSIONS, type TextureCompression } from '../../../../apps/viewer/src/scene/texture-compression-mode.ts';
import { DEFAULT_DDGI_VIEWER_SETTINGS } from '../../../../apps/viewer/src/techniques/ddgi-technique.ts';
import { defaultExposureEv } from '../../../../scenes/src/manifest.ts';

const env = import.meta.env as Record<string, unknown>;
const text = (value: unknown): string | null => (typeof value === 'string' && value !== '' ? value : null);
const SCENE = text(env.VITE_DDGI_SCENE) ?? 'sponza';
const MODE: TextureCompression = TEXTURE_COMPRESSIONS.find((mode) => mode === env.VITE_DDGI_TEXTURES) ?? 'compressed';
const MAX_TEXTURE_SIZE = text(env.VITE_DDGI_MAX_TEXTURE_SIZE) === null ? undefined : Number(env.VITE_DDGI_MAX_TEXTURE_SIZE);
const CAMERA = Number(text(env.VITE_DDGI_CAMERA) ?? 0);
const SETTLE_FRAMES = Number(text(env.VITE_DDGI_FRAMES) ?? 300);
const TIMED_FRAMES = Number(text(env.VITE_DDGI_TIMED_FRAMES) ?? 60);
const WIDTH = 1280;
const HEIGHT = 720;
/** How long DDGI may take to start (the scene copy and trees; Bistro takes tens of seconds). */
const DDGI_START_TIMEOUT_MS = 300_000;

const startTime = performance.now();
const logElement = document.getElementById('log') as HTMLElement;
const logLines: string[] = [];
const log = (message: string): void => {
    logLines.push(message);
    logElement.textContent = logLines.join('\n');
    console.log(message);
};
const pageErrors: string[] = [];
window.addEventListener('error', (event) => pageErrors.push(`page error: ${event.message}`));
window.addEventListener('unhandledrejection', (event) => pageErrors.push(`unhandled rejection: ${String(event.reason)}`));

function isTexture(value: unknown): value is Texture {
    return typeof value === 'object' && value !== null && (value as { isTexture?: boolean }).isTexture === true;
}

/** Levels in a full mip chain down to 1×1. */
function fullChainLevels(width: number, height: number): number {
    return Math.floor(Math.log2(Math.max(width, height))) + 1;
}

/** Every texture in the loaded model's materials, by slot, checked against the mode. Returns the problems and counts. */
function checkTextures(materials: Set<Material>, mode: TextureCompression): { problems: string[]; compressed: number; uncompressed: number } {
    const problems: string[] = [];
    let compressed = 0;
    let uncompressed = 0;
    for (const material of materials) {
        for (const [slot, value] of Object.entries(material)) {
            if (!isTexture(value)) continue;
            const where = `${material.name || material.type}.${slot}`;
            if (!(value instanceof CompressedTexture)) {
                uncompressed++;
                continue;
            }
            compressed++;
            if (mode === 'uncompressed') problems.push(`${where} is compressed in uncompressed mode`);
            if (mode === 'compressed' && GI_AVERAGED_SLOTS.includes(slot)) problems.push(`${where} is compressed, but GI reads its colours until M4`);
            const expectedFormat = slot === 'normalMap' ? RED_GREEN_RGTC2_Format : RGBA_BPTC_Format;
            if (value.format !== expectedFormat) problems.push(`${where} has format ${value.format}, expected ${expectedFormat}`);
            // gputex pads each level to whole 4x4 blocks (a 1x1 image is one block), and keeps the image's own size in userData.
            const width = Number(value.userData.logicalWidth);
            const height = Number(value.userData.logicalHeight);
            const levels = value.mipmaps?.length ?? 0;
            if (levels !== fullChainLevels(width, height)) problems.push(`${where} has ${levels} mip levels for ${width}x${height}`);
        }
    }
    return { problems, compressed, uncompressed };
}

function percentile(values: number[], fraction: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))];
}

try {
    const canvas = document.getElementById('view') as HTMLCanvasElement;
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    canvas.style.width = `${WIDTH}px`;
    canvas.style.height = `${HEIGHT}px`;
    const core = await ViewerCore.create(canvas);
    core.resize(WIDTH, HEIGHT);
    window.addEventListener('pagehide', () => core.dispose());

    const manifest = loadAllManifests().find((candidate) => candidate.id === SCENE);
    if (!manifest) throw new Error(`no scene ${SCENE}`);
    const bookmark = manifest.cameras[CAMERA];
    if (!bookmark) throw new Error(`${SCENE} has no camera ${CAMERA}`);

    core.layout = { left: 'none', right: null, giMode: 'on', split: 0.5 };
    core.configureDdgi(DEFAULT_DDGI_VIEWER_SETTINGS);
    const loadStartedAt = performance.now();
    // Each loading stage once, without its running counts ("(3 of 23)"), so the log stays short.
    let lastStage = '';
    const logStage = (stage: string): void => {
        const name = stage.replace(/ \([^)]*\)$/, '');
        if (name !== lastStage) log(stage);
        lastStage = name;
    };
    const loaded = await core.loadScene(manifest, (progress) => logStage(progress.stage), {
        textureCompression: MODE,
        ...(MAX_TEXTURE_SIZE === undefined ? {} : { maxTextureSize: MAX_TEXTURE_SIZE }),
    });
    const loadMs = performance.now() - loadStartedAt;
    for (const part of loaded.textureCompression) {
        log(
            `compression (${part.mode}): ${part.compressed} textures (${part.normalMaps} normal maps), kept ${part.keptForGi} for GI and ${part.keptOther} others; ` +
                `${(part.uncompressedBytes / 1e6).toFixed(0)} MB as RGBA8 -> ${(part.compressedBytes / 1e6).toFixed(0)} MB in ${part.milliseconds.toFixed(0)} ms` +
                (part.unavailable ? `; not compressed: ${part.unavailable}` : ''),
        );
    }

    const problems: string[] = [];
    if (MODE !== 'uncompressed') {
        const unavailable = loaded.textureCompression.find((part) => part.unavailable)?.unavailable;
        if (unavailable) problems.push(`textures not compressed: ${unavailable}`);
    }
    const materials = new Set<Material>();
    loaded.content.traverse((object) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh) return;
        for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) materials.add(material);
    });
    const textureCheck = checkTextures(materials, MODE);
    problems.push(...textureCheck.problems.slice(0, 10));
    // The textures still uncompressed: the ones the plan keeps (for GI, not an image, a size), nothing else.
    if (MODE !== 'uncompressed' && problems.length === 0) {
        const notPlanned = [...textureSlots(loaded.content)].filter(([texture, slots]) => !(texture instanceof CompressedTexture) && planTexture(texture, slots, MODE).action === 'compress');
        if (notPlanned.length > 0) problems.push(`${notPlanned.length} textures were planned for compression and are still uncompressed`);
    }
    log(`textures in materials: ${textureCheck.compressed} compressed, ${textureCheck.uncompressed} uncompressed (counted per slot)`);

    core.setExposureEv(defaultExposureEv(manifest));
    core.camera.position.set(...bookmark.position);
    core.camera.fov = bookmark.fovYDeg;
    core.camera.lookAt(...bookmark.target);
    core.camera.updateProjectionMatrix();
    core.camera.updateMatrixWorld(true);
    core.setLayout({ left: 'ddgi', right: null, giMode: 'on' });
    const ddgi = core.ddgi;
    if (!ddgi) throw new Error('DDGI was not created');
    const ddgiStartedAt = performance.now();
    while (!ddgi.running) {
        if (ddgi.failed) throw new Error(`DDGI failed: ${ddgi.failed}`);
        if (performance.now() - ddgiStartedAt > DDGI_START_TIMEOUT_MS) throw new Error(`DDGI did not start within ${DDGI_START_TIMEOUT_MS / 1000} s`);
        await core.renderThroughLoop();
        await core.pumpPrepare(40);
    }
    const ddgiStartMs = performance.now() - ddgiStartedAt;
    log(`DDGI running after ${(ddgiStartMs / 1000).toFixed(1)} s; drawing ${SETTLE_FRAMES} frames`);
    for (let frame = 0; frame < SETTLE_FRAMES; frame++) await core.renderThroughLoop();

    // Each frame drawn and waited for: wall time includes three's CPU recording and any wait on the GPU, as T2.17 read it.
    const frameMs: number[] = [];
    for (let frame = 0; frame < TIMED_FRAMES; frame++) {
        const startedAt = performance.now();
        await core.renderThroughLoop();
        await core.device.queue.onSubmittedWorkDone();
        frameMs.push(performance.now() - startedAt);
    }
    const frames = { p50: percentile(frameMs, 0.5), p95: percentile(frameMs, 0.95), largest: Math.max(...frameMs) };
    log(`GI-on frames, wall: p50 ${frames.p50.toFixed(1)} ms, p95 ${frames.p95.toFixed(1)} ms, largest ${frames.largest.toFixed(1)} ms`);

    // The picture of one more frame, for comparison between modes. Snapshot it right after the frame is recorded,
    // with no await in between: once the browser has shown it, a WebGPU canvas reads back blank.
    await core.renderThroughLoop();
    const snapshot = await createImageBitmap(canvas);
    // The runner's own screenshot then shows the picture, not the log over it.
    logElement.style.display = 'none';
    const copy = new OffscreenCanvas(snapshot.width, snapshot.height);
    copy.getContext('2d')!.drawImage(snapshot, 0, 0);
    const blob = await copy.convertToBlob({ type: 'image/png' });
    const picture = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.readAsDataURL(blob);
    });

    await core.device.queue.onSubmittedWorkDone();
    const allProblems = [...problems, ...pageErrors];
    reportGpuTest({
        status: allProblems.length === 0 ? 'pass' : 'fail',
        message: allProblems.length === 0 ? undefined : allProblems.join(' | '),
        data: {
            scene: SCENE,
            mode: MODE,
            maxTextureSize: MAX_TEXTURE_SIZE ?? null,
            camera: bookmark.name,
            compression: loaded.textureCompression,
            texturesInMaterials: { compressed: textureCheck.compressed, uncompressed: textureCheck.uncompressed },
            threeTexturesMb: core.renderer.info.memory.texturesSize / 1e6,
            loadMs,
            ddgiStartMs,
            frameWallMs: frames,
            picture,
        },
        deviceErrors: core.gpuErrors,
        startTime,
        adapter: {
            vendor: core.adapterInfo.vendor,
            architecture: core.adapterInfo.architecture,
            description: core.adapterInfo.description,
            device: core.adapterInfo.device,
            features: [...core.device.features].sort(),
            isFallbackAdapter: false,
        },
        logLines,
    });
} catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    reportGpuTest({ status: 'fail', message: [message, ...pageErrors].join(' | '), startTime, logLines });
}
