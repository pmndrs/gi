/**
 * RC4.1: the radiance cache's temporal screen probes on the GPU, checked
 * against CPU twins of Capsaicin's rules (packages/ddgi-core/src/techniques/
 * radiance-cache/probe-cache.ts, capsaicin-math.ts). 1280 × 720, the
 * engine-free page screen (radiance-cache-probes/screen.ts, with velocity),
 * on a device held at the platform's 8 storage buffers per stage.
 *
 * `sky-plane` (a floor under a constant white sky):
 * 1. **Still camera reprojects every probe to itself:** from a reset, each
 *    frame's tiles that weren't re-spawned keep last frame's seed, SH (bit for
 *    bit) and atlas cells (within the ×1e4 quantization and a half step); no
 *    tile loses its probe; once every tile has one, no tile is empty.
 * 2. **Ray guiding follows the CDF:** a history of our own (one bright cell
 *    of luminance 10, the others 1) is written into the previous atlas; the
 *    next frame's rays, binned by cell, against the CDF Capsaicin builds from
 *    it (luminance × the cell's cosine): a chi-square over the 64 cells.
 * 3. **A whole probe of sky rays in one cell doesn't wrap the sum
 *    (register D48):** a history that is black but for one cell sends every
 *    ray there; they all miss, and that cell's hit distance must read the
 *    capped 6,710.9 m (wrapped, 64 misses would read 5,106 m).
 *
 * `pillar` (the floor and a 2.5 m box, the camera panning past it):
 * 4. **A pan marks the disoccluded tiles empty, and the patch moves probes
 *    into them:** each pan frame's empty-tile list against a CPU twin of
 *    `ReprojectScreenProbes`' search (velocity, previous mask, plane and normal
 *    tests, in f64; tiles with a test within 1e-3 of its threshold are
 *    counted, not failed); every probe in the list is its spawn tile's seed or
 *    a patched seed in an empty tile at the frame's 8-pixel jitter.
 * 5. **The probe cache keeps and evicts as Capsaicin's rules say:** each
 *    frame's per-tile lists (`CountScreenProbes`, the scan,
 *    `ScatterScreenProbes`) against the CPU twin from last frame's LRU and
 *    cached positions, and the next frame's LRU (`ReorderScreenProbes`) against
 *    the CPU twin from this frame's LRU, flags and MRU, exactly.
 * 6. **An evicted probe off screen and back is found:** the probes evicted at
 *    the pillar's edges during the pan are not listed while the camera looks
 *    away, stay allocated, and when it turns back, new probes find them
 *    (`updated`), each on its own plane within a cell's size.
 *
 * `cornell-box` and `sponza` (first bookmark, still):
 * 7. **Convergence and flicker (information, T1.12's measure):** per-pixel
 *    frame-to-frame change of the interpolated irradiance E, mean and p95,
 *    over a still minute's last 32 frames, against DDGI's thresholds (mean ≤
 *    0.1%, p95 ≤ 0.5%), on E rather than albedo / π × E (the page screen has
 *    no albedo); the change falls from the first frames to the last; and the
 *    image mean with history against the mean of no-history frames (the
 *    screen probes reset each time; the world cache warm).
 *
 * Correctness only, no timing (DECISIONS 2026-11-09). Run through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-temporal`.
 */

import type { FrameCamera, FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { invertMat4, multiplyMat4 } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/camera.ts';
import {
    calculateHaltonSequence,
    floatToHalfBits,
    INVALID_ID,
    mapToHemiOctahedron,
    probeCellOf,
    spawnJitter,
    spawnScreenProbes,
    unpackHalf3,
    unpackSeed,
} from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { CACHED_TILE_UNALLOCATED, listCachedTiles, reorderCachedTiles } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/probe-cache.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { screenProbeCellSize } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/screen-irradiance.ts';
import { PROBE_CELLS, PROBE_SIZE, PROBE_SPAWN_TILE_SIZE } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { COUNTER_INDEX, PROBE_RAY_BYTES, PROBE_RAYS_HEADER_BYTES, PROBE_SPAWN_SEEDS_OFFSET } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/screen-probes.ts';
import { LARGEST_QUANTIZED_HIT_DISTANCE } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/capsaicin.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { describeScene, loadManifest, readBuffer } from '../ddgi-shared/harness.ts';
import { requestBudgetDevice } from '../radiance-cache-probes/budget-device.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';
import { bookmarkCamera, describeSkyPlane, describeSponza } from '../radiance-cache-probes/scenes.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
const TILES_X = Math.ceil(WIDTH / PROBE_SIZE);
const TILES_Y = Math.ceil(HEIGHT / PROBE_SIZE);
const TILE_COUNT = TILES_X * TILES_Y;

/** Still frames after a reset in check 1. */
const STILL_FRAMES = 12;
/** From this frame after a reset every tile has had a probe (Halton visits all four 8 × 8 tiles of a spawn tile by frame 7). */
const FULL_FROM_FRAME = 8;
/** The cell the guiding checks make bright (row 3, column 2: neither the pole nor the rim). */
const BRIGHT_CELL = 2 + 3 * PROBE_SIZE;
/** The synthetic history's hit distance: far, so the neighbours' parallax keeps every cell in its cell. */
const SYNTHETIC_HIT_DISTANCE = 5000;
/** Chi-square over 64 cells (63 degrees of freedom): the mean plus six standard deviations. */
const CHI_SQUARE_LIMIT = 63 + 6 * Math.sqrt(2 * 63);
/** The capped hit distance (register D48), and how far a half may sit from it (the half step there is 4). */
const CAPPED_HIT_DISTANCE = LARGEST_QUANTIZED_HIT_DISTANCE / 1e4;
const CAPPED_HIT_DISTANCE_TOLERANCE = 4.01;
/** The pillar scene: warm-up, pan, look-away and return frames. */
const PILLAR_WARM_FRAMES = 10;
const PAN_FRAMES = 24;
const PAN_STEP = 0.06;
const AWAY_FRAMES = 4;
const RETURN_FRAMES = 6;
/** How near its threshold a CPU twin's test may be and still be counted rather than failed. */
const THRESHOLD_MARGIN = 1e-3;
/** Check 7: warm-up frames (the world cache's 16-sample cap twice), measured frames, no-history samples. */
const CONVERGE_WARM_FRAMES = 32;
const CONVERGE_MEASURE_FRAMES = 32;
const NO_HISTORY_SAMPLES = 8;

type Vector = [number, number, number];

interface Run {
    readonly name: string;
    readonly technique: RadianceCacheTechnique;
    readonly platform: GIPlatformImpl;
    readonly screen: PageScreen;
    camera: PageCamera;
    /** Runs one frame at `camera`: the host's encode, the page's screen, the technique's screen stage. */
    frame(): Promise<void>;
    dispose(): void;
}

function frameInput(camera: PageCamera): FrameInput {
    return { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
}

async function startRun(device: GPUDevice, name: string, camera: PageCamera, describe: (platform: GIPlatformImpl) => Promise<(() => void) | void>): Promise<Run> {
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-temporal ${name}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique());
    platform.setOutput(technique);
    await platform.prepare();
    const beforeFrame = (await describe(platform)) ?? (() => {});
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const run: Run = {
        name,
        technique,
        platform,
        screen,
        camera,
        async frame() {
            beforeFrame();
            const encoder = device.createCommandEncoder({ label: `radiance-cache-temporal ${name}` });
            platform.encode(encoder, frameInput(run.camera));
            screen.encode(encoder, run.camera);
            platform.encodeScreen(encoder, screen.screen);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
        },
        dispose() {
            screen.dispose();
            platform.dispose();
        },
    };
    // A frame places the scene; then the mesh trees.
    await run.frame();
    await platform.rays.whenBuilt();
    return run;
}

// ---- Readbacks ------------------------------------------------------------------------------

async function readWords(device: GPUDevice, buffer: GPUBuffer, words: number): Promise<Uint32Array> {
    return new Uint32Array(await readBuffer(device, buffer, Math.max(words, 1) * 4));
}

async function readU32Texture(device: GPUDevice, texture: GPUTexture, width: number, height: number, channels = 1): Promise<Uint32Array> {
    const bytes = await readTexture(device, texture, 4 * channels, width, height);
    return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

async function readF32Texture(device: GPUDevice, texture: GPUTexture, width: number, height: number): Promise<Float32Array> {
    const bytes = await readTexture(device, texture, 4, width, height);
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

async function readHalf4Texture(device: GPUDevice, texture: GPUTexture, width: number, height: number): Promise<Float32Array> {
    return halves(await readTexture(device, texture, 8, width, height));
}

/** The technique's state after a frame: what the next frame reads as history, and what this one wrote. */
interface Snapshot {
    readonly frameIndex: number;
    readonly mask: Uint32Array;
    readonly atlas: Float32Array;
    readonly sh: Uint32Array;
    /** The probe list: the count, then the seeds. */
    readonly list: Uint32Array;
    readonly counters: Uint32Array;
    readonly words: Uint32Array;
    readonly nextLRUHalf: 0 | 1;
    readonly listCounts: Uint32Array;
    readonly listIndex: Uint32Array;
    readonly lruFlags: Uint32Array;
    /** Cached probes' entries (rgba32uint: position bits, packed normal). */
    readonly cachedTileIndex: Uint32Array;
    readonly previousMask: Uint32Array;
    readonly previousDepth: Float32Array;
    readonly previousNormal: Float32Array;
}

async function snapshot(device: GPUDevice, run: Run): Promise<Snapshot> {
    const probes = run.technique.probes;
    const history = run.technique.probeHistory;
    if (!probes || !history) throw new Error(`${run.name}: no probes after encodeScreen`);
    const layout = history.probeCacheLayout;
    return {
        frameIndex: probes.frameIndex,
        mask: await readU32Texture(device, probes.probeMask, TILES_X, TILES_Y),
        atlas: await readHalf4Texture(device, probes.probeAtlas, probes.layout.atlasSize[0], probes.layout.atlasSize[1]),
        sh: await readU32Texture(device, probes.probeSH, TILES_X * 3, TILES_Y * 3, 2),
        list: await readWords(device, probes.probeSpawn, PROBE_SPAWN_SEEDS_OFFSET / 4 + layout.spawnTiles),
        counters: await readWords(device, history.counters, 16),
        words: await readWords(device, history.probeCacheWords, layout.words),
        nextLRUHalf: history.nextLRUHalf,
        listCounts: await readWords(device, history.listCounts, layout.slots),
        listIndex: await readWords(device, history.listIndex, layout.slots),
        lruFlags: await readWords(device, history.lruFlags, layout.slots + 1),
        cachedTileIndex: await readU32Texture(device, history.cachedTileIndex, TILES_X, TILES_Y, 4),
        previousMask: await readU32Texture(device, history.previousProbeMask, TILES_X, TILES_Y),
        previousDepth: await readF32Texture(device, history.previousDepth, WIDTH, HEIGHT),
        previousNormal: await readHalf4Texture(device, history.previousNormal, WIDTH, HEIGHT),
    };
}

/** The page screen as the technique read it this frame. */
interface ScreenRead {
    readonly depth: Float32Array;
    readonly normal: Float32Array;
    readonly velocity: Float32Array;
}

async function readScreen(device: GPUDevice, run: Run): Promise<ScreenRead> {
    const screen = run.screen.screen;
    if (!screen.velocity) throw new Error('the page screen has no velocity');
    return {
        depth: await readF32Texture(device, screen.depth, WIDTH, HEIGHT),
        normal: await readHalf4Texture(device, screen.normal, WIDTH, HEIGHT),
        velocity: await readHalf4Texture(device, screen.velocity, WIDTH, HEIGHT),
    };
}

// ---- f64 camera math ------------------------------------------------------------------------

/** World position of a pixel at a depth through a camera (transformPointProjection with the inverse view-projection). */
function unprojector(camera: FrameCamera): (pixel: readonly [number, number], depth: number) => Vector {
    const inverse = invertMat4(multiplyMat4(camera.projection, camera.view));
    if (!inverse) throw new Error('camera without an inverse');
    return ([x, y], depth) => {
        const ndc = [((x + 0.5) / WIDTH) * 2 - 1, (1 - (y + 0.5) / HEIGHT) * 2 - 1, depth, 1];
        const h = [0, 1, 2, 3].map((row) => inverse[row] * ndc[0] + inverse[4 + row] * ndc[1] + inverse[8 + row] * ndc[2] + inverse[12 + row] * ndc[3]);
        return [h[0] / h[3], h[1] / h[3], h[2] / h[3]];
    };
}

function projector(camera: FrameCamera): (world: readonly [number, number, number]) => [number, number, number] {
    const viewProjection = multiplyMat4(camera.projection, camera.view);
    return ([x, y, z]) => {
        const h = [0, 1, 2, 3].map((row) => viewProjection[row] * x + viewProjection[4 + row] * y + viewProjection[8 + row] * z + viewProjection[12 + row]);
        return [h[0] / h[3], h[1] / h[3], h[2] / h[3]];
    };
}

const dot = (a: readonly number[], b: readonly number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const subtract = (a: readonly number[], b: readonly number[]): Vector => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const normalize = (a: readonly number[]): Vector => {
    const length = Math.hypot(a[0], a[1], a[2]);
    return [a[0] / length, a[1] / length, a[2] / length];
};

function normalAt(normals: Float32Array, x: number, y: number): Vector {
    const base = (y * WIDTH + x) * 4;
    return normalize([normals[base], normals[base + 1], normals[base + 2]]);
}

function f32BitsToFloat(bits: number): number {
    return new Float32Array(new Uint32Array([bits]).buffer)[0];
}

function cachedPosition(entries: Uint32Array, cachedTile: number): Vector {
    return [f32BitsToFloat(entries[cachedTile * 4]), f32BitsToFloat(entries[cachedTile * 4 + 1]), f32BitsToFloat(entries[cachedTile * 4 + 2])];
}

function unpackNormal(packed: number): Vector {
    const field = (shift: number): number => ((packed << (22 - shift)) >> 22) / 511;
    return normalize([field(0), field(10), field(20)]);
}

// ---- Check 1: still camera ------------------------------------------------------------------

interface StillResult {
    readonly failures: string[];
    readonly summary: Record<string, unknown>;
}

async function checkStill(device: GPUDevice, run: Run): Promise<StillResult> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`still: ${message}`);
    };
    run.technique.resetHistory();
    await run.frame();
    let before = await snapshot(device, run);
    let keptTiles = 0;
    let movedSeeds = 0;
    let lostProbes = 0;
    let shMismatches = 0;
    let atlasMismatches = 0;
    let worstAtlas = 0;
    const emptyTiles: number[] = [];
    for (let frame = 1; frame <= STILL_FRAMES; frame++) {
        await run.frame();
        const after = await snapshot(device, run);
        emptyTiles.push(after.counters[COUNTER_INDEX.emptyTileCount]);
        if (frame >= FULL_FROM_FRAME && after.counters[COUNTER_INDEX.emptyTileCount] !== 0) fail(`frame ${frame}: ${after.counters[COUNTER_INDEX.emptyTileCount]} empty tiles with every tile holding a probe`);
        // Tiles this frame blended again are new probes; the rest only went through reprojection.
        const listed = new Set<number>();
        for (let probe = 0; probe < after.list[0]; probe++) {
            const [x, y] = unpackSeed(after.list[1 + probe]);
            listed.add(Math.floor(y / PROBE_SIZE) * TILES_X + Math.floor(x / PROBE_SIZE));
        }
        for (let tile = 0; tile < TILE_COUNT; tile++) {
            if (listed.has(tile)) continue;
            const old = before.mask[tile];
            const now = after.mask[tile];
            if (old === INVALID_ID) {
                if (now !== INVALID_ID) fail(`frame ${frame} tile ${tile}: a probe appeared without being spawned`);
                continue;
            }
            if (now === INVALID_ID) {
                lostProbes++;
                fail(`frame ${frame} tile ${tile}: the still camera lost its probe`);
                continue;
            }
            keptTiles++;
            if (now !== old) {
                movedSeeds++;
                fail(`frame ${frame} tile ${tile}: seed ${unpackSeed(old).join(',')} reprojected to ${unpackSeed(now).join(',')}`);
                continue;
            }
            const tx = tile % TILES_X;
            const ty = Math.floor(tile / TILES_X);
            for (let coefficient = 0; coefficient < 9; coefficient++) {
                const texel = ((ty * 3 + Math.floor(coefficient / 3)) * TILES_X * 3 + tx * 3 + (coefficient % 3)) * 2;
                if (after.sh[texel] !== before.sh[texel] || after.sh[texel + 1] !== before.sh[texel + 1]) shMismatches++;
            }
            for (let cell = 0; cell < PROBE_CELLS; cell++) {
                const texel = ((ty * PROBE_SIZE + Math.floor(cell / PROBE_SIZE)) * TILES_X * PROBE_SIZE + tx * PROBE_SIZE + (cell % PROBE_SIZE)) * 4;
                for (let channel = 0; channel < 4; channel++) {
                    // A hit distance past the cap comes back as the cap (register D48): the reprojected
                    // length is quantized like any sample.
                    const a = channel === 3 ? Math.min(before.atlas[texel + channel], CAPPED_HIT_DISTANCE) : before.atlas[texel + channel];
                    const b = after.atlas[texel + channel];
                    // ×1e4 quantization (5e-5) and two half steps.
                    const allowed = 1e-4 + Math.abs(a) * 2 ** -9 + (channel === 3 ? CAPPED_HIT_DISTANCE_TOLERANCE : 0);
                    const error = Math.abs(a - b);
                    worstAtlas = Math.max(worstAtlas, error / Math.max(Math.abs(a), 1e-6));
                    if (error > allowed) {
                        atlasMismatches++;
                        if (atlasMismatches <= 3) fail(`frame ${frame} tile ${tx},${ty} cell ${cell} channel ${channel}: ${a} became ${b}`);
                    }
                }
            }
        }
        before = after;
    }
    if (shMismatches > 0) fail(`${shMismatches} SH coefficients of reprojected probes changed`);
    if (atlasMismatches > 0) fail(`${atlasMismatches} atlas values of reprojected probes moved by more than the quantization`);
    if (keptTiles === 0) fail('no tile was reprojected');
    return { failures, summary: { keptTiles, movedSeeds, lostProbes, shMismatches, atlasMismatches, worstAtlasRelativeChange: worstAtlas, emptyTilesPerFrame: emptyTiles } };
}

// ---- Checks 2 and 3: ray guiding and the capped sum -----------------------------------------

/** Writes a history into the previous atlas: every probe tile's cells as `radiance(cell)`, all at the synthetic hit distance. */
function writeSyntheticHistory(device: GPUDevice, run: Run, radiance: (cell: number) => number): void {
    const history = run.technique.probeHistory;
    if (!history) throw new Error('no history');
    const width = TILES_X * PROBE_SIZE;
    const height = TILES_Y * PROBE_SIZE;
    const texels = new Uint16Array(width * height * 4);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const cell = (x % PROBE_SIZE) + (y % PROBE_SIZE) * PROBE_SIZE;
            const value = floatToHalfBits(radiance(cell));
            texels.set([value, value, value, floatToHalfBits(SYNTHETIC_HIT_DISTANCE)], (y * width + x) * 4);
        }
    }
    device.queue.writeTexture({ texture: history.previousProbeAtlas }, texels, { bytesPerRow: width * 8 }, [width, height]);
}

/** Each probe's 64 rays, binned by the cell BlendScreenProbes puts them in. */
async function rayCells(device: GPUDevice, run: Run, screen: ScreenRead): Promise<{ list: Uint32Array; cells: Int32Array }> {
    const probes = run.technique.probes;
    if (!probes) throw new Error('no probes');
    const list = await readWords(device, probes.probeSpawn, PROBE_SPAWN_SEEDS_OFFSET / 4 + probes.layout.maxProbeSpawnCount);
    const count = list[0];
    const rays = new Uint32Array(await readBuffer(device, probes.probeRays, PROBE_RAYS_HEADER_BYTES + Math.max(count, 1) * PROBE_CELLS * PROBE_RAY_BYTES));
    const cells = new Int32Array(count * PROBE_CELLS);
    for (let probe = 0; probe < count; probe++) {
        const [x, y] = unpackSeed(list[1 + probe]);
        const normal = normalAt(screen.normal, x, y);
        for (let cell = 0; cell < PROBE_CELLS; cell++) {
            const base = (PROBE_RAYS_HEADER_BYTES + (probe * PROBE_CELLS + cell) * PROBE_RAY_BYTES) / 4;
            cells[probe * PROBE_CELLS + cell] = probeCellOf(unpackHalf3([rays[base + 4], rays[base + 5]]), normal);
        }
    }
    return { list, cells };
}

async function checkGuiding(device: GPUDevice, run: Run): Promise<StillResult> {
    const failures: string[] = [];
    // 2. One bright cell: rays in proportion to luminance x the cell's cosine (gi1.comp:944-947).
    const pattern = (cell: number): number => (cell === BRIGHT_CELL ? 10 : 1);
    writeSyntheticHistory(device, run, pattern);
    await run.frame();
    const screen = await readScreen(device, run);
    const { cells } = await rayCells(device, run, screen);
    const weights = Array.from({ length: PROBE_CELLS }, (_, cell) => pattern(cell) * mapToHemiOctahedron(((cell % PROBE_SIZE) + 0.5) / PROBE_SIZE, (Math.floor(cell / PROBE_SIZE) + 0.5) / PROBE_SIZE)[2]);
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    const observed = new Array<number>(PROBE_CELLS).fill(0);
    for (const cell of cells) observed[cell]++;
    const rays = cells.length;
    let chiSquare = 0;
    for (let cell = 0; cell < PROBE_CELLS; cell++) {
        const expected = (rays * weights[cell]) / total;
        chiSquare += (observed[cell] - expected) ** 2 / expected;
    }
    if (rays === 0) failures.push('guiding: no rays');
    if (!(chiSquare < CHI_SQUARE_LIMIT)) failures.push(`guiding: chi-square ${chiSquare.toFixed(1)} over 63 degrees of freedom (limit ${CHI_SQUARE_LIMIT.toFixed(1)})`);
    const brightShare = observed[BRIGHT_CELL] / Math.max(rays, 1);
    const expectedBrightShare = weights[BRIGHT_CELL] / total;

    // 3. All of a probe's rays into one sky cell: the capped hit distance (register D48).
    writeSyntheticHistory(device, run, (cell) => (cell === BRIGHT_CELL ? 1000 : 0));
    await run.frame();
    const skyScreen = await readScreen(device, run);
    const sky = await rayCells(device, run, skyScreen);
    const probes = run.technique.probes;
    if (!probes) throw new Error('no probes');
    const atlas = await readHalf4Texture(device, probes.probeAtlas, probes.layout.atlasSize[0], probes.layout.atlasSize[1]);
    let fullProbes = 0;
    let checkedProbes = 0;
    let worstDistance = 0;
    let worstRadiance = 0;
    for (let probe = 0; probe < sky.list[0]; probe++) {
        let inBright = 0;
        for (let cell = 0; cell < PROBE_CELLS; cell++) if (sky.cells[probe * PROBE_CELLS + cell] === BRIGHT_CELL) inBright++;
        if (inBright === PROBE_CELLS) fullProbes++;
        if (inBright < PROBE_CELLS - 4) continue;
        checkedProbes++;
        const [x, y] = unpackSeed(sky.list[1 + probe]);
        const texel = ((Math.floor(y / PROBE_SIZE) * PROBE_SIZE + Math.floor(BRIGHT_CELL / PROBE_SIZE)) * TILES_X * PROBE_SIZE + Math.floor(x / PROBE_SIZE) * PROBE_SIZE + (BRIGHT_CELL % PROBE_SIZE)) * 4;
        worstDistance = Math.max(worstDistance, Math.abs(atlas[texel + 3] - CAPPED_HIT_DISTANCE));
        worstRadiance = Math.max(worstRadiance, Math.abs(atlas[texel] - 1));
    }
    if (checkedProbes === 0) failures.push('sky cell: no probe sent its rays into one cell');
    if (!(worstDistance <= CAPPED_HIT_DISTANCE_TOLERANCE)) failures.push(`sky cell: a hit distance is ${worstDistance.toFixed(2)} m off the capped ${CAPPED_HIT_DISTANCE.toFixed(3)} m (a wrapped sum reads 5,106 m)`);
    if (!(worstRadiance <= 0.01)) failures.push(`sky cell: radiance ${worstRadiance.toFixed(4)} off the sky's 1`);
    return {
        failures,
        summary: {
            guiding: { rays, chiSquare, limit: CHI_SQUARE_LIMIT, brightShare, expectedBrightShare, observedOutsideBright: rays - observed[BRIGHT_CELL] },
            skyCell: { probes: sky.list[0], probesWithAll64InOneCell: fullProbes, checkedProbes, worstHitDistanceError: worstDistance, worstRadianceError: worstRadiance, cappedHitDistance: CAPPED_HIT_DISTANCE },
        },
    };
}

// ---- Checks 4-6: the pillar ------------------------------------------------------------------

async function describePillar(platform: GIPlatformImpl): Promise<void> {
    await describeSkyPlane(platform);
    const scene = platform.scene;
    // A 0.6 x 2.5 x 0.6 m box standing on the floor 5 m ahead (each face its own four vertices, normals out).
    const [cx, cz, half, height] = [0, -5, 0.3, 2.5];
    const positions: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];
    const face = (normal: Vector, corners: Vector[]): void => {
        const base = positions.length / 3;
        for (const corner of corners) {
            positions.push(...corner);
            normals.push(...normal);
        }
        indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    };
    const [x0, x1, z0, z1] = [cx - half, cx + half, cz - half, cz + half];
    face([0, 0, 1], [[x0, 0, z1], [x1, 0, z1], [x1, height, z1], [x0, height, z1]]);
    face([0, 0, -1], [[x1, 0, z0], [x0, 0, z0], [x0, height, z0], [x1, height, z0]]);
    face([1, 0, 0], [[x1, 0, z1], [x1, 0, z0], [x1, height, z0], [x1, height, z1]]);
    face([-1, 0, 0], [[x0, 0, z0], [x0, 0, z1], [x0, height, z1], [x0, height, z0]]);
    face([0, 1, 0], [[x0, height, z1], [x1, height, z1], [x1, height, z0], [x0, height, z0]]);
    const mesh = scene.createMesh({ positions: new Float32Array(positions), normals: new Float32Array(normals), indices: new Uint32Array(indices), label: 'pillar' });
    const material = scene.createMaterial({ albedo: [0.7, 0.3, 0.2], label: 'pillar' });
    scene.createInstance({ mesh, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], label: 'pillar' });
}

/** The tiles ReprojectScreenProbes should find no previous probe for, and those it can't decide in f64 against f32. */
function expectedEmptyTiles(
    screen: ScreenRead,
    before: Snapshot,
    camera: FrameCamera,
    previousCamera: FrameCamera,
    frameIndex: number,
): { empty: Set<number>; undecided: Set<number>; found: number; nonSky: number } {
    const current = unprojector(camera);
    const previous = unprojector(previousCamera);
    const cellSize = screenProbeCellSize(camera, WIDTH, HEIGHT);
    const isSky = (x: number, y: number): boolean => screen.depth[y * WIDTH + x] === 1;
    const [jitterX, jitterY] = spawnJitter(frameIndex);
    const spawnQuadrant = [Math.floor(jitterX / PROBE_SIZE), Math.floor(jitterY / PROBE_SIZE)];
    const empty = new Set<number>();
    const undecided = new Set<number>();
    let found = 0;
    let nonSky = 0;
    for (let tile = 0; tile < TILE_COUNT; tile++) {
        const tx = tile % TILES_X;
        const ty = Math.floor(tile / TILES_X);
        let tileFound = false;
        let tileUndecided = false;
        for (let py = ty * PROBE_SIZE; py < Math.min((ty + 1) * PROBE_SIZE, HEIGHT); py++) {
            for (let px = tx * PROBE_SIZE; px < Math.min((tx + 1) * PROBE_SIZE, WIDTH); px++) {
                if (isSky(px, py)) continue;
                const velocityBase = (py * WIDTH + px) * 4;
                const previousU = (px + 0.5) / WIDTH - 0.5 * screen.velocity[velocityBase];
                const previousV = (py + 0.5) / HEIGHT + 0.5 * screen.velocity[velocityBase + 1];
                const scaledX = previousU * WIDTH;
                const scaledY = previousV * HEIGHT;
                if (Math.abs(scaledX - Math.round(scaledX)) < THRESHOLD_MARGIN || Math.abs(scaledY - Math.round(scaledY)) < THRESHOLD_MARGIN) tileUndecided = true;
                const previousX = Math.trunc(scaledX);
                const previousY = Math.trunc(scaledY);
                if (previousX < 0 || previousY < 0 || previousX >= WIDTH || previousY >= HEIGHT) continue;
                const probeMask = before.previousMask[Math.floor(previousY / PROBE_SIZE) * TILES_X + Math.floor(previousX / PROBE_SIZE)];
                if (probeMask === INVALID_ID) continue;
                const [sx, sy] = unpackSeed(probeMask);
                const world = current([px, py], screen.depth[py * WIDTH + px]);
                const normal = normalAt(screen.normal, px, py);
                const probePosition = previous([sx, sy], before.previousDepth[sy * WIDTH + sx]);
                const probeNormal = normalAt(before.previousNormal, sx, sy);
                const cell = Math.hypot(...subtract(world, camera.position)) * cellSize;
                const plane = Math.abs(dot(subtract(probePosition, world), normal));
                const facing = dot(normal, probeNormal);
                if (Math.abs(plane - cell) < THRESHOLD_MARGIN * cell || Math.abs(facing - 0.95) < THRESHOLD_MARGIN) tileUndecided = true;
                if (plane < cell && facing > 0.95) tileFound = true;
            }
        }
        const originSky = isSky(tx * PROBE_SIZE, ty * PROBE_SIZE);
        if (!originSky) nonSky++;
        if (tileFound && !originSky) found++;
        if (tileUndecided && !tileFound) undecided.add(tile);
        const spawnedHere = tx % 2 === spawnQuadrant[0] && ty % 2 === spawnQuadrant[1];
        if (!tileFound && !originSky && !spawnedHere) empty.add(tile);
    }
    return { empty, undecided, found, nonSky };
}

async function checkPillar(device: GPUDevice, run: Run, start: PageCamera): Promise<StillResult> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 16) failures.push(`pillar: ${message}`);
    };
    run.camera = start;
    run.technique.resetHistory();
    for (let frame = 0; frame < PILLAR_WARM_FRAMES; frame++) await run.frame();
    const history = run.technique.probeHistory;
    if (!history) throw new Error('no history');
    const cacheLayout = history.probeCacheLayout;
    let before = await snapshot(device, run);
    let previousCamera = frameCamera(run.camera, WIDTH, HEIGHT);

    const evictedTiles = new Set<number>();
    let emptyMismatches = 0;
    let emptyUndecided = 0;
    let reuseShareMin = 1;
    let patchedProbes = 0;
    let badListEntries = 0;
    let listMismatches = 0;
    let listUndecided = 0;
    let reorderMismatches = 0;
    let evictions = 0;
    let panFramesWithEmpty = 0;
    const perPhase: Record<string, Array<Record<string, number>>> = { pan: [], away: [], back: [] };
    let awayListedEvicted = 0;
    let evictedStillAllocated = 0;
    let foundEvicted = 0;
    let foundInvalid = 0;
    let foundTotal = 0;

    const cameras: Array<{ phase: 'pan' | 'away' | 'back'; camera: PageCamera }> = [];
    for (let step = 1; step <= PAN_FRAMES; step++) {
        cameras.push({ phase: 'pan', camera: { ...start, position: [start.position[0] + step * PAN_STEP, start.position[1], start.position[2]], target: [start.target[0] + step * PAN_STEP, start.target[1], start.target[2]] } });
    }
    for (let step = 0; step < AWAY_FRAMES; step++) cameras.push({ phase: 'away', camera: { ...start, target: [start.position[0], start.position[1] - 0.5, start.position[2] + 5] } });
    for (let step = 0; step < RETURN_FRAMES; step++) cameras.push({ phase: 'back', camera: start });

    for (const { phase, camera: pageCamera } of cameras) {
        run.camera = pageCamera;
        await run.frame();
        const after = await snapshot(device, run);
        const screen = await readScreen(device, run);
        const camera = frameCamera(pageCamera, WIDTH, HEIGHT);
        const currentHalf = before.nextLRUHalf;
        const lruBefore = before.words.subarray(cacheLayout.lru[currentHalf], cacheLayout.lru[currentHalf] + cacheLayout.slots);
        const lruNow = after.words.subarray(cacheLayout.lru[currentHalf], cacheLayout.lru[currentHalf] + cacheLayout.slots);
        const lruNext = after.words.subarray(cacheLayout.lru[after.nextLRUHalf], cacheLayout.lru[after.nextLRUHalf] + cacheLayout.slots);
        const counters = after.counters;
        const probeCount = after.list[0];
        const record: Record<string, number> = { probes: probeCount, empty: counters[COUNTER_INDEX.emptyTileCount], override: counters[COUNTER_INDEX.overrideTileCount], newSlots: counters[COUNTER_INDEX.cachedTileLRUCount], listed: counters[COUNTER_INDEX.cachedTileListElementCount] };

        // 4. The empty tiles against the CPU twin of the reprojection's search.
        const expected = expectedEmptyTiles(screen, before, camera, previousCamera, after.frameIndex);
        const gpuEmpty = new Set(Array.from(after.words.subarray(cacheLayout.emptyTile, cacheLayout.emptyTile + counters[COUNTER_INDEX.emptyTileCount])));
        let frameMismatches = 0;
        for (const tile of new Set([...gpuEmpty, ...expected.empty])) {
            if (gpuEmpty.has(tile) === expected.empty.has(tile)) continue;
            if (expected.undecided.has(tile)) {
                emptyUndecided++;
                continue;
            }
            frameMismatches++;
            if (frameMismatches <= 2) fail(`${phase} frame ${after.frameIndex} tile ${tile}: the GPU ${gpuEmpty.has(tile) ? 'lists' : "doesn't list"} it empty, the CPU twin ${expected.empty.has(tile) ? 'does' : "doesn't"}`);
        }
        emptyMismatches += frameMismatches;
        record.cpuEmpty = expected.empty.size;
        record.reuseShare = expected.found / Math.max(expected.nonSky, 1);
        if (phase === 'pan') {
            reuseShareMin = Math.min(reuseShareMin, record.reuseShare);
            if (gpuEmpty.size > 0) panFramesWithEmpty++;
        }

        // 4. The probe list: each entry its spawn tile's seed, or a patched seed in an empty tile.
        const spawn = spawnScreenProbes(WIDTH, HEIGHT, after.frameIndex, (x, y) => screen.depth[y * WIDTH + x] === 1).filter((tile) => tile.spawned);
        const [haltonX, haltonY] = calculateHaltonSequence(after.frameIndex);
        const patchJitter = [Math.min(Math.floor(haltonX * PROBE_SIZE), PROBE_SIZE - 1), Math.min(Math.floor(haltonY * PROBE_SIZE), PROBE_SIZE - 1)];
        if (spawn.length !== probeCount) fail(`${phase} frame ${after.frameIndex}: ${probeCount} probes listed, the spawn rule gives ${spawn.length}`);
        let framePatched = 0;
        for (let probe = 0; probe < Math.min(probeCount, spawn.length); probe++) {
            const [x, y] = unpackSeed(after.list[1 + probe]);
            if (x === spawn[probe].seed[0] && y === spawn[probe].seed[1]) continue;
            const tile = Math.floor(y / PROBE_SIZE) * TILES_X + Math.floor(x / PROBE_SIZE);
            const patchedSeed = [Math.min((tile % TILES_X) * PROBE_SIZE + patchJitter[0], WIDTH - 1), Math.min(Math.floor(tile / TILES_X) * PROBE_SIZE + patchJitter[1], HEIGHT - 1)];
            if (gpuEmpty.has(tile) && x === patchedSeed[0] && y === patchedSeed[1] && after.mask[tile] === after.list[1 + probe]) {
                framePatched++;
            } else {
                badListEntries++;
                fail(`${phase} frame ${after.frameIndex} probe ${probe}: seed ${x},${y} is neither its spawn tile's nor a patched empty tile's`);
            }
        }
        if (framePatched > counters[COUNTER_INDEX.overrideTileCount]) fail(`${phase} frame ${after.frameIndex}: ${framePatched} patched probes, only ${counters[COUNTER_INDEX.overrideTileCount]} override tiles`);
        // PatchScreenProbes skips an empty tile whose seed at this jitter is sky (gi1.comp:633-636).
        const patchable = [...gpuEmpty].filter((tile) => {
            const x = Math.min((tile % TILES_X) * PROBE_SIZE + patchJitter[0], WIDTH - 1);
            const y = Math.min(Math.floor(tile / TILES_X) * PROBE_SIZE + patchJitter[1], HEIGHT - 1);
            return screen.depth[y * WIDTH + x] !== 1;
        }).length;
        record.patchable = patchable;
        if (patchable > 0 && counters[COUNTER_INDEX.overrideTileCount] > 0 && framePatched === 0) fail(`${phase} frame ${after.frameIndex}: ${patchable} patchable empty tiles and override tiles, but nothing patched`);
        patchedProbes += framePatched;
        record.patched = framePatched;

        // 5. The per-tile lists against the CPU twin of the count, scan and scatter.
        const lists = listCachedTiles(lruBefore, (cachedTile) => cachedPosition(before.cachedTileIndex, cachedTile), projector(camera), [TILES_X, TILES_Y]);
        const undecidedSlots = new Set(lists.undecided);
        const gpuList = after.words.subarray(cacheLayout.list, cacheLayout.list + cacheLayout.slots);
        for (let tile = 0; tile < TILE_COUNT; tile++) {
            const gpuSlots = Array.from(gpuList.subarray(after.listIndex[tile], after.listIndex[tile] + after.listCounts[tile])).sort((a, b) => a - b);
            const cpuSlots = lists.slotsByTile.get(tile) ?? [];
            if (gpuSlots.length === cpuSlots.length && gpuSlots.every((slot, index) => slot === cpuSlots[index])) continue;
            if ([...gpuSlots, ...cpuSlots].some((slot) => undecidedSlots.has(slot))) {
                listUndecided++;
                continue;
            }
            listMismatches++;
            if (listMismatches <= 2) fail(`${phase} frame ${after.frameIndex} tile ${tile}: the GPU lists slots [${gpuSlots.join(',')}], the CPU twin [${cpuSlots.join(',')}]`);
        }
        if (Math.abs(counters[COUNTER_INDEX.cachedTileListElementCount] - lists.elementCount) > undecidedSlots.size) fail(`${phase} frame ${after.frameIndex}: ${counters[COUNTER_INDEX.cachedTileListElementCount]} listed on the GPU, ${lists.elementCount} on the CPU`);

        // 5. The next frame's LRU against the CPU twin of the scan and reorder.
        const mru = after.words.subarray(cacheLayout.mru, cacheLayout.mru + cacheLayout.slots);
        const reordered = reorderCachedTiles(lruNow, after.lruFlags, mru, cacheLayout.slots);
        for (let slot = 0; slot < cacheLayout.slots; slot++) {
            if (reordered[slot] !== lruNext[slot]) {
                reorderMismatches++;
                if (reorderMismatches <= 2) fail(`${phase} frame ${after.frameIndex} LRU slot ${slot}: GPU ${lruNext[slot]}, CPU twin ${reordered[slot]}`);
            }
        }

        // 6. Evictions and finds, by cached tile (the slot of a cached probe moves as the LRU is reordered).
        const lruCount = counters[COUNTER_INDEX.cachedTileLRUCount];
        let frameFound = 0;
        for (let probe = 0; probe < probeCount; probe++) {
            const evicted = after.words[cacheLayout.spawnProbe + 2 * probe];
            const updated = after.words[cacheLayout.spawnProbe + 2 * probe + 1];
            if (evicted !== INVALID_ID && evicted >= lruCount) {
                const slot = (evicted & ~CACHED_TILE_UNALLOCATED) >>> 0;
                if (phase === 'pan') {
                    evictedTiles.add(lruNow[slot] & ~CACHED_TILE_UNALLOCATED);
                    evictions++;
                }
            }
            if (updated !== INVALID_ID && updated >= lruCount) {
                frameFound++;
                const cachedTile = lruNow[updated] & ~CACHED_TILE_UNALLOCATED;
                const [x, y] = unpackSeed(after.list[1 + probe]);
                const world = unprojector(camera)([x, y], screen.depth[y * WIDTH + x]);
                const normal = normalAt(screen.normal, x, y);
                const position = cachedPosition(before.cachedTileIndex, cachedTile);
                const cachedNormal = unpackNormal(before.cachedTileIndex[cachedTile * 4 + 3]);
                const cell = Math.hypot(...subtract(world, camera.position)) * screenProbeCellSize(camera, WIDTH, HEIGHT);
                const distance = Math.hypot(...subtract(position, world));
                if (!(distance < cell * (1 + THRESHOLD_MARGIN)) || !(dot(normal, cachedNormal) > 0.95 - THRESHOLD_MARGIN)) {
                    foundInvalid++;
                    fail(`${phase} frame ${after.frameIndex} probe ${probe}: found cached tile ${cachedTile} ${distance.toFixed(4)} m away (cell ${cell.toFixed(4)} m), normals ${dot(normal, cachedNormal).toFixed(3)}`);
                }
                if (phase === 'back' && evictedTiles.has(cachedTile)) foundEvicted++;
            }
        }
        foundTotal += frameFound;
        record.found = frameFound;
        if (phase === 'away') {
            // Off screen, the pan's evicted probes are listed nowhere.
            for (let tile = 0; tile < TILE_COUNT; tile++) {
                for (const slot of gpuList.subarray(after.listIndex[tile], after.listIndex[tile] + after.listCounts[tile])) {
                    if (evictedTiles.has(lruNow[slot] & ~CACHED_TILE_UNALLOCATED)) awayListedEvicted++;
                }
            }
        }
        perPhase[phase].push(record);
        before = after;
        previousCamera = camera;
    }
    // Still held by the cache after the look-away: allocated entries for the pan's cached tiles.
    const lruAfter = before.words.subarray(cacheLayout.lru[before.nextLRUHalf], cacheLayout.lru[before.nextLRUHalf] + cacheLayout.slots);
    for (const entry of lruAfter) if ((entry & CACHED_TILE_UNALLOCATED) === 0 && evictedTiles.has(entry)) evictedStillAllocated++;

    if (emptyMismatches > 0) fail(`${emptyMismatches} empty-tile decisions differ from the CPU twin`);
    if (panFramesWithEmpty === 0) fail('the pan disoccluded no tile');
    if (patchedProbes === 0) fail('no empty tile was patched');
    if (!(reuseShareMin >= 0.8)) fail(`a pan frame reused history in only ${(reuseShareMin * 100).toFixed(1)}% of the tiles`);
    if (listMismatches > 0) fail(`${listMismatches} probe tiles' cache lists differ from the CPU twin`);
    if (reorderMismatches > 0) fail(`${reorderMismatches} LRU slots differ from the CPU twin`);
    if (evictions === 0) fail('the pan evicted no probe into the cache');
    if (awayListedEvicted > 0) fail(`${awayListedEvicted} evicted probes listed while the camera looked away`);
    if (evictedStillAllocated === 0) fail("none of the pan's cached probes is still in the cache after the look-away");
    if (foundEvicted === 0) fail("no probe found one of the pan's cached probes on the way back");
    if (foundInvalid > 0) fail(`${foundInvalid} probes found a cached probe off their plane or facing away`);
    return {
        failures,
        summary: {
            emptyMismatches,
            emptyUndecided,
            reuseShareMin,
            patchedProbes,
            badListEntries,
            listMismatches,
            listUndecided,
            reorderMismatches,
            evictions,
            evictedCachedTiles: evictedTiles.size,
            awayListedEvicted,
            evictedStillAllocated,
            foundTotal,
            foundEvicted,
            foundInvalid,
            frames: perPhase,
        },
    };
}

// ---- Check 7: convergence and flicker -------------------------------------------------------

function percentile(sorted: Float32Array, share: number): number {
    if (sorted.length === 0) return 0;
    return sorted[Math.min(sorted.length - 1, Math.floor(share * sorted.length))];
}

async function readIrradiance(device: GPUDevice, run: Run): Promise<Float32Array> {
    const probes = run.technique.probes;
    if (!probes) throw new Error('no probes');
    return readHalf4Texture(device, probes.irradiance, WIDTH, HEIGHT);
}

/** T1.12's measure between two frames of E, over surface pixels: mean and p95 of |ΔE| (rgb mean) over the image's mean E. */
function frameChange(previous: Float32Array, current: Float32Array, surface: Uint32Array): { mean: number; p95: number; imageMean: number; meanAbsolute: number } {
    let sum = 0;
    for (const pixel of surface) sum += (current[pixel * 4] + current[pixel * 4 + 1] + current[pixel * 4 + 2]) / 3;
    const imageMean = sum / Math.max(surface.length, 1);
    const changes = new Float32Array(surface.length);
    let total = 0;
    let absolute = 0;
    surface.forEach((pixel, index) => {
        const change = (Math.abs(current[pixel * 4] - previous[pixel * 4]) + Math.abs(current[pixel * 4 + 1] - previous[pixel * 4 + 1]) + Math.abs(current[pixel * 4 + 2] - previous[pixel * 4 + 2])) / 3;
        absolute += change;
        changes[index] = change / imageMean;
        total += changes[index];
    });
    changes.sort();
    return { mean: total / Math.max(surface.length, 1), p95: percentile(changes, 0.95), imageMean, meanAbsolute: absolute / Math.max(surface.length, 1) };
}

function imageMean(image: Float32Array, surface: Uint32Array): number {
    let sum = 0;
    for (const pixel of surface) sum += (image[pixel * 4] + image[pixel * 4 + 1] + image[pixel * 4 + 2]) / 3;
    return sum / Math.max(surface.length, 1);
}

async function checkConvergence(device: GPUDevice, run: Run): Promise<StillResult> {
    const failures: string[] = [];
    run.technique.resetHistory();
    const screen = await readScreenAfter(device, run);
    const surface = Uint32Array.from({ length: WIDTH * HEIGHT }, (_, pixel) => pixel).filter((pixel) => screen.depth[pixel] !== 1);
    let previous = await readIrradiance(device, run);
    const early: number[] = [];
    const late: Array<ReturnType<typeof frameChange>> = [];
    const lateMeans: number[] = [];
    for (let frame = 1; frame < CONVERGE_WARM_FRAMES + CONVERGE_MEASURE_FRAMES; frame++) {
        await run.frame();
        const current = await readIrradiance(device, run);
        const change = frameChange(previous, current, surface);
        if (frame <= 4) early.push(change.meanAbsolute);
        if (frame >= CONVERGE_WARM_FRAMES) {
            late.push(change);
            lateMeans.push(change.imageMean);
        }
        previous = current;
    }
    // No-history frames with the world cache warm: the screen probes forget theirs each time.
    const noHistoryMeans: number[] = [];
    for (let sample = 0; sample < NO_HISTORY_SAMPLES; sample++) {
        run.technique.resetScreenProbes();
        await run.frame();
        noHistoryMeans.push(imageMean(await readIrradiance(device, run), surface));
    }
    const mean = (values: number[]): number => values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
    const deviation = (values: number[]): number => Math.sqrt(mean(values.map((value) => (value - mean(values)) ** 2)));
    const earlyChange = mean(early);
    const lateChange = mean(late.map((change) => change.meanAbsolute));
    if (!(lateChange < earlyChange)) failures.push(`${run.name}: E changes ${lateChange.toExponential(3)} a frame once settled, ${earlyChange.toExponential(3)} in the first frames`);
    const withHistory = mean(lateMeans);
    const noHistory = mean(noHistoryMeans);
    return {
        failures,
        summary: {
            flickerMean: mean(late.map((change) => change.mean)),
            flickerP95: mean(late.map((change) => change.p95)),
            ddgiThresholds: { mean: 0.001, p95: 0.005 },
            earlyMeanAbsoluteChange: earlyChange,
            lateMeanAbsoluteChange: lateChange,
            withHistoryMean: withHistory,
            noHistoryMean: noHistory,
            noHistorySd: deviation(noHistoryMeans),
            withOverNoHistory: withHistory / noHistory,
            surfacePixels: surface.length,
        },
    };
}

/** One frame (the first after a reset), then the page screen as it read it. */
async function readScreenAfter(device: GPUDevice, run: Run): Promise<ScreenRead> {
    await run.frame();
    return readScreen(device, run);
}

// ---- The page -------------------------------------------------------------------------------

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    const { device, adapterStorageBuffersPerStage } = await requestBudgetDevice(adapter);
    window.addEventListener('pagehide', () => device.destroy(), { once: true });
    deviceErrors = watchDevice(device);
    const results: Record<string, unknown> = {};
    const failures: string[] = [];

    const skyCamera: PageCamera = { position: [0, 3, 0], target: [0, 1, -10], fovYDeg: 60, near: 0.1, far: 1000 };
    const sky = await startRun(device, 'sky-plane', skyCamera, describeSkyPlane);
    try {
        const still = await checkStill(device, sky);
        failures.push(...still.failures);
        const guiding = await checkGuiding(device, sky);
        failures.push(...guiding.failures);
        // Under a constant sky the converged E against pi L, for information (the gate is on the scenes below).
        const convergence = await checkConvergence(device, sky);
        failures.push(...convergence.failures);
        results['sky-plane'] = { still: still.summary, ...guiding.summary, convergence: { ...convergence.summary, withHistoryOverPiL: Number(convergence.summary.withHistoryMean) / Math.PI, noHistoryOverPiL: Number(convergence.summary.noHistoryMean) / Math.PI } };
    } finally {
        sky.dispose();
        await device.queue.onSubmittedWorkDone();
    }

    const pillarCamera: PageCamera = { position: [-1.2, 1.6, 0], target: [-0.2, 0.6, -5], fovYDeg: 60, near: 0.1, far: 1000 };
    const pillar = await startRun(device, 'pillar', pillarCamera, describePillar);
    try {
        const checks = await checkPillar(device, pillar, pillarCamera);
        failures.push(...checks.failures);
        results.pillar = checks.summary;
    } finally {
        pillar.dispose();
        await device.queue.onSubmittedWorkDone();
    }

    const cornell = loadManifest('cornell-box');
    const sponza = loadManifest('sponza');
    const stillScenes: { name: string; camera: PageCamera; describe: (platform: GIPlatformImpl) => Promise<(() => void) | void> }[] = [
        { name: 'cornell-box', camera: bookmarkCamera(cornell, 100), describe: async (platform) => void describeScene(platform, cornell) },
        { name: 'sponza', camera: bookmarkCamera(sponza, 1000), describe: (platform) => describeSponza(platform, device, sponza) },
    ];
    for (const scene of stillScenes) {
        const run = await startRun(device, scene.name, scene.camera, scene.describe);
        try {
            const checks = await checkConvergence(device, run);
            failures.push(...checks.failures);
            results[scene.name] = checks.summary;
        } finally {
            run.dispose();
            await device.queue.onSubmittedWorkDone();
        }
    }

    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data: { width: WIDTH, height: HEIGHT, adapterStorageBuffersPerStage, spawnTileSize: PROBE_SPAWN_TILE_SIZE, scenes: results },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}

