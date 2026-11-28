/**
 * M4, T4.3 review: `flipY` textures through the three adapter give the same
 * albedo at ray hits as the same image stored the other way up.
 *
 * Two 1 m `PlaneGeometry` quads in one three scene, mirrored by the adapter's
 * SceneSync, each a `MeshStandardMaterial` (white, metalness 0) with a 256²
 * linear `DataTexture`:
 *
 * - **flipped:** an asymmetric image P (red ramps along x, green along the
 *   rows, blue marks the top-left quarter) with `flipY = true`. three uploads
 *   it upside down, and the adapter turns v upside down in the uv mapping;
 * - **pre-flipped:** P with its rows reversed and `flipY = false`, which
 *   three draws the same way up and the adapter passes unchanged.
 *
 * Rays hit a 16 × 16 grid on each quad head-on (`giSurfaceAtCone`, spread
 * 0, so mip 0). Passes when every pair of hits has the same albedo within
 * 0.01, and the picture isn't symmetric (the top rows of the flipped quad
 * read green near 0, as three draws P's first row at the top), so a missing
 * flip would show.
 */

import { DataTexture, Mesh, MeshStandardMaterial, NoColorSpace, PlaneGeometry, Scene } from 'three';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { SceneSync } from '../../../../packages/three/src/scene-sync.ts';
import { assert, runGpuTest } from '../../client.ts';
import { frame, ShadeKernel } from '../hit-shading-shared/harness.ts';

const SIZE = 256;
const GRID = 16;
/** Where the second quad sits along x. */
const SECOND_X = 3;
const RAY_DISTANCE = 2;
const TOLERANCE = 0.01;

/** The asymmetric image P, rows top first: red = x ramp, green = row ramp, blue = 255 in the top-left quarter. */
function image(): Uint8Array {
    const pixels = new Uint8Array(SIZE * SIZE * 4);
    for (let row = 0; row < SIZE; row++) {
        for (let column = 0; column < SIZE; column++) {
            const topLeft = row < SIZE / 2 && column < SIZE / 2;
            pixels.set([column, row, topLeft ? 255 : 0, 255], (row * SIZE + column) * 4);
        }
    }
    return pixels;
}

/** `pixels` with its rows in reverse order. */
function rowsReversed(pixels: Uint8Array): Uint8Array {
    const reversed = new Uint8Array(pixels.length);
    const rowBytes = SIZE * 4;
    for (let row = 0; row < SIZE; row++) reversed.set(pixels.subarray(row * rowBytes, (row + 1) * rowBytes), (SIZE - 1 - row) * rowBytes);
    return reversed;
}

function quad(pixels: Uint8Array, flipY: boolean, x: number): Mesh {
    const texture = new DataTexture(pixels, SIZE, SIZE);
    texture.flipY = flipY;
    texture.colorSpace = NoColorSpace;
    texture.needsUpdate = true;
    const mesh = new Mesh(new PlaneGeometry(1, 1), new MeshStandardMaterial({ map: texture, metalness: 0 }));
    mesh.position.x = x;
    return mesh;
}

const KERNEL = /* wgsl */ `
@group(1) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> outputs: array<vec4f>;

// Per ray, in: (origin, spread), (direction, 0). Out: (albedo, t).
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= arrayLength(&inputs) / 2u) {
        return;
    }
    let originAndSpread = inputs[2u * index];
    let ray = GiRay(originAndSpread.xyz, 0.0, normalize(inputs[2u * index + 1u].xyz), 100.0);
    let hit = giTraceClosest(ray);
    var albedo = vec3f(-1.0);
    if (hit.t >= 0.0) {
        albedo = giSurfaceAtCone(ray, hit, originAndSpread.w).albedo;
    }
    outputs[index] = vec4f(albedo, hit.t);
}
`;

await runGpuTest(async ({ device, log }) => {
    const platform = new GIPlatformImpl(device, { label: 'three-m4-flip-y' });
    await platform.prepare();
    const warnings: string[] = [];
    const sync = new SceneSync(platform.scene, { device, warn: (message) => warnings.push(message) });
    try {
        const pixels = image();
        const scene = new Scene();
        scene.add(quad(pixels, true, 0), quad(rowsReversed(pixels), false, SECOND_X));
        scene.updateMatrixWorld(true);
        sync.update(scene);
        frame(device, platform);
        await platform.rays.whenBuilt();
        frame(device, platform);
        await device.queue.onSubmittedWorkDone();

        // The same grid point on each quad, head-on from +z; rows top first.
        const inputs = new Float32Array(2 * GRID * GRID * 8);
        for (const [quadIndex, centreX] of [0, SECOND_X].entries()) {
            for (let row = 0; row < GRID; row++) {
                for (let column = 0; column < GRID; column++) {
                    const x = centreX + ((column + 0.5) / GRID - 0.5) * 0.98;
                    const y = (0.5 - (row + 0.5) / GRID) * 0.98;
                    inputs.set([x, y, RAY_DISTANCE, 0, 0, 0, -1, 0], ((quadIndex * GRID + row) * GRID + column) * 8);
                }
            }
        }
        const kernel = await ShadeKernel.create(device, platform, 'three-m4-flip-y', KERNEL);
        const output = new Float32Array(await kernel.run(inputs, 2 * GRID * GRID * 16, 2 * GRID * GRID));

        let worst = 0;
        let topGreen = 0;
        let bottomGreen = 0;
        for (let point = 0; point < GRID * GRID; point++) {
            const flipped = output.subarray(point * 4, point * 4 + 4);
            const preFlipped = output.subarray((GRID * GRID + point) * 4, (GRID * GRID + point) * 4 + 4);
            assert(flipped[3] >= 0 && preFlipped[3] >= 0, `grid point ${point} missed a quad`);
            for (let channel = 0; channel < 3; channel++) worst = Math.max(worst, Math.abs(flipped[channel] - preFlipped[channel]));
            const row = Math.floor(point / GRID);
            if (row === 0) topGreen += flipped[1] / GRID;
            if (row === GRID - 1) bottomGreen += flipped[1] / GRID;
        }
        log(`flipY = true against pre-flipped pixels: largest albedo difference ${worst.toFixed(5)} over ${GRID * GRID} hits; flipped quad's green, top row ${topGreen.toFixed(3)}, bottom row ${bottomGreen.toFixed(3)}`);
        assert(warnings.length === 0, `the adapter warned: ${warnings.join(' | ')}`);
        assert(worst <= TOLERANCE, `the flipY texture's albedo is ${worst} from the pre-flipped one's (limit ${TOLERANCE})`);
        // P's first row (green 0) is at the top, as three draws it; without the flip it would be at the bottom.
        assert(topGreen < 0.1 && bottomGreen > 0.85, `the flipped quad is upside down (green: top ${topGreen}, bottom ${bottomGreen})`);
        return { hits: GRID * GRID, largestDifference: worst, topRowGreen: topGreen, bottomRowGreen: bottomGreen };
    } finally {
        sync.dispose();
        platform.dispose();
    }
});
