/**
 * T4.2 done-when 2 (second half): light through an alpha wall. T4.4's
 * `alpha-checker` scene is not on `m4` yet, so the case is built here: the
 * thin-wall scene (one room split by a 3 cm wall, a lamp on the west side
 * only) with its partition replaced by a partition *mesh* that has a large
 * window (3 x 2.2 m), filled by a single quad (slot 1) of one of these
 * materials:
 *
 * - **open:** `blocksLight: false`, the same wall fully open;
 * - **checker cutout:** a 16 x 16 checkerboard of clear and opaque texels
 *   (one texel = 1/16 of the window), `alphaMode: 'cutout'`: 50% open;
 * - **stochastic:** `alphaMode: 'stochastic'`, opacity 0.5, no texture: a
 *   ray passes with probability 0.5.
 *
 * The quad is black (albedo 0), so its opaque parts reflect nothing: the
 * check is about how much light goes *through* the window, not about light
 * the checker's lit face sends back into the lit room. What then reaches
 * the dark side is light that crossed the window once (exactly half of it
 * for both 50% walls) plus light that crossed three times or more (an
 * eighth or less of it), which the window's small solid angle makes small.
 *
 * DDGI on the manifest's volume, every probe every frame, FRAMES frames per
 * variant; measured with `ddgiSampleIrradiance` (FieldProbe) at a grid of
 * floor points on the dark side, facing up. The bar: the dark side's mean
 * for each 50% wall is 50% ± 3% of the open wall's.
 */

import type { FrameInput } from '../../../../packages/core/src/platform/contract/frame.ts';
import type { MaterialHandle } from '../../../../packages/core/src/platform/contract/handles.ts';
import type { MaterialDescriptor } from '../../../../packages/core/src/platform/contract/scene.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { checkerImage, CHECKER_SIZE } from '../../../../packages/core/src/platform/trace/oracle-fixtures.ts';
import { DDGITechnique } from '../../../../packages/core/src/techniques/ddgi/ddgi-technique.ts';
import { buildProceduralParts, mergeScenes } from '../../../../scenes/src/build-scene.ts';
import { THIN_WALL_EMBED_M, THIN_WALL_ROOM_SIZE_M, THIN_WALL_THICKNESS_M } from '../../../../scenes/src/procedural/thin-wall.ts';
import { assert, runGpuTest } from '../../client.ts';
import { FieldProbe, loadManifest, releaseDeviceOnPageHide, runFrames, type SurfacePoint } from '../ddgi-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;
/** Frames every probe is updated, per variant; enough for a ratio (the leak page's 3000 are for a 1% bar). */
const FRAMES = Number(env.VITE_DDGI_FRAMES) > 0 ? Math.round(Number(env.VITE_DDGI_FRAMES)) : 1200;
/** Done-when 2: each 50% wall lets through 50% ± 3% of the open wall's light. */
const EXPECTED_SHARE = 0.5;
const SHARE_TOLERANCE = 0.03;

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const IDLE_FRAME: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: { position: [0, 0, 0], view: IDENTITY, projection: IDENTITY, near: 0.1, far: 100 } };

/** The window in the partition (z across the room, y up), in metres. */
const WINDOW_Z: readonly [number, number] = [0.5, 3.5];
const WINDOW_Y: readonly [number, number] = [0.3, 2.5];

type Vec3 = [number, number, number];

/** A mesh being collected: boxes with outward faces and a quad, a material slot per triangle, uvs for the quad. */
class PartitionMesh {
    readonly positions: number[] = [];
    readonly normals: number[] = [];
    readonly uvs: number[] = [];
    readonly indices: number[] = [];
    private readonly slots: number[] = [];

    /** An axis-aligned box, each face counter-clockwise seen from outside. Its uvs are 0: its material has no texture. */
    box(min: Vec3, max: Vec3, slot: number): void {
        const corner = (x: number, y: number, z: number): Vec3 => [x ? max[0] : min[0], y ? max[1] : min[1], z ? max[2] : min[2]];
        const faces: [Vec3, Vec3[]][] = [
            [[1, 0, 0], [corner(1, 0, 1), corner(1, 0, 0), corner(1, 1, 0), corner(1, 1, 1)]],
            [[-1, 0, 0], [corner(0, 0, 0), corner(0, 0, 1), corner(0, 1, 1), corner(0, 1, 0)]],
            [[0, 1, 0], [corner(0, 1, 1), corner(1, 1, 1), corner(1, 1, 0), corner(0, 1, 0)]],
            [[0, -1, 0], [corner(0, 0, 0), corner(1, 0, 0), corner(1, 0, 1), corner(0, 0, 1)]],
            [[0, 0, 1], [corner(0, 0, 1), corner(1, 0, 1), corner(1, 1, 1), corner(0, 1, 1)]],
            [[0, 0, -1], [corner(1, 0, 0), corner(0, 0, 0), corner(0, 1, 0), corner(1, 1, 0)]],
        ];
        for (const [normal, corners] of faces) this.face(normal, corners, [0, 0, 0, 0, 0, 0, 0, 0], slot);
    }

    /** The window's quad in the plane x = `x`, its texture upright seen from the lit (west) side: u along +z, v down. */
    windowQuad(x: number, slot: number): void {
        const [z0, z1] = WINDOW_Z;
        const [y0, y1] = WINDOW_Y;
        this.face([-1, 0, 0], [[x, y0, z1], [x, y0, z0], [x, y1, z0], [x, y1, z1]], [1, 1, 0, 1, 0, 0, 1, 0], slot);
    }

    private face(normal: Vec3, corners: Vec3[], uvs: number[], slot: number): void {
        const first = this.positions.length / 3;
        for (const point of corners) {
            this.positions.push(...point);
            this.normals.push(...normal);
        }
        this.uvs.push(...uvs);
        this.indices.push(first, first + 1, first + 2, first, first + 2, first + 3);
        this.slots.push(slot, slot);
    }

    /** Runs of triangles with one slot, as `MeshDescriptor.groups`. */
    groups(): { start: number; count: number; materialSlot: number }[] {
        const runs: { start: number; count: number; materialSlot: number }[] = [];
        this.slots.forEach((slot, triangle) => {
            const last = runs[runs.length - 1];
            if (last && last.materialSlot === slot) last.count += 3;
            else runs.push({ start: triangle * 3, count: 3, materialSlot: slot });
        });
        return runs;
    }
}

/** The thin-wall partition with a window (slot 0), and the window's quad (slot 1). */
function partitionWithWindow(): PartitionMesh {
    const [, sizeY, sizeZ] = THIN_WALL_ROOM_SIZE_M;
    const middleX = THIN_WALL_ROOM_SIZE_M[0] / 2;
    const x0 = middleX - THIN_WALL_THICKNESS_M / 2;
    const x1 = middleX + THIN_WALL_THICKNESS_M / 2;
    const bottom = -THIN_WALL_EMBED_M;
    const top = sizeY + THIN_WALL_EMBED_M;
    const south = -THIN_WALL_EMBED_M;
    const north = sizeZ + THIN_WALL_EMBED_M;
    const mesh = new PartitionMesh();
    // The wall around the window: below, above, and either side, meeting the window's edges exactly.
    mesh.box([x0, bottom, south], [x1, WINDOW_Y[0], north], 0);
    mesh.box([x0, WINDOW_Y[1], south], [x1, top, north], 0);
    mesh.box([x0, WINDOW_Y[0], south], [x1, WINDOW_Y[1], WINDOW_Z[0]], 0);
    mesh.box([x0, WINDOW_Y[0], WINDOW_Z[1]], [x1, WINDOW_Y[1], north], 0);
    mesh.windowQuad(middleX, 1);
    return mesh;
}

/** Floor points of the dark (east) side, facing up, seen from above. */
function darkFloorPoints(): SurfacePoint[] {
    const points: SurfacePoint[] = [];
    const [sizeX, , sizeZ] = THIN_WALL_ROOM_SIZE_M;
    const xFrom = sizeX / 2 + 0.2;
    const xTo = sizeX - 0.2;
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            const x = xFrom + ((i + 0.5) / 8) * (xTo - xFrom);
            const z = ((j + 0.5) / 8) * sizeZ;
            points.push({ position: [x, 0.01, z], normal: [0, 1, 0], viewDirection: [0, -1, 0] });
        }
    }
    return points;
}

function meanOf(values: readonly [number, number, number][]): number {
    let sum = 0;
    for (const value of values) sum += (value[0] + value[1] + value[2]) / 3;
    return sum / values.length;
}

type Variant = 'open' | 'checker cutout' | 'stochastic 0.5';

/** The window quad's material for a variant; black, so its opaque parts reflect nothing. */
function windowMaterial(variant: Variant, checker: Parameters<GIPlatformImpl['scene']['createMaterial']>[0]['albedoTexture']): MaterialDescriptor {
    const black: MaterialDescriptor = { albedo: [0, 0, 0], label: `window (${variant})` };
    if (variant === 'open') return { ...black, blocksLight: false };
    if (variant === 'checker cutout') return { ...black, albedoTexture: checker, alphaMode: 'cutout', alphaCutoff: 0.5 };
    return { ...black, alphaMode: 'stochastic', opacity: 0.5 };
}

/** The thin-wall scene with the windowed partition and the variant's window. Converged, then measured. */
async function runVariant(device: GPUDevice, variant: Variant, log: (line: string) => void) {
    const manifest = loadManifest('thin-wall');
    const data = mergeScenes(buildProceduralParts(manifest));
    const platform = new GIPlatformImpl(device, { label: `alpha wall ${variant}` });
    try {
        const scene = platform.scene;
        const meshes = data.meshes.map((mesh) => scene.createMesh({ positions: mesh.positions, indices: mesh.indices, normals: mesh.normals, label: mesh.name }));
        const materials: MaterialHandle[] = data.materials.map((material) => scene.createMaterial({ albedo: material.albedo, emissive: material.emissive, label: material.name }));
        let partitionMaterial: MaterialHandle | null = null;
        for (const instance of data.instances) {
            if (instance.name === 'thin-wall') {
                partitionMaterial = materials[instance.material];
                continue; // replaced by the windowed partition below
            }
            scene.createInstance({ mesh: meshes[instance.mesh], materials: [materials[instance.material]], transform: instance.transform, label: instance.name });
        }
        assert(partitionMaterial !== null, 'the thin-wall scene has no instance named thin-wall');
        const checkerPixels = checkerImage();
        const checker = scene.createTexture({ source: { data: checkerPixels.data, width: checkerPixels.width, height: checkerPixels.height }, srgb: true, label: 'checker' });
        const partition = partitionWithWindow();
        const partitionMesh = scene.createMesh({
            positions: Float32Array.from(partition.positions),
            normals: Float32Array.from(partition.normals),
            uvs: Float32Array.from(partition.uvs),
            indices: Uint32Array.from(partition.indices),
            groups: partition.groups(),
            label: 'partition with a window',
        });
        const window = scene.createMaterial(windowMaterial(variant, checker));
        scene.createInstance({ mesh: partitionMesh, materials: [partitionMaterial, window], transform: IDENTITY, label: 'partition' });
        // The lamp, without the range window (as the leak page lights it).
        const lamp = manifest.lights[0];
        assert(lamp.type === 'point', 'the thin-wall lamp is not a point light');
        scene.createLight({ type: 'point', color: lamp.color, intensity: lamp.intensity, position: lamp.position, range: 0, castsShadows: true, label: lamp.label });
        scene.setEnvironment({ type: 'none' });

        const volume = { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing };
        const ddgi = platform.addTechnique(new DDGITechnique({ warmUpPasses: 0, volume }));
        platform.setOutput(ddgi);
        await platform.prepare();
        const frame = (): void => {
            const encoder = device.createCommandEncoder({ label: 'alpha wall frame' });
            platform.encode(encoder, IDLE_FRAME);
            device.queue.submit([encoder.finish()]);
        };
        frame();
        await platform.rays.whenBuilt();
        frame();
        const textures = platform.sceneCopy.textureCounts();
        assert(textures.built === 1, `the checker texture isn't built: ${JSON.stringify(textures)}`);
        const field = ddgi.field;
        assert(field, 'no field after prepare');
        await runFrames(device, platform, FRAMES, field.grid.probeCount);
        // Read before the test's own probe kernel compiles, which counts as one.
        const recompiles = platform.stats.recompiles;

        const probe = await FieldProbe.create(device, platform);
        const dark = meanOf(await probe.sample(field, darkFloorPoints()));
        log(`${variant}: dark-side floor E ${dark.toFixed(6)}`);
        return { darkMean: dark, flagRewrites: platform.rays.triangleFlagRewrites, recompiles };
    } finally {
        platform.dispose();
    }
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const open = await runVariant(device, 'open', log);
        const checker = await runVariant(device, 'checker cutout', log);
        const stochastic = await runVariant(device, 'stochastic 0.5', log);
        const checkerShare = checker.darkMean / open.darkMean;
        const stochasticShare = stochastic.darkMean / open.darkMean;
        log(`dark side against the open wall: checker cutout ${(checkerShare * 100).toFixed(2)}%, stochastic 0.5 ${(stochasticShare * 100).toFixed(2)}% (bar ${EXPECTED_SHARE * 100}% ± ${SHARE_TOLERANCE * 100})`);
        const failures: string[] = [];
        if (Math.abs(checkerShare - EXPECTED_SHARE) > SHARE_TOLERANCE) failures.push(`the checker cutout lets through ${(checkerShare * 100).toFixed(2)}% of the open wall's light`);
        if (Math.abs(stochasticShare - EXPECTED_SHARE) > SHARE_TOLERANCE) failures.push(`the stochastic wall lets through ${(stochasticShare * 100).toFixed(2)}% of the open wall's light`);
        assert(failures.length === 0, failures.join('\n'));
        return { frames: FRAMES, window: { z: WINDOW_Z, y: WINDOW_Y, checkerTexels: CHECKER_SIZE }, open, checker, stochastic, checkerShare, stochasticShare };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
