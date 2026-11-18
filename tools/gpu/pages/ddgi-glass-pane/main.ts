/**
 * T4.2 done-when 2: light through a glass pane. The thin-wall scene (one room
 * split by a 3 cm wall, a lamp on the west side only) with its partition
 * replaced by a partition *mesh* that has a 2 x 1.4 m window: the wall is
 * material slot 0 (opaque) and a 1 cm pane filling the window is slot 1, one
 * mesh, two slots. DDGI on the manifest's volume, every probe every frame,
 * for FRAMES frames, twice:
 *
 * - **see-through:** the pane's material has `blocksLight: false`, so probe
 *   rays and shadow rays go through it (triangle flags, `giMaterialPasses`);
 * - **opaque:** the same pane blocks, as a wall would.
 *
 * Measured with `ddgiSampleIrradiance` (FieldProbe) at a grid of points on the
 * floor of each side, facing up. Reported, no exact bar (done-when 2): the
 * dark side's mean against the lit side's, for both, and the dark side with
 * the pane see-through against opaque. The page fails only if the
 * see-through dark side isn't brighter than the opaque one.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { MaterialHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { buildProceduralParts, mergeScenes } from '../../../../scenes/src/build-scene.ts';
import { THIN_WALL_EMBED_M, THIN_WALL_ROOM_SIZE_M, THIN_WALL_THICKNESS_M } from '../../../../scenes/src/procedural/thin-wall.ts';
import { assert, runGpuTest } from '../../client.ts';
import { FieldProbe, loadManifest, releaseDeviceOnPageHide, runFrames, type SurfacePoint } from '../ddgi-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;
/** Frames every probe is updated, per variant; enough for a ratio (the leak page's 3000 are for a 1% bar). */
const FRAMES = Number(env.VITE_DDGI_FRAMES) > 0 ? Math.round(Number(env.VITE_DDGI_FRAMES)) : 1200;

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const IDLE_FRAME: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: { position: [0, 0, 0], view: IDENTITY, projection: IDENTITY, near: 0.1, far: 100 } };

/** The window in the partition (z across the room, y up), in metres. */
const WINDOW_Z: readonly [number, number] = [1.0, 3.0];
const WINDOW_Y: readonly [number, number] = [0.8, 2.2];
const PANE_THICKNESS_M = 0.01;

type Vec3 = [number, number, number];

/** A mesh being collected: boxes with outward faces, a material slot per triangle. */
class PartitionMesh {
    readonly positions: number[] = [];
    readonly normals: number[] = [];
    readonly indices: number[] = [];
    private readonly slots: number[] = [];

    /** An axis-aligned box, each face counter-clockwise seen from outside. */
    box(min: Vec3, max: Vec3, slot: number): void {
        const corner = (x: number, y: number, z: number): Vec3 => [x ? max[0] : min[0], y ? max[1] : min[1], z ? max[2] : min[2]];
        // Each face: its outward normal and four corners counter-clockwise seen from outside.
        const faces: [Vec3, Vec3[]][] = [
            [[1, 0, 0], [corner(1, 0, 1), corner(1, 0, 0), corner(1, 1, 0), corner(1, 1, 1)]],
            [[-1, 0, 0], [corner(0, 0, 0), corner(0, 0, 1), corner(0, 1, 1), corner(0, 1, 0)]],
            [[0, 1, 0], [corner(0, 1, 1), corner(1, 1, 1), corner(1, 1, 0), corner(0, 1, 0)]],
            [[0, -1, 0], [corner(0, 0, 0), corner(1, 0, 0), corner(1, 0, 1), corner(0, 0, 1)]],
            [[0, 0, 1], [corner(0, 0, 1), corner(1, 0, 1), corner(1, 1, 1), corner(0, 1, 1)]],
            [[0, 0, -1], [corner(1, 0, 0), corner(0, 0, 0), corner(0, 1, 0), corner(1, 1, 0)]],
        ];
        for (const [normal, corners] of faces) {
            const first = this.positions.length / 3;
            for (const point of corners) {
                this.positions.push(...point);
                this.normals.push(...normal);
            }
            this.indices.push(first, first + 1, first + 2, first, first + 2, first + 3);
            this.slots.push(slot, slot);
        }
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

/** The thin-wall partition with a window, and the pane filling it (slot 1). */
function partitionWithPane(): PartitionMesh {
    const [, sizeY, sizeZ] = THIN_WALL_ROOM_SIZE_M;
    const middleX = THIN_WALL_ROOM_SIZE_M[0] / 2;
    const x0 = middleX - THIN_WALL_THICKNESS_M / 2;
    const x1 = middleX + THIN_WALL_THICKNESS_M / 2;
    const bottom = -THIN_WALL_EMBED_M;
    const top = sizeY + THIN_WALL_EMBED_M;
    const south = -THIN_WALL_EMBED_M;
    const north = sizeZ + THIN_WALL_EMBED_M;
    const mesh = new PartitionMesh();
    // The wall around the window: below, above, and either side. They overlap nowhere and
    // meet the window's edges exactly; the pane sits inside the window.
    mesh.box([x0, bottom, south], [x1, WINDOW_Y[0], north], 0);
    mesh.box([x0, WINDOW_Y[1], south], [x1, top, north], 0);
    mesh.box([x0, WINDOW_Y[0], south], [x1, WINDOW_Y[1], WINDOW_Z[0]], 0);
    mesh.box([x0, WINDOW_Y[0], WINDOW_Z[1]], [x1, WINDOW_Y[1], north], 0);
    mesh.box([middleX - PANE_THICKNESS_M / 2, WINDOW_Y[0], WINDOW_Z[0]], [middleX + PANE_THICKNESS_M / 2, WINDOW_Y[1], WINDOW_Z[1]], 1);
    return mesh;
}

/** Floor points of one side, facing up, seen from above. */
function floorPoints(xFrom: number, xTo: number): SurfacePoint[] {
    const points: SurfacePoint[] = [];
    const sizeZ = THIN_WALL_ROOM_SIZE_M[2];
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

/** The thin-wall scene with the windowed partition; the pane's material see-through or opaque. Converged, then measured. */
async function runVariant(device: GPUDevice, paneSeeThrough: boolean, log: (line: string) => void) {
    const manifest = loadManifest('thin-wall');
    const data = mergeScenes(buildProceduralParts(manifest));
    const platform = new GIPlatformImpl(device, { label: `glass pane ${paneSeeThrough ? 'see-through' : 'opaque'}` });
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
        const partition = partitionWithPane();
        const partitionMesh = scene.createMesh({
            positions: Float32Array.from(partition.positions),
            normals: Float32Array.from(partition.normals),
            indices: Uint32Array.from(partition.indices),
            groups: partition.groups(),
            label: 'partition with a window',
        });
        const pane = scene.createMaterial({ albedo: [0.8, 0.8, 0.8], blocksLight: !paneSeeThrough, label: 'pane' });
        scene.createInstance({ mesh: partitionMesh, materials: [partitionMaterial, pane], transform: IDENTITY, label: 'partition' });
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
            const encoder = device.createCommandEncoder({ label: 'glass pane frame' });
            platform.encode(encoder, IDLE_FRAME);
            device.queue.submit([encoder.finish()]);
        };
        frame();
        await platform.rays.whenBuilt();
        frame();
        const field = ddgi.field;
        assert(field, 'no field after prepare');
        await runFrames(device, platform, FRAMES, field.grid.probeCount);
        // Read before the test's own probe kernel compiles, which counts as one.
        const recompiles = platform.stats.recompiles;

        const probe = await FieldProbe.create(device, platform);
        const middleX = THIN_WALL_ROOM_SIZE_M[0] / 2;
        const lit = meanOf(await probe.sample(field, floorPoints(0.2, middleX - 0.2)));
        const dark = meanOf(await probe.sample(field, floorPoints(middleX + 0.2, THIN_WALL_ROOM_SIZE_M[0] - 0.2)));
        const flagRewrites = platform.rays.triangleFlagRewrites;
        log(`pane ${paneSeeThrough ? 'see-through' : 'opaque'}: lit-side floor E ${lit.toFixed(5)}, dark-side floor E ${dark.toFixed(6)}, dark / lit ${(dark / lit).toFixed(5)}`);
        return { litMean: lit, darkMean: dark, darkOverLit: dark / lit, flagRewrites, recompiles };
    } finally {
        platform.dispose();
    }
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const seeThrough = await runVariant(device, true, log);
        const opaque = await runVariant(device, false, log);
        const darkRatio = seeThrough.darkMean / opaque.darkMean;
        log(`dark side, pane see-through against opaque: ${darkRatio.toFixed(2)}x`);
        assert(seeThrough.darkMean > opaque.darkMean, `the dark side is no brighter with the pane see-through (${seeThrough.darkMean} against ${opaque.darkMean})`);
        return { frames: FRAMES, window: { z: WINDOW_Z, y: WINDOW_Y }, seeThrough, opaque, darkSeeThroughOverOpaque: darkRatio };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
