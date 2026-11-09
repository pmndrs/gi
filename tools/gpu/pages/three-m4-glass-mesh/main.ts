/**
 * M4, T4.3 done-when 3b: one three mesh that mixes an opaque wall material
 * and a glass material lets light through its glass only.
 *
 * The thin-wall scene (one room split by a 3 cm wall, a lamp on the west
 * side only), built in three, with its partition replaced by one
 * `BufferGeometry` with two groups: group 0 the wall around a 2 × 1.4 m
 * window, group 1 a 1 cm pane filling the window. The mesh's material array
 * is [wall, pane]. Mirrored by the adapter's SceneSync, DDGI updates every
 * probe every frame for FRAMES frames, three times:
 *
 * - **opaque pane:** the pane is a plain `MeshStandardMaterial`;
 * - **glass pane:** the pane is `MeshPhysicalMaterial({ transmission: 1 })`;
 * - **all glass:** both groups are the glass material.
 *
 * Before T4.3 the adapter sent the mixed mesh whole and opaque (with a
 * warning), so the glass pane read as the opaque one. Passing: the dark
 * side's floor is brighter with the glass pane than with the opaque one
 * (light through the glass), and darker than with the whole partition glass
 * (the wall around it still blocks). The ratios are reported; T4.2's
 * core-only `ddgi-glass-pane` read dark/lit 0.167 with the pane see-through
 * and 0.0126 opaque.
 */

import { BufferAttribute, BufferGeometry, Mesh, MeshPhysicalMaterial, MeshStandardMaterial, type Material } from 'three';
import { THIN_WALL_EMBED_M, THIN_WALL_ROOM_SIZE_M, THIN_WALL_THICKNESS_M } from '../../../../scenes/src/procedural/thin-wall.ts';
import { assert, runGpuTest } from '../../client.ts';
import { FieldProbe, releaseDeviceOnPageHide, type SurfacePoint } from '../ddgi-shared/harness.ts';
import { startThreeGi, threeProceduralScene } from '../three-m4-shared/three-scene.ts';

const env = import.meta.env as Record<string, unknown>;
/** Frames every probe is updated, per variant; as ddgi-glass-pane (enough for a ratio). */
const FRAMES = Number(env.VITE_DDGI_FRAMES) > 0 ? Math.round(Number(env.VITE_DDGI_FRAMES)) : 1200;

/** The window in the partition (z across the room, y up), in metres; as ddgi-glass-pane. */
const WINDOW_Z: readonly [number, number] = [1.0, 3.0];
const WINDOW_Y: readonly [number, number] = [0.8, 2.2];
const PANE_THICKNESS_M = 0.01;

type Vec3 = [number, number, number];
type Variant = 'opaque pane' | 'glass pane' | 'all glass';

/** The partition being collected: boxes with outward faces, a group (material slot) per box. */
class PartitionGeometry {
    private readonly positions: number[] = [];
    private readonly normals: number[] = [];
    private readonly indices: number[] = [];
    private readonly groups: { start: number; count: number; materialIndex: number }[] = [];

    /** An axis-aligned box, each face counter-clockwise seen from outside, in material slot `slot`. */
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
        const start = this.indices.length;
        for (const [normal, corners] of faces) {
            const first = this.positions.length / 3;
            for (const point of corners) {
                this.positions.push(...point);
                this.normals.push(...normal);
            }
            this.indices.push(first, first + 1, first + 2, first, first + 2, first + 3);
        }
        const last = this.groups[this.groups.length - 1];
        if (last && last.materialIndex === slot) last.count += this.indices.length - start;
        else this.groups.push({ start, count: this.indices.length - start, materialIndex: slot });
    }

    build(): BufferGeometry {
        const geometry = new BufferGeometry();
        geometry.setAttribute('position', new BufferAttribute(Float32Array.from(this.positions), 3));
        geometry.setAttribute('normal', new BufferAttribute(Float32Array.from(this.normals), 3));
        geometry.setIndex(new BufferAttribute(Uint32Array.from(this.indices), 1));
        for (const group of this.groups) geometry.addGroup(group.start, group.count, group.materialIndex);
        geometry.name = 'partition with a window';
        return geometry;
    }
}

/** The thin-wall partition with a window (slot 0) and the pane filling it (slot 1). */
function partitionWithPane(): BufferGeometry {
    const [, sizeY, sizeZ] = THIN_WALL_ROOM_SIZE_M;
    const middleX = THIN_WALL_ROOM_SIZE_M[0] / 2;
    const x0 = middleX - THIN_WALL_THICKNESS_M / 2;
    const x1 = middleX + THIN_WALL_THICKNESS_M / 2;
    const bottom = -THIN_WALL_EMBED_M;
    const top = sizeY + THIN_WALL_EMBED_M;
    const south = -THIN_WALL_EMBED_M;
    const north = sizeZ + THIN_WALL_EMBED_M;
    const partition = new PartitionGeometry();
    partition.box([x0, bottom, south], [x1, WINDOW_Y[0], north], 0);
    partition.box([x0, WINDOW_Y[1], south], [x1, top, north], 0);
    partition.box([x0, WINDOW_Y[0], south], [x1, WINDOW_Y[1], WINDOW_Z[0]], 0);
    partition.box([x0, WINDOW_Y[0], WINDOW_Z[1]], [x1, WINDOW_Y[1], north], 0);
    partition.box([middleX - PANE_THICKNESS_M / 2, WINDOW_Y[0], WINDOW_Z[0]], [middleX + PANE_THICKNESS_M / 2, WINDOW_Y[1], WINDOW_Z[1]], 1);
    return partition.build();
}

/** Floor points of one side, facing up. */
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

async function runVariant(device: GPUDevice, variant: Variant, log: (line: string) => void, warnings: string[]) {
    const built = threeProceduralScene('thin-wall', ['thin-wall']);
    const wallData = built.data.materials[built.data.instances.find((instance) => instance.name === 'thin-wall')!.material];
    const wall = new MeshStandardMaterial({ name: 'wall', roughness: 1, metalness: 0 });
    wall.color.setRGB(wallData.albedo[0], wallData.albedo[1], wallData.albedo[2]);
    const glass = new MeshPhysicalMaterial({ name: 'glass', transmission: 1, roughness: 0, metalness: 0 });
    const opaquePane = new MeshStandardMaterial({ name: 'opaque pane', color: 0xcccccc, roughness: 1, metalness: 0 });
    const materials: Material[] = variant === 'opaque pane' ? [wall, opaquePane] : variant === 'glass pane' ? [wall, glass] : [glass, glass];
    const partition = new Mesh(partitionWithPane(), materials);
    partition.name = 'partition';
    built.scene.add(partition);
    built.scene.updateMatrixWorld(true);

    const run = await startThreeGi(device, built.scene, built.manifest, `three-m4 glass ${variant}`, (message) => warnings.push(message));
    try {
        const sent = materials.map((material) => run.sync.sentMaterial(material)?.blocksLight);
        const field = run.ddgi.field;
        assert(field, 'DDGI has no field after prepare');
        await run.frames(FRAMES, field.grid.probeCount);
        const probe = await FieldProbe.create(device, run.platform);
        const middleX = THIN_WALL_ROOM_SIZE_M[0] / 2;
        const lit = meanOf(await probe.sample(field, floorPoints(0.2, middleX - 0.2)));
        const dark = meanOf(await probe.sample(field, floorPoints(middleX + 0.2, THIN_WALL_ROOM_SIZE_M[0] - 0.2)));
        log(`${variant}: blocksLight sent [wall slot, pane slot] ${JSON.stringify(sent)}; lit-side floor E ${lit.toFixed(5)}, dark-side ${dark.toFixed(6)}, dark / lit ${(dark / lit).toFixed(5)}`);
        return { blocksLightSent: sent, litMean: lit, darkMean: dark, darkOverLit: dark / lit, flagRewrites: run.platform.rays.triangleFlagRewrites };
    } finally {
        run.dispose();
    }
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const warnings: string[] = [];
        const opaque = await runVariant(device, 'opaque pane', log, warnings);
        const glassPane = await runVariant(device, 'glass pane', log, warnings);
        const allGlass = await runVariant(device, 'all glass', log, warnings);
        log(`dark side: glass pane / opaque pane ${(glassPane.darkMean / opaque.darkMean).toFixed(2)}x, all glass / glass pane ${(allGlass.darkMean / glassPane.darkMean).toFixed(2)}x; adapter warnings ${warnings.length}`);
        assert(JSON.stringify(glassPane.blocksLightSent) === '[true,false]', `the glass-pane mesh was sent with blocksLight ${JSON.stringify(glassPane.blocksLightSent)}`);
        assert(warnings.length === 0, `the adapter warned: ${warnings.join(' | ')}`);
        assert(glassPane.darkMean > 2 * opaque.darkMean, `the dark side is not clearly brighter through the glass pane (${glassPane.darkMean} against ${opaque.darkMean} opaque)`);
        assert(allGlass.darkMean > 1.5 * glassPane.darkMean, `the wall around the pane doesn't block: all glass ${allGlass.darkMean} against the glass pane's ${glassPane.darkMean}`);
        return { frames: FRAMES, opaque, glassPane, allGlass };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
