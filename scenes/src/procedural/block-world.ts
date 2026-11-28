/**
 * Block world: a generated town of blocks and courtyards on the open ground,
 * about 200 m across, for example X2 (how DDGI and the radiance cache cover
 * a big world). The manifest puts it on `open-ground`'s 200 m slab.
 *
 * An 8 × 8 grid of 24 m lots, each an 18 m block with half of a 6 m street
 * on every side, so the town spans 186 m between its outer walls and the
 * streets meet at the origin (where the open ground's cube stands). Each lot
 * is one of three kinds:
 *
 * - **building:** one or two boxes on the lot, 6 to 18 m tall, some with a
 *   narrower tower on top, up to {@link BLOCK_WORLD_MAX_HEIGHT_M};
 * - **courtyard:** four 5 m deep wings, 8 to 14 m tall, around an 8 m yard,
 *   with a 3 m gateway through one wing, so the yard sees the sky from above
 *   and the street through the gateway only;
 * - **plaza:** a paved square with a few low pavilions.
 *
 * Everything comes from a seed (the `seed` option, default
 * {@link BLOCK_WORLD_DEFAULT_SEED}) through a small fixed generator, so every
 * run, test and renderer builds the same town. A sun lights it from the
 * south-east, low enough that the streets between tall blocks are in shade
 * and lit mostly by the sky and by the light the walls bounce.
 *
 * Solids run 5 cm into the ground and into each other, so no seam is flush.
 */

import type { BuilderOptions, ProceduralScene, SceneLightData } from '../scene-data';
import { SceneBuilder } from './geometry';

/** Lots per side of the grid. */
export const BLOCK_WORLD_LOTS = 8;
/** One lot: an 18 m block and half of the 6 m street on each side. */
export const BLOCK_WORLD_LOT_M = 24;
export const BLOCK_WORLD_STREET_M = 6;
export const BLOCK_WORLD_BLOCK_M = BLOCK_WORLD_LOT_M - BLOCK_WORLD_STREET_M;
/** The tallest a building may be, tower included. */
export const BLOCK_WORLD_MAX_HEIGHT_M = 32;
export const BLOCK_WORLD_DEFAULT_SEED = 1;
/** Where the sun is, from the ground (not normalised); the light shines the opposite way. */
export const BLOCK_WORLD_TO_SUN = [0.5, 0.62, 0.6] as const;

const SINK_M = 0.05;
const WING_DEPTH_M = 5;
const GATEWAY_WIDTH_M = 3;
const GATEWAY_HEIGHT_M = 4;

/** A small seeded generator (mulberry32): the same numbers for the same seed, everywhere. */
function seededRandom(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/** A whole number of half metres between `low` and `high`. */
function halfMetres(random: () => number, low: number, high: number): number {
    return Math.round((low + random() * (high - low)) * 2) / 2;
}

const WALL_COLOURS: ReadonlyArray<{ name: string; albedo: [number, number, number] }> = [
    { name: 'plaster', albedo: [0.72, 0.66, 0.55] },
    { name: 'brick', albedo: [0.55, 0.32, 0.25] },
    { name: 'concrete', albedo: [0.6, 0.6, 0.58] },
    { name: 'limestone', albedo: [0.76, 0.72, 0.64] },
    { name: 'slate-blue', albedo: [0.42, 0.48, 0.55] },
    { name: 'ochre', albedo: [0.7, 0.52, 0.32] },
];

/** The sun the builder adds, as the manifest lists it. */
export function blockWorldSun(): SceneLightData {
    return { type: 'directional', label: 'sun', color: [1, 0.95, 0.85], intensity: 3, direction: [-BLOCK_WORLD_TO_SUN[0], -BLOCK_WORLD_TO_SUN[1], -BLOCK_WORLD_TO_SUN[2]] };
}

export function buildBlockWorld(options: BuilderOptions = {}): ProceduralScene {
    const seed = typeof options.seed === 'number' ? options.seed : BLOCK_WORLD_DEFAULT_SEED;
    const random = seededRandom(seed);
    const builder = new SceneBuilder();
    const walls = WALL_COLOURS.map((colour) => builder.addDiffuse(`wall-${colour.name}`, colour.albedo));
    const paving = builder.addDiffuse('paving', [0.42, 0.39, 0.34]);
    const pickWall = (): number => walls[Math.floor(random() * walls.length)];

    const half = (BLOCK_WORLD_LOTS * BLOCK_WORLD_LOT_M) / 2;
    const block = BLOCK_WORLD_BLOCK_M;
    for (let row = 0; row < BLOCK_WORLD_LOTS; row++) {
        for (let column = 0; column < BLOCK_WORLD_LOTS; column++) {
            const x0 = -half + column * BLOCK_WORLD_LOT_M + BLOCK_WORLD_STREET_M / 2;
            const z0 = -half + row * BLOCK_WORLD_LOT_M + BLOCK_WORLD_STREET_M / 2;
            const x1 = x0 + block;
            const z1 = z0 + block;
            const name = `lot-${column}-${row}`;
            const roll = random();

            if (roll < 0.4) {
                // A courtyard: four wings around the yard, a gateway through one of them.
                const height = halfMetres(random, 8, 14);
                const wall = pickWall();
                const d = WING_DEPTH_M;
                const side = Math.floor(random() * 4); // 0 north (−z), 1 east, 2 south, 3 west
                const wings: Array<{ id: string; min: [number, number, number]; max: [number, number, number]; along: 'x' | 'z' }> = [
                    { id: 'north', min: [x0, -SINK_M, z0], max: [x1, height, z0 + d], along: 'x' },
                    { id: 'east', min: [x1 - d, -SINK_M, z0 + d - SINK_M], max: [x1, height, z1 - d + SINK_M], along: 'z' },
                    { id: 'south', min: [x0, -SINK_M, z1 - d], max: [x1, height, z1], along: 'x' },
                    { id: 'west', min: [x0, -SINK_M, z0 + d - SINK_M], max: [x0 + d, height, z1 - d + SINK_M], along: 'z' },
                ];
                wings.forEach((wing, index) => {
                    if (index !== side) {
                        builder.addBox(`${name}-wing-${wing.id}`, wall, wing.min, wing.max);
                        return;
                    }
                    // The gateway: a gap through the middle of the wing at street level, the wing bridging it above.
                    const axis = wing.along === 'x' ? 0 : 2;
                    const middle = (wing.min[axis] + wing.max[axis]) / 2;
                    const gapLow = middle - GATEWAY_WIDTH_M / 2;
                    const gapHigh = middle + GATEWAY_WIDTH_M / 2;
                    const before = { min: [...wing.min] as [number, number, number], max: [...wing.max] as [number, number, number] };
                    before.max[axis] = gapLow;
                    const after = { min: [...wing.min] as [number, number, number], max: [...wing.max] as [number, number, number] };
                    after.min[axis] = gapHigh;
                    const over = { min: [...wing.min] as [number, number, number], max: [...wing.max] as [number, number, number] };
                    over.min[axis] = gapLow - SINK_M;
                    over.max[axis] = gapHigh + SINK_M;
                    over.min[1] = GATEWAY_HEIGHT_M;
                    builder.addBox(`${name}-wing-${wing.id}-a`, wall, before.min, before.max);
                    builder.addBox(`${name}-wing-${wing.id}-b`, wall, after.min, after.max);
                    builder.addBox(`${name}-wing-${wing.id}-over-gateway`, wall, over.min, over.max);
                });
            } else if (roll < 0.8) {
                // Buildings: the lot whole, or split in two along x or z.
                const split = random() < 0.5;
                const alongX = random() < 0.5;
                const cut = split ? halfMetres(random, 7, 11) : block;
                const parts: Array<[number, number, number, number]> = split
                    ? alongX
                        ? [
                              [x0, z0, x0 + cut, z1],
                              [x0 + cut + 2, z0, x1, z1],
                          ]
                        : [
                              [x0, z0, x1, z0 + cut],
                              [x0, z0 + cut + 2, x1, z1],
                          ]
                    : [[x0, z0, x1, z1]];
                parts.forEach(([ax, az, bx, bz], index) => {
                    const height = halfMetres(random, 6, 18);
                    const wall = pickWall();
                    builder.addBox(`${name}-building-${index}`, wall, [ax, -SINK_M, az], [bx, height, bz]);
                    if (random() < 0.35) {
                        // A tower set back from the edges, up to the tallest height.
                        const inset = halfMetres(random, 1.5, 3);
                        const top = Math.min(BLOCK_WORLD_MAX_HEIGHT_M, height + halfMetres(random, 6, 14));
                        builder.addBox(`${name}-building-${index}-tower`, pickWall(), [ax + inset, height - SINK_M, az + inset], [bx - inset, top, bz - inset]);
                    }
                });
            } else {
                // A plaza: paving a little above the ground, and low pavilions on it.
                builder.addBox(`${name}-paving`, paving, [x0, -SINK_M, z0], [x1, 0.1, z1]);
                const pavilions = 1 + Math.floor(random() * 3);
                for (let index = 0; index < pavilions; index++) {
                    const sizeX = halfMetres(random, 1.5, 4);
                    const sizeZ = halfMetres(random, 1.5, 4);
                    const height = halfMetres(random, 1, 3.5);
                    const cx = x0 + 2 + random() * (block - 4 - sizeX);
                    const cz = z0 + 2 + random() * (block - 4 - sizeZ);
                    builder.addBox(`${name}-pavilion-${index}`, pickWall(), [cx, 0.05, cz], [cx + sizeX, height, cz + sizeZ]);
                }
            }
        }
    }

    builder.addLight(blockWorldSun());
    return builder.build();
}
