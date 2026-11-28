import { describe, expect, it } from 'vitest';
import { CUBE_FACE_COUNT, FLIP_Y_LAYER, flipPixels, mirroredCubeFaces, type CubeFacePixels } from './cube-mirror.ts';

/**
 * WebGPU's cube face selection (the WebGPU spec's table, as Vulkan and D3D):
 * the face of the major axis, then (s, t) in 0..1 with t = 0 at the top row.
 */
function cubeLookup(direction: readonly [number, number, number]): { face: number; s: number; t: number } {
    const [x, y, z] = direction;
    const [ax, ay, az] = [Math.abs(x), Math.abs(y), Math.abs(z)];
    let face: number;
    let sc: number;
    let tc: number;
    let ma: number;
    if (ax >= ay && ax >= az) {
        ma = ax;
        face = x > 0 ? 0 : 1;
        sc = x > 0 ? -z : z;
        tc = -y;
    } else if (ay >= az) {
        ma = ay;
        face = y > 0 ? 2 : 3;
        sc = x;
        tc = y > 0 ? z : -z;
    } else {
        ma = az;
        face = z > 0 ? 4 : 5;
        sc = z > 0 ? x : -x;
        tc = -y;
    }
    return { face, s: (sc / ma + 1) / 2, t: (tc / ma + 1) / 2 };
}

/** The texel value a nearest lookup returns along `direction`. */
function sample(faces: readonly CubeFacePixels<Float32Array>[], direction: readonly [number, number, number]): number {
    const { face, s, t } = cubeLookup(direction);
    const size = faces[face].width;
    const column = Math.min(Math.floor(s * size), size - 1);
    const row = Math.min(Math.floor(t * size), size - 1);
    return faces[face].data[row * size + column];
}

/** A cube whose every texel holds a different number. */
function codedCube(size: number): CubeFacePixels<Float32Array>[] {
    return Array.from({ length: CUBE_FACE_COUNT }, (_, face) => {
        const data = new Float32Array(size * size);
        for (let texel = 0; texel < size * size; texel++) data[texel] = face * 1000 + texel;
        return { data, width: size, height: size };
    });
}

/** Directions spread over the sphere, away from face edges (where rounding could pick a neighbour). */
function directions(count: number): [number, number, number][] {
    const result: [number, number, number][] = [];
    for (let index = 0; index < count; index++) {
        // A Fibonacci sphere.
        const y = 1 - (2 * (index + 0.5)) / count;
        const radius = Math.sqrt(1 - y * y);
        const angle = index * Math.PI * (3 - Math.sqrt(5));
        result.push([Math.cos(angle) * radius, y, Math.sin(angle) * radius]);
    }
    return result;
}

describe('three cube maps mirrored in x', () => {
    const size = 8;
    const three = codedCube(size);

    it('shows along d what three shows along d', () => {
        // three r187 looks a cube up at (-x, y, z); the platform at d.
        const mirrored = mirroredCubeFaces(three, 1);
        let checked = 0;
        for (const direction of directions(2000)) {
            const threeShows = sample(three, [-direction[0], direction[1], direction[2]]);
            expect(sample(mirrored, direction)).toBe(threeShows);
            checked++;
        }
        expect(checked).toBe(2000);
    });

    it('would be wrong without the mirror', () => {
        let differing = 0;
        for (const direction of directions(200)) {
            if (sample(three, direction) !== sample(three, [-direction[0], direction[1], direction[2]])) differing++;
        }
        expect(differing).toBeGreaterThan(150);
    });

    it('applies three\'s flipY too', () => {
        // With flipY three uploads image i upside down into layer FLIP_Y_LAYER[i] (the ±Y faces swap).
        const uploadedByThree: CubeFacePixels<Float32Array>[] = [];
        three.forEach((face, image) => (uploadedByThree[FLIP_Y_LAYER[image]] = flipPixels(face, 1, false, true)));
        const mirrored = mirroredCubeFaces(three, 1, true);
        for (const direction of directions(500)) {
            expect(sample(mirrored, direction)).toBe(sample(uploadedByThree, [-direction[0], direction[1], direction[2]]));
        }
    });

    it('keeps every channel of a texel together', () => {
        const face = { data: new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]), width: 2, height: 1 };
        expect(Array.from(flipPixels(face, 4, true, false).data)).toEqual([5, 6, 7, 8, 1, 2, 3, 4]);
    });

    it('needs six faces', () => {
        expect(() => mirroredCubeFaces(three.slice(0, 5), 1)).toThrow(RangeError);
    });
});
