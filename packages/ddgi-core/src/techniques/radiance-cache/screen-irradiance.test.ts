/**
 * RC2.2 on the CPU: fixed reference values for Capsaicin's hemi-octahedral
 * map and SH (worked out by hand from the HLSL, so a map mirrored or
 * transposed the same way in the WGSL and the CPU port still fails), the
 * projection's units (a constant sky of radiance L reads E = πL; Capsaicin's
 * own weight reads 4L, register D25), the uniform blocks against their WGSL
 * structs, and the camera's jitter.
 */

import { describe, expect, it } from 'vitest';
import { cameraRelativeInverseViewProjection, jitteredProjection, multiplyMat4 } from './camera.ts';
import {
    CAPSAICIN_SH_CELL_WEIGHT,
    getOrthoVectors,
    integrateHemisphereSH,
    mapToHemiOctahedron,
    mapToHemiOctahedronInverse,
    probeCellDirection,
    projectProbe,
    SH_CELL_SOLID_ANGLE,
    SH_CELL_SUBSAMPLES,
    shGetCoefficients,
    shGetCoefficientsClampedCosineCone,
    shIrradiance,
    type Direction,
    CpuRandom,
    interpolateScreenProbes,
    INVALID_ID,
    type InterpolationInputs,
} from './capsaicin-math.ts';
import { screenProbeCellSize, viewForward } from './screen-irradiance.ts';
import { PROBE_SIZE } from './settings.ts';
import { DEBUG_UNIFORM_BYTES, DEBUG_UNIFORM_OFFSETS, DEBUG_WGSL } from './wgsl/debug.ts';
import { FRAME_UNIFORM_BYTES, FRAME_UNIFORM_OFFSETS, FRAME_WGSL } from './wgsl/screen-probes.ts';
import { INTERPOLATE_WGSL, SCREEN_PROBES_CONSTANTS_BYTES, SCREEN_PROBES_CONSTANTS_OFFSETS, SH_CELL_SOLID_ANGLE as WGSL_SH_CELL_SOLID_ANGLE, SH_CELL_SUBSAMPLES as WGSL_SH_CELL_SUBSAMPLES } from './wgsl/screen-irradiance.ts';

const expectDirection = (actual: readonly number[], expected: readonly number[], digits = 9): void => {
    expected.forEach((value, axis) => expect(actual[axis]).toBeCloseTo(value, digits));
};

describe("Capsaicin's hemi-octahedral map, fixed values (sampling.hlsl:61-116)", () => {
    // Each value below is worked through the HLSL by hand: st = 2 uv - 1, turned 45 degrees to
    // ((s + t) / 2, (s - t) / 2); distance = 1 - |st.x| - |st.y|; radius = 1 - |distance|;
    // phi = pi/4 ((|st.y| - |st.x|) / radius + 1); sin(theta) = radius sqrt(2 - radius^2);
    // direction = (sin(theta) sign(st.x) cos(phi), sin(theta) sign(st.y) sin(phi), sign(distance)(1 - radius^2)).
    it('sends the square centre to the normal and the corners to the horizon, +x at (1, 1), +y at (1, 0)', () => {
        expectDirection(mapToHemiOctahedron(0.5, 0.5), [0, 0, 1]);
        // (1, 1): st = (1, 0), radius 1, phi 0.
        expectDirection(mapToHemiOctahedron(1, 1), [1, 0, 0]);
        // (1, 0): st = (0, 1), phi pi/2.
        expectDirection(mapToHemiOctahedron(1, 0), [0, 1, 0]);
        // (0, 0): st = (-1, 0).
        expectDirection(mapToHemiOctahedron(0, 0), [-1, 0, 0]);
        // (0, 1): st = (0, -1).
        expectDirection(mapToHemiOctahedron(0, 1), [0, -1, 0]);
    });

    it('gives the worked interior points', () => {
        // (0.75, 0.5): st = (0.25, 0.25), distance 0.5, radius 0.5, phi pi/4, sin(theta) = 0.5 sqrt(1.75), z = 0.75.
        const diagonal = 0.5 * Math.sqrt(1.75) * Math.SQRT1_2;
        expectDirection(mapToHemiOctahedron(0.75, 0.5), [diagonal, diagonal, 0.75]);
        // (0.5, 0.25): st = (-0.25, 0.25): x mirrored.
        expectDirection(mapToHemiOctahedron(0.5, 0.25), [-diagonal, diagonal, 0.75]);
        // (0.625, 0.375): st = (0, 0.25), radius 0.25, phi pi/2, sin(theta) = 0.25 sqrt(1.9375), z = 0.9375.
        expectDirection(mapToHemiOctahedron(0.625, 0.375), [0, 0.25 * Math.sqrt(1.9375), 0.9375]);
        // (0.9, 0.2): st = (0.1, 0.7), distance 0.2, radius 0.8, phi = pi/4 (0.6 / 0.8 + 1) = 7 pi / 16,
        // sin(theta) = 0.8 sqrt(1.36), z = 1 - 0.64.
        const sinTheta = 0.8 * Math.sqrt(1.36);
        expectDirection(mapToHemiOctahedron(0.9, 0.2), [sinTheta * Math.cos((7 * Math.PI) / 16), sinTheta * Math.sin((7 * Math.PI) / 16), 0.36]);
    });

    it('inverts the worked points (the inverse map, sampling.hlsl:90-116)', () => {
        expectDirection(mapToHemiOctahedronInverse([0, 0, 1]), [0.5, 0.5]);
        expectDirection(mapToHemiOctahedronInverse([1, 0, 0]), [1, 1]);
        // (0, 1, 0): radius 1, phi = 1 - 0 (|x| < |y|), t = 1, s = 0, st = (0, 1), turned to (1, -1).
        expectDirection(mapToHemiOctahedronInverse([0, 1, 0]), [1, 0]);
        expectDirection(mapToHemiOctahedronInverse([0, -1, 0]), [0, 1]);
        const sinTheta = 0.8 * Math.sqrt(1.36);
        expectDirection(mapToHemiOctahedronInverse([sinTheta * Math.cos((7 * Math.PI) / 16), sinTheta * Math.sin((7 * Math.PI) / 16), 0.36]), [0.9, 0.2], 6);
    });
});

describe("Capsaicin's SH (spherical_harmonics.hlsl)", () => {
    it('SH_GetCoefficients gives the fixed basis values: band 1 is (-y, z, -x)', () => {
        expectDirection(shGetCoefficients([0, 0, 1]), [0.2820947917738781, 0, 0.4886025119029199, 0, 0, 0, 0.6307831305050400, 0, 0]);
        expectDirection(shGetCoefficients([1, 0, 0]), [0.2820947917738781, 0, 0, -0.48860251190292, 0, 0, -0.3153915652525201, 0, 0.5462742152960395]);
        expectDirection(shGetCoefficients([0, 1, 0]), [0.2820947917738781, -0.48860251190292, 0, 0, 0, 0, -0.3153915652525201, 0, -0.5462742152960395]);
        const d = Math.SQRT1_2;
        // (d, d, 0): fS1 = 2 x y = 1.
        expectDirection(shGetCoefficients([d, d, 0]), [0.2820947917738781, -0.48860251190292 * d, 0, -0.48860251190292 * d, 0.5462742152960395, 0, -0.3153915652525201, 0, 0]);
    });

    it('the clamped cosine at a whole hemisphere (ao 1) is pi, 2 pi / 3 and pi / 4 times the basis', () => {
        const lobe = shGetCoefficientsClampedCosineCone([0, 0, 1], Math.PI / 2);
        expectDirection(lobe, [Math.PI * 0.2820947917738781, 0, ((2 * Math.PI) / 3) * 0.4886025119029199, 0, 0, 0, (Math.PI / 4) * 0.6307831305050400, 0, 0], 6);
        const direction: Direction = [0.36, -0.48, 0.8];
        const basis = shGetCoefficients(direction);
        const bands = [Math.PI, (2 * Math.PI) / 3, (2 * Math.PI) / 3, (2 * Math.PI) / 3, Math.PI / 4, Math.PI / 4, Math.PI / 4, Math.PI / 4, Math.PI / 4];
        expectDirection(shGetCoefficientsClampedCosineCone(direction, Math.PI / 2), basis.map((value, index) => value * bands[index]), 6);
    });
});

describe('the projection and its units (ProjectScreenProbes, InterpolateScreenProbes, ResolveGI1)', () => {
    const normals: Direction[] = [
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [0.36, -0.48, 0.8],
    ];

    it('reads a constant sky of radiance L as E = pi L along the normal; Capsaicin projection reads 1.30 pi L with no history (register D25, D34)', () => {
        expect(WGSL_SH_CELL_SOLID_ANGLE).toBeCloseTo(SH_CELL_SOLID_ANGLE, 12);
        expect(WGSL_SH_CELL_SUBSAMPLES).toBe(SH_CELL_SUBSAMPLES);
        const radiance = 0.7;
        for (const normal of normals) {
            // Ours: each cell's solid angle times its mean SH basis (4 x 4 points): within 0.2% of pi L.
            const ours = shIrradiance(projectProbe(() => radiance, normal), normal);
            expect(Math.abs(ours / (Math.PI * radiance) - 1)).toBeLessThan(0.002);
            // The basis at the cell centre alone leans towards the normal: 2.1% high even with the solid angle.
            const centres = shIrradiance(projectProbe(() => radiance, normal, SH_CELL_SOLID_ANGLE, 1), normal);
            expect(centres / (Math.PI * radiance)).toBeCloseTo(1.021, 3);
            // Capsaicin: 1 / probe_size and the centre (gi1.comp:1466-1474): 4 / pi times that again, 4.08 L.
            const capsaicin = shIrradiance(projectProbe(() => radiance, normal, CAPSAICIN_SH_CELL_WEIGHT, 1), normal);
            expect(capsaicin / centres).toBeCloseTo(4 / Math.PI, 9);
            expect(capsaicin / radiance).toBeCloseTo(4.084, 2);
        }
        // ResolveGI1 applies evaluateLambert (albedo / pi, material_evaluation.hlsl:157-160) to this E, as three's
        // composite does: a white surface under a white sky of radiance L should reflect L.
    });

    it('projects a direction-varying sky to its SH (against a fine integral), and a transposed probe fails', () => {
        // Brighter towards +x than -x, and towards -z than +z: no mirror symmetry across either tangent.
        const sky = (direction: Direction): number => 1 + 0.6 * direction[0] - 0.3 * direction[2];
        // A cell's value is its mean radiance (the GPU's is one random ray in the cell: the same on average).
        const cellMean = (cell: number, normal: Direction): number => {
            const { b1, b2 } = getOrthoVectors(normal);
            let sum = 0;
            for (let row = 0; row < 8; row++) {
                for (let column = 0; column < 8; column++) {
                    const [x, y, z] = mapToHemiOctahedron(((cell % 8) + (column + 0.5) / 8) / 8, (Math.floor(cell / 8) + (row + 0.5) / 8) / 8);
                    sum += sky([x * b1[0] + y * b2[0] + z * normal[0], x * b1[1] + y * b2[1] + z * normal[1], x * b1[2] + y * b2[2] + z * normal[2]]);
                }
            }
            return sum / 64;
        };
        for (const normal of normals) {
            const reference = integrateHemisphereSH(sky, normal, 192);
            const projected = projectProbe((cell) => cellMean(cell, normal), normal);
            const scale = Math.max(...reference.map(Math.abs));
            // A cell holds one value, so the product of means differs a little from the mean of products (up to 1.7%
            // of the largest coefficient where the sky changes along the normal): within 2.5%.
            reference.forEach((value, index) => expect(Math.abs(projected[index] - value) / scale).toBeLessThan(0.025));
            // Each cell read at its transpose (cell x, y from y, x) mirrors the sky across a tangent: it must show.
            const transposed = projectProbe((cell) => cellMean((cell % 8) * 8 + Math.floor(cell / 8), normal), normal);
            const worst = Math.max(...reference.map((value, index) => Math.abs(transposed[index] - value) / scale));
            expect(worst).toBeGreaterThan(0.2);
        }
        // The cell-centre direction a cell's ray is drawn about (probeCellDirection) lies in that cell.
        expect(probeCellDirection(9, [0, 0, 1])[2]).toBeGreaterThan(0);
    });
});

/**
 * Byte offsets of a WGSL struct's fields by the layout rules (WGSL 13.4.3),
 * for the scalar, vector and matrix types the uniform blocks use.
 */
function wgslStructOffsets(source: string, structName: string): { offsets: Record<string, number>; size: number } {
    const body = new RegExp(`struct ${structName} \\{([^}]*)\\}`).exec(source)?.[1];
    if (!body) throw new Error(`no struct ${structName}`);
    const types: Record<string, { size: number; align: number }> = {
        f32: { size: 4, align: 4 },
        u32: { size: 4, align: 4 },
        vec2f: { size: 8, align: 8 },
        vec2u: { size: 8, align: 8 },
        vec3f: { size: 12, align: 16 },
        vec4f: { size: 16, align: 16 },
        mat4x4f: { size: 64, align: 16 },
    };
    const offsets: Record<string, number> = {};
    let offset = 0;
    let structAlign = 1;
    for (const match of body.matchAll(/(\w+)\s*:\s*(\w+)\s*,/g)) {
        const type = types[match[2]];
        if (!type) throw new Error(`unknown type ${match[2]}`);
        offset = Math.ceil(offset / type.align) * type.align;
        offsets[match[1]] = offset;
        offset += type.size;
        structAlign = Math.max(structAlign, type.align);
    }
    return { offsets, size: Math.ceil(offset / structAlign) * structAlign };
}

describe('uniform blocks are written by named offsets that match their WGSL structs', () => {
    it('RcFrame', () => {
        const layout = wgslStructOffsets(FRAME_WGSL, 'RcFrame');
        expect(layout.offsets).toEqual(FRAME_UNIFORM_OFFSETS);
        expect(layout.size).toBe(FRAME_UNIFORM_BYTES);
    });

    it('RcScreenProbesConstants', () => {
        const layout = wgslStructOffsets(INTERPOLATE_WGSL, 'RcScreenProbesConstants');
        expect(layout.offsets).toEqual(SCREEN_PROBES_CONSTANTS_OFFSETS);
        expect(layout.size).toBeLessThanOrEqual(SCREEN_PROBES_CONSTANTS_BYTES);
    });

    it('RcDebug', () => {
        const layout = wgslStructOffsets(DEBUG_WGSL, 'RcDebug');
        expect(layout.offsets).toEqual(DEBUG_UNIFORM_OFFSETS);
        expect(layout.size).toBeLessThanOrEqual(DEBUG_UNIFORM_BYTES);
    });
});

describe('the camera the screen stage reads', () => {
    const perspective = (fovY: number, aspect: number, near: number, far: number): number[] => {
        const f = 1 / Math.tan(fovY / 2);
        return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0];
    };
    // Looking down -z from eye, turned 30 degrees about y.
    const angle = Math.PI / 6;
    const eye: [number, number, number] = [3, 2, 10];
    const rotation = [Math.cos(angle), 0, Math.sin(angle), 0, 0, 1, 0, 0, -Math.sin(angle), 0, Math.cos(angle), 0, 0, 0, 0, 1];
    const translation = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -eye[0], -eye[1], -eye[2], 1];
    const view = multiplyMat4(rotation, translation);
    const projection = perspective(1, 16 / 9, 0.1, 100);

    const project = (matrix: readonly number[], point: readonly number[]): number[] => {
        const clip = [0, 1, 2, 3].map((row) => matrix[row] * point[0] + matrix[4 + row] * point[1] + matrix[8 + row] * point[2] + matrix[12 + row]);
        return [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];
    };

    it('jitters the projection by an NDC offset (FrameCamera.jitter)', () => {
        const point = [2, 1, 0];
        const plain = project(multiplyMat4(projection, view), point);
        const jitter: [number, number] = [0.002, -0.0015];
        const jittered = project(multiplyMat4(jitteredProjection({ projection, jitter }), view), point);
        expect(jittered[0] - plain[0]).toBeCloseTo(jitter[0], 12);
        expect(jittered[1] - plain[1]).toBeCloseTo(jitter[1], 12);
        expect(jittered[2]).toBeCloseTo(plain[2], 12);
        expect(jitteredProjection({ projection })).toEqual(projection);
    });

    it('rebuilds a point from depth drawn with the jitter where it was drawn', () => {
        const point = [2, 1, 0];
        const jitter: [number, number] = [1 / 640, -1 / 360];
        const camera = { position: eye, view, projection, jitter };
        const ndc = project(multiplyMat4(jitteredProjection(camera), view), point);
        const inverse = cameraRelativeInverseViewProjection(camera);
        const h = [0, 1, 2, 3].map((row) => inverse[row] * ndc[0] + inverse[4 + row] * ndc[1] + inverse[8 + row] * ndc[2] + inverse[12 + row]);
        [0, 1, 2].forEach((axis) => expect(eye[axis] + h[axis] / h[3]).toBeCloseTo(point[axis], 6));
        // Without the jitter the same depth sample lands elsewhere (RC2.1's half-pixel error).
        const unjittered = cameraRelativeInverseViewProjection({ position: eye, view, projection });
        const u = [0, 1, 2, 3].map((row) => unjittered[row] * ndc[0] + unjittered[4 + row] * ndc[1] + unjittered[8 + row] * ndc[2] + unjittered[12 + row]);
        const error = Math.hypot(...[0, 1, 2].map((axis) => eye[axis] + u[axis] / u[3] - point[axis]));
        expect(error).toBeGreaterThan(1e-3);
    });

    it("gives the view's forward axis and Capsaicin's cell size", () => {
        // Turned 30 degrees about y, -z goes to (-sin, 0, -cos).
        expectDirection(viewForward({ view }), [-Math.sin(angle), 0, -Math.cos(angle)]);
        // gi1.cpp:1839-1843 at 1280 x 720: tan(fovY x 8 x max(1/720, 720/1280^2)).
        expect(screenProbeCellSize({ projection }, 1280, 720)).toBeCloseTo(Math.tan((1 * 8) / 720), 9);
    });
});

describe('InterpolateScreenProbes on the CPU (the GPU page compares the kernel with it)', () => {
    // A 64 x 32 screen of a wall at z = 0 facing the camera at z = 10: pixel (x, y) is at (x, -y, 0) / 10.
    const width = 64;
    const height = 32;
    const tiles = [width / PROBE_SIZE, height / PROBE_SIZE];
    const mipCount = Math.floor(Math.log2(Math.max(tiles[0], tiles[1]))) + 1;
    const seedAt = (tileX: number, tileY: number): number => ((tileX * PROBE_SIZE + 3) << 16) | (tileY * PROBE_SIZE + 4);
    function inputs(options: { probeTiles: Array<[number, number]>; offPlane?: Set<number> }): InterpolationInputs {
        // Level 0 from the probe tiles, then FilterProbeMask's rule.
        const levels: Uint32Array[] = [];
        let size = [...tiles];
        levels.push(new Uint32Array(size[0] * size[1]).fill(INVALID_ID));
        for (const [tx, ty] of options.probeTiles) levels[0][ty * size[0] + tx] = seedAt(tx, ty);
        const sizes = [size];
        for (let level = 1; level < mipCount; level++) {
            const next = [Math.max(size[0] >> 1, 1), Math.max(size[1] >> 1, 1)];
            const texels = new Uint32Array(next[0] * next[1]).fill(INVALID_ID);
            for (let y = 0; y < next[1]; y++) {
                for (let x = 0; x < next[0]; x++) {
                    for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
                        const cx = 2 * x + dx;
                        const cy = 2 * y + dy;
                        if (cx < size[0] && cy < size[1] && levels[level - 1][cy * size[0] + cx] !== INVALID_ID) {
                            texels[y * next[0] + x] = levels[level - 1][cy * size[0] + cx];
                            break;
                        }
                    }
                }
            }
            levels.push(texels);
            sizes.push(next);
            size = next;
        }
        const offPlane = options.offPlane ?? new Set<number>();
        return {
            width,
            height,
            frameIndex: 7,
            cellSize: 0.01,
            eye: [3.2, -1.6, 10],
            viewForward: [0, 0, -1],
            probeMaskMipCount: mipCount,
            isSky: () => false,
            // A probe's seed moved off the wall by a metre when its tile is in 'offPlane'.
            worldPosition: (x, y) => [x / 10, -y / 10, offPlane.has(Math.floor(y / PROBE_SIZE) * tiles[0] + Math.floor(x / PROBE_SIZE)) && x % PROBE_SIZE === 3 && y % PROBE_SIZE === 4 ? 1 : 0],
            normal: () => [0, 0, 1],
            probeMask: (level, x, y) => levels[level][y * sizes[level][0] + x],
            probeIrradiance: (tileX, tileY) => 1 + tileX + 10 * tileY,
        };
    }

    it('mixes up to four probes with weights summing to 1, each probe its own value', () => {
        const all: Array<[number, number]> = [];
        for (let y = 0; y < tiles[1]; y++) for (let x = 0; x < tiles[0]; x++) all.push([x, y]);
        const result = interpolateScreenProbes(inputs({ probeTiles: all }), 20, 12);
        expect(result).not.toBeNull();
        expect(result!.hint).toBe(1);
        expect(result!.weights.reduce((sum, w) => sum + w, 0)).toBeCloseTo(1, 12);
        const values = result!.probes.filter((probe) => probe !== INVALID_ID).map((probe) => 1 + Math.floor((probe >>> 16) / PROBE_SIZE) + 10 * Math.floor((probe & 0xffff) / PROBE_SIZE));
        expect(result!.irradiance).toBeGreaterThanOrEqual(Math.min(...values) - 1e-12);
        expect(result!.irradiance).toBeLessThanOrEqual(Math.max(...values) + 1e-12);
        expect(new Set(result!.probes).size).toBe(4);
    });

    it('gives a probe off the pixel plane no weight, and takes the backup when every probe is off it', () => {
        const probeTiles: Array<[number, number]> = [];
        for (let y = 0; y < tiles[1]; y++) for (let x = 0; x < tiles[0]; x++) probeTiles.push([x, y]);
        // One of the neighbours the pixel mixes, moved off the wall.
        const onPlane = interpolateScreenProbes(inputs({ probeTiles }), 20, 12)!;
        const neighbour = onPlane.probes.find((probe, index) => index > 0 && probe !== INVALID_ID)!;
        const neighbourTile = Math.floor((neighbour & 0xffff) / PROBE_SIZE) * tiles[0] + Math.floor((neighbour >>> 16) / PROBE_SIZE);
        expect(onPlane.weights[onPlane.probes.indexOf(neighbour)]).toBeGreaterThan(0);
        const someOff = interpolateScreenProbes(inputs({ probeTiles, offPlane: new Set([neighbourTile]) }), 20, 12)!;
        const offIndex = someOff.probes.indexOf(neighbour);
        expect(offIndex).toBeGreaterThanOrEqual(0);
        expect(someOff.weights[offIndex]).toBe(0);
        expect(someOff.hint).toBe(1);
        const allOff = interpolateScreenProbes(inputs({ probeTiles, offPlane: new Set(probeTiles.map(([x, y]) => y * tiles[0] + x)) }), 20, 12)!;
        expect(allOff.hint).toBe(0);
        const used = allOff.probes.filter((probe) => probe !== INVALID_ID).length;
        for (const [index, w] of allOff.weights.entries()) expect(w).toBeCloseTo(allOff.probes[index] === INVALID_ID ? 0 : 1 / used, 12);
    });

    it("draws the jitter from Capsaicin's PCG as rcMakeRandom starts it (D18): the same numbers on every call", () => {
        const a = new CpuRandom(1234, 5).rand2();
        const b = new CpuRandom(1234, 5).rand2();
        expect(a).toEqual(b);
        expect(new CpuRandom(1234, 6).rand2()).not.toEqual(a);
        for (const value of a) expect(value >= 0 && value < 1).toBe(true);
    });
});
