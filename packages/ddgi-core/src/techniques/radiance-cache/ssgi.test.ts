/**
 * RC4.2 on the CPU: the SSGI twin's pieces against values worked out from
 * Capsaicin's formulas (ssgi.comp), its constants against ssgi.cpp, a flat
 * wall seen head-on (open: occlusion 1, bent normal the normal, no near
 * light), a floor beside a wall (closed in: occlusion below 1, the bent
 * normal leaning away from the wall, the wall's light), and the probe
 * filter's twin on a small screen (a constant atlas stays constant, a probe
 * off the plane is left out).
 */

import { describe, expect, it } from 'vitest';
import { cameraRelativeInverseViewProjection, multiplyMat4 } from './camera.ts';
import { INVALID_ID, packSeed } from './capsaicin-math.ts';
import { filterScreenProbes, type ProbeFilterInputs } from './probe-filter.ts';
import { screenProbeCellSize, viewForward } from './screen-irradiance.ts';
import { DEFAULT_RADIANCE_CACHE_SETTINGS, PROBE_SIZE } from './settings.ts';
import { ssgiConstants } from './ssgi.ts';
import { integrateBentNormal, integrateHalfArc, intersectDirPlaneOneSided, rotateFromTo, ssgiPixel, type SSGIScreen } from './ssgi-cpu.ts';
import { SSGI_CONSTANTS_BYTES, SSGI_CONSTANTS_OFFSETS, SSGI_WGSL } from './wgsl/ssgi.ts';
import { BLUR_DIRECTION_BYTES, FILTER_ANGLE_THRESHOLD, FILTER_PROBES_WGSL, FILTER_RADIUS } from './wgsl/screen-irradiance.ts';

type Vector = [number, number, number];

const perspective = (fovY: number, aspect: number, near: number, far: number): number[] => {
    // WebGPU's 0..1 depth, column-major.
    const f = 1 / Math.tan(fovY / 2);
    return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0];
};

/** A look-at view (column-major), as the GPU pages build it. */
function lookAt(eye: Vector, target: Vector): number[] {
    const sub = (a: Vector, b: Vector): Vector => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const norm = (a: Vector): Vector => {
        const size = Math.hypot(...a);
        return [a[0] / size, a[1] / size, a[2] / size];
    };
    const crossV = (a: Vector, b: Vector): Vector => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dotV = (a: Vector, b: Vector) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const forward = norm(sub(target, eye));
    const right = norm(crossV(forward, Math.abs(forward[1]) > 0.999 ? [0, 0, -1] : [0, 1, 0]));
    const up = crossV(right, forward);
    return [right[0], up[0], -forward[0], 0, right[1], up[1], -forward[1], 0, right[2], up[2], -forward[2], 0, -dotV(right, eye), -dotV(up, eye), dotV(forward, eye), 1];
}

interface Plane {
    readonly point: Vector;
    readonly normal: Vector;
    /** Last frame's lit colour on it. */
    readonly lighting: Vector;
}

/** A screen ray-cast against planes (the nearest hit in front of the eye): depth, normals, lighting. */
function castScreen(width: number, height: number, eye: Vector, target: Vector, planes: readonly Plane[], fovY = Math.PI / 3): { screen: SSGIScreen; view: number[]; projection: number[] } {
    const near = 0.05;
    const far = 100;
    const view = lookAt(eye, target);
    const projection = perspective(fovY, width / height, near, far);
    const viewProjection = multiplyMat4(projection, view);
    const inverse = cameraRelativeInverseViewProjection({ position: eye, view, projection });
    const depth = new Float32Array(width * height).fill(1);
    const normal = new Float32Array(width * height * 4);
    const lighting = new Float32Array(width * height * 4);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            // The ray through the pixel centre: a point at depth 0.5 rebuilt, minus the eye.
            const u = (x + 0.5) / width;
            const v = (y + 0.5) / height;
            const ndc = [2 * u - 1, 2 * (1 - v) - 1, 0.5, 1];
            const h = [0, 1, 2, 3].map((row) => inverse[row] * ndc[0] + inverse[4 + row] * ndc[1] + inverse[8 + row] * ndc[2] + inverse[12 + row] * ndc[3]);
            const direction: Vector = [h[0] / h[3], h[1] / h[3], h[2] / h[3]];
            let best = Infinity;
            let hit: Plane | null = null;
            for (const plane of planes) {
                const denominator = direction[0] * plane.normal[0] + direction[1] * plane.normal[1] + direction[2] * plane.normal[2];
                if (Math.abs(denominator) < 1e-12) continue;
                const t = ((plane.point[0] - eye[0]) * plane.normal[0] + (plane.point[1] - eye[1]) * plane.normal[1] + (plane.point[2] - eye[2]) * plane.normal[2]) / denominator;
                if (t > 0 && t < best) {
                    best = t;
                    hit = plane;
                }
            }
            const pixel = x + y * width;
            if (!hit) continue;
            const world = [eye[0] + direction[0] * best, eye[1] + direction[1] * best, eye[2] + direction[2] * best, 1];
            const clip = [0, 1, 2, 3].map((row) => viewProjection[row] * world[0] + viewProjection[4 + row] * world[1] + viewProjection[8 + row] * world[2] + viewProjection[12 + row]);
            depth[pixel] = Math.fround(clip[2] / clip[3]);
            normal.set([...hit.normal, 0], pixel * 4);
            lighting.set([...hit.lighting, 1], pixel * 4);
        }
    }
    const screen: SSGIScreen = { width, height, depth, normal, lighting, lightingWidth: width, lightingHeight: height, farDepth: 1, inverseViewProjection: inverse, eye, frameIndex: 3 };
    return { screen, view, projection };
}

describe("SSGI's pieces (ssgi.comp)", () => {
    it('IntegrateHalfArc: an open slice about a normal along the view is 1, and cos n + n sin n tilted', () => {
        expect(integrateHalfArc(Math.PI / 2, 0) + integrateHalfArc(-Math.PI / 2, 0)).toBeCloseTo(1, 12);
        // h = n ± pi/2: each side's -cos(2h - n) is cos n, so the sum is cos n + n sin n.
        const n = 0.4;
        expect(integrateHalfArc(n + Math.PI / 2, n) + integrateHalfArc(n - Math.PI / 2, n)).toBeCloseTo(Math.cos(n) + n * Math.sin(n), 12);
        // A closed side (horizon at the normal) adds the half before the normal only.
        expect(integrateHalfArc(0, 0)).toBeCloseTo(0, 12);
    });

    it('RotFromToMatrix takes from onto to and keeps lengths; near-parallel vectors are left alone', () => {
        const from: Vector = [0, 0, 1];
        const to: Vector = [Math.sin(0.3), 0, Math.cos(0.3)];
        const turned = rotateFromTo(from, to, from);
        to.forEach((value, axis) => expect(turned[axis]).toBeCloseTo(value, 12));
        const other = rotateFromTo(from, to, [0, 1, 0]);
        expect(Math.hypot(...other)).toBeCloseTo(1, 12);
        expect(other[1]).toBeCloseTo(1, 12);
        // f > 1 - 0.0003: identity.
        expect(rotateFromTo(from, [0.01, 0, Math.sqrt(1 - 1e-4)], [1, 2, 3])).toEqual([1, 2, 3]);
    });

    it("IntersectDirPlaneOneSided measures from the origin, as h3r2tic's view space does", () => {
        // The plane z = -2 facing +z, a ray from the origin down -z: t = 2.
        expect(intersectDirPlaneOneSided([0, 0, -1], [0, 0, 1], [0, 0, -2])).toBeCloseTo(2, 12);
        // Facing away: the denominator is held at 1e-5.
        expect(intersectDirPlaneOneSided([0, 0, 1], [0, 0, 1], [0, 0, -2])).toBeCloseTo(2 / 1e-5, 6);
    });

    it('IntegrateBentNormal of an open slice about a normal along the view is that normal', () => {
        const constants = { inverseView: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], forward: [0, 0, -1] as const };
        const bent = integrateBentNormal(Math.PI / 2, -Math.PI / 2, 0, [0, 0, 1], [1, 0, 0], constants);
        expect(bent[0]).toBeCloseTo(0, 12);
        expect(bent[1]).toBeCloseTo(0, 12);
        expect(bent[2]).toBeCloseTo(2 / 3, 12);
    });

    it("has Capsaicin's constants (ssgi.cpp:135-156) at its defaults (ssgi.h:48-51)", () => {
        expect(DEFAULT_RADIANCE_CACHE_SETTINGS).toMatchObject({ ssgiSliceCount: 1, ssgiStepCount: 2, ssgiViewRadius: 0.2, ssgiFalloffRange: 0.05 });
        const projection = perspective(Math.PI / 3, 16 / 9, 0.1, 100);
        const constants = ssgiConstants({ view: lookAt([1, 2, 3], [1, 2, 0]), projection }, DEFAULT_RADIANCE_CACHE_SETTINGS);
        // falloff over the last 0.05 of 0.2 m: mul -1 / 0.01, add 0.19 / 0.01 + 1.
        expect(constants.falloffMul).toBeCloseTo(-100, 9);
        expect(constants.falloffAdd).toBeCloseTo(20, 9);
        // max(proj[0][0], proj[1][1]) is proj[1][1] = 1 / tan(30 deg) for a wide screen.
        expect(constants.uvRadius).toBeCloseTo(0.2 * 0.5 * Math.sqrt(3), 9);
        expect(constants.forward).toEqual(viewForward({ view: lookAt([1, 2, 3], [1, 2, 0]) }));
        expect([constants.sliceCount, constants.stepCount]).toEqual([1, 2]);
    });
});

describe('the SSGI twin on simple screens', () => {
    const constantsFor = (view: number[], projection: number[]) => ssgiConstants({ view, projection }, DEFAULT_RADIANCE_CACHE_SETTINGS);

    it('a wall seen head-on is open: occlusion 1, the bent normal the normal, no near-field light', () => {
        const { screen, view, projection } = castScreen(64, 64, [0, 0, 2], [0, 0, 0], [{ point: [0, 0, 0], normal: [0, 0, 1], lighting: [5, 5, 5] }]);
        const pixel = ssgiPixel(screen, constantsFor(view, projection), 32, 32);
        expect(pixel).not.toBeNull();
        if (!pixel) return;
        expect(pixel.ambientOcclusion).toBeCloseTo(1, 3);
        expect(pixel.bentNormal[2]).toBeCloseTo(1, 3);
        // Every sample lies in the plane: it raises no horizon, so no light is gathered.
        expect(Math.max(...pixel.nearField)).toBeLessThan(1e-3);
    });

    it('a floor beside a wall is closed in: occlusion below 1, the bent normal leaning away from the wall, the wall lit', () => {
        // A floor (y = 0) and a wall (z = 0, facing +z) with lit colour 2, seen from above and in front.
        const planes: Plane[] = [
            { point: [0, 0, 0], normal: [0, 1, 0], lighting: [0, 0, 0] },
            { point: [0, 0, 0], normal: [0, 0, 1], lighting: [2, 2, 2] },
        ];
        const { screen, view, projection } = castScreen(160, 120, [0, 1.2, 1.6], [0, 0, 0], planes);
        const constants = constantsFor(view, projection);
        let nearWall = 0;
        let leansAway = 0;
        let occluded = 0;
        let lit = 0;
        for (let y = 0; y < 120; y++) {
            for (let x = 0; x < 160; x++) {
                const pixel = ssgiPixel(screen, constants, x, y);
                if (!pixel || screen.normal[4 * (x + y * 160) + 1] !== 1) continue;
                // The floor point under this pixel: within 5 cm of the wall.
                const u = (x + 0.5) / 160;
                const v = (y + 0.5) / 120;
                const m = screen.inverseViewProjection;
                const ndc = [2 * u - 1, 2 * (1 - v) - 1, screen.depth[x + y * 160], 1];
                const h = [0, 1, 2, 3].map((row) => m[row] * ndc[0] + m[4 + row] * ndc[1] + m[8 + row] * ndc[2] + m[12 + row] * ndc[3]);
                const z = screen.eye[2] + h[2] / h[3];
                if (z > 0.05) continue;
                nearWall++;
                if (pixel.bentNormal[2] > 0) leansAway++;
                if (pixel.ambientOcclusion < 0.99) occluded++;
                if (pixel.nearField[0] > 0) lit++;
            }
        }
        expect(nearWall).toBeGreaterThan(100);
        // One slice of random direction a pixel: most, not all, of them see the wall.
        expect(leansAway / nearWall).toBeGreaterThan(0.6);
        expect(occluded / nearWall).toBeGreaterThan(0.6);
        expect(lit / nearWall).toBeGreaterThan(0.5);
    });
});

describe("the probe filter's twin (FilterScreenProbes)", () => {
    const width = 4 * PROBE_SIZE;
    const height = PROBE_SIZE;

    /**
     * Four probe tiles in a row, seeds at the tiles' centres, on a wall seen head-on through a
     * narrow view (as a 4 × 1 corner of a wide screen would see it: Capsaicin's cell size is
     * about a tile's pixels, 6 cm here); probe 3 pushed 1 m back when `step` is set.
     */
    function inputs(step: boolean): ProbeFilterInputs {
        const eye: Vector = [0, 0, 3];
        const planes: Plane[] = [{ point: [0, 0, 0], normal: [0, 0, 1], lighting: [0, 0, 0] }];
        const { screen, view, projection } = castScreen(width, height, eye, [0, 0, 0], planes, 0.02);
        const depth = Float32Array.from(screen.depth);
        if (step) {
            // The fourth tile drawn 1 m further back (a depth edge).
            const viewProjection = multiplyMat4(projection, view);
            const z = (viewProjection[10] * -1 + viewProjection[14]) / (viewProjection[11] * -1 + viewProjection[15]);
            for (let y = 0; y < height; y++) for (let x = 3 * PROBE_SIZE; x < width; x++) depth[x + y * width] = Math.fround(z);
        }
        const seeds = [0, 1, 2, 3].map((tile) => packSeed(tile * PROBE_SIZE + 4, 4));
        const level0 = Uint32Array.from(seeds);
        const maskLevels = [
            { width: 4, height: 1, texels: level0 },
            { width: 2, height: 1, texels: Uint32Array.from([seeds[0], seeds[2]]) },
            { width: 1, height: 1, texels: Uint32Array.from([seeds[0]]) },
        ];
        return {
            width,
            height,
            depth,
            normal: screen.normal,
            maskLevels,
            seeds,
            inverseViewProjection: screen.inverseViewProjection,
            eye,
            viewForward: viewForward({ view }),
            cellSize: screenProbeCellSize({ projection }, width, height),
        };
    }

    const atlasOf = (value: (tile: number) => number, hitDistance: number) => {
        const texels = new Float32Array(width * height * 4);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) texels.set([value(Math.floor(x / PROBE_SIZE)), 0.5, 0.25, hitDistance], 4 * (x + y * width));
        }
        return { width, height, texels };
    };

    it("is Capsaicin's: radius 3, a 3.6 degree reprojection angle", () => {
        expect(FILTER_RADIUS).toBe(3);
        expect(FILTER_ANGLE_THRESHOLD).toBeCloseTo(Math.cos(0.02 * Math.PI), 12);
        expect(INVALID_ID).toBe(0xffffffff);
    });

    it('keeps a constant atlas constant, with neighbours accepted', () => {
        const result = filterScreenProbes(inputs(false), atlasOf(() => 1, 50));
        expect(result.accepted[0]).toBeGreaterThan(0);
        expect(result.accepted[1]).toBe(0); // one row of probes: nothing above or below
        for (let i = 0; i < width * height; i++) {
            expect(result.atlas.texels[4 * i]).toBeCloseTo(1, 3);
            expect(result.atlas.texels[4 * i + 3]).toBeCloseTo(50, 1);
        }
    });

    it('blends neighbours on the plane and leaves out a probe across a depth edge', () => {
        // Tile values 0, 1, 2 on the wall and 100 on the step back.
        const result = filterScreenProbes(inputs(true), atlasOf((tile) => (tile === 3 ? 100 : tile), 50));
        const red = (tile: number) => result.atlas.texels[4 * (tile * PROBE_SIZE + 4 + 4 * width)];
        // Tile 2 mixes with 0 and 1 (on its plane, same depth: weight 1 each) but not the step (off its plane).
        expect(red(2)).toBeLessThan(2);
        expect(red(2)).toBeGreaterThan(0.9);
        // The step back keeps its own value.
        expect(red(3)).toBeCloseTo(100, 0);
    });
});

describe('uniform blocks of the RC4.2 kernels match their WGSL structs', () => {
    /** Field offsets by WGSL's layout rules for the types these structs use. */
    function offsets(source: string, structName: string): { offsets: Record<string, number>; size: number } {
        const body = new RegExp(`struct ${structName} \\{([^}]*)\\}`).exec(source)?.[1];
        if (!body) throw new Error(`no struct ${structName}`);
        const types: Record<string, { size: number; align: number }> = {
            f32: { size: 4, align: 4 },
            u32: { size: 4, align: 4 },
            vec2i: { size: 8, align: 8 },
            vec3f: { size: 12, align: 16 },
            mat4x4f: { size: 64, align: 16 },
        };
        const result: Record<string, number> = {};
        let offset = 0;
        let align = 1;
        for (const match of body.matchAll(/(\w+)\s*:\s*(\w+)\s*,/g)) {
            const type = types[match[2]];
            if (!type) throw new Error(`unknown type ${match[2]}`);
            offset = Math.ceil(offset / type.align) * type.align;
            result[match[1]] = offset;
            offset += type.size;
            align = Math.max(align, type.align);
        }
        return { offsets: result, size: Math.ceil(offset / align) * align };
    }

    it('RcSSGIConstants', () => {
        const layout = offsets(SSGI_WGSL, 'RcSSGIConstants');
        expect(layout.offsets).toEqual(SSGI_CONSTANTS_OFFSETS);
        expect(layout.size).toBe(SSGI_CONSTANTS_BYTES);
    });

    it('RcBlurDirection', () => {
        const layout = offsets(FILTER_PROBES_WGSL, 'RcBlurDirection');
        expect(layout.offsets).toEqual({ direction: 0 });
        expect(layout.size).toBeLessThanOrEqual(BLUR_DIRECTION_BYTES);
    });
});
