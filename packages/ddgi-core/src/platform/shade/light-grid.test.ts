import { describe, expect, it } from 'vitest';
import type { ComputeKernel, ComputeKernelOptions, KernelBuilder, WgslModule } from '../contract/wgsl.ts';
import { createRandom } from '../trace/oracle-fixtures.ts';
import {
    LIGHT_GRID_CELLS_PER_GROUP,
    LIGHT_GRID_MAX_CELLS,
    LIGHT_GRID_PICKS_PER_CELL,
    LIGHT_GRID_RUNS_PER_CELL,
    LIGHT_GRID_TEXTURE_WIDTH,
    LIGHT_GRID_WORKGROUP_SIZE,
    lightGridBuildWgsl,
    lightGridLayout,
    lightGridLookupWgsl,
    lightGridTextureRows,
    PlatformLightGrid,
} from './light-grid.ts';

const STUB_MODULE: WgslModule = { name: 'stub', dependencies: [], code: () => '', entries: [], resources: () => [], version: 0, storageBufferCount: 0 };

/** Just enough of a device for the grid, recording textures, writes and dispatches. */
function fakeDevice(features: string[] = []) {
    const textures: GPUTextureDescriptor[] = [];
    const writes: Array<{ offset: number; data: ArrayBuffer }> = [];
    const dispatches: number[] = [];
    const device = {
        features: new Set(features),
        createBuffer: (descriptor: GPUBufferDescriptor) => ({ ...descriptor, destroy() {} }),
        createTexture: (descriptor: GPUTextureDescriptor) => {
            textures.push(descriptor);
            const size = descriptor.size as GPUExtent3DDict;
            return { ...descriptor, width: size.width, height: size.height ?? 1, createView: () => ({ texture: descriptor }), destroy() {} };
        },
        createBindGroup: (descriptor: GPUBindGroupDescriptor) => ({ ...descriptor }),
        queue: { writeBuffer: (_buffer: GPUBuffer, offset: number, data: ArrayBuffer) => writes.push({ offset, data: data.slice(0) }) },
    } as unknown as GPUDevice;
    const encoder = {
        beginComputePass: () => ({ setPipeline() {}, setBindGroup() {}, dispatchWorkgroups: (count: number) => dispatches.push(count), end() {} }),
    } as unknown as GPUCommandEncoder;
    return { device, encoder, textures, writes, dispatches };
}

function fakeKernels(): KernelBuilder & { readonly built: ComputeKernelOptions[] } {
    const built: ComputeKernelOptions[] = [];
    return {
        built,
        computeKernel: (options: ComputeKernelOptions): Promise<ComputeKernel> => {
            built.push(options);
            return Promise.resolve({ pipeline: {} as GPUComputePipeline, ownLayouts: [{} as GPUBindGroupLayout], bind() {} });
        },
    };
}

const SPONZA_BOUNDS = { min: [-16.4, -0.2, -9.6] as [number, number, number], max: [20.1, 19.8, 14.5] as [number, number, number] };

describe('the light grid layout', () => {
    it('covers the bounds with cubic cells, as many as fit in the cap', () => {
        for (const maxCells of [1, 64, 4096, 32768]) {
            const layout = lightGridLayout(SPONZA_BOUNDS, maxCells);
            expect(layout.cellCount).toBeLessThanOrEqual(maxCells);
            expect(layout.cellCount).toBe(layout.cells[0] * layout.cells[1] * layout.cells[2]);
            for (let axis = 0; axis < 3; axis++) {
                expect(layout.origin[axis]).toBeLessThan(SPONZA_BOUNDS.min[axis]);
                expect(layout.origin[axis] + layout.cells[axis] * layout.cellSize).toBeGreaterThan(SPONZA_BOUNDS.max[axis]);
            }
            // Not much room left: a cell 2% smaller would not fit.
            const side = layout.cellSize * 0.98;
            const finer = [0, 1, 2].map((axis) => Math.ceil((SPONZA_BOUNDS.max[axis] - SPONZA_BOUNDS.min[axis]) / side)).reduce((a, b) => a * b, 1);
            if (maxCells > 1) expect(finer).toBeGreaterThan(maxCells * 0.9);
        }
        // Sponza at the default cap: cells of about 1.7 m, close to the design's two probe spacings (2 m).
        const sponza = lightGridLayout(SPONZA_BOUNDS, 4096);
        expect(sponza.cellSize).toBeGreaterThan(1.4);
        expect(sponza.cellSize).toBeLessThan(2);
    });

    it('sizes the cells of a flat scene by its area, and copes with an empty one', () => {
        const ground = lightGridLayout({ min: [-200, 0, -200], max: [200, 0, 200] }, 4096);
        expect(ground.cells[1]).toBe(1);
        expect(ground.cellCount).toBeGreaterThan(3000);
        const empty = lightGridLayout({ min: [0, 0, 0], max: [0, 0, 0] }, 4096);
        expect(empty.cellCount).toBeGreaterThanOrEqual(1);
        expect(empty.cellSize).toBeGreaterThan(0);
        expect(lightGridLayout(SPONZA_BOUNDS, 1e9).cellCount).toBeLessThanOrEqual(LIGHT_GRID_MAX_CELLS);
    });

    it('fits every cell and the outside cell in the picks texture, each cell in one row', () => {
        expect(LIGHT_GRID_TEXTURE_WIDTH % LIGHT_GRID_PICKS_PER_CELL).toBe(0);
        expect(lightGridTextureRows(4096)).toBe(Math.ceil((4097 * LIGHT_GRID_PICKS_PER_CELL) / 2048));
        expect(lightGridTextureRows(LIGHT_GRID_MAX_CELLS) * LIGHT_GRID_TEXTURE_WIDTH).toBeGreaterThanOrEqual((LIGHT_GRID_MAX_CELLS + 1) * LIGHT_GRID_PICKS_PER_CELL);
    });
});

/**
 * A twin of the build's weight (`giLightGridWeight`) and of the falloff the
 * hit uses, to check the method: the weight over a box is never below the
 * light's unshadowed irradiance at any point of the box. The GPU pages check
 * the WGSL itself end to end (hit-shading-many-lights).
 */
function falloff(distance: number, range: number): number {
    const inverseSquare = 1 / Math.max(distance * distance, 0.01);
    if (range <= 0) return inverseSquare;
    const window = Math.min(Math.max(1 - (distance / range) ** 4, 0), 1);
    return inverseSquare * window * window;
}
function cone(cosine: number, cosOuter: number, cosInner: number): number {
    if (cosInner <= cosOuter) return cosine > cosOuter ? 1 : 0;
    const t = Math.min(Math.max((cosine - cosOuter) / (cosInner - cosOuter), 0), 1);
    return t * t * (3 - 2 * t);
}
interface TwinLight {
    position: number[];
    direction: number[];
    range: number;
    spot: boolean;
    cosOuter: number;
    cosInner: number;
}
function irradianceAt(light: TwinLight, point: number[]): number {
    const toPoint = point.map((value, axis) => value - light.position[axis]);
    const distance = Math.hypot(...toPoint);
    let value = falloff(distance, light.range);
    if (light.spot) value *= cone(toPoint.reduce((sum, value, axis) => sum + (value / distance) * light.direction[axis], 0), light.cosOuter, light.cosInner);
    return value;
}
function boxWeight(light: TwinLight, boxMin: number[], boxMax: number[]): number {
    const nearest = light.position.map((value, axis) => Math.min(Math.max(value, boxMin[axis]), boxMax[axis]));
    let bound = falloff(Math.hypot(...nearest.map((value, axis) => value - light.position[axis])), light.range);
    if (light.spot && bound > 0) {
        const centre = boxMin.map((value, axis) => 0.5 * (value + boxMax[axis]));
        const radius = 0.5 * Math.hypot(...boxMax.map((value, axis) => value - boxMin[axis]));
        const toCentre = centre.map((value, axis) => value - light.position[axis]);
        const centreDistance = Math.hypot(...toCentre);
        if (centreDistance > radius) {
            const axisAngle = Math.acos(Math.min(Math.max(toCentre.reduce((sum, value, axis) => sum + (value / centreDistance) * light.direction[axis], 0), -1), 1));
            const nearestAngle = Math.max(axisAngle - Math.asin(radius / centreDistance), 0);
            bound *= cone(Math.cos(nearestAngle), light.cosOuter, light.cosInner);
        }
    }
    return bound;
}

describe("the light grid's weights", () => {
    it('never fall below a light at any point of the cell (so every light that can light a hit can be picked)', () => {
        const random = createRandom(2303);
        let checked = 0;
        for (let trial = 0; trial < 2000; trial++) {
            const size = 0.2 + random() * 3;
            const boxMin = [random() * 10 - 5, random() * 10 - 5, random() * 10 - 5];
            const boxMax = boxMin.map((value) => value + size);
            const direction = [random() - 0.5, random() - 0.5, random() - 0.5];
            const length = Math.hypot(...direction);
            const outer = 0.1 + random() * 1.2;
            const light: TwinLight = {
                position: [random() * 14 - 7, random() * 14 - 7, random() * 14 - 7],
                direction: direction.map((value) => value / length),
                range: random() < 0.4 ? 0 : 0.5 + random() * 8,
                spot: random() < 0.5,
                cosOuter: Math.cos(outer),
                cosInner: Math.cos(outer * (random() < 0.3 ? 1 : random())),
            };
            const bound = boxWeight(light, boxMin, boxMax);
            for (let sample = 0; sample < 50; sample++) {
                const point = boxMin.map((value) => value + random() * size);
                const actual = irradianceAt(light, point);
                expect(bound).toBeGreaterThanOrEqual(actual * (1 - 1e-9));
                if (actual > 0) checked++;
            }
        }
        expect(checked).toBeGreaterThan(20000);
    });
});

describe('the light grid kernels', () => {
    it('declare every platform global with the gi prefix, and the subgroup scan only in its own variant', () => {
        const lookup = lightGridLookupWgsl('giLightGridPicks');
        for (const code of [lightGridBuildWgsl('subgroups'), lightGridBuildWgsl('workgroup'), lookup]) {
            for (const match of code.matchAll(/^(?:fn|struct|const|var<[^>]*>|var|override)\s+([A-Za-z_]\w*)/gm)) {
                expect(match[1].toLowerCase().startsWith('gi'), match[1]).toBe(true);
            }
        }
        const subgroups = lightGridBuildWgsl('subgroups');
        const fallback = lightGridBuildWgsl('workgroup');
        expect(subgroups.startsWith('enable subgroups;')).toBe(true);
        expect(subgroups).toContain('subgroupInclusiveAdd');
        // The subgroup variant falls back to the serial scan when subgroups don't hold whole cells of consecutive threads.
        expect(subgroups).toContain('giLightGridSerialScan(sum, thread)');
        expect(fallback).not.toMatch(/subgroup[A-Z]|enable subgroups/);
        expect(fallback).toContain('giLightGridSerialScan');
        expect(LIGHT_GRID_WORKGROUP_SIZE).toBe(LIGHT_GRID_RUNS_PER_CELL * LIGHT_GRID_CELLS_PER_GROUP);
    });

    it('chooses subgroups when the device has them, and refuses them when it has not', () => {
        expect(new PlatformLightGrid(fakeDevice(['subgroups']).device, fakeKernels(), STUB_MODULE, STUB_MODULE).reduction).toBe('subgroups');
        expect(new PlatformLightGrid(fakeDevice().device, fakeKernels(), STUB_MODULE, STUB_MODULE).reduction).toBe('workgroup');
        expect(() => new PlatformLightGrid(fakeDevice().device, fakeKernels(), STUB_MODULE, STUB_MODULE, { reduction: 'subgroups' })).toThrow(/subgroups/);
    });

    it('stays off up to the exact limit, then builds one workgroup a cell and the outside cell each frame', async () => {
        const { device, encoder, textures, writes, dispatches } = fakeDevice();
        const kernels = fakeKernels();
        const grid = new PlatformLightGrid(device, kernels, STUB_MODULE, STUB_MODULE);
        // A one-texel texture until a scene needs it.
        expect(textures).toHaveLength(1);
        expect(grid.version).toBe(0);
        const frame = { frame: 7, bounds: SPONZA_BOUNDS, localLights: 8, exactLightLimit: 8, maxCells: 4096 };
        // Before the kernel compiles the grid stays off: direct light keeps the exact path.
        expect(grid.encode(encoder, { ...frame, localLights: 1024 }).active).toBe(false);
        await grid.prepare();
        // The light packing and the build: one code, two entry points.
        expect(kernels.built.map((options) => options.entryPoint)).toEqual(['giLightGridPackLights', 'giLightGridBuildCell']);
        expect(kernels.built[0].modules).toEqual([STUB_MODULE, STUB_MODULE]);
        expect(grid.encode(encoder, frame).active).toBe(false);
        expect(dispatches).toHaveLength(0);

        const lookup = grid.encode(encoder, { ...frame, localLights: 9 });
        expect(lookup.active).toBe(true);
        // One thread a light to pack them, then 16 cells a workgroup, the outside cell last.
        expect(dispatches).toEqual([1, Math.ceil((lookup.layout.cellCount + 1) / LIGHT_GRID_CELLS_PER_GROUP)]);
        expect(grid.version).toBe(1);
        const texture = textures[1].size as GPUExtent3DDict;
        expect(texture.width).toBe(LIGHT_GRID_TEXTURE_WIDTH);
        expect(texture.height).toBe(lightGridTextureRows(lookup.layout.cellCount));
        // The frame number is the picks' seed.
        expect(new Uint32Array(writes.at(-1)!.data)[8]).toBe(7);

        grid.encode(encoder, { ...frame, frame: 8, localLights: 9 });
        expect(new Uint32Array(writes.at(-1)!.data)[8]).toBe(8);
        // Same cells: no new texture.
        expect(grid.version).toBe(1);
        expect(dispatches).toHaveLength(4);
    });
});
