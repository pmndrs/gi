/**
 * T1.7: three's sky and environment, through the adapter, on a real GPU.
 *
 * **A. GI's sky is the sky three shows.** For three sky textures, three
 * draws `scene.background` and the page reads the pixel at the centre of a
 * narrow camera looking along d; the adapter's scene sync mirrors the same
 * background into a platform, and a hit-shading kernel reads
 * `giSkyRadiance(d)`. They must agree within `SKY_TOLERANCE` along
 * directions chosen at the centres of uniform regions of the texture:
 * - a half-float **cube** whose faces are split left and right into two
 *   colours, so a missing x mirror (left and right swapped, ±X faces swapped)
 *   fails;
 * - the same cube from **canvas images** (8-bit sRGB, the image path);
 * - a half-float **equirect** in eight coloured blocks;
 * each also with `backgroundRotation.y` and `backgroundIntensity`.
 *
 * **B. Environment diffuse isn't counted twice.** A white rough
 * `MeshStandardMaterial` floor inside a DDGI volume under a flat
 * `scene.environment`. GI sees that environment as its sky, so on the open
 * floor GI's irradiance is about the environment's own. Measured at the
 * floor's centre (coverage 1):
 * - environment alone (GI light at intensity 0) = env diffuse + specular;
 * - GI with the workaround (the default) should read about the same: GI
 *   replaces the env diffuse (within `REPLACE_TOLERANCE`);
 * - a GI light without the workaround (`replaceEnvironmentDiffuse: false`)
 *   reads about twice the diffuse: the double count the workaround removes.
 * - A Lambert floor gets no diffuse light from the environment at all (its
 *   lighting model only reflects it), so it has nothing to double count.
 */

import {
    BoxGeometry,
    CubeTexture,
    DataTexture,
    EquirectangularReflectionMapping,
    HalfFloatType,
    LinearFilter,
    Mesh,
    MeshLambertMaterial,
    MeshStandardMaterial,
    PerspectiveCamera,
    PlaneGeometry,
    RGBAFormat,
    RenderTarget,
    Scene,
    SRGBColorSpace,
    Vector3,
    type Texture,
    type WebGPURenderer,
} from 'three/webgpu';
import { DataUtils, FloatType } from 'three';
import { assert } from '../../client.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { SceneSync } from '../../../../packages/ddgi-three/src/scene-sync.ts';
import { GILight } from '../../../../packages/ddgi-three/src/gi-light.ts';
import { ThreeGI } from '../../../../packages/ddgi-three/src/three-gi.ts';
import { frame, ready, ShadeKernel, type Vec3 } from '../hit-shading-shared/harness.ts';
import { runSteps, runThreePage } from '../three-shared/harness.ts';

/** Three's background against GI's sky, relative per channel (plus an absolute floor of 1% of the brightest channel). */
const SKY_TOLERANCE = 0.02;
/** GI replacing the environment's diffuse light: the floor's brightness against the environment's own. */
const REPLACE_TOLERANCE = 0.05;
/** Without the workaround the floor must read at least this much brighter than the environment alone. */
const DOUBLE_COUNT_AT_LEAST = 1.6;

const CUBE_SIZE = 4;
const EQUIRECT_WIDTH = 16;
const EQUIRECT_HEIGHT = 8;

// ---------------------------------------------------------------------------
// Test textures
// ---------------------------------------------------------------------------

/** Linear colours of the left and right halves of cube face `face`, all different. */
function faceColours(face: number): [Vec3, Vec3] {
    return [
        [0.1 + 0.12 * face, 0.25, 0.9 - 0.1 * face],
        [0.85 - 0.1 * face, 0.6, 0.15 + 0.12 * face],
    ];
}

function halfFaceTexture(face: number): DataTexture {
    const [left, right] = faceColours(face);
    const data = new Uint16Array(CUBE_SIZE * CUBE_SIZE * 4);
    for (let row = 0; row < CUBE_SIZE; row++) {
        for (let column = 0; column < CUBE_SIZE; column++) {
            const colour = column < CUBE_SIZE / 2 ? left : right;
            data.set([...colour.map((value) => DataUtils.toHalfFloat(value)), DataUtils.toHalfFloat(1)], (row * CUBE_SIZE + column) * 4);
        }
    }
    const texture = new DataTexture(data, CUBE_SIZE, CUBE_SIZE, RGBAFormat, HalfFloatType);
    texture.needsUpdate = true;
    return texture;
}

function dataCube(): CubeTexture {
    const cube = new CubeTexture(Array.from({ length: 6 }, (_, face) => halfFaceTexture(face)) as never);
    cube.type = HalfFloatType;
    cube.format = RGBAFormat;
    cube.magFilter = LinearFilter;
    cube.minFilter = LinearFilter;
    cube.generateMipmaps = false;
    cube.needsUpdate = true;
    return cube;
}

function srgbByte(linear: number): number {
    const encoded = linear <= 0.0031308 ? linear * 12.92 : 1.055 * Math.pow(linear, 1 / 2.4) - 0.055;
    return Math.round(Math.min(Math.max(encoded, 0), 1) * 255);
}

function imageCube(): CubeTexture {
    const faces = Array.from({ length: 6 }, (_, face) => {
        const canvas = document.createElement('canvas');
        canvas.width = CUBE_SIZE;
        canvas.height = CUBE_SIZE;
        const context = canvas.getContext('2d')!;
        faceColours(face).forEach((colour, half) => {
            context.fillStyle = `rgb(${colour.map(srgbByte).join(',')})`;
            context.fillRect((half * CUBE_SIZE) / 2, 0, CUBE_SIZE / 2, CUBE_SIZE);
        });
        return canvas;
    });
    const cube = new CubeTexture(faces as never);
    cube.colorSpace = SRGBColorSpace;
    cube.magFilter = LinearFilter;
    cube.minFilter = LinearFilter;
    cube.generateMipmaps = false;
    cube.needsUpdate = true;
    return cube;
}

/** Eight blocks: 4 across (u) by 2 down (v), each a different colour. */
function blockColour(column: number, row: number): Vec3 {
    return [0.1 + 0.2 * column, 0.2 + 0.5 * row, 0.9 - 0.15 * column - 0.3 * row];
}

function equirect(): DataTexture {
    const data = new Uint16Array(EQUIRECT_WIDTH * EQUIRECT_HEIGHT * 4);
    for (let row = 0; row < EQUIRECT_HEIGHT; row++) {
        for (let column = 0; column < EQUIRECT_WIDTH; column++) {
            const colour = blockColour(Math.floor((column * 4) / EQUIRECT_WIDTH), Math.floor((row * 2) / EQUIRECT_HEIGHT));
            data.set([...colour.map((value) => DataUtils.toHalfFloat(value)), DataUtils.toHalfFloat(1)], (row * EQUIRECT_WIDTH + column) * 4);
        }
    }
    const texture = new DataTexture(data, EQUIRECT_WIDTH, EQUIRECT_HEIGHT, RGBAFormat, HalfFloatType);
    texture.mapping = EquirectangularReflectionMapping;
    texture.magFilter = LinearFilter;
    texture.minFilter = LinearFilter;
    texture.needsUpdate = true;
    return texture;
}

// ---------------------------------------------------------------------------
// Directions at the centres of uniform regions
// ---------------------------------------------------------------------------

/** The direction to (s, t) on a cube face, WebGPU's face table (t = 0 at the top). Mirroring maps these centres onto centres. */
function cubeDirection(face: number, s: number, t: number): Vec3 {
    const sc = 2 * s - 1;
    const tc = 2 * t - 1;
    const directions: Vec3[] = [
        [1, -tc, -sc],
        [-1, -tc, sc],
        [sc, 1, tc],
        [sc, -1, -tc],
        [sc, -tc, 1],
        [-sc, -tc, -1],
    ];
    return normalize(directions[face]);
}

function cubeRegionDirections(): Vec3[] {
    const directions: Vec3[] = [];
    for (let face = 0; face < 6; face++) for (const s of [0.25, 0.75]) directions.push(cubeDirection(face, s, 0.5));
    return directions;
}

/** three's equirectUV, inverted: u = atan2(z, x) / 2π + 0.5, v = asin(y) / π + 0.5. */
function equirectDirection(u: number, v: number): Vec3 {
    const longitude = (u - 0.5) * 2 * Math.PI;
    const latitude = (v - 0.5) * Math.PI;
    return [Math.cos(latitude) * Math.cos(longitude), Math.sin(latitude), Math.cos(latitude) * Math.sin(longitude)];
}

function equirectRegionDirections(): Vec3[] {
    const directions: Vec3[] = [];
    for (let row = 0; row < 2; row++) for (let column = 0; column < 4; column++) directions.push(equirectDirection((column + 0.5) / 4, (row + 0.5) / 2 + (row === 0 ? 0.1 : -0.1)));
    return directions;
}

function normalize(vector: readonly number[]): Vec3 {
    const length = Math.hypot(vector[0], vector[1], vector[2]);
    return [vector[0] / length, vector[1] / length, vector[2] / length];
}

/** R_y(angle) d: where a rotation by `angle` moves what the unturned texture shows along d. */
function rotateY(direction: Vec3, angle: number): Vec3 {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return [c * direction[0] + s * direction[2], direction[1], -s * direction[0] + c * direction[2]];
}

// ---------------------------------------------------------------------------
// Reading three's background and GI's sky
// ---------------------------------------------------------------------------

const PROBE_SIZE = 8;

/** What three draws along each direction: the centre of a narrow camera's view of the background. */
async function threeBackground(renderer: WebGPURenderer, scene: Scene, directions: readonly Vec3[]): Promise<Vec3[]> {
    const camera = new PerspectiveCamera(2, 1, 0.1, 10);
    const target = new RenderTarget(PROBE_SIZE, PROBE_SIZE, { type: FloatType });
    const results: Vec3[] = [];
    for (const direction of directions) {
        camera.quaternion.setFromUnitVectors(new Vector3(0, 0, -1), new Vector3(...direction));
        camera.updateMatrixWorld();
        renderer.setRenderTarget(target);
        renderer.render(scene, camera);
        renderer.setRenderTarget(null);
        const pixels = (await renderer.readRenderTargetPixelsAsync(target, PROBE_SIZE / 2 - 1, PROBE_SIZE / 2 - 1, 2, 2)) as Float32Array;
        const mean: Vec3 = [0, 0, 0];
        for (let pixel = 0; pixel < 4; pixel++) for (let channel = 0; channel < 3; channel++) mean[channel] += pixels[pixel * 4 + channel] / 4;
        results.push(mean);
    }
    target.dispose();
    return results;
}

const SKY_KERNEL = /* wgsl */ `
@group(1) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> outputs: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= arrayLength(&inputs)) {
        return;
    }
    outputs[id.x] = vec4f(giSkyRadiance(normalize(inputs[id.x].xyz)), 0.0);
}
`;

async function giSky(kernel: ShadeKernel, directions: readonly Vec3[]): Promise<Vec3[]> {
    const inputs = new Float32Array(directions.length * 4);
    directions.forEach((direction, index) => inputs.set(direction, index * 4));
    const out = new Float32Array(await kernel.run(inputs, directions.length * 16, directions.length));
    return directions.map((_, index) => [out[index * 4], out[index * 4 + 1], out[index * 4 + 2]]);
}

function worstDifference(found: readonly Vec3[], expected: readonly Vec3[]): number {
    let worst = 0;
    found.forEach((colour, index) => {
        const reference = expected[index];
        const floor = 0.01 * Math.max(...reference);
        for (let channel = 0; channel < 3; channel++) worst = Math.max(worst, Math.abs(colour[channel] - reference[channel]) / Math.max(Math.abs(reference[channel]), floor));
    });
    return worst;
}

function format(colours: readonly Vec3[]): string {
    return colours.map((colour) => `(${colour.map((value) => value.toFixed(3)).join(' ')})`).join(' ');
}

await runThreePage(async ({ device, renderer, data, log }) => {
    // ---- A. GI's sky is the sky three shows ---------------------------------
    const platform = new GIPlatformImpl(device, { label: 'three environment' });
    await ready(device, platform);
    const sync = new SceneSync(platform.scene, { device, warn: (message) => log(`warning: ${message}`) });
    const kernel = await ShadeKernel.create(device, platform, 'three sky check', SKY_KERNEL);
    const skyChecks: Record<string, unknown>[] = [];
    data.sky = skyChecks;

    const cases: { name: string; texture: Texture; directions: Vec3[]; rotation: number; intensity: number }[] = [];
    for (const [name, make, directions] of [
        ['half-float cube', dataCube, cubeRegionDirections()],
        ['8-bit sRGB image cube', imageCube, cubeRegionDirections()],
        ['half-float equirect', equirect, equirectRegionDirections()],
    ] as const) {
        const texture = make();
        cases.push({ name, texture, directions: [...directions], rotation: 0, intensity: 1 });
        // Turned and brightened: the directions turn with the sky, so they stay at region centres.
        cases.push({ name: `${name}, rotated 0.6 rad, intensity 1.5`, texture, directions: directions.map((direction) => rotateY(direction, 0.6)), rotation: 0.6, intensity: 1.5 });
    }

    for (const check of cases) {
        const scene = new Scene();
        scene.background = check.texture;
        scene.backgroundRotation.set(0, check.rotation, 0);
        scene.backgroundIntensity = check.intensity;
        sync.update(scene);
        frame(device, platform); // projects the sky
        const three = await threeBackground(renderer, scene, check.directions);
        const gi = await giSky(kernel, check.directions);
        const worst = worstDifference(gi, three);
        skyChecks.push({ name: check.name, worst, three, gi });
        log(`${check.name}: worst ${(worst * 100).toFixed(2)}%`);
        assert(Math.max(...three.flat()) > 0.05, `${check.name}: three drew a black background`);
        assert(worst <= SKY_TOLERANCE, `${check.name}: GI's sky differs from three's background by up to ${(worst * 100).toFixed(1)}%. three ${format(three)}; GI ${format(gi)}`);
    }
    data.environmentUpdates = sync.environmentUpdates;
    sync.dispose();
    platform.dispose();

    // ---- B. Environment diffuse isn't counted twice -------------------------
    const scene = new Scene();
    const flat = new DataTexture(new Uint16Array(8 * 4 * 4).fill(DataUtils.toHalfFloat(0.5)), 8, 4, RGBAFormat, HalfFloatType);
    flat.mapping = EquirectangularReflectionMapping;
    flat.magFilter = LinearFilter;
    flat.minFilter = LinearFilter;
    flat.needsUpdate = true;
    scene.environment = flat;
    const standard = new MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 });
    const floor = new Mesh<PlaneGeometry, MeshStandardMaterial | MeshLambertMaterial>(new PlaneGeometry(40, 40), standard);
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);
    // A thin box under the floor, so rays from below see a closed underside, not the sky.
    const slab = new Mesh(new BoxGeometry(40, 0.1, 40), new MeshLambertMaterial({ color: 0x000000 }));
    slab.position.y = -0.06;
    scene.add(slab);
    const camera = new PerspectiveCamera(20, 1, 0.1, 100);
    camera.position.set(0, 4, 3);
    camera.lookAt(0, 0, 0);

    const giPlatform = new GIPlatformImpl(device, { label: 'three environment diffuse' });
    const ddgi = giPlatform.addTechnique(new DDGITechnique({ volume: { center: [0, 1.25, 0], size: [4, 2, 4], spacing: 0.5 } }));
    giPlatform.setOutput(ddgi);
    await giPlatform.prepare();
    const threeGI = new ThreeGI(giPlatform, scene, { budget: { units: 0 }, warn: (message) => log(`warning: ${message}`) });
    threeGI.update(camera, 1 / 60);
    await giPlatform.rays.whenBuilt();
    threeGI.update(camera, 1 / 60);
    threeGI.budget = { units: ddgi.field!.grid.probeCount };
    await runSteps(device, 300, () => threeGI.update(camera, 1 / 60));
    assert(threeGI.light.computeCoverage, 'coverage should be computed when scene.environment is set');

    const target = new RenderTarget(64, 64, { type: FloatType, depthBuffer: true });
    const floorCentre = async (): Promise<number> => {
        await renderer.compileAsync(scene, camera);
        renderer.setRenderTarget(target);
        renderer.render(scene, camera);
        renderer.setRenderTarget(null);
        const pixels = (await renderer.readRenderTargetPixelsAsync(target, 28, 28, 8, 8)) as Float32Array;
        let sum = 0;
        for (let pixel = 0; pixel < 64; pixel++) sum += pixels[pixel * 4 + 1];
        return sum / 64;
    };
    threeGI.light.intensity = 0;
    const environmentAlone = await floorCentre();
    threeGI.light.intensity = 1;
    const withWorkaround = await floorCentre();
    // A second GI light without the workaround, as a host could add; the adapter's own goes dark.
    // (A light's workaround setting is fixed for its life: three caches shaders by light id.)
    const plainLight = new GILight(giPlatform.shading, { replaceEnvironmentDiffuse: false });
    plainLight.computeCoverage = true;
    threeGI.light.intensity = 0;
    scene.add(plainLight);
    const withoutWorkaround = await floorCentre();
    plainLight.removeFromParent();
    threeGI.light.intensity = 1;
    // GI alone, without the environment: what GI adds to the floor.
    scene.environment = null;
    const giAlone = await floorCentre();
    scene.environment = flat;
    floor.material = new MeshLambertMaterial({ color: 0xffffff });
    const lambertWithGI = await floorCentre();
    threeGI.light.intensity = 0;
    const lambertEnvironmentAlone = await floorCentre();
    target.dispose();
    threeGI.dispose();
    giPlatform.dispose();

    const diffuse = {
        environmentAlone,
        giAlone,
        withWorkaround,
        withoutWorkaround,
        workaroundOverEnvironment: withWorkaround / environmentAlone,
        noWorkaroundOverEnvironment: withoutWorkaround / environmentAlone,
        lambertWithGI,
        lambertEnvironmentAlone,
    };
    data.environmentDiffuse = diffuse;
    log(`floor (green): env alone ${environmentAlone.toFixed(4)}, GI + workaround ${withWorkaround.toFixed(4)}, GI without ${withoutWorkaround.toFixed(4)}; Lambert: GI ${lambertWithGI.toFixed(4)}, env alone ${lambertEnvironmentAlone.toFixed(4)}`);
    assert(environmentAlone > 0.1, `the environment alone lit the floor with only ${environmentAlone}`);
    assert(Math.abs(diffuse.workaroundOverEnvironment - 1) <= REPLACE_TOLERANCE, `with the workaround the floor reads ${(diffuse.workaroundOverEnvironment * 100).toFixed(1)}% of the environment alone; GI should replace its diffuse (±${REPLACE_TOLERANCE * 100}%)`);
    assert(diffuse.noWorkaroundOverEnvironment >= DOUBLE_COUNT_AT_LEAST, `without the workaround the floor reads only ${diffuse.noWorkaroundOverEnvironment.toFixed(2)}x the environment alone; expected the double count`);
    assert(lambertEnvironmentAlone < 0.02 * lambertWithGI, `a Lambert floor got ${lambertEnvironmentAlone} from the environment alone; expected no environment diffuse`);
});
