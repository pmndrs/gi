/**
 * T1.4: direct light at ray hits on a real GPU, through `giSurfaceAt` and
 * `giDirectIrradiance` compiled the way a technique compiles them.
 *
 * 1. **A sun on a plane** gives E cos(theta) at 0 to 89 degrees (relative 1e-4),
 *    with its shadow ray on (so the plane must not shadow itself), and nothing
 *    on the plane's far side.
 * 2. **Point and spot falloff match three.js r187** at sample distances: the
 *    lights are described as three's `PointLight` / `SpotLight` (intensity,
 *    distance, decay 2, angle, penumbra), converted the way the three adapter
 *    will, and compared with three's own formulas (`three-reference.ts`).
 *    Covers `range: 0`, a range window, a penumbra and a hard-edged cone, and
 *    three's 0.01 clamp on d^2.
 * 3. **Light groups:** each group alone, all four together (the sum), mask 0
 *    and bits above 3 (nothing).
 * 4. **`giSurfaceAt`:** position, interpolated normal, frontFace, uv, material
 *    and `bounceAlbedoCap` (a live setting).
 * 6. **The binding budget:** the scene copy and hit shading bind 2 storage
 *    buffers together; the test kernel (tracer, scene copy, hit shading, 2 of
 *    its own) uses 7 of a default device's 8.
 * 5. **The stochastic light choice is unbiased:** 12 shadowed point lights
 *    around a box, at most 2 or 5 shadowed per hit; the mean over 32,768
 *    random numbers per point must be within 1% and 4 standard errors of the
 *    all-lights answer (or within 1e-5, float rounding, where every light is
 *    picked every time). With the cap at or above the light count the answer is
 *    exact and the same for every random number. This is the exact path's
 *    choice (`exactLightLimit` raised to 16 for it); the light grid's is
 *    checked by `hit-shading-many-lights`.
 */

import type { LightHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import type { LightDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { countStorageBuffers, orderModules } from '../../../../packages/ddgi-core/src/platform/kernel-builder.ts';
import { createRandom } from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';
import { boxMesh } from '../trace-oracle/cases.ts';
import { assert, runGpuTest } from '../../client.ts';
import { frame, normalize, ready, relativeError, ShadeKernel, type Vec3 } from '../hit-shading-shared/harness.ts';
import { threeLightIrradiance, toPlatformLight, type ThreeLight } from './three-reference.ts';

/** Relative tolerance for closed-form answers in f32. */
const EXACT_TOLERANCE = 1e-4;
/** Below this relative difference a mean is the reference up to float rounding. */
const FLOAT_NOISE = 1e-5;
/** Random numbers per point for the unbiasedness check. */
const SEEDS_PER_POINT = 32768;

const ALL_GROUPS = 0b1111;

/**
 * Traces one ray per query and shades the hit. Inputs, 3 vec4 per query:
 * (origin, unused), (direction, unused), (random.xy, groupMask as bits, unused).
 * Outputs, 4 vec4 per query: (direct irradiance, t), (position, frontFace),
 * (shading normal, material), (albedo, instance); t < 0 for a miss.
 */
const SHADE_KERNEL = /* wgsl */ `
@group(1) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> outputs: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let query = id.x;
    if (3u * query >= arrayLength(&inputs)) {
        return;
    }
    let ray = GiRay(inputs[3u * query].xyz, 0.0, normalize(inputs[3u * query + 1u].xyz), 1.0e6);
    let extra = inputs[3u * query + 2u];
    let hit = giTraceClosest(ray);
    if (hit.t < 0.0) {
        outputs[4u * query] = vec4f(0.0, 0.0, 0.0, -1.0);
        return;
    }
    let surface = giSurfaceAt(ray, hit);
    let direct = giDirectIrradiance(surface, bitcast<u32>(extra.z), extra.xy);
    outputs[4u * query] = vec4f(direct, hit.t);
    outputs[4u * query + 1u] = vec4f(surface.position, select(0.0, 1.0, surface.frontFace));
    outputs[4u * query + 2u] = vec4f(surface.normal, f32(surface.material));
    outputs[4u * query + 3u] = vec4f(surface.albedo, f32(surface.instance));
}
`;

interface Query {
    origin: Vec3;
    direction: Vec3;
    random?: [number, number];
    groupMask?: number;
}

interface Shaded {
    direct: Vec3;
    t: number;
    position: Vec3;
    frontFace: boolean;
    normal: Vec3;
    material: number;
    albedo: Vec3;
    instance: number;
}

async function shade(kernel: ShadeKernel, queries: readonly Query[]): Promise<Shaded[]> {
    const inputs = new Float32Array(queries.length * 12);
    const words = new Uint32Array(inputs.buffer);
    queries.forEach((query, index) => {
        inputs.set(query.origin, index * 12);
        inputs.set(query.direction, index * 12 + 4);
        inputs.set(query.random ?? [0.5, 0.5], index * 12 + 8);
        words[index * 12 + 10] = query.groupMask ?? ALL_GROUPS;
    });
    const out = new Float32Array(await kernel.run(inputs, queries.length * 64, queries.length));
    return queries.map((_, index) => {
        const base = index * 16;
        const vec = (offset: number): Vec3 => [out[base + offset], out[base + offset + 1], out[base + offset + 2]];
        return {
            direct: vec(0),
            t: out[base + 3],
            position: vec(4),
            frontFace: out[base + 7] === 1,
            normal: vec(8),
            material: out[base + 11],
            albedo: vec(12),
            instance: out[base + 15],
        };
    });
}

/** A ray straight down onto the ground at (x, z). */
const down = (x: number, z: number, extra: Partial<Query> = {}): Query => ({ origin: [x, 10, z], direction: [0, -1, 0], ...extra });

/** A 400 x 400 m ground at y = 0, facing +y, with normals and uvs (uv = (x, z) / 400 + 0.5). */
function groundMesh(): { positions: Float32Array; indices: Uint32Array; normals: Float32Array; uvs: Float32Array } {
    const h = 200;
    return {
        positions: new Float32Array([-h, 0, -h, -h, 0, h, h, 0, h, h, 0, -h]),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
        normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
        uvs: new Float32Array([0, 0, 0, 1, 1, 1, 1, 0]),
    };
}

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const translation = (x: number, y: number, z: number): number[] => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];

function luminance(rgb: readonly number[]): number {
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

await runGpuTest(async ({ device, log }) => {
    const results: Record<string, unknown> = {};
    const platform = new GIPlatformImpl(device, { label: 'hit-shading-lights' });
    try {
        const scene = platform.scene;
        const ground = scene.createMesh(groundMesh());
        const groundAlbedo: Vec3 = [0.8, 0.6, 0.4];
        const groundMaterial = scene.createMaterial({ albedo: groundAlbedo });
        const groundInstance = scene.createInstance({ mesh: ground, materials: [groundMaterial], transform: IDENTITY });
        await ready(device, platform);
        const kernel = await ShadeKernel.create(device, platform, 'shade hits', SHADE_KERNEL);
        // The binding budget: this kernel is DDGI-shaped (tracer, scene copy, hit shading and two
        // buffers of its own) and compiled on a default-limits device.
        const platformBuffers = countStorageBuffers(orderModules([platform.shade.module]));
        results.storageBuffers = { platform: platformBuffers, sceneCopyAndHitShading: platform.sceneCopy.module.storageBufferCount + platform.shade.module.storageBufferCount, kernel: platformBuffers + 2, limit: device.limits.maxStorageBuffersPerShaderStage };
        assert(platform.sceneCopy.module.storageBufferCount + platform.shade.module.storageBufferCount <= 2, 'the scene copy and hit shading use more than 2 storage buffers');
        assert(platformBuffers + 2 === 7 && device.limits.maxStorageBuffersPerShaderStage === 8, `storage buffers: ${JSON.stringify(results.storageBuffers)}`);
        log(`binding budget: scene copy + hit shading ${platform.sceneCopy.module.storageBufferCount + platform.shade.module.storageBufferCount}, with the tracer ${platformBuffers}, this kernel ${platformBuffers + 2} of ${device.limits.maxStorageBuffersPerShaderStage}`);
        const groundSlot = platform.sceneCopy.instanceSlot(groundInstance);

        // 1. A sun on a plane.
        const sunColor: Vec3 = [1, 0.9, 0.8];
        const sunIrradiance = 3;
        const sun = scene.createLight({ type: 'directional', color: sunColor, intensity: sunIrradiance, direction: [0, -1, 0], castsShadows: true });
        const groundPoints = [down(0, 0), down(3.7, -1.2), down(-20, 40), down(150, -150), down(0.01, 0.02)];
        let worstSun = 0;
        for (const degrees of [0, 30, 60, 75, 89]) {
            const angle = (degrees * Math.PI) / 180;
            // Travelling down and towards +x and -z, at 'angle' from straight down.
            scene.updateLight(sun, { direction: [Math.sin(angle) * 0.6, -Math.cos(angle), -Math.sin(angle) * 0.8] });
            frame(device, platform);
            const shaded = await shade(kernel, groundPoints);
            for (const hit of shaded) {
                for (let channel = 0; channel < 3; channel++) {
                    const expected = sunIrradiance * sunColor[channel] * Math.cos(angle);
                    worstSun = Math.max(worstSun, relativeError(hit.direct[channel], expected, 1e-6));
                }
            }
            // From below, the ray meets the ground's back; the sun can't reach that side.
            const fromBelow = await shade(kernel, [{ origin: [1, -5, 2], direction: [0, 1, 0] }]);
            assert(fromBelow[0].t > 0 && !fromBelow[0].frontFace, 'a ray from below must hit the back of the ground');
            assert(fromBelow[0].direct.every((value) => value === 0), `sun at ${degrees} degrees lights the back of the ground`);
        }
        results.sunOnPlaneWorstRelativeError = worstSun;
        log(`sun on a plane, 0-89 degrees, 5 points each: worst relative error ${worstSun.toExponential(2)}`);
        assert(worstSun <= EXACT_TOLERANCE, `sun on a plane: off E cos(theta) by ${worstSun} (limit ${EXACT_TOLERANCE})`);
        scene.destroyLight(sun);

        // 4. The surface itself.
        const surfaceCheck = (await shade(kernel, [down(50, -100)]))[0];
        // The point stays on the plane, and moves along it by at most hit shading's in-surface nudge:
        // 64 rounding steps (2^-23) of its largest coordinate, here 100.
        const nudge = 64 * 100 * 2 ** -23 * 1.01;
        const along = Math.hypot(surfaceCheck.position[0] - 50, surfaceCheck.position[2] + 100);
        results.surfaceNudge = along;
        assert(along <= nudge && Math.abs(surfaceCheck.position[1]) < 1e-6, `giSurfaceAt position ${surfaceCheck.position} (allowed ${nudge} along the plane)`);
        assert(surfaceCheck.frontFace && Math.abs(surfaceCheck.normal[1] - 1) < 1e-4, `giSurfaceAt normal ${surfaceCheck.normal} / frontFace ${surfaceCheck.frontFace}`);
        assert(surfaceCheck.instance === groundSlot && surfaceCheck.material >= 0, 'giSurfaceAt instance or material');
        assert(surfaceCheck.albedo.every((value, channel) => Math.abs(value - groundAlbedo[channel]) < 1e-6), `albedo ${surfaceCheck.albedo}`);
        const backSurface = (await shade(kernel, [{ origin: [1, -5, 2], direction: [0, 1, 0] }]))[0];
        assert(Math.abs(backSurface.normal[1] + 1) < 1e-4, `a back-face hit's shading normal must face the ray: ${backSurface.normal}`);
        platform.shade.settings.bounceAlbedoCap = 0.7;
        frame(device, platform);
        const capped = (await shade(kernel, [down(1, 1)]))[0];
        assert(capped.albedo.every((value, channel) => Math.abs(value - Math.min(0.7, groundAlbedo[channel])) < 1e-6), `bounceAlbedoCap 0.7 gave albedo ${capped.albedo}`);
        platform.shade.settings.bounceAlbedoCap = 1;
        results.surface = 'ok';

        // 2. Point and spot lights, as three.js describes them.
        const threeLights: ThreeLight[] = [
            // Group 0: no range window; 5 cm above the ground, so the point under it is inside three's d^2 >= 0.01 clamp.
            { type: 'point', color: [1, 0.9, 0.7], intensity: 20, position: [0, 0.05, 0], distance: 0, decay: 2 },
            // Group 1: windowed at 8 m.
            { type: 'point', color: [0.6, 0.8, 1], intensity: 40, position: [0, 2, 0], distance: 8, decay: 2 },
            // Group 2: a spot with a penumbra, pointing straight down.
            { type: 'spot', color: [1, 1, 1], intensity: 60, position: [0, 4, 0], target: [0, 0, 0], distance: 12, decay: 2, angle: 0.6, penumbra: 0.4 },
            // Group 3: a hard-edged spot, tilted, no range.
            { type: 'spot', color: [1, 0.5, 0.25], intensity: 30, position: [1, 3, 1], target: [2, 0, 0.5], distance: 0, decay: 2, angle: 0.5, penumbra: 0 },
        ];
        const handles: LightHandle[] = threeLights.map((light, group) => scene.createLight({ ...toPlatformLight(light), group, castsShadows: true } as LightDescriptor));
        frame(device, platform);
        // Points on the ground at many distances and directions from the lights (none exactly on a cone edge).
        const offsets = [0, 0.03, 0.1, 0.5, 1, 1.7, 2.5, 3.3, 4.1, 5, 6.2, 7.5, 7.9, 8.5, 10, 13, 25];
        const falloffPoints: Query[] = [];
        offsets.forEach((offset, index) => {
            const azimuth = index * 2.39996;
            falloffPoints.push(down(offset * Math.cos(azimuth), offset * Math.sin(azimuth)));
        });
        let worstFalloff = 0;
        const perGroup: Vec3[][] = [];
        for (let group = 0; group < 4; group++) {
            const shaded = await shade(kernel, falloffPoints.map((query) => ({ ...query, groupMask: 1 << group })));
            perGroup.push(shaded.map((hit) => hit.direct));
            shaded.forEach((hit, index) => {
                const expected = threeLightIrradiance(threeLights[group], hit.position, [0, 1, 0]);
                const scale = threeLights[group].intensity;
                for (let channel = 0; channel < 3; channel++) {
                    worstFalloff = Math.max(worstFalloff, relativeError(hit.direct[channel], expected[channel], 1e-6 * scale));
                }
                if (index === 0 && group === 0) results.clampedPointUnderLight = { found: hit.direct, expected };
            });
        }
        results.pointSpotWorstRelativeError = worstFalloff;
        log(`point and spot lights against three.js r187's formulas, ${offsets.length} points x 4 lights: worst relative error ${worstFalloff.toExponential(2)}`);
        assert(worstFalloff <= EXACT_TOLERANCE, `point/spot falloff differs from three.js by ${worstFalloff}`);

        // 3. Light groups.
        const allGroups = await shade(kernel, falloffPoints);
        let worstSum = 0;
        allGroups.forEach((hit, index) => {
            for (let channel = 0; channel < 3; channel++) {
                const sum = perGroup.reduce((total, group) => total + group[index][channel], 0);
                worstSum = Math.max(worstSum, relativeError(hit.direct[channel], sum, 1e-4));
            }
        });
        const noGroups = await shade(kernel, falloffPoints.map((query) => ({ ...query, groupMask: 0 })));
        const highBits = await shade(kernel, falloffPoints.map((query) => ({ ...query, groupMask: 0xfffffff0 })));
        results.groupSumWorstRelativeError = worstSum;
        assert(worstSum <= EXACT_TOLERANCE, `all four groups together differ from the sum of each by ${worstSum}`);
        assert(noGroups.every((hit) => hit.direct.every((value) => value === 0)), 'group mask 0 must give no light');
        assert(highBits.every((hit) => hit.direct.every((value) => value === 0)), 'group bits above 3 must give no light');
        log(`light groups: the four together equal their sum (worst ${worstSum.toExponential(2)}); mask 0 and bits 4-31 give nothing`);
        for (const handle of handles) scene.destroyLight(handle);

        // 5. The stochastic light choice.
        const boxGeometry = boxMesh([0.6, 0.6, 0.6]);
        const box = scene.createMesh({ positions: boxGeometry.positions, indices: boxGeometry.indices as Uint32Array });
        const boxMaterial = scene.createMaterial({ albedo: [0.5, 0.5, 0.5] });
        scene.createInstance({ mesh: box, materials: [boxMaterial], transform: translation(0.4, 0.6, -0.2) });
        const random = createRandom(1234);
        const stochasticLights: LightHandle[] = [];
        for (let index = 0; index < 12; index++) {
            const angle = (index / 12) * 2 * Math.PI + random() * 0.4;
            const radius = 1.2 + random() * 3;
            stochasticLights.push(scene.createLight({
                type: 'point',
                color: [0.5 + random() * 0.5, 0.5 + random() * 0.5, 0.5 + random() * 0.5],
                intensity: 1 + random() * 29,
                position: [Math.cos(angle) * radius, 0.4 + random() * 2.5, Math.sin(angle) * radius],
                range: index % 3 === 0 ? 0 : 6 + random() * 6,
                castsShadows: true,
            }));
        }
        // A sun too: directional lights are always evaluated, outside the cap.
        stochasticLights.push(scene.createLight({ type: 'directional', color: [1, 1, 1], intensity: 2, direction: normalize([0.3, -1, 0.2]), castsShadows: true }));
        // This checks the exact path's stochastic choice over every light, so its 12 lights stay below
        // exactLightLimit. With the light grid (more than 8 by default) one frame's grid is a single random
        // draw, and only a new grid every frame averages it out (hit-shading-many-lights checks that).
        platform.shade.settings.exactLightLimit = 16;
        await ready(device, platform);
        const points: Query[] = [down(0, 0), down(-1, 0.5), down(1.5, -1), down(0.4, 1.1), down(-2, -2), down(2.5, 0.3), down(0.9, -0.2), down(-0.3, -1.3)];

        platform.shade.settings.maxShadowedLightsPerHit = 64;
        frame(device, platform);
        const exact = await shade(kernel, points);
        // At or above the light count nothing is random: another random number gives the very same answer.
        const exactAgain = await shade(kernel, points.map((query) => ({ ...query, random: [0.123, 0.987] })));
        assert(exact.every((hit, index) => hit.direct.every((value, channel) => value === exactAgain[index].direct[channel])), 'with the cap above the light count the answer must not depend on the random numbers');
        // The box must really shadow some of these points from some lights, or visibility would play no part.
        for (const handle of stochasticLights) scene.updateLight(handle, { castsShadows: false });
        frame(device, platform);
        const unshadowed = await shade(kernel, points);
        const shadowedPoints = exact.filter((hit, index) => luminance(hit.direct) < 0.99 * luminance(unshadowed[index].direct)).length;
        results.pointsPartlyInShadow = shadowedPoints;
        assert(shadowedPoints >= 3, `only ${shadowedPoints} of ${points.length} points are partly in the box's shadow`);
        for (const handle of stochasticLights) scene.updateLight(handle, { castsShadows: true });

        const unbiased: Record<string, unknown> = {};
        for (const cap of [2, 5]) {
            platform.shade.settings.maxShadowedLightsPerHit = cap;
            frame(device, platform);
            const queries: Query[] = [];
            const seeds = createRandom(99 + cap);
            for (const point of points) {
                for (let seed = 0; seed < SEEDS_PER_POINT; seed++) {
                    // Exactly representable in f32 and below 1, as the contract's random is [0, 1).
                    queries.push({ ...point, random: [Math.floor(seeds() * 2 ** 24) / 2 ** 24, Math.floor(seeds() * 2 ** 24) / 2 ** 24] });
                }
            }
            const shaded = await shade(kernel, queries);
            let worstRelative = 0;
            let worstStandardErrors = 0;
            points.forEach((_, pointIndex) => {
                const reference = luminance(exact[pointIndex].direct);
                const values = Array.from({ length: SEEDS_PER_POINT }, (_, seed) => luminance(shaded[pointIndex * SEEDS_PER_POINT + seed].direct));
                const mean = values.reduce((total, value) => total + value, 0) / SEEDS_PER_POINT;
                const variance = values.reduce((total, value) => total + (value - mean) ** 2, 0) / (SEEDS_PER_POINT - 1);
                const standardError = Math.sqrt(variance / SEEDS_PER_POINT);
                const relative = Math.abs(mean / reference - 1);
                // Where every light that reaches the point is picked every time (few contributing lights, or
                // each holding a share of at least 1/K), the samples differ only by float rounding, around
                // 1e-7, and so does the mean: within FLOAT_NOISE of the reference counts as exact there.
                const inStandardErrors = standardError > 0 ? Math.abs(mean - reference) / standardError : 0;
                const withinFloatNoise = relative <= FLOAT_NOISE;
                worstRelative = Math.max(worstRelative, relative);
                if (!withinFloatNoise) worstStandardErrors = Math.max(worstStandardErrors, inStandardErrors);
                assert(withinFloatNoise || (relative <= 0.01 && inStandardErrors <= 4), `cap ${cap}, point ${pointIndex}: mean ${mean} against the all-lights ${reference} (${(relative * 100).toFixed(3)}%, ${inStandardErrors.toFixed(2)} standard errors)`);
            });
            unbiased[`cap ${cap}`] = { worstRelative, worstStandardErrors };
            log(`stochastic choice, at most ${cap} of 12 shadowed point lights: mean over ${SEEDS_PER_POINT} random numbers within ${(worstRelative * 100).toFixed(3)}% (${worstStandardErrors.toFixed(2)} standard errors) of the all-lights answer`);
        }
        results.unbiased = unbiased;
        results.exactLuminance = exact.map((hit) => luminance(hit.direct));
    } finally {
        platform.dispose();
    }
    return results;
});
