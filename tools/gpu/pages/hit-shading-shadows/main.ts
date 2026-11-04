/**
 * T1.4: shadow rays agree with the oracle. For every trace-oracle scene
 * (thin walls 1 mm apart, mirrored boxes, a closed thin box, instancing,
 * random soups, a deep mesh), each oracle ray is traced and shaded on the GPU
 * by `giSurfaceAt` and `giDirectIrradiance`, for three lights in turn (light
 * groups): a point light outside the scene, a point light inside the first
 * instance, and a sun.
 *
 * Visibility on the GPU is the shadowed answer over the unshadowed one (the
 * same lights with `castsShadows: false`), which must be exactly 0 or 1. It is
 * compared with the CPU reference tracer (double precision, every triangle)
 * from the very origin hit shading used (`giShadeOffsetRayOrigin`, read back)
 * towards the same light. As in the trace oracle, a disagreement is excused
 * only where the reference marks the ray ambiguous (within rounding distance
 * of an edge).
 *
 * Extra checks the comparison alone can't make:
 * - **No self-shadowing:** the reference's nearest hit along a shadow ray must
 *   never be the triangle the ray left from (an origin on the wrong side of
 *   its own triangle would hit it; the light is always on the ray's side).
 * - **The closed thin box doesn't leak:** every hit inside it is dark under the
 *   outside light and the sun, ambiguous or not.
 * - Each scene has both lit and shadowed hits, so the comparison means something.
 */

import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { buildReferenceScene, traceAnyReference, traceClosestReference } from '../../../../packages/ddgi-core/src/platform/trace/reference-tracer.ts';
import { invertAffine } from '../../../../packages/ddgi-core/src/platform/trace/top-level.ts';
import { assert, runGpuTest } from '../../client.ts';
import { buildOracleCases, FULL_SIZES, transformPoint, type OracleCase } from '../trace-oracle/cases.ts';
import { frame, normalize, ready, ShadeKernel, type Vec3 } from '../hit-shading-shared/harness.ts';

/** Share of shaded light samples per scene that may be excused as ambiguous. */
const MAX_AMBIGUOUS_SHARE = 0.01;

const LIGHT_NAMES = ['point light outside', 'point light inside the first instance', 'sun'] as const;
const SUN_DIRECTION = normalize([-0.3, -1, -0.45]);

/**
 * Per oracle ray: trace, shade, and for each light group alone, direct
 * irradiance. Inputs: GiRay (2 vec4). Outputs, 5 vec4 per ray: the three
 * groups' irradiance (in w: t, or -1 for a miss; the instance slot; the
 * triangle), the shadow-ray origin hit shading uses, the shading position.
 */
const SHADOW_KERNEL = /* wgsl */ `
@group(1) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> outputs: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (2u * index >= arrayLength(&inputs)) {
        return;
    }
    let ray = GiRay(inputs[2u * index].xyz, inputs[2u * index].w, inputs[2u * index + 1u].xyz, inputs[2u * index + 1u].w);
    let hit = giTraceClosest(ray);
    if (hit.t < 0.0) {
        outputs[5u * index] = vec4f(0.0, 0.0, 0.0, -1.0);
        return;
    }
    let surface = giSurfaceAt(ray, hit);
    let random = vec2f(0.5);
    outputs[5u * index] = vec4f(giDirectIrradiance(surface, 1u, random), hit.t);
    outputs[5u * index + 1u] = vec4f(giDirectIrradiance(surface, 2u, random), 0.0);
    outputs[5u * index + 2u] = vec4f(giDirectIrradiance(surface, 4u, random), 0.0);
    outputs[5u * index + 3u] = vec4f(giShadeOffsetRayOrigin(surface.position, giShadeRaySide(surface)), 0.0);
    outputs[5u * index + 4u] = vec4f(surface.position, 0.0);
    outputs[5u * index + 1u].w = f32(hit.instance);
    outputs[5u * index + 2u].w = f32(hit.triangle);
    outputs[5u * index + 3u].w = hit.barycentrics.x;
    outputs[5u * index + 4u].w = hit.barycentrics.y;
}
`;

interface CaseLights {
    outside: Vec3;
    inside: Vec3;
}

/** Lights placed from the scene's world bounds and its first instance. */
function lightsFor(oracleCase: OracleCase): CaseLights {
    const min: Vec3 = [Infinity, Infinity, Infinity];
    const max: Vec3 = [-Infinity, -Infinity, -Infinity];
    for (const instance of oracleCase.instances) {
        const positions = oracleCase.meshes[instance.meshIndex].positions;
        for (let vertex = 0; vertex < positions.length; vertex += 3) {
            const point = transformPoint(Array.from(instance.objectToWorld), [positions[vertex], positions[vertex + 1], positions[vertex + 2]]);
            for (let axis = 0; axis < 3; axis++) {
                min[axis] = Math.min(min[axis], point[axis]);
                max[axis] = Math.max(max[axis], point[axis]);
            }
        }
    }
    const centre = min.map((value, axis) => (value + max[axis]) / 2) as Vec3;
    const extent = max.map((value, axis) => value - min[axis]) as Vec3;
    // Above and to one side, well clear of every surface.
    const outside: Vec3 = [centre[0] + 0.3 * extent[0] + 0.7, max[1] + 0.6 * extent[1] + 1.3, centre[2] + 0.2 * extent[2] + 0.9];
    // Near the first instance's own origin, nudged off any plane through it
    // (on thin-wall it sits between the two skins, 0.5 mm from each).
    const origin = transformPoint(Array.from(oracleCase.instances[0].objectToWorld), [0, 0, 0]);
    const inside: Vec3 = [origin[0] + 0.0005, origin[1] + 0.0131, origin[2] + 0.0213];
    return { outside, inside };
}

const luminance = (rgb: ArrayLike<number>, offset: number): number => 0.2126 * rgb[offset] + 0.7152 * rgb[offset + 1] + 0.0722 * rgb[offset + 2];

interface CaseReport {
    rays: number;
    hits: number;
    lit: number[];
    shadowed: number[];
    ambiguous: number;
    mismatches: number;
    selfHits: number;
    leaks: number;
}

async function checkCase(device: GPUDevice, oracleCase: OracleCase, log: (message: string) => void): Promise<CaseReport> {
    const platform = new GIPlatformImpl(device, { label: `shadows ${oracleCase.name}` });
    try {
        const scene = platform.scene;
        const material = scene.createMaterial({ albedo: [0.5, 0.5, 0.5] });
        const meshes = oracleCase.meshes.map((mesh) =>
            scene.createMesh({ positions: mesh.positions, indices: mesh.indices ?? Uint32Array.from({ length: mesh.positions.length / 3 }, (_, index) => index) }),
        );
        const caseIndexBySlot = new Map<number, number>();
        for (const instance of oracleCase.instances) {
            const handle = scene.createInstance({ mesh: meshes[instance.meshIndex], materials: [material], transform: instance.objectToWorld, visible: instance.visible, contributesToGI: instance.contributesToGi });
            caseIndexBySlot.set(platform.sceneCopy.instanceSlot(handle), instance.instanceIndex);
        }
        const lights = lightsFor(oracleCase);
        const handles = [
            scene.createLight({ type: 'point', color: [1, 1, 1], intensity: 10, position: lights.outside, range: 0, group: 0, castsShadows: true }),
            scene.createLight({ type: 'point', color: [1, 1, 1], intensity: 10, position: lights.inside, range: 0, group: 1, castsShadows: true }),
            scene.createLight({ type: 'directional', color: [1, 1, 1], intensity: 3, direction: SUN_DIRECTION, group: 2, castsShadows: true }),
        ];
        await ready(device, platform);
        const kernel = await ShadeKernel.create(device, platform, `shadows ${oracleCase.name}`, SHADOW_KERNEL);
        const rayCount = oracleCase.rays.length / 8;
        const shadowed = new Float32Array(await kernel.run(oracleCase.rays, rayCount * 80, rayCount));
        for (const handle of handles) scene.updateLight(handle, { castsShadows: false });
        frame(device, platform);
        const unshadowed = new Float32Array(await kernel.run(oracleCase.rays, rayCount * 80, rayCount));

        const reference = buildReferenceScene(oracleCase.meshes, oracleCase.instances);
        const report: CaseReport = { rays: rayCount, hits: 0, lit: [0, 0, 0], shadowed: [0, 0, 0], ambiguous: 0, mismatches: 0, selfHits: 0, leaks: 0 };
        const examples: string[] = [];
        const leakExamples: string[] = [];
        let samples = 0;
        for (let ray = 0; ray < rayCount; ray++) {
            const base = ray * 20;
            if (!(shadowed[base + 3] >= 0)) continue;
            report.hits++;
            const hitInstance = caseIndexBySlot.get(shadowed[base + 7]);
            const hitTriangle = shadowed[base + 11];
            const origin: Vec3 = [shadowed[base + 12], shadowed[base + 13], shadowed[base + 14]];
            for (let light = 0; light < 3; light++) {
                const full = luminance(unshadowed, base + light * 4);
                if (!(full > 0)) continue;
                samples++;
                const visibility = luminance(shadowed, base + light * 4) / full;
                assert(Math.abs(visibility) < 1e-5 || Math.abs(visibility - 1) < 1e-5, `${oracleCase.name}, ray ${ray}: visibility ${visibility} for one hard-shadowed light is neither 0 nor 1`);
                const gpuBlocked = visibility < 0.5;
                if (gpuBlocked) report.shadowed[light]++;
                else report.lit[light]++;

                let direction: Vec3;
                let tMax: number;
                if (light === 2) {
                    direction = [-SUN_DIRECTION[0], -SUN_DIRECTION[1], -SUN_DIRECTION[2]];
                    tMax = 1e27;
                } else {
                    const target = light === 0 ? lights.outside : lights.inside;
                    const toLight: Vec3 = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
                    tMax = Math.hypot(...toLight);
                    direction = [toLight[0] / tMax, toLight[1] / tMax, toLight[2] / tMax];
                }
                const shadowRay = { origin, direction, tMin: 0, tMax };
                const expected = traceAnyReference(reference, shadowRay);
                if (expected.blocked !== gpuBlocked) {
                    if (expected.ambiguous) {
                        report.ambiguous++;
                    } else {
                        report.mismatches++;
                        if (examples.length < 3) examples.push(`ray ${ray}, ${LIGHT_NAMES[light]}: GPU ${gpuBlocked ? 'blocked' : 'lit'}, reference ${expected.blocked ? 'blocked' : 'lit'}`);
                    }
                }
                const nearest = traceClosestReference(reference, shadowRay);
                if (nearest && nearest.instanceIndex === hitInstance && nearest.triangleIndex === hitTriangle) report.selfHits++;
                // Every hit of the closed thin box is inside it: light from outside must never get in.
                if (oracleCase.everyRayMustHit && light !== 1 && !gpuBlocked) {
                    report.leaks++;
                    if (leakExamples.length < 4) {
                        const instance = oracleCase.instances.find((candidate) => candidate.instanceIndex === hitInstance);
                        const rows = invertAffine(Array.from(instance?.objectToWorld ?? []))?.rows ?? new Float64Array(12);
                        const toObject = (point: Vec3): Vec3 => [0, 1, 2].map((row) => rows[row * 4] * point[0] + rows[row * 4 + 1] * point[1] + rows[row * 4 + 2] * point[2] + rows[row * 4 + 3]) as Vec3;
                        const local = toObject(origin);
                        const position = toObject([shadowed[base + 16], shadowed[base + 17], shadowed[base + 18]]);
                        leakExamples.push(`ray ${ray} ${LIGHT_NAMES[light]}: instance ${hitInstance} triangle ${hitTriangle} bary ${shadowed[base + 15]}, ${shadowed[base + 19]}; position (object) ${position.map((v) => v.toFixed(7))}; origin (object) ${local.map((v) => v.toFixed(7))}`);
                    }
                }
            }
        }
        log(
            `${oracleCase.name}: ${report.hits} hits; lit/shadowed ${report.lit.map((lit, light) => `${lit}/${report.shadowed[light]}`).join(', ')} ` +
                `(outside, inside, sun); ${report.mismatches} mismatches, ${report.ambiguous} ambiguous, ${report.selfHits} self-hits, ${report.leaks} leaks`,
        );
        assert(report.mismatches === 0, `${oracleCase.name}: ${report.mismatches} shadow rays disagree with the reference (${examples.join('; ')})`);
        assert(report.ambiguous <= MAX_AMBIGUOUS_SHARE * samples, `${oracleCase.name}: ${report.ambiguous} of ${samples} disagreements excused as ambiguous (limit 1%)`);
        assert(report.selfHits === 0, `${oracleCase.name}: ${report.selfHits} shadow rays hit the triangle they leave from`);
        assert(report.leaks === 0, `${oracleCase.name}: ${report.leaks} hits inside the closed box are lit from outside (${leakExamples.join('; ')})`);
        const litTotal = report.lit.reduce((sum, value) => sum + value, 0);
        const shadowedTotal = report.shadowed.reduce((sum, value) => sum + value, 0);
        assert(litTotal > 0 && shadowedTotal > 0, `${oracleCase.name}: lit ${litTotal}, shadowed ${shadowedTotal}: the scene doesn't exercise shadows`);
        return report;
    } finally {
        platform.dispose();
    }
}

await runGpuTest(async ({ device, log }) => {
    const reports: Record<string, CaseReport> = {};
    for (const oracleCase of buildOracleCases(FULL_SIZES)) {
        reports[oracleCase.name] = await checkCase(device, oracleCase, log);
        // The CPU reference is slow on the big scenes; let the page breathe between cases.
        await new Promise((resolve) => setTimeout(resolve, 0));
    }
    return { cases: reports };
});
