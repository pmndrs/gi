/**
 * RC1.1 on Sponza with its curtains (`sponza`), at its first bookmark: the
 * three adapter fills the platform's `FrameScreen` from the host's own MRT
 * scene pass (`screenInputs`, packages/ddgi-three/src/screen-inputs.ts), and
 * a test technique reads it inside `encodeScreen` (screen-probe.ts).
 *
 * Checks, each by read-back at a grid of sample pixels:
 * 1. **Handover:** what the technique read equals three's own read-back of
 *    the same MRT targets (normal, albedo, velocity, instance), so the
 *    platform got three's textures of this very frame, with no copy.
 * 2. **Normals mean world space:** the world normal equals the camera's
 *    rotation applied to the pass's view-space normal (three-composite's
 *    `normal` output), within half-float precision; each is unit length.
 * 3. **Instance ids:** each pixel's id is the platform instance index + 1
 *    of the object three's `Raycaster` finds under the pixel's centre (the
 *    adapter's mesh-to-instance map, `SceneSync.mirroredInstances`).
 *    Pixels at an object's edge (a 3 × 3 neighbourhood with two ids) and
 *    pixels on alpha-tested or transparent surfaces (the raycaster ignores
 *    alpha) are left out and counted.
 * 4. **Velocity:** zero over the whole target for a still camera; non-zero
 *    at the surface samples in the frame after a camera turn; zero again
 *    once still.
 * 5. **Every frame:** a box placed in front of the camera shows at the
 *    centre pixel in the next frame (its id, its albedo with metalness
 *    folded in, its normal facing the camera); moved away, the centre shows
 *    what is behind it again.
 * 6. **Resize:** after the canvas changes size, the screen has the new size,
 *    new textures, and the handover check holds again.
 * 7. **Debug views** (the core's `ScreenInputsView`): each channel drawn,
 *    read back, checked at the samples (instance colours by hash, albedo as
 *    sRGB, normals as n / 2 + 1/2), and laid out as a contact sheet that the
 *    runner's screenshot captures.
 * 8. **The opaque surface under the depth sample:** a transparent plane
 *    (no depth write) over the centre, under normal, premultiplied normal,
 *    additive and multiply blending, changes the picture but no channel.
 * Every frame also checks that exactly one screen stage ran.
 *
 * Manual (it needs the downloaded Sponza):
 *   pnpm test:gpu --gpu nvidia --filter three-screen-inputs
 */

import { AdditiveBlending, BlendMode, BoxGeometry, MaterialBlending, Matrix3, Mesh, MeshStandardMaterial, MultiplyBlending, NormalBlending, PlaneGeometry, Raycaster, RenderPipeline, Vector2, Vector3, type Material, type Node, type Object3D } from 'three/webgpu';
import { mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import { assert } from '../../client.ts';
import { instanceDebugColor, SCREEN_CHANNELS, type ScreenChannel } from '../../../../packages/ddgi-core/src/platform/screen/screen-inputs-view.ts';
import type { FrameScreen } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { GI_SCREEN_ALBEDO_OUTPUT, GI_SCREEN_INSTANCE_OUTPUT, GI_SCREEN_NORMAL_OUTPUT, GI_SCREEN_VELOCITY_OUTPUT, GIScreenChannelNode, screenInputs } from '../../../../packages/ddgi-three/src/screen-inputs.ts';
import { GI_NORMAL_OUTPUT } from '../../../../packages/ddgi-three/src/composite.ts';
import { runThreePage, type ThreePageContext } from '../three-shared/harness.ts';
import { advanceThreeFrame, loadCompositeScene, type CompositeScene } from '../three-composite-shared/composite-check.ts';
import { readInstanceIds, readPassTarget, readRgba8, type TargetImage } from './readback.ts';
import { ScreenProbeTechnique, type ScreenSample } from './screen-probe.ts';

const SIZE = { width: 1280, height: 720 } as const;
const RESIZED = { width: 960, height: 540 } as const;
/** Sample grid: columns × rows of pixels, spread over the view. */
const GRID = { columns: 10, rows: 6 } as const;
/** The probe reads the same texels three reads back: any difference means another texture or another frame. */
const HANDOVER_TOLERANCE = 1e-6;
/** World normal against the rotated view normal: both went through half floats. */
const NORMAL_SPACE_TOLERANCE = 0.01;
/** A still camera's velocity: exactly zero in practice (the same matrices both frames). */
const STILL_VELOCITY = 1e-5;
/** NDC motion that counts as moving after a 1.5° turn (about 0.03 at Sponza's field of view). */
const MOVING_VELOCITY = 1e-3;
/** Instance checks needed for the check to mean something. */
const MIN_INSTANCE_CHECKS = 20;
/** 8-bit views and targets: one step, plus rounding. */
const BYTE_TOLERANCE = 1.5 / 255;

function samplePixels(width: number, height: number): Array<readonly [number, number]> {
    const pixels: Array<readonly [number, number]> = [];
    for (let row = 0; row < GRID.rows; row++) {
        for (let column = 0; column < GRID.columns; column++) {
            pixels.push([Math.floor(((column + 0.5) / GRID.columns) * width), Math.floor(((row + 0.5) / GRID.rows) * height)]);
        }
    }
    return pixels;
}

/** Whether a pixel shows a surface (the depth buffer was written; it clears to 1 at the far plane). */
function isSurface(sample: ScreenSample): boolean {
    return sample.depth < 1;
}

interface Rig {
    readonly scene: CompositeScene;
    readonly probe: ScreenProbeTechnique;
    readonly screen: ReturnType<typeof screenInputs>;
    readonly scenePass: ReturnType<typeof pass>;
    readonly pipeline: RenderPipeline;
}

/** One frame the way an app draws it: GI's world-space update (paused budget: scene changes only), then the pipeline. */
function drawFrame(context: ThreePageContext, rig: Rig, pipeline: RenderPipeline = rig.pipeline): void {
    rig.scene.threeGI.update(rig.scene.camera, 1 / 60);
    advanceThreeFrame(context.renderer);
    const stagesBefore = rig.screen.stages;
    const probeStagesBefore = rig.probe.stages;
    pipeline.render();
    // Exactly one screen stage per frame, and the technique's screen stage ran in it.
    const ran = rig.screen.stages - stagesBefore;
    const probeRan = rig.probe.stages - probeStagesBefore;
    assert(ran === 1 && probeRan === 1, `a frame ran ${ran} screen stages (the technique saw ${probeRan}); expected exactly 1`);
}

async function sampleFrame(context: ThreePageContext, rig: Rig, pixels: Array<readonly [number, number]>): Promise<ScreenSample[]> {
    rig.probe.arm(pixels);
    drawFrame(context, rig);
    return rig.probe.collect();
}

/** Check 1: the technique's reads equal three's read-back of the same targets. */
async function checkHandover(context: ThreePageContext, rig: Rig, samples: readonly ScreenSample[], label: string): Promise<Record<string, number>> {
    const { renderer } = context;
    const normal = await readPassTarget(renderer, rig.scenePass, GI_SCREEN_NORMAL_OUTPUT);
    const albedo = await readPassTarget(renderer, rig.scenePass, GI_SCREEN_ALBEDO_OUTPUT);
    const velocity = await readPassTarget(renderer, rig.scenePass, GI_SCREEN_VELOCITY_OUTPUT);
    const instance = await readInstanceIds(renderer, rig.scenePass, GI_SCREEN_INSTANCE_OUTPUT);
    let normalError = 0;
    let albedoError = 0;
    let velocityError = 0;
    let instanceMismatches = 0;
    for (const sample of samples) {
        const [x, y] = sample.pixel;
        for (let channel = 0; channel < 3; channel++) {
            normalError = Math.max(normalError, Math.abs(sample.normal[channel] - normal.at(x, y, channel)));
            albedoError = Math.max(albedoError, Math.abs(sample.albedo[channel] - albedo.at(x, y, channel)));
        }
        for (let channel = 0; channel < 2; channel++) velocityError = Math.max(velocityError, Math.abs(sample.velocity[channel] - velocity.at(x, y, channel)));
        if (sample.instance !== instance.at(x, y, 0)) instanceMismatches++;
    }
    context.log(`${label} handover: max |technique - three| normal ${normalError}, albedo ${albedoError}, velocity ${velocityError}; instance mismatches ${instanceMismatches} of ${samples.length}`);
    assert(normalError <= HANDOVER_TOLERANCE, `${label}: the normal the technique read differs from three's target by ${normalError}`);
    assert(albedoError <= HANDOVER_TOLERANCE, `${label}: the albedo the technique read differs from three's target by ${albedoError}`);
    assert(velocityError <= HANDOVER_TOLERANCE, `${label}: the velocity the technique read differs from three's target by ${velocityError}`);
    assert(instanceMismatches === 0, `${label}: ${instanceMismatches} instance ids differ from three's target`);
    return { normalError, albedoError, velocityError, instanceMismatches };
}

/** Check 2: world normals are the view normals turned by the camera, and unit length. */
async function checkNormalSpace(context: ThreePageContext, rig: Rig, samples: readonly ScreenSample[]): Promise<Record<string, number>> {
    const viewNormals = await readPassTarget(context.renderer, rig.scenePass, GI_NORMAL_OUTPUT);
    const rotation = new Matrix3().setFromMatrix4(rig.scene.camera.matrixWorld);
    let worstAngle = 0;
    let worstLength = 0;
    let checked = 0;
    for (const sample of samples) {
        if (!isSurface(sample)) continue;
        const [x, y] = sample.pixel;
        // packNormalToRGB: n × 0.5 + 0.5.
        const fromView = new Vector3(viewNormals.at(x, y, 0) * 2 - 1, viewNormals.at(x, y, 1) * 2 - 1, viewNormals.at(x, y, 2) * 2 - 1).applyMatrix3(rotation).normalize();
        const world = new Vector3(...sample.normal);
        worstLength = Math.max(worstLength, Math.abs(world.length() - 1));
        worstAngle = Math.max(worstAngle, world.normalize().distanceTo(fromView));
        checked++;
    }
    context.log(`normal space: ${checked} surface samples; max |world - R × view| ${worstAngle.toFixed(5)}, max |length - 1| ${worstLength.toFixed(5)}`);
    assert(checked >= MIN_INSTANCE_CHECKS, `only ${checked} surface samples`);
    assert(worstAngle <= NORMAL_SPACE_TOLERANCE, `world normals differ from the camera-rotated view normals by up to ${worstAngle}`);
    assert(worstLength <= NORMAL_SPACE_TOLERANCE, `world normals are not unit length (off by up to ${worstLength})`);
    return { worstAngle, worstLength, checked };
}

/** Whether the 3 × 3 pixels around (x, y) all hold one id: away from an object's edge. */
function inside(ids: TargetImage, x: number, y: number): boolean {
    const id = ids.at(x, y, 0);
    for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
            const nx = Math.min(ids.width - 1, Math.max(0, x + dx));
            const ny = Math.min(ids.height - 1, Math.max(0, y + dy));
            if (ids.at(nx, ny, 0) !== id) return false;
        }
    }
    return true;
}

function alphaCut(material: Material | Material[] | undefined, group: number | undefined): boolean {
    const single = Array.isArray(material) ? material[group ?? 0] : material;
    if (!single) return false;
    const lit = single as Material & { alphaMap?: unknown; map?: { format?: number } | null };
    return single.alphaTest > 0 || single.transparent || lit.alphaMap != null;
}

/** Check 3: instance ids against three's raycaster and the adapter's map. */
async function checkInstances(context: ThreePageContext, rig: Rig, samples: readonly ScreenSample[]): Promise<Record<string, number>> {
    const { scene, camera, threeGI } = rig.scene;
    const ids = await readInstanceIds(context.renderer, rig.scenePass, GI_SCREEN_INSTANCE_OUTPUT);
    const instancesOf = new Map<Object3D, readonly number[]>();
    for (const [object, handles] of threeGI.sync.mirroredInstances()) instancesOf.set(object, handles.map((handle) => threeGI.gi.scene.instanceIndex(handle) + 1));
    const raycaster = new Raycaster();
    raycaster.layers.mask = camera.layers.mask;
    let checked = 0;
    let atEdges = 0;
    let cutouts = 0;
    let background = 0;
    const mismatches: string[] = [];
    const started = performance.now();
    for (const sample of samples) {
        const [x, y] = sample.pixel;
        if (!isSurface(sample)) {
            background++;
            assert(sample.instance === 0, `pixel ${x},${y} shows no surface but has instance id ${sample.instance}`);
            continue;
        }
        if (!inside(ids, x, y)) {
            atEdges++;
            continue;
        }
        const ndc = new Vector2(((x + 0.5) / ids.width) * 2 - 1, 1 - ((y + 0.5) / ids.height) * 2);
        raycaster.setFromCamera(ndc, camera);
        const hit = raycaster.intersectObject(scene, true).find((candidate) => (candidate.object as Mesh).isMesh && candidate.object.visible);
        if (hit && alphaCut((hit.object as Mesh).material, hit.face?.materialIndex)) {
            cutouts++;
            continue;
        }
        const expected = hit ? (instancesOf.get(hit.object)?.[hit.instanceId ?? 0] ?? 0) : 0;
        checked++;
        if (sample.instance !== expected) mismatches.push(`${x},${y}: id ${sample.instance}, expected ${expected} ('${hit?.object.name ?? 'nothing'}')`);
    }
    context.log(
        `instance ids: ${checked} checked against the raycaster (${(performance.now() - started).toFixed(0)} ms), ${mismatches.length} mismatches; left out ${atEdges} at edges, ${cutouts} on alpha-tested or transparent surfaces, ${background} background`,
    );
    for (const line of mismatches) context.log(`  mismatch ${line}`);
    assert(checked >= MIN_INSTANCE_CHECKS, `only ${checked} instance samples could be checked`);
    assert(mismatches.length === 0, `${mismatches.length} instance ids differ from the object under the pixel: ${mismatches.slice(0, 3).join('; ')}`);
    return { checked, mismatches: mismatches.length, atEdges, cutouts, background };
}

/** Check 4: velocity over the whole target. */
async function maxVelocity(context: ThreePageContext, rig: Rig): Promise<number> {
    const velocity = await readPassTarget(context.renderer, rig.scenePass, GI_SCREEN_VELOCITY_OUTPUT);
    let largest = 0;
    for (let y = 0; y < velocity.height; y++) {
        for (let x = 0; x < velocity.width; x++) largest = Math.max(largest, Math.abs(velocity.at(x, y, 0)), Math.abs(velocity.at(x, y, 1)));
    }
    return largest;
}

async function checkVelocity(context: ThreePageContext, rig: Rig, pixels: Array<readonly [number, number]>): Promise<Record<string, number>> {
    const camera = rig.scene.camera;
    drawFrame(context, rig);
    await sampleFrame(context, rig, pixels);
    const still = await maxVelocity(context, rig);
    camera.rotateY((1.5 * Math.PI) / 180);
    camera.updateMatrixWorld(true);
    const turned = await sampleFrame(context, rig, pixels);
    const surfaces = turned.filter(isSurface);
    const moving = surfaces.filter((sample) => Math.hypot(sample.velocity[0], sample.velocity[1]) > MOVING_VELOCITY).length;
    drawFrame(context, rig);
    await sampleFrame(context, rig, pixels);
    const stillAgain = await maxVelocity(context, rig);
    context.log(`velocity: still max ${still}; after a 1.5° turn ${moving} of ${surfaces.length} surface samples move by more than ${MOVING_VELOCITY}; still again max ${stillAgain}`);
    assert(still <= STILL_VELOCITY, `a still camera has velocity up to ${still}`);
    assert(moving >= surfaces.length * 0.9 && surfaces.length > 0, `after a camera turn only ${moving} of ${surfaces.length} surface samples moved`);
    assert(stillAgain <= STILL_VELOCITY, `the camera stopped but velocity reads up to ${stillAgain}`);
    return { still, moving, surfaces: surfaces.length, stillAgain };
}

/** Check 5: a new object shows in the next frame, and its pixels change when it moves away. */
async function checkEveryFrame(context: ThreePageContext, rig: Rig): Promise<Record<string, number>> {
    const { scene, camera, threeGI } = rig.scene;
    const centre: Array<readonly [number, number]> = [[Math.floor(SIZE.width / 2), Math.floor(SIZE.height / 2)]];
    const before = (await sampleFrame(context, rig, centre))[0];

    // Base colour (0.8, 0.4, 0.2) at metalness 0.5: the screen's albedo is half of it (FrameScreen.albedo).
    const material = new MeshStandardMaterial({ roughness: 1, metalness: 0.5 });
    material.color.setRGB(0.8, 0.4, 0.2);
    const box = new Mesh(new BoxGeometry(0.4, 0.4, 0.4), material);
    box.name = 'screen-inputs test box';
    const forward = camera.getWorldDirection(new Vector3());
    box.position.copy(camera.position).addScaledVector(forward, 1.2);
    box.lookAt(camera.position);
    box.updateMatrixWorld(true);
    scene.add(box);
    threeGI.markChanged(box);
    const shown = (await sampleFrame(context, rig, centre))[0];
    const handles = [...threeGI.sync.mirroredInstances()].find(([object]) => object === box)?.[1];
    assert(handles && handles.length === 1, 'the box was not mirrored into the platform');
    const boxId = threeGI.gi.scene.instanceIndex(handles[0]) + 1;
    const toCamera = camera.position.clone().sub(box.position).normalize();
    const facing = new Vector3(...shown.normal).normalize().dot(toCamera);
    const albedoError = Math.max(Math.abs(shown.albedo[0] - 0.4), Math.abs(shown.albedo[1] - 0.2), Math.abs(shown.albedo[2] - 0.1));
    context.log(`box in front: centre id ${shown.instance} (box ${boxId}, before ${before.instance}), albedo ${shown.albedo.map((value) => value.toFixed(4)).join(', ')} (expected 0.4, 0.2, 0.1), normal·toCamera ${facing.toFixed(4)}`);
    assert(shown.instance === boxId, `the centre pixel shows id ${shown.instance}, not the box's ${boxId}, in the frame after it was added`);
    assert(albedoError <= BYTE_TOLERANCE, `the box's albedo reads ${shown.albedo.join(', ')}, expected base colour × (1 − metalness) = 0.4, 0.2, 0.1`);
    assert(facing > 0.98, `the box's normal doesn't face the camera (dot ${facing})`);

    box.position.addScaledVector(new Vector3().crossVectors(forward, camera.up).normalize(), 50);
    box.updateMatrixWorld(true);
    threeGI.markChanged(box);
    const moved = (await sampleFrame(context, rig, centre))[0];
    context.log(`box moved away: centre id ${moved.instance} (before the box ${before.instance}), albedo ${moved.albedo.map((value) => value.toFixed(4)).join(', ')}`);
    assert(moved.instance === before.instance, `after the box moved away the centre shows id ${moved.instance}, expected ${before.instance} again`);
    assert(moved.albedo.every((value, channel) => Math.abs(value - before.albedo[channel]) <= HANDOVER_TOLERANCE), 'after the box moved away the centre albedo is not what it was');
    scene.remove(box);
    threeGI.markChanged(box);
    box.geometry.dispose();
    material.dispose();
    return { boxId, facing, albedoError };
}

/**
 * Check 8 (RC1.1 review): a transparent surface over a known opaque one
 * leaves every channel as the opaque surface's (it writes no depth, as
 * glTF's BLEND materials don't), under each of three's blendings, while it
 * does change the picture.
 */
async function checkTransparency(context: ThreePageContext, rig: Rig): Promise<Record<string, number>> {
    const { scene, camera, threeGI } = rig.scene;
    const centre: Array<readonly [number, number]> = [[Math.floor(SIZE.width / 2), Math.floor(SIZE.height / 2)]];
    const [x, y] = centre[0];
    const colourAt = async () => {
        const picture = await readPassTarget(context.renderer, rig.scenePass, 'output');
        return [picture.at(x, y, 0), picture.at(x, y, 1), picture.at(x, y, 2)];
    };
    const opaque = (await sampleFrame(context, rig, centre))[0];
    const opaqueColour = await colourAt();
    const blendings = [
        { name: 'normal', blending: NormalBlending, premultipliedAlpha: false },
        { name: 'normal premultiplied', blending: NormalBlending, premultipliedAlpha: true },
        { name: 'additive', blending: AdditiveBlending, premultipliedAlpha: false },
        { name: 'multiply', blending: MultiplyBlending, premultipliedAlpha: true },
    ];
    const results: Record<string, number> = {};
    for (const { name, blending, premultipliedAlpha } of blendings) {
        const material = new MeshStandardMaterial({ color: 0x20c040, transparent: true, opacity: 0.5, depthWrite: false, blending, premultipliedAlpha });
        const plane = new Mesh(new PlaneGeometry(0.4, 0.4), material);
        plane.name = `screen-inputs transparent plane (${name})`;
        plane.position.copy(camera.position).addScaledVector(camera.getWorldDirection(new Vector3()), 1.2);
        plane.lookAt(camera.position);
        plane.updateMatrixWorld(true);
        scene.add(plane);
        threeGI.markChanged(plane);
        const covered = (await sampleFrame(context, rig, centre))[0];
        const coveredColour = await colourAt();
        // Relative: multiply blending darkens a dark pixel by a small absolute amount.
        const pictureChange = Math.max(...coveredColour.map((value, channel) => Math.abs(value - opaqueColour[channel]) / Math.max(Math.abs(opaqueColour[channel]), 1e-3)));
        const channelChange = Math.max(
            Math.abs(covered.depth - opaque.depth),
            ...covered.normal.map((value, channel) => Math.abs(value - opaque.normal[channel])),
            ...covered.velocity.map((value, channel) => Math.abs(value - opaque.velocity[channel])),
            ...covered.albedo.map((value, channel) => Math.abs(value - opaque.albedo[channel])),
        );
        context.log(`transparent plane, ${name} blending: picture changed by ${(pictureChange * 100).toFixed(1)}%; channels changed by ${channelChange}; id ${covered.instance} (opaque ${opaque.instance})`);
        assert(pictureChange > 0.05, `the ${name} transparent plane doesn't cover the centre pixel (the check would prove nothing)`);
        assert(channelChange === 0, `a transparent plane (${name} blending) changed the screen channels by ${channelChange}`);
        assert(covered.instance === opaque.instance, `a transparent plane (${name} blending) changed the instance id to ${covered.instance}`);
        results[name] = channelChange;
        scene.remove(plane);
        threeGI.markChanged(plane);
        plane.geometry.dispose();
        material.dispose();
    }
    return results;
}

function setSize(context: ThreePageContext, rig: Rig, size: { width: number; height: number }): void {
    context.renderer.setSize(size.width, size.height, false);
    rig.scene.camera.aspect = size.width / size.height;
    rig.scene.camera.updateProjectionMatrix();
}

/** Check 6: a resize gives the screen the new size and new textures, and the handover holds. */
async function checkResize(context: ThreePageContext, rig: Rig): Promise<Record<string, number>> {
    const before = rig.screen.screen as FrameScreen;
    setSize(context, rig, RESIZED);
    drawFrame(context, rig);
    const pixels = samplePixels(RESIZED.width, RESIZED.height);
    const samples = await sampleFrame(context, rig, pixels);
    const after = rig.screen.screen as FrameScreen;
    const probeScreen = rig.probe.lastScreen as FrameScreen;
    context.log(`resize: ${before.width} x ${before.height} -> ${after.width} x ${after.height}; the technique saw ${probeScreen.width} x ${probeScreen.height}; normal texture ${after.normal.width} x ${after.normal.height}`);
    assert(after.width === RESIZED.width && after.height === RESIZED.height, `after the resize the screen is ${after.width} x ${after.height}`);
    assert(probeScreen.width === RESIZED.width && probeScreen.normal === after.normal, "the technique didn't get the resized screen");
    for (const channel of ['depth', 'normal', 'velocity', 'albedo', 'instance'] as const) {
        const texture = after[channel] as GPUTexture;
        assert(texture.width === RESIZED.width && texture.height === RESIZED.height, `the ${channel} texture is ${texture.width} x ${texture.height} after the resize`);
        assert(texture !== before[channel], `the ${channel} texture is the one from before the resize`);
    }
    const handover = await checkHandover(context, rig, samples, 'resized');
    setSize(context, rig, SIZE);
    drawFrame(context, rig);
    return { ...handover, width: after.width, height: after.height };
}

function linearToSrgb(value: number): number {
    return value <= 0.0031308 ? value * 12.92 : 1.055 * value ** (1 / 2.4) - 0.055;
}

/** Check 7: each channel's debug view, checked at the samples and laid out as a contact sheet. */
async function checkViews(context: ThreePageContext, rig: Rig, samples: readonly ScreenSample[]): Promise<Record<string, number>> {
    const { renderer, device } = context;
    const sheet = document.createElement('canvas');
    const columns = 3;
    const tileWidth = Math.floor(SIZE.width / columns);
    const tileHeight = Math.floor(SIZE.height / 2);
    sheet.width = SIZE.width;
    sheet.height = SIZE.height;
    const drawing = sheet.getContext('2d') as CanvasRenderingContext2D;
    drawing.fillStyle = '#202020';
    drawing.fillRect(0, 0, sheet.width, sheet.height);
    const errors: Record<string, number> = {};
    for (const [index, channel] of SCREEN_CHANNELS.entries()) {
        const pipeline = new RenderPipeline(renderer, rig.screen.view(channel));
        pipeline.outputColorTransform = false;
        for (let frame = 0; frame < 3; frame++) drawFrame(context, rig, pipeline);
        await device.queue.onSubmittedWorkDone();
        const texture = rig.screen.viewTexture;
        assert(texture, `the ${channel} view drew nothing`);
        const pixels = await readRgba8(device, texture);
        errors[channel] = viewError(channel, pixels, texture.width, samples, rig.scene.camera);
        context.log(`view '${channel}': max error at the samples ${(errors[channel] * 255).toFixed(2)} / 255`);
        assert(errors[channel] <= 2.5 / 255, `the ${channel} view differs from what the channel holds by ${(errors[channel] * 255).toFixed(2)} / 255`);
        const tile = await createImageBitmap(new ImageData(pixels, texture.width, texture.height));
        const left = (index % columns) * tileWidth;
        const top = Math.floor(index / columns) * tileHeight;
        drawing.drawImage(tile, left, top, tileWidth, tileHeight);
        drawing.fillStyle = '#ffffff';
        drawing.font = '20px sans-serif';
        drawing.fillText(`Screen inputs: ${channel}`, left + 10, top + 28);
        pipeline.dispose();
    }
    drawing.fillStyle = '#ffffff';
    drawing.font = '18px sans-serif';
    drawing.fillText('RC1.1 three-screen-inputs, Sponza bookmark 1', tileWidth * 2 + 16, tileHeight + 40);
    // The runner screenshots the page: show the sheet instead of the WebGPU canvas.
    renderer.domElement.style.display = 'none';
    sheet.style.display = 'block';
    sheet.style.width = '100%';
    document.body.style.margin = '0';
    document.body.prepend(sheet);
    return errors;
}

/** The largest difference at the samples between a view and what its channel should look like there. */
function viewError(channel: ScreenChannel, pixels: Uint8ClampedArray, width: number, samples: readonly ScreenSample[], camera: { near: number; far: number }): number {
    let worst = 0;
    for (const sample of samples) {
        const [x, y] = sample.pixel;
        const at = (y * width + x) * 4;
        const shown = [pixels[at] / 255, pixels[at + 1] / 255, pixels[at + 2] / 255];
        let expected: number[] | null = null;
        if (channel === 'instance') expected = instanceDebugColor(sample.instance);
        else if (channel === 'albedo') expected = sample.albedo.map((value) => linearToSrgb(Math.min(1, Math.max(0, value))));
        else if (channel === 'normal' && isSurface(sample)) {
            const length = Math.hypot(...sample.normal);
            expected = sample.normal.map((value) => (value / length) * 0.5 + 0.5);
        } else if (channel === 'depth' && isSurface(sample)) {
            // The view's distance shade (ScreenInputsView, default mid grey at 10 m), from the pass camera's planes.
            const distance = (camera.near * camera.far) / (camera.near + (1 - sample.depth) * (camera.far - camera.near));
            expected = [1, 1, 1].map(() => 1 / (1 + distance / 10));
        } else if (channel === 'velocity') expected = [...sample.velocity.map((value) => Math.min(1, Math.max(-1, value * 20)) * 0.5 + 0.5), 0.5];
        if (!expected) continue;
        for (let component = 0; component < 3; component++) worst = Math.max(worst, Math.abs(shown[component] - expected[component]));
    }
    return worst;
}

await runThreePage(
    async (context) => {
        const { renderer, device, data, log } = context;
        renderer.shadowMap.enabled = true;
        const probe = new ScreenProbeTechnique();
        const scene = await loadCompositeScene(device, 'sponza', { ...SIZE, giFrames: 4, probesPerFrame: 1024, log, techniques: [probe] });
        const scenePass = pass(scene.scene, scene.camera);
        const screen = screenInputs(scenePass, scene.threeGI);
        log(`screen needs: ${JSON.stringify(scene.platform.screenNeeds)}; channels rendered: ${JSON.stringify(screen.channels)}; outputs ${Object.keys(screen.outputs).join(', ')}`);
        assert(screen.channels.velocity && screen.channels.albedo && screen.channels.instance, "the pass doesn't render every channel the probe asked for");
        // Check 2's reference, the view-space normal, describes the opaque surface under the depth sample
        // too (written like the screen channels), or a decal's normal would differ from the screen's.
        const viewNormal = new GIScreenChannelNode(() => packNormalToRGB(normalView) as never);
        const outputs = mrt({ output, [GI_NORMAL_OUTPUT]: viewNormal, ...screen.outputs });
        outputs.setBlendMode(GI_NORMAL_OUTPUT, new BlendMode(MaterialBlending));
        scenePass.setMRT(outputs);
        // The view-space normal (check 2's reference): an MRT output is drawn only once its texture exists (NOTES).
        scenePass.getTexture(GI_NORMAL_OUTPUT);
        const pipeline = new RenderPipeline(renderer, screen.encodeBefore(scenePass.getTextureNode('output') as unknown as Node));
        const rig: Rig = { scene, probe, screen, scenePass, pipeline };

        // Compile: draw until three's pipelines are ready (objects are skipped until theirs is).
        for (let frame = 0; frame < 8; frame++) {
            drawFrame(context, rig);
            await device.queue.onSubmittedWorkDone();
        }
        const stagesBefore = probe.stages;
        const pixels = samplePixels(SIZE.width, SIZE.height);
        const samples = await sampleFrame(context, rig, pixels);
        data.handover = await checkHandover(context, rig, samples, 'bookmark 1');
        data.normalSpace = await checkNormalSpace(context, rig, samples);
        data.instances = await checkInstances(context, rig, samples);
        data.velocity = await checkVelocity(context, rig, pixels);
        data.everyFrame = await checkEveryFrame(context, rig);
        data.transparency = await checkTransparency(context, rig);
        data.resize = await checkResize(context, rig);
        const freshSamples = await sampleFrame(context, rig, pixels);
        data.views = await checkViews(context, rig, freshSamples);
        data.stages = { probe: probe.stages - stagesBefore, adapter: screen.stages };
        const formats = screen.screen as FrameScreen;
        data.formats = { depth: formats.depth.format, normal: formats.normal.format, velocity: formats.velocity?.format, albedo: formats.albedo?.format, instance: formats.instance?.format };
        log(`formats: ${JSON.stringify(data.formats)}; screen stages seen by the technique ${probe.stages - stagesBefore}, run by the adapter ${screen.stages}`);
        pipeline.dispose();
        log('done');
    },
    SIZE,
);
