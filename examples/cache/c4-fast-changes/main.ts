/**
 * Fast changes: lag against ghosting. The radiance cache, on its default
 * settings, in a dark room: a ceiling lamp on a switch, a flashlight on the
 * camera and a red box to drag. A chart shows the mean GI light on screen
 * over the last 120 frames, with a marker at each change, and the ghost
 * view shows where the GI light is still moving.
 *
 * Nothing is called on a change: the adapter sees the light switched, the
 * flashlight moved with the camera and the box moved, and the cache resets
 * its history near them. The panel turns the world cache's reset,
 * `cellChangeReset` (on by default, register E9), off and on: it is read
 * every frame.
 *
 * The chart and the ghost view are this page's own: each frame it copies the
 * cache's irradiance (`cache.probes.denoisedIrradiance`), reads back a few
 * rows of it for the mean, and blends it into a running average.
 */

import * as THREE from 'three/webgpu';
import { abs, luminance, mix, mrt, normalView, output, packNormalToRGB, pass, screenUV, select, texture, uniform, vec3, vec4 } from 'three/tsl';
import GUI from 'three/addons/libs/lil-gui.module.min.js';
import { createGI, halfToFloat } from '@pmndrs/gi/core';
import { RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';
import { ddgi, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT, screenInputs, ThreeGI, wrapGpuTexture } from '@pmndrs/gi/three';
import { createRendererAndDevice } from '../../shared/renderer.ts';
import { loadScene } from '../../shared/scene.ts';
import { showCacheStats } from '../../shared/cache-stats.ts';
import { createTimeline } from '../../shared/timeline.ts';

// The screen inputs need a single-sampled pass.
const { renderer, device } = await createRendererAndDevice({ antialias: false });
const { scene, camera, controls, lights } = loadScene('dark-room', renderer);
const [lamp, flashlight] = lights as [THREE.PointLight, THREE.SpotLight];
const box = scene.getObjectByName('drag-box');
if (!box) throw new Error('The dark-room scene has no drag-box.');
// The scene's objects keep the matrix they were built with; the box moves by its position instead.
box.matrix.decompose(box.position, box.quaternion, box.scale);
box.matrixAutoUpdate = true;
// Orbit the room's middle, close enough that the camera stays inside its walls (5 × 2.7 × 5 m).
controls.maxDistance = 2.3;
controls.minPolarAngle = 0.7;
controls.maxPolarAngle = 1.45;
controls.enablePan = false;

// The flashlight rides on the camera, a little below and right of the eye, pointing where it looks.
scene.add(camera);
camera.add(flashlight, flashlight.target);
flashlight.position.set(0.15, -0.15, 0);
flashlight.target.position.set(0, 0, -1);

const gi = createGI(device);
const cache = gi.addTechnique(new RadianceCacheTechnique()); // the defaults: change resets on
gi.setOutput(cache);
const preparing = gi.prepare();
const threeGI = new ThreeGI(gi, scene, { preparing });
await preparing;

// The ghost view: |this frame's irradiance - its average over about the last 8 frames|, relative. Where GI
// has settled the two agree and the view is black; light still fading out or filling in shows. The loop below
// copies the irradiance (`current`) and blends it into the average, two render targets used in turn.
let copy: GPUTexture | null = null;
const current = texture(new THREE.Texture(), screenUV);
const averages = [0, 1].map(() => new THREE.RenderTarget(1, 1, { type: THREE.HalfFloatType, depthBuffer: false }));
const average = texture(averages[0].texture, screenUV);
const blending = new THREE.NodeMaterial();
blending.colorNode = mix(average, current, 1 / 8);
const blend = new THREE.QuadMesh(blending);
const ghost = uniform(0);

/** A copy of the irradiance and two averages of its size, made again when its size changes. */
function resizeGhost(width: number, height: number): void {
    copy?.destroy();
    copy = device.createTexture({ label: 'c4: irradiance copy', size: [width, height], format: 'rgba16float', usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING });
    current.value = wrapGpuTexture(copy, { dimension: '2d', sampleType: 'float' });
    for (const target of averages) target.setSize(width, height);
}
const size = renderer.getDrawingBufferSize(new THREE.Vector2());
resizeGhost(size.x, size.y);

const pipeline = new THREE.RenderPipeline(renderer);
const scenePass = pass(scene, camera);
const screen = screenInputs(scenePass, threeGI);
scenePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), ...screen.outputs }));
// As a share of the light there, so bright and dim places read alike. Under 4% (the noise) stays black;
// white is about a quarter.
const difference = luminance(abs(current.rgb.sub(average.rgb))).div(luminance(average.rgb).add(0.05));
const ghostView = vec4(vec3(difference.sub(0.04).max(0).mul(5)), 1);
pipeline.outputNode = screen.encodeBefore(select(ghost.greaterThan(0.5), ghostView, ddgi(scenePass, threeGI)));

// The chart: the mean irradiance of a few rows of the screen, read back every frame.
const timeline = createTimeline({ label: 'Mean GI light on screen (E)', frames: 120, settleTolerance: 0.05 });
const ROWS = 16;
const readbacks: Array<{ buffer: GPUBuffer; busy: boolean }> = [];
let queue = Promise.resolve();
let pendingMark: string | null = null;

/** Marks the chart at the next frame: the frame the change first shows in. */
function changed(label: string): void {
    pendingMark = label;
}

function texelLuminance(halves: Uint16Array, first: number): number {
    return 0.2126 * halfToFloat(halves[first]) + 0.7152 * halfToFloat(halves[first + 1]) + 0.0722 * halfToFloat(halves[first + 2]);
}

/** Copies this frame's irradiance for the ghost view and reads back its mean for the chart. */
function afterFrame(frame: number): void {
    const irradiance = cache.probes?.denoisedIrradiance;
    if (!irradiance) return;
    const { width, height } = irradiance;
    if (copy?.width !== width || copy?.height !== height) {
        resizeGhost(width, height);
        for (const readback of readbacks.splice(0)) readback.buffer.destroy();
    }
    const bytesPerRow = Math.ceil((width * 8) / 256) * 256;
    let readback = readbacks.find((candidate) => !candidate.busy);
    if (!readback && readbacks.length < 6) {
        readback = { buffer: device.createBuffer({ label: 'c4: mean readback', size: bytesPerRow * ROWS, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }), busy: false };
        readbacks.push(readback);
    }

    const encoder = device.createCommandEncoder();
    encoder.copyTextureToTexture({ texture: irradiance }, { texture: copy! }, [width, height]);
    for (let row = 0; readback && row < ROWS; row++) {
        const y = Math.floor(((row + 0.5) / ROWS) * height);
        encoder.copyTextureToBuffer({ texture: irradiance, origin: [0, y] }, { buffer: readback.buffer, offset: row * bytesPerRow, bytesPerRow }, [width, 1]);
    }
    device.queue.submit([encoder.finish()]);
    // The average takes an eighth of this frame: written to one target, read from the other next frame.
    const [from, to] = frame % 2 === 0 ? averages : [averages[1], averages[0]];
    average.value = from.texture;
    renderer.setRenderTarget(to);
    blend.render(renderer);
    renderer.setRenderTarget(null);
    average.value = to.texture;

    const mark = pendingMark;
    pendingMark = null;
    let mean: Promise<number | null> = Promise.resolve(null); // no buffer free: a gap in the chart
    if (readback) {
        const { buffer } = readback;
        readback.busy = true;
        mean = buffer.mapAsync(GPUMapMode.READ).then(
            () => {
                const halves = new Uint16Array(buffer.getMappedRange());
                let sum = 0;
                for (let row = 0; row < ROWS; row++) for (let x = 0; x < width; x++) sum += texelLuminance(halves, (row * bytesPerRow) / 2 + x * 4);
                buffer.unmap();
                readback.busy = false;
                return sum / (ROWS * width);
            },
            () => null, // destroyed on a resize
        );
    }
    // In frame order, the marker after its frame's value.
    queue = queue.then(async () => {
        timeline.push(await mean);
        if (mark) timeline.mark(mark);
    });
}

// The panel: switches change the scene itself; that is all GI needs.
const settings = { flashlight: true, lamp: true, resets: true, ghost: false };
const gui = new GUI({ title: 'Fast changes' });
gui.add(settings, 'flashlight').name('flashlight').onChange((on: boolean) => {
    flashlight.visible = on;
    changed(on ? 'flashlight on' : 'flashlight off');
});
gui.add(settings, 'lamp').name('room light').onChange((on: boolean) => {
    lamp.visible = on;
    changed(on ? 'light on' : 'light off');
});
gui.add(settings, 'resets').name('change resets').onChange((on: boolean) => (cache.settings.cellChangeReset = on));
gui.add(settings, 'ghost').name('ghost view').onChange((on: boolean) => (ghost.value = on ? 1 : 0));

// Drag the box across the floor. Taken before the orbit controls see the press.
const canvas = renderer.domElement;
const raycaster = new THREE.Raycaster();
const floor = new THREE.Plane(new THREE.Vector3(0, 1, 0), -box.position.y);
const hit = new THREE.Vector3();
const grab = new THREE.Vector3();
let dragging = false;
const aim = (event: PointerEvent): void => {
    const rect = canvas.getBoundingClientRect();
    raycaster.setFromCamera(new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1), camera);
};
canvas.addEventListener(
    'pointerdown',
    (event) => {
        aim(event);
        if (raycaster.intersectObject(box).length === 0 || !raycaster.ray.intersectPlane(floor, hit)) return;
        event.stopImmediatePropagation();
        canvas.setPointerCapture(event.pointerId);
        grab.subVectors(box.position, hit);
        dragging = true;
    },
    { capture: true },
);
canvas.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    aim(event);
    if (!raycaster.ray.intersectPlane(floor, hit)) return;
    // Inside the room's walls (5 × 5 m), the box being 0.6 m.
    box.position.x = THREE.MathUtils.clamp(hit.x + grab.x, 0.3, 4.7);
    box.position.z = THREE.MathUtils.clamp(hit.z + grab.z, 0.3, 4.7);
});
canvas.addEventListener('pointerup', () => {
    if (!dragging) return;
    dragging = false;
    changed('box dropped');
});
controls.addEventListener('end', () => flashlight.visible && changed('flashlight moved'));

showCacheStats(device, gi, cache, () => (settings.resets ? 'change resets on' : 'change resets off'));
let frame = 0;
renderer.setAnimationLoop(() => {
    threeGI.update(camera); // the frame's GI work; the screen stage runs inside the pipeline
    pipeline.render();
    afterFrame(frame++);
});
