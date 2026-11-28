/**
 * Balls in the courtyard: a scene that keeps changing, under one GI
 * technique at a time. Big balls, 1–2 m across, come over the wall, bounce,
 * roll and settle. Half of them glow (some pulse, some flicker); the rest
 * are plain PBR materials. Right-click, or `throw`, adds one, up to 30.
 *
 * Nothing special is done for GI:
 * - a ball is an ordinary three mesh, added and removed with `scene.add` and
 *   `removeFromParent`, and moved by setting its position and rotation;
 *   the adapter finds both in three's render (docs/api/three.md, "How
 *   changes are found");
 * - a glowing ball's `emissiveIntensity` is edited each frame, and
 *   `threeGI.markChanged(mesh)` has GI see the edit in the next update, as
 *   in C5. A settled ball is left alone, so GI sees it still.
 *
 * The technique: DDGI (a `DDGIVolume` over the courtyard, GI per fragment),
 * or the radiance cache in three of its variants
 * (`RADIANCE_CACHE_VARIANT_PRESETS`), wired as in C1/C6. A switch disposes
 * one platform and builds the next, as C3 does: two platforms can't share
 * one scene pass (their shading modules declare the same WGSL names).
 *
 * The scene is the repo's procedural courtyard, or Sponza's courtyard
 * loaded from Khronos's glTF at run time with `?place=sponza` (place.ts).
 * The physics is the page's own (physics.ts); the readouts are too
 * (readouts.ts).
 */

import * as THREE from 'three/webgpu';
import GUI from 'three/addons/libs/lil-gui.module.min.js';
import { mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import { createGI, type GIPlatform } from '@pmndrs/gi/core';
import { DDGITechnique } from '@pmndrs/gi/ddgi';
import { applyCompareValues, DEFAULT_RADIANCE_CACHE_SETTINGS, RadianceCacheTechnique, variantPreset, type RadianceCacheSettings } from '@pmndrs/gi/radiance-cache';
import { DDGIVolume, ddgi, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT, screenInputs, ThreeGI } from '@pmndrs/gi/three';
import { createRendererAndDevice } from '../../shared/renderer.ts';
import { showPassTiming } from '../../shared/pass-timing.ts';
import { createTimeline } from '../../shared/timeline.ts';
import { loadSponza, proceduralCourtyard, SPONZA_CREDIT, type Place } from './place.ts';
import { SphereWorld, type Ball } from './physics.ts';
import { measureGIMean, PictureMean, showStatsLine, type Running } from './readouts.ts';
import './x3.css';

/** The techniques on offer: DDGI, and the cache's variants that run on the site (C6 has all ten). */
const TECHNIQUES = {
    ddgi: 'DDGI',
    'gi12-ddgi': 'Cache: GI-1.2 grid over DDGI',
    sharc: 'Cache: SHaRC alone',
    budgeted: 'Cache: budgeted cells over DDGI',
} as const;
type TechniqueId = keyof typeof TECHNIQUES;
const MAX_BALLS = 30;
const FIRST_THROWS = 6;

const params = new URLSearchParams(location.search);
const asked = params.get('technique') ?? '';
const settings = {
    technique: (asked in TECHNIQUES ? asked : 'ddgi') as TechniqueId,
    glow: true,
    freeze: false,
};

// The screen inputs need a single-sampled pass. The cache's work is all per pixel, at full resolution, so the canvas
// draws at most about 1280 × 720 pixels' worth (the browser scales it to the window), the same for every technique.
const { renderer, device } = await createRendererAndDevice({ antialias: false });
const MAX_PIXELS = 1280 * 720;
const fitPixels = (): void => renderer.setPixelRatio(Math.min(window.devicePixelRatio, Math.sqrt(MAX_PIXELS / (window.innerWidth * window.innerHeight))));
fitPixels();
window.addEventListener('resize', fitPixels);

const note = document.createElement('div');
note.className = 'x3-place';
document.body.append(note);
const placeNote = (html: string): void => void (note.innerHTML = html);

// The place: Sponza's atrium, loaded from GitHub; the repo's procedural courtyard if that fails (or with `?place=courtyard`).
let place: Place;
if (params.get('place') !== 'courtyard') {
    placeNote('Loading Sponza from GitHub (about 53 MB)…');
    try {
        place = await loadSponza(renderer, (share) => placeNote(`Loading Sponza from GitHub (about 53 MB)… ${Math.round(share * 100)}%`));
        placeNote(`${SPONZA_CREDIT.text} <a href="${SPONZA_CREDIT.href}">Source and licence</a>`);
    } catch (error) {
        console.warn('x3: Sponza could not be loaded; using the procedural courtyard.', error);
        place = proceduralCourtyard(renderer);
        placeNote("Sponza couldn't be loaded (offline, or GitHub unreachable), so this is the repo's small procedural courtyard instead.");
    }
} else {
    place = proceduralCourtyard(renderer);
    placeNote(`The repo's small procedural courtyard, the page's offline fallback. <a href="?technique=${settings.technique}">Load Sponza</a>`);
}
const { scene, camera, controls } = place;
controls.enablePan = false; // the right button throws

/** One technique, its platform and how a frame is drawn with it. */
interface Rig extends Running {
    readonly threeGI: ThreeGI;
    render(): void;
    dispose(): void;
}

async function makeRig(id: TechniqueId): Promise<Rig> {
    if (id === 'ddgi') {
        // GI per fragment: the volume's light lights every material, and GI updates inside three's render.
        const volume = DDGIVolume.fromSettings(place.probes);
        scene.add(volume);
        await volume.start(renderer, scene, { camera });
        return {
            label: TECHNIQUES.ddgi,
            platform: volume.platform!,
            threeGI: volume.threeGI!,
            cache: null,
            render: () => renderer.render(scene, camera),
            dispose() {
                volume.dispose();
                volume.removeFromParent();
            },
        };
    }
    // A cache variant: every compare toggle at the preset's value; a DDGI volume under it where the preset asks for one.
    const preset = variantPreset(id)!;
    const cacheSettings: RadianceCacheSettings = { ...DEFAULT_RADIANCE_CACHE_SETTINGS };
    applyCompareValues(cacheSettings, preset.values);
    const gi: GIPlatform = createGI(device, { gpuTiming: device.features.has('timestamp-query') });
    const cache = gi.addTechnique(new RadianceCacheTechnique(cacheSettings));
    gi.setOutput(cache);
    let under: DDGITechnique | null = null;
    if (preset.needs.ddgiVolume) {
        under = gi.addTechnique(new DDGITechnique({ volume: place.probes }));
        gi.setFallback(cache, under); // before prepare: the cache builds its DDGI lookup then
    }
    const preparing = gi.prepare();
    const threeGI = new ThreeGI(gi, scene, { preparing, budget: { units: 256 } }); // DDGI's probes a frame
    await preparing;
    while (under?.warmingUp) {
        threeGI.update(camera, 0);
        await device.queue.onSubmittedWorkDone();
    }
    const pipeline = new THREE.RenderPipeline(renderer);
    const scenePass = pass(scene, camera);
    const screen = screenInputs(scenePass, threeGI);
    scenePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), ...screen.outputs }));
    pipeline.outputNode = screen.encodeBefore(ddgi(scenePass, threeGI));
    return {
        label: TECHNIQUES[id],
        platform: gi,
        threeGI,
        cache,
        render() {
            threeGI.update(camera); // the frame's GI work; the screen stage runs inside the pipeline
            pipeline.render();
        },
        dispose() {
            pipeline.dispose();
            screen.dispose();
            scenePass.dispose();
            threeGI.dispose();
            gi.dispose();
        },
    };
}

// The balls: one shared sphere, scaled per ball; each its own material.
const world = new SphereWorld(place.boxes);
const sphere = new THREE.SphereGeometry(1, 48, 24);
interface Glow {
    readonly base: number;
    readonly mode: 'steady' | 'pulse' | 'flicker';
    readonly rate: number;
    readonly phase: number;
    level: number;
    target: number;
}
interface Thrown {
    readonly body: Ball;
    readonly mesh: THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>;
    readonly glow: Glow | null;
}
const thrown: Thrown[] = [];
const random = THREE.MathUtils.randFloat;

function newMaterial(): { material: THREE.MeshStandardMaterial; glow: Glow | null } {
    const hue = Math.random();
    if (Math.random() < 0.5) {
        const mode = (['steady', 'pulse', 'flicker'] as const)[Math.floor(Math.random() * 3)];
        const glow: Glow = { base: random(1.2, 2.5), mode, rate: random(0.4, 1.2), phase: random(0, Math.PI * 2), level: 1, target: 1 };
        const material = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.4, emissive: new THREE.Color().setHSL(hue, 0.95, 0.5), emissiveIntensity: settings.glow ? glow.base : 0 });
        return { material, glow };
    }
    const metal = Math.random() < 0.3;
    const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color().setHSL(hue, random(0.3, 0.9), random(0.3, 0.7)),
        metalness: metal ? random(0.7, 1) : random(0, 0.2),
        roughness: random(0.08, 0.9),
    });
    return { material, glow: null };
}

/** Throws a ball in over the wall, aimed near `aim` on the floor if given. */
function throwBall(aim: THREE.Vector3 | null = null): void {
    if (thrown.length >= MAX_BALLS) removeBall(thrown[0]); // the oldest goes
    const radius = random(0.5, 1);
    const { from, to, seconds } = place.throwPath(radius, aim);
    // Reach `to` after `seconds` under gravity.
    const velocity = new THREE.Vector3().subVectors(to, from).divideScalar(seconds);
    velocity.y += 0.5 * 9.81 * seconds;
    const body = world.add(from, velocity, radius);
    const { material, glow } = newMaterial();
    const mesh = new THREE.Mesh(sphere, material);
    mesh.scale.setScalar(radius);
    mesh.position.copy(from);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    scene.add(mesh);
    thrown.push({ body, mesh, glow });
    changed('throw');
}

function removeBall(ball: Thrown): void {
    thrown.splice(thrown.indexOf(ball), 1);
    world.remove(ball.body);
    ball.mesh.removeFromParent();
    ball.mesh.material.dispose();
}

/** A glowing ball's brightness at `seconds`: steady, a slow pulse, or a candle's flicker. */
function glowLevel(glow: Glow, seconds: number): number {
    if (glow.mode === 'pulse') return 0.15 + 0.85 * (0.5 + 0.5 * Math.sin(seconds * glow.rate * Math.PI * 2 + glow.phase));
    if (glow.mode === 'flicker') {
        if (Math.random() < 0.12) glow.target = random(0.25, 1);
        glow.level += (glow.target - glow.level) * 0.35;
    }
    return glow.level;
}

// The readouts: the chart of the picture's mean, with a marker at each throw; GPU time per pass; the stats line.
const timeline = createTimeline({ label: 'Mean picture brightness (display value)', frames: 240, format: (value) => value.toFixed(3), settleTolerance: 0.03, settleHold: 20 });
let pendingMark: string | null = null;
const changed = (label: string): void => void (pendingMark = label);
let rig: Rig | null = null;
const passes = showPassTiming(() => rig?.platform ?? null);
const pictureMean = new PictureMean(device, renderer.domElement);
let chartQueue = Promise.resolve();
let frameMs = 16.7;
showStatsLine(() => ({ running: rig, gpuMs: passes.totalMs, balls: thrown.length, moving: world.moving, frameMs, place: place.name === 'sponza' ? 'Sponza' : 'procedural courtyard', pixels: `${renderer.domElement.width}×${renderer.domElement.height}` }));

// The GI mean for the site check, every two seconds; a switch waits for one in flight, since it reads the old platform's buffers.
let measuring = Promise.resolve();
setInterval(() => {
    const running = rig;
    if (running) measuring = measuring.then(() => measureGIMean(device, running));
}, 2000);

async function switchTo(id: TechniqueId): Promise<void> {
    const old = rig;
    rig = null; // direct light only meanwhile
    await measuring;
    old?.dispose();
    timeline.clear();
    rig = await makeRig(id);
    const url = new URL(location.href);
    url.searchParams.set('technique', id);
    history.replaceState(null, '', url);
}

// The panel.
let switching = switchTo(settings.technique);
const gui = new GUI({ title: 'Balls in the courtyard' });
const techniqueChoices = Object.fromEntries(Object.entries(TECHNIQUES).map(([id, label]) => [label, id])) as Record<string, TechniqueId>;
gui.add(settings, 'technique', techniqueChoices)
    .name('GI')
    .onChange((id: TechniqueId) => (switching = switching.then(() => switchTo(id)))); // one switch at a time
gui.add({ throw: () => throwBall() }, 'throw').name('throw a ball (or right-click)');
gui.add(
    {
        clear: () => {
            for (const ball of [...thrown]) removeBall(ball);
            changed('clear');
        },
    },
    'clear',
).name('clear');
gui.add(settings, 'glow').name('glowing balls glow').onChange((on: boolean) => {
    for (const ball of thrown) {
        if (!ball.glow) continue;
        ball.mesh.material.emissiveIntensity = on ? ball.glow.base * ball.glow.level : 0;
        rig?.threeGI.markChanged(ball.mesh);
    }
    changed(on ? 'glow on' : 'glow off');
});
gui.add(settings, 'freeze').name('freeze the balls');

// Right-click throws, aimed at the floor under the pointer.
const raycaster = new THREE.Raycaster();
const floor = new THREE.Plane(new THREE.Vector3(0, 1, 0), -place.floorY);
renderer.domElement.addEventListener('contextmenu', (event) => {
    event.preventDefault();
    const rect = renderer.domElement.getBoundingClientRect();
    raycaster.setFromCamera(new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1), camera);
    throwBall(raycaster.ray.intersectPlane(floor, new THREE.Vector3()));
});

// The first few balls, once GI is running.
await switching;
for (let index = 0; index < FIRST_THROWS; index++) setTimeout(() => throwBall(), index * 450);

const timer = new THREE.Timer();
renderer.setAnimationLoop(() => {
    timer.update();
    const delta = timer.getDelta();
    frameMs += (delta * 1000 - frameMs) * 0.05;
    if (!settings.freeze) {
        for (const body of world.step(delta)) {
            const ball = thrown.find((candidate) => candidate.body === body);
            if (ball) removeBall(ball);
        }
        // Only moving balls are touched: a settled ball's matrix stays as it is, so GI sees it still.
        for (const { body, mesh } of thrown) {
            if (body.asleep) continue;
            mesh.position.copy(body.position);
            mesh.quaternion.copy(body.quaternion);
        }
    }
    if (settings.glow) {
        const seconds = timer.getElapsed();
        for (const { mesh, glow } of thrown) {
            if (!glow || glow.mode === 'steady') continue;
            mesh.material.emissiveIntensity = glow.base * glowLevel(glow, seconds);
            rig?.threeGI.markChanged(mesh); // a material edit: seen by GI in the next update
        }
    }
    if (rig) rig.render();
    else renderer.render(scene, camera); // between techniques: three's direct light only
    // The chart, in frame order, the marker after its frame's value.
    const mean = pictureMean.read();
    const mark = pendingMark;
    pendingMark = null;
    chartQueue = chartQueue.then(async () => {
        timeline.push(await mean);
        if (mark) timeline.mark(mark);
    });
});
