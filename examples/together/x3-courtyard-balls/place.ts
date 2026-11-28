/**
 * Where the balls land: Sponza's atrium, loaded at run time, or the repo's
 * small procedural `courtyard` when it can't be (offline) or with
 * `?place=courtyard`.
 *
 * Each place gives the three scene, its camera, the DDGI box around it, the
 * boxes the physics collides with, and where a thrown ball comes from and
 * may land.
 *
 * **Sponza** is Crytek's model as the Khronos glTF Sample Assets repository
 * publishes it (`Models/Sponza`, 262,267 triangles, 69 textures, about
 * 53 MB), fetched from GitHub at a pinned commit. The site doesn't ship it.
 * Its licence: see SPONZA_CREDIT below and the PR notes.
 *
 * Its collision boxes were measured from the glTF itself (node scale 0.008;
 * the file's units, about metres). Its triangles that cross the band 0.3 to
 * 2.4 m above the floor, mapped on a 0.1 m grid of the floor plan, give the
 * walls, columns and pots; the lowest triangle over each cell gives the
 * vaults. The ground floor, long axis along x:
 * - The floor is at y = −0.02. The outer walls' inner faces are at
 *   x = −11.4 and +10.4, z = −4.9 and +4.3.
 * - The atrium, open to the sky, is the strip between the two colonnades.
 *   Each colonnade (x −8.6 to +7.6) is six columns with a curtain hanging
 *   to the floor in each of its five bays: the curtains' faces are at
 *   z = −2.12 and +1.51 (the columns' backs at −2.55 and +1.96), and the
 *   column bases stand 0.6 m into the strip.
 * - The ground-floor arcade runs all round, under vaults whose lowest point
 *   is about 2.6 m: along both long sides behind the curtains, and across
 *   both ends, where it opens onto the atrium (the sky is open only over
 *   x −7.75 to +6.25). A ball can roll in under it at the ends, and round
 *   the corners behind the curtains.
 * - In the arcade: a big plant pot in each corner (1.3 m square), a
 *   fountain basin against each end wall (x −11.4 to −10.8 and +9.7 to
 *   +10.4, z −1.3 to +0.8), and pilasters on the end walls. The pilasters
 *   along the long walls stand out 0.1 to 0.2 m and are left out.
 * - Above: the first-floor gallery's floor is at about 4 m, the second's at
 *   about 7.4 m, and the roof's edge at about 10.3 to 10.9 m. The
 *   colonnades' boxes run up to 12 m: a ball can't leave the atrium.
 */

import * as THREE from 'three/webgpu';
import { SkyMesh } from 'three/addons/objects/SkyMesh.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import type { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { DDGIVolumeSettings } from '@pmndrs/gi/ddgi';
import { lightToThree } from '../../../scenes/src/three-scene.ts';
import { COURTYARD_HALF_SIZE_M, COURTYARD_WALL_HEIGHT_M, COURTYARD_WALL_THICKNESS_M, GATEWAY_CENTRE_X_M, GATEWAY_HEIGHT_M, GATEWAY_WIDTH_M } from '../../../scenes/src/procedural/courtyard.ts';
import { orbitCamera } from '../../shared/renderer.ts';
import { loadScene } from '../../shared/scene.ts';
import type { Box } from './physics.ts';

/** Khronos glTF Sample Assets, pinned to the commit last touching `Models/Sponza` (2026-09-26). */
const SPONZA_COMMIT = '4995a638c96dbc94376d3d6164e70fee3ac7f7a2';
export const SPONZA_URL = `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/${SPONZA_COMMIT}/Models/Sponza/glTF/Sponza.gltf`;
export const SPONZA_CREDIT = {
    text: 'Sponza: © 2016 Crytek, under the CRYENGINE Limited License Agreement, from the Khronos glTF Sample Assets (Models/Sponza), loaded from GitHub when you open this page. Not part of this site.',
    href: 'https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Sponza',
};

export interface Place {
    readonly name: 'sponza' | 'courtyard';
    readonly scene: THREE.Scene;
    readonly camera: THREE.PerspectiveCamera;
    readonly controls: OrbitControls;
    /** The DDGI box: around the courtyard and what the camera sees of it. */
    readonly probes: DDGIVolumeSettings;
    /** What the balls collide with. */
    readonly boxes: readonly Box[];
    /** A thrown ball's start, where it lands and its flight time, for a ball of `radius`; `aim` is a floor point to aim near, if any. */
    throwPath(radius: number, aim: THREE.Vector3 | null): { from: THREE.Vector3; to: THREE.Vector3; seconds: number };
    /** The floor's height, for aiming. */
    readonly floorY: number;
}

const box = (min: [number, number, number], max: [number, number, number]): Box => ({ min, max });
const random = THREE.MathUtils.randFloat;

/** Sponza's ground floor (see the comment at the top). */
const SPONZA_FLOOR = -0.02;
const SPONZA_CURTAINS_Z: readonly [number, number] = [-2.12, 1.51];
const SPONZA_OPEN_X: readonly [number, number] = [-7.75, 6.25];
const SPONZA_COLUMNS_X: ReadonlyArray<readonly [number, number]> = [
    [-8.6, -7.4],
    [-5.4, -4.5],
    [-2.5, -1.5],
    [0.4, 1.4],
    [3.5, 4.4],
    [6.4, 7.6],
];
const VAULT = 2.6;
const TOP = 12;
const SPONZA_BOXES: readonly Box[] = [
    box([-13, -2, -7], [12, SPONZA_FLOOR, 6]), // the floor
    // The outer walls.
    box([-13, SPONZA_FLOOR, -7], [-11.4, TOP, 6]),
    box([10.4, SPONZA_FLOOR, -7], [12, TOP, 6]),
    box([-13, SPONZA_FLOOR, -7], [12, TOP, -4.9]),
    box([-13, SPONZA_FLOOR, 4.3], [12, TOP, 6]),
    // The two colonnades: columns and curtains, up through the galleries.
    box([-8.6, SPONZA_FLOOR, -2.55], [7.6, TOP, SPONZA_CURTAINS_Z[0]]),
    box([-8.6, SPONZA_FLOOR, SPONZA_CURTAINS_Z[1]], [7.6, TOP, 1.96]),
    // The column bases, into the atrium.
    ...SPONZA_COLUMNS_X.flatMap(([x0, x1]) => [box([x0, SPONZA_FLOOR, -2.2], [x1, 1.8, -1.5]), box([x0, SPONZA_FLOOR, 0.9], [x1, 1.8, 1.6])]),
    // The vaults over the arcade: everywhere but the open atrium.
    box([-13, VAULT, -7], [SPONZA_OPEN_X[0], TOP, 6]),
    box([SPONZA_OPEN_X[1], VAULT, -7], [12, TOP, 6]),
    box([SPONZA_OPEN_X[0], VAULT, -7], [SPONZA_OPEN_X[1], TOP, SPONZA_CURTAINS_Z[0]]),
    box([SPONZA_OPEN_X[0], VAULT, SPONZA_CURTAINS_Z[1]], [SPONZA_OPEN_X[1], TOP, 6]),
    // The corner pots.
    box([-10.3, SPONZA_FLOOR, -4.2], [-9.0, 1.6, -2.9]),
    box([8.3, SPONZA_FLOOR, -4.2], [9.6, 1.6, -2.9]),
    box([-10.3, SPONZA_FLOOR, 2.6], [-9.0, 1.6, 3.9]),
    box([8.3, SPONZA_FLOOR, 2.6], [9.6, 1.6, 3.9]),
    // The fountain basins and the end walls' pilasters.
    box([-11.4, SPONZA_FLOOR, -1.3], [-10.8, 0.9, 0.8]),
    box([9.7, SPONZA_FLOOR, -1.3], [10.4, 0.9, 0.8]),
    box([-11.4, SPONZA_FLOOR, -2.7], [-11.1, TOP, -1.8]),
    box([-11.4, SPONZA_FLOOR, 1.2], [-11.1, TOP, 1.9]),
    box([10.1, SPONZA_FLOOR, -2.7], [10.4, TOP, -1.8]),
    box([10.1, SPONZA_FLOOR, 1.2], [10.4, TOP, 1.9]),
];

/** Sponza from Khronos's glTF, under the sky with a high sun that reaches the courtyard's floor. Throws if it can't be fetched. */
export async function loadSponza(renderer: THREE.WebGPURenderer, onProgress: (share: number) => void): Promise<Place> {
    const gltf = await new GLTFLoader().loadAsync(SPONZA_URL, (event) => {
        if (event.lengthComputable) onProgress(event.loaded / event.total);
    });
    const scene = new THREE.Scene();
    gltf.scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
            object.castShadow = true;
            object.receiveShadow = true;
        }
    });
    scene.add(gltf.scene);
    // The sun steep enough to reach the floor between 11 m walls 3.6 m apart, and the sky to match.
    const sun = lightToThree({ type: 'directional', label: 'sun', color: [1, 0.95, 0.85], intensity: 3, direction: [0.4, -0.9, 0.12] }, { center: [-0.5, 5.2, -0.3], radius: 18 }) as THREE.DirectionalLight;
    scene.add(sun, sun.target);
    const sky = new SkyMesh();
    sky.scale.setScalar(1000);
    sky.turbidity.value = 3;
    sky.sunPosition.value.set(-0.4, 0.9, -0.12).normalize();
    scene.add(sky);
    renderer.toneMappingExposure = 2;

    // From the first-floor gallery at the west end, down the atrium.
    const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.05, 2000);
    camera.position.set(-8.2, 5.2, -0.3);
    const controls = orbitCamera(renderer, camera, new THREE.Vector3(3, 0.6, -0.3));
    return {
        name: 'sponza',
        scene,
        camera,
        controls,
        // The ground floor, its arcades and the galleries above, a probe every metre.
        probes: { center: [-0.5, 5.8, -0.2], size: [22.2, 11.8, 10.4], spacing: 1 },
        boxes: SPONZA_BOXES,
        floorY: SPONZA_FLOOR,
        throwPath(radius, aim) {
            // Over the second-floor gallery's railing (about 8.6 m) or the roof's edge (about 11.6 m), on either side, down into the atrium.
            const z = Math.random() < 0.5 ? SPONZA_CURTAINS_Z[0] + radius + 0.05 : SPONZA_CURTAINS_Z[1] - radius - 0.05;
            const height = (Math.random() < 0.5 ? 8.6 : 11.6) + radius;
            const inside = (x: number): number => THREE.MathUtils.clamp(x, SPONZA_OPEN_X[0] + radius, SPONZA_OPEN_X[1] - radius);
            const to = new THREE.Vector3(inside(aim?.x ?? random(-6.5, 5)), SPONZA_FLOOR + radius, THREE.MathUtils.clamp(aim?.z ?? random(-1.4, 0.8), SPONZA_CURTAINS_Z[0] + radius, SPONZA_CURTAINS_Z[1] - radius));
            // Thrown along the atrium too, so it lands rolling: some roll on in under the arcade at the ends.
            const from = new THREE.Vector3(inside(to.x + (Math.random() < 0.5 ? -1 : 1) * random(1, 4)), height, z);
            return { from, to, seconds: Math.sqrt((2 * (height - to.y)) / 9.81) + 0.2 };
        },
    };
}

/** The repo's procedural courtyard (scenes/src/procedural/courtyard.ts): 10 m square, 2.6 m walls, one gateway. */
export function proceduralCourtyard(renderer: THREE.WebGPURenderer): Place {
    const { scene, camera, controls, probes, lights, sky } = loadScene('courtyard-tree', renderer);
    // Dusk: a low sun on the wall tops and a dim blue sky (GI reads a background colour as a uniform sky), so the glowing balls' light shows.
    const sun = lights[0] as THREE.DirectionalLight;
    const toSun = new THREE.Vector3(-0.9, 0.22, 0.38).normalize();
    sun.position.copy(sun.target.position).addScaledVector(toSun, sun.position.distanceTo(sun.target.position));
    sun.color.setRGB(1, 0.7, 0.45);
    sky?.removeFromParent();
    scene.background = new THREE.Color(0.03, 0.05, 0.1);
    // Above the south-east corner, looking down into the yard.
    camera.position.set(5.5, 8.5, 9);
    controls.target.set(-0.5, 0, -0.5);
    controls.update();
    const half = COURTYARD_HALF_SIZE_M;
    const outer = half + COURTYARD_WALL_THICKNESS_M;
    const top = COURTYARD_WALL_HEIGHT_M;
    const gateway = [GATEWAY_CENTRE_X_M - GATEWAY_WIDTH_M / 2, GATEWAY_CENTRE_X_M + GATEWAY_WIDTH_M / 2];
    // The builder's own boxes: the paving, three walls, the south wall either side of the gateway and its lintel, the bench.
    const boxes: Box[] = [
        box([-outer, -0.3, -outer], [outer, 0, outer]),
        box([-outer, -0.3, -outer], [-half, top, outer]),
        box([half, -0.3, -outer], [outer, top, outer]),
        box([-outer, -0.3, -outer], [outer, top, -half]),
        box([-outer, -0.3, half], [gateway[0], top, outer]),
        box([gateway[1], -0.3, half], [outer, top, outer]),
        box([gateway[0], GATEWAY_HEIGHT_M, half], [gateway[1], top, outer]),
        box([-4.4, 0, -half], [-2.6, 0.48, -4.5]),
    ];
    return {
        name: 'courtyard',
        scene,
        camera,
        controls,
        // The scene's suggested box: the yard and its walls, a probe every 0.5 m.
        probes,
        boxes,
        floorY: 0,
        throwPath(radius, aim) {
            const inside = half - radius - 0.1;
            const to = new THREE.Vector3(THREE.MathUtils.clamp(aim?.x ?? random(-3.5, 3.5), -inside, inside), radius, THREE.MathUtils.clamp(aim?.z ?? random(-3.5, 3.5), -inside, inside));
            // From 2.5 m outside the west or north wall, 4.5 m up: the walls across from the camera, so a ball is seen coming over.
            const along = random(-3, 3);
            const from = Math.random() < 0.5 ? new THREE.Vector3(-outer - 2.5, 4.5, along) : new THREE.Vector3(along, 4.5, -outer - 2.5);
            return { from, to, seconds: 1.3 };
        },
    };
}
