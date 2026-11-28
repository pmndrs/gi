import * as THREE from 'three/webgpu';

/** One frame per simulation tick, independent of the refresh rate or GPU speed. */
export const REPLAY_FRAMES = 720;
export type LampCue = 'sequence' | 'amber' | 'blue' | 'off';
export type CameraPose = 'rail' | 'receiver';
export type PassagePreset = 'gallery' | 'polished' | 'roughness' | 'reflection-motion';
export type EmitterMotion = 'sequence' | 'left' | 'still' | 'right';
export const FLOOR_ROUGHNESSES = [0.12, 0.25, 0.4, 0.55, 0.8] as const;

const limestone = 0xd8c9b2;
const granite = 0x514f52;

/** A semicircular shell running along z, with outward normals on both its interior and exterior. */
function vault(radius: number, thickness: number, length: number): THREE.ExtrudeGeometry {
    const shape = new THREE.Shape();
    shape.moveTo(radius, 0);
    shape.absarc(0, 0, radius, 0, Math.PI, false);
    shape.lineTo(-radius - thickness, 0);
    shape.absarc(0, 0, radius + thickness, Math.PI, 0, true);
    shape.closePath();
    return new THREE.ExtrudeGeometry(shape, { depth: length, bevelEnabled: false, curveSegments: 24 });
}

export function makePassage(preset: PassagePreset = 'gallery') {
    const polished = preset !== 'gallery';
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x08090c);
    const stone = new THREE.MeshStandardMaterial({ color: limestone, roughness: 0.91 });
    const trim = new THREE.MeshStandardMaterial({ color: granite, roughness: 0.78 });
    const ceiling = new THREE.MeshStandardMaterial({ color: 0xa3a09c, roughness: 0.9 });
    const floor = new THREE.MeshStandardMaterial(polished
        ? { color: 0x9fa7ac, metalness: 0.8, roughness: 0.24 }
        : { color: 0xc3b6a7, roughness: 0.86 });
    const tile = new THREE.MeshStandardMaterial({ color: 0x807d77, roughness: 0.8 });
    const unit = new THREE.BoxGeometry();
    function block(x: number, y: number, z: number, w: number, h: number, d: number, material: THREE.MeshStandardMaterial) {
        const mesh = new THREE.Mesh(unit, material);
        mesh.scale.set(w, h, d);
        mesh.position.set(x, y, z);
        scene.add(mesh);
        return mesh;
    }

    if (preset === 'roughness') {
        const edges = [-5, -1.8, -0.6, 0.6, 1.8, 5];
        FLOOR_ROUGHNESSES.forEach((roughness, i) => {
            const material = floor.clone();
            material.roughness = roughness;
            block((edges[i] + edges[i + 1]) / 2, -0.14, 0, edges[i + 1] - edges[i], 0.28, 22, material);
        });
    } else {
        block(0, -0.14, 0, 10, 0.28, 22, floor);
    }
    for (const x of [-1.95, 1.95]) block(x, 0.002, 0, 0.07, 0.005, 22, tile);
    for (let z = -10; z <= 10; z += 2) block(0, 0.002, z, 3.84, 0.004, 0.018, tile);
    const shell = new THREE.Mesh(vault(2.9, 0.24, 22), ceiling);
    shell.position.set(0, 2.35, -11);
    scene.add(shell);
    const rib = vault(2.77, 0.14, 0.28);
    for (const z of [-9, -5, -1, 3, 7]) {
        const arch = new THREE.Mesh(rib, stone);
        arch.position.set(0, 2.35, z - 0.14);
        scene.add(arch);
        for (const side of [-1, 1]) {
            block(side * 2.9, 1.17, z, 0.32, 2.35, 0.48, stone);
            block(side * 2.86, 0.12, z, 0.48, 0.24, 0.62, trim);
            block(side * 2.87, 2.28, z, 0.49, 0.17, 0.62, stone);
        }
    }

    // Side chapels have fixed returns framing recessed emitters. Their spill
    // reaches the vault and the stone plinths through the openings.
    const warm = new THREE.MeshStandardMaterial({ color: 0x32251b, emissive: 0xff833b, emissiveIntensity: 12 });
    const blue = new THREE.MeshStandardMaterial({ color: 0x162737, emissive: 0x55afff, emissiveIntensity: 0 });
    for (const side of [-1, 1]) {
        block(side * 4.75, 1.2, 0, 0.24, 2.4, 22, stone);
        block(side * 3.82, 2.45, 0, 1.94, 0.22, 22, stone);
        for (const z of [-11, -9, -5, -1, 3, 7, 11]) {
            block(side * 3.82, 1.18, z, 1.94, 2.36, 0.26, stone);
        }
        for (const z of [-7, -3, 1, 5, 9]) {
            block(side * 2.98, 1.18, z + 0.8, 0.22, 2.36, 1.65, stone);
            block(side * 4.55, 1.15, z + 0.65, 0.08, 1.75, 2.25, side === -1 ? warm : blue);
        }
    }
    block(0, 2.75, -11, 10, 5.5, 0.3, stone);
    block(0, 2.75, 11, 10, 5.5, 0.3, stone);
    const plinth = new THREE.CylinderGeometry(0.62, 0.76, 0.46, 32);
    const sculpture = new THREE.TorusKnotGeometry(0.48, 0.13, 96, 12);
    for (const [x, z] of [[-1.75, -3], [1.72, 1.1], [0, -8.5]]) {
        const base = new THREE.Mesh(plinth, trim);
        base.position.set(x, 0.23, z);
        scene.add(base);
        const knot = new THREE.Mesh(sculpture, stone);
        knot.position.set(x, 1.13, z);
        knot.rotation.set(0.24, z * 0.24, 0.5);
        scene.add(knot);
    }
    // A fixed, dim axial lamp gives the direct-only image a geometric reference.
    const guide = new THREE.PointLight(0xffe4c2, 18, 13);
    guide.position.set(0, 3.6, -7.8);
    scene.add(guide);

    // At the fixed floor camera this broad source is above the frustum. Its
    // virtual image across y=0 remains inside the bottom half of the viewport.
    const panel = polished ? block(0, 3.75, -0.4, 1.9, 0.2, 2.2,
        new THREE.MeshStandardMaterial({ color: 0x22272c, emissive: 0x8dd9ff, emissiveIntensity: 12 })) : null;
    if (panel) panel.name = 'Moving overhead emitter';
    const projection = new THREE.Matrix4();
    const frustum = new THREE.Frustum();
    const panelBounds = new THREE.Box3();
    const projected = new THREE.Vector3();
    function updateReflection(frame: number, motion: EmitterMotion, roughness: number) {
        if (!panel) return;
        floor.roughness = roughness;
        const offset = motion === 'sequence' ? Math.sin(2 * Math.PI * frame / REPLAY_FRAMES)
            : motion === 'left' ? -1 : motion === 'right' ? 1 : 0;
        panel.position.x = 1.32 * offset;
        panel.updateMatrixWorld(true);
    }
    function reflectionState(camera: THREE.PerspectiveCamera) {
        if (!panel) return null;
        camera.updateMatrixWorld();
        projection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
        frustum.setFromProjectionMatrix(projection, THREE.WebGPUCoordinateSystem);
        projected.copy(panel.position);
        projected.y = -projected.y;
        projected.project(camera);
        return {
            emitterPosition: panel.position.toArray(), cameraPosition: camera.position.toArray(),
            emitterOnScreen: frustum.intersectsBox(panelBounds.setFromObject(panel)), mirrorUv: [(projected.x + 1) / 2, (1 - projected.y) / 2],
        };
    }

    function pose(camera: THREE.PerspectiveCamera, frame: number, view: CameraPose) {
        if (view === 'receiver' && polished) {
            camera.position.set(0, 1.35, 5.55);
            camera.lookAt(0, 0, 2.05);
        } else if (view === 'receiver') {
            camera.position.set(-0.6, 1.38, 3.3);
            camera.lookAt(1.65, 1.12, 0.75);
        } else {
            const angle = (frame / REPLAY_FRAMES) * 2 * Math.PI;
            const z = 1.25 + 5.4 * Math.cos(angle);
            camera.position.set(0.7 * Math.sin(angle), 1.48 + 0.09 * Math.sin(angle * 2), z);
            camera.lookAt(0.52 * Math.sin(angle + 0.7), 1.65, z - 5.4);
        }
    }
    function light(frame: number, requested: LampCue): string {
        const phase = frame % REPLAY_FRAMES;
        const cue = requested === 'sequence' ? (phase < 240 ? 'amber' : phase < 480 ? 'blue' : 'off') : requested;
        // The fixed-camera motion fixture isolates a single transported source.
        warm.emissiveIntensity = preset !== 'reflection-motion' && cue === 'amber' ? 16 : 0;
        blue.emissiveIntensity = preset !== 'reflection-motion' && cue === 'blue' ? 16 : 0;
        if (panel) {
            panel.material.emissive.setHex(cue === 'blue' ? 0x8dd9ff : 0xffbc74);
            panel.material.emissiveIntensity = cue === 'off' ? 0 : 14;
        }
        return cue;
    }
    return { scene, pose, light, panel, updateReflection, reflectionState };
}
