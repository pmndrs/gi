import * as THREE from 'three/webgpu';
import { REPLAY_FRAMES, type CameraPose, type LampCue } from './scene.ts';

export type DoorMode = 'sequence' | 'open' | 'closed';
export const DOOR_TIMELINE = {
    closeStart: 120, closed: 150, openStart: 390, opened: 420,
} as const;

function sequenceOpenness(frame: number): number {
    const t = ((frame % REPLAY_FRAMES) + REPLAY_FRAMES) % REPLAY_FRAMES;
    const { closeStart, closed, openStart, opened } = DOOR_TIMELINE;
    if (t < closeStart || t >= opened) return 1;
    if (t >= closed && t < openStart) return 0;
    const closing = t < closed;
    const u = closing ? (t - closeStart) / (closed - closeStart) : (t - openStart) / (opened - openStart);
    const eased = u * u * (3 - 2 * u);
    return closing ? 1 - eased : eased;
}

/** A fixed lamp chamber at the side of a gallery. Only the shutter transform changes. */
export function makeDoorChamber() {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0);
    const plaster = new THREE.MeshStandardMaterial({ color: 0xc6bdab, roughness: 1 });
    const limestone = new THREE.MeshStandardMaterial({ color: 0xe1dccf, roughness: 1 });
    const slate = new THREE.MeshStandardMaterial({ color: 0x3d4348, roughness: 0.93 });
    const bronze = new THREE.MeshStandardMaterial({ color: 0x665345, roughness: 0.86 });
    const lamp = new THREE.MeshStandardMaterial({ color: 0x21201c, emissive: 0xffdbac, emissiveIntensity: 12 });
    const unit = new THREE.BoxGeometry();
    function box(name: string, material: THREE.MeshStandardMaterial, x: number, y: number, z: number, width: number, height: number, depth: number) {
        const mesh = new THREE.Mesh(unit, material);
        mesh.name = name;
        mesh.position.set(x, y, z);
        mesh.scale.set(width, height, depth);
        scene.add(mesh);
        return mesh;
    }

    // Floor, ceiling, and exterior walls seal the two spaces. The illuminated
    // annex occupies x=2.48..4.20. The receiver remains on the opposite side.
    box('gallery-floor', slate, 0.8, -0.14, 0, 7.2, 0.28, 6.6);
    box('gallery-ceiling', plaster, 0.8, 3.34, 0, 7.2, 0.28, 6.6);
    box('gallery-west', plaster, -2.68, 1.6, 0, 0.24, 3.2, 6.6);
    box('annex-east', plaster, 4.28, 1.6, 0, 0.24, 3.2, 6.6);
    box('receiver-wall', limestone, 0.8, 1.6, -3.18, 7.2, 3.2, 0.24);
    box('gallery-entry', plaster, 0.8, 1.6, 3.18, 7.2, 3.2, 0.24);
    // The center third looks at this fixed wall panel and its surrounding
    // stone. No emissive or moving surface contributes directly to that ROI.
    box('receiver-panel', limestone, -0.10, 1.53, -3.01, 2.9, 2.37, 0.08);
    for (const x of [-2.25, 1.85]) {
        box('pilaster', plaster, x, 1.6, -2.88, 0.32, 3.2, 0.36);
        box('pilaster-foot', slate, x, 0.12, -2.86, 0.43, 0.24, 0.44);
    }
    // Wall x=2.36 contains a 2.05 × 2.40 m opening in z and y. The
    // closed leaf overlaps its sill, head and jambs; the open leaf slides
    // behind the south jamb, leaving the aperture completely clear.
    const aperture = { lowZ: -1.85, highZ: 0.2, lowY: 0.32, highY: 2.72 };
    const apertureZ = (aperture.lowZ + aperture.highZ) / 2;
    const apertureY = (aperture.lowY + aperture.highY) / 2;
    box('annex-north-jamb', plaster, 2.36, 1.6, (-3.3 + aperture.lowZ) / 2, 0.28, 3.2, aperture.lowZ + 3.3);
    box('annex-south-jamb', plaster, 2.36, 1.6, (3.3 + aperture.highZ) / 2, 0.28, 3.2, 3.3 - aperture.highZ);
    box('annex-sill', slate, 2.36, aperture.lowY / 2, apertureZ, 0.28, aperture.lowY, 2.05);
    box('annex-lintel', plaster, 2.36, (3.2 + aperture.highY) / 2, apertureZ, 0.28, 3.2 - aperture.highY, 2.05);
    const mesh = box('sliding-lamp-shutter', bronze, 2.55, apertureY, apertureZ, 0.16, 2.58, 2.23);
    box('constant-emitter', lamp, 4.08, 1.61, -0.32, 0.12, 2.65, 4.35);
    // A very dim guide in the gallery makes the closed chamber visible. Its
    // finite range stops short of the center receiver.
    const guide = new THREE.PointLight(0xffdfb4, 0.5, 2.1);
    guide.position.set(-2.12, 2.75, 1.7);
    scene.add(guide);

    let openness = 1;
    function setOpenness(value: number) {
        openness = THREE.MathUtils.clamp(value, 0, 1);
        mesh.position.z = apertureZ + 2.30 * openness;
        mesh.updateMatrixWorld(true);
        return openness;
    }
    setOpenness(1);
    const door = {
        mesh, setOpenness,
        get openness() { return openness; },
        update(frame: number, mode: DoorMode) {
            return setOpenness(mode === 'sequence' ? sequenceOpenness(frame) : mode === 'open' ? 1 : 0);
        },
    };
    function pose(camera: THREE.PerspectiveCamera, frame: number, view: CameraPose | 'door') {
        if (view === 'rail') {
            const angle = (frame / REPLAY_FRAMES) * 2 * Math.PI;
            camera.position.set(-0.82 + 0.52 * Math.sin(angle), 1.45, 2.0 + 0.48 * Math.cos(angle));
            camera.lookAt(0.50, 1.4, -1.48);
        } else {
            camera.position.set(-0.48, 1.46, 2.30);
            camera.lookAt(-0.10, 1.48, -3.01);
        }
    }
    function light(_frame: number, _cue: LampCue) {
        return 'constant amber';
    }
    return { scene, pose, light, door };
}
