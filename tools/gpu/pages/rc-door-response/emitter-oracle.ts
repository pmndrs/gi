/**
 * From the GI workspace root:
 *   node tools/gpu/pages/rc-door-response/emitter-oracle.ts
 *
 * CPU midpoint quadrature of the constant panel's single diffuse bounce.
 * Samples the same 16 × 9 receiver region as measurement.ts. Output is
 * scene-linear outgoing luminance. Additional bounces and the dim guide
 * lamp are excluded. No renderer, browser or GPU is required.
 */
import assert from 'node:assert/strict';
import * as THREE from 'three/webgpu';
import { makeDoorChamber } from '../../../../examples/cascades/11-radiance-cascades/door-scene.ts';

const chamber = makeDoorChamber();
const camera = new THREE.PerspectiveCamera(62, 16 / 9, 0.05, 40);
chamber.pose(camera, 0, 'receiver');
camera.updateMatrixWorld();
type ChamberMesh = THREE.Mesh<THREE.BoxGeometry, THREE.MeshStandardMaterial>;
const meshes: ChamberMesh[] = [];
chamber.scene.traverse(object => {
    if (!(object instanceof THREE.Mesh)) return;
    assert(object.geometry instanceof THREE.BoxGeometry && object.material instanceof THREE.MeshStandardMaterial && !object.material.transparent,
        `The CPU visibility model requires opaque box meshes; found ${object.name}`);
    meshes.push(object as ChamberMesh);
});
const emitter = meshes.find(mesh => mesh.name === 'constant-emitter');
assert(emitter, 'missing constant emitter');
const emission = emitter.material.emissive.clone().multiplyScalar(emitter.material.emissiveIntensity);

function receiverSamples() {
    const raycaster = new THREE.Raycaster();
    const samples = [];
    for (let y = 9; y < 18; y++) {
        for (let x = 16; x < 32; x++) {
            raycaster.setFromCamera(new THREE.Vector2(2 * (x + 0.5) / 48 - 1, 1 - 2 * (y + 0.5) / 27), camera);
            const first = raycaster.intersectObjects<ChamberMesh>(meshes, false)[0];
            assert(first?.face && first.object !== emitter && first.object !== chamber.door.mesh, 'receiver contains moving or emissive geometry');
            const normal = first.face.normal.clone().applyNormalMatrix(new THREE.Matrix3().getNormalMatrix(first.object.matrixWorld));
            const reflected = first.object.material.color.clone().multiply(emission);
            const lambertLuma = (0.2126 * reflected.r + 0.7152 * reflected.g + 0.0722 * reflected.b) / Math.PI;
            samples.push({ point: first.point, normal, lambertLuma });
        }
    }
    return samples;
}

/** Intersect the interior of the receiver-to-emitter segment with the opaque boxes. */
function blocked(origin: THREE.Vector3, delta: THREE.Vector3, boxes: THREE.Box3[]) {
    return boxes.some(box => {
        let enter = 0.0001, exit = 0.9999;
        for (let axis = 0; axis < 3; axis++) {
            const o = origin.getComponent(axis), d = delta.getComponent(axis);
            const min = box.min.getComponent(axis), max = box.max.getComponent(axis);
            if (Math.abs(d) < 1e-10) {
                if (o < min || o > max) return false;
            } else {
                const a = (min - o) / d, b = (max - o) / d;
                enter = Math.max(enter, Math.min(a, b));
                exit = Math.min(exit, Math.max(a, b));
                if (exit < enter) return false;
            }
        }
        return true;
    });
}

function integrate(state: 'open' | 'closed', steps: number) {
    chamber.door.update(0, state);
    chamber.scene.updateMatrixWorld(true);
    for (const mesh of meshes) {
        assert([1,2,4,6,8,9].every(i => Math.abs(mesh.matrixWorld.elements[i]) < 1e-10),
            `The CPU visibility model requires axis-aligned geometry: ${mesh.name}`);
    }
    const blockers = meshes.filter(mesh => mesh !== emitter).map(mesh => new THREE.Box3().setFromObject(mesh));
    const area = new THREE.Box3().setFromObject(emitter!);
    const center = area.getCenter(new THREE.Vector3()), size = area.getSize(new THREE.Vector3());
    const toLight = new THREE.Vector3();
    const samples = receiverSamples();
    let total = 0;
    for (const { point, normal, lambertLuma } of samples) {
        for (const [axis, a, b] of [[0,1,2], [1,0,2], [2,0,1]]) {
            const faceArea = size.getComponent(a) * size.getComponent(b);
            for (const sign of [-1, 1]) {
                const plane = center.getComponent(axis) + sign * size.getComponent(axis) / 2;
                if ((point.getComponent(axis) - plane) * sign <= 0) continue;
                for (let y = 0; y < steps; y++) {
                    for (let x = 0; x < steps; x++) {
                        toLight.copy(center);
                        toLight.setComponent(axis, plane);
                        toLight.setComponent(a, center.getComponent(a) + ((x + 0.5) / steps - 0.5) * size.getComponent(a));
                        toLight.setComponent(b, center.getComponent(b) + ((y + 0.5) / steps - 0.5) * size.getComponent(b));
                        toLight.sub(point);
                        const cosReceiverTimesR = normal.dot(toLight);
                        const cosEmitterTimesR = -sign * toLight.getComponent(axis);
                        if (cosReceiverTimesR <= 0 || blocked(point, toLight, blockers)) continue;
                        // The unnormalized cosines each contribute a factor of r.
                        total += lambertLuma * cosReceiverTimesR * cosEmitterTimesR * faceArea / (toLight.lengthSq() ** 2 * steps ** 2);
                    }
                }
            }
        }
    }
    return total / samples.length;
}

const rows = [12, 24, 48].map(steps => ({ stepsPerFaceAxis: steps, open: integrate('open', steps), closed: integrate('closed', steps) }));
console.log(JSON.stringify({ units: 'scene-linear outgoing emitter-only luminance', receiverPixels: 144, rows }, null, 2));
assert(rows.every(row => row.open > 0 && row.closed === 0), 'shutter must admit open-state light and block all closed-state emitter sightlines');
assert(Math.abs(rows[2].open - rows[1].open) / rows[2].open < 0.01, 'receiver integral changed by more than 1% at the finest quadrature');
