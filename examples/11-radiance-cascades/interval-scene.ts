import * as THREE from 'three/webgpu';
import {
    createRcTransport, type RcPointLight, type RcRGB, type RcSurfaceMaterial,
} from '../../packages/core/src/techniques/radiance-cascades/transport.ts';

/** Converts an opaque Three scene into one static BLAS and one object-space BLAS per moving mesh. */
export async function createIntervalScene(
    device: GPUDevice,
    scene: THREE.Scene,
    options: { movingMeshes?: readonly THREE.Mesh[] } = {},
) {
    const movingSet = new Set(options.movingMeshes ?? []);
    const materials: THREE.MeshStandardMaterial[] = [];
    const materialIndices = new Map<THREE.Material, number>();
    const pointLights: THREE.PointLight[] = [];
    type Part = { mesh?: THREE.Mesh; positions: number[]; ids: number[] };
    const fixed: Part = { positions: [], ids: [] };
    const parts = [fixed];
    const vertex = new THREE.Vector3();
    scene.updateMatrixWorld(true);
    scene.traverseVisible(object => {
        if (object instanceof THREE.PointLight) { pointLights.push(object); return; }
        if (object instanceof THREE.Light) throw new Error('Interval scene supports emissive geometry and point lights. Unsupported ' + object.type);
        if (!(object instanceof THREE.Mesh)) return;
        if (object instanceof THREE.SkinnedMesh || object instanceof THREE.InstancedMesh || object instanceof THREE.BatchedMesh) throw new Error('Interval scene requires rigid meshes: ' + object.name);
        const moving = movingSet.has(object) || object.userData.rcMovable === true;
        const part: Part = moving ? { mesh: object, positions: [], ids: [] } : fixed;
        if (moving) parts.push(part);
        const geo = object.geometry, positions = geo.getAttribute('position'), indices = geo.getIndex();
        if (!positions) return;
        const count = indices?.count ?? positions.count;
        const groups = Array.isArray(object.material) ? geo.groups : [{ start: 0, count, materialIndex: 0 }];
        const order = !moving && object.matrixWorld.determinant() < 0 ? [0, 2, 1] : [0, 1, 2];
        for (const group of groups) {
            const material = Array.isArray(object.material) ? object.material[group.materialIndex ?? 0] : object.material;
            if (!material?.visible) continue;
            if (!(material instanceof THREE.MeshStandardMaterial) || material.transparent || material.alphaTest > 0 || material.map || material.emissiveMap) {
                throw new Error('Interval scene expects opaque, untextured MeshStandardMaterial: ' + material.name);
            }
            let materialId = materialIndices.get(material);
            if (materialId === undefined) {
                materialId = materials.length;
                if (materialId >= 65535) throw new RangeError('Too many RC materials');
                materials.push(material);
                materialIndices.set(material, materialId);
            }
            const first = Math.max(group.start, geo.drawRange.start);
            const end = Math.min(count, group.start + group.count, geo.drawRange.start + geo.drawRange.count);
            for (let i = first; i + 2 < end; i += 3) {
                for (const corner of order) {
                    vertex.fromBufferAttribute(positions, indices ? indices.getX(i + corner) : i + corner);
                    if (!moving) vertex.applyMatrix4(object.matrixWorld);
                    part.positions.push(vertex.x, vertex.y, vertex.z);
                }
                part.ids.push(materialId);
            }
        }
    });

    const rgb = (c: THREE.Color, scale = 1): RcRGB => [c.r * scale, c.g * scale, c.b * scale];
    const surface = (m: THREE.MeshStandardMaterial): RcSurfaceMaterial => ({
        albedo: rgb(m.color, 1 - m.metalness),
        emission: rgb(m.emissive, m.emissiveIntensity),
        side: m.side === THREE.DoubleSide ? 'double' : m.side === THREE.BackSide ? 'back' : 'front',
    });
    const light = (l: THREE.PointLight): RcPointLight => {
        vertex.setFromMatrixPosition(l.matrixWorld);
        return { position: [vertex.x, vertex.y, vertex.z], intensity: rgb(l.color, l.visible ? l.intensity : 0), distance: l.distance, decay: l.decay };
    };
    const used = parts.filter(part => part.ids.length > 0);
    const identity = new THREE.Matrix4();
    const transport = await createRcTransport(device, {
        meshes: used.map(part => ({ positions: new Float32Array(part.positions), triangleMaterials: new Uint16Array(part.ids) })),
        instances: used.map((part, index) => ({ meshIndex: index, instanceIndex: index, objectToWorld: (part.mesh?.matrixWorld ?? identity).elements })),
        materials: materials.map(surface), lights: pointLights.map(light),
        sky: scene.background instanceof THREE.Color ? rgb(scene.background) : [0, 0, 0],
    });
    const moving = used.flatMap(({ mesh }, index) => mesh ? [{ mesh, index, last: mesh.matrixWorld.clone() }] : []);
    // Track the few material/light fields actually consumed, without hashing the geometry or reading GPU buffers.
    const previousMaterials = materials.map(m => [m.color.r, m.color.g, m.color.b, m.metalness, m.emissive.r, m.emissive.g, m.emissive.b, m.emissiveIntensity, m.side]);
    const previousLights = pointLights.map(() => Array<number>(10).fill(NaN));
    function updateRow(row: number[], values: number[]) {
        let changed = false;
        for (let i = 0; i < values.length; i++) {
            if (row[i] !== values[i]) { row[i] = values[i]; changed = true; }
        }
        return changed;
    }
    return {
        transport,
        sync() {
            let materialsChanged = false, lightsChanged = false;
            for (const { mesh, index, last } of moving) {
                mesh.updateWorldMatrix(true, false);
                if (last.equals(mesh.matrixWorld)) continue;
                last.copy(mesh.matrixWorld);
                transport.setTransform(index, last.elements);
            }
            materials.forEach((m, i) => {
                if (updateRow(previousMaterials[i], [m.color.r, m.color.g, m.color.b, m.metalness, m.emissive.r, m.emissive.g, m.emissive.b, m.emissiveIntensity, m.side])) materialsChanged = true;
            });
            pointLights.forEach((l, i) => {
                l.updateWorldMatrix(true, false);
                vertex.setFromMatrixPosition(l.matrixWorld);
                if (updateRow(previousLights[i], [vertex.x, vertex.y, vertex.z, l.color.r, l.color.g, l.color.b, l.intensity, l.distance, l.decay, Number(l.visible)])) lightsChanged = true;
            });
            if (materialsChanged) transport.setMaterials(materials.map(surface));
            if (lightsChanged) transport.setLights(pointLights.map(light));
        },
        dispose: () => transport.dispose(),
    };
}
