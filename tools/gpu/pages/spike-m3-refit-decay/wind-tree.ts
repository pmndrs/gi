/**
 * T3.11's wind tree (tools/assets/convert/wind-tree.ts, 50,474 triangles in
 * three meshes: trunk, branches, leaf cards) as one swaying mesh, for the
 * T3.0 refit-decay spike. The three meshes are merged into one, because a
 * deforming tree would be refit as one mesh tree.
 *
 * The sway is a stand-in for the viewer's wind (T3.8), driven by the asset's
 * own wind weight (COLOR_0 alpha: 0 at the trunk's foot, up to 0.6 on
 * branches, 0.7 to 1 on the cards, the same for a card's four corners):
 * - the whole tree bends along the wind, by weight;
 * - a slow wave runs through the crown (its phase follows position, so
 *   neighbouring branches move together), by weight;
 * - each card flutters on its own phase: two of its corners move along its
 *   normal, so it tilts as one piece.
 * Amplitudes come from tree.ts's `SwayStrength`, scaled to this 10 m tree
 * (its top moves about as far as the procedural tree's top).
 */

import { Vector3, type BufferGeometry, type Mesh } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { findModelUrl } from '../../../../apps/viewer/src/scene/model-url.ts';
import type { SwayingMesh } from './tree.ts';

/** How far the weight-1 top bends, as a multiple of `trunkMetres` (the procedural tree's top reaches ~2.25x). */
const TOP_BEND_SCALE = 2.5;

export async function loadWindTree(): Promise<SwayingMesh> {
    const url = await findModelUrl('wind-tree', 'wind-tree.glb');
    const gltf = await new GLTFLoader().loadAsync(url);
    gltf.scene.updateMatrixWorld(true);
    const parts: { geometry: BufferGeometry; isLeaves: boolean }[] = [];
    gltf.scene.traverse((object) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh) return;
        parts.push({ geometry: mesh.geometry.clone().applyMatrix4(mesh.matrixWorld), isLeaves: /leaves/i.test(mesh.name) || /leaves/i.test(mesh.parent?.name ?? '') });
    });
    if (parts.length === 0) throw new Error('the wind tree has no meshes');

    const vertexTotal = parts.reduce((sum, part) => sum + part.geometry.getAttribute('position').count, 0);
    const restPositions = new Float32Array(vertexTotal * 3);
    const weights = new Float32Array(vertexTotal);
    const normals = new Float32Array(vertexTotal * 3);
    /** Per vertex: 1 for the two flapping corners of a card, else 0. */
    const flaps = new Float32Array(vertexTotal);
    const cardPhases = new Float32Array(vertexTotal);
    const indices: number[] = [];
    let base = 0;
    for (const part of parts) {
        const positions = part.geometry.getAttribute('position');
        const colours = part.geometry.getAttribute('color');
        const normal = part.geometry.getAttribute('normal');
        if (!colours || colours.itemSize < 4) throw new Error('the wind tree has no wind weight (COLOR_0 alpha)');
        for (let vertex = 0; vertex < positions.count; vertex++) {
            restPositions.set([positions.getX(vertex), positions.getY(vertex), positions.getZ(vertex)], (base + vertex) * 3);
            weights[base + vertex] = colours.getW(vertex);
            normals.set([normal.getX(vertex), normal.getY(vertex), normal.getZ(vertex)], (base + vertex) * 3);
            if (part.isLeaves) {
                // The generator writes each card's four corners in a row; corners 2 and 3 flap.
                const card = Math.floor(vertex / 4);
                flaps[base + vertex] = vertex % 4 >= 2 ? 1 : 0;
                cardPhases[base + vertex] = ((card * 2654435761) % 1000) / 1000 * Math.PI * 2;
            }
        }
        const index = part.geometry.getIndex();
        if (index) for (let entry = 0; entry < index.count; entry++) indices.push(base + index.getX(entry));
        else for (let vertex = 0; vertex < positions.count; vertex++) indices.push(base + vertex);
        base += positions.count;
    }

    const windX = 0.8;
    const windZ = 0.6;
    const waveDirection = new Vector3(0.2, 1, 0.2).normalize();
    // A box around every pose: the rest box grown by the strongest sway's reach.
    const reach = 0.4 * TOP_BEND_SCALE * 1.3 + 0.25 + 0.04 + 0.1;
    const swayBounds = new Float32Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
    for (let vertex = 0; vertex < vertexTotal; vertex++) {
        for (let axis = 0; axis < 3; axis++) {
            swayBounds[axis] = Math.min(swayBounds[axis], restPositions[vertex * 3 + axis] - reach);
            swayBounds[axis + 3] = Math.max(swayBounds[axis + 3], restPositions[vertex * 3 + axis] + reach);
        }
    }

    return {
        name: 'wind tree (T3.11)',
        restPositions,
        indices: new Uint32Array(indices),
        triangleCount: indices.length / 3,
        swayBounds,
        swayInto(out, seconds, strength) {
            const bendAlong = strength.trunkMetres * TOP_BEND_SCALE * Math.sin(2 * Math.PI * 0.4 * seconds);
            const bendAcross = 0.3 * strength.trunkMetres * TOP_BEND_SCALE * Math.sin(2 * Math.PI * 0.27 * seconds + 1);
            for (let vertex = 0; vertex < vertexTotal; vertex++) {
                const weight = weights[vertex];
                const x0 = restPositions[vertex * 3];
                const y0 = restPositions[vertex * 3 + 1];
                const z0 = restPositions[vertex * 3 + 2];
                // The crown's wave: about one wavelength across 6 m of crown, 1 Hz.
                const wave = strength.branchMetres * weight * Math.sin(2 * Math.PI * seconds + (x0 * 0.7 + y0 * 0.3 + z0 * 0.5) * 1.0);
                let x = x0 + weight * (bendAlong * windX - bendAcross * windZ) + waveDirection.x * wave;
                let y = y0 + waveDirection.y * wave;
                let z = z0 + weight * (bendAlong * windZ + bendAcross * windX) + waveDirection.z * wave;
                if (flaps[vertex] > 0) {
                    const flutter = strength.leafMetres * Math.sin(2 * Math.PI * 4 * seconds + cardPhases[vertex]);
                    x += normals[vertex * 3] * flutter;
                    y += normals[vertex * 3 + 1] * flutter;
                    z += normals[vertex * 3 + 2] * flutter;
                }
                out[vertex * 3] = x;
                out[vertex * 3 + 1] = y;
                out[vertex * 3 + 2] = z;
            }
        },
    };
}
