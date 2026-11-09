/**
 * The cabinet scenario's "add" and "copy" (T3.8, docs/design/m3-moving.md §8,
 * "Add and duplicate cabinets"): small white cabinets put on the cabinet
 * row's floor in front of the run, in rows, up to 100.
 *
 * - **Add** makes a new mesh each time (its own `BoxGeometry`): GI copies a
 *   new shape and builds a ray-tracing tree for it on the GPU (T3.2).
 * - **Copy** adds another of the last added cabinet's mesh (the same
 *   geometry): GI only places that shape once more (a new instance).
 *
 * The three adapter finds added and removed objects by itself (T3.7), so
 * nothing here tells GI anything. Neither changes a GPU program: the
 * recompile counter stays 0.
 */

import { BoxGeometry, Mesh, MeshStandardMaterial, type BufferGeometry, type Object3D } from 'three';
import { LAYER_VIEW_GEOMETRY } from './layers';

/** The most the viewer adds (m3-moving.md §8: "up to 100"). */
export const MAX_ADDED_CABINETS = 100;
/** Metres: a small cabinet, 18 mm short of each grid step so neighbours never touch. */
const CABINET_SIZE: readonly [number, number, number] = [0.22, 0.25, 0.22];
/**
 * Where they go: a grid on the floor in front of the run, 1 m out from it (the
 * cabinet row is x 0 to 3 along the back wall at z 0, 0.58 m deep; the room is
 * 4.8 by 3.2 m). 10 across, 5 deep, then a second layer on top: 0.5 m high at
 * most, so the `from-the-right` camera still sees the opening door over them.
 */
const GRID = { first: [0.3, 0, 1.6] as const, step: [0.25, 0.25, 0.25] as const, across: 10, deep: 5 };
/** The cabinets' white, as the row's own (`cabinet-white`, linear). */
const CABINET_COLOR: readonly [number, number, number] = [0.78, 0.78, 0.75];

export class AddedCabinets {
    private readonly added: Mesh[] = [];
    /** Geometries this made, one per "add", disposed with it. */
    private readonly geometries: BufferGeometry[] = [];
    private readonly material = new MeshStandardMaterial({ name: 'added cabinet', roughness: 1, metalness: 0 });

    constructor(private readonly parent: Object3D) {
        this.material.color.setRGB(...CABINET_COLOR);
    }

    get count(): number {
        return this.added.length;
    }

    /** How many have their own mesh (each "add"), against copies of one (each "copy"). */
    get shapes(): number {
        return this.geometries.length;
    }

    get full(): boolean {
        return this.added.length >= MAX_ADDED_CABINETS;
    }

    /** A new cabinet with a mesh of its own. */
    add(): void {
        if (this.full) return;
        const geometry = new BoxGeometry(...CABINET_SIZE);
        this.geometries.push(geometry);
        this.place(geometry);
    }

    /** One more of the last added cabinet's mesh; the first is an "add". */
    copy(): void {
        if (this.full) return;
        const last = this.added[this.added.length - 1];
        if (!last) return this.add();
        this.place(last.geometry);
    }

    /** Takes every added cabinet away again. */
    clear(): void {
        for (const mesh of this.added) mesh.removeFromParent();
        this.added.length = 0;
        for (const geometry of this.geometries) geometry.dispose();
        this.geometries.length = 0;
    }

    dispose(): void {
        this.clear();
        this.material.dispose();
    }

    private place(geometry: BufferGeometry): void {
        const index = this.added.length;
        const perLayer = GRID.across * GRID.deep;
        const layer = Math.floor(index / perLayer);
        const column = index % GRID.across;
        const row = Math.floor((index % perLayer) / GRID.across);
        const mesh = new Mesh(geometry, this.material);
        mesh.name = `added-cabinet-${index + 1}`;
        mesh.position.set(
            GRID.first[0] + column * GRID.step[0],
            // Standing on the floor (y 0), or on the layer below.
            CABINET_SIZE[1] / 2 + layer * GRID.step[1],
            GRID.first[2] + row * GRID.step[2],
        );
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        // Layer 0 for GI, and the viewing cameras' layer, as the scene's own meshes (layers.ts).
        mesh.layers.enable(LAYER_VIEW_GEOMETRY);
        this.parent.add(mesh);
        // World matrix now, so GI's update this frame sees it where it stands.
        mesh.updateMatrixWorld(true);
        this.added.push(mesh);
    }
}
