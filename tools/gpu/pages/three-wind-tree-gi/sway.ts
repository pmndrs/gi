/**
 * A CPU sway for the wind tree (T3.11's asset), driven by its wind weight:
 * COLOR_0's alpha, 0 at the trunk's foot, up to 0.6 on the tubes and 0.7 to 1
 * on the leaf cards (tools/assets/convert/wind-tree.ts). Each vertex moves
 * sideways by its weight times the sway, so the foot stays put, the crown
 * moves most, and a leaf card moves as one piece.
 *
 * The positions are written into the mesh's own position attribute
 * (`needsUpdate`), as m3-moving.md §7's route 1 has it: three draws the sway,
 * and the three adapter sends the same positions to GI
 * (`userData.giDeforming`). A sway of `s` (−1 to 1) moves a vertex of weight
 * `w` by `s × w × reach` along `direction`, in the mesh's own space.
 *
 * Kept small and standalone so the viewer's wind scenario (T3.8) can take it.
 */

import type { BufferAttribute, Mesh } from 'three/webgpu';

export interface SwayOptions {
    /** Metres a vertex of weight 1 moves at full sway. */
    readonly reach: number;
    /** Unit direction of the sway in the mesh's own space. */
    readonly direction: readonly [number, number, number];
}

export class WindSway {
    readonly mesh: Mesh;
    private readonly rest: Float32Array;
    private readonly weights: Float32Array;
    private readonly options: SwayOptions;

    /** Throws if the mesh has no RGBA vertex colour (no wind weight). */
    constructor(mesh: Mesh, options: SwayOptions) {
        const position = mesh.geometry.getAttribute('position') as BufferAttribute;
        const color = mesh.geometry.getAttribute('color') as BufferAttribute | undefined;
        if (!color || color.itemSize !== 4) throw new Error(`wind sway: mesh "${mesh.name}" has no RGBA vertex colour to read the wind weight from`);
        this.mesh = mesh;
        this.options = options;
        this.rest = Float32Array.from(position.array as Float32Array);
        this.weights = new Float32Array(position.count);
        for (let vertex = 0; vertex < position.count; vertex++) this.weights[vertex] = color.getW(vertex);
    }

    /** The largest wind weight on the mesh. */
    get largestWeight(): number {
        let largest = 0;
        for (const weight of this.weights) largest = Math.max(largest, weight);
        return largest;
    }

    /**
     * The box every sway stays in, in the mesh's own space: the rest box
     * stretched by the farthest any vertex goes each way. For
     * `userData.giDeformBounds`.
     */
    swayBounds(): { min: [number, number, number]; max: [number, number, number] } {
        const min: [number, number, number] = [Infinity, Infinity, Infinity];
        const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
        const reach = this.options.reach;
        for (let vertex = 0; vertex < this.weights.length; vertex++) {
            for (let axis = 0; axis < 3; axis++) {
                const rest = this.rest[vertex * 3 + axis];
                const offset = Math.abs(this.weights[vertex] * reach * this.options.direction[axis]);
                min[axis] = Math.min(min[axis], rest - offset);
                max[axis] = Math.max(max[axis], rest + offset);
            }
        }
        return { min, max };
    }

    /** Poses the mesh at sway `s` (−1 to 1). */
    pose(s: number): void {
        const position = this.mesh.geometry.getAttribute('position') as BufferAttribute;
        const array = position.array as Float32Array;
        const [dx, dy, dz] = this.options.direction;
        const reach = this.options.reach;
        for (let vertex = 0; vertex < this.weights.length; vertex++) {
            const move = s * this.weights[vertex] * reach;
            array[vertex * 3] = this.rest[vertex * 3] + move * dx;
            array[vertex * 3 + 1] = this.rest[vertex * 3 + 1] + move * dy;
            array[vertex * 3 + 2] = this.rest[vertex * 3 + 2] + move * dz;
        }
        position.needsUpdate = true;
        // three culls with the bounding sphere: keep it holding the pose.
        this.mesh.geometry.boundingSphere = null;
    }
}
