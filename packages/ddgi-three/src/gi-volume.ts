/**
 * A GI volume shaped like three's `LightProbeGrid` (SPEC, "Matching the
 * LightProbeGrid shape"), so its users can switch to DDGI with the same
 * numbers:
 *
 * ```ts
 * const volume = new GIVolume(width, height, depth, nx, ny, nz); // like new LightProbeGrid(...)
 * volume.position.set(x, y, z);
 * scene.add(volume);
 * scene.updateMatrixWorld();
 * const ddgi = gi.addTechnique(new DDGITechnique({ volume: volume.ddgiVolume() }));
 * gi.setOutput(ddgi);
 * await gi.prepare();
 * const threeGI = new ThreeGI(gi, scene, { volume });
 * ```
 *
 * As in `LightProbeGrid`, the box is `width × height × depth`, centred on the
 * object's world position and axis-aligned (rotation and scale are not
 * used), split into `nx × ny × nz` cells with a probe at each cell's centre.
 * `boundingBox`, `priority` and `intensity` mean what they mean there;
 * `ThreeGI` copies `priority` and `intensity` onto its GI light each frame,
 * so turning GI down or off works on the volume too.
 *
 * DDGI places probes on a grid from a box and a spacing, `floor(size /
 * spacing) + 1` per axis, centred in the box. Cell centres are that grid with
 * spacing = width / nx and a box of width − width / nx (the outermost probes
 * sit half a cell inside the volume), which {@link GIVolume.ddgiVolume} gives.
 * Changing the volume after `prepare` needs a new `DDGITechnique` settings
 * and another `prepare`, as for any DDGI volume.
 *
 * The volume itself adds no light: the GI light does (gi-light.ts). It is a
 * `Light` because `LightProbeGrid` is one; its registered node does nothing.
 */

import { AnalyticLightNode, Box3, Light, Vector3, type Node } from 'three/webgpu';
import type { DDGIVolumeSettings, Vec3 } from '@pmndrs/ddgi-core';
import { GI_LIGHT_PRIORITY } from './gi-light.ts';

/** `LightProbeGrid`'s default probe count for an axis: one per unit of length, at least one. */
function defaultProbes(length: number): number {
    return Math.max(1, Math.round(length));
}

export class GIVolume extends Light {
    readonly isGIVolume = true;
    /** Full size of the volume along each axis. */
    width: number;
    height: number;
    depth: number;
    /** Probes along each axis. */
    readonly resolution: Vector3;
    /** The box the volume covers, in world space. Updated by {@link updateBoundingBox} (and by `ddgiVolume`). */
    readonly boundingBox = new Box3();
    /** Where probe-like lights overlap, the highest priority lights the fragment. */
    priority = GI_LIGHT_PRIORITY;

    constructor(width = 1, height = 1, depth = 1, widthProbes?: number, heightProbes?: number, depthProbes?: number) {
        super(0xffffff, 1);
        (this as { type: string }).type = 'GIVolume';
        this.width = width;
        this.height = height;
        this.depth = depth;
        this.resolution = new Vector3(widthProbes ?? defaultProbes(width), heightProbes ?? defaultProbes(height), depthProbes ?? defaultProbes(depth));
    }

    /** The volume's centre: its world position (`matrixWorld` must be current). */
    private center(): Vec3 {
        const e = this.matrixWorld.elements;
        return [e[12], e[13], e[14]];
    }

    /** Sets `boundingBox` from the world position and the size. */
    updateBoundingBox(): Box3 {
        const [x, y, z] = this.center();
        this.boundingBox.min.set(x - this.width / 2, y - this.height / 2, z - this.depth / 2);
        this.boundingBox.max.set(x + this.width / 2, y + this.height / 2, z + this.depth / 2);
        return this.boundingBox;
    }

    /**
     * The DDGI volume settings with a probe at the centre of each of the
     * `resolution` cells: spacing = size / probes per axis, and a box one
     * cell smaller than the volume, so `floor(size / spacing) + 1` gives the
     * probe counts back. World matrices must be current.
     */
    ddgiVolume(): DDGIVolumeSettings {
        this.updateBoundingBox();
        const lengths: Vec3 = [this.width, this.height, this.depth];
        const probes: Vec3 = [this.resolution.x, this.resolution.y, this.resolution.z];
        const spacing = lengths.map((length, axis) => length / probes[axis]) as unknown as Vec3;
        const size = lengths.map((length, axis) => length - spacing[axis]) as unknown as Vec3;
        return { center: this.center(), size, spacing };
    }
}

/** The volume describes; the GI light lights. LightsNode still needs a node for every light. */
class GIVolumeNode extends AnalyticLightNode<GIVolume> {
    static get type(): string {
        return 'GIVolumeNode';
    }

    setup(): Node | undefined {
        return undefined;
    }
}

GIVolume.registerNode(GIVolumeNode);
