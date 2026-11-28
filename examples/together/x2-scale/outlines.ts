/**
 * "Show volumes": each DDGI volume's box of probes as an outline, read from
 * the technique's field every frame. A camera-following volume's box moves
 * by whole planes of probes as its window follows the camera
 * (`DDGIFieldVolume.window`, in lattice steps from where its settings put
 * it); a fixed volume's stays put.
 */

import * as THREE from 'three/webgpu';
import type { DDGITechnique } from '@pmndrs/gi/ddgi';

/** The outlines' colours, as the panel names them. */
export const VOLUME_COLOURS = { near: '#ffd166', middle: '#06d6a0', town: '#4cc9f0' } as const;

export class VolumeOutlines {
    readonly group = new THREE.Group();
    private readonly helpers: THREE.Box3Helper[] = [];

    constructor(scene: THREE.Scene) {
        scene.add(this.group);
    }

    update(technique: DDGITechnique | null, visible: boolean): void {
        const volumes = visible ? (technique?.field?.volumes ?? []) : [];
        this.group.visible = volumes.length > 0;
        volumes.forEach((volume, index) => {
            const helper = (this.helpers[index] ??= this.add());
            // Highest priority first: the near window, then the middle one; the fixed town volume last.
            (helper.material as THREE.LineBasicMaterial).color.set(!volume.followsCamera ? VOLUME_COLOURS.town : index === 0 ? VOLUME_COLOURS.near : VOLUME_COLOURS.middle);
            const { origin, spacing, counts } = volume.grid;
            for (let axis = 0; axis < 3; axis++) {
                // The window's first probe, then the last probe on each axis.
                const first = origin[axis] + volume.window[axis] * spacing[axis];
                helper.box.min.setComponent(axis, first);
                helper.box.max.setComponent(axis, first + (counts[axis] - 1) * spacing[axis]);
            }
        });
        this.helpers.forEach((helper, index) => (helper.visible = index < volumes.length));
    }

    private add(): THREE.Box3Helper {
        const helper = new THREE.Box3Helper(new THREE.Box3());
        this.group.add(helper);
        return helper;
    }
}
