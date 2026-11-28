/**
 * Objects move, GI follows. Two rooms joined by a doorway: a lamp lights the
 * near room, the far room has no light of its own. A door swings open and
 * shut on its hinge, and a box slides across the near room.
 *
 * Nothing is called: change the scene as you always would, and the adapter
 * finds the moved objects on the next render. The probes near what moved
 * are traced again. With the door shut, no light reaches the far room, so it
 * goes dark; open, light spills through the doorway.
 *
 * `volume.threeGI.markChanged(object)` makes a change count at once, for
 * edits three's render doesn't see (docs/api/three.md, "Moving, adding and
 * removing objects"). Nothing here needs it.
 */

import * as THREE from 'three/webgpu';
import { DDGIVolume } from '@pmndrs/gi/three';
import { createRenderer } from '../../shared/renderer.ts';
import { loadScene } from '../../shared/scene.ts';
import { showStats } from '../../shared/stats.ts';

const renderer = await createRenderer();
const { scene, camera, probes } = loadScene('closed-door', renderer, { cameraBookmark: 1, exposureEv: -1 });

// The door turns about its hinge: put it in a group at the hinge and turn the group.
const door = scene.getObjectByName('door-panel');
if (!door) throw new Error('The closed-door scene has no door-panel.');
const hinge = new THREE.Group();
hinge.position.set(4.08, 0, 1.546);
scene.add(hinge);
hinge.attach(door);

// A plain three box, added like any other object.
const box = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.5, 0.5), new THREE.MeshStandardMaterial({ color: 0x4f86c6 }));
box.castShadow = true;
box.receiveShadow = true;
scene.add(box);

const volume = DDGIVolume.fromSettings(probes);
scene.add(volume);
await volume.start(renderer, scene);

/** How far the door is open, 0 (shut) to 1 (open): open 4 s, swing shut in 1 s, shut 4 s, swing open in 1 s. */
function doorOpening(seconds: number): number {
    const time = seconds % 10;
    const swing = (share: number) => share * share * (3 - 2 * share); // smoothstep: no jolt at either end
    if (time < 4) return 1;
    if (time < 5) return 1 - swing(time - 4);
    if (time < 9) return 0;
    return swing(time - 9);
}

const timer = new THREE.Timer();
showStats(volume, { note: () => (doorOpening(timer.getElapsed()) === 0 ? 'door shut' : 'door open') });
renderer.setAnimationLoop(() => {
    timer.update();
    const seconds = timer.getElapsed();
    hinge.rotation.y = (-Math.PI / 2) * doorOpening(seconds);
    // Back and forth across the view, in front of the doorway.
    box.position.set(2.6, 0.25, 2 + 1.1 * Math.sin(seconds * 0.7));
    renderer.render(scene, camera);
});
