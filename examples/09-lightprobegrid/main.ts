/**
 * GI exported as three's own `LightProbeGrid` (r187), a static fallback for
 * where DDGI can't run. `volume.exportLightProbeGrid()` projects every probe
 * to L2 spherical harmonics on the GPU and builds the grid from them; its
 * probes sit exactly where ours do. `lightProbeGridToJSON` and
 * `lightProbeGridFromJSON` save and load it (our own format, version 1).
 *
 * The picture is wiped: live GI left of the line, the exported grid right.
 * Each frame draws the scene twice with a scissor, once with the GI light on
 * and the grid off, once the other way round. What the grid loses: detail
 * finer than L2, and visibility (no distances, so light leaks through walls
 * between probes more easily).
 *
 * The grid is a snapshot: re-export after the light changes.
 */

import * as THREE from 'three/webgpu';
import GUI from 'three/addons/libs/lil-gui.module.min.js';
import { DDGIVolume, lightProbeGridToJSON } from '@pmndrs/gi/three';
import { createRenderer } from '../shared/renderer.ts';
import { loadScene } from '../shared/scene.ts';
import { showStats } from '../shared/stats.ts';

const renderer = await createRenderer();
const { scene, camera, probes } = loadScene('cornell-box-point', renderer);

const volume = DDGIVolume.fromDDGIVolume(probes);
scene.add(volume);
await volume.start(renderer, scene);

let grid = await volume.exportLightProbeGrid();
scene.add(grid);

async function exportAgain(): Promise<void> {
    const fresh = await volume.exportLightProbeGrid();
    scene.remove(grid);
    grid.dispose();
    grid = fresh;
    scene.add(grid);
}

function downloadJSON(): void {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([JSON.stringify(lightProbeGridToJSON(grid))], { type: 'application/json' }));
    link.download = 'cornell-box.lightprobegrid.json';
    link.click();
    URL.revokeObjectURL(link.href);
}

const gui = new GUI({ title: 'LightProbeGrid' });
gui.add({ exportAgain }, 'exportAgain').name('export again');
gui.add({ downloadJSON }, 'downloadJSON').name('download the JSON');

// The wipe follows the mouse; it starts in the middle.
let split = 0.5;
renderer.domElement.addEventListener('pointermove', (event) => {
    split = event.clientX / window.innerWidth;
});

// The line between the two halves.
const line = document.createElement('div');
line.style.cssText = 'position:fixed;top:0;bottom:0;width:2px;margin-left:-1px;background:rgba(255,255,255,0.7);pointer-events:none;';
document.body.append(line);

const size = new THREE.Vector2();
renderer.autoClear = false;
showStats(volume, { note: () => 'left: live GI · right: LightProbeGrid' });
renderer.setAnimationLoop(() => {
    renderer.getSize(size);
    const splitX = Math.round(size.x * split);
    line.style.left = `${splitX}px`;
    renderer.clear();
    renderer.setScissorTest(true);

    // Left: live GI. The GI light outranks the grid while it's on.
    volume.light.intensity = 1;
    grid.intensity = 0;
    renderer.setScissor(0, 0, splitX, size.y);
    renderer.render(scene, camera);

    // Right: the exported grid alone.
    volume.light.intensity = 0;
    grid.intensity = 1;
    renderer.setScissor(splitX, 0, size.x - splitX, size.y);
    renderer.render(scene, camera);

    renderer.setScissorTest(false);
});
