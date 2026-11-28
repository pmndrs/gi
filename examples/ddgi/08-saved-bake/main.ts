/**
 * Saved bakes: save what the probes have learned, and start from it later,
 * so a page opens with finished GI and keeps updating from there.
 *
 * - The first visit warms the probes up as usual, then saves them with
 *   `volume.saveBake()` (a `.ddgibake` file's bytes; one GPU readback) and
 *   keeps the bytes in this browser's IndexedDB.
 * - Reload: `volume.start(renderer, scene, { bake })` starts from the saved
 *   probes, with no warm-up.
 *
 * A bake records a fingerprint of the scene it was made for. If the scene
 * changed since, it still loads, as a starting point (`loadedBake.stale`),
 * and every probe updates from it. A bake for another probe layout is
 * refused with a `SavedFieldError`; this page then forgets it and warms up.
 * On a site of your own you'd ship the file next to the scene and `fetch` it.
 */

import GUI from 'three/addons/libs/lil-gui.module.min.js';
import { SavedFieldError } from '@pmndrs/gi/ddgi';
import { DDGIVolume } from '@pmndrs/gi/three';
import { createRenderer } from '../../shared/renderer.ts';
import { loadScene } from '../../shared/scene.ts';
import { showStats } from '../../shared/stats.ts';

// --- Keeping the bake in IndexedDB: one database, one store, one key. ---

const DATABASE = 'ddgi-examples';
const STORE = 'bakes';
const KEY = '08-saved-bake/two-room';

function openDatabase(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DATABASE, 1);
        request.onupgradeneeded = () => request.result.createObjectStore(STORE);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

async function inStore<T>(mode: IDBTransactionMode, action: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
    const database = await openDatabase();
    try {
        return await new Promise<T>((resolve, reject) => {
            const request = action(database.transaction(STORE, mode).objectStore(STORE));
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    } finally {
        database.close();
    }
}

async function readBake(): Promise<Uint8Array | null> {
    const value = await inStore<unknown>('readonly', (store) => store.get(KEY));
    return value instanceof ArrayBuffer ? new Uint8Array(value) : null;
}

async function writeBake(bytes: Uint8Array<ArrayBuffer>): Promise<void> {
    await inStore('readwrite', (store) => store.put(bytes.slice().buffer, KEY));
}

async function forgetBake(): Promise<void> {
    await inStore('readwrite', (store) => store.delete(KEY));
}

const stateLine = document.getElementById('bake-state') as HTMLElement;
const renderer = await createRenderer();
const { scene, camera, probes } = loadScene('two-room', renderer, { cameraBookmark: 1, exposureEv: -1 });

let volume = DDGIVolume.fromSettings(probes);
scene.add(volume);
const saved = await readBake();
const startedAt = performance.now();
try {
    await volume.start(renderer, scene, saved ? { bake: saved } : {});
} catch (error) {
    if (!(error instanceof SavedFieldError)) throw error;
    // A bake from an older layout (another grid or tile size): forget it and warm up instead.
    await forgetBake();
    volume.dispose();
    scene.remove(volume);
    volume = DDGIVolume.fromSettings(probes);
    scene.add(volume);
    await volume.start(renderer, scene);
}
const seconds = ((performance.now() - startedAt) / 1000).toFixed(1);

const loaded = volume.loadedBake;
if (loaded) {
    const when = new Date(loaded.savedAt).toLocaleString();
    stateLine.textContent = loaded.stale
        ? `Started in ${seconds} s from the bake saved ${when}. The scene changed since (${loaded.changedParts.join(', ')}), so every probe is updating from it.`
        : `Started in ${seconds} s from the bake saved ${when}, with no warm-up.`;
} else {
    const bytes = await volume.saveBake();
    await writeBake(bytes);
    stateLine.textContent = `Warmed up in ${seconds} s and saved a bake (${(bytes.byteLength / 2 ** 20).toFixed(1)} MB). Reload to start from it.`;
}

const gui = new GUI({ title: 'Saved bake' });
gui.add({ reload: () => location.reload() }, 'reload').name('reload the page');
gui.add({ forget: () => forgetBake().then(() => location.reload()) }, 'forget').name('forget the bake and reload');
gui.add({ download: () => downloadBake() }, 'download').name('download the .ddgibake');

showStats(volume, { note: () => (loaded ? 'from the saved bake' : 'warmed up') });
renderer.setAnimationLoop(() => renderer.render(scene, camera));

async function downloadBake(): Promise<void> {
    const bytes = await volume.saveBake();
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([bytes], { type: 'application/octet-stream' }));
    link.download = 'two-room.ddgibake';
    link.click();
    URL.revokeObjectURL(link.href);
}
