/**
 * The scene copy's GPU mesh packing (T2.13) against the CPU packer, on real
 * scenes and a real GPU.
 *
 * Every mesh of each scene is described the way the three adapter describes
 * it (`meshDescriptor`, uv1 included so that path is packed too), created in
 * a `PlatformScene` with GPU packing on, uploaded with one `encode`, and the
 * vertex buffer read back. Each mesh's words are compared with `packMesh`
 * (mesh-packing.ts), the reference: indices, positions, uvs and material slots
 * bit for bit, octahedral normals within one step (1/32767) per component,
 * because the GPU encodes in f32 and the CPU in f64.
 *
 * Also reported, for information: the main-thread time of the `createMesh`
 * calls (checking, the raw upload, the tree builders' copies), the kernel's
 * GPU time (`gi/pack-mesh`, when the device has timestamps), and how many
 * normals came out one step away from the CPU's.
 *
 * Scenes: every procedural manifest by default (in the suite). Sponza and the
 * other downloaded scenes on request (they need `pnpm assets:fetch`):
 *   VITE_DDGI_SCENES=sponza,bistro-exterior pnpm test:gpu --filter scene-gpu-packing
 */

import type { BufferGeometry, Mesh, Object3D } from 'three';
import type { MeshDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import { NONE } from '../../../../packages/ddgi-core/src/platform/scene/layout.ts';
import { packMesh } from '../../../../packages/ddgi-core/src/platform/scene/mesh-packing.ts';
import { PACK_MESH_PASS } from '../../../../packages/ddgi-core/src/platform/scene/gpu-mesh-packing.ts';
import { PlatformScene, type UploadedMesh } from '../../../../packages/ddgi-core/src/platform/scene/platform-scene.ts';
import { GpuTimer } from '../../../../packages/ddgi-core/src/platform/timing/gpu-timer.ts';
import { meshDescriptor } from '../../../../packages/ddgi-three/src/convert.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { assert } from '../../client.ts';
import { runThreePage } from '../three-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;

function isProcedural(manifest: SceneManifest): boolean {
    return manifest.parts.every((part) => part.kind === 'procedural');
}

const manifests = loadAllManifests();
const SCENES =
    typeof env.VITE_DDGI_SCENES === 'string' && env.VITE_DDGI_SCENES.trim() !== ''
        ? env.VITE_DDGI_SCENES.split(',').map((id) => id.trim())
        : manifests.filter(isProcedural).map((manifest) => manifest.id);

/** Every mesh's geometry in the scene, once each (instances share one mesh). */
function sceneGeometries(root: Object3D): BufferGeometry[] {
    const geometries = new Set<BufferGeometry>();
    root.traverse((object) => {
        if ((object as Mesh).isMesh) geometries.add((object as Mesh).geometry);
    });
    return [...geometries];
}

/** The signed snorm16 halves of an octahedral word. */
function halves(word: number): [number, number] {
    const toSigned = (bits: number) => (bits >= 0x8000 ? bits - 0x10000 : bits);
    return [toSigned(word & 0xffff), toSigned(word >>> 16)];
}

function octahedralSteps(a: number, b: number): number {
    const [au, av] = halves(a);
    const [bu, bv] = halves(b);
    return Math.max(Math.abs(au - bu), Math.abs(av - bv));
}

interface MeshComparison {
    /** Words other than normals that differ: must be 0. */
    readonly wrongWords: number;
    readonly firstWrong: string | null;
    readonly normals: number;
    /** Normals one step away from the CPU's. */
    readonly normalsOneStep: number;
    readonly worstNormalSteps: number;
}

function compareMesh(descriptor: MeshDescriptor, uploaded: UploadedMesh, vertices: Uint32Array): MeshComparison {
    const reference = packMesh(descriptor);
    const normalsStart = reference.offsets.normals;
    const normalsEnd = normalsStart === NONE ? NONE : normalsStart + reference.vertexCount;
    let wrongWords = 0;
    let firstWrong: string | null = null;
    let normalsOneStep = 0;
    let worstNormalSteps = 0;
    for (let word = 0; word < reference.words.length; word++) {
        const actual = vertices[uploaded.vertexOffset + word];
        const expected = reference.words[word];
        if (normalsStart !== NONE && word >= normalsStart && word < normalsEnd) {
            const steps = octahedralSteps(actual, expected);
            if (steps > 0) normalsOneStep++;
            worstNormalSteps = Math.max(worstNormalSteps, steps);
        } else if (actual !== expected) {
            wrongWords++;
            firstWrong ??= `word ${word} of ${reference.words.length}: ${actual.toString(16)} against ${expected.toString(16)}`;
        }
    }
    return { wrongWords, firstWrong, normals: normalsStart === NONE ? 0 : reference.vertexCount, normalsOneStep, worstNormalSteps };
}

async function readWords(device: GPUDevice, buffer: GPUBuffer, words: number): Promise<Uint32Array> {
    const readable = device.createBuffer({ label: 'scene-gpu-packing readback', size: words * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readable, 0, words * 4);
    device.queue.submit([encoder.finish()]);
    await readable.mapAsync(GPUMapMode.READ);
    const result = new Uint32Array(readable.getMappedRange().slice(0));
    readable.destroy();
    return result;
}

await runThreePage(async ({ device, log, data }) => {
    const failures: string[] = [];
    const results: Record<string, unknown> = {};
    data.scenes = results;
    for (const id of SCENES) {
        const manifest = manifests.find((candidate) => candidate.id === id);
        assert(manifest, `no manifest '${id}'`);
        const loaded = await loadScene(manifest, () => {});
        const descriptors = sceneGeometries(loaded.content)
            .map((geometry) => meshDescriptor(geometry, { includeUv1: true }))
            .filter((descriptor): descriptor is MeshDescriptor => descriptor !== null);

        const scene = new PlatformScene(device, { label: `scene-gpu-packing ${id}` });
        await scene.prepareGpuPacking();
        const createdAt = performance.now();
        const handles = descriptors.map((descriptor) => scene.createMesh(descriptor));
        const createMeshMs = performance.now() - createdAt;

        const timer = new GpuTimer(device);
        await timer.ready;
        timer.beginFrame(0);
        const encoder = device.createCommandEncoder();
        const upload = scene.encode(encoder, timer);
        timer.resolve(encoder);
        device.queue.submit([encoder.finish()]);
        timer.readback();
        await timer.drain();
        const packGpuMs = timer.timings.get(PACK_MESH_PASS);
        timer.dispose();

        const vertexBuffer = (scene.module.resources()[0] as GPUBufferBinding).buffer;
        const wordsUsed = Math.max(0, ...upload.meshesUploaded.map((mesh) => mesh.vertexOffset + mesh.layout.wordCount));
        const vertices = await readWords(device, vertexBuffer, wordsUsed);

        let meshes = 0;
        let wrongWords = 0;
        let normals = 0;
        let normalsOneStep = 0;
        let worstNormalSteps = 0;
        for (const uploaded of upload.meshesUploaded) {
            const descriptor = descriptors[handles.indexOf(uploaded.handle)];
            const comparison = compareMesh(descriptor, uploaded, vertices);
            meshes++;
            wrongWords += comparison.wrongWords;
            normals += comparison.normals;
            normalsOneStep += comparison.normalsOneStep;
            worstNormalSteps = Math.max(worstNormalSteps, comparison.worstNormalSteps);
            if (comparison.firstWrong) failures.push(`${id}, mesh ${uploaded.slot}: ${comparison.firstWrong} (${comparison.wrongWords} words wrong)`);
        }
        if (meshes !== descriptors.length) failures.push(`${id}: ${meshes} meshes uploaded of ${descriptors.length}`);
        if (worstNormalSteps > 1) failures.push(`${id}: a normal is ${worstNormalSteps} octahedral steps from the CPU packer's`);
        const result = {
            meshes,
            vertexMB: Number((scene.memory.vertexBytesUsed / 1e6).toFixed(2)),
            wrongWords,
            normals,
            normalsOneStep,
            worstNormalSteps,
            createMeshMs: Number(createMeshMs.toFixed(1)),
            packGpuMs: packGpuMs === undefined ? null : Number(packGpuMs.toFixed(3)),
        };
        results[id] = result;
        log(`${id}: ${JSON.stringify(result)}`);
        scene.dispose();
    }
    assert(failures.length === 0, failures.slice(0, 10).join('; '));
});
