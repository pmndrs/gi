/**
 * T3.0 spike, question 1 (docs/design/m3-moving.md §7, route 2): can a mesh
 * that three deforms on the GPU feed our refit without a CPU copy, and
 * without reaching into three's backend?
 *
 * The route under test:
 * 1. We create a `rgba32float` GPUTexture with STORAGE_BINDING on the shared
 *    device and wrap it in three's `ExternalTexture` (format RGBA, type
 *    Float, so three's WGSL declares `texture_storage_2d<rgba32float, write>`
 *    and its bind-group layout reads the real texture's format; the two match).
 * 2. A TSL compute runs three's own `computeSkinning` on a `SkinnedMesh` and
 *    `textureStore`s each deformed vertex at texel (i mod W, i / W).
 * 3. three draws a plain mesh whose `positionNode` loads the same texel by
 *    `vertexIndex`; its picture must match the `SkinnedMesh` drawn normally.
 * 4. A GPU copy (texture to buffer, same queue) gives `GpuMeshRefit` the
 *    positions as a vec4 stream (stride 4 floats); it refits a tinybvh tree
 *    of the bind pose, and the tree must be the deformed mesh's exactly
 *    (layout checker, corners bit for bit against what the compute wrote).
 *
 * Readbacks here are checks only: the refit itself never touches the CPU.
 * Two poses are run, to show the route follows the skeleton frame to frame.
 *
 * Also tried, for the record: the storage-buffer route. A TSL compute writes a
 * three `StorageBufferAttribute`, and the GPUBuffer is taken from
 * `renderer.backend.get(attribute).buffer`. It works, but that is three's
 * backend internals (m3-moving.md §7 rules it out for the adapter).
 */

import {
    Bone,
    CylinderGeometry,
    ExternalTexture,
    FloatType,
    Mesh,
    MeshBasicNodeMaterial,
    NearestFilter,
    PerspectiveCamera,
    RenderTarget,
    RGBAFormat,
    Scene,
    Skeleton,
    SkinnedMesh,
    StorageBufferAttribute,
    type Node,
    type StorageTexture,
    Uint16BufferAttribute,
    Float32BufferAttribute,
    UnsignedByteType,
    Vector3,
    type WebGPURenderer,
} from 'three/webgpu';
import { computeSkinning, Fn, If, instanceIndex, int, ivec2, storage, texture, textureStore, uvec2, vec4, vertexIndex } from 'three/tsl';
import { GpuMeshRefit } from '@pmndrs/bvh';
import { checkGpuLayout } from '@pmndrs/bvh';
import { createTinyBvhBuilder } from '@pmndrs/bvh';
import type { GpuLayoutMeshBvh } from '@pmndrs/bvh';
import { assert } from '../../client.ts';
import { runThreePage } from '../three-shared/harness.ts';

/**
 * three's `computeSkinning(mesh)`: the skinned position of vertex `instanceIndex`.
 * @types/three 0.187 types it as returning void, but r187 returns the position
 * node (Skinning.js). Its `toPosition` parameter doesn't help: the call's
 * result is then unused, TSL leaves the call out, and nothing is written.
 */
const skinnedPosition = computeSkinning as unknown as (mesh: SkinnedMesh) => Node<'vec3'>;

/** Texels per row of the positions texture: rows of 256 rgba32float texels are 4,096 bytes, a multiple of copyTextureToBuffer's 256. */
const POSITIONS_WIDTH = 256;
const TEXEL_BYTES = 16;
const PICTURE_SIZE = 128;
/** Two poses (bone bends in radians, middle and top bone), to show the route follows the skeleton. */
const POSES = [
    { middle: 0.8, top: 0.6 },
    { middle: -0.5, top: 1.1 },
];

/** A 2 m cylinder with three bones along it; each vertex blends the two bones around its height. */
function makeSkinnedCylinder(): SkinnedMesh {
    const geometry = new CylinderGeometry(0.15, 0.15, 2, 16, 40, false);
    const positions = geometry.getAttribute('position');
    const boneSpan = 2 / 3;
    const skinIndices: number[] = [];
    const skinWeights: number[] = [];
    for (let vertex = 0; vertex < positions.count; vertex++) {
        const along = (positions.getY(vertex) + 1) / boneSpan;
        const lower = Math.min(2, Math.floor(along));
        const upper = Math.min(2, lower + 1);
        const blend = upper === lower ? 0 : along - lower;
        skinIndices.push(lower, upper, 0, 0);
        skinWeights.push(1 - blend, blend, 0, 0);
    }
    geometry.setAttribute('skinIndex', new Uint16BufferAttribute(skinIndices, 4));
    geometry.setAttribute('skinWeight', new Float32BufferAttribute(skinWeights, 4));

    const root = new Bone();
    root.position.y = -1;
    const middle = new Bone();
    middle.position.y = boneSpan;
    const top = new Bone();
    top.position.y = boneSpan;
    root.add(middle);
    middle.add(top);
    const mesh = new SkinnedMesh(geometry, new MeshBasicNodeMaterial({ color: 0xffffff }));
    mesh.add(root);
    mesh.bind(new Skeleton([root, middle, top]));
    return mesh;
}

function pose(mesh: SkinnedMesh, bends: (typeof POSES)[number]): void {
    const [, middle, top] = mesh.skeleton.bones;
    middle.rotation.z = bends.middle;
    top.rotation.z = bends.top;
    mesh.updateMatrixWorld(true);
    mesh.skeleton.update();
}

/** The deformed vertices as three's CPU skinning gives them (local space, like computeSkinning's output). */
function cpuSkinned(mesh: SkinnedMesh): Float32Array {
    const positions = mesh.geometry.getAttribute('position');
    const out = new Float32Array(positions.count * 3);
    const vertex = new Vector3();
    for (let index = 0; index < positions.count; index++) {
        mesh.applyBoneTransform(index, vertex.fromBufferAttribute(positions, index));
        out.set([vertex.x, vertex.y, vertex.z], index * 3);
    }
    return out;
}

async function readBuffer(device: GPUDevice, buffer: GPUBuffer, size = buffer.size): Promise<ArrayBuffer> {
    const readback = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.destroy();
    return copy;
}

function storageBuffer(device: GPUDevice, label: string, data: ArrayBufferView<ArrayBuffer>): GPUBuffer {
    const buffer = device.createBuffer({ label, size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    return buffer;
}

/** xyz of a vec4 stream, as a tight xyz array. */
function xyzOf(vec4s: Float32Array, vertexCount: number): Float32Array {
    const out = new Float32Array(vertexCount * 3);
    for (let vertex = 0; vertex < vertexCount; vertex++) out.set(vec4s.subarray(vertex * 4, vertex * 4 + 3), vertex * 3);
    return out;
}

function largestDifference(a: Float32Array, b: Float32Array): number {
    let largest = 0;
    for (let index = 0; index < a.length; index++) largest = Math.max(largest, Math.abs(a[index] - b[index]));
    return largest;
}

/** The mesh drawn white on black from a fixed camera; returns which pixels it covers. */
async function coverage(renderer: WebGPURenderer, scene: Scene, camera: PerspectiveCamera, target: RenderTarget): Promise<Uint8Array> {
    renderer.setRenderTarget(target);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    const pixels = (await renderer.readRenderTargetPixelsAsync(target, 0, 0, PICTURE_SIZE, PICTURE_SIZE)) as Uint8Array;
    const covered = new Uint8Array(PICTURE_SIZE * PICTURE_SIZE);
    for (let pixel = 0; pixel < covered.length; pixel++) covered[pixel] = pixels[pixel * 4] > 127 ? 1 : 0;
    return covered;
}

await runThreePage(async ({ device, renderer, log, data }) => {
    assert(GpuMeshRefit.isSupported(device), 'the refit needs subgroups');
    const mesh = makeSkinnedCylinder();
    const geometry = mesh.geometry;
    const vertexCount = geometry.getAttribute('position').count;
    const rows = Math.ceil(vertexCount / POSITIONS_WIDTH);
    const restPositions = new Float32Array(geometry.getAttribute('position').array as Float32Array);
    const indices = new Uint32Array(geometry.getIndex()!.array);
    data.vertices = vertexCount;
    data.triangles = indices.length / 3;

    // 1. Our texture, wrapped for three.
    const positionsTexture = device.createTexture({
        label: 'spike deformed positions',
        size: [POSITIONS_WIDTH, rows],
        format: 'rgba32float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    const external = new ExternalTexture(positionsTexture);
    external.image = { width: POSITIONS_WIDTH, height: rows } as unknown as typeof external.image;
    external.format = RGBAFormat;
    external.type = FloatType;
    external.magFilter = NearestFilter;
    external.minFilter = NearestFilter;
    external.generateMipmaps = false;

    // 2. three's GPU skinning, stored into our texture.
    const skinIntoTexture = Fn(() => {
        If(instanceIndex.lessThan(vertexCount), () => {
            const skinned = skinnedPosition(mesh);
            // @types/three only lists StorageTexture here; at runtime three takes any texture whose format/type
            // name a storage format, and an ExternalTexture is exactly what this spike asks about.
            const target = external as unknown as StorageTexture;
            textureStore(target, uvec2(instanceIndex.mod(POSITIONS_WIDTH), instanceIndex.div(POSITIONS_WIDTH)), vec4(skinned, 1));
        });
    })().compute(vertexCount);

    // 3. A plain mesh that draws from our texture.
    const fromTexture = new Mesh(geometry, new MeshBasicNodeMaterial({ color: 0xffffff }));
    const texel = ivec2(int(vertexIndex).mod(POSITIONS_WIDTH), int(vertexIndex).div(POSITIONS_WIDTH));
    (fromTexture.material as MeshBasicNodeMaterial).positionNode = texture(external).load(texel).xyz;
    // three can't compute bounds from a texture: culling would use the rest pose, so turn it off for the check.
    fromTexture.frustumCulled = false;
    mesh.frustumCulled = false;
    const skinnedScene = new Scene();
    skinnedScene.add(mesh);
    const textureScene = new Scene();
    textureScene.add(fromTexture);
    const camera = new PerspectiveCamera(50, 1, 0.1, 20);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    const target = new RenderTarget(PICTURE_SIZE, PICTURE_SIZE, { type: UnsignedByteType });

    // 4. The refit, on a tinybvh tree of the bind pose.
    const tinyBvh = await createTinyBvhBuilder();
    const restLayout: GpuLayoutMeshBvh = tinyBvh.buildLayout({ positions: restPositions, indices }).layout;
    const nodes = storageBuffer(device, 'spike nodes', new Uint8Array(restLayout.nodes));
    const triangles = storageBuffer(device, 'spike triangles', new Uint8Array(restLayout.triangles));
    const indexBuffer = storageBuffer(device, 'spike indices', indices);
    const refitPositions = device.createBuffer({ label: 'spike refit positions', size: POSITIONS_WIDTH * rows * TEXEL_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const refit = new GpuMeshRefit(device, { label: 'spike refit', diagnostics: true });
    await refit.prepare();
    refit.setTree(restLayout);

    const results: Record<string, unknown>[] = [];
    for (const [poseIndex, bends] of POSES.entries()) {
        pose(mesh, bends);
        await renderer.computeAsync(skinIntoTexture);

        // GPU to GPU, on the shared queue after three's compute: no CPU in the path.
        const encoder = device.createCommandEncoder({ label: 'spike refit' });
        encoder.copyTextureToBuffer({ texture: positionsTexture }, { buffer: refitPositions, bytesPerRow: POSITIONS_WIDTH * TEXEL_BYTES, rowsPerImage: rows }, [POSITIONS_WIDTH, rows]);
        refit.pollStats();
        const pass = encoder.beginComputePass({ label: 'spike refit' });
        refit.encode(pass, { nodes, triangles }, { positionBuffer: refitPositions, vertexCount, positionStrideFloats: 4, indexBuffer, indexFormat: 'uint32' });
        pass.end();
        refit.copyStats(encoder);
        device.queue.submit([encoder.finish()]);

        // Checks (readbacks for the test only).
        const gpuPositions = xyzOf(new Float32Array(await readBuffer(device, refitPositions)), vertexCount);
        const cpuPositions = cpuSkinned(mesh);
        const skinningDifference = largestDifference(gpuPositions, cpuPositions);
        const movedFromRest = largestDifference(gpuPositions, restPositions);
        const refitNodes = await readBuffer(device, nodes);
        const refitTriangles = await readBuffer(device, triangles);
        const rootFloats = new Float32Array(refitNodes, 0, 16);
        const bounds = new Float32Array([0, 1, 2].map((axis) => Math.min(rootFloats[axis], rootFloats[8 + axis])).concat([0, 1, 2].map((axis) => Math.max(rootFloats[4 + axis], rootFloats[12 + axis]))));
        const treeProblems = checkGpuLayout({ ...restLayout, nodes: refitNodes, triangles: refitTriangles, bounds }, { positions: gpuPositions, indices });

        const skinnedPicture = await coverage(renderer, skinnedScene, camera, target);
        const texturePicture = await coverage(renderer, textureScene, camera, target);
        let coveredPixels = 0;
        let differentPixels = 0;
        for (let pixel = 0; pixel < skinnedPicture.length; pixel++) {
            coveredPixels += skinnedPicture[pixel];
            if (skinnedPicture[pixel] !== texturePicture[pixel]) differentPixels++;
        }

        const result = { pose: poseIndex, skinningDifference, movedFromRest, treeProblems: treeProblems.length, coveredPixels, differentPixels };
        results.push(result);
        log(`pose ${poseIndex}: GPU skinning vs CPU ${skinningDifference.toExponential(2)} m (moved up to ${movedFromRest.toFixed(3)} m); refit tree problems ${treeProblems.length}; pictures differ in ${differentPixels} of ${coveredPixels} covered pixels`);
        assert(movedFromRest > 0.1, `pose ${poseIndex}: the mesh didn't deform (${movedFromRest} m); the compute didn't follow the skeleton`);
        assert(skinningDifference < 1e-4, `pose ${poseIndex}: the texture's positions differ from three's CPU skinning by ${skinningDifference} m`);
        assert(treeProblems.length === 0, `pose ${poseIndex}: the refit tree isn't the deformed mesh's: ${treeProblems.slice(0, 3).join('; ')}`);
        assert(coveredPixels > 200, `pose ${poseIndex}: the skinned mesh covers only ${coveredPixels} pixels`);
        assert(differentPixels <= coveredPixels * 0.01, `pose ${poseIndex}: drawing from the texture differs from three's skinning in ${differentPixels} of ${coveredPixels} pixels`);
    }
    data.textureRoute = results;
    for (let tries = 0; tries < 50 && !refit.latestStats; tries++) {
        await new Promise((resolve) => setTimeout(resolve, 20));
        refit.pollStats();
    }
    data.refitStats = refit.latestStats;

    // For the record: the storage-buffer route, through three's backend internals.
    const attribute = new StorageBufferAttribute(new Float32Array(vertexCount * 4), 4);
    const skinIntoBuffer = Fn(() => {
        If(instanceIndex.lessThan(vertexCount), () => {
            const skinned = skinnedPosition(mesh);
            storage(attribute, 'vec4', vertexCount).element(instanceIndex).assign(vec4(skinned, 1));
        });
    })().compute(vertexCount);
    await renderer.computeAsync(skinIntoBuffer);
    const backend = renderer.backend as unknown as { get(object: object): { buffer?: GPUBuffer } };
    const threeBuffer = backend.get(attribute).buffer;
    let bufferRoute = 'no GPUBuffer found';
    if (threeBuffer) {
        const viaBuffer = xyzOf(new Float32Array(await readBuffer(device, threeBuffer, vertexCount * 16)), vertexCount);
        const difference = largestDifference(viaBuffer, cpuSkinned(mesh));
        // Bind it to the refit as it is (usage ${threeBuffer.usage}): a validation error would show in the runner.
        const encoder = device.createCommandEncoder({ label: 'spike refit from three buffer' });
        const pass = encoder.beginComputePass();
        refit.encode(pass, { nodes, triangles }, { positionBuffer: threeBuffer, vertexCount, positionStrideFloats: 4, indexBuffer, indexFormat: 'uint32' });
        pass.end();
        device.queue.submit([encoder.finish()]);
        bufferRoute = `works through renderer.backend.get(attribute).buffer (usage 0x${threeBuffer.usage.toString(16)}), positions within ${difference.toExponential(2)} m of CPU skinning`;
    }
    data.bufferRoute = bufferRoute;
    log(`storage-buffer route: ${bufferRoute}`);

    refit.dispose();
    for (const buffer of [nodes, triangles, indexBuffer, refitPositions]) buffer.destroy();
    target.dispose();
    positionsTexture.destroy();
});
