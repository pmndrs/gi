/**
 * Skinned meshes posed on the GPU by three's own skinning (T3.3; m3-moving.md
 * §7, route 2, shown by T3.0's spike).
 *
 * A TSL compute runs three's `computeSkinning` on the `SkinnedMesh` and
 * stores each vertex's skinned position into an `rgba32float` storage
 * texture we own, wrapped as a three `ExternalTexture`: vertex `i` at texel
 * `(i mod width, floor(i / width))`. The texture goes to the platform as a
 * `DeformedPositions` texture: the platform copies it into its scene copy on
 * the GPU and refits the mesh's tree from it. No position reaches the CPU, and
 * nothing reaches into three's backend (NOTES: "A TSL compute can write a
 * storage `ExternalTexture` we own").
 *
 * **When it runs.** `nextPose` compares the pose's inputs (the bind matrices
 * and every bone's world matrix) with the last ones, as the CPU pose does;
 * when they changed, it runs the compute with `renderer.compute`, which three
 * submits at once. The platform's copy comes later on the same queue (the
 * host's frame is submitted after the GI update), so it reads this pose.
 * Before the compute it calls `skeleton.update()` itself: three's skinning
 * refreshes the bone matrices once per animation frame, which can leave them
 * a pose behind (NOTES: "`computeSkinning` can skin the previous pose").
 *
 * Skinned meshes with position morph targets stay on the CPU pose:
 * `computeSkinning` skins the rest positions only.
 */

import type { SkinnedMesh } from 'three';
import { ExternalTexture, FloatType, NearestFilter, RGBAFormat, type ComputeNode, type Node, type StorageTexture, type WebGPURenderer } from 'three/webgpu';
import { computeSkinning, Fn, If, instanceIndex, textureStore, uvec2, vec4 } from 'three/tsl';
import type { DeformedPositions } from '@pmndrs/ddgi-core';
import type { PoseSource } from './deforming.ts';

/**
 * three's `computeSkinning(mesh)`: the skinned position of vertex
 * `instanceIndex`. @types/three 0.187 types it as returning void, but r187
 * returns the position node; its `toPosition` argument doesn't work instead
 * (NOTES).
 */
const skinnedPosition = computeSkinning as unknown as (mesh: SkinnedMesh) => Node<'vec3'>;

/**
 * Texels per row of the positions texture: 256 rgba32float texels are 4,096
 * bytes, a whole number of the 256-byte rows the platform's copy needs.
 */
const POSITIONS_WIDTH = 256;

/** WebGPU texture usage bits (numbers, so this file also loads where the GPU globals don't exist). */
const TEXTURE_USAGE = { COPY_SRC: 0x01, TEXTURE_BINDING: 0x04, STORAGE_BINDING: 0x08 } as const;

/** What a GPU pose source needs from the host: the device three renders with, and the renderer to run the compute. */
export interface GpuPoseContext {
    readonly device: GPUDevice;
    readonly renderer: WebGPURenderer;
}

export class GpuSkinningPose implements PoseSource {
    private readonly mesh: SkinnedMesh;
    private readonly context: GpuPoseContext;
    private texture: GPUTexture | null = null;
    private compute: ComputeNode | null = null;
    private positions: DeformedPositions | null = null;
    /** The pose's inputs when last read: the bind matrices, then every bone's world matrix. */
    private key = new Float64Array(0);
    /** readKey's place in `key`, and whether a value it read differed (fields, so no closure is made each update). */
    private keyAt = 0;
    private keyChanged = false;

    constructor(mesh: SkinnedMesh, context: GpuPoseContext) {
        this.mesh = mesh;
        this.context = context;
    }

    nextPose(): DeformedPositions | null {
        if (!this.readKey()) return null;
        if (!this.compute) this.build();
        // `computeSkinning` refreshes `skeleton.boneMatrices` (each bone's world matrix times its inverse bind
        // matrix) only once per three frame id, and the frame id advances only in three's own
        // requestAnimationFrame loop, not per render or compute. A render in the same frame id as this compute has
        // already marked the skeleton updated, with the bones as they were then, so the compute would skin the
        // previous pose. The bones' world matrices are the ones the key just read: flatten them now.
        this.mesh.skeleton.update();
        this.context.renderer.compute(this.compute!);
        return this.positions;
    }

    reset(): void {
        // A new shape may have another vertex count: the texture and the compute are made again at the next pose.
        this.dispose();
        this.key = new Float64Array(0);
    }

    dispose(): void {
        this.compute?.dispose();
        this.texture?.destroy();
        this.compute = null;
        this.texture = null;
        this.positions = null;
    }

    /** The texture, its three wrapper, and the compute that stores three's skinned positions into it. */
    private build(): void {
        const mesh = this.mesh;
        const vertexCount = mesh.geometry.getAttribute('position').count;
        const rows = Math.max(1, Math.ceil(vertexCount / POSITIONS_WIDTH));
        const texture = this.context.device.createTexture({
            label: `ddgi skinned positions ${mesh.name || mesh.uuid}`,
            size: [POSITIONS_WIDTH, rows],
            format: 'rgba32float',
            usage: TEXTURE_USAGE.STORAGE_BINDING | TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_SRC,
        });
        // three names the storage format from `format` and `type`, and takes the bind-group layout from the real
        // texture: the two must agree (NOTES).
        const external = new ExternalTexture(texture as never);
        (external as unknown as { image: unknown }).image = { width: POSITIONS_WIDTH, height: rows };
        external.format = RGBAFormat;
        external.type = FloatType;
        external.magFilter = NearestFilter;
        external.minFilter = NearestFilter;
        external.generateMipmaps = false;
        // @types/three lists only StorageTexture for textureStore; three takes any texture that names a storage format.
        const target = external as unknown as StorageTexture;
        this.compute = Fn(() => {
            If(instanceIndex.lessThan(vertexCount), () => {
                textureStore(target, uvec2(instanceIndex.mod(POSITIONS_WIDTH), instanceIndex.div(POSITIONS_WIDTH)), vec4(skinnedPosition(mesh), 1));
            });
        })().compute(vertexCount);
        this.texture = texture;
        this.positions = { texture };
    }

    /** Reads the pose's inputs into the key; returns whether any changed. */
    private readKey(): boolean {
        const mesh = this.mesh;
        const bones = mesh.skeleton.bones;
        const length = 32 + 16 * bones.length;
        this.keyChanged = false;
        this.keyAt = 0;
        if (this.key.length !== length) {
            this.key = new Float64Array(length);
            this.keyChanged = true;
        }
        this.writeKey(mesh.bindMatrix.elements);
        this.writeKey(mesh.bindMatrixInverse.elements);
        for (const bone of bones) this.writeKey(bone.matrixWorld.elements);
        return this.keyChanged;
    }

    /** A matrix's 16 numbers into the key, noting any that differ from last time. */
    private writeKey(elements: ArrayLike<number>): void {
        for (let index = 0; index < 16; index++, this.keyAt++) {
            if (this.key[this.keyAt] !== elements[index]) {
                this.key[this.keyAt] = elements[index];
                this.keyChanged = true;
            }
        }
    }
}
