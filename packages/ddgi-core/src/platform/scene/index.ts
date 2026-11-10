// The platform's scene copy (T1.2): SceneApi for hosts, SceneCopy for techniques.
// Internal: platform/index.ts does not re-export this, so nothing here reaches
// `@pmndrs/gi/core`. The platform object and the tracer (T1.3) import it by path.
// Layout constants stay in layout.ts: techniques read the scene through the WGSL
// accessors only.
export { PlatformScene, DEFAULT_SCENE_CAPACITIES } from './platform-scene.ts';
export type {
    InstanceSnapshot,
    PackedGiInstances,
    PlatformSceneOptions,
    SceneCapacities,
    SceneMemory,
    SceneUpload,
    UploadedMesh,
} from './platform-scene.ts';
export { SCENE_MODULE_ENTRIES, sceneModuleWgsl } from './wgsl.ts';
export { packMesh, packMeshWords, packedIndices, packedPositions, packedTriangleVertices, planMesh } from './mesh-packing.ts';
export type { MeshLayout, MeshPlan, PackedMesh, PackedMeshOffsets } from './mesh-packing.ts';
export { decodeOctahedral, encodeOctahedral } from './octahedral.ts';
