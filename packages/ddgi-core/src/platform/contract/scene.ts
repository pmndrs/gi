/**
 * How a host describes its scene to the platform.
 *
 * The host owns its scene and builds it however it likes (code, procedural,
 * loaded files, a live editor). The adapter mirrors it into the platform with
 * these calls and then sends only what changed. The platform keeps its own,
 * deliberately simplified GPU copy: geometry, instances, a four-field material
 * model, lights and the environment.
 *
 * Nothing here recompiles shaders. Lights, meshes, materials and instances live
 * in fixed-capacity GPU tables that grow by reallocation, never by changing
 * pipeline code.
 */

import type { InstanceHandle, LightHandle, MaterialHandle, MeshHandle, TextureHandle } from './handles';
import type { Aabb, ColorRGB, Mat4, Vec3 } from './math';

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/** A run of triangles in a mesh that uses one material slot. */
export interface MeshGroup {
    /** First index (not triangle) of the run. */
    readonly start: number;
    /** Number of indices in the run; a multiple of 3. */
    readonly count: number;
    /** Which entry of the instance's `materials` array the run uses. */
    readonly materialSlot: number;
}

export interface MeshDescriptor {
    /** xyz per vertex, in the mesh's local space. */
    readonly positions: Float32Array;
    /** Triangle list. Non-indexed meshes are given `0..n-1` by the adapter. */
    readonly indices: Uint32Array | Uint16Array;
    /**
     * xyz unit normal per vertex, in the mesh's local space. When present, ray
     * hits shade with the interpolated normal, so curved low-poly surfaces
     * (columns, spheres) don't light in facets. When absent, the triangle's own
     * normal is used. Inside/outside tests always use the triangle normal.
     * Costs 4 bytes per vertex on the GPU: stored octahedral-encoded (two
     * 16-bit values), within about 1e-4 radians of the given direction.
     * Normal maps are deliberately not supported: their detail is far below
     * probe resolution.
     */
    readonly normals?: Float32Array;
    /** uv per vertex (UV channel 0). Needed for textured materials and alpha cutouts. */
    readonly uvs?: Float32Array;
    /** A second uv set (UV channel 1), for textures whose {@link UvMapping.channel} is 1. */
    readonly uvs1?: Float32Array;
    /**
     * Material runs. Omit for a single run covering every triangle using
     * material slot 0.
     */
    readonly groups?: readonly MeshGroup[];
    /**
     * The vertex positions will change every frame (foliage, cloth, a fan)
     * while the triangle list stays the same. The platform then keeps the BVH
     * shape and refits its boxes on the GPU from positions supplied through
     * {@link SceneApi.setDeformedPositions}.
     */
    readonly deforming?: boolean;
    /** Shown in debug output and errors. */
    readonly label?: string;
}

/**
 * Deformed positions supplied by the host each frame, already on the GPU.
 *
 * The buffer belongs to the host and must come from the same device, with
 * `STORAGE` usage. The platform reads it inside {@link GIPlatform.encode} and
 * never writes or destroys it.
 */
export interface DeformedPositions {
    readonly buffer: GPUBuffer;
    /** Byte offset of the first vertex. Must be a multiple of 4. */
    readonly offset?: number;
    /** Bytes between consecutive vertices. Defaults to 12 (tightly packed xyz). */
    readonly stride?: number;
}

// ---------------------------------------------------------------------------
// Textures and materials
// ---------------------------------------------------------------------------

/** Raw pixels in linear or sRGB 8-bit RGBA. */
export interface PixelData {
    readonly data: Uint8Array | Uint8ClampedArray;
    readonly width: number;
    readonly height: number;
}

/**
 * Anything the platform can read an image from. It always makes its own
 * downsized copy in its atlas, so the host can free the source afterwards.
 *
 * A same-device `GPUTexture` covers images the host only has on the GPU (for
 * example compressed KTX2 textures after upload).
 */
export type TextureSource =
    | GPUTexture
    | ImageBitmap
    | HTMLImageElement
    | HTMLCanvasElement
    | OffscreenCanvas
    | ImageData
    | PixelData;

export interface TextureDescriptor {
    readonly source: TextureSource;
    /** The stored values are sRGB-encoded (true for typical albedo images). */
    readonly srgb: boolean;
    readonly label?: string;
}

/** How a texture wraps outside 0..1, per axis. The same three choices as WebGPU's address modes. */
export type UvWrap = 'repeat' | 'clamp' | 'mirror';

/**
 * Where a material's texture is sampled on the mesh. Engines tile, offset and
 * rotate textures, and may read a second uv set. Without this, the GI copy of
 * a tiled floor would sample the wrong texels.
 *
 * Every field is optional; omitted means channel 0, no transform, repeat.
 */
export interface UvMapping {
    /** Which uv set: 0 = `MeshDescriptor.uvs`, 1 = `MeshDescriptor.uvs1`. */
    readonly channel?: 0 | 1;
    /**
     * A 2D affine transform applied to the uv before sampling:
     * `uv' = (m[0]·u + m[2]·v + m[4], m[1]·u + m[3]·v + m[5])`.
     * That's a 3×2 matrix, column-major. From three.js, pass
     * `[e[0], e[1], e[3], e[4], e[6], e[7]]` of `texture.matrix.elements`,
     * after `texture.updateMatrix()`.
     */
    readonly matrix?: readonly [number, number, number, number, number, number];
    readonly wrapU?: UvWrap;
    readonly wrapV?: UvWrap;
}

/**
 * The platform's material model. Each adapter maps its engine's materials onto
 * these four ideas; nothing else about a material affects GI.
 */
export interface MaterialDescriptor {
    /** Diffuse albedo, multiplied with `albedoTexture` when both are set. */
    readonly albedo?: ColorRGB;
    readonly albedoTexture?: TextureHandle;
    /** Where `albedoTexture` is sampled. Its alpha (for `alphaCutoff`) uses the same mapping. */
    readonly albedoUv?: UvMapping;
    /**
     * Emitted radiance, multiplied with `emissiveTexture` and
     * `emissiveIntensity`. In the platform's light units (see {@link LightDescriptor}).
     */
    readonly emissive?: ColorRGB;
    readonly emissiveTexture?: TextureHandle;
    /** Where `emissiveTexture` is sampled. */
    readonly emissiveUv?: UvMapping;
    readonly emissiveIntensity?: number;
    /**
     * Alpha below this value is a hole for rays. The alpha comes from
     * `albedoTexture`'s alpha channel. Omit for opaque.
     */
    readonly alphaCutoff?: number;
    /**
     * Whether rays stop at this surface. False for glass and similar: light
     * passes through, and the surface neither shadows nor bounces.
     * Defaults to true.
     */
    readonly blocksLight?: boolean;
    /**
     * Rays that hit the back side count as hitting a surface. Defaults to true.
     * Back-face hits are also how probes find out they are inside geometry, so
     * single-sided walls should still usually be double-sided here.
     */
    readonly doubleSided?: boolean;
    readonly label?: string;
}

// ---------------------------------------------------------------------------
// Instances
// ---------------------------------------------------------------------------

export interface InstanceDescriptor {
    readonly mesh: MeshHandle;
    /** One material per material slot used by the mesh's groups. */
    readonly materials: readonly MaterialHandle[];
    /** Local-to-world transform. */
    readonly transform: Mat4;
    /**
     * Whether this object blocks and bounces light for GI. Off for characters
     * and small props, which still *receive* GI through the shading module but
     * are left out of GI rays (see SPEC "What moving means"). Defaults to true.
     */
    readonly contributesToGI?: boolean;
    /** Hidden instances are left out of rays but keep their data. Defaults to true. */
    readonly visible?: boolean;
    readonly label?: string;
}

// ---------------------------------------------------------------------------
// Lights and environment
// ---------------------------------------------------------------------------

/**
 * Light units: the platform's units match three.js's physically based lights,
 * so the three adapter passes values straight through and other adapters
 * convert once.
 *
 * - Directional: `intensity` is irradiance at normal incidence (a white
 *   diffuse surface facing the light reflects radiance `color * intensity / π`).
 * - Point and spot: `intensity` is radiant intensity; irradiance falls off as
 *   `intensity / distance²`, windowed to zero at `range`. `range: 0` means no
 *   window, as three.js's `distance = 0` does.
 * - Emissive surfaces and the sky: radiance.
 */
interface LightCommon {
    readonly color: ColorRGB;
    readonly intensity: number;
    /** Lights that don't cast shadows are cheaper but leak; default true. */
    readonly castsShadows?: boolean;
    /**
     * Light-group index (0..3). Techniques that support groups (DDGI) can
     * re-weight a whole group instantly. Group assignment is scene data, so
     * it lives here; the weights belong to the technique. Defaults to 0.
     */
    readonly group?: number;
    readonly label?: string;
}

export interface DirectionalLightDescriptor extends LightCommon {
    readonly type: 'directional';
    /** Direction the light travels, world space; normalized by the platform. */
    readonly direction: Vec3;
    /** Angular radius of the source in radians, for soft shadows. Defaults to 0. */
    readonly angularRadius?: number;
}

export interface PointLightDescriptor extends LightCommon {
    readonly type: 'point';
    readonly position: Vec3;
    /**
     * Distance at which the light reaches zero. Required, so culling is a
     * choice: `0` means no limit (three.js's `distance = 0`). Such a light is
     * never culled and costs every probe.
     */
    readonly range: number;
    /** Radius of the emitting sphere, for soft shadows. Defaults to 0. */
    readonly radius?: number;
}

export interface SpotLightDescriptor extends LightCommon {
    readonly type: 'spot';
    readonly position: Vec3;
    readonly direction: Vec3;
    /** As for point lights: `0` means no limit. */
    readonly range: number;
    /** Half-angle where the falloff starts, radians. */
    readonly innerConeAngle: number;
    /** Half-angle where the light reaches zero, radians. */
    readonly outerConeAngle: number;
    readonly radius?: number;
}

export type LightDescriptor = DirectionalLightDescriptor | PointLightDescriptor | SpotLightDescriptor;

/**
 * Changes to one light, for {@link SceneApi.updateLight}: any fields of the
 * light's own type except `type`.
 *
 * Why a union of per-type partials: `Partial<Omit<LightDescriptor, 'type'>>`
 * keeps only the keys every light type shares (`Omit` over a union), so it
 * couldn't name a point light's `position` or `range` (T1.2).
 */
export type LightChanges =
    | Partial<Omit<DirectionalLightDescriptor, 'type'>>
    | Partial<Omit<PointLightDescriptor, 'type'>>
    | Partial<Omit<SpotLightDescriptor, 'type'>>;

/**
 * The sky, seen by rays that miss all geometry. The platform projects it into
 * a small engine-neutral form by compute: irradiance as spherical harmonics
 * (bands 0, 1, 2 and 4) and a 256 x 128 equirectangular radiance map for rays
 * that miss, so tracing never samples the host's environment format directly.
 * The projection runs in the next `encode` after a `setEnvironment` call, and
 * only then: a texture is read at that moment, so after changing its contents
 * call `setEnvironment` again (the same object is fine; every call re-projects).
 *
 * World space is right-handed with +Y up, as in three.js.
 *
 * **`equirect` layout** (three.js's `equirectUV`, so three's equirect
 * textures can be passed straight in): a world direction d is found at the
 * WebGPU texture coordinate
 *
 *     u = atan2(d.z, d.x) / (2 pi) + 0.5,    v = asin(d.y) / pi + 0.5
 *
 * - v = 0, the texture's first row in memory, is the nadir (-Y); the last row
 *   is the zenith (+Y). (three stores its equirect textures this way: EXR and
 *   HDR loaders write rows bottom-up, and images with `flipY` are flipped on
 *   upload.) An image stored top row first, as image files are, must be
 *   flipped when uploaded.
 * - u = 0 and 1 are -X, u = 0.25 is -Z, u = 0.5 is +X, u = 0.75 is +Z, so u
 *   runs clockwise seen from above (+Y). u wraps at -X.
 *
 * **`cube` layout:** WebGPU's own cube addressing, looked up with the world
 * direction d as it is: layers +X, -X, +Y, -Y, +Z, -Z, each face's (s, t)
 * as in the WebGPU spec's face-selection table (the same as Vulkan's and
 * D3D's). **three.js differs:** r187 looks every cube texture up at
 * (-d.x, d.y, d.z) on WebGPU (`nodes/accessors/CubeTextureNode.js`,
 * `setupUV`), so a three cube texture passed straight in comes out mirrored
 * in x. The three adapter has to account for that (T1.7); an equirect
 * texture has no such difference.
 *
 * **`rotationY`** turns the environment about +Y, counter-clockwise seen from
 * above (right-handed): what the unturned texture shows along d appears along
 * R_y(rotationY) d, as three's `scene.environmentRotation.y` does.
 *
 * Float textures that can't be filtered (`rgba32float` and the like) are read
 * with nearest samples. Formats that can't be sampled as float, and textures
 * of the wrong shape, are refused with a console warning and give a black sky.
 */
export type EnvironmentDescriptor =
    | { readonly type: 'none' }
    /** One radiance in every direction, below the horizon too. */
    | { readonly type: 'color'; readonly color: ColorRGB; readonly intensity?: number }
    | {
          readonly type: 'texture';
          /** Same-device texture with TEXTURE_BINDING usage: a cube map (6 layers), or a 2D equirectangular image (see above). */
          readonly texture: GPUTexture;
          readonly layout: 'cube' | 'equirect';
          /** Multiplies the texture's radiance. Defaults to 1. */
          readonly intensity?: number;
          /** Rotation about +Y, radians (see above). Defaults to 0. */
          readonly rotationY?: number;
      }
    /**
     * The Preetham clear sky (Preetham, Shirley and Smits 1999), made by the
     * same code as the repo's shared sky files: no sun disc (the sun is your
     * directional light), black below the horizon, and calibrated so that flat
     * open ground gets the Ineichen-Perez clear-sky share of the sun's light
     * (DECISIONS 2026-11-04).
     */
    | {
          readonly type: 'procedural';
          /** Direction toward the sun, world space. Below 5 degrees of elevation the sky is made for a sun at 5 degrees (time of day is M2). */
          readonly sunDirection: Vec3;
          /** Atmospheric haze, roughly 2 (clear) to 10 (hazy); clamped to the model's 1.7 to 20. Defaults to 2. */
          readonly turbidity?: number;
          /**
           * The illuminance, at normal incidence, of the sun the sky is
           * calibrated to: for a directional light, its `intensity` times the
           * Rec. 709 luminance of its `color`. The sky is made for a sun of
           * illuminance 1 and scaled by this, the same quantity as
           * pmndrs/sky's `exposure` / `luminanceScale` (which computes its sky
           * for a sun of illuminance 1, Hillaire's ILLUMINANCE_IS_ONE, then
           * scales). Defaults to 1.
           */
          readonly sunIlluminance?: number;
          /**
           * A plain multiplier on top, as for `color` and `texture`. Defaults to 1.
           * The sky's radiance is the calibrated sky x `sunIlluminance` x `intensity`.
           */
          readonly intensity?: number;
      };

// ---------------------------------------------------------------------------
// The scene API
// ---------------------------------------------------------------------------

/**
 * Incremental scene description. Every call is cheap on the CPU: the platform
 * queues the change and uploads it in the next {@link GIPlatform.encode}.
 * Calls made between two frames are merged.
 */
export interface SceneApi {
    createMesh(descriptor: MeshDescriptor): MeshHandle;
    /** Replace the shape. Rebuilds that mesh's BVH in a worker; the old one is used until it's ready. */
    updateMesh(mesh: MeshHandle, descriptor: MeshDescriptor): void;
    /** Supply this frame's positions for a mesh created with `deforming: true`. */
    setDeformedPositions(mesh: MeshHandle, positions: DeformedPositions): void;
    destroyMesh(mesh: MeshHandle): void;

    createTexture(descriptor: TextureDescriptor): TextureHandle;
    destroyTexture(texture: TextureHandle): void;

    createMaterial(descriptor: MaterialDescriptor): MaterialHandle;
    /** Partial updates: only the fields given change. */
    updateMaterial(material: MaterialHandle, changes: Partial<MaterialDescriptor>): void;
    destroyMaterial(material: MaterialHandle): void;

    createInstance(descriptor: InstanceDescriptor): InstanceHandle;
    setTransform(instance: InstanceHandle, transform: Mat4): void;
    /**
     * Many transforms in one call: `matrices` holds 16 numbers per entry of
     * `instances` (column-major, as `setTransform` takes them), in the same
     * order. The same result as calling `setTransform` for each, without the
     * per-call bookkeeping, for hosts that move many objects a frame.
     * Throws before changing anything if an entry is invalid.
     */
    setTransforms(instances: Uint32Array, matrices: Float32Array): void;
    setVisible(instance: InstanceHandle, visible: boolean): void;
    setMaterials(instance: InstanceHandle, materials: readonly MaterialHandle[]): void;
    destroyInstance(instance: InstanceHandle): void;

    createLight(descriptor: LightDescriptor): LightHandle;
    /** Partial updates. Changing `type` is not allowed: destroy and recreate. */
    updateLight(light: LightHandle, changes: LightChanges): void;
    destroyLight(light: LightHandle): void;

    setEnvironment(environment: EnvironmentDescriptor): void;

    /** World-space bounds of everything that contributes to GI. */
    readonly bounds: Aabb;
}
