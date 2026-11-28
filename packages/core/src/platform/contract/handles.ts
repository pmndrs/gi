/**
 * Opaque handles for things the host has described to the platform.
 *
 * Handles are plain numbers at runtime (cheap to store in engine objects and
 * to pass across a worker boundary) but are branded in TypeScript so a mesh
 * handle can't be passed where a material handle is expected.
 *
 * A handle stays valid until the matching `destroy*` call. Using a destroyed
 * handle throws; handle numbers may be reused after destruction.
 */

declare const handleBrand: unique symbol;

/** A number tagged with the kind of object it refers to. */
export type Handle<Kind extends string> = number & { readonly [handleBrand]: Kind };

/** Geometry: positions and indices, optionally UVs, owned by the platform. */
export type MeshHandle = Handle<'mesh'>;

/** A simplified material: albedo, emission, cutout, whether it blocks light. */
export type MaterialHandle = Handle<'material'>;

/** An image the platform downsizes into its own texture atlas. */
export type TextureHandle = Handle<'texture'>;

/** One placement of a mesh in the world, with a transform and materials. */
export type InstanceHandle = Handle<'instance'>;

/** A directional, point or spot light. */
export type LightHandle = Handle<'light'>;
