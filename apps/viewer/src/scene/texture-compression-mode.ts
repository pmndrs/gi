/**
 * The viewer's texture compression setting (scene/compress-textures.ts does the work).
 * No engine imports, so the link parser (ui/url-state.ts) can use it in unit tests.
 *
 * - `uncompressed`: as three loads them (RGBA8 with mipmaps).
 * - `compressed`: BC7 and BC5, except the slots in `GI_AVERAGED_SLOTS`
 *   (compress-textures.ts), empty since M4's T4.3; so the same as `all-compressed`.
 * - `all-compressed`: every texture (the default). GI reads the compressed mips (M4, T4.3).
 */
export type TextureCompression = 'uncompressed' | 'compressed' | 'all-compressed';
export const TEXTURE_COMPRESSIONS: readonly TextureCompression[] = ['uncompressed', 'compressed', 'all-compressed'];
