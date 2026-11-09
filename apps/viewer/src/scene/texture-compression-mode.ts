/**
 * The viewer's texture compression setting (scene/compress-textures.ts does the work).
 * No engine imports, so the link parser (ui/url-state.ts) can use it in unit tests.
 *
 * - `uncompressed`: as three loads them (RGBA8 with mipmaps).
 * - `compressed`: BC7 and BC5, except the maps GI reads colours from on the CPU
 *   until M4 (`GI_AVERAGED_SLOTS` in compress-textures.ts).
 * - `all-compressed`: every texture, those maps included. GI then takes the
 *   materials' plain colours until M4.
 */
export type TextureCompression = 'uncompressed' | 'compressed' | 'all-compressed';
export const TEXTURE_COMPRESSIONS: readonly TextureCompression[] = ['uncompressed', 'compressed', 'all-compressed'];
