/**
 * The `gi/texture-store` kernel: makes one 256 × 256 layer of the texture
 * store, with its full mip chain, from a source on the GPU
 * (texture-sources.ts). Three entry points:
 *
 * - `giTextureStoreFillEncoded` / `giTextureStoreFillLinear`: level 0 from the
 *   source. Each store texel averages the source texels under it (a box,
 *   when the source is larger) or blends the nearest four (bilinear, when it
 *   is smaller; small and non-square textures are stretched to 256²), in
 *   linear light: the "encoded" version decodes sRGB before averaging.
 * - `giTextureStoreMip`: level k from level k - 1, a 2 × 2 average in linear light.
 *
 * Every average weights colour by alpha (T4.2; Alpha-weighted averages
 * below): a clear texel's colour doesn't bleed into the coarser texels it
 * shares with opaque ones, so the coarse mips of cutout foliage keep the
 * leaves' colour, not the colour painted under the holes. Alpha itself is
 * the plain average. A fully opaque texture gives exactly the plain average.
 *
 * Each level is written twice: in linear light as floats into a scratch
 * buffer (`giBuildLinear`), which the next level averages without the 8-bit
 * rounding of the stored texels, and as sRGB-encoded rgba8 texels into a
 * byte buffer (`giBuildBytes`), rows padded to 256 bytes, which the store
 * copies into its layer with `copyBufferToTexture`.
 *
 * *Why buffers and copies, not storage writes into the layer:* the store's
 * texture is `rgba8unorm-srgb`, which can't be a storage texture, and an
 * `rgba8unorm` texture with an sRGB view would need a view per level and
 * layer bound as storage and as a sampled texture side by side, which
 * compatibility-mode devices refuse (they bind a texture only whole, with the
 * view dimension it was made with). Copies have neither problem.
 *
 * Which level a dispatch works on is a `GiTextureBuildStep` picked by a
 * dynamic offset into one uniform buffer, written once: nothing about a
 * texture changes the kernel or its uniforms.
 */

/** The store's layers are this many texels on a side. */
export const STORE_LAYER_SIZE = 256;

/** Levels of a layer's mip chain: 256, 128, ..., 1. */
export const STORE_MIP_LEVELS = Math.log2(STORE_LAYER_SIZE) + 1;

/** Layers one `gi/texture-store` pass makes before its byte buffer is copied out: the byte buffer holds this many layers. */
export const BUILD_LAYERS_PER_PASS = 8;

/** `copyBufferToTexture` needs rows padded to 256 bytes. */
const COPY_ROW_ALIGNMENT = 256;

/** Bytes between `GiTextureBuildStep`s in the uniform buffer: WebGPU's default `minUniformBufferOffsetAlignment`. */
export const BUILD_STEP_STRIDE = 256;

/** Bytes of one `GiTextureBuildStep` (five u32s, rounded up to its 16-byte alignment): the size of its uniform binding. */
export const BUILD_STEP_BYTES = 32;

/** Threads per workgroup side; dispatches cover a level in 8 × 8 tiles. */
export const BUILD_WORKGROUP_SIZE = 8;

/** Width (and height) of mip `level`. */
export function levelSize(level: number): number {
    return STORE_LAYER_SIZE >> level;
}

/** Bytes per row of mip `level` in the byte buffer: four per texel, padded for `copyBufferToTexture`. */
export function levelRowBytes(level: number): number {
    return Math.ceil((levelSize(level) * 4) / COPY_ROW_ALIGNMENT) * COPY_ROW_ALIGNMENT;
}

function sumOverLevels(below: number, value: (level: number) => number): number {
    let sum = 0;
    for (let level = 0; level < below; level++) sum += value(level);
    return sum;
}

/** First texel of mip `level` in the linear scratch (levels one after another, rows unpadded). */
export function linearLevelTexel(level: number): number {
    return sumOverLevels(level, (above) => levelSize(above) * levelSize(above));
}

/** Texels of the linear scratch: every level of one layer. */
export const LINEAR_SCRATCH_TEXELS = linearLevelTexel(STORE_MIP_LEVELS);

/** Byte offset of mip `level` inside one layer's part of the byte buffer. */
export function levelByteOffset(level: number): number {
    return sumOverLevels(level, (above) => levelRowBytes(above) * levelSize(above));
}

/** Bytes of one layer's part of the byte buffer (every level, rows padded). */
export const LAYER_BUILD_BYTES = levelByteOffset(STORE_MIP_LEVELS);

/**
 * The uniform buffer's contents: one `GiTextureBuildStep` per (place in the
 * pass, level), `BUILD_STEP_STRIDE` bytes apart. Step `place × levels + level`.
 */
export function buildStepWords(): Uint32Array<ArrayBuffer> {
    const strideWords = BUILD_STEP_STRIDE / 4;
    const words = new Uint32Array(BUILD_LAYERS_PER_PASS * STORE_MIP_LEVELS * strideWords);
    for (let place = 0; place < BUILD_LAYERS_PER_PASS; place++) {
        for (let level = 0; level < STORE_MIP_LEVELS; level++) {
            const base = (place * STORE_MIP_LEVELS + level) * strideWords;
            words[base + 0] = levelSize(level);
            words[base + 1] = level > 0 ? linearLevelTexel(level - 1) : 0;
            words[base + 2] = linearLevelTexel(level);
            words[base + 3] = (place * LAYER_BUILD_BYTES + levelByteOffset(level)) / 4;
            words[base + 4] = levelRowBytes(level) / 4;
        }
    }
    return words;
}

/** The dynamic offset of step (place, level) in the uniform buffer. */
export function buildStepOffset(place: number, level: number): number {
    return (place * STORE_MIP_LEVELS + level) * BUILD_STEP_STRIDE;
}

/** Bytes of the uniform buffer of every step. */
export const BUILD_STEP_UNIFORM_BYTES = BUILD_LAYERS_PER_PASS * STORE_MIP_LEVELS * BUILD_STEP_STRIDE;

export const FILL_ENCODED_ENTRY_POINT = 'giTextureStoreFillEncoded';
export const FILL_LINEAR_ENTRY_POINT = 'giTextureStoreFillLinear';
export const MIP_ENTRY_POINT = 'giTextureStoreMip';

export const TEXTURE_STORE_WGSL = /* wgsl */ `
// ---- gi/texture-store (platform/textures) -----------------------------------------------

// What one dispatch works on: one level of one layer of the pass.
struct GiTextureBuildStep {
    size: u32,          // the level's width and height in texels
    sourceTexel: u32,   // first texel of the level above it in giBuildLinear (mips only)
    targetTexel: u32,   // first texel of this level in giBuildLinear
    byteWord: u32,      // first word of this level in giBuildBytes
    rowWords: u32,      // words between its rows there (rows padded to 256 bytes)
}

// The layer being made, in linear light: every level, one after another.
@group(0) @binding(0) var<storage, read_write> giBuildLinear: array<vec4f>;
// The pass's layers as rgba8 texels, sRGB-encoded (alpha linear), for copyBufferToTexture.
@group(0) @binding(1) var<storage, read_write> giBuildBytes: array<u32>;
@group(0) @binding(2) var<uniform> giBuildStep: GiTextureBuildStep;
// The source (fill entry points only): one level of a 2D texture, read with textureLoad.
@group(1) @binding(0) var giBuildSource: texture_2d<f32>;

const GI_STORE_LAYER_SIZE: u32 = ${STORE_LAYER_SIZE}u;

// The sRGB transfer functions (IEC 61966-2-1), per channel.
fn giSrgbToLinear(encoded: vec3f) -> vec3f {
    return select(encoded / 12.92, pow((encoded + 0.055) / 1.055, vec3f(2.4)), encoded > vec3f(0.04045));
}

fn giLinearToSrgb(linear: vec3f) -> vec3f {
    return select(linear * 12.92, 1.055 * pow(linear, vec3f(1.0 / 2.4)) - 0.055, linear > vec3f(0.0031308));
}

// Stores one texel of the current level: linear for the next level, encoded for the layer.
// Clamped to 0..1 first, so the next level averages what the layer holds (an HDR source's
// values above 1 can't be stored in 8 bits anyway).
fn giBuildWrite(texel: vec2u, value: vec4f) {
    let linear = clamp(value, vec4f(0.0), vec4f(1.0));
    giBuildLinear[giBuildStep.targetTexel + texel.y * giBuildStep.size + texel.x] = linear;
    giBuildBytes[giBuildStep.byteWord + texel.y * giBuildStep.rowWords + texel.x] = pack4x8unorm(vec4f(giLinearToSrgb(linear.rgb), linear.a));
}

// ---- Alpha-weighted averages ----------------------------------------------------
//
// Colour averaged with each texel weighted by its alpha: sum(w a rgb) / sum(w a),
// so texels a ray passes through (alpha 0) add nothing to the colour of a coarser
// texel. Alpha is the plain average sum(w a), as coverage. Where every texel is
// clear the plain colour average stands (nothing opaque to prefer).
//
// The weighted mean is scaled by sum(w) (1 for a full set of taps, as the plain
// sum(w rgb) was). Where every tap is opaque, sum(w a) equals sum(w) exactly
// (multiplying by 1.0 is exact), and the plain sum(w rgb) is used as it is, so
// opaque textures make the same layer as before alpha weighting, to the bit.
struct GiBuildAverage {
    weightedColour: vec3f,  // sum of w a rgb
    alphaWeight: f32,       // sum of w a
    plainColour: vec3f,     // sum of w rgb
    weight: f32,            // sum of w
}

fn giBuildAdd(sum: ptr<function, GiBuildAverage>, value: vec4f, weight: f32) {
    let alpha = clamp(value.a, 0.0, 1.0);
    (*sum).weightedColour += value.rgb * (weight * alpha);
    (*sum).alphaWeight += weight * alpha;
    (*sum).plainColour += value.rgb * weight;
    (*sum).weight += weight;
}

fn giBuildAverageOf(sum: GiBuildAverage) -> vec4f {
    if (sum.alphaWeight == sum.weight) {
        return vec4f(sum.plainColour, sum.alphaWeight);
    }
    if (sum.alphaWeight > 0.0) {
        return vec4f(sum.weightedColour * (sum.weight / sum.alphaWeight), sum.alphaWeight);
    }
    return vec4f(sum.plainColour, 0.0);
}

// The source texels one store texel reads along one axis, and each one's weight.
//
// Downsizing (the source at least as large as the layer on this axis): a box
// over the source texels whose centres fall inside the store texel's
// footprint, every one weighted the same; exact boxes for the usual
// power-of-two sizes (2048 -> 8 texels). Upsizing: the two texels around
// the store texel's centre, weighted bilinearly (clamped at the edges).
struct GiBuildTaps {
    first: i32,
    count: i32,
    firstWeight: f32,   // the box's every weight, or the bilinear weight of 'first'
    bilinear: bool,
}

fn giBuildAxisTaps(texel: u32, sourceSize: u32) -> GiBuildTaps {
    let scale = f32(sourceSize) / f32(GI_STORE_LAYER_SIZE);
    if (sourceSize >= GI_STORE_LAYER_SIZE) {
        // Source texel i's centre is i + 0.5; the footprint is [texel, texel + 1) x scale.
        let first = i32(ceil(f32(texel) * scale - 0.5));
        let end = i32(ceil(f32(texel + 1u) * scale - 0.5));
        let count = max(end - first, 1);
        return GiBuildTaps(first, count, 1.0 / f32(count), false);
    }
    let centre = (f32(texel) + 0.5) * scale - 0.5;
    let first = floor(centre);
    return GiBuildTaps(i32(first), 2, 1.0 - (centre - first), true);
}

fn giBuildTapWeight(taps: GiBuildTaps, tap: i32) -> f32 {
    if (taps.bilinear && tap == 1) {
        return 1.0 - taps.firstWeight;
    }
    return taps.firstWeight;
}

fn giBuildFill(texel: vec2u, decodeSrgb: bool) {
    if (any(texel >= vec2u(GI_STORE_LAYER_SIZE))) {
        return;
    }
    let sourceSize = textureDimensions(giBuildSource);
    let lastTexel = vec2i(sourceSize) - 1;
    let tapsX = giBuildAxisTaps(texel.x, sourceSize.x);
    let tapsY = giBuildAxisTaps(texel.y, sourceSize.y);
    var sum = GiBuildAverage(vec3f(0.0), 0.0, vec3f(0.0), 0.0);
    for (var y = 0; y < tapsY.count; y++) {
        let row = clamp(tapsY.first + y, 0, lastTexel.y);
        let rowWeight = giBuildTapWeight(tapsY, y);
        for (var x = 0; x < tapsX.count; x++) {
            let column = clamp(tapsX.first + x, 0, lastTexel.x);
            var value = textureLoad(giBuildSource, vec2i(column, row), 0);
            // Averages are taken in linear light: an sRGB-encoded source is decoded first (alpha is linear).
            if (decodeSrgb) {
                value = vec4f(giSrgbToLinear(clamp(value.rgb, vec3f(0.0), vec3f(1.0))), value.a);
            }
            giBuildAdd(&sum, value, rowWeight * giBuildTapWeight(tapsX, x));
        }
    }
    giBuildWrite(texel, giBuildAverageOf(sum));
}

@compute @workgroup_size(${BUILD_WORKGROUP_SIZE}, ${BUILD_WORKGROUP_SIZE})
fn ${FILL_ENCODED_ENTRY_POINT}(@builtin(global_invocation_id) id: vec3u) {
    giBuildFill(id.xy, true);
}

@compute @workgroup_size(${BUILD_WORKGROUP_SIZE}, ${BUILD_WORKGROUP_SIZE})
fn ${FILL_LINEAR_ENTRY_POINT}(@builtin(global_invocation_id) id: vec3u) {
    giBuildFill(id.xy, false);
}

// Level k from level k - 1 (twice the size): the alpha-weighted average of the 2 x 2 texels it covers.
@compute @workgroup_size(${BUILD_WORKGROUP_SIZE}, ${BUILD_WORKGROUP_SIZE})
fn ${MIP_ENTRY_POINT}(@builtin(global_invocation_id) id: vec3u) {
    let size = giBuildStep.size;
    if (any(id.xy >= vec2u(size))) {
        return;
    }
    let parentSize = size * 2u;
    let topLeft = giBuildStep.sourceTexel + (id.y * 2u) * parentSize + id.x * 2u;
    var sum = GiBuildAverage(vec3f(0.0), 0.0, vec3f(0.0), 0.0);
    giBuildAdd(&sum, giBuildLinear[topLeft], 0.25);
    giBuildAdd(&sum, giBuildLinear[topLeft + 1u], 0.25);
    giBuildAdd(&sum, giBuildLinear[topLeft + parentSize], 0.25);
    giBuildAdd(&sum, giBuildLinear[topLeft + parentSize + 1u], 0.25);
    giBuildWrite(id.xy, giBuildAverageOf(sum));
}
`;
