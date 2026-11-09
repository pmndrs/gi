import {
    BoxGeometry,
    BufferAttribute,
    BufferGeometry,
    ClampToEdgeWrapping,
    Color,
    CompressedTexture,
    DataTexture,
    DirectionalLight,
    DoubleSide,
    FloatType,
    FrontSide,
    InterleavedBuffer,
    InterleavedBufferAttribute,
    MeshBasicMaterial,
    MeshLambertMaterial,
    MeshPhongMaterial,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
    MeshToonMaterial,
    MirroredRepeatWrapping,
    NoColorSpace,
    PointLight,
    RED_GREEN_RGTC2_Format,
    RGBA_ASTC_4x4_Format,
    RGBA_BPTC_Format,
    RGBA_ETC2_EAC_Format,
    RGBA_S3TC_DXT1_Format,
    RGBA_S3TC_DXT5_Format,
    RGB_BPTC_UNSIGNED_Format,
    RGB_ETC2_Format,
    RawShaderMaterial,
    RenderTarget,
    RepeatWrapping,
    ShaderMaterial,
    ShadowMaterial,
    SpotLight,
    SRGBColorSpace,
    Texture,
    type CompressedPixelFormat,
    type Material,
} from 'three';
import {
    MeshBasicNodeMaterial,
    MeshLambertNodeMaterial,
    MeshPhongNodeMaterial,
    MeshPhysicalNodeMaterial,
    MeshStandardNodeMaterial,
    MeshToonNodeMaterial,
} from 'three/webgpu';
import { color as colorNode } from 'three/tsl';
import { describe, expect, it } from 'vitest';
import type { CompressedPixelData, MaterialDescriptor, PixelData, TextureHandle } from '@pmndrs/ddgi-core';
import {
    CUSTOM_SHADING_ALBEDO,
    LIGHT_NUMBER_COUNT,
    lightDescriptor,
    materialDescriptor,
    MaterialSnapshot,
    meshDescriptor,
    readFloatAttribute,
    readLightNumbers,
    textureDescriptor,
    unreadableTextureReason,
    usesUvChannel1,
    uvMapping,
} from './convert.ts';

const HANDLE = 7 as TextureHandle;
const noTextures = () => undefined;

function expectClose(actual: readonly number[] | undefined, expected: readonly number[], digits = 6): void {
    expect(actual).toBeDefined();
    expect(actual!.length).toBe(expected.length);
    actual!.forEach((value, index) => expect(value).toBeCloseTo(expected[index], digits));
}

/** A 2 × 2 8-bit RGBA texture, as a loaded glTF texture would be oriented (no `flipY`). */
function dataTexture(): DataTexture {
    const texture = new DataTexture(new Uint8Array(2 * 2 * 4).fill(255), 2, 2);
    texture.flipY = false;
    return texture;
}

/** Hands out one handle per texture, so a test can tell which texture went where. */
function handles(): { readonly of: (texture: Texture) => TextureHandle; readonly handleOf: Map<Texture, TextureHandle> } {
    const handleOf = new Map<Texture, TextureHandle>();
    return {
        handleOf,
        of: (texture) => {
            if (!handleOf.has(texture)) handleOf.set(texture, (100 + handleOf.size) as TextureHandle);
            return handleOf.get(texture)!;
        },
    };
}

/** A converted material and the warnings it gave. */
function convert(material: Material): { descriptor: MaterialDescriptor; warnings: string[]; handleOf: Map<Texture, TextureHandle> } {
    const warnings: string[] = [];
    const { of, handleOf } = handles();
    const descriptor = materialDescriptor(material, of, { warn: (message) => warnings.push(message) });
    return { descriptor, warnings, handleOf };
}

// ---------------------------------------------------------------------------
// The material matrix (docs/design/m4-materials.md §4)
// ---------------------------------------------------------------------------

interface MaterialType {
    readonly name: string;
    readonly make: () => Material;
    /** Has `metalness` and `metalnessMap` (Standard, Physical and their node versions). */
    readonly metalness: boolean;
    /** Has `emissive` (every lit material; not Basic). */
    readonly emission: boolean;
    /** Has `transmission` (Physical). */
    readonly transmission: boolean;
    readonly node: boolean;
}

const MATERIAL_TYPES: readonly MaterialType[] = [
    { name: 'MeshStandardMaterial', make: () => new MeshStandardMaterial(), metalness: true, emission: true, transmission: false, node: false },
    { name: 'MeshPhysicalMaterial', make: () => new MeshPhysicalMaterial(), metalness: true, emission: true, transmission: true, node: false },
    { name: 'MeshLambertMaterial', make: () => new MeshLambertMaterial(), metalness: false, emission: true, transmission: false, node: false },
    { name: 'MeshPhongMaterial', make: () => new MeshPhongMaterial(), metalness: false, emission: true, transmission: false, node: false },
    { name: 'MeshToonMaterial', make: () => new MeshToonMaterial(), metalness: false, emission: true, transmission: false, node: false },
    { name: 'MeshBasicMaterial', make: () => new MeshBasicMaterial(), metalness: false, emission: false, transmission: false, node: false },
    { name: 'MeshStandardNodeMaterial', make: () => new MeshStandardNodeMaterial(), metalness: true, emission: true, transmission: false, node: true },
    { name: 'MeshPhysicalNodeMaterial', make: () => new MeshPhysicalNodeMaterial(), metalness: true, emission: true, transmission: true, node: true },
    { name: 'MeshLambertNodeMaterial', make: () => new MeshLambertNodeMaterial(), metalness: false, emission: true, transmission: false, node: true },
    { name: 'MeshPhongNodeMaterial', make: () => new MeshPhongNodeMaterial(), metalness: false, emission: true, transmission: false, node: true },
    { name: 'MeshToonNodeMaterial', make: () => new MeshToonNodeMaterial(), metalness: false, emission: true, transmission: false, node: true },
    { name: 'MeshBasicNodeMaterial', make: () => new MeshBasicNodeMaterial(), metalness: false, emission: false, transmission: false, node: true },
];

type Fields = Material & {
    color: Color;
    map: Texture | null;
    metalness?: number;
    metalnessMap?: Texture | null;
    emissive?: Color;
    emissiveIntensity?: number;
    emissiveMap?: Texture | null;
    transmission?: number;
    colorNode?: unknown;
};

describe.each(MATERIAL_TYPES)('$name', (type) => {
    const make = () => type.make() as Fields;

    it('sends its colour as the albedo, with metalness apart (the platform applies it)', () => {
        const material = make();
        material.color.setRGB(0.8, 0.6, 0.4);
        if (type.metalness) material.metalness = 0.25;
        const { descriptor, warnings } = convert(material);
        // Not multiplied by (1 − metalness): the platform does that, with the metalness map.
        expectClose(descriptor.albedo, [0.8, 0.6, 0.4]);
        expect(descriptor.metalness).toBe(type.metalness ? 0.25 : undefined);
        expect(descriptor).toMatchObject({ alphaMode: 'opaque', blocksLight: true, doubleSided: true });
        expect(descriptor.opacity).toBeUndefined();
        expect(descriptor.alphaCutoff).toBeUndefined();
        expect(warnings).toEqual([]);
    });

    it('sends its map as the albedo texture, with its uv mapping', () => {
        const material = make();
        const map = dataTexture();
        map.repeat.set(2, 3);
        material.map = map;
        const { descriptor, handleOf } = convert(material);
        expect(descriptor.albedoTexture).toBe(handleOf.get(map));
        expectClose(descriptor.albedoUv?.matrix, [2, 0, 0, 3, 0, 0]);
        // The colour still multiplies it, unfolded.
        expectClose(descriptor.albedo, [1, 1, 1]);
    });

    it(type.metalness ? 'sends its metalnessMap as blue-channel metalness with its own mapping' : 'has no metalness to send', () => {
        const material = make();
        const metalnessMap = dataTexture();
        metalnessMap.offset.set(0.5, 0);
        material.metalnessMap = metalnessMap;
        if (type.metalness) material.metalness = 1;
        const { descriptor, handleOf } = convert(material);
        if (type.metalness) {
            expect(descriptor.metalness).toBe(1);
            expect(descriptor.metalnessTexture).toMatchObject({ texture: handleOf.get(metalnessMap), channel: 'b' });
            expectClose(descriptor.metalnessTexture?.uv?.matrix, [1, 0, 0, 1, 0.5, 0]);
            // Albedo unchanged: no texture average is folded into it.
            expectClose(descriptor.albedo, [1, 1, 1]);
        } else {
            expect(descriptor.metalness).toBeUndefined();
            expect(descriptor.metalnessTexture).toBeUndefined();
            expect(handleOf.size).toBe(0);
        }
    });

    it('turns alphaTest into a cutout at alphaTest, alpha = opacity × the map\'s alpha', () => {
        const material = make();
        material.alphaTest = 0.4;
        material.opacity = 0.8;
        const { descriptor } = convert(material);
        expect(descriptor).toMatchObject({ alphaMode: 'cutout', opacity: 0.8, blocksLight: true });
        expect(descriptor.alphaCutoff).toBeCloseTo(0.4);
        // Alpha-tested and transparent is still a cutout: three tests alpha first.
        material.transparent = true;
        expect(convert(material).descriptor.alphaMode).toBe('cutout');
    });

    it('makes a blended (transparent) material stochastic: it blocks light by its alpha', () => {
        const material = make();
        material.transparent = true;
        material.opacity = 0.3;
        const { descriptor } = convert(material);
        expect(descriptor).toMatchObject({ alphaMode: 'stochastic', opacity: 0.3, blocksLight: true });
        expect(descriptor.alphaCutoff).toBeUndefined();
    });

    it(type.transmission ? 'lets light through as glass (transmission > 0)' : 'has no transmission', () => {
        const material = make();
        if (type.transmission) material.transmission = 1;
        expect(convert(material).descriptor.blocksLight).toBe(!type.transmission);
    });

    it(type.emission ? 'sends emissive × emissiveIntensity × emissiveMap' : 'has no emission (Basic)', () => {
        const material = make();
        const emissiveMap = dataTexture();
        if (type.emission) {
            material.emissive!.setRGB(1, 0.5, 0.25);
            material.emissiveIntensity = 12;
            material.emissiveMap = emissiveMap;
        }
        const { descriptor, handleOf } = convert(material);
        if (type.emission) {
            expectClose(descriptor.emissive, [1, 0.5, 0.25]);
            expect(descriptor.emissiveIntensity).toBe(12);
            expect(descriptor.emissiveTexture).toBe(handleOf.get(emissiveMap));
            expectClose(descriptor.emissiveUv?.matrix, [1, 0, 0, 1, 0, 0]);
        } else {
            expect(descriptor.emissive).toBeUndefined();
            expect(descriptor.emissiveIntensity).toBeUndefined();
        }
    });

    it('takes userData.giAlbedo, giAlbedoMap and giEmissive over its own', () => {
        const material = make();
        material.color.setRGB(1, 0, 0);
        material.map = dataTexture();
        const override = dataTexture();
        material.userData.giAlbedo = [0.2, 0.3, 0.4];
        material.userData.giAlbedoMap = override;
        material.userData.giEmissive = new Color().setRGB(2, 1, 0);
        const { descriptor, handleOf } = convert(material);
        expectClose(descriptor.albedo, [0.2, 0.3, 0.4]);
        expect(descriptor.albedoTexture).toBe(handleOf.get(override));
        expect(handleOf.has(material.map!)).toBe(false);
        expectClose(descriptor.emissive, [2, 1, 0]);
        expect(descriptor.emissiveIntensity).toBe(1);
    });

    if (type.node) {
        it('guesses grey 0.5 for a colorNode graph with nothing to go by, and warns naming it', () => {
            const material = make();
            material.name = 'swirl';
            material.colorNode = colorNode(1, 0, 0);
            const { descriptor, warnings } = convert(material);
            expectClose(descriptor.albedo, CUSTOM_SHADING_ALBEDO);
            expect(warnings).toHaveLength(1);
            expect(warnings[0]).toContain('"swirl"');
            expect(warnings[0]).toContain('grey 0.5');
            // Its map is left out: three ignores it under a colorNode.
            material.map = dataTexture();
            const withMap = convert(material);
            expectClose(withMap.descriptor.albedo, CUSTOM_SHADING_ALBEDO);
            expect(withMap.descriptor.albedoTexture).toBeUndefined();
            expect(withMap.handleOf.size).toBe(0);
            // A colour that isn't the default white is used instead of grey.
            material.color.setRGB(0.2, 0.4, 0.6);
            expectClose(convert(material).descriptor.albedo, [0.2, 0.4, 0.6]);
            // giAlbedoMap says what to use: its texture, and no warning.
            const override = dataTexture();
            material.userData.giAlbedoMap = override;
            const told = convert(material);
            expect(told.descriptor.albedoTexture).toBe(told.handleOf.get(override));
            expect(told.warnings).toEqual([]);
        });
    }
});

describe('custom shading', () => {
    it('gives a ShaderMaterial or RawShaderMaterial grey 0.5 and a warning, unless userData says what to use', () => {
        for (const material of [new ShaderMaterial({ name: 'water' }), new RawShaderMaterial({ name: 'raw' })]) {
            const { descriptor, warnings } = convert(material);
            expectClose(descriptor.albedo, CUSTOM_SHADING_ALBEDO);
            expect(descriptor.emissive).toBeUndefined();
            expect(warnings).toHaveLength(1);
            expect(warnings[0]).toContain(`"${material.name}"`);
            material.userData.giAlbedo = 0x336699;
            const told = convert(material);
            expectClose(told.descriptor.albedo, new Color(0x336699).toArray());
            expect(told.warnings).toEqual([]);
        }
    });

    it('reads a node material with only an emissiveNode as plain: its color and map, no warning', () => {
        const map = dataTexture();
        const material = new MeshStandardNodeMaterial({ name: 'glow', map });
        material.emissiveNode = colorNode(5, 5, 5);
        const { descriptor, warnings, handleOf } = convert(material);
        // White colour × map, as three draws it; not the grey guess.
        expectClose(descriptor.albedo, [1, 1, 1]);
        expect(descriptor.albedoTexture).toBe(handleOf.get(map));
        expect(warnings).toEqual([]);
    });

    it('treats node materials with plain properties like their plain counterparts', () => {
        const material = new MeshStandardNodeMaterial({ color: new Color().setRGB(0.5, 0.25, 0.125), metalness: 0.5, roughness: 0.3 });
        const { descriptor, warnings } = convert(material);
        expectClose(descriptor.albedo, [0.5, 0.25, 0.125]);
        expect(descriptor.metalness).toBe(0.5);
        expect(warnings).toEqual([]);
    });
});

describe('alpha and light passing', () => {
    it('lets light through hidden materials and shadow catchers, whatever else they are', () => {
        expect(convert(new MeshStandardMaterial({ visible: false })).descriptor.blocksLight).toBe(false);
        expect(convert(new ShadowMaterial()).descriptor.blocksLight).toBe(false);
        const hidden = new MeshStandardMaterial({ transparent: true, visible: false });
        hidden.userData.giBlocksLight = true;
        expect(convert(hidden).descriptor.blocksLight).toBe(false);
    });

    it('makes three\'s alphaHash stochastic too', () => {
        expect(convert(new MeshStandardMaterial({ alphaHash: true, opacity: 0.5 })).descriptor).toMatchObject({ alphaMode: 'stochastic', opacity: 0.5 });
    });

    it('blocks Sponza\'s dirt_decal by its alpha (glTF BLEND, metalness 1 from no metallicFactor, a map)', () => {
        const map = dataTexture();
        const decal = new MeshStandardMaterial({ transparent: true, metalness: 1, map, name: 'dirt_decal' });
        const { descriptor, handleOf } = convert(decal);
        expect(descriptor).toMatchObject({ alphaMode: 'stochastic', opacity: 1, blocksLight: true, metalness: 1, albedoTexture: handleOf.get(map) });
    });

    it('makes a see-through material opaque for GI when it opts in (userData.giBlocksLight)', () => {
        const blended = new MeshStandardMaterial({ transparent: true, opacity: 0.2, map: dataTexture() });
        blended.userData.giBlocksLight = true;
        expect(convert(blended).descriptor).toMatchObject({ alphaMode: 'opaque', blocksLight: true });
        expect(convert(blended).descriptor.opacity).toBeUndefined();
        const glass = new MeshPhysicalMaterial({ transmission: 1 });
        glass.userData.giBlocksLight = true;
        expect(convert(glass).descriptor.blocksLight).toBe(true);
        // A cutout keeps its holes.
        const leaves = new MeshStandardMaterial({ alphaTest: 0.5 });
        leaves.userData.giBlocksLight = true;
        expect(convert(leaves).descriptor.alphaMode).toBe('cutout');
        // Only true is read.
        const notTrue = new MeshStandardMaterial({ transparent: true });
        notTrue.userData.giBlocksLight = 'yes';
        expect(convert(notTrue).descriptor.alphaMode).toBe('stochastic');
    });

    it('clamps alphaTest and opacity to 0..1', () => {
        const material = new MeshStandardMaterial({ alphaTest: 1.5, opacity: 1.2 });
        expect(convert(material).descriptor).toMatchObject({ alphaCutoff: 1, opacity: 1 });
    });

    it('warns once that alphaMap is ignored, and about uv channels the platform lacks', () => {
        const material = new MeshStandardMaterial({ alphaMap: dataTexture(), alphaTest: 0.5 });
        const map = dataTexture();
        map.channel = 2;
        material.map = map;
        const { descriptor, warnings } = convert(material);
        expect(descriptor.alphaMode).toBe('cutout');
        expect(warnings.some((warning) => warning.includes('alphaMap'))).toBe(true);
        expect(warnings.some((warning) => warning.includes('uv channel 2'))).toBe(true);
        expect(descriptor.albedoUv?.channel).toBe(0);
    });

    it('makes every material double-sided for GI whatever its side, unless it opts out', () => {
        expect(convert(new MeshStandardMaterial({ side: DoubleSide })).descriptor.doubleSided).toBe(true);
        expect(convert(new MeshStandardMaterial({ side: FrontSide })).descriptor.doubleSided).toBe(true);
        const optedOut = new MeshStandardMaterial({ side: DoubleSide });
        optedOut.userData.giSingleSided = true;
        expect(convert(optedOut).descriptor.doubleSided).toBe(false);
    });
});

describe('userData colour overrides', () => {
    it('reads Colors and [r, g, b] as linear, hex numbers and CSS strings as sRGB, and ignores anything else', () => {
        const material = new MeshLambertMaterial();
        const albedoFor = (value: unknown) => {
            material.userData.giAlbedo = value;
            return convert(material).descriptor.albedo;
        };
        expectClose(albedoFor(new Color().setRGB(0.1, 0.2, 0.3)), [0.1, 0.2, 0.3]);
        expectClose(albedoFor([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3]);
        expectClose(albedoFor(0x808080), new Color(0x808080).toArray());
        expectClose(albedoFor('#ff8000'), new Color('#ff8000').toArray());
        // Not colours: the material's own colour (white) is used.
        expectClose(albedoFor([1, 2]), [1, 1, 1]);
        expectClose(albedoFor({ r: 'red' }), [1, 1, 1]);
        expectClose(albedoFor(Number.NaN), [1, 1, 1]);
    });

    it('needs uv1 when an override map samples channel 1', () => {
        const material = new MeshLambertMaterial();
        expect(usesUvChannel1(material)).toBe(false);
        const map = dataTexture();
        map.channel = 1;
        material.userData.giAlbedoMap = map;
        expect(usesUvChannel1(material)).toBe(true);
    });
});

// ---------------------------------------------------------------------------
// Texture sources
// ---------------------------------------------------------------------------

/** A CompressedTexture with three mips of made-up blocks (16 bytes per 4×4 block). */
function compressedTexture(format: CompressedPixelFormat, colorSpace: string): CompressedTexture {
    const mipmaps = [512, 256, 128].map((size) => ({ data: new Uint8Array((size / 4) * (size / 4) * 16).fill(size & 0xff), width: size, height: size }));
    const texture = new CompressedTexture(mipmaps, 512, 512, format);
    texture.colorSpace = colorSpace;
    return texture;
}

describe('texture sources', () => {
    it('sends a BC7 CompressedTexture as compressed mips, sRGB or linear by its colour space', () => {
        const srgb = compressedTexture(RGBA_BPTC_Format, SRGBColorSpace);
        const descriptor = textureDescriptor(srgb)!;
        const source = descriptor.source as CompressedPixelData;
        expect(source.format).toBe('bc7-rgba-unorm-srgb');
        expect(descriptor.srgb).toBe(true);
        expect(source.levels.map((level) => [level.width, level.height])).toEqual([
            [512, 512],
            [256, 256],
            [128, 128],
        ]);
        // The mips' own bytes, not copies.
        expect(source.levels[1].data).toBe(srgb.mipmaps[1].data);
        const linear = textureDescriptor(compressedTexture(RGBA_BPTC_Format, NoColorSpace))!;
        expect((linear.source as CompressedPixelData).format).toBe('bc7-rgba-unorm');
        expect(linear.srgb).toBe(false);
        expect(unreadableTextureReason(srgb)).toBeNull();
    });

    it('maps BC1, BC3, BC5, ETC2 and ASTC 4×4 as three\'s backend does', () => {
        const formatOf = (format: CompressedPixelFormat, colorSpace: string = SRGBColorSpace) => (textureDescriptor(compressedTexture(format, colorSpace))!.source as CompressedPixelData).format;
        expect(formatOf(RGBA_S3TC_DXT1_Format)).toBe('bc1-rgba-unorm-srgb');
        expect(formatOf(RGBA_S3TC_DXT5_Format, NoColorSpace)).toBe('bc3-rgba-unorm');
        expect(formatOf(RGB_ETC2_Format)).toBe('etc2-rgb8unorm-srgb');
        expect(formatOf(RGBA_ETC2_EAC_Format)).toBe('etc2-rgba8unorm-srgb');
        expect(formatOf(RGBA_ASTC_4x4_Format)).toBe('astc-4x4-unorm-srgb');
        // BC5 has no sRGB variant: read linear, as three reads it, whatever the colour space says.
        const bc5 = textureDescriptor(compressedTexture(RED_GREEN_RGTC2_Format, SRGBColorSpace))!;
        expect((bc5.source as CompressedPixelData).format).toBe('bc5-rg-unorm');
        expect(bc5.srgb).toBe(false);
    });

    it('skips compressed formats GI doesn\'t read, and compressed textures with no mips yet, saying why only for the first', () => {
        const bc6h = compressedTexture(RGB_BPTC_UNSIGNED_Format, NoColorSpace);
        expect(textureDescriptor(bc6h)).toBeNull();
        expect(unreadableTextureReason(bc6h)).toContain(`format ${RGB_BPTC_UNSIGNED_Format}`);
        const empty = new CompressedTexture([], 512, 512, RGBA_BPTC_Format);
        expect(textureDescriptor(empty)).toBeNull();
        expect(unreadableTextureReason(empty)).toBeNull();
    });

    it('sends an 8-bit RGBA data texture as PixelData, sRGB by its colour space', () => {
        const texture = dataTexture();
        const linear = textureDescriptor(texture)!;
        expect(linear.srgb).toBe(false);
        expect((linear.source as PixelData).data).toBe(texture.image.data);
        texture.colorSpace = SRGBColorSpace;
        expect(textureDescriptor(texture)!.srgb).toBe(true);
    });

    it('skips float data textures and render targets with a reason, and images still loading without one', () => {
        const float = new DataTexture(new Float32Array(16), 2, 2, undefined, FloatType);
        expect(textureDescriptor(float)).toBeNull();
        expect(unreadableTextureReason(float)).toContain('8-bit RGBA');
        const target = new RenderTarget(4, 4).texture;
        expect(textureDescriptor(target)).toBeNull();
        expect(unreadableTextureReason(target)).toContain('render-target');
        const loading = new Texture();
        expect(textureDescriptor(loading)).toBeNull();
        expect(unreadableTextureReason(loading)).toBeNull();
    });
});

// ---------------------------------------------------------------------------
// UV mappings
// ---------------------------------------------------------------------------

describe('UV mappings', () => {
    /** A texture as glTF loads it (`flipY` false), so the matrix passes through unchanged. */
    function unflipped(): Texture {
        const texture = new Texture();
        texture.flipY = false;
        return texture;
    }

    it('passes texture.matrix as the contract\'s 3x2', () => {
        const texture = unflipped();
        texture.repeat.set(2, 3);
        texture.offset.set(0.25, 0.5);
        expectClose(uvMapping(texture).matrix, [2, 0, 0, 3, 0.25, 0.5]);
    });

    it('matches three\'s own matrix with rotation and centre', () => {
        const texture = unflipped();
        texture.repeat.set(4, 0.5);
        texture.offset.set(-0.1, 0.3);
        texture.rotation = 0.7;
        texture.center.set(0.5, 0.25);
        const mapping = uvMapping(texture);
        const e = texture.matrix.elements;
        expectClose(mapping.matrix, [e[0], e[1], e[3], e[4], e[6], e[7]]);
        // A point maps the same way through both: uv' = M (u, v, 1).
        const [u, v] = [0.3, 0.8];
        const viaContract = [mapping.matrix![0] * u + mapping.matrix![2] * v + mapping.matrix![4], mapping.matrix![1] * u + mapping.matrix![3] * v + mapping.matrix![5]];
        const viaThree = [e[0] * u + e[3] * v + e[6], e[1] * u + e[4] * v + e[7]];
        expectClose(viaContract, viaThree);
    });

    it('turns v upside down after three\'s matrix for a flipY texture (three uploads it flipped; the platform never flips)', () => {
        const texture = new Texture();
        expect(texture.flipY).toBe(true);
        texture.repeat.set(2, 3);
        texture.offset.set(0.25, 0.5);
        texture.rotation = 0.4;
        const mapping = uvMapping(texture);
        const e = texture.matrix.elements;
        const [u, v] = [0.3, 0.8];
        const threeV = e[1] * u + e[4] * v + e[7];
        const platformV = mapping.matrix![1] * u + mapping.matrix![3] * v + mapping.matrix![5];
        // three's v counts from the image's bottom row, the platform's from its top.
        expect(platformV).toBeCloseTo(1 - threeV, 12);
        expect(mapping.matrix![0] * u + mapping.matrix![2] * v + mapping.matrix![4]).toBeCloseTo(e[0] * u + e[3] * v + e[6], 12);
        // Compressed textures are never flipped by three, so neither is their mapping.
        const compressed = compressedTexture(RGBA_BPTC_Format, SRGBColorSpace);
        compressed.flipY = true;
        expectClose(uvMapping(compressed).matrix, [1, 0, 0, 1, 0, 0]);
    });

    it('uses a matrix three doesn\'t update as it is', () => {
        const texture = unflipped();
        texture.matrixAutoUpdate = false;
        texture.matrix.set(1, 2, 3, 4, 5, 6, 0, 0, 1);
        expectClose(uvMapping(texture).matrix, [1, 4, 2, 5, 3, 6]);
    });

    it('passes the uv channel and the wraps', () => {
        const texture = new Texture();
        texture.channel = 1;
        texture.wrapS = MirroredRepeatWrapping;
        texture.wrapT = ClampToEdgeWrapping;
        const mapping = uvMapping(texture);
        expect(mapping.channel).toBe(1);
        expect(mapping.wrapU).toBe('mirror');
        expect(mapping.wrapV).toBe('clamp');
        texture.wrapS = RepeatWrapping;
        expect(uvMapping(texture).wrapU).toBe('repeat');
    });

    it('falls back to channel 0 for channels the platform lacks', () => {
        const texture = new Texture();
        texture.channel = 2;
        const mapping = uvMapping(texture);
        expect(mapping.channel).toBe(0);
        expect(mapping.channelSupported).toBe(false);
    });

    it('leaves the mapping out when the texture has no handle yet', () => {
        const material = new MeshStandardMaterial({ map: new Texture() });
        const descriptor = materialDescriptor(material, noTextures);
        expect(descriptor.albedoTexture).toBeUndefined();
        expect(descriptor.albedoUv).toBeUndefined();
        expect(materialDescriptor(material, () => HANDLE).albedoUv).toBeDefined();
    });
});

describe('vertex normals are passed through', () => {
    it('passes the normal attribute', () => {
        const geometry = new BoxGeometry(1, 2, 3);
        const descriptor = meshDescriptor(geometry)!;
        expect(descriptor.normals).toEqual(geometry.getAttribute('normal').array);
        expect(descriptor.positions).toEqual(geometry.getAttribute('position').array);
        expect(descriptor.uvs).toEqual(geometry.getAttribute('uv').array);
        expect(descriptor.indices).toEqual(geometry.index!.array);
    });

    it('de-interleaves normals', () => {
        // position xyz, normal xyz per vertex, in one buffer.
        const values = new Float32Array([0, 0, 0, 0, 1, 0, 1, 0, 0, 0.6, 0.8, 0, 0, 0, 1, 0, 0, -1]);
        const buffer = new InterleavedBuffer(values, 6);
        const geometry = new BufferGeometry();
        geometry.setAttribute('position', new InterleavedBufferAttribute(buffer, 3, 0));
        geometry.setAttribute('normal', new InterleavedBufferAttribute(buffer, 3, 3));
        const descriptor = meshDescriptor(geometry)!;
        expectClose(Array.from(descriptor.normals!), [0, 1, 0, 0.6, 0.8, 0, 0, 0, -1]);
        expectClose(Array.from(descriptor.positions), [0, 0, 0, 1, 0, 0, 0, 0, 1]);
    });

    it('undoes normalized integer normals', () => {
        const attribute = new BufferAttribute(new Int8Array([0, 127, 0, -127, 0, 0]), 3, true);
        expectClose(Array.from(readFloatAttribute(attribute, 3)), [0, 1, 0, -1, 0, 0], 5);
    });

    it('leaves normals out when the geometry has none', () => {
        const geometry = new BufferGeometry();
        geometry.setAttribute('position', new BufferAttribute(new Float32Array(9), 3));
        expect(meshDescriptor(geometry)!.normals).toBeUndefined();
    });
});

describe('meshes', () => {
    it('makes up an index for non-indexed geometry', () => {
        const geometry = new BufferGeometry();
        geometry.setAttribute('position', new BufferAttribute(new Float32Array(18), 3));
        const descriptor = meshDescriptor(geometry)!;
        expect(Array.from(descriptor.indices)).toEqual([0, 1, 2, 3, 4, 5]);
        expect(descriptor.indices).toBeInstanceOf(Uint16Array);
    });

    it('passes groups as material slots, clipped to whole triangles', () => {
        const geometry = new BoxGeometry();
        const descriptor = meshDescriptor(geometry)!;
        expect(descriptor.groups).toHaveLength(6);
        expect(descriptor.groups![5]).toEqual({ start: 30, count: 6, materialSlot: 5 });
        geometry.clearGroups();
        geometry.addGroup(0, Infinity, 2);
        expect(meshDescriptor(geometry)!.groups).toEqual([{ start: 0, count: 36, materialSlot: 2 }]);
    });

    it('passes uv1 only when asked', () => {
        const geometry = new BoxGeometry();
        geometry.setAttribute('uv1', geometry.getAttribute('uv').clone());
        expect(meshDescriptor(geometry)!.uvs1).toBeUndefined();
        expect(meshDescriptor(geometry, { includeUv1: true })!.uvs1).toEqual(geometry.getAttribute('uv1').array);
    });

    it('refuses a geometry without positions', () => {
        expect(meshDescriptor(new BufferGeometry())).toBeNull();
    });
});

describe('lights', () => {
    it('sends a point light with distance 0 as range 0 (no limit)', () => {
        const light = new PointLight(0xffffff, 15, 0, 2);
        light.position.set(1, 2, 3);
        light.updateMatrixWorld();
        const descriptor = lightDescriptor(light)!;
        expect(descriptor.type).toBe('point');
        if (descriptor.type !== 'point') return;
        expect(descriptor.range).toBe(0);
        expect(descriptor.position).toEqual([1, 2, 3]);
        expect(descriptor.intensity).toBe(15);
    });

    it('passes a point light\'s distance as its range', () => {
        const light = new PointLight(0xffffff, 1, 7.5);
        light.updateMatrixWorld();
        const descriptor = lightDescriptor(light)!;
        expect(descriptor.type === 'point' && descriptor.range).toBe(7.5);
    });

    it('points a directional light from its position to its target', () => {
        const light = new DirectionalLight(new Color().setRGB(1, 0.5, 0.25), 3);
        light.position.set(0, 10, 0);
        light.target.position.set(10, 0, 0);
        light.updateMatrixWorld();
        light.target.updateMatrixWorld();
        const descriptor = lightDescriptor(light)!;
        expect(descriptor.type).toBe('directional');
        if (descriptor.type !== 'directional') return;
        expectClose(descriptor.direction, [Math.SQRT1_2, -Math.SQRT1_2, 0]);
        expectClose(descriptor.color, [1, 0.5, 0.25]);
        expect(descriptor.intensity).toBe(3);
    });

    it('maps a spot light\'s angle and penumbra to outer and inner cones', () => {
        const light = new SpotLight(0xffffff, 100, 0, Math.PI / 4, 0.5, 2);
        light.position.set(0, 3, 0);
        light.target.position.set(0, 0, 0);
        light.updateMatrixWorld();
        light.target.updateMatrixWorld();
        const descriptor = lightDescriptor(light)!;
        expect(descriptor.type).toBe('spot');
        if (descriptor.type !== 'spot') return;
        expect(descriptor.outerConeAngle).toBeCloseTo(Math.PI / 4);
        expect(descriptor.innerConeAngle).toBeCloseTo(Math.PI / 8);
        expect(descriptor.range).toBe(0);
        expectClose(descriptor.direction, [0, -1, 0]);
    });

    it('takes the light group from userData and scales hidden lights to 0', () => {
        const light = new PointLight(0xffffff, 4);
        light.userData.giLightGroup = 2;
        light.updateMatrixWorld();
        expect(lightDescriptor(light)!.group).toBe(2);
        expect(lightDescriptor(light, 0)!.intensity).toBe(0);
        light.userData.giLightGroup = 9;
        expect(lightDescriptor(light)!.group).toBe(0);
    });

    it("shadows a light in GI by three's castShadow, unless userData.giCastsShadows says otherwise", () => {
        const light = new PointLight(0xffffff, 4);
        light.updateMatrixWorld();
        const numbers = new Float64Array(LIGHT_NUMBER_COUNT);
        const changed = new Float64Array(LIGHT_NUMBER_COUNT);
        expect(lightDescriptor(light)!.castsShadows).toBe(false);
        readLightNumbers(light, 1, numbers);
        // A candle three draws unshadowed (too many for shadow maps), shadowed in GI.
        light.userData.giCastsShadows = true;
        expect(lightDescriptor(light)!.castsShadows).toBe(true);
        readLightNumbers(light, 1, changed);
        expect([...changed]).not.toEqual([...numbers]);
        light.castShadow = true;
        light.userData.giCastsShadows = false;
        expect(lightDescriptor(light)!.castsShadows).toBe(false);
        // Only a boolean counts.
        light.userData.giCastsShadows = 'yes';
        expect(lightDescriptor(light)!.castsShadows).toBe(true);
    });
});

describe('material snapshots', () => {
    it('see colour, metalness and texture edits that three\'s version misses', () => {
        const map = new DataTexture(new Uint8Array(4), 1, 1);
        const material = new MeshStandardMaterial({ color: 0xff0000, map });
        const before = new MaterialSnapshot();
        const after = new MaterialSnapshot();
        before.read(material);
        after.read(material);
        expect(after.equals(before)).toBe(true);
        material.color.setRGB(0, 1, 0);
        after.read(material);
        expect(after.equals(before)).toBe(false);
        before.read(material);
        material.metalness = 0.5;
        after.read(material);
        expect(after.equals(before)).toBe(false);
        before.read(material);
        map.offset.set(0.5, 0);
        after.read(material);
        expect(after.equals(before)).toBe(false);
        before.read(material);
        material.map = null;
        after.read(material);
        expect(after.equals(before)).toBe(false);
    });

    it('see texture versions, flipY, alpha modes, custom shading and the userData overrides', () => {
        const map = dataTexture();
        const material = new MeshStandardNodeMaterial({ map });
        const before = new MaterialSnapshot();
        const after = new MaterialSnapshot();
        const edits: (() => void)[] = [
            () => (map.needsUpdate = true),
            () => (map.flipY = true),
            () => (material.alphaHash = true),
            () => (material.colorNode = colorNode(1, 0, 0)),
            () => (material.userData.giAlbedo = [0.1, 0.2, 0.3]),
            () => (material.userData.giAlbedo = [0.1, 0.2, 0.4]),
            () => (material.userData.giEmissive = 0xff0000),
            () => (material.userData.giAlbedoMap = dataTexture()),
            () => (material.userData.giBlocksLight = true),
        ];
        for (const edit of edits) {
            before.read(material);
            edit();
            after.read(material);
            expect(after.equals(before)).toBe(false);
        }
        // Reading again without an edit: equal.
        before.read(material);
        after.read(material);
        expect(after.equals(before)).toBe(true);
    });
});
