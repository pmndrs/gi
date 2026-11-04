// Splits a .glb with embedded textures into a .gltf + .bin + image files.
//
// Why: the native test harness refuses .glb files that contain textures
// ("GLB format not supported for scenes with texture data") because it loads
// textures from disk by name. A .gltf with the images written next to it is
// accepted. Nothing else about the scene changes: the geometry stays in the
// .bin, and every texture, material and node index stays the same.
//
// WebP textures (the Blender 4.5 exporter writes them for the Bistro scenes, as
// the EXT_texture_webp extension) are converted to PNG with ImageMagick, because
// the harness reads PNG and JPEG only and silently ignores textures it cannot
// find a plain `source` for. Without this step the Bistro materials point at
// textures that do not exist.
//
// Normal maps need tangents. The harness builds its tangent frame from the
// mesh's TANGENT attribute and shades with garbage (black and purple surfaces,
// NaN in the probes) when a normal map is used on a mesh without one. The Bistro
// files have no tangents, so the normal textures are removed from the materials
// in that case. The scene geometry and all other textures stay as they are.
//
// Usage: node glb-to-gltf.mjs <input.glb> <output folder> <name>
// Writes <output folder>/<name>.gltf, <name>.bin and images/<n>.<ext>.
// The output name must be letters, digits, "_" and "-" only: the harness
// rejects anything else.

import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, readdirSync, unlinkSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const GLB_MAGIC = 0x46546c67; // "glTF"
const JSON_CHUNK = 0x4e4f534a; // "JSON"
const BIN_CHUNK = 0x004e4942; // "BIN\0"

const IMAGE_EXTENSIONS = {
    'image/png': 'png',
    'image/jpeg': 'jpg',
    'image/webp': 'webp',
};

function readGlb(path) {
    const file = readFileSync(path);
    if (file.readUInt32LE(0) !== GLB_MAGIC) throw new Error(`${path} is not a GLB file`);
    let offset = 12;
    let json = null;
    let bin = null;
    while (offset < file.length) {
        const length = file.readUInt32LE(offset);
        const type = file.readUInt32LE(offset + 4);
        const data = file.subarray(offset + 8, offset + 8 + length);
        if (type === JSON_CHUNK) json = JSON.parse(data.toString('utf8'));
        else if (type === BIN_CHUNK) bin = data;
        offset += 8 + length;
    }
    if (!json || !bin) throw new Error(`${path} has no JSON or no binary chunk`);
    return { json, bin };
}

function convertWebpToPng(json, imageFolder) {
    const webpFiles = readdirSync(imageFolder).filter((file) => file.endsWith('.webp'));
    if (webpFiles.length === 0) return;
    console.log(`Converting ${webpFiles.length} WebP images to PNG with ImageMagick...`);
    // mogrify converts in place into the same folder, one file after the other
    for (let start = 0; start < webpFiles.length; start += 40) {
        const batch = webpFiles.slice(start, start + 40);
        execFileSync('magick', ['mogrify', '-format', 'png', ...batch], { cwd: imageFolder, stdio: 'inherit' });
    }
    webpFiles.forEach((file) => unlinkSync(join(imageFolder, file)));
    for (const image of json.images ?? []) {
        if (image.uri?.endsWith('.webp')) image.uri = image.uri.replace(/\.webp$/, '.png');
    }
    // Point textures at their image the plain way and drop the extension.
    for (const texture of json.textures ?? []) {
        const webp = texture.extensions?.EXT_texture_webp;
        if (webp) {
            texture.source = webp.source;
            delete texture.extensions.EXT_texture_webp;
            if (Object.keys(texture.extensions).length === 0) delete texture.extensions;
        }
    }
    for (const list of ['extensionsUsed', 'extensionsRequired']) {
        if (json[list]) json[list] = json[list].filter((name) => name !== 'EXT_texture_webp');
    }
}

function dropNormalMapsWithoutTangents(json) {
    const hasTangents = (json.meshes ?? []).some((mesh) => mesh.primitives.some((primitive) => primitive.attributes.TANGENT !== undefined));
    if (hasTangents) return;
    let removed = 0;
    for (const material of json.materials ?? []) {
        if (material.normalTexture) {
            delete material.normalTexture;
            removed++;
        }
    }
    console.log(`No TANGENT attributes in the meshes: removed the normal map from ${removed} materials.`);
}

function main() {
    const [input, outputFolder, name] = process.argv.slice(2);
    if (!input || !outputFolder || !name) {
        console.error('Usage: node glb-to-gltf.mjs <input.glb> <output folder> <name>');
        process.exit(2);
    }
    if (!/^[\w-]+$/.test(name)) throw new Error(`name "${name}" must be letters, digits, "_" and "-" only`);

    const { json, bin } = readGlb(input);
    mkdirSync(join(outputFolder, 'images'), { recursive: true });

    const images = json.images ?? [];
    images.forEach((image, index) => {
        if (image.bufferView === undefined) return; // already an external file
        const view = json.bufferViews[image.bufferView];
        const extension = IMAGE_EXTENSIONS[image.mimeType];
        if (!extension) throw new Error(`image ${index} has unsupported type ${image.mimeType}`);
        const bytes = bin.subarray(view.byteOffset ?? 0, (view.byteOffset ?? 0) + view.byteLength);
        const relativePath = `images/${index}.${extension}`;
        writeFileSync(join(outputFolder, relativePath), bytes);
        image.uri = relativePath;
        delete image.bufferView;
        delete image.mimeType;
    });

    convertWebpToPng(json, join(outputFolder, 'images'));
    dropNormalMapsWithoutTangents(json);

    // The GLB has exactly one buffer: the binary chunk.
    json.buffers = [{ uri: `${name}.bin`, byteLength: bin.length }];
    writeFileSync(join(outputFolder, `${name}.bin`), bin);
    writeFileSync(join(outputFolder, `${name}.gltf`), JSON.stringify(json));
    console.log(`${images.length} images, ${(bin.length / 1048576).toFixed(1)} MB binary -> ${join(outputFolder, name + '.gltf')}`);
}

main();
