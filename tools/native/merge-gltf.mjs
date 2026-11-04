// Merges the model parts of one scene manifest into a single .gltf for the native test harness.
//
// Why: the harness loads exactly one glTF file per scene and cannot place a model, but our
// manifests can list several model parts (Sponza = Intel's base package + the curtains add-on),
// each with its own transform. The viewer and Cycles load all parts with their transforms; this
// writes one file that the harness reads the same way. Nothing in assets/ is touched: the merged
// file only points at the parts' own buffers and images, by relative paths, so no geometry or
// texture is copied.
//
// What the merge does:
//   - every top-level list (buffers, bufferViews, accessors, images, samplers, textures,
//     materials, meshes, nodes, cameras, KHR_lights_punctual lights) is concatenated, and every
//     index into those lists is shifted by the size of the parts before it;
//   - each part's default-scene root nodes go into one scene. A part with a non-identity manifest
//     transform gets one extra root node that carries it (translation, rotation as a quaternion
//     from the manifest's XYZ Euler degrees, uniform scale: the same order three.js uses), with
//     the part's roots as children. Identity parts get no extra node;
//   - buffer and image URIs are rewritten relative to the output folder.
// Animations and skins are not supported (none of our model parts have them); a part with either
// is refused rather than merged wrongly.
//
// Usage: node merge-gltf.mjs <spec.json>
//   spec = { "output": "<folder>", "name": "<file name without .gltf>",
//            "parts": [ { "path": "<part.gltf>", "label": "...", "transform": { position, rotationDeg, scale } } ] }
// Exit code 0: the merged file was already up to date (same content, no source file newer).
// Exit code 3: the merged file was written (new or changed), so the caller must drop the old scene cache.
// Any other exit code is an error.

import { existsSync, readFileSync, statSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';

const EXIT_WRITTEN = 3;
// Upstream's Vulkan harness has room for 2048 buffer descriptors: 4 plus an index and a vertex
// buffer per mesh. The fork sizes that array from the scene, but say it when a merge crosses it.
const UPSTREAM_VULKAN_MESH_LIMIT = 1022;

function isDataUri(uri) {
    return typeof uri === 'string' && uri.startsWith('data:');
}

// The relative path from the output folder to a file a part refers to, with forward slashes.
// The harness only decodes %20 in image URIs, so spaces are the only character that is encoded.
function rebaseUri(uri, partFolder, outputFolder) {
    if (uri === undefined || isDataUri(uri)) return uri;
    const absolute = resolve(partFolder, decodeURIComponent(uri));
    if (!existsSync(absolute)) throw new Error(`${absolute} (referenced by a part) does not exist`);
    return relative(outputFolder, absolute).replaceAll('\\', '/').replaceAll(' ', '%20');
}

// Three.js Euler 'XYZ' (degrees) to a glTF quaternion [x, y, z, w].
function quaternionFromEulerXyzDegrees([xDeg, yDeg, zDeg]) {
    const half = Math.PI / 360;
    const [c1, c2, c3] = [Math.cos(xDeg * half), Math.cos(yDeg * half), Math.cos(zDeg * half)];
    const [s1, s2, s3] = [Math.sin(xDeg * half), Math.sin(yDeg * half), Math.sin(zDeg * half)];
    return [
        s1 * c2 * c3 + c1 * s2 * s3,
        c1 * s2 * c3 - s1 * c2 * s3,
        c1 * c2 * s3 + s1 * s2 * c3,
        c1 * c2 * c3 - s1 * s2 * s3,
    ];
}

function isIdentity(transform) {
    return transform.position.every((v) => v === 0) && transform.rotationDeg.every((v) => v === 0) && transform.scale === 1;
}

// Material texture references are objects with an `index` under a key ending in "Texture"
// (baseColorTexture, normalTexture, ..., also inside material extensions). Shift them all.
function shiftTextureIndices(value, textureOffset) {
    if (Array.isArray(value)) {
        value.forEach((item) => shiftTextureIndices(item, textureOffset));
        return;
    }
    if (value === null || typeof value !== 'object') return;
    for (const [key, child] of Object.entries(value)) {
        if (key.endsWith('Texture') && child && typeof child.index === 'number') child.index += textureOffset;
        shiftTextureIndices(child, textureOffset);
    }
}

function newestModifiedTime(paths) {
    return Math.max(...paths.map((path) => statSync(path).mtimeMs));
}

function main() {
    const specPath = process.argv[2];
    if (!specPath) {
        console.error('Usage: node merge-gltf.mjs <spec.json>');
        process.exit(2);
    }
    const spec = JSON.parse(readFileSync(specPath, 'utf8'));
    if (!/^[\w-]+$/.test(spec.name)) throw new Error(`name "${spec.name}" must be letters, digits, "_" and "-" only`);
    const outputFolder = resolve(spec.output);
    mkdirSync(outputFolder, { recursive: true });

    const merged = {
        asset: { version: '2.0', generator: 'webgpu-ddgi tools/native/merge-gltf.mjs' },
        scene: 0,
        scenes: [{ name: spec.name, nodes: [] }],
        nodes: [], meshes: [], accessors: [], bufferViews: [], buffers: [],
        materials: [], textures: [], images: [], samplers: [], cameras: [],
    };
    const lights = [];
    const extensionsUsed = new Set();
    const extensionsRequired = new Set();
    // The files the merged result depends on: when one is newer than the merged file, it is
    // rewritten (and the caller drops the scene cache). Spec changes show up in the content check.
    const sourceFiles = [];
    let primitives = 0;

    for (const part of spec.parts) {
        const partPath = resolve(part.path);
        const partFolder = dirname(partPath);
        const json = JSON.parse(readFileSync(partPath, 'utf8'));
        sourceFiles.push(partPath);
        if ((json.animations ?? []).length > 0 || (json.skins ?? []).length > 0) {
            throw new Error(`${partPath} has animations or skins; merging those is not supported`);
        }

        // Where this part's entries start in the merged lists.
        const offset = {
            buffer: merged.buffers.length, bufferView: merged.bufferViews.length, accessor: merged.accessors.length,
            image: merged.images.length, sampler: merged.samplers.length, texture: merged.textures.length,
            material: merged.materials.length, mesh: merged.meshes.length, node: merged.nodes.length,
            camera: merged.cameras.length, light: lights.length,
        };

        for (const buffer of json.buffers ?? []) {
            if (buffer.uri && !isDataUri(buffer.uri)) sourceFiles.push(resolve(partFolder, decodeURIComponent(buffer.uri)));
            merged.buffers.push({ ...buffer, uri: rebaseUri(buffer.uri, partFolder, outputFolder) });
        }
        for (const view of json.bufferViews ?? []) merged.bufferViews.push({ ...view, buffer: view.buffer + offset.buffer });
        for (const accessor of json.accessors ?? []) {
            const copy = structuredClone(accessor);
            if (copy.bufferView !== undefined) copy.bufferView += offset.bufferView;
            if (copy.sparse) {
                copy.sparse.indices.bufferView += offset.bufferView;
                copy.sparse.values.bufferView += offset.bufferView;
            }
            merged.accessors.push(copy);
        }
        for (const image of json.images ?? []) {
            const copy = { ...image };
            if (copy.bufferView !== undefined) copy.bufferView += offset.bufferView;
            if (copy.uri && !isDataUri(copy.uri)) sourceFiles.push(resolve(partFolder, decodeURIComponent(copy.uri)));
            copy.uri = rebaseUri(copy.uri, partFolder, outputFolder);
            if (copy.uri === undefined) delete copy.uri;
            merged.images.push(copy);
        }
        for (const sampler of json.samplers ?? []) merged.samplers.push({ ...sampler });
        for (const texture of json.textures ?? []) {
            const copy = structuredClone(texture);
            if (copy.source !== undefined) copy.source += offset.image;
            if (copy.sampler !== undefined) copy.sampler += offset.sampler;
            // Image extensions (EXT_texture_webp, KHR_texture_basisu, ...) carry their own source.
            for (const extension of Object.values(copy.extensions ?? {})) {
                if (typeof extension.source === 'number') extension.source += offset.image;
            }
            merged.textures.push(copy);
        }
        for (const material of json.materials ?? []) {
            const copy = structuredClone(material);
            shiftTextureIndices(copy, offset.texture);
            merged.materials.push(copy);
        }
        for (const mesh of json.meshes ?? []) {
            const copy = structuredClone(mesh);
            for (const primitive of copy.primitives) {
                primitives++;
                for (const name of Object.keys(primitive.attributes)) primitive.attributes[name] += offset.accessor;
                for (const target of primitive.targets ?? []) {
                    for (const name of Object.keys(target)) target[name] += offset.accessor;
                }
                if (primitive.indices !== undefined) primitive.indices += offset.accessor;
                if (primitive.material !== undefined) primitive.material += offset.material;
            }
            merged.meshes.push(copy);
        }
        for (const camera of json.cameras ?? []) merged.cameras.push(structuredClone(camera));
        for (const light of json.extensions?.KHR_lights_punctual?.lights ?? []) lights.push(structuredClone(light));
        for (const node of json.nodes ?? []) {
            const copy = structuredClone(node);
            if (copy.mesh !== undefined) copy.mesh += offset.mesh;
            if (copy.camera !== undefined) copy.camera += offset.camera;
            if (copy.children) copy.children = copy.children.map((child) => child + offset.node);
            const light = copy.extensions?.KHR_lights_punctual;
            if (light && typeof light.light === 'number') light.light += offset.light;
            merged.nodes.push(copy);
        }
        (json.extensionsUsed ?? []).forEach((name) => extensionsUsed.add(name));
        (json.extensionsRequired ?? []).forEach((name) => extensionsRequired.add(name));

        const roots = (json.scenes?.[json.scene ?? 0]?.nodes ?? []).map((root) => root + offset.node);
        if (isIdentity(part.transform)) {
            merged.scenes[0].nodes.push(...roots);
        } else {
            // One extra node carries the manifest transform; the part's own roots hang under it.
            merged.nodes.push({
                name: `part:${part.label}`,
                translation: part.transform.position,
                rotation: quaternionFromEulerXyzDegrees(part.transform.rotationDeg),
                scale: [part.transform.scale, part.transform.scale, part.transform.scale],
                children: roots,
            });
            merged.scenes[0].nodes.push(merged.nodes.length - 1);
        }
        console.log(`  part ${part.label}: ${(json.meshes ?? []).length} meshes, ${(json.images ?? []).length} images${isIdentity(part.transform) ? '' : ', placed with its manifest transform'}`);
    }

    if (lights.length > 0) merged.extensions = { KHR_lights_punctual: { lights } };
    if (extensionsUsed.size > 0) merged.extensionsUsed = [...extensionsUsed];
    if (extensionsRequired.size > 0) merged.extensionsRequired = [...extensionsRequired];
    for (const list of ['cameras', 'samplers', 'images', 'textures', 'materials']) {
        if (merged[list].length === 0) delete merged[list];
    }

    console.log(`  merged: ${merged.meshes.length} meshes (${primitives} primitives), ${merged.images?.length ?? 0} images, ${merged.nodes.length} nodes`);
    if (merged.meshes.length > UPSTREAM_VULKAN_MESH_LIMIT) {
        console.log(`  note: more than ${UPSTREAM_VULKAN_MESH_LIMIT} meshes. Upstream's Vulkan harness loses the GPU on this; the fork (1df1f97 and later) sizes its buffer array from the scene.`);
    }

    // Write only when something changed, so an unchanged scene keeps its texture cache.
    const outputPath = join(outputFolder, `${spec.name}.gltf`);
    const text = JSON.stringify(merged);
    const upToDate = existsSync(outputPath) && readFileSync(outputPath, 'utf8') === text &&
        statSync(outputPath).mtimeMs >= newestModifiedTime(sourceFiles);
    if (upToDate) {
        console.log(`  ${outputPath} is up to date`);
        process.exit(0);
    }
    writeFileSync(outputPath, text);
    console.log(`  wrote ${outputPath}`);
    process.exit(EXIT_WRITTEN);
}

main();
