/**
 * The shared sky in three.js: the manifest's procedural sky is the committed
 * file `scenes/sky/<scene id>.exr` (made by `tools/sky/generate.ts`), the same
 * file Cycles and the platform read. Nothing here computes a sky.
 *
 * Two ways the sky reaches the picture:
 *  - {@link createSkyDome}: what cameras see, and what the light probe grid's
 *    bake cameras capture. This is how the viewer always shows it, as the
 *    baselines did with three's `SkyMesh`.
 *  - {@link applySkyEnvironmentLighting}: the same file as `scene.environment`,
 *    three's own image-based lighting (prefiltered by PMREM). The viewer never
 *    lights surfaces this way: three r187 adds environment diffuse on top of
 *    any probe GI, so the probe grid would count the sky twice (docs/NOTES.md,
 *    "Env diffuse counted twice"). Only `tools/sky/pmrem-env-render.ts` uses it,
 *    to measure three's PMREM on its own.
 */

import { BackSide, BoxGeometry, EquirectangularReflectionMapping, HalfFloatType, LinearFilter, Mesh, RepeatWrapping, type Scene, type DataTexture } from 'three';
import { EXRLoader } from 'three/addons/loaders/EXRLoader.js';
import { cameraPosition, equirectUV, Fn, modelViewProjection, positionWorld, texture, vec4 } from 'three/tsl';
import { NodeMaterial, type Node } from 'three/webgpu';
import { skyFileName } from '../../../../scenes/src/sky/sky-file.ts';
import { GI_USER_DATA } from '../ddgi/library';

/**
 * Every committed sky file, as a URL vite serves (the dev server and a build
 * both). Keyed by the path below; see {@link skyFileUrl}.
 */
const SKY_FILE_URLS = import.meta.glob('../../../../scenes/sky/*.exr', { query: '?url', import: 'default', eager: true }) as Record<string, string>;

export function skyFileUrl(sceneId: string): string {
    const url = SKY_FILE_URLS[`../../../../scenes/sky/${skyFileName(sceneId)}`];
    if (!url) throw new Error(`no sky file for scene "${sceneId}" in scenes/sky; run "node tools/sky/generate.ts"`);
    return url;
}

/**
 * Loads a scene's sky as a three.js texture: half floats as stored (no
 * conversion), linear filtering, no mipmaps, wrapping across the -X seam.
 * `EXRLoader` puts the file's last row (the nadir) first in memory, so
 * three's `equirectUV` (v = 1 at the zenith) reads it the right way up.
 */
export async function loadSkyTexture(sceneId: string): Promise<DataTexture> {
    const loader = new EXRLoader();
    loader.setDataType(HalfFloatType);
    const sky = await loader.loadAsync(skyFileUrl(sceneId));
    sky.name = `sky ${sceneId}`;
    sky.mapping = EquirectangularReflectionMapping;
    sky.wrapS = RepeatWrapping;
    sky.minFilter = LinearFilter;
    sky.magFilter = LinearFilter;
    sky.generateMipmaps = false;
    sky.needsUpdate = true;
    return sky;
}

/**
 * Size of the sky box. Its vertices go to the far plane, so the size only has
 * to keep the box around every camera (the value three's `SkyMesh` examples use).
 */
const SKY_BOX_SIZE = 450_000;

/**
 * A box around the world showing the sky file, drawn behind everything. It is
 * a mesh rather than `scene.background` so it keeps the layer rules of
 * `scene/layers.ts` (bake cameras on layer 0 see it; so do viewing cameras),
 * as three's `SkyMesh` did before.
 */
export function createSkyDome(sky: DataTexture): Mesh {
    const material = new NodeMaterial();
    material.side = BackSide;
    material.depthWrite = false;
    material.fog = false;
    // Put every vertex on the far plane (z = w, so depth is 1 after the divide), as SkyMesh does.
    material.vertexNode = Fn(() => {
        // Typed as a plain Node; it is the vec4 clip-space position.
        const clipPosition = modelViewProjection as unknown as Node<'vec4'>;
        clipPosition.z.assign(clipPosition.w);
        return clipPosition;
    })();
    // The view direction from whichever camera is drawing (cube cameras too).
    const viewDirection = positionWorld.sub(cameraPosition).normalize();
    material.colorNode = vec4(texture(sky, equirectUV(viewDirection)).rgb, 1);

    // DDGI's screen debug views replace every material (scene.overrideMaterial); the sky stays the sky.
    material.allowOverride = false;

    const dome = new Mesh(new BoxGeometry(1, 1, 1), material);
    dome.name = 'manifest-sky';
    dome.scale.setScalar(SKY_BOX_SIZE);
    dome.frustumCulled = false;
    // A picture of the sky, not geometry: DDGI's rays get the same sky from the platform's
    // own environment (platformSkyEnvironment), and must not hit a 450 km box instead.
    dome.userData[GI_USER_DATA.contributes] = false;
    return dome;
}

/** Lights the scene's materials with the sky through three's environment map (PMREM). Not used by the viewer; see the file comment. */
export function applySkyEnvironmentLighting(scene: Scene, sky: DataTexture): void {
    scene.environment = sky;
    scene.environmentIntensity = 1;
}
