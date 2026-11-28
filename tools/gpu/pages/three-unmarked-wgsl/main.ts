/**
 * T1.14: the composite leaves GI per fragment exactly as it was. The
 * fragment WGSL three builds for a Standard and a Lambert material lit by
 * the GI light, in an unmarked render, must be the same text:
 *
 * 1. before any composite exists;
 * 2. after a composite pipeline (`ddgi()` on another pass, with its MRT and
 *    mark) has been built and drawn, so every composite code path (the
 *    coverage-only entry, the winner flag, `giAlbedo`) has run in this page.
 *
 * And the unmarked text must hold none of the composite's own symbols. (A
 * one-off check during T1.14 also found it byte-identical to the build from
 * before T1.14; that comparison needs the old code, so it isn't repeated
 * here.)
 */

import { BoxGeometry, Mesh, MeshLambertMaterial, MeshStandardMaterial, NodeMaterial, PerspectiveCamera, PointLight, QuadMesh, RenderTarget, Scene } from 'three/webgpu';
import { mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import { assert } from '../../client.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi/src/ddgi-technique.ts';
import { ThreeGI } from '../../../../packages/three/src/three-gi.ts';
import { ddgi, giAlbedo } from '../../../../packages/three/src/composite.ts';
import { advanceThreeFrame } from '../three-composite-shared/composite-check.ts';
import { runThreePage } from '../three-shared/harness.ts';

/** Names only composited builds and the composite's own shader use. */
const COMPOSITE_SYMBOLS = ['giThreeCoverage', 'ThreeGiCompositeWon', 'threeGiCompositeClaimed', 'threeGiSingleScattering'];

await runThreePage(async ({ device, renderer, data, log }) => {
    const scene = new Scene();
    const standard = new Mesh(new BoxGeometry(), new MeshStandardMaterial({ color: 0x808080, roughness: 0.6 }));
    const lambert = new Mesh(new BoxGeometry(), new MeshLambertMaterial({ color: 0x808080 }));
    lambert.position.x = 2;
    scene.add(standard, lambert, new PointLight(0xffffff, 10));
    const camera = new PerspectiveCamera(50, 1, 0.1, 100);
    camera.position.set(0, 0, 5);
    const platform = new GIPlatformImpl(device, { label: 'unmarked wgsl' });
    platform.setOutput(platform.addTechnique(new DDGITechnique({ volume: { center: [0, 0, 0], size: [4, 4, 4], spacing: 1 }, warmUpPasses: 0 })));
    await platform.prepare();
    const threeGI = new ThreeGI(platform, scene, { budget: { units: 0 } });
    threeGI.update(camera);

    const unmarked = async () => {
        const shaders: Record<string, string> = {};
        for (const [name, mesh] of [['standard', standard], ['lambert', lambert]] as const) shaders[name] = (await renderer.debug.getShaderAsync(scene, camera, mesh)).fragmentShader ?? '';
        return shaders;
    };
    const before = await unmarked();

    // A composite pipeline over the same scene, built and drawn: the marked builds and the composite's own shader.
    const scenePass = pass(scene, camera);
    scenePass.setMRT(mrt({ output, giAlbedo, normal: packNormalToRGB(normalView) }));
    const material = new NodeMaterial();
    material.fragmentNode = ddgi(scenePass, threeGI);
    const quad = new QuadMesh(material);
    const target = new RenderTarget(64, 64);
    renderer.setRenderTarget(target);
    for (let frame = 0; frame < 3; frame++) {
        advanceThreeFrame(renderer);
        quad.render(renderer);
        await device.queue.onSubmittedWorkDone();
    }
    renderer.setRenderTarget(null);

    const after = await unmarked();
    for (const name of Object.keys(before)) {
        data[`${name}Length`] = before[name].length;
        assert(before[name] === after[name], `${name}: the unmarked WGSL changed once a composite existed`);
        for (const symbol of COMPOSITE_SYMBOLS) assert(!before[name].includes(symbol), `${name}: the unmarked WGSL holds the composite's ${symbol}`);
        assert(before[name].includes('giThreeLighting'), `${name}: the unmarked WGSL doesn't call the per-fragment GI entry`);
    }
    log(`unmarked Standard and Lambert WGSL identical before and after a composite (${String(data.standardLength)} and ${String(data.lambertLength)} characters), with no composite symbols`);
    target.dispose();
    threeGI.dispose();
    platform.dispose();
});
