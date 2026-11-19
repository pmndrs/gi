import * as THREE from 'three/webgpu';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { createNativeAdapter } from '../../../../examples/11-radiance-cascades/native-adapter.ts';
import { RealtimeRadianceCascades } from '../../../../examples/11-radiance-cascades/RealtimeRadianceCascades.js';
import { assert, drainDevice, reportGpuTest, summarizeAdapter, watchDevice } from '../../client.ts';
import type { AdapterSummary } from '../../contract.ts';

const width = 640, height = 360, frames = 128;
const sceneScale = 0.0038; // Fits the entire model inside the bounded native gallery volume.
const inputBase = '/rc-chandra-input';
const expectedObj = 'ec7b65b793692db211e3011d71f8b02f3d3b57ec5c7da90e5805360ce44f452b';

async function loadScene() {
    const response = await fetch(inputBase + '/sponza.obj');
    const obj = (await response.text()).replaceAll('\r\n', '\n');
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(obj));
    const hash = Array.from(new Uint8Array(digest), n => n.toString(16).padStart(2, '0')).join('');
    assert(hash === expectedObj, 'Sponza differs from the Chandra 9af00e8 asset');
    const source = new OBJLoader().parse(obj);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0);
    const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(0.7, 0.7, 0.7), roughness: 0.15,
        side: THREE.DoubleSide, flatShading: true
    });
    const red = material.clone(), cyan = material.clone();
    // Chandra get_emission multiplies its custom emitter radiance by the hit diffuse.
    red.emissive.setRGB(7, 0, 0);
    cyan.emissive.setRGB(0, 4.9, 4.9);
    let groups = 0, triangles = 0;
    source.traverse(object => {
        if (!(object instanceof THREE.Mesh)) return;
        if (groups === 1) assert(object.name.includes('sponza_328'), 'red emitter group changed');
        if (groups === 9) assert(object.name.includes('sponza_322'), 'cyan emitter group changed');
        object.material = groups === 1 ? red : groups === 9 ? cyan : material;
        object.geometry.computeVertexNormals(); // OBJLoader emits deindexed triangles.
        triangles += object.geometry.getAttribute('position').count / 3;
        groups++;
    });
    assert(triangles === 262267, 'Sponza triangle count differs from upstream');
    // Chandra permutes OBJ (x,y,z) to (z,x,y). Then turn its z-up world into Three's y-up.
    source.rotation.y = Math.PI / 2;
    source.scale.setScalar(sceneScale);
    scene.add(source);
    const camera = new THREE.PerspectiveCamera(THREE.MathUtils.radToDeg(1.15), width / height, 0.001, 40);
    camera.position.set(-94.45221, 613.4341, 937.2442).multiplyScalar(sceneScale);
    const yaw = 6.437008, pitch = 0.47492924;
    const direction = new THREE.Vector3(Math.sin(yaw) * Math.cos(pitch), -Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch));
    camera.lookAt(direction.add(camera.position));
    return { scene, camera, groups, triangles };
}

function stats(image: Float32Array) {
    const sums = [0, 0, 0];
    for (let i = 0; i < image.length; i++) {
        assert(Number.isFinite(image[i]), 'non-finite linear pixel');
        sums[i % 3] += image[i];
    }
    return sums.map(v => v / (width * height));
}

function compare(a: Float32Array, b: Float32Array) {
    assert(a.length === b.length, 'linear comparison size mismatch');
    let absolute = 0, squared = 0, signal = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        absolute += Math.abs(d);
        squared += d * d;
        signal += b[i] * b[i];
    }
    return {
        meanAbsoluteError: absolute / a.length, rmse: Math.sqrt(squared / a.length),
        relativeRms: signal > 0 ? Math.sqrt(squared / signal) : null
    };
}

function show(label: string, image: Float32Array) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d')!;
    const pixels = ctx.createImageData(width, height);
    for (let i = 0; i < width * height; i++) {
        for (let c = 0; c < 3; c++) {
            const value = Math.max(0, image[3 * i + c]);
            const mapped = value / (1 + value);
            pixels.data[4 * i + c] = 255 * (mapped <= 0.0031308 ? 12.92 * mapped : 1.055 * mapped ** (1 / 2.4) - 0.055);
        }
        pixels.data[4 * i + 3] = 255;
    }
    ctx.putImageData(pixels, 0, 0);
    const figure = document.createElement('figure'), caption = document.createElement('figcaption');
    caption.textContent = label;
    figure.append(canvas, caption);
    document.querySelector('main')!.append(figure);
}

const started = performance.now();
let adapter: AdapterSummary | null = null;
let deviceErrors: string[] = [];
const data: Record<string, unknown> = {};
try {
    const inputs: string[] = await (await fetch(inputBase + '/inputs.json')).json();
    assert(inputs.includes('/sponza.obj'), 'Set GI_CHANDRA_ROOT to the pinned Chandra source checkout');
    assert(['/diffuse.f32', '/specular.f32', '/primary.u8'].every(file => inputs.includes(file)),
        'Run tools/reference/chandra/run.ps1 to capture the original diffuse, specular and primary images');
    const gpu = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    assert(gpu && gpu.limits.maxStorageBuffersPerShaderStage >= 9, 'S5 requires at least 9 storage buffers per shader stage');
    adapter = summarizeAdapter(gpu);
    const renderer = new THREE.WebGPURenderer({
        powerPreference: 'high-performance', requiredLimits: {
            maxStorageBuffersPerShaderStage: gpu.limits.maxStorageBuffersPerShaderStage,
            maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize, maxBufferSize: gpu.limits.maxBufferSize,
        }
    });
    await renderer.init();
    renderer.setSize(width, height, false);
    const { scene, camera, groups, triangles } = await loadScene();
    const device = (renderer.backend as unknown as { device: GPUDevice }).device;
    deviceErrors = watchDevice(device);
    const target = new THREE.RenderTarget(width, height, { type: THREE.HalfFloatType, depthBuffer: false });
    const mat = new THREE.NodeMaterial();
    mat.toneMapped = false;
    const quad = new THREE.QuadMesh(mat);
    async function read(node: THREE.Node) {
        mat.fragmentNode = node;
        mat.needsUpdate = true;
        renderer.setRenderTarget(target);
        quad.render(renderer);
        renderer.setRenderTarget(null);
        const raw = await renderer.readRenderTargetPixelsAsync(target, 0, 0, width, height);
        const half = new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2);
        const rgb = new Float32Array(width * height * 3);
        for (let pixel = 0; pixel < width * height; pixel++) {
            for (let channel = 0; channel < 3; channel++) {
                rgb[pixel * 3 + channel] = THREE.DataUtils.fromHalfFloat(half[4 * pixel + channel]);
            }
        }
        return rgb;
    }
    async function render(arm: 'three' | 'webgpu') {
        const reference = arm === 'three' ? new RealtimeRadianceCascades({
            spacing: 0.10, levelDistance: 1.5, levels: 10,
            maxProbes: 4096, blendingRadius: 2, orientations: false, rayStride: 2, secondary: false, qualityDecay: 0.94,
            quality: 1, adaptiveQuality: false, velocity: false
        }) : null;
        const glossy = reference ? null : await createNativeAdapter(renderer, scene, 2, [], 'glossy');
        const rc = reference ?? glossy!;
        try {
            data.stage = arm;
            reference?.setScene(renderer, scene);
            rc.mode.value = glossy ? 3 : 1; // Diffuse reflected light in both adapters.
            for (let i = 0; i < frames; i++) {
                // A directional query reads the completed cache and does not feed the solver.
                if (glossy) glossy.specularEnabled.value = Number(i === frames - 1);
                rc.update(renderer, scene, camera);
                if ((i + 1) % 8 === 0) await device.queue.onSubmittedWorkDone();
            }
            const indirect = await read(rc.getCompositeNode());
            const specularNode = glossy?.getPrefilteredSpecularNode();
            const reflected = specularNode ? await read(specularNode) : null;
            rc.mode.value = 0;
            rc.intensity.value = 0;
            const direct = await read(rc.getCompositeNode());
            assert(stats(indirect).some(v => v > 0.001), arm + ' returned dark indirect light');
            return { indirect, direct, reflected };
        } finally {
            rc.dispose();
        }
    }
    const native = await render('webgpu');
    const three = await render('three');
    const directDifference = compare(native.direct, three.direct);
    data.directDifference = directDifference;
    assert(directDifference.meanAbsoluteError < 0.001, 'paired WebGPU primary emission disagrees');
    show('GI WebGPU · diffuse', native.indirect);
    show('Three.js S5 · diffuse', three.indirect);
    Object.assign(data, {
        width, height, frames, groups, triangles, objSha256: expectedObj,
        nativeMean: stats(native.indirect), threeMean: stats(three.indirect), nativeVsThree: compare(native.indirect, three.indirect), directDifference
    });
    const original = new Float32Array(await (await fetch(inputBase + '/diffuse.f32')).arrayBuffer());
    assert(original.length === width * height * 3, 'Chandra capture must be top-down 640×360 RGB float32');
    const mean = stats(original);
    assert(mean.some(v => v > 0.001), 'no upstream diffuse signal');
    data.chandraMean = mean;
    data.nativeVsChandra = compare(native.indirect, original);
    data.threeVsChandra = compare(three.indirect, original);
    show('Chandra · upstream', original);
    const reflected = new Float32Array(await (await fetch(inputBase + '/specular.f32')).arrayBuffer());
    assert(reflected.length === width * height * 3, 'Chandra specular capture size mismatch');
    data.chandraReflectedRadianceMean = stats(reflected);
    assert(native.reflected, 'Missing prefiltered WebGPU reflection');
    data.nativeReflectedRadianceMean = stats(native.reflected);
    data.glossyVsChandraDirection = compare(native.reflected, reflected);
    show('WebGPU · roughness 0.15, before Fresnel', native.reflected);
    show('Chandra · original reflected direction', reflected);
    const primary = new Uint8Array(await (await fetch(inputBase + '/primary.u8')).arrayBuffer());
    assert(primary.length === width * height, 'Chandra primary mask size mismatch');
    let matched = 0, union = 0;
    for (let i = 0; i < primary.length; i++) {
        const nativeId = native.direct[3 * i] > 1 ? 2 : native.direct[3 * i + 1] > 1 ? 3 : 0;
        if (nativeId || primary[i] >= 2) {
            union++;
            if (nativeId === primary[i]) matched++;
        }
    }
    data.emitterMaskAgreement = matched / Math.max(union, 1);
    assert(union > 0 && Number(data.emitterMaskAgreement) > 0.95,
        'Upstream and raster camera emitter agreement: ' + matched + '/' + union);
    await drainDevice(device);
    // Disposing Three deliberately destroys its GPUDevice. Freeze observed run-time errors first.
    const observedErrors = [...deviceErrors];
    delete data.stage;
    mat.dispose();
    target.dispose();
    renderer.dispose();
    reportGpuTest({ status: 'pass', adapter, startTime: started, data, deviceErrors: observedErrors });
} catch (error) {
    reportGpuTest({ status: 'fail', adapter, startTime: started, data, deviceErrors, message: error instanceof Error ? error.message : String(error) });
}
