import * as THREE from 'three/webgpu';
import { cameraPosition, diffuseColor, materialMetalness, materialRoughness, mix, mrt, normalWorld, output, positionWorld, screenUV, select, texture, uniform, vec3, vec4, velocity } from 'three/tsl';
import { NativeRadianceCascades, type NativeRcOptions } from '@pmndrs/gi/radiance-cascades';
import { createIntervalScene } from './interval-scene.ts';

export interface RawBackend {
    device: GPUDevice;
    get(texture: THREE.Texture): { texture: GPUTexture };
}

// Reconstruct the core's world-position input from the raster depth buffer,
// leaving room for motion vectors on devices with a 32-byte MRT limit.
const reconstruct = `
@group(0) @binding(0) var depth: texture_depth_2d;
@group(0) @binding(1) var position: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> worldFromClip: mat4x4f;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id:vec3u) {
    let size = textureDimensions(depth);
    if (any(id.xy >= size)) { return; }
    let p = vec2i(id.xy);
    let z = textureLoad(depth,p,0);
    let uv = (vec2f(id.xy)+0.5)/vec2f(size);
    let h = worldFromClip * vec4f(uv * vec2f(2,-2) + vec2f(-1,1), z, 1);
    textureStore(position,p,select(vec4f(0),vec4f(h.xyz/h.w,1),z < 1.0));
}`;

/** Solver settings a diagnostic page may change, one at a time, on top of a profile. */
export type NativeAdapterOverrides = Partial<Pick<NativeRcOptions, 'grid' | 'firstInterval' | 'maxDistance' | 'history' | 'normalBias'>>;

export async function createNativeAdapter(renderer: THREE.WebGPURenderer, scene: THREE.Scene, stride: 1 | 2, movingMeshes: THREE.Mesh[],
    profile: 'gallery' | 'door' | 'glossy', overrides: NativeAdapterOverrides = {}) {
    const backend = renderer.backend as unknown as RawBackend;
    const { device } = backend;
    const door = profile === 'door';
    const glossy = profile === 'glossy';
    const solver = await NativeRadianceCascades.create(device, {
        bounds: door ? { min: [-2.9,-0.4,-3.4], max: [4.5,3.6,3.4] } : { min: [-5,-0.4,-11.5], max: [5,5.7,11.5] },
        grid: door ? [20,12,20] : [20,12,46], firstInterval: 0.6, maxDistance: door ? 12 : 28, rayStride: stride, roughSpecular: glossy,
        ...overrides,
    });
    const geometry = await createIntervalScene(device, scene, { movingMeshes });
    const positionPass = await device.createComputePipelineAsync({
        label: 'RC gallery world position', layout: 'auto', compute: { module: device.createShaderModule({ code: reconstruct }) },
    });
    const matrix = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const clipInverse = new THREE.Matrix4();
    const packedMatrix = new Float32Array(16);
    const target = new THREE.RenderTarget(1, 1, { count: 4, type: THREE.HalfFloatType });
    for (const [i, name] of ['output', 'normal', 'albedo', 'velocity'].entries()) target.textures[i].name = name;
    target.textures[2].type = THREE.UnsignedByteType;
    target.textures[3].format = THREE.RGFormat;
    const depth = new THREE.DepthTexture(1,1,THREE.FloatType);
    target.depthTexture = depth;
    const motion = texture(target.textures[3]);
    const depthNode = texture(depth);
    const attachments = glossy ? mrt({
        output: vec4(output.rgb, normalWorld.dot(cameraPosition.sub(positionWorld).normalize()).saturate()),
        normal: vec4(normalWorld, materialRoughness), albedo: vec4(diffuseColor.rgb, materialMetalness), velocity,
    }) : mrt({ output, normal: vec4(normalWorld,1), albedo: vec4(diffuseColor.rgb,1), velocity });
    const irradiance = new THREE.StorageTexture();
    irradiance.type = THREE.HalfFloatType;
    irradiance.generateMipmaps = false;
    const mode = uniform(0);
    const intensity = uniform(1);
    const specularEnabled = uniform(1);
    const albedo = texture(target.textures[2], screenUV);
    const beauty = texture(target.texture, screenUV);
    const diffuseReflectance = glossy ? albedo.rgb.mul(albedo.a.oneMinus()) : albedo.rgb;
    const diffuse = texture(irradiance, screenUV).rgb.mul(diffuseReflectance, intensity, 1 / Math.PI);
    const reflected = glossy ? new THREE.StorageTexture() : null;
    const eye = glossy ? device.createBuffer({ label: 'RC reflection eye', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }) : null;
    const eyeData = new Float32Array([0,0,0,1]);
    const eyeWorld = new THREE.Vector3();
    let composite: THREE.Node = vec4(select(mode.equal(1), diffuse, beauty.rgb.add(diffuse)), 1);
    if (reflected) {
        reflected.type = THREE.HalfFloatType;
        reflected.generateMipmaps = false;
        const f0 = mix(vec3(0.04), albedo.rgb, albedo.a);
        const fresnel = mix(f0, vec3(1), beauty.a.oneMinus().pow(5));
        const specular = texture(reflected, screenUV).rgb.mul(fresnel, specularEnabled, intensity);
        const indirect = diffuse.add(specular);
        composite = vec4(select(mode.equal(2), specular, select(mode.equal(3), diffuse,
            select(mode.equal(1), indirect, beauty.rgb.add(indirect)))), 1);
    }
    const size = new THREE.Vector2();
    let positions: GPUTexture | undefined;
    let positionBindings: GPUBindGroup;
    let outputView: GPUTextureView;
    let normalView: GPUTextureView;
    let positionView: GPUTextureView;
    let reflectedView: GPUTextureView | undefined;
    return {
        velocity: false, renderSize: null as { x: number; y: number } | null,
        mode, intensity, specularEnabled,
        get memory() { return solver.probeBytes; },
        get hitScratchBytes() { return solver.rayScratchBytes + geometry.transport.scratchBytes; },
        getCompositeNode: () => composite,
        getPrefilteredSpecularNode: () => reflected ? texture(reflected, screenUV).rgb : null,
        getTextureNode: (name: 'depth' | 'velocity') => name === 'depth' ? depthNode : motion,
        update(r: THREE.WebGPURenderer, input: THREE.Scene, camera: THREE.PerspectiveCamera) {
            const { x: w, y: h } = this.renderSize ?? r.getDrawingBufferSize(size);
            const resized = !positions || target.width !== w || target.height !== h;
            if (resized) {
                target.setSize(w,h);
                irradiance.setSize(w,h,1);
                r.initTexture(irradiance);
                if (reflected) {
                    reflected.setSize(w,h,1);
                    r.initTexture(reflected);
                    reflectedView = backend.get(reflected).texture.createView({ baseMipLevel: 0, mipLevelCount: 1 });
                }
                solver.resize(w,h);
            }
            const previousTarget = r.getRenderTarget(), previousMRT = r.getMRT();
            try {
                r.setRenderTarget(target);
                r.setMRT(attachments);
                r.render(input,camera);
            } finally {
                r.setMRT(previousMRT);
                r.setRenderTarget(previousTarget);
            }
            if (resized) {
                positions?.destroy();
                positions = device.createTexture({ label: 'RC visible positions', size:[w,h], format:'rgba16float',
                    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
                positionView = positions.createView();
                normalView = backend.get(target.textures[1]).texture.createView();
                outputView = backend.get(irradiance).texture.createView({ baseMipLevel:0, mipLevelCount:1 });
                positionBindings = device.createBindGroup({ layout:positionPass.getBindGroupLayout(0), entries:[
                    { binding:0, resource:backend.get(depth).texture.createView() },
                    { binding:1, resource:positionView }, { binding:2, resource:{buffer:matrix} },
                ] });
            }
            clipInverse.multiplyMatrices(camera.matrixWorld,camera.projectionMatrixInverse).toArray(packedMatrix);
            device.queue.writeBuffer(matrix,0,packedMatrix);
            const encoder = device.createCommandEncoder({ label: 'RC gallery frame' });
            const pass = encoder.beginComputePass();
            pass.setPipeline(positionPass); pass.setBindGroup(0,positionBindings);
            pass.dispatchWorkgroups(Math.ceil(w/8),Math.ceil(h/8)); pass.end();
            geometry.sync();
            const gather = eye && specularEnabled.value !== 0 && reflectedView ? { view: eye, outputView: reflectedView } : undefined;
            if (gather) {
                camera.getWorldPosition(eyeWorld).toArray(eyeData);
                device.queue.writeBuffer(gather.view, 0, eyeData);
            }
            solver.encode(encoder,{worldPosition:positionView,worldNormal:normalView,specular:gather},geometry.transport,outputView);
            device.queue.submit([encoder.finish()]);
        },
        dispose() {
            solver.dispose(); geometry.dispose(); positions?.destroy(); matrix.destroy(); eye?.destroy(); reflected?.dispose(); irradiance.dispose(); target.dispose();
        },
    };
}
