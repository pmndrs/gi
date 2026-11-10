/**
 * DDGI's debug outputs drawn in three.js, with the library's public API.
 * The core only hands out data (the technique's `debugViews` and its shading
 * module); drawing it is the engine's job, so this is the three.js side.
 * Everything stays on the GPU: three reads the same `GPUTexture`s DDGI
 * writes, wrapped with `wrapGpuTexture`, and nothing is read back.
 *
 * - `ShadingResources`: the shading module's parameters as three nodes.
 * - `ScreenDebugMaterial`: the `screen` debug views (GI only, coverage, the
 *   states of the probes near each point), drawn through the module's
 *   `giDebugColor` as an override material.
 * - `ProbeDots`: the `points` debug view, a dot at each probe where it really
 *   sits, coloured by its state.
 * - `IrradianceAtlas`: the irradiance atlas, one picture per row of probes.
 */

import * as THREE from 'three/webgpu';
import { float, instanceIndex, int, ivec2, normalView, positionLocal, positionWorld, normalWorld, select, texture, texture3D, textureLoad, uniform, uv, varying, vec2, vec3, vec4, wgsl, wgslFn } from 'three/tsl';
import { type DebugView, type ShadingModule } from '@pmndrs/gi/core';
import { DDGI_DEBUG_COLORS, DDGI_SHADING_PARAMETER_NAMES, PROBE_STATE, type DDGITechnique } from '@pmndrs/gi/ddgi';
import { GI_USER_DATA, wgslParameterType, wrapGpuTexture } from '@pmndrs/gi/three';

type Node<Type = unknown> = THREE.Node<Type>;

/** The shading module's textures, samplers and vec4 values as three nodes, by parameter name. */
export class ShadingResources {
    readonly byName: Record<string, Node> = {};
    private readonly uniforms: Array<{ vector: THREE.Vector4; value: () => Float32Array }> = [];

    constructor(readonly module: ShadingModule) {
        const textureNodes = new Map<string, Node>();
        for (const parameter of module.parameters) {
            if (parameter.kind !== 'texture') continue;
            // A texture's sampler (if it has one) says how three should filter it.
            const sampler = module.parameters.find((other) => other.kind === 'sampler' && other.texture === parameter.name);
            const sampling = sampler?.kind === 'sampler' ? { filtering: sampler.filtering, addressMode: sampler.addressMode } : {};
            const wrapped = wrapGpuTexture(parameter.texture(), { dimension: parameter.dimension, sampleType: parameter.sampleType, ...sampling });
            textureNodes.set(parameter.name, parameter.dimension === '3d' ? texture3D(wrapped) : texture(wrapped));
        }
        for (const parameter of module.parameters) {
            if (parameter.kind === 'vec4f') {
                const vector = new THREE.Vector4().fromArray(parameter.value());
                this.uniforms.push({ vector, value: parameter.value });
                this.byName[parameter.name] = uniform(vector);
            } else {
                // three's wgslFn feeds a sampler parameter from its texture's node.
                this.byName[parameter.name] = textureNodes.get(parameter.kind === 'sampler' ? parameter.texture : parameter.name)!;
            }
        }
    }

    /** Copies the module's current vec4 values. Once per frame. */
    refresh(): void {
        for (const { vector, value } of this.uniforms) vector.fromArray(value());
    }
}

/**
 * Every surface drawn in a screen debug view's colours instead of its own
 * material: set it as `scene.overrideMaterial`. Unlit, because a lit
 * material would build the GI light too, which includes the same WGSL, and
 * WGSL included twice is a redeclaration error.
 */
export class ScreenDebugMaterial {
    readonly material = new THREE.NodeMaterial();
    private readonly viewId = uniform(0);

    constructor(resources: ShadingResources) {
        const module = resources.module;
        // giDebugColor takes the view id as a u32; three hands uniforms over as floats.
        const declarations = module.parameters.map((parameter) => `    ${parameter.name}: ${wgslParameterType(parameter)},`).join('\n');
        const names = module.parameters.map((parameter) => parameter.name).join(', ');
        const entry = wgslFn(
            `fn exampleDebugColor(position: vec3f, normal: vec3f, view: f32,\n${declarations}\n) -> vec3f {\n` +
                `    return giDebugColor(position, normal, u32(view + 0.5), ${names});\n}`,
            [wgsl(module.code)],
        );
        const color = entry({ position: positionWorld, normal: normalWorld, view: this.viewId, ...resources.byName }) as unknown as Node<'vec3'>;
        this.material.lights = false;
        this.material.side = THREE.DoubleSide;
        this.material.colorNode = vec4(color, 1);
    }

    /** Which screen view to draw: an id from `DDGI_DEBUG_VIEW`. */
    set view(id: number) {
        this.viewId.value = id;
    }
}

/** DDGI's own colour for each probe state, as a TSL colour. */
function stateColor(state: Node<'int'>): Node<'vec3'> {
    const color = (rgb: readonly [number, number, number]) => vec3(...rgb);
    return select(
        state.equal(PROBE_STATE.active),
        color(DDGI_DEBUG_COLORS.stateActive),
        select(
            state.equal(PROBE_STATE.offInsideGeometry),
            color(DDGI_DEBUG_COLORS.stateInsideGeometry),
            select(state.equal(PROBE_STATE.offNoSurfaceNearby), color(DDGI_DEBUG_COLORS.stateNoSurfaceNearby), color(DDGI_DEBUG_COLORS.stateNotUpdatedYet)),
        ),
    );
}

/**
 * A small sphere at each probe. The `points` view is a texture with one texel
 * per probe (xyz where it sits after relocation, w its state), so one
 * instanced mesh places each sphere in its vertex shader.
 */
export class ProbeDots {
    readonly mesh: THREE.Mesh;

    constructor(view: Extract<DebugView, { kind: 'points' }>, radius: number) {
        const points = view.texture();
        const index = int(instanceIndex);
        const texel = ivec2(index.mod(points.width), index.div(points.width));
        const point = textureLoad(wrapGpuTexture(points, { dimension: '2d', sampleType: 'unfilterable-float' }), texel);
        const material = new THREE.NodeMaterial();
        material.lights = false;
        // The screen debug views replace every material; the dots keep theirs.
        material.allowOverride = false;
        material.positionNode = positionLocal.mul(radius).add(point.xyz);
        // A little darker towards the rim, so the dots read as spheres.
        const color = varying(stateColor(int(point.w.add(0.5)))) as unknown as Node<'vec3'>;
        material.colorNode = vec4(color.mul(float(0.55).add(normalView.z.abs().mul(0.45))), 1);
        this.mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), material);
        this.mesh.count = view.count();
        // Placed in the shader, so three's bounds say nothing about where they are.
        this.mesh.frustumCulled = false;
        // A picture of the probes, not part of the scene: GI's rays must not see them.
        this.mesh.userData[GI_USER_DATA.contributes] = false;
    }
}

/** Pixels between the atlas pictures, and between them and the window's edge. */
const ATLAS_GAP = 4;
const ATLAS_MARGIN = 12;

/**
 * The irradiance atlas: one array layer per row of probes (y), in which probe
 * (x, z) owns a small square of directions. Each layer is a picture at one
 * screen pixel per texel, laid out in the bottom-right corner, in a scene of
 * its own drawn over the picture with an orthographic camera in pixels.
 */
export class IrradianceAtlas {
    readonly scene = new THREE.Scene();
    readonly camera = new THREE.OrthographicCamera(0, 1, 1, 0, -1, 1);
    private readonly pictures: THREE.Mesh[] = [];
    private readonly layerSize: { width: number; height: number };

    constructor(technique: DDGITechnique, resources: ShadingResources) {
        const field = technique.field;
        if (!field) throw new Error('The technique has no field before prepare.');
        const atlas = field.layout.irradiance;
        this.layerSize = { width: atlas.width, height: atlas.height };
        const atlasNode = resources.byName[DDGI_SHADING_PARAMETER_NAMES.irradiance];
        for (let layer = 0; layer < atlas.layers; layer++) {
            const material = new THREE.MeshBasicNodeMaterial();
            // The atlas holds irradiance E; a white matte surface shows E / π.
            const texel = (atlasNode as ReturnType<typeof texture>).sample(vec2(uv().x, uv().y.oneMinus())).depth(int(layer)) as unknown as Node<'vec4'>;
            material.colorNode = vec4(texel.rgb.div(Math.PI), 1);
            const picture = new THREE.Mesh(new THREE.PlaneGeometry(atlas.width, atlas.height), material);
            this.pictures.push(picture);
            this.scene.add(picture);
        }
    }

    /** Places the pictures for the canvas's size (CSS pixels). */
    layout(width: number, height: number): void {
        this.camera.right = width;
        this.camera.top = height;
        this.camera.updateProjectionMatrix();
        const columns = Math.ceil(Math.sqrt(this.pictures.length));
        const rows = Math.ceil(this.pictures.length / columns);
        const right = width - ATLAS_MARGIN;
        const bottom = ATLAS_MARGIN + (rows - 1) * (this.layerSize.height + ATLAS_GAP);
        this.pictures.forEach((picture, layer) => {
            // The lowest row of probes first, top left.
            const column = layer % columns;
            const row = Math.floor(layer / columns);
            const x = right - (columns - column) * (this.layerSize.width + ATLAS_GAP) + this.layerSize.width / 2;
            const y = bottom - row * (this.layerSize.height + ATLAS_GAP) + this.layerSize.height / 2;
            picture.position.set(x, y, 0);
        });
    }
}
