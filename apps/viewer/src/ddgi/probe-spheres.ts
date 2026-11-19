/**
 * DDGI's probes as small spheres, coloured by state: the technique's
 * `points` debug view drawn in three.js.
 *
 * The data stays on the GPU. DDGI writes one texel per probe into an
 * `rgba32float` texture (xyz: where the probe really sits after relocation;
 * w: its state), and every frame's update keeps it current. One instanced
 * sphere mesh reads it in the vertex shader: instance i loads texel
 * (i mod width, i div width), moves the sphere there and passes the state's
 * colour on to the fragment. Nothing is read back, and the spheres follow
 * relocation as it happens.
 *
 * The colours are DDGI's own debug colours (`DDGI_DEBUG_COLORS`), the ones
 * its "probes near this point" view uses, so the two views agree.
 *
 * Or, coloured by history (`colorBy = 'history'`): how many updates each
 * active probe's light is averaged over (adaptive hysteresis' n), red for
 * few, green for the full count (`history-colors.ts`); probes that aren't
 * active are grey. The counts come from `ProbeHistoryTexture`, one texel per
 * probe in the same order as the points.
 *
 * Or by the probe's own light (`colorBy = 'irradiance'`; TU.5, DV-03 in
 * docs/divergence.md), RTXGI's standard probe view: each point of a sphere
 * shows the irradiance its probe holds for a surface facing that way (the
 * sphere's normal), from the probe's own tile alone, as a white matte surface
 * would show it (E / pi), and a thin rim in the probe's state colour, as
 * RTXGI's view colours the rim by state. The light comes from the technique's
 * shading module (the function the points view names in `irradianceFunction`),
 * so it is the field the picture is lit by.
 */

import { Mesh, NoBlending, SphereGeometry } from 'three';
import { NodeMaterial, type Node } from 'three/webgpu';
import { clamp, float, instanceIndex, int, ivec2, log, mix, normalView, normalWorld, positionLocal, positionView, select, textureLoad, uniform, varying, vec3, vec4, wgsl, wgslFn } from 'three/tsl';
import { DDGI_DEBUG_COLORS, GI_USER_DATA, PROBE_STATE, wgslParameterType, wrapGpuTexture, type DebugView, type ShadingModule } from './library';
import { LAYER_DDGI_PROBES } from '../scene/layers';
import { HISTORY_COLORS } from './history-colors';
import { ShadingNodes } from './shading-nodes';

type PointsView = Extract<DebugView, { kind: 'points' }>;

/** What the spheres' colour shows: the probe's state, how settled its history is, or its own light. */
export type ProbeColorBy = 'state' | 'history' | 'irradiance';

/** The colour modes as the material's uniform holds them. */
const COLOR_MODE: Readonly<Record<ProbeColorBy, number>> = { state: 0, history: 1, irradiance: 2 };

/** Where each probe's history count is: a texture laid out like the points texture (`ProbeHistoryTexture`). */
export interface ProbeHistorySource {
    texture(): GPUTexture;
    /** The full count (`hysteresisMaxSamples`): green. */
    maxSamples(): number;
}

/** Linear RGB per probe state, from DDGI's own debug colours. */
export const PROBE_STATE_COLORS: Readonly<Record<number, readonly [number, number, number]>> = {
    [PROBE_STATE.active]: DDGI_DEBUG_COLORS.stateActive,
    [PROBE_STATE.offInsideGeometry]: DDGI_DEBUG_COLORS.stateInsideGeometry,
    [PROBE_STATE.offNoSurfaceNearby]: DDGI_DEBUG_COLORS.stateNoSurfaceNearby,
    [PROBE_STATE.notUpdatedYet]: DDGI_DEBUG_COLORS.stateNotUpdatedYet,
};

/**
 * Sphere radius as a share of the smallest probe spacing: big enough to see
 * from across a room, small enough that neighbours never touch.
 */
const RADIUS_SHARE_OF_SPACING = 0.08;

/** Segments of the sphere; the spheres are small on screen, so a coarse one looks round. */
const SPHERE_WIDTH_SEGMENTS = 12;
const SPHERE_HEIGHT_SEGMENTS = 8;

/**
 * Debug colours are meant to be read, not lit, but a flat disc hides that a
 * sphere is a sphere. A little shading from the view direction: the rim is
 * this much darker than the middle.
 */
const RIM_DARKENING = 0.45;

/**
 * Coloured by the probe's own light: where the sphere turns away from the
 * view this steeply (|normal . view| under this), the rim shows the probe's
 * state instead, as RTXGI's probe view does (0.45, `ProbesRGS.hlsl`).
 */
const IRRADIANCE_STATE_RIM = 0.45;

/** The WGSL function the spheres call, wrapping the shading module's per-probe irradiance. */
const IRRADIANCE_ENTRY_NAME = 'viewerProbeIrradiance';

/**
 * A wrapper three's `wgslFn` can call: the point index arrives as a float
 * (an instance's index, passed to the fragment as a varying) and becomes the
 * module's `u32`. The resources are the module's parameters, matched by name
 * (the same construction as debug-material.ts).
 */
function irradianceEntryWgsl(shading: ShadingModule, functionName: string): string {
    const declarations = shading.parameters.map((parameter) => `    ${parameter.name}: ${wgslParameterType(parameter)},`).join('\n');
    const resources = shading.parameters.map((parameter) => parameter.name).join(', ');
    return /* wgsl */ `fn ${IRRADIANCE_ENTRY_NAME}(
    point: f32,
    direction: vec3f,
${declarations}
) -> vec3f {
    // Point indices are whole numbers; + 0.5 so the conversion can't round down past one.
    return ${functionName}(u32(point + 0.5), direction, ${resources});
}`;
}

function colorNode(color: readonly [number, number, number]): Node<'vec3'> {
    return vec3(color[0], color[1], color[2]) as unknown as Node<'vec3'>;
}

export class ProbeSpheres {
    readonly mesh: Mesh;
    private readonly material: NodeMaterial;
    private readonly geometry = new SphereGeometry(1, SPHERE_WIDTH_SEGMENTS, SPHERE_HEIGHT_SEGMENTS);
    private readonly textureWidth = uniform(1, 'int');
    private gpuTexture: GPUTexture;
    private readonly pointsNode: ReturnType<typeof textureLoad>;
    private historyTexture: GPUTexture;
    private readonly historyNode: ReturnType<typeof textureLoad>;
    /** COLOR_MODE: by state, by history or by the probe's light. A uniform, so switching costs nothing. */
    private readonly colorMode = uniform(COLOR_MODE.state);
    private readonly maxSamples = uniform(1);
    /** The shading module's resources, for the probe's own light; null when the view names no irradiance function. */
    private readonly shadingNodes: ShadingNodes | null;

    /**
     * `shading`: the technique's shading module, whose function
     * `view.irradianceFunction` gives each probe's own light. Without either,
     * `irradiance` colours by state.
     */
    constructor(
        private readonly view: PointsView,
        smallestSpacing: number,
        private readonly history: ProbeHistorySource,
        shading: ShadingModule | null = null,
    ) {
        this.gpuTexture = view.texture();
        this.textureWidth.value = this.gpuTexture.width;

        // The probe of this instance: texel (i mod width, i div width).
        const index = int(instanceIndex);
        const texel = ivec2(index.mod(this.textureWidth), index.div(this.textureWidth));
        this.pointsNode = textureLoad(wrapGpuTexture(this.gpuTexture, { dimension: '2d', sampleType: 'unfilterable-float' }), texel);
        const point = this.pointsNode as unknown as Node<'vec4'>;
        const radius = float(RADIUS_SHARE_OF_SPACING * smallestSpacing);

        // The state's colour, chosen per vertex (one state per instance) and passed on.
        const state = int(point.w.add(0.5));
        const byState = select(
            state.equal(PROBE_STATE.active),
            colorNode(PROBE_STATE_COLORS[PROBE_STATE.active]),
            select(
                state.equal(PROBE_STATE.offInsideGeometry),
                colorNode(PROBE_STATE_COLORS[PROBE_STATE.offInsideGeometry]),
                select(
                    state.equal(PROBE_STATE.offNoSurfaceNearby),
                    colorNode(PROBE_STATE_COLORS[PROBE_STATE.offNoSurfaceNearby]),
                    colorNode(PROBE_STATE_COLORS[PROBE_STATE.notUpdatedYet]),
                ),
            ),
        );

        // The history colour: the mirror of `historyColor` in history-colors.ts (log scale, red to yellow to green).
        this.historyTexture = history.texture();
        this.historyNode = textureLoad(wrapGpuTexture(this.historyTexture, { dimension: '2d', sampleType: 'unfilterable-float' }), texel);
        const samples = (this.historyNode as unknown as Node<'vec4'>).x;
        const settled = clamp(log(samples.max(0).add(1)).div(log(this.maxSamples.add(1))), 0, 1);
        const towardsHalfway = mix(colorNode(HISTORY_COLORS.fewUpdates), colorNode(HISTORY_COLORS.halfway), settled.mul(2));
        const towardsSettled = mix(colorNode(HISTORY_COLORS.halfway), colorNode(HISTORY_COLORS.settled), settled.mul(2).sub(1));
        const activeHistory = select(settled.lessThan(0.5), towardsHalfway, towardsSettled);
        const byHistory = select(state.equal(PROBE_STATE.active), activeHistory, colorNode(HISTORY_COLORS.notActive));
        const stateColor = select(this.colorMode.equal(COLOR_MODE.history), byHistory, byState);
        // Darker towards the rim, so a sphere reads as round. How squarely the surface faces the eye,
        // from the direction to the eye at this pixel (not the view axis, which would put the rim
        // off-centre on spheres near the picture's edges).
        const facing = normalView.dot(positionView.normalize()).abs();
        const shadedStateColor = varying(stateColor).mul(float(1 - RIM_DARKENING).add(facing.mul(RIM_DARKENING)));

        // The probe's own light (RTXGI's view): its irradiance for the sphere's normal, on a white matte
        // surface (E / pi), and its state's colour, unshaded, on the rim.
        const functionName = view.irradianceFunction;
        this.shadingNodes = shading && functionName ? new ShadingNodes(shading) : null;
        let color = shadedStateColor as unknown as Node<'vec3'>;
        if (shading && functionName && this.shadingNodes) {
            const entry = wgslFn(irradianceEntryWgsl(shading, functionName), [wgsl(shading.code)]);
            const point = varying(float(instanceIndex));
            const irradiance = entry({ point, direction: normalWorld, ...this.shadingNodes.byName }) as unknown as Node<'vec3'>;
            const ownLight = select(facing.lessThan(IRRADIANCE_STATE_RIM), varying(byState), irradiance.div(Math.PI));
            color = select(this.colorMode.equal(COLOR_MODE.irradiance), ownLight, shadedStateColor) as unknown as Node<'vec3'>;
        }

        const material = new NodeMaterial();
        material.name = 'ddgi probe spheres';
        material.lights = false;
        // The screen debug views replace every material through scene.overrideMaterial; the spheres keep theirs.
        material.allowOverride = false;
        material.positionNode = positionLocal.mul(radius).add(point.xyz);
        // Alpha 0 marks the spheres' pixels in DDGI's picture, so the change-per-frame view can leave
        // them in their own colours (frame-change.ts); everything else there is alpha 1. Nothing else
        // reads that alpha. NoBlending writes the alpha as given: with three's default (NormalBlending)
        // an opaque material's alpha is forced to 1 (NodeMaterial, builder.isOpaque()).
        material.blending = NoBlending;
        material.colorNode = vec4(color, 0);
        this.material = material;

        const mesh = new Mesh(this.geometry, material);
        mesh.name = 'ddgi probe spheres';
        mesh.count = view.count();
        // Instances are placed in the shader, so three's bounds (one sphere at the origin) say nothing.
        mesh.frustumCulled = false;
        mesh.castShadow = false;
        mesh.receiveShadow = false;
        mesh.layers.set(LAYER_DDGI_PROBES);
        // A picture of the probes, not part of the scene: GI must not see them.
        mesh.userData[GI_USER_DATA.contributes] = false;
        this.mesh = mesh;
    }

    /** Colour by state, by history or by the probe's own light. Free: it's a uniform. */
    set colorBy(choice: ProbeColorBy) {
        // Without the shading module's per-probe light, 'irradiance' falls back to the state colours.
        this.colorMode.value = choice === 'irradiance' && !this.shadingNodes ? COLOR_MODE.state : COLOR_MODE[choice];
    }

    /** Once per frame: follow a reallocated texture (a new `prepare`) and the probe count. */
    refresh(): void {
        const current = this.view.texture();
        if (current !== this.gpuTexture) {
            this.gpuTexture = current;
            this.pointsNode.value = wrapGpuTexture(current, { dimension: '2d', sampleType: 'unfilterable-float' });
            this.textureWidth.value = current.width;
        }
        const history = this.history.texture();
        if (history !== this.historyTexture) {
            this.historyTexture = history;
            this.historyNode.value = wrapGpuTexture(history, { dimension: '2d', sampleType: 'unfilterable-float' });
        }
        this.maxSamples.value = this.history.maxSamples();
        this.shadingNodes?.refresh();
        this.mesh.count = this.view.count();
    }

    dispose(): void {
        this.mesh.removeFromParent();
        this.geometry.dispose();
        this.material.dispose();
    }
}
