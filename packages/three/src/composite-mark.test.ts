import { MeshBasicMaterial, MeshLambertMaterial, MeshPhongMaterial, MeshPhysicalMaterial, MeshStandardMaterial, MeshToonMaterial, PerspectiveCamera, Scene, type Material } from 'three';
import { MeshBasicNodeMaterial, MeshStandardNodeMaterial } from 'three/webgpu';
import { builtinAOContext, context, float, pass } from 'three/tsl';
import { describe, expect, it } from 'vitest';
import { GI_COMPOSITE_MARK, compositeTakesGI, compositedModel, flowContextData, hasCompositeMark, markCompositePass, type MarkablePass } from './composite-mark.ts';

function scenePass() {
    return pass(new Scene(), new PerspectiveCamera());
}

/** The context a material build sees in the marked pass (PassNode merges the pass's context into the renderer's). */
function builderFor(material: Material, contextData: object) {
    return { material, context: contextData } as unknown as Parameters<typeof compositeTakesGI>[0];
}

describe('the composite mark', () => {
    it('marks a pass that had no context', () => {
        const target = scenePass();
        markCompositePass(target as unknown as MarkablePass);
        expect(target.contextNode).not.toBeNull();
        expect(hasCompositeMark(flowContextData(target.contextNode!))).toBe(true);
    });

    it('keeps an existing context: GTAO-style builtinAOContext still reaches the materials', () => {
        const target = scenePass();
        const aoContext = builtinAOContext(float(0.5));
        target.contextNode = aoContext;
        markCompositePass(target as unknown as MarkablePass);
        const data = flowContextData(target.contextNode!);
        expect(hasCompositeMark(data)).toBe(true);
        expect(typeof data.getAO).toBe('function');
        // The AO context's own function, not a copy that could drift.
        expect(data.getAO).toBe(flowContextData(aoContext).getAO);
    });

    it('keeps every key of a user context, and marking twice changes nothing', () => {
        const target = scenePass();
        const getUV = () => null;
        target.contextNode = context({ getUV, custom: 7 });
        markCompositePass(target as unknown as MarkablePass);
        const once = target.contextNode;
        markCompositePass(target as unknown as MarkablePass);
        expect(target.contextNode).toBe(once);
        const data = flowContextData(target.contextNode!);
        expect(data.getUV).toBe(getUV);
        expect(data.custom).toBe(7);
        expect(data[GI_COMPOSITE_MARK]).toBe(true);
    });

    it('bumps the pass version, so PassNode rebuilds its cached merged context', () => {
        const target = scenePass();
        const before = target.version;
        markCompositePass(target as unknown as MarkablePass);
        expect(target.version).toBeGreaterThan(before);
    });
});

describe('which surfaces the composite lights', () => {
    const marked = { [GI_COMPOSITE_MARK]: true };

    it('lights opaque Standard, Physical, Lambert and Phong surfaces in a marked pass', () => {
        expect(compositeTakesGI(builderFor(new MeshStandardMaterial(), marked))).toBe(true);
        expect(compositeTakesGI(builderFor(new MeshStandardNodeMaterial(), marked))).toBe(true);
        expect(compositeTakesGI(builderFor(new MeshPhysicalMaterial(), marked))).toBe(true);
        expect(compositeTakesGI(builderFor(new MeshLambertMaterial(), marked))).toBe(true);
        expect(compositeTakesGI(builderFor(new MeshPhongMaterial(), marked))).toBe(true);
        expect(compositedModel(new MeshPhysicalMaterial())).toBe('physical');
        expect(compositedModel(new MeshLambertMaterial())).toBe('lambert');
    });

    it('leaves every surface per fragment in an unmarked pass (the path before T1.14)', () => {
        for (const material of [new MeshStandardMaterial(), new MeshLambertMaterial(), new MeshPhysicalMaterial()]) {
            expect(compositeTakesGI(builderFor(material, {}))).toBe(false);
            // Another pass's context (GTAO's) is not the mark.
            expect(compositeTakesGI(builderFor(material, flowContextData(builtinAOContext(float(1)))))).toBe(false);
        }
    });

    it('leaves transparent surfaces per fragment', () => {
        expect(compositeTakesGI(builderFor(new MeshStandardMaterial({ transparent: true, opacity: 0.5 }), marked))).toBe(false);
    });

    it('leaves unlit and unsupported models out', () => {
        expect(compositeTakesGI(builderFor(new MeshBasicMaterial(), marked))).toBe(false);
        expect(compositeTakesGI(builderFor(new MeshBasicNodeMaterial(), marked))).toBe(false);
        expect(compositeTakesGI(builderFor(new MeshToonMaterial(), marked))).toBe(false);
        // Physical features that change the indirect-diffuse weight in ways giAlbedo doesn't reproduce.
        expect(compositedModel(new MeshPhysicalMaterial({ clearcoat: 1 }))).toBeNull();
        expect(compositedModel(new MeshPhysicalMaterial({ sheen: 1 }))).toBeNull();
        expect(compositedModel(new MeshPhysicalMaterial({ iridescence: 1 }))).toBeNull();
        expect(compositedModel(new MeshPhysicalMaterial({ transmission: 1 }))).toBeNull();
    });
});
