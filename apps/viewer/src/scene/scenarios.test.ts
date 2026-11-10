import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { motionTarget, type SceneManifest } from '../../../../scenes/src/manifest';
import { findScenario, sceneAddsCabinets, SCENARIOS } from './scenarios';

const manifestDirectory = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..', 'scenes', 'manifests');
const manifests = new Map<string, SceneManifest>(
    readdirSync(manifestDirectory)
        .filter((file) => file.endsWith('.json'))
        .map((file) => {
            const manifest = JSON.parse(readFileSync(join(manifestDirectory, file), 'utf8')) as SceneManifest;
            return [manifest.id, manifest];
        }),
);

describe('the moving-geometry scenarios (T3.8)', () => {
    it("each names a scene, one of its camera views, and its motions; the step one jumps", () => {
        for (const scenario of SCENARIOS) {
            const manifest = manifests.get(scenario.scene);
            expect(manifest, scenario.id).toBeDefined();
            expect(manifest!.cameras.map((camera) => camera.name), scenario.id).toContain(scenario.camera);
            const motion = manifest!.motions?.find((candidate) => candidate.id === scenario.motion);
            expect(motion, `${scenario.id} ${scenario.motion}`).toBeDefined();
            if (scenario.stepMotion !== null) {
                const step = manifest!.motions?.find((candidate) => candidate.id === scenario.stepMotion);
                expect(step?.interpolation, `${scenario.id} ${scenario.stepMotion}`).toBe('step');
                // The meter can't measure a camera motion (it needs a still camera).
                expect(motionTarget(step!)?.kind).not.toBe('camera');
            }
        }
    });

    it('covers the design list: furniture, door, cabinets, wind tree and the fly-through', () => {
        expect(SCENARIOS.map((scenario) => scenario.id)).toEqual(['furniture', 'door', 'cabinets', 'wind-tree', 'fly-through']);
        // Only the fly-through moves the camera, so only it needs the camera-following probes.
        for (const scenario of SCENARIOS) {
            const motion = manifests.get(scenario.scene)!.motions!.find((candidate) => candidate.id === scenario.motion)!;
            expect(scenario.followCamera, scenario.id).toBe(motionTarget(motion)?.kind === 'camera');
        }
        expect(findScenario('door')?.scene).toBe('closed-door');
        expect(findScenario('nothing')).toBeNull();
        expect(sceneAddsCabinets('cabinet-row')).toBe(true);
        expect(sceneAddsCabinets('two-room')).toBe(false);
    });
});
