/**
 * Not a gate: what the warm-up does under the `low` preset on a weak GPU. A procedural scene (two-room by default, its volume refined by VITE_WARMUP_SPACING_DIVISOR; the harness builds procedural scenes only, so sponza and Bistro are empty here), the preset's rays and tiles, the
 * technique's own warm-up, the host's budget as `low` gives it (1 ms). Each frame
 * is waited on, so a frame's wall time is its GPU time plus a little. A device
 * loss is reported, not thrown away. `VITE_WARMUP_MAX` sets `maxProbesPerFrame`
 * (the default is 4,096), `VITE_DDGI_PRESET` the preset (default low).
 */

import { assert, runGpuTest } from '../../client.ts';
import { ddgiPreset, isDDGIPresetName } from '../../../../packages/ddgi-core/src/techniques/ddgi/presets.ts';
import { createDdgiScene, frameInput, halfToFloat, loadManifest, readTextureBytes, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;

await runGpuTest(async ({ device, log: logLine }) => {
    releaseDeviceOnPageHide(device);
    const log = (line: string): void => {
        logLine(line);
        console.log(line);
    };
    let lost: string | null = null;
    void device.lost.then((info) => {
        lost = `${info.reason}: ${info.message}`;
    });
    const presetName = typeof env.VITE_DDGI_PRESET === 'string' && isDDGIPresetName(env.VITE_DDGI_PRESET) ? env.VITE_DDGI_PRESET : 'low';
    const preset = ddgiPreset(presetName);
    const sceneId = typeof env.VITE_WARMUP_SCENE === 'string' ? env.VITE_WARMUP_SCENE : 'two-room';
    const divisor = Number(env.VITE_WARMUP_SPACING_DIVISOR ?? 1);
    const base = loadManifest(sceneId).volume;
    const maxProbes = Number(env.VITE_WARMUP_MAX ?? 4096);
    const scene = await createDdgiScene(device, sceneId, { ...preset.ddgi, volume: { center: base.center, size: base.size, spacing: base.probeSpacing / divisor }, warmUpPasses: 16, maxProbesPerFrame: maxProbes }, { gpuTiming: true, textureLayerSize: preset.textureLayerSize });
    const probes = scene.ddgi.field?.grid.probeCount ?? 0;
    const frames: { ms: number; updated: number; passes: number }[] = [];
    let guard = 0;
    while (scene.ddgi.warmingUp && lost === null && guard++ < 2000) {
        const start = performance.now();
        const encoder = device.createCommandEncoder();
        scene.platform.encode(encoder, { ...frameInput({ gpuMs: preset.budget.gpuMs }), deltaTime: 1 / 60 });
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        frames.push({ ms: performance.now() - start, updated: scene.ddgi.counters['probes updated'] ?? 0, passes: scene.ddgi.counters['warm-up passes'] ?? 0 });
    }
    let atlasMean = -1;
    if (lost === null) {
        const halves = new Uint16Array((await readTextureBytes(device, scene.ddgi.field!.irradiance, 8)).buffer);
        let sum = 0;
        for (let texel = 0; texel < halves.length; texel += 4) sum += halfToFloat(halves[texel]) + halfToFloat(halves[texel + 1]) + halfToFloat(halves[texel + 2]);
        atlasMean = sum / ((halves.length / 4) * 3);
    }
    const sorted = frames.map((frame) => frame.ms).sort((a, b) => a - b);
    const at = (q: number): number => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] ?? 0;
    const summary = {
        scene: sceneId,
        spacingDivisor: divisor,
        atlasMean,
        preset: presetName,
        maxProbesPerFrame: maxProbes,
        probes,
        warmFrames: frames.length,
        totalMs: frames.reduce((sum, frame) => sum + frame.ms, 0),
        medianMs: at(0.5),
        p90Ms: at(0.9),
        maxMs: sorted[sorted.length - 1] ?? 0,
        probesUpdatedPerFrameMax: Math.max(0, ...frames.map((frame) => frame.updated)),
        finalPasses: frames[frames.length - 1]?.passes ?? 0,
        stillWarming: scene.ddgi.warmingUp,
        deviceLost: lost,
        first10Ms: frames.slice(0, 10).map((frame) => Math.round(frame.ms * 10) / 10),
    };
    log(JSON.stringify(summary));
    assert(lost === null, `the device was lost during warm-up: ${lost}`);
    return summary;
});
