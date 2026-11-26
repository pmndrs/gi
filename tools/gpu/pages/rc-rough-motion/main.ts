import { assert, reportGpuTest, summarizeAdapter } from '../../client.ts';
import { EXAMPLES_SITE_BASE, type AdapterSummary } from '../../contract.ts';
import { waitForRailControl, type RailCapture, type RailControl } from '../rc-rail/frame.ts';

type Options = Omit<Parameters<RailControl['render']>[0], 'view'> & {
    view?: 'lit' | 'direct' | 'diffuse' | 'specular';
    roughness?: number;
    specular?: 'on' | 'off';
    motion?: 'left' | 'right' | 'still' | 'sequence';
};
type Sample = RailCapture & { reflection: {
    enabled: boolean; roughness: number | number[]; motion: string;
    emitterPosition: number[]; cameraPosition: number[]; emitterOnScreen: boolean; mirrorUv: number[];
} };
type Control = { render(options: Options): Promise<Sample> };

const data: Record<string, unknown> = {};
const startTime = performance.now();
let adapter: AdapterSummary | null = null;
const iframe = document.createElement('iframe');
iframe.style.cssText = 'width:480px;height:270px;border:0;display:block;';
document.body.append(iframe);

async function open(scene: string, backend = 'interval') {
    const loaded = new Promise(resolve => iframe.addEventListener('load', resolve, { once: true }));
    const options = `verify=1&scene=${scene}&backend=${backend}&camera=receiver&cue=blue&motion=still&resolve=native&view=specular`;
    iframe.src = new URL(`${EXAMPLES_SITE_BASE}11-radiance-cascades/?${options}`, location.origin).href;
    await loaded;
    return await waitForRailControl(iframe) as Control;
}

function floorDelta(a: Sample, b: Sample) {
    assert(a.width === b.width && a.height === b.height, 'mismatched floor readbacks');
    let difference = 0, samples = 0, flux = 0, momentX = 0, momentXX = 0, peak = 0;
    for (let y = Math.floor(a.height * 0.63); y < Math.floor(a.height * 0.98); y++) {
        for (let x = Math.floor(a.width * 0.1); x < Math.floor(a.width * 0.9); x++) {
            const i = (y * a.width + x) * 4;
            const dr = a.pixels[i] - b.pixels[i], dg = a.pixels[i + 1] - b.pixels[i + 1], db = a.pixels[i + 2] - b.pixels[i + 2];
            difference += (Math.abs(dr) + Math.abs(dg) + Math.abs(db)) / 3;
            // Blue-emitter response after subtracting the same material with its lamp off.
            const energy = Math.max(0, (0.35 * dg + 0.65 * db) / 255);
            const u = (x + 0.5) / a.width;
            flux += energy; momentX += u * energy; momentXX += u * u * energy;
            peak = Math.max(peak, energy);
            samples++;
        }
    }
    const centerX = momentX / Math.max(flux, 1e-12);
    const width = Math.sqrt(Math.max(0, momentXX / Math.max(flux, 1e-12) - centerX * centerX));
    return { rgbMae255: difference / samples, mean: flux / samples, centerX, width, peak };
}

try {
    const deviceAdapter = await navigator.gpu?.requestAdapter();
    if (deviceAdapter) adapter = summarizeAdapter(deviceAdapter);
    const control = await open('reflection-motion');
    const fixed = { frames: 36, frame: 180, advance: false, camera: 'receiver', view: 'specular', specular: 'on', captureSize: 'display' } as const;
    const dark = await control.render({ ...fixed, roughness: 0.15, cue: 'off', motion: 'still' });
    const left = await control.render({ ...fixed, roughness: 0.15, cue: 'blue', motion: 'left' });
    const right = await control.render({ ...fixed, roughness: 0.15, cue: 'blue', motion: 'right' });
    const rough = await control.render({ ...fixed, roughness: 0.85, cue: 'blue', motion: 'right' });
    const roughDark = await control.render({ ...fixed, roughness: 0.85, cue: 'off', motion: 'right' });
    const off = await control.render({ ...fixed, frames: 2, specular: 'off', cue: 'blue', motion: 'right' });
    const l = floorDelta(left, dark), r = floorDelta(right, dark), roughResponse = floorDelta(rough, roughDark);
    const material = floorDelta(rough, right);
    let offPeak = 0;
    for (let i = 0; i < off.pixels.length; i += 4) offPeak = Math.max(offPeak, off.pixels[i], off.pixels[i + 1], off.pixels[i + 2]);
    data.fixed = { left: l, right: r, rough: roughResponse, material, offPeak,
        predictedLeftUv: left.reflection.mirrorUv, predictedRightUv: right.reflection.mirrorUv,
        leftOnScreen: left.reflection.emitterOnScreen, rightOnScreen: right.reflection.emitterOnScreen };
    assert(left.reflection.enabled && right.reflection.enabled && !off.reflection.enabled, 'specular toggle state did not track the requests');
    assert(!left.reflection.emitterOnScreen && !right.reflection.emitterOnScreen, 'test lamp entered the camera frustum');
    assert(left.reflection.cameraPosition.join() === right.reflection.cameraPosition.join(), 'receiver camera moved during the lamp comparison');
    assert(l.mean > 1 / 255 && r.mean > 1 / 255, 'offscreen emitter contributed no measurable reflected light to the floor');
    assert(r.centerX - l.centerX > 0.01, `reflected blue light did not shift toward the moving lamp: ${l.centerX} → ${r.centerX}`);
    assert(material.rgbMae255 > 1, 'changing only floor roughness did not change its reflection');
    assert(roughResponse.width > r.width + 0.015 && roughResponse.peak < r.peak - 0.04,
        'increasing roughness did not broaden and dim the reflected blue highlight');
    assert(offPeak <= 1, `disabled specular output still contains light (peak ${offPeak})`);

    const rail = await open('polished');
    const moving = { frames: 40, view: 'specular', camera: 'rail', cue: 'blue', specular: 'on', motion: 'still', roughness: 0.24, captureSize: 'display' } as const;
    const first = await rail.render({ ...moving, frame: 115, advance: false });
    const later = await rail.render({ ...moving, advance: true });
    assert(later.frame - first.frame === moving.frames, 'camera rail skipped a simulation tick');
    assert(later.reflection.emitterPosition.join() === first.reflection.emitterPosition.join(), 'reference light moved during the camera-only comparison');
    assert(later.reflection.cameraPosition.join() !== first.reflection.cameraPosition.join(), 'rail camera remained still');
    const railDelta = floorDelta(later, first);
    data.rail = { ticks: [first.frame, later.frame], floorRgbMae255: railDelta.rgbMae255 };
    assert(railDelta.rgbMae255 > 1, 'reflected floor image did not change along the rail');
    const sweep = await open('roughness');
    const strips = await sweep.render({ frames: 16, view: 'specular', advance: false });
    data.roughnessStrips = strips.reflection.roughness;
    assert(Array.isArray(strips.reflection.roughness) && strips.reflection.roughness.length === 5, 'roughness sweep did not load its five finishes');
    const reference = await open('polished', 'three');
    const direct = await reference.render({ frames: 1, view: 'direct', advance: false });
    const specularControl = iframe.contentDocument!.querySelector<HTMLSelectElement>('#specular')!;
    assert(direct.backend === 'three' && !direct.reflection.enabled && specularControl.disabled && specularControl.value === 'off',
        'Three diffuse reference advertised a specular gather');
    reportGpuTest({ status: 'pass', data, adapter, startTime });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? error.message : String(error), data, adapter, startTime });
}
