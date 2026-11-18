/**
 * Quality presets (docs/design/m5-coverage.md §1): a host picks one word
 * instead of thirty settings.
 *
 * A preset is plain data: the DDGI settings it changes (`ddgi`, spread into
 * `DDGITechnique`'s settings), the per-frame GPU budget to pass as
 * `frame.budget`, the texture store's layer size for `createGI`, and the probe
 * spacing to use where the host doesn't choose one. Every call returns a new
 * object, so a host can change single fields:
 *
 * ```ts
 * const preset = ddgiPreset(recommendPreset(adapter.info, adapter.limits));
 * const gi = createGI(device, { textureLayerSize: preset.textureLayerSize });
 * const ddgi = gi.addTechnique(new DDGITechnique({ ...preset.ddgi, raysPerProbe: 160, volume }));
 * gi.encode(encoder, { deltaTime, budget: preset.budget, camera });
 * ```
 *
 * `high` is today's defaults exactly: what M1 to M4 were checked with on the
 * RTX 3060 and what the M5 Pro runs (presets.test.ts holds it to that).
 * `low` is for the correctness devices (the Intel UHD, the iPad, older iOS),
 * and `medium` sits in between. Why these numbers, and what each costs in
 * look, is in docs/api/platform.md ("Quality presets").
 *
 * Rays and tile sizes need a new `prepare`, as they always have: switching
 * presets on a running technique means a new technique (or new settings and
 * `prepare` again). That's how DDGI's resources are sized, not a recompile
 * bug.
 */

import type { GIPlatformOptions } from '../../platform/contract/platform.ts';
import type { DDGISettings } from './settings.ts';

export type DDGIPresetName = 'low' | 'medium' | 'high';

/** The presets from cheapest to best. */
export const DDGI_PRESET_NAMES: readonly DDGIPresetName[] = ['low', 'medium', 'high'];

/** The DDGI settings a preset sets; every other setting keeps its default. */
export type DDGIPresetSettings = Pick<DDGISettings, 'raysPerProbe' | 'irradianceTexels' | 'distanceTexels'>;

export interface DDGIPreset {
    readonly name: DDGIPresetName;
    /** DDGI settings over the defaults: spread them into `new DDGITechnique({ ...preset.ddgi, volume })`. */
    ddgi: DDGIPresetSettings;
    /** The budget to pass as `frame.budget` (three: `start`'s `budget`). */
    budget: { gpuMs: number };
    /** `createGI`'s `textureLayerSize`: texels on a side of the platform's copy of each material texture. */
    textureLayerSize: NonNullable<GIPlatformOptions['textureLayerSize']>;
    /**
     * Metres between probes where the host doesn't set a spacing (three: a
     * `DDGIVolume` made without probe counts). A host that sets one keeps it.
     */
    probeSpacing: number;
}

/**
 * The table. `ddgi` names only what differs between presets; `high`'s
 * values are the defaults written out (DEFAULT_DDGI_SETTINGS, the adapters'
 * 2 ms budget, the store's 256² layers, `LightProbeGrid`'s one probe a metre),
 * so the test can hold them to it.
 *
 * - **Rays** (64 to 512, a multiple of 32): the first 32 are the fixed rays
 *   that are never blended, so 128 still leaves 96 random rays an update.
 *   Fewer rays: each update is noisier, so the field flickers a little more
 *   and takes a little longer to settle. Ray work scales with them.
 * - **Distance tiles** (the visibility test that stops leaks) take most of a
 *   probe's memory and its blend: 14² interior texels against 6² for
 *   irradiance. 10² and 8² still resolve the shadow edges of walls a spacing
 *   away; finer occluders blur. Irradiance stays at 6²: it is already small,
 *   and 4² would make the light's direction visibly blocky.
 * - **Budget:** fewer GPU milliseconds a frame for probe updates, so a weak
 *   GPU keeps its frame rate; the field then follows changes more slowly
 *   (the platform's floor still sweeps every probe within a few seconds).
 * - **Texture store:** 128² layers take a quarter of the memory. GI rays
 *   mostly read coarse mips (a Sponza wall 3 m away reads mip 6 of 256), so
 *   only hits up close see fewer texels.
 * - **Spacing:** 2 m has an eighth of 1 m's probes; light changes over
 *   less than 2 m (a doorway, a corner) blur, and thin walls leak sooner.
 */
const PRESETS: Readonly<Record<DDGIPresetName, DDGIPreset>> = {
    low: { name: 'low', ddgi: { raysPerProbe: 128, irradianceTexels: 6, distanceTexels: 8 }, budget: { gpuMs: 1 }, textureLayerSize: 128, probeSpacing: 2 },
    medium: { name: 'medium', ddgi: { raysPerProbe: 192, irradianceTexels: 6, distanceTexels: 10 }, budget: { gpuMs: 1.5 }, textureLayerSize: 256, probeSpacing: 1.5 },
    high: { name: 'high', ddgi: { raysPerProbe: 256, irradianceTexels: 6, distanceTexels: 14 }, budget: { gpuMs: 2 }, textureLayerSize: 256, probeSpacing: 1 },
};

export function isDDGIPresetName(value: unknown): value is DDGIPresetName {
    return typeof value === 'string' && (DDGI_PRESET_NAMES as readonly string[]).includes(value);
}

/** A preset's settings, as a new object each call (change any field you like). */
export function ddgiPreset(name: DDGIPresetName): DDGIPreset {
    const preset = PRESETS[name];
    if (!preset) throw new RangeError(`@pmndrs/gi: no quality preset "${String(name)}"; the presets are ${DDGI_PRESET_NAMES.join(', ')}`);
    return { ...preset, ddgi: { ...preset.ddgi }, budget: { ...preset.budget } };
}

/** What `recommendPreset` reads of the adapter's info: any `GPUAdapterInfo` will do. */
export interface PresetAdapterInfo {
    readonly vendor: string;
    readonly architecture: string;
    readonly device?: string;
    readonly description?: string;
    /** Chrome's `GPUAdapterInfo.isFallbackAdapter` (a software adapter). */
    readonly isFallbackAdapter?: boolean;
}

/** What `recommendPreset` reads of the adapter's limits. */
export type PresetAdapterLimits = Pick<GPUSupportedLimits, 'maxBufferSize'>;

export interface RecommendPresetHints {
    /**
     * The page runs on a phone or tablet. Default: read from `navigator`
     * (`userAgentData.mobile`, an Android, iPhone or iPad user agent, or an
     * iPad that reports itself as a Mac with a touch screen).
     */
    readonly mobile?: boolean;
}

/** Mobile GPU makers, as `GPUAdapterInfo.vendor` names them (Adreno, Mali, PowerVR, Xclipse, Mali in MediaTek's chips). */
const MOBILE_VENDORS = ['qualcomm', 'arm', 'imagination', 'samsung', 'mediatek'];

/** Intel's discrete GPUs (Arc), by `architecture`; every other Intel GPU is integrated. */
const INTEL_DISCRETE_ARCHITECTURES = ['xe-hpg', 'xe2-hpg', 'alchemist', 'battlemage'];

/** Software adapters, by name, for browsers that don't set `isFallbackAdapter`. */
const SOFTWARE_ADAPTERS = /swiftshader|llvmpipe|lavapipe|microsoft basic render/i;

/**
 * An adapter that offers less than this per buffer is taken to be a
 * small-memory device. Both test laptops' GPUs offer 2 GiB (NOTES), and the
 * WebGPU default is 256 MiB.
 */
const SMALL_DEVICE_MAX_BUFFER_BYTES = 1024 * 1024 * 1024;

/**
 * The preset to start from on this adapter: `low` for integrated and mobile
 * GPUs, `high` otherwise. Advice only: nothing applies it behind the host's
 * back, and a host may start from it and change single settings.
 *
 * Browsers say little about the GPU (Chrome gives a vendor and an
 * architecture, and usually no device name), so the rules are guesses from
 * what they do say, in this order:
 *
 * 1. a software adapter, a phone or tablet, or a mobile GPU maker: `low`;
 * 2. an adapter whose buffers are capped under 1 GiB: `low`;
 * 3. Intel: `low` unless the architecture is an Arc card's;
 * 4. Apple on a Mac: `high` (the Pro and Max chips are the target hardware).
 *    A description that names a base M chip ("Apple M2", no Pro, Max or
 *    Ultra) gives `low`, but Chrome doesn't show the chip, so a base Mac
 *    usually gets `high`;
 * 5. AMD: `low` when the description names integrated Radeon graphics, else
 *    `high`; NVIDIA and anything else: `high`.
 */
export function recommendPreset(adapterInfo: PresetAdapterInfo, limits: PresetAdapterLimits, hints: RecommendPresetHints = {}): DDGIPresetName {
    const vendor = adapterInfo.vendor.toLowerCase();
    const architecture = adapterInfo.architecture.toLowerCase();
    const description = `${adapterInfo.description ?? ''} ${adapterInfo.device ?? ''}`.trim();
    const mobile = hints.mobile ?? detectMobile();

    if (adapterInfo.isFallbackAdapter === true || SOFTWARE_ADAPTERS.test(`${vendor} ${description}`)) return 'low';
    if (mobile || MOBILE_VENDORS.includes(vendor)) return 'low';
    if (limits.maxBufferSize < SMALL_DEVICE_MAX_BUFFER_BYTES) return 'low';
    if (vendor === 'intel') return INTEL_DISCRETE_ARCHITECTURES.some((name) => architecture.includes(name)) ? 'high' : 'low';
    if (vendor === 'apple') return /\bM\d+\b(?!\s*(Pro|Max|Ultra))/i.test(description) ? 'low' : 'high';
    if (vendor === 'amd' && /radeon(\(tm\))?\s+(graphics|vega)/i.test(description)) return 'low';
    return 'high';
}

/** Whether the page runs on a phone or tablet, from `navigator` (false outside a browser). */
function detectMobile(): boolean {
    const browser = (globalThis as { navigator?: { userAgent?: string; maxTouchPoints?: number; userAgentData?: { mobile?: boolean } } }).navigator;
    if (!browser) return false;
    if (browser.userAgentData?.mobile === true) return true;
    const agent = browser.userAgent ?? '';
    if (/Android|iPhone|iPad|iPod/i.test(agent)) return true;
    // iPadOS asks for desktop pages as a Mac; only its touch screen gives it away.
    return /Macintosh/.test(agent) && (browser.maxTouchPoints ?? 0) > 1;
}
