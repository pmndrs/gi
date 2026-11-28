/**
 * RC1.4 (TASKS.md, radiance-cache track): functional checks of the features
 * the radiance cache needs, on three devices from the same GPU:
 *
 * 1. **Ours:** made by `requestGIDevice` (packages/three/src/host.ts),
 *    the adapter's host helper: every feature the adapter offers and the
 *    platform's `requiredLimits`.
 * 2. **Without tier2:** the same request minus `texture-formats-tier2`, to
 *    show what tier1 gives, that read-write `rgba16float` is refused cleanly
 *    (a captured validation error), and that D5's ping-pong fallback works.
 * 3. **three's own:** `new WebGPURenderer()` + `init()` with no device
 *    passed, so three requests the adapter and device itself
 *    (WebGPUBackend.js:213-250 in r187, quoted below).
 *
 * Gaps (a feature the adapter doesn't offer, or three not enabling or not
 * raising something) are data, not failures: the page fails only when a
 * check that ran gives a wrong result, when our device lacks a feature its
 * adapter offers, or when "without tier2" accepts read-write `rgba16float`.
 */

import { WebGPURenderer } from 'three/webgpu';
import { drainDevice, reportGpuTest, summarizeAdapter, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import type { AdapterSummary } from '../../contract.ts';
import { deviceLimits, requestGIDevice } from '../../../../packages/three/src/host.ts';
import { optionalFeatures } from '../../../../packages/core/src/platform/device-requirements.ts';
import { fullBrowserVersion } from '../trace-bench/run.ts';
import {
    CHECKED_FEATURES,
    FEATURE_NEEDED,
    REPORTED_LIMITS,
    acceptsReadWriteRgba16Float,
    checkWithoutTier2,
    reportedLimits,
    runFeatureChecks,
    type CheckedFeature,
    type FeatureResult,
} from './checks.ts';

/**
 * What three r187 requests when it makes its own device, read from its source
 * (node_modules/three/src/renderers/webgpu/WebGPUBackend.js):
 * - :215-219 `requestAdapter({ powerPreference, featureLevel: 'compatibility', xrCompatible })`;
 * - :231-243 every name in `GPUFeatureName` (utils/WebGPUConstants.js:328-349) the adapter has,
 *   which includes `timestamp-query`, `shader-f16`, `subgroups`, `texture-formats-tier1` and
 *   `texture-formats-tier2` (and `core-features-and-limits`);
 * - :245-248 `requiredLimits: parameters.requiredLimits`, which defaults to `{}` (:97): no raised
 *   limits unless the host passes them to `new WebGPURenderer({ requiredLimits })`.
 *
 * Read from source, not measured, so it goes in the data as text only (no checks.csv row); the
 * measured `three/features/<feature>/enabled` and `three/limits/*` rows show the same thing.
 */
const THREE_SOURCE_NOTE =
    'three r187 WebGPUBackend.js: compatibility adapter (:215-219); every GPUFeatureName the adapter has (:231-243, WebGPUConstants.js:328-349); requiredLimits defaults to {} (:97, :245-248)';

/** Our feature results as plain data (numbers and booleans go to checks.csv; `problems` strings don't). */
function featureData(results: Record<CheckedFeature, FeatureResult>): Record<string, unknown> {
    return Object.fromEntries(CHECKED_FEATURES.map((feature) => [feature, results[feature]]));
}

/** Whether the platform's own advice (`optionalFeatures`, ddgi-core) asks for the feature. */
function platformAdvises(adapter: GPUAdapter): Record<string, boolean> {
    const advised = new Set<string>(optionalFeatures(adapter));
    return Object.fromEntries(
        CHECKED_FEATURES.map((feature) => {
            const needed = FEATURE_NEEDED[feature];
            return [feature, needed === null || advised.has(needed)];
        }),
    );
}

function problemsOf(label: string, results: Record<CheckedFeature, FeatureResult>): string[] {
    return CHECKED_FEATURES.flatMap((feature) => results[feature].problems.map((problem) => `${label} ${feature}: ${problem}`));
}

const startTime = performance.now();
const logLines: string[] = [];
const log = (line: string): void => {
    logLines.push(line);
};
const data: Record<string, unknown> = { userAgent: navigator.userAgent };
const deviceErrorLists: string[][] = [];
const devices: GPUDevice[] = [];
let adapterSummary: AdapterSummary | null = null;
const failures: string[] = [];
const gaps: string[] = [];

try {
    data.browserVersion = await fullBrowserVersion();
    log(`browser: ${navigator.userAgent} (${String(data.browserVersion)})`);

    // 1. Our device, the platform's way.
    const { adapter, device } = await requestGIDevice({ label: 'rc1.4 ours' });
    devices.push(device);
    deviceErrorLists.push(watchDevice(device));
    adapterSummary = summarizeAdapter(adapter);
    const ours = await runFeatureChecks(device, adapter.features);
    data.ours = {
        coreFeaturesAndLimits: device.features.has('core-features-and-limits'),
        platformOptionalFeaturesAdvise: platformAdvises(adapter),
        features: featureData(ours),
        readWriteRgba16FloatLayoutAccepted: await acceptsReadWriteRgba16Float(device),
        limits: reportedLimits(device.limits),
        adapterLimits: reportedLimits(adapter.limits),
    };
    failures.push(...problemsOf('ours', ours));
    for (const feature of CHECKED_FEATURES) {
        const result = ours[feature];
        if (!result.offered) gaps.push(`${feature}: the adapter doesn't offer it`);
        else if (!result.enabled) failures.push(`ours ${feature}: offered by the adapter but not enabled on our device`);
        log(`ours  ${feature}: offered ${result.offered}, enabled ${result.enabled}, ${result.ran ? (result.pass ? 'PASS' : 'FAIL') : 'not run'}`);
    }

    // 2. The same request without tier2.
    const plainAdapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!plainAdapter) throw new Error('requestAdapter() returned null for the device without tier2');
    const withoutTier2Device = await plainAdapter.requestDevice({
        label: 'rc1.4 without tier2',
        requiredFeatures: [...plainAdapter.features].filter((feature) => feature !== 'texture-formats-tier2') as GPUFeatureName[],
        requiredLimits: deviceLimits(plainAdapter),
    });
    devices.push(withoutTier2Device);
    deviceErrorLists.push(watchDevice(withoutTier2Device));
    const withoutTier2 = await checkWithoutTier2(withoutTier2Device);
    data.withoutTier2 = { ...withoutTier2, tier2Enabled: withoutTier2Device.features.has('texture-formats-tier2') };
    failures.push(...withoutTier2.problems.map((problem) => `without tier2: ${problem}`));
    log(
        `without tier2: tier1 ${withoutTier2.tier1Enabled}, rg16float write-only ${withoutTier2.rg16floatWriteOnlyPass}, ` +
            `read-write rgba16float refused ${withoutTier2.rgba16floatReadWriteRefused} ("${withoutTier2.refusalMessage.split('\n')[0]}"), ` +
            `ping-pong fallback ${withoutTier2.pingPongFallbackPass}`,
    );

    // 3. three's own device. The compatibility adapter is requested only to read what it offers three.
    const compatibilityAdapter = await navigator.gpu.requestAdapter({ featureLevel: 'compatibility' });
    if (!compatibilityAdapter) throw new Error('requestAdapter({ featureLevel: "compatibility" }) returned null');
    const renderer = new WebGPURenderer();
    await renderer.init();
    const threeDevice = (renderer.backend as unknown as { device: GPUDevice }).device;
    devices.push(threeDevice);
    deviceErrorLists.push(watchDevice(threeDevice));
    const threeAdapter = summarizeAdapterInfo(threeDevice.adapterInfo, threeDevice.features);
    if (adapterSummary && threeAdapter.vendor !== adapterSummary.vendor) {
        failures.push(`three's device is on ${threeAdapter.vendor}/${threeAdapter.architecture}, ours on ${adapterSummary.vendor}/${adapterSummary.architecture}`);
    }
    const three = await runFeatureChecks(threeDevice, compatibilityAdapter.features);
    const threeLimits = reportedLimits(threeDevice.limits);
    const ourLimits = reportedLimits(device.limits);
    data.three = {
        adapter: `${threeAdapter.vendor}/${threeAdapter.architecture}`,
        coreFeaturesAndLimits: threeDevice.features.has('core-features-and-limits'),
        source: THREE_SOURCE_NOTE,
        features: featureData(three),
        readWriteRgba16FloatLayoutAccepted: await acceptsReadWriteRgba16Float(threeDevice),
        limits: threeLimits,
        limitsBelowOurs: REPORTED_LIMITS.filter((name) => threeLimits[name] < ourLimits[name]).length,
    };
    failures.push(...problemsOf('three', three));
    for (const feature of CHECKED_FEATURES) {
        const result = three[feature];
        if (result.offered && !result.enabled) gaps.push(`three ${feature}: offered but not enabled on three's own device`);
        log(`three ${feature}: offered ${result.offered}, enabled ${result.enabled}, ${result.ran ? (result.pass ? 'PASS' : 'FAIL') : 'not run'}`);
    }
    for (const name of REPORTED_LIMITS) {
        if (threeLimits[name] < ourLimits[name]) gaps.push(`three ${name}: ${threeLimits[name]} on three's own device against ${ourLimits[name]} on ours`);
        log(`limit ${name}: ours ${ourLimits[name]}, three ${threeLimits[name]}, adapter ${(adapter.limits as unknown as Record<string, number>)[name]}`);
    }
    data.gaps = gaps;
    data.gapCount = gaps.length;
    for (const gap of gaps) log(`GAP ${gap}`);

    for (const device of devices) await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data,
        adapter: adapterSummary,
        deviceErrors: deviceErrorLists.flat(),
        startTime,
        logLines,
    });
} catch (error) {
    reportGpuTest({
        status: 'fail',
        message: [error instanceof Error ? error.message : String(error), ...failures].join(' | '),
        data,
        adapter: adapterSummary,
        deviceErrors: deviceErrorLists.flat(),
        startTime,
        logLines,
    });
}
