/**
 * What the browser says about itself and its WebGPU support. Collected before
 * the test runs and never throws: a browser without WebGPU still has to be
 * able to report "WebGPU: no" and its user agent.
 */

export interface AdapterDetails {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
    isFallbackAdapter: boolean;
    featureCount: number;
    hasTimestampQuery: boolean;
}

export interface DeviceEnvironment {
    userAgent: string;
    platform: string;
    isSecureContext: boolean;
    hasNavigatorGpu: boolean;
    /** `requestAdapter()` gave something. */
    adapterAvailable: boolean;
    adapter: AdapterDetails | null;
    /** Why there is no adapter, when we know. */
    problem: string | null;
}

export async function collectEnvironment(): Promise<DeviceEnvironment> {
    const environment: DeviceEnvironment = {
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        isSecureContext: window.isSecureContext,
        hasNavigatorGpu: 'gpu' in navigator && navigator.gpu !== undefined,
        adapterAvailable: false,
        adapter: null,
        problem: null,
    };

    if (!environment.hasNavigatorGpu) {
        environment.problem = environment.isSecureContext
            ? 'This browser has no navigator.gpu (WebGPU is not supported or is switched off).'
            : 'The page is not a secure context, so the browser hides WebGPU. Open it over https.';
        return environment;
    }

    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            environment.problem = 'requestAdapter() returned null (no usable GPU).';
            return environment;
        }
        environment.adapterAvailable = true;
        // Older Safari builds have no `adapter.info`; read it defensively.
        const info = (adapter as { info?: GPUAdapterInfo }).info;
        environment.adapter = {
            vendor: info?.vendor ?? '',
            architecture: info?.architecture ?? '',
            device: info?.device ?? '',
            description: info?.description ?? '',
            isFallbackAdapter: info?.isFallbackAdapter ?? false,
            featureCount: adapter.features.size,
            hasTimestampQuery: adapter.features.has('timestamp-query'),
        };
    } catch (error) {
        environment.problem = `requestAdapter() threw: ${error instanceof Error ? error.message : String(error)}`;
    }
    return environment;
}
