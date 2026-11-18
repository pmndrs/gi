/**
 * Typed errors, following the `@pmndrs/upscaler` convention: a stable `code`
 * for consumers that can't use `instanceof` across bundles.
 */

/** A call that needs the platform ready was made before `prepare()` finished, or after the device was lost. */
export class GINotReadyError extends Error {
    readonly code = 'GI_NOT_READY';

    constructor(readonly reason: 'preparing' | 'device-lost') {
        super(
            reason === 'device-lost'
                ? '@pmndrs/gi: the GPU device was lost. Create a new platform on the new device.'
                : '@pmndrs/gi: await prepare() before encode().',
        );
        this.name = 'GINotReadyError';
    }
}

/**
 * The scene or settings need more than the device allows. The message names
 * the limit and the value to request through `requiredLimits`.
 */
export class GILimitError extends Error {
    readonly code = 'GI_LIMIT';

    constructor(
        readonly limit: Exclude<keyof GPUSupportedLimits, '__brand'>,
        readonly needed: number,
        readonly available: number,
    ) {
        super(
            `@pmndrs/gi: needs ${limit} = ${needed}, but the device allows ${available}. ` +
                'Request it when creating the device; requiredLimits(adapter) lists what to ask for.',
        );
        this.name = 'GILimitError';
    }
}
